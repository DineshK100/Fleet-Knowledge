"""Pure-Python fleet memory store (no ROS dependency).

Implements Section 3.5 of the proposal:

* every record holds text, text embedding, position, heading, timestamps,
  hazard type, severity and a confidence score;
* a **spatial index** (uniform grid / geohash-style buckets) over (x, y);
* a **vector index** over description embeddings (numpy, optionally FAISS);
* **merging** of a new report into an existing record that is close, of the
  same type, travelling in the same direction and has similar text;
* **confidence decay** ``c <- c * exp(-lambda * dt)``, **confirmations** and
  **refutations**, and **pruning** of records whose confidence falls below a
  threshold.

Times are plain floats in seconds (the ROS node feeds simulation time).
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .embeddings import HashingEmbedder

HAZARD_TYPES = ('pedestrian_crossing', 'lane_closure', 'road_damage', 'stalled_vehicle', 'other')


@dataclass
class MemoryRecord:
    id: str = ''
    hazard_type: str = 'other'
    description: str = ''
    lane: str = 'unknown'
    severity: int = 3
    advice: str = ''
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    road_id: int = -1
    lane_id: int = 0
    first_seen: float = 0.0
    last_confirmed: float = 0.0
    reporter_id: str = ''
    confidence: float = 0.0
    count: int = 1
    refutations: int = 0
    source: str = 'vlm'
    trigger: str = ''
    # bookkeeping: time at which ``confidence`` was last brought up to date
    conf_time: float = 0.0

    def text_for_embedding(self) -> str:
        parts = [self.hazard_type.replace('_', ' '), self.description]
        if self.lane and self.lane != 'unknown':
            parts.append(f'{self.lane} lane')
        if self.advice:
            parts.append(self.advice)
        return '. '.join(p for p in parts if p)


@dataclass
class MemoryConfig:
    merge_radius_m: float = 20.0
    merge_text_similarity: float = 0.3
    merge_heading_cos: float = 0.0          # require cos(dheading) >= this to merge
    initial_confidence: float = 0.6
    confirm_gain: float = 0.5                # c <- c + gain * (1 - c)
    refute_factor: float = 0.5               # c <- c * factor
    decay_lambda: float = 1.0 / 3600.0       # 1/s
    prune_threshold: float = 0.1
    grid_cell_m: float = 25.0
    decay_exempt_sources: Tuple[str, ...] = ('oracle',)


class SpatialGrid:
    """Uniform grid buckets over (x, y) for radius queries."""

    def __init__(self, cell: float = 25.0):
        self.cell = float(cell)
        self._cells: Dict[Tuple[int, int], set] = {}
        self._pos: Dict[str, Tuple[float, float]] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return int(math.floor(x / self.cell)), int(math.floor(y / self.cell))

    def insert(self, rid: str, x: float, y: float):
        self.remove(rid)
        self._pos[rid] = (x, y)
        self._cells.setdefault(self._key(x, y), set()).add(rid)

    def remove(self, rid: str):
        pos = self._pos.pop(rid, None)
        if pos is None:
            return
        k = self._key(*pos)
        bucket = self._cells.get(k)
        if bucket is not None:
            bucket.discard(rid)
            if not bucket:
                del self._cells[k]

    def query(self, x: float, y: float, radius: float) -> List[Tuple[str, float]]:
        out = []
        r = max(0.0, radius)
        kx0, ky0 = self._key(x - r, y - r)
        kx1, ky1 = self._key(x + r, y + r)
        for kx in range(kx0, kx1 + 1):
            for ky in range(ky0, ky1 + 1):
                for rid in self._cells.get((kx, ky), ()):
                    px, py = self._pos[rid]
                    d = math.hypot(px - x, py - y)
                    if d <= r:
                        out.append((rid, d))
        return out

    def clear(self):
        self._cells.clear()
        self._pos.clear()


class VectorIndex:
    """Cosine-similarity index over L2-normalised embeddings.

    The memory is small (hundreds of records), so similarities for the spatial
    candidates are computed exactly with numpy. If FAISS is installed it is
    used for the global (radius <= 0) top-k search.
    """

    def __init__(self):
        self._vecs: Dict[str, np.ndarray] = {}
        try:
            import faiss  # noqa: F401
            self._faiss = faiss
        except Exception:
            self._faiss = None

    def set(self, rid: str, vec: np.ndarray):
        self._vecs[rid] = np.asarray(vec, dtype=np.float32)

    def remove(self, rid: str):
        self._vecs.pop(rid, None)

    def similarity(self, rid: str, q: np.ndarray) -> float:
        v = self._vecs.get(rid)
        if v is None or q is None or v.shape != q.shape:
            return 0.0
        return float(np.dot(v, q))

    def top_k(self, q: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self._vecs:
            return []
        ids = list(self._vecs.keys())
        mat = np.stack([self._vecs[i] for i in ids]).astype(np.float32)
        k = min(k, len(ids)) if k > 0 else len(ids)
        if self._faiss is not None:
            index = self._faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            sims, idx = index.search(q.reshape(1, -1).astype(np.float32), k)
            return [(ids[j], float(s)) for j, s in zip(idx[0], sims[0]) if j >= 0]
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [(ids[j], float(sims[j])) for j in order]

    def clear(self):
        self._vecs.clear()


class FleetMemoryStore:
    """Thread-safe shared hazard memory."""

    def __init__(self, config: Optional[MemoryConfig] = None, embedder=None):
        self.cfg = config or MemoryConfig()
        self.embedder = embedder or HashingEmbedder()
        self.records: Dict[str, MemoryRecord] = {}
        self.grid = SpatialGrid(self.cfg.grid_cell_m)
        self.vectors = VectorIndex()
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ utils
    def __len__(self):
        return len(self.records)

    def _decay(self, rec: MemoryRecord, now: float):
        if rec.source in self.cfg.decay_exempt_sources:
            rec.conf_time = now
            return
        dt = now - rec.conf_time
        if dt > 0:
            rec.confidence *= math.exp(-self.cfg.decay_lambda * dt)
        rec.conf_time = now

    def refresh(self, now: float):
        with self.lock:
            for rec in self.records.values():
                self._decay(rec, now)

    def _index(self, rec: MemoryRecord):
        self.grid.insert(rec.id, rec.x, rec.y)
        self.vectors.set(rec.id, self.embedder.embed(rec.text_for_embedding()))

    def _remove(self, rid: str) -> Optional[MemoryRecord]:
        rec = self.records.pop(rid, None)
        self.grid.remove(rid)
        self.vectors.remove(rid)
        return rec

    # ------------------------------------------------------------------ write
    def find_merge_target(self, rec: MemoryRecord) -> Optional[MemoryRecord]:
        q = self.embedder.embed(rec.text_for_embedding())
        best, best_key = None, None
        for rid, d in self.grid.query(rec.x, rec.y, self.cfg.merge_radius_m):
            other = self.records[rid]
            if other.hazard_type != rec.hazard_type:
                continue
            if math.cos(rec.heading - other.heading) < self.cfg.merge_heading_cos:
                continue
            sim = self.vectors.similarity(rid, q)
            text_ok = (sim >= self.cfg.merge_text_similarity
                       or not rec.description or not other.description)
            if not text_ok:
                continue
            key = (d, -sim)
            if best_key is None or key < best_key:
                best, best_key = other, key
        return best

    def write(self, rec: MemoryRecord, now: float, allow_merge: bool = True
              ) -> Tuple[MemoryRecord, bool]:
        """Insert ``rec`` or merge it into an existing record. Returns (record, merged)."""
        with self.lock:
            rec.severity = int(min(5, max(1, rec.severity)))
            if allow_merge:
                target = self.find_merge_target(rec)
                if target is not None:
                    self._merge(target, rec, now)
                    return target, True
            if not rec.id or rec.id in self.records:
                rec.id = uuid.uuid4().hex[:12]
            if rec.confidence <= 0.0:
                rec.confidence = self.cfg.initial_confidence
            rec.confidence = float(min(1.0, rec.confidence))
            rec.first_seen = rec.first_seen or now
            rec.last_confirmed = rec.last_confirmed or now
            rec.conf_time = now
            rec.count = max(1, rec.count)
            self.records[rec.id] = rec
            self._index(rec)
            return rec, False

    def _merge(self, target: MemoryRecord, new: MemoryRecord, now: float):
        self._decay(target, now)
        n = target.count
        target.x = (target.x * n + new.x) / (n + 1)
        target.y = (target.y * n + new.y) / (n + 1)
        target.z = (target.z * n + new.z) / (n + 1)
        target.count += 1
        target.confidence = target.confidence + self.cfg.confirm_gain * (1.0 - target.confidence)
        target.last_confirmed = now
        target.severity = max(target.severity, new.severity)
        # prefer a VLM description over a template / empty one, or a more severe report
        if new.description and (not target.description or
                                (target.source != 'vlm' and new.source == 'vlm') or
                                new.severity > target.severity):
            target.description = new.description
            target.advice = new.advice or target.advice
            if new.source == 'vlm':
                target.source = 'vlm'
        if target.lane in ('', 'unknown') and new.lane not in ('', 'unknown'):
            target.lane = new.lane
        self._index(target)

    # ---------------------------------------------------------------- feedback
    def feedback(self, rid: str, confirm: bool, now: float) -> Tuple[Optional[MemoryRecord], bool]:
        """Confirm or refute a record. Returns (record, pruned)."""
        with self.lock:
            rec = self.records.get(rid)
            if rec is None:
                return None, False
            self._decay(rec, now)
            if confirm:
                rec.confidence = rec.confidence + self.cfg.confirm_gain * (1.0 - rec.confidence)
                rec.count += 1
                rec.last_confirmed = now
            else:
                rec.confidence *= self.cfg.refute_factor
                rec.refutations += 1
            if rec.confidence < self.cfg.prune_threshold:
                self._remove(rid)
                return rec, True
            return rec, False

    def prune(self, now: float) -> List[MemoryRecord]:
        with self.lock:
            pruned = []
            for rid in list(self.records):
                rec = self.records[rid]
                self._decay(rec, now)
                if rec.confidence < self.cfg.prune_threshold:
                    pruned.append(self._remove(rid))
            return pruned

    # ------------------------------------------------------------------ query
    def query(self, x: float, y: float, radius: float, now: float,
              query_text: str = '', max_results: int = 0,
              min_confidence: float = 0.0) -> List[Tuple[MemoryRecord, float, float]]:
        """Return ``(record, similarity, distance)`` sorted by distance."""
        with self.lock:
            if radius > 0:
                hits = self.grid.query(x, y, radius)
            else:
                hits = [(rid, math.hypot(r.x - x, r.y - y)) for rid, r in self.records.items()]
            q = self.embedder.embed(query_text) if query_text else None
            out = []
            for rid, d in hits:
                rec = self.records[rid]
                self._decay(rec, now)
                if rec.confidence < min_confidence:
                    continue
                sim = self.vectors.similarity(rid, q) if q is not None else 0.0
                out.append((rec, sim, d))
            out.sort(key=lambda t: t[2])
            if max_results > 0:
                out = out[:max_results]
            return out

    def semantic_search(self, text: str, k: int = 5) -> List[Tuple[MemoryRecord, float]]:
        with self.lock:
            q = self.embedder.embed(text)
            return [(self.records[rid], s) for rid, s in self.vectors.top_k(q, k)]

    # ------------------------------------------------------------ persistence
    def reset(self):
        with self.lock:
            self.records.clear()
            self.grid.clear()
            self.vectors.clear()

    def all_records(self) -> List[MemoryRecord]:
        with self.lock:
            return list(self.records.values())

    def save(self, path: str):
        with self.lock:
            data = [asdict(r) for r in self.records.values()]
        with open(path, 'w') as f:
            json.dump({'records': data}, f, indent=2)

    def load(self, path: str, now: Optional[float] = None) -> int:
        with open(path) as f:
            data = json.load(f)
        with self.lock:
            for d in data.get('records', []):
                rec = MemoryRecord(**{k: v for k, v in d.items()
                                      if k in MemoryRecord.__dataclass_fields__})
                if now is not None:
                    # times from an earlier run are meaningless in this clock
                    rec.first_seen = rec.last_confirmed = rec.conf_time = now
                self.records[rec.id] = rec
                self._index(rec)
            return len(self.records)

    def bulk_write(self, recs: Iterable[MemoryRecord], now: float, allow_merge=True):
        return [self.write(r, now, allow_merge) for r in recs]
