"""Time-indexed ring buffer of camera frames (about 3 s of history)."""

import threading
from collections import deque
from typing import Any, List, Sequence, Tuple


class FrameBuffer:
    def __init__(self, max_age_s: float = 3.0, max_len: int = 200):
        self.max_age_s = float(max_age_s)
        self._buf: deque = deque(maxlen=max_len)
        self._lock = threading.Lock()

    def append(self, t: float, frame: Any):
        with self._lock:
            if self._buf and t < self._buf[-1][0]:
                self._buf.clear()           # clock jumped backwards (new run)
            self._buf.append((t, frame))
            while self._buf and t - self._buf[0][0] > self.max_age_s + 2.0:
                self._buf.popleft()

    def __len__(self):
        return len(self._buf)

    def latest_time(self) -> float:
        with self._lock:
            return self._buf[-1][0] if self._buf else float('-inf')

    def select(self, times: Sequence[float]) -> List[Tuple[float, Any]]:
        """Return the frame closest to each requested time (duplicates removed, ordered)."""
        with self._lock:
            items = list(self._buf)
        if not items:
            return []
        chosen = []
        for t in times:
            best = min(items, key=lambda it: abs(it[0] - t))
            if not chosen or chosen[-1][0] != best[0]:
                chosen.append(best)
        chosen.sort(key=lambda it: it[0])
        out = []
        for it in chosen:
            if not out or out[-1][0] != it[0]:
                out.append(it)
        return out

    def keyframes(self, t_event: float, n: int = 4, before_s: float = 2.0,
                  after_s: float = 0.5) -> List[Tuple[float, Any]]:
        if n <= 1:
            return self.select([t_event])
        start, end = t_event - before_s, t_event + after_s
        times = [start + i * (end - start) / (n - 1) for i in range(n)]
        return self.select(times)

    def clear(self):
        with self._lock:
            self._buf.clear()
