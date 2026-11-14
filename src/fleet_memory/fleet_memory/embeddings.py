"""Text embedders for the fleet memory vector index.

Two back-ends are provided:

* ``SentenceTransformerEmbedder`` - a real sentence-embedding model
  (default ``all-MiniLM-L6-v2``), used when ``sentence-transformers`` is installed.
* ``HashingEmbedder`` - a dependency-free, deterministic hashed bag of
  unigrams + bigrams. It is much weaker semantically but lets the whole system
  run (and be unit-tested) without downloading a model.

All embedders return L2-normalised ``float32`` vectors so cosine similarity is a
plain dot product.
"""

import hashlib
import re
from typing import List

import numpy as np

_TOKEN_RE = re.compile(r'[a-z0-9]+')

# Very small normalisation table so that common driving synonyms land in the
# same hash bucket when the hashing embedder is used.
_SYNONYMS = {
    'pedestrians': 'pedestrian', 'walker': 'pedestrian', 'walkers': 'pedestrian',
    'person': 'pedestrian', 'people': 'pedestrian',
    'cones': 'cone', 'barriers': 'barrier', 'closed': 'closure', 'blocked': 'closure',
    'roadworks': 'construction', 'works': 'construction',
    'potholes': 'pothole', 'damaged': 'damage', 'bump': 'pothole', 'bumps': 'pothole',
    'cars': 'vehicle', 'car': 'vehicle', 'vehicles': 'vehicle', 'van': 'vehicle',
    'truck': 'vehicle', 'stopped': 'stalled', 'broken': 'stalled', 'parked': 'stalled',
    'intersections': 'intersection', 'junction': 'intersection', 'crossroads': 'intersection',
    'lanes': 'lane',
}


def tokenize(text: str) -> List[str]:
    toks = _TOKEN_RE.findall((text or '').lower())
    return [_SYNONYMS.get(t, t) for t in toks]


class HashingEmbedder:
    """Deterministic hashed n-gram embedding (no external dependencies)."""

    name = 'hashing'

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def _bucket(self, feature: str):
        h = hashlib.md5(feature.encode('utf-8')).digest()
        idx = int.from_bytes(h[:4], 'little') % self.dim
        sign = 1.0 if (h[4] & 1) else -1.0
        return idx, sign

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = list(toks) + [a + '_' + b for a, b in zip(toks, toks[1:])]
        for f in feats:
            idx, sign = self._bucket(f)
            vec[idx] += sign
        n = float(np.linalg.norm(vec))
        if n > 0:
            vec /= n
        return vec


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer  # noqa: import inside
        self.model = SentenceTransformer(model_name)
        self.name = 'st:' + model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)


def make_embedder(backend: str = 'auto',
                  model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                  logger=None):
    """Create an embedder. ``backend`` is ``auto``, ``sentence_transformers`` or ``hashing``."""
    backend = (backend or 'auto').lower()
    if backend in ('auto', 'sentence_transformers', 'st'):
        try:
            emb = SentenceTransformerEmbedder(model_name)
            if logger:
                logger.info(f'Using sentence-transformers embedder ({model_name}, dim={emb.dim})')
            return emb
        except Exception as exc:  # ImportError, download failure, ...
            if backend != 'auto':
                raise
            if logger:
                logger.warning(f'sentence-transformers unavailable ({exc!r}); '
                               'falling back to the hashing embedder')
    return HashingEmbedder()
