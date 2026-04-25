import hashlib
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
 
 
@dataclass
class CacheEntry:
    response: str
    embedding: Optional[np.ndarray]   # None for exact strategy
    chunk_fp: str = ""                 # SHA-256 prefix of retrieved chunks
    llm_latency_s: float = 0.0
    hits: int = 0
    created_at: float = field(default_factory=time.time)
 
 
def _normalize(query: str) -> str:
    """Lowercase + collapse whitespace."""
    return " ".join(query.lower().split())
 
def _chunk_fingerprint(chunks) -> str:
    """Short hash of the retrieved chunks so we can detect context changes."""
    if not chunks:
        return ""
    texts = []
    for c in chunks:
        if isinstance(c, dict):
            texts.append(c.get("content", ""))
        elif isinstance(c, tuple):
            texts.append(c[0])
        else:
            texts.append(str(c))
    return hashlib.sha256("\n".join(texts).encode("utf-8", errors="replace")).hexdigest()[:16]
 
def _embed(text: str, embed_fn: Callable) -> np.ndarray:
    vec = np.array(embed_fn(text), dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
 
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # both already unit-normalised
 
 
class PromptCache:
    """
    Parameters
    ----------
    strategy  : "exact", "semantic", or "context"
    threshold : cosine similarity cutoff for semantic/context (default 0.90)
    embed_fn  : required for semantic/context — function(str) -> list[float]
    max_size  : max entries before LRU eviction
    """
 
    def __init__(
        self,
        strategy: str = "semantic",
        threshold: float = 0.90,
        embed_fn: Optional[Callable] = None,
        max_size: int = 512,
    ):
        assert strategy in ("exact", "semantic", "context"), f"Unknown strategy: {strategy}"
        self.strategy  = strategy
        self.threshold = threshold
        self.embed_fn  = embed_fn
        self.max_size  = max_size
 
        self._store: dict[str, CacheEntry] = {}
        self._order: list[str] = []
 
        self._hits   = 0
        self._misses = 0
        self._total_latency_saved = 0.0
 
    def get(self, query: str, chunks=None) -> Optional[str]:
        """Return cached response string, or None on miss."""
        if self.strategy == "exact":
            result = self._get_exact(query, chunks)
        elif self.strategy == "semantic":
            result = self._get_semantic(query)
        else:
            result = self._get_context(query, chunks)
 
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
 
    def put(self, query: str, response: str, chunks=None, llm_latency_s: float = 0.0):
        """Store a response."""
        # For exact+context, key encodes both query and chunk fingerprint
        fp  = _chunk_fingerprint(chunks) if chunks else ""
        key = _normalize(query) + ("|" + fp if fp and self.strategy in ("exact", "context") else "")
 
        embedding = None
        if self.strategy in ("semantic", "context") and self.embed_fn:
            embedding = _embed(query, self.embed_fn)
 
        entry = CacheEntry(
            response=response,
            embedding=embedding,
            chunk_fp=fp,
            llm_latency_s=llm_latency_s,
        )
 
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self.max_size:
            oldest = self._order.pop(0)
            del self._store[oldest]
 
        self._store[key] = entry
        self._order.append(key)
 
    def clear(self):
        self._store.clear()
        self._order.clear()
        self._hits = 0
        self._misses = 0
        self._total_latency_saved = 0.0
 
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "strategy":        self.strategy,
            "hits":            self._hits,
            "misses":          self._misses,
            "total":           total,
            "hit_rate":        round(self._hits / total, 4) if total else 0.0,
            "latency_saved_s": round(self._total_latency_saved, 3),
            "size":            len(self._store),
        }
 
    # ------------------------------------------------------------------
    # Internal lookups
    # ------------------------------------------------------------------
 
    def _get_exact(self, query: str, chunks=None) -> Optional[str]:
        fp  = _chunk_fingerprint(chunks) if chunks else ""
        key = _normalize(query) + ("|" + fp if fp else "")
        entry = self._store.get(key)
        if entry is None:
            return None
        self._touch(entry, key)
        return entry.response
 
    def _get_semantic(self, query: str) -> Optional[str]:
        if not self.embed_fn or not self._store:
            return None
        q_vec    = _embed(query, self.embed_fn)
        best_sim = -1.0
        best_key = None
        for key, entry in self._store.items():
            if entry.embedding is None:
                continue
            sim = _cosine(q_vec, entry.embedding)
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= self.threshold:
            self._touch(self._store[best_key], best_key)
            return self._store[best_key].response
        return None
 
    def _get_context(self, query: str, chunks=None) -> Optional[str]:
        """Semantic lookup that also requires the chunk fingerprint to match."""
        if not self.embed_fn or not self._store:
            return None
        incoming_fp = _chunk_fingerprint(chunks) if chunks else ""
        q_vec       = _embed(query, self.embed_fn)
        best_sim    = -1.0
        best_key    = None
        for key, entry in self._store.items():
            if entry.embedding is None:
                continue
            if entry.chunk_fp != incoming_fp:   # context must match
                continue
            sim = _cosine(q_vec, entry.embedding)
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= self.threshold:
            self._touch(self._store[best_key], best_key)
            return self._store[best_key].response
        return None
 
    def _touch(self, entry: CacheEntry, key: str):
        entry.hits += 1
        self._total_latency_saved += entry.llm_latency_s
        self._order.remove(key)
        self._order.append(key)