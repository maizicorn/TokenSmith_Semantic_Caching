import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class CacheEntry:
    response: str
    embedding: Optional[np.ndarray]   # None for exact strategy
    llm_latency_s: float = 0.0        # how long the original LLM call took
    hits: int = 0
    created_at: float = field(default_factory=time.time)


def _normalize(query: str) -> str:
    """Lowercase + collapse whitespace so near-identical queries share a key."""
    return " ".join(query.lower().split())

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
    strategy  : "exact" or "semantic"
    threshold : cosine similarity cutoff for semantic (default 0.90)
    embed_fn  : required for semantic — function(str) -> list[float]
    max_size  : max entries before LRU eviction
    """

    def __init__(
        self,
        strategy: str = "semantic",
        threshold: float = 0.90,
        embed_fn: Optional[Callable] = None,
        max_size: int = 512,
    ):
        assert strategy in ("exact", "semantic"), f"Unknown strategy: {strategy}"
        self.strategy  = strategy
        self.threshold = threshold
        self.embed_fn  = embed_fn
        self.max_size  = max_size

        self._store: dict[str, CacheEntry] = {}
        self._order: list[str] = []   # insertion order for LRU eviction

        self._hits   = 0
        self._misses = 0
        self._total_latency_saved = 0.0

    def get(self, query: str) -> Optional[str]:
        """Return cached response string, or None on miss."""
        if self.strategy == "exact":
            result = self._get_exact(query)
        else:
            result = self._get_semantic(query)

        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, query: str, response: str, llm_latency_s: float = 0.0):
        """Store a response. Pass llm_latency_s for latency-saved reporting."""
        key = _normalize(query)

        embedding = None
        if self.strategy == "semantic" and self.embed_fn:
            embedding = _embed(query, self.embed_fn)

        entry = CacheEntry(
            response=response,
            embedding=embedding,
            llm_latency_s=llm_latency_s,
        )

        # LRU eviction
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
    # Internal lookup
    # ------------------------------------------------------------------

    def _get_exact(self, query: str) -> Optional[str]:
        key   = _normalize(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        entry.hits += 1
        self._total_latency_saved += entry.llm_latency_s
        self._order.remove(key)
        self._order.append(key)
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
            entry = self._store[best_key]
            entry.hits += 1
            self._total_latency_saved += entry.llm_latency_s
            self._order.remove(best_key)
            self._order.append(best_key)
            return entry.response

        return None
