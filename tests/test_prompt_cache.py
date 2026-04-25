import time
import pytest
import numpy as np
from prompt_cache import PromptCache, _normalize, _chunk_fingerprint


# ---------------------------------------------------------------------------
# Mock embed_fn for deterministic testing
# ---------------------------------------------------------------------------

def make_embed_fn(mapping: dict):
    """
    Returns an embed_fn that maps keywords to fixed unit vectors.
    Any text not matching a keyword returns the default vector.
    """
    def embed_fn(text):
        text_lower = text.lower()
        for keyword, vec in mapping.items():
            if keyword in text_lower:
                arr = np.array(vec, dtype=np.float32)
                return (arr / np.linalg.norm(arr)).tolist()
        default = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return default.tolist()
    return embed_fn


BTREE_EMBED = make_embed_fn({
    "b-tree":  [1.0, 0.0, 0.0],
    "b tree":  [1.0, 0.0, 0.0],
    "btree":   [1.0, 0.0, 0.0],
    "hashing": [0.0, 1.0, 0.0],
    "hash":    [0.0, 1.0, 0.0],
})


# ===========================================================================
# Helper tests
# ===========================================================================

class TestHelpers:

    def test_normalize_lowercase(self):
        assert _normalize("What is a B-Tree?") == "what is a b-tree?"

    def test_normalize_whitespace(self):
        assert _normalize("  what   is   hashing  ") == "what is hashing"

    def test_normalize_idempotent(self):
        q = "what is a b-tree?"
        assert _normalize(q) == _normalize(_normalize(q))

    def test_chunk_fingerprint_empty(self):
        assert _chunk_fingerprint([]) == ""
        assert _chunk_fingerprint(None) == ""

    def test_chunk_fingerprint_stable(self):
        chunks = ["B-trees store sorted data.", "Leaf nodes are linked."]
        assert _chunk_fingerprint(chunks) == _chunk_fingerprint(chunks)

    def test_chunk_fingerprint_different_chunks(self):
        chunks_a = ["B-trees store sorted data."]
        chunks_b = ["Hashing distributes data."]
        assert _chunk_fingerprint(chunks_a) != _chunk_fingerprint(chunks_b)

    def test_chunk_fingerprint_tuple_chunks(self):
        # chunks can be (text, score) tuples from the retriever
        chunks = [("B-trees store sorted data.", 0.9)]
        plain  = ["B-trees store sorted data."]
        assert _chunk_fingerprint(chunks) == _chunk_fingerprint(plain)


# ===========================================================================
# Exact strategy
# ===========================================================================

class TestExactCache:

    def setup_method(self):
        self.cache = PromptCache(strategy="exact")

    def test_basic_hit(self):
        self.cache.put("What is a B-tree?", "B-trees are balanced trees.")
        assert self.cache.get("What is a B-tree?") == "B-trees are balanced trees."

    def test_case_insensitive(self):
        self.cache.put("What is a B-tree?", "B-trees are balanced trees.")
        assert self.cache.get("what is a b-tree?") is not None

    def test_whitespace_insensitive(self):
        self.cache.put("What is a B-tree?", "B-trees are balanced trees.")
        assert self.cache.get("  What  is  a  B-tree?  ") is not None

    def test_miss_different_query(self):
        self.cache.put("What is a B-tree?", "B-trees are balanced trees.")
        assert self.cache.get("What is hashing?") is None

    def test_overwrite_existing(self):
        self.cache.put("What is a B-tree?", "First answer.")
        self.cache.put("What is a B-tree?", "Updated answer.")
        assert self.cache.get("What is a B-tree?") == "Updated answer."

    def test_stats_hits_misses(self):
        self.cache.put("What is a B-tree?", "answer", llm_latency_s=10.0)
        self.cache.get("What is a B-tree?")   # hit
        self.cache.get("What is hashing?")    # miss
        s = self.cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["latency_saved_s"] == 10.0

    def test_lru_eviction(self):
        cache = PromptCache(strategy="exact", max_size=2)
        cache.put("q1", "a1")
        cache.put("q2", "a2")
        cache.put("q3", "a3")   # evicts q1 (oldest)
        assert cache.get("q1") is None
        assert cache.get("q2") is not None
        assert cache.get("q3") is not None

    def test_lru_hit_refreshes_order(self):
        cache = PromptCache(strategy="exact", max_size=2)
        cache.put("q1", "a1")
        cache.put("q2", "a2")
        cache.get("q1")         # touch q1 → now q2 is oldest
        cache.put("q3", "a3")   # should evict q2, not q1
        assert cache.get("q1") is not None
        assert cache.get("q2") is None

    def test_clear_resets_everything(self):
        self.cache.put("What is a B-tree?", "answer", llm_latency_s=5.0)
        self.cache.get("What is a B-tree?")
        self.cache.clear()
        assert self.cache.get("What is a B-tree?") is None
        s = self.cache.stats()
        assert s["hits"] == 0
        assert s["misses"] == 1   # the get after clear counts
        assert s["size"] == 0


# ===========================================================================
# Semantic strategy
# ===========================================================================

class TestSemanticCache:

    def setup_method(self):
        self.cache = PromptCache(
            strategy="semantic",
            threshold=0.90,
            embed_fn=BTREE_EMBED,
        )

    def test_exact_query_hit(self):
        self.cache.put("What is a B-tree?", "B-trees are balanced.")
        assert self.cache.get("What is a B-tree?") is not None

    def test_similar_query_hit(self):
        # "explain b-tree" and "what is a b-tree" map to same vector
        self.cache.put("What is a B-tree?", "B-trees are balanced.")
        assert self.cache.get("explain b-tree structure") is not None

    def test_different_query_miss(self):
        self.cache.put("What is a B-tree?", "B-trees are balanced.")
        assert self.cache.get("What is hashing?") is None

    def test_below_threshold_miss(self):
        cache = PromptCache(strategy="semantic", threshold=0.99, embed_fn=BTREE_EMBED)
        # store a b-tree entry, query with slight variation that maps to same vector
        # but test a truly different domain query that won't reach 0.99
        cache.put("What is a B-tree?", "B-trees are balanced.")
        assert cache.get("What is hashing?") is None

    def test_no_embed_fn_always_miss(self):
        cache = PromptCache(strategy="semantic", threshold=0.90, embed_fn=None)
        cache.put("What is a B-tree?", "answer")
        assert cache.get("What is a B-tree?") is None

    def test_stats_latency_saved(self):
        self.cache.put("What is a B-tree?", "answer", llm_latency_s=8.0)
        self.cache.get("What is a B-tree?")   # hit
        self.cache.get("What is hashing?")    # miss
        s = self.cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["latency_saved_s"] == 8.0

    def test_empty_cache_miss(self):
        assert self.cache.get("What is a B-tree?") is None


# ===========================================================================
# Context-aware strategy
# ===========================================================================

class TestContextCache:

    def setup_method(self):
        self.cache = PromptCache(
            strategy="context",
            threshold=0.90,
            embed_fn=BTREE_EMBED,
        )
        self.chunks_a = ["B-trees store keys in sorted order across pages."]
        self.chunks_b = ["Hashing distributes records into buckets via a hash function."]

    def test_hit_same_query_same_chunks(self):
        self.cache.put("What is a B-tree?", "answer A", chunks=self.chunks_a)
        assert self.cache.get("What is a B-tree?", self.chunks_a) is not None

    def test_miss_same_query_different_chunks(self):
        self.cache.put("What is a B-tree?", "answer A", chunks=self.chunks_a)
        # same query, different retrieved context → should miss
        assert self.cache.get("What is a B-tree?", self.chunks_b) is None

    def test_miss_different_query_same_chunks(self):
        self.cache.put("What is a B-tree?", "answer A", chunks=self.chunks_a)
        assert self.cache.get("What is hashing?", self.chunks_a) is None

    def test_similar_query_same_chunks_hit(self):
        self.cache.put("What is a B-tree?", "answer A", chunks=self.chunks_a)
        # "explain b-tree" maps to same embedding → hit if chunks match
        assert self.cache.get("explain b-tree structure", self.chunks_a) is not None

    def test_similar_query_different_chunks_miss(self):
        self.cache.put("What is a B-tree?", "answer A", chunks=self.chunks_a)
        # similar query but different context → must miss
        assert self.cache.get("explain b-tree structure", self.chunks_b) is None

    def test_no_chunks_treated_as_empty_fp(self):
        self.cache.put("What is a B-tree?", "answer", chunks=None)
        # both stored and queried with no chunks → same fingerprint (empty)
        assert self.cache.get("What is a B-tree?", None) is not None

    def test_stats(self):
        self.cache.put("What is a B-tree?", "answer", chunks=self.chunks_a, llm_latency_s=9.0)
        self.cache.get("What is a B-tree?", self.chunks_a)   # hit
        self.cache.get("What is a B-tree?", self.chunks_b)   # miss (different chunks)
        s = self.cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["latency_saved_s"] == 9.0
