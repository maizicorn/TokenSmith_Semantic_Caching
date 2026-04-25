import json
import time
import pytest
from pathlib import Path
from datetime import datetime
from prompt_cache import PromptCache
  
 
# ---------------------------------------------------------------------------
# Main benchmark test
# ---------------------------------------------------------------------------
 
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cache_benchmarks(benchmarks, config, cache_results_dir):
    """
    Evaluate all three prompt caching strategies against the benchmark set.
 
    For each strategy:
      - Pass 1 (cold): queries run through the real TokenSmith pipeline,
        populating the cache and recording LLM latency per query.
      - Pass 2 (warm): same queries replayed; hits are served from cache
        and measured for latency.
 
    Reports hit rate, latency saved, and answer accuracy (semantic similarity
    vs. expected answer) for each strategy.
    """
    from tests.test_benchmarks import get_tokensmith_answer, clean_answer
 
    strategies = _build_strategies(config)
 
    all_results = {}
 
    for strategy_name, cache in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Cache Strategy: {strategy_name.upper()}")
        print(f"{'='*60}")
 
        strategy_results = _run_strategy(
            strategy_name=strategy_name,
            cache=cache,
            benchmarks=benchmarks,
            config=config,
            get_answer_fn=get_tokensmith_answer,
        )
 
        all_results[strategy_name] = strategy_results
        _print_strategy_summary(strategy_name, strategy_results)
 
    _print_comparison_table(all_results)
 
 
# ---------------------------------------------------------------------------
# Strategy builder
# ---------------------------------------------------------------------------
 
def _build_strategies(config):
    """
    Build one PromptCache per strategy.
    embed_fn is taken from the retriever if available in config.
    """
    embed_fn = _get_embed_fn(config)
 
    return {
        "exact": PromptCache(
            strategy="exact",
            max_size=512,
        ),
        "semantic": PromptCache(
            strategy="semantic",
            threshold=config.get("cache_threshold", 0.90),
            embed_fn=embed_fn,
            max_size=512,
        ),
        "context": PromptCache(
            strategy="context",
            threshold=config.get("cache_threshold", 0.90),
            embed_fn=embed_fn,
            max_size=512,
        ),
    }
 
 
def _get_embed_fn(config):
    """Load the embed function using the same CachedEmbedder used by the retriever."""
    try:
        from src.embedder import CachedEmbedder
        embedder = CachedEmbedder(config.get("embed_model"))
        # encode() expects a list and returns a 2D array; wrap to match embed_fn(str) -> list[float]
        return lambda text: embedder.encode([text])[0].tolist()
    except Exception as e:
        print(f"  ⚠️  Could not load embed_fn ({e}); semantic/context will always miss.")
        return None
 
 
# ---------------------------------------------------------------------------
# Per-strategy runner
# ---------------------------------------------------------------------------
 
def _run_strategy(strategy_name, cache, benchmarks, config, get_answer_fn):
    """
    Run two passes over all benchmarks for a single cache strategy.
    Returns a list of per-benchmark result dicts.
    """
    results = []
 
    # ---- Pass 1: cold cache, run LLM, populate cache ----
    print(f"\n  Pass 1 (cold cache) — running LLM for all {len(benchmarks)} benchmarks...")
    cold_answers  = {}   # benchmark_id → answer
    cold_chunks   = {}   # benchmark_id → chunks_info
    cold_latencies = {}  # benchmark_id → seconds
 
    for benchmark in benchmarks:
        bid      = benchmark.get("id", "unknown")
        question = benchmark["question"]
        golden   = benchmark.get("golden_chunks") if config.get("use_golden_chunks") else None
 
        t0 = time.time()
        try:
            answer, chunks_info, _ = get_answer_fn(
                question=question,
                config=config,
                golden_chunks=golden,
            )
        except Exception as e:
            print(f"    ❌ {bid}: error — {e}")
            answer, chunks_info = "", []
        latency = time.time() - t0
 
        cold_answers[bid]   = answer
        cold_chunks[bid]    = chunks_info or []
        cold_latencies[bid] = latency
 
        # Store in cache
        cache.put(question, answer, chunks=chunks_info, llm_latency_s=latency)
        print(f"    {bid}: LLM latency {latency:.1f}s → cached")
 
    # ---- Pass 2: warm cache, serve from cache ----
    print(f"\n  Pass 2 (warm cache) — replaying queries...")
    cache_pass_stats = {"hits": 0, "misses": 0, "latency_saved": 0.0}
 
    for benchmark in benchmarks:
        bid      = benchmark.get("id", "unknown")
        question = benchmark["question"]
        chunks   = cold_chunks[bid]
 
        t0 = time.time()
        cached = cache.get(question, chunks)
        warm_latency = time.time() - t0
 
        hit = cached is not None
        if hit:
            cache_pass_stats["hits"] += 1
            cache_pass_stats["latency_saved"] += cold_latencies[bid]
            status = "✅ HIT"
        else:
            cache_pass_stats["misses"] += 1
            status = "❌ MISS"
 
        print(f"    {bid}: {status}  (lookup {warm_latency*1000:.1f}ms)")
 
        # Accuracy: compare cached answer vs expected (simple overlap check)
        expected = benchmark.get("expected_answer", "")
        retrieved_answer = cached if cached else cold_answers[bid]
        accuracy = _simple_accuracy(retrieved_answer, expected)
 
        results.append({
            "benchmark_id":    bid,
            "question":        question,
            "expected_answer": expected,
            "cold_answer":     cold_answers[bid],
            "cached_answer":   cached,
            "cache_hit":       hit,
            "cold_latency_s":  round(cold_latencies[bid], 3),
            "warm_latency_s":  round(warm_latency, 6),
            "latency_saved_s": round(cold_latencies[bid], 3) if hit else 0.0,
            "accuracy":        round(accuracy, 3),
            "strategy":        strategy_name,
            "timestamp":       datetime.now().isoformat(),
        })
 
    return results
 
 
# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------
 
def _simple_accuracy(answer: str, expected: str) -> float:
    """
    Word overlap accuracy — fraction of expected words found in the answer.
    Used when the full scorer is not available in this test context.
    """
    if not answer or not expected:
        return 0.0
    answer_words   = set(answer.lower().split())
    expected_words = set(expected.lower().split())
    if not expected_words:
        return 0.0
    return len(answer_words & expected_words) / len(expected_words)
 
 
# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
 
def _print_strategy_summary(strategy_name, results):
    total    = len(results)
    hits     = sum(1 for r in results if r["cache_hit"])
    hit_rate = hits / total if total else 0.0
    saved    = sum(r["latency_saved_s"] for r in results)
    avg_acc  = sum(r["accuracy"] for r in results) / total if total else 0.0
 
    print(f"\n  {'─'*58}")
    print(f"  Strategy:        {strategy_name.upper()}")
    print(f"  Hit rate:        {hits}/{total} ({hit_rate:.0%})")
    print(f"  Latency saved:   {saved:.1f}s total")
    print(f"  Avg accuracy:    {avg_acc:.3f}")
    print(f"  {'─'*58}")
 
 
def _print_comparison_table(all_results):
    print(f"\n{'='*60}")
    print(f"  CACHE STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Strategy':<12} {'Hit Rate':>10} {'Saved (s)':>12} {'Avg Accuracy':>14}")
    print(f"  {'─'*52}")
 
    for strategy_name, results in all_results.items():
        total    = len(results)
        hits     = sum(1 for r in results if r["cache_hit"])
        hit_rate = hits / total if total else 0.0
        saved    = sum(r["latency_saved_s"] for r in results)
        avg_acc  = sum(r["accuracy"] for r in results) / total if total else 0.0
        print(f"  {strategy_name:<12} {hit_rate:>9.0%} {saved:>11.1f}s {avg_acc:>14.3f}")
 
    print(f"{'='*60}\n")