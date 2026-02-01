"""
Yuno Integrations RAG - Latency Evaluation
Compares retrieval and generation latency with different filter configurations.

Measures:
- Retrieval latency (vector search time)
- Generation latency (LLM response time)
- Total end-to-end latency
- Comparison: filtered vs unfiltered

Target: <2 seconds end-to-end latency for 95th percentile
"""

import sys
from pathlib import Path
import time
import statistics
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from retrieval import retrieve_with_filter
from generation import generate_answer

# Evaluation configuration
DEFAULT_K = 5
DEFAULT_RUNS = 3  # Number of runs per test for averaging

# Test cases with different filter configurations
LATENCY_TEST_CASES = [
    {
        "id": "no_filter",
        "question": "How to configure payment webhooks?",
        "filters": {},
        "description": "No filters (baseline)"
    },
    {
        "id": "provider_filter",
        "question": "How to configure Fintoc webhooks?",
        "filters": {"providers": ["fintoc"]},
        "description": "Single provider filter"
    },
    {
        "id": "method_filter",
        "question": "How to implement PIX payments?",
        "filters": {"payment_methods": ["PIX"]},
        "description": "Payment method filter"
    },
    {
        "id": "country_filter",
        "question": "Payment integration for Chile?",
        "filters": {"countries": ["CL"]},
        "description": "Country filter"
    },
    {
        "id": "source_filter",
        "question": "Recent payment incidents",
        "filters": {"document_source": "jira"},
        "description": "Document source filter (Jira)"
    },
    {
        "id": "severity_filter",
        "question": "Critical payment failures",
        "filters": {"document_source": "jira", "severity": "S1"},
        "description": "Severity filter (S1)"
    },
    {
        "id": "error_codes_filter",
        "question": "What error codes are returned?",
        "filters": {"has_error_codes": True},
        "description": "Has error codes filter"
    },
    {
        "id": "combined_simple",
        "question": "Stripe card payments for Chile",
        "filters": {
            "providers": ["stripe"],
            "payment_methods": ["CARD"],
            "countries": ["CL"]
        },
        "description": "Combined filters (3 fields)"
    },
    {
        "id": "combined_complex",
        "question": "Critical Fintoc incidents in Chile with error codes",
        "filters": {
            "providers": ["fintoc"],
            "countries": ["CL"],
            "document_source": "jira",
            "severity": ["S1", "S2"],
            "has_error_codes": True
        },
        "description": "Combined filters (5 fields)"
    },
]


def measure_retrieval_latency(
    query: str,
    k: int = DEFAULT_K,
    filters: dict = None
) -> Tuple[List, float]:
    """
    Measure retrieval latency and return documents.

    Returns:
        Tuple of (documents, latency_ms)
    """
    start_time = time.perf_counter()

    documents = retrieve_with_filter(
        query=query,
        top_k=k,
        **(filters if filters else {}),
        verbose=False
    )

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return documents, latency_ms


def measure_generation_latency(
    question: str,
    filters: dict = None,
    k: int = DEFAULT_K
) -> Tuple[Dict, float]:
    """
    Measure generation latency and return answer.

    Returns:
        Tuple of (result_dict, latency_ms)
    """
    start_time = time.perf_counter()

    result = generate_answer(
        question=question,
        top_k=k,
        **(filters if filters else {}),
        verbose=False
    )

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    return result, latency_ms


def run_single_test(
    question: str,
    filters: dict,
    k: int = DEFAULT_K,
    include_generation: bool = True
) -> dict:
    """
    Run a single latency test.

    Returns:
        Dictionary with latency measurements
    """
    # Measure retrieval only
    documents, retrieval_latency = measure_retrieval_latency(
        query=question,
        k=k,
        filters=filters if filters else {}
    )

    result = {
        "retrieval_latency_ms": retrieval_latency,
        "documents_retrieved": len(documents),
    }

    # Measure full generation if requested
    if include_generation:
        _, generation_latency = measure_generation_latency(
            question=question,
            filters=filters if filters else {},
            k=k
        )

        result["generation_latency_ms"] = generation_latency
        result["total_latency_ms"] = generation_latency  # Generation includes retrieval
    else:
        result["generation_latency_ms"] = 0
        result["total_latency_ms"] = retrieval_latency

    return result


def run_latency_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    num_runs: int = DEFAULT_RUNS,
    include_generation: bool = True
) -> dict:
    """
    Run latency evaluation across all test cases.

    Args:
        test_cases: List of test cases to run
        k: Number of documents to retrieve
        num_runs: Number of runs per test for averaging
        include_generation: Whether to measure generation latency

    Returns:
        Dictionary with evaluation results
    """
    if test_cases is None:
        test_cases = LATENCY_TEST_CASES

    print("=" * 70)
    print("Yuno Integrations RAG - Latency Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • k (documents): {k}")
    print(f"  • Runs per test: {num_runs}")
    print(f"  • Include generation: {include_generation}")
    print(f"  • Test cases: {len(test_cases)}")
    print(f"  • Target: <2000ms end-to-end")
    print("\n" + "-" * 70)

    results = []

    for test_case in test_cases:
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]

        print(f"\n📝 {test_id}: {description}")

        # Run multiple times and collect measurements
        retrieval_times = []
        generation_times = []
        total_times = []
        docs_retrieved = []

        for run in range(num_runs):
            try:
                run_result = run_single_test(
                    question=question,
                    filters=filters,
                    k=k,
                    include_generation=include_generation
                )

                retrieval_times.append(run_result["retrieval_latency_ms"])
                generation_times.append(run_result["generation_latency_ms"])
                total_times.append(run_result["total_latency_ms"])
                docs_retrieved.append(run_result["documents_retrieved"])
            except Exception as e:
                print(f"   ❌ Error on run {run + 1}: {e}")
                continue

        if not retrieval_times:
            print(f"   ❌ All runs failed")
            continue

        # Calculate statistics
        result = {
            "test_id": test_id,
            "description": description,
            "filters": filters,
            "num_runs": len(retrieval_times),
            "documents_retrieved": docs_retrieved[0] if docs_retrieved else 0,
            "retrieval": {
                "mean_ms": statistics.mean(retrieval_times),
                "median_ms": statistics.median(retrieval_times),
                "min_ms": min(retrieval_times),
                "max_ms": max(retrieval_times),
                "stdev_ms": statistics.stdev(retrieval_times) if len(retrieval_times) > 1 else 0
            },
            "generation": {
                "mean_ms": statistics.mean(generation_times),
                "median_ms": statistics.median(generation_times),
                "min_ms": min(generation_times),
                "max_ms": max(generation_times),
            },
            "total": {
                "mean_ms": statistics.mean(total_times),
                "median_ms": statistics.median(total_times),
                "min_ms": min(total_times),
                "max_ms": max(total_times),
            }
        }

        results.append(result)

        # Print summary for this test
        print(f"   Retrieval: {result['retrieval']['mean_ms']:.0f}ms (±{result['retrieval']['stdev_ms']:.0f}ms)")
        if include_generation:
            print(f"   Generation: {result['generation']['mean_ms']:.0f}ms")
            print(f"   Total: {result['total']['mean_ms']:.0f}ms", end="")
            if result['total']['mean_ms'] < 2000:
                print(" ✅")
            else:
                print(" ⚠️ (>2s)")
        print(f"   Docs retrieved: {result['documents_retrieved']}")

    # Calculate summary statistics
    baseline = next((r for r in results if r["test_id"] == "no_filter"), results[0] if results else None)

    if baseline:
        baseline_retrieval = baseline["retrieval"]["mean_ms"]

        # Print comparison
        print("\n" + "=" * 70)
        print("LATENCY COMPARISON")
        print("=" * 70)
        print(f"\n{'Test Case':<30} {'Retrieval':<12} {'vs Baseline':<18} {'Docs':<8}")
        print("-" * 70)

        for r in results:
            retrieval = r["retrieval"]["mean_ms"]
            diff = retrieval - baseline_retrieval
            diff_pct = (diff / baseline_retrieval * 100) if baseline_retrieval > 0 else 0

            if diff < -50:
                diff_str = f"🟢 {diff:+.0f}ms ({diff_pct:+.0f}%)"
            elif diff > 50:
                diff_str = f"🔴 {diff:+.0f}ms ({diff_pct:+.0f}%)"
            elif r["test_id"] == "no_filter":
                diff_str = "baseline"
            else:
                diff_str = f"{diff:+.0f}ms ({diff_pct:+.0f}%)"

            print(f"{r['test_id']:<30} {retrieval:<12.0f} {diff_str:<18} {r['documents_retrieved']:<8}")

        if include_generation:
            print(f"\n📊 End-to-End Latency Summary:")
            total_times_list = [r['total']['mean_ms'] for r in results]
            avg_total = statistics.mean(total_times_list)
            max_total = max(total_times_list)

            print(f"   Average: {avg_total:.0f}ms")
            print(f"   Max: {max_total:.0f}ms")

            under_2s = sum(1 for t in total_times_list if t < 2000)
            print(f"   Under 2s: {under_2s}/{len(results)} ({under_2s/len(results)*100:.0f}%)")

            if avg_total < 2000:
                print(f"   ✅ Target met: Average latency under 2 seconds")
            else:
                print(f"   ⚠️ Target not met: Average latency over 2 seconds")

    summary = {
        "k": k,
        "num_runs": num_runs,
        "baseline_retrieval_ms": baseline["retrieval"]["mean_ms"] if baseline else None,
        "results": results
    }

    return summary


def main():
    """Run the latency evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run latency evaluation on Yuno Integrations RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-n", "--num-runs", type=int, default=DEFAULT_RUNS, help=f"Number of runs per test (default: {DEFAULT_RUNS})")
    parser.add_argument("--no-generation", action="store_true", help="Skip generation latency measurement")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    summary = run_latency_evaluation(
        k=args.k,
        num_runs=args.num_runs,
        include_generation=not args.no_generation
    )

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
