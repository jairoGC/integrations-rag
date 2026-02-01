"""
Yuno Integrations RAG - Precision Delta Evaluation

Compares retrieval precision with and without metadata filters.
Measures the improvement in precision when using targeted filters
for provider-specific or payment-method-specific questions.

Precision Delta = Filtered Precision - Unfiltered Precision

Target: Filters should improve precision by ≥20% for targeted queries
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from retrieval import retrieve_with_filter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEFAULT_K = 5
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge for payment integration documentation.
Determine if the retrieved document is relevant to answering the given question.

A document is RELEVANT if it contains information that would help answer the question about:
- Payment integration implementation
- Provider configuration
- Error handling
- Payment methods
- Country-specific requirements
- Testing procedures
- Incident resolution

A document is NOT RELEVANT if it contains no useful information for the question.

Question: {question}

Retrieved Document:
{document}

Is this document relevant? Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""


# Test cases comparing filtered vs unfiltered retrieval
PRECISION_DELTA_TEST_CASES = [
    {
        "id": "provider_fintoc",
        "question": "How do I configure Fintoc webhooks?",
        "filter_type": "provider",
        "filters": {"providers": ["fintoc"]},
        "description": "Fintoc-specific question with provider filter"
    },
    {
        "id": "provider_stripe",
        "question": "What error codes does Stripe return for declined cards?",
        "filter_type": "provider",
        "filters": {"providers": ["stripe"]},
        "description": "Stripe error codes with provider filter"
    },
    {
        "id": "method_pix",
        "question": "How to implement PIX payments?",
        "filter_type": "payment_method",
        "filters": {"payment_methods": ["PIX"]},
        "description": "PIX payment method with filter"
    },
    {
        "id": "method_pse",
        "question": "How to test PSE payments in sandbox?",
        "filter_type": "payment_method",
        "filters": {"payment_methods": ["PSE"]},
        "description": "PSE testing with method filter"
    },
    {
        "id": "country_chile",
        "question": "Payment integration requirements for Chile?",
        "filter_type": "country",
        "filters": {"countries": ["CL"]},
        "description": "Chile-specific integration with country filter"
    },
    {
        "id": "country_brazil",
        "question": "Payment methods available in Brazil?",
        "filter_type": "country",
        "filters": {"countries": ["BR"]},
        "description": "Brazil payment methods with country filter"
    },
    {
        "id": "severity_critical",
        "question": "Recent critical payment incidents?",
        "filter_type": "severity",
        "filters": {"document_source": "jira", "severity": ["S1", "S2"]},
        "description": "Critical incidents with severity filter"
    },
    {
        "id": "error_codes",
        "question": "What error codes should we handle?",
        "filter_type": "content",
        "filters": {"has_error_codes": True},
        "description": "Error handling with content filter"
    },
    {
        "id": "testing_instructions",
        "question": "How to test payment flows?",
        "filter_type": "content",
        "filters": {"has_testing_instructions": True},
        "description": "Testing documentation with content filter"
    },
    {
        "id": "combined_provider_method",
        "question": "Stripe card payment implementation?",
        "filter_type": "combined",
        "filters": {"providers": ["stripe"], "payment_methods": ["CARD"]},
        "description": "Combined provider + method filter"
    },
    {
        "id": "combined_provider_country",
        "question": "Fintoc integration for Chile?",
        "filter_type": "combined",
        "filters": {"providers": ["fintoc"], "countries": ["CL"]},
        "description": "Combined provider + country filter"
    },
    {
        "id": "combined_complex",
        "question": "Critical Izipay incidents in Peru with error codes?",
        "filter_type": "combined",
        "filters": {
            "providers": ["izipay"],
            "countries": ["PE"],
            "document_source": "jira",
            "severity": ["S1", "S2"],
            "has_error_codes": True
        },
        "description": "Complex multi-filter combination"
    },
]


def create_relevance_judge():
    """Create the LLM judge for assessing relevance."""
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)

    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def judge_relevance(judge_chain, question: str, document_content: str) -> bool:
    """Use LLM to judge if a document is relevant to the question."""
    try:
        response = judge_chain.invoke({
            "question": question,
            "document": document_content
        })

        return "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()
    except Exception as e:
        print(f"   ⚠️ Judge error: {e}")
        return False


def calculate_precision(question: str, documents: List, judge) -> tuple[float, List[bool]]:
    """
    Calculate precision for a set of retrieved documents.

    Returns:
        Tuple of (precision_score, list_of_relevance_judgments)
    """
    if not documents:
        return 0.0, []

    judgments = []
    for doc in documents:
        is_relevant = judge_relevance(judge, question, doc.page_content)
        judgments.append(is_relevant)

    precision = sum(judgments) / len(judgments)
    return precision, judgments


def evaluate_precision_delta(
    question: str,
    filters: dict,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Compare precision with and without filters for a single question.

    Returns:
        Dictionary with precision comparison results
    """
    judge = create_relevance_judge()

    # Retrieve WITHOUT filters (baseline)
    try:
        unfiltered_docs = retrieve_with_filter(
            query=question,
            top_k=k,
            verbose=False
        )
    except Exception as e:
        print(f"   ❌ Unfiltered retrieval error: {e}")
        unfiltered_docs = []

    unfiltered_precision, unfiltered_judgments = calculate_precision(
        question, unfiltered_docs, judge
    )

    # Retrieve WITH filters
    try:
        filtered_docs = retrieve_with_filter(
            query=question,
            top_k=k,
            **filters,
            verbose=False
        )
    except Exception as e:
        print(f"   ❌ Filtered retrieval error: {e}")
        filtered_docs = []

    filtered_precision, filtered_judgments = calculate_precision(
        question, filtered_docs, judge
    )

    # Calculate delta
    precision_delta = filtered_precision - unfiltered_precision

    result = {
        "question": question,
        "filters": filters,
        "k": k,
        "unfiltered": {
            "precision": unfiltered_precision,
            "relevant_count": sum(unfiltered_judgments),
            "total_docs": len(unfiltered_docs),
            "judgments": unfiltered_judgments
        },
        "filtered": {
            "precision": filtered_precision,
            "relevant_count": sum(filtered_judgments),
            "total_docs": len(filtered_docs),
            "judgments": filtered_judgments
        },
        "precision_delta": precision_delta,
        "improvement": precision_delta > 0
    }

    if verbose:
        # Extract identifier from each document
        unfiltered_ids = []
        for doc in unfiltered_docs:
            ticket_id = doc.metadata.get("ticket_id", "")
            page_id = doc.metadata.get("page_id", "")
            identifier = ticket_id or f"Page-{page_id}" or "Unknown"
            unfiltered_ids.append(identifier)

        filtered_ids = []
        for doc in filtered_docs:
            ticket_id = doc.metadata.get("ticket_id", "")
            page_id = doc.metadata.get("page_id", "")
            identifier = ticket_id or f"Page-{page_id}" or "Unknown"
            filtered_ids.append(identifier)

        result["unfiltered_sources"] = unfiltered_ids
        result["filtered_sources"] = filtered_ids

    return result


def run_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Run precision delta evaluation on all test cases.

    Returns:
        Dictionary with aggregate results
    """
    if test_cases is None:
        test_cases = PRECISION_DELTA_TEST_CASES

    print("=" * 70)
    print("Yuno Integrations RAG - Precision Delta Evaluation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • k (documents): {k}")
    print(f"  • Judge model: {JUDGE_MODEL}")
    print(f"  • Test cases: {len(test_cases)}")
    print(f"  • Target: ≥20% improvement with filters")
    print("\n" + "-" * 70)

    results = []
    improvements = 0
    no_change = 0
    regressions = 0

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        description = test_case["description"]
        filter_type = test_case["filter_type"]

        print(f"\n[{i}/{len(test_cases)}] 📝 {test_id}")
        print(f"    Type: {filter_type}")
        print(f"    Q: {question}")

        result = evaluate_precision_delta(
            question=question,
            filters=filters,
            k=k,
            verbose=verbose
        )
        result["test_id"] = test_id
        result["description"] = description
        result["filter_type"] = filter_type

        results.append(result)

        # Track outcomes
        delta = result["precision_delta"]
        if delta > 0:
            improvements += 1
            status = f"🟢 +{delta:.0%}"
        elif delta < 0:
            regressions += 1
            status = f"🔴 {delta:.0%}"
        else:
            no_change += 1
            status = "⚪ 0%"

        print(f"    Unfiltered: {result['unfiltered']['precision']:.0%} ({result['unfiltered']['relevant_count']}/{result['unfiltered']['total_docs']})")
        print(f"    Filtered:   {result['filtered']['precision']:.0%} ({result['filtered']['relevant_count']}/{result['filtered']['total_docs']})")
        print(f"    Delta:      {status}")

    # Calculate aggregate metrics
    avg_unfiltered = sum(r["unfiltered"]["precision"] for r in results) / len(results)
    avg_filtered = sum(r["filtered"]["precision"] for r in results) / len(results)
    avg_delta = sum(r["precision_delta"] for r in results) / len(results)

    # Group by filter type
    by_filter_type = {}
    for r in results:
        ft = r["filter_type"]
        if ft not in by_filter_type:
            by_filter_type[ft] = []
        by_filter_type[ft].append(r["precision_delta"])

    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "avg_unfiltered_precision": avg_unfiltered,
        "avg_filtered_precision": avg_filtered,
        "avg_precision_delta": avg_delta,
        "improvements": improvements,
        "no_change": no_change,
        "regressions": regressions,
        "by_filter_type": {
            ft: sum(deltas) / len(deltas)
            for ft, deltas in by_filter_type.items()
        },
        "results": results
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(f"\n📊 Aggregate Metrics:")
    print(f"   Average Unfiltered Precision: {avg_unfiltered:.2%}")
    print(f"   Average Filtered Precision:   {avg_filtered:.2%}")
    print(f"   Average Precision Delta:      {avg_delta:+.2%}")

    print(f"\n📈 Outcomes:")
    print(f"   🟢 Improvements: {improvements}/{len(results)} ({100*improvements/len(results):.0f}%)")
    print(f"   ⚪ No Change:    {no_change}/{len(results)} ({100*no_change/len(results):.0f}%)")
    print(f"   🔴 Regressions:  {regressions}/{len(results)} ({100*regressions/len(results):.0f}%)")

    print(f"\n📋 By Filter Type:")
    for ft, avg in summary["by_filter_type"].items():
        indicator = "🟢" if avg > 0 else "🔴" if avg < 0 else "⚪"
        print(f"   {indicator} {ft}: {avg:+.2%} avg delta")

    print(f"\n📋 Per-Query Breakdown:")
    for r in results:
        delta = r["precision_delta"]
        if delta > 0:
            indicator = "🟢"
        elif delta < 0:
            indicator = "🔴"
        else:
            indicator = "⚪"
        print(f"   {indicator} {r['test_id']}: {r['unfiltered']['precision']:.0%} → {r['filtered']['precision']:.0%} ({delta:+.0%})")

    # Check if target met
    target_met = avg_delta >= 0.20
    target_icon = "✅" if target_met else "⚠️"
    print(f"\n{target_icon} Target (≥20% improvement): {'MET' if target_met else 'NOT MET'}")

    return summary


def main():
    """Run the precision delta evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run precision delta evaluation on Yuno Integrations RAG")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include document sources in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    summary = run_evaluation(k=args.k, verbose=args.verbose)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
