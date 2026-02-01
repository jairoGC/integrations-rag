"""
Test Hybrid Retrieval Performance

Compares standard filtered retrieval vs hybrid retrieval on problematic queries.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retrieval import retrieve_with_filter, retrieve_hybrid


# Test cases that showed regressions with standard filtering
TEST_CASES = [
    {
        "id": "stripe_errors",
        "query": "What error codes does Stripe return for declined cards?",
        "filters": {"providers": ["stripe"], "has_error_codes": True},
        "description": "Stripe error codes (was 100% → 60%)"
    },
    {
        "id": "combined_stripe_card",
        "query": "Stripe card payment implementation?",
        "filters": {"providers": ["stripe"], "payment_methods": ["CARD"]},
        "description": "Stripe + CARD (was 80% → 40%)"
    },
    {
        "id": "combined_complex",
        "query": "Critical Izipay incidents in Peru with error codes?",
        "filters": {
            "providers": ["izipay"],
            "countries": ["PE"],
            "document_source": "jira",
            "severity": ["S1", "S2"],
            "has_error_codes": True
        },
        "description": "Complex multi-filter (was 100% → 0%)"
    },
]


def test_retrieval_comparison():
    """Compare standard vs hybrid retrieval on problematic queries."""
    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)
    print("\nComparing standard filtering vs hybrid retrieval...\n")

    for test_case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"Query: {test_case['query']}")
        print(f"Description: {test_case['description']}")
        print(f"Filters: {test_case['filters']}")
        print(f"{'='*70}\n")

        # Standard filtered retrieval
        print("📋 Standard Filtered Retrieval:")
        try:
            standard_docs = retrieve_with_filter(
                query=test_case['query'],
                top_k=5,
                **test_case['filters'],
                verbose=False
            )
            print(f"   Retrieved: {len(standard_docs)} documents")
            for i, doc in enumerate(standard_docs[:3], 1):
                doc_type = doc.metadata.get("document_type", "Unknown")
                identifier = doc.metadata.get("ticket_id") or doc.metadata.get("page_id", "Unknown")
                providers = ', '.join(doc.metadata.get("providers", [])[:2])
                print(f"   [{i}] {identifier} ({doc_type}, Providers: {providers})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            standard_docs = []

        # Hybrid retrieval
        print(f"\n🔀 Hybrid Retrieval (60% filtered + 40% unfiltered):")
        try:
            hybrid_docs = retrieve_hybrid(
                query=test_case['query'],
                top_k=5,
                filtered_ratio=0.6,
                **test_case['filters'],
                verbose=True
            )
            print(f"   Retrieved: {len(hybrid_docs)} documents")
            for i, doc in enumerate(hybrid_docs[:3], 1):
                doc_type = doc.metadata.get("document_type", "Unknown")
                identifier = doc.metadata.get("ticket_id") or doc.metadata.get("page_id", "Unknown")
                providers = ', '.join(doc.metadata.get("providers", [])[:2])
                print(f"   [{i}] {identifier} ({doc_type}, Providers: {providers})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            hybrid_docs = []

        # Comparison
        print(f"\n📊 Comparison:")
        print(f"   Standard: {len(standard_docs)} documents")
        print(f"   Hybrid: {len(hybrid_docs)} documents")

        if len(hybrid_docs) > len(standard_docs):
            improvement = len(hybrid_docs) - len(standard_docs)
            print(f"   ✅ Hybrid retrieved {improvement} more documents")
        elif len(hybrid_docs) == len(standard_docs):
            print(f"   ⚪ Same number of documents")
        else:
            print(f"   ⚠️  Hybrid retrieved fewer documents")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nHybrid retrieval combines:")
    print("  • 60% filtered results (high specificity)")
    print("  • 40% unfiltered results (high recall)")
    print("  • Deduplication to prevent duplicates")
    print("\nThis ensures you get targeted docs while not missing relevant content!")
    print("=" * 70)


if __name__ == "__main__":
    test_retrieval_comparison()
