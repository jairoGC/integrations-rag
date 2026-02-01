"""
Integration Precision Evaluation

Measures precision@5 for Yuno integration-specific queries to validate
that the RAG system retrieves relevant documentation.

Precision@K = (Number of relevant documents in top K) / K

Target: >80% precision@5 across all test queries
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from retrieval import retrieve_with_filter
from typing import List, Dict, Set


# Test queries with ground truth (expected relevant document sources)
TEST_QUERIES = [
    {
        "query": "How do I configure Fintoc webhooks for Chile?",
        "filters": {"providers": ["fintoc"], "countries": ["CL"]},
        "expected_keywords": ["fintoc", "webhook", "chile"],
        "expected_types": ["integration_guide", "api_reference"],
        "description": "Fintoc webhook configuration for Chile"
    },
    {
        "query": "What error codes does Stripe return for declined cards?",
        "filters": {"providers": ["stripe"], "has_error_codes": True},
        "expected_keywords": ["stripe", "error", "card", "declined"],
        "expected_types": ["integration_guide", "troubleshooting"],
        "description": "Stripe card decline error codes"
    },
    {
        "query": "How to test PSE payments in sandbox?",
        "filters": {"payment_methods": ["PSE"], "has_testing_instructions": True},
        "expected_keywords": ["pse", "test", "sandbox"],
        "expected_types": ["testing_guide", "integration_guide"],
        "description": "PSE payment testing guide"
    },
    {
        "query": "Recent Izipay incidents affecting Peru merchants",
        "filters": {"providers": ["izipay"], "countries": ["PE"], "document_source": "jira"},
        "expected_keywords": ["izipay", "peru", "incident"],
        "expected_types": ["post_mortem"],
        "description": "Izipay incidents in Peru"
    },
    {
        "query": "How to implement PIX payments for Brazil?",
        "filters": {"payment_methods": ["PIX"], "countries": ["BR"]},
        "expected_keywords": ["pix", "brazil", "payment"],
        "expected_types": ["integration_guide", "api_reference"],
        "description": "PIX payment implementation for Brazil"
    },
    {
        "query": "High severity payment incidents and failures",
        "filters": {"document_source": "jira", "severity": ["S1", "S2"]},
        "expected_keywords": ["payment", "incident"],
        "expected_types": ["post_mortem"],
        "description": "High severity payment incidents"
    },
    {
        "query": "How to integrate Adyen card payments?",
        "filters": {"providers": ["adyen"], "payment_methods": ["CARD"]},
        "expected_keywords": ["adyen", "card", "payment"],
        "expected_types": ["integration_guide", "api_reference"],
        "description": "Adyen card payment integration"
    },
    {
        "query": "Stripe refund and reversal process",
        "filters": {"providers": ["stripe"]},
        "expected_keywords": ["stripe", "refund"],
        "expected_types": ["integration_guide", "api_reference"],
        "description": "Stripe refund documentation"
    },
    {
        "query": "MercadoPago payment error codes and failures",
        "filters": {"providers": ["mercadopago"], "has_error_codes": True},
        "expected_keywords": ["mercadopago", "error"],
        "expected_types": ["api_reference", "troubleshooting"],
        "description": "MercadoPago error code documentation"
    },
    {
        "query": "DLocal payment integration for Colombia",
        "filters": {"providers": ["dlocal"], "countries": ["CO"]},
        "expected_keywords": ["dlocal", "colombia"],
        "expected_types": ["integration_guide", "api_reference"],
        "description": "DLocal configuration for Colombia"
    }
]


def check_relevance(doc, expected_keywords: List[str], expected_types: List[str]) -> bool:
    """
    Check if a retrieved document is relevant based on keywords and document type.

    Args:
        doc: LangChain Document object (has .page_content and .metadata)
        expected_keywords: Keywords expected in relevant documents
        expected_types: Document types expected to be relevant

    Returns:
        True if document is considered relevant
    """
    # Check document type
    doc_type = doc.metadata.get("document_type", "")
    type_match = doc_type in expected_types if expected_types else True

    # Check keyword presence in content (case-insensitive)
    content = doc.page_content.lower()  # <-- FIX: Use page_content instead of metadata
    providers = [p.lower() for p in doc.metadata.get("providers", [])]
    payment_methods = [m.lower() for m in doc.metadata.get("payment_methods", [])]
    countries = [c.lower() for c in doc.metadata.get("countries", [])]

    # Combine all searchable text
    searchable = f"{content} {' '.join(providers)} {' '.join(payment_methods)} {' '.join(countries)}".lower()

    # Check if at least 2 expected keywords are present
    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in searchable)
    keyword_relevant = keyword_matches >= min(2, len(expected_keywords))

    return type_match and keyword_relevant


def evaluate_query(test_case: Dict, top_k: int = 5) -> Dict:
    """
    Evaluate a single test query.

    Args:
        test_case: Test case dictionary with query, filters, and expected results
        top_k: Number of documents to retrieve

    Returns:
        Evaluation results dictionary
    """
    query = test_case["query"]
    filters = test_case["filters"]
    expected_keywords = test_case["expected_keywords"]
    expected_types = test_case["expected_types"]

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"Description: {test_case['description']}")
    print(f"Filters: {filters}")

    try:
        # Retrieve documents
        documents = retrieve_with_filter(
            query=query,
            top_k=top_k,
            **filters,
            verbose=False
        )

        if not documents:
            print(f"⚠️  No documents retrieved")
            return {
                "query": query,
                "precision": 0.0,
                "relevant_count": 0,
                "retrieved_count": 0,
                "error": None
            }

        # Check relevance for each document
        relevant_count = 0
        for i, doc in enumerate(documents, 1):
            is_relevant = check_relevance(doc, expected_keywords, expected_types)

            if is_relevant:
                relevant_count += 1

            # Display document info
            doc_type = doc.metadata.get("document_type", "Unknown")
            providers = doc.metadata.get("providers", [])
            ticket_id = doc.metadata.get("ticket_id", "")
            page_id = doc.metadata.get("page_id", "")
            identifier = ticket_id or f"Page {page_id}" or "Unknown"

            relevance_icon = "✅" if is_relevant else "❌"
            print(f"  {relevance_icon} [{i}] {doc_type} | {identifier} | Providers: {', '.join(providers[:2])}")

        # Calculate precision
        precision = relevant_count / len(documents)

        print(f"\n📊 Precision@{len(documents)}: {precision:.2%} ({relevant_count}/{len(documents)} relevant)")

        return {
            "query": query,
            "precision": precision,
            "relevant_count": relevant_count,
            "retrieved_count": len(documents),
            "error": None
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "query": query,
            "precision": 0.0,
            "relevant_count": 0,
            "retrieved_count": 0,
            "error": str(e)
        }


def run_precision_evaluation():
    """Run precision evaluation on all test queries."""
    print("="*70)
    print("Yuno Integrations RAG - Precision Evaluation")
    print("="*70)
    print(f"\nEvaluating {len(TEST_QUERIES)} test queries...")
    print("Target: >80% precision@5 per query\n")

    results = []
    total_precision = 0.0
    total_relevant = 0
    total_retrieved = 0
    errors = 0

    for test_case in TEST_QUERIES:
        result = evaluate_query(test_case, top_k=5)
        results.append(result)

        if result["error"]:
            errors += 1
        else:
            total_precision += result["precision"]
            total_relevant += result["relevant_count"]
            total_retrieved += result["retrieved_count"]

    # Calculate aggregate metrics
    successful_queries = len(TEST_QUERIES) - errors
    avg_precision = total_precision / successful_queries if successful_queries > 0 else 0.0
    overall_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0.0

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\n📊 Test Queries: {len(TEST_QUERIES)}")
    print(f"✅ Successful: {successful_queries}")
    print(f"❌ Errors: {errors}")
    print(f"\n📈 Average Precision@5: {avg_precision:.2%}")
    print(f"📈 Overall Precision: {overall_precision:.2%} ({total_relevant}/{total_retrieved} relevant)")

    # Check if target met
    target_met = avg_precision >= 0.80
    target_icon = "✅" if target_met else "❌"
    print(f"\n{target_icon} Target (>80% precision): {'MET' if target_met else 'NOT MET'}")

    # Show queries that need improvement
    low_precision_queries = [r for r in results if not r["error"] and r["precision"] < 0.60]
    if low_precision_queries:
        print(f"\n⚠️  Queries needing improvement ({len(low_precision_queries)}):")
        for r in low_precision_queries:
            print(f"   • {r['query'][:60]}... (Precision: {r['precision']:.2%})")

    print("\n" + "="*70)

    return {
        "results": results,
        "avg_precision": avg_precision,
        "overall_precision": overall_precision,
        "target_met": target_met
    }


if __name__ == "__main__":
    results = run_precision_evaluation()
