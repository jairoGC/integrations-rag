"""
Diagnostic script to understand why filters are hurting precision.
Analyzes metadata quality for queries where filters caused regressions.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retrieval import retrieve_with_filter

# Test cases where filters caused regressions
FAILING_QUERIES = [
    {
        "id": "stripe_error_codes",
        "query": "What error codes does Stripe return for declined cards?",
        "filters": {"providers": ["stripe"]},
        "expected": "Should find Stripe error documentation"
    },
    {
        "id": "pse_testing",
        "query": "How to test PSE payments in sandbox?",
        "filters": {"payment_methods": ["PSE"]},
        "expected": "Should find PSE testing documentation"
    },
    {
        "id": "stripe_card_implementation",
        "query": "Stripe card payment implementation?",
        "filters": {"providers": ["stripe"], "payment_methods": ["CARD"]},
        "expected": "Should find Stripe card documentation"
    },
    {
        "id": "izipay_complex",
        "query": "Critical Izipay incidents in Peru with error codes?",
        "filters": {
            "providers": ["izipay"],
            "countries": ["PE"],
            "document_source": "jira",
            "severity": ["S1", "S2"],
            "has_error_codes": True
        },
        "expected": "Should find Izipay incidents in Peru"
    }
]


def diagnose_query(test_case):
    """Diagnose why a query's filters are causing issues."""
    query = test_case["query"]
    filters = test_case["filters"]

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Expected: {test_case['expected']}")
    print(f"Filters: {filters}")
    print(f"{'='*80}")

    # Retrieve WITHOUT filters
    print("\n📊 UNFILTERED RESULTS:")
    unfiltered_docs = retrieve_with_filter(query=query, top_k=10, verbose=False)

    if not unfiltered_docs:
        print("   ❌ No unfiltered results found!")
        return

    print(f"   Retrieved {len(unfiltered_docs)} documents\n")

    for i, doc in enumerate(unfiltered_docs[:10], 1):
        providers = doc.metadata.get("providers", [])
        methods = doc.metadata.get("payment_methods", [])
        countries = doc.metadata.get("countries", [])
        doc_source = doc.metadata.get("document_source", "")
        doc_type = doc.metadata.get("document_type", "")
        severity = doc.metadata.get("severity", "")
        has_errors = doc.metadata.get("has_error_codes", False)
        has_testing = doc.metadata.get("has_testing_instructions", False)
        ticket_id = doc.metadata.get("ticket_id", "")
        page_id = doc.metadata.get("page_id", "")

        identifier = ticket_id or f"Page-{page_id}" or "Unknown"

        # Check if this doc matches the filter
        matches_filter = True
        missing_fields = []

        if "providers" in filters:
            expected_providers = filters["providers"]
            if not any(p in providers for p in expected_providers):
                matches_filter = False
                missing_fields.append(f"providers (has {providers}, need {expected_providers})")

        if "payment_methods" in filters:
            expected_methods = filters["payment_methods"]
            if not any(m in methods for m in expected_methods):
                matches_filter = False
                missing_fields.append(f"methods (has {methods}, need {expected_methods})")

        if "countries" in filters:
            expected_countries = filters["countries"]
            if not any(c in countries for c in expected_countries):
                matches_filter = False
                missing_fields.append(f"countries (has {countries}, need {expected_countries})")

        if "document_source" in filters:
            if doc_source != filters["document_source"]:
                matches_filter = False
                missing_fields.append(f"source (has '{doc_source}', need '{filters['document_source']}')")

        if "severity" in filters:
            if severity not in filters["severity"]:
                matches_filter = False
                missing_fields.append(f"severity (has '{severity}', need {filters['severity']})")

        if "has_error_codes" in filters:
            if has_errors != filters["has_error_codes"]:
                matches_filter = False
                missing_fields.append(f"has_error_codes (has {has_errors}, need {filters['has_error_codes']})")

        # Print result
        match_icon = "✅" if matches_filter else "❌"
        print(f"   {match_icon} [{i}] {identifier} ({doc_type})")
        print(f"       Providers: {providers}")
        print(f"       Methods: {methods}")
        print(f"       Countries: {countries}")
        print(f"       Source: {doc_source}, Severity: {severity}")
        print(f"       Has errors: {has_errors}, Has testing: {has_testing}")

        if not matches_filter:
            print(f"       ⚠️  Missing: {', '.join(missing_fields)}")

        # Show snippet of content
        content_snippet = doc.page_content[:150].replace('\n', ' ')
        print(f"       Content: {content_snippet}...")
        print()

    # Count how many match the filter
    matching_count = sum(1 for doc in unfiltered_docs if would_match_filter(doc, filters))
    print(f"\n📈 ANALYSIS:")
    print(f"   Total unfiltered results: {len(unfiltered_docs)}")
    print(f"   Results that match filter: {matching_count}")
    print(f"   Filtered out: {len(unfiltered_docs) - matching_count}")

    # Now retrieve WITH filters
    print(f"\n📊 FILTERED RESULTS:")
    try:
        filtered_docs = retrieve_with_filter(query=query, top_k=10, **filters, verbose=False)
        print(f"   Retrieved {len(filtered_docs)} documents")

        if len(filtered_docs) < 5:
            print(f"   ⚠️  Only {len(filtered_docs)} results - filters too restrictive!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        filtered_docs = []

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if matching_count == 0:
        print("   🔴 CRITICAL: No unfiltered results match the filter criteria!")
        print("   → The filter is excluding ALL relevant documents")
        print("   → Problem: Metadata extraction is missing key fields")
        print("   → Fix: Improve metadata extraction in ingestion.py")
    elif matching_count < len(unfiltered_docs) / 2:
        print(f"   ⚠️  Only {matching_count}/{len(unfiltered_docs)} results match filter")
        print("   → Many relevant documents lack proper metadata")
        print("   → Fix: Improve metadata extraction or relax filters")
    else:
        print(f"   ✅ {matching_count}/{len(unfiltered_docs)} results match filter")
        print("   → Metadata quality looks good")


def would_match_filter(doc, filters):
    """Check if a document would match the given filters."""
    providers = doc.metadata.get("providers", [])
    methods = doc.metadata.get("payment_methods", [])
    countries = doc.metadata.get("countries", [])
    doc_source = doc.metadata.get("document_source", "")
    severity = doc.metadata.get("severity", "")
    has_errors = doc.metadata.get("has_error_codes", False)

    # Check each filter
    if "providers" in filters:
        if not any(p in providers for p in filters["providers"]):
            return False

    if "payment_methods" in filters:
        if not any(m in methods for m in filters["payment_methods"]):
            return False

    if "countries" in filters:
        if not any(c in countries for c in filters["countries"]):
            return False

    if "document_source" in filters:
        if doc_source != filters["document_source"]:
            return False

    if "severity" in filters:
        if severity not in filters["severity"]:
            return False

    if "has_error_codes" in filters:
        if has_errors != filters["has_error_codes"]:
            return False

    return True


def main():
    """Run diagnostics on all failing queries."""
    print("=" * 80)
    print("FILTER REGRESSION DIAGNOSTICS")
    print("=" * 80)
    print("\nAnalyzing why filters are hurting precision...\n")

    for test_case in FAILING_QUERIES:
        diagnose_query(test_case)
        print("\n")


if __name__ == "__main__":
    main()
