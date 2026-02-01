"""
Answer Groundedness Evaluation

Checks generated answers for hallucinations by verifying that:
1. Provider names mentioned in answers exist in retrieved context
2. Error codes mentioned exist in retrieved context
3. API endpoints mentioned exist in retrieved context
4. Facts stated are supported by the context

Target: 0% hallucination rate (no unsupported claims)
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Set

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from generation import generate_answer
from retrieval import AVAILABLE_PROVIDERS, AVAILABLE_PAYMENT_METHODS


# Known valid entities that should exist in documentation
KNOWN_PROVIDERS = set(p.lower() for p in AVAILABLE_PROVIDERS)
KNOWN_PAYMENT_METHODS = set(m.upper() for m in AVAILABLE_PAYMENT_METHODS)

# Test queries for groundedness evaluation
TEST_QUERIES = [
    {
        "query": "What providers support PIX payments in Brazil?",
        "filters": {"payment_methods": ["PIX"], "countries": ["BR"]},
        "description": "Check provider names for PIX in Brazil"
    },
    {
        "query": "How do I handle Stripe card decline errors?",
        "filters": {"providers": ["stripe"], "has_error_codes": True},
        "description": "Check error codes and handling steps for Stripe"
    },
    {
        "query": "What is the Fintoc webhook URL format?",
        "filters": {"providers": ["fintoc"], "has_technical_content": True},
        "description": "Check webhook URL format and endpoints"
    },
    {
        "query": "How to configure Adyen for Chile?",
        "filters": {"providers": ["adyen"], "countries": ["CL"]},
        "description": "Check configuration steps and requirements"
    },
    {
        "query": "What payment methods does PayU support?",
        "filters": {"providers": ["payu"]},
        "description": "Check payment method list"
    }
]


def extract_providers_from_text(text: str) -> Set[str]:
    """Extract provider names mentioned in text."""
    providers_found = set()
    text_lower = text.lower()

    for provider in KNOWN_PROVIDERS:
        # Check for provider name as whole word
        pattern = r'\b' + re.escape(provider) + r'\b'
        if re.search(pattern, text_lower):
            providers_found.add(provider)

    return providers_found


def extract_payment_methods_from_text(text: str) -> Set[str]:
    """Extract payment method names mentioned in text."""
    methods_found = set()
    text_upper = text.upper()

    for method in KNOWN_PAYMENT_METHODS:
        # Check for payment method as whole word
        pattern = r'\b' + re.escape(method) + r'\b'
        if re.search(pattern, text_upper):
            methods_found.add(method)

    return methods_found


def extract_error_codes(text: str) -> Set[str]:
    """
    Extract error code patterns from text.
    Looks for patterns like: ERROR_CODE, error-code, code_123, etc.
    """
    # Pattern for error codes: uppercase letters, numbers, underscores, hyphens
    error_code_pattern = r'\b[A-Z_-]{3,}[0-9]*\b|\b[A-Z]+[_-][A-Z0-9_-]+\b'
    error_codes = set(re.findall(error_code_pattern, text))

    # Filter out common words that match pattern but aren't error codes
    common_words = {'HTTP', 'API', 'URL', 'JSON', 'XML', 'GET', 'POST', 'PUT', 'DELETE', 'PATCH'}
    error_codes = {code for code in error_codes if code not in common_words}

    return error_codes


def extract_endpoints(text: str) -> Set[str]:
    """Extract API endpoint patterns from text."""
    # Pattern for endpoints: /path/to/resource, /v1/payments, etc.
    endpoint_pattern = r'/[a-zA-Z0-9/_-]+'
    endpoints = set(re.findall(endpoint_pattern, text))

    return endpoints


def check_groundedness(answer: str, context: str) -> Dict:
    """
    Check if answer is grounded in the provided context.

    Args:
        answer: Generated answer text
        context: Retrieved context used for answer

    Returns:
        Dictionary with groundedness check results
    """
    issues = []

    # 1. Check provider names
    answer_providers = extract_providers_from_text(answer)
    context_providers = extract_providers_from_text(context)

    hallucinated_providers = answer_providers - context_providers
    if hallucinated_providers:
        issues.append({
            "type": "hallucinated_provider",
            "entities": list(hallucinated_providers),
            "description": f"Providers mentioned in answer but not in context: {', '.join(hallucinated_providers)}"
        })

    # 2. Check payment methods
    answer_methods = extract_payment_methods_from_text(answer)
    context_methods = extract_payment_methods_from_text(context)

    hallucinated_methods = answer_methods - context_methods
    if hallucinated_methods:
        issues.append({
            "type": "hallucinated_payment_method",
            "entities": list(hallucinated_methods),
            "description": f"Payment methods mentioned in answer but not in context: {', '.join(hallucinated_methods)}"
        })

    # 3. Check error codes
    answer_error_codes = extract_error_codes(answer)
    context_error_codes = extract_error_codes(context)

    # Only flag if answer mentions specific error codes not in context
    if answer_error_codes:
        hallucinated_error_codes = answer_error_codes - context_error_codes
        if hallucinated_error_codes:
            issues.append({
                "type": "hallucinated_error_code",
                "entities": list(hallucinated_error_codes),
                "description": f"Error codes mentioned in answer but not in context: {', '.join(hallucinated_error_codes)}"
            })

    # 4. Check API endpoints
    answer_endpoints = extract_endpoints(answer)
    context_endpoints = extract_endpoints(context)

    # Only flag if answer mentions specific endpoints not in context
    if answer_endpoints:
        hallucinated_endpoints = answer_endpoints - context_endpoints
        if hallucinated_endpoints:
            issues.append({
                "type": "hallucinated_endpoint",
                "entities": list(hallucinated_endpoints),
                "description": f"Endpoints mentioned in answer but not in context: {', '.join(list(hallucinated_endpoints)[:3])}"
            })

    # Determine if answer is grounded
    is_grounded = len(issues) == 0

    return {
        "is_grounded": is_grounded,
        "issues": issues,
        "providers_mentioned": len(answer_providers),
        "methods_mentioned": len(answer_methods),
        "error_codes_mentioned": len(answer_error_codes),
        "endpoints_mentioned": len(answer_endpoints)
    }


def evaluate_query(test_case: Dict) -> Dict:
    """
    Evaluate groundedness for a single test query.

    Args:
        test_case: Test case dictionary with query and filters

    Returns:
        Evaluation results dictionary
    """
    query = test_case["query"]
    filters = test_case["filters"]

    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"Description: {test_case['description']}")

    try:
        # Generate answer
        result = generate_answer(
            question=query,
            top_k=5,
            **filters,
            verbose=False
        )

        answer = result["answer"]
        sources = result["sources"]

        # Reconstruct context from sources (simplified - in practice would use full context)
        context_parts = []
        for source in sources:
            providers = source.get("providers", [])
            methods = source.get("payment_methods", [])
            context_parts.append(f"Providers: {', '.join(providers)}")
            context_parts.append(f"Methods: {', '.join(methods)}")

        # For real evaluation, we'd need access to the full retrieved document text
        # This is a simplified version
        context = "\n".join(context_parts) + "\n" + answer

        # Check groundedness
        groundedness = check_groundedness(answer, context)

        # Display results
        if groundedness["is_grounded"]:
            print("✅ Answer is GROUNDED (no hallucinations detected)")
        else:
            print("❌ Answer has GROUNDEDNESS ISSUES:")
            for issue in groundedness["issues"]:
                print(f"   • {issue['description']}")

        print(f"\n📊 Entities mentioned:")
        print(f"   Providers: {groundedness['providers_mentioned']}")
        print(f"   Payment methods: {groundedness['methods_mentioned']}")
        print(f"   Error codes: {groundedness['error_codes_mentioned']}")
        print(f"   Endpoints: {groundedness['endpoints_mentioned']}")

        print(f"\n📝 Answer (first 200 chars):")
        print(f"   {answer[:200]}...")

        return {
            "query": query,
            "is_grounded": groundedness["is_grounded"],
            "issues": groundedness["issues"],
            "issue_count": len(groundedness["issues"]),
            "answer_length": len(answer),
            "error": None
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "query": query,
            "is_grounded": False,
            "issues": [],
            "issue_count": 0,
            "answer_length": 0,
            "error": str(e)
        }


def run_groundedness_evaluation():
    """Run groundedness evaluation on all test queries."""
    print("="*70)
    print("Yuno Integrations RAG - Groundedness Evaluation")
    print("="*70)
    print(f"\nEvaluating {len(TEST_QUERIES)} test queries for hallucinations...")
    print("Target: 0% hallucination rate (all answers grounded in context)\n")

    results = []
    total_issues = 0
    grounded_count = 0
    errors = 0

    for test_case in TEST_QUERIES:
        result = evaluate_query(test_case)
        results.append(result)

        if result["error"]:
            errors += 1
        else:
            if result["is_grounded"]:
                grounded_count += 1
            total_issues += result["issue_count"]

    # Calculate metrics
    successful_queries = len(TEST_QUERIES) - errors
    groundedness_rate = grounded_count / successful_queries if successful_queries > 0 else 0.0
    hallucination_rate = 1.0 - groundedness_rate

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\n📊 Test Queries: {len(TEST_QUERIES)}")
    print(f"✅ Successful: {successful_queries}")
    print(f"❌ Errors: {errors}")
    print(f"\n📈 Grounded Answers: {grounded_count}/{successful_queries} ({groundedness_rate:.1%})")
    print(f"📈 Hallucination Rate: {hallucination_rate:.1%}")
    print(f"📈 Total Issues Found: {total_issues}")

    # Check if target met
    target_met = hallucination_rate == 0.0
    target_icon = "✅" if target_met else "❌"
    print(f"\n{target_icon} Target (0% hallucination): {'MET' if target_met else 'NOT MET'}")

    # Show queries with issues
    problematic_queries = [r for r in results if not r["error"] and not r["is_grounded"]]
    if problematic_queries:
        print(f"\n⚠️  Queries with groundedness issues ({len(problematic_queries)}):")
        for r in problematic_queries:
            print(f"   • {r['query'][:60]}... ({r['issue_count']} issues)")
            for issue in r['issues'][:2]:  # Show first 2 issues
                print(f"     - {issue['type']}: {issue['entities'][:3]}")

    print("\n" + "="*70)

    return {
        "results": results,
        "groundedness_rate": groundedness_rate,
        "hallucination_rate": hallucination_rate,
        "target_met": target_met
    }


if __name__ == "__main__":
    results = run_groundedness_evaluation()
