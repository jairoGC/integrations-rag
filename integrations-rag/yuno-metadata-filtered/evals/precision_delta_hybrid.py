"""
Precision Delta with Hybrid Retrieval

Tests if hybrid retrieval improves precision over standard filtering.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retrieval import retrieve_with_filter, retrieve_hybrid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "gpt-4o-mini"

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

Question: {question}

Retrieved Document:
{document}

Is this document relevant? Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""


# Critical test cases where standard filtering failed
TEST_CASES = [
    {
        "id": "combined_complex",
        "question": "Critical Izipay incidents in Peru with error codes?",
        "filters": {
            "providers": ["izipay"],
            "countries": ["PE"],
            "document_source": "jira",
            "severity": ["S1", "S2"],
            "has_error_codes": True
        },
        "description": "Complex multi-filter (standard returned 0 docs)"
    },
    {
        "id": "stripe_card",
        "question": "Stripe card payment implementation?",
        "filters": {"providers": ["stripe"], "payment_methods": ["CARD"]},
        "description": "Stripe + CARD combination"
    },
    {
        "id": "stripe_errors",
        "question": "What error codes does Stripe return for declined cards?",
        "filters": {"providers": ["stripe"], "has_error_codes": True},
        "description": "Stripe error codes"
    },
]


def create_judge():
    """Create LLM judge."""
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)
    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    return prompt | llm | StrOutputParser()


def judge_relevance(judge_chain, question: str, document_content: str) -> bool:
    """Judge if document is relevant."""
    try:
        response = judge_chain.invoke({"question": question, "document": document_content})
        return "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()
    except:
        return False


def calculate_precision(question: str, documents: list, judge) -> float:
    """Calculate precision."""
    if not documents:
        return 0.0
    relevant = sum(1 for doc in documents if judge_relevance(judge, question, doc.page_content))
    return relevant / len(documents)


def run_comparison():
    """Compare standard vs hybrid retrieval precision."""
    print("=" * 70)
    print("PRECISION DELTA: Standard vs Hybrid Retrieval")
    print("=" * 70)
    print("\nTesting hybrid retrieval on problematic queries...\n")

    judge = create_judge()
    results = []

    for test_case in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"Query: {test_case['question']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*70}\n")

        # Standard filtering
        print("📋 Standard Filtered Retrieval:")
        standard_docs = retrieve_with_filter(
            query=test_case['question'],
            top_k=5,
            **test_case['filters'],
            verbose=False
        )
        standard_precision = calculate_precision(test_case['question'], standard_docs, judge)
        print(f"   Docs: {len(standard_docs)}")
        print(f"   Precision: {standard_precision:.0%}")

        # Hybrid retrieval
        print(f"\n🔀 Hybrid Retrieval:")
        hybrid_docs = retrieve_hybrid(
            query=test_case['question'],
            top_k=5,
            filtered_ratio=0.6,
            **test_case['filters'],
            verbose=False
        )
        hybrid_precision = calculate_precision(test_case['question'], hybrid_docs, judge)
        print(f"   Docs: {len(hybrid_docs)}")
        print(f"   Precision: {hybrid_precision:.0%}")

        # Delta
        delta = hybrid_precision - standard_precision
        if delta > 0:
            status = f"🟢 +{delta:.0%}"
        elif delta < 0:
            status = f"🔴 {delta:.0%}"
        else:
            status = "⚪ 0%"

        print(f"\n📊 Delta: {status}")

        results.append({
            "test_id": test_case['id'],
            "standard_precision": standard_precision,
            "hybrid_precision": hybrid_precision,
            "delta": delta,
            "improvement": delta > 0
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    improvements = sum(1 for r in results if r['improvement'])
    avg_standard = sum(r['standard_precision'] for r in results) / len(results)
    avg_hybrid = sum(r['hybrid_precision'] for r in results) / len(results)
    avg_delta = sum(r['delta'] for r in results) / len(results)

    print(f"\n📊 Average Precision:")
    print(f"   Standard: {avg_standard:.2%}")
    print(f"   Hybrid: {avg_hybrid:.2%}")
    print(f"   Delta: {avg_delta:+.2%}")

    print(f"\n📈 Outcomes:")
    print(f"   Improvements: {improvements}/{len(results)}")

    if avg_delta > 0:
        print(f"\n✅ Hybrid retrieval improves precision by {avg_delta:.1%}!")
    else:
        print(f"\n⚪ Hybrid retrieval maintains similar precision")

    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
