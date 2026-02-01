"""
Yuno Integrations RAG - Generation Module
Uses filtered retrieval to generate answers from Yuno's payment integration documentation.

Supports answering questions with constraints like:
- "How to configure Fintoc webhooks in Chile?"
- "Recent Stripe incidents with S1 severity..."
- "PSE payment integration guide for Colombia..."
- "Error codes for Adyen card payments..."
"""

import os
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import (
    retrieve_with_filter,
    retrieve_hybrid,
    format_retrieved_context,
    get_available_providers,
    get_available_payment_methods,
    get_available_countries,
    get_provider_counts,
    get_payment_method_counts,
    get_document_type_counts,
    AVAILABLE_DOCUMENT_SOURCES,
    AVAILABLE_DOCUMENT_TYPES,
    AVAILABLE_PROVIDERS,
    AVAILABLE_PAYMENT_METHODS,
    AVAILABLE_COUNTRIES,
    AVAILABLE_SEVERITY_LEVELS,
    AVAILABLE_PLATFORMS
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template with filter context
RAG_PROMPT_TEMPLATE = """You are a technical documentation assistant for Yuno's payment integration system. You help engineers troubleshoot payment integrations, understand provider configurations, and resolve payment-related incidents.

{filter_context}

When answering:
- Cite specific providers, error codes, endpoints, or services when present in the context
- Be precise with technical details (API endpoints, webhook URLs, configuration parameters)
- If the context mentions specific Jira tickets or Confluence pages, reference them
- If the information is insufficient, clearly state what additional documentation would be needed

Use ONLY the information from the context below. Do not make assumptions about provider behavior or API details not present in the documentation.

Context:
{context}

Question: {question}

Answer:"""


def create_rag_chain():
    """Create the RAG chain with prompt template and LLM."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    return chain


def build_filter_context(
    document_source: Optional[str | list[str]] = None,
    document_type: Optional[str | list[str]] = None,
    providers: Optional[str | list[str]] = None,
    payment_methods: Optional[str | list[str]] = None,
    countries: Optional[str | list[str]] = None,
    services: Optional[str | list[str]] = None,
    severity: Optional[str | list[str]] = None,
    platform: Optional[str] = None,
    has_error_codes: Optional[bool] = None,
    has_testing_instructions: Optional[bool] = None,
    has_technical_content: Optional[bool] = None,
    created_date_range: Optional[tuple[str, str]] = None,
    updated_date_range: Optional[tuple[str, str]] = None,
    ticket_id: Optional[str] = None,
    page_id: Optional[str] = None,
) -> str:
    """Build a human-readable description of the applied Yuno-specific filters."""
    parts = []

    if document_source is not None:
        if isinstance(document_source, list):
            parts.append(f"from {' and '.join(document_source)} documents")
        else:
            parts.append(f"from {document_source} documents")

    if document_type is not None:
        if isinstance(document_type, list):
            types_str = ', '.join([t.replace('_', ' ') for t in document_type])
            parts.append(f"document types: {types_str}")
        else:
            parts.append(f"document type: {document_type.replace('_', ' ')}")

    if providers is not None:
        if isinstance(providers, str):
            providers = [providers]
        parts.append(f"providers: {', '.join(providers)}")

    if payment_methods is not None:
        if isinstance(payment_methods, str):
            payment_methods = [payment_methods]
        parts.append(f"payment methods: {', '.join(payment_methods)}")

    if countries is not None:
        if isinstance(countries, str):
            countries = [countries]
        parts.append(f"countries: {', '.join(countries)}")

    if services is not None:
        if isinstance(services, str):
            services = [services]
        parts.append(f"services: {', '.join(services)}")

    if severity is not None:
        if isinstance(severity, list):
            parts.append(f"severity: {', '.join(severity)}")
        else:
            parts.append(f"severity: {severity}")

    if platform is not None:
        parts.append(f"platform: {platform}")

    if has_error_codes is True:
        parts.append("containing error codes")

    if has_testing_instructions is True:
        parts.append("with testing instructions")

    if has_technical_content is True:
        parts.append("with technical content (API endpoints, credentials)")

    if created_date_range is not None:
        parts.append(f"created between {created_date_range[0]} and {created_date_range[1]}")

    if updated_date_range is not None:
        parts.append(f"updated between {updated_date_range[0]} and {updated_date_range[1]}")

    if ticket_id is not None:
        parts.append(f"Jira ticket: {ticket_id}")

    if page_id is not None:
        parts.append(f"Confluence page: {page_id}")

    if parts:
        return f"You are answering based on a filtered subset of integration documentation: {'; '.join(parts)}."
    else:
        return "You are answering based on all available Yuno integration documentation."


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    use_hybrid: bool = True,
    document_source: Optional[str | list[str]] = None,
    document_type: Optional[str | list[str]] = None,
    providers: Optional[str | list[str]] = None,
    payment_methods: Optional[str | list[str]] = None,
    countries: Optional[str | list[str]] = None,
    services: Optional[str | list[str]] = None,
    severity: Optional[str | list[str]] = None,
    platform: Optional[str] = None,
    has_error_codes: Optional[bool] = None,
    has_testing_instructions: Optional[bool] = None,
    has_technical_content: Optional[bool] = None,
    created_date_range: Optional[tuple[str, str]] = None,
    updated_date_range: Optional[tuple[str, str]] = None,
    ticket_id: Optional[str] = None,
    page_id: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Generate an answer using the Yuno integrations RAG pipeline.

    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        use_hybrid: Use hybrid retrieval (combines filtered + unfiltered, default: True)
        document_source: Filter by source (jira, confluence)
        document_type: Filter by type (post_mortem, integration_guide, etc.)
        providers: Filter by provider(s)
        payment_methods: Filter by payment method(s)
        countries: Filter by country code(s)
        services: Filter by service name(s)
        severity: Filter by severity level(s)
        platform: Filter by platform (web, mobile, both)
        has_error_codes: Filter for documents with error codes
        has_testing_instructions: Filter for documents with testing instructions
        has_technical_content: Filter for documents with technical content
        created_date_range: Filter by creation date range
        updated_date_range: Filter by update date range
        ticket_id: Filter by specific Jira ticket ID
        page_id: Filter by specific Confluence page ID
        verbose: Include retrieved documents in response

    Returns:
        Dictionary containing the answer and metadata
    """
    # Step 1: Retrieve with filters (hybrid or standard mode)
    retrieval_func = retrieve_hybrid if use_hybrid else retrieve_with_filter

    documents = retrieval_func(
        query=question,
        top_k=top_k,
        document_source=document_source,
        document_type=document_type,
        providers=providers,
        payment_methods=payment_methods,
        countries=countries,
        services=services,
        severity=severity,
        platform=platform,
        has_error_codes=has_error_codes,
        has_testing_instructions=has_testing_instructions,
        has_technical_content=has_technical_content,
        created_date_range=created_date_range,
        updated_date_range=updated_date_range,
        ticket_id=ticket_id,
        page_id=page_id,
        verbose=verbose
    )

    if not documents:
        return {
            "answer": "I couldn't find any relevant integration documentation matching your criteria. Try broadening your filters or rephrasing your question.",
            "sources": [],
            "filters_applied": build_filter_context(
                document_source, document_type, providers, payment_methods,
                countries, services, severity, platform, has_error_codes,
                has_testing_instructions, has_technical_content,
                created_date_range, updated_date_range, ticket_id, page_id
            )
        }

    # Step 2: Format context
    context = format_retrieved_context(documents)
    filter_context = build_filter_context(
        document_source, document_type, providers, payment_methods,
        countries, services, severity, platform, has_error_codes,
        has_testing_instructions, has_technical_content,
        created_date_range, updated_date_range, ticket_id, page_id
    )

    # Step 3: Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "filter_context": filter_context
    })

    # Prepare response
    response = {
        "answer": answer,
        "sources": [
            {
                "source": doc.metadata.get("document_source", "Unknown"),
                "type": doc.metadata.get("document_type", "Unknown"),
                "identifier": doc.metadata.get("ticket_id") or doc.metadata.get("page_id", "Unknown"),
                "providers": doc.metadata.get("providers", []),
                "payment_methods": doc.metadata.get("payment_methods", []),
                "countries": doc.metadata.get("countries", []),
                "services": doc.metadata.get("services", []),
                "severity": doc.metadata.get("severity", "N/A")
            }
            for doc in documents
        ],
        "filters_applied": filter_context
    }
    
    if verbose:
        response["retrieved_documents"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
    
    return response


def interactive_mode():
    """Run an interactive Q&A session with Yuno-specific filter support."""
    print("\n" + "=" * 60)
    print("Yuno Integrations RAG - Interactive Q&A")
    print("=" * 60)
    print("\nFilter commands:")
    print("  source:jira            - Filter by document source (jira, confluence)")
    print("  type:post_mortem       - Filter by document type")
    print("  provider:fintoc        - Filter by provider (comma-separated for multiple)")
    print("  method:PIX             - Filter by payment method")
    print("  country:CL             - Filter by country code")
    print("  service:izipay-int     - Filter by service name")
    print("  severity:S1            - Filter by severity level (S1/S2/S3/S4)")
    print("  platform:web           - Filter by platform (web, mobile, both)")
    print("  errors:on              - Show only docs with error codes")
    print("  testing:on             - Show only docs with testing instructions")
    print("  clear                  - Reset all filters")
    print("  filters                - Show active filters")
    print("  help                   - Show available providers, methods, countries")
    print("  quit                   - Exit")
    print("-" * 60)

    # Active filters
    active_filters = {
        "document_source": None,
        "document_type": None,
        "providers": None,
        "payment_methods": None,
        "countries": None,
        "services": None,
        "severity": None,
        "platform": None,
        "has_error_codes": None,
        "has_testing_instructions": None,
    }
    
    while True:
        print()
        user_input = input(">> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'clear':
            active_filters = {k: None for k in active_filters}
            print("✅ All filters cleared")
            continue
        
        if user_input.lower() == 'filters':
            active = {k: v for k, v in active_filters.items() if v is not None}
            if active:
                print("🔍 Active filters:")
                for k, v in active.items():
                    print(f"   {k}: {v}")
            else:
                print("No active filters")
            continue
        
        if user_input.lower() == 'help':
            print("\n💳 Available Providers:")
            for provider, count in get_provider_counts():
                print(f"   {provider} ({count} chunks)")
            print("\n💵 Available Payment Methods:")
            for method, count in get_payment_method_counts():
                print(f"   {method} ({count} chunks)")
            print("\n🌎 Available Countries:")
            print(f"   {', '.join(get_available_countries())}")
            print("\n📋 Document Types:")
            for doc_type, count in get_document_type_counts():
                print(f"   {doc_type} ({count} chunks)")
            continue

        # Parse filter commands
        if user_input.startswith('source:'):
            sources = user_input.split(':')[1].split(',')
            active_filters["document_source"] = [s.strip() for s in sources] if len(sources) > 1 else sources[0].strip()
            print(f"✅ Document source filter: {active_filters['document_source']}")
            continue

        if user_input.startswith('type:'):
            types = user_input.split(':')[1].split(',')
            active_filters["document_type"] = [t.strip() for t in types] if len(types) > 1 else types[0].strip()
            print(f"✅ Document type filter: {active_filters['document_type']}")
            continue

        if user_input.startswith('provider:'):
            providers = user_input.split(':')[1].split(',')
            active_filters["providers"] = [p.strip() for p in providers]
            print(f"✅ Provider filter: {active_filters['providers']}")
            continue

        if user_input.startswith('method:'):
            methods = user_input.split(':')[1].split(',')
            active_filters["payment_methods"] = [m.strip() for m in methods]
            print(f"✅ Payment method filter: {active_filters['payment_methods']}")
            continue

        if user_input.startswith('country:'):
            countries = user_input.split(':')[1].split(',')
            active_filters["countries"] = [c.strip() for c in countries]
            print(f"✅ Country filter: {active_filters['countries']}")
            continue

        if user_input.startswith('service:'):
            services = user_input.split(':')[1].split(',')
            active_filters["services"] = [s.strip() for s in services]
            print(f"✅ Service filter: {active_filters['services']}")
            continue

        if user_input.startswith('severity:'):
            severities = user_input.split(':')[1].split(',')
            active_filters["severity"] = [s.strip() for s in severities] if len(severities) > 1 else severities[0].strip()
            print(f"✅ Severity filter: {active_filters['severity']}")
            continue

        if user_input.startswith('platform:'):
            active_filters["platform"] = user_input.split(':')[1].strip()
            print(f"✅ Platform filter: {active_filters['platform']}")
            continue

        if user_input.startswith('errors:'):
            val = user_input.split(':')[1].lower()
            active_filters["has_error_codes"] = val in ['on', 'true', 'yes', '1']
            print(f"✅ Error codes filter: {'ON' if active_filters['has_error_codes'] else 'OFF'}")
            continue

        if user_input.startswith('testing:'):
            val = user_input.split(':')[1].lower()
            active_filters["has_testing_instructions"] = val in ['on', 'true', 'yes', '1']
            print(f"✅ Testing instructions filter: {'ON' if active_filters['has_testing_instructions'] else 'OFF'}")
            continue
        
        # Treat as question
        question = user_input
        
        # Show active filters
        active = {k: v for k, v in active_filters.items() if v is not None}
        if active:
            print(f"🔍 Searching with filters: {active}")
        
        print("🤖 Generating answer...\n")
        
        try:
            result = generate_answer(
                question,
                **active_filters,
                verbose=False
            )
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print(f"\n📚 Sources ({len(result['sources'])} documents):")
            for source in result["sources"]:
                providers_str = ', '.join(source['providers'][:2]) if source['providers'] else 'N/A'
                methods_str = ', '.join(source['payment_methods'][:2]) if source['payment_methods'] else 'N/A'
                print(f"  • {source['source']} | {source['type']} | {source['identifier']}")
                print(f"    Providers: {providers_str} | Methods: {methods_str} | Severity: {source['severity']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Yuno Integrations RAG - Generation Pipeline")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Example queries with filters
    example_queries = [
        {
            "question": "How do I configure Fintoc webhooks for Chile?",
            "providers": ["fintoc"],
            "countries": ["CL"],
            "has_technical_content": True,
            "description": "Provider + country with technical content"
        },
        {
            "question": "What error codes does Stripe return for declined cards?",
            "providers": ["stripe"],
            "payment_methods": ["CARD"],
            "has_error_codes": True,
            "description": "Provider + method with error codes"
        },
        {
            "question": "How to test PSE payments in sandbox?",
            "payment_methods": ["PSE"],
            "has_testing_instructions": True,
            "description": "Payment method with testing docs"
        },
        {
            "question": "Recent critical payment incidents affecting Izipay in Peru",
            "providers": ["izipay"],
            "countries": ["PE"],
            "document_source": "jira",
            "severity": ["S1", "S2"],
            "description": "Provider + country with high severity"
        },
    ]
    
    print("\n📋 Example Queries with Filters:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q['question'][:45]}... ({q['description']})")
    
    print("\n" + "-" * 50)
    choice = input("Enter 1-4 for examples, 'i' for interactive mode, or your question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3', '4']:
        q = example_queries[int(choice) - 1]
        print(f"\n📝 Question: {q['question']}")
        print(f"📁 Filters: {q['description']}")
        print("\n🔍 Retrieving with filters...")
        print("🤖 Generating answer...\n")
        
        # Build kwargs from example
        kwargs = {k: v for k, v in q.items() if k not in ['question', 'description']}
        
        result = generate_answer(q['question'], **kwargs, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for source in result["sources"]:
            identifier = source['identifier']
            doc_type = source['type']
            doc_source = source['source']
            providers = ', '.join(source['providers'][:2]) if source['providers'] else 'N/A'
            print(f"  • {identifier} ({doc_type} from {doc_source}, Providers: {providers})")
            # Show additional metadata if available
            if source.get('payment_methods'):
                methods = ', '.join(source['payment_methods'][:3])
                print(f"    Methods: {methods}")
            if source.get('countries'):
                countries = ', '.join(source['countries'][:3])
                print(f"    Countries: {countries}")

        print(f"\n🔍 {result['filters_applied']}")
    elif choice:
        print(f"\n📝 Question: {choice}")
        print("\n🔍 Retrieving (no filters)...")
        print("🤖 Generating answer...\n")
        
        result = generate_answer(choice, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\n📚 Sources:")
        for source in result["sources"]:
            identifier = source['identifier']
            doc_type = source['type']
            doc_source = source['source']
            providers = ', '.join(source['providers'][:2]) if source['providers'] else 'N/A'
            print(f"  • {identifier} ({doc_type} from {doc_source}, Providers: {providers})")
    else:
        print("\n👋 No input provided. Run again to try!")


if __name__ == "__main__":
    main()
