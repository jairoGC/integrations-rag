"""
Yuno Integrations RAG - Retrieval Module
Performs metadata filtering BEFORE vector search for more targeted results.

Supported filters:
- document_source: Filter by source (jira, confluence)
- document_type: Filter by type (post_mortem, integration_guide, api_reference, etc.)
- providers: Filter by payment providers (fintoc, stripe, adyen, payu, etc.)
- payment_methods: Filter by payment methods (PIX, PSE, CARD, BANK_TRANSFER, etc.)
- countries: Filter by country codes (CL, MX, BR, CO, PE, AR, etc.)
- services: Filter by microservice names (izipay-int, fintoc-int, etc.)
- severity: Filter by severity level (S1, S2, S3, S4)
- platform: Filter by platform (web, mobile, both)
- has_error_codes: Filter for documents with error codes
- has_testing_instructions: Filter for documents with testing instructions
- has_technical_content: Filter for documents with API endpoints/credentials
- created_date_range: Filter by creation date range
- updated_date_range: Filter by last update date range
"""

import os
import certifi
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (must match ingestion.py)
DB_NAME = "yuno_integrations_rag"
COLLECTION_NAME = "integration_docs"
INDEX_NAME = "integration_docs_index"

# Retrieval configuration
DEFAULT_TOP_K = 5

# Available filter options (for reference)
AVAILABLE_DOCUMENT_SOURCES = ["jira", "confluence"]

AVAILABLE_DOCUMENT_TYPES = [
    "post_mortem", "integration_guide", "api_reference",
    "troubleshooting", "technical_specs", "testing_guide"
]

AVAILABLE_PROVIDERS = [
    "fintoc", "stripe", "adyen", "payu", "izipay", "dlocal", "mercadopago",
    "paypal", "coinflow", "paymentes", "ebanx", "rappi", "nuvei", "kushki", "niubiz"
]

AVAILABLE_PAYMENT_METHODS = [
    "CARD", "BANK_TRANSFER", "PIX", "PSE", "OXXO", "BOLETO",
    "WALLET", "CRYPTO", "SPEI", "CASH", "DEBIT_CARD", "CREDIT_CARD"
]

AVAILABLE_COUNTRIES = [
    "CL", "MX", "BR", "CO", "PE", "AR", "EC", "UY", "PA", "CR"
]

AVAILABLE_SEVERITY_LEVELS = ["S1", "S2", "S3", "S4"]

AVAILABLE_PLATFORMS = ["web", "mobile", "both"]


def get_vector_store():
    """Connect to the MongoDB vector store."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return vector_store, client


def build_pre_filter(
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
) -> dict:
    """
    Build a MongoDB pre-filter for vector search with Yuno-specific filters.

    Args:
        document_source: Source type (jira, confluence) - single or list
        document_type: Document type (post_mortem, integration_guide, etc.) - single or list
        providers: Provider name(s) (fintoc, stripe, etc.) - single or list (OR matching)
        payment_methods: Payment method(s) (PIX, PSE, CARD, etc.) - single or list (OR matching)
        countries: Country code(s) (CL, MX, BR, etc.) - single or list (OR matching)
        services: Service name(s) (izipay-int, fintoc-int, etc.) - single or list (OR matching)
        severity: Severity level(s) (S1, S2, S3, S4) - single or list
        platform: Platform type (web, mobile, both)
        has_error_codes: Filter for documents with error codes
        has_testing_instructions: Filter for documents with testing instructions
        has_technical_content: Filter for documents with API endpoints/credentials
        created_date_range: Tuple of (start_date, end_date) as ISO strings
        updated_date_range: Tuple of (start_date, end_date) as ISO strings
        ticket_id: Specific Jira ticket ID (e.g., "PFU-152")
        page_id: Specific Confluence page ID

    Returns:
        MongoDB filter dictionary
    """
    conditions = []

    # Document source filter
    if document_source is not None:
        if isinstance(document_source, list):
            conditions.append({"document_source": {"$in": document_source}})
        else:
            conditions.append({"document_source": {"$eq": document_source}})

    # Document type filter
    if document_type is not None:
        if isinstance(document_type, list):
            conditions.append({"document_type": {"$in": document_type}})
        else:
            conditions.append({"document_type": {"$eq": document_type}})

    # Providers filter (OR matching - document mentions at least one provider)
    if providers is not None:
        if isinstance(providers, str):
            providers = [providers]
        conditions.append({"providers": {"$in": providers}})

    # Payment methods filter (OR matching)
    if payment_methods is not None:
        if isinstance(payment_methods, str):
            payment_methods = [payment_methods]
        conditions.append({"payment_methods": {"$in": payment_methods}})

    # Countries filter (OR matching)
    if countries is not None:
        if isinstance(countries, str):
            countries = [countries]
        conditions.append({"countries": {"$in": countries}})

    # Services filter (OR matching)
    if services is not None:
        if isinstance(services, str):
            services = [services]
        conditions.append({"services": {"$in": services}})

    # Severity filter
    if severity is not None:
        if isinstance(severity, list):
            conditions.append({"severity": {"$in": severity}})
        else:
            conditions.append({"severity": {"$eq": severity}})

    # Platform filter
    if platform is not None:
        conditions.append({"platform": {"$eq": platform}})

    # Boolean filters
    if has_error_codes is not None:
        conditions.append({"has_error_codes": {"$eq": has_error_codes}})

    if has_testing_instructions is not None:
        conditions.append({"has_testing_instructions": {"$eq": has_testing_instructions}})

    if has_technical_content is not None:
        conditions.append({"has_technical_content": {"$eq": has_technical_content}})

    # Created date range filter
    if created_date_range is not None:
        start_date, end_date = created_date_range
        conditions.append({
            "$and": [
                {"created_date": {"$gte": start_date}},
                {"created_date": {"$lte": end_date}}
            ]
        })

    # Updated date range filter
    if updated_date_range is not None:
        start_date, end_date = updated_date_range
        conditions.append({
            "$and": [
                {"updated_date": {"$gte": start_date}},
                {"updated_date": {"$lte": end_date}}
            ]
        })

    # Ticket ID filter (exact match)
    if ticket_id is not None:
        conditions.append({"ticket_id": {"$eq": ticket_id}})

    # Page ID filter (exact match)
    if page_id is not None:
        conditions.append({"page_id": {"$eq": page_id}})

    # Combine all conditions with AND
    if not conditions:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def retrieve_with_filter(
    query: str,
    top_k: int = DEFAULT_TOP_K,
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
) -> list:
    """
    Retrieve documents with metadata pre-filtering for Yuno integration docs.

    Args:
        query: The search query string
        top_k: Number of documents to retrieve
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
        verbose: Print filter details

    Returns:
        List of relevant document chunks
    """
    vector_store, client = get_vector_store()

    try:
        # Build pre-filter
        pre_filter = build_pre_filter(
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
        )

        if verbose and pre_filter:
            print(f"   Pre-filter: {pre_filter}")

        # Perform filtered similarity search
        if pre_filter:
            results = vector_store.similarity_search(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search(
                query=query,
                k=top_k
            )

        return results
    finally:
        client.close()


def retrieve_with_filter_and_scores(
    query: str,
    top_k: int = DEFAULT_TOP_K,
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
) -> list:
    """
    Retrieve documents with metadata pre-filtering and similarity scores.

    Returns:
        List of tuples (document, score)
    """
    vector_store, client = get_vector_store()

    try:
        pre_filter = build_pre_filter(
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
        )

        if pre_filter:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )

        return results
    finally:
        client.close()


def retrieve_hybrid(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filtered_ratio: float = 0.6,
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
) -> list:
    """
    Hybrid retrieval: Combines filtered and unfiltered results for best precision.

    This approach ensures that:
    1. You get targeted results from filters (high specificity)
    2. You don't miss relevant cross-domain docs (high recall)
    3. Deduplication prevents showing the same doc twice

    Args:
        query: The search query string
        top_k: Total number of documents to return (default: 5)
        filtered_ratio: Ratio of filtered to unfiltered docs (default: 0.6 = 60% filtered, 40% unfiltered)
        ... (same filter parameters as retrieve_with_filter)
        verbose: Print retrieval details

    Returns:
        List of top_k deduplicated documents, combining filtered + unfiltered results

    Example:
        # Get 5 docs: 3 filtered (60%) + 2 unfiltered (40%)
        docs = retrieve_hybrid(
            query="Stripe webhooks",
            top_k=5,
            providers=["stripe"],
            has_technical_content=True
        )
    """
    # Check if any filters are actually provided
    has_filters = any([
        document_source, document_type, providers, payment_methods, countries,
        services, severity, platform, has_error_codes, has_testing_instructions,
        has_technical_content, created_date_range, updated_date_range, ticket_id, page_id
    ])

    # If no filters, just do regular unfiltered retrieval
    if not has_filters:
        if verbose:
            print("   📋 Hybrid mode: No filters provided, using unfiltered retrieval")
        return retrieve_with_filter(query=query, top_k=top_k, verbose=verbose)

    # Calculate split: how many filtered vs unfiltered docs to retrieve
    filtered_k = max(1, int(top_k * filtered_ratio))  # At least 1 filtered doc
    unfiltered_k = max(1, top_k - filtered_k)  # At least 1 unfiltered doc

    if verbose:
        print(f"   📋 Hybrid mode: Retrieving {filtered_k} filtered + {unfiltered_k} unfiltered docs")

    # Step 1: Get filtered results
    filtered_docs = retrieve_with_filter(
        query=query,
        top_k=filtered_k * 2,  # Retrieve more to account for potential overlap
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
        verbose=False
    )

    # Step 2: Get unfiltered results (for broader coverage)
    unfiltered_docs = retrieve_with_filter(
        query=query,
        top_k=unfiltered_k * 2,  # Retrieve more to account for overlap
        verbose=False
    )

    # Step 3: Deduplicate based on page_content (exact match)
    # Track seen content to avoid duplicates
    seen_content = set()
    merged_docs = []

    # First, add filtered docs (higher priority)
    for doc in filtered_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            merged_docs.append(doc)
            if len(merged_docs) >= top_k:
                break

    # Then, add unfiltered docs if we need more
    if len(merged_docs) < top_k:
        for doc in unfiltered_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                merged_docs.append(doc)
                if len(merged_docs) >= top_k:
                    break

    if verbose:
        filtered_count = min(len(filtered_docs), filtered_k)
        unfiltered_count = len(merged_docs) - filtered_count
        print(f"   ✅ Retrieved {len(merged_docs)} docs: {filtered_count} filtered + {unfiltered_count} unfiltered")

    return merged_docs[:top_k]


def format_retrieved_context(documents: list) -> str:
    """Format retrieved documents into a context string for the LLM."""
    context_parts = []

    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("document_source", "Unknown")
        doc_type = doc.metadata.get("document_type", "Unknown")
        providers = doc.metadata.get("providers", [])
        methods = doc.metadata.get("payment_methods", [])
        countries = doc.metadata.get("countries", [])
        services = doc.metadata.get("services", [])
        severity = doc.metadata.get("severity", "N/A")
        ticket_id = doc.metadata.get("ticket_id", "")
        page_id = doc.metadata.get("page_id", "")

        # Build identifier (ticket ID for Jira, page ID for Confluence)
        identifier = ticket_id if ticket_id else (f"Page {page_id}" if page_id else "Unknown")

        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source} | Type: {doc_type} | ID: {identifier}\n"
            f"Providers: {', '.join(providers) if providers else 'N/A'}\n"
            f"Payment Methods: {', '.join(methods) if methods else 'N/A'}\n"
            f"Countries: {', '.join(countries) if countries else 'N/A'}\n"
            f"Services: {', '.join(services) if services else 'N/A'}\n"
            f"Severity: {severity}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n---\n".join(context_parts)


def get_available_providers():
    """Get list of providers available in the collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        providers = collection.distinct("providers")
        return sorted([p for p in providers if p])
    finally:
        client.close()


def get_available_payment_methods():
    """Get list of payment methods available in the collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        methods = collection.distinct("payment_methods")
        return sorted([m for m in methods if m])
    finally:
        client.close()


def get_available_countries():
    """Get list of countries available in the collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        countries = collection.distinct("countries")
        return sorted([c for c in countries if c])
    finally:
        client.close()


def get_provider_counts():
    """Get counts of provider mentions in the collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        pipeline = [
            {"$unwind": "$providers"},
            {"$group": {"_id": "$providers", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results]
    finally:
        client.close()


def get_payment_method_counts():
    """Get counts of payment method mentions in the collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        pipeline = [
            {"$unwind": "$payment_methods"},
            {"$group": {"_id": "$payment_methods", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results]
    finally:
        client.close()


def get_document_type_counts():
    """Get counts of document types in the collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    try:
        pipeline = [
            {"$group": {"_id": "$document_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results]
    finally:
        client.close()


def debug_collection():
    """Debug function to check MongoDB collection status."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]

    print("=" * 50)
    print("DEBUG: MongoDB Collection Status")
    print("=" * 50)

    doc_count = collection.count_documents({})
    print(f"\nTotal documents in collection: {doc_count}")

    if doc_count > 0:
        sample = collection.find_one()
        print(f"\nSample document fields: {list(sample.keys())}")
        print(f"\nSample metadata:")
        print(f"  document_source: {sample.get('document_source', 'N/A')}")
        print(f"  document_type: {sample.get('document_type', 'N/A')}")
        print(f"  providers: {sample.get('providers', [])}")
        print(f"  payment_methods: {sample.get('payment_methods', [])}")
        print(f"  countries: {sample.get('countries', [])}")
        print(f"  services: {sample.get('services', [])}")
        print(f"  severity: {sample.get('severity', 'N/A')}")
        print(f"  has_error_codes: {sample.get('has_error_codes', 'N/A')}")
        print(f"  has_testing_instructions: {sample.get('has_testing_instructions', 'N/A')}")

        # Show document type counts
        print("\nDocument type counts:")
        for doc_type, count in get_document_type_counts():
            print(f"  {doc_type}: {count}")

        # Show provider counts
        print("\nTop 10 provider mentions:")
        for provider, count in get_provider_counts()[:10]:
            print(f"  {provider}: {count}")

        # Show payment method counts
        print("\nPayment method counts:")
        for method, count in get_payment_method_counts():
            print(f"  {method}: {count}")

    client.close()
    return doc_count


def main():
    """Test retrieval with various Yuno-specific filters."""
    print("=" * 60)
    print("Yuno Integrations RAG - Retrieval Test")
    print("=" * 60)

    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Debug collection first
    doc_count = debug_collection()

    if doc_count == 0:
        print("\n❌ No documents found! Run ingestion.py first.")
        return

    # Test queries with different filters
    print("\n" + "=" * 60)
    print("Running Filtered Retrieval Tests")
    print("=" * 60)

    # Test 1: No filter (baseline)
    test_query = "How do I configure Fintoc webhooks?"
    print("\n📝 Test 1: No filter (baseline)")
    print(f"   Query: {test_query}")
    results = retrieve_with_filter(test_query, top_k=3, verbose=True)
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        providers = doc.metadata.get('providers', [])
        doc_type = doc.metadata.get('document_type', 'N/A')
        print(f"   - Type: {doc_type} | Providers: {', '.join(providers[:2])}")

    # Test 2: Filter by provider
    print("\n📝 Test 2: Filter by providers=['fintoc']")
    test_query = "How to handle payment errors?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        providers=["fintoc"],
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        doc_type = doc.metadata.get('document_type', 'N/A')
        ticket_id = doc.metadata.get('ticket_id', '')
        print(f"   - Type: {doc_type} | Ticket: {ticket_id if ticket_id else 'N/A'}")

    # Test 3: Filter by payment method
    print("\n📝 Test 3: Filter by payment_methods=['PIX']")
    test_query = "How to test PIX payments?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        payment_methods=["PIX"],
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        providers = doc.metadata.get('providers', [])
        countries = doc.metadata.get('countries', [])
        print(f"   - Providers: {', '.join(providers)} | Countries: {', '.join(countries)}")

    # Test 4: Filter by document source + severity
    print("\n📝 Test 4: Filter by document_source='jira' + severity='S1'")
    test_query = "Critical payment incidents"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        document_source="jira",
        severity="S1",
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        ticket_id = doc.metadata.get('ticket_id', 'N/A')
        services = doc.metadata.get('services', [])
        print(f"   - Ticket: {ticket_id} | Services: {', '.join(services)}")

    # Test 5: Combined filters (multiple providers + country)
    print("\n📝 Test 5: Combined filters (providers=['stripe', 'adyen'] + countries=['CL'])")
    test_query = "How to integrate card payments in Chile?"
    results = retrieve_with_filter(
        test_query,
        top_k=3,
        providers=["stripe", "adyen"],
        countries=["CL"],
        verbose=True
    )
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        doc_type = doc.metadata.get('document_type', 'N/A')
        methods = doc.metadata.get('payment_methods', [])
        print(f"   - Type: {doc_type} | Methods: {', '.join(methods)}")


if __name__ == "__main__":
    main()
