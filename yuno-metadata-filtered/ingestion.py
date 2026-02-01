"""
Yuno Integrations RAG - Ingestion Pipeline
Loads PDFs from Jira and Confluence, extracts metadata, chunks, and stores in MongoDB.

Enhanced metadata includes:
- source_file: Original PDF filename
- document_source: jira or confluence
- document_type: post_mortem, integration_guide, api_reference, troubleshooting, etc.
- page: Page number within the PDF
- chunk_index: Position of chunk within the document
- word_count: Number of words in chunk
- providers: Payment providers mentioned (fintoc, stripe, adyen, etc.)
- payment_methods: Payment methods mentioned (CARD, PIX, PSE, etc.)
- countries: Country codes mentioned (CL, MX, BR, etc.)
- services: Microservices mentioned (*-int pattern)
- severity: S1/S2/S3/S4 for Jira tickets
- platform: web, mobile, both
- has_error_codes: Whether chunk contains error codes
- has_testing_instructions: Whether chunk contains test scenarios
- has_technical_content: Whether chunk contains API endpoints/credentials
- ticket_id: For Jira documents (e.g., PFU-152)
- page_id: For Confluence documents
- created_date: Document creation date
- updated_date: Document last update date
"""

import os
import re
import certifi
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration
DB_NAME = "yuno_integrations_rag"
COLLECTION_NAME = "integration_docs"
INDEX_NAME = "integration_docs_index"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# =============================================================================
# DOCUMENT TYPES - Based on content analysis
# =============================================================================
DOCUMENT_TYPES = {
    "post_mortem": [
        "post mortem", "incident", "root cause", "affected services",
        "corrective measures", "preventive measures", "detection",
        "severity", "impact time"
    ],
    "integration_guide": [
        "dashboard section", "connection required fields", "credentials",
        "webhook", "payment method", "provider name", "integration"
    ],
    "api_reference": [
        "endpoint", "request", "response", "api reference",
        "method", "headers", "body", "parameters"
    ],
    "troubleshooting": [
        "error", "failed", "debug", "fix", "issue", "problem", "solution"
    ],
    "technical_specs": [
        "technical specs", "required fields", "environments",
        "production", "sandbox", "testing"
    ],
    "testing_guide": [
        "how to test", "test credentials", "test mode", "simulate"
    ]
}

# =============================================================================
# PAYMENT PROVIDERS - Payment gateways and processors
# =============================================================================
PROVIDERS = {
    "fintoc": ["fintoc", "fintoc-int"],
    "stripe": ["stripe", "stripe-int"],
    "adyen": ["adyen", "adyen-int"],
    "payu": ["payu", "payu-int"],
    "izipay": ["izipay", "izipay-int"],
    "dlocal": ["dlocal", "dlocal-int"],
    "mercadopago": ["mercadopago", "mercado pago", "mercadopago-int"],
    "paypal": ["paypal", "paypal-int"],
    "coinflow": ["coinflow", "coinflow-int"],
    "paymentes": ["paymentes", "paymentes-int"],
    "kushki": ["kushki", "kushki-int"],
    "wompi": ["wompi", "wompi-int"],
    "nuvei": ["nuvei", "nuvei-int"],
    "ebanx": ["ebanx", "ebanx-int"],
    "rappi": ["rappi", "rappi-int"],
    "whop": ["whop", "whop-int"],
}

# =============================================================================
# PAYMENT METHODS - Types of payment methods
# =============================================================================
PAYMENT_METHODS = [
    "CARD", "BANK_TRANSFER", "PIX", "PSE", "OXXO", "BOLETO",
    "WALLET", "CRYPTO", "SPEI", "CASH", "DEBIT_CARD", "CREDIT_CARD"
]

# =============================================================================
# COUNTRIES - ISO 3166-1 alpha-2 country codes
# =============================================================================
COUNTRIES = {
    "CL": ["chile", "chilean", "clp"],
    "MX": ["mexico", "mexican", "mxn"],
    "BR": ["brazil", "brazilian", "brl"],
    "CO": ["colombia", "colombian", "cop"],
    "PE": ["peru", "peruvian", "pen"],
    "AR": ["argentina", "argentinian", "ars"],
    "UY": ["uruguay", "uruguayan", "uyu"],
    "EC": ["ecuador", "ecuadorian", "usd"],
}

# =============================================================================
# SEVERITY LEVELS - For Jira tickets
# =============================================================================
SEVERITY_LEVELS = ["S1", "S2", "S3", "S4"]

# =============================================================================
# PLATFORM - Web or Mobile
# =============================================================================
PLATFORMS = ["web", "mobile"]


def get_pdf_files_recursive(root_dir: Path) -> dict[str, list[Path]]:
    """Get all PDF files from jira/ and confluence/ subdirectories."""
    pdf_files = {"jira": [], "confluence": []}

    jira_dir = root_dir / "jira"
    confluence_dir = root_dir / "confluence"

    if jira_dir.exists():
        pdf_files["jira"] = sorted(jira_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files['jira'])} Jira PDF files")

    if confluence_dir.exists():
        pdf_files["confluence"] = sorted(confluence_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files['confluence'])} Confluence PDF files")

    total = len(pdf_files["jira"]) + len(pdf_files["confluence"])
    print(f"Total: {total} PDF files")

    return pdf_files


def detect_document_type(text: str, document_source: str) -> str:
    """Detect document type based on content and source."""
    text_lower = text.lower()

    # Default types based on source
    if document_source == "jira":
        default_type = "post_mortem"
    else:  # confluence
        default_type = "integration_guide"

    # Score each document type
    scores = {}
    for doc_type, keywords in DOCUMENT_TYPES.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        scores[doc_type] = score

    # Return type with highest score, or default if no matches
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return default_type


def detect_providers(text: str) -> list[str]:
    """Detect payment providers mentioned in text."""
    text_lower = text.lower()
    detected = []

    for provider, variations in PROVIDERS.items():
        for variation in variations:
            if variation in text_lower:
                detected.append(provider)
                break

    return detected


def detect_payment_methods(text: str) -> list[str]:
    """Detect payment methods mentioned in text."""
    text_lower = text.lower()
    detected = []

    for method in PAYMENT_METHODS:
        # Check for the method name (case-insensitive)
        if method.lower() in text_lower or method.replace("_", " ").lower() in text_lower:
            detected.append(method)

    return detected


def detect_countries(text: str) -> list[str]:
    """Detect country codes mentioned in text."""
    text_lower = text.lower()
    detected = []

    for country_code, keywords in COUNTRIES.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(country_code)
                break

    return detected


def detect_services(text: str) -> list[str]:
    """Detect microservice names (pattern: *-int)."""
    # Find all words ending with -int
    pattern = r'\b(\w+)-int\b'
    matches = re.findall(pattern, text.lower())

    # Return unique service names
    services = [f"{match}-int" for match in set(matches)]
    return services


def detect_severity(text: str) -> str:
    """Detect severity level (S1/S2/S3/S4) from Jira tickets."""
    text_upper = text.upper()

    for severity in SEVERITY_LEVELS:
        if severity in text_upper or f"{severity} -" in text_upper:
            return severity

    return None


def detect_platform(text: str) -> list[str]:
    """Detect platform support (web, mobile, both)."""
    text_lower = text.lower()
    detected = []

    if "mobile" in text_lower or "app" in text_lower:
        detected.append("mobile")
    if "web" in text_lower or "website" in text_lower or "browser" in text_lower:
        detected.append("web")

    if len(detected) == 2:
        return ["both"]
    return detected if detected else []


def has_error_codes(text: str) -> bool:
    """Check if text contains error codes or status mappings."""
    text_lower = text.lower()
    indicators = [
        "error code", "error_code", "status code", "status mapping",
        "error.error_code", "error_reason", "failed", "declined"
    ]
    return any(indicator in text_lower for indicator in indicators)


def has_testing_instructions(text: str) -> bool:
    """Check if text contains testing instructions."""
    text_lower = text.lower()
    indicators = [
        "how to test", "test credential", "test mode", "sandbox",
        "test account", "simulate", "test payment"
    ]
    return any(indicator in text_lower for indicator in indicators)


def has_technical_content(text: str) -> bool:
    """Check if text contains technical content (endpoints, credentials)."""
    text_lower = text.lower()
    indicators = [
        "endpoint", "api", "credential", "secret key", "public key",
        "webhook", "https://", "authorization", "bearer"
    ]
    return any(indicator in text_lower for indicator in indicators)


def extract_ticket_id(filename: str) -> str:
    """Extract Jira ticket ID from filename (e.g., PFU-152, CORECM-13628)."""
    # Pattern: Letters-Numbers
    match = re.match(r'([A-Z]+-\d+)', filename.upper())
    return match.group(1) if match else None


def extract_page_id(filename: str) -> str:
    """Extract Confluence page ID from filename (e.g., 1025376259)."""
    # Pattern: Numbers at start of filename
    match = re.match(r'(\d+)', filename)
    return match.group(1) if match else None


def extract_dates_from_text(text: str) -> tuple[str, str]:
    """Extract created and updated dates from document header."""
    created_date = None
    updated_date = None

    # Look for "Created:" or "Created Date:" patterns
    created_match = re.search(r'Created:?\s*(\d{4}-\d{2}-\d{2}T[\d:.-]+)', text)
    if created_match:
        created_date = created_match.group(1)

    # Look for "Updated:" patterns
    updated_match = re.search(r'Updated:?\s*(\d{4}-\d{2}-\d{2}T[\d:.-]+)', text)
    if updated_match:
        updated_date = updated_match.group(1)

    return created_date, updated_date


def extract_metadata(
    text: str,
    pdf_path: Path,
    document_source: str,
    page: int,
    chunk_idx: int
) -> dict:
    """Extract all metadata for a chunk using fast string matching."""

    # Basic metadata
    word_count = len(text.split())

    # Document type detection
    document_type = detect_document_type(text, document_source)

    # Provider and payment method detection
    providers = detect_providers(text)
    payment_methods = detect_payment_methods(text)
    countries = detect_countries(text)
    services = detect_services(text)

    # Severity (for Jira documents)
    severity = detect_severity(text) if document_source == "jira" else None

    # Platform detection
    platform = detect_platform(text)

    # Content flags
    has_errors = has_error_codes(text)
    has_testing = has_testing_instructions(text)
    has_technical = has_technical_content(text)

    # Extract ticket/page ID
    ticket_id = None
    page_id = None
    if document_source == "jira":
        ticket_id = extract_ticket_id(pdf_path.stem)
    else:
        page_id = extract_page_id(pdf_path.stem)

    # Extract dates (only from first chunk for efficiency)
    created_date, updated_date = None, None
    if chunk_idx == 0:
        created_date, updated_date = extract_dates_from_text(text)

    return {
        "source_file": pdf_path.name,
        "document_source": document_source,
        "document_type": document_type,
        "page": page,
        "chunk_index": chunk_idx,
        "word_count": word_count,
        "providers": providers,
        "payment_methods": payment_methods,
        "countries": countries,
        "services": services,
        "severity": severity,
        "platform": platform,
        "has_error_codes": has_errors,
        "has_testing_instructions": has_testing,
        "has_technical_content": has_technical,
        "ticket_id": ticket_id,
        "page_id": page_id,
        "created_date": created_date,
        "updated_date": updated_date,
    }


def load_and_chunk_pdfs_with_metadata(
    pdf_files_by_source: dict[str, list[Path]]
) -> list:
    """Load PDFs and split into chunks with rich metadata."""
    all_documents = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    for document_source, pdf_files in pdf_files_by_source.items():
        print(f"\n📁 Processing {document_source.upper()} documents...")

        for pdf_path in pdf_files:
            print(f"  Processing: {pdf_path.name}")
            try:
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()

                # Split documents into chunks
                chunks = text_splitter.split_documents(documents)

                # PHASE 1: Extract metadata per-chunk
                for chunk_idx, chunk in enumerate(chunks):
                    # Get page from existing metadata
                    page = int(chunk.metadata.get("page", 0))

                    # Extract all metadata
                    metadata = extract_metadata(
                        text=chunk.page_content,
                        pdf_path=pdf_path,
                        document_source=document_source,
                        page=page,
                        chunk_idx=chunk_idx
                    )

                    # Update chunk metadata
                    chunk.metadata.update(metadata)

                # PHASE 2: Aggregate metadata across ALL chunks of this document
                # Collect all providers, methods, countries, services mentioned ANYWHERE in the document
                doc_level_providers = set()
                doc_level_methods = set()
                doc_level_countries = set()
                doc_level_services = set()
                doc_level_severity = None
                doc_level_has_errors = False
                doc_level_has_testing = False
                doc_level_has_technical = False

                for chunk in chunks:
                    doc_level_providers.update(chunk.metadata.get("providers", []))
                    doc_level_methods.update(chunk.metadata.get("payment_methods", []))
                    doc_level_countries.update(chunk.metadata.get("countries", []))
                    doc_level_services.update(chunk.metadata.get("services", []))

                    # Take first non-None severity found
                    if not doc_level_severity and chunk.metadata.get("severity"):
                        doc_level_severity = chunk.metadata.get("severity")

                    # OR logic for boolean flags
                    if chunk.metadata.get("has_error_codes"):
                        doc_level_has_errors = True
                    if chunk.metadata.get("has_testing_instructions"):
                        doc_level_has_testing = True
                    if chunk.metadata.get("has_technical_content"):
                        doc_level_has_technical = True

                # PHASE 3: Apply document-level metadata to ALL chunks
                for chunk in chunks:
                    chunk.metadata["providers"] = list(doc_level_providers)
                    chunk.metadata["payment_methods"] = list(doc_level_methods)
                    chunk.metadata["countries"] = list(doc_level_countries)
                    chunk.metadata["services"] = list(doc_level_services)
                    chunk.metadata["severity"] = doc_level_severity
                    chunk.metadata["has_error_codes"] = doc_level_has_errors
                    chunk.metadata["has_testing_instructions"] = doc_level_has_testing
                    chunk.metadata["has_technical_content"] = doc_level_has_technical

                    all_documents.append(chunk)

                # Summary for this PDF
                print(f"    -> {len(chunks)} chunks")
                if doc_level_providers:
                    print(f"       Providers: {list(doc_level_providers)}")
                if doc_level_methods:
                    methods_list = list(doc_level_methods)
                    print(f"       Methods: {methods_list[:3]}{'...' if len(methods_list) > 3 else ''}")
                if doc_level_services:
                    print(f"       Services: {list(doc_level_services)}")

            except Exception as e:
                print(f"    -> Error processing {pdf_path.name}: {e}")

    print(f"\n✅ Total chunks created: {len(all_documents)}")
    return all_documents


def setup_mongodb_collection():
    """Set up MongoDB collection for vector storage."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    # Create collection if it doesn't exist
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        # Clear existing documents for fresh ingestion
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in: {COLLECTION_NAME}")

    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: list):
    """Create embeddings and store in MongoDB Atlas Vector Search."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    print("\n🔄 Creating embeddings and storing in MongoDB...")

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )

    print(f"✅ Successfully stored {len(documents)} document chunks in MongoDB")
    return vector_store


def print_metadata_summary(collection):
    """Print summary of metadata in the collection."""
    print("\n📊 Metadata Summary:")

    # Count by document source
    pipeline = [
        {"$group": {"_id": "$document_source", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    sources = list(collection.aggregate(pipeline))
    print(f"\n   By Document Source:")
    for s in sources:
        if s["_id"]:
            print(f"   {s['_id']}: {s['count']} chunks")

    # Count by document type
    pipeline = [
        {"$group": {"_id": "$document_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    doc_types = list(collection.aggregate(pipeline))
    print(f"\n   By Document Type:")
    for t in doc_types:
        if t["_id"]:
            print(f"   {t['_id']}: {t['count']} chunks")

    # Top providers
    pipeline = [
        {"$unwind": "$providers"},
        {"$group": {"_id": "$providers", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    providers = list(collection.aggregate(pipeline))
    print(f"\n   Top Providers:")
    for p in providers:
        print(f"   {p['_id']}: {p['count']} chunks")

    # Top payment methods
    pipeline = [
        {"$unwind": "$payment_methods"},
        {"$group": {"_id": "$payment_methods", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ]
    methods = list(collection.aggregate(pipeline))
    print(f"\n   Top Payment Methods:")
    for m in methods:
        print(f"   {m['_id']}: {m['count']} chunks")

    # Countries
    pipeline = [
        {"$unwind": "$countries"},
        {"$group": {"_id": "$countries", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    countries = list(collection.aggregate(pipeline))
    print(f"\n   Countries:")
    for c in countries:
        print(f"   {c['_id']}: {c['count']} chunks")

    # Services
    pipeline = [
        {"$unwind": "$services"},
        {"$group": {"_id": "$services", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    services = list(collection.aggregate(pipeline))
    if services:
        print(f"\n   Top Services:")
        for svc in services[:10]:
            print(f"   {svc['_id']}: {svc['count']} chunks")

    # Content flags
    error_codes_count = collection.count_documents({"has_error_codes": True})
    testing_count = collection.count_documents({"has_testing_instructions": True})
    technical_count = collection.count_documents({"has_technical_content": True})
    total_count = collection.count_documents({})

    print(f"\n   Content Flags:")
    print(f"   Has Error Codes: {error_codes_count}/{total_count} chunks ({100*error_codes_count/total_count:.1f}%)")
    print(f"   Has Testing Instructions: {testing_count}/{total_count} chunks ({100*testing_count/total_count:.1f}%)")
    print(f"   Has Technical Content: {technical_count}/{total_count} chunks ({100*technical_count/total_count:.1f}%)")


def print_vector_search_index_instructions():
    """Print instructions for creating the vector search index in MongoDB Atlas."""
    print("\n" + "=" * 70)
    print("IMPORTANT: Create Vector Search Index in MongoDB Atlas")
    print("=" * 70)
    print(f"""
Create a vector search index with these filter fields:

1. Go to MongoDB Atlas → Your Cluster → Atlas Search → Create Search Index
2. Select "Atlas Vector Search" → JSON Editor
3. Use this configuration:

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }},
    {{
      "type": "filter",
      "path": "document_source"
    }},
    {{
      "type": "filter",
      "path": "document_type"
    }},
    {{
      "type": "filter",
      "path": "providers"
    }},
    {{
      "type": "filter",
      "path": "payment_methods"
    }},
    {{
      "type": "filter",
      "path": "countries"
    }},
    {{
      "type": "filter",
      "path": "services"
    }},
    {{
      "type": "filter",
      "path": "severity"
    }},
    {{
      "type": "filter",
      "path": "platform"
    }},
    {{
      "type": "filter",
      "path": "has_error_codes"
    }},
    {{
      "type": "filter",
      "path": "has_testing_instructions"
    }},
    {{
      "type": "filter",
      "path": "has_technical_content"
    }}
  ]
}}

4. Set the index name to: {INDEX_NAME}
5. Select database: {DB_NAME}
6. Select collection: {COLLECTION_NAME}

Wait for the index to become "Active" before running queries.
""")
    print("=" * 70)


def main():
    """Main ingestion pipeline for Yuno integration documentation."""
    print("=" * 70)
    print("Yuno Integrations RAG - Ingestion Pipeline")
    print("Processing Jira and Confluence PDFs")
    print("=" * 70)

    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Get path to rag-knowledge-base directory
    script_dir = Path(__file__).parent
    knowledge_base_dir = script_dir.parent.parent / "rag-knowledge-base" / "data"

    if not knowledge_base_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")

    # Step 1: Get PDF files from jira/ and confluence/ subdirectories
    pdf_files_by_source = get_pdf_files_recursive(knowledge_base_dir)

    total_files = sum(len(files) for files in pdf_files_by_source.values())
    if total_files == 0:
        raise ValueError("No PDF files found in knowledge base directory")

    # Step 2: Load, chunk PDFs, and extract metadata
    print("\n🔍 Extracting metadata using fast string matching...")
    documents = load_and_chunk_pdfs_with_metadata(pdf_files_by_source)

    # Step 3: Setup MongoDB
    print("\n🗄️  Setting up MongoDB...")
    client, collection = setup_mongodb_collection()

    try:
        # Step 4: Create embeddings and store in vector database
        vector_store = create_vector_store(collection, documents)

        # Show metadata summary
        print_metadata_summary(collection)

        print("\n✅ Ingestion complete!")
        print(f"   Database: {DB_NAME}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Documents stored: {len(documents)}")

        # Print instructions for creating the vector search index
        print_vector_search_index_instructions()

    finally:
        client.close()


if __name__ == "__main__":
    main()
