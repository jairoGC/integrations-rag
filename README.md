# Yuno RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying payment provider documentation and Jira tickets using metadata-filtered vector search.

## Overview

This project builds a RAG pipeline that:
- Ingests PDF documents (provider documentation and Jira tickets)
- Chunks documents into semantic units
- Indexes content in a vector database (MongoDB Atlas)
- Retrieves relevant information using metadata-filtered similarity search
- Generates natural language answers using LLMs (Claude/GPT-4)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- MongoDB Atlas account (for vector database)
- OpenAI API key or Anthropic API key (for embeddings and LLM)

### Setup

1. Clone the repository:
```bash
cd /path/to/integrations-rag
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt indicating the virtual environment is active.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (create a `.env` file):
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
MONGODB_URI=your_mongodb_atlas_connection_string
```

## Usage

### 1. Document Ingestion

Ingest PDF files from a directory:

```bash
python -m src.ingestion.pdf_ingestion /path/to/pdf/directory
```

**Filename Convention:**

For proper metadata extraction, name your PDF files following this pattern:
```
{document_type}_{provider_name}_{date}.pdf
```

Examples:
- `provider_doc_Stripe_20240101.pdf`
- `jira_ticket_PayPal_2024-01-15.pdf`

**Document Types:**
- `provider_doc` - Payment provider documentation
- `jira_ticket` - Jira ticket exports

**Python API:**
```python
from src.ingestion import PDFIngestionPipeline

pipeline = PDFIngestionPipeline()
documents = pipeline.ingest_directory("/path/to/pdfs")

# Access documents
for doc in documents:
    print(f"File: {doc.metadata['filename']}")
    print(f"Provider: {doc.metadata['provider_name']}")
    print(f"Type: {doc.metadata['document_type']}")
    print(f"Content length: {len(doc.content)} chars")
```

### 2. Document Chunking

Chunk documents into semantic units:

**Python API:**
```python
from src.chunking import TextChunker

chunker = TextChunker(chunk_size_tokens=1000, overlap_tokens=200)
chunks = chunker.chunk_document(content, metadata)

# Get statistics
stats = chunker.get_chunk_statistics()
print(f"Created {stats['count']} chunks")
print(f"Average size: {stats['avg_size']} characters")
```

### 3. Vector Database Indexing

Index chunks with embeddings in MongoDB Atlas:

**Python API:**
```python
from src.indexing import VectorStore

# Initialize vector store (requires OPENAI_API_KEY and MONGODB_URI env vars)
vector_store = VectorStore(
    database_name="yuno_rag",
    collection_name="chunks"
)

# Index a single chunk
vector_store.index_chunk(
    chunk_id="chunk_001",
    content="Your chunk content here",
    metadata={"document_type": "provider_doc", "provider_name": "Stripe"}
)

# Index multiple chunks in batch
chunks = [
    {
        "chunk_id": "chunk_001",
        "content": "Content 1",
        "metadata": {"document_type": "provider_doc", "provider_name": "Stripe"}
    },
    # ... more chunks
]
indexed_ids = vector_store.index_batch(chunks)

# Query chunks by metadata
stripe_chunks = vector_store.get_chunks_by_metadata({"provider_name": "Stripe"})

# Clean up
vector_store.close()
```

### 4. Metadata-Filtered Retrieval

Retrieve relevant chunks using metadata-filtered vector search:

**Python API:**
```python
from src.retrieval import Retriever
from src.indexing import VectorStore

# Initialize vector store and retriever
vector_store = VectorStore()
retriever = Retriever(vector_store)

# Simple retrieval with metadata filtering
results = retriever.retrieve("How do Stripe refunds work?", top_k=5)

# The query classifier automatically detects "Stripe" and applies metadata filters
for result in results:
    print(f"Score: {result.similarity_score:.4f}")
    print(f"Provider: {result.metadata['provider_name']}")
    print(f"Content: {result.content[:100]}...")
    print()

# Retrieval with fallback (tries filtered, falls back to unfiltered)
results = retriever.retrieve_with_fallback("How do refunds work?", top_k=5)

# Manual filter control
results = retriever.retrieve(
    "What is the API documentation?",
    top_k=10,
    apply_filters=False,  # Disable automatic filtering
    min_similarity=0.7     # Minimum similarity threshold
)

# Query classification only
classification = retriever.query_classifier.classify_query("Stripe and PayPal fees")
print(classification)
# {'providers': {'Stripe', 'Paypal'}, 'document_type': None, 'has_filters': True}
```

### 5. LLM-based Answer Generation

Generate natural language answers with source citations:

**Python API:**
```python
from src.generation import AnswerGenerator
from src.retrieval import Retriever
from src.indexing import VectorStore

# Initialize components
vector_store = VectorStore()
retriever = Retriever(vector_store)

# Initialize answer generator (supports both Claude and GPT-4)
# Using Claude:
generator = AnswerGenerator(model="claude-3-sonnet-20240229")
# Or using GPT-4:
# generator = AnswerGenerator(model="gpt-4")

# Retrieve relevant chunks
query = "How do Stripe refunds work?"
results = retriever.retrieve(query, top_k=5)

# Convert retrieval results to chunks format
chunks = [
    {
        "chunk_id": r.chunk_id,
        "content": r.content,
        "metadata": r.metadata,
    }
    for r in results
]

# Generate answer with citations
answer = generator.generate_answer(
    query=query,
    retrieved_chunks=chunks,
    min_chunks=1  # Minimum chunks required
)

# Display formatted answer with citations
print(generator.format_answer_with_citations(answer))

# Access answer components
print(f"Has answer: {answer.has_answer}")
print(f"Model used: {answer.model}")
print(f"Number of sources: {len(answer.sources)}")
```

### 6. Query Interface

(Coming soon - US-006)

## Project Structure

```
integrations-rag/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_ingestion.py      # PDF document ingestion pipeline
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── text_chunker.py        # Text chunking strategy
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── vector_store.py        # MongoDB Atlas vector store
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py           # Metadata-filtered retrieval
│   └── generation/
│       ├── __init__.py
│       └── answer_generator.py    # LLM-based answer generation
├── tests/
│   ├── test_pdf_ingestion.py     # Tests for PDF ingestion
│   ├── test_text_chunker.py      # Tests for text chunking
│   ├── test_vector_store.py      # Tests for vector store
│   ├── test_retriever.py         # Tests for retrieval
│   └── test_answer_generator.py  # Tests for answer generation
├── integrations-rag/
│   ├── prd.json                   # Product requirements document
│   └── progress.txt               # Development progress log
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Tool configurations
└── README.md                      # This file
```

## Development

### Running Tests

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_pdf_ingestion.py
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

Run tests in verbose mode:
```bash
pytest -v
```

### Type Checking

Run mypy type checker:
```bash
mypy src/
```

Check specific file:
```bash
mypy src/ingestion/pdf_ingestion.py
```

### Linting

Run pylint:
```bash
pylint src/
```

Check specific file:
```bash
pylint src/ingestion/pdf_ingestion.py
```

### Code Formatting

Format code with black:
```bash
black src/ tests/
```

Check formatting without making changes:
```bash
black --check src/ tests/
```

## Configuration

### pyproject.toml

Tool configurations are defined in `pyproject.toml`:

- **mypy**: Strict type checking with Python 3.10
- **pylint**: Custom rules for code quality
- **black**: Code formatting (88 character line length)
- **pytest**: Test discovery and execution

### Quality Standards

- All code must pass mypy type checking (no errors)
- Pylint score must be 9.0 or higher
- All tests must pass
- Test coverage should be >80%

## Current Status

### Completed Features

✅ **US-001**: Document ingestion pipeline for PDFs
- Accepts directory of PDF files
- Extracts text content and metadata
- Handles errors gracefully
- Comprehensive test coverage

✅ **US-002**: Document chunking strategy
- Recursive character splitter (1000 token chunks, 200 token overlap)
- Section header extraction and preservation
- Metadata inheritance
- Chunk statistics

✅ **US-003**: Vector database setup and indexing
- MongoDB Atlas integration
- OpenAI text-embedding-3-small embeddings (1536 dimensions)
- Metadata indexes on document_type and provider_name
- Batch upsert operations
- 23 comprehensive tests

✅ **US-004**: Metadata-filtered retrieval
- Query classification (provider and document type extraction)
- Cosine similarity search
- Automatic metadata filtering
- Fallback to unfiltered search
- Top-k retrieval with configurable similarity threshold
- 36 comprehensive tests

✅ **US-005**: LLM-based answer generation
- Support for Claude (3-Opus, 3-Sonnet) and GPT (4, 3.5-turbo)
- System prompt enforcing context-only answers
- Automatic source citation formatting
- Graceful handling of insufficient information
- Configurable minimum chunk threshold
- 21 comprehensive tests

### In Progress

The following user stories are planned:

- **US-006**: Query interface (CLI/API)
- **US-007-011**: Evaluation metrics (precision, recall, latency, groundedness)

See `integrations-rag/prd.json` for full details.

## Example Workflow

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Prepare your PDFs:**
   ```bash
   mkdir -p data/pdfs
   # Add your PDFs with proper naming convention
   ```

3. **Ingest documents:**
   ```bash
   python -m src.ingestion.pdf_ingestion data/pdfs
   ```

4. **Expected output:**
   ```
   ============================================================
   Ingestion Summary
   ============================================================
   Total documents ingested: 5

   By document type:
     provider_doc: 3
     jira_ticket: 2

   Sample documents:
   1. provider_doc_Stripe_20240101.pdf
      Type: provider_doc
      Provider: Stripe
      Date: 2024-01-01
      Content length: 15234 characters
   ```

## Troubleshooting

### Virtual Environment Issues

**Activating the virtual environment:**

Make sure your virtual environment is activated before running any commands:
```bash
# Check if virtual environment is active (you should see (venv) in prompt)
# If not active, activate it:
source venv/bin/activate
```

**Deactivating the virtual environment:**
```bash
deactivate
```

**Virtual environment not found:**

If you get an error that `venv` doesn't exist:
```bash
# Create it first
python3 -m venv venv

# Then activate it
source venv/bin/activate
```

**Wrong Python version in virtual environment:**

Make sure you create the venv with Python 3.10+:
```bash
# Check your Python version
python3 --version

# Create venv with specific Python version
python3.10 -m venv venv
```

### PDF Text Extraction Fails

If text extraction fails for a PDF:
- Ensure the PDF is not encrypted or password-protected
- Check that the PDF contains actual text (not just images)
- Try re-exporting the PDF with text layer enabled

### Metadata Not Extracted Correctly

If metadata is not parsed correctly:
- Check your filename follows the convention: `{type}_{provider}_{date}.pdf`
- Ensure document type is exactly `provider_doc` or `jira_ticket`
- Date format should be `YYYYMMDD` or `YYYY-MM-DD`

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root
cd /path/to/integrations-rag

# Install dependencies
pip install -r requirements.txt

# Run Python modules with -m flag
python -m src.ingestion.pdf_ingestion
```

## Contributing

1. Create a feature branch from `main`
2. Implement your feature following the quality standards
3. Run tests and quality checks
4. Commit with descriptive messages
5. Submit for review

## License

(Add your license here)

## Contact

For questions or issues, please contact the Yuno team.
