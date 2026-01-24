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

(Coming soon - US-002)

### 3. Vector Database Indexing

(Coming soon - US-003)

### 4. Query Interface

(Coming soon - US-006)

## Project Structure

```
integrations-rag/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pdf_ingestion.py      # PDF document ingestion pipeline
│   ├── chunking/                  # (Coming soon)
│   ├── vectordb/                  # (Coming soon)
│   ├── retrieval/                 # (Coming soon)
│   └── generation/                # (Coming soon)
├── tests/
│   └── test_pdf_ingestion.py     # Tests for PDF ingestion
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

### In Progress

The following user stories are planned:

- **US-002**: Document chunking strategy
- **US-003**: Vector database setup and indexing
- **US-004**: Metadata-filtered retrieval
- **US-005**: LLM-based answer generation
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
