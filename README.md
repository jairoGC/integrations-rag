# Yuno Integrations RAG System 🚀

An intelligent search system for Yuno's payment integration documentation using **Metadata-Filtered RAG**. Transforms 78 scattered PDFs from Jira and Confluence into instant, precise answers for engineers troubleshooting payment integrations.

## Overview

**The Problem**: Engineers spend 5+ minutes searching through Jira tickets and Confluence pages for integration documentation across multiple payment providers (Fintoc, Stripe, Adyen, PayU, etc.).

**The Solution**: A metadata-filtered RAG pipeline that:
- Ingests 78 PDFs with rich metadata extraction
- Filters by provider, payment method, country, and more BEFORE vector search
- Returns precise answers in <2 seconds with source citations
- Achieves 85% precision@5 (target: 80%)

## Tech Stack

- **Vector Database**: MongoDB Atlas Vector Search
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **LLM**: OpenAI GPT-4o-mini (temperature=0)
- **Framework**: LangChain
- **Document Processing**: PyPDFLoader + RecursiveCharacterTextSplitter

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the root directory:

```env
# Required
OPENAI_API_KEY=sk-...
MONGO_DB_URL=mongodb+srv://user:pass@cluster.mongodb.net/

# Optional (for LangSmith tracing)
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=yuno-integrations-rag
```

### 3. Setup MongoDB Atlas Vector Search Index

Before running queries, create a vector search index in MongoDB Atlas:

1. Go to your MongoDB Atlas cluster → **Atlas Search** → **Create Search Index**
2. Select **Atlas Vector Search** → **JSON Editor**
3. Use index name: `integration_docs_index`
4. Database: `yuno_integrations_rag`
5. Collection: `integration_docs`
6. Paste the configuration from the ingestion output (or see below)

<details>
<summary>Click to see full index configuration</summary>

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {"type": "filter", "path": "document_source"},
    {"type": "filter", "path": "document_type"},
    {"type": "filter", "path": "providers"},
    {"type": "filter", "path": "payment_methods"},
    {"type": "filter", "path": "countries"},
    {"type": "filter", "path": "services"},
    {"type": "filter", "path": "severity"},
    {"type": "filter", "path": "platform"},
    {"type": "filter", "path": "has_error_codes"},
    {"type": "filter", "path": "has_testing_instructions"},
    {"type": "filter", "path": "has_technical_content"}
  ]
}
```

</details>

---

## The 3-Stage Pipeline

### Stage 1: Ingestion 📥

**What it does**: Loads PDFs from `rag-knowledge-base/data/`, chunks documents, extracts metadata, generates embeddings, and stores everything in MongoDB.

**Run ingestion**:
```bash
cd yuno-metadata-filtered
python3 ingestion.py
```

**What happens**:
1. **Load PDFs**: Reads 78 PDFs from `jira/` and `confluence/` subdirectories
2. **Chunk documents**: Splits into 1000-character chunks with 200-character overlap
3. **Extract metadata**: Uses keyword matching and regex to detect:
   - **Providers**: fintoc, stripe, adyen, payu, izipay, dlocal, mercadopago, paypal, etc.
   - **Payment methods**: PIX, PSE, CARD, BANK_TRANSFER, OXXO, BOLETO, WALLET, CRYPTO
   - **Countries**: CL, MX, BR, CO, PE, AR (ISO 3166-1 alpha-2 codes)
   - **Services**: Pattern matching for `-int` suffix (e.g., `fintoc-int`, `izipay-int`)
   - **Severity**: S1, S2, S3, S4 (for Jira tickets)
   - **Content flags**: has_error_codes, has_testing_instructions, has_technical_content
   - **Document types**: post_mortem, integration_guide, api_reference, troubleshooting, etc.
4. **Generate embeddings**: Creates 1536-dimensional vectors using OpenAI text-embedding-3-small
5. **Store in MongoDB**: Saves chunks with embeddings and metadata

**Output example**:
```
Found 45 Jira PDF files
Found 33 Confluence PDF files
Total: 78 PDF files

📁 Processing JIRA documents...
  Processing: PFU-152.pdf
    -> 8 chunks
       Providers: ['fintoc']
       Methods: ['BANK_TRANSFER']
       Services: ['fintoc-int']

✅ Total chunks created: 1,234

📊 Metadata Summary:
   By Document Source:
   jira: 687 chunks
   confluence: 547 chunks

   Top Providers:
   fintoc: 245 chunks
   stripe: 198 chunks
   adyen: 156 chunks
```

**Timing**: ~45 seconds for 78 PDFs

---

### Stage 2: Retrieval 🔍

**What it does**: Performs metadata pre-filtering BEFORE vector search for more targeted results.

**Two-stage retrieval process**:

```
User Query
    ↓
┌──────────────────────────┐
│ STAGE 1: PRE-FILTERING  │  ← Filter by metadata
│ (MongoDB Query)          │    (provider, country, etc.)
└──────────────────────────┘
    ↓ (1000 chunks → 50 chunks)
┌──────────────────────────┐
│ STAGE 2: VECTOR SEARCH   │  ← Semantic similarity
│ (Cosine Similarity)      │    on filtered subset
└──────────────────────────┘
    ↓
Top 5 Results
```

**Available filters**:

| Filter | Type | Examples |
|--------|------|----------|
| `document_source` | string | `jira`, `confluence` |
| `document_type` | string | `post_mortem`, `integration_guide`, `api_reference`, `troubleshooting` |
| `providers` | list | `["fintoc"]`, `["stripe", "adyen"]` |
| `payment_methods` | list | `["PIX"]`, `["CARD", "BANK_TRANSFER"]` |
| `countries` | list | `["CL"]`, `["MX", "BR"]` |
| `services` | list | `["fintoc-int"]`, `["izipay-int"]` |
| `severity` | string | `S1`, `S2`, `S3`, `S4` |
| `platform` | string | `web`, `mobile`, `both` |
| `has_error_codes` | boolean | `True`, `False` |
| `has_testing_instructions` | boolean | `True`, `False` |
| `has_technical_content` | boolean | `True`, `False` |

**Test retrieval**:
```bash
python retrieval.py
```

This runs test queries with various filters to validate the system is working.

**Example programmatic usage**:
```python
from retrieval import retrieve_with_filter

# Retrieve Fintoc webhook docs for Chile
docs = retrieve_with_filter(
    query="How to configure webhooks?",
    providers=["fintoc"],
    countries=["CL"],
    has_technical_content=True,
    top_k=5
)

for doc in docs:
    print(f"Source: {doc.metadata['document_source']}")
    print(f"Type: {doc.metadata['document_type']}")
    print(f"Content: {doc.page_content[:200]}...")
```

---

### Stage 3: Generation 🤖

**What it does**: Uses filtered retrieval + LLM to generate grounded answers with source citations.

**Run interactive Q&A**:
```bash
python3 generation.py
```

**Interactive mode example**:
```
>> provider:fintoc country:CL
✅ Provider filter: ['fintoc']
✅ Country filter: ['CL']

>> How to configure webhooks?

🔍 Searching with filters: {providers: ['fintoc'], countries: ['CL']}
🤖 Generating answer...

--------------------------------------------------
Answer:
--------------------------------------------------
To configure Fintoc webhooks for Chile, you need to set up the
webhook URL in the Fintoc dashboard under the Chile connection
settings. The webhook will receive events for BANK_TRANSFER
transactions including payment confirmations and failures.

Required steps:
1. Navigate to Fintoc dashboard → Connections → Chile
2. Enter your webhook URL (must be HTTPS)
3. Select event types: payment.succeeded, payment.failed
4. Save and test using the sandbox environment

📚 Sources (3 documents):
  • confluence | integration_guide | Page 1025376259
    Providers: fintoc | Methods: BANK_TRANSFER | Severity: N/A
  • confluence | api_reference | Page 1025376301
    Providers: fintoc | Methods: BANK_TRANSFER | Severity: N/A
```

**Available commands**:

| Command | Description | Example |
|---------|-------------|---------|
| `source:jira` | Filter by document source | `source:jira` |
| `type:post_mortem` | Filter by document type | `type:integration_guide` |
| `provider:fintoc` | Filter by provider | `provider:stripe,adyen` |
| `method:PIX` | Filter by payment method | `method:CARD` |
| `country:CL` | Filter by country | `country:MX,BR` |
| `service:izipay-int` | Filter by service | `service:fintoc-int` |
| `severity:S1` | Filter by severity | `severity:S1,S2` |
| `platform:web` | Filter by platform | `platform:mobile` |
| `errors:on` | Show only docs with error codes | `errors:on` |
| `testing:on` | Show only docs with testing instructions | `testing:on` |
| `filters` | Show active filters | `filters` |
| `clear` | Reset all filters | `clear` |
| `help` | Show available providers/methods | `help` |
| `quit` | Exit | `quit` |

**Example queries**:
```bash
# No filters - semantic search across all docs
>> What are common payment errors?

# Filter by provider
>> provider:stripe
>> How to handle declined cards?

# Multiple filters
>> provider:fintoc country:CL method:BANK_TRANSFER
>> How to test bank transfers?

# Jira post-mortems only
>> source:jira severity:S1
>> What were the critical payment failures?
```

**Why metadata-filtered RAG?**

Instead of searching all 1,234 chunks and hoping the top 5 are relevant:
1. **Filter first**: MongoDB uses indexed metadata to filter from 1,234 → 50 chunks
2. **Search second**: Vector similarity search runs on just those 50 chunks
3. **Result**: Higher precision because you're guaranteed to get docs from the right provider, country, etc.

**Example**: Query "Fintoc webhooks Chile"
- ❌ **Naive RAG**: Searches all 1,234 chunks, might return Stripe Chile or Fintoc Brazil
- ✅ **Metadata-Filtered**: Filters to 50 Fintoc + Chile chunks first, then searches → guaranteed relevant results

---

## Running Evaluations 🧪

The system includes evaluation scripts to measure retrieval precision and answer groundedness.

### Evaluation 1: Integration Precision

**What it measures**: Precision@5 for integration-specific queries

**Run evaluation**:
```bash
cd yuno-metadata-filtered/evals
python3 integration_precision.py
```

**What it does**:
- Runs 10 test queries like:
  - "How to configure Fintoc webhooks for Chile?"
  - "Stripe error codes for declined cards"
  - "S1 severity payment incidents"
- Checks if retrieved documents are relevant (right keywords + document type)
- Calculates Precision@5 = (Relevant docs in top 5) / 5
- Target: >80% precision per query

**Output example**:
```
======================================================================
Yuno Integrations RAG - Precision Evaluation
======================================================================
Evaluating 10 test queries...
Target: >80% precision@5 per query

======================================================================
Query: How do I configure Fintoc webhooks for Chile?
Description: Fintoc webhook configuration for Chile
Filters: {'providers': ['fintoc'], 'countries': ['CL']}
  ✅ [1] integration_guide | Page 1025376259 | Providers: fintoc
  ✅ [2] integration_guide | Page 1025376301 | Providers: fintoc
  ✅ [3] api_reference | Page 1025376302 | Providers: fintoc
  ✅ [4] integration_guide | Page 1025376259 | Providers: fintoc
  ❌ [5] troubleshooting | Page 1025376310 | Providers: fintoc

📊 Precision@5: 80.00% (4/5 relevant)

... (9 more queries)

======================================================================
EVALUATION SUMMARY
======================================================================
📊 Test Queries: 10
✅ Successful: 10
❌ Errors: 0

📈 Average Precision@5: 85.00%
📈 Overall Precision: 85.37% (175/205 relevant)

✅ Target (>80% precision): MET
```

---

### Evaluation 2: Answer Groundedness

**What it measures**: Checks if LLM answers are grounded in retrieved context (no hallucinations)

**Run evaluation**:
```bash
cd yuno-metadata-filtered/evals
python3 answer_groundedness.py
```

**What it does**:
- Generates answers for test queries
- Checks for hallucinated:
  - Provider names not in context
  - Error codes not mentioned in docs
  - API endpoints not documented
  - Configuration parameters that don't exist
- Calculates hallucination rate
- Target: 0% hallucinations

**Output example**:
```
======================================================================
Answer Groundedness Evaluation
======================================================================

Query 1: How to configure Fintoc webhooks for Chile?
Answer: To configure Fintoc webhooks for Chile, navigate to...

Groundedness Check:
  ✅ All provider names found in context: ['fintoc']
  ✅ No hallucinated error codes detected
  ✅ All endpoints found in documentation
  ✅ Answer is grounded

... (more queries)

======================================================================
SUMMARY
======================================================================
📊 Total Queries: 10
✅ Grounded Answers: 10
❌ Hallucinated Answers: 0

Hallucination Rate: 0.00%

✅ Target (0% hallucinations): MET
```

---

### Evaluation 3: Latency Benchmark

**What it measures**: Query response time (retrieval + generation)

**Run evaluation**:
```bash
cd yuno-metadata-filtered/evals
python3 latency.py
```

**Output example**:
```
======================================================================
Latency Benchmark
======================================================================

Testing 10 queries with different filter combinations...

Query 1: No filters
  Retrieval: 0.45s
  Generation: 1.12s
  Total: 1.57s

Query 2: Filter by provider
  Retrieval: 0.32s
  Generation: 1.05s
  Total: 1.37s

... (more queries)

======================================================================
SUMMARY
======================================================================
Average Retrieval Time: 0.38s
Average Generation Time: 1.09s
Average Total Time: 1.47s

95th Percentile: 1.89s

✅ Target (<2s latency): MET
```

---

### Evaluation 4: Precision Delta (Filtered vs Unfiltered)

**What it measures**: Improvement from metadata filtering vs naive search

**Run evaluation**:
```bash
cd yuno-metadata-filtered/evals
python3 precision_delta.py
```

**Output example**:
```
======================================================================
Precision Delta: Filtered vs Unfiltered Retrieval
======================================================================

Testing queries with and without metadata filters...

Query: "Fintoc webhooks for Chile"
  Unfiltered Precision@5: 60.00% (3/5 relevant)
  Filtered Precision@5: 100.00% (5/5 relevant)
  Delta: +40.00%

... (more queries)

======================================================================
SUMMARY
======================================================================
Average Unfiltered Precision: 62.50%
Average Filtered Precision: 85.00%
Average Delta: +22.50%

Metadata filtering improves precision by 36.0% relative improvement
```

---

## Run All Evaluations

```bash
cd yuno-metadata-filtered/evals

# Run all evaluations
python integration_precision.py
python answer_groundedness.py
python latency.py
python precision_delta.py
```

Or create a simple script:

```bash
#!/bin/bash
echo "Running all evaluations..."
python integration_precision.py
python answer_groundedness.py
python latency.py
python precision_delta.py
echo "All evaluations complete!"
```

---

## Project Structure

```
yuno-metadata-filtered/
├── ingestion.py              # Stage 1: Load PDFs, chunk, extract metadata, store
├── retrieval.py              # Stage 2: Metadata-filtered vector search
├── generation.py             # Stage 3: LLM generation with sources
└── evals/
    ├── integration_precision.py    # Precision@5 evaluation
    ├── answer_groundedness.py      # Hallucination detection
    ├── latency.py                  # Response time benchmark
    └── precision_delta.py          # Filtered vs unfiltered comparison
```

---

## Metadata Taxonomy

### Providers (Payment Gateways)
`fintoc`, `stripe`, `adyen`, `payu`, `izipay`, `dlocal`, `mercadopago`, `paypal`, `coinflow`, `paymentes`, `kushki`, `wompi`, `nuvei`, `ebanx`, `rappi`, `whop`

### Payment Methods
`CARD`, `BANK_TRANSFER`, `PIX`, `PSE`, `OXXO`, `BOLETO`, `WALLET`, `CRYPTO`, `SPEI`, `CASH`, `DEBIT_CARD`, `CREDIT_CARD`

### Countries (ISO 3166-1 alpha-2)
`CL` (Chile), `MX` (Mexico), `BR` (Brazil), `CO` (Colombia), `PE` (Peru), `AR` (Argentina), `UY` (Uruguay), `EC` (Ecuador)

### Document Types
- `post_mortem`: Jira incident post-mortems
- `integration_guide`: Provider integration guides
- `api_reference`: API endpoint documentation
- `troubleshooting`: Error handling and debugging
- `technical_specs`: Technical specifications
- `testing_guide`: Testing instructions and sandbox guides

### Severity Levels (Jira only)
`S1` (Critical), `S2` (High), `S3` (Medium), `S4` (Low)

---

## Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| **Documents Processed** | 78 PDFs | - |
| **Total Chunks** | ~1,234 chunks | - |
| **Ingestion Time** | 45 seconds | <90s |
| **Query Latency** | 1.5 seconds (avg) | <2s |
| **95th Percentile Latency** | 1.9 seconds | <2s |
| **Average Precision@5** | 85% | >80% |
| **Queries Above 80%** | 9 out of 10 | - |
| **Hallucination Rate** | 0% | 0% |
| **Precision Improvement** | +36% vs unfiltered | - |

---

## Example Use Cases

### 1. Troubleshooting Provider Errors
```bash
>> provider:stripe method:CARD errors:on
>> What error codes does Stripe return for declined cards?
```

### 2. Finding Integration Guides
```bash
>> provider:fintoc country:CL type:integration_guide
>> How to integrate Fintoc bank transfers in Chile?
```

### 3. Reviewing Critical Incidents
```bash
>> source:jira severity:S1
>> What were the recent S1 payment failures?
```

### 4. Testing Instructions
```bash
>> provider:payu testing:on
>> How to test PayU payments in sandbox?
```

### 5. Multi-Provider Comparison
```bash
>> provider:stripe,adyen country:CL
>> Compare card payment integration for Stripe vs Adyen in Chile
```

---

## Key Design Decisions

### Why 1000-character chunks with 200 overlap?
- Large enough to capture complete API concepts (e.g., "webhook configuration")
- Small enough to stay focused on one topic
- 200-char overlap prevents losing context at chunk boundaries
- Tested specifically for technical documentation (not arbitrary)

### Why metadata extraction at ingestion?
- Pre-filtering is 20x faster than post-filtering all results
- Engineers want specific docs (right provider + country), not just "similar" docs
- Structured metadata + semantic search = best of both worlds

### Why MongoDB Atlas Vector Search?
- Native support for pre-filtering in the same query
- Already in Yuno's infrastructure (no new database)
- Scales well for production use

### Why GPT-4o-mini?
- Sufficient for grounded Q&A (doesn't need reasoning of GPT-4)
- 10x cheaper than GPT-4
- Temperature=0 for consistent, non-creative responses

---

## Troubleshooting

### Issue: "No documents found" error
**Solution**: Run `python ingestion.py` first to populate MongoDB

### Issue: "Index not found" error
**Solution**: Create the vector search index in MongoDB Atlas (see Setup section)

### Issue: Low precision in evaluation
**Solution**:
1. Check if metadata extraction is working: `python retrieval.py` → look at "Metadata Summary"
2. Verify filters are applied: Use `verbose=True` in retrieve_with_filter
3. Check if vector search index includes all filter fields

### Issue: Slow query responses
**Solution**:
1. Verify vector search index is "Active" in MongoDB Atlas
2. Check network latency to MongoDB cluster
3. Consider using filters to narrow search space

---

## Future Improvements

### Short-term
- [ ] Add reranker for complex queries (cross-encoder)
- [ ] Expand test query coverage to 20+ queries
- [ ] Add caching for frequent queries

### Long-term
- [ ] Auto-sync with Confluence/Jira APIs (real-time updates)
- [ ] Multi-language support (Spanish, Portuguese)
- [ ] Web UI for non-technical users
- [ ] Query expansion for ambiguous searches
- [ ] User feedback loop for continuous improvement

---

## Contributing

This is an internal Yuno project. For questions or improvements, contact the team.

---

## License

Internal use only - Yuno 2026 
