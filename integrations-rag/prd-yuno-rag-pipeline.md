# PRD: Juno RAG Pipeline for Payment Provider Documentation

## Introduction

Build a Retrieval-Augmented Generation (RAG) system for Yuno that enables engineers to query hundreds of internal documents including payment provider integration documentation and historical Jira tickets. The system will answer questions about provider integrations, status mappings, error handling, and past feature implementations, reducing time spent searching through scattered documentation.

## Goals

- Enable natural language querying of 50-75 payment provider docs and Jira tickets (MVP scope)
- Implement metadata-filtered RAG pattern for precise document type and provider filtering
- Achieve high precision (>80%) and recall (>85%) for integration-related queries
- Provide sub-3-second query response latency for good user experience
- Support three primary query types: integration details, status/error mappings, and historical features
- Implement comprehensive evaluation suite (precision, recall, classification accuracy, latency, groundedness)

## User Stories

### US-001: Document ingestion pipeline for PDFs
**Description:** As a developer, I need to ingest payment provider docs and Jira tickets (in PDF format) so they can be processed and indexed.

**Acceptance Criteria:**
- [ ] Script accepts directory of PDF files as input
- [ ] Extracts text content from each PDF
- [ ] Extracts metadata: document_type (provider_doc | jira_ticket), provider_name, date, filename
- [ ] Handles malformed PDFs gracefully with error logging
- [ ] Outputs structured documents with content + metadata
- [ ] Typecheck/lint passes

### US-002: Implement chunking strategy
**Description:** As a developer, I need to chunk documents into semantic units so retrieval returns focused, relevant content.

**Acceptance Criteria:**
- [ ] Implements recursive character splitter with 1000 token chunks, 200 token overlap
- [ ] Preserves section headers within chunks for context
- [ ] Each chunk inherits parent document metadata
- [ ] Chunks maintain provider_name, document_type, section_title metadata
- [ ] Outputs chunk statistics (count, avg size, size distribution)
- [ ] Typecheck/lint passes

### US-003: Vector database setup and indexing
**Description:** As a developer, I need to embed chunks and store them in a vector database so they can be retrieved via similarity search.

**Acceptance Criteria:**
- [ ] Connects to vector database MongoDB Atlas
- [ ] Uses text-embedding-3-small or equivalent embedding model
- [ ] Indexes chunks with embeddings and full metadata
- [ ] Creates metadata indexes on document_type and provider_name fields
- [ ] Supports batch upsert for efficient indexing
- [ ] Typecheck/lint passes

### US-004: Metadata-filtered retrieval
**Description:** As a user, I want queries to be filtered by document type and provider so I get precise, relevant results.

**Acceptance Criteria:**
- [ ] Extracts provider names from queries (e.g., "Stripe", "PayPal")
- [ ] Applies metadata filters before vector search when provider detected
- [ ] Retrieves top-k chunks (k=5 default, configurable)
- [ ] Falls back to unfiltered search if no metadata detected
- [ ] Returns chunks with similarity scores and metadata
- [ ] Typecheck/lint passes

### US-005: LLM-based answer generation
**Description:** As a user, I want to receive natural language answers based on retrieved documentation so I don't have to read raw chunks.

**Acceptance Criteria:**
- [ ] Uses Claude or GPT-4 for generation
- [ ] Includes system prompt: "Answer based only on provided context. Cite sources."
- [ ] Formats retrieved chunks as context with source metadata
- [ ] Returns answer with source citations (filename, provider, document type)
- [ ] Handles "no relevant information found" gracefully
- [ ] Typecheck/lint passes

### US-006: Precision evaluation
**Description:** As a developer, I need to measure retrieval precision so I know the system returns relevant documents.

**Acceptance Criteria:**
- [ ] Implements precision = (relevant retrieved docs) / (total retrieved docs)
- [ ] Uses test set of 20+ queries with ground truth relevant docs
- [ ] Calculates precision@k for k=5 (top 5 results)
- [ ] Outputs per-query precision and average precision score
- [ ] Target: >80% average precision
- [ ] Typecheck/lint passes

### US-007: Recall evaluation
**Description:** As a developer, I need to measure retrieval recall so I know the system doesn't miss relevant documents.

**Acceptance Criteria:**
- [ ] Implements recall = (relevant retrieved docs) / (all relevant docs)
- [ ] Uses same test set as precision eval with complete ground truth
- [ ] Calculates recall@k for k=5, k=10, k=20
- [ ] Outputs per-query recall and average recall score
- [ ] Target: >85% recall@10
- [ ] Typecheck/lint passes

### US-008: Latency tracking
**Description:** As a developer, I need to track query latency so I can ensure fast response times.

**Acceptance Criteria:**
- [ ] Measures end-to-end query time (retrieval + generation)
- [ ] Logs separate timings for: embedding, retrieval, LLM generation
- [ ] Calculates p50, p95, p99 latencies across test queries
- [ ] Target: p95 < 3 seconds
- [ ] Outputs latency report with breakdown
- [ ] Typecheck/lint passes

### US-009: Groundedness evaluation
**Description:** As a developer, I need to verify LLM answers are grounded in retrieved content so we avoid hallucinations.

**Acceptance Criteria:**
- [ ] Uses LLM-as-judge to verify answer claims against retrieved chunks
- [ ] Calculates groundedness score: % of answer claims supported by context
- [ ] Tests on sample of 20+ query-answer pairs
- [ ] Target: >90% groundedness score
- [ ] Flags specific ungrounded claims for review
- [ ] Typecheck/lint passes

### US-010: Query classification accuracy evaluation
**Description:** As a developer, I need to validate that provider name extraction works correctly so metadata filtering performs as expected.

**Acceptance Criteria:**
- [ ] Test set includes queries with explicit providers ("Stripe refunds")
- [ ] Test set includes queries without providers ("How do refunds work?")
- [ ] Test set includes multi-provider queries ("Compare Stripe and PayPal")
- [ ] Calculates classification accuracy: correct detections / total queries
- [ ] Target: >90% classification accuracy
- [ ] Outputs confusion matrix showing detection errors
- [ ] Typecheck/lint passes

### US-011: Query interface (CLI or API)
**Description:** As a user, I want to submit queries and receive answers so I can access documentation knowledge.

**Acceptance Criteria:**
- [ ] CLI accepts query string as input OR API endpoint accepts POST with query
- [ ] Displays answer with source citations
- [ ] Shows metadata filters applied (if any)
- [ ] Returns response within 3 seconds for typical queries
- [ ] Handles errors gracefully with user-friendly messages
- [ ] Typecheck/lint passes

## Functional Requirements

**Ingestion & Processing:**
- FR-1: System must ingest PDF files from specified directory
- FR-2: System must extract metadata (document_type, provider_name, date) from filenames or content
- FR-3: System must chunk documents into 1000-token segments with 200-token overlap
- FR-4: System must preserve document structure (headers, sections) in chunks

**Indexing & Storage:**
- FR-5: System must embed chunks using text-embedding-3-small or equivalent
- FR-6: System must store embeddings and metadata in vector database (MongoDB Atlas or Pinecone)
- FR-7: System must create indexes on document_type and provider_name for filtering

**Retrieval:**
- FR-8: System must detect provider names in queries for metadata filtering
- FR-9: System must apply metadata filters before vector similarity search
- FR-10: System must retrieve top-5 most similar chunks by default
- FR-11: System must return chunks with similarity scores and full metadata

**Generation:**
- FR-12: System must generate answers using Claude or GPT-4
- FR-13: System must cite sources (filename, provider, document type) in answers
- FR-14: System must only answer based on retrieved context, not general knowledge

**Evaluation:**
- FR-15: System must calculate precision@5 on test set (target: >80%)
- FR-16: System must calculate recall@10 on test set (target: >85%)
- FR-17: System must track p95 latency (target: <3 seconds)
- FR-18: System must evaluate groundedness using LLM-as-judge (target: >90%)
- FR-19: System must validate query classification accuracy for metadata extraction (target: >90%)

## Non-Goals (Out of Scope)

- No user interface or web frontend (CLI/API only for MVP)
- No real-time document updates or live syncing with Jira
- No multi-turn conversational interface or chat history
- No user authentication or access control
- No deployment to production infrastructure (local/dev environment only)
- No support for non-PDF document formats (no Word, HTML, Markdown)
- No automatic metadata extraction via ML (manual metadata from filenames)
- No query suggestion or autocomplete features
- No advanced RAG patterns (no Graph RAG, no query rewriting, no re-ranking)

## Technical Considerations

**Document Types:**
1. **Provider Documentation PDFs**
   - Integration details: API endpoints, authentication, request/response formats
   - Status mappings: how provider statuses map to Juno internal statuses
   - Error handling: error codes, retry logic, fallback behaviors
   - Metadata: provider_name (Stripe, PayPal, Adyen, etc.), document_type=provider_doc

2. **Jira Ticket PDFs**
   - Feature implementation history: what was built, how it was built
   - Technical decisions and tradeoffs
   - Code changes and approaches
   - Metadata: ticket_id, document_type=jira_ticket, related_provider (if applicable)

**Architecture Stack:**
- **Framework**: LangChain (recommended) or custom implementation
- **Vector DB**: MongoDB Atlas Vector Search
- **Embedding Model**: OpenAI text-embedding-3-small (1536 dims)
- **LLM**: Claude 3.5 Sonnet or GPT-4
- **PDF Processing**: PyPDF2 or pdfplumber
- **Language**: Python 3.10+

**Chunking Strategy Considerations:**
- Technical docs benefit from larger chunks (1000 tokens) to preserve context
- Section headers should be preserved in chunks for semantic coherence
- Overlap (200 tokens) prevents information loss at chunk boundaries
- Consider sentence-aware splitting to avoid mid-sentence cuts

**Metadata Schema:**
```json
{
  "chunk_id": "uuid",
  "content": "chunk text",
  "document_type": "provider_doc | jira_ticket",
  "provider_name": "Stripe | PayPal | Adyen | etc.",
  "filename": "original_filename.pdf",
  "section_title": "extracted_header",
  "date": "YYYY-MM-DD",
  "chunk_index": 0,
  "parent_document_id": "uuid"
}
```

**Query Types & Examples:**

1. **Integration Details**
   - "How do we integrate Stripe refunds?"
   - "What API endpoint does PayPal use for authorization?"
   - "Show me Adyen authentication flow"

2. **Status/Error Mappings**
   - "What does Stripe status 'succeeded' map to in Juno?"
   - "How do we handle PayPal error code 10001?"
   - "List all Adyen status mappings"

3. **Historical Features**
   - "How did we implement recurring payments for Stripe?"
   - "What approach did we use for PayPal dispute handling?"
   - "Show me past tickets about payment retry logic"

## Evaluation Strategy

**Primary Metrics (Retrieval-based):**
1. **Precision@5**: Measures relevance of top 5 retrieved chunks
   - Target: >80%
   - Method: Manual labeling of test set (20+ queries)

2. **Recall@10**: Measures completeness of retrieval
   - Target: >85%
   - Method: Ground truth set of all relevant docs per query

**Pattern-Specific Metrics (Metadata-Filtered RAG):**
3. **Query Classification Accuracy**: Validates provider name extraction for filtering
   - Target: >90%
   - Method: Test queries with known expected filters
   - Critical for metadata-filtered RAG pattern success

**Secondary Metrics (System Performance & Quality):**
4. **Latency (p95)**: End-to-end query response time
   - Target: <3 seconds
   - Method: Instrumented timing of retrieval + generation

5. **Groundedness**: Answer accuracy vs. retrieved content
   - Target: >90%
   - Method: LLM-as-judge evaluation of claims

**Test Set Requirements:**
- Minimum 20 test queries covering all query types
- Ground truth labels: relevant documents for each query
- Include edge cases: ambiguous queries, multi-provider queries, no-match queries

## Success Metrics

- **Retrieval Quality**: Precision >80%, Recall >85%
- **Metadata Filtering**: Query classification accuracy >90%
- **Performance**: p95 latency <3 seconds
- **Answer Quality**: Groundedness >90%
- **Coverage**: 50-75 documents indexed (MVP milestone)
- **User Impact**: Reduces documentation search time from 15+ minutes to <1 minute per query
- **Evaluation Completeness**: All 5 metrics (precision, recall, classification accuracy, latency, groundedness) implemented and passing targets

## Implementation Phases

**Phase 1: Core Pipeline (Week 1)**
- US-001: Document ingestion
- US-002: Chunking strategy
- US-003: Vector database setup

**Phase 2: Retrieval & Generation (Week 1)**
- US-004: Metadata-filtered retrieval
- US-005: LLM answer generation
- US-011: Query interface

**Phase 3: Evaluation (Week 2)**
- US-006: Precision evaluation
- US-007: Recall evaluation
- US-008: Latency tracking
- US-009: Groundedness evaluation
- US-010: Query classification accuracy evaluation

## Open Questions

1. Should we implement automatic provider name extraction from PDFs, or rely on filename metadata?
2. Do we need different chunking strategies for provider docs vs. Jira tickets?
3. Should we support filtering by date ranges (e.g., "features implemented in 2024")?
4. How should we handle queries that span multiple providers (e.g., "compare Stripe and PayPal refund flows")?
5. Should we implement query rewriting or expansion for better recall?
6. Do we need to support non-English documentation in the future?

## Deliverables (Per Assignment Requirements)

1. **Code Repository**
   - Ingestion and chunking logic
   - Retrieval logic with metadata filtering
   - Evaluation logic (precision, recall, classification accuracy, latency, groundedness)
   - Clear documentation and structure

2. **Video Walkthrough (3-5 minutes)**
   - Pipeline architecture overview
   - Chunking and retrieval decisions explained
   - Metadata-filtered RAG pattern justification
   - Evaluation results and insights
   - Live demo (optional)

3. **Test Results**
   - Precision@5, Recall@10 scores on test set
   - Query classification accuracy with confusion matrix
   - Latency benchmarks (p50, p95, p99)
   - Groundedness evaluation results
   - Example queries with retrieved sources and applied filters

## References

- Gauntlet AI Catalyst 3 Assignment
- RAG Cookbook (inspiration for architecture patterns)
- Assignment due: January 29, 2026 at 11:59 PM CT
