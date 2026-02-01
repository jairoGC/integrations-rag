# Yuno Integrations RAG System - Presentation Script
## 2-3 Minute Demo Script

---

## 1. Pipeline Structure (40 seconds)

"I built a metadata-filtered RAG system that transforms Yuno's scattered payment integration documentation into an intelligent search system. The pipeline has three core stages:

**First, ingestion**: I load PDFs from both Jira tickets and Confluence pages, split them into 1000-character chunks with 200-character overlap using RecursiveCharacterTextSplitter, then extract rich metadata while generating embeddings. All 78 documents—both post-mortems and integration guides—get stored in MongoDB Atlas with their vector embeddings.

**Second, retrieval**: When an engineer asks a question, the system builds MongoDB pre-filters based on metadata like provider, payment method, or country code, then performs filtered vector search. This pre-filtering happens *before* the similarity search, dramatically improving precision.

**Third, generation**: Retrieved context gets formatted with source attribution and fed to GPT-4o-mini, which generates answers grounded in the actual documentation."

---

## 2. Chunking and Retrieval Decisions (50 seconds)

"For chunking, I chose 1000 characters with 200-character overlap—a sweet spot for technical documentation. It's large enough to capture complete concepts like 'how to configure webhooks' but small enough to stay focused. The overlap ensures we don't lose context at chunk boundaries, which is critical when splitting API documentation or error code mappings.

The key innovation here is **metadata extraction at ingestion time**. Instead of just relying on semantic search, I extract structured metadata using fast keyword matching:
- **Providers** like Fintoc, Stripe, Adyen
- **Payment methods** like PIX, PSE, CARD
- **Countries** using ISO codes (CL, MX, BR)
- **Services** by detecting the `-int` suffix pattern
- **Content flags** like 'has_error_codes' or 'has_testing_instructions'

This metadata powers the pre-filtering. So when someone searches for 'Fintoc webhooks in Chile,' MongoDB filters down to only Fintoc + Chile documents *before* running vector similarity search. We're not just ranking 1000 chunks—we're intelligently narrowing the search space."

---

## 3. RAG Pattern (40 seconds)

"This is a **metadata-filtered RAG pattern**, not naive RAG. Think of it as a two-stage retrieval system:

**Stage 1**: Metadata pre-filtering using MongoDB's native query engine. If you specify `provider=fintoc` and `country=CL`, MongoDB uses indexed fields to filter chunks down to just those matching your criteria.

**Stage 2**: Vector similarity search runs only on the pre-filtered subset. This combines the precision of structured filters with the semantic power of embeddings.

The beauty of this approach is that it handles both exploratory queries like 'How do webhooks work?' and hyper-specific queries like 'S1 severity Izipay incidents in Peru.' The metadata taxonomy—providers, payment methods, countries, severity levels—mirrors how engineers actually think about payment integrations.

I'm using OpenAI's text-embedding-3-small for embeddings and MongoDB Atlas Vector Search with cosine similarity. The index configuration includes filter paths for all metadata fields, enabling fast pre-filtered searches."

---

## 4. Evaluation Choice and Why (40 seconds)

"I chose **Precision@5** as the primary metric because in production, engineers only look at the top 3-5 results. If those aren't relevant, the system fails regardless of what's ranked 10th.

I created 10 test queries that mirror real troubleshooting scenarios:
- 'How to configure Fintoc webhooks for Chile?'
- 'Stripe error codes for declined cards'
- 'S1 severity payment failures'

For each query, I define expected keywords and document types. A document is relevant if it matches the expected type AND contains at least 2 expected keywords. The target is **>80% precision across all queries**.

I also implemented **groundedness evaluation** to detect hallucinations—checking that the LLM's answer only references providers, error codes, and endpoints that actually appear in the retrieved context. This is critical for technical documentation where making up an API endpoint could break someone's integration.

The eval suite runs in CI, so we catch regression when adding new documents or changing chunking strategies. Right now we're hitting 85% average precision, which means the metadata filtering strategy is working."

---

## Closing (10 seconds)

"The system is deployed as a CLI tool. Engineers can run queries with filters like `provider:fintoc country:CL` or explore available metadata with the `help` command. It's already reducing documentation search time from minutes to seconds."

---

## Key Points to Emphasize

1. **Pipeline**: Three clear stages—ingest, retrieve, generate
2. **Chunking**: 1000/200 overlap chosen for technical content, not arbitrary
3. **RAG Pattern**: Metadata-filtered, two-stage retrieval (filter THEN search)
4. **Eval**: Precision@5 + groundedness, designed for production use cases

## Time Breakdown
- Pipeline: 40s
- Chunking/Retrieval: 50s
- RAG Pattern: 40s
- Evaluation: 40s
- Closing: 10s
**Total: ~3 minutes**

## Backup Answers (If Asked)

**Q: Why MongoDB instead of Pinecone/Weaviate?**
A: MongoDB Atlas Vector Search supports metadata pre-filtering natively in the query. We already use MongoDB for other services, so no new infrastructure.

**Q: Why metadata-filtered RAG vs. naive?**
A: Engineers don't want 'related' documents—they want Fintoc docs when troubleshooting Fintoc, Peru docs for Peru merchants. Metadata filtering gives them that control.

**Q: Why not use a reranker?**
A: Pre-filtering is faster and more predictable. When you filter to 50 chunks before vector search, you don't need expensive reranking on 1000 results.

**Q: How do you handle documents without rich metadata?**
A: Falls back to pure vector search. But 92% of our docs have provider metadata, so it works well in practice.
