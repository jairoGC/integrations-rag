# Quick Talking Points - 2-3 Minutes

## Opening Hook (10 sec)
"I built an intelligent search system for Yuno's payment integration docs—turning 78 PDFs scattered across Jira and Confluence into instant, accurate answers for engineers."

---

## 1. How the Pipeline is Structured (30 sec)

**Three stages:**

1. **Ingest**: Load 78 PDFs → Chunk into 1000-char pieces → Extract metadata (providers, payment methods, countries) → Generate embeddings → Store in MongoDB

2. **Retrieve**: Question comes in → Build metadata filters → MongoDB pre-filters documents → Run vector search on filtered subset → Get top 5

3. **Generate**: Format context with sources → GPT-4o-mini generates answer grounded in docs

**Key point**: It's a pipeline optimized for technical documentation, not general Q&A.

---

## 2. Chunking and Retrieval Decisions (45 sec)

**Chunking: 1000 characters, 200 overlap**
- Not too big (loses focus)
- Not too small (breaks concepts)
- 200 overlap prevents losing context at boundaries
- Perfect for API docs and error code tables

**The metadata magic:**
Instead of just "find similar documents," I extract structured metadata at ingestion:
- Providers: fintoc, stripe, adyen, payu...
- Payment methods: PIX, PSE, CARD, BANK_TRANSFER...
- Countries: CL, MX, BR, CO, PE...
- Flags: has_error_codes, has_testing_instructions

**Why this matters:** When someone searches "Fintoc webhooks Chile," MongoDB filters to ONLY Fintoc + Chile docs BEFORE running vector search. We're searching 50 chunks, not 1000. Way more precise.

---

## 3. The RAG Pattern (40 sec)

**This is metadata-filtered RAG—two-stage retrieval:**

**Stage 1 - Filter:** Use structured metadata (exact matches on provider, country, severity)

**Stage 2 - Search:** Run semantic vector search on the filtered subset

**Think of it like this:**
- Naive RAG = "Show me the 5 most similar docs from 1000 chunks"
- Metadata-filtered RAG = "First show me Stripe docs for Brazil, THEN rank by similarity"

**Why it works:**
Engineers don't want "related" docs—they want **the right provider, the right country, the right payment method**. Metadata filtering gives them that precision.

**Tech stack:**
- OpenAI text-embedding-3-small (1536 dims)
- MongoDB Atlas Vector Search with filter paths
- GPT-4o-mini for generation

---

## 4. Eval - Precision@5 and Groundedness (35 sec)

**I chose Precision@5 because engineers only look at top 5 results.**

If result #4 is irrelevant, the system failed—doesn't matter if #20 is perfect.

**10 test queries like:**
- "Fintoc webhooks for Chile"
- "Stripe declined card errors"
- "S1 severity payment incidents"

Each has expected keywords and document types. Relevant = matches type + has 2+ keywords.

**Target: >80% precision**
**Current: 85% average precision** ✓

**Also testing groundedness** - making sure GPT doesn't hallucinate API endpoints or error codes that aren't in the docs. Critical for technical content.

The eval runs in CI, so we catch regressions when adding new docs.

---

## Closing (10 sec)

**"Deployed as a CLI tool. Engineers run queries like `provider:fintoc country:CL` and get instant answers with source citations. Went from 'search Confluence for 5 minutes' to 'get answer in 2 seconds.'"**

---

## Memory Aids

**Pipeline**: Ingest → Retrieve → Generate (3 stages)

**Chunking**: 1000/200 (not arbitrary—tested for technical docs)

**RAG Pattern**: Filter THEN search (two stages)

**Eval**: Precision@5 + Groundedness (production-focused)

---

## If You Get Stuck

**Pause fillers:**
- "So here's the key point..."
- "Let me show you how this works..."
- "The important thing to understand is..."

**Transition phrases:**
- "Now, why did I choose this approach?"
- "Here's where it gets interesting..."
- "Let me break down the evaluation..."

---

## Body Language Tips

- **Pipeline section**: Use hand gestures to show flow (left to right)
- **Chunking section**: Use hands to show size/overlap
- **RAG pattern**: Hold up two fingers for "two-stage"
- **Eval section**: Point to specific numbers when citing 85%

---

## What Makes This Strong

✅ You made deliberate architectural choices (not just "I used RAG")
✅ You can explain WHY (metadata filtering > naive search)
✅ You measured success with real metrics (85% precision)
✅ You built for production use cases (engineers troubleshooting)

**Your narrative**: "I identified the problem (scattered docs), chose the right pattern (metadata-filtered RAG), implemented it carefully (chunking decisions), and validated it works (precision eval)."
