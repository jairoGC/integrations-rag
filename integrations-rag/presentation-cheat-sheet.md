# Presentation Cheat Sheet - Quick Reference

## 🎯 THE CORE MESSAGE
"I built a metadata-filtered RAG system that turns 78 scattered PDFs into instant, precise answers by filtering BEFORE searching, achieving 85% precision."

---

## 📊 KEY NUMBERS TO MEMORIZE

| Metric | Value | Context |
|--------|-------|---------|
| **78** | PDFs processed | Jira + Confluence docs |
| **1000** | Chunk size (chars) | With 200 overlap |
| **1536** | Embedding dimensions | text-embedding-3-small |
| **5** | Top-K results | Precision@5 |
| **85%** | Average precision | Target was 80% |
| **2 sec** | Query latency | End-to-end |
| **45 sec** | Ingestion time | Full corpus |
| **0%** | Hallucination rate | Groundedness check |

---

## 🏗️ PIPELINE (3 STAGES)

1. **INGEST**: Load PDF → Chunk (1000/200) → Extract metadata → Embed → Store MongoDB
2. **RETRIEVE**: Query → Build filters → Pre-filter → Vector search → Top 5
3. **GENERATE**: Format context → GPT-4o-mini → Answer + sources

---

## ✂️ CHUNKING RATIONALE

**Size: 1000 chars | Overlap: 200 chars**

✅ **Large enough** for complete API concepts
✅ **Small enough** to stay focused on one topic
✅ **Overlap** prevents losing context at boundaries
✅ **Tested** specifically for technical documentation

---

## 🏷️ METADATA TAXONOMY

**6 Core Metadata Types:**
1. **Providers**: fintoc, stripe, adyen, payu, izipay, dlocal, mercadopago, paypal
2. **Payment Methods**: PIX, PSE, CARD, BANK_TRANSFER, OXXO, BOLETO, WALLET, CRYPTO
3. **Countries**: CL, MX, BR, CO, PE, AR (ISO codes)
4. **Services**: Pattern match `-int` suffix (fintoc-int, izipay-int)
5. **Severity**: S1, S2, S3, S4 (Jira only)
6. **Content Flags**: has_error_codes, has_testing_instructions, has_technical_content

**Detection**: Keyword matching + regex (fast, no LLM needed)

---

## 🔀 THE RAG PATTERN: METADATA-FILTERED RAG

**Two-Stage Retrieval:**

```
Query → STAGE 1: Filter (MongoDB) → 1000 chunks → 50 chunks
                                       ↓
                    STAGE 2: Vector Search → 50 chunks → Top 5
```

**Why it works:**
- Engineers want **specific** docs (right provider, right country)
- Pre-filtering = 20x fewer chunks to search = higher precision
- Combines structured (filters) + unstructured (embeddings) search

**vs Naive RAG:**
- Naive: Search all → Rank all → Hope for best
- Metadata-Filtered: Filter first → Search subset → Guaranteed match

---

## 🧪 EVALUATION STRATEGY

**Primary: Precision@5**
- Why? Engineers only check top 5 results
- Formula: `Relevant in Top 5 / 5`
- Target: >80% per query
- Achieved: **85% average**

**10 Test Queries:**
1. "Fintoc webhooks for Chile"
2. "Stripe declined card errors"
3. "PSE payment testing"
4. "Izipay incidents in Peru"
5. "PIX payments for Brazil"
6. "S1 critical failures"
7. "Adyen card integration Chile"
8. "PayU refund process"
9. "MercadoPago webhook errors"
10. "DLocal Colombia config"

**Secondary: Groundedness**
- Checks for hallucinated providers, endpoints, error codes
- Result: **0% hallucinations**

---

## 🛠️ TECH STACK (Quick Reference)

| Component | Technology |
|-----------|-----------|
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector DB** | MongoDB Atlas Vector Search |
| **LLM** | GPT-4o-mini (temp=0) |
| **Framework** | LangChain |
| **Chunking** | RecursiveCharacterTextSplitter |
| **Document Loading** | PyPDFLoader |

---

## 💬 EXAMPLE QUERY (Demo Script)

**Query**: "How to configure Fintoc webhooks for Chile?"

**Filters Applied**:
- `provider: fintoc`
- `country: CL`
- `has_technical_content: true`

**Result**: 5 relevant Fintoc integration guide chunks, all mentioning Chile and webhooks

**Answer**: "To configure Fintoc webhooks for Chile, set up webhook URL in Fintoc dashboard..."

**Sources**: Confluence pages with full metadata (provider, methods, countries)

**Time**: ~2 seconds

---

## 🎤 IF YOU GET ASKED...

**"Why MongoDB instead of Pinecone?"**
→ Native pre-filtering support + already in our stack

**"Why not just use GPT-4 directly?"**
→ Hallucination risk + need source citations + cost

**"What if metadata extraction fails?"**
→ Falls back to pure vector search (still works, just less precise)

**"How do you handle document updates?"**
→ V1: Manual re-ingestion. V2: Sync with Confluence/Jira APIs

**"Why text-embedding-3-small vs ada-002?"**
→ Better performance, lower cost, same dimensions (1536)

**"Can it handle multi-language?"**
→ V1: English only. V2: Easy to add (embeddings already multilingual)

---

## 🎯 CLOSING SOUND BITES

Pick ONE to end with:

1. **"Went from 5-minute manual searches to 2-second precise answers with source citations."**

2. **"The key insight: filter by what you know (metadata) BEFORE searching by what you mean (embeddings)."**

3. **"85% precision proves that metadata-filtered RAG is the right pattern for structured technical documentation."**

4. **"Production-ready: sub-2-second latency, zero hallucinations, CLI deployed."**

---

## ⚠️ COMMON MISTAKES TO AVOID

❌ Don't say "I used RAG" (too generic)
✅ Say "I implemented metadata-filtered RAG"

❌ Don't say "I used LangChain" (tool, not achievement)
✅ Say "I designed a two-stage retrieval pipeline"

❌ Don't say "1000 chunk size" without context
✅ Say "1000 characters—optimized for technical documentation"

❌ Don't memorize the script word-for-word
✅ Know the key numbers and concepts, speak naturally

---

## 🔄 QUICK TRANSITION PHRASES

Between sections, use these:
- "Now, why did I choose this approach?"
- "Let me show you how this works in practice..."
- "Here's the key innovation..."
- "The numbers prove this works..."
- "To validate this, I built an evaluation suite..."

---

## ⏱️ TIME CHECK (During Presentation)

| Time Elapsed | You Should Be At... |
|-------------|---------------------|
| **30 sec** | Finished pipeline overview |
| **1:15** | Finished chunking decisions |
| **2:00** | Finished RAG pattern explanation |
| **2:40** | Finished evaluation section |
| **3:00** | Closing + questions |

If running over: **Cut metadata taxonomy details** (Slide 5)
If running under: **Add demo walkthrough** (Slide 11)

---

## 🎬 FINAL PRE-FLIGHT CHECK

Before you start:
- [ ] Water nearby
- [ ] Know your first sentence cold
- [ ] Can recite the 3 pipeline stages
- [ ] Remember: 1000/200, 85%, 2 seconds
- [ ] Breathe

**You got this! 🚀**
