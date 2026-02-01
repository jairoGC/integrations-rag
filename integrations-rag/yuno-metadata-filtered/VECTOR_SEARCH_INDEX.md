# MongoDB Atlas Vector Search Index Configuration

## Overview

This document describes the required MongoDB Atlas Vector Search index configuration for the Yuno Integrations RAG system.

## Database and Collection

- **Database Name**: `yuno_integrations_rag`
- **Collection Name**: `integration_docs`
- **Index Name**: `integration_docs_index`

## Index Configuration

To create the vector search index in MongoDB Atlas:

1. Navigate to your MongoDB Atlas cluster
2. Go to the "Search" tab
3. Click "Create Search Index"
4. Select "JSON Editor"
5. Use the following JSON configuration:

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      },
      "document_source": {
        "type": "token"
      },
      "document_type": {
        "type": "token"
      },
      "providers": {
        "type": "token"
      },
      "payment_methods": {
        "type": "token"
      },
      "countries": {
        "type": "token"
      },
      "services": {
        "type": "token"
      },
      "severity": {
        "type": "token"
      },
      "platform": {
        "type": "token"
      },
      "has_error_codes": {
        "type": "boolean"
      },
      "has_testing_instructions": {
        "type": "boolean"
      },
      "has_technical_content": {
        "type": "boolean"
      },
      "created_date": {
        "type": "date"
      },
      "updated_date": {
        "type": "date"
      },
      "ticket_id": {
        "type": "token"
      },
      "page_id": {
        "type": "token"
      },
      "text": {
        "type": "string"
      }
    }
  }
}
```

**IMPORTANT**: All metadata filter fields use `type: "token"` (not `"string"`) to support array filtering with the `$in` operator. Only the document text content field uses `type: "string"` for full-text search.

## Field Descriptions

### Vector Field
- **embedding**: 1536-dimensional vector from OpenAI's `text-embedding-3-small` model, using cosine similarity

### Filter Fields

#### Document Classification
- **document_source**: Source of the document (`jira`, `confluence`)
- **document_type**: Type of document (`post_mortem`, `integration_guide`, `api_reference`, `troubleshooting`, `technical_specs`, `testing_guide`)

#### Payment Integration Metadata
- **providers**: Array of payment provider names (`fintoc`, `stripe`, `adyen`, `payu`, `izipay`, `dlocal`, `mercadopago`, `paypal`, etc.)
- **payment_methods**: Array of payment method types (`CARD`, `BANK_TRANSFER`, `PIX`, `PSE`, `OXXO`, `BOLETO`, `WALLET`, `CRYPTO`, etc.)
- **countries**: Array of ISO 3166-1 alpha-2 country codes (`CL`, `MX`, `BR`, `CO`, `PE`, `AR`, etc.)
- **services**: Array of microservice names (e.g., `izipay-int`, `fintoc-int`, `stripe-int`)

#### Incident Management
- **severity**: Severity level for Jira tickets (`S1`, `S2`, `S3`, `S4`)
- **ticket_id**: Jira ticket ID (e.g., `PFU-152`, `CORECM-13628`)

#### Technical Content Flags
- **has_error_codes**: Boolean indicating presence of error codes or status mappings
- **has_testing_instructions**: Boolean indicating presence of testing instructions
- **has_technical_content**: Boolean indicating presence of API endpoints, credentials, or webhook configurations
- **platform**: Platform support (`web`, `mobile`, `both`)

#### Document Metadata
- **created_date**: Document creation date (ISO 8601 format)
- **updated_date**: Document last update date (ISO 8601 format)
- **page_id**: Confluence page ID for Confluence documents

## Removed Fields (from Warren Buffett System)

The following fields from the original metadata-filtered RAG system have been removed:

- `year` - Replaced with `created_date` and `updated_date`
- `decade` - No longer relevant for technical documentation
- `source_file` - Replaced with `document_source` and document IDs
- `topic_buckets` - Replaced with Yuno-specific metadata (providers, methods, etc.)
- `companies_mentioned` - Not relevant for internal technical documentation
- `has_financials` - Not relevant for payment integration docs

## Setup Instructions

### 1. Create the Database and Collection

If not already created, the ingestion script will automatically create the database and collection when run.

### 2. Create the Vector Search Index

1. Log in to [MongoDB Atlas](https://cloud.mongodb.com)
2. Navigate to your cluster
3. Click "Search" in the left sidebar
4. Click "Create Search Index"
5. Select "JSON Editor"
6. Paste the JSON configuration above
7. Set the index name to `integration_docs_index`
8. Select database `yuno_integrations_rag` and collection `integration_docs`
9. Click "Create Search Index"

### 3. Wait for Index Build

The index build can take several minutes depending on the number of documents (typically 5-10 minutes for 3,000+ chunks).

Monitor the index status in the Atlas UI. The index is ready when the status shows "Active".

### 4. Verify Index

Run the retrieval test to verify the index is working:

```bash
cd yuno-metadata-filtered
python3 retrieval.py
```

This will test various filter combinations and display results.

## Index Performance Considerations

- **Index Size**: Approximately 1536 dimensions × 4 bytes × number of chunks ≈ 18 MB for 3,000 chunks
- **Query Latency**: Expected <500ms for vector search with pre-filtering
- **Filter Selectivity**: Pre-filtering reduces the vector search space, improving performance
- **Concurrent Queries**: Atlas Vector Search supports concurrent queries from multiple engineers

## Common Pitfall: String vs Token Type

**CRITICAL**: Do not use `type: "string"` for metadata filter fields!

MongoDB Atlas Vector Search uses different field types:

- **`token`**: For exact-match filtering on strings and arrays (use for all metadata fields)
- **`string`**: For full-text search (use only for document content)
- **`boolean`**: For true/false fields
- **`date`**: For date fields
- **`knnVector`**: For embeddings

### Why Token Type for Arrays?

Our metadata fields like `providers`, `payment_methods`, `countries` are **arrays** because a single document can mention multiple providers, support multiple payment methods, etc.

When you use:
```python
retrieve_with_filter(query="...", providers=["fintoc", "stripe"])
```

The system generates a MongoDB query:
```json
{"providers": {"$in": ["fintoc", "stripe"]}}
```

The `$in` operator requires the field to be indexed as `token`, not `string`. If you use `string` type, you'll get this error:
```
Path 'providers' needs to be indexed as token
```

## Troubleshooting

### "Path needs to be indexed as token" Error

**Error:** `Path 'providers' needs to be indexed as token`

**Solution:**
1. Delete the existing index in Atlas Search
2. Recreate it with `type: "token"` for ALL metadata fields (not `"string"`)
3. Wait for the index to rebuild (1-2 minutes)

### Index Not Found Error

If you receive an error about the index not being found:

1. Verify the index name is exactly `integration_docs_index`
2. Check that the database and collection names match
3. Ensure the index status is "Active" in Atlas
4. Wait a few minutes after creation for the index to build

### Slow Query Performance

If queries are slower than expected:

1. Check the index status - partial builds can cause slowness
2. Verify filter fields are indexed correctly
3. Consider reducing `top_k` if retrieving many documents
4. Monitor Atlas metrics for query performance

### Connection Errors

If you cannot connect to MongoDB Atlas:

1. Verify your IP address is whitelisted in Atlas Network Access
2. Check that your connection string in `.env` is correct
3. Ensure you have the `srv` connection format: `mongodb+srv://...`
4. Verify your database user credentials

## Migration Notes

When migrating from the Warren Buffett system:

1. Create the new index with the Yuno-specific configuration
2. Run the Yuno ingestion pipeline to populate the new collection
3. The old index (`metadata_filtered_index`) can be deleted once migration is complete
4. The old collection (`metadata_filtered_rag`) can be archived or deleted

## Additional Resources

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [LangChain MongoDB Integration](https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas)
