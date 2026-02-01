"""
Vector Store for MongoDB Atlas
Handles embedding generation and vector database operations.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEmbedding:
    """Represents a chunk with its embedding vector."""

    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]


class EmbeddingProvider:
    """Handles text embedding generation using OpenAI."""

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the embedding provider.

        Args:
            model: OpenAI embedding model name (default: text-embedding-3-small)
            api_key: OpenAI API key (if not provided, reads from OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        logger.info("Initialized EmbeddingProvider with model: %s", model)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector (list of floats)
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding: %s", str(e))
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")

        try:
            response = self.client.embeddings.create(
                input=valid_texts, model=self.model
            )
            embeddings = [item.embedding for item in response.data]
            logger.info("Generated embeddings for %d texts", len(embeddings))
            return embeddings

        except Exception as e:
            logger.error("Failed to generate batch embeddings: %s", str(e))
            raise


class VectorStore:
    """
    MongoDB Atlas Vector Store for RAG pipeline.

    Handles:
    - Connection to MongoDB Atlas
    - Embedding generation
    - Vector indexing with metadata
    - Batch operations
    """

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: str = "yuno_rag",
        collection_name: str = "chunks",
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """
        Initialize the vector store.

        Args:
            mongodb_uri: MongoDB connection URI (if not provided, reads from MONGODB_URI env var)
            database_name: Name of the database
            collection_name: Name of the collection
            embedding_provider: EmbeddingProvider instance (if None, creates default)
        """
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")

        if not self.mongodb_uri:
            raise ValueError(
                "MongoDB URI not provided. Set MONGODB_URI environment variable."
            )

        # Initialize MongoDB client
        try:
            self.client: MongoClient = MongoClient(self.mongodb_uri)
            self.database = self.client[database_name]
            self.collection: Collection = self.database[collection_name]
            logger.info(
                "Connected to MongoDB: database=%s, collection=%s",
                database_name,
                collection_name,
            )
        except Exception as e:
            logger.error("Failed to connect to MongoDB: %s", str(e))
            raise

        # Initialize embedding provider
        self.embedding_provider = embedding_provider or EmbeddingProvider()

        # Create indexes
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create metadata indexes for efficient filtering."""
        try:
            # Create index on document_type
            self.collection.create_index(
                [("metadata.document_type", pymongo.ASCENDING)],
                name="document_type_index",
            )

            # Create index on provider_name
            self.collection.create_index(
                [("metadata.provider_name", pymongo.ASCENDING)],
                name="provider_name_index",
            )

            # Create compound index for both fields
            self.collection.create_index(
                [
                    ("metadata.document_type", pymongo.ASCENDING),
                    ("metadata.provider_name", pymongo.ASCENDING),
                ],
                name="document_type_provider_index",
            )

            logger.info("Created metadata indexes on collection")

        except Exception as e:
            logger.error("Failed to create indexes: %s", str(e))
            raise

    def index_chunk(
        self, chunk_id: str, content: str, metadata: Dict[str, Any]
    ) -> str:
        """
        Index a single chunk with embedding.

        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content to embed and index
            metadata: Metadata dictionary

        Returns:
            MongoDB document ID
        """
        if not content or not content.strip():
            raise ValueError("Cannot index empty content")

        try:
            # Generate embedding
            embedding = self.embedding_provider.embed_text(content)

            # Create document
            document = {
                "chunk_id": chunk_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            }

            # Upsert document (insert or update if exists)
            result = self.collection.update_one(
                {"chunk_id": chunk_id}, {"$set": document}, upsert=True
            )

            doc_id = str(result.upserted_id) if result.upserted_id else chunk_id
            logger.info("Indexed chunk: %s", chunk_id)

            return doc_id

        except Exception as e:
            logger.error("Failed to index chunk %s: %s", chunk_id, str(e))
            raise

    def index_batch(
        self, chunks: List[Dict[str, Any]], batch_size: int = 100
    ) -> List[str]:
        """
        Index multiple chunks in batches.

        Args:
            chunks: List of chunk dictionaries with keys: chunk_id, content, metadata
            batch_size: Number of chunks to process at once (default: 100)

        Returns:
            List of indexed chunk IDs
        """
        if not chunks:
            return []

        indexed_ids = []
        total_chunks = len(chunks)

        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            logger.info(
                "Processing batch %d/%d (%d chunks)",
                batch_num,
                total_batches,
                len(batch),
            )

            try:
                # Extract texts for batch embedding
                texts = [chunk["content"] for chunk in batch]

                # Generate embeddings for batch
                embeddings = self.embedding_provider.embed_batch(texts)

                # Prepare documents for bulk upsert
                operations = []
                for chunk, embedding in zip(batch, embeddings):
                    chunk_id = chunk["chunk_id"]
                    document = {
                        "chunk_id": chunk_id,
                        "content": chunk["content"],
                        "embedding": embedding,
                        "metadata": chunk["metadata"],
                    }

                    # Use update_one with upsert for each document
                    operations.append(
                        pymongo.UpdateOne(
                            {"chunk_id": chunk_id}, {"$set": document}, upsert=True
                        )
                    )
                    indexed_ids.append(chunk_id)

                # Execute bulk operation
                if operations:
                    result = self.collection.bulk_write(operations)
                    logger.info(
                        "Batch %d complete: inserted=%d, modified=%d",
                        batch_num,
                        result.upserted_count,
                        result.modified_count,
                    )

            except Exception as e:
                logger.error("Failed to process batch %d: %s", batch_num, str(e))
                raise

        logger.info("Successfully indexed %d chunks", len(indexed_ids))
        return indexed_ids

    def get_chunk_count(self) -> int:
        """
        Get total number of indexed chunks.

        Returns:
            Count of documents in collection
        """
        return self.collection.count_documents({})

    def get_chunks_by_metadata(
        self, metadata_filter: Dict[str, Any], limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks by metadata filter.

        Args:
            metadata_filter: Dictionary of metadata field filters
            limit: Maximum number of chunks to return

        Returns:
            List of chunk documents
        """
        # Build query with metadata prefix
        query = {f"metadata.{key}": value for key, value in metadata_filter.items()}

        try:
            chunks = list(self.collection.find(query).limit(limit))
            logger.info("Retrieved %d chunks with filter: %s", len(chunks), query)
            return chunks

        except Exception as e:
            logger.error("Failed to retrieve chunks: %s", str(e))
            raise

    def delete_all_chunks(self) -> int:
        """
        Delete all chunks from the collection.

        Returns:
            Number of deleted documents
        """
        result = self.collection.delete_many({})
        logger.info("Deleted %d chunks from collection", result.deleted_count)
        return result.deleted_count

    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Closed MongoDB connection")


def main() -> None:
    """Example usage of the vector store."""
    # Example chunks
    sample_chunks = [
        {
            "chunk_id": "chunk_001",
            "content": "Stripe provides a comprehensive API for payment processing. "
            "The refund endpoint allows merchants to return funds to customers.",
            "metadata": {
                "document_type": "provider_doc",
                "provider_name": "Stripe",
                "filename": "stripe_api.pdf",
                "section_title": "REFUND API",
            },
        },
        {
            "chunk_id": "chunk_002",
            "content": "PayPal integration requires OAuth 2.0 authentication. "
            "Developers must register their application to obtain credentials.",
            "metadata": {
                "document_type": "provider_doc",
                "provider_name": "PayPal",
                "filename": "paypal_integration.pdf",
                "section_title": "AUTHENTICATION",
            },
        },
    ]

    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore(
        database_name="yuno_rag_test", collection_name="chunks_test"
    )

    # Index sample chunks
    print("\nIndexing sample chunks...")
    indexed_ids = vector_store.index_batch(sample_chunks)
    print(f"Indexed {len(indexed_ids)} chunks: {indexed_ids}")

    # Get chunk count
    count = vector_store.get_chunk_count()
    print(f"\nTotal chunks in collection: {count}")

    # Retrieve chunks by metadata
    print("\nRetrieving Stripe chunks...")
    stripe_chunks = vector_store.get_chunks_by_metadata({"provider_name": "Stripe"})
    for chunk in stripe_chunks:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Provider: {chunk['metadata']['provider_name']}")
        print(f"Content: {chunk['content'][:100]}...")

    # Clean up
    print("\nCleaning up test data...")
    deleted = vector_store.delete_all_chunks()
    print(f"Deleted {deleted} chunks")

    vector_store.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
