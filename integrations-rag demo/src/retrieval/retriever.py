"""
Metadata-Filtered Retrieval System
Performs vector similarity search with optional metadata filtering.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np
from src.indexing.vector_store import VectorStore, EmbeddingProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with similarity score."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
        }


class QueryClassifier:
    """
    Extracts metadata filters from queries.

    Identifies provider names and document types in user queries
    to enable metadata-filtered retrieval.
    """

    # Common payment provider names
    PROVIDER_NAMES = {
        "stripe",
        "paypal",
        "adyen",
        "braintree",
        "square",
        "checkout",
        "worldpay",
        "authorize.net",
        "klarna",
        "affirm",
    }

    # Document type keywords
    DOCUMENT_TYPE_KEYWORDS = {
        "documentation": "provider_doc",
        "docs": "provider_doc",
        "api": "provider_doc",
        "integration": "provider_doc",
        "ticket": "jira_ticket",
        "jira": "jira_ticket",
        "issue": "jira_ticket",
        "feature": "jira_ticket",
    }

    def __init__(self, additional_providers: Optional[List[str]] = None) -> None:
        """
        Initialize the query classifier.

        Args:
            additional_providers: Optional list of additional provider names to recognize
        """
        self.providers = self.PROVIDER_NAMES.copy()
        if additional_providers:
            self.providers.update(p.lower() for p in additional_providers)
        logger.info("Initialized QueryClassifier with %d providers", len(self.providers))

    def extract_providers(self, query: str) -> Set[str]:
        """
        Extract provider names from a query.

        Args:
            query: User query string

        Returns:
            Set of detected provider names (lowercased)
        """
        query_lower = query.lower()
        detected_providers = set()

        # Check for exact provider name matches (word boundaries)
        for provider in self.providers:
            # Use word boundary matching to avoid false positives
            pattern = r"\b" + re.escape(provider) + r"\b"
            if re.search(pattern, query_lower):
                detected_providers.add(provider.capitalize())

        return detected_providers

    def extract_document_type(self, query: str) -> Optional[str]:
        """
        Extract document type from a query.

        Args:
            query: User query string

        Returns:
            Document type ('provider_doc' or 'jira_ticket') or None
        """
        query_lower = query.lower()

        for keyword, doc_type in self.DOCUMENT_TYPE_KEYWORDS.items():
            if keyword in query_lower:
                return doc_type

        return None

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify a query and extract metadata filters.

        Args:
            query: User query string

        Returns:
            Dictionary with detected filters:
            {
                'providers': Set[str],
                'document_type': Optional[str],
                'has_filters': bool
            }
        """
        providers = self.extract_providers(query)
        document_type = self.extract_document_type(query)

        return {
            "providers": providers,
            "document_type": document_type,
            "has_filters": bool(providers or document_type),
        }


class Retriever:
    """
    Metadata-filtered retrieval system.

    Performs vector similarity search with optional metadata filtering
    based on provider names and document types extracted from queries.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        query_classifier: Optional[QueryClassifier] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance for similarity search
            query_classifier: QueryClassifier instance (if None, creates default)
            embedding_provider: EmbeddingProvider instance (if None, creates default)
        """
        self.vector_store = vector_store
        self.query_classifier = query_classifier or QueryClassifier()
        self.embedding_provider = (
            embedding_provider or self.vector_store.embedding_provider
        )
        logger.info("Initialized Retriever")

    def _compute_cosine_similarity(
        self, query_embedding: List[float], chunk_embedding: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            query_embedding: Query embedding vector
            chunk_embedding: Chunk embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        query_vec = np.array(query_embedding)
        chunk_vec = np.array(chunk_embedding)

        # Compute cosine similarity
        dot_product = np.dot(query_vec, chunk_vec)
        query_norm = np.linalg.norm(query_vec)
        chunk_norm = np.linalg.norm(chunk_vec)

        if query_norm == 0 or chunk_norm == 0:
            return 0.0

        similarity = dot_product / (query_norm * chunk_norm)

        # Ensure similarity is in [0, 1] range
        return float(max(0.0, min(1.0, similarity)))

    def _build_metadata_filter(
        self, classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build MongoDB query filter from classification results.

        Args:
            classification: Query classification results

        Returns:
            MongoDB filter dictionary
        """
        filters = {}

        # Add document type filter
        if classification["document_type"]:
            filters["document_type"] = classification["document_type"]

        # Add provider filter (support multiple providers with $in)
        if classification["providers"]:
            if len(classification["providers"]) == 1:
                filters["provider_name"] = list(classification["providers"])[0]
            else:
                filters["provider_name"] = {"$in": list(classification["providers"])}

        return filters

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        apply_filters: bool = True,
        min_similarity: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query string
            top_k: Number of top results to return (default: 5)
            apply_filters: Whether to apply metadata filters (default: True)
            min_similarity: Minimum similarity threshold (default: 0.0)

        Returns:
            List of RetrievalResult objects, sorted by similarity score (descending)
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        # Generate query embedding
        logger.info("Generating embedding for query: %s", query[:100])
        query_embedding = self.embedding_provider.embed_text(query)

        # Classify query to extract filters
        classification = self.query_classifier.classify_query(query)
        logger.info("Query classification: %s", classification)

        # Build metadata filter
        metadata_filter = {}
        if apply_filters and classification["has_filters"]:
            metadata_filter = self._build_metadata_filter(classification)
            logger.info("Applying metadata filters: %s", metadata_filter)

        # Retrieve candidate chunks from vector store
        # If we have filters, retrieve filtered chunks; otherwise retrieve all
        if metadata_filter:
            candidates = self.vector_store.get_chunks_by_metadata(
                metadata_filter, limit=1000
            )
        else:
            # Retrieve all chunks (up to reasonable limit)
            candidates = list(
                self.vector_store.collection.find().limit(1000)
            )

        if not candidates:
            logger.warning("No candidate chunks found")
            return []

        logger.info("Retrieved %d candidate chunks", len(candidates))

        # Compute similarity scores for each candidate
        results = []
        for chunk in candidates:
            # Skip chunks without embeddings
            if "embedding" not in chunk:
                continue

            similarity = self._compute_cosine_similarity(
                query_embedding, chunk["embedding"]
            )

            # Apply minimum similarity threshold
            if similarity < min_similarity:
                continue

            result = RetrievalResult(
                chunk_id=chunk["chunk_id"],
                content=chunk["content"],
                metadata=chunk["metadata"],
                similarity_score=similarity,
            )
            results.append(result)

        # Sort by similarity score (descending) and return top-k
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        top_results = results[:top_k]

        logger.info(
            "Returning %d results (from %d candidates)", len(top_results), len(results)
        )

        # Log top result for debugging
        if top_results:
            logger.info(
                "Top result: score=%.4f, chunk_id=%s",
                top_results[0].similarity_score,
                top_results[0].chunk_id,
            )

        return top_results

    def retrieve_with_fallback(
        self, query: str, top_k: int = 5, min_similarity: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve with automatic fallback to unfiltered search.

        First attempts filtered retrieval. If no results found,
        falls back to unfiltered retrieval.

        Args:
            query: User query string
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of RetrievalResult objects
        """
        # Try filtered retrieval first
        results = self.retrieve(
            query, top_k=top_k, apply_filters=True, min_similarity=min_similarity
        )

        # Fallback to unfiltered search if no results
        if not results:
            logger.info("No results with filters, falling back to unfiltered search")
            results = self.retrieve(
                query, top_k=top_k, apply_filters=False, min_similarity=min_similarity
            )

        return results


def main() -> None:
    """Example usage of the retrieval system."""
    import os

    # Check for required environment variables
    if not os.getenv("MONGODB_URI") or not os.getenv("OPENAI_API_KEY"):
        print("Error: MONGODB_URI and OPENAI_API_KEY environment variables required")
        return

    # Initialize vector store
    print("Initializing vector store...")
    vector_store = VectorStore(
        database_name="yuno_rag_test", collection_name="chunks_test"
    )

    # Create retriever
    print("Creating retriever...")
    retriever = Retriever(vector_store)

    # Example queries
    queries = [
        "How do Stripe refunds work?",
        "What is PayPal's authentication flow?",
        "Show me Jira tickets about payment retry logic",
        "How do we handle errors?",  # No provider specified
    ]

    print("\n" + "=" * 60)
    print("Retrieval Examples")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        # Classify query
        classification = retriever.query_classifier.classify_query(query)
        print(f"Classification: {classification}")

        # Note: This example won't return real results without indexed data
        # In a real scenario, you would have chunks indexed in the vector store
        print("(No actual retrieval - vector store is empty)")

    # Clean up
    vector_store.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
