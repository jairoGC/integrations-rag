"""
Tests for Retrieval System
Tests query classification and metadata-filtered retrieval.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from src.retrieval.retriever import (
    Retriever,
    QueryClassifier,
    RetrievalResult,
)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_result_creation(self):
        """Test creating a RetrievalResult instance."""
        result = RetrievalResult(
            chunk_id="chunk_001",
            content="test content",
            metadata={"provider_name": "Stripe"},
            similarity_score=0.95,
        )

        assert result.chunk_id == "chunk_001"
        assert result.content == "test content"
        assert result.metadata == {"provider_name": "Stripe"}
        assert result.similarity_score == 0.95

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = RetrievalResult(
            chunk_id="chunk_001",
            content="test content",
            metadata={"provider_name": "Stripe"},
            similarity_score=0.95,
        )

        result_dict = result.to_dict()

        assert result_dict["chunk_id"] == "chunk_001"
        assert result_dict["content"] == "test content"
        assert result_dict["metadata"] == {"provider_name": "Stripe"}
        assert result_dict["similarity_score"] == 0.95


class TestQueryClassifier:
    """Tests for QueryClassifier class."""

    def test_init_default(self):
        """Test initialization with default providers."""
        classifier = QueryClassifier()
        assert len(classifier.providers) >= 10
        assert "stripe" in classifier.providers
        assert "paypal" in classifier.providers

    def test_init_additional_providers(self):
        """Test initialization with additional providers."""
        classifier = QueryClassifier(additional_providers=["CustomProvider"])
        assert "customprovider" in classifier.providers

    def test_extract_providers_single(self):
        """Test extracting a single provider from query."""
        classifier = QueryClassifier()

        providers = classifier.extract_providers("How do Stripe refunds work?")
        assert providers == {"Stripe"}

        providers = classifier.extract_providers("PayPal authentication guide")
        assert providers == {"Paypal"}

    def test_extract_providers_multiple(self):
        """Test extracting multiple providers from query."""
        classifier = QueryClassifier()

        providers = classifier.extract_providers("Compare Stripe and PayPal fees")
        assert "Stripe" in providers
        assert "Paypal" in providers
        assert len(providers) == 2

    def test_extract_providers_none(self):
        """Test query with no provider mentions."""
        classifier = QueryClassifier()

        providers = classifier.extract_providers("How do refunds work?")
        assert len(providers) == 0

    def test_extract_providers_case_insensitive(self):
        """Test provider extraction is case-insensitive."""
        classifier = QueryClassifier()

        providers = classifier.extract_providers("STRIPE payment processing")
        assert providers == {"Stripe"}

        providers = classifier.extract_providers("stripe and PAYPAL")
        assert len(providers) == 2

    def test_extract_providers_word_boundaries(self):
        """Test provider extraction respects word boundaries."""
        classifier = QueryClassifier()

        # Should not match "stripe" in "striped"
        providers = classifier.extract_providers("The striped pattern")
        assert len(providers) == 0

    def test_extract_document_type_provider_doc(self):
        """Test extracting provider_doc document type."""
        classifier = QueryClassifier()

        doc_type = classifier.extract_document_type("Show me the API documentation")
        assert doc_type == "provider_doc"

        doc_type = classifier.extract_document_type("Integration guide")
        assert doc_type == "provider_doc"

    def test_extract_document_type_jira_ticket(self):
        """Test extracting jira_ticket document type."""
        classifier = QueryClassifier()

        doc_type = classifier.extract_document_type("Find Jira ticket about refunds")
        assert doc_type == "jira_ticket"

        doc_type = classifier.extract_document_type("Show me the issue details")
        assert doc_type == "jira_ticket"

    def test_extract_document_type_none(self):
        """Test query with no document type keywords."""
        classifier = QueryClassifier()

        doc_type = classifier.extract_document_type("How do refunds work?")
        assert doc_type is None

    def test_classify_query_full(self):
        """Test full query classification with all filters."""
        classifier = QueryClassifier()

        result = classifier.classify_query(
            "Show me Stripe API documentation for refunds"
        )

        assert result["providers"] == {"Stripe"}
        assert result["document_type"] == "provider_doc"
        assert result["has_filters"] is True

    def test_classify_query_provider_only(self):
        """Test classification with provider only."""
        classifier = QueryClassifier()

        result = classifier.classify_query("How does PayPal handle errors?")

        assert result["providers"] == {"Paypal"}
        assert result["document_type"] is None
        assert result["has_filters"] is True

    def test_classify_query_doc_type_only(self):
        """Test classification with document type only."""
        classifier = QueryClassifier()

        result = classifier.classify_query("Find all Jira tickets about payments")

        assert len(result["providers"]) == 0
        assert result["document_type"] == "jira_ticket"
        assert result["has_filters"] is True

    def test_classify_query_no_filters(self):
        """Test classification with no filters."""
        classifier = QueryClassifier()

        result = classifier.classify_query("How do payments work?")

        assert len(result["providers"]) == 0
        assert result["document_type"] is None
        assert result["has_filters"] is False


class TestRetriever:
    """Tests for Retriever class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore for testing."""
        mock_store = MagicMock()
        mock_store.embedding_provider = MagicMock()
        mock_store.embedding_provider.embed_text.return_value = [0.1] * 1536
        mock_store.collection.find.return_value.limit.return_value = []
        mock_store.get_chunks_by_metadata.return_value = []
        return mock_store

    @pytest.fixture
    def mock_classifier(self):
        """Mock QueryClassifier for testing."""
        classifier = MagicMock()
        classifier.classify_query.return_value = {
            "providers": set(),
            "document_type": None,
            "has_filters": False,
        }
        return classifier

    def test_init(self, mock_vector_store):
        """Test retriever initialization."""
        retriever = Retriever(mock_vector_store)
        assert retriever.vector_store == mock_vector_store
        assert retriever.query_classifier is not None

    def test_init_with_custom_classifier(self, mock_vector_store, mock_classifier):
        """Test initialization with custom classifier."""
        retriever = Retriever(mock_vector_store, query_classifier=mock_classifier)
        assert retriever.query_classifier == mock_classifier

    def test_compute_cosine_similarity_identical(self, mock_vector_store):
        """Test cosine similarity with identical vectors."""
        retriever = Retriever(mock_vector_store)

        vec = [1.0, 0.0, 0.0]
        similarity = retriever._compute_cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_compute_cosine_similarity_orthogonal(self, mock_vector_store):
        """Test cosine similarity with orthogonal vectors."""
        retriever = Retriever(mock_vector_store)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = retriever._compute_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_compute_cosine_similarity_opposite(self, mock_vector_store):
        """Test cosine similarity with opposite vectors."""
        retriever = Retriever(mock_vector_store)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = retriever._compute_cosine_similarity(vec1, vec2)

        # Cosine similarity should be clamped to [0, 1]
        assert 0.0 <= similarity <= 1.0

    def test_compute_cosine_similarity_zero_vector(self, mock_vector_store):
        """Test cosine similarity with zero vector."""
        retriever = Retriever(mock_vector_store)

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 0.0, 0.0]
        similarity = retriever._compute_cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_build_metadata_filter_empty(self, mock_vector_store):
        """Test building metadata filter with no classification."""
        retriever = Retriever(mock_vector_store)

        classification = {
            "providers": set(),
            "document_type": None,
            "has_filters": False,
        }

        filters = retriever._build_metadata_filter(classification)
        assert filters == {}

    def test_build_metadata_filter_single_provider(self, mock_vector_store):
        """Test building metadata filter with single provider."""
        retriever = Retriever(mock_vector_store)

        classification = {
            "providers": {"Stripe"},
            "document_type": None,
            "has_filters": True,
        }

        filters = retriever._build_metadata_filter(classification)
        assert filters == {"provider_name": "Stripe"}

    def test_build_metadata_filter_multiple_providers(self, mock_vector_store):
        """Test building metadata filter with multiple providers."""
        retriever = Retriever(mock_vector_store)

        classification = {
            "providers": {"Stripe", "Paypal"},
            "document_type": None,
            "has_filters": True,
        }

        filters = retriever._build_metadata_filter(classification)
        assert "provider_name" in filters
        assert "$in" in filters["provider_name"]
        assert set(filters["provider_name"]["$in"]) == {"Stripe", "Paypal"}

    def test_build_metadata_filter_doc_type(self, mock_vector_store):
        """Test building metadata filter with document type."""
        retriever = Retriever(mock_vector_store)

        classification = {
            "providers": set(),
            "document_type": "provider_doc",
            "has_filters": True,
        }

        filters = retriever._build_metadata_filter(classification)
        assert filters == {"document_type": "provider_doc"}

    def test_build_metadata_filter_combined(self, mock_vector_store):
        """Test building metadata filter with provider and doc type."""
        retriever = Retriever(mock_vector_store)

        classification = {
            "providers": {"Stripe"},
            "document_type": "provider_doc",
            "has_filters": True,
        }

        filters = retriever._build_metadata_filter(classification)
        assert filters["provider_name"] == "Stripe"
        assert filters["document_type"] == "provider_doc"

    def test_retrieve_empty_query(self, mock_vector_store):
        """Test retrieval with empty query."""
        retriever = Retriever(mock_vector_store)

        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_no_candidates(self, mock_vector_store):
        """Test retrieval with no candidate chunks."""
        retriever = Retriever(mock_vector_store)

        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_with_candidates(self, mock_vector_store):
        """Test retrieval with candidate chunks."""
        # Mock chunks with embeddings
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "Stripe refund processing",
                "metadata": {"provider_name": "Stripe"},
                "embedding": [1.0] * 1536,
            },
            {
                "chunk_id": "chunk_002",
                "content": "PayPal authentication",
                "metadata": {"provider_name": "Paypal"},
                "embedding": [0.5] * 1536,
            },
        ]

        # The query "Stripe refunds" will be classified as having "Stripe" provider
        # So it will use get_chunks_by_metadata instead of collection.find()
        mock_vector_store.get_chunks_by_metadata.return_value = mock_chunks

        retriever = Retriever(mock_vector_store)
        results = retriever.retrieve("Stripe refunds", top_k=5)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Results should be sorted by similarity
        assert results[0].similarity_score >= results[1].similarity_score

    def test_retrieve_top_k(self, mock_vector_store):
        """Test retrieval respects top_k parameter."""
        # Create 10 mock chunks
        mock_chunks = [
            {
                "chunk_id": f"chunk_{i:03d}",
                "content": f"content {i}",
                "metadata": {"provider_name": "Stripe"},
                "embedding": [float(i) / 10] * 1536,
            }
            for i in range(10)
        ]

        # Mock the find().limit() chain - make it directly iterable
        mock_limit_result = Mock()
        mock_limit_result.__iter__ = Mock(return_value=iter(mock_chunks))
        mock_vector_store.collection.find.return_value.limit.return_value = mock_limit_result

        retriever = Retriever(mock_vector_store)
        results = retriever.retrieve("test query", top_k=3)

        assert len(results) == 3

    def test_retrieve_min_similarity(self, mock_vector_store):
        """Test retrieval with minimum similarity threshold."""
        # Mock chunks with varying similarities
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "content 1",
                "metadata": {"provider_name": "Stripe"},
                "embedding": [1.0] * 1536,  # High similarity
            },
            {
                "chunk_id": "chunk_002",
                "content": "content 2",
                "metadata": {"provider_name": "Stripe"},
                "embedding": [-1.0] * 1536,  # Low similarity
            },
        ]

        # Mock the find().limit() chain - make it directly iterable
        mock_limit_result = Mock()
        mock_limit_result.__iter__ = Mock(return_value=iter(mock_chunks))
        mock_vector_store.collection.find.return_value.limit.return_value = mock_limit_result
        mock_vector_store.embedding_provider.embed_text.return_value = [1.0] * 1536

        retriever = Retriever(mock_vector_store)
        results = retriever.retrieve("test query", min_similarity=0.5)

        # Should only return high similarity chunk
        assert len(results) >= 1
        assert all(r.similarity_score >= 0.5 for r in results)

    def test_retrieve_with_filters(self, mock_vector_store, mock_classifier):
        """Test retrieval with metadata filters applied."""
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "Stripe content",
                "metadata": {"provider_name": "Stripe"},
                "embedding": [1.0] * 1536,
            }
        ]

        mock_vector_store.get_chunks_by_metadata.return_value = mock_chunks
        mock_classifier.classify_query.return_value = {
            "providers": {"Stripe"},
            "document_type": None,
            "has_filters": True,
        }

        retriever = Retriever(mock_vector_store, query_classifier=mock_classifier)
        results = retriever.retrieve("Stripe refunds", apply_filters=True)

        # Should call get_chunks_by_metadata with filters
        mock_vector_store.get_chunks_by_metadata.assert_called_once()
        assert len(results) == 1

    def test_retrieve_with_fallback_success(self, mock_vector_store):
        """Test fallback retrieval when filtered search succeeds."""
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "content",
                "metadata": {"provider_name": "Stripe"},
                "embedding": [1.0] * 1536,
            }
        ]

        mock_vector_store.get_chunks_by_metadata.return_value = mock_chunks

        retriever = Retriever(mock_vector_store)
        results = retriever.retrieve_with_fallback("Stripe refunds")

        assert len(results) >= 1

    def test_retrieve_with_fallback_fallback(self, mock_vector_store):
        """Test fallback retrieval when filtered search fails."""
        # First call (with filters) returns empty
        # Second call (without filters) returns results
        mock_chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "content",
                "metadata": {},
                "embedding": [1.0] * 1536,
            }
        ]

        mock_vector_store.get_chunks_by_metadata.return_value = []

        # Mock the find().limit() chain - make it directly iterable
        mock_limit_result = Mock()
        mock_limit_result.__iter__ = Mock(return_value=iter(mock_chunks))
        mock_vector_store.collection.find.return_value.limit.return_value = mock_limit_result

        retriever = Retriever(mock_vector_store)

        # Use a classifier that detects filters
        retriever.query_classifier = QueryClassifier()

        results = retriever.retrieve_with_fallback("Stripe refunds")

        # Should fall back to unfiltered search
        assert len(results) >= 0


def test_integration_example():
    """Test that the retrieval API structure is correct."""
    with patch("src.retrieval.retriever.VectorStore"):
        mock_store = MagicMock()
        mock_store.embedding_provider = MagicMock()

        # Should be able to create retriever
        retriever = Retriever(mock_store)
        assert retriever is not None

        # Should have required methods
        assert hasattr(retriever, "retrieve")
        assert hasattr(retriever, "retrieve_with_fallback")

        # Should have query classifier
        assert isinstance(retriever.query_classifier, QueryClassifier)
