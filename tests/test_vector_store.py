"""
Tests for Vector Store
Tests MongoDB Atlas integration and embedding generation.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.indexing.vector_store import VectorStore, EmbeddingProvider, ChunkWithEmbedding


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("src.indexing.vector_store.OpenAI") as mock_openai:
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.return_value.embeddings.create.return_value = mock_response
        yield mock_openai


@pytest.fixture
def mock_mongo_client():
    """Mock MongoDB client for testing."""
    with patch("src.indexing.vector_store.MongoClient") as mock_client:
        # Mock collection operations
        mock_collection = MagicMock()
        mock_collection.create_index.return_value = None
        mock_collection.count_documents.return_value = 0
        mock_collection.update_one.return_value = Mock(upserted_id="test_id")
        mock_collection.bulk_write.return_value = Mock(
            upserted_count=2, modified_count=0
        )
        mock_collection.find.return_value.limit.return_value = []
        mock_collection.delete_many.return_value = Mock(deleted_count=0)

        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        mock_client.return_value.__getitem__.return_value = mock_db
        mock_client.return_value.close.return_value = None

        yield mock_client


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider class."""

    def test_init_with_api_key(self, mock_openai_client):
        """Test initialization with provided API key."""
        provider = EmbeddingProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.model == "text-embedding-3-small"

    def test_init_with_env_var(self, mock_openai_client, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env_key")
        provider = EmbeddingProvider()
        assert provider.api_key == "env_key"

    def test_init_without_api_key(self, mock_openai_client, monkeypatch):
        """Test initialization fails without API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            EmbeddingProvider()

    def test_embed_text_success(self, mock_openai_client):
        """Test successful text embedding."""
        provider = EmbeddingProvider(api_key="test_key")
        embedding = provider.embed_text("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_text_empty_string(self, mock_openai_client):
        """Test embedding empty string raises error."""
        provider = EmbeddingProvider(api_key="test_key")

        with pytest.raises(ValueError, match="Cannot embed empty text"):
            provider.embed_text("")

        with pytest.raises(ValueError, match="Cannot embed empty text"):
            provider.embed_text("   ")

    def test_embed_batch_success(self, mock_openai_client):
        """Test successful batch embedding."""
        # Mock batch response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        provider = EmbeddingProvider(api_key="test_key")
        embeddings = provider.embed_batch(["text 1", "text 2"])

        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)

    def test_embed_batch_empty_list(self, mock_openai_client):
        """Test embedding empty list returns empty list."""
        provider = EmbeddingProvider(api_key="test_key")
        embeddings = provider.embed_batch([])
        assert embeddings == []

    def test_embed_batch_filters_empty_texts(self, mock_openai_client):
        """Test batch embedding filters out empty strings."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        provider = EmbeddingProvider(api_key="test_key")
        embeddings = provider.embed_batch(["valid text", "", "   "])

        assert len(embeddings) == 1


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_init_with_uri(self, mock_mongo_client, mock_openai_client, monkeypatch):
        """Test initialization with provided MongoDB URI."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        store = VectorStore(mongodb_uri="mongodb://localhost:27017")
        assert store.mongodb_uri == "mongodb://localhost:27017"

    def test_init_with_env_var(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test initialization with environment variable."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://env:27017")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        store = VectorStore()
        assert store.mongodb_uri == "mongodb://env:27017"

    def test_init_without_uri(self, mock_mongo_client, mock_openai_client, monkeypatch):
        """Test initialization fails without MongoDB URI."""
        monkeypatch.delenv("MONGODB_URI", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        with pytest.raises(ValueError, match="MongoDB URI not provided"):
            VectorStore()

    def test_create_indexes(self, mock_mongo_client, mock_openai_client, monkeypatch):
        """Test that metadata indexes are created."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()

        # Check that create_index was called
        mock_collection = store.collection
        assert mock_collection.create_index.call_count == 3

    def test_index_chunk_success(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test successful chunk indexing."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()
        chunk_id = store.index_chunk(
            chunk_id="test_001",
            content="test content",
            metadata={"document_type": "provider_doc", "provider_name": "Stripe"},
        )

        assert chunk_id is not None
        store.collection.update_one.assert_called_once()

    def test_index_chunk_empty_content(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test indexing empty content raises error."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()

        with pytest.raises(ValueError, match="Cannot index empty content"):
            store.index_chunk("test_001", "", {})

    def test_index_batch_success(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test successful batch indexing."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Mock batch embedding response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * 1536),
            Mock(embedding=[0.2] * 1536),
        ]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        store = VectorStore()
        chunks = [
            {
                "chunk_id": "chunk_001",
                "content": "content 1",
                "metadata": {"document_type": "provider_doc"},
            },
            {
                "chunk_id": "chunk_002",
                "content": "content 2",
                "metadata": {"document_type": "jira_ticket"},
            },
        ]

        indexed_ids = store.index_batch(chunks)

        assert len(indexed_ids) == 2
        assert "chunk_001" in indexed_ids
        assert "chunk_002" in indexed_ids
        store.collection.bulk_write.assert_called_once()

    def test_index_batch_empty_list(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test indexing empty list returns empty list."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()
        indexed_ids = store.index_batch([])
        assert indexed_ids == []

    def test_index_batch_batching(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test batch indexing processes in batches."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Mock multiple batch responses
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536) for _ in range(3)]
        mock_openai_client.return_value.embeddings.create.return_value = mock_response

        store = VectorStore()

        # Create 3 chunks with batch_size=2
        chunks = [
            {
                "chunk_id": f"chunk_{i:03d}",
                "content": f"content {i}",
                "metadata": {"document_type": "provider_doc"},
            }
            for i in range(3)
        ]

        indexed_ids = store.index_batch(chunks, batch_size=2)

        assert len(indexed_ids) == 3
        # Should call bulk_write twice (batch 1: 2 items, batch 2: 1 item)
        assert store.collection.bulk_write.call_count == 2

    def test_get_chunk_count(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test getting chunk count."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()
        store.collection.count_documents.return_value = 42

        count = store.get_chunk_count()
        assert count == 42

    def test_get_chunks_by_metadata(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test retrieving chunks by metadata."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        mock_chunks = [
            {
                "chunk_id": "test_001",
                "content": "test",
                "metadata": {"provider_name": "Stripe"},
            }
        ]
        mock_mongo_client.return_value[
            "test_db"
        ].chunks.find.return_value.limit.return_value = mock_chunks

        store = VectorStore()
        store.collection.find.return_value.limit.return_value = mock_chunks

        chunks = store.get_chunks_by_metadata({"provider_name": "Stripe"})

        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "test_001"
        store.collection.find.assert_called_once()

    def test_delete_all_chunks(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test deleting all chunks."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()
        store.collection.delete_many.return_value = Mock(deleted_count=10)

        deleted = store.delete_all_chunks()
        assert deleted == 10
        store.collection.delete_many.assert_called_once()

    def test_close_connection(
        self, mock_mongo_client, mock_openai_client, monkeypatch
    ):
        """Test closing MongoDB connection."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        store = VectorStore()
        store.close()

        store.client.close.assert_called_once()


class TestChunkWithEmbedding:
    """Tests for ChunkWithEmbedding dataclass."""

    def test_chunk_creation(self):
        """Test creating a ChunkWithEmbedding instance."""
        chunk = ChunkWithEmbedding(
            chunk_id="test_001",
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
        )

        assert chunk.chunk_id == "test_001"
        assert chunk.content == "test content"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.metadata == {"key": "value"}


def test_integration_example():
    """Test that the example code structure is correct."""
    # This test verifies the API structure without actual connections
    with patch("src.indexing.vector_store.MongoClient"), patch(
        "src.indexing.vector_store.OpenAI"
    ):
        with patch.dict(
            os.environ, {"MONGODB_URI": "mongodb://test", "OPENAI_API_KEY": "test_key"}
        ):
            # Should be able to create store
            store = VectorStore()
            assert store is not None

            # Should have required methods
            assert hasattr(store, "index_chunk")
            assert hasattr(store, "index_batch")
            assert hasattr(store, "get_chunk_count")
            assert hasattr(store, "get_chunks_by_metadata")
            assert hasattr(store, "delete_all_chunks")
            assert hasattr(store, "close")
