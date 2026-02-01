"""
Tests for CLI Query Interface
Tests the command-line interface and RAG pipeline integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.cli import RAGPipeline, format_response
from src.generation.answer_generator import Answer


class TestRAGPipeline:
    """Tests for RAGPipeline class."""

    @pytest.fixture
    def mock_components(self):
        """Mock all pipeline components."""
        with patch("src.cli.VectorStore") as mock_vs, patch(
            "src.cli.Retriever"
        ) as mock_ret, patch("src.cli.AnswerGenerator") as mock_gen:

            # Mock vector store
            mock_vs_instance = MagicMock()
            mock_vs.return_value = mock_vs_instance

            # Mock retriever
            mock_ret_instance = MagicMock()
            mock_ret.return_value = mock_ret_instance

            # Mock query classifier
            mock_classifier = MagicMock()
            mock_classifier.classify_query.return_value = {
                "providers": {"Stripe"},
                "document_type": None,
                "has_filters": True,
            }
            mock_ret_instance.query_classifier = mock_classifier

            # Mock retrieval results
            mock_result = Mock()
            mock_result.chunk_id = "chunk_001"
            mock_result.content = "Test content"
            mock_result.metadata = {"filename": "test.pdf", "provider_name": "Stripe"}
            mock_ret_instance.retrieve_with_fallback.return_value = [mock_result]

            # Mock answer generator
            mock_gen_instance = MagicMock()
            mock_answer = Answer(
                query="Test query",
                answer="Test answer",
                sources=[
                    {
                        "filename": "test.pdf",
                        "provider_name": "Stripe",
                        "document_type": "provider_doc",
                        "chunk_id": "chunk_001",
                    }
                ],
                model="gpt-4",
                has_answer=True,
            )
            mock_gen_instance.generate_answer.return_value = mock_answer
            mock_gen.return_value = mock_gen_instance

            yield {
                "vector_store": mock_vs,
                "retriever": mock_ret,
                "generator": mock_gen,
            }

    def test_init(self, mock_components, monkeypatch):
        """Test pipeline initialization."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        pipeline = RAGPipeline(model="gpt-4")

        assert pipeline.vector_store is not None
        assert pipeline.retriever is not None
        assert pipeline.answer_generator is not None

    def test_init_custom_params(self, mock_components, monkeypatch):
        """Test initialization with custom parameters."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        pipeline = RAGPipeline(
            model="gpt-3.5-turbo",
            mongodb_uri="mongodb://custom",
            database_name="custom_db",
            collection_name="custom_collection",
        )

        # Verify VectorStore was called with custom params
        mock_components["vector_store"].assert_called_once_with(
            mongodb_uri="mongodb://custom",
            database_name="custom_db",
            collection_name="custom_collection",
        )

    def test_query_success(self, mock_components, monkeypatch):
        """Test successful query processing."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        pipeline = RAGPipeline()
        response = pipeline.query("Test query", top_k=5)

        assert "answer" in response
        assert "classification" in response
        assert "num_chunks_retrieved" in response
        assert "timing" in response
        assert response["answer"].query == "Test query"
        assert response["num_chunks_retrieved"] == 1

    def test_query_timing(self, mock_components, monkeypatch):
        """Test that query includes timing information."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        pipeline = RAGPipeline()
        response = pipeline.query("Test query")

        assert "retrieval" in response["timing"]
        assert "generation" in response["timing"]
        assert "total" in response["timing"]
        assert response["timing"]["total"] > 0

    def test_query_with_params(self, mock_components, monkeypatch):
        """Test query with custom parameters."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        pipeline = RAGPipeline()
        response = pipeline.query("Test query", top_k=10, min_similarity=0.7)

        # Verify retriever was called with correct params
        mock_retriever = mock_components["retriever"].return_value
        mock_retriever.retrieve_with_fallback.assert_called_once_with(
            "Test query", top_k=10, min_similarity=0.7
        )

    def test_close(self, mock_components, monkeypatch):
        """Test closing pipeline connections."""
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        pipeline = RAGPipeline()
        pipeline.close()

        # Verify vector store close was called
        mock_components["vector_store"].return_value.close.assert_called_once()


class TestFormatResponse:
    """Tests for format_response function."""

    @pytest.fixture
    def sample_response(self):
        """Sample response for testing."""
        answer = Answer(
            query="How do Stripe refunds work?",
            answer="Stripe refunds are processed through the API.",
            sources=[
                {
                    "filename": "stripe_refunds.pdf",
                    "provider_name": "Stripe",
                    "document_type": "provider_doc",
                    "chunk_id": "chunk_001",
                }
            ],
            model="gpt-4",
            has_answer=True,
        )

        return {
            "answer": answer,
            "classification": {
                "providers": {"Stripe"},
                "document_type": None,
                "has_filters": True,
            },
            "num_chunks_retrieved": 3,
            "timing": {"retrieval": 0.5, "generation": 1.2, "total": 1.7},
        }

    def test_format_response_structure(self, sample_response):
        """Test formatted response contains all sections."""
        formatted = format_response(sample_response)

        assert "YUNO RAG PIPELINE RESPONSE" in formatted
        assert "Query:" in formatted
        assert "How do Stripe refunds work?" in formatted
        assert "Filters Applied:" in formatted
        assert "ANSWER:" in formatted
        assert "Stripe refunds are processed" in formatted
        assert "SOURCES:" in formatted
        assert "stripe_refunds.pdf" in formatted
        assert "STATS:" in formatted

    def test_format_response_filters(self, sample_response):
        """Test filters section formatting."""
        formatted = format_response(sample_response)

        assert "Filters Applied:" in formatted
        assert "Providers: Stripe" in formatted

    def test_format_response_no_filters(self, sample_response):
        """Test formatting when no filters applied."""
        sample_response["classification"]["has_filters"] = False
        sample_response["classification"]["providers"] = set()

        formatted = format_response(sample_response)

        assert "Filters Applied:" not in formatted

    def test_format_response_sources(self, sample_response):
        """Test sources section formatting."""
        formatted = format_response(sample_response)

        assert "SOURCES:" in formatted
        assert "1. stripe_refunds.pdf" in formatted
        assert "Provider: Stripe" in formatted
        assert "Type: provider_doc" in formatted

    def test_format_response_no_sources(self, sample_response):
        """Test formatting when no sources."""
        sample_response["answer"].sources = []

        formatted = format_response(sample_response)

        # Sources section should still exist but be empty or minimal
        assert "STATS:" in formatted

    def test_format_response_timing(self, sample_response):
        """Test timing information formatting."""
        formatted = format_response(sample_response)

        assert "Retrieval Time: 0.500s" in formatted
        assert "Generation Time: 1.200s" in formatted
        assert "Total Time: 1.700s" in formatted

    def test_format_response_stats(self, sample_response):
        """Test stats section formatting."""
        formatted = format_response(sample_response)

        assert "Chunks Retrieved: 3" in formatted
        assert "Model: gpt-4" in formatted
        assert "Has Answer: True" in formatted


class TestCLIMain:
    """Tests for main CLI function."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock RAGPipeline."""
        with patch("src.cli.RAGPipeline") as mock_pipe:
            mock_instance = MagicMock()

            answer = Answer(
                query="Test query",
                answer="Test answer",
                sources=[],
                model="gpt-4",
                has_answer=True,
            )

            mock_instance.query.return_value = {
                "answer": answer,
                "classification": {"providers": set(), "has_filters": False},
                "num_chunks_retrieved": 1,
                "timing": {"retrieval": 0.1, "generation": 0.2, "total": 0.3},
            }

            mock_pipe.return_value = mock_instance
            yield mock_pipe

    def test_main_requires_query(self, monkeypatch):
        """Test that CLI requires a query argument."""
        from src.cli import main

        # Mock sys.argv
        with patch("sys.argv", ["cli.py"]):
            result = main()
            assert result == 1

    def test_main_requires_mongodb_uri(self, monkeypatch, mock_pipeline):
        """Test that CLI requires MONGODB_URI."""
        from src.cli import main

        monkeypatch.delenv("MONGODB_URI", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        with patch("sys.argv", ["cli.py", "test query"]):
            result = main()
            assert result == 1

    def test_main_requires_api_key_for_gpt(self, monkeypatch, mock_pipeline):
        """Test that CLI requires OPENAI_API_KEY for GPT models."""
        from src.cli import main

        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("sys.argv", ["cli.py", "test query", "--model", "gpt-4"]):
            result = main()
            assert result == 1

    def test_main_requires_api_key_for_claude(self, monkeypatch, mock_pipeline):
        """Test that CLI requires ANTHROPIC_API_KEY for Claude models."""
        from src.cli import main

        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch(
            "sys.argv", ["cli.py", "test query", "--model", "claude-3-sonnet-20240229"]
        ):
            result = main()
            assert result == 1

    def test_main_success(self, monkeypatch, mock_pipeline):
        """Test successful CLI execution."""
        from src.cli import main

        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        with patch("sys.argv", ["cli.py", "test query"]):
            result = main()
            assert result == 0

    def test_main_with_options(self, monkeypatch, mock_pipeline):
        """Test CLI with various options."""
        from src.cli import main

        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        with patch(
            "sys.argv",
            [
                "cli.py",
                "test query",
                "--model",
                "gpt-3.5-turbo",
                "--top-k",
                "10",
                "--min-similarity",
                "0.7",
            ],
        ):
            result = main()
            assert result == 0

            # Verify pipeline was called with correct params
            mock_instance = mock_pipeline.return_value
            mock_instance.query.assert_called_once()
            call_args = mock_instance.query.call_args
            assert call_args[1]["top_k"] == 10
            assert call_args[1]["min_similarity"] == 0.7


def test_integration_structure():
    """Test that CLI components are properly structured."""
    from src.cli import RAGPipeline, format_response, main

    # Check that main components exist
    assert callable(RAGPipeline)
    assert callable(format_response)
    assert callable(main)

    # Check RAGPipeline has required methods
    assert hasattr(RAGPipeline, "query")
    assert hasattr(RAGPipeline, "close")
