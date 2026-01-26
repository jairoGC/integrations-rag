"""
Tests for Answer Generator
Tests LLM-based answer generation with source citations.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generation.answer_generator import AnswerGenerator, Answer


class TestAnswer:
    """Tests for Answer dataclass."""

    def test_answer_creation(self):
        """Test creating an Answer instance."""
        sources = [
            {
                "filename": "test.pdf",
                "provider_name": "Stripe",
                "document_type": "provider_doc",
                "chunk_id": "chunk_001",
            }
        ]

        answer = Answer(
            query="Test query",
            answer="Test answer",
            sources=sources,
            model="gpt-4",
            has_answer=True,
        )

        assert answer.query == "Test query"
        assert answer.answer == "Test answer"
        assert answer.sources == sources
        assert answer.model == "gpt-4"
        assert answer.has_answer is True

    def test_answer_to_dict(self):
        """Test converting answer to dictionary."""
        sources = [{"filename": "test.pdf"}]
        answer = Answer(
            query="Test query",
            answer="Test answer",
            sources=sources,
            model="gpt-4",
            has_answer=True,
        )

        result = answer.to_dict()

        assert result["query"] == "Test query"
        assert result["answer"] == "Test answer"
        assert result["sources"] == sources
        assert result["model"] == "gpt-4"
        assert result["has_answer"] is True


class TestAnswerGenerator:
    """Tests for AnswerGenerator class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        with patch("src.generation.answer_generator.OpenAI") as mock_openai:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "This is a test answer."
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            yield mock_openai

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing."""
        with patch("src.generation.answer_generator.anthropic.Anthropic") as mock_anthropic:
            mock_response = Mock()
            mock_content = Mock()
            mock_content.text = "This is a test answer from Claude."
            mock_response.content = [mock_content]
            mock_anthropic.return_value.messages.create.return_value = mock_response
            yield mock_anthropic

    @pytest.fixture
    def sample_chunks(self):
        """Sample retrieved chunks for testing."""
        return [
            {
                "chunk_id": "chunk_001",
                "content": "Stripe provides a refund API.",
                "metadata": {
                    "filename": "stripe_refunds.pdf",
                    "provider_name": "Stripe",
                    "document_type": "provider_doc",
                },
            },
            {
                "chunk_id": "chunk_002",
                "content": "PayPal also supports refunds.",
                "metadata": {
                    "filename": "paypal_refunds.pdf",
                    "provider_name": "PayPal",
                    "document_type": "provider_doc",
                },
            },
        ]

    def test_init_gpt(self, mock_openai_client, monkeypatch):
        """Test initialization with GPT model."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        assert generator.model == "gpt-4"
        assert generator.is_claude is False

    def test_init_claude(self, mock_anthropic_client, monkeypatch):
        """Test initialization with Claude model."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        generator = AnswerGenerator(model="claude-3-sonnet-20240229")

        assert generator.model == "claude-3-sonnet-20240229"
        assert generator.is_claude is True

    def test_init_without_openai_key(self, mock_openai_client, monkeypatch):
        """Test initialization fails without OpenAI API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError, match="OpenAI API key required"):
            AnswerGenerator(model="gpt-4")

    def test_init_without_anthropic_key(self, mock_anthropic_client, monkeypatch):
        """Test initialization fails without Anthropic API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="Anthropic API key required"):
            AnswerGenerator(model="claude-3-sonnet-20240229")

    def test_format_context(self, mock_openai_client, sample_chunks, monkeypatch):
        """Test formatting chunks into context."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        context = generator._format_context(sample_chunks)

        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "stripe_refunds.pdf" in context
        assert "paypal_refunds.pdf" in context
        assert "Stripe" in context
        assert "PayPal" in context

    def test_format_context_empty(self, mock_openai_client, monkeypatch):
        """Test formatting empty chunks."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        context = generator._format_context([])

        assert context == "No relevant documentation found."

    def test_extract_sources(self, mock_openai_client, sample_chunks, monkeypatch):
        """Test extracting source metadata."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        sources = generator._extract_sources(sample_chunks)

        assert len(sources) == 2
        assert sources[0]["filename"] == "stripe_refunds.pdf"
        assert sources[0]["provider_name"] == "Stripe"
        assert sources[0]["chunk_id"] == "chunk_001"
        assert sources[1]["filename"] == "paypal_refunds.pdf"

    def test_generate_answer_gpt(
        self, mock_openai_client, sample_chunks, monkeypatch
    ):
        """Test generating answer with GPT."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        answer = generator.generate_answer(
            query="How do refunds work?",
            retrieved_chunks=sample_chunks,
        )

        assert isinstance(answer, Answer)
        assert answer.query == "How do refunds work?"
        assert answer.answer == "This is a test answer."
        assert len(answer.sources) == 2
        assert answer.model == "gpt-4"
        assert answer.has_answer is True

    def test_generate_answer_claude(
        self, mock_anthropic_client, sample_chunks, monkeypatch
    ):
        """Test generating answer with Claude."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        generator = AnswerGenerator(model="claude-3-sonnet-20240229")

        answer = generator.generate_answer(
            query="How do refunds work?",
            retrieved_chunks=sample_chunks,
        )

        assert isinstance(answer, Answer)
        assert answer.query == "How do refunds work?"
        assert answer.answer == "This is a test answer from Claude."
        assert len(answer.sources) == 2
        assert answer.has_answer is True

    def test_generate_answer_empty_query(
        self, mock_openai_client, sample_chunks, monkeypatch
    ):
        """Test generating answer with empty query raises error."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            generator.generate_answer(query="", retrieved_chunks=sample_chunks)

    def test_generate_answer_insufficient_chunks(
        self, mock_openai_client, monkeypatch
    ):
        """Test generating answer with insufficient chunks."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        answer = generator.generate_answer(
            query="Test query",
            retrieved_chunks=[],
            min_chunks=1,
        )

        assert answer.has_answer is False
        assert "don't have enough information" in answer.answer
        assert len(answer.sources) == 0

    def test_generate_answer_min_chunks_threshold(
        self, mock_openai_client, sample_chunks, monkeypatch
    ):
        """Test minimum chunks threshold."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        # Only 2 chunks, but require 3
        answer = generator.generate_answer(
            query="Test query",
            retrieved_chunks=sample_chunks,
            min_chunks=3,
        )

        assert answer.has_answer is False
        assert "don't have enough information" in answer.answer

    def test_generate_answer_no_info_detection(
        self, mock_openai_client, sample_chunks, monkeypatch
    ):
        """Test detection of 'no information' responses."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

        # Mock response indicating no information
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "I don't have enough information to answer this question."
        )
        mock_openai_client.return_value.chat.completions.create.return_value = mock_response

        generator = AnswerGenerator(model="gpt-4")

        answer = generator.generate_answer(
            query="Test query",
            retrieved_chunks=sample_chunks,
        )

        assert answer.has_answer is False

    def test_format_answer_with_citations(
        self, mock_openai_client, sample_chunks, monkeypatch
    ):
        """Test formatting answer with citations."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        answer = Answer(
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

        formatted = generator.format_answer_with_citations(answer)

        assert "Query: Test query" in formatted
        assert "Test answer" in formatted
        assert "Sources:" in formatted
        assert "test.pdf" in formatted
        assert "Provider: Stripe" in formatted
        assert "Type: provider_doc" in formatted

    def test_format_answer_no_sources(
        self, mock_openai_client, monkeypatch
    ):
        """Test formatting answer without sources."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        answer = Answer(
            query="Test query",
            answer="No information found",
            sources=[],
            model="gpt-4",
            has_answer=False,
        )

        formatted = generator.format_answer_with_citations(answer)

        assert "Query: Test query" in formatted
        assert "No information found" in formatted
        assert "Sources:" not in formatted

    def test_system_prompt_exists(self, mock_openai_client, monkeypatch):
        """Test that system prompt is defined."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        assert generator.SYSTEM_PROMPT is not None
        assert "based on provided" in generator.SYSTEM_PROMPT.lower()
        assert "cite" in generator.SYSTEM_PROMPT.lower()

    def test_gpt_api_call_structure(
        self, mock_openai_client, sample_chunks, monkeypatch
    ):
        """Test that GPT API is called with correct structure."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")
        generator = AnswerGenerator(model="gpt-4")

        generator.generate_answer(
            query="Test query",
            retrieved_chunks=sample_chunks,
        )

        # Verify API was called
        mock_openai_client.return_value.chat.completions.create.assert_called_once()

        # Get the call arguments
        call_args = mock_openai_client.return_value.chat.completions.create.call_args

        # Check model
        assert call_args.kwargs["model"] == "gpt-4"

        # Check messages structure
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test query" in messages[1]["content"]

    def test_claude_api_call_structure(
        self, mock_anthropic_client, sample_chunks, monkeypatch
    ):
        """Test that Claude API is called with correct structure."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
        generator = AnswerGenerator(model="claude-3-sonnet-20240229")

        generator.generate_answer(
            query="Test query",
            retrieved_chunks=sample_chunks,
        )

        # Verify API was called
        mock_anthropic_client.return_value.messages.create.assert_called_once()

        # Get the call arguments
        call_args = mock_anthropic_client.return_value.messages.create.call_args

        # Check model
        assert call_args.kwargs["model"] == "claude-3-sonnet-20240229"

        # Check system prompt
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == generator.SYSTEM_PROMPT

        # Check messages
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Test query" in messages[0]["content"]


def test_integration_example():
    """Test that the generation API structure is correct."""
    with patch("src.generation.answer_generator.OpenAI"), patch.dict(
        os.environ, {"OPENAI_API_KEY": "test_key"}
    ):
        # Should be able to create generator
        generator = AnswerGenerator(model="gpt-4")
        assert generator is not None

        # Should have required methods
        assert hasattr(generator, "generate_answer")
        assert hasattr(generator, "format_answer_with_citations")
        assert hasattr(generator, "_format_context")
        assert hasattr(generator, "_extract_sources")
