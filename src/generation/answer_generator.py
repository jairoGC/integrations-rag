"""
LLM-based Answer Generation
Generates natural language answers from retrieved context using Claude or GPT-4.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from openai import OpenAI
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """Represents a generated answer with source citations."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    has_answer: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert answer to dictionary format."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "model": self.model,
            "has_answer": self.has_answer,
        }


class AnswerGenerator:
    """
    LLM-based answer generator.

    Generates natural language answers from retrieved chunks using
    Claude or GPT-4, with proper source citations.
    """

    # System prompt for answer generation
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided documentation.

IMPORTANT RULES:
1. Answer ONLY based on the provided context below. Do not use external knowledge.
2. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documentation to answer this question."
3. Always cite your sources by referencing the document filename or provider name.
4. Be concise and clear in your answers.
5. If multiple sources provide relevant information, mention all of them."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the answer generator.

        Args:
            model: Model to use ('gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet')
            api_key: OpenAI API key (if using GPT models)
            anthropic_api_key: Anthropic API key (if using Claude models)
        """
        self.model = model
        self.is_claude = model.startswith("claude")
        self.client: Union[OpenAI, anthropic.Anthropic]

        if self.is_claude:
            self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.anthropic_api_key:
                raise ValueError(
                    "Anthropic API key required for Claude models. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            logger.info("Initialized AnswerGenerator with Claude model: %s", model)
        else:
            self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError(
                    "OpenAI API key required for GPT models. "
                    "Set OPENAI_API_KEY environment variable."
                )
            self.client = OpenAI(api_key=self.openai_api_key)
            logger.info("Initialized AnswerGenerator with GPT model: %s", model)

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: List of retrieved chunks with content and metadata

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant documentation found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            provider = metadata.get("provider_name", "Unknown")
            doc_type = metadata.get("document_type", "Unknown")

            context_parts.append(
                f"[Source {i}] (File: {filename}, Provider: {provider}, Type: {doc_type})\n"
                f"{chunk.get('content', '')}\n"
            )

        return "\n".join(context_parts)

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source metadata from chunks.

        Args:
            chunks: List of retrieved chunks

        Returns:
            List of source dictionaries
        """
        sources = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source = {
                "filename": metadata.get("filename", "Unknown"),
                "provider_name": metadata.get("provider_name"),
                "document_type": metadata.get("document_type"),
                "chunk_id": chunk.get("chunk_id"),
            }
            sources.append(source)
        return sources

    def _generate_with_claude(self, query: str, context: str) -> str:
        """
        Generate answer using Claude.

        Args:
            query: User query
            context: Formatted context

        Returns:
            Generated answer
        """
        message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        response = self.client.messages.create(  # type: ignore
            model=self.model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )

        # Extract text from first content block
        first_block = response.content[0]
        if hasattr(first_block, 'text'):
            return first_block.text
        return str(first_block)

    def _generate_with_gpt(self, query: str, context: str) -> str:
        """
        Generate answer using GPT.

        Args:
            query: User query
            context: Formatted context

        Returns:
            Generated answer
        """
        message = f"Context:\n{context}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(  # type: ignore
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=1024,
            temperature=0.7,
        )

        return response.choices[0].message.content or ""

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        min_chunks: int = 1,
    ) -> Answer:
        """
        Generate an answer from retrieved chunks.

        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks (with 'content' and 'metadata')
            min_chunks: Minimum number of chunks required to generate answer

        Returns:
            Answer object with generated text and source citations
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Check if we have enough chunks
        if len(retrieved_chunks) < min_chunks:
            logger.warning(
                "Insufficient chunks for query: %d < %d", len(retrieved_chunks), min_chunks
            )
            return Answer(
                query=query,
                answer="I don't have enough information in the provided documentation to answer this question.",
                sources=[],
                model=self.model,
                has_answer=False,
            )

        # Format context
        context = self._format_context(retrieved_chunks)
        logger.info("Generating answer for query: %s", query[:100])

        # Generate answer
        try:
            if self.is_claude:
                answer_text = self._generate_with_claude(query, context)
            else:
                answer_text = self._generate_with_gpt(query, context)

            # Extract sources
            sources = self._extract_sources(retrieved_chunks)

            # Check if answer indicates insufficient information
            no_info_phrases = [
                "don't have enough information",
                "not enough information",
                "cannot answer",
                "unable to answer",
                "not found in the documentation",
            ]
            has_answer = not any(
                phrase in answer_text.lower() for phrase in no_info_phrases
            )

            logger.info("Answer generated successfully (has_answer=%s)", has_answer)

            return Answer(
                query=query,
                answer=answer_text,
                sources=sources,
                model=self.model,
                has_answer=has_answer,
            )

        except Exception as e:
            logger.error("Failed to generate answer: %s", str(e))
            raise

    def format_answer_with_citations(self, answer: Answer) -> str:
        """
        Format answer with source citations for display.

        Args:
            answer: Answer object

        Returns:
            Formatted string with answer and citations
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Query: {answer.query}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(answer.answer)
        lines.append("")

        if answer.sources:
            lines.append("-" * 60)
            lines.append("Sources:")
            for i, source in enumerate(answer.sources, 1):
                lines.append(f"{i}. {source['filename']}")
                if source.get("provider_name"):
                    lines.append(f"   Provider: {source['provider_name']}")
                if source.get("document_type"):
                    lines.append(f"   Type: {source['document_type']}")
            lines.append("-" * 60)

        return "\n".join(lines)


def main() -> None:
    """Example usage of the answer generator."""
    # Example chunks
    sample_chunks = [
        {
            "chunk_id": "chunk_001",
            "content": "Stripe provides a comprehensive refund API. To process a refund, "
            "use the POST /v1/refunds endpoint with the charge ID. Refunds are processed "
            "immediately and funds are returned to the customer within 5-10 business days.",
            "metadata": {
                "filename": "stripe_refunds.pdf",
                "provider_name": "Stripe",
                "document_type": "provider_doc",
            },
        }
    ]

    print("Initializing answer generator...")
    print("(This example requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable)")

    # Try to use GPT-4 if available, otherwise skip
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("No API keys found. Skipping example.")
        return

    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            generator = AnswerGenerator(model="claude-3-sonnet-20240229")
        else:
            generator = AnswerGenerator(model="gpt-4")

        print("\nGenerating answer...")
        answer = generator.generate_answer(
            query="How do I process a refund with Stripe?",
            retrieved_chunks=sample_chunks,
        )

        print("\n" + generator.format_answer_with_citations(answer))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
