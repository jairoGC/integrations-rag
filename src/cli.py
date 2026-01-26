"""
CLI Query Interface for Yuno RAG Pipeline
Provides command-line interface for querying documentation.
"""

import sys
import os
import argparse
import logging
import time
from typing import Optional
from src.indexing import VectorStore
from src.retrieval import Retriever
from src.generation import AnswerGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline integrating all components.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        mongodb_uri: Optional[str] = None,
        database_name: str = "yuno_rag",
        collection_name: str = "chunks",
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            model: LLM model to use (default: gpt-4)
            mongodb_uri: MongoDB connection URI
            database_name: MongoDB database name
            collection_name: MongoDB collection name
        """
        logger.info("Initializing RAG pipeline...")

        # Initialize vector store
        self.vector_store = VectorStore(
            mongodb_uri=mongodb_uri,
            database_name=database_name,
            collection_name=collection_name,
        )
        logger.info("Vector store initialized")

        # Initialize retriever
        self.retriever = Retriever(self.vector_store)
        logger.info("Retriever initialized")

        # Initialize answer generator
        self.answer_generator = AnswerGenerator(model=model)
        logger.info("Answer generator initialized with model: %s", model)

        logger.info("RAG pipeline ready!")

    def query(
        self, query_text: str, top_k: int = 5, min_similarity: float = 0.0
    ) -> dict:
        """
        Process a query through the complete RAG pipeline.

        Args:
            query_text: User query
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            Dictionary with answer, sources, filters, and timing
        """
        start_time = time.time()

        # Classify query to show filters
        classification = self.retriever.query_classifier.classify_query(query_text)

        # Retrieve relevant chunks
        retrieval_start = time.time()
        results = self.retriever.retrieve_with_fallback(
            query_text, top_k=top_k, min_similarity=min_similarity
        )
        retrieval_time = time.time() - retrieval_start

        # Convert to chunks format
        chunks = [
            {"chunk_id": r.chunk_id, "content": r.content, "metadata": r.metadata}
            for r in results
        ]

        # Generate answer
        generation_start = time.time()
        answer = self.answer_generator.generate_answer(
            query=query_text, retrieved_chunks=chunks, min_chunks=1
        )
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        return {
            "answer": answer,
            "classification": classification,
            "num_chunks_retrieved": len(results),
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": total_time,
            },
        }

    def close(self) -> None:
        """Close connections."""
        self.vector_store.close()


def format_response(response: dict) -> str:
    """
    Format pipeline response for display.

    Args:
        response: Response dictionary from pipeline.query()

    Returns:
        Formatted string
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("YUNO RAG PIPELINE RESPONSE")
    lines.append("=" * 70)
    lines.append("")

    # Query and filters
    lines.append(f"Query: {response['answer'].query}")
    lines.append("")

    classification = response["classification"]
    if classification["has_filters"]:
        lines.append("Filters Applied:")
        if classification["providers"]:
            lines.append(f"  - Providers: {', '.join(classification['providers'])}")
        if classification["document_type"]:
            lines.append(f"  - Document Type: {classification['document_type']}")
        lines.append("")

    # Answer
    lines.append("-" * 70)
    lines.append("ANSWER:")
    lines.append("-" * 70)
    lines.append("")
    lines.append(response["answer"].answer)
    lines.append("")

    # Sources
    if response["answer"].sources:
        lines.append("-" * 70)
        lines.append("SOURCES:")
        lines.append("-" * 70)
        for i, source in enumerate(response["answer"].sources, 1):
            lines.append(f"\n{i}. {source['filename']}")
            if source.get("provider_name"):
                lines.append(f"   Provider: {source['provider_name']}")
            if source.get("document_type"):
                lines.append(f"   Type: {source['document_type']}")

    # Stats
    lines.append("")
    lines.append("-" * 70)
    lines.append("STATS:")
    lines.append("-" * 70)
    lines.append(f"Chunks Retrieved: {response['num_chunks_retrieved']}")
    lines.append(f"Retrieval Time: {response['timing']['retrieval']:.3f}s")
    lines.append(f"Generation Time: {response['timing']['generation']:.3f}s")
    lines.append(f"Total Time: {response['timing']['total']:.3f}s")
    lines.append(f"Model: {response['answer'].model}")
    lines.append(f"Has Answer: {response['answer'].has_answer}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Query the Yuno RAG documentation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "How do Stripe refunds work?"
  %(prog)s "What is PayPal's authentication flow?" --model claude-3-sonnet-20240229
  %(prog)s "Show me error handling" --top-k 10 --min-similarity 0.7
  %(prog)s --query "How do we integrate Adyen?"

Environment Variables:
  OPENAI_API_KEY      - Required for GPT models
  ANTHROPIC_API_KEY   - Required for Claude models
  MONGODB_URI         - MongoDB connection string (required)
        """,
    )

    parser.add_argument(
        "query", nargs="?", help="Query string (can also use --query flag)"
    )
    parser.add_argument("--query", "-q", help="Query string (alternative to positional)")
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4",
        help="LLM model to use (default: gpt-4)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--min-similarity",
        "-s",
        type=float,
        default=0.0,
        help="Minimum similarity threshold (default: 0.0)",
    )
    parser.add_argument(
        "--database",
        "-d",
        default="yuno_rag",
        help="MongoDB database name (default: yuno_rag)",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default="chunks",
        help="MongoDB collection name (default: chunks)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get query from args
    query_text = args.query or args.query
    if not query_text:
        parser.print_help()
        print("\nError: Query is required", file=sys.stderr)
        return 1

    # Check for required environment variables
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("Error: MONGODB_URI environment variable is required", file=sys.stderr)
        return 1

    # Check for API keys based on model
    if args.model.startswith("claude"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            print(
                "Error: ANTHROPIC_API_KEY environment variable is required for Claude models",
                file=sys.stderr,
            )
            return 1
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "Error: OPENAI_API_KEY environment variable is required for GPT models",
                file=sys.stderr,
            )
            return 1

    try:
        # Initialize pipeline
        print("Initializing RAG pipeline...\n")
        pipeline = RAGPipeline(
            model=args.model,
            mongodb_uri=mongodb_uri,
            database_name=args.database,
            collection_name=args.collection,
        )

        # Process query
        print(f"Processing query: {query_text}\n")
        response = pipeline.query(
            query_text, top_k=args.top_k, min_similarity=args.min_similarity
        )

        # Display response
        print(format_response(response))

        # Check if response time is acceptable
        if response["timing"]["total"] > 3.0:
            print(
                f"\nWarning: Response time ({response['timing']['total']:.3f}s) "
                "exceeded 3 second target",
                file=sys.stderr,
            )

        # Close pipeline
        pipeline.close()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        logger.exception("Error processing query")
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
