"""
Text Chunking Strategy
Implements recursive character splitter for semantic document chunking.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    content: str
    metadata: Dict[str, Any]
    chunk_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }


class TextChunker:
    """
    Implements recursive character-based text chunking strategy.

    Chunks are created with:
    - Target size: 1000 tokens (approximately 4000 characters)
    - Overlap: 200 tokens (approximately 800 characters)
    """

    # Approximate characters per token (average for English text)
    CHARS_PER_TOKEN = 4

    def __init__(
        self, chunk_size_tokens: int = 1000, overlap_tokens: int = 200
    ) -> None:
        """
        Initialize the text chunker.

        Args:
            chunk_size_tokens: Target chunk size in tokens (default 1000)
            overlap_tokens: Overlap between chunks in tokens (default 200)
        """
        self.chunk_size = chunk_size_tokens * self.CHARS_PER_TOKEN
        self.overlap = overlap_tokens * self.CHARS_PER_TOKEN
        self.chunks: List[Chunk] = []

        # Separators for recursive splitting (from largest to smallest unit)
        self.separators = [
            "\n\n\n",  # Multiple blank lines (section breaks)
            "\n\n",  # Double newlines (paragraph breaks)
            "\n",  # Single newlines
            ". ",  # Sentences
            "! ",  # Exclamation sentences
            "? ",  # Question sentences
            "; ",  # Semicolons
            ", ",  # Commas
            " ",  # Words
            "",  # Characters
        ]

    def extract_section_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract section headers from text.

        Looks for common header patterns:
        - Lines with all caps
        - Lines ending with colons
        - Lines that are short and followed by blank line
        - Markdown-style headers (# Header)

        Args:
            text: Input text

        Returns:
            List of dictionaries with header text and position
        """
        headers = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Check for markdown headers
            if line_stripped.startswith("#"):
                headers.append(
                    {"text": line_stripped, "position": text.find(line), "line": i}
                )
                continue

            # Check for all caps headers (at least 3 words)
            if (
                line_stripped.isupper()
                and len(line_stripped.split()) >= 3
                and len(line_stripped) < 100
            ):
                headers.append(
                    {"text": line_stripped, "position": text.find(line), "line": i}
                )
                continue

            # Check for headers ending with colon
            if line_stripped.endswith(":") and len(line_stripped) < 100:
                headers.append(
                    {"text": line_stripped, "position": text.find(line), "line": i}
                )

        return headers

    def find_section_for_position(
        self, position: int, headers: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find the section header that applies to a given position in text.

        Args:
            position: Character position in text
            headers: List of header dictionaries

        Returns:
            Section header text or None
        """
        applicable_header = None

        for header in headers:
            if header["position"] <= position:
                applicable_header = header["text"]
            else:
                break

        return applicable_header

    def split_text_recursive(
        self, text: str, separators: List[str], start_pos: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Recursively split text using separators from coarse to fine.

        Args:
            text: Text to split
            separators: List of separators to try
            start_pos: Starting position in original text

        Returns:
            List of text chunks with position information
        """
        if not text or len(text) <= self.chunk_size:
            return [{"text": text, "start_pos": start_pos}]

        # Try each separator
        for i, separator in enumerate(separators):
            if separator and separator in text:
                splits = text.split(separator)
                chunks = []
                current_pos = start_pos

                for split in splits:
                    # Add separator back except for last split
                    if split != splits[-1]:
                        split = split + separator

                    # If split is still too large, recurse with finer separators
                    if len(split) > self.chunk_size and i < len(separators) - 1:
                        sub_chunks = self.split_text_recursive(
                            split, separators[i + 1 :], current_pos
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append({"text": split, "start_pos": current_pos})

                    current_pos += len(split)

                return chunks

        # No separators found, return as-is
        return [{"text": text, "start_pos": start_pos}]

    def merge_small_chunks(
        self, chunks: List[Dict[str, Any]], min_size: int
    ) -> List[Dict[str, Any]]:
        """
        Merge chunks that are smaller than minimum size.

        Args:
            chunks: List of chunk dictionaries
            min_size: Minimum chunk size in characters

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            # If current chunk is too small, merge with next
            if len(current_chunk["text"]) < min_size:
                current_chunk["text"] += next_chunk["text"]
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        # Add the last chunk
        merged.append(current_chunk)

        return merged

    def add_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add overlap between consecutive chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            List of chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]

            # Add overlap from previous chunk
            if i > 0 and self.overlap > 0:
                prev_text = chunks[i - 1]["text"]
                overlap_text = prev_text[-self.overlap :]
                chunk_text = overlap_text + chunk_text

            overlapped.append({"text": chunk_text, "start_pos": chunk["start_pos"]})

        return overlapped

    def chunk_document(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Chunk a document into smaller pieces with metadata.

        Args:
            content: Document text content
            metadata: Document metadata to inherit

        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            logger.warning("Empty content provided for chunking")
            return []

        # Extract section headers
        headers = self.extract_section_headers(content)

        # Split text recursively
        raw_chunks = self.split_text_recursive(content, self.separators)

        # Merge very small chunks
        min_chunk_size = self.chunk_size // 4
        merged_chunks = self.merge_small_chunks(raw_chunks, min_chunk_size)

        # Add overlap between chunks
        overlapped_chunks = self.add_overlap(merged_chunks)

        # Create Chunk objects with metadata
        chunks = []
        for i, chunk_data in enumerate(overlapped_chunks):
            # Find applicable section header
            section_title = self.find_section_for_position(
                chunk_data["start_pos"], headers
            )

            # Create chunk metadata (inherit from document + add chunk-specific)
            chunk_metadata = metadata.copy()
            chunk_metadata["section_title"] = section_title
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(overlapped_chunks)

            chunk = Chunk(
                content=chunk_data["text"].strip(),
                metadata=chunk_metadata,
                chunk_id=i,
            )
            chunks.append(chunk)

        self.chunks.extend(chunks)
        logger.info(
            "Created %d chunks from document %s",
            len(chunks),
            metadata.get("filename", "unknown"),
        )

        return chunks

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about the chunks.

        Returns:
            Dictionary with chunk statistics
        """
        if not self.chunks:
            return {
                "count": 0,
                "avg_size": 0,
                "min_size": 0,
                "max_size": 0,
                "size_distribution": {},
            }

        sizes = [len(chunk.content) for chunk in self.chunks]
        avg_size = sum(sizes) / len(sizes)

        # Calculate size distribution (buckets of 1000 chars)
        distribution: Dict[str, int] = {}
        for size in sizes:
            bucket = (size // 1000) * 1000
            bucket_label = f"{bucket}-{bucket+999}"
            distribution[bucket_label] = distribution.get(bucket_label, 0) + 1

        return {
            "count": len(self.chunks),
            "avg_size": int(avg_size),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "size_distribution": distribution,
        }


def main() -> None:
    """Example usage of the text chunker."""
    # Example document
    sample_doc = """
INTRODUCTION

This is a sample document to demonstrate the chunking functionality.
It contains multiple sections and paragraphs.

SECTION ONE: GETTING STARTED

Here is the first section with some content. This section talks about
how to get started with the system. It has multiple sentences that
provide detailed information.

The section continues with more paragraphs. Each paragraph contains
useful information that should be preserved in chunks.

SECTION TWO: ADVANCED FEATURES

This section describes advanced features. The content here is more
technical and detailed. It requires careful chunking to maintain context.

Additional paragraphs provide more details about specific features
and how to use them effectively.
    """.strip()

    sample_metadata = {
        "document_type": "provider_doc",
        "provider_name": "Example",
        "filename": "sample.pdf",
        "date": "2024-01-01",
    }

    # Create chunker and process document
    chunker = TextChunker(chunk_size_tokens=50, overlap_tokens=10)
    chunks = chunker.chunk_document(sample_doc, sample_metadata)

    # Print results
    print("\n" + "=" * 60)
    print("Chunking Results")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Size: {len(chunk.content)} characters")
        print(f"  Section: {chunk.metadata.get('section_title', 'N/A')}")
        print(f"  Content preview: {chunk.content[:100]}...")

    # Print statistics
    stats = chunker.get_chunk_statistics()
    print("\n" + "=" * 60)
    print("Chunk Statistics")
    print("=" * 60)
    print(f"Total chunks: {stats['count']}")
    print(f"Average size: {stats['avg_size']} characters")
    print(f"Min size: {stats['min_size']} characters")
    print(f"Max size: {stats['max_size']} characters")
    print("\nSize distribution:")
    for bucket, count in sorted(stats["size_distribution"].items()):
        print(f"  {bucket} chars: {count} chunks")


if __name__ == "__main__":
    main()
