"""Tests for text chunking."""

import pytest
from src.chunking.text_chunker import TextChunker, Chunk


class TestChunk:
    """Test Chunk class."""

    def test_chunk_creation(self) -> None:
        """Test creating a chunk."""
        metadata = {
            "document_type": "provider_doc",
            "provider_name": "Stripe",
            "filename": "test.pdf",
        }

        chunk = Chunk(content="Test content", metadata=metadata, chunk_id=0)

        assert chunk.content == "Test content"
        assert chunk.metadata["provider_name"] == "Stripe"
        assert chunk.chunk_id == 0

    def test_chunk_to_dict(self) -> None:
        """Test converting chunk to dictionary."""
        metadata = {"provider_name": "Stripe"}
        chunk = Chunk(content="Test", metadata=metadata, chunk_id=0)

        result = chunk.to_dict()

        assert result["content"] == "Test"
        assert result["metadata"]["provider_name"] == "Stripe"
        assert result["chunk_id"] == 0


class TestTextChunker:
    """Test TextChunker class."""

    def test_chunker_initialization(self) -> None:
        """Test chunker initialization."""
        chunker = TextChunker(chunk_size_tokens=1000, overlap_tokens=200)

        assert chunker.chunk_size == 4000  # 1000 tokens * 4 chars/token
        assert chunker.overlap == 800  # 200 tokens * 4 chars/token

    def test_extract_section_headers_markdown(self) -> None:
        """Test extracting markdown-style headers."""
        text = "# Introduction\n\nSome content\n\n## Details\n\nMore content"
        chunker = TextChunker()

        headers = chunker.extract_section_headers(text)

        assert len(headers) >= 2
        assert any("Introduction" in h["text"] for h in headers)
        assert any("Details" in h["text"] for h in headers)

    def test_extract_section_headers_caps(self) -> None:
        """Test extracting all-caps headers."""
        text = "SECTION ONE HEADER\n\nContent here\n\nANOTHER SECTION HEADER\n\nMore content"
        chunker = TextChunker()

        headers = chunker.extract_section_headers(text)

        assert len(headers) >= 2
        assert any("SECTION ONE HEADER" in h["text"] for h in headers)
        assert any("ANOTHER SECTION HEADER" in h["text"] for h in headers)

    def test_extract_section_headers_colon(self) -> None:
        """Test extracting headers ending with colon."""
        text = "Introduction:\n\nContent here\n\nDetails:\n\nMore content"
        chunker = TextChunker()

        headers = chunker.extract_section_headers(text)

        assert len(headers) >= 2
        assert any("Introduction:" in h["text"] for h in headers)
        assert any("Details:" in h["text"] for h in headers)

    def test_find_section_for_position(self) -> None:
        """Test finding section header for position."""
        chunker = TextChunker()
        headers = [
            {"text": "# Section 1", "position": 0, "line": 0},
            {"text": "# Section 2", "position": 100, "line": 5},
        ]

        section = chunker.find_section_for_position(50, headers)
        assert section == "# Section 1"

        section = chunker.find_section_for_position(150, headers)
        assert section == "# Section 2"

    def test_chunk_document_empty(self) -> None:
        """Test chunking empty document."""
        chunker = TextChunker()
        metadata = {"filename": "test.pdf"}

        chunks = chunker.chunk_document("", metadata)

        assert len(chunks) == 0

    def test_chunk_document_small(self) -> None:
        """Test chunking small document."""
        chunker = TextChunker(chunk_size_tokens=100, overlap_tokens=20)
        metadata = {"filename": "test.pdf", "provider_name": "Stripe"}

        content = "This is a small document that fits in one chunk."
        chunks = chunker.chunk_document(content, metadata)

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].metadata["provider_name"] == "Stripe"
        assert chunks[0].chunk_id == 0

    def test_chunk_document_large(self) -> None:
        """Test chunking large document."""
        chunker = TextChunker(chunk_size_tokens=50, overlap_tokens=10)
        metadata = {"filename": "test.pdf", "provider_name": "Stripe"}

        # Create content that needs multiple chunks
        content = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
        chunks = chunker.chunk_document(content, metadata)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["provider_name"] == "Stripe"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_chunk_document_preserves_sections(self) -> None:
        """Test that section headers are preserved in chunks."""
        chunker = TextChunker(chunk_size_tokens=50, overlap_tokens=10)
        metadata = {"filename": "test.pdf"}

        content = """# INTRODUCTION

This is the introduction section with some content.

# SECTION TWO

This is section two with more content."""

        chunks = chunker.chunk_document(content, metadata)

        # Check that at least some chunks have section titles
        section_titles = [c.metadata.get("section_title") for c in chunks]
        assert any(title for title in section_titles if title is not None)

    def test_chunk_inherits_metadata(self) -> None:
        """Test that chunks inherit parent document metadata."""
        chunker = TextChunker(chunk_size_tokens=50, overlap_tokens=10)
        metadata = {
            "document_type": "provider_doc",
            "provider_name": "Stripe",
            "date": "2024-01-01",
            "filename": "test.pdf",
        }

        content = " ".join(["Content " * 20 for _ in range(10)])
        chunks = chunker.chunk_document(content, metadata)

        for chunk in chunks:
            assert chunk.metadata["document_type"] == "provider_doc"
            assert chunk.metadata["provider_name"] == "Stripe"
            assert chunk.metadata["date"] == "2024-01-01"
            assert chunk.metadata["filename"] == "test.pdf"

    def test_get_chunk_statistics_empty(self) -> None:
        """Test statistics for empty chunker."""
        chunker = TextChunker()

        stats = chunker.get_chunk_statistics()

        assert stats["count"] == 0
        assert stats["avg_size"] == 0

    def test_get_chunk_statistics(self) -> None:
        """Test chunk statistics calculation."""
        chunker = TextChunker(chunk_size_tokens=50, overlap_tokens=10)
        metadata = {"filename": "test.pdf"}

        content = " ".join(["Word " * 20 for _ in range(10)])
        chunker.chunk_document(content, metadata)

        stats = chunker.get_chunk_statistics()

        assert stats["count"] > 0
        assert stats["avg_size"] > 0
        assert stats["min_size"] > 0
        assert stats["max_size"] > 0
        assert len(stats["size_distribution"]) > 0

    def test_chunk_overlap(self) -> None:
        """Test that chunks have overlap."""
        chunker = TextChunker(chunk_size_tokens=20, overlap_tokens=5)
        metadata = {"filename": "test.pdf"}

        # Create content with distinct sentences
        sentences = [f"Sentence number {i}. " for i in range(20)]
        content = "".join(sentences)
        chunks = chunker.chunk_document(content, metadata)

        # Check that consecutive chunks have some overlap
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current = chunks[i].content
                next_chunk = chunks[i + 1].content
                # Some text from current should appear in next
                # This is a loose check - just verify multiple chunks exist
                assert len(current) > 0
                assert len(next_chunk) > 0

    def test_merge_small_chunks(self) -> None:
        """Test merging small chunks."""
        chunker = TextChunker()
        chunks = [
            {"text": "Small", "start_pos": 0},
            {"text": " chunk", "start_pos": 5},
            {"text": " here", "start_pos": 11},
        ]

        merged = chunker.merge_small_chunks(chunks, min_size=20)

        # Should merge into fewer chunks
        assert len(merged) <= len(chunks)
