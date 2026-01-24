"""Tests for PDF ingestion pipeline."""

import os
import tempfile
from pathlib import Path
import pytest
from PyPDF2 import PdfWriter

from src.ingestion.pdf_ingestion import PDFIngestionPipeline, Document


class TestDocument:
    """Test Document class."""

    def test_document_creation(self) -> None:
        """Test creating a document."""
        doc = Document(
            content="Test content",
            document_type="provider_doc",
            provider_name="Stripe",
            date="2024-01-01",
            filename="test.pdf",
        )

        assert doc.content == "Test content"
        assert doc.metadata["document_type"] == "provider_doc"
        assert doc.metadata["provider_name"] == "Stripe"
        assert doc.metadata["date"] == "2024-01-01"
        assert doc.metadata["filename"] == "test.pdf"

    def test_document_to_dict(self) -> None:
        """Test converting document to dictionary."""
        doc = Document(
            content="Test content",
            document_type="provider_doc",
            provider_name="Stripe",
            date="2024-01-01",
            filename="test.pdf",
        )

        result = doc.to_dict()

        assert result["content"] == "Test content"
        assert result["metadata"]["document_type"] == "provider_doc"
        assert result["metadata"]["provider_name"] == "Stripe"


class TestPDFIngestionPipeline:
    """Test PDFIngestionPipeline class."""

    def test_extract_metadata_with_full_format(self) -> None:
        """Test metadata extraction from well-formatted filename."""
        pipeline = PDFIngestionPipeline()

        metadata = pipeline.extract_metadata("provider_doc_Stripe_20240101.pdf")

        assert metadata["document_type"] == "provider_doc"
        assert metadata["provider_name"] == "Stripe"
        assert metadata["date"] == "2024-01-01"

    def test_extract_metadata_with_jira_ticket(self) -> None:
        """Test metadata extraction for Jira ticket."""
        pipeline = PDFIngestionPipeline()

        metadata = pipeline.extract_metadata("jira_ticket_PayPal_2024-01-15.pdf")

        assert metadata["document_type"] == "jira_ticket"
        assert metadata["provider_name"] == "PayPal"
        assert metadata["date"] == "2024-01-15"

    def test_extract_metadata_partial_format(self) -> None:
        """Test metadata extraction from partial filename."""
        pipeline = PDFIngestionPipeline()

        metadata = pipeline.extract_metadata("Stripe_documentation.pdf")

        assert metadata["document_type"] == "provider_doc"
        assert metadata["provider_name"] == "Stripe"

    def test_extract_metadata_minimal(self) -> None:
        """Test metadata extraction from minimal filename."""
        pipeline = PDFIngestionPipeline()

        metadata = pipeline.extract_metadata("random_file.pdf")

        assert metadata["document_type"] == "provider_doc"
        assert metadata["provider_name"] == "random"
        assert metadata["date"] is not None

    def test_ingest_directory_not_exists(self) -> None:
        """Test ingesting from non-existent directory."""
        pipeline = PDFIngestionPipeline()

        with pytest.raises(ValueError, match="Directory does not exist"):
            pipeline.ingest_directory("/nonexistent/path")

    def test_ingest_directory_empty(self) -> None:
        """Test ingesting from empty directory."""
        pipeline = PDFIngestionPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            documents = pipeline.ingest_directory(tmpdir)

            assert len(documents) == 0

    def test_get_documents(self) -> None:
        """Test getting documents as dictionaries."""
        pipeline = PDFIngestionPipeline()

        # Add a sample document
        doc = Document(
            content="Test",
            document_type="provider_doc",
            provider_name="Stripe",
            date="2024-01-01",
            filename="test.pdf",
        )
        pipeline.documents.append(doc)

        documents = pipeline.get_documents()

        assert len(documents) == 1
        assert documents[0]["content"] == "Test"
        assert documents[0]["metadata"]["provider_name"] == "Stripe"
