"""
PDF Document Ingestion Pipeline
Extracts text and metadata from PDF files for RAG indexing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Represents a document with content and metadata."""

    def __init__(
        self,
        content: str,
        document_type: str,
        provider_name: Optional[str],
        date: Optional[str],
        filename: str,
    ) -> None:
        self.content = content
        self.metadata = {
            "document_type": document_type,
            "provider_name": provider_name,
            "date": date,
            "filename": filename,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata,
        }


class PDFIngestionPipeline:
    """Pipeline for ingesting PDF documents."""

    VALID_DOCUMENT_TYPES = ["provider_doc", "jira_ticket"]

    def __init__(self) -> None:
        self.documents: List[Document] = []

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []

                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)

                return "\n".join(text_content)

        except (IOError, PyPDF2.errors.PdfReadError) as e:
            logger.error("Failed to extract text from %s: %s", pdf_path, str(e))
            return None

    def extract_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from filename.
        Expected format: {document_type}_{provider_name}_{date}.pdf

        Args:
            filename: Name of the PDF file

        Returns:
            Dictionary containing extracted metadata
        """
        # Remove .pdf extension
        base_name = filename.replace('.pdf', '')

        # Default metadata
        metadata = {
            "document_type": "provider_doc",
            "provider_name": None,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

        # Try to parse filename components
        # Check for document types that contain underscores
        doc_type_extracted = False
        remaining_name = base_name

        for doc_type in self.VALID_DOCUMENT_TYPES:
            if base_name.startswith(doc_type + '_'):
                metadata["document_type"] = doc_type
                remaining_name = base_name[len(doc_type) + 1:]  # +1 for underscore
                doc_type_extracted = True
                break

        # Split remaining parts
        parts = remaining_name.split('_')

        # Extract provider name (first part after doc type)
        if len(parts) > 0 and parts[0]:
            metadata["provider_name"] = parts[0]

        # Extract date (second part after doc type)
        if len(parts) > 1:
            try:
                # Try to parse date
                date_str = parts[1]
                # Accept formats like YYYY-MM-DD or YYYYMMDD
                if len(date_str) == 8 and date_str.isdigit():
                    metadata["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                elif len(date_str) == 10 and date_str.count('-') == 2:
                    metadata["date"] = date_str
            except (ValueError, IndexError):
                pass

        return metadata

    def ingest_directory(self, directory_path: str) -> List[Document]:
        """
        Ingest all PDF files from a directory.

        Args:
            directory_path: Path to directory containing PDF files

        Returns:
            List of ingested documents
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("**/*.pdf"))

        if not pdf_files:
            logger.warning("No PDF files found in %s", directory_path)
            return []

        logger.info("Found %d PDF files to process", len(pdf_files))

        for pdf_path in pdf_files:
            try:
                # Extract text content
                text_content = self.extract_text_from_pdf(pdf_path)

                if text_content is None:
                    logger.warning("Skipping %s - text extraction failed", pdf_path.name)
                    continue

                if not text_content.strip():
                    logger.warning("Skipping %s - no text content found", pdf_path.name)
                    continue

                # Extract metadata
                metadata = self.extract_metadata(pdf_path.name)

                # Create document
                document = Document(
                    content=text_content,
                    document_type=metadata["document_type"],
                    provider_name=metadata["provider_name"],
                    date=metadata["date"],
                    filename=pdf_path.name,
                )

                self.documents.append(document)
                logger.info("Successfully ingested: %s", pdf_path.name)

            except (IOError, ValueError) as e:
                logger.error("Error processing %s: %s", pdf_path.name, str(e))
                continue

        logger.info("Ingestion complete: %d documents processed", len(self.documents))
        return self.documents

    def get_documents(self) -> List[Dict[str, Any]]:
        """Return all ingested documents as dictionaries."""
        return [doc.to_dict() for doc in self.documents]


def main() -> None:
    """Example usage of the PDF ingestion pipeline."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_ingestion.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    pipeline = PDFIngestionPipeline()
    documents = pipeline.ingest_directory(directory_path)

    # Print summary
    print("\n" + "="*60)
    print("Ingestion Summary")
    print("="*60)
    print(f"Total documents ingested: {len(documents)}")

    # Group by document type
    type_counts: Dict[str, int] = {}
    for doc in documents:
        doc_type = doc.metadata["document_type"]
        if doc_type:
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

    print("\nBy document type:")
    for doc_type, count in type_counts.items():
        print(f"  {doc_type}: {count}")

    # Show sample documents
    print("\nSample documents:")
    for i, doc in enumerate(documents[:3]):
        print(f"\n{i+1}. {doc.metadata['filename']}")
        print(f"   Type: {doc.metadata['document_type']}")
        print(f"   Provider: {doc.metadata['provider_name']}")
        print(f"   Date: {doc.metadata['date']}")
        print(f"   Content length: {len(doc.content)} characters")


if __name__ == "__main__":
    main()
