"""PDF document ingestion module using Unstructured library."""

import logging
from pathlib import Path
from typing import List, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from ..models import PDFParagraph
from ..config import Config


class PDFIngester:
    """Ingests and processes PDF documents using Unstructured library with chunking by title."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """
        Initialize the PDFIngester.
        
        Args:
            config: Configuration object containing PDF directory path
            logger: Logger instance for logging operations
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pdf_dir = config.pdf_dir
    
    def ingest_directory(self, directory: Optional[Path] = None) -> List[PDFParagraph]:
        """
        Ingest all PDF files from the specified directory.
        
        Args:
            directory: Directory path to ingest from. If None, uses config.pdf_dir
            
        Returns:
            List of PDFParagraph objects extracted from all PDFs
        """
        target_dir = directory or self.pdf_dir
        
        if not target_dir.exists():
            self.logger.error(f"PDF directory does not exist: {target_dir}")
            return []
        
        if not target_dir.is_dir():
            self.logger.error(f"PDF path is not a directory: {target_dir}")
            return []
        
        all_paragraphs = []
        pdf_files = list(target_dir.glob("*.pdf"))
        
        self.logger.info(f"Found {len(pdf_files)} PDF files in {target_dir}")
        
        for pdf_file in pdf_files:
            paragraphs = self.ingest_file(pdf_file)
            if paragraphs:
                all_paragraphs.extend(paragraphs)
                self.logger.info(f"Extracted {len(paragraphs)} chunks from {pdf_file.name}")
        
        self.logger.info(f"Successfully extracted {len(all_paragraphs)} total chunks from {len(pdf_files)} PDF files")
        return all_paragraphs
    
    def ingest_file(self, file_path: Path) -> List[PDFParagraph]:
        """
        Extract text and chunks from a single PDF file using Unstructured.
        
        Uses Unstructured's partition_pdf to extract elements, then chunks by title
        to keep headings together with their content. Extracts the title/heading
        for each chunk.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of PDFParagraph objects extracted from the PDF
        """
        paragraphs = []
        
        try:
            self.logger.info(f"Parsing PDF with Unstructured: {file_path.name}")
            
            # Step 1: Partition the PDF into elements
            elements = partition_pdf(
                filename=str(file_path),
                strategy="fast",  # Use fast strategy for better performance
                infer_table_structure=False  # Disable table detection for speed
            )
            
            self.logger.info(f"Extracted {len(elements)} elements from {file_path.name}")
            
            # Step 2: Chunk by title to keep headings with their content
            chunks = chunk_by_title(
                elements,
                max_characters=1000,  # Maximum chunk size
                combine_text_under_n_chars=100,  # Combine small chunks
                new_after_n_chars=800  # Prefer breaking at this size
            )
            
            self.logger.info(f"Created {len(chunks)} chunks from elements")
            
            # Step 3: Convert chunks to PDFParagraph objects with titles
            for chunk_idx, chunk in enumerate(chunks):
                # Get metadata from the chunk
                metadata = chunk.metadata
                page_number = metadata.page_number if hasattr(metadata, 'page_number') and metadata.page_number else 1
                
                # Get the text content
                text = chunk.text.strip()
                
                # Extract title from the chunk
                # Unstructured's chunk_by_title includes the title at the beginning of the chunk
                # We'll extract it by looking for Title elements in the original elements
                title = self._extract_title_from_chunk(chunk, elements)
                
                if text and len(text) >= 20:  # Filter very short chunks
                    paragraph = PDFParagraph(
                        pdf_filename=file_path.name,
                        page_number=page_number,
                        paragraph_index=chunk_idx,
                        text=text,
                        title=title
                    )
                    paragraphs.append(paragraph)
                    
                    title_info = f" (Title: {title})" if title else ""
                    self.logger.debug(
                        f"Chunk {chunk_idx} (page {page_number}){title_info}: {text[:100]}..."
                    )
            
            if not paragraphs:
                self.logger.warning(f"No chunks extracted from {file_path.name}")
            else:
                self.logger.info(
                    f"Successfully extracted {len(paragraphs)} chunks from {file_path.name}"
                )
            
            return paragraphs
            
        except FileNotFoundError:
            self.logger.error(f"PDF file not found: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing PDF file {file_path}: {e}")
            self.logger.exception("Full traceback:")
            return []
    
    def _extract_title_from_chunk(self, chunk, elements: List) -> Optional[str]:
        """
        Extract the title/heading from a chunk.
        
        Looks for Title elements at the beginning of the chunk text.
        
        Args:
            chunk: The chunk object from chunk_by_title
            elements: Original list of elements from partition_pdf
            
        Returns:
            Title string if found, None otherwise
        """
        try:
            # Get the chunk text
            chunk_text = chunk.text.strip()
            
            # Look for Title elements in the original elements that match the beginning of this chunk
            for element in elements:
                element_type = type(element).__name__
                
                # Check if this is a Title element
                if element_type == 'Title':
                    title_text = element.text.strip()
                    
                    # Check if this title appears at the start of the chunk
                    if chunk_text.startswith(title_text):
                        return title_text
            
            # If no Title element found, try to extract the first line as a potential heading
            # (often headings are in ALL CAPS or have specific formatting)
            first_line = chunk_text.split('\n')[0].strip()
            
            # Heuristic: if first line is short (<100 chars) and in ALL CAPS, treat as title
            if len(first_line) < 100 and first_line.isupper() and len(first_line) > 3:
                return first_line
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error extracting title: {e}")
            return None
