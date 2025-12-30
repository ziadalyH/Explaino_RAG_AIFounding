"""PDF document ingestion module using PyMuPDF library.

This module extracts all elements per page and groups them into semantic chunks:
- Hierarchical structure preservation (sections → paragraphs)
- Optimal chunk sizes (512-1024 tokens)
- Title context maintained with content
- Chunk overlap for context continuity
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import fitz  # PyMuPDF
import tiktoken

from ..models import PDFParagraph
from config.config import Config


class PDFIngester:
    """
    Ingests and processes PDF documents using PyMuPDF library.
    
    Features:
    - Extracts text blocks per page with font information
    - Groups content into semantic chunks (512-1024 tokens)
    - Detects titles based on font size and formatting
    - Adds overlap between chunks for context continuity
    - Optimized for RAG retrieval quality
    """
    
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
        
        # Chunking parameters (paragraph-level with overlap)
        self.target_chunk_size = 512  # tokens per chunk
        self.max_chunk_size = 768     # tokens (allow some flexibility)
        self.chunk_overlap = 128      # tokens overlap between chunks
        
        # Paragraph extraction settings
        self.min_paragraph_length = 20  # characters (very permissive)
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception as e:
            self.logger.warning(f"Failed to load tiktoken, using character approximation: {e}")
            self.tokenizer = None
    
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or character approximation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximation: ~4 characters per token
            return len(text) // 4
    
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
        Extract semantic chunks from a single PDF file using PyMuPDF.
        
        This method:
        1. Extracts text blocks with font information using PyMuPDF
        2. Detects titles based on font size
        3. Creates semantic chunks (512-1024 tokens)
        4. Preserves title hierarchy
        5. Adds overlap between chunks
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of PDFParagraph objects (semantic chunks)
        """
        paragraphs = []
        
        try:
            self.logger.info(f"Parsing PDF with PyMuPDF: {file_path.name}")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(str(file_path))
            self.logger.info(f"Opened PDF with {len(doc)} pages")
            
            # Process each page and create semantic chunks
            total_chunks_created = 0
            pages_with_no_chunks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with font information
                blocks = self._extract_blocks_from_page(page)
                
                if not blocks:
                    pages_with_no_chunks.append(page_num + 1)
                    continue
                
                # Create semantic chunks from this page (paragraph index resets per page)
                # Note: page_num is 0-indexed (PyMuPDF), we convert to 1-indexed for storage
                page_chunks = self._create_semantic_chunks(
                    file_path.name,
                    page_num + 1,  # Physical page number (1-indexed)
                    blocks,
                    start_index=0  # Reset to 0 for each page
                )
                
                if page_chunks:
                    self.logger.debug(
                        f"Page {page_num + 1}: Created {len(page_chunks)} chunks from {len(blocks)} blocks"
                    )
                
                if not page_chunks:
                    pages_with_no_chunks.append(page_num + 1)
                
                paragraphs.extend(page_chunks)
                total_chunks_created += len(page_chunks)
                
                # Log every 50 pages
                if (page_num + 1) % 50 == 0:
                    self.logger.info(
                        f"Progress: Page {page_num + 1}/{len(doc)} - "
                        f"Created {total_chunks_created} chunks so far"
                    )
            
            doc.close()
            
            if pages_with_no_chunks:
                self.logger.warning(
                    f"Pages with no chunks: {len(pages_with_no_chunks)} pages "
                    f"(e.g., {pages_with_no_chunks[:10]})"
                )
            
            if not paragraphs:
                self.logger.warning(f"No chunks created from {file_path.name}")
            else:
                # Calculate statistics
                total_tokens = sum(self._count_tokens(p.text) for p in paragraphs)
                avg_tokens = total_tokens / len(paragraphs) if paragraphs else 0
                chunks_with_titles = sum(1 for p in paragraphs if p.title)
                
                self.logger.info(
                    f"Successfully created {len(paragraphs)} semantic chunks from {file_path.name}"
                )
                self.logger.info(
                    f"  Average chunk size: {avg_tokens:.0f} tokens"
                )
                self.logger.info(
                    f"  Chunks with titles: {chunks_with_titles}/{len(paragraphs)} "
                    f"({chunks_with_titles/len(paragraphs)*100:.1f}%)"
                )
            
            return paragraphs
            
        except FileNotFoundError:
            self.logger.error(f"PDF file not found: {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing PDF file {file_path}: {e}")
            self.logger.exception("Full traceback:")
            return []
    
    def _extract_blocks_from_page(self, page) -> List[Dict]:
        """
        Extract text blocks from a page with font information.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of block dictionaries with 'text', 'font_size', and 'is_title' keys
        """
        blocks = []
        
        # Get text blocks with detailed information
        text_dict = page.get_text("dict")
        
        # Calculate average font size for title detection
        font_sizes = []
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(span.get("size", 0))
        
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        title_threshold = avg_font_size * 1.15  # 15% larger than average (more sensitive)
        
        self.logger.debug(
            f"Page {page.number + 1}: avg_font_size={avg_font_size:.1f}, "
            f"title_threshold={title_threshold:.1f}"
        )
        
        # Extract blocks with title detection
        for block in text_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_text = ""
                max_font_size = 0
                
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                        max_font_size = max(max_font_size, span.get("size", 0))
                    block_text += line_text + " "
                
                block_text = block_text.strip()
                if block_text:  # Keep all blocks, including potential titles
                    is_likely_title = (
                        max_font_size >= title_threshold and 
                        len(block_text) < 200 and  # Titles are usually short
                        not block_text.endswith('.')  # Titles often don't end with period
                    )
                    blocks.append({
                        "text": block_text,
                        "font_size": max_font_size,
                        "is_title": is_likely_title
                    })
        
        return blocks
    
    
    def _create_semantic_chunks(
        self,
        pdf_filename: str,
        page_num: int,
        blocks: List[Dict],
        start_index: int
    ) -> List[PDFParagraph]:
        """
        Create chunks from page blocks - PARAGRAPH-LEVEL with sliding window.
        
        Strategy for comprehensive coverage:
        1. Extract EVERY paragraph as individual chunks
        2. For long paragraphs (>max_chunk_size), split with overlap
        3. Preserve title context with each paragraph
        4. No aggressive grouping - capture all content
        
        Args:
            pdf_filename: Name of the PDF file
            page_num: Page number
            blocks: List of block dictionaries from PyMuPDF
            start_index: Starting paragraph index
            
        Returns:
            List of PDFParagraph objects (one per paragraph or split)
        """
        chunks = []
        current_title = None
        paragraph_index = start_index
        
        filtered_count = 0
        processed_count = 0
        
        # Process each block
        for block in blocks:
            text = block["text"].strip()
            is_title = block["is_title"]
            
            # Skip empty blocks
            if not text:
                continue
            
            # Skip very short blocks (likely noise)
            if len(text) < self.min_paragraph_length:
                filtered_count += 1
                continue
            
            # Handle title blocks - update current title context
            if is_title:
                current_title = text
                self.logger.debug(f"Detected title: '{text[:100]}...' (font_size={block['font_size']:.1f})")
                continue
            
            # Handle content blocks - each becomes its own chunk(s)
            processed_count += 1
            text_tokens = self._count_tokens(text)
            
            # If paragraph fits in one chunk, create it directly
            if text_tokens <= self.max_chunk_size:
                chunks.append(PDFParagraph(
                    pdf_filename=pdf_filename,
                    page_number=page_num,
                    paragraph_index=paragraph_index,
                    text=text,
                    title=current_title
                ))
                paragraph_index += 1
            else:
                # Long paragraph - split with sliding window overlap
                sub_chunks = self._split_long_paragraph(text)
                for sub_text in sub_chunks:
                    chunks.append(PDFParagraph(
                        pdf_filename=pdf_filename,
                        page_number=page_num,
                        paragraph_index=paragraph_index,
                        text=sub_text,
                        title=current_title
                    ))
                    paragraph_index += 1
        
        # Log if page has very few chunks (potential issue)
        if len(blocks) > 5 and len(chunks) == 0:
            self.logger.warning(
                f"Page {page_num}: {len(blocks)} blocks but 0 chunks created "
                f"(filtered: {filtered_count}, processed: {processed_count})"
            )
        
        return chunks
    
    def _split_long_paragraph(self, text: str) -> List[str]:
        """
        Split a long paragraph into overlapping chunks.
        
        Uses sliding window approach to ensure no content is lost.
        
        Args:
            text: Long paragraph text
            
        Returns:
            List of text chunks with overlap
        """
        if not self.tokenizer:
            # Fallback: split by characters with overlap
            chunks = []
            chunk_chars = self.target_chunk_size * 4  # ~4 chars per token
            overlap_chars = self.chunk_overlap * 4
            
            start = 0
            while start < len(text):
                end = start + chunk_chars
                chunk = text[start:end]
                
                # Try to break at sentence boundary
                if end < len(text):
                    last_period = chunk.rfind('. ')
                    if last_period > len(chunk) // 2:  # Only if in second half
                        chunk = chunk[:last_period + 1]
                
                chunks.append(chunk.strip())
                start += chunk_chars - overlap_chars
            
            return chunks
        
        # Use tokenizer for accurate splitting
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.target_chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text.strip())
            
            # Move forward with overlap
            start += self.target_chunk_size - self.chunk_overlap
        
        return chunks
    
    def _get_overlap_text(self, content_list: List[str], overlap_tokens: int) -> str:
        """
        Get overlap text from the end of content list.
        
        Args:
            content_list: List of text segments
            overlap_tokens: Number of tokens to overlap
            
        Returns:
            Overlap text (last N tokens from content)
        """
        if not content_list:
            return ""
        
        # Start from the end and accumulate until we have enough tokens
        overlap_parts = []
        token_count = 0
        
        for text in reversed(content_list):
            text_tokens = self._count_tokens(text)
            if token_count + text_tokens <= overlap_tokens:
                overlap_parts.insert(0, text)
                token_count += text_tokens
            else:
                # Take partial text to reach overlap target
                if self.tokenizer:
                    tokens = self.tokenizer.encode(text)
                    remaining = overlap_tokens - token_count
                    if remaining > 0:
                        partial_tokens = tokens[-remaining:]
                        partial_text = self.tokenizer.decode(partial_tokens)
                        overlap_parts.insert(0, partial_text)
                break
        
        return ' '.join(overlap_parts)

