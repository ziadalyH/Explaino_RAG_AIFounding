"""Main RAG system orchestrator.

This module provides the RAGSystem class which coordinates all components
of the RAG chatbot backend: ingestion, processing, retrieval, and response generation.
"""

import logging
from pathlib import Path
from typing import Union, Optional
from opensearchpy import OpenSearch

from .config import Config
from .models import VideoResponse, PDFResponse, NoAnswerResponse
from .ingestion.transcript_ingester import TranscriptIngester
from .ingestion.pdf_ingester import PDFIngester
from .processing.chunking import ChunkingModule
from .processing.embedding import EmbeddingEngine
from .processing.indexing import VectorIndexBuilder
from .retrieval.query_processor import QueryProcessor
from .retrieval.retrieval_engine import RetrievalEngine
from .retrieval.response_generator import ResponseGenerator
from .knowledge_summary import KnowledgeSummaryGenerator


class RAGSystem:
    """
    Main orchestrator for the RAG Chatbot Backend.
    
    This class coordinates all components of the system:
    - Data ingestion (video transcripts and PDFs)
    - Text chunking and embedding generation
    - Vector index building and management
    - Query processing and retrieval
    - Response generation with LLM-based answers
    
    The system implements a two-tier retrieval strategy:
    1. Search video transcripts first (primary source)
    2. Fall back to PDF documents if no relevant video content found
    3. Return no-answer response if neither source has relevant content
    """
    
    def __init__(self, config: Config):
        """
        Initialize the RAG system with all components.
        
        Args:
            config: Configuration object containing all system settings
        """
        self.config = config
        self.logger = self._setup_logger()
        
        self.logger.info("Initializing RAG System")
        self.logger.info(f"Configuration: {config.opensearch_host}:{config.opensearch_port}")
        
        # Initialize core components
        self.embedding_engine = EmbeddingEngine(config, self.logger)
        self.vector_index_builder = VectorIndexBuilder(config, self.logger)
        self.opensearch_client = self.vector_index_builder.opensearch_client
        
        # Initialize ingestion components
        self.transcript_ingester = TranscriptIngester(config, self.logger)
        self.pdf_ingester = PDFIngester(config, self.logger)
        self.chunking_module = ChunkingModule(config, self.logger)
        
        # Initialize retrieval components
        self.query_processor = QueryProcessor(self.embedding_engine, self.logger)
        self.retrieval_engine = RetrievalEngine(
            self.opensearch_client,
            config,
            self.logger
        )
        self.response_generator = ResponseGenerator(config, self.logger)
        
        # Initialize knowledge summary generator
        self.knowledge_summary_generator = KnowledgeSummaryGenerator(config, self.logger)
        
        self.logger.info("RAG System initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger with configured log level.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(self.config.log_level)
        
        # Create console handler if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self.config.log_level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
        
        return logger
    
    def answer_question(
        self,
        question: str
    ) -> Union[VideoResponse, PDFResponse, NoAnswerResponse]:
        """
        Main entry point for answering questions.
        
        This method implements the complete query pipeline:
        1. Process and embed the query
        2. Retrieve relevant content using two-tier strategy
        3. Generate natural language answer with LLM
        4. Return structured response with citations
        
        Args:
            question: User's question as a string
            
        Returns:
            VideoResponse if video content found above threshold
            PDFResponse if PDF content found above threshold
            NoAnswerResponse if no relevant content found
            
        Raises:
            ValueError: If question is empty
            Exception: If any component fails during processing
        """
        self.logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Step 1: Process and embed the query
            self.logger.info("Step 1: Processing query")
            query_embedding = self.query_processor.process_query(question)
            
            # Step 2: Retrieve relevant content
            self.logger.info("Step 2: Retrieving relevant content")
            retrieval_result = self.retrieval_engine.retrieve(query_embedding, question)
            
            # Step 3: Generate response with LLM-based answer
            self.logger.info("Step 3: Generating response")
            response = self.response_generator.generate_response(
                query=question,
                result=retrieval_result
            )
            
            self.logger.info(f"Successfully generated {response.answer_type} response")
            return response
            
        except ValueError as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}", exc_info=True)
            raise
    
    def build_index(
        self, 
        force_rebuild: bool = False,
        videos_only: bool = False,
        pdfs_only: bool = False
    ) -> None:
        """
        Build or rebuild the vector index from data files.
        
        This method:
        1. Checks what's already indexed (unless force_rebuild=True)
        2. Only indexes missing PDFs and transcripts (resume capability)
        3. Ingests video transcripts and PDF documents
        4. Chunks the content into segments
        5. Generates embeddings for all chunks
        6. Builds the OpenSearch vector index
        
        Args:
            force_rebuild: If True, delete and rebuild entire index from scratch
            videos_only: If True, only index video transcripts (skip PDFs)
            pdfs_only: If True, only index PDF documents (skip videos)
            
        Raises:
            Exception: If indexing fails
        """
        self.logger.info("Starting index building process")
        
        # Validate mutually exclusive options
        if videos_only and pdfs_only:
            raise ValueError("Cannot use videos_only and pdfs_only together")
        
        try:
            # Check if we should rebuild from scratch
            if force_rebuild:
                self.logger.info("Force rebuild requested - deleting existing indices")
                
                if not pdfs_only and self.opensearch_client.indices.exists(index=self.config.opensearch_video_index):
                    self.opensearch_client.indices.delete(index=self.config.opensearch_video_index)
                    self.logger.info(f"Deleted video index: {self.config.opensearch_video_index}")
                
                if not videos_only and self.opensearch_client.indices.exists(index=self.config.opensearch_pdf_index):
                    self.opensearch_client.indices.delete(index=self.config.opensearch_pdf_index)
                    self.logger.info(f"Deleted PDF index: {self.config.opensearch_pdf_index}")
            
            # Create indices if they don't exist
            if not pdfs_only:
                self.vector_index_builder.create_index_if_not_exists(self.config.opensearch_video_index, "video")
            if not videos_only:
                self.vector_index_builder.create_index_if_not_exists(self.config.opensearch_pdf_index, "pdf")
            
            # Get already indexed files (for resume capability)
            indexed_pdfs, indexed_videos = self._get_indexed_files()
            
            self.logger.info(f"Already indexed: {len(indexed_pdfs)} PDFs, {len(indexed_videos)} videos")
            
            # Step 1: Ingest video transcripts (unless pdfs_only)
            all_transcripts = []
            if not pdfs_only:
                self.logger.info("Step 1: Ingesting video transcripts")
                all_transcripts = self.transcript_ingester.ingest_directory()
            else:
                self.logger.info("Step 1: Skipping video transcripts (pdfs_only mode)")
            
            # Filter out already indexed transcripts
            transcripts = [t for t in all_transcripts if t.video_id not in indexed_videos]
            
            self.logger.info(
                f"Ingested {len(all_transcripts)} video transcripts "
                f"({len(transcripts)} new, {len(indexed_videos)} already indexed)"
            )
            
            # Step 2: Ingest PDF documents (unless videos_only)
            all_pdf_paragraphs = []
            if not videos_only:
                self.logger.info("Step 2: Ingesting PDF documents")
                all_pdf_paragraphs = self.pdf_ingester.ingest_directory()
            else:
                self.logger.info("Step 2: Skipping PDF documents (videos_only mode)")
            
            # Filter out already indexed PDFs
            pdf_paragraphs = [p for p in all_pdf_paragraphs if p.pdf_filename not in indexed_pdfs]
            
            # Count paragraphs per PDF for logging
            pdf_counts = {}
            for p in all_pdf_paragraphs:
                pdf_counts[p.pdf_filename] = pdf_counts.get(p.pdf_filename, 0) + 1
            
            new_pdf_count = len(set(p.pdf_filename for p in pdf_paragraphs))
            
            if not videos_only:
                self.logger.info(
                    f"Ingested {len(all_pdf_paragraphs)} PDF paragraphs from {len(pdf_counts)} PDFs "
                    f"({new_pdf_count} new PDFs, {len(indexed_pdfs)} already indexed)"
                )
            
            # Check if there's anything to index
            if not transcripts and not pdf_paragraphs:
                self.logger.info("All files are already indexed. Nothing to do.")
                self.logger.info("Use force_rebuild=True to reindex everything.")
                return
            
            # Step 3: Chunk transcripts
            self.logger.info("Step 3: Chunking transcripts")
            transcript_chunks = []
            for transcript in transcripts:
                chunks = self.chunking_module.chunk_transcript(transcript)
                transcript_chunks.extend(chunks)
            self.logger.info(f"Created {len(transcript_chunks)} transcript chunks")
            
            # Step 4: Chunk PDF paragraphs
            self.logger.info("Step 4: Chunking PDF paragraphs")
            pdf_chunks = self.chunking_module.chunk_pdf_paragraphs(pdf_paragraphs)
            self.logger.info(f"Created {len(pdf_chunks)} PDF chunks")
            
            # Step 5: Build vector index (only for new content)
            self.logger.info("Step 5: Building vector index for new content")
            self.vector_index_builder.build_index(
                transcript_chunks=transcript_chunks,
                pdf_chunks=pdf_chunks,
                embedding_engine=self.embedding_engine
            )
            
            # Show final status
            final_indexed_pdfs, final_indexed_videos = self._get_indexed_files()
            
            self.logger.info("Index building completed successfully")
            self.logger.info(
                f"Total indexed: {len(final_indexed_pdfs)} PDFs, "
                f"{len(final_indexed_videos)} videos"
            )
            
            # Generate knowledge summary
            self.logger.info("Generating knowledge summary...")
            try:
                # Get sample chunks for summary generation
                sample_chunks = self._get_sample_chunks(pdf_chunks, transcript_chunks)
                
                # Generate summary
                self.knowledge_summary_generator.generate_summary(
                    pdf_files=list(final_indexed_pdfs),
                    video_ids=list(final_indexed_videos),
                    sample_chunks=sample_chunks
                )
            except Exception as e:
                self.logger.warning(f"Failed to generate knowledge summary: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}", exc_info=True)
            raise
    
    def _get_sample_chunks(self, pdf_chunks, transcript_chunks):
        """Get sample chunks for summary generation."""
        sample_chunks = {
            'pdf': [],
            'video': []
        }
        
        # Get sample PDF chunks
        if pdf_chunks:
            sample_size = min(20, len(pdf_chunks))
            step = max(1, len(pdf_chunks) // sample_size)
            sample_chunks['pdf'] = [chunk.text for chunk in pdf_chunks[::step][:sample_size]]
        
        # Get sample video chunks
        if transcript_chunks:
            sample_size = min(20, len(transcript_chunks))
            step = max(1, len(transcript_chunks) // sample_size)
            sample_chunks['video'] = [chunk.text for chunk in transcript_chunks[::step][:sample_size]]
        
        return sample_chunks
    
    def _get_indexed_files(self):
        """
        Get list of completely indexed PDFs and videos using separate indices.
        
        For PDFs: Checks max page number in rag-pdf-index
        For Videos: Checks max token ID in rag-video-index
        
        Returns:
            Tuple of (set of PDF filenames, set of video IDs)
        """
        indexed_pdfs = set()
        indexed_videos = set()
        
        try:
            # Check PDF index
            if self.opensearch_client.indices.exists(index=self.config.opensearch_pdf_index):
                pdf_response = self.opensearch_client.search(
                    index=self.config.opensearch_pdf_index,
                    body={
                        "size": 0,
                        "aggs": {
                            "pdfs": {
                                "terms": {
                                    "field": "pdf_filename",
                                    "size": 10000
                                },
                                "aggs": {
                                    "max_page": {"max": {"field": "page_number"}},
                                    "unique_pages": {"cardinality": {"field": "page_number"}},
                                    "total_chunks": {"value_count": {"field": "page_number"}}
                                }
                            }
                        }
                    }
                )
                
                if 'aggregations' in pdf_response:
                    buckets = pdf_response['aggregations']['pdfs']['buckets']
                    for bucket in buckets:
                        pdf_filename = bucket['key']
                        max_page = int(bucket['max_page']['value'])
                        unique_pages = bucket['unique_pages']['value']
                        total_chunks = bucket['doc_count']
                        
                        # Consider complete if has documents
                        if total_chunks > 0:
                            indexed_pdfs.add(pdf_filename)
                            self.logger.debug(
                                f"PDF {pdf_filename}: {total_chunks} chunks, "
                                f"{unique_pages} pages, max page {max_page}"
                            )
            
            # Check video index
            if self.opensearch_client.indices.exists(index=self.config.opensearch_video_index):
                video_response = self.opensearch_client.search(
                    index=self.config.opensearch_video_index,
                    body={
                        "size": 0,
                        "aggs": {
                            "videos": {
                                "terms": {
                                    "field": "video_id",
                                    "size": 10000
                                },
                                "aggs": {
                                    "max_token": {"max": {"field": "end_token_id"}},
                                    "max_timestamp": {"max": {"field": "end_timestamp"}},
                                    "total_chunks": {"value_count": {"field": "video_id"}}
                                }
                            }
                        }
                    }
                )
                
                if 'aggregations' in video_response:
                    buckets = video_response['aggregations']['videos']['buckets']
                    for bucket in buckets:
                        video_id = bucket['key']
                        max_token = int(bucket['max_token']['value'])
                        max_timestamp = bucket['max_timestamp']['value']
                        total_chunks = bucket['doc_count']
                        
                        # Consider complete if has chunks
                        if total_chunks > 0:
                            indexed_videos.add(video_id)
                            self.logger.debug(
                                f"Video {video_id}: {total_chunks} chunks, "
                                f"max token {max_token}, max time {max_timestamp:.1f}s"
                            )
            
        except Exception as e:
            self.logger.warning(f"Error checking indexed files: {e}")
        
        return indexed_pdfs, indexed_videos
    
    def check_index_exists(self) -> bool:
        """
        Check if OpenSearch indices exist and have documents.
        
        Returns:
            True if either PDF or video index exists and contains documents
            False if both indices don't exist or are empty
        """
        try:
            pdf_exists = False
            video_exists = False
            
            # Check PDF index
            if self.opensearch_client.indices.exists(index=self.config.opensearch_pdf_index):
                count_response = self.opensearch_client.count(index=self.config.opensearch_pdf_index)
                pdf_count = count_response.get('count', 0)
                if pdf_count > 0:
                    pdf_exists = True
                    self.logger.info(f"PDF index exists with {pdf_count} documents")
            
            # Check video index
            if self.opensearch_client.indices.exists(index=self.config.opensearch_video_index):
                count_response = self.opensearch_client.count(index=self.config.opensearch_video_index)
                video_count = count_response.get('count', 0)
                if video_count > 0:
                    video_exists = True
                    self.logger.info(f"Video index exists with {video_count} documents")
            
            return pdf_exists or video_exists
            
        except Exception as e:
            self.logger.error(f"Error checking index existence: {str(e)}")
            return False
