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
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the vector index from data files.
        
        This method:
        1. Checks if index already exists (unless force_rebuild=True)
        2. Ingests video transcripts and PDF documents
        3. Chunks the content into segments
        4. Generates embeddings for all chunks
        5. Builds the OpenSearch vector index
        
        Args:
            force_rebuild: If True, rebuild index even if it exists
            
        Raises:
            Exception: If indexing fails
        """
        self.logger.info("Starting index building process")
        
        try:
            # Check if index exists and has documents
            if not force_rebuild:
                index_exists = self.check_index_exists()
                if index_exists:
                    self.logger.info(
                        "Index already exists with documents. "
                        "Use force_rebuild=True to rebuild."
                    )
                    return
            
            # Step 1: Ingest video transcripts
            self.logger.info("Step 1: Ingesting video transcripts")
            transcripts = self.transcript_ingester.ingest_directory()
            self.logger.info(f"Ingested {len(transcripts)} video transcripts")
            
            # Step 2: Ingest PDF documents
            self.logger.info("Step 2: Ingesting PDF documents")
            pdf_paragraphs = self.pdf_ingester.ingest_directory()
            self.logger.info(f"Ingested {len(pdf_paragraphs)} PDF paragraphs")
            
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
            
            # Step 5: Build vector index
            self.logger.info("Step 5: Building vector index")
            self.vector_index_builder.build_index(
                transcript_chunks=transcript_chunks,
                pdf_chunks=pdf_chunks,
                embedding_engine=self.embedding_engine
            )
            
            self.logger.info("Index building completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}", exc_info=True)
            raise
    
    def check_index_exists(self) -> bool:
        """
        Check if OpenSearch index exists and has documents.
        
        Returns:
            True if index exists and contains at least one document
            False if index doesn't exist or is empty
        """
        try:
            # Check if index exists
            index_exists = self.opensearch_client.indices.exists(
                index=self.config.opensearch_index_name
            )
            
            if not index_exists:
                self.logger.info(
                    f"Index '{self.config.opensearch_index_name}' does not exist"
                )
                return False
            
            # Check document count
            count_response = self.opensearch_client.count(
                index=self.config.opensearch_index_name
            )
            doc_count = count_response.get('count', 0)
            
            self.logger.info(
                f"Index '{self.config.opensearch_index_name}' exists with "
                f"{doc_count} documents"
            )
            
            return doc_count > 0
            
        except Exception as e:
            self.logger.error(f"Error checking index existence: {str(e)}")
            return False
