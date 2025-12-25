"""Vector index building module using OpenSearch."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError, ConnectionError as OpenSearchConnectionError

from ..models import TranscriptChunk, PDFChunk
from ..config import Config
from .embedding import EmbeddingEngine


class VectorIndexBuilder:
    """
    Build and manage a searchable vector index in OpenSearch with metadata.
    
    This class handles:
    - Connecting to OpenSearch cluster
    - Creating k-NN enabled indices
    - Bulk indexing of document embeddings with metadata
    - Error handling and logging
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize the VectorIndexBuilder.
        
        Args:
            config: Configuration object containing OpenSearch settings
            logger: Logger instance for logging operations
        """
        self.config = config
        self.logger = logger
        self.index_name = config.opensearch_index_name
        self.embedding_dimension = config.embedding_dimension
        self.opensearch_client = self._initialize_opensearch_client()
    
    def _initialize_opensearch_client(self) -> OpenSearch:
        """
        Initialize OpenSearch client with connection details from config.
        
        Returns:
            OpenSearch client instance
            
        Raises:
            OpenSearchConnectionError: If connection to OpenSearch fails
        """
        self.logger.info(
            f"Initializing OpenSearch client: {self.config.opensearch_host}:"
            f"{self.config.opensearch_port}"
        )
        
        # Build connection parameters
        connection_params = {
            'hosts': [{'host': self.config.opensearch_host, 'port': self.config.opensearch_port}],
            'http_compress': True,
            'use_ssl': self.config.opensearch_use_ssl,
            'verify_certs': self.config.opensearch_verify_certs,
            'ssl_show_warn': False,
            'timeout': 30,
            'max_retries': 3,
            'retry_on_timeout': True
        }
        
        # Add authentication if provided
        if self.config.opensearch_username and self.config.opensearch_password:
            connection_params['http_auth'] = (
                self.config.opensearch_username,
                self.config.opensearch_password
            )
            self.logger.info("Using authentication for OpenSearch connection")
        
        try:
            client = OpenSearch(**connection_params)
            
            # Test connection
            info = client.info()
            self.logger.info(
                f"Successfully connected to OpenSearch cluster: {info.get('cluster_name', 'unknown')}"
            )
            
            return client
            
        except OpenSearchConnectionError as e:
            self.logger.error(f"Failed to connect to OpenSearch: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error initializing OpenSearch client: {str(e)}")
            raise
    
    def create_index_if_not_exists(self) -> None:
        """
        Create OpenSearch index with k-NN configuration if it doesn't exist.
        
        The index is configured with:
        - k-NN plugin enabled
        - HNSW algorithm for efficient similarity search
        - Cosine similarity space type
        - Proper field mappings for metadata
        
        Raises:
            RequestError: If index creation fails
        """
        if self.opensearch_client.indices.exists(index=self.index_name):
            self.logger.info(f"Index '{self.index_name}' already exists")
            return
        
        self.logger.info(f"Creating index '{self.index_name}' with k-NN configuration")
        
        # Define index settings and mappings
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    # Vector embedding field with k-NN configuration
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    # Source type for filtering (video or pdf)
                    "source_type": {"type": "keyword"},
                    
                    # Video transcript metadata
                    "video_id": {"type": "keyword"},
                    "start_timestamp": {"type": "float"},
                    "end_timestamp": {"type": "float"},
                    "start_token_id": {"type": "integer"},
                    "end_token_id": {"type": "integer"},
                    "transcript_snippet": {"type": "text"},
                    
                    # PDF document metadata
                    "pdf_filename": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "paragraph_index": {"type": "integer"},
                    "title": {"type": "text"},  # Section title/heading
                    
                    # Common text field
                    "text": {"type": "text"}
                }
            }
        }
        
        try:
            response = self.opensearch_client.indices.create(
                index=self.index_name,
                body=index_body
            )
            
            self.logger.info(
                f"Successfully created index '{self.index_name}' with k-NN enabled"
            )
            self.logger.debug(f"Index creation response: {response}")
            
        except RequestError as e:
            self.logger.error(f"Failed to create index: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error creating index: {str(e)}")
            raise
    
    def build_index(
        self,
        transcript_chunks: List[TranscriptChunk],
        pdf_chunks: List[PDFChunk],
        embedding_engine: EmbeddingEngine
    ) -> None:
        """
        Build vector index in OpenSearch from all chunks.
        
        This method:
        1. Creates the index if it doesn't exist
        2. Generates embeddings for all chunks
        3. Bulk indexes documents with embeddings and metadata
        
        Args:
            transcript_chunks: List of video transcript chunks
            pdf_chunks: List of PDF chunks
            embedding_engine: EmbeddingEngine instance for generating embeddings
            
        Raises:
            Exception: If indexing fails
        """
        self.logger.info(
            f"Building index with {len(transcript_chunks)} transcript chunks "
            f"and {len(pdf_chunks)} PDF chunks"
        )
        
        # Create index if needed
        self.create_index_if_not_exists()
        
        # Prepare documents for indexing
        documents = []
        
        # Process transcript chunks
        if transcript_chunks:
            self.logger.info(f"Processing {len(transcript_chunks)} transcript chunks")
            
            # Extract texts for batch embedding
            transcript_texts = [chunk.text for chunk in transcript_chunks]
            
            try:
                # Generate embeddings in batch
                transcript_embeddings = embedding_engine.embed_batch(transcript_texts)
                
                # Create documents with embeddings and metadata
                for chunk, embedding in zip(transcript_chunks, transcript_embeddings):
                    doc = {
                        "embedding": embedding.tolist(),
                        "source_type": "video",
                        "video_id": chunk.video_id,
                        "start_timestamp": chunk.start_timestamp,
                        "end_timestamp": chunk.end_timestamp,
                        "start_token_id": chunk.start_token_id,
                        "end_token_id": chunk.end_token_id,
                        "transcript_snippet": chunk.text,
                        "text": chunk.text
                    }
                    documents.append(doc)
                
                self.logger.info(
                    f"Successfully created {len(documents)} video documents with embeddings"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to process transcript chunks: {str(e)}")
                raise
        
        # Process PDF chunks
        if pdf_chunks:
            self.logger.info(f"Processing {len(pdf_chunks)} PDF chunks")
            
            # Extract texts for batch embedding
            # Combine title and text for better semantic search
            pdf_texts = []
            for chunk in pdf_chunks:
                if chunk.title:
                    # Embed both title and content together
                    combined_text = f"{chunk.title}\n\n{chunk.text}"
                else:
                    combined_text = chunk.text
                pdf_texts.append(combined_text)
            
            try:
                # Generate embeddings in batch
                pdf_embeddings = embedding_engine.embed_batch(pdf_texts)
                
                # Create documents with embeddings and metadata
                for chunk, embedding in zip(pdf_chunks, pdf_embeddings):
                    doc = {
                        "embedding": embedding.tolist(),
                        "source_type": "pdf",
                        "pdf_filename": chunk.pdf_filename,
                        "page_number": chunk.page_number,
                        "paragraph_index": chunk.paragraph_index,
                        "text": chunk.text,
                        "title": chunk.title  # Store title separately
                    }
                    documents.append(doc)
                
                self.logger.info(
                    f"Successfully created {len(pdf_chunks)} PDF documents with embeddings"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to process PDF chunks: {str(e)}")
                raise
        
        # Index all documents
        if documents:
            self.index_documents(documents)
        else:
            self.logger.warning("No documents to index")
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Bulk index documents into OpenSearch.
        
        Uses the bulk API for efficient indexing with proper error handling.
        
        Args:
            documents: List of document dictionaries with embeddings and metadata
            
        Raises:
            Exception: If bulk indexing fails
        """
        if not documents:
            self.logger.warning("No documents provided for indexing")
            return
        
        self.logger.info(f"Bulk indexing {len(documents)} documents into '{self.index_name}'")
        
        # Prepare bulk actions
        actions = []
        for doc in documents:
            action = {
                "_index": self.index_name,
                "_source": doc
            }
            actions.append(action)
        
        try:
            # Execute bulk indexing
            success_count, errors = helpers.bulk(
                self.opensearch_client,
                actions,
                chunk_size=500,
                request_timeout=60,
                raise_on_error=False,
                raise_on_exception=False
            )
            
            if errors:
                self.logger.error(f"Bulk indexing encountered {len(errors)} errors")
                # Log first few errors for debugging
                for i, error in enumerate(errors[:5]):
                    self.logger.error(f"Error {i+1}: {error}")
                
                # Raise exception if too many errors
                error_rate = len(errors) / len(documents)
                if error_rate > 0.1:  # More than 10% errors
                    raise Exception(
                        f"Bulk indexing failed with {len(errors)} errors "
                        f"({error_rate:.1%} error rate)"
                    )
            
            self.logger.info(
                f"Successfully indexed {success_count} documents "
                f"({len(errors)} errors)"
            )
            
            # Refresh index to make documents searchable immediately
            self.opensearch_client.indices.refresh(index=self.index_name)
            self.logger.info(f"Refreshed index '{self.index_name}'")
            
        except Exception as e:
            self.logger.error(f"Bulk indexing failed: {str(e)}")
            raise

    def index_pdf_chunks(
        self,
        pdf_chunks: List[PDFChunk],
        pdf_embeddings: List[np.ndarray]
    ) -> None:
        """
        Index PDF chunks with pre-generated embeddings.
        
        This method is used for reindexing PDFs without regenerating embeddings.
        
        Args:
            pdf_chunks: List of PDF chunks
            pdf_embeddings: List of embeddings corresponding to the chunks
            
        Raises:
            Exception: If indexing fails
        """
        if len(pdf_chunks) != len(pdf_embeddings):
            raise ValueError(
                f"Mismatch between chunks ({len(pdf_chunks)}) "
                f"and embeddings ({len(pdf_embeddings)})"
            )
        
        self.logger.info(f"Indexing {len(pdf_chunks)} PDF chunks with embeddings")
        
        # Create documents with embeddings and metadata
        documents = []
        for chunk, embedding in zip(pdf_chunks, pdf_embeddings):
            doc = {
                "embedding": embedding.tolist(),
                "source_type": "pdf",
                "pdf_filename": chunk.pdf_filename,
                "page_number": chunk.page_number,
                "paragraph_index": chunk.paragraph_index,
                "text": chunk.text,
                "title": chunk.title  # Store title separately
            }
            documents.append(doc)
        
        # Index all documents
        self.index_documents(documents)
