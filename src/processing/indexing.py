"""Vector index building module using OpenSearch."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import RequestError, ConnectionError as OpenSearchConnectionError

from ..models import TranscriptChunk, PDFChunk
from config.config import Config
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
        self.pdf_index_name = config.opensearch_pdf_index
        self.video_index_name = config.opensearch_video_index
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
    
    def create_index_if_not_exists(self, index_name: str, index_type: str = "general") -> None:
        """
        Create OpenSearch index with k-NN configuration if it doesn't exist.
        
        The index is configured with:
        - k-NN plugin enabled
        - HNSW algorithm for efficient similarity search
        - Cosine similarity space type
        - Proper field mappings for metadata
        
        Args:
            index_name: Name of the index to create
            index_type: Type of index ("pdf" or "video") for specific mappings
        
        Raises:
            RequestError: If index creation fails
        """
        if self.opensearch_client.indices.exists(index=index_name):
            self.logger.info(f"Index '{index_name}' already exists")
            return
        
        self.logger.info(f"Creating index '{index_name}' with k-NN configuration")
        
        # Base index settings
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
                    # Common text field
                    "text": {"type": "text"}
                }
            }
        }
        
        # Add type-specific mappings
        if index_type == "pdf":
            index_body["mappings"]["properties"].update({
                # Title embedding (separate for better title matching)
                "title_embedding": {
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
                # PDF document metadata
                "pdf_filename": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "paragraph_index": {"type": "integer"},
                "title": {"type": "text"}
            })
        elif index_type == "video":
            index_body["mappings"]["properties"].update({
                # Video transcript metadata
                "video_id": {"type": "keyword"},
                "start_timestamp": {"type": "float"},
                "end_timestamp": {"type": "float"},
                "start_token_id": {"type": "integer"},
                "end_token_id": {"type": "integer"},
                "transcript_snippet": {"type": "text"}
            })
        
        try:
            response = self.opensearch_client.indices.create(
                index=index_name,
                body=index_body
            )
            
            self.logger.info(
                f"Successfully created index '{index_name}' with k-NN enabled"
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
        Build vector indices in OpenSearch from all chunks.
        
        This method:
        1. Creates separate indices for PDFs and videos if they don't exist
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
            f"Building indices with {len(transcript_chunks)} transcript chunks "
            f"and {len(pdf_chunks)} PDF chunks"
        )
        
        # Create indices if needed
        if transcript_chunks:
            self.create_index_if_not_exists(self.video_index_name, "video")
        if pdf_chunks:
            self.create_index_if_not_exists(self.pdf_index_name, "pdf")
        
        # Process transcript chunks
        if transcript_chunks:
            self.logger.info("="*80)
            self.logger.info(f"Processing {len(transcript_chunks)} transcript chunks")
            self.logger.info("="*80)
            
            # Extract texts for batch embedding
            transcript_texts = [chunk.text for chunk in transcript_chunks]
            
            try:
                # Generate embeddings in batch with progress tracking
                self.logger.info("Generating embeddings for transcript chunks...")
                import time
                start_time = time.time()
                
                transcript_embeddings = embedding_engine.embed_batch(transcript_texts)
                
                elapsed = time.time() - start_time
                embeddings_per_sec = len(transcript_texts) / elapsed if elapsed > 0 else 0
                
                self.logger.info(
                    f"✓ Generated {len(transcript_embeddings)} embeddings in {elapsed:.2f}s "
                    f"({embeddings_per_sec:.1f} embeddings/sec)"
                )
                
                # Create documents with embeddings and metadata
                self.logger.info("Creating video documents...")
                video_documents = []
                for chunk, embedding in zip(transcript_chunks, transcript_embeddings):
                    doc = {
                        "embedding": embedding.tolist(),
                        "video_id": chunk.video_id,
                        "start_timestamp": chunk.start_timestamp,
                        "end_timestamp": chunk.end_timestamp,
                        "start_token_id": chunk.start_token_id,
                        "end_token_id": chunk.end_token_id,
                        "transcript_snippet": chunk.text,
                        "text": chunk.text
                    }
                    video_documents.append(doc)
                
                self.logger.info(f"✓ Created {len(video_documents)} video documents")
                
                # Index video documents
                self.index_documents(video_documents, self.video_index_name)
                
            except Exception as e:
                self.logger.error(f"Failed to process transcript chunks: {str(e)}")
                raise
        
        # Process PDF chunks with dual embeddings
        if pdf_chunks:
            self.logger.info("="*80)
            self.logger.info(f"Processing {len(pdf_chunks)} PDF chunks with dual embeddings")
            self.logger.info("="*80)
            
            # Separate chunks with and without titles
            chunks_with_titles = [chunk for chunk in pdf_chunks if chunk.title]
            chunks_without_titles = [chunk for chunk in pdf_chunks if not chunk.title]
            
            self.logger.info(f"  - {len(chunks_with_titles)} chunks with titles (dual embeddings)")
            self.logger.info(f"  - {len(chunks_without_titles)} chunks without titles (content only)")
            
            # Extract texts for batch embedding
            pdf_texts = [chunk.text for chunk in pdf_chunks]
            
            try:
                import time
                
                # Generate content embeddings for all chunks
                self.logger.info("Generating content embeddings for all PDF chunks...")
                start_time = time.time()
                
                pdf_embeddings = embedding_engine.embed_batch(pdf_texts)
                
                elapsed = time.time() - start_time
                embeddings_per_sec = len(pdf_texts) / elapsed if elapsed > 0 else 0
                
                self.logger.info(
                    f"✓ Generated {len(pdf_embeddings)} content embeddings in {elapsed:.2f}s "
                    f"({embeddings_per_sec:.1f} embeddings/sec)"
                )
                
                # Generate title embeddings only for chunks with titles
                title_embeddings_map = {}
                if chunks_with_titles:
                    self.logger.info(f"Generating title embeddings for {len(chunks_with_titles)} chunks...")
                    start_time = time.time()
                    
                    pdf_titles = [chunk.title for chunk in chunks_with_titles]
                    title_embeddings = embedding_engine.embed_batch(pdf_titles)
                    
                    elapsed = time.time() - start_time
                    embeddings_per_sec = len(pdf_titles) / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(
                        f"✓ Generated {len(title_embeddings)} title embeddings in {elapsed:.2f}s "
                        f"({embeddings_per_sec:.1f} embeddings/sec)"
                    )
                    
                    # Create a map of chunk to title embedding
                    for chunk, title_emb in zip(chunks_with_titles, title_embeddings):
                        # Use a unique key (filename + page + index)
                        key = f"{chunk.pdf_filename}_{chunk.page_number}_{chunk.paragraph_index}"
                        title_embeddings_map[key] = title_emb
                
                # Create documents with embeddings and metadata
                self.logger.info("Creating PDF documents...")
                pdf_documents = []
                for chunk, content_emb in zip(pdf_chunks, pdf_embeddings):
                    # Check if this chunk has a title embedding
                    key = f"{chunk.pdf_filename}_{chunk.page_number}_{chunk.paragraph_index}"
                    
                    doc = {
                        "embedding": content_emb.tolist(),  # Content embedding (always present)
                        "pdf_filename": chunk.pdf_filename,
                        "page_number": chunk.page_number,
                        "paragraph_index": chunk.paragraph_index,
                        "text": chunk.text,
                        "title": chunk.title  # Store title separately (can be None)
                    }
                    
                    # Only add title_embedding if chunk has a title
                    if key in title_embeddings_map:
                        doc["title_embedding"] = title_embeddings_map[key].tolist()
                    
                    pdf_documents.append(doc)
                
                self.logger.info(
                    f"✓ Created {len(pdf_chunks)} PDF documents "
                    f"({len(chunks_with_titles)} with dual embeddings)"
                )
                
                # Index PDF documents
                self.index_documents(pdf_documents, self.pdf_index_name)
                
            except Exception as e:
                self.logger.error(f"Failed to process PDF chunks: {str(e)}")
                raise
    
    def index_documents(self, documents: List[Dict[str, Any]], index_name: str) -> None:
        """
        Bulk index documents into OpenSearch with detailed progress tracking.
        
        Uses the bulk API for efficient indexing with proper error handling
        and progress logging.
        
        Args:
            documents: List of document dictionaries with embeddings and metadata
            index_name: Name of the index to insert documents into
            
        Raises:
            Exception: If bulk indexing fails
        """
        if not documents:
            self.logger.warning("No documents provided for indexing")
            return
        
        total_docs = len(documents)
        self.logger.info(f"Starting bulk indexing of {total_docs} documents into '{index_name}'")
        
        # Prepare bulk actions
        actions = []
        for doc in documents:
            action = {
                "_index": index_name,
                "_source": doc
            }
            actions.append(action)
        
        try:
            # Track progress
            import time
            start_time = time.time()
            
            # Execute bulk indexing with progress tracking
            success_count = 0
            error_count = 0
            errors = []
            
            # Use smaller chunk size for better progress tracking
            chunk_size = 100
            total_chunks = (total_docs + chunk_size - 1) // chunk_size
            
            self.logger.info(f"Indexing in {total_chunks} batches of {chunk_size} documents")
            
            for chunk_num in range(total_chunks):
                start_idx = chunk_num * chunk_size
                end_idx = min(start_idx + chunk_size, total_docs)
                chunk_actions = actions[start_idx:end_idx]
                
                chunk_start_time = time.time()
                
                # Index this chunk
                chunk_success, chunk_errors = helpers.bulk(
                    self.opensearch_client,
                    chunk_actions,
                    chunk_size=chunk_size,
                    request_timeout=60,
                    raise_on_error=False,
                    raise_on_exception=False
                )
                
                chunk_elapsed = time.time() - chunk_start_time
                
                success_count += chunk_success
                if chunk_errors:
                    error_count += len(chunk_errors)
                    errors.extend(chunk_errors)
                
                # Calculate progress
                progress_pct = ((chunk_num + 1) / total_chunks) * 100
                docs_per_sec = chunk_success / chunk_elapsed if chunk_elapsed > 0 else 0
                
                # Log progress
                self.logger.info(
                    f"Batch {chunk_num + 1}/{total_chunks} ({progress_pct:.1f}%): "
                    f"Indexed {chunk_success} docs in {chunk_elapsed:.2f}s "
                    f"({docs_per_sec:.1f} docs/sec) - "
                    f"Total: {success_count}/{total_docs}"
                )
            
            # Calculate total time
            total_elapsed = time.time() - start_time
            avg_docs_per_sec = success_count / total_elapsed if total_elapsed > 0 else 0
            
            # Log errors if any
            if errors:
                self.logger.error(f"Bulk indexing encountered {error_count} errors")
                # Log first few errors for debugging
                for i, error in enumerate(errors[:5]):
                    self.logger.error(f"Error {i+1}: {error}")
                
                # Raise exception if too many errors
                error_rate = error_count / total_docs
                if error_rate > 0.1:  # More than 10% errors
                    raise Exception(
                        f"Bulk indexing failed with {error_count} errors "
                        f"({error_rate:.1%} error rate)"
                    )
            
            # Final summary
            self.logger.info("="*80)
            self.logger.info(f"Bulk indexing completed for '{index_name}'")
            self.logger.info(f"  Total documents: {total_docs}")
            self.logger.info(f"  Successfully indexed: {success_count}")
            self.logger.info(f"  Errors: {error_count}")
            self.logger.info(f"  Total time: {total_elapsed:.2f}s")
            self.logger.info(f"  Average speed: {avg_docs_per_sec:.1f} docs/sec")
            self.logger.info("="*80)
            
            # Refresh index to make documents searchable immediately
            self.logger.info(f"Refreshing index '{index_name}'...")
            self.opensearch_client.indices.refresh(index=index_name)
            self.logger.info(f"✓ Index '{index_name}' refreshed and ready for search")
            
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
