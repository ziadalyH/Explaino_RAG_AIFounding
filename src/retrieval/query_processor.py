"""Query processing module for user queries.

This module handles processing user queries, including validation,
preprocessing, and embedding generation using OpenAI.
"""

import logging
import numpy as np

from ..processing.embedding import EmbeddingEngine


class QueryProcessor:
    """Process user queries and generate query embeddings.
    
    This class handles query validation, text preprocessing, and embedding
    generation using the same embedding model as content chunks to ensure
    semantic similarity search works correctly.
    """
    
    def __init__(self, embedding_engine: EmbeddingEngine, logger: logging.Logger):
        """Initialize the query processor.
        
        Args:
            embedding_engine: EmbeddingEngine instance for generating embeddings
            logger: Logger instance for logging operations
        """
        self.embedding_engine = embedding_engine
        self.logger = logger
    
    def process_query(self, query: str) -> np.ndarray:
        """Process and embed a user query.
        
        This method validates the query, preprocesses the text, and generates
        a vector embedding using OpenAI's embedding API. The same embedding
        model is used as for content chunks to ensure compatibility.
        
        Args:
            query: User's question as a string
            
        Returns:
            Numpy array containing the query embedding vector
            
        Raises:
            ValueError: If query is empty or whitespace-only
            Exception: If embedding generation fails
        """
        # Validate non-empty query
        if not query or not query.strip():
            self.logger.error("Received empty or whitespace-only query")
            raise ValueError("Query cannot be empty")
        
        self.logger.info(f"Processing query: {query[:100]}...")
        
        # Preprocess the query text
        preprocessed_query = self.preprocess_text(query)
        self.logger.debug(f"Preprocessed query: {preprocessed_query[:100]}...")
        
        # Generate embedding using the embedding engine
        try:
            query_embedding = self.embedding_engine.embed_text(preprocessed_query)
            self.logger.info(
                f"Successfully generated query embedding of dimension {len(query_embedding)}"
            )
            return query_embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Normalize and clean query text.
        
        This method performs minimal preprocessing to preserve user intent
        while ensuring consistent formatting. It strips leading/trailing
        whitespace and normalizes internal whitespace.
        
        Args:
            text: Raw query text
            
        Returns:
            Preprocessed query text
        """
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Normalize internal whitespace (replace multiple spaces with single space)
        text = ' '.join(text.split())
        
        self.logger.debug(f"Preprocessed text: '{text[:100]}...'")
        
        return text
