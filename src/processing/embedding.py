"""Embedding generation module using OpenAI or local models."""

import logging
import time
import string
from typing import List, Optional
import numpy as np
import openai
import nltk
from nltk.corpus import stopwords

from config.config import Config

# Download stopwords on first import (will be cached)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class EmbeddingEngine:
    """Generate vector embeddings for text chunks using OpenAI's API or local models."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the embedding engine.
        
        Args:
            config: Configuration object containing embedding settings
            logger: Logger instance for logging operations
        """
        self.config = config
        self.logger = logger
        self.model_name = config.embedding_model
        self.embedding_dimension = config.embedding_dimension
        self.embedding_provider = config.embedding_provider
        self._embedding_cache = {}  # Cache to reduce API costs
        
        # Load English stop words for preprocessing before embedding
        self.stop_words = set(stopwords.words('english'))
        
        # Log configuration
        self.logger.info("=" * 60)
        self.logger.info("EMBEDDING ENGINE CONFIGURATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Provider: {self.embedding_provider}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Dimension: {self.embedding_dimension}")
        self.logger.info("=" * 60)
        
        if self.embedding_provider == "openai":
            self._initialize_openai_client()
        elif self.embedding_provider == "local":
            self._initialize_local_model()
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
        
    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client with API key from config.
        
        Raises:
            ValueError: If API key is not configured
        """
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.logger.info(f"Initializing OpenAI client with model: {self.model_name}")
        openai.api_key = self.config.openai_api_key
    
    def _initialize_local_model(self) -> None:
        """Initialize local sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            self.logger.info(f"🔄 Loading local embedding model: {self.model_name}")
            
            # Check if model exists locally in ./models directory
            local_model_path = "./models"
            model_exists = False
            
            if os.path.exists(local_model_path):
                # Check if this specific model is cached
                model_cache_name = self.model_name.replace('/', '_')
                model_cache_path = os.path.join(local_model_path, model_cache_name)
                
                if os.path.exists(model_cache_path):
                    model_exists = True
                    self.logger.info(f"✅ Found cached model at: {model_cache_path}")
                    self.logger.info(f"📦 Loading from cache (fast)...")
                else:
                    self.logger.info(f"⚠️  Model not in cache")
                    self.logger.info(f"🌐 Will download from HuggingFace: {self.model_name}")
                
                self.local_model = SentenceTransformer(self.model_name, cache_folder=local_model_path)
            else:
                self.logger.info(f"📁 Cache directory not found, creating: {local_model_path}")
                self.logger.info(f"🌐 Downloading model from HuggingFace: {self.model_name}")
                os.makedirs(local_model_path, exist_ok=True)
                self.local_model = SentenceTransformer(self.model_name, cache_folder=local_model_path)
            
            # Get actual model info
            actual_dimension = self.local_model.get_sentence_embedding_dimension()
            max_seq_length = self.local_model.max_seq_length
            
            self.logger.info("=" * 60)
            self.logger.info("✅ MODEL LOADED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Model Name: {self.model_name}")
            self.logger.info(f"Model Type: {type(self.local_model).__name__}")
            self.logger.info(f"Embedding Dimension: {actual_dimension}")
            self.logger.info(f"Max Sequence Length: {max_seq_length} tokens")
            self.logger.info(f"Cached: {'Yes' if model_exists else 'No (just downloaded)'}")
            self.logger.info(f"Cache Location: {local_model_path}")
            self.logger.info("=" * 60)
            
            # Validate dimension matches config
            if actual_dimension != self.embedding_dimension:
                self.logger.error("=" * 60)
                self.logger.error("❌ DIMENSION MISMATCH ERROR")
                self.logger.error("=" * 60)
                self.logger.error(f"Expected dimension: {self.embedding_dimension}")
                self.logger.error(f"Actual dimension: {actual_dimension}")
                self.logger.error("")
                self.logger.error("Fix: Update EMBEDDING_DIMENSION in .env to match model:")
                self.logger.error(f"EMBEDDING_DIMENSION={actual_dimension}")
                self.logger.error("=" * 60)
                raise ValueError(
                    f"Dimension mismatch: Expected {self.embedding_dimension}, "
                    f"but model produces {actual_dimension}. "
                    f"Update EMBEDDING_DIMENSION={actual_dimension} in .env"
                )
            
        except ImportError:
            self.logger.error("❌ sentence-transformers not installed")
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load local model: {str(e)}")
    
    def preprocess_for_embedding(self, text: str) -> str:
        """
        Preprocess text before embedding by removing stop words and punctuation.
        
        This focuses the embedding on important keywords while preserving
        the full text for LLM context.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text with stop words and punctuation removed
        """
        # Strip whitespace
        text = text.strip()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize and remove stop words
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        
        # Join back together
        text = ' '.join(filtered_words)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        The text is preprocessed (stop words removed) before embedding,
        but the original text is preserved in the index for LLM context.
        
        Args:
            text: Text to embed (will be preprocessed automatically)
            
        Returns:
            Numpy array containing the embedding vector
            
        Raises:
            ValueError: If text is empty
            Exception: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        # Preprocess text for embedding (remove stop words and punctuation)
        preprocessed_text = self.preprocess_for_embedding(text)
        
        # If preprocessing removed everything, use original text
        if not preprocessed_text:
            self.logger.warning(f"Preprocessing removed all content, using original: '{text[:50]}'")
            preprocessed_text = text.strip()
        
        # Check cache first (using preprocessed text as key)
        cache_key = preprocessed_text
        if cache_key in self._embedding_cache:
            self.logger.debug(f"Using cached embedding for text: {preprocessed_text[:50]}...")
            return self._embedding_cache[cache_key]
        
        # Generate embedding based on provider (using preprocessed text)
        if self.embedding_provider == "openai":
            embedding = self._embed_text_openai(preprocessed_text)
        elif self.embedding_provider == "local":
            embedding = self._embed_text_local(preprocessed_text)
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
        
        # Cache the result
        self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _embed_text_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Generating OpenAI embedding for text: {text[:50]}...")
                response = openai.Embedding.create(
                    model=self.model_name,
                    input=text
                )
                
                embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
                
                # Validate embedding dimension
                if len(embedding) != self.embedding_dimension:
                    raise ValueError(
                        f"Expected embedding dimension {self.embedding_dimension}, "
                        f"got {len(embedding)}"
                    )
                
                self.logger.debug(f"Successfully generated embedding of dimension {len(embedding)}")
                return embedding
                
            except openai.error.RateLimitError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise
                    
            except openai.error.APIError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"API error: {str(e)}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"API error after {max_retries} attempts: {str(e)}")
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error generating embedding: {str(e)}")
                raise
    
    def _embed_text_local(self, text: str) -> np.ndarray:
        """Generate embedding using local sentence transformer model.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        try:
            self.logger.debug(f"Generating local embedding for text: {text[:50]}...")
            
            # Generate embedding
            embedding = self.local_model.encode(text, convert_to_numpy=True)
            embedding = embedding.astype(np.float32)
            
            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.embedding_dimension}, "
                    f"got {len(embedding)}"
                )
            
            self.logger.debug(f"Successfully generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating local embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of shape (n_texts, embedding_dimension) containing embeddings
            
        Raises:
            ValueError: If texts list is empty or contains empty strings
            Exception: If embedding generation fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")
        
        # Preprocess and filter out empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                preprocessed = self.preprocess_for_embedding(text)
                if not preprocessed:
                    self.logger.warning(f"Preprocessing removed all content at index {i}, using original")
                    preprocessed = text.strip()
                valid_texts.append(preprocessed)
                valid_indices.append(i)
            else:
                self.logger.warning(f"Skipping empty text at index {i}")
        
        if not valid_texts:
            raise ValueError("All texts are empty")
        
        self.logger.info(f"Generating embeddings for {len(valid_texts)} texts")
        
        # Check cache for existing embeddings
        embeddings_list = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(valid_texts):
            if text in self._embedding_cache:
                embeddings_list.append((i, self._embedding_cache[text]))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        self.logger.info(
            f"Found {len(embeddings_list)} cached embeddings, "
            f"need to generate {len(texts_to_embed)} new embeddings"
        )
        
        # Generate embeddings based on provider
        if self.embedding_provider == "openai":
            new_embeddings = self._embed_batch_openai(texts_to_embed, text_indices)
        elif self.embedding_provider == "local":
            new_embeddings = self._embed_batch_local(texts_to_embed, text_indices)
        else:
            raise ValueError(f"Unknown embedding provider: {self.embedding_provider}")
        
        # Combine cached and new embeddings
        embeddings_list.extend(new_embeddings)
        
        # Sort embeddings by original index and convert to array
        embeddings_list.sort(key=lambda x: x[0])
        embeddings_array = np.array([emb for _, emb in embeddings_list], dtype=np.float32)
        
        self.logger.info(
            f"Successfully generated embeddings array of shape {embeddings_array.shape}"
        )
        
        return embeddings_array
    
    def _embed_batch_local(self, texts: List[str], text_indices: List[int]) -> List[tuple]:
        """Generate embeddings for multiple texts using local model.
        
        Args:
            texts: List of texts to embed
            text_indices: Original indices of texts
            
        Returns:
            List of (index, embedding) tuples
        """
        if not texts:
            return []
        
        try:
            from tqdm import tqdm
            
            self.logger.info(f"Generating {len(texts)} embeddings with local model")
            
            # Generate all embeddings at once with progress bar
            embeddings = self.local_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,  # Always show progress bar
                batch_size=32  # Process in batches for better progress tracking
            )
            embeddings = embeddings.astype(np.float32)
            
            # Create result list with indices
            result = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                # Validate dimension
                if len(embedding) != self.embedding_dimension:
                    raise ValueError(
                        f"Expected embedding dimension {self.embedding_dimension}, "
                        f"got {len(embedding)}"
                    )
                
                original_index = text_indices[i]
                result.append((original_index, embedding))
                
                # Cache the embedding
                self._embedding_cache[text] = embedding
            
            self.logger.info(f"Successfully generated {len(result)} local embeddings")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating local embeddings: {str(e)}")
            raise
    
    def _embed_batch_openai(self, texts: List[str], text_indices: List[int]) -> List[tuple]:
        """Generate embeddings for multiple texts using OpenAI API.
        
        Args:
            texts: List of texts to embed
            text_indices: Original indices of texts
            
        Returns:
            List of (index, embedding) tuples
        """
        if not texts:
            return []
        
        result = []
        
        # Process texts in batches (OpenAI limit is 2048)
        batch_size = 2048
        max_retries = 3
        base_delay = 1.0
        
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_indices = text_indices[batch_start:batch_end]
            
            self.logger.debug(
                f"Processing batch {batch_start // batch_size + 1}: "
                f"{len(batch_texts)} texts"
            )
            
            # Retry logic for batch
            for attempt in range(max_retries):
                try:
                    response = openai.Embedding.create(
                        model=self.model_name,
                        input=batch_texts
                    )
                    
                    # Extract embeddings and add to list
                    for i, embedding_data in enumerate(response['data']):
                        embedding = np.array(embedding_data['embedding'], dtype=np.float32)
                        
                        # Validate dimension
                        if len(embedding) != self.embedding_dimension:
                            raise ValueError(
                                f"Expected embedding dimension {self.embedding_dimension}, "
                                f"got {len(embedding)}"
                            )
                        
                        original_index = batch_indices[i]
                        result.append((original_index, embedding))
                        
                        # Cache the embedding
                        self._embedding_cache[batch_texts[i]] = embedding
                    
                    self.logger.debug(f"Successfully processed batch of {len(batch_texts)} texts")
                    break  # Success, exit retry loop
                    
                except openai.error.RateLimitError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(
                            f"Rate limit hit on batch, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"Rate limit exceeded after {max_retries} attempts on batch"
                        )
                        raise
                        
                except openai.error.APIError as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(
                            f"API error on batch: {str(e)}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        self.logger.error(
                            f"API error after {max_retries} attempts on batch: {str(e)}"
                        )
                        raise
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error processing batch: {str(e)}")
                    raise
        
        return result
