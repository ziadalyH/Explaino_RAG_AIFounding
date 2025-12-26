"""Configuration management for the RAG Chatbot Backend."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class Config:
    """Configuration for the RAG Chatbot Backend."""
    
    # Data paths
    transcript_dir: Path
    pdf_dir: Path
    
    # OpenSearch configuration
    opensearch_host: str
    opensearch_port: int
    opensearch_username: Optional[str]
    opensearch_password: Optional[str]
    opensearch_use_ssl: bool
    opensearch_verify_certs: bool
    opensearch_index_name: str  # Legacy - kept for backward compatibility
    opensearch_pdf_index: str   # Separate index for PDFs
    opensearch_video_index: str # Separate index for videos
    
    # OpenAI configuration
    openai_api_key: str
    embedding_model: str
    embedding_dimension: int
    embedding_provider: str  # "openai" or "local"
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    
    # Retrieval configuration
    relevance_threshold: float
    max_results: int
    
    # Chunking configuration
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str  # "sliding_window" or "all_combinations"
    max_chunk_window: int  # For all_combinations strategy: max tokens per chunk
    min_pdf_paragraphs_per_page: int  # Minimum paragraphs to extract per PDF page
    
    # Logging
    log_level: str
    
    @classmethod
    def from_env(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from environment variables and optional YAML file.
        
        Environment variables take priority over YAML values.
        If no YAML file is provided, all values come from environment variables with sensible defaults.
        
        Args:
            config_path: Optional path to YAML config file (defaults to None)
        """
        # Load YAML config if provided
        config_data = {}
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                # YAML file specified but doesn't exist - just use defaults
                pass
        
        # Override with environment variables (or use defaults)
        return cls(
            # Data paths
            transcript_dir=Path(os.getenv("TRANSCRIPT_DIR", config_data.get("data", {}).get("transcript_dir", "data/transcripts"))),
            pdf_dir=Path(os.getenv("PDF_DIR", config_data.get("data", {}).get("pdf_dir", "data/pdfs"))),
            
            # OpenSearch configuration
            opensearch_host=os.getenv("OPENSEARCH_HOST", config_data.get("opensearch", {}).get("host", "localhost")),
            opensearch_port=int(os.getenv("OPENSEARCH_PORT", config_data.get("opensearch", {}).get("port", 9200))),
            opensearch_username=os.getenv("OPENSEARCH_USERNAME", config_data.get("opensearch", {}).get("username")),
            opensearch_password=os.getenv("OPENSEARCH_PASSWORD", config_data.get("opensearch", {}).get("password")),
            opensearch_use_ssl=os.getenv("OPENSEARCH_USE_SSL", str(config_data.get("opensearch", {}).get("use_ssl", False))).lower() == "true",
            opensearch_verify_certs=os.getenv("OPENSEARCH_VERIFY_CERTS", str(config_data.get("opensearch", {}).get("verify_certs", True))).lower() == "true",
            opensearch_index_name=os.getenv("OPENSEARCH_INDEX_NAME", config_data.get("opensearch", {}).get("index_name", "rag-index")),
            opensearch_pdf_index=os.getenv("OPENSEARCH_PDF_INDEX", config_data.get("opensearch", {}).get("pdf_index", "rag-pdf-index")),
            opensearch_video_index=os.getenv("OPENSEARCH_VIDEO_INDEX", config_data.get("opensearch", {}).get("video_index", "rag-video-index")),
            
            # OpenAI configuration
            openai_api_key=os.getenv("OPENAI_API_KEY", config_data.get("openai", {}).get("api_key", "")),
            embedding_model=os.getenv("EMBEDDING_MODEL", config_data.get("openai", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", config_data.get("openai", {}).get("embedding_dimension", 384))),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", config_data.get("openai", {}).get("embedding_provider", "local")),
            llm_model=os.getenv("LLM_MODEL", config_data.get("openai", {}).get("llm_model", "gpt-4o-mini")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", config_data.get("openai", {}).get("llm_temperature", 0.3))),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", config_data.get("openai", {}).get("llm_max_tokens", 500))),
            
            # Retrieval configuration
            relevance_threshold=float(os.getenv("RELEVANCE_THRESHOLD", config_data.get("retrieval", {}).get("relevance_threshold", 0.5))),
            max_results=int(os.getenv("MAX_RESULTS", config_data.get("retrieval", {}).get("max_results", 5))),
            
            # Chunking configuration
            chunk_size=int(os.getenv("CHUNK_SIZE", config_data.get("chunking", {}).get("chunk_size", 100))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", config_data.get("chunking", {}).get("chunk_overlap", 20))),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", config_data.get("chunking", {}).get("strategy", "sliding_window")),
            max_chunk_window=int(os.getenv("MAX_CHUNK_WINDOW", config_data.get("chunking", {}).get("max_window", 30))),
            min_pdf_paragraphs_per_page=int(os.getenv("MIN_PDF_PARAGRAPHS_PER_PAGE", config_data.get("chunking", {}).get("min_pdf_paragraphs_per_page", 4))),
            
            # Logging
            log_level=os.getenv("LOG_LEVEL", config_data.get("logging", {}).get("log_level", "INFO")),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI embeddings")
        
        if self.embedding_provider not in ["openai", "local"]:
            raise ValueError("embedding_provider must be 'openai' or 'local'")
        
        if self.relevance_threshold < 0 or self.relevance_threshold > 1:
            raise ValueError("relevance_threshold must be between 0 and 1")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
        
        if self.chunking_strategy not in ["sliding_window", "all_combinations"]:
            raise ValueError("chunking_strategy must be 'sliding_window' or 'all_combinations'")
        
        if self.max_chunk_window <= 0:
            raise ValueError("max_chunk_window must be positive")
