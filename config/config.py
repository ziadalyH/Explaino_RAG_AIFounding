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
    
    # LLM configuration (OpenSearch connector)
    llm_provider: str  # openai, bedrock, cohere, azure_openai, vertexai, sagemaker, deepseek, custom
    llm_endpoint: Optional[str]  # Full endpoint URL (for OpenAI/custom)
    llm_api_key: Optional[str]
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    
    # AWS Bedrock/SageMaker specific
    aws_region: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    sagemaker_endpoint: Optional[str]
    
    # Azure OpenAI specific
    azure_api_key: Optional[str]
    azure_api_base: Optional[str]
    azure_api_version: Optional[str]
    azure_deployment_name: Optional[str]
    
    # Cohere specific
    cohere_api_key: Optional[str]
    
    # Google VertexAI specific
    vertexai_project: Optional[str]
    vertexai_location: Optional[str]
    vertexai_access_token: Optional[str]
    
    # DeepSeek specific
    deepseek_api_key: Optional[str]
    
    # Custom headers and templates
    llm_headers: Optional[str]  # JSON string
    llm_request_template: Optional[str]  # JSON string
    
    # Legacy OpenAI configuration (for backward compatibility)
    openai_api_key: str
    
    # Embedding configuration
    embedding_model: str
    embedding_dimension: int
    embedding_provider: str  # "openai" or "local"
    
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
        
        # Determine workspace root (for Docker: /app/workspace, for local: current dir)
        workspace_root = Path("/app/workspace") if Path("/app/workspace").exists() else Path.cwd()
        
        # Helper function to resolve data paths
        def resolve_data_path(path_str: str) -> Path:
            """Resolve data path relative to workspace root."""
            path = Path(path_str)
            if path.is_absolute():
                return path
            else:
                # Relative path - resolve from workspace root
                return workspace_root / path
        
        # Override with environment variables (or use defaults)
        return cls(
            # Data paths - resolve relative to workspace
            transcript_dir=resolve_data_path(os.getenv("TRANSCRIPT_DIR", config_data.get("data", {}).get("transcript_dir", "data/transcripts"))),
            pdf_dir=resolve_data_path(os.getenv("PDF_DIR", config_data.get("data", {}).get("pdf_dir", "data/pdfs"))),
            
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
            
            # LLM configuration (OpenSearch connector)
            llm_provider=os.getenv("LLM_PROVIDER", config_data.get("llm", {}).get("provider", "openai")),
            llm_endpoint=os.getenv("LLM_ENDPOINT", config_data.get("llm", {}).get("endpoint")),
            llm_api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", config_data.get("llm", {}).get("api_key"))),
            llm_model=os.getenv("LLM_MODEL", config_data.get("llm", {}).get("model", "gpt-4o-mini")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", config_data.get("llm", {}).get("temperature", 0.3))),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", config_data.get("llm", {}).get("max_tokens", 500))),
            
            # AWS Bedrock/SageMaker
            aws_region=os.getenv("AWS_REGION", config_data.get("aws", {}).get("region")),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", config_data.get("aws", {}).get("access_key_id")),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", config_data.get("aws", {}).get("secret_access_key")),
            sagemaker_endpoint=os.getenv("SAGEMAKER_ENDPOINT", config_data.get("aws", {}).get("sagemaker_endpoint")),
            
            # Azure OpenAI
            azure_api_key=os.getenv("AZURE_API_KEY", config_data.get("azure", {}).get("api_key")),
            azure_api_base=os.getenv("AZURE_API_BASE", config_data.get("azure", {}).get("api_base")),
            azure_api_version=os.getenv("AZURE_API_VERSION", config_data.get("azure", {}).get("api_version", "2023-05-15")),
            azure_deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", config_data.get("azure", {}).get("deployment_name")),
            
            # Cohere
            cohere_api_key=os.getenv("COHERE_API_KEY", config_data.get("cohere", {}).get("api_key")),
            
            # Google VertexAI
            vertexai_project=os.getenv("VERTEXAI_PROJECT", config_data.get("vertexai", {}).get("project")),
            vertexai_location=os.getenv("VERTEXAI_LOCATION", config_data.get("vertexai", {}).get("location", "us-central1")),
            vertexai_access_token=os.getenv("VERTEXAI_ACCESS_TOKEN", config_data.get("vertexai", {}).get("access_token")),
            
            # DeepSeek
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", config_data.get("deepseek", {}).get("api_key")),
            
            # Custom headers and templates
            llm_headers=os.getenv("LLM_HEADERS", config_data.get("llm", {}).get("headers")),
            llm_request_template=os.getenv("LLM_REQUEST_TEMPLATE", config_data.get("llm", {}).get("request_template")),
            
            # Legacy OpenAI configuration (backward compatibility)
            openai_api_key=os.getenv("OPENAI_API_KEY", config_data.get("openai", {}).get("api_key", "")),
            
            # Embedding configuration
            embedding_model=os.getenv("EMBEDDING_MODEL", config_data.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", config_data.get("embedding", {}).get("dimension", 384))),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", config_data.get("embedding", {}).get("provider", "local")),
            
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
        # Validate LLM provider
        valid_providers = ["openai", "bedrock", "cohere", "azure_openai", "vertexai", "sagemaker", "deepseek", "custom"]
        if self.llm_provider not in valid_providers:
            raise ValueError(f"llm_provider must be one of: {valid_providers}")
        
        # Provider-specific validation
        if self.llm_provider == "openai":
            if not self.llm_api_key:
                raise ValueError("LLM_API_KEY required for OpenAI provider")
        
        elif self.llm_provider == "bedrock":
            if not all([self.aws_access_key_id, self.aws_secret_access_key, self.aws_region]):
                raise ValueError("AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) required for Bedrock")
        
        elif self.llm_provider == "cohere":
            if not self.cohere_api_key:
                raise ValueError("COHERE_API_KEY required for Cohere provider")
        
        elif self.llm_provider == "azure_openai":
            if not all([self.azure_api_key, self.azure_api_base]):
                raise ValueError("Azure credentials (AZURE_API_KEY, AZURE_API_BASE) required for Azure OpenAI")
        
        elif self.llm_provider == "vertexai":
            if not all([self.vertexai_project, self.vertexai_access_token]):
                raise ValueError("VertexAI credentials (VERTEXAI_PROJECT, VERTEXAI_ACCESS_TOKEN) required")
        
        elif self.llm_provider == "sagemaker":
            if not all([self.sagemaker_endpoint, self.aws_access_key_id, self.aws_secret_access_key, self.aws_region]):
                raise ValueError("SageMaker credentials (SAGEMAKER_ENDPOINT, AWS credentials) required")
        
        elif self.llm_provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY required for DeepSeek provider")
        
        elif self.llm_provider == "custom":
            if not all([self.llm_endpoint, self.llm_api_key]):
                raise ValueError("LLM_ENDPOINT and LLM_API_KEY required for custom provider")
        
        # Validate embedding configuration
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
