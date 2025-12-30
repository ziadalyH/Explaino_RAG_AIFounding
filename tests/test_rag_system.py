"""Tests for RAGSystem orchestrator."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.rag_system import RAGSystem
from config.config import Config
from src.models import VideoResponse, PDFResponse, NoAnswerResponse


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.transcript_dir = Mock()
    config.pdf_dir = Mock()
    config.opensearch_host = "localhost"
    config.opensearch_port = 9200
    config.opensearch_username = None
    config.opensearch_password = None
    config.opensearch_use_ssl = False
    config.opensearch_verify_certs = False
    config.opensearch_index_name = "test-index"
    config.opensearch_pdf_index = "test-pdf-index"
    config.opensearch_video_index = "test-video-index"
    # LLM configuration
    config.llm_provider = "openai"
    config.llm_endpoint = None
    config.llm_api_key = "test-key"
    config.llm_model = "gpt-4o-mini"
    config.llm_temperature = 0.3
    config.llm_max_tokens = 500
    # AWS credentials
    config.aws_region = None
    config.aws_access_key_id = None
    config.aws_secret_access_key = None
    config.aws_session_token = None
    config.sagemaker_endpoint = None
    # Azure OpenAI
    config.azure_api_key = None
    config.azure_api_base = None
    config.azure_api_version = None
    config.azure_deployment_name = None
    # Cohere
    config.cohere_api_key = None
    # VertexAI
    config.vertexai_project = None
    config.vertexai_location = None
    config.vertexai_access_token = None
    # DeepSeek
    config.deepseek_api_key = None
    # Custom
    config.llm_headers = None
    config.llm_request_template = None
    # Legacy
    config.openai_api_key = "test-key"
    # Embedding
    config.embedding_model = "text-embedding-3-small"
    config.embedding_dimension = 1536
    config.embedding_provider = "local"
    # Retrieval
    config.relevance_threshold = 0.5
    config.max_results = 5
    # Chunking
    config.chunk_size = 100
    config.chunk_overlap = 20
    config.chunking_strategy = "sliding_window"
    config.max_chunk_window = 30
    config.min_pdf_paragraphs_per_page = 4
    # Logging
    config.log_level = "INFO"
    return config


def test_rag_system_initialization(mock_config):
    """Test that RAGSystem can be initialized with all components."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine') as mock_embedding:
        
        # Mock the OpenSearch client
        mock_opensearch_client = MagicMock()
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Verify all components are initialized
        assert rag_system.config == mock_config
        assert rag_system.logger is not None
        assert rag_system.embedding_engine is not None
        assert rag_system.vector_index_builder is not None
        assert rag_system.transcript_ingester is not None
        assert rag_system.pdf_ingester is not None
        assert rag_system.chunking_module is not None
        assert rag_system.query_processor is not None
        assert rag_system.retrieval_engine is not None
        assert rag_system.response_generator is not None


def test_check_index_exists_when_index_missing(mock_config):
    """Test check_index_exists returns False when index doesn't exist."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine'):
        
        # Mock OpenSearch client
        mock_opensearch_client = MagicMock()
        mock_opensearch_client.indices.exists.return_value = False
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Check index exists
        result = rag_system.check_index_exists()
        
        assert result is False
        # Should be called twice (once for pdf index, once for video index)
        assert mock_opensearch_client.indices.exists.call_count == 2


def test_check_index_exists_when_index_empty(mock_config):
    """Test check_index_exists returns False when index exists but is empty."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine'):
        
        # Mock OpenSearch client
        mock_opensearch_client = MagicMock()
        mock_opensearch_client.indices.exists.return_value = True
        mock_opensearch_client.count.return_value = {'count': 0}
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Check index exists
        result = rag_system.check_index_exists()
        
        assert result is False
        # Should be called twice (once for pdf index, once for video index)
        assert mock_opensearch_client.count.call_count == 2


def test_check_index_exists_when_index_has_documents(mock_config):
    """Test check_index_exists returns True when index has documents."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine'):
        
        # Mock OpenSearch client
        mock_opensearch_client = MagicMock()
        mock_opensearch_client.indices.exists.return_value = True
        mock_opensearch_client.count.return_value = {'count': 100}
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Check index exists
        result = rag_system.check_index_exists()
        
        assert result is True


def test_answer_question_raises_on_empty_query(mock_config):
    """Test that answer_question raises ValueError for empty query."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine'):
        
        mock_opensearch_client = MagicMock()
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Try to answer empty question
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_system.answer_question("")


def test_answer_question_returns_no_answer_when_no_results(mock_config):
    """Test that answer_question returns NoAnswerResponse when no results found."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine') as mock_embedding:
        
        mock_opensearch_client = MagicMock()
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Mock query processor to return embedding
        import numpy as np
        mock_embedding_array = np.zeros(1536)
        rag_system.query_processor.process_query = Mock(return_value=mock_embedding_array)
        
        # Mock retrieval engine to return None
        rag_system.retrieval_engine.retrieve = Mock(return_value=None)
        
        # Mock response generator
        rag_system.response_generator.generate_response = Mock(
            return_value=NoAnswerResponse()
        )
        
        # Answer question
        response = rag_system.answer_question("What is the meaning of life?")
        
        # Verify response
        assert isinstance(response, NoAnswerResponse)
        assert response.answer_type == "no_answer"


def test_build_index_skips_when_index_exists(mock_config):
    """Test that build_index processes when index exists but has no new files."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine'):
        
        mock_opensearch_client = MagicMock()
        mock_opensearch_client.indices.exists.return_value = True
        mock_opensearch_client.count.return_value = {'count': 100}
        mock_opensearch_client.search.return_value = {
            'hits': {
                'hits': []
            }
        }
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Mock ingesters to return empty lists (no new files)
        rag_system.transcript_ingester.ingest_directory = Mock(return_value=[])
        rag_system.pdf_ingester.ingest_directory = Mock(return_value=[])
        
        # Build index (should process but find no new files)
        rag_system.build_index(force_rebuild=False)
        
        # Verify ingesters were called (to check for new files)
        rag_system.transcript_ingester.ingest_directory.assert_called_once()
        rag_system.pdf_ingester.ingest_directory.assert_called_once()


def test_build_index_rebuilds_when_forced(mock_config):
    """Test that build_index rebuilds when force_rebuild=True."""
    with patch('src.rag_system.VectorIndexBuilder') as mock_builder, \
         patch('src.rag_system.EmbeddingEngine'):
        
        mock_opensearch_client = MagicMock()
        mock_opensearch_client.indices.exists.return_value = True
        mock_opensearch_client.count.return_value = {'count': 100}
        mock_builder.return_value.opensearch_client = mock_opensearch_client
        mock_builder.return_value.build_index = Mock()
        
        # Initialize RAGSystem
        rag_system = RAGSystem(mock_config)
        
        # Mock ingesters
        rag_system.transcript_ingester.ingest_directory = Mock(return_value=[])
        rag_system.pdf_ingester.ingest_directory = Mock(return_value=[])
        rag_system.chunking_module.chunk_transcript = Mock(return_value=[])
        rag_system.chunking_module.chunk_pdf_paragraphs = Mock(return_value=[])
        
        # Build index with force_rebuild
        rag_system.build_index(force_rebuild=True)
        
        # Verify ingesters were called
        rag_system.transcript_ingester.ingest_directory.assert_called_once()
        rag_system.pdf_ingester.ingest_directory.assert_called_once()
