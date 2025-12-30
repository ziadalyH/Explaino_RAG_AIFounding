"""Tests for the VectorIndexBuilder module."""

import pytest
import logging
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.processing.indexing import VectorIndexBuilder
from src.processing.embedding import EmbeddingEngine
from src.models import TranscriptChunk, PDFChunk
from config.config import Config


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.opensearch_host = "localhost"
    config.opensearch_port = 9200
    config.opensearch_username = None
    config.opensearch_password = None
    config.opensearch_use_ssl = False
    config.opensearch_verify_certs = False
    config.opensearch_index_name = "test-index"
    config.opensearch_pdf_index = "test-pdf-index"
    config.opensearch_video_index = "test-video-index"
    config.embedding_dimension = 1536
    config.embedding_provider = "local"
    return config


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return logging.getLogger("test")


@pytest.fixture
def sample_transcript_chunks():
    """Create sample transcript chunks for testing."""
    return [
        TranscriptChunk(
            video_id="video_001",
            start_token_id=0,
            end_token_id=10,
            start_timestamp=0.0,
            end_timestamp=5.0,
            text="This is a test transcript chunk"
        ),
        TranscriptChunk(
            video_id="video_001",
            start_token_id=8,
            end_token_id=18,
            start_timestamp=4.0,
            end_timestamp=9.0,
            text="Another test transcript chunk with overlap"
        )
    ]


@pytest.fixture
def sample_pdf_chunks():
    """Create sample PDF chunks for testing."""
    return [
        PDFChunk(
            pdf_filename="test.pdf",
            page_number=1,
            paragraph_index=0,
            text="This is a test PDF paragraph"
        ),
        PDFChunk(
            pdf_filename="test.pdf",
            page_number=1,
            paragraph_index=1,
            text="Another test PDF paragraph"
        )
    ]


def test_initialize_opensearch_client(mock_config, mock_logger):
    """Test OpenSearch client initialization."""
    with patch('src.processing.indexing.OpenSearch') as mock_opensearch:
        # Mock the OpenSearch client
        mock_client = MagicMock()
        mock_client.info.return_value = {"cluster_name": "test-cluster"}
        mock_opensearch.return_value = mock_client
        
        # Create VectorIndexBuilder
        builder = VectorIndexBuilder(mock_config, mock_logger)
        
        # Verify client was initialized
        assert builder.opensearch_client is not None
        mock_opensearch.assert_called_once()
        mock_client.info.assert_called_once()


def test_create_index_if_not_exists_new_index(mock_config, mock_logger):
    """Test creating a new index when it doesn't exist."""
    with patch('src.processing.indexing.OpenSearch') as mock_opensearch:
        # Mock the OpenSearch client
        mock_client = MagicMock()
        mock_client.info.return_value = {"cluster_name": "test-cluster"}
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        mock_opensearch.return_value = mock_client
        
        # Create VectorIndexBuilder
        builder = VectorIndexBuilder(mock_config, mock_logger)
        
        # Create index
        builder.create_index_if_not_exists("test-index", "video")
        
        # Verify index creation was called
        mock_client.indices.exists.assert_called_with(index="test-index")
        mock_client.indices.create.assert_called_once()


def test_create_index_if_not_exists_existing_index(mock_config, mock_logger):
    """Test that existing index is not recreated."""
    with patch('src.processing.indexing.OpenSearch') as mock_opensearch:
        # Mock the OpenSearch client
        mock_client = MagicMock()
        mock_client.info.return_value = {"cluster_name": "test-cluster"}
        mock_client.indices.exists.return_value = True
        mock_opensearch.return_value = mock_client
        
        # Create VectorIndexBuilder
        builder = VectorIndexBuilder(mock_config, mock_logger)
        
        # Try to create index
        builder.create_index_if_not_exists("test-index", "video")
        
        # Verify index creation was NOT called
        mock_client.indices.exists.assert_called_with(index="test-index")
        mock_client.indices.create.assert_not_called()


def test_index_documents(mock_config, mock_logger):
    """Test bulk indexing of documents."""
    with patch('src.processing.indexing.OpenSearch') as mock_opensearch, \
         patch('src.processing.indexing.helpers') as mock_helpers:
        
        # Mock the OpenSearch client
        mock_client = MagicMock()
        mock_client.info.return_value = {"cluster_name": "test-cluster"}
        mock_opensearch.return_value = mock_client
        
        # Mock bulk helper
        mock_helpers.bulk.return_value = (2, [])
        
        # Create VectorIndexBuilder
        builder = VectorIndexBuilder(mock_config, mock_logger)
        
        # Create test documents
        documents = [
            {
                "embedding": [0.1] * 1536,
                "source_type": "video",
                "video_id": "test_video",
                "text": "test text"
            },
            {
                "embedding": [0.2] * 1536,
                "source_type": "pdf",
                "pdf_filename": "test.pdf",
                "text": "test pdf text"
            }
        ]
        
        # Index documents
        builder.index_documents(documents, "test-index")
        
        # Verify bulk was called
        mock_helpers.bulk.assert_called()
        mock_client.indices.refresh.assert_called_once_with(index="test-index")


def test_build_index(mock_config, mock_logger, sample_transcript_chunks, sample_pdf_chunks):
    """Test building the complete index."""
    with patch('src.processing.indexing.OpenSearch') as mock_opensearch, \
         patch('src.processing.indexing.helpers') as mock_helpers:
        
        # Mock the OpenSearch client
        mock_client = MagicMock()
        mock_client.info.return_value = {"cluster_name": "test-cluster"}
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        mock_opensearch.return_value = mock_client
        
        # Mock bulk helper
        mock_helpers.bulk.return_value = (2, [])
        
        # Create VectorIndexBuilder
        builder = VectorIndexBuilder(mock_config, mock_logger)
        
        # Create mock embedding engine
        mock_embedding_engine = Mock(spec=EmbeddingEngine)
        mock_embedding_engine.embed_batch.side_effect = [
            np.array([[0.1] * 1536, [0.2] * 1536], dtype=np.float32),  # transcript embeddings
            np.array([[0.3] * 1536, [0.4] * 1536], dtype=np.float32)   # pdf content embeddings
        ]
        
        # Build index
        builder.build_index(sample_transcript_chunks, sample_pdf_chunks, mock_embedding_engine)
        
        # Verify indices were created (one for video, one for pdf)
        assert mock_client.indices.create.call_count == 2
        
        # Verify embeddings were generated
        assert mock_embedding_engine.embed_batch.call_count == 2
        
        # Verify documents were indexed (called twice, once for videos, once for pdfs)
        assert mock_helpers.bulk.call_count >= 2


def test_index_documents_empty_list(mock_config, mock_logger):
    """Test indexing with empty document list."""
    with patch('src.processing.indexing.OpenSearch') as mock_opensearch:
        # Mock the OpenSearch client
        mock_client = MagicMock()
        mock_client.info.return_value = {"cluster_name": "test-cluster"}
        mock_opensearch.return_value = mock_client
        
        # Create VectorIndexBuilder
        builder = VectorIndexBuilder(mock_config, mock_logger)
        
        # Index empty list (should not raise error)
        builder.index_documents([], "test-index")
        
        # Verify refresh was not called
        mock_client.indices.refresh.assert_not_called()
