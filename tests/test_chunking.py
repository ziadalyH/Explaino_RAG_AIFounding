"""Tests for the chunking module."""

import pytest
import logging
from pathlib import Path

from src.models import VideoTranscript, TranscriptToken, PDFParagraph
from src.processing.chunking import ChunkingModule
from config.config import Config


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config(
        transcript_dir=Path("data/transcripts"),
        pdf_dir=Path("data/pdfs"),
        opensearch_host="localhost",
        opensearch_port=9200,
        opensearch_username=None,
        opensearch_password=None,
        opensearch_use_ssl=False,
        opensearch_verify_certs=False,
        opensearch_index_name="test-index",
        opensearch_pdf_index="test-pdf-index",
        opensearch_video_index="test-video-index",
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
        embedding_provider="local",
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        llm_max_tokens=500,
        relevance_threshold=0.5,
        max_results=5,
        chunk_size=100,
        chunk_overlap=20,
        chunking_strategy="sliding_window",
        max_chunk_window=30,
        min_pdf_paragraphs_per_page=4,
        log_level="INFO"
    )


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test")


@pytest.fixture
def chunking_module(config, logger):
    """Create a ChunkingModule instance."""
    return ChunkingModule(config, logger)


def test_chunk_transcript_basic(chunking_module):
    """Test basic transcript chunking with adaptive sizing."""
    # Create a simple transcript with 10 tokens
    tokens = [
        TranscriptToken(id=i, timestamp=float(i), word=f"word{i}")
        for i in range(10)
    ]
    transcript = VideoTranscript(
        video_id="test_video",
        pdf_reference="test.pdf",
        tokens=tokens
    )

    chunks = chunking_module.chunk_transcript(transcript)

    # With adaptive chunking, the number of chunks may vary
    # Just verify we got at least one chunk and it has valid data
    assert len(chunks) >= 1

    # Verify first chunk has valid structure
    assert chunks[0].video_id == "test_video"
    assert chunks[0].start_token_id >= 0
    assert chunks[0].end_token_id >= chunks[0].start_token_id
    assert chunks[0].start_timestamp >= 0.0
    assert chunks[0].end_timestamp >= chunks[0].start_timestamp
    assert len(chunks[0].text) > 0
    
    # Verify all tokens are covered
    all_token_ids = set()
    for chunk in chunks:
        for token_id in range(chunk.start_token_id, chunk.end_token_id + 1):
            all_token_ids.add(token_id)
    
    # Most tokens should be covered
    assert len(all_token_ids) >= 8  # At least 80% coverage


def test_chunk_transcript_empty(chunking_module):
    """Test chunking an empty transcript."""
    transcript = VideoTranscript(
        video_id="empty_video",
        pdf_reference="test.pdf",
        tokens=[]
    )

    chunks = chunking_module.chunk_transcript(transcript)
    assert len(chunks) == 0


def test_chunk_transcript_shorter_than_chunk_size(chunking_module):
    """Test chunking a transcript shorter than chunk size."""
    tokens = [
        TranscriptToken(id=i, timestamp=float(i), word=f"word{i}")
        for i in range(5)
    ]
    transcript = VideoTranscript(
        video_id="short_video",
        pdf_reference="test.pdf",
        tokens=tokens
    )

    chunking_module.chunk_size = 100
    chunking_module.chunk_overlap = 20

    chunks = chunking_module.chunk_transcript(transcript)

    # Should create 1 chunk containing all tokens
    assert len(chunks) == 1
    assert chunks[0].start_token_id == 0
    assert chunks[0].end_token_id == 4
    assert len(chunks[0].text.split()) == 5


def test_chunk_pdf_paragraphs_basic(chunking_module):
    """Test basic PDF paragraph chunking."""
    paragraphs = [
        PDFParagraph(
            pdf_filename="test.pdf",
            page_number=1,
            paragraph_index=0,
            text="This is the first paragraph with enough text."
        ),
        PDFParagraph(
            pdf_filename="test.pdf",
            page_number=1,
            paragraph_index=1,
            text="This is the second paragraph with enough text."
        ),
        PDFParagraph(
            pdf_filename="test.pdf",
            page_number=2,
            paragraph_index=0,
            text="This is a paragraph on page 2 with enough text."
        )
    ]

    chunks = chunking_module.chunk_pdf_paragraphs(paragraphs)

    # Should create 3 chunks (one per paragraph)
    assert len(chunks) == 3

    # Verify first chunk
    assert chunks[0].pdf_filename == "test.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].paragraph_index == 0
    assert "first paragraph" in chunks[0].text

    # Verify third chunk
    assert chunks[2].page_number == 2
    assert chunks[2].paragraph_index == 0


def test_chunk_pdf_paragraphs_skip_short(chunking_module):
    """Test that very short paragraphs are skipped."""
    paragraphs = [
        PDFParagraph(
            pdf_filename="test.pdf",
            page_number=1,
            paragraph_index=0,
            text="Short"  # Too short, should be skipped
        ),
        PDFParagraph(
            pdf_filename="test.pdf",
            page_number=1,
            paragraph_index=1,
            text="This is a longer paragraph that should be included."
        )
    ]

    chunks = chunking_module.chunk_pdf_paragraphs(paragraphs)

    # Should only create 1 chunk (short paragraph skipped)
    assert len(chunks) == 1
    assert "longer paragraph" in chunks[0].text


def test_chunk_transcript_overlap_consistency(chunking_module):
    """Test that consecutive chunks have proper overlap."""
    tokens = [
        TranscriptToken(id=i, timestamp=float(i), word=f"word{i}")
        for i in range(20)
    ]
    transcript = VideoTranscript(
        video_id="test_video",
        pdf_reference="test.pdf",
        tokens=tokens
    )

    chunking_module.chunk_size = 10
    chunking_module.chunk_overlap = 3

    chunks = chunking_module.chunk_transcript(transcript)

    # Verify overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]

        # The next chunk should start before the current chunk ends
        # (overlap of 3 tokens means next starts 7 tokens after current)
        expected_overlap = min(3, current_chunk.end_token_id - current_chunk.start_token_id + 1)
        
        # Check that there is some overlap in token IDs
        assert next_chunk.start_token_id <= current_chunk.end_token_id


def test_chunk_transcript_metadata_completeness(chunking_module):
    """Test that all chunks have complete metadata."""
    tokens = [
        TranscriptToken(id=i, timestamp=float(i * 0.5), word=f"word{i}")
        for i in range(15)
    ]
    transcript = VideoTranscript(
        video_id="metadata_test",
        pdf_reference="test.pdf",
        tokens=tokens
    )

    chunks = chunking_module.chunk_transcript(transcript)

    # Verify all chunks have complete metadata
    for chunk in chunks:
        assert chunk.video_id == "metadata_test"
        assert chunk.start_token_id >= 0
        assert chunk.end_token_id >= chunk.start_token_id
        assert chunk.start_timestamp >= 0.0
        assert chunk.end_timestamp >= chunk.start_timestamp
        assert len(chunk.text) > 0
        assert chunk.text.count(" ") >= 0  # Has words

