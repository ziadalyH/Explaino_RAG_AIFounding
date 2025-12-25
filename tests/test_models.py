"""Tests for data models."""

import json
from src.models import (
    TranscriptToken,
    VideoTranscript,
    PDFParagraph,
    TranscriptChunk,
    PDFChunk,
    VideoResult,
    PDFResult,
    VideoResponse,
    PDFResponse,
    NoAnswerResponse
)


def test_transcript_token_serialization():
    """Test TranscriptToken JSON serialization."""
    token = TranscriptToken(id=1, timestamp=0.5, word="hello")
    
    # Test to_dict
    token_dict = token.to_dict()
    assert token_dict == {"id": 1, "timestamp": 0.5, "word": "hello"}
    
    # Test to_json
    token_json = token.to_json()
    assert json.loads(token_json) == {"id": 1, "timestamp": 0.5, "word": "hello"}
    
    # Test from_dict
    token_restored = TranscriptToken.from_dict(token_dict)
    assert token_restored == token


def test_video_transcript_serialization():
    """Test VideoTranscript JSON serialization."""
    tokens = [
        TranscriptToken(id=1, timestamp=0.5, word="hello"),
        TranscriptToken(id=2, timestamp=1.0, word="world")
    ]
    transcript = VideoTranscript(
        video_id="video123",
        pdf_reference="doc.pdf",
        tokens=tokens
    )
    
    # Test to_dict
    transcript_dict = transcript.to_dict()
    assert transcript_dict["video_id"] == "video123"
    assert transcript_dict["pdf_reference"] == "doc.pdf"
    assert len(transcript_dict["tokens"]) == 2
    
    # Test to_json
    transcript_json = transcript.to_json()
    parsed = json.loads(transcript_json)
    assert parsed["video_id"] == "video123"
    
    # Test from_dict
    transcript_restored = VideoTranscript.from_dict(transcript_dict)
    assert transcript_restored.video_id == transcript.video_id
    assert len(transcript_restored.tokens) == len(transcript.tokens)


def test_pdf_paragraph_serialization():
    """Test PDFParagraph JSON serialization."""
    paragraph = PDFParagraph(
        pdf_filename="test.pdf",
        page_number=1,
        paragraph_index=0,
        text="This is a test paragraph."
    )
    
    # Test to_dict
    para_dict = paragraph.to_dict()
    assert para_dict["pdf_filename"] == "test.pdf"
    assert para_dict["page_number"] == 1
    
    # Test to_json
    para_json = paragraph.to_json()
    assert json.loads(para_json)["text"] == "This is a test paragraph."
    
    # Test from_dict
    para_restored = PDFParagraph.from_dict(para_dict)
    assert para_restored == paragraph


def test_transcript_chunk_serialization():
    """Test TranscriptChunk JSON serialization."""
    chunk = TranscriptChunk(
        video_id="video123",
        start_token_id=1,
        end_token_id=10,
        start_timestamp=0.5,
        end_timestamp=5.0,
        text="hello world test"
    )
    
    # Test to_dict
    chunk_dict = chunk.to_dict()
    assert chunk_dict["video_id"] == "video123"
    assert chunk_dict["start_token_id"] == 1
    
    # Test to_json
    chunk_json = chunk.to_json()
    parsed = json.loads(chunk_json)
    assert parsed["end_timestamp"] == 5.0
    
    # Test from_dict
    chunk_restored = TranscriptChunk.from_dict(chunk_dict)
    assert chunk_restored == chunk


def test_pdf_chunk_serialization():
    """Test PDFChunk JSON serialization."""
    chunk = PDFChunk(
        pdf_filename="test.pdf",
        page_number=1,
        paragraph_index=0,
        text="This is a chunk."
    )
    
    # Test to_dict
    chunk_dict = chunk.to_dict()
    assert chunk_dict["pdf_filename"] == "test.pdf"
    
    # Test to_json
    chunk_json = chunk.to_json()
    assert json.loads(chunk_json)["text"] == "This is a chunk."
    
    # Test from_dict
    chunk_restored = PDFChunk.from_dict(chunk_dict)
    assert chunk_restored == chunk


def test_video_result_serialization():
    """Test VideoResult JSON serialization."""
    result = VideoResult(
        video_id="video123",
        start_timestamp=0.5,
        end_timestamp=5.0,
        start_token_id=1,
        end_token_id=10,
        transcript_snippet="hello world",
        score=0.95
    )
    
    # Test to_dict
    result_dict = result.to_dict()
    assert result_dict["video_id"] == "video123"
    assert result_dict["score"] == 0.95
    
    # Test to_json
    result_json = result.to_json()
    parsed = json.loads(result_json)
    assert parsed["transcript_snippet"] == "hello world"
    
    # Test from_dict
    result_restored = VideoResult.from_dict(result_dict)
    assert result_restored == result


def test_pdf_result_serialization():
    """Test PDFResult JSON serialization."""
    result = PDFResult(
        pdf_filename="test.pdf",
        page_number=1,
        paragraph_index=0,
        source_snippet="This is a snippet.",
        score=0.85
    )
    
    # Test to_dict
    result_dict = result.to_dict()
    assert result_dict["pdf_filename"] == "test.pdf"
    assert result_dict["score"] == 0.85
    
    # Test to_json
    result_json = result.to_json()
    parsed = json.loads(result_json)
    assert parsed["source_snippet"] == "This is a snippet."
    
    # Test from_dict
    result_restored = PDFResult.from_dict(result_dict)
    assert result_restored == result


def test_video_response_serialization():
    """Test VideoResponse JSON serialization."""
    response = VideoResponse(
        video_id="video123",
        start_timestamp=0.5,
        end_timestamp=5.0,
        start_token_id=1,
        end_token_id=10,
        transcript_snippet="hello world",
        generated_answer="The answer is hello world."
    )
    
    # Test to_dict
    response_dict = response.to_dict()
    assert response_dict["answer_type"] == "video"
    assert response_dict["video_id"] == "video123"
    assert response_dict["generated_answer"] == "The answer is hello world."
    
    # Test to_json
    response_json = response.to_json()
    parsed = json.loads(response_json)
    assert parsed["answer_type"] == "video"
    assert parsed["transcript_snippet"] == "hello world"
    
    # Test from_dict
    response_restored = VideoResponse.from_dict(response_dict)
    assert response_restored.video_id == response.video_id
    assert response_restored.generated_answer == response.generated_answer


def test_pdf_response_serialization():
    """Test PDFResponse JSON serialization."""
    response = PDFResponse(
        pdf_filename="test.pdf",
        page_number=1,
        paragraph_index=0,
        source_snippet="This is a snippet.",
        generated_answer="The answer is in the snippet."
    )
    
    # Test to_dict
    response_dict = response.to_dict()
    assert response_dict["answer_type"] == "pdf"
    assert response_dict["pdf_filename"] == "test.pdf"
    assert response_dict["generated_answer"] == "The answer is in the snippet."
    
    # Test to_json
    response_json = response.to_json()
    parsed = json.loads(response_json)
    assert parsed["answer_type"] == "pdf"
    assert parsed["source_snippet"] == "This is a snippet."
    
    # Test from_dict
    response_restored = PDFResponse.from_dict(response_dict)
    assert response_restored.pdf_filename == response.pdf_filename
    assert response_restored.generated_answer == response.generated_answer


def test_no_answer_response_serialization():
    """Test NoAnswerResponse JSON serialization."""
    response = NoAnswerResponse()
    
    # Test to_dict
    response_dict = response.to_dict()
    assert response_dict["answer_type"] == "no_answer"
    assert response_dict["message"] == "No relevant answer found in the knowledge base."
    
    # Test to_json
    response_json = response.to_json()
    parsed = json.loads(response_json)
    assert parsed["answer_type"] == "no_answer"
    
    # Test from_dict
    response_restored = NoAnswerResponse.from_dict(response_dict)
    assert response_restored == response


def test_video_response_default_values():
    """Test VideoResponse with default values."""
    response = VideoResponse()
    assert response.answer_type == "video"
    assert response.video_id == ""
    assert response.start_timestamp == 0.0
    assert response.generated_answer == ""


def test_pdf_response_default_values():
    """Test PDFResponse with default values."""
    response = PDFResponse()
    assert response.answer_type == "pdf"
    assert response.pdf_filename == ""
    assert response.page_number == 0
    assert response.generated_answer == ""
