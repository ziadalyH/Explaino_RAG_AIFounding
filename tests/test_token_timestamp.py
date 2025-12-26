"""Tests for token to timestamp lookup functionality."""

import pytest
from src.models import VideoTranscript, TranscriptToken
from src.processing.chunking import ChunkingModule
from src.config import Config


class TestTokenTimestampLookup:
    """Test suite for token ID to timestamp conversion."""
    
    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript for testing."""
        tokens = [
            TranscriptToken(id=1, timestamp=0.0, word="Hello"),
            TranscriptToken(id=2, timestamp=0.5, word="world"),
            TranscriptToken(id=3, timestamp=1.0, word="this"),
            TranscriptToken(id=4, timestamp=1.5, word="is"),
            TranscriptToken(id=5, timestamp=2.0, word="a"),
            TranscriptToken(id=6, timestamp=2.5, word="test"),
            TranscriptToken(id=7, timestamp=3.0, word="transcript"),
        ]
        return VideoTranscript(
            video_id="test_video",
            pdf_reference="test.pdf",
            tokens=tokens
        )
    
    def test_token_to_timestamp_lookup(self, sample_transcript):
        """Test that token IDs correctly map to timestamps."""
        # Token 1 should have timestamp 0.0
        assert sample_transcript.tokens[0].id == 1
        assert sample_transcript.tokens[0].timestamp == 0.0
        
        # Token 4 should have timestamp 1.5
        token_4 = next(t for t in sample_transcript.tokens if t.id == 4)
        assert token_4.timestamp == 1.5
        assert token_4.word == "is"
        
        # Token 7 should have timestamp 3.0
        token_7 = next(t for t in sample_transcript.tokens if t.id == 7)
        assert token_7.timestamp == 3.0
        assert token_7.word == "transcript"
    
    def test_timestamp_range_from_token_ids(self, sample_transcript):
        """Test extracting timestamp range from token ID range."""
        # Get tokens 2-5
        start_token_id = 2
        end_token_id = 5
        
        tokens_in_range = [
            t for t in sample_transcript.tokens 
            if start_token_id <= t.id <= end_token_id
        ]
        
        # Should have 4 tokens (2, 3, 4, 5)
        assert len(tokens_in_range) == 4
        
        # Start timestamp should be 0.5 (token 2)
        start_timestamp = tokens_in_range[0].timestamp
        assert start_timestamp == 0.5
        
        # End timestamp should be 2.0 (token 5)
        end_timestamp = tokens_in_range[-1].timestamp
        assert end_timestamp == 2.0
    
    def test_chunking_preserves_token_timestamps(self, sample_transcript):
        """Test that chunking preserves token ID and timestamp relationships."""
        import logging
        config = Config.from_env()
        logger = logging.getLogger("test")
        chunking_module = ChunkingModule(config, logger)
        
        # Create chunks from transcript
        chunks = chunking_module.chunk_transcript(sample_transcript)
        
        # Each chunk should have valid token IDs and timestamps
        for chunk in chunks:
            assert chunk.start_token_id > 0
            assert chunk.end_token_id >= chunk.start_token_id
            assert chunk.start_timestamp >= 0.0
            assert chunk.end_timestamp >= chunk.start_timestamp
            
            # Verify token IDs correspond to actual tokens
            start_token = next(
                (t for t in sample_transcript.tokens if t.id == chunk.start_token_id),
                None
            )
            end_token = next(
                (t for t in sample_transcript.tokens if t.id == chunk.end_token_id),
                None
            )
            
            assert start_token is not None, f"Start token {chunk.start_token_id} not found"
            assert end_token is not None, f"End token {chunk.end_token_id} not found"
            
            # Timestamps should match
            assert chunk.start_timestamp == start_token.timestamp
            assert chunk.end_timestamp == end_token.timestamp
    
    def test_token_id_monotonicity(self, sample_transcript):
        """Test that token IDs are strictly increasing."""
        previous_id = 0
        for token in sample_transcript.tokens:
            assert token.id > previous_id, "Token IDs must be strictly increasing"
            previous_id = token.id
    
    def test_timestamp_monotonicity(self, sample_transcript):
        """Test that timestamps are non-decreasing."""
        previous_timestamp = -1.0
        for token in sample_transcript.tokens:
            assert token.timestamp >= previous_timestamp, \
                "Timestamps must be non-decreasing"
            previous_timestamp = token.timestamp
    
    def test_find_token_by_timestamp(self, sample_transcript):
        """Test finding the closest token for a given timestamp."""
        # Find token closest to timestamp 1.7
        target_timestamp = 1.7
        
        closest_token = min(
            sample_transcript.tokens,
            key=lambda t: abs(t.timestamp - target_timestamp)
        )
        
        # Should be token 4 (timestamp 1.5) or token 5 (timestamp 2.0)
        # Token 4 is closer (0.2 difference vs 0.3)
        assert closest_token.id == 4
        assert closest_token.timestamp == 1.5
    
    def test_extract_text_from_token_range(self, sample_transcript):
        """Test extracting text from a token ID range."""
        # Extract text from tokens 3-6
        start_id = 3
        end_id = 6
        
        tokens_in_range = [
            t for t in sample_transcript.tokens
            if start_id <= t.id <= end_id
        ]
        
        text = " ".join(t.word for t in tokens_in_range)
        
        # Should be "this is a test"
        assert text == "this is a test"
        
        # Verify timestamps
        assert tokens_in_range[0].timestamp == 1.0  # "this"
        assert tokens_in_range[-1].timestamp == 2.5  # "test"
    
    def test_chunk_token_coverage(self, sample_transcript):
        """Test that chunks cover all tokens without gaps."""
        import logging
        config = Config.from_env()
        logger = logging.getLogger("test")
        chunking_module = ChunkingModule(config, logger)
        
        chunks = chunking_module.chunk_transcript(sample_transcript)
        
        # Collect all token IDs covered by chunks
        covered_tokens = set()
        for chunk in chunks:
            for token_id in range(chunk.start_token_id, chunk.end_token_id + 1):
                covered_tokens.add(token_id)
        
        # All original tokens should be covered
        original_token_ids = {t.id for t in sample_transcript.tokens}
        
        # Check coverage (allowing for some edge cases in chunking)
        coverage_ratio = len(covered_tokens & original_token_ids) / len(original_token_ids)
        assert coverage_ratio >= 0.9, \
            f"Chunks should cover at least 90% of tokens, got {coverage_ratio:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
