"""Chunking module for segmenting video transcripts and PDF paragraphs."""

from typing import List
import logging

from ..models import VideoTranscript, PDFParagraph, TranscriptChunk, PDFChunk
from ..config import Config


class ChunkingModule:
    """
    Handles chunking of video transcripts and PDF paragraphs into segments
    suitable for embedding and retrieval.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize the ChunkingModule.

        Args:
            config: Configuration object containing chunking parameters
            logger: Logger instance for logging operations
        """
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.chunking_strategy = config.chunking_strategy
        self.max_chunk_window = config.max_chunk_window
        self.logger = logger

    def chunk_transcript(self, transcript: VideoTranscript) -> List[TranscriptChunk]:
        """
        Create chunks from video transcript tokens based on configured strategy.
        
        Automatically adjusts chunk size based on video duration for optimal retrieval:
        - Short videos (< 5 min): Smaller chunks for precision
        - Medium videos (5-20 min): Standard chunks
        - Long videos (> 20 min): Larger chunks for coverage

        Args:
            transcript: VideoTranscript object containing tokens

        Returns:
            List of TranscriptChunk objects with metadata
        """
        # Calculate video duration from tokens
        if not transcript.tokens:
            self.logger.warning(f"Empty transcript for video_id: {transcript.video_id}")
            return []
        
        video_duration = transcript.tokens[-1].timestamp  # Last token's timestamp
        total_tokens = len(transcript.tokens)
        
        # Adaptive chunk sizing based on video duration
        chunk_size, chunk_overlap = self._calculate_adaptive_chunk_size(
            video_duration, 
            total_tokens
        )
        
        self.logger.info(
            f"Video {transcript.video_id}: duration={video_duration:.1f}s, "
            f"tokens={total_tokens}, adaptive_chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )
        
        if self.chunking_strategy == "sliding_window":
            return self._chunk_transcript_sliding_window(
                transcript, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        elif self.chunking_strategy == "all_combinations":
            return self._chunk_transcript_all_combinations(transcript)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
    
    def _calculate_adaptive_chunk_size(self, duration_seconds: float, total_tokens: int) -> tuple:
        """
        Calculate optimal chunk size and overlap based on video duration.
        
        Strategy:
        - Short videos (< 5 min / 300s): 30-50 tokens, 10 overlap
        - Medium videos (5-20 min): 50-100 tokens, 15 overlap  
        - Long videos (20-60 min): 100-150 tokens, 20 overlap
        - Very long videos (> 60 min): 150-200 tokens, 30 overlap
        
        Args:
            duration_seconds: Video duration in seconds
            total_tokens: Total number of tokens in transcript
            
        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        duration_minutes = duration_seconds / 60
        
        if duration_minutes < 5:
            # Short videos: smaller chunks for precision
            chunk_size = min(50, max(30, total_tokens // 3))
            chunk_overlap = 10
            category = "short"
        elif duration_minutes < 20:
            # Medium videos: balanced chunks
            chunk_size = min(100, max(50, total_tokens // 5))
            chunk_overlap = 15
            category = "medium"
        elif duration_minutes < 60:
            # Long videos: larger chunks for coverage
            chunk_size = min(150, max(100, total_tokens // 8))
            chunk_overlap = 20
            category = "long"
        else:
            # Very long videos: largest chunks
            chunk_size = min(200, max(150, total_tokens // 10))
            chunk_overlap = 30
            category = "very long"
        
        # Ensure chunk_size doesn't exceed total tokens
        chunk_size = min(chunk_size, total_tokens)
        
        # Ensure overlap is less than chunk_size
        chunk_overlap = min(chunk_overlap, chunk_size // 2)
        
        self.logger.debug(
            f"Adaptive chunking: {category} video ({duration_minutes:.1f} min) → "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
        
        return chunk_size, chunk_overlap
    
    def _chunk_transcript_sliding_window(
        self, 
        transcript: VideoTranscript,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[TranscriptChunk]:
        """
        Create overlapping chunks using sliding window strategy.

        Uses a fixed-window with overlap strategy to maintain temporal coherence
        and ensure context isn't lost at boundaries.

        Args:
            transcript: VideoTranscript object containing tokens
            chunk_size: Override chunk size (uses config default if None)
            chunk_overlap: Override chunk overlap (uses config default if None)

        Returns:
            List of TranscriptChunk objects with metadata
        """
        chunks = []
        tokens = transcript.tokens

        if not tokens:
            self.logger.warning(f"Empty transcript for video_id: {transcript.video_id}")
            return chunks

        # Use provided values or fall back to config
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        # Calculate step size (how many tokens to advance for each chunk)
        step_size = chunk_size - chunk_overlap

        # Ensure step size is at least 1 to avoid infinite loops
        if step_size <= 0:
            step_size = 1
            self.logger.warning(
                f"chunk_overlap ({chunk_overlap}) >= chunk_size ({chunk_size}). "
                f"Using step_size=1 to avoid infinite loop."
            )

        # Create chunks with sliding window
        start_idx = 0
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + chunk_size, len(tokens))

            # Extract tokens for this chunk
            chunk_tokens = tokens[start_idx:end_idx]

            # Build chunk metadata
            start_token_id = chunk_tokens[0].id
            end_token_id = chunk_tokens[-1].id
            start_timestamp = chunk_tokens[0].timestamp
            end_timestamp = chunk_tokens[-1].timestamp

            # Concatenate words to form text
            text = " ".join(token.word for token in chunk_tokens)

            # Create chunk object
            chunk = TranscriptChunk(
                video_id=transcript.video_id,
                start_token_id=start_token_id,
                end_token_id=end_token_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                text=text
            )

            chunks.append(chunk)

            # Move to next chunk position
            start_idx += step_size

            # If we've reached the end, break
            if end_idx >= len(tokens):
                break

        self.logger.info(
            f"Created {len(chunks)} chunks from transcript {transcript.video_id} "
            f"with {len(tokens)} tokens using sliding_window strategy "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

        return chunks
    
    def _chunk_transcript_all_combinations(self, transcript: VideoTranscript) -> List[TranscriptChunk]:
        """
        Create chunks for all possible token combinations within max window size.
        
        This strategy creates a chunk for every possible token range up to max_chunk_window.
        For example, with max_window=30:
        - Tokens 1-1, 1-2, 1-3, ..., 1-30
        - Tokens 2-2, 2-3, 2-4, ..., 2-31
        - And so on...
        
        This enables precise timestamp extraction for any segment of the video.

        Args:
            transcript: VideoTranscript object containing tokens

        Returns:
            List of TranscriptChunk objects with metadata
        """
        chunks = []
        tokens = transcript.tokens

        if not tokens:
            self.logger.warning(f"Empty transcript for video_id: {transcript.video_id}")
            return chunks

        total_tokens = len(tokens)
        
        self.logger.info(
            f"Creating all combinations for transcript {transcript.video_id} "
            f"with {total_tokens} tokens and max_window={self.max_chunk_window}"
        )
        
        # For each starting position
        for start_idx in range(total_tokens):
            # For each window size from 1 to max_chunk_window
            for window_size in range(1, self.max_chunk_window + 1):
                end_idx = start_idx + window_size
                
                # Stop if we exceed the transcript length
                if end_idx > total_tokens:
                    break
                
                # Extract tokens for this chunk
                chunk_tokens = tokens[start_idx:end_idx]
                
                # Build chunk metadata
                start_token_id = chunk_tokens[0].id
                end_token_id = chunk_tokens[-1].id
                start_timestamp = chunk_tokens[0].timestamp
                end_timestamp = chunk_tokens[-1].timestamp
                
                # Concatenate words to form text
                text = " ".join(token.word for token in chunk_tokens)
                
                # Create chunk object
                chunk = TranscriptChunk(
                    video_id=transcript.video_id,
                    start_token_id=start_token_id,
                    end_token_id=end_token_id,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    text=text
                )
                
                chunks.append(chunk)
        
        self.logger.info(
            f"Created {len(chunks)} chunks from transcript {transcript.video_id} "
            f"with {total_tokens} tokens using all_combinations strategy"
        )

        return chunks

    def chunk_pdf_paragraphs(self, paragraphs: List[PDFParagraph]) -> List[PDFChunk]:
        """
        Create chunks from PDF paragraphs.

        Uses a paragraph-based strategy where each paragraph becomes a chunk
        if it's under the maximum size. Long paragraphs are split into
        sentence-based sub-chunks with overlap. Preserves title information.

        Args:
            paragraphs: List of PDFParagraph objects

        Returns:
            List of PDFChunk objects
        """
        chunks = []

        for paragraph in paragraphs:
            # For simplicity, treat each paragraph as a single chunk
            # In a more sophisticated implementation, we could split long paragraphs
            # into sentence-based sub-chunks with overlap

            # Skip very short paragraphs (likely noise)
            if len(paragraph.text.strip()) < 20:
                self.logger.debug(
                    f"Skipping short paragraph in {paragraph.pdf_filename} "
                    f"page {paragraph.page_number}, paragraph {paragraph.paragraph_index}"
                )
                continue

            # Create chunk from paragraph, preserving title
            chunk = PDFChunk(
                pdf_filename=paragraph.pdf_filename,
                page_number=paragraph.page_number,
                paragraph_index=paragraph.paragraph_index,
                text=paragraph.text,
                title=paragraph.title
            )

            chunks.append(chunk)

        self.logger.info(f"Created {len(chunks)} chunks from {len(paragraphs)} PDF paragraphs")

        return chunks

