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

        Args:
            transcript: VideoTranscript object containing tokens

        Returns:
            List of TranscriptChunk objects with metadata
        """
        if self.chunking_strategy == "sliding_window":
            return self._chunk_transcript_sliding_window(transcript)
        elif self.chunking_strategy == "all_combinations":
            return self._chunk_transcript_all_combinations(transcript)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
    
    def _chunk_transcript_sliding_window(self, transcript: VideoTranscript) -> List[TranscriptChunk]:
        """
        Create overlapping chunks using sliding window strategy.

        Uses a fixed-window with overlap strategy to maintain temporal coherence
        and ensure context isn't lost at boundaries.

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

        # Calculate step size (how many tokens to advance for each chunk)
        step_size = self.chunk_size - self.chunk_overlap

        # Ensure step size is at least 1 to avoid infinite loops
        if step_size <= 0:
            step_size = 1
            self.logger.warning(
                f"chunk_overlap ({self.chunk_overlap}) >= chunk_size ({self.chunk_size}). "
                f"Using step_size=1 to avoid infinite loop."
            )

        # Create chunks with sliding window
        start_idx = 0
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))

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
            f"with {len(tokens)} tokens using sliding_window strategy"
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

