"""Video transcript ingestion module."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..models import VideoTranscript, TranscriptToken
from ..config import Config


class TranscriptIngester:
    """Ingests and validates video transcript JSON files."""
    
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """
        Initialize the TranscriptIngester.
        
        Args:
            config: Configuration object containing transcript directory path
            logger: Logger instance for logging operations
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.transcript_dir = config.transcript_dir
    
    def ingest_directory(self, directory: Optional[Path] = None) -> List[VideoTranscript]:
        """
        Ingest all JSON files from the specified directory.
        
        Args:
            directory: Directory path to ingest from. If None, uses config.transcript_dir
            
        Returns:
            List of successfully parsed VideoTranscript objects
        """
        target_dir = directory or self.transcript_dir
        
        if not target_dir.exists():
            self.logger.error(f"Transcript directory does not exist: {target_dir}")
            return []
        
        if not target_dir.is_dir():
            self.logger.error(f"Transcript path is not a directory: {target_dir}")
            return []
        
        transcripts = []
        json_files = list(target_dir.glob("*.json"))
        
        self.logger.info(f"Found {len(json_files)} JSON files in {target_dir}")
        
        for json_file in json_files:
            transcript = self.ingest_file(json_file)
            if transcript:
                transcripts.append(transcript)
        
        self.logger.info(f"Successfully ingested {len(transcripts)} out of {len(json_files)} transcript files")
        return transcripts
    
    def ingest_file(self, file_path: Path) -> Optional[VideoTranscript]:
        """
        Ingest a single JSON transcript file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            VideoTranscript object if successful, None if parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate the transcript structure
            if not self.validate_transcript(data):
                self.logger.error(f"Validation failed for file: {file_path}")
                return None
            
            # Extract required fields
            video_id = data.get('video_id')
            pdf_reference = data.get('pdf_reference')
            video_transcripts = data.get('video_transcripts', [])
            
            # Parse tokens
            tokens = []
            for token_data in video_transcripts:
                if not self.validate_token(token_data):
                    self.logger.error(f"Invalid token in file {file_path}: {token_data}")
                    return None
                
                token = TranscriptToken(
                    id=token_data['id'],
                    timestamp=token_data['timestamp'],
                    word=token_data['word']
                )
                tokens.append(token)
            
            # Create VideoTranscript object
            transcript = VideoTranscript(
                video_id=video_id,
                pdf_reference=pdf_reference,
                tokens=tokens
            )
            
            self.logger.info(f"Successfully ingested transcript from {file_path.name} with {len(tokens)} tokens")
            return transcript
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in file {file_path}: {e}")
            return None
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error ingesting file {file_path}: {e}")
            return None
    
    def validate_transcript(self, data: Dict[str, Any]) -> bool:
        """
        Validate the structure of a transcript JSON object.
        
        Args:
            data: Dictionary containing transcript data
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields exist
        if 'video_id' not in data:
            self.logger.error("Missing required field: video_id")
            return False
        
        if 'pdf_reference' not in data:
            self.logger.error("Missing required field: pdf_reference")
            return False
        
        if 'video_transcripts' not in data:
            self.logger.error("Missing required field: video_transcripts")
            return False
        
        # Check video_transcripts is a list
        if not isinstance(data['video_transcripts'], list):
            self.logger.error("video_transcripts must be a list")
            return False
        
        # Validate each token
        video_transcripts = data['video_transcripts']
        if len(video_transcripts) == 0:
            self.logger.warning("video_transcripts is empty")
            return True  # Empty is valid, just warn
        
        # Check that token IDs are strictly increasing
        prev_id = None
        for token_data in video_transcripts:
            if not self.validate_token(token_data):
                return False
            
            current_id = token_data['id']
            if prev_id is not None and current_id <= prev_id:
                self.logger.error(f"Token IDs are not strictly increasing: {prev_id} -> {current_id}")
                return False
            prev_id = current_id
        
        return True
    
    def validate_token(self, token_data: Dict[str, Any]) -> bool:
        """
        Validate a single token's structure and data types.
        
        Args:
            token_data: Dictionary containing token data
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields exist
        if 'id' not in token_data:
            self.logger.error("Token missing required field: id")
            return False
        
        if 'timestamp' not in token_data:
            self.logger.error("Token missing required field: timestamp")
            return False
        
        if 'word' not in token_data:
            self.logger.error("Token missing required field: word")
            return False
        
        # Validate id is an integer
        if not isinstance(token_data['id'], int):
            self.logger.error(f"Token id must be an integer, got {type(token_data['id'])}")
            return False
        
        # Validate timestamp is a number (int or float) and non-negative
        timestamp = token_data['timestamp']
        if not isinstance(timestamp, (int, float)):
            self.logger.error(f"Token timestamp must be a number, got {type(timestamp)}")
            return False
        
        if timestamp < 0:
            self.logger.error(f"Token timestamp must be non-negative, got {timestamp}")
            return False
        
        # Validate word is a string and non-empty
        word = token_data['word']
        if not isinstance(word, str):
            self.logger.error(f"Token word must be a string, got {type(word)}")
            return False
        
        if len(word) == 0:
            self.logger.error("Token word cannot be empty")
            return False
        
        return True
