"""Data models for the RAG Chatbot Backend."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class TranscriptToken:
    """Represents a single token in a video transcript."""
    id: int
    timestamp: float
    word: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptToken':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VideoTranscript:
    """Represents a complete video transcript."""
    video_id: str
    pdf_reference: str
    tokens: List[TranscriptToken]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'video_id': self.video_id,
            'pdf_reference': self.pdf_reference,
            'tokens': [token.to_dict() for token in self.tokens]
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoTranscript':
        """Create from dictionary."""
        return cls(
            video_id=data['video_id'],
            pdf_reference=data['pdf_reference'],
            tokens=[TranscriptToken.from_dict(t) for t in data['tokens']]
        )


@dataclass
class PDFParagraph:
    """Represents a paragraph extracted from a PDF."""
    pdf_filename: str
    page_number: int
    paragraph_index: int
    text: str
    title: Optional[str] = None  # Section title/heading for this chunk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PDFParagraph':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TranscriptChunk:
    """Represents a chunk of video transcript for embedding."""
    video_id: str
    start_token_id: int
    end_token_id: int
    start_timestamp: float
    end_timestamp: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptChunk':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PDFChunk:
    """Represents a chunk of PDF content for embedding."""
    pdf_filename: str
    page_number: int
    paragraph_index: int
    text: str
    title: Optional[str] = None  # Section title/heading for this chunk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PDFChunk':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VideoResult:
    """Represents a video-based retrieval result."""
    video_id: str
    start_timestamp: float
    end_timestamp: float
    start_token_id: int
    end_token_id: int
    transcript_snippet: str
    score: float
    document_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PDFResult:
    """Represents a PDF-based retrieval result."""
    pdf_filename: str
    page_number: int
    paragraph_index: int
    source_snippet: str
    score: float
    document_id: str = ""
    title: Optional[str] = None  # Section title/heading for this result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PDFResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VideoResponse:
    """Represents a video-based answer response."""
    answer_type: str = "video"
    video_id: str = ""
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    start_token_id: int = 0
    end_token_id: int = 0
    transcript_snippet: str = ""
    generated_answer: str = ""
    score: float = 0.0
    document_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoResponse':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PDFResponse:
    """Represents a PDF-based answer response."""
    answer_type: str = "pdf"
    pdf_filename: str = ""
    page_number: int = 0
    paragraph_index: int = 0
    source_snippet: str = ""
    generated_answer: str = ""
    score: float = 0.0
    document_id: str = ""
    title: Optional[str] = None  # Section title/heading for this response

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PDFResponse':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class NoAnswerResponse:
    """Represents a no-answer response."""
    answer_type: str = "no_answer"
    message: str = "No relevant answer found in the knowledge base."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoAnswerResponse':
        """Create from dictionary."""
        return cls(**data)
