# Explaino RAG System - Technical Architecture

## Overview

This document provides an in-depth technical overview of the Explaino RAG system architecture, design decisions, and implementation details.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                  (CLI / REST API / Python SDK)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      RAGSystem Orchestrator                      │
│  • Coordinates all components                                    │
│  • Manages lifecycle (init, index, query, shutdown)             │
│  • Implements resume capability for indexing                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│  Ingestion     │  │   Processing    │  │   Retrieval    │
│  Pipeline      │  │   Pipeline      │  │   Pipeline     │
└───────┬────────┘  └────────┬────────┘  └───────┬────────┘
        │                    │                    │
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│ PDFIngester    │  │ ChunkingModule  │  │ QueryProcessor │
│ • PyMuPDF      │  │ • Paragraph     │  │ • Preprocessing│
│ • Font detect  │  │ • Sentence      │  │ • Embedding    │
│ • Title extract│  │ • Overlap       │  │                │
└────────────────┘  └─────────────────┘  └────────────────┘
                                                  │
┌────────────────┐  ┌─────────────────┐  ┌──────▼─────────┐
│TranscriptIngest│  │ EmbeddingEngine │  │RetrievalEngine │
│ • JSON parse   │  │ • MPNet model   │  │ • Two-tier     │
│ • Timestamp    │  │ • Stop words    │  │ • k-NN search  │
│ • Validation   │  │ • Batch process │  │ • Hybrid search│
└────────────────┘  └─────────────────┘  └────────────────┘
                                                  │
                    ┌─────────────────┐  ┌───────▼────────┐
                    │VectorIndexBuild │  │ResponseGenerat │
                    │ • OpenSearch    │  │ • GPT-4o-mini  │
                    │ • k-NN config   │  │ • Context      │
                    │ • Dual embed    │  │ • Citations    │
                    └────────┬────────┘  └────────────────┘
                             │
                    ┌────────▼────────┐
                    │   OpenSearch    │
                    │ Vector Database │
                    │                 │
                    │ • rag-pdf-index │
                    │ • rag-video-idx │
                    └─────────────────┘
```

## Component Details

### 1. Ingestion Layer

#### PDFIngester

**Purpose**: Extract structured text from PDF documents with semantic understanding

**Technology**: PyMuPDF (fitz)

**Key Features**:

- Font-based title detection (>1.2x average font size)
- Block-level text extraction with metadata
- Filtering of headers/footers (<50 chars)
- Page-by-page processing for memory efficiency

**Algorithm**:

```python
for each page in PDF:
    1. Extract text blocks with font information
    2. Calculate average font size for page
    3. Identify titles (font_size > avg * 1.2)
    4. Filter short blocks (< 50 chars)
    5. Create paragraph objects with:
       - Text content
       - Page number
       - Paragraph index
       - Associated title (if any)
```

**Output**: List[PDFParagraph]

#### TranscriptIngester

**Purpose**: Parse video transcript JSON files with word-level timestamps

**Technology**: Python JSON parser

**Schema Validation**:

```json
{
  "video_id": "string (required, unique)",
  "pdf_reference": "string (required, fallback document)",
  "video_transcripts": [
    {
      "id": "integer (sequential token ID)",
      "timestamp": "float (seconds)",
      "word": "string (spoken word)"
    }
  ]
}
```

**Key Features**:

- Schema validation with clear error messages
- Monotonic timestamp verification
- Sequential token ID validation
- PDF reference tracking for fallback

**Output**: List[Transcript]

### 2. Processing Layer

#### ChunkingModule

**Purpose**: Create optimal-sized semantic chunks for embedding and retrieval

**Strategies**:

**PDF Chunking (Paragraph-Level)**:

```
Parameters:
- target_chunk_size: 512 tokens
- max_chunk_size: 768 tokens
- chunk_overlap: 128 tokens (25%)
- min_paragraph_length: 20 chars

Algorithm:
1. For each paragraph:
   a. Count tokens using tiktoken
   b. If tokens <= max_chunk_size:
      - Create single chunk with title context
   c. If tokens > max_chunk_size:
      - Split using sliding window
      - Overlap = 128 tokens
      - Preserve title with each sub-chunk
2. Maintain paragraph index per page
```

**Video Chunking (Sentence-Based)**:

```
Parameters:
- target_words: 40 words
- min_words: 30 words
- max_words: 50 words

Algorithm:
1. Reconstruct sentences from word list
2. Detect boundaries (. ! ? + pause)
3. Group into 30-50 word chunks
4. Record timestamp ranges
5. Preserve token IDs for highlighting
```

**Output**: List[PDFChunk], List[TranscriptChunk]
