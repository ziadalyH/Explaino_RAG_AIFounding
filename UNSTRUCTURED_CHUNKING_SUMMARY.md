# Unstructured Chunking Implementation Summary

## Overview

Implemented Unstructured library's chunking by title to improve PDF retrieval quality by preserving document structure and context.

## Key Changes

### 1. **PDF Ingestion with Unstructured** (`src/ingestion/pdf_ingester.py`)

- Replaced PyPDF2 with Unstructured library
- Uses `partition_pdf()` to extract document elements
- Uses `chunk_by_title()` to keep headings with their content
- Extracts titles/headings for each chunk
- Heuristic: Treats ALL CAPS first lines as titles

### 2. **Title Field Added to Models** (`src/models.py`)

- Added `title: Optional[str]` field to:
  - `PDFParagraph`
  - `PDFChunk`
  - `PDFResult`
  - `PDFResponse`

### 3. **Embedding Strategy** (`src/processing/indexing.py`)

- **Embeds title + text together** for better semantic search
- Format: `"{title}\n\n{text}"` when title exists
- Stores title separately in OpenSearch for display
- Added `title` field to index mapping

### 4. **Chunking Module** (`src/processing/chunking.py`)

- Updated to preserve title field when creating PDFChunk objects

### 5. **Retrieval Engine** (`src/retrieval/retrieval_engine.py`)

- Extracts title field from OpenSearch results
- Passes title through to PDFResult objects

### 6. **Response Generator** (`src/retrieval/response_generator.py`)

- Displays section title in formatted output
- Passes title through to PDFResponse

### 7. **Dependencies** (`requirements.txt`)

- Added `unstructured[pdf]==0.11.8`
- Added `pillow==10.2.0`
- Added `pillow-heif==0.15.0`

## Benefits

### Better Context Preservation

- Headings stay with their content
- "OPENSTAX" heading now grouped with description
- Reduces fragmentation of related information

### Improved Semantic Search

- Embedding title + text provides richer context
- Query "What is OpenStax?" will match chunks containing both the heading and description
- Better ranking for definitional queries

### Enhanced User Experience

- Users see the section title in results
- Easier to understand context of retrieved information

## Reindexing Process

Use the new script to reindex PDFs:

```bash
docker exec rag-backend python reindex_pdfs_with_unstructured.py
```

This will:

1. Delete existing PDF documents (preserves video data)
2. Re-ingest PDFs with Unstructured chunking
3. Extract titles from chunks
4. Embed title + text together
5. Reindex with new structure

## Example

**Before (PyPDF2):**

- Chunk 1: "OPENSTAX"
- Chunk 2: "OpenStax provides free, peer-reviewed..."

**After (Unstructured):**

- Chunk 1: Title="OPENSTAX", Text="OpenStax provides free, peer-reviewed..."
- Embedding: "OPENSTAX\n\nOpenStax provides free, peer-reviewed..."

## Next Steps

1. Rebuild Docker image: `docker-compose build rag-backend`
2. Recreate container: `docker-compose up -d rag-backend`
3. Run reindex script: `docker exec rag-backend python reindex_pdfs_with_unstructured.py`
4. Test query: `docker exec rag-backend python main.py query --question "What is OpenStax?"`
