# Data Flow Guide: From Files to Answers

This guide explains the complete data flow in the Explaino RAG system, showing how documents are processed and how queries find answers in the latent space.

## 🔄 Two-Way Pipeline Overview

The system has **two parallel pipelines** that meet in the **latent space** (vector embeddings):

```
📁 INDEXING PIPELINE (Offline)          🔍 QUERY PIPELINE (Real-time)
Documents → Embeddings                   Question → Embedding
        ↓                                        ↓
    Vector DB ←─────── LATENT SPACE ──────→ Search
                    (768-dimensional)
```

---

## 📥 Pipeline 1: Indexing Pipeline (Offline)

This pipeline processes your documents and stores them as vectors.

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Data Ingestion                                          │
└─────────────────────────────────────────────────────────────────┘

📁 data/transcripts/          📁 data/pdfs/
    video_001.json                document.pdf
    video_002.json                guide.pdf
         ↓                             ↓
    [Transcript Ingester]        [PDF Ingester]
         ↓                             ↓
    VideoTranscript              PDFParagraph
    objects                      objects


┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Text Chunking                                           │
└─────────────────────────────────────────────────────────────────┘

VideoTranscript                PDFParagraph
    ↓                             ↓
[Chunking Module]             [Chunking Module]
    ↓                             ↓
TranscriptChunk               PDFChunk
(100 tokens each)             (100 tokens each)
with overlap                  with overlap

Example Video Chunk:
{
  "video_id": "v001",
  "start_timestamp": 10.5,
  "end_timestamp": 25.3,
  "text": "Machine learning is a subset of AI..."
}

Example PDF Chunk:
{
  "pdf_filename": "guide.pdf",
  "page_number": 5,
  "paragraph_index": 2,
  "text": "Neural networks consist of layers..."
}


┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Embedding Generation (ENTERING LATENT SPACE)            │
└─────────────────────────────────────────────────────────────────┘

Text Chunks
    ↓
[Embedding Engine]
(sentence-transformers/all-mpnet-base-v2)
    ↓
768-dimensional vectors

Example:
Text: "Machine learning is a subset of AI..."
    ↓
Vector: [0.234, -0.567, 0.891, ..., 0.123]  (768 numbers)
        └─────────────────────────────────┘
              LATENT SPACE
        (Semantic meaning encoded)


┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Vector Storage                                          │
└─────────────────────────────────────────────────────────────────┘

Embeddings + Metadata
    ↓
[OpenSearch Indexer]
    ↓
┌──────────────────────────────────────┐
│ OpenSearch Vector Database           │
│                                      │
│ rag-video-index:                    │
│   - embedding: [0.234, -0.567, ...] │
│   - video_id: "v001"                │
│   - start_timestamp: 10.5           │
│   - text: "Machine learning..."     │
│                                      │
│ rag-pdf-index:                      │
│   - embedding: [0.891, 0.234, ...]  │
│   - pdf_filename: "guide.pdf"       │
│   - page_number: 5                  │
│   - text: "Neural networks..."      │
└──────────────────────────────────────┘
```

---

## 🔍 Pipeline 2: Query Pipeline (Real-time)

This pipeline processes user questions and finds relevant documents.

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Query Input                                             │
└─────────────────────────────────────────────────────────────────┘

User Question:
"What is machine learning?"
    ↓
[Query Processor]


┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Query Embedding (ENTERING LATENT SPACE)                 │
└─────────────────────────────────────────────────────────────────┘

Question Text
    ↓
[Same Embedding Engine]
(sentence-transformers/all-mpnet-base-v2)
    ↓
768-dimensional query vector

Example:
Question: "What is machine learning?"
    ↓
Query Vector: [0.245, -0.543, 0.876, ..., 0.134]  (768 numbers)
              └─────────────────────────────────┘
                    LATENT SPACE
              (Same space as documents!)


┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Vector Search (MEETING IN LATENT SPACE)                 │
└─────────────────────────────────────────────────────────────────┘

Query Vector: [0.245, -0.543, 0.876, ...]
    ↓
[OpenSearch k-NN Search]
    ↓
Compares query vector with ALL document vectors
using cosine similarity:

similarity = (query_vector · doc_vector) / (||query|| × ||doc||)

Result: Similarity scores (0.0 to 1.0)

Example Results:
┌────────────────────────────────────────────────┐
│ Document                    | Similarity Score │
├────────────────────────────────────────────────┤
│ "Machine learning is..."    | 0.92 ✅ (High!)  │
│ "Neural networks consist..." | 0.78 ✅          │
│ "The weather today is..."   | 0.23 ❌ (Low)    │
└────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Retrieval & Ranking                                     │
└─────────────────────────────────────────────────────────────────┘

[Retrieval Engine]
    ↓
1. Search rag-video-index (Tier 1)
   - Get top 5 results above threshold (0.5)
   - If found → proceed to LLM
   - If not found or LLM refuses → Tier 2

2. Search rag-pdf-index (Tier 2)
   - Get top 5 results above threshold (0.5)
   - If found → proceed to LLM
   - If not found or LLM refuses → Tier 3

3. Return NoAnswer with knowledge summary (Tier 3)


┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: LLM Answer Generation                                   │
└─────────────────────────────────────────────────────────────────┘

Retrieved Context + Question
    ↓
[Response Generator]
    ↓
[OpenSearch RAG Pipeline]
    ↓
[LLM (via OpenSearch ML Connector)]
    ↓
Generated Answer

Example:
Context: "Machine learning is a subset of AI that enables
          computers to learn from data..."
Question: "What is machine learning?"
    ↓
Answer: "Machine learning is a subset of artificial
         intelligence that allows computers to learn
         and improve from experience without being
         explicitly programmed."
```

---

## 🎯 The Latent Space: Where Pipelines Meet

### What is Latent Space?

The **latent space** is a 768-dimensional mathematical space where:

- Each dimension represents a learned semantic feature
- Similar meanings are close together
- Different meanings are far apart

```
Latent Space Visualization (simplified to 2D):

                    "AI"
                     ●
                    / \
                   /   \
    "Machine Learning" "Deep Learning"
           ●               ●
           |               |
           |               |
    "Neural Networks"  "Transformers"
           ●               ●


    Far away:
    "Weather" ●                    ● "Cooking"
```

### How Documents and Queries Meet

```
┌─────────────────────────────────────────────────────────────────┐
│                        LATENT SPACE                              │
│                     (768 dimensions)                             │
│                                                                  │
│  Document Vectors              Query Vector                     │
│  (from indexing)               (from query)                     │
│                                                                  │
│     Doc1: [0.234, -0.567, ...]                                 │
│     Doc2: [0.891, 0.234, ...]                                  │
│     Doc3: [0.123, -0.789, ...]                                 │
│                                                                  │
│                    ↓                                             │
│                                                                  │
│              COSINE SIMILARITY                                   │
│         (How close are they?)                                    │
│                                                                  │
│                    ↓                                             │
│                                                                  │
│     Query: [0.245, -0.543, ...]                                │
│                                                                  │
│     Similarity Scores:                                           │
│     Doc1 ↔ Query: 0.92 ✅ (Very similar!)                      │
│     Doc2 ↔ Query: 0.78 ✅ (Similar)                            │
│     Doc3 ↔ Query: 0.23 ❌ (Not similar)                        │
│                                                                  │
│     Return: Doc1, Doc2 (above threshold 0.5)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Same Embedding Model**: Both pipelines use the same model

   - Documents: `all-mpnet-base-v2`
   - Queries: `all-mpnet-base-v2` (same!)

2. **Same Vector Space**: Both produce 768-dimensional vectors

   - Documents: 768 numbers
   - Queries: 768 numbers (same space!)

3. **Semantic Similarity**: Similar meanings → similar vectors
   - "What is ML?" ≈ "Machine learning is..."
   - Even with different words!

---

## 📊 Complete End-to-End Example

### Scenario: User asks "What is machine learning?"

```
┌─────────────────────────────────────────────────────────────────┐
│ INDEXING PIPELINE (Already completed)                           │
└─────────────────────────────────────────────────────────────────┘

1. Document: "Machine learning is a subset of AI that enables
              computers to learn from data without explicit
              programming."

2. Chunked: Same (fits in one chunk)

3. Embedded: [0.234, -0.567, 0.891, ..., 0.123] (768 dims)

4. Stored in OpenSearch:
   {
     "embedding": [0.234, -0.567, ...],
     "text": "Machine learning is...",
     "video_id": "intro_to_ml",
     "start_timestamp": 15.2
   }


┌─────────────────────────────────────────────────────────────────┐
│ QUERY PIPELINE (Real-time)                                      │
└─────────────────────────────────────────────────────────────────┘

1. User Question: "What is machine learning?"

2. Embedded: [0.245, -0.543, 0.876, ..., 0.134] (768 dims)
   (Very similar to document vector!)

3. k-NN Search in OpenSearch:
   - Compare query vector with all document vectors
   - Calculate cosine similarity
   - Find: Document has 0.92 similarity ✅

4. Retrieve Context:
   {
     "text": "Machine learning is a subset of AI...",
     "video_id": "intro_to_ml",
     "start_timestamp": 15.2,
     "score": 0.92
   }

5. Generate Answer with LLM:
   Input to LLM:
   - Context: "Machine learning is a subset of AI..."
   - Question: "What is machine learning?"

   Output from LLM:
   "Machine learning is a subset of artificial intelligence
    that enables computers to learn and improve from experience
    without being explicitly programmed."

6. Return Response:
   {
     "answer_type": "video",
     "video_id": "intro_to_ml",
     "start_timestamp": 15.2,
     "generated_answer": "Machine learning is...",
     "score": 0.92
   }
```

---

## 🔑 Key Concepts

### 1. Embedding Model

- **Purpose**: Converts text to vectors
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Output**: 768-dimensional vectors
- **Property**: Similar meanings → similar vectors

### 2. Latent Space

- **Definition**: Mathematical space where vectors live
- **Dimensions**: 768 (each represents a learned feature)
- **Property**: Semantic similarity = geometric proximity

### 3. Cosine Similarity

- **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Range**: 0.0 (completely different) to 1.0 (identical)
- **Threshold**: 0.5 (configurable via `RELEVANCE_THRESHOLD`)

### 4. k-NN Search

- **Algorithm**: HNSW (Hierarchical Navigable Small World)
- **Purpose**: Find k nearest neighbors in vector space
- **Speed**: Sub-linear time complexity
- **Result**: Top-k most similar documents

---

## 🎨 Visual Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                                 │
└──────────────────────────────────────────────────────────────────────┘

INDEXING (Offline)                    QUERY (Real-time)
═══════════════════                   ═══════════════════

📁 Files                              ❓ Question
  ↓                                     ↓
📝 Parse                              🔤 Text
  ↓                                     ↓
✂️  Chunk
  ↓
🧮 Embed ──────────┐                  🧮 Embed
  ↓                │                    ↓
💾 Store           │                    │
                   │                    │
                   ↓                    ↓
              ┌─────────────────────────────┐
              │     LATENT SPACE            │
              │   (768 dimensions)          │
              │                             │
              │  Document Vectors           │
              │        ↕                    │
              │  Query Vector               │
              │        ↓                    │
              │  Cosine Similarity          │
              │        ↓                    │
              │  Top-k Results              │
              └─────────────────────────────┘
                         ↓
                    🔍 Retrieve
                         ↓
                    🤖 LLM Generate
                         ↓
                    ✅ Answer
```

---

## 📈 Performance Characteristics

### Indexing Pipeline

- **Speed**: ~230 documents/second
- **Embedding**: ~90 embeddings/second (CPU)
- **Storage**: ~1KB per document (vector + metadata)
- **Time**: One-time cost (or when adding new data)

### Query Pipeline

- **Embedding**: ~10ms (single query)
- **k-NN Search**: ~50-100ms (depends on index size)
- **LLM Generation**: ~1-2 seconds
- **Total**: < 2 seconds end-to-end

---

## 🔧 Configuration

### Embedding Settings

```bash
# config/.env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
EMBEDDING_PROVIDER=local
```

### Retrieval Settings

```bash
# Minimum similarity score to consider relevant
RELEVANCE_THRESHOLD=0.5

# Number of results to retrieve
MAX_RESULTS=5
```

### Chunking Settings

```bash
# Tokens per chunk
CHUNK_SIZE=100

# Overlap between chunks
CHUNK_OVERLAP=20
```

---

## 🎯 Why This Architecture Works

1. **Semantic Understanding**: Embeddings capture meaning, not just keywords
2. **Fast Retrieval**: Vector search is much faster than full-text search
3. **Scalability**: Can handle millions of documents
4. **Flexibility**: Works with any text (videos, PDFs, web pages)
5. **Accuracy**: LLM generates natural answers from retrieved context

---
