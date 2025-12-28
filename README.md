# Explaino RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions from video transcripts and PDF documents using semantic search and LLM-powered responses.

## 🚀 Quick Start

Get up and running in 3 steps:

```bash
# 1. Clone and navigate
git clone https://github.com/ziadalyH/Explaino_RAG-based-chatbot.git
cd Explaino_RAG-based-chatbot

# 2. Add your OpenAI API key
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=your_key_here

# 3. Start with your preferred mode
```

### Choose Your Mode

The system supports two modes in one docker-compose file:

#### CLI Mode (Default)

For command-line interactions and scripting:

```bash
# Start CLI mode
docker-compose --profile cli up -d

# ⏳ IMPORTANT: Wait for indexing to complete before querying!
# Check the logs to see indexing progress:
docker-compose logs -f rag-backend-cli

# Look for these messages in the logs:
# - "✓ Generated X embeddings" (embedding generation)
# - "✓ Index 'rag-pdf-index' refreshed and ready for search"
# - "✓ Index 'rag-video-index' refreshed and ready for search"
# - "Index building completed successfully"

# Once indexing is complete (usually 2-3 minutes), query the system:
docker-compose exec rag-backend-cli python main.py query -q "What is a database?"

# Index management
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

#### API Mode

For REST API access (ideal for frontend integration):

```bash
# Start API mode
docker-compose --profile api up -d

# ⏳ IMPORTANT: Wait for indexing to complete before querying!
# Check the logs to see indexing progress:
docker-compose logs -f rag-backend-api

# Look for these messages in the logs:
# - "✓ Generated X embeddings" (embedding generation)
# - "✓ Index 'rag-pdf-index' refreshed and ready for search"
# - "✓ Index 'rag-video-index' refreshed and ready for search"
# - "Index building completed successfully"

# Once indexing is complete (usually 2-3 minutes), the API is ready at http://localhost:8000
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a database?"}'
```

#### Both Modes Together

```bash
# Start both CLI and API
docker-compose --profile cli --profile api up -d

# ⏳ IMPORTANT: Wait for indexing to complete before querying!
# Monitor both services:
docker-compose logs -f rag-backend-cli rag-backend-api

# Or check specific service:
docker-compose logs -f rag-backend-cli

# Once you see "Index building completed successfully" in the logs:

# Use CLI
docker-compose exec rag-backend-cli python main.py query -q "test"

# Use API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
```

**💡 Pro Tip:** To check if indexing is complete without watching logs:

```bash
# Check index status via CLI
docker-compose exec rag-backend-cli python main.py status

# Or via API
curl http://localhost:8000/index/status
```

The system will:

- Download the MPNet embedding model (~420MB, first time only)
- Start OpenSearch vector database
- Automatically index sample data (6 PDFs + 8 video transcripts)
- Be ready to answer questions in ~2-3 minutes

## 📋 What's Included

The repository includes sample data for testing:

**Video Transcripts** (`data/transcripts/`):

- Database fundamentals
- Machine learning intro
- Python programming basics
- Cloud computing overview
- Web development trends
- Deep learning tutorial (15min)
- Data science tutorial (20min)
- Python advanced tutorial (30min)

**PDF Documents** (`data/pdfs/`):

- Database systems textbook
- Machine learning fundamentals
- Python programming guide
- Cloud infrastructure guide
- Modern web development
- Principles of Data Science (32MB)

## 🎯 Key Features

- **Two-Tier Retrieval**: Searches videos first, falls back to PDFs
- **Precise Citations**: Exact timestamps for videos, page/paragraph for PDFs
- **Dual Indices**: Separate `rag-pdf-index` and `rag-video-index`
- **Flexible Embeddings**: Choose any sentence-transformers model via `.env` - no Docker rebuild needed!
- **Flexible Data Paths**: Use custom data directories via `.env` - no docker-compose.yml editing needed!
- **Local Embeddings**: Uses MPNet (768-dim) by default - no API costs
- **Pure k-NN Search**: Consistent vector similarity scoring (0.0 to 1.0) for both videos and PDFs
- **Auto-Indexing**: Data indexed automatically on startup
- **Resume Capability**: Only processes new files on re-indexing

> 💡 **New Features!**
>
> - Change embedding models: Edit `.env` and restart - see [QUICK_MODEL_CHANGE.md](QUICK_MODEL_CHANGE.md)
> - Change data paths: Edit `.env` and restart - see [DYNAMIC_PATHS.md](DYNAMIC_PATHS.md)

## 🏗️ Architecture

```
User Query
    ↓
Query Processor (MPNet embeddings)
    ↓
Two-Tier Retrieval Engine
    ├─→ Video Index (k-NN search)
    └─→ PDF Index (k-NN search)
    ↓
Response Generator (GPT-4o-mini)
    ↓
Structured Answer with Citations
```

## 📊 Adding Your Own Data

### Video Transcripts

Place JSON files in `data/transcripts/`:

```json
{
  "video_id": "your_video_id",
  "pdf_reference": "related_document.pdf",
  "video_transcripts": [
    {
      "id": 1,
      "timestamp": 0.0,
      "word": "Hello"
    }
  ]
}
```

### PDF Documents

Place PDF files in `data/pdfs/`:

- Must contain extractable text (not scanned images)
- Filename should match `pdf_reference` in video transcripts

### Re-index Data

```bash
# Index new files (only processes new/modified files)
docker-compose exec rag-backend python main.py index

# Force rebuild entire index
docker-compose exec rag-backend python main.py index --rebuild

# Reindex only videos (force rebuild)
docker-compose exec rag-backend python main.py index --force-rebuild --videos-only

# Reindex only PDFs (force rebuild)
docker-compose exec rag-backend python main.py index --force-rebuild --pdfs-only

# Add new videos (incremental)
docker-compose exec rag-backend python main.py index --videos-only

# Add new PDFs (incremental)
docker-compose exec rag-backend python main.py index --pdfs-only

# See all options
docker-compose exec rag-backend python main.py index --help
```

## ⚙️ Configuration

All configuration is done via the `.env` file. Key settings:

```bash
# Required
OPENAI_API_KEY=your_key_here

# OpenSearch (defaults work with docker-compose)
OPENSEARCH_HOST=opensearch-node1
OPENSEARCH_PORT=9200

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768

# LLM Configuration
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500

# Retrieval
RELEVANCE_THRESHOLD=0.5  # Lower = more results (0.3-0.7 recommended for k-NN scoring)
MAX_RESULTS=5

# Auto-indexing
AUTO_INDEX_ON_STARTUP=true
```

### Changing Configuration

**No rebuild needed!** Just edit `.env` and restart:

```bash
# Edit configuration
nano .env

# Restart to apply changes
docker-compose restart
```

Changes take effect immediately on restart.

### Changing Embedding Models

**New!** You can now change embedding models without rebuilding Docker:

```bash
# 1. Edit .env file
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# 2. Restart container (no rebuild!)
docker-compose restart

# 3. Reindex with new model
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

**Popular Models:**

- `all-MiniLM-L6-v2` (384-dim) - Fast, good for development
- `all-mpnet-base-v2` (768-dim) - **Default**, balanced quality/speed
- `all-roberta-large-v1` (1024-dim) - Highest quality, slower

**📖 Detailed Guides:**

- [QUICK_MODEL_CHANGE.md](QUICK_MODEL_CHANGE.md) - Quick reference for changing models
- [MODEL_OPTIONS.md](MODEL_OPTIONS.md) - Complete list of available models
- [MODEL_COMPARISON_CHART.md](MODEL_COMPARISON_CHART.md) - Visual comparison and decision guide

### Changing Data Paths

**New!** You can now use custom data directories without editing docker-compose.yml:

```bash
# 1. Create your custom directories
mkdir -p my_data/pdfs my_data/transcripts

# 2. Copy your files
cp data/pdfs/* my_data/pdfs/
cp data/transcripts/* my_data/transcripts/

# 3. Edit .env file
PDF_DIR=my_data/pdfs
TRANSCRIPT_DIR=my_data/transcripts

# 4. Restart container (no rebuild!)
docker-compose restart

# 5. Reindex with new data
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

**📖 Detailed Guide:**

- [DYNAMIC_PATHS.md](DYNAMIC_PATHS.md) - Complete guide for custom data paths

## 🔧 Advanced Usage

### Selective Reindexing

When you update only videos or only PDFs, you can reindex just that content type:

**Scenario 1: Updated Video Transcripts**

```bash
# 1. Update video JSON files in data/transcripts/
cp updated_video.json data/transcripts/

# 2. Reindex only videos (faster than full rebuild)
docker-compose exec rag-backend python main.py index --force-rebuild --videos-only

# 3. Query to test
docker-compose exec rag-backend python main.py query -q "Question about updated video"
```

**Scenario 2: Updated PDF Documents**

```bash
# 1. Update PDF files in data/pdfs/
cp updated_document.pdf data/pdfs/

# 2. Reindex only PDFs (faster than full rebuild)
docker-compose exec rag-backend python main.py index --force-rebuild --pdfs-only

# 3. Query to test
docker-compose exec rag-backend python main.py query -q "Question about updated PDF"
```

**Scenario 3: Adding New Content**

```bash
# Add new videos only (incremental - doesn't rebuild existing)
docker-compose exec rag-backend python main.py index --videos-only

# Add new PDFs only (incremental - doesn't rebuild existing)
docker-compose exec rag-backend python main.py index --pdfs-only
```

**Performance Comparison:**

| Command                               | Time (example) | Use Case                   |
| ------------------------------------- | -------------- | -------------------------- |
| `index`                               | ~30s           | Add new files (both types) |
| `index --videos-only`                 | ~10s           | Add/update videos only     |
| `index --pdfs-only`                   | ~20s           | Add/update PDFs only       |
| `index --force-rebuild`               | ~60s           | Rebuild everything         |
| `index --force-rebuild --videos-only` | ~15s           | Rebuild videos only        |
| `index --force-rebuild --pdfs-only`   | ~35s           | Rebuild PDFs only          |

### CLI Commands

```bash
# Query the system
docker-compose exec rag-backend-cli python main.py query --question "Your question"

# Build/rebuild index
docker-compose exec rag-backend-cli python main.py index
docker-compose exec rag-backend-cli python main.py index --rebuild

# Check system status
docker-compose exec rag-backend-cli python main.py status

# Generate knowledge summary
docker-compose exec rag-backend-cli python main.py summarize
```

**Selective Reindexing:**

```bash
# Reindex only videos (useful when updating video transcripts)
docker-compose exec rag-backend-cli python main.py index --force-rebuild --videos-only

# Reindex only PDFs (useful when updating PDF documents)
docker-compose exec rag-backend-cli python main.py index --force-rebuild --pdfs-only

# Incremental indexing for specific type
docker-compose exec rag-backend-cli python main.py index --videos-only  # Only new videos
docker-compose exec rag-backend-cli python main.py index --pdfs-only    # Only new PDFs
```

**Why use selective reindexing?**

- ⚡ **Faster** - Only processes one content type
- 🎯 **Targeted** - Update specific content without affecting the other
- 💾 **Efficient** - Saves time when you only changed videos or PDFs

See [CLI_COMMANDS.md](CLI_COMMANDS.md) for complete command reference.

### API Server

The API mode runs a REST API server on port 8000:

```bash
# Start API mode
docker-compose --profile api up -d

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a database?"}'

# Check index status
curl http://localhost:8000/index/status

# Build index via API
curl -X POST http://localhost:8000/index/build \
  -H "Content-Type: application/json" \
  -d '{"force_rebuild": true}'

# Get knowledge summary
curl http://localhost:8000/knowledge/summary
```

**API Endpoints:**

- `GET /health` - Health check
- `POST /query` - Ask a question
- `GET /index/status` - Check index status
- `POST /index/build` - Build/rebuild index
- `GET /knowledge/summary` - Get knowledge summary and suggested questions

### Local Development (Without Docker)

```bash
# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Start OpenSearch separately
docker run -d -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:latest

# Set environment
export OPENSEARCH_HOST=localhost
export OPENAI_API_KEY=your_key_here

# Run commands
python main.py index
python main.py query --question "What is a database?"
```

## 🐛 Troubleshooting

### Indexing Not Complete / "No answer found" Errors

If you get "No answer found" responses immediately after starting:

```bash
# 1. Check if indexing is still in progress
docker-compose logs rag-backend-cli | tail -50
# or for API mode:
docker-compose logs rag-backend-api | tail -50

# 2. Look for completion messages:
# ✓ "Index building completed successfully"
# ✓ "rag-pdf-index refreshed and ready for search"
# ✓ "rag-video-index refreshed and ready for search"

# 3. Check index status
docker-compose exec rag-backend-cli python main.py status

# 4. Verify indices have documents
curl http://localhost:9200/rag-pdf-index/_count
curl http://localhost:9200/rag-video-index/_count

# Expected output: {"count": <number>, ...}
# If count is 0, indexing hasn't completed yet

# 5. Wait for indexing to complete (usually 2-3 minutes on first run)
# Then try your query again
```

### Docker Build Fails

```bash
# Clean Docker cache and rebuild
docker system prune -a --volumes -f
docker-compose build --no-cache
docker-compose up
```

### No Results Found (After Indexing Complete)

```bash
# Check indices exist and have data
curl http://localhost:9200/rag-pdf-index/_count
curl http://localhost:9200/rag-video-index/_count

# Lower relevance threshold in .env
RELEVANCE_THRESHOLD=0.3

# Rebuild index
docker-compose exec rag-backend-cli python main.py index --rebuild
```

### OpenSearch Connection Issues

```bash
# Check OpenSearch is healthy
curl http://localhost:9200/_cluster/health

# View OpenSearch logs
docker-compose logs opensearch-node1

# Restart services
docker-compose restart
```

## 📦 Tech Stack

- **Vector Database**: OpenSearch 2.11+ with k-NN plugin
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768-dim)
- **LLM**: OpenAI GPT-4o-mini
- **PDF Processing**: PyMuPDF
- **Search**: HNSW algorithm (cosine similarity)
- **Backend**: Python 3.11+
- **Deployment**: Docker + Docker Compose

## 📝 Data Format Details

### Video Transcript Schema

```json
{
  "video_id": "string (required, unique)",
  "pdf_reference": "string (required, matches PDF filename)",
  "video_transcripts": [
    {
      "id": "integer (sequential, starting from 1)",
      "timestamp": "float (seconds, non-decreasing)",
      "word": "string (single word/token)"
    }
  ]
}
```

### PDF Requirements

- ✅ Extractable text (not scanned images)
- ✅ Multi-page documents supported
- ✅ Tables and lists extracted as text
- ❌ Password-protected PDFs not supported
- ❌ Image-only PDFs not supported

## 🔍 How It Works

### Chunking Strategy

**PDFs**: Paragraph-level chunks (512 tokens target, 128 token overlap)

- Preserves semantic coherence
- Maintains title hierarchy
- Dual embeddings (content + title)

**Videos**: Sentence-based chunks (30-50 words)

- Natural sentence boundaries
- Precise timestamp ranges
- Token IDs for highlighting

### Retrieval Strategy

1. **Query Processing**: Stop words are removed from the query before embedding (same as during indexing)
2. **Tier 1 - Videos**: k-NN search on video index
   - If score ≥ threshold → return video results
3. **Tier 2 - PDFs**: k-NN search on PDF index
   - If score ≥ threshold → return PDF results
4. **No Results**: Return "No answer found" message

Both tiers use pure k-NN vector similarity search for consistent scoring (0.0 to 1.0).

### Response Generation

- Assembles top-k retrieved chunks
- Includes source metadata (timestamps/pages)
- Generates natural language answer via GPT-4o-mini
- Preserves citations in response

## 📈 Performance

- **Indexing Speed**: ~230 docs/sec
- **Query Time**: < 2 seconds (including LLM)
- **Embedding Generation**: ~90 embeddings/sec (CPU)
- **Model Size**: 420MB (MPNet)
- **Memory Usage**: ~2GB (with model loaded)

## 📁 Project Structure

```
Explaino_RAG_AIFounding/
├── data/                           # Sample data for testing
│   ├── pdfs/                       # PDF documents (6 files)
│   │   ├── database_systems_textbook.pdf
│   │   ├── machine_learning_fundamentals.pdf
│   │   ├── python_programming_guide.pdf
│   │   ├── cloud_infrastructure_guide.pdf
│   │   ├── modern_web_development.pdf
│   │   └── Principles-of-Data-Science-WEB.pdf
│   ├── transcripts/                # Video transcripts (8 files)
│   │   ├── database_fundamentals.json
│   │   ├── machine_learning_intro.json
│   │   ├── python_programming_basics.json
│   │   ├── cloud_computing_overview.json
│   │   ├── web_development_trends.json
│   │   ├── deep_learning_tutorial_15min.json
│   │   ├── data_science_tutorial_20min.json
│   │   └── python_advanced_tutorial_30min.json
│   └── knowledge_summary.json      # Auto-generated knowledge summary
│
├── src/                            # Source code
│   ├── __main__.py                 # CLI entry point
│   ├── api.py                      # FastAPI REST API
│   ├── cli.py                      # CLI command handlers
│   ├── config.py                   # Configuration management
│   ├── models.py                   # Data models (Pydantic)
│   ├── rag_system.py               # Main RAG orchestrator
│   │
│   ├── ingestion/                  # Data ingestion modules
│   │   ├── transcript_ingester.py  # Video transcript parser
│   │   └── pdf_ingester.py         # PDF document parser
│   │
│   ├── processing/                 # Data processing modules
│   │   ├── chunking.py             # Text chunking strategies
│   │   ├── embedding.py            # Embedding generation
│   │   └── indexing.py             # OpenSearch indexing
│   │
│   └── retrieval/                  # Retrieval modules
│       ├── query_processor.py      # Query embedding
│       ├── retrieval_engine.py     # Vector search
│       └── response_generator.py   # LLM response generation
│
├── tests/                          # Test suite (41 tests)
│   ├── test_chunking.py
│   ├── test_indexing.py
│   ├── test_models.py
│   ├── test_rag_system.py
│   └── test_token_timestamp.py
│
├── .env.example                    # Environment variables template
├── docker-compose.yml              # Docker orchestration
├── Dockerfile                      # Container definition
├── entrypoint.sh                   # Container startup script
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Test configuration
└── README.md                       # This file
```

## 🔄 Indexing Pipeline

When you add files to the `data/` directory, here's what happens:

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. FILE DETECTION                            │
│  System scans data/pdfs/ and data/transcripts/ directories     │
│  Identifies new/modified files not yet indexed                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. INGESTION                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  PDF Ingestion       │      │  Video Ingestion     │        │
│  │  • Parse PDF text    │      │  • Parse JSON        │        │
│  │  • Extract pages     │      │  • Extract tokens    │        │
│  │  • Detect paragraphs │      │  • Map timestamps    │        │
│  │  • Extract titles    │      │  • Validate format   │        │
│  └──────────────────────┘      └──────────────────────┘        │
└────────────────────┬────────────────────┬───────────────────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. CHUNKING                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  PDF Chunks          │      │  Video Chunks        │        │
│  │  • Paragraph-level   │      │  • Sentence-based    │        │
│  │  • 512 tokens target │      │  • 30-50 words       │        │
│  │  • 128 token overlap │      │  • Adaptive sizing   │        │
│  │  • Title extraction  │      │  • Token ID ranges   │        │
│  └──────────────────────┘      └──────────────────────┘        │
└────────────────────┬────────────────────┬───────────────────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4. EMBEDDING GENERATION                      │
│  Uses: sentence-transformers/all-mpnet-base-v2 (768-dim)       │
│  Stop words removed before embedding for better semantic focus  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  PDF Embeddings      │      │  Video Embeddings    │        │
│  │  • Content embedding │      │  • Text embedding    │        │
│  │  • Title embedding   │      │  • Single vector     │        │
│  │  • Dual vectors      │      │  • ~90 emb/sec       │        │
│  └──────────────────────┘      └──────────────────────┘        │
└────────────────────┬────────────────────┬───────────────────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    5. INDEXING                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  rag-pdf-index       │      │  rag-video-index     │        │
│  │  • k-NN enabled      │      │  • k-NN enabled      │        │
│  │  • HNSW algorithm    │      │  • HNSW algorithm    │        │
│  │  • Cosine similarity │      │  • Cosine similarity │        │
│  │  • Page metadata     │      │  • Timestamp ranges  │        │
│  └──────────────────────┘      └──────────────────────┘        │
└────────────────────┬────────────────────┬───────────────────────┘
                     │                    │
                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    6. READY FOR SEARCH                          │
│  System can now answer questions using indexed content          │
│  • Video-first retrieval (Tier 1)                              │
│  • PDF fallback retrieval (Tier 2)                             │
│  • GPT-4o-mini response generation                             │
└─────────────────────────────────────────────────────────────────┘
```

### Adding New Files

To add your own content:

1. **Add PDFs**: Place PDF files in `data/pdfs/`
2. **Add Videos**: Place transcript JSON files in `data/transcripts/`
3. **Reindex**: Run `docker-compose exec rag-backend-cli python main.py index`
4. **Query**: System automatically includes new content

**Video Transcript Format:**

```json
{
  "video_id": "unique_video_id",
  "pdf_reference": "related_document.pdf",
  "video_transcripts": [
    { "id": 1, "timestamp": 0.0, "word": "Hello" },
    { "id": 2, "timestamp": 0.5, "word": "world" }
  ]
}
```

**Indexing Time Estimates:**

- Small files (< 10 pages): ~5-10 seconds
- Medium files (10-50 pages): ~20-30 seconds
- Large files (> 50 pages): ~1-2 minutes
- Full sample dataset (6 PDFs + 8 videos): ~2-3 minutes
