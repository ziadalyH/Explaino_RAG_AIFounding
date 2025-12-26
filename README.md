# Explaino RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions from video transcripts and PDF documents using semantic search and LLM-powered responses.

## 🚀 Quick Start

Get up and running in 3 steps:

```bash
# 1. Clone and navigate
git clone https://github.com/ziadalyH/Explaino_RAG_AIFounding.git
cd Explaino_RAG_AIFounding

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

# Query the system
docker-compose exec rag-backend-cli python main.py query -q "What is a database?"

# Index management
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

#### API Mode

For REST API access (ideal for frontend integration):

```bash
# Start API mode
docker-compose --profile api up -d

# API available at http://localhost:8000
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a database?"}'
```

#### Both Modes Together

```bash
# Start both CLI and API
docker-compose --profile cli --profile api up -d

# Use CLI
docker-compose exec rag-backend-cli python main.py query -q "test"

# Use API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
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
- **Local Embeddings**: Uses MPNet (768-dim) - no API costs
- **Pure k-NN Search**: Consistent vector similarity scoring (0.0 to 1.0) for both videos and PDFs
- **Auto-Indexing**: Data indexed automatically on startup
- **Resume Capability**: Only processes new files on re-indexing

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
docker-compose restart rag-backend
```

Changes take effect immediately on restart.

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

### Docker Build Fails

```bash
# Clean Docker cache and rebuild
docker system prune -a --volumes -f
docker-compose build --no-cache
docker-compose up
```

### No Results Found

```bash
# Check indices exist and have data
curl http://localhost:9200/rag-pdf-index/_count
curl http://localhost:9200/rag-video-index/_count

# Lower relevance threshold in .env
RELEVANCE_THRESHOLD=0.3

# Rebuild index
docker-compose exec rag-backend python main.py index --rebuild
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

1. **Tier 1 - Videos**: k-NN search on video index
   - If score ≥ threshold → return video results
2. **Tier 2 - PDFs**: k-NN search on PDF index
   - If score ≥ threshold → return PDF results
3. **No Results**: Return "No answer found" message

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

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Add your license here]

## 🙋 Support

For issues or questions:

- Open an issue on GitHub
- Check existing issues for solutions
- Review troubleshooting section above

---

**Note**: This system requires an OpenAI API key for response generation. Embedding generation uses local models (no API costs).
