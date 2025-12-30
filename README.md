# Explaino RAG System

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions from video transcripts and PDF documents using semantic search and OpenSearch-managed LLM connections.

## ✨ Key Features

- **🔌 Dynamic LLM Providers** - Support for 8+ providers (OpenAI, Bedrock, Cohere, Azure, VertexAI, SageMaker, DeepSeek, Custom)
- **🚀 OpenSearch-Native RAG** - All LLM connections managed by OpenSearch ML Commons
- **🎯 Centralized LLM Service** - Single initialization point for all LLM operations
- **⚡ Zero Code Changes** - Switch providers by updating `.env` only
- **🔧 Simple Configuration** - Provider selection + credentials in one file
- **🔄 Automatic Setup** - Connector, model, and pipeline created automatically on first run
- **✅ Smart Verification** - Ensures model is truly ready before use

## 🚀 Quick Start

### One-Command Setup (Recommended)

```bash
# 1. Configure your LLM provider
cp config/.env.example config/.env
# Edit config/.env and set your LLM_PROVIDER and LLM_API_KEY

# 2. Start everything with one command
docker-compose --profile cli up -d
```

This automatically:

1. ✅ Starts OpenSearch
2. ✅ Creates LLM connector (first time only)
3. ✅ Registers and deploys model (first time only)
4. ✅ Creates RAG pipeline (first time only)
5. ✅ Indexes your data (first time or new files)
6. ✅ Starts CLI backend

**Then query:**

```bash
docker-compose exec rag-backend-cli python main.py query -q "What is machine learning?"
```

### What Happens on First Run

When you start the system for the first time, you'll see detailed logs showing:

```
================================================================================
STEP 1: Creating LLM Connector
================================================================================
📡 Creating OPENAI connector
Model: gpt-4o-mini
→ Sending connector creation request to OpenSearch...
✓ Connector created with ID: abc123

================================================================================
STEP 2: Registering Model
================================================================================
📝 Registering model with OpenSearch ML
→ Sending model registration request...
✓ Model registration initiated

================================================================================
STEP 3: Deploying Model
================================================================================
🚀 Deploying model
→ Sending deployment request...
⏳ Waiting for model deployment and readiness...
   Model state: DEPLOYING
   Model state: DEPLOYED
   ✓ Model state is DEPLOYED
   🧪 Testing model with inference call...
   ✓ Model responded successfully!
   ✓ Model is ready for inference!

================================================================================
STEP 4: Creating RAG Pipeline
================================================================================
🔧 Creating RAG search pipeline
→ Sending pipeline creation request...
✓ RAG pipeline created successfully

✓ OpenSearch RAG setup completed successfully
```

The system verifies the model is truly ready by testing it with an actual inference call before proceeding.

## 🤖 Supported LLM Providers

| Provider             | Models                        | Auth Type       | Status   |
| -------------------- | ----------------------------- | --------------- | -------- |
| **OpenAI**           | gpt-4, gpt-4o, gpt-3.5-turbo  | API Key         | ✅ Ready |
| **Amazon Bedrock**   | Claude v2/v3, Jurassic-2      | AWS Credentials | ✅ Ready |
| **Azure OpenAI**     | gpt-4, gpt-35-turbo           | API Key         | ✅ Ready |
| **Cohere**           | command, command-light        | API Key         | ✅ Ready |
| **Google VertexAI**  | chat-bison, gemini-pro        | GCP Token       | ✅ Ready |
| **Amazon SageMaker** | Custom models                 | AWS Credentials | ✅ Ready |
| **DeepSeek**         | deepseek-chat, deepseek-coder | API Key         | ✅ Ready |
| **Custom**           | Any OpenAI-compatible         | Configurable    | ✅ Ready |

**Switching Providers**: Just update `config/.env`, delete `.opensearch_rag_config`, and restart - no code changes needed!

**📖 Complete Provider Guide:** See [LLM_PROVIDERS.md](LLM_PROVIDERS.md) for detailed configuration examples for each provider.

## ⚙️ Configuration

### LLM Configuration

Edit `config/.env` to configure your LLM provider:

```bash
# ============================================
# LLM Configuration (OpenSearch Connector)
# ============================================
# Supported providers: openai, bedrock, cohere, azure_openai, vertexai, sagemaker, deepseek, custom

# Common settings (all providers)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...your-key...
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500

# Provider-specific settings (see LLM_PROVIDERS.md for details)
```

**📖 Detailed Configuration:** See [LLM_PROVIDERS.md](LLM_PROVIDERS.md) for:

- Complete configuration examples for each provider
- Required credentials and endpoints
- Model recommendations
- Troubleshooting tips

### Embedding Configuration

```bash
# ============================================
# Embedding Configuration
# ============================================
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
```

**Popular Models:**

- `all-MiniLM-L6-v2` (384-dim) - Fast, good for development
- `all-mpnet-base-v2` (768-dim) - **Default**, balanced quality/speed
- `all-roberta-large-v1` (1024-dim) - Highest quality, slower

**📖 Complete Model List:** See [MODEL_OPTIONS.md](MODEL_OPTIONS.md) for all available embedding models.

### Other Settings

```bash
# OpenSearch
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200

# Retrieval
RELEVANCE_THRESHOLD=0.5
MAX_RESULTS=5

# System
AUTO_INDEX_ON_STARTUP=true
LOG_LEVEL=INFO
```

### Changing Configuration

**No rebuild needed!** Just edit `config/.env` and restart:

```bash
# Edit configuration
nano config/.env

# Restart to apply changes
docker-compose restart
```

## 🔄 Switching LLM Providers

To switch to a different LLM provider:

```bash
# 1. Update config/.env
LLM_PROVIDER=bedrock
LLM_MODEL=anthropic.claude-v2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# 2. Delete old configuration
rm .opensearch_rag_config

# 3. Restart (setup runs automatically)
docker-compose restart rag-backend-cli

# 4. Query as normal
docker-compose exec rag-backend-cli python main.py query -q "Your question"
```

The system will automatically:

- Delete old connector, model, and pipeline
- Create new connector for the new provider
- Register and deploy the new model
- Create new RAG pipeline
- Verify the model is ready

## 🏗️ Architecture

```
User Query
    ↓
Python Application
    ├── Centralized LLM Service (single initialization)
    │   ├── Response Generator
    │   └── Knowledge Summary Generator
    ↓
OpenSearch
├── Vector Search (finds relevant documents)
├── RAG Pipeline (combines context + query)
├── ML Connector (provider-specific)
└── ML Inference (calls LLM)
    ↓
LLM Provider API (OpenAI/Bedrock/Cohere/etc.)
    ↓
Generated Answer
```

**Key Features:**

- **Centralized LLM Service**: Single initialization point for all LLM operations
- **Smart Verification**: Tests model with actual inference before proceeding
- **Automatic Setup**: Connector, model, and pipeline created on first run
- **Provider Agnostic**: All LLM communication through OpenSearch ML connectors

**📖 Architecture Details:** See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) and [CENTRALIZED_LLM_SERVICE.md](CENTRALIZED_LLM_SERVICE.md) for complete technical documentation.

## 📋 CLI Commands

```bash
# Query the system
docker-compose exec rag-backend-cli python main.py query -q "Your question"

# Build/rebuild index
docker-compose exec rag-backend-cli python main.py index
docker-compose exec rag-backend-cli python main.py index --force-rebuild

# Selective reindexing
docker-compose exec rag-backend-cli python main.py index --videos-only
docker-compose exec rag-backend-cli python main.py index --pdfs-only

# Check system status
docker-compose exec rag-backend-cli python main.py status

# Verify LLM setup
docker-compose exec rag-backend-cli python verify_setup.py
```

## 🔧 API Mode

For REST API access:

```bash
# Start API mode
docker-compose --profile api up -d

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Check index status
curl http://localhost:8000/index/status

# Get knowledge summary
curl http://localhost:8000/knowledge/summary
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
docker-compose exec rag-backend-cli python main.py index

# Force rebuild entire index
docker-compose exec rag-backend-cli python main.py index --force-rebuild

# Reindex only videos
docker-compose exec rag-backend-cli python main.py index --videos-only

# Reindex only PDFs
docker-compose exec rag-backend-cli python main.py index --pdfs-only
```

## 🐛 Troubleshooting

### Setup Issues

If setup fails or times out:

```bash
# 1. Check logs for detailed error messages
docker-compose logs rag-backend-cli | grep "❌"

# 2. Verify LLM credentials in config/.env
cat config/.env | grep LLM_

# 3. Delete config and retry
rm .opensearch_rag_config
docker-compose restart rag-backend-cli

# 4. Verify setup manually
docker-compose exec rag-backend-cli python verify_setup.py
```

### Model Not Ready

If you see "Model not ready" errors:

```bash
# The system automatically waits up to 120 seconds for the model
# If it still fails, check:

# 1. OpenSearch ML plugin logs
docker-compose logs opensearch-node1 | grep -i "ml"

# 2. Model status
curl http://localhost:9200/_plugins/_ml/models/<model_id>

# 3. Try increasing timeout in setup_opensearch_rag.py
# Change: max_wait=120 to max_wait=300
```

### No Results Found

```bash
# Check indices exist and have data
curl http://localhost:9200/rag-pdf-index/_count
curl http://localhost:9200/rag-video-index/_count

# Lower relevance threshold in config/.env
RELEVANCE_THRESHOLD=0.3

# Rebuild index
docker-compose exec rag-backend-cli python main.py index --force-rebuild
```

## 📖 Documentation

- **[LLM_PROVIDERS.md](LLM_PROVIDERS.md)** - Complete LLM provider configuration guide
- **[CENTRALIZED_LLM_SERVICE.md](CENTRALIZED_LLM_SERVICE.md)** - Centralized LLM service architecture
- **[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)** - Complete system architecture
- **[MODEL_OPTIONS.md](MODEL_OPTIONS.md)** - Available embedding models
- **[SETUP_LOGS_GUIDE.md](SETUP_LOGS_GUIDE.md)** - Understanding setup logs
- **[FIX_MODEL_READINESS.md](FIX_MODEL_READINESS.md)** - Model readiness verification

## 📦 Tech Stack

- **Vector Database**: OpenSearch 2.11+ with k-NN plugin
- **LLM Management**: OpenSearch ML Commons plugin
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768-dim)
- **LLM**: Configurable (OpenAI, Bedrock, Cohere, etc.)
- **PDF Processing**: PyMuPDF
- **Search**: HNSW algorithm (cosine similarity)
- **Backend**: Python 3.11+
- **Deployment**: Docker + Docker Compose

## 🎯 Key Features

- **Centralized LLM Service**: Single initialization point for all LLM operations
- **Smart Model Verification**: Tests model with actual inference before proceeding
- **Two-Tier Retrieval**: Searches videos first, falls back to PDFs
- **Precise Citations**: Exact timestamps for videos, page/paragraph for PDFs
- **Dual Indices**: Separate `rag-pdf-index` and `rag-video-index`
- **Flexible Configuration**: Change providers, models, and settings via `.env`
- **Auto-Indexing**: Data indexed automatically on startup
- **Resume Capability**: Only processes new files on re-indexing

## 📈 Performance

- **Indexing Speed**: ~230 docs/sec
- **Query Time**: < 2 seconds (including LLM)
- **Embedding Generation**: ~90 embeddings/sec (CPU)
- **Model Size**: 420MB (MPNet)
- **Memory Usage**: ~2GB (with model loaded)

## 📁 Project Structure

```
Explaino_RAG-based-chatbot/
├── config/                         # Configuration modules
│   ├── .env                        # Environment configuration
│   ├── config.py                   # Config management
│   ├── connector_manager.py        # LLM connector management
│   ├── pipeline_manager.py         # RAG pipeline management
│   └── knowledge_summary.py        # Knowledge summary generator
│
├── src/                            # Source code
│   ├── llm_inference.py            # Centralized LLM service
│   ├── rag_system.py               # Main RAG orchestrator
│   ├── models.py                   # Data models
│   │
│   ├── ingestion/                  # Data ingestion
│   │   ├── transcript_ingester.py
│   │   └── pdf_ingester.py
│   │
│   ├── processing/                 # Data processing
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   └── indexing.py
│   │
│   └── retrieval/                  # Retrieval modules
│       ├── query_processor.py
│       ├── retrieval_engine.py
│       └── response_generator.py
│
├── data/                           # Sample data
│   ├── pdfs/                       # PDF documents
│   └── transcripts/                # Video transcripts
│
├── setup_opensearch_rag.py         # LLM setup script
├── verify_setup.py                 # Setup verification script
├── main.py                         # CLI entry point
├── docker-compose.yml              # Docker orchestration
└── README.md                       # This file
```

## 🔄 How It Works

### Setup Phase (First Run)

1. **Connector Creation**: Creates provider-specific ML connector in OpenSearch
2. **Model Registration**: Registers the LLM model with OpenSearch ML
3. **Model Deployment**: Deploys the model and waits for it to be ready
4. **Inference Verification**: Tests the model with actual inference call
5. **Pipeline Creation**: Creates RAG search pipeline
6. **Configuration Save**: Saves connector/model/pipeline IDs to `.opensearch_rag_config`

### Query Phase

1. **Query Processing**: Embeds user query using MPNet
2. **Vector Search**: Searches video and PDF indices
3. **Context Retrieval**: Retrieves top-k relevant chunks
4. **LLM Generation**: Generates answer using centralized LLM service
5. **Response Formatting**: Returns structured response with citations

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.
