# RAG Chatbot Backend

A production-grade, backend-only RAG (Retrieval-Augmented Generation) chatbot system that answers questions from video transcripts and PDF documents. The system uses OpenSearch for vector storage and OpenAI for embeddings and LLM-based answer generation.

## Features

- **Two-tier retrieval strategy**: Prioritizes video transcripts, falls back to PDF documents
- **Precise citations**: Provides timestamps for video content, page/paragraph references for PDFs
- **Vector similarity search**: Uses OpenSearch with k-NN for semantic search
- **LLM-powered answers**: Generates natural language answers using OpenAI GPT models
- **Docker support**: Easy deployment with Docker Compose
- **Automatic indexing**: Discovers and indexes new files on startup
- **Configurable**: All parameters externalized to configuration files

## Architecture

The system consists of several modular components:

- **Ingestion Layer**: Parses video transcript JSON files and PDF documents
- **Processing Layer**: Chunks content, generates embeddings, builds vector index
- **Query Layer**: Processes queries, retrieves relevant content, generates responses
- **Storage Layer**: OpenSearch vector database with metadata

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- OpenSearch cluster (provided via Docker Compose)

## Installation

### Option 1: Docker Compose (Recommended)

1. Clone the repository
2. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. Start the services:
   ```bash
   docker-compose up -d
   ```

The system will:

- Start OpenSearch on port 9200
- Start the RAG backend
- Automatically index data if `AUTO_INDEX_ON_STARTUP=true`

### Option 2: Local Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up OpenSearch (or use Docker):

   ```bash
   docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.11.1
   ```

3. Configure environment variables:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   export OPENSEARCH_HOST=localhost
   export OPENSEARCH_PORT=9200
   ```

4. Build the index:
   ```bash
   python main.py index
   ```

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENSEARCH_HOST`: OpenSearch host (default: localhost)
- `OPENSEARCH_PORT`: OpenSearch port (default: 9200)
- `OPENSEARCH_INDEX_NAME`: Index name (default: rag-index)
- `AUTO_INDEX_ON_STARTUP`: Auto-index on startup (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)

### Configuration File

Edit `config/config.example.yaml` to customize:

- Embedding model (default: text-embedding-3-small)
- LLM model (default: gpt-4o-mini)
- Relevance threshold (default: 0.5)
- Chunk size and overlap
- Data directories

## Data Management

### Adding Video Transcripts

1. Place JSON files in `data/transcripts/` directory
2. Each file must follow this schema:

```json
{
  "video_id": "unique_video_identifier",
  "pdf_reference": "reference_to_related_pdf",
  "video_transcripts": [
    {
      "id": 0,
      "timestamp": 0.0,
      "word": "Hello"
    },
    {
      "id": 1,
      "timestamp": 0.5,
      "word": "world"
    }
  ]
}
```

3. Run indexing to process new files:
   ```bash
   python main.py index
   ```

### Adding PDF Documents

1. Place PDF files in `data/pdfs/` directory
2. PDFs must contain extractable text (not scanned images)
3. Run indexing to process new files:
   ```bash
   python main.py index
   ```

### Re-indexing

To update the index with new or modified files:

```bash
# Local
python main.py index --rebuild

# Docker
docker-compose exec rag-backend python main.py index --rebuild
```

The system will:

- Discover all files in configured directories
- Generate embeddings for new content
- Update the OpenSearch index
- Preserve existing data for unchanged files

## Usage

### Programmatic Usage

You can use the RAGSystem class directly in your Python code:

```python
from src.rag_system import RAGSystem
from src.config import Config

# Load configuration
config = Config.from_env()

# Initialize RAG system
rag_system = RAGSystem(config)

# Build index (first time or when data changes)
rag_system.build_index(force_rebuild=False)

# Answer questions
response = rag_system.answer_question("How do I add a new customer?")

# Access response data
if response.answer_type == "video":
    print(f"Video: {response.video_id}")
    print(f"Timestamp: {response.start_timestamp}s - {response.end_timestamp}s")
    print(f"Answer: {response.generated_answer}")
elif response.answer_type == "pdf":
    print(f"PDF: {response.pdf_filename}, Page {response.page_number}")
    print(f"Answer: {response.generated_answer}")
else:
    print(response.message)
```

### Querying the Chatbot

```bash
# Local
python main.py query --question "How do I add a new customer?"

# Docker
docker-compose exec rag-backend python main.py query --question "How do I add a new customer?"

# Verbose output
python main.py query --question "What is the pricing?" --verbose
```

### Response Format

The system returns structured responses:

**Video Response:**

```json
{
  "answer_type": "video",
  "video_id": "video_123",
  "start_timestamp": 45.2,
  "end_timestamp": 52.8,
  "start_token_id": 120,
  "end_token_id": 145,
  "transcript_snippet": "To add a new customer, click the Add button...",
  "generated_answer": "You can add a new customer by clicking the Add button in the customer management section."
}
```

**PDF Response:**

```json
{
  "answer_type": "pdf",
  "pdf_filename": "user_guide.pdf",
  "page_number": 5,
  "paragraph_index": 2,
  "source_snippet": "Customer management is available in the admin panel...",
  "generated_answer": "Customer management features are located in the admin panel, as described on page 5."
}
```

**No Answer:**

```json
{
  "answer_type": "no_answer",
  "message": "No relevant answer found in the knowledge base."
}
```

## Docker Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f rag-backend

# Query the chatbot
docker-compose exec rag-backend python main.py query --question "Your question here"

# Rebuild index
docker-compose exec rag-backend python main.py index --rebuild

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# Property-based tests only
pytest -m property

# Integration tests only
pytest -m integration

# With coverage
pytest --cov=src --cov-report=html
```

## Project Structure

```
rag-chatbot-backend/
├── config/                      # Configuration files
│   └── config.example.yaml
├── data/                        # User data
│   ├── transcripts/            # Video transcript JSON files
│   └── pdfs/                   # PDF documents
├── mock_data/                   # Sample data for testing
│   ├── transcripts/
│   └── pdfs/
├── src/                         # Source code
│   ├── ingestion/              # Data ingestion modules
│   ├── processing/             # Chunking, embedding, indexing
│   ├── retrieval/              # Query processing and retrieval
│   ├── models.py               # Data models
│   ├── config.py               # Configuration management
│   ├── rag_system.py           # Main orchestrator
│   └── cli.py                  # Command-line interface
├── tests/                       # Test suite
├── docker-compose.yml           # Docker Compose configuration
├── Dockerfile                   # Docker image definition
├── entrypoint.sh               # Docker entrypoint script
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point
└── README.md                   # This file
```

## Troubleshooting

### OpenSearch Connection Issues

If you see connection errors:

1. Check OpenSearch is running:

   ```bash
   curl http://localhost:9200/_cluster/health
   ```

2. Verify environment variables:

   ```bash
   echo $OPENSEARCH_HOST
   echo $OPENSEARCH_PORT
   ```

3. Check Docker logs:
   ```bash
   docker-compose logs opensearch
   ```

### Indexing Issues

If indexing fails:

1. Check data directory permissions
2. Verify JSON schema matches requirements
3. Check OpenAI API key is valid
4. Review logs for specific errors

### Query Issues

If queries return no results:

1. Verify index has documents:

   ```bash
   curl http://localhost:9200/rag-index/_count
   ```

2. Try lowering relevance threshold in config
3. Check query embedding generation
4. Review logs for errors

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
