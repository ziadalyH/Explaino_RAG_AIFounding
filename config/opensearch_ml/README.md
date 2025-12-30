# OpenSearch ML Infrastructure Module

This module contains all OpenSearch ML Commons setup and management code for connecting external LLM providers to OpenSearch.

## 📁 Module Structure

```
config/opensearch_ml/
├── __init__.py              # Package initialization
├── setup.py                 # Main setup script (creates connectors, models, pipelines)
├── verify.py                # Verification script (checks setup status)
├── connector_manager.py     # Connector creation for all LLM providers
├── pipeline_manager.py      # RAG pipeline configuration
└── README.md               # This file
```

## 🎯 Purpose

This module handles the **infrastructure layer** between OpenSearch and external LLM providers:

1. **Connector Creation** - Connects OpenSearch to LLM APIs (OpenAI, DeepSeek, Azure, etc.)
2. **Model Registration** - Registers LLM models with OpenSearch ML Commons
3. **Model Deployment** - Deploys models for inference
4. **Pipeline Configuration** - Sets up RAG (Retrieval-Augmented Generation) pipelines

## 🚀 Usage

### Setup (First Time or Provider Change)

```bash
# From project root
python -m config.opensearch_ml.setup
```

This will:

1. ✅ Create ML connector for your configured LLM provider
2. ✅ Register the model with OpenSearch
3. ✅ Deploy the model (make it ready for inference)
4. ✅ Create RAG search pipeline
5. ✅ Save configuration to `.opensearch_rag_config`

### Verify Setup

```bash
# Check if everything is working
python -m config.opensearch_ml.verify
```

This will:

- ✅ Check connector exists and is accessible
- ✅ Verify model is registered and DEPLOYED
- ✅ Test model inference (actual API call)
- ✅ Confirm RAG pipeline is configured
- ✅ Display all IDs and configuration

### Use in Code

```python
from config.opensearch_ml import OpenSearchConnectorManager, RAGPipelineManager

# Create managers
connector_manager = OpenSearchConnectorManager(client, logger)
pipeline_manager = RAGPipelineManager(client, logger)

# Create connector
connector_id = connector_manager.create_connector(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=500,
    api_key="sk-..."
)
```

## 📋 Supported LLM Providers

All providers are configured via `config/.env`:

| Provider              | Configuration               | Blueprint                 |
| --------------------- | --------------------------- | ------------------------- |
| **OpenAI**            | `LLM_PROVIDER=openai`       | Standard OpenAI API       |
| **DeepSeek**          | `LLM_PROVIDER=deepseek`     | OpenAI-compatible         |
| **Cohere**            | `LLM_PROVIDER=cohere`       | Cohere Chat API           |
| **Azure OpenAI**      | `LLM_PROVIDER=azure_openai` | Azure-specific format     |
| **Amazon Bedrock**    | `LLM_PROVIDER=bedrock`      | AWS SigV4 auth            |
| **Google VertexAI**   | `LLM_PROVIDER=vertexai`     | Google Cloud              |
| **Amazon SageMaker**  | `LLM_PROVIDER=sagemaker`    | AWS SageMaker             |
| **Amazon Comprehend** | `LLM_PROVIDER=comprehend`   | NLP tasks only            |
| **Custom**            | `LLM_PROVIDER=custom`       | Any OpenAI-compatible API |

See `MODEL_PROVIDER_GUIDE.md` for detailed configuration.

## 🔧 Module Components

### 1. `setup.py`

**Main setup script** - Orchestrates the entire setup process

**Key Functions:**

- `create_opensearch_client()` - Creates OpenSearch client
- `get_provider_kwargs()` - Extracts provider-specific configuration
- `cleanup_existing_resources()` - Removes old connectors/models
- `wait_for_model_deployment()` - Waits for model to be ready
- `main()` - Main setup flow

**When to run:**

- First time setup
- Switching LLM providers
- Recreating infrastructure

### 2. `verify.py`

**Verification script** - Checks setup status

**What it checks:**

- Configuration file exists
- OpenSearch connection works
- Connector is accessible
- Model is DEPLOYED
- Model can perform inference
- RAG pipeline is configured

**When to run:**

- After setup to confirm success
- When troubleshooting issues
- To check current configuration

### 3. `connector_manager.py`

**Connector management** - Creates and manages ML connectors

**Key Class:** `OpenSearchConnectorManager`

**Methods:**

- `create_connector()` - Creates connector for any provider
- `register_model()` - Registers model with OpenSearch
- `deploy_model()` - Deploys model for inference
- `get_model_status()` - Checks model deployment status
- `delete_connector()` - Removes connector
- `undeploy_model()` - Undeploys model

**Provider-specific methods:**

- `_create_openai_connector()`
- `_create_deepseek_connector()`
- `_create_cohere_connector()`
- `_create_azure_openai_connector()`
- `_create_bedrock_connector()`
- `_create_vertexai_connector()`
- `_create_sagemaker_connector()`
- `_create_comprehend_connector()`
- `_create_custom_connector()`

### 4. `pipeline_manager.py`

**Pipeline management** - Creates and manages RAG pipelines

**Key Class:** `RAGPipelineManager`

**Methods:**

- `create_rag_pipeline()` - Creates RAG search pipeline
- `get_pipeline()` - Retrieves pipeline configuration
- `delete_pipeline()` - Removes pipeline

## 🔄 Typical Workflow

```
1. Configure provider in config/.env
   ↓
2. Run setup.py
   ↓
3. Setup creates: Connector → Model → Pipeline
   ↓
4. Run verify.py to confirm
   ↓
5. Application uses LLM via OpenSearch
```

## 📝 Configuration File

Setup creates `.opensearch_rag_config` with:

```
CONNECTOR_ID=abc123...
MODEL_ID=xyz789...
PIPELINE_ID=rag-pipeline
PROVIDER=openai
MODEL_NAME=gpt-4o-mini
```

This file is used by:

- `src/llm_inference.py` - To load model ID for inference
- `verify.py` - To check existing setup
- `setup.py` - To cleanup old resources

## 🐛 Troubleshooting

### Setup fails at connector creation

```bash
# Check your API key and provider configuration
cat config/.env | grep LLM_

# Verify OpenSearch is running
curl http://localhost:9200
```

### Model deployment times out

```bash
# Increase timeout in setup.py
# Change: max_wait=120 to max_wait=300
```

### Verification fails

```bash
# Check OpenSearch logs
docker logs opensearch-node

# Re-run setup
python -m config.opensearch_ml.setup
```

## 🔗 Related Documentation

- `MODEL_PROVIDER_GUIDE.md` - Detailed provider configuration
- `QUICK_PROVIDER_REFERENCE.md` - Quick reference card
- `CONNECTOR_UPDATES.md` - Recent connector changes
- `config/.env.example` - Example configuration

## 🏗️ Architecture Benefits

### Separation of Concerns

```
config/opensearch_ml/  → Infrastructure setup (one-time)
src/                   → Application logic (runtime)
```

### Scalability

Easy to add new functionality:

- `cleanup.py` - Clean up old models
- `migrate.py` - Migrate between providers
- `monitor.py` - Monitor model health
- `benchmark.py` - Benchmark different models

### Maintainability

- All OpenSearch ML code in one place
- Clear module boundaries
- Easy to find and update

## 📚 Further Reading

- [OpenSearch ML Commons Documentation](https://opensearch.org/docs/latest/ml-commons-plugin/)
- [Connector Blueprints](https://opensearch.org/docs/latest/ml-commons-plugin/remote-models/blueprints/)
- [RAG with OpenSearch](https://opensearch.org/docs/latest/search-plugins/conversational-search/)
