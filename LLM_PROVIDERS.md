# LLM Provider Configuration Guide

This guide explains how to configure different LLM providers for the RAG system. The system uses OpenSearch ML connectors to communicate with various LLM providers.

## Supported Providers

The system supports **chat and completion models only** (no embedding models):

- **OpenAI** - Chat and completion models (gpt-4, gpt-3.5-turbo, text-davinci-003)
- **Amazon Bedrock** - Claude, Jurassic-2, and other chat models
- **Azure OpenAI** - Chat models (gpt-4, gpt-35-turbo)
- **Cohere** - Chat models (command, command-light)
- **Google VertexAI** - Chat models (chat-bison)
- **Amazon SageMaker** - Custom hosted chat models
- **DeepSeek** - Chat models (deepseek-chat, deepseek-coder)
- **Custom** - Any OpenAI-compatible endpoint

## Configuration

All configuration is done through environment variables in `config/.env`. Set the `LLM_PROVIDER` variable and the corresponding provider-specific credentials.

---

## 1. OpenAI

**Models**: gpt-4, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, text-davinci-003, etc.

**Configuration**:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...your-api-key...
LLM_ENDPOINT=https://api.openai.com/v1/chat/completions
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)

---

## 2. Amazon Bedrock

**Models**:

- Anthropic Claude (claude-v2, claude-v3, claude-3-sonnet, claude-3-opus)
- AI21 Labs Jurassic-2 (ai21.j2-mid, ai21.j2-ultra)
- Amazon Titan (amazon.titan-text-express)

**Configuration**:

```bash
LLM_PROVIDER=bedrock
LLM_MODEL=anthropic.claude-v2
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Requirements**:

- AWS account with Bedrock access
- IAM credentials with `bedrock:InvokeModel` permission
- Model access enabled in Bedrock console

**Connector Blueprint**: [Bedrock Claude Blueprint](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/connectors/#amazon-bedrock-connector)

---

## 3. Azure OpenAI

**Models**: gpt-4, gpt-35-turbo, gpt-4-32k, etc.

**Configuration**:

```bash
LLM_PROVIDER=azure_openai
LLM_MODEL=gpt-4
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-resource.openai.azure.com
AZURE_API_VERSION=2023-05-15
AZURE_DEPLOYMENT_NAME=your-deployment-name
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Requirements**:

- Azure subscription
- Azure OpenAI resource created
- Model deployed in Azure OpenAI Studio

**Connector Blueprint**: [Azure OpenAI Blueprint](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/connectors/#azure-openai-connector)

---

## 4. Cohere

**Models**: command, command-light, command-nightly

**Configuration**:

```bash
LLM_PROVIDER=cohere
LLM_MODEL=command
COHERE_API_KEY=your-cohere-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Get API Key**: [Cohere Dashboard](https://dashboard.cohere.com/api-keys)

**Connector Blueprint**: [Cohere Chat Blueprint](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/connectors/#cohere-chat-connector)

---

## 5. Google VertexAI

**Models**: chat-bison, codechat-bison, gemini-pro

**Configuration**:

```bash
LLM_PROVIDER=vertexai
LLM_MODEL=chat-bison
VERTEXAI_PROJECT=your-gcp-project-id
VERTEXAI_LOCATION=us-central1
VERTEXAI_ACCESS_TOKEN=your-access-token
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Requirements**:

- Google Cloud Platform account
- VertexAI API enabled
- Service account with VertexAI permissions
- Access token from `gcloud auth print-access-token`

**Connector Blueprint**: [VertexAI Blueprint](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/connectors/#google-vertexai-connector)

---

## 6. Amazon SageMaker

**Models**: Custom models deployed on SageMaker endpoints

**Configuration**:

```bash
LLM_PROVIDER=sagemaker
LLM_MODEL=your-model-name
SAGEMAKER_ENDPOINT=your-endpoint-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Requirements**:

- AWS account
- Model deployed to SageMaker endpoint
- IAM credentials with `sagemaker:InvokeEndpoint` permission

**Connector Blueprint**: [SageMaker Blueprint](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/connectors/#amazon-sagemaker-connector)

---

## 7. DeepSeek

**Models**: deepseek-chat, deepseek-coder

**Configuration**:

```bash
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=your-deepseek-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500
```

**Get API Key**: [DeepSeek Platform](https://platform.deepseek.com/)

**Connector Blueprint**: [DeepSeek Blueprint](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/connectors/#deepseek-connector)

---

## 8. Custom OpenAI-Compatible Endpoint

For any service that implements the OpenAI API format (e.g., LocalAI, Ollama with OpenAI compatibility, vLLM, etc.)

**Configuration**:

```bash
LLM_PROVIDER=custom
LLM_MODEL=your-model-name
LLM_ENDPOINT=https://your-custom-endpoint.com/v1/chat/completions
LLM_API_KEY=your-api-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500

# Optional: Custom headers (JSON format)
LLM_HEADERS={"X-Custom-Header": "value"}

# Optional: Custom request template
LLM_REQUEST_TEMPLATE={"model": "${parameters.model}", "messages": ${parameters.messages}}
```

---

## Setup Process

After configuring your provider in `config/.env`:

1. **Start OpenSearch**:

   ```bash
   docker-compose up -d opensearch-node1
   ```

2. **Run Setup** (creates connector and registers model):

   ```bash
   docker-compose --profile setup up rag-setup
   ```

3. **Query the System**:
   ```bash
   python query_opensearch_rag.py "Your question here"
   ```

The setup script will:

- Read your provider configuration from `.env`
- Create the appropriate OpenSearch ML connector
- Register and deploy the model
- Create the RAG pipeline
- Save configuration to `.opensearch_rag_config`

---

## Switching Providers

To switch providers:

1. Update `LLM_PROVIDER` and provider-specific credentials in `config/.env`
2. Re-run the setup: `docker-compose --profile setup up rag-setup`
3. The old connector will be cleaned up and a new one created

---

## Common Parameters

All providers support these common parameters:

- **LLM_TEMPERATURE** (0.0-1.0): Controls randomness. Lower = more deterministic
- **LLM_MAX_TOKENS**: Maximum tokens in the response
- **LLM_MODEL**: Model identifier (provider-specific)

---

## Troubleshooting

### Authentication Errors

- Verify API keys are correct and not expired
- Check IAM permissions for AWS services
- Ensure service accounts have proper roles for GCP

### Model Not Found

- Verify model name matches provider's model ID
- Check if model access is enabled (Bedrock, Azure)
- Ensure model is deployed (Azure, SageMaker)

### Connection Errors

- Verify endpoint URLs are correct
- Check network connectivity to provider
- Ensure OpenSearch can reach external APIs

### View Logs

```bash
docker-compose logs rag-setup
docker-compose logs opensearch-node1
```

---

## Reference

- [OpenSearch ML Commons Documentation](https://docs.opensearch.org/latest/ml-commons-plugin/)
- [Supported Connectors](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/supported-connectors/)
- [Connector Blueprints](https://docs.opensearch.org/latest/ml-commons-plugin/remote-models/blueprints/)
