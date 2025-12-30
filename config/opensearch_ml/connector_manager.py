"""OpenSearch ML Connector Manager supporting chat and completion models.

Supports:
- OpenAI (chat, completion)
- Amazon Bedrock (Claude, Jurassic, etc.)
- Azure OpenAI (chat)
- Cohere (chat)
- Google VertexAI (chat)
- Amazon SageMaker (chat)
- DeepSeek (chat)
- Custom OpenAI-compatible endpoints
"""

import logging
import json
from typing import Dict, Any, Optional
from opensearchpy import OpenSearch


class OpenSearchConnectorManager:
    """Manages OpenSearch ML connectors for all supported LLM providers."""
    
    def __init__(self, client: OpenSearch, logger: Optional[logging.Logger] = None):
        """
        Initialize connector manager.
        
        Args:
            client: OpenSearch client instance
            logger: Optional logger instance
        """
        self.client = client
        self.logger = logger or logging.getLogger(__name__)
    
    def create_connector(
        self,
        provider: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Create a connector based on provider type.
        
        Args:
            provider: Provider name (openai, bedrock, cohere, azure_openai, vertexai, sagemaker, deepseek, custom)
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Provider-specific parameters
            
        Returns:
            Connector ID
        """
        provider_methods = {
            "openai": self._create_openai_connector,
            "bedrock": self._create_bedrock_connector,
            "cohere": self._create_cohere_connector,
            "azure_openai": self._create_azure_openai_connector,
            "vertexai": self._create_vertexai_connector,
            "sagemaker": self._create_sagemaker_connector,
            "deepseek": self._create_deepseek_connector,
            "comprehend": self._create_comprehend_connector,
            "custom": self._create_custom_connector,
        }
        
        if provider not in provider_methods:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(provider_methods.keys())}")
        
        self.logger.info("=" * 80)
        self.logger.info(f"📡 Creating {provider.upper()} connector")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Temperature: {temperature}")
        self.logger.info(f"Max Tokens: {max_tokens}")
        
        connector_id = provider_methods[provider](model, temperature, max_tokens, **kwargs)
        
        self.logger.info(f"✓ Connector created successfully: {connector_id}")
        return connector_id
    
    def _create_openai_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        endpoint: str = "https://api.openai.com/v1/chat/completions",
        **kwargs
    ) -> str:
        """Create OpenAI connector."""
        connector_config = {
            "name": f"OpenAI {model} Connector",
            "description": f"Connector for OpenAI {model}",
            "version": 2,
            "protocol": "http",
            "parameters": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "credential": {
                "openAI_key": api_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": endpoint,
                    "headers": {
                        "Authorization": "Bearer ${credential.openAI_key}",
                        "Content-Type": "application/json"
                    },
                    "request_body": """{
                        "model": "${parameters.model}",
                        "messages": ${parameters.messages},
                        "temperature": ${parameters.temperature},
                        "max_tokens": ${parameters.max_tokens}
                    }"""
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_bedrock_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        **kwargs
    ) -> str:
        """Create Amazon Bedrock connector."""
        connector_config = {
            "name": f"Bedrock {model} Connector",
            "description": f"Connector for Amazon Bedrock {model}",
            "version": 2,
            "protocol": "aws_sigv4",
            "parameters": {
                "region": aws_region,
                "service_name": "bedrock",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "credential": {
                "access_key": aws_access_key_id,
                "secret_key": aws_secret_access_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://bedrock-runtime.{aws_region}.amazonaws.com/model/{model}/invoke",
                    "headers": {
                        "Content-Type": "application/json",
                        "x-amz-content-sha256": "required"
                    },
                    "request_body": json.dumps({
                        "messages": "${parameters.messages}",
                        "temperature": "${parameters.temperature}",
                        "max_tokens": "${parameters.max_tokens}",
                        "anthropic_version": "bedrock-2023-05-31"
                    })
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_cohere_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        **kwargs
    ) -> str:
        """Create Cohere connector.
        
        Cohere Chat API uses 'message' (string) not 'messages' (array).
        Official blueprint: https://github.com/opensearch-project/ml-commons/blob/main/docs/remote_inference_blueprints/cohere_connector_chat_blueprint.md
        """
        connector_config = {
            "name": f"Cohere {model} Connector",
            "description": f"Connector for Cohere {model}",
            "version": 1,
            "protocol": "http",
            "parameters": {
                "model": model
            },
            "credential": {
                "cohere_key": api_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://api.cohere.ai/v1/chat",
                    "headers": {
                        "Authorization": "Bearer ${credential.cohere_key}",
                        "Request-Source": "unspecified:opensearch"
                    },
                    "request_body": "{ \"message\": \"${parameters.message}\", \"model\": \"${parameters.model}\" }"
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_azure_openai_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        api_base: str,
        api_version: str = "2023-05-15",
        deployment_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create Azure OpenAI connector.
        
        Azure OpenAI uses deployment names in the URL path and 'api-key' header (not 'Authorization').
        Official blueprint: https://github.com/opensearch-project/ml-commons/blob/main/docs/remote_inference_blueprints/azure_openai_connector_chat_blueprint.md
        """
        deployment = deployment_name or model
        # Extract endpoint from api_base (remove https:// and trailing /)
        endpoint = api_base.replace("https://", "").replace("http://", "").rstrip("/")
        
        connector_config = {
            "name": f"Azure OpenAI {model} Connector",
            "description": f"Connector for Azure OpenAI {model}",
            "version": 1,
            "protocol": "http",
            "parameters": {
                "endpoint": endpoint,
                "deploy-name": deployment,
                "model": model,
                "api-version": api_version,
                "temperature": temperature
            },
            "credential": {
                "openAI_key": api_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://${parameters.endpoint}/openai/deployments/${parameters.deploy-name}/chat/completions?api-version=${parameters.api-version}",
                    "headers": {
                        "api-key": "${credential.openAI_key}"
                    },
                    "request_body": "{ \"messages\": ${parameters.messages}, \"temperature\": ${parameters.temperature} }"
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_vertexai_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        project_id: str,
        location: str = "us-central1",
        access_token: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create Google VertexAI connector."""
        endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:predict"
        
        connector_config = {
            "name": f"VertexAI {model} Connector",
            "description": f"Connector for Google VertexAI {model}",
            "version": 2,
            "protocol": "http",
            "parameters": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "credential": {
                "access_token": access_token
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": endpoint,
                    "headers": {
                        "Authorization": "Bearer ${credential.access_token}",
                        "Content-Type": "application/json"
                    },
                    "request_body": json.dumps({
                        "instances": [{"content": "${parameters.prompt}"}],
                        "parameters": {
                            "temperature": "${parameters.temperature}",
                            "maxOutputTokens": "${parameters.max_tokens}"
                        }
                    })
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_sagemaker_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        endpoint_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        **kwargs
    ) -> str:
        """Create Amazon SageMaker connector."""
        connector_config = {
            "name": f"SageMaker {model} Connector",
            "description": f"Connector for Amazon SageMaker {model}",
            "version": 2,
            "protocol": "aws_sigv4",
            "parameters": {
                "region": aws_region,
                "service_name": "sagemaker",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "credential": {
                "access_key": aws_access_key_id,
                "secret_key": aws_secret_access_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": f"https://runtime.sagemaker.{aws_region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "request_body": json.dumps({
                        "inputs": "${parameters.prompt}",
                        "parameters": {
                            "temperature": "${parameters.temperature}",
                            "max_new_tokens": "${parameters.max_tokens}"
                        }
                    })
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_deepseek_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        **kwargs
    ) -> str:
        """Create DeepSeek connector.
        
        DeepSeek uses OpenAI-compatible API format.
        """
        connector_config = {
            "name": f"DeepSeek {model} Connector",
            "description": f"Connector for DeepSeek {model}",
            "version": 1,
            "protocol": "http",
            "parameters": {
                "endpoint": "api.deepseek.com",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "credential": {
                "deepSeek_key": api_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://${parameters.endpoint}/v1/chat/completions",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer ${credential.deepSeek_key}"
                    },
                    "request_body": "{ \"model\": \"${parameters.model}\", \"messages\": ${parameters.messages}, \"temperature\": ${parameters.temperature}, \"max_tokens\": ${parameters.max_tokens} }"
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_comprehend_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        session_token: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create Amazon Comprehend connector.
        
        Note: Comprehend is not a chat model - it's for language detection and NLP tasks.
        Official blueprint: https://github.com/opensearch-project/ml-commons/blob/main/docs/remote_inference_blueprints/amazon_comprehend_connector_blueprint.md
        """
        credential = {
            "access_key": aws_access_key_id,
            "secret_key": aws_secret_access_key
        }
        if session_token:
            credential["session_token"] = session_token
        
        connector_config = {
            "name": f"Amazon Comprehend Connector",
            "description": f"Connector for Amazon Comprehend",
            "version": 1,
            "protocol": "aws_sigv4",
            "credential": credential,
            "parameters": {
                "service_name": "comprehend",
                "region": aws_region,
                "api_version": "20171127",
                "api_name": "DetectDominantLanguage",
                "api": "Comprehend_${parameters.api_version}.${parameters.api_name}",
                "response_filter": "$"
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": "https://${parameters.service_name}.${parameters.region}.amazonaws.com",
                    "headers": {
                        "X-Amz-Target": "${parameters.api}",
                        "content-type": "application/x-amz-json-1.1"
                    },
                    "request_body": "{ \"Text\": \"${parameters.Text}\"}"
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_custom_connector(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        endpoint: str,
        api_key: str,
        custom_headers: Optional[Dict[str, str]] = None,
        request_template: Optional[str] = None,
        **kwargs
    ) -> str:
        """Create custom OpenAI-compatible connector."""
        headers = {
            "Authorization": f"Bearer {{credential.api_key}}",
            "Content-Type": "application/json"
        }
        
        if custom_headers:
            headers.update(custom_headers)
        
        if request_template is None:
            request_template = json.dumps({
                "model": "${parameters.model}",
                "messages": "${parameters.messages}",
                "temperature": "${parameters.temperature}",
                "max_tokens": "${parameters.max_tokens}"
            })
        
        connector_config = {
            "name": f"Custom {model} Connector",
            "description": f"Custom connector to {endpoint}",
            "version": 2,
            "protocol": "http",
            "parameters": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            "credential": {
                "api_key": api_key
            },
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": endpoint,
                    "headers": headers,
                    "request_body": request_template
                }
            ]
        }
        
        return self._create_connector_from_config(connector_config)
    
    def _create_connector_from_config(self, connector_config: Dict[str, Any]) -> str:
        """Create connector from configuration."""
        self.logger.info(f"→ Sending connector creation request to OpenSearch...")
        self.logger.debug(f"Connector config: {json.dumps(connector_config, indent=2)}")
        
        try:
            response = self.client.transport.perform_request(
                "POST",
                "/_plugins/_ml/connectors/_create",
                body=connector_config
            )
            
            connector_id = response["connector_id"]
            self.logger.info(f"✓ Connector created with ID: {connector_id}")
            return connector_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create connector: {e}")
            raise
    
    def register_model(self, connector_id: str, name: str = "RAG Model") -> str:
        """
        Register a model with OpenSearch ML.
        
        Args:
            connector_id: Connector ID
            name: Model name
            
        Returns:
            Model ID
        """
        self.logger.info("=" * 80)
        self.logger.info("📝 Registering model with OpenSearch ML")
        self.logger.info("=" * 80)
        self.logger.info(f"Model name: {name}")
        self.logger.info(f"Connector ID: {connector_id}")
        
        register_config = {
            "name": name,
            "function_name": "remote",
            "description": "Model for RAG conversational search",
            "connector_id": connector_id
        }
        
        try:
            self.logger.info("→ Sending model registration request...")
            response = self.client.transport.perform_request(
                "POST",
                "/_plugins/_ml/models/_register",
                body=register_config
            )
            
            task_id = response["task_id"]
            model_id = response.get("model_id")
            
            self.logger.info(f"✓ Model registration initiated")
            self.logger.info(f"  Task ID: {task_id}")
            self.logger.info(f"  Model ID: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def deploy_model(self, model_id: str) -> Dict[str, Any]:
        """
        Deploy a registered model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Deployment response
        """
        self.logger.info("=" * 80)
        self.logger.info("🚀 Deploying model")
        self.logger.info("=" * 80)
        self.logger.info(f"Model ID: {model_id}")
        
        try:
            self.logger.info("→ Sending deployment request...")
            response = self.client.transport.perform_request(
                "POST",
                f"/_plugins/_ml/models/{model_id}/_deploy"
            )
            
            self.logger.info(f"✓ Deployment initiated")
            self.logger.debug(f"Response: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deploy model: {e}")
            raise
    
    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get model deployment status.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model status
        """
        response = self.client.transport.perform_request(
            "GET",
            f"/_plugins/_ml/models/{model_id}"
        )
        return response
    
    def list_connectors(self) -> Dict[str, Any]:
        """
        List all connectors.
        
        Returns:
            List of connectors
        """
        response = self.client.transport.perform_request(
            "GET",
            "/_plugins/_ml/connectors/_search",
            body={"query": {"match_all": {}}}
        )
        return response
    
    def delete_connector(self, connector_id: str) -> Dict[str, Any]:
        """
        Delete a connector.
        
        Args:
            connector_id: Connector ID
            
        Returns:
            Deletion response
        """
        self.logger.info(f"Deleting connector: {connector_id}")
        response = self.client.transport.perform_request(
            "DELETE",
            f"/_plugins/_ml/connectors/{connector_id}"
        )
        return response
    
    def undeploy_model(self, model_id: str) -> Dict[str, Any]:
        """
        Undeploy a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Undeploy response
        """
        self.logger.info(f"Undeploying model: {model_id}")
        response = self.client.transport.perform_request(
            "POST",
            f"/_plugins/_ml/models/{model_id}/_undeploy"
        )
        return response
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """
        Delete a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Deletion response
        """
        self.logger.info(f"Deleting model: {model_id}")
        response = self.client.transport.perform_request(
            "DELETE",
            f"/_plugins/_ml/models/{model_id}"
        )
        return response
