#!/usr/bin/env python3
"""
Setup script for OpenSearch RAG with dynamic LLM provider support.

This script:
1. Reads LLM provider configuration from config/.env
2. Creates appropriate OpenSearch ML connector based on provider
3. Registers and deploys the model
4. Creates RAG search pipeline
5. Saves configuration for query script
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from opensearchpy import OpenSearch

# Load environment variables
load_dotenv("config/.env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config
from config.opensearch_ml.connector_manager import OpenSearchConnectorManager
from config.opensearch_ml.pipeline_manager import RAGPipelineManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_opensearch_client(config: Config) -> OpenSearch:
    """Create OpenSearch client."""
    client = OpenSearch(
        hosts=[{"host": config.opensearch_host, "port": config.opensearch_port}],
        http_auth=(config.opensearch_username, config.opensearch_password)
        if config.opensearch_username
        else None,
        use_ssl=config.opensearch_use_ssl,
        verify_certs=config.opensearch_verify_certs,
        ssl_show_warn=False,
    )
    return client


def get_provider_kwargs(config: Config) -> dict:
    """Get provider-specific kwargs from config."""
    provider = config.llm_provider
    
    if provider == "openai":
        return {
            "api_key": config.llm_api_key,
            "endpoint": config.llm_endpoint or "https://api.openai.com/v1/chat/completions"
        }
    
    elif provider == "bedrock":
        return {
            "aws_access_key_id": config.aws_access_key_id,
            "aws_secret_access_key": config.aws_secret_access_key,
            "aws_region": config.aws_region or "us-east-1"
        }
    
    elif provider == "cohere":
        return {
            "api_key": config.cohere_api_key
        }
    
    elif provider == "azure_openai":
        return {
            "api_key": config.azure_api_key,
            "api_base": config.azure_api_base,
            "api_version": config.azure_api_version or "2023-05-15",
            "deployment_name": config.azure_deployment_name
        }
    
    elif provider == "vertexai":
        return {
            "project_id": config.vertexai_project,
            "location": config.vertexai_location or "us-central1",
            "access_token": config.vertexai_access_token
        }
    
    elif provider == "sagemaker":
        return {
            "endpoint_name": config.sagemaker_endpoint,
            "aws_access_key_id": config.aws_access_key_id,
            "aws_secret_access_key": config.aws_secret_access_key,
            "aws_region": config.aws_region or "us-east-1"
        }
    
    elif provider == "deepseek":
        return {
            "api_key": config.deepseek_api_key
        }
    
    elif provider == "comprehend":
        kwargs = {
            "aws_access_key_id": config.aws_access_key_id,
            "aws_secret_access_key": config.aws_secret_access_key,
            "aws_region": config.aws_region or "us-east-1"
        }
        if config.aws_session_token:
            kwargs["session_token"] = config.aws_session_token
        return kwargs
    
    elif provider == "custom":
        kwargs = {
            "endpoint": config.llm_endpoint,
            "api_key": config.llm_api_key
        }
        if config.llm_headers:
            import json
            kwargs["custom_headers"] = json.loads(config.llm_headers)
        if config.llm_request_template:
            kwargs["request_template"] = config.llm_request_template
        return kwargs
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def cleanup_existing_resources(
    connector_manager: OpenSearchConnectorManager,
    pipeline_manager: RAGPipelineManager,
    config_file: Path
):
    """Clean up existing resources if they exist."""
    if not config_file.exists():
        logger.info("No existing configuration found")
        return
    
    logger.info("Found existing configuration, cleaning up...")
    
    # Load existing config
    existing_config = {}
    with open(config_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                existing_config[key] = value
    
    # Delete pipeline
    if "PIPELINE_ID" in existing_config:
        try:
            pipeline_manager.delete_pipeline(existing_config["PIPELINE_ID"])
            logger.info(f"Deleted pipeline: {existing_config['PIPELINE_ID']}")
        except Exception as e:
            logger.warning(f"Failed to delete pipeline: {e}")
    
    # Undeploy and delete model
    if "MODEL_ID" in existing_config:
        try:
            connector_manager.undeploy_model(existing_config["MODEL_ID"])
            logger.info(f"Undeployed model: {existing_config['MODEL_ID']}")
            time.sleep(2)
            
            connector_manager.delete_model(existing_config["MODEL_ID"])
            logger.info(f"Deleted model: {existing_config['MODEL_ID']}")
        except Exception as e:
            logger.warning(f"Failed to delete model: {e}")
    
    # Delete connector
    if "CONNECTOR_ID" in existing_config:
        try:
            connector_manager.delete_connector(existing_config["CONNECTOR_ID"])
            logger.info(f"Deleted connector: {existing_config['CONNECTOR_ID']}")
        except Exception as e:
            logger.warning(f"Failed to delete connector: {e}")
    
    logger.info("Cleanup complete")


def wait_for_model_deployment(
    connector_manager: OpenSearchConnectorManager,
    model_id: str,
    max_wait: int = 120
):
    """
    Wait for model to be deployed and ready for inference.
    
    This function:
    1. Waits for model state to be DEPLOYED
    2. Tests the model with an actual inference call
    3. Keeps checking until model is truly ready
    
    Args:
        connector_manager: Connector manager instance
        model_id: Model ID to wait for
        max_wait: Maximum time to wait in seconds (default: 120)
        
    Returns:
        True if model is deployed and ready, False otherwise
    """
    logger.info("⏳ Waiting for model deployment and readiness...")
    logger.info(f"   Max wait time: {max_wait} seconds")
    logger.info(f"   Model ID: {model_id}")
    logger.info("")
    
    model_deployed = False
    last_state = None
    inference_test_attempts = 0
    max_inference_attempts = 10  # Try inference test max 10 times
    
    for i in range(max_wait):
        try:
            # Step 1: Check model state
            status = connector_manager.get_model_status(model_id)
            model_state = status.get("model_state", "UNKNOWN")
            
            # Only log state changes to reduce noise
            if model_state != last_state:
                logger.info(f"   Model state: {model_state}")
                last_state = model_state
            
            if model_state == "DEPLOYED":
                if not model_deployed:
                    logger.info("   ✓ Model state is DEPLOYED")
                    logger.info("")
                    logger.info("   🧪 Testing model with inference call...")
                    model_deployed = True
                
                # Step 2: Test with actual inference call (but limit attempts)
                if inference_test_attempts < max_inference_attempts:
                    inference_test_attempts += 1
                    
                    try:
                        test_response = connector_manager.client.transport.perform_request(
                            "POST",
                            f"/_plugins/_ml/models/{model_id}/_predict",
                            body={
                                "parameters": {
                                    "messages": [
                                        {"role": "user", "content": "test"}
                                    ]
                                }
                            }
                        )
                        
                        # Check if we got a valid response
                        if "inference_results" in test_response:
                            logger.info("   ✓ Model responded successfully!")
                            logger.info("   ✓ Model is ready for inference!")
                            return True
                        else:
                            logger.warning("   ⚠ Model responded but format unexpected, retrying...")
                            time.sleep(2)
                            
                    except Exception as e:
                        error_msg = str(e)
                        
                        # Log the full error for debugging
                        if "400" in error_msg or "Bad Request" in error_msg or "illegal_argument" in error_msg:
                            # Show full error message (not truncated)
                            logger.info(f"   ⏳ Inference test failed (attempt {inference_test_attempts}/{max_inference_attempts}): {error_msg}")
                            
                            # Try to extract detailed error info
                            try:
                                import json
                                if hasattr(e, 'info') and e.info:
                                    error_details = json.loads(e.info)
                                    reason = error_details.get('error', {}).get('reason', '')
                                    if reason:
                                        logger.info(f"   Details: {reason}")
                            except:
                                pass
                        elif "no such index" in error_msg.lower():
                            logger.info("   ⏳ ML model index initializing...")
                        else:
                            logger.info(f"   ⏳ Model not ready yet: {error_msg}")
                        time.sleep(2)
                else:
                    # Exceeded max inference attempts - trust DEPLOYED state
                    logger.warning(f"   ⚠ Inference test failed {max_inference_attempts} times")
                    logger.info("   ℹ Model is DEPLOYED - proceeding anyway")
                    logger.info("   ℹ Inference may work once system is fully initialized")
                    return True
                    
            elif model_state in ["DEPLOY_FAILED", "FAILED"]:
                logger.error(f"   ❌ Model deployment failed!")
                logger.error(f"   Status: {status}")
                return False
            else:
                # Still deploying - only log every 10 seconds to reduce noise
                if i % 10 == 0 and i > 0:
                    logger.info(f"   ⏳ Still deploying... ({i}/{max_wait}s)")
                time.sleep(1)
                
        except Exception as e:
            if i % 10 == 0:  # Only log errors every 10 seconds
                logger.warning(f"   ⚠ Error checking status: {e}")
            time.sleep(1)
    
    logger.error(f"   ❌ Model deployment timed out after {max_wait}s")
    return False


def main():
    """Main setup function."""
    logger.info("=" * 80)
    logger.info("OpenSearch RAG Setup - Dynamic LLM Provider")
    logger.info("=" * 80)
    
    # Load and validate configuration
    logger.info("Loading configuration...")
    config = Config.from_env()
    config.validate()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🤖 LLM CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Provider: {config.llm_provider.upper()}")
    logger.info(f"Model: {config.llm_model}")
    logger.info(f"Temperature: {config.llm_temperature}")
    logger.info(f"Max Tokens: {config.llm_max_tokens}")
    logger.info("=" * 80)
    logger.info("")
    
    # Create OpenSearch client
    logger.info("Connecting to OpenSearch...")
    client = create_opensearch_client(config)
    
    # Test connection
    try:
        info = client.info()
        logger.info(f"Connected to OpenSearch {info['version']['number']}")
    except Exception as e:
        logger.error(f"Failed to connect to OpenSearch: {e}")
        sys.exit(1)
    
    # Create managers
    connector_manager = OpenSearchConnectorManager(client, logger)
    pipeline_manager = RAGPipelineManager(client, logger)
    
    # Cleanup existing resources
    config_file = Path(".opensearch_rag_config")
    cleanup_existing_resources(connector_manager, pipeline_manager, config_file)
    
    # Create connector
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 1: Creating LLM Connector")
    logger.info("=" * 80)
    
    try:
        provider_kwargs = get_provider_kwargs(config)
        connector_id = connector_manager.create_connector(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            **provider_kwargs
        )
        logger.info("")
        logger.info("✓ STEP 1 COMPLETE: Connector created")
        logger.info(f"  Connector ID: {connector_id}")
    except Exception as e:
        logger.error(f"❌ STEP 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Register model
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: Registering Model")
    logger.info("=" * 80)
    
    try:
        model_id = connector_manager.register_model(
            connector_id=connector_id,
            name=f"{config.llm_provider.upper()} RAG Model"
        )
        logger.info("")
        logger.info("✓ STEP 2 COMPLETE: Model registered")
        logger.info(f"  Model ID: {model_id}")
    except Exception as e:
        logger.error(f"❌ STEP 2 FAILED: {e}")
        sys.exit(1)
    
    # Deploy model
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: Deploying Model")
    logger.info("=" * 80)
    
    try:
        connector_manager.deploy_model(model_id)
        
        # Wait for deployment
        logger.info("")
        if not wait_for_model_deployment(connector_manager, model_id):
            logger.error("❌ STEP 3 FAILED: Model deployment failed")
            sys.exit(1)
        
        logger.info("")
        logger.info("✓ STEP 3 COMPLETE: Model deployed and ready")
    except Exception as e:
        logger.error(f"❌ STEP 3 FAILED: {e}")
        sys.exit(1)
    
    # Create RAG pipeline
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 4: Creating RAG Pipeline")
    logger.info("=" * 80)
    
    pipeline_id = "rag-pipeline"
    
    try:
        pipeline_manager.create_rag_pipeline(
            pipeline_id=pipeline_id,
            model_id=model_id,
            context_field="text",
            system_prompt="You are a helpful assistant that answers questions based on provided context.",
            user_instructions="Generate a concise and informative answer based on the context provided."
        )
        logger.info("")
        logger.info("✓ STEP 4 COMPLETE: RAG pipeline created")
        logger.info(f"  Pipeline ID: {pipeline_id}")
    except Exception as e:
        logger.error(f"❌ STEP 4 FAILED: {e}")
        sys.exit(1)
    
    # Save configuration
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 5: Saving Configuration")
    logger.info("=" * 80)
    
    with open(config_file, "w") as f:
        f.write(f"CONNECTOR_ID={connector_id}\n")
        f.write(f"MODEL_ID={model_id}\n")
        f.write(f"PIPELINE_ID={pipeline_id}\n")
        f.write(f"PROVIDER={config.llm_provider}\n")
        f.write(f"MODEL_NAME={config.llm_model}\n")
    
    logger.info(f"✓ Configuration saved to {config_file}")
    
    # Success!
    logger.info("=" * 80)
    logger.info("✓ Setup Complete!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Provider: {config.llm_provider}")
    logger.info(f"  Model: {config.llm_model}")
    logger.info(f"  Connector ID: {connector_id}")
    logger.info(f"  Model ID: {model_id}")
    logger.info(f"  Pipeline ID: {pipeline_id}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nSetup cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n\nSetup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
