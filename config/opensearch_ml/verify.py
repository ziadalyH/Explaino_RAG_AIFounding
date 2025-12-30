#!/usr/bin/env python3
"""Verify OpenSearch RAG setup and display current configuration."""

import sys
import logging
from pathlib import Path

from config.config import Config
from config.opensearch_ml.connector_manager import OpenSearchConnectorManager
from config.opensearch_ml.pipeline_manager import RAGPipelineManager
from opensearchpy import OpenSearch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def create_opensearch_client(config: Config) -> OpenSearch:
    """Create OpenSearch client."""
    return OpenSearch(
        hosts=[{
            'host': config.opensearch_host,
            'port': config.opensearch_port
        }],
        http_auth=(config.opensearch_user, config.opensearch_password),
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )


def verify_setup():
    """Verify OpenSearch RAG setup."""
    logger.info("=" * 80)
    logger.info("OpenSearch RAG Setup Verification")
    logger.info("=" * 80)
    logger.info("")
    
    # Check config file
    config_file = Path(".opensearch_rag_config")
    if not config_file.exists():
        logger.error("❌ .opensearch_rag_config not found")
        logger.info("Run 'python -m config.opensearch_ml.setup' to create configuration")
        return False
    
    # Load config
    logger.info("📋 Loading configuration from .opensearch_rag_config")
    config_data = {}
    with open(config_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config_data[key] = value
    
    logger.info("")
    logger.info("Configuration:")
    for key, value in config_data.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # Connect to OpenSearch
    logger.info("🔌 Connecting to OpenSearch...")
    config = Config.from_env()
    client = create_opensearch_client(config)
    
    try:
        info = client.info()
        logger.info(f"✓ Connected to OpenSearch {info['version']['number']}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to OpenSearch: {e}")
        return False
    
    logger.info("")
    
    # Create managers
    connector_manager = OpenSearchConnectorManager(client, logger)
    pipeline_manager = RAGPipelineManager(client, logger)
    
    # Verify connector
    connector_id = config_data.get("CONNECTOR_ID")
    if connector_id:
        logger.info("=" * 80)
        logger.info("📡 Verifying Connector")
        logger.info("=" * 80)
        logger.info(f"Connector ID: {connector_id}")
        
        try:
            response = client.transport.perform_request(
                "GET",
                f"/_plugins/_ml/connectors/{connector_id}"
            )
            
            logger.info(f"✓ Connector exists")
            logger.info(f"  Name: {response.get('name', 'N/A')}")
            logger.info(f"  Protocol: {response.get('protocol', 'N/A')}")
            logger.info(f"  Version: {response.get('version', 'N/A')}")
            
            # Show parameters
            params = response.get('parameters', {})
            if params:
                logger.info(f"  Parameters:")
                for key, value in params.items():
                    logger.info(f"    {key}: {value}")
            
        except Exception as e:
            logger.error(f"❌ Connector not found or error: {e}")
    
    logger.info("")
    
    # Verify model
    model_id = config_data.get("MODEL_ID")
    if model_id:
        logger.info("=" * 80)
        logger.info("🤖 Verifying Model")
        logger.info("=" * 80)
        logger.info(f"Model ID: {model_id}")
        
        try:
            status = connector_manager.get_model_status(model_id)
            
            model_state = status.get("model_state", "UNKNOWN")
            logger.info(f"✓ Model exists")
            logger.info(f"  Name: {status.get('name', 'N/A')}")
            logger.info(f"  State: {model_state}")
            logger.info(f"  Algorithm: {status.get('algorithm', 'N/A')}")
            
            if model_state == "DEPLOYED":
                logger.info(f"  ✓ Model is DEPLOYED and ready")
                
                # Test inference
                logger.info("")
                logger.info("🧪 Testing model inference...")
                try:
                    test_response = client.transport.perform_request(
                        "POST",
                        f"/_plugins/_ml/models/{model_id}/_predict",
                        body={
                            "parameters": {
                                "messages": [{"role": "user", "content": "test"}],
                                "max_tokens": 5
                            }
                        }
                    )
                    
                    if "inference_results" in test_response:
                        logger.info("  ✓ Model inference working correctly")
                    else:
                        logger.warning("  ⚠ Model responded but format unexpected")
                        
                except Exception as e:
                    logger.error(f"  ❌ Model inference failed: {e}")
            else:
                logger.warning(f"  ⚠ Model state is {model_state} (not DEPLOYED)")
            
        except Exception as e:
            logger.error(f"❌ Model not found or error: {e}")
    
    logger.info("")
    
    # Verify pipeline
    pipeline_id = config_data.get("PIPELINE_ID")
    if pipeline_id:
        logger.info("=" * 80)
        logger.info("🔧 Verifying Pipeline")
        logger.info("=" * 80)
        logger.info(f"Pipeline ID: {pipeline_id}")
        
        try:
            response = pipeline_manager.get_pipeline(pipeline_id)
            
            logger.info(f"✓ Pipeline exists")
            
            # Show pipeline details
            if pipeline_id in response:
                pipeline_config = response[pipeline_id]
                logger.info(f"  Description: {pipeline_config.get('description', 'N/A')}")
                
                # Show response processors
                processors = pipeline_config.get('response_processors', [])
                if processors:
                    logger.info(f"  Response Processors:")
                    for proc in processors:
                        if 'retrieval_augmented_generation' in proc:
                            rag_config = proc['retrieval_augmented_generation']
                            logger.info(f"    Type: RAG")
                            logger.info(f"    Model ID: {rag_config.get('model_id', 'N/A')}")
                            logger.info(f"    Context Fields: {rag_config.get('context_field_list', [])}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline not found or error: {e}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ Verification Complete")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  Provider: {config_data.get('PROVIDER', 'N/A')}")
    logger.info(f"  Model: {config_data.get('MODEL_NAME', 'N/A')}")
    logger.info(f"  Connector: {'✓' if connector_id else '❌'}")
    logger.info(f"  Model: {'✓' if model_id else '❌'}")
    logger.info(f"  Pipeline: {'✓' if pipeline_id else '❌'}")
    logger.info("")
    
    return True


if __name__ == "__main__":
    try:
        success = verify_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
