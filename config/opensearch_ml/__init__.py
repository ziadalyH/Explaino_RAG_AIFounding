"""
OpenSearch ML Infrastructure Module

This module contains all OpenSearch ML Commons setup and management code:
- Connector creation for various LLM providers
- Model registration and deployment
- RAG pipeline configuration
- Setup and verification scripts

Usage:
    # Setup OpenSearch ML infrastructure
    python -m config.opensearch_ml.setup
    
    # Verify setup
    python -m config.opensearch_ml.verify
    
    # Use in code
    from config.opensearch_ml.connector_manager import OpenSearchConnectorManager
    from config.opensearch_ml.pipeline_manager import RAGPipelineManager
"""

from .connector_manager import OpenSearchConnectorManager
from .pipeline_manager import RAGPipelineManager

__all__ = [
    'OpenSearchConnectorManager',
    'RAGPipelineManager',
]
