"""OpenSearch RAG Pipeline Manager."""

import logging
from typing import Dict, Any, Optional
from opensearchpy import OpenSearch


class RAGPipelineManager:
    """Manages RAG search pipelines in OpenSearch."""
    
    def __init__(self, client: OpenSearch, logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline manager.
        
        Args:
            client: OpenSearch client instance
            logger: Optional logger instance
        """
        self.client = client
        self.logger = logger or logging.getLogger(__name__)
    
    def create_rag_pipeline(
        self,
        pipeline_id: str,
        model_id: str,
        context_field: str = "text",
        system_prompt: Optional[str] = None,
        user_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a RAG search pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            model_id: Model ID to use for generation
            context_field: Field name containing the context text
            system_prompt: Optional system prompt
            user_instructions: Optional user instructions
            
        Returns:
            Pipeline creation response
        """
        self.logger.info("=" * 80)
        self.logger.info("🔧 Creating RAG search pipeline")
        self.logger.info("=" * 80)
        self.logger.info(f"Pipeline ID: {pipeline_id}")
        self.logger.info(f"Model ID: {model_id}")
        self.logger.info(f"Context field: {context_field}")
        
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        
        if user_instructions is None:
            user_instructions = "Generate a concise and informative answer in less than 100 words for the given question"
        
        self.logger.info(f"System prompt: {system_prompt[:50]}...")
        self.logger.info(f"User instructions: {user_instructions[:50]}...")
        
        pipeline_config = {
            "description": f"RAG pipeline using model {model_id}",
            "response_processors": [
                {
                    "retrieval_augmented_generation": {
                        "model_id": model_id,
                        "context_field_list": [context_field],
                        "system_prompt": system_prompt,
                        "user_instructions": user_instructions
                    }
                }
            ]
        }
        
        try:
            self.logger.info("→ Sending pipeline creation request...")
            response = self.client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{pipeline_id}",
                body=pipeline_config
            )
            
            self.logger.info(f"✓ RAG pipeline created successfully")
            self.logger.debug(f"Response: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create pipeline: {e}")
            raise
    
    def get_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Get pipeline configuration.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Pipeline configuration
        """
        response = self.client.transport.perform_request(
            "GET",
            f"/_search/pipeline/{pipeline_id}"
        )
        return response
    
    def delete_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """
        Delete a pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Deletion response
        """
        self.logger.info(f"Deleting pipeline: {pipeline_id}")
        response = self.client.transport.perform_request(
            "DELETE",
            f"/_search/pipeline/{pipeline_id}"
        )
        return response
    
    def search_with_rag(
        self,
        index: str,
        query: Dict[str, Any],
        llm_question: str,
        pipeline_id: str,
        memory_id: Optional[str] = None,
        context_size: Optional[int] = None,
        message_size: int = 10,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a search with RAG pipeline.
        
        Args:
            index: Index name
            query: OpenSearch query
            llm_question: Question for the LLM
            pipeline_id: Pipeline to use
            memory_id: Optional conversation memory ID
            context_size: Number of search results to send to LLM
            message_size: Number of conversation messages to include
            timeout: Timeout in seconds
            
        Returns:
            Search response with RAG answer
        """
        # Build generative QA parameters
        qa_params = {
            "llm_question": llm_question,
            "message_size": message_size,
            "timeout": timeout
        }
        
        if memory_id:
            qa_params["memory_id"] = memory_id
        
        if context_size:
            qa_params["context_size"] = context_size
        
        # Build search body
        search_body = {
            **query,
            "ext": {
                "generative_qa_parameters": qa_params
            }
        }
        
        self.logger.info(f"Executing RAG search on index: {index}")
        self.logger.debug(f"Question: {llm_question}")
        
        response = self.client.search(
            index=index,
            body=search_body,
            params={"search_pipeline": pipeline_id}
        )
        
        # Extract RAG answer
        rag_answer = response.get("ext", {}).get("retrieval_augmented_generation", {})
        
        if rag_answer:
            self.logger.info("RAG answer generated successfully")
            self.logger.debug(f"Answer: {rag_answer.get('answer', '')[:100]}...")
        else:
            self.logger.warning("No RAG answer in response")
        
        return response
    
    def list_pipelines(self) -> Dict[str, Any]:
        """
        List all search pipelines.
        
        Returns:
            List of pipelines
        """
        response = self.client.transport.perform_request(
            "GET",
            "/_search/pipeline"
        )
        return response
