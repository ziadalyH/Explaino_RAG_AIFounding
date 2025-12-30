"""Centralized LLM inference service using OpenSearch ML connector.

This module provides a single interface for all LLM inference calls,
replacing direct OpenAI API calls throughout the system.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from opensearchpy import OpenSearch

from config.config import Config


class LLMInferenceService:
    """
    Centralized service for LLM inference using OpenSearch ML connector.
    
    This service:
    - Loads model_id from .opensearch_rag_config
    - Provides a unified interface for LLM calls
    - Handles different response formats from various providers
    - Logs all LLM interactions
    """
    
    def __init__(
        self,
        config: Config,
        opensearch_client: OpenSearch,
        logger: logging.Logger,
        model_id: Optional[str] = None
    ):
        """
        Initialize the LLM inference service.
        
        Args:
            config: Configuration object
            opensearch_client: OpenSearch client
            logger: Logger instance
            model_id: Optional model ID (loaded from config if not provided)
        """
        self.config = config
        self.client = opensearch_client
        self.logger = logger
        self._model_id = model_id
        self._model_id_loaded = False
    
    @property
    def model_id(self) -> str:
        """Get model ID, loading from config file if not already loaded."""
        if not self._model_id_loaded:
            if self._model_id is None:
                self._model_id = self._load_model_id()
                self.logger.info(f"📋 Loaded Model ID: {self._model_id}")
            self._model_id_loaded = True
        return self._model_id
    
    def _load_model_id(self) -> str:
        """Load model ID from OpenSearch RAG config file."""
        config_file = Path(".opensearch_rag_config")
        if not config_file.exists():
            raise FileNotFoundError(
                "OpenSearch RAG config not found. Run 'python -m config.opensearch_ml.setup' first."
            )
        
        with open(config_file, "r") as f:
            for line in f:
                if line.startswith("MODEL_ID="):
                    return line.strip().split("=", 1)[1]
        
        raise ValueError("MODEL_ID not found in .opensearch_rag_config")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (default: helpful assistant)
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Generated text
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        
        if temperature is None:
            temperature = self.config.llm_temperature
        
        if max_tokens is None:
            max_tokens = self.config.llm_max_tokens
        
        self.logger.info("=" * 80)
        self.logger.info(f"🤖 Using LLM Model: {self.model_id}")
        self.logger.info(f"📝 Prompt length: {len(prompt)} characters")
        self.logger.info(f"🌡️  Temperature: {temperature}")
        self.logger.info(f"📊 Max tokens: {max_tokens}")
        self.logger.info("=" * 80)
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            self.logger.info("Calling OpenSearch ML inference...")
            response = self.client.transport.perform_request(
                "POST",
                f"/_plugins/_ml/models/{self.model_id}/_predict",
                body={
                    "parameters": {
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
            )
            
            self.logger.info("✓ Received response from LLM")
            
            # Extract text from response
            text = self._extract_response_text(response)
            
            self.logger.info(f"✓ Generated text ({len(text)} characters)")
            return text
            
        except Exception as e:
            self.logger.error(f"❌ ML inference failed: {str(e)}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text with context (for RAG).
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."
        
        prompt = f"""Based on the following context, answer the user's question concisely and accurately.

IMPORTANT: If the context does not contain information to answer the question, respond with "I cannot answer this question based on the provided context."

Context: {context}

Question: {query}

Answer:"""
        
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            
        Returns:
            Parsed JSON dictionary
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that always responds with valid JSON."
        
        text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract JSON from markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        import json
        return json.loads(text)
    
    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Extract text from OpenSearch ML response."""
        if "inference_results" in response:
            results = response["inference_results"]
            if results and len(results) > 0:
                output = results[0].get("output", [])
                if output and len(output) > 0:
                    result = output[0]
                    
                    # Try different response formats
                    if "result" in result:
                        return result["result"].strip()
                    elif "response" in result:
                        return result["response"].strip()
                    elif "dataAsMap" in result:
                        data = result["dataAsMap"]
                        if "response" in data:
                            return data["response"].strip()
                        elif "choices" in data:
                            # OpenAI format
                            choices = data["choices"]
                            if choices and len(choices) > 0:
                                message = choices[0].get("message", {})
                                return message.get("content", "").strip()
        
        self.logger.error(f"Unexpected response format: {response}")
        raise ValueError("Could not extract text from LLM response")
    
    def is_available(self) -> bool:
        """
        Check if LLM inference is available and ready.
        
        This checks:
        1. Config file exists
        2. Model ID can be loaded
        3. Model exists in OpenSearch
        4. Model can actually perform inference
        
        Returns:
            True if LLM is ready, False otherwise
        """
        try:
            # Check if config file exists
            config_file = Path(".opensearch_rag_config")
            if not config_file.exists():
                self.logger.debug("LLM inference not available: .opensearch_rag_config not found")
                return False
            
            # Try to load model_id
            try:
                model_id = self.model_id
            except Exception as e:
                self.logger.debug(f"LLM inference not available: Cannot load model_id - {e}")
                return False
            
            # Check if model exists and is deployed
            try:
                status = self.client.transport.perform_request(
                    "GET",
                    f"/_plugins/_ml/models/{model_id}"
                )
                
                model_state = status.get("model_state", "UNKNOWN")
                
                if model_state != "DEPLOYED":
                    self.logger.debug(f"LLM inference not available: Model state is {model_state}")
                    return False
                
                # Try a quick test inference to ensure it's truly ready
                try:
                    self.client.transport.perform_request(
                        "POST",
                        f"/_plugins/_ml/models/{model_id}/_predict",
                        body={
                            "parameters": {
                                "messages": [{"role": "user", "content": "test"}]
                            }
                        }
                    )
                    return True
                except Exception as e:
                    error_msg = str(e)
                    if "400" in error_msg or "Bad Request" in error_msg:
                        self.logger.debug(f"LLM inference not available: Model not ready for inference - {error_msg[:100]}")
                    else:
                        self.logger.debug(f"LLM inference not available: Test inference failed - {e}")
                    return False
                    
            except Exception as e:
                self.logger.debug(f"LLM inference not available: Model check failed - {e}")
                return False
                
        except Exception as e:
            self.logger.debug(f"LLM inference not available: {e}")
            return False
