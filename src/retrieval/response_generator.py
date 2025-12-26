"""Response generator module for formatting retrieval results and generating LLM answers."""

import logging
from typing import Union, List, Optional
from pathlib import Path
import openai

from ..models import (
    VideoResult,
    PDFResult,
    VideoResponse,
    PDFResponse,
    NoAnswerResponse
)
from ..config import Config


class ResponseGenerator:
    """
    Format retrieval results into structured responses and generate natural language answers.
    
    This class handles:
    - Generating natural language answers using OpenAI's LLM
    - Formatting video-based responses with timestamps and citations
    - Formatting PDF-based responses with page/paragraph citations
    - Handling no-answer scenarios
    - Converting responses to human-readable display format
    """
    
    def __init__(self, config: Config, logger: logging.Logger, knowledge_summary_path: Optional[str] = None):
        """
        Initialize the ResponseGenerator.
        
        Args:
            config: Configuration object containing LLM settings
            logger: Logger instance for logging operations
            knowledge_summary_path: Optional path to knowledge summary file
        """
        self.config = config
        self.logger = logger
        self.llm_model = config.llm_model
        self.llm_temperature = config.llm_temperature
        self.llm_max_tokens = config.llm_max_tokens
        self.knowledge_summary_path = Path(knowledge_summary_path) if knowledge_summary_path else Path("data/knowledge_summary.json")
        self._initialize_openai_client()
    
    def _initialize_openai_client(self) -> None:
        """
        Initialize OpenAI client for LLM calls.
        
        Raises:
            ValueError: If API key is not configured
        """
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.logger.info(f"Initializing OpenAI client for LLM with model: {self.llm_model}")
        openai.api_key = self.config.openai_api_key
    
    def generate_response(
        self,
        query: str,
        result: Union[VideoResult, PDFResult, List[VideoResult], List[PDFResult], None]
    ) -> Union[VideoResponse, PDFResponse, NoAnswerResponse]:
        """
        Generate structured response with LLM-generated answer from retrieval result.
        
        Args:
            query: Original user query
            result: Retrieval result (VideoResult, PDFResult, List of results, or None)
            
        Returns:
            VideoResponse if result is VideoResult or List[VideoResult]
            PDFResponse if result is PDFResult or List[PDFResult]
            NoAnswerResponse if result is None
        """
        self.logger.info(f"Generating response for query: {query[:50]}...")
        
        if result is None:
            self.logger.info("No retrieval result, returning NoAnswerResponse with knowledge summary")
            return self._generate_no_answer_response()
        
        # Handle list of results (Option 2: LLM-based selection)
        elif isinstance(result, list):
            if not result:
                self.logger.info("Empty result list, returning NoAnswerResponse with knowledge summary")
                return self._generate_no_answer_response()
            
            # Check if it's a list of VideoResult or PDFResult
            if isinstance(result[0], VideoResult):
                self.logger.info(f"Generating VideoResponse from {len(result)} video results")
                return self._generate_video_response_from_multiple(query, result)
            elif isinstance(result[0], PDFResult):
                self.logger.info(f"Generating PDFResponse from {len(result)} PDF results")
                return self._generate_pdf_response_from_multiple(query, result)
        
        # Handle single result (backward compatibility)
        elif isinstance(result, VideoResult):
            self.logger.info(f"Generating VideoResponse for video_id={result.video_id}")
            return self._generate_video_response_from_single(query, result)
        
        elif isinstance(result, PDFResult):
            self.logger.info(
                f"Generating PDFResponse for {result.pdf_filename}, "
                f"page {result.page_number}"
            )
            return self._generate_pdf_response_from_single(query, result)
        
        else:
            self.logger.error(f"Unexpected result type: {type(result)}")
            return self._generate_no_answer_response(
                message="An error occurred while generating the response."
            )
    
    def _generate_no_answer_response(self, message: str = "No relevant answer found in the knowledge base.") -> NoAnswerResponse:
        """
        Generate NoAnswerResponse with knowledge summary.
        
        Args:
            message: Custom message for the no-answer response
            
        Returns:
            NoAnswerResponse with knowledge summary included
        """
        knowledge_summary = None
        
        # Try to load knowledge summary
        try:
            if self.knowledge_summary_path.exists():
                import json
                with open(self.knowledge_summary_path, 'r') as f:
                    knowledge_summary = json.load(f)
                self.logger.info("Loaded knowledge summary for no-answer response")
            else:
                self.logger.warning(f"Knowledge summary file not found: {self.knowledge_summary_path}")
        except Exception as e:
            self.logger.error(f"Failed to load knowledge summary: {e}")
        
        return NoAnswerResponse(
            message=message,
            knowledge_summary=knowledge_summary
        )
    
    def _generate_video_response_from_single(self, query: str, result: VideoResult) -> VideoResponse:
        """Generate VideoResponse from a single VideoResult."""
        generated_answer = self.generate_answer_with_llm(
            query=query,
            context=result.transcript_snippet
        )
        
        return VideoResponse(
            answer_type="video",
            video_id=result.video_id,
            start_timestamp=result.start_timestamp,
            end_timestamp=result.end_timestamp,
            start_token_id=result.start_token_id,
            end_token_id=result.end_token_id,
            transcript_snippet=result.transcript_snippet,
            generated_answer=generated_answer,
            score=result.score,
            document_id=result.document_id
        )
    
    def _generate_pdf_response_from_single(self, query: str, result: PDFResult) -> PDFResponse:
        """Generate PDFResponse from a single PDFResult."""
        generated_answer = self.generate_answer_with_llm(
            query=query,
            context=result.source_snippet
        )
        
        return PDFResponse(
            answer_type="pdf",
            pdf_filename=result.pdf_filename,
            page_number=result.page_number,
            paragraph_index=result.paragraph_index,
            source_snippet=result.source_snippet,
            generated_answer=generated_answer,
            score=result.score,
            document_id=result.document_id,
            title=result.title  # Pass through title
        )
    
    def _generate_video_response_from_multiple(
        self,
        query: str,
        results: List[VideoResult]
    ) -> Union[VideoResponse, NoAnswerResponse]:
        """
        Generate VideoResponse from multiple VideoResults.
        Let LLM choose the best context and generate answer.
        Returns NoAnswerResponse if LLM refuses to answer.
        """
        # Combine contexts with metadata
        combined_context = self._combine_contexts_for_llm(
            query=query,
            results=results,
            result_type="video"
        )
        
        # Generate answer with LLM (it will pick the best context)
        generated_answer, best_idx = self.generate_answer_with_llm_selection(
            query=query,
            contexts=combined_context,
            num_results=len(results)
        )
        
        # Check if LLM refused to answer
        if generated_answer is None:
            self.logger.info("LLM refused to answer, returning NoAnswerResponse")
            return self._generate_no_answer_response(
                message="I couldn't find relevant information to answer your question. Please try rephrasing or asking a different question."
            )
        
        # Return response with the best result's metadata
        best_result = results[best_idx]
        
        return VideoResponse(
            answer_type="video",
            video_id=best_result.video_id,
            start_timestamp=best_result.start_timestamp,
            end_timestamp=best_result.end_timestamp,
            start_token_id=best_result.start_token_id,
            end_token_id=best_result.end_token_id,
            transcript_snippet=best_result.transcript_snippet,
            generated_answer=generated_answer,
            score=best_result.score,
            document_id=best_result.document_id
        )
    
    def _generate_pdf_response_from_multiple(
        self,
        query: str,
        results: List[PDFResult]
    ) -> Union[PDFResponse, NoAnswerResponse]:
        """
        Generate PDFResponse from multiple PDFResults.
        Let LLM choose the best context and generate answer.
        Returns NoAnswerResponse if LLM refuses to answer.
        """
        # Log the results being processed
        self.logger.info("="*80)
        self.logger.info(f"GENERATING ANSWER FROM {len(results)} PDF RESULTS:")
        self.logger.info("="*80)
        for i, result in enumerate(results, 1):
            self.logger.info(f"\nResult {i}:")
            self.logger.info(f"  PDF: {result.pdf_filename}")
            self.logger.info(f"  Page: {result.page_number}, Paragraph: {result.paragraph_index}")
            self.logger.info(f"  Title: {result.title or 'No title'}")
            self.logger.info(f"  Score: {result.score:.4f}")
            self.logger.info(f"  Snippet length: {len(result.source_snippet)} chars")
            self.logger.info(f"  Snippet preview: {result.source_snippet[:300]}...")
        self.logger.info("="*80)
        
        # Combine contexts with metadata
        combined_context = self._combine_contexts_for_llm(
            query=query,
            results=results,
            result_type="pdf"
        )
        
        self.logger.info(f"Combined context length: {len(combined_context)} chars")
        self.logger.info(f"Combined context preview:\n{combined_context[:500]}...")
        
        # Generate answer with LLM (it will pick the best context)
        generated_answer, best_idx = self.generate_answer_with_llm_selection(
            query=query,
            contexts=combined_context,
            num_results=len(results)
        )
        
        # Check if LLM refused to answer
        if generated_answer is None:
            self.logger.info("LLM refused to answer, returning NoAnswerResponse")
            return self._generate_no_answer_response(
                message="I couldn't find relevant information to answer your question. Please try rephrasing or asking a different question."
            )
        
        self.logger.info(f"LLM selected result {best_idx + 1}")
        self.logger.info(f"Generated answer: {generated_answer}")
        
        # Return response with the best result's metadata
        best_result = results[best_idx]
        
        return PDFResponse(
            answer_type="pdf",
            pdf_filename=best_result.pdf_filename,
            page_number=best_result.page_number,
            paragraph_index=best_result.paragraph_index,
            source_snippet=best_result.source_snippet,
            generated_answer=generated_answer,
            score=best_result.score,
            document_id=best_result.document_id,
            title=best_result.title  # Pass through title
        )
    
    def _combine_contexts_for_llm(
        self,
        query: str,
        results: List[Union[VideoResult, PDFResult]],
        result_type: str
    ) -> str:
        """Combine multiple contexts into a single prompt for LLM."""
        contexts = []
        
        for i, result in enumerate(results):
            if result_type == "video":
                context_text = result.transcript_snippet
            else:  # pdf
                context_text = result.source_snippet
            
            contexts.append(f"[Context {i+1}]\n{context_text}\n")
        
        return "\n".join(contexts)
    
    def generate_answer_with_llm_selection(
        self,
        query: str,
        contexts: str,
        num_results: int
    ) -> tuple[str, int]:
        """
        Use OpenAI LLM to select best context and generate answer.
        
        Returns:
            Tuple of (generated_answer, best_context_index)
        """
        self.logger.debug(f"Generating LLM answer with selection from {num_results} contexts")
        
        # Build prompt template
        prompt = f"""You are given multiple context snippets. Your task is to:
1. Identify which context (if any) best answers the question
2. Generate a concise answer based on that context

IMPORTANT: If none of the contexts contain information to answer the question, respond with "I cannot answer this question based on the provided context."

{contexts}

Question: {query}

Instructions:
- First, identify the best context number (1-{num_results}) that answers the question
- Then provide your answer based on that context
- Format: Start with "[Using Context X]" then provide the answer

Answer:"""
        
        try:
            # Call OpenAI API
            self.logger.debug(
                f"Calling OpenAI API with model={self.llm_model}, "
                f"temperature={self.llm_temperature}, "
                f"max_tokens={self.llm_max_tokens}"
            )
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate. Always indicate which context you used."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )
            
            # Extract generated answer
            answer = response['choices'][0]['message']['content'].strip()
            
            # Parse which context was used
            best_idx = 0  # Default to first
            if "[Using Context" in answer:
                try:
                    import re
                    match = re.search(r'\[Using Context (\d+)\]', answer)
                    if match:
                        best_idx = int(match.group(1)) - 1  # Convert to 0-indexed
                        # Remove the context indicator from the answer
                        answer = re.sub(r'\[Using Context \d+\]\s*', '', answer).strip()
                except:
                    pass
            
            # Check if LLM refused to answer
            refusal_phrases = [
                "i cannot answer",
                "i can't answer",
                "cannot answer this question",
                "can't answer this question",
                "not enough information",
                "insufficient information",
                "no information",
                "don't have enough",
                "doesn't contain"
            ]
            
            answer_lower = answer.lower()
            if any(phrase in answer_lower for phrase in refusal_phrases):
                self.logger.info("LLM refused to answer - returning None to trigger NoAnswerResponse")
                return None, best_idx
            
            self.logger.debug(f"LLM selected context {best_idx + 1}: {answer[:100]}...")
            self.logger.info("LLM answer generated successfully")
            
            return answer, best_idx
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM answer: {str(e)}")
            # Return a fallback message
            return "I found relevant information but couldn't generate a detailed answer. Please refer to the source snippet.", 0
    
    def generate_answer_with_llm(self, query: str, context: str) -> str:
        """
        Use OpenAI LLM to generate natural language answer from context.
        
        Args:
            query: User's question
            context: Retrieved context snippet
            
        Returns:
            Natural language answer generated by LLM
        """
        self.logger.debug(f"Generating LLM answer for query: {query[:50]}...")
        
        # Build prompt template
        prompt = f"""Based on the following context, answer the user's question concisely and accurately.

IMPORTANT: If the context does not contain information to answer the question, respond with "I cannot answer this question based on the provided context."

Context: {context}

Question: {query}

Answer:"""
        
        try:
            # Call OpenAI API
            self.logger.debug(
                f"Calling OpenAI API with model={self.llm_model}, "
                f"temperature={self.llm_temperature}, "
                f"max_tokens={self.llm_max_tokens}"
            )
            
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )
            
            # Extract generated answer
            answer = response['choices'][0]['message']['content'].strip()
            
            self.logger.debug(f"LLM generated answer: {answer[:100]}...")
            self.logger.info("LLM answer generated successfully")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM answer: {str(e)}")
            # Return a fallback message if LLM fails
            return "I found relevant information but couldn't generate a detailed answer. Please refer to the source snippet."
    
    def format_for_display(
        self,
        response: Union[VideoResponse, PDFResponse, NoAnswerResponse]
    ) -> str:
        """
        Format response for human-readable display.
        
        Args:
            response: Response object to format
            
        Returns:
            Human-readable string representation
        """
        self.logger.debug(f"Formatting response for display: {response.answer_type}")
        
        if isinstance(response, VideoResponse):
            # Format video response with timestamps and citation
            formatted = f"""Answer Type: Video
Video ID: {response.video_id}
Timestamp: {response.start_timestamp:.2f}s - {response.end_timestamp:.2f}s
Token Range: {response.start_token_id} - {response.end_token_id}

Answer:
{response.generated_answer}

Source Transcript:
"{response.transcript_snippet}"
"""
            
        elif isinstance(response, PDFResponse):
            # Format PDF response with page/paragraph citation
            title_line = f"Section: {response.title}\n" if response.title else ""
            formatted = f"""Answer Type: PDF
Document: {response.pdf_filename}
{title_line}Page: {response.page_number}
Paragraph: {response.paragraph_index}

Answer:
{response.generated_answer}

Source Text:
"{response.source_snippet}"
"""
            
        elif isinstance(response, NoAnswerResponse):
            # Format no-answer response
            formatted = f"""Answer Type: No Answer
Message: {response.message}
"""
            
        else:
            formatted = f"Unknown response type: {type(response)}"
        
        self.logger.debug("Response formatted for display")
        return formatted
