"""Retrieval engine module for searching OpenSearch."""

import logging
from typing import List, Union, Optional, Dict, Any
import numpy as np
from opensearchpy import OpenSearch

from ..models import VideoResult, PDFResult
from ..config import Config


class RetrievalEngine:
    """
    Search OpenSearch vector index and implement two-tier retrieval strategy.
    
    This class handles:
    - Searching video transcript chunks using k-NN similarity
    - Falling back to PDF chunks when video results are insufficient
    - Filtering results by relevance threshold
    - Parsing OpenSearch responses and extracting metadata
    
    Both videos and PDFs use pure k-NN search for consistent scoring (0.0 to 1.0).
    """
    
    def __init__(
        self,
        opensearch_client: OpenSearch,
        config: Config,
        logger: logging.Logger
    ):
        """
        Initialize the RetrievalEngine.
        
        Args:
            opensearch_client: OpenSearch client instance
            config: Configuration object containing retrieval settings
            logger: Logger instance for logging operations
        """
        self.opensearch_client = opensearch_client
        self.pdf_index_name = config.opensearch_pdf_index
        self.video_index_name = config.opensearch_video_index
        self.relevance_threshold = config.relevance_threshold
        self.max_results = config.max_results
        self.logger = logger
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str = ""
    ) -> Union[VideoResult, PDFResult, List[VideoResult], List[PDFResult], None]:
        """
        Execute two-tier retrieval strategy using OpenSearch k-NN search.
        
        Strategy:
        1. Search video transcript embeddings first
        2. If top result score >= relevance_threshold, return top N video results
        3. Otherwise, search PDF embeddings using k-NN search
        4. If top result score >= relevance_threshold, return top N PDF results
        5. Otherwise, return None (no answer)
        
        Both searches use pure k-NN for consistent scoring (0.0 to 1.0).
        
        Args:
            query_embedding: Query vector embedding
            query_text: Original query text for BM25 search
            
        Returns:
            List[VideoResult] if video matches found above threshold
            List[PDFResult] if PDF matches found above threshold
            None if no results above threshold
        """
        self.logger.info("Starting two-tier retrieval strategy")
        
        # Tier 1: Search video transcripts
        self.logger.debug("Searching video transcripts")
        video_results = self.search_videos(query_embedding)
        
        if video_results:
            top_video = video_results[0]
            self.logger.info(
                f"Top video result: video_id={top_video.video_id}, "
                f"score={top_video.score:.4f}, threshold={self.relevance_threshold}"
            )
            
            if top_video.score >= self.relevance_threshold:
                self.logger.info(f"Video result exceeds threshold, returning top {len(video_results)} video results")
                return video_results
            else:
                self.logger.info(
                    f"Video result below threshold ({top_video.score:.4f} < "
                    f"{self.relevance_threshold}), falling back to PDFs"
                )
        else:
            self.logger.info("No video results found, falling back to PDFs")
        
        # Tier 2: Search PDF documents with k-NN search
        self.logger.debug("Searching PDF documents with k-NN search")
        pdf_results = self.search_pdfs(query_embedding, query_text)
        
        if pdf_results:
            top_pdf = pdf_results[0]
            self.logger.info(
                f"Top PDF result: filename={top_pdf.pdf_filename}, "
                f"score={top_pdf.score:.4f}, threshold={self.relevance_threshold}"
            )
            
            if top_pdf.score >= self.relevance_threshold:
                self.logger.info(f"PDF result exceeds threshold, returning top {len(pdf_results)} PDF results")
                return pdf_results
            else:
                self.logger.info(
                    f"PDF result below threshold ({top_pdf.score:.4f} < "
                    f"{self.relevance_threshold}), no answer found"
                )
        else:
            self.logger.info("No PDF results found")
        
        self.logger.info("No results above threshold in either source")
        return None
    
    def search_videos(self, query_embedding: np.ndarray) -> List[VideoResult]:
        """
        Search video transcript chunks in OpenSearch using k-NN.
        
        Args:
            query_embedding: Query vector embedding
            
        Returns:
            List of VideoResult objects, sorted by score (descending)
        """
        self.logger.debug("Executing k-NN search for video transcripts")
        
        try:
            # Execute k-NN search with source_type filter
            raw_results = self.knn_search(
                query_embedding=query_embedding,
                source_type="video",
                k=self.max_results
            )
            
            # Parse results into VideoResult objects
            video_results = []
            for hit in raw_results:
                source = hit.get("_source", {})
                score = hit.get("_score", 0.0)
                document_id = hit.get("_id", "")
                
                video_result = VideoResult(
                    video_id=source.get("video_id", ""),
                    start_timestamp=source.get("start_timestamp", 0.0),
                    end_timestamp=source.get("end_timestamp", 0.0),
                    start_token_id=source.get("start_token_id", 0),
                    end_token_id=source.get("end_token_id", 0),
                    transcript_snippet=source.get("transcript_snippet", ""),
                    score=score,
                    document_id=document_id
                )
                video_results.append(video_result)
            
            self.logger.info(f"Found {len(video_results)} video results")
            
            # Print all results before filtering
            if video_results:
                self.logger.info("="*80)
                self.logger.info("VIDEO SEARCH RESULTS (before threshold filtering):")
                self.logger.info("="*80)
                for i, result in enumerate(video_results, 1):
                    self.logger.info(f"\nResult {i}:")
                    self.logger.info(f"  Video ID: {result.video_id}")
                    self.logger.info(f"  Score: {result.score:.4f}")
                    self.logger.info(f"  Timestamp: {result.start_timestamp:.2f}s - {result.end_timestamp:.2f}s")
                    self.logger.info(f"  Tokens: {result.start_token_id} - {result.end_token_id}")
                    self.logger.info(f"  Snippet: {result.transcript_snippet[:200]}...")
                self.logger.info("="*80)
            
            # Filter by threshold
            filtered_results = self.filter_by_threshold(video_results, self.relevance_threshold)
            self.logger.info(
                f"After threshold filtering ({self.relevance_threshold}): {len(filtered_results)} video results"
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching videos: {str(e)}")
            return []
    
    def search_pdfs(self, query_embedding: np.ndarray, query_text: str = "") -> List[PDFResult]:
        """
        Search PDF document chunks in OpenSearch using k-NN search.
        
        Args:
            query_embedding: Query vector embedding
            query_text: Original query text (not used, kept for backward compatibility)
            
        Returns:
            List of PDFResult objects, sorted by score (descending)
        """
        self.logger.debug("Executing k-NN search for PDF documents")
        
        try:
            # Execute k-NN search with source_type filter
            raw_results = self.knn_search(
                query_embedding=query_embedding,
                source_type="pdf",
                k=self.max_results
            )
            
            # Parse results into PDFResult objects
            pdf_results = []
            for hit in raw_results:
                source = hit.get("_source", {})
                score = hit.get("_score", 0.0)
                document_id = hit.get("_id", "")
                
                pdf_result = PDFResult(
                    pdf_filename=source.get("pdf_filename", ""),
                    page_number=source.get("page_number", 0),
                    paragraph_index=source.get("paragraph_index", 0),
                    source_snippet=source.get("text", ""),
                    score=score,
                    document_id=document_id,
                    title=source.get("title")  # Extract title
                )
                pdf_results.append(pdf_result)
            
            self.logger.info(f"Found {len(pdf_results)} PDF results")
            
            # Print all results before filtering
            if pdf_results:
                self.logger.info("="*80)
                self.logger.info("PDF SEARCH RESULTS (before threshold filtering):")
                self.logger.info("="*80)
                for i, result in enumerate(pdf_results, 1):
                    self.logger.info(f"\nResult {i}:")
                    self.logger.info(f"  PDF: {result.pdf_filename}")
                    self.logger.info(f"  Score: {result.score:.4f}")
                    self.logger.info(f"  Page: {result.page_number}, Paragraph: {result.paragraph_index}")
                    self.logger.info(f"  Title: {result.title or 'No title'}")
                    self.logger.info(f"  Snippet: {result.source_snippet[:200]}...")
                self.logger.info("="*80)
            
            # Filter by threshold
            filtered_results = self.filter_by_threshold(pdf_results, self.relevance_threshold)
            self.logger.info(
                f"After threshold filtering ({self.relevance_threshold}): {len(filtered_results)} PDF results"
            )
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error searching PDFs: {str(e)}")
            return []
    
    def knn_search(
        self,
        query_embedding: np.ndarray,
        source_type: str,
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform k-NN search in OpenSearch using separate indices.
        
        Args:
            query_embedding: Query vector embedding
            source_type: Source type to search ("video" or "pdf")
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of raw hit dictionaries from OpenSearch response
        """
        # Determine which index to search
        index_name = self.video_index_name if source_type == "video" else self.pdf_index_name
        
        self.logger.debug(
            f"Executing k-NN search: index={index_name}, k={k}"
        )
        
        # Convert numpy array to list for JSON serialization
        query_vector = query_embedding.tolist()
        
        # Build k-NN query (no filter needed since we have separate indices)
        query_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        
        try:
            # Execute search
            response = self.opensearch_client.search(
                index=index_name,
                body=query_body
            )
            
            # Extract hits
            hits = response.get("hits", {}).get("hits", [])
            
            self.logger.debug(
                f"k-NN search returned {len(hits)} results from {index_name}"
            )
            
            return hits
            
        except Exception as e:
            self.logger.error(
                f"k-NN search failed for index {index_name}: {str(e)}"
            )
            raise
    
    def filter_by_threshold(
        self,
        results: List[Union[VideoResult, PDFResult]],
        threshold: float
    ) -> List[Union[VideoResult, PDFResult]]:
        """
        Filter results below relevance threshold.
        
        Args:
            results: List of VideoResult or PDFResult objects
            threshold: Minimum score threshold
            
        Returns:
            Filtered list containing only results with score >= threshold
        """
        if not results:
            return []
        
        filtered = [r for r in results if r.score >= threshold]
        
        self.logger.debug(
            f"Filtered {len(results)} results to {len(filtered)} "
            f"above threshold {threshold}"
        )
        
        return filtered
