"""Command-line interface for the RAG Chatbot Backend.

This module provides a CLI for querying the chatbot, building/rebuilding the index,
and running the system as a service.
"""

import argparse
import sys
import logging
import json
from typing import Optional, Union

from .config import Config
from src.rag_system import RAGSystem
from src.models import VideoResponse, PDFResponse, NoAnswerResponse


class CLI:
    """
    Command-line interface for the RAG Chatbot Backend.
    
    Provides commands for:
    - query: Ask questions and get answers with citations
    - index: Build or rebuild the vector index
    - serve: Run as a service (for Docker deployment)
    """
    
    def __init__(self, config: Config):
        """
        Initialize CLI with configuration.
        
        Args:
            config: Configuration object for the RAG system
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for CLI operations.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self.config.log_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run(self, args: argparse.Namespace) -> int:
        """
        Execute CLI command based on parsed arguments.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if args.command == "query":
                return self.query_command(
                    question=args.question,
                    verbose=args.verbose
                )
            elif args.command == "index":
                return self.index_command(
                    force_rebuild=args.force_rebuild,
                    videos_only=args.videos_only,
                    pdfs_only=args.pdfs_only
                )
            elif args.command == "serve":
                return self.serve_command()
            elif args.command == "status":
                return self.status_command(detailed=args.detailed)
            else:
                self.logger.error(f"Unknown command: {args.command}")
                return 1
                
        except KeyboardInterrupt:
            self.logger.info("\nOperation cancelled by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=args.debug if hasattr(args, 'debug') else False)
            return 1
    
    def query_command(self, question: str, verbose: bool = False) -> int:
        """
        Handle query command - ask a question and display the answer.
        
        Args:
            question: User's question
            verbose: If True, display additional debug information
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Validate question
            if not question or not question.strip():
                print("Error: Question cannot be empty", file=sys.stderr)
                return 1
            
            if verbose:
                self.logger.info(f"Processing question: {question}")
            
            # Get answer from RAG system
            response = self.rag_system.answer_question(question)
            
            # Format and display response
            formatted_response = self._format_response(response, verbose)
            print(formatted_response)
            
            return 0
            
        except ValueError as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            return 1
        except Exception as e:
            self.logger.error(f"Failed to process query: {str(e)}", exc_info=verbose)
            print(f"Error: Failed to process query - {str(e)}", file=sys.stderr)
            return 1
    
    def index_command(
        self, 
        force_rebuild: bool = False,
        videos_only: bool = False,
        pdfs_only: bool = False
    ) -> int:
        """
        Handle index command - build or rebuild the vector index.
        
        Args:
            force_rebuild: If True, rebuild index even if it exists
            videos_only: If True, only index video transcripts
            pdfs_only: If True, only index PDF documents
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Validate mutually exclusive options
            if videos_only and pdfs_only:
                print("Error: Cannot use --videos-only and --pdfs-only together", file=sys.stderr)
                return 1
            
            # Determine what to index
            if videos_only:
                print("Indexing video transcripts only...")
                index_type = "videos"
            elif pdfs_only:
                print("Indexing PDF documents only...")
                index_type = "pdfs"
            else:
                print("Indexing both videos and PDFs...")
                index_type = "all"
            
            if force_rebuild:
                print(f"Force rebuild enabled - will reindex {index_type}")
            else:
                print(f"Incremental indexing - will only process new/modified {index_type}")
            
            self.rag_system.build_index(
                force_rebuild=force_rebuild,
                videos_only=videos_only,
                pdfs_only=pdfs_only
            )
            
            print(f"✓ Index building completed successfully ({index_type})")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}", exc_info=True)
            print(f"Error: Failed to build index - {str(e)}", file=sys.stderr)
            return 1
    
    def status_command(self, detailed: bool = False) -> int:
        """
        Handle status command - show system configuration and status.
        
        Args:
            detailed: If True, show detailed configuration
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            print("=" * 80)
            print("RAG SYSTEM STATUS")
            print("=" * 80)
            print()
            
            # Embedding Configuration
            print("🔧 EMBEDDING CONFIGURATION")
            print("-" * 80)
            print(f"Provider:  {self.config.embedding_provider}")
            print(f"Model:     {self.config.embedding_model}")
            print(f"Dimension: {self.config.embedding_dimension}")
            print()
            
            # LLM Configuration
            print("🤖 LLM CONFIGURATION")
            print("-" * 80)
            print(f"Model:       {self.config.llm_model}")
            print(f"Temperature: {self.config.llm_temperature}")
            print(f"Max Tokens:  {self.config.llm_max_tokens}")
            print()
            
            # Data Paths
            print("📁 DATA PATHS")
            print("-" * 80)
            print(f"PDFs:        {self.config.pdf_dir}")
            print(f"Transcripts: {self.config.transcript_dir}")
            print()
            
            # OpenSearch Configuration
            print("🔍 OPENSEARCH CONFIGURATION")
            print("-" * 80)
            print(f"Host:        {self.config.opensearch_host}:{self.config.opensearch_port}")
            print(f"PDF Index:   {self.config.opensearch_pdf_index}")
            print(f"Video Index: {self.config.opensearch_video_index}")
            print(f"SSL:         {self.config.opensearch_use_ssl}")
            print()
            
            # Retrieval Configuration
            print("⚙️  RETRIEVAL CONFIGURATION")
            print("-" * 80)
            print(f"Relevance Threshold: {self.config.relevance_threshold}")
            print(f"Max Results:         {self.config.max_results}")
            print()
            
            if detailed:
                # Chunking Configuration
                print("📄 CHUNKING CONFIGURATION")
                print("-" * 80)
                print(f"Strategy:     {self.config.chunking_strategy}")
                print(f"Chunk Size:   {self.config.chunk_size}")
                print(f"Chunk Overlap: {self.config.chunk_overlap}")
                print(f"Max Window:   {self.config.max_chunk_window}")
                print()
                
                # Check if model is actually loaded
                try:
                    from .processing.embedding import EmbeddingEngine
                    engine = EmbeddingEngine(self.config, self.logger)
                    
                    if hasattr(engine, 'local_model'):
                        print("✅ MODEL VERIFICATION")
                        print("-" * 80)
                        actual_dim = engine.local_model.get_sentence_embedding_dimension()
                        max_seq = engine.local_model.max_seq_length
                        print(f"Model Loaded:        Yes")
                        print(f"Actual Dimension:    {actual_dim}")
                        print(f"Max Sequence Length: {max_seq} tokens")
                        print(f"Dimension Match:     {'✅ Yes' if actual_dim == self.config.embedding_dimension else '❌ No'}")
                        print()
                except Exception as e:
                    print("⚠️  MODEL VERIFICATION")
                    print("-" * 80)
                    print(f"Could not load model: {str(e)}")
                    print()
            
            print("=" * 80)
            print("✅ System configuration loaded successfully")
            print()
            print("💡 Tip: Use --detailed flag for more information")
            print("=" * 80)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Status command failed: {str(e)}")
            print(f"Error: Failed to get status - {str(e)}", file=sys.stderr)
            return 1
    
    def serve_command(self) -> int:
        """
        Handle serve command - run as a service (for Docker).
        
        This command keeps the process running and can be extended
        to include a REST API or other service functionality.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            print("RAG Chatbot Backend service started")
            print("System is ready to process queries")
            print("Press Ctrl+C to stop")
            
            # Keep the process running
            # In a real implementation, this would start a web server or message queue listener
            import time
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nService stopped")
            return 0
        except Exception as e:
            self.logger.error(f"Service error: {str(e)}", exc_info=True)
            print(f"Error: Service failed - {str(e)}", file=sys.stderr)
            return 1
    
    def _format_response(
        self,
        response: Union[VideoResponse, PDFResponse, NoAnswerResponse],
        verbose: bool = False
    ) -> str:
        """
        Format response for human-readable display.
        
        Args:
            response: Response object from RAG system
            verbose: If True, include additional details
            
        Returns:
            Formatted string for display
        """
        lines = []
        lines.append("=" * 80)
        
        if isinstance(response, VideoResponse):
            # Check if LLM couldn't answer
            is_no_answer = "cannot answer" in response.generated_answer.lower()
            
            if is_no_answer:
                lines.append("❌ NO RELEVANT ANSWER FOUND")
                lines.append("=" * 80)
                lines.append(f"\n{response.generated_answer}\n")
                lines.append("Suggestion: The retrieved content wasn't relevant enough.")
                lines.append("            Try rephrasing your question.")
            else:
                lines.append("📹 VIDEO ANSWER")
                lines.append("=" * 80)
                lines.append(f"\nAnswer: {response.generated_answer}\n")
                lines.append("-" * 80)
                lines.append("Source Information:")
                lines.append(f"  Video ID: {response.video_id}")
                lines.append(f"  Timestamp: {response.start_timestamp:.2f}s - {response.end_timestamp:.2f}s")
                lines.append(f"  Token Range: {response.start_token_id} - {response.end_token_id}")
                lines.append(f"  Relevance Score: {response.score:.4f}")
                lines.append(f"  Document ID: {response.document_id}")
                
                if verbose:
                    lines.append(f"\nTranscript Snippet:")
                    lines.append(f"  {response.transcript_snippet}")
            
        elif isinstance(response, PDFResponse):
            # Check if LLM couldn't answer
            is_no_answer = "cannot answer" in response.generated_answer.lower()
            
            if is_no_answer:
                lines.append("❌ NO RELEVANT ANSWER FOUND")
                lines.append("=" * 80)
                lines.append(f"\n{response.generated_answer}\n")
                lines.append("Suggestion: The retrieved content wasn't relevant enough.")
                lines.append("            Try rephrasing your question.")
            else:
                lines.append("📄 PDF ANSWER")
                lines.append("=" * 80)
                lines.append(f"\nAnswer: {response.generated_answer}\n")
                lines.append("-" * 80)
                lines.append("Source Information:")
                lines.append(f"  PDF File: {response.pdf_filename}")
                lines.append(f"  Page: {response.page_number}")
                lines.append(f"  Paragraph: {response.paragraph_index}")
                lines.append(f"  Relevance Score: {response.score:.4f}")
                lines.append(f"  Document ID: {response.document_id}")
                
                if verbose:
                    lines.append(f"\nSource Snippet:")
                    lines.append(f"  {response.source_snippet}")
            
        elif isinstance(response, NoAnswerResponse):
            lines.append("❌ NO ANSWER FOUND")
            lines.append("=" * 80)
            lines.append(f"\n{response.message}\n")
            
            # Display knowledge summary if available
            if response.knowledge_summary:
                lines.append("-" * 80)
                lines.append("📚 AVAILABLE KNOWLEDGE BASE:")
                lines.append("-" * 80)
                
                summary = response.knowledge_summary
                
                # Show indexed content
                if "indexed_content" in summary:
                    content = summary["indexed_content"]
                    lines.append(f"\n📄 PDFs: {content.get('total_pdfs', 0)} documents")
                    if content.get('pdf_files'):
                        for pdf in content['pdf_files'][:5]:  # Show first 5
                            lines.append(f"  • {pdf}")
                        if len(content.get('pdf_files', [])) > 5:
                            lines.append(f"  ... and {len(content['pdf_files']) - 5} more")
                    
                    lines.append(f"\n📹 Videos: {content.get('total_videos', 0)} transcripts")
                    if content.get('video_ids'):
                        for vid in content['video_ids'][:5]:  # Show first 5
                            lines.append(f"  • {vid}")
                        if len(content.get('video_ids', [])) > 5:
                            lines.append(f"  ... and {len(content['video_ids']) - 5} more")
                
                # Show suggested questions
                if "suggested_questions" in summary and summary["suggested_questions"]:
                    lines.append("\n💡 TRY ASKING:")
                    for i, q in enumerate(summary["suggested_questions"][:5], 1):
                        lines.append(f"  {i}. {q}")
                lines.append("")
            
            lines.append("Suggestion: Try rephrasing your question or check if relevant")
            lines.append("            content has been indexed.")
        
        lines.append("=" * 80)
        
        if verbose:
            lines.append("\nJSON Response:")
            lines.append(json.dumps(response.__dict__, indent=2))
        
        return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="RAG Chatbot Backend - Answer questions from video transcripts and PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query the chatbot
  python main.py query --question "How do I add a new customer?"
  
  # Query with verbose output
  python main.py query -q "What is the pricing?" --verbose
  
  # Check system status and configuration
  python main.py status
  
  # Check detailed status (including model verification)
  python main.py status --detailed
  
  # Build the index (incremental - only new files)
  python main.py index
  
  # Force rebuild the entire index
  python main.py index --force-rebuild
  
  # Reindex only videos
  python main.py index --force-rebuild --videos-only
  
  # Reindex only PDFs
  python main.py index --force-rebuild --pdfs-only
  
  # Run as a service (for Docker)
  python main.py serve
        """
    )
    
    # Add global flags
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed error messages"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a question and get an answer with citations"
    )
    query_parser.add_argument(
        "--question", "-q",
        required=True,
        help="Question to ask the chatbot"
    )
    query_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Display verbose output including source snippets and JSON"
    )
    
    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Build or rebuild the vector index from data files"
    )
    index_parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of index even if it already exists"
    )
    index_parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Only index video transcripts (skip PDFs)"
    )
    index_parser.add_argument(
        "--pdfs-only",
        action="store_true",
        help="Only index PDF documents (skip videos)"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the system as a service (for Docker deployment)"
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show system status and configuration"
    )
    status_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed configuration information"
    )
    
    return parser
