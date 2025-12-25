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
from .rag_system import RAGSystem
from .models import VideoResponse, PDFResponse, NoAnswerResponse


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
                    force_rebuild=args.force_rebuild
                )
            elif args.command == "serve":
                return self.serve_command()
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
    
    def index_command(self, force_rebuild: bool = False) -> int:
        """
        Handle index command - build or rebuild the vector index.
        
        Args:
            force_rebuild: If True, rebuild index even if it exists
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if force_rebuild:
                print("Rebuilding index from scratch...")
            else:
                print("Building index (will skip if already exists)...")
            
            self.rag_system.build_index(force_rebuild=force_rebuild)
            
            print("✓ Index building completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}", exc_info=True)
            print(f"Error: Failed to build index - {str(e)}", file=sys.stderr)
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
  
  # Build the index (skip if exists)
  python main.py index
  
  # Force rebuild the index
  python main.py index --force-rebuild
  
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
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the system as a service (for Docker deployment)"
    )
    
    return parser
