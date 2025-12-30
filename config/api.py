"""Flask API for RAG System."""

import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.rag_system import RAGSystem
from .config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config: Config = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config: Configuration object (if None, loads from environment)
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend access
    
    # Initialize RAG system
    if config is None:
        config = Config.from_env()
    
    rag_system = RAGSystem(config)
    
    # Store in app context
    app.rag_system = rag_system
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "RAG System API"
        }), 200

    @app.route('/query', methods=['POST'])
    def query():
        """
        Query endpoint for asking questions.
        
        Request body:
        {
            "question": "What is OpenStax?"
        }
        
        Response:
        {
            "answer_type": "pdf" | "video" | "no_answer",
            "answer": "Generated answer text",
            "source": {...}
        }
        """
        try:
            # Get question from request
            data = request.get_json()
            
            if not data or 'question' not in data:
                return jsonify({
                    "error": "Missing 'question' in request body"
                }), 400
            
            question = data['question']
            
            if not question or not question.strip():
                return jsonify({
                    "error": "Question cannot be empty"
                }), 400
            
            logger.info(f"Received query: {question}")
            
            # Process query through RAG system
            response = app.rag_system.answer_question(question)
            
            # Format response based on type
            if response.answer_type == "pdf":
                return jsonify({
                    "answer_type": "pdf",
                    "answer": response.generated_answer,
                    "source": {
                        "pdf_filename": response.pdf_filename,
                        "page_number": response.page_number,
                        "paragraph_index": response.paragraph_index,
                        "title": response.title,
                        "snippet": response.source_snippet,
                        "score": response.score,
                        "document_id": response.document_id
                    }
                }), 200
            
            elif response.answer_type == "video":
                return jsonify({
                    "answer_type": "video",
                    "answer": response.generated_answer,
                    "source": {
                        "video_id": response.video_id,
                        "start_timestamp": response.start_timestamp,
                        "end_timestamp": response.end_timestamp,
                        "start_token_id": response.start_token_id,
                        "end_token_id": response.end_token_id,
                        "transcript_snippet": response.transcript_snippet,
                        "score": response.score,
                        "document_id": response.document_id
                    }
                }), 200
            
            else:  # no_answer
                response_data = {
                    "answer_type": "no_answer",
                    "answer": response.message,
                    "suggestion": "The retrieved content wasn't relevant enough. Try rephrasing your question."
                }
                
                # Add knowledge summary if available
                if response.knowledge_summary:
                    response_data["knowledge_base"] = response.knowledge_summary
                
                return jsonify(response_data), 200
        
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({
                "error": str(e)
            }), 400
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Internal server error",
                "message": str(e)
            }), 500

    @app.route('/index/status', methods=['GET'])
    def index_status():
        """Check if index exists and get statistics."""
        try:
            exists = app.rag_system.check_index_exists()
            
            if exists:
                indexed_pdfs, indexed_videos = app.rag_system._get_indexed_files()
                return jsonify({
                    "index_exists": True,
                    "statistics": {
                        "total_pdfs": len(indexed_pdfs),
                        "total_videos": len(indexed_videos),
                        "pdf_files": list(indexed_pdfs),
                        "video_ids": list(indexed_videos)
                    }
                }), 200
            else:
                return jsonify({
                    "index_exists": False,
                    "message": "Index not found. Please build the index first."
                }), 200
        
        except Exception as e:
            logger.error(f"Error checking index status: {str(e)}")
            return jsonify({
                "error": "Failed to check index status",
                "message": str(e)
            }), 500

    @app.route('/index/build', methods=['POST'])
    def build_index():
        """
        Build or rebuild the index.
        
        Request body (optional):
        {
            "force_rebuild": true,
            "videos_only": false,
            "pdfs_only": false
        }
        """
        try:
            data = request.get_json() or {}
            force_rebuild = data.get('force_rebuild', False)
            videos_only = data.get('videos_only', False)
            pdfs_only = data.get('pdfs_only', False)
            
            logger.info(
                f"Starting index build (force_rebuild={force_rebuild}, "
                f"videos_only={videos_only}, pdfs_only={pdfs_only})"
            )
            
            # Build index
            app.rag_system.build_index(
                force_rebuild=force_rebuild,
                videos_only=videos_only,
                pdfs_only=pdfs_only
            )
            
            # Get final statistics
            indexed_pdfs, indexed_videos = app.rag_system._get_indexed_files()
            
            return jsonify({
                "status": "success",
                "message": "Index built successfully",
                "statistics": {
                    "total_pdfs": len(indexed_pdfs),
                    "total_videos": len(indexed_videos)
                }
            }), 200
        
        except Exception as e:
            logger.error(f"Error building index: {str(e)}", exc_info=True)
            return jsonify({
                "error": "Failed to build index",
                "message": str(e)
            }), 500

    @app.route('/knowledge/summary', methods=['GET'])
    def get_knowledge_summary():
        """
        Get knowledge summary and suggested questions.
        
        Response:
        {
            "overview": "Brief overview of topics...",
            "topics": ["Topic 1", "Topic 2", ...],
            "suggested_questions": ["Question 1?", "Question 2?", ...]
        }
        """
        try:
            summary = app.rag_system.knowledge_summary_generator.load_summary()
            
            if summary:
                return jsonify(summary), 200
            else:
                return jsonify({
                    "error": "No knowledge summary available",
                    "message": "Please build the index first to generate a knowledge summary."
                }), 404
        
        except Exception as e:
            logger.error(f"Error loading knowledge summary: {str(e)}")
            return jsonify({
                "error": "Failed to load knowledge summary",
                "message": str(e)
            }), 500
    
    return app


def run_api_server(host: str = '0.0.0.0', port: int = 8000, debug: bool = False):
    """
    Run the Flask API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    logger.info("Starting RAG System API...")
    logger.info(f"API will be available at http://{host}:{port}")
    logger.info("Endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /query - Ask a question")
    logger.info("  GET  /index/status - Check index status")
    logger.info("  POST /index/build - Build/rebuild index")
    logger.info("  GET  /knowledge/summary - Get knowledge summary")
    
    config = Config.from_env()
    app = create_app(config)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_api_server(debug=True)
