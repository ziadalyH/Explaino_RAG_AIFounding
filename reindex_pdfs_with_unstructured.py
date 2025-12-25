"""
Reindex PDFs using Unstructured chunking by title.

This script:
1. Deletes only PDF documents from OpenSearch (preserves video data)
2. Re-ingests PDFs using Unstructured's chunking by title
3. Re-indexes the new chunks

This approach keeps headings together with their content for better retrieval.
"""

import logging
from pathlib import Path
from opensearchpy import OpenSearch

from src.config import Config
from src.ingestion.pdf_ingester import PDFIngester
from src.processing.chunking import ChunkingModule
from src.processing.embedding import EmbeddingEngine
from src.processing.indexing import VectorIndexBuilder


def setup_logger():
    """Set up logger for the reindexing process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def delete_pdf_documents(opensearch_client: OpenSearch, index_name: str, logger: logging.Logger):
    """
    Delete only PDF documents from OpenSearch index.
    
    Args:
        opensearch_client: OpenSearch client instance
        index_name: Name of the index
        logger: Logger instance
    """
    logger.info("Deleting PDF documents from OpenSearch...")
    
    try:
        # Delete by query - only documents with source_type="pdf"
        delete_query = {
            "query": {
                "term": {
                    "source_type": "pdf"
                }
            }
        }
        
        response = opensearch_client.delete_by_query(
            index=index_name,
            body=delete_query
        )
        
        deleted_count = response.get('deleted', 0)
        logger.info(f"Successfully deleted {deleted_count} PDF documents")
        
        # Refresh index to make changes visible
        opensearch_client.indices.refresh(index=index_name)
        logger.info("Index refreshed")
        
    except Exception as e:
        logger.error(f"Error deleting PDF documents: {e}")
        raise


def main():
    """Main reindexing process."""
    logger = setup_logger()
    
    logger.info("=" * 80)
    logger.info("Starting PDF reindexing with Unstructured chunking by title")
    logger.info("=" * 80)
    
    # Load configuration
    config = Config.from_env()
    logger.info(f"Configuration loaded: {config.opensearch_host}:{config.opensearch_port}")
    
    # Initialize components
    logger.info("Initializing components...")
    embedding_engine = EmbeddingEngine(config, logger)
    vector_index_builder = VectorIndexBuilder(config, logger)
    opensearch_client = vector_index_builder.opensearch_client
    pdf_ingester = PDFIngester(config, logger)
    chunking_module = ChunkingModule(config, logger)
    
    # Step 1: Delete existing PDF documents
    logger.info("\nStep 1: Deleting existing PDF documents...")
    delete_pdf_documents(opensearch_client, config.opensearch_index_name, logger)
    
    # Step 2: Ingest PDFs with Unstructured
    logger.info("\nStep 2: Ingesting PDFs with Unstructured chunking by title...")
    pdf_paragraphs = pdf_ingester.ingest_directory()
    logger.info(f"Ingested {len(pdf_paragraphs)} PDF chunks")
    
    # Step 3: Chunk PDF paragraphs (this will just pass them through since they're already chunked)
    logger.info("\nStep 3: Processing chunks...")
    pdf_chunks = chunking_module.chunk_pdf_paragraphs(pdf_paragraphs)
    logger.info(f"Created {len(pdf_chunks)} PDF chunks")
    
    # Step 4: Generate embeddings and index
    logger.info("\nStep 4: Generating embeddings and indexing...")
    
    # Generate embeddings for PDF chunks
    # Combine title and text for better semantic search
    logger.info("Generating embeddings for PDF chunks (title + text)...")
    pdf_texts = []
    for chunk in pdf_chunks:
        if chunk.title:
            # Embed both title and content together
            combined_text = f"{chunk.title}\n\n{chunk.text}"
        else:
            combined_text = chunk.text
        pdf_texts.append(combined_text)
    
    pdf_embeddings = embedding_engine.generate_embeddings(pdf_texts)
    logger.info(f"Generated {len(pdf_embeddings)} PDF embeddings")
    
    # Index PDF chunks
    logger.info("Indexing PDF chunks...")
    vector_index_builder.index_pdf_chunks(pdf_chunks, pdf_embeddings)
    logger.info("PDF chunks indexed successfully")
    
    # Verify final count
    count_response = opensearch_client.count(index=config.opensearch_index_name)
    total_docs = count_response.get('count', 0)
    
    logger.info("\n" + "=" * 80)
    logger.info("Reindexing complete!")
    logger.info(f"Total documents in index: {total_docs}")
    logger.info(f"PDF chunks: {len(pdf_chunks)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
