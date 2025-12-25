#!/usr/bin/env python3
"""Script to reindex only PDF documents without touching video transcripts."""

import logging
from src.config import Config
from src.ingestion.pdf_ingester import PDFIngester
from src.processing.chunking import ChunkingModule
from src.processing.embedding import EmbeddingEngine
from opensearchpy import OpenSearch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reindex_pdfs_only():
    """Reindex only PDF documents, keeping video transcripts intact."""
    
    logger.info("Starting PDF-only reindexing...")
    
    # Load configuration
    config = Config.from_env()
    
    # Initialize components
    embedding_engine = EmbeddingEngine(config, logger)
    pdf_ingester = PDFIngester(config, logger)
    chunking_module = ChunkingModule(config, logger)
    
    # Initialize OpenSearch client
    opensearch_client = OpenSearch(
        hosts=[{'host': config.opensearch_host, 'port': config.opensearch_port}],
        http_auth=(config.opensearch_username, config.opensearch_password) if config.opensearch_username else None,
        use_ssl=config.opensearch_use_ssl,
        verify_certs=config.opensearch_verify_certs,
        ssl_show_warn=False
    )
    
    # Step 1: Delete only PDF documents from the index
    logger.info("Step 1: Deleting existing PDF documents from index...")
    try:
        delete_query = {
            "query": {
                "term": {
                    "source_type": "pdf"
                }
            }
        }
        
        response = opensearch_client.delete_by_query(
            index=config.opensearch_index_name,
            body=delete_query
        )
        
        deleted_count = response.get('deleted', 0)
        logger.info(f"Deleted {deleted_count} PDF documents from index")
        
    except Exception as e:
        logger.error(f"Error deleting PDF documents: {str(e)}")
        raise
    
    # Step 2: Ingest PDF documents
    logger.info("Step 2: Ingesting PDF documents...")
    pdf_paragraphs = pdf_ingester.ingest_directory()
    logger.info(f"Ingested {len(pdf_paragraphs)} PDF paragraphs")
    
    # Step 3: Chunk PDF paragraphs
    logger.info("Step 3: Chunking PDF paragraphs...")
    pdf_chunks = chunking_module.chunk_pdf_paragraphs(pdf_paragraphs)
    logger.info(f"Created {len(pdf_chunks)} PDF chunks")
    
    # Step 4: Generate embeddings and index
    logger.info("Step 4: Generating embeddings and indexing PDF chunks...")
    
    bulk_data = []
    for i, chunk in enumerate(pdf_chunks):
        if i % 100 == 0:
            logger.info(f"Processing chunk {i}/{len(pdf_chunks)}...")
        
        # Generate embedding
        embedding = embedding_engine.embed_text(chunk.text)
        
        # Prepare document for indexing
        doc = {
            "embedding": embedding.tolist(),
            "source_type": "pdf",
            "pdf_filename": chunk.pdf_filename,
            "page_number": chunk.page_number,
            "paragraph_index": chunk.paragraph_index,
            "text": chunk.text
        }
        
        # Add to bulk data
        bulk_data.append({"index": {"_index": config.opensearch_index_name}})
        bulk_data.append(doc)
        
        # Bulk index every 100 documents
        if len(bulk_data) >= 200:  # 100 docs * 2 lines each
            opensearch_client.bulk(body=bulk_data)
            bulk_data = []
    
    # Index remaining documents
    if bulk_data:
        opensearch_client.bulk(body=bulk_data)
    
    logger.info(f"Successfully indexed {len(pdf_chunks)} PDF chunks")
    
    # Step 5: Verify counts
    logger.info("Step 5: Verifying index counts...")
    
    # Count videos
    video_count = opensearch_client.count(
        index=config.opensearch_index_name,
        body={"query": {"term": {"source_type": "video"}}}
    )['count']
    
    # Count PDFs
    pdf_count = opensearch_client.count(
        index=config.opensearch_index_name,
        body={"query": {"term": {"source_type": "pdf"}}}
    )['count']
    
    # Total count
    total_count = opensearch_client.count(
        index=config.opensearch_index_name
    )['count']
    
    logger.info(f"Index verification:")
    logger.info(f"  Video documents: {video_count}")
    logger.info(f"  PDF documents: {pdf_count}")
    logger.info(f"  Total documents: {total_count}")
    
    logger.info("PDF reindexing completed successfully!")

if __name__ == "__main__":
    try:
        reindex_pdfs_only()
    except Exception as e:
        logger.error(f"Failed to reindex PDFs: {str(e)}", exc_info=True)
        exit(1)
