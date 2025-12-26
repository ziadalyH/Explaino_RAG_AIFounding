#!/bin/bash
set -e

echo "=========================================="
echo "Starting RAG Chatbot Backend..."
echo "=========================================="

# Configuration
OPENSEARCH_HOST="${OPENSEARCH_HOST:-localhost}"
OPENSEARCH_PORT="${OPENSEARCH_PORT:-9200}"
OPENSEARCH_PDF_INDEX="${OPENSEARCH_PDF_INDEX:-rag-pdf-index}"
OPENSEARCH_VIDEO_INDEX="${OPENSEARCH_VIDEO_INDEX:-rag-video-index}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Build OpenSearch URL
OPENSEARCH_URL="http://${OPENSEARCH_HOST}:${OPENSEARCH_PORT}"

# Function to check OpenSearch health
check_opensearch_health() {
    local url="$1"
    if curl -s -f "${url}/_cluster/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to check if index exists and get document count
check_index_status() {
    local index_name="$1"
    local status=$(curl -s -o /dev/null -w "%{http_code}" "${OPENSEARCH_URL}/${index_name}")
    
    if [ "$status" = "404" ]; then
        echo "missing"
        return 0
    elif [ "$status" = "200" ]; then
        local count_response=$(curl -s "${OPENSEARCH_URL}/${index_name}/_count")
        local doc_count=$(echo "$count_response" | grep -o '"count":[0-9]*' | grep -o '[0-9]*' | head -1)
        
        if [ -z "$doc_count" ]; then
            echo "0"
        else
            echo "$doc_count"
        fi
        return 0
    else
        echo "error"
        return 1
    fi
}

# Wait for OpenSearch to be ready with timeout
echo "Waiting for OpenSearch at ${OPENSEARCH_URL}..."
RETRY_COUNT=0
until check_opensearch_health "${OPENSEARCH_URL}"; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: OpenSearch failed to become ready after ${MAX_RETRIES} attempts"
        echo "Please check OpenSearch logs and configuration"
        exit 1
    fi
    echo "OpenSearch is unavailable - attempt ${RETRY_COUNT}/${MAX_RETRIES} - sleeping ${RETRY_INTERVAL}s"
    sleep $RETRY_INTERVAL
done

echo "✓ OpenSearch is up and running!"

# Check if auto-indexing is enabled
if [ "${AUTO_INDEX_ON_STARTUP}" = "true" ]; then
    echo ""
    echo "AUTO_INDEX_ON_STARTUP is enabled. Checking for new data..."
    
    # Always run indexing - the Python code will handle checking for new files
    # and only process what's new (resume capability)
    echo ""
    echo "→ Running indexing (will only process new/modified files)..."
    if python main.py index; then
        echo "✓ Indexing completed successfully"
    else
        echo "ERROR: Indexing failed"
        exit 1
    fi
else
    echo ""
    echo "AUTO_INDEX_ON_STARTUP is disabled - skipping automatic indexing"
fi

echo ""
echo "=========================================="
echo "Startup complete!"
echo "=========================================="

# Check if we should run in API mode or CLI mode
if [ "$RUN_MODE" = "api" ]; then
    echo "Starting in API mode on port ${API_PORT:-8000}..."
    echo ""
    exec python -m src.api
else
    echo "Starting in CLI mode..."
    echo "Executing command: $@"
    echo ""
    exec python main.py "$@"
fi
