#!/bin/bash
set -e

echo "=========================================="
echo "Starting RAG Chatbot Backend..."
echo "=========================================="

# Configuration
OPENSEARCH_HOST="${OPENSEARCH_HOST:-localhost}"
OPENSEARCH_PORT="${OPENSEARCH_PORT:-9200}"
OPENSEARCH_INDEX_NAME="${OPENSEARCH_INDEX_NAME:-rag-index}"
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
    echo "AUTO_INDEX_ON_STARTUP is enabled. Checking index status..."
    
    # Check if index exists
    INDEX_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${OPENSEARCH_URL}/${OPENSEARCH_INDEX_NAME}")
    
    if [ "$INDEX_STATUS" = "404" ]; then
        echo "✗ Index '${OPENSEARCH_INDEX_NAME}' does not exist"
        echo "→ Running initial indexing..."
        if python main.py index; then
            echo "✓ Indexing completed successfully"
        else
            echo "ERROR: Indexing failed"
            exit 1
        fi
    elif [ "$INDEX_STATUS" = "200" ]; then
        echo "✓ Index '${OPENSEARCH_INDEX_NAME}' exists"
        
        # Check document count
        COUNT_RESPONSE=$(curl -s "${OPENSEARCH_URL}/${OPENSEARCH_INDEX_NAME}/_count")
        DOC_COUNT=$(echo "$COUNT_RESPONSE" | grep -o '"count":[0-9]*' | grep -o '[0-9]*' | head -1)
        
        if [ -z "$DOC_COUNT" ]; then
            echo "⚠ Warning: Could not determine document count, assuming empty"
            DOC_COUNT=0
        fi
        
        if [ "$DOC_COUNT" -eq 0 ]; then
            echo "✗ Index is empty (0 documents)"
            echo "→ Running indexing..."
            if python main.py index; then
                echo "✓ Indexing completed successfully"
            else
                echo "ERROR: Indexing failed"
                exit 1
            fi
        else
            echo "✓ Index contains ${DOC_COUNT} documents - skipping indexing"
        fi
    else
        echo "⚠ Warning: Unexpected HTTP status ${INDEX_STATUS} when checking index"
        echo "→ Attempting to run indexing anyway..."
        if python main.py index; then
            echo "✓ Indexing completed successfully"
        else
            echo "ERROR: Indexing failed"
            exit 1
        fi
    fi
else
    echo ""
    echo "AUTO_INDEX_ON_STARTUP is disabled - skipping automatic indexing"
fi

echo ""
echo "=========================================="
echo "Startup complete!"
echo "Executing command: $@"
echo "=========================================="
echo ""

# Execute the main command
exec python main.py "$@"
