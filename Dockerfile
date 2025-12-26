FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Unstructured and OpenCV
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Download the sentence-transformers model during build
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading MPNet model...'); \
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/app/models'); \
    print('Model downloaded successfully')"

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create data directories
RUN mkdir -p data/transcripts data/pdfs mock_data/transcripts mock_data/pdfs

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["serve"]
