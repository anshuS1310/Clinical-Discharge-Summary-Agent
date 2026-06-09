FROM python:3.11-slim

WORKDIR /app

# System libraries needed by PDF and image packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies — lite build (no torch) since we use a cloud API key on Railway
COPY requirements-lite.txt .
RUN pip install --no-cache-dir -r requirements-lite.txt

# Copy all project files
COPY . .

# Create output directories
RUN mkdir -p output/drafts output/traces output/plots data/raw_patients

EXPOSE 8000

# Railway injects $PORT at runtime
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
