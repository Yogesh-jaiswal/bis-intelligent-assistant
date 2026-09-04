# ==============================================================================
# BIS Intelligent Assistant - Application Dockerfile (Dockerfile.app)
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.factory:create_app

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Build-time flag to toggle Tesseract OCR installation
# To skip OCR during build, pass: --build-arg INSTALL_OCR=false
ARG INSTALL_OCR=true
RUN if [ "$INSTALL_OCR" = "true" ]; then \
        echo "Installing Tesseract OCR and Hindi language data..." && \
        apt-get update && apt-get install -y --no-install-recommends \
            tesseract-ocr \
            tesseract-ocr-hin \
        && rm -rf /var/lib/apt/lists/* ; \
    else \
        echo "Skipping Tesseract OCR installation (INSTALL_OCR=false)..." ; \
    fi

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/

# Expose API port
EXPOSE 5000

# Health check against API health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:5000/v1/health || exit 1

# Start development / prototype server
CMD ["python", "run.py"]
