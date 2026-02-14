FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY backend/requirements.txt ./backend/
COPY ml/requirements.txt ./ml/
COPY workers/requirements.txt ./workers/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r ml/requirements.txt
RUN pip install --no-cache-dir -r workers/requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY workers/ ./workers/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/models

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["celery", "-A", "workers.celery_app", "worker", "--loglevel=info"]
