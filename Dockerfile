# Use official lightweight Python 3.12 slim image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install essential build dependencies for tokenizers / sentencepiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application source modules and scripts
COPY app.py .
COPY multimodal_embedder.py .
COPY demo_dashboard.py .
COPY ProjectD.py .
COPY projectd_closed_loop.py .
COPY start.sh .

# Ensure startup script has execution permissions
RUN chmod +x start.sh

# Pre-cache CLIP model during build so first request is instant
RUN python -c "from transformers import CLIPModel, CLIPProcessor; CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

EXPOSE 8501 8000

# Execute orchestration startup script
CMD ["./start.sh"]
