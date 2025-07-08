# Use an official PyTorch image with CUDA for GPU support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Set environment variable for CUDA (optional, for some models)
ENV CUDA_VISIBLE_DEVICES=0

# Run the FastAPI app with uvicorn, using all available GPUs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
