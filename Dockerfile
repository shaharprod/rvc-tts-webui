FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Downgrade pip to version that can install omegaconf
RUN pip install --upgrade "pip<24.1"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models from Google Drive if GOOGLE_DRIVE_MODEL_ID is set
RUN python download_models.py || echo "Models download skipped or failed"

# Expose port
EXPOSE 7865

# Run the application
CMD ["python", "app.py"]

