FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Expose dynamic port
EXPOSE $PORT

# Run gunicorn with single worker and increased timeout
CMD ["sh", "-c", "gunicorn --workers=1 --timeout 120 --bind 0.0.0.0:$PORT app:app"]