FROM python:3.11-slim

# Install system dependencies for blis and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Expose port (Render will override this with its PORT env variable)
EXPOSE 8000

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]