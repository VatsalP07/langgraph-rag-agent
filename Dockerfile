# Dockerfile for RAG Agent Deployment

# Use a standard Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install necessary system packages
# These are often required for libraries like numpy, scipy, etc., which are dependencies of your RAG libraries.
RUN apt-get update && apt-get install -y \
    gcc \
    # Clean up the cache to keep the image small
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the generated data directory (including the ChromaDB)
# IMPORTANT: The 'data/chroma_db' folder must exist on your local machine before building!
COPY . /app/
COPY data/ /app/data/

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

# Command to run the Gradio UI application
CMD ["python", "app_ui.py"]