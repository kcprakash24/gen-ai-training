FROM python:3.12-slim

WORKDIR /app

# Set Python to use centralized pycache directory
ENV PYTHONPYCACHEPREFIX=/app/.pycache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt supervisor

# Copy application code
COPY . .

# Make bootstrap script executable
RUN chmod +x project_bootstrap.py

# Create directories including centralized pycache
RUN mkdir -p database/vector_store database/loan_data logs .pycache

# Expose FastAPI and Streamlit ports only
EXPOSE 8000 8501

# Start FastAPI and Streamlit using supervisor (bootstrap temporarily disabled for debugging)
CMD ["/bin/bash", "-c", "supervisord -n -c supervisord.conf"]
