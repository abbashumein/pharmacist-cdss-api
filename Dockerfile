FROM python:3.11-slim

WORKDIR /code

# Install system dependencies needed for compiling certain python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application directories and files
COPY ./app /code/app
COPY ./frontend /code/frontend
COPY ./ingest_fda.py /code/ingest_fda.py

# Copy local database storage so the container has access to your vectors
COPY ./chroma_storage /code/chroma_storage

EXPOSE 8000

# FIXED: Points to main_demo instead of main
CMD ["uvicorn", "app.main_demo:app", "--host", "0.0.0.0", "--port", "8000"]