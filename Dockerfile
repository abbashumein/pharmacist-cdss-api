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

# No COPY of ./chroma_db: it is gitignored, so a clean checkout does not have it.
# V2 builds the FDA vector DB itself on first startup (see startup_ingest).

COPY ./static /code/static

EXPOSE 8000

# This image serves V2 (RAG pipeline). V3 (agentic, app.main_agentic) reads the
# same ./chroma_db, so run it after V2 has built the DB, e.g.:
#   uvicorn app.main_agentic:app --host 0.0.0.0 --port 8001
CMD ["uvicorn", "app.main_demo:app", "--host", "0.0.0.0", "--port", "8000"]
