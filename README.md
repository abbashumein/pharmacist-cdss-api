Pharmacist CDSS: AI-Powered Clinical Decision Support System API

Pharmacist CDSS is an asynchronous, high-performance API microservice engineered for clinical triage and decision support. The system implements a modular, containerized multi-service architecture that seamlessly bridges high-speed local deep learning inference with Retrieval-Augmented Generation (RAG) and large language model orchestration.

🚀 Part 1: Current Codebase Implementation

The files currently residing in this repository constitute the complete, functional core engine of the CDSS microservice. It is fully containerized, tested, and ready to run locally or inside a single cloud container.

Core Components Built:

Asynchronous FastAPI Gateway (app/main.py): Manages sub-second request-response lifecycles, fielding incoming clinician payloads and managing concurrent service threads.

Local Deep Learning Sentiment Node (app/services/ml_service.py): Runs a localized, fine-tuned PyTorch DistilBERT model mapping inputs across 27 emotional parameters to evaluate clinical severity and patient distress.

Deterministic Semantic Search Engine (app/services/rag_service.py): Implements LangChain-Chroma vector storage locally to execute high-speed similarity queries over medical reference logs, pulling exact pharmaceutical safety boundaries.

Context-Aware Reasoning Agent (app/services/gemini_service.py): Orchestrates the Gemini API to consume local PyTorch metrics alongside retrieved RAG guidelines, synthesizing a secure, structured guidance note for the clinician.

Infrastructure & Containerization (Dockerfile): Wraps the entire Python runtime, dependencies, and local parameters into a reproducible, production-ready virtual image.

CI/CD Build Automation (.github/workflows/deploy.yml): Automates testing of container compilations on every code change to prevent build regressions.

🏗️ Part 2: Target Enterprise Production Blueprint

To scale this service for real-world hospital environments, the codebase has been architected to deploy directly into the following enterprise cloud and data orchestrator infrastructure.

┌─────────────────────┐
│     User (UI)       │
│  Gradio Frontend    │
└──────────┬──────────┘
           │
           │ User Message
           ▼
┌─────────────────────┐
│      app.py         │
│ Request Handler     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Router Service    │
│ Decision Logic      │
└──────┬──────┬───────┘
       │      │
       │      │
       ▼      ▼
┌──────────┐ ┌──────────────┐
│ ML Model │ │ RAG Service  │
│DistilBERT│ │ Vector Search│
└─────┬────┘ └──────┬───────┘
      │             │
      └──────┬──────┘
             │
             ▼
┌─────────────────────┐
│   Gemini Service    │
│ LLM Orchestration   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Safety & Validation │
│ Response Filtering  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Final AI Response   │
│ Returned to User    │
└─────────────────────┘


      Knowledge Update Pipeline

┌─────────────────────┐
│ Mental Health Docs  │
│ PDFs / Articles     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Document Processing │
│ Chunking & Cleaning │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Embedding Generator │
│ SentenceTransformers│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Vector Database     │
│ Chroma / FAISS      │
└─────────────────────┘


Architectural Design Logic:

1. Cloud-Native Scalability (AWS ECS & Fargate)

The core FastAPI Docker container is designed to run on AWS ECS Fargate. As traffic peaks, AWS automatically provisions resources up to 1 vCPU and 2 GB RAM per instance, ensuring consistent api throughput without requiring manual server administration.

2. Scheduled Data Ingestion (Apache Airflow)

In production, clinical reference manuals and drug warning databases are updated constantly. To prevent downtime or out-of-date model contexts:

Apache Airflow acts as our central orchestration framework.

Airflow runs scheduled pipelines that extract fresh drug interaction data, run them through localized embedding models, and update the Chroma vector indexes.

These updated indexes are periodically pushed to production storage, keeping the active RAG retrieval engine accurate without interrupting the core API runtime.

🛠️ Installation & Local Execution

To run, inspect, and verify the core microservice architecture on your local system:

1. Requirements & Dependencies

Install the required packages:

pip install -r requirements.txt


2. Set Up Environment Variables

Create a .env file in the root directory and add your secret API key:

GEMINI_API_KEY="your_actual_gemini_api_key_here"


3. Start the FastAPI Microservice

Run the local Uvicorn development server:

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload


Once running, access the interactive Swagger API documentation at: http://localhost:8000/docs

4. Build the Local Docker Container

To test container packaging:

docker build -t pharmacist-cdss-api .
docker run -d -p 8000:8000 --env GEMINI_API_KEY="your_api_key" pharmacist-cdss-api
