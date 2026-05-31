# Pharmacist CDSS: AI-Powered Clinical Decision Support System API

Pharmacist CDSS is a modular, production-ready microservice architecture designed to assist clinicians and pharmacists in high-stress triage environments. The system acts as an intelligent dual-engine gateway: it uses a highly optimized PyTorch NLP model to parse user distress text across dozens of emotional variables, while concurrently running an advanced Retrieval-Augmented Generation (RAG) pipeline to cross-reference pharmaceutical knowledge bases, ensuring safe decision-making and preventing adverse drug events.

🚀 Features

- **Hybrid AI Architecture**: Dual-engine intelligence combining fine-grained sentiment classifications with context-aware semantic retrieval.
- **Deep NLP Sentiment Engine**: Powered by a fine-tuned DistilBERT transformer (via Hugging Face & PyTorch) to evaluate 27 emotional classes and output explicit clinical severity metrics.
- **Context-Aware Knowledge Retrieval**: Uses LangChain and a localized vector storage setup to ingest clinical medical guidelines, injecting authoritative medical context into the processing pipeline via Retrieval-Augmented Generation (RAG).
- **Orchestrated LLM Synthesis**: Seamlessly integrates Gemini API orchestration to synthesize clinical documentation, cross-reference data points, and generate clear decision pathways.
- **High-Performance Microservice Layer**: Engineered completely on FastAPI using asynchronous design patterns for low-latency request processing.
- **Containerized for the Cloud**: Fully configured with an enterprise-ready Dockerfile for seamless, environment-agnostic deployment onto production clusters (AWS ECS Fargate, Kubernetes).
- **Automated CI/CD**: Built-in GitHub Actions workflow to automatically test, validate, and preview container builds on every code change.

📁 Project Structure

├── .github/
│   └── workflows/
│       └── deploy.yml        # CI/CD pipeline automation for cloud deployment
├── app/
│   ├── main.py               # FastAPI application entry point and async routing
│   └── services/
│       ├── ml_service.py     # PyTorch DistilBERT sentiment & severity execution
│       ├── rag_service.py    # LangChain vector indexing and document retrieval
│       └── gemini_service.py # Gemini LLM orchestration and final response synthesis
├── .gitignore                # Production-safe tracking exclusions (ignores data/env)
├── Dockerfile                # Multi-stage container build configuration
└── requirements.txt          # Python dependency pinning matrix

🔁 Core Execution Flow

1. Client or clinical interface fires a payload containing patient history or raw text to the FastAPI gateway.
2. The endpoint triggers parallel service actions:
   - `ml_service.py` feeds text to DistilBERT to classify clinical severity.
   - `rag_service.py` queries localized vector indexes to extract explicit pharmaceutical constraints.
3. `gemini_service.py` acts as the master orchestrator, synthesizing the raw context arrays and severity values into an authoritative clinical insight model.
4. FastAPI serializes the clean payload and streams it back to the interface with ultra-low latency.

🧠 Architecture Overview

     ┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                                           
│                                                                                        │
│  ┌───────────────────────┐       ┌─────────────────────────┐       ┌─────────────────┐ │
│  │    CLIENT REQUEST     │ ───>  │     AWS API GATEWAY     │ ───>  │     AWS ECS     │ │
│  │ (Clinical Dashboard)  │       │ (Security & Throttling) │       │  (Fargate App)  │ │
│  └───────────────────────┘       └─────────────────────────┘       └────────┬────────┘ │
│                                                                             │          │
│                                            ┌────────────────────────────────┴────────┐ │
│                                            ▼                                         ▼ │
│                                  ┌───────────────────┐                     ┌───────────────────┐ │
│                                  │   ML_SERVICE.PY   │                     │   RAG_SERVICE.PY  │ │
│                                  │PyTorch(DistilBERT)│                     │ LangChain Vector  │ │
│                                  └─────────┬─────────┘                     └─────────┬─────────┘ │
│                                            │                                         │          │
│                                            └────────────────────┬────────────────────┘          │
│                                                                 ▼                               │
│                                                      ┌───────────────────┐                     │
│                                                      │ GEMINI_SERVICE.PY │                     │
│                                                      │(LLMOrchestration) │                     │
│                                                      └─────────┬─────────┘                     │
│                                                                │                               │
│                                                                ▼                               │
│                                                      ┌───────────────────┐                     │
│                                                      │ MLOPS MONITORING  │                     │
│                                                      │(Prometheus/Loggg) │                     │
│                                                      └───────────────────┘                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                           ▲ (Scheduled Updates)
                                           │
┌──────────────────────────────────────────┴─────────────────────────────────────────────┐
│                             DATA ORCHESTRATION PIPELINES                               │
│                                                                                        │
│  ┌───────────────────────┐       ┌─────────────────────────┐       ┌─────────────────┐ │
│  │   RAW CLINICAL DATA   │ ───>  │     APACHE AIRFLOW      │ ───>  │ EMBEDDING LOOP  │ │
│  │ (Medical Docs / Logs) │       │    (Data Ingestion)     │       │ (Vector Update) │ │
│  └───────────────────────┘       └─────────────────────────┘       └─────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘



📦 Technologies Used

FastAPI: Asynchronous Python microservice framework.

PyTorch & Hugging Face: Transformer runtime infrastructure executing the localized DistilBERT model.

LangChain: AI orchestration primitives for systemic Retrieval-Augmented Generation.

Docker: Containerization and isolation platform.

GitHub Actions: Continuous Integration and Deployment automation matrix.

```bash
docker build -t pharmacist-cdss-api .
docker run -d -p 8000:8000 --env GEMINI_API_KEY="your_api_key_here" pharmacist-cdss-api
