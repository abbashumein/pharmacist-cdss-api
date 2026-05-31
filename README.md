# Pharmacist CDSS: AI-Powered Clinical Decision Support System

Pharmacist CDSS is a production-oriented Clinical Decision Support System (CDSS) designed to assist healthcare professionals by combining machine learning, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs) into a unified AI workflow.

The system is engineered using modern AI engineering principles including modular service design, API-first development, containerization, retrieval pipelines, and scalable deployment patterns.

Rather than functioning as a standalone machine learning model, the platform demonstrates how multiple AI components can collaborate to support clinical reasoning and pharmaceutical decision-making.

---

## 🚀 Current Implementation

The current repository contains the core backend architecture and AI workflow.

### Core Components

#### FastAPI API Gateway

* Asynchronous API endpoints
* Request validation using Pydantic
* Structured request/response handling
* RESTful architecture

#### Clinical Severity Classification

* Fine-tuned DistilBERT model
* PyTorch inference pipeline
* Emotional and clinical risk assessment
* Local inference without external model hosting

#### Retrieval-Augmented Generation (RAG)

* LangChain integration
* ChromaDB vector storage
* Semantic similarity search
* Retrieval of relevant clinical references

#### LLM Orchestration Layer

* Gemini API integration
* Context-aware prompt construction
* Combination of:

  * ML predictions
  * Retrieved medical references
  * User-provided information

#### Containerization

* Dockerized deployment
* Reproducible runtime environment
* Environment-based configuration

#### CI/CD Foundations

* GitHub Actions workflow
* Automated build validation
* Continuous integration pipeline

---

## 🏗️ System Architecture

```text
User Interface
      │
      ▼
FastAPI API Layer
      │
      ▼
Routing & Request Validation
      │
 ┌────┴────┐
 ▼         ▼
ML Service  RAG Service
 │           │
 ▼           ▼
DistilBERT   ChromaDB
 │           │
 └────┬──────┘
      ▼
 Gemini Service
      │
      ▼
 Safety Layer
      │
      ▼
 Structured Clinical Guidance
```

---

## 📁 Project Structure

```text
app/
│
├── main.py
│
├── services/
│   ├── ml_service.py
│   ├── rag_service.py
│   └── gemini_service.py
│
├── models/
├── utils/
├── core/
│
├── data/
├── vectorstore/
│
└── tests/

.github/
└── workflows/

Dockerfile
requirements.txt
README.md
```

---

## 🧠 AI Workflow

1. Clinician submits patient information.
2. FastAPI validates incoming data.
3. DistilBERT performs local severity analysis.
4. ChromaDB retrieves relevant pharmaceutical references.
5. Gemini receives:

   * Patient information
   * Severity indicators
   * Retrieved context
6. Safety rules are applied.
7. Structured guidance is returned.

---

## 🔄 Knowledge Update Pipeline (Planned)

```text
Medical Documents
       │
       ▼
Document Processing
       │
       ▼
Chunking
       │
       ▼
Embeddings
       │
       ▼
ChromaDB
       │
       ▼
RAG Retrieval Layer
```

Future versions will support:

* PDF ingestion
* Automated chunking
* Embedding generation
* Knowledge base updates
* Continuous RAG refresh

---

## ☁️ Planned Production Architecture

The project is being designed toward a scalable cloud-native architecture.

Future enhancements include:

### Infrastructure

* AWS ECS/Fargate deployment
* Container orchestration
* Auto-scaling services

### Data Pipelines

* Apache Airflow
* Scheduled document ingestion
* Automated vector database updates

### Observability

* Structured logging
* Request tracing
* Latency monitoring
* Error analytics

### LLMOps

* Response evaluation
* Prompt versioning
* Quality monitoring

---

## 🛠️ Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
```

### Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

Build image:

```bash
docker build -t pharmacist-cdss-api .
```

Run container:

```bash
docker run -d -p 8000:8000 \
-e GEMINI_API_KEY=your_gemini_api_key \
pharmacist-cdss-api
```

---

## 📦 Technologies Used

* Python
* FastAPI
* PyTorch
* DistilBERT
* LangChain
* ChromaDB
* Gemini API
* Docker
* GitHub Actions
* Pydantic
* Uvicorn

---

## 🎯 Learning Objectives

This project is designed to demonstrate practical AI Engineering concepts:

* LLM Integration
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* API Engineering
* AI System Design
* Containerization
* CI/CD
* MLOps & LLMOps Foundations
* Production-Oriented AI Development
