# Pharmacist CDSS: AI-Powered Clinical Decision Support System

Pharmacist CDSS is a production-oriented Clinical Decision Support System (CDSS) that assists pharmacists and healthcare professionals with medication safety checks, pharmaceutical information retrieval, and evidence-informed clinical decision support.

The platform combines Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), semantic search, and rule-based safety mechanisms to provide contextual pharmaceutical guidance during clinical consultations.

Unlike traditional chatbots, the system integrates a structured retrieval pipeline with a vector database of pharmaceutical references, enabling responses to be grounded in retrieved medical knowledge rather than relying solely on LLM-generated content.

The project demonstrates modern AI Engineering practices including API-first development, retrieval systems, vector databases, containerization, CI/CD foundations, modular service architecture, and scalable deployment workflows.

---

## Key Features

### Clinical Decision Support

* Medication information assistance
* Drug safety guidance
* Context-aware pharmaceutical recommendations
* Structured consultation responses

### Retrieval-Augmented Generation (RAG)

* Semantic retrieval using embeddings
* ChromaDB vector database
* Evidence-grounded response generation
* Context injection into LLM prompts

### LLM-Powered Reasoning

* Gemini-powered response generation
* Context-aware clinical guidance
* Multi-stage prompt orchestration
* Safety-focused response construction

### AI Engineering Infrastructure

* FastAPI backend services
* Docker containerization
* Environment-based configuration
* Modular service architecture
* GitHub Actions CI/CD pipeline

### Production-Oriented Design

* API-first architecture
* Retrieval pipelines
* Scalable deployment strategy
* Structured validation layer
* Extensible knowledge base architecture

---

## Project Goal

The objective of this project is to demonstrate how modern AI systems can combine retrieval mechanisms, vector databases, LLM reasoning, and backend engineering to support healthcare workflows in a safe and scalable manner.

The repository focuses on practical AI Engineering concepts rather than standalone model training, showcasing how multiple AI components can be orchestrated into a deployable decision-support application.


---

## 🚀 Current Implementation

The current repository contains the core backend architecture and AI workflow.

### Core Components

#### FastAPI API Gateway

* Asynchronous API endpoints
* Request validation using Pydantic
* Structured request/response handling
* RESTful architecture

#### Clinical Safety Classification

* Fine-tuned DistilBERT model
* Local PyTorch inference pipeline
* Consultation intent and risk categorization

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

1. Pharmacist submits medication-related query
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
* Production-Oriented AI Development.

## Production Features

- FastAPI REST API
- Pydantic Request Validation
- Dockerized Deployment
- ChromaDB Vector Store
- Gemini LLM Integration
- Structured Logging
- Environment-Based Configuration
- CI/CD Foundations
- Modular Service Architecture
- Retrieval-Augmented Generation (RAG)



### 🔬 Research & Development

The exploratory data analysis, severity classification modeling, and initial workflow prototyping for this clinical support tool were developed in Google Colab. You can view and run the experimental notebook directly via the link below:

* [Launch Active Colab Notebook](https://colab.research.google.com/drive/1KwWJRlIynOMbfM8f3zcUym4lCNytyoj8#scrollTo=MUOtHKZA2L7n)
