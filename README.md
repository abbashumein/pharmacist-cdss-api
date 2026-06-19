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
---text
## 🏗️ System Architecture

User Query (Frontend UI)
        │
        ▼
FastAPI API Gateway (Azure Container Apps)
        │
        ▼
LangGraph 3-Node Pipeline
        │
   ┌────┴────┐
   ▼         ▼
Triage    ChromaDB
Node      Vector DB
   │      (FDA Records)
   └────┬──────┘
        ▼
  Gemini 2.5 Flash
        │
        ▼
  Telemetry Node
  (Risk + Emotion + Confidence)
        │
        ▼
  Structured Clinical Response
  
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

## 🔄 Knowledge Pipeline 

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

## ☁️ Production Infrastructure

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| AI Orchestration | LangGraph 3-node pipeline |
| Vector Database | ChromaDB + Gemini Embeddings |
| LLM | Gemini 2.5 Flash |
| Containerization | Docker |
| Cloud Hosting | Azure Container Apps |
| CI/CD | GitHub Actions |
| Frontend | Custom HTML/JS (served via FastAPI) |
| Security | API Key middleware |
| Monitoring | Structured audit logging |

## 🔮 Future Enhancements

### Data Pipelines
* Scheduled document ingestion
* Automated vector database updates

### Observability
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
uvicorn app.main_demo:app --host 0.0.0.0 --port 8000 --reload
```

API documentation:

```text
http://localhost:8000/docs
```

---

## 📦 Technologies Used

* Python
* FastAPI
* PyTorch
* DistilBERT
* LangGraph
* LangChain
* ChromaDB
* Gemini API
* GitHub Actions
* Pydantic
* Uvicorn

---

## 🎯 What This Project Demonstrates

| AI Engineering Concept | Implementation |
|---|---|
| LLM Integration | Gemini 2.5 Flash via google-genai SDK |
| Retrieval-Augmented Generation | ChromaDB + Gemini Embeddings + openFDA API |
| AI Orchestration | LangGraph 3-node pipeline |
| Vector Database | ChromaDB PersistentClient |
| API Engineering | FastAPI + Pydantic validation |
| Containerization | Docker + Azure Container Registry |
| CI/CD | GitHub Actions automated deployment |
| Cloud Deployment | Azure Container Apps |
| MLOps | Auto-ingestion on startup + audit logging |
| Security | API Key middleware |
| Frontend Integration | Custom HTML/JS served from FastAPI |



## 📸 System in Action

### Clinical AI Interface
![Home Screen](screenshots/home.png)

### Drug Interaction Detection — HIGH Risk Triggered
![Prompt 1 Response](screenshots/prompt_1_response.png)

### Drug Interaction Detection — HIGH Risk Triggered
![Prompt 2 Response](screenshots/prompt_2_response.png)

### RAG Evidence Sources + Audit Trail
![Telemetry Panel](screenshots/telemetry.png)

### API Documentation
![Swagger Docs](screenshots/swagger.png)

### 🔬 Research & Development

The exploratory data analysis, severity classification modeling, and initial workflow prototyping for this clinical support tool were developed in Google Colab. You can view and run the experimental notebook directly via the link below:


* [Launch Active Colab Notebook](https://colab.research.google.com/drive/1KwWJRlIynOMbfM8f3zcUym4lCNytyoj8#scrollTo=MUOtHKZA2L7n)
