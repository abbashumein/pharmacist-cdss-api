# Pharmacist CDSS — AI-Powered Clinical Decision Support System

A production-deployed Clinical Decision Support System that assists pharmacists and healthcare professionals with drug interaction checks, medication safety assessment, and evidence-grounded clinical guidance.

The system combines a **LangGraph 3-node RAG pipeline**, **ChromaDB vector database**, and **Gemini 2.5 Flash LLM** to retrieve real FDA drug records and generate structured clinical responses — with live risk assessment, emotion detection, and full audit logging.
---

## Key Features

### Clinical Decision Support
* Drug interaction safety checks
* Medication side effect guidance
* Patient symptom triage with severity scoring (LOW/MODERATE/HIGH)
* Structured clinical responses with follow-up recommendations

### Retrieval-Augmented Generation (RAG)
* Semantic search via Gemini Embeddings
* ChromaDB vector database with FDA drug records
* Evidence-grounded responses with source attribution
* Auto-ingestion from openFDA API on every startup

### LangGraph AI Orchestration
* 3-node pipeline: Triage → RAG → Generation → Telemetry
* Clinical keyword detection and drug name extraction
* Multi-stage prompt construction with retrieved context
* Session-based memory with MemorySaver

### Live Telemetry Dashboard
* Real-time clinical risk badge (LOW/MODERATE/HIGH)
* Confidence score per response
* Patient emotion state detection
* Full audit trail per query

### Production Infrastructure
* FastAPI backend with API key security
* Docker containerized deployment
* Azure Container Apps cloud hosting
* GitHub Actions CI/CD pipeline
* Custom HTML/JS frontend served from same container

## 📸 System in Action

**Clinical AI Interface — Ready State**
![Home Screen](screenshots/home.PNG)

**Drug Interaction Query — HIGH Risk Triggered**
![Prompt 1 Response](screenshots/prompt_1_response.PNG)

**Patient Symptom Triage — Clinical Summary Generated**
![Prompt 2 Response](screenshots/prompt_2%20_response.PNG)

**Live Telemetry — RAG Evidence Sources + Audit Trail**
![Telemetry Panel](screenshots/telemetry.PNG)

**API Documentation — Swagger UI**
![Swagger Docs](screenshots/swagger.PNG)

## Project Goal

To demonstrate how modern AI engineering combines RAG pipelines, vector databases, LLM reasoning, and cloud infrastructure to build a deployable, production-ready clinical decision support application.

This project focuses on **practical AI system building** — not standalone model training — showing how multiple AI components can be orchestrated into a real deployed product.

---

## 🚀 Current Implementation

The system is fully built and deployed. All components are live and functional.

### Core Components

**FastAPI API Gateway**
* Asynchronous API endpoints
* Pydantic request validation
* API key security middleware
* RESTful architecture with Swagger docs

**LangGraph Orchestration Pipeline**
* 3-node workflow: Triage → Generation → Telemetry
* Clinical keyword and drug name detection
* Session memory with MemorySaver
* Conditional evidence retrieval

**Retrieval-Augmented Generation (RAG)**
* ChromaDB vector storage
* Gemini Embedding (models/gemini-embedding-001)
* Semantic similarity search
* openFDA API auto-ingestion on startup

**Gemini 2.5 Flash LLM**
* Context-aware clinical prompt construction
* Retrieved FDA context injection
* Structured response format enforcement
* Safety disclaimer enforcement

**Frontend UI**
* Custom HTML/CSS/JS chat interface
* Live telemetry panel (risk, confidence, emotion)
* Real-time audit trail display
* Served directly from FastAPI

**Containerization & Deployment**
* Dockerized with Azure Container Registry
* Azure Container Apps hosting
* GitHub Actions CI/CD pipeline
* Auto-deploy on every git push

---
## 🏗️ System Architecture

```text
User Query (Frontend UI)
        │
        ▼
FastAPI API Gateway
(API Key Security + Pydantic Validation)
        │
        ▼
LangGraph Pipeline
        │
   ┌────┴──────────┐
   ▼               ▼
Triage Node     ChromaDB
(keyword +      Vector DB
drug detection) (FDA Records)
   │               │
   └────┬──────────┘
        ▼
Generation Node
(Gemini 2.5 Flash +
Retrieved FDA Context)
        │
        ▼
Telemetry Node
(Risk + Confidence + Emotion)
        │
        ▼
Structured Clinical Response
+ Full Audit Trail
```

---

## 📁 Project Structure

```text
app/
│
├── main_demo.py          # FastAPI app + LangGraph pipeline
│
└── utils/
    └── logger.py         # Structured logging

static/
└── index.html            # Frontend UI

screenshots/              # Testing proof screenshots

.github/
└── workflows/
    └── deploy.yml        # GitHub Actions CI/CD

Dockerfile                # Container build
requirements.txt          # Python dependencies
README.md
```


---

## 🧠 AI Workflow

1. Pharmacist submits query via frontend UI
2. FastAPI validates and routes the request
3. **Triage Node** — detects clinical keywords, identifies drug names
4. **RAG Node** — ChromaDB semantic search retrieves relevant FDA drug records
5. **Generation Node** — Gemini 2.5 Flash receives:
   * Retrieved FDA context
   * Patient medication profile
   * Clinical triage signals
6. **Telemetry Node** — extracts risk level, confidence score, emotion state
7. Structured clinical response returned with full audit trail
---

## 🔄 Knowledge Ingestion Pipeline

```text
openFDA API

│

▼

Auto-ingest on Startup

│

▼

Text Chunking (drug name + interactions + contraindications)

│

▼

Gemini Embedding (models/gemini-embedding-001)

│

▼

ChromaDB Vector Store

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
| Frontend | Custom HTML/JS served via FastAPI |
| Security | API Key middleware |
| Monitoring | Structured audit logging |

---
---

## 📦 Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| API Framework | FastAPI + Uvicorn |
| AI Orchestration | LangGraph |
| LLM | Gemini 2.5 Flash (google-genai SDK) |
| Embeddings | Gemini Embedding (models/gemini-embedding-001) |
| Vector Database | ChromaDB |
| Data Validation | Pydantic |
| Containerization | Docker |
| Cloud Platform | Azure Container Apps |
| Container Registry | Azure Container Registry |
| CI/CD | GitHub Actions |
| Data Source | openFDA API |
| Frontend | HTML + CSS + JavaScript |

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

## Performance Metrics

| Metric | V1 (Baseline) | V2 (RAG Pipeline) | V3 (Agentic) |
|---|---|---|---|
| Direct Lookup Accuracy | 80% (4/5) | 100% (7/7) | 100% |
| Drug Interaction Accuracy | 100% (4/4) | 83% (5/6) | 80% (4/5) |
| Contraindication Accuracy | 100% (4/4) | 100% (6/6) | 100% (3/3) |
| Out-of-Scope Rejection | Not tested | 100% (3/3) | 100% (4/4) |
| Tool Routing Accuracy | N/A | N/A | 100% (20/20) |
| Grounded Responses | N/A | N/A | 90% (18/20) |
| Records in ChromaDB | 19 FDA labels | 444 FDA labels | 806 FDA labels |
| Embedding Model | Gemini cloud API | Local all-MiniLM-L6-v2 | Local all-MiniLM-L6-v2 |
| Reranker | None | CrossEncoder | CrossEncoder |
| Similarity Threshold | None | 0.95 cosine distance | 0.95 cosine distance |
| Eval Test Cases | 10 queries | 24 queries | 20 queries |
| p95 Latency | ~16s (cold start) | ~5-8s (local) | ~5-10s (local) |
| Cost per Request | ~$0.0008 | ~$0.0004 | ~$0.0004 |

## V3 Agentic Key Findings
- Gemini correctly routes ALL 20 queries — tool called for clinical, refused for out-of-scope
- 90% of clinical responses grounded in retrieved FDA evidence
- 2 misses: ibuprofen data quality gap + ambiguous query without drug context

## Known Limitations
* Multi-drug interaction queries occasionally favor one drug's chunk over the other
* Ibuprofen FDA records in corpus have incomplete interaction data
* Gemini free tier limits evaluation to 20 requests/day

## 🔧 Engineering Challenges & Solutions

| Problem | Root Cause | Fix |
|---|---|---|
| Container OOM crash | ChromaDB + ONNX loading 3GB+ RAM on startup | Switched to cloud Gemini embeddings, removed local ONNX runtime entirely |
| 503 timeout on ingest | 130MB ZIP upload exhausting Azure proxy timeout | Replaced file upload with direct openFDA API + FastAPI BackgroundTasks |
| GeminiEmbeddingFunction not found | ChromaDB 0.5.3 naming convention change | Wrote custom `GeminiEmbeddingFunction(EmbeddingFunction)` class using google-genai SDK |
| Static files missing in container | Dockerfile missing `COPY ./static` line | Added `COPY ./static /code/static` to Dockerfile |
| Git push conflicts | CI/CD auto-commits diverging from local branch | Used `git pull --rebase` + force push to resolve |
| Hardcoded API key fallback in auth logic | `os.getenv("CDSS_API_KEY", "prod-secret-fallback-key")` — silent fallback to a public, committed string if env var unset | Removed fallback; auth now fails loudly at startup via `RuntimeError` if `CDSS_API_KEY` is missing |
| Eval script reported false 100% accuracy | Hit detection checked only HTTP gateway status, never verified the correct drug was actually retrieved | Rewrote eval to check expected drug term presence in retrieved evidence, added 5-category breakdown (direct lookup, interaction, contraindication, ambiguous, out-of-scope) |
| Ingested drugs didn't match tested drugs | openFDA ingestion pulled a random top-50 batch with no guarantee target drugs (warfarin, aspirin, etc.) were included | Added explicit per-drug openFDA fetch before the general batch, guaranteeing core test drugs are always in the corpus |
| Corrupted API key at runtime | Manual shell `export` left a stale, multi-line-corrupted value in the terminal session; `load_dotenv()` doesn't override existing env vars by default | Switched to `load_dotenv(override=True)` so `.env` always takes priority over stray shell state |
| Windows local dev blocked by native build | `chroma-hnswlib==0.7.3` (via `chromadb==0.5.3`) has no prebuilt Windows wheel for Python 3.12, requires MSVC compiler | Upgraded to `chromadb==0.5.4` (prebuilt wheel available), decoupled `langchain-chroma` install with `--no-deps` to resolve a metadata-only version conflict |
| Out-of-scope queries returning irrelevant chunks | No similarity threshold — ChromaDB always returned nearest neighbor even for unrelated queries | Added cosine distance threshold (0.8) in triage node — rejects chunks above threshold, returns 'No relevant FDA evidence found' |



### 🔬 Research & Development

The exploratory data analysis, severity classification modeling, and initial workflow prototyping for this clinical support tool were developed in Google Colab. You can view and run the experimental notebook directly via the link below:

* [Launch Active Colab Notebook](https://colab.research.google.com/drive/1KwWJRlIynOMbfM8f3zcUym4lCNytyoj8#scrollTo=MUOtHKZA2L7n)
