# Pharmacist CDSS — Controlled Agentic RAG Clinical Decision Support System

A Clinical Decision Support System that helps pharmacists check drug interactions, dosing, warnings and side effects, with every answer grounded in FDA drug label text. It evolved from a fixed RAG pipeline (V2) into a controlled agentic system (V3): safety-critical routing is deterministic, and the LLM only writes the final answer from retrieved evidence. When evidence is missing, it refuses with a fixed message instead of guessing.

Built to be measured, not just demoed: a 50-question evaluation (including made-up drugs and prompt-injection attacks), regression tests for every bug found, and cost tracking (at most 1 LLM call per question).

The system exists in two versions:

- **V2 (Advanced RAG Pipeline)** — LangGraph 3-node pipeline (triage, retrieval, generation) over 830 openFDA drug label records, indexed as 11,898 section-level chunks (about 900 characters, 150 overlap, each tagged with drug and section). Local all-MiniLM-L6-v2 embeddings in ChromaDB (L2 distance), a drug-name guard that stops one drug's chunks answering for another, and CrossEncoder reranking (40 candidates narrowed to 8, near-duplicates removed).

- **V3 (Agentic RAG)** — LangGraph agent with deterministic safety routing and LLM generation. Local keyword routing (0 calls) sends out-of-scope questions to a fixed scope reply and no-evidence questions to a fixed "insufficient FDA data" refusal. Intent detection maps the question to label sections and ignores injected "SYSTEM:" text. Interaction searches include the partner drug. At most 1 Gemini call per question (0.68 on average). `ROUTER_MODE=llm` restores the old LLM router. I kept routing deterministic on purpose: predictable cost, latency and safety matter more than autonomy in a medical setting.

Both versions return structured clinical responses with a risk level and full audit logging.

---

## Key Features

### Clinical Decision Support
* Drug interaction safety checks
* Medication side effect guidance
* Risk level per answer (LOW / MODERATE / HIGH); risk is capped at MODERATE when there is no evidence
* Answers grounded only in retrieved FDA label text
* Out-of-scope queries get a fixed reply (0 model calls), with an emergency line
* Fixed refusal ("I don't have sufficient FDA data") when no usable evidence is found (0 model calls)

### Retrieval-Augmented Generation (RAG)
* Local all-MiniLM-L6-v2 embeddings: free, no API quota
* ChromaDB with 830 openFDA drug label records, split into 11,898 section chunks (about 900 characters, 150 overlap)
* Each chunk starts with `Drug: X | Section: Y |`, so the section and drug travel with the text
* Retrieval pipeline: 40 candidates, drug-name guard, intent-to-section filter, CrossEncoder rerank (ms-marco-MiniLM-L-6-v2) down to 8, near-duplicate removal
* Interaction searches include the partner drug's name
* Distance metric is L2 (Chroma default)

### Controlled Agent (V3)
* LangGraph state graph; safety-critical routing is deterministic and local (free)
* Gemini is called at most once per question, only to write the final answer
* Prompt rules: answer only from evidence, refuse when evidence is missing, never infer an interaction because two drug names appear
* Injection-safe intent detection: text after "SYSTEM:" cannot steer retrieval
* `ROUTER_MODE=llm` keeps the older LLM router as an option

### Evaluation Framework
* 50 questions in 7 categories: easy 14, hard 10, ambiguous 4, out-of-scope 6, made-up drugs 5, real drugs not in DB 5, prompt injection 6
* Measures retrieval, faithfulness (lexical check + Gemini judge), correctness, latency, tokens and calls
* Errors and quota failures are marked INVALID, never scored as pass or fail
* Resumable and quota-aware (default 18 Gemini calls per day)
* Regression test for every bug the eval found

### Production Infrastructure
* FastAPI backend with API key middleware and input validation
* Docker image; earlier deployed on Azure Container Apps (free tier ended, now runs locally or in Docker)
* GitHub Actions CI runs the pytest suite on every push
* Custom HTML/JS frontend served from the same container
* Audit trail per query: retrieval distance, latency, token counts, calls

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

**Drug Interaction Check — FDA Evidence Retrieved**
![Drug Interaction Check](screenshots/Drug_Check_2.PNG)

**FDA Evidence Search — Direct ChromaDB Retrieval**
![FDA Evidence Search](screenshots/FDA_Database_2.PNG)

**V2 — RAG Pipeline: Clinical Query Response**
![V2 AI Assistant](screenshots/AI_Assitant_Prompt_2_response.PNG)

**V3 — Agentic Pipeline: Tool-Calling Clinical Response**
![V3 Agentic Assistant](screenshots/v3-agentic-atorvastatin-contraindications-test2.PNG)

## Problem Statement

Pharmacists in high-volume settings make 50+ drug interaction checks daily under time pressure. Preventable medication errors are a major source of patient harm. Existing tools are either too slow, too generic, or not grounded in verified FDA data.

This system gives fast, FDA-grounded drug guidance with an explicit risk level and audit logging. If the FDA evidence is missing, it refuses instead of guessing. It is a decision support tool, not a replacement for clinical judgment.

## Project Goal

To show how RAG, a controlled agent, a vector database and LLM generation can be combined into a safe, measured clinical decision support API.

This project focuses on **practical AI system building**, not model training. It was built in two stages:

- **V2**: RAG pipeline over 830 FDA drug label records with evaluation, audit logging and safety controls.
- **V3**: Controlled agent (LangGraph). Safety routing is deterministic and the LLM writes the final answer from retrieved evidence, with at most 1 Gemini call per question. It was measured on a 50-question eval covering made-up drugs and prompt injection.

---

## 🚀 Current Implementation

The system exists in two versions. All components are built, evaluated, and pushed to GitHub.

### V2 — RAG Pipeline (main_demo.py)

**FastAPI API Gateway**
* Synchronous (`def`) endpoints, so requests don't block the event loop
* Pydantic request validation with Optional typing
* API key security middleware
* Input validation: empty/oversized messages rejected
* RESTful architecture with Swagger docs

**LangGraph 3-Node Pipeline**
* Triage Node: clinical keyword detection, drug name extraction, query rewriting
* Generation Node: FDA context injection, Gemini JSON output, grounding instruction
* Telemetry Node: JSON parsing with regex fallback, risk level, confidence, emotion

**Retrieval-Augmented Generation**
* ChromaDB vector storage: 830 openFDA label records, split into 11,898 section chunks
* Local all-MiniLM-L6-v2 embeddings (free, unlimited)
* CrossEncoder reranking: ms-marco-MiniLM-L-6-v2
* Retrieval guards: question must name a drug found in the chunk's drug name, a named drug never gets another drug's chunks, near-duplicate chunks dropped
* Section-aware retrieval: intent (dosage, interactions, pregnancy, etc.) selects label sections first, then widens if too little is found
* Query rewriting: drug names + clinical intent keywords + partner drug for interactions
* openFDA API auto-ingestion on startup with duplicate removal

**Observability**
* Retrieval distance logged per query
* Latency tracked per request
* Audit trail: session, risk tier, confidence, evidence blocks used

### V3 — Controlled Agentic Pipeline (main_agentic.py)

**LangGraph Agent**
* Deterministic local routing (free): out-of-scope gets a fixed scope reply, no usable evidence gets a fixed refusal, both with 0 Gemini calls
* Gemini writes the final answer and risk level from retrieved evidence (max 1 call per question)
* `ROUTER_MODE=llm` restores the old LLM router (experimental)
* MemorySaver for session continuity, with per-session evidence isolation

**FDA Tool (check_fda_database)**
* Query rewriting: drug names + clinical intent
* ChromaDB retrieval: 40 candidates, CrossEncoder reranks to the best 8
* Returns ranked evidence with reranker scores and vector distances

**Evaluation Framework**
* 50 questions, 7 categories (easy, hard, ambiguous, out-of-scope, made-up drugs, drugs not in DB, prompt injection)
* Retrieval 100%, correct 100% (lexical check), judge faithfulness 0.984, judge correctness 1.0 (Gemini judge, 31 answered questions)
* 0.68 Gemini calls per question (max 1), latency p50 7.2s / p95 14.3s
* Limits: tuned on this set, judge is Gemini judging Gemini, held-out set pending
* Scripts in /evaluation

**Frontend UI**
* Custom HTML/CSS/JS chat interface
* Live telemetry panel (risk, confidence, emotion)
* Audit trail display
* Served directly from FastAPI

**Containerization & Deployment**
* Dockerized with Azure Container Registry
* Azure Container Apps hosting (was live; free tier ended, deploy steps disabled)
* GitHub Actions CI runs the regression tests on every push (no auto-deploy now)
  
---


## 🏗️ System Architecture

### V2 — RAG Pipeline

```text
User Query (Frontend UI)
        │
        ▼
FastAPI API Gateway
(API Key Security + Pydantic Validation + Input Sanitization)
        │
        ▼
LangGraph 3-Node Pipeline
        │
        ▼
Triage Node
(Query Rewriting + Drug Extraction + Keyword Detection)
        │
        ▼
ChromaDB Vector Search
(830 FDA Label Records → 11,898 Section Chunks + Local Embeddings + L2 Distance Threshold)
        │
        ▼
CrossEncoder Reranking
(ms-marco-MiniLM-L-6-v2)
        │
        ▼
Generation Node
(Gemini 2.5 Flash + FDA Context Injection + JSON Output)
        │
        ▼
Telemetry Node
(JSON Parsing + Risk Level + Confidence + Emotion)
        │
        ▼
Structured Clinical Response + Full Audit Trail
(retrieval_distance + latency_ms + evidence_sources)
```

### V3 — Controlled Agentic Pipeline

```text

User Query
        │
        ▼
FastAPI API Gateway
(API Key Security + Pydantic Validation)
        │
        ▼
LangGraph Agent
(Local router, 0 Gemini calls)
        │
   ┌────┴─────────────────┬───────────────────────┐
   ▼                      ▼                       ▼
Clinical Query       Out-of-Scope Query      Ambiguous Query
   │                      │                       │
   ▼                      ▼                       ▼
check_fda_database()  Fixed scope reply      Clarifying reply
 1. Intent → label sections (Dosage, Interactions...)
 2. Drug-name guard (wrong-drug chunks removed)
 3. ChromaDB: 40 candidates
 4. CrossEncoder rerank → top 8, duplicates dropped
   │
   ├── No usable evidence → Fixed "insufficient FDA data" reply (0 calls)
   │
   ▼
FDA Evidence
   │
   ▼
Agent Node
(Gemini, 1 call max: grounded answer + risk level)
        │
        ▼
Response + llm_calls + prompt_tokens + output_tokens + risk_level + latency
```

---
## 📁 Project Structure

```text
app/
├── main_demo.py          # V2: FastAPI + LangGraph RAG (port 8000, ingest)
├── main_agentic.py       # V3: FastAPI + LangGraph controlled agent (port 8001)
├── chunking.py           # Section chunking (900 chars, 150 overlap, metadata)
├── frontend.py           # Frontend routes
├── main_demo_backup.py   # V2 backup (delete in cleanup)
└── utils/
    └── logger.py         # Structured logging

evaluation/
├── eval_set.json         # 50-question eval (7 categories)
├── run_eval.py           # Retrieval, faithfulness, correctness, cost, judge
└── runs/                 # Saved results (retrieval_*.json, judge.json)

tests/
├── test_routing.py
├── test_retrieval_guards.py
├── test_eval_scoring.py
├── test_call_budget.py
├── test_agentic_session.py
├── test_chunking.py
├── test_evidence_selection.py
└── test_v2_startup.py

static/
└── index.html            # Frontend UI

screenshots/              # Testing proof

.github/
└── workflows/
    └── deploy.yml        # GitHub Actions CI (deploy disabled)

Dockerfile
requirements.txt
RULES.md                  # Project rules
README.md
```


---

## 🧠 AI Workflow

### V2 — RAG Pipeline (main_demo.py)
1. Pharmacist submits query via frontend UI
2. FastAPI validates and routes the request
3. **Triage Node**: detects clinical keywords, rewrites query with drug names + intent keywords
4. **RAG Node**: ChromaDB semantic search over 11,898 section chunks (from 830 FDA label records), CrossEncoder reranks them
5. **Generation Node**: Gemini 2.5 Flash receives the retrieved FDA context and generates a structured JSON response
6. **Telemetry Node**: parses JSON, extracts risk level, confidence score, emotion state
7. Structured clinical response returned with audit trail including retrieval distance and latency

### V3 — Controlled Agentic Pipeline (main_agentic.py)
1. Pharmacist submits query via FastAPI
2. **Local router (0 LLM calls)**: keyword check for clinical scope
   - Out of scope → fixed scope reply with emergency line (0 calls)
3. **Intent detection (0 calls)**: maps the question to label sections (Dosage, Interactions, Boxed Warning, Pregnancy, etc.). Injected text after "SYSTEM:" is ignored.
4. **Retrieval**: ChromaDB returns 40 candidates. Drug-name guard: the question must name a drug in the chunk, and a named drug never receives another drug's chunks. Interaction searches include the partner drug.
5. **Rerank + dedupe**: CrossEncoder narrows to the best 8 chunks, near-duplicate chunks are dropped
6. No usable evidence → fixed "I don't have sufficient FDA data" reply (0 calls). Risk level is capped at moderate.
7. **Generation (1 Gemini call)**: answers only from the retrieved evidence, with risk level
8. Response returned with risk level, `llm_calls`, `prompt_tokens`, `output_tokens` and latency

`ROUTER_MODE=local` is the default. `ROUTER_MODE=llm` is an experimental LLM router.

---
## 🔄 Knowledge Ingestion Pipeline

```text
openFDA API (830 FDA drug labels)
│
▼
Auto-ingest on Startup (BackgroundTasks)
│
▼
Text Chunking (drug + interactions + contraindications + side effects + warnings + dosage)
│
▼
Duplicate removal (seen_documents set)
│
▼
Local all-MiniLM-L6-v2 Embeddings (free, unlimited, no API quota)
│
▼
ChromaDB Vector Store (PersistentClient)
│
▼
Query Rewriting + CrossEncoder Reranking
│
▼
RAG or Agentic Retrieval Layer
```

---

## ☁️ Production Infrastructure

| Component | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| AI Orchestration | LangGraph 3-node pipeline |
| Vector Database | ChromaDB + Gemini Embeddings |
| LLM | Gemini 2.5 Flash |
| Containerization | Docker |
| Cloud Hosting | Azure Container Apps (was live; free tier ended) |
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
| AI Orchestration | LangGraph (RAG pipeline + Agentic loop) |
| LLM | Gemini 2.5 Flash (google-genai SDK) |
| Embeddings | Local all-MiniLM-L6-v2 (sentence-transformers) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Vector Database | ChromaDB PersistentClient |
| Data Validation | Pydantic v2 |
| Containerization | Docker |
| Cloud Platform | Azure Container Apps (was live; free tier ended) |
| Container Registry | Azure Container Registry |
| CI/CD | GitHub Actions |
| Data Source | openFDA API (830 FDA drug labels) |
| Frontend | HTML + CSS + JavaScript |
| Evaluation | Custom eval framework (retrieval + agent grounding metrics) |

---

## 🎯 What This Project Demonstrates

| AI Engineering Concept | Implementation |
|---|---|
| LLM Integration | Gemini 2.5 Flash via google-genai SDK |
| Retrieval-Augmented Generation | ChromaDB + local embeddings + 830 openFDA label records (11,898 section chunks) |
| Agentic AI | LangGraph controlled agent (V3): deterministic safety routing, 1 Gemini call max per question |
| AI Orchestration | LangGraph state graph: V2 3-node pipeline, V3 controlled agent (no open-ended loop) |
| Vector Database | ChromaDB PersistentClient |
| Semantic Search | Local all-MiniLM-L6-v2 sentence-transformers |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Query Rewriting | Drug-name extraction, intent-to-section mapping, partner-drug-aware interaction search |
| Structured Output | Gemini JSON output + Pydantic validation |
| Evaluation | 50-question eval (7 categories incl. made-up drugs, prompt injection): retrieval, faithfulness (lexical + Gemini judge), correctness, cost |
| API Engineering | FastAPI + Pydantic validation + input sanitization |
| Containerization | Docker + Azure Container Registry |
| CI | GitHub Actions (tests on every push; deploy disabled) |
| Cloud Deployment | Azure Container Apps (earlier; now disabled) |
| MLOps | Auto-ingestion on startup + audit logging + retrieval distance tracking |
| Security | API Key middleware + secret management |
| Frontend Integration | Custom HTML/JS served from FastAPI |
| Reliability & Safety | Drug-name guard, fixed refusal when no evidence, injection-safe intent detection, regression tests for every bug |
| Cost & Latency | 0.68 calls/question (max 1), p50 7.2s / p95 14.3s, ~1,900 prompt tokens |

## Performance Metrics

| Metric | V1 (Baseline) | V2 (RAG Pipeline) | V3 (Agentic) |
|---|---|---|---|
| Direct Lookup Accuracy | 80% (4/5) | 100% (7/7) | 100% (14/14) |
| Drug Interaction / Hard Questions | 100% (4/4) | 83% (5/6) | 100% (10/10) |
| Out-of-Scope Rejection | Not tested | 100% (3/3) | 100% (6/6) |
| Records in ChromaDB | 19 FDA labels | 444 FDA labels | 830 FDA labels (11,898 section chunks) |
| Similarity Threshold | None | 0.95 L2 distance | Drug-name guard + rerank to top 8 (distance cutoff replaced) |
| Eval Test Cases | 10 queries | 24 queries | 50 queries (7 categories) |
| p95 Latency | ~16s (cold start) | ~5-8s (local) | 14.3s (p50 8.6s, includes Gemini call) |
| Cost per Request | ~$0.0008 | ~$0.0004 | 0.68 Gemini calls (max 1), ~1,900 prompt + ~190 output tokens |
| Retrieval Accuracy | N/A | N/A | 100% |
| Correctness (judge) | N/A | N/A | 1.0 (Gemini judge, 31 answers) |
| Faithfulness (judge) | N/A | N/A | 0.984 (Gemini judge, 31 answers) |



### V2 — additional robustness testing

A harder eval set (`eval_v2_hard.py`, 18 queries) was added to test paraphrased and indirectly-referenced queries, beyond the original direct-name test set:

| Category | Result |
|---|---|
| Direct lookup / interaction / contraindication / out-of-scope | 9/9 (100%) — consistent with original eval |
| Paraphrased drug queries (e.g. "does lisinopril mess with your kidneys?") | 100% (4/4) |
| Indirect drug references (e.g. "my grandma's blood thinner", "blood pressure medicine") | 100% (3/3) |
| Ambiguous, no drug name (e.g. "can I mix my two prescriptions?") | Correctly flagged for manual review, not auto-scored (2/2) |


Indirect references initially scored 33% (1/3) due to two compounding bugs: the retrieval step only ran the LLM-based drug resolver when zero drugs were found directly (missed multi-drug queries where one drug was named and another only described), and the reranker scored chunks
against the raw query text instead of the resolved drug name, discarding correctly-retrieved evidence. Both fixed, and confirmed by re-running all 18 queries
(executed in slices to stay within).


## V3 Agentic Key Findings (updated Oct 2026)

V3 is a controlled agent: safety-critical steps are deterministic (local scope routing, drug-name guard, fixed refusals), and Gemini is used for one thing,writing the answer from retrieved evidence (max 1 call per question). `ROUTER_MODE=local` is the default; `ROUTER_MODE=llm` restores the earlier Gemini-routed version.

**Index:** 830 openFDA label records, split into 11,898 section chunks (~900 characters, 150 overlap; each chunk carries drug and section). Distance metric is L2.

**Evaluation:** 50 questions in 7 categories (easy 14, hard 10, ambiguous 4, out-of-scope 6, made-up drugs 5, real drugs not in the database 5, prompt injection 6). All 50 ran end to end.

| Metric | Result |
|---|---|
| Retrieval OK (38 scored questions) | 100% |
| Correct (rule-based fact check) | 100% |
| Faithfulness, Gemini judge (31 answered questions) | 0.984 |
| Correctness, Gemini judge (31 answered questions) | 1.0 |
| Gemini calls per question | 0.68 average, 1 max |
| Latency (answered questions) | p50 8.6 s, p95 14.3 s |
| Tokens per answered question | ~1,900 prompt, ~190 output |

Retrieval progression on the same questions: 81.6% (old chunks) → 73.7%
(after rebuild, junk matches on the larger index) → 100% (after
drug-name guard, reranking, and section filtering).

**Bugs found by the eval and fixed (each has a regression test):**
- Duplicate chunks from repeated manufacturer text wasted evidence slots.
- Interaction searches did not include the partner drug.
- A missing routing keyword sent "Is tramadol addictive?" to out-of-scope.
- Injected text after "SYSTEM:" steered intent detection to the wrong
  section; intent is now detected on the clean question only.
- The Gemini judge was seeing only half the evidence (3,500 characters),
  which gave a false faithfulness score of 0.79. After the fix: 0.984.

**Guardrails:**
- Out-of-scope questions get a fixed scope reply with an emergency line (0 calls).
- No usable evidence gives a fixed "I don't have sufficient FDA data" reply (0 calls).
- A question must name a drug present in a chunk's drug name, and a named drug never receives another drug's chunks.
- The system does not infer an interaction because two drug names appear together (e.g. metformin + lisinopril: both are in the corpus, but no chunk documents an interaction, so it declines). This was observed in an earlier version and is not one of the 50 eval questions.

**Known limitations:**
- The system was tuned on this eval set, so the numbers are not a held-out measurement. A fresh held-out set is planned.
- The judge is Gemini grading Gemini, on a small sample (31 answers).
- One answer (#3, warfarin boxed warning) was judged partial: the bleeding-risk statement is not word-for-word in the retrieved text.
- Combination-product labels (e.g. amoxicillin/clavulanate) can still appear as evidence for a single-drug question.
- Routing keyword lists are hand-written, so a medical word missing from them can be treated as out of scope.
- Earlier figures (95% routing, 80% grounded, based on Gemini function-calling and a weaker metric) no longer apply.


## How grounded is measured
`evaluation/run_eval.py` runs a 50-question set (`evaluation/eval_set.json`) in 7 categories: easy, hard, ambiguous, out-of-scope, made-up drugs, real drugs possibly not in the database, and prompt injection.

Every question is scored on four things:

* **Retrieval:** the expected drug, section and fact terms must appear in the retrieved chunks.
* **Faithfulness:** a free lexical check (drug names, doses and terms in the answer must appear in the
  evidence), plus an optional Gemini judge (`--judge`) that reads the **full** evidence and rates each
  answer yes / partial / no.
* **Correctness:** expected facts or expected behaviour (answer, refuse, or resist the injection).
* **Cost:** Gemini calls, tokens and latency per question.

Errors and quota failures are marked INVALID, never pass or fail. The judge scores only answers the model actually generated. Fixed refusals and out-of-scope replies (0 calls) are not judged.

The lexical check only looks at words. It cannot catch a subtly wrong claim, which is why the Gemini judge exists. Neither is a human clinical review.

## Known Limitations
* Retrieval covers all 830 indexed FDA label records (11,898 section chunks), not a fixed list of 12 drugs. A question must name a drug that appears in a chunk's drug name, and a named drug never receives another drug's chunks. This guard matches on name, so a combination product (e.g. amoxicillin/clavulanate) can
  still leak into a single-drug question. Planned fix: match the whole drug name.
* "Grounded" is checked by a lexical check plus a Gemini judge on 31 answered questions. The judge is
  Gemini judging Gemini, and the sample is small, so treat the scores as a strong signal, not proof.
  One answer (#3) was rated "partial" because a true bleeding-risk statement was not word-for-word in the
  boxed-warning text.
* The eval set was used to find and fix bugs, so its scores are not a held-out result. A fresh held-out
  set is planned.
* Sessions use in-memory `MemorySaver`: history is lost on restart and not shared across instances.
* Corpus coverage is per-drug, not per-pair. An interaction can go undocumented even when
  both drugs are individually well-represented, and the system correctly refuses to guess
  rather than hallucinate.
* Gemini free tier limits evaluation to 20 requests/day. The eval is resumable and stops at a daily call
  cap (default 18), so a full run with the judge can take more than one day.
* ChromaDB collection handles can go stale after an external rebuild, requiring a server restart to re-resolve (see Engineering Challenges).
* V3 routing is local and rule-based by default (keyword lists for scope and for the label sections a
  question needs), at most 1 Gemini call per question (0.68 on average). The old LLM router is still
  available with `ROUTER_MODE=llm`. The earlier 95% (19/20) tool-routing figure is superseded by the
  50-question eval. Because the keyword lists are hand-written, a medical word missing from them can be
  wrongly treated as out of scope (this happened with "addictive" before it was added).
* V3 is a controlled agent: deterministic safety routing plus LLM answer generation. It has no open-ended
  agent loop and does not retry when evidence is weak.
* Not load-tested. Latency is p50 7.2 s / p95 14.3 s for a single user, and the free Gemini quota
  would not support many concurrent users.



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
| ChromaDB collection handle goes stale after rebuild | `get_or_create_collection()` resolves a collection by internal UUID at server startup; if the DB is rebuilt afterward (e.g. via `main_demo.py`'s ingest), that UUID no longer exists — confirmed via `Collection [uuid] does not exist` error in logs | Restart the agentic server after any DB rebuild so it re-resolves a fresh collection handle; documented as a required step in deployment/ops notes |
| Gemini returning JSON wrapped in markdown backticks | Gemini occasionally wraps JSON in ` ```json ``` ` despite prompt instructions | Added `except Exception` catch in telemetry node — falls back to regex parsing when JSON parsing fails |
| Local embedding quota hit during large ingest | Gemini free tier embedding API has 1000 requests/day limit — 200+ records exhausted it | Switched to local `all-MiniLM-L6-v2` sentence-transformers model — free, unlimited, no API calls |
| Pydantic validation crash on null warnings field | `warnings: str = None` not valid in Pydantic v2 — `None` is not a string | Changed to `warnings: Optional[str] = None` with proper `Optional` typing |
| Indirect drug references ("blood thinner") silently dropped when another drug was already named | Resolver only ran when zero drugs were found via substring match — multi-drug queries with one named + one described drug skipped resolution entirely | Removed the gating condition; resolver now runs on every clinical query regardless of what substring matching already found, and appends any newly-resolved drug |
| Correctly-resolved indirect drug still missing from final answer | CrossEncoder reranked candidate chunks against the raw user query text, which never contained the resolved drug name (e.g. "blood pressure medicine" never lexically matches "lisinopril") — so retrieved lisinopril chunks lost the rerank despite being fetched correctly | Reranking query now includes resolved drug names, e.g. `"{user_msg} (lisinopril)"`, so resolved-but-unnamed drugs get proper credit during reranking |
| Single 429 failure corrupted entire eval batches | Gemini calls failed immediately on rate-limit errors with no retry, so one quota hit mid-batch reported false negatives across the rest of the run | Added `call_gemini_with_retry()` — parses Gemini's own suggested `retryDelay` from the error and retries up to 2 times before failing; eval script timeout increased from 30s to 180s to accommodate |
| Eval reported false 100% accuracy | Hit detection checked only the HTTP status, never whether the right drug was retrieved | Rewrote scoring to check drug, section and fact terms per question. Now 50 questions in 7 categories; errors and quota failures are INVALID, never pass or fail |
| Session state leak | A second turn reused the first turn's evidence | Per-request state, plus a regression test |
| Endpoints blocked the server | Async endpoints ran blocking retrieval and LLM code | Switched to plain `def` so FastAPI runs them in a threadpool |
| Warfarin's aspirin warning was cut off | Whole-section text was truncated before embedding | Section chunking (900 chars, 150 overlap, each chunk prefixed with drug and section). 828 to 11,898 chunks (830 label records) |
| Junk matches after the DB grew 10x | The fixed L2 distance cutoff (0.95) stopped filtering at the larger size | Drug-name guard: the question must name a drug in the chunk's drug name, and a named drug never gets another drug's chunks |
| Duplicate chunks wasted evidence slots (#13) | Manufacturers repeat identical label text | Near-duplicate removal after reranking |
| Interaction answers missed the partner drug (#16, #17) | Search used only the first drug | Interaction search now includes the partner drug's name |
| "Is tramadol addictive?" got the out-of-scope reply (#42) | Routing keyword list had no "addictive" or "withdrawal" | Extended the keywords, added a routing test |
| Prompt injection steered retrieval (#47) | Injected "any dose" text triggered the Dosage intent, so the model answered from its own knowledge | Intent detection ignores everything after "SYSTEM:", plus a regression test |
| LLM judge scored 0.79 faithfulness | The judge saw only the first 3,500 characters of evidence | Raised to 9,000 characters and re-judged: faithfulness 0.984, correctness 1.0 (31 answers; Gemini judges Gemini) |
| Too many Gemini calls on a 20/day quota | Old design used 2 calls per question (LLM router plus answer) | Local routing and a fixed refusal when evidence is missing. 0.68 calls per question, max 1. Eval is resumable with an 18-call daily cap. `ROUTER_MODE=llm` kept as an option |


### 🔬 Research & Development

The exploratory data analysis, severity classification modeling, and initial workflow prototyping for this clinical support tool were developed in Google Colab. You can view and run the experimental notebook directly via the link below:

* [Launch Active Colab Notebook](https://colab.research.google.com/drive/1KwWJRlIynOMbfM8f3zcUym4lCNytyoj8#scrollTo=MUOtHKZA2L7n)
