import os
import re
import time
import json
import urllib.request
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, validator
from fastapi import FastAPI, Depends, HTTPException, status, Security, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from google import genai
from google.genai import types
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.utils.logger import sys_logger
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# CONFIGURATION
# ============================================================

app = FastAPI(title="Pharmacist CDSS — Agentic V3")

# Allow the static frontend (served from V2 on :8000, or from this app on
# :8001) to call this API directly from the browser when the user switches
# modes in Settings. Additive only — does not change any existing route.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Same local embedding model used by CDSS V2
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to the existing local ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="langchain"
)
# ============================================================
# FDA DATABASE TOOL
# ============================================================

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

TARGET_DRUGS = [
    "warfarin", "amiodarone", "aspirin", "ibuprofen",
    "metformin", "lisinopril", "atorvastatin", "omeprazole",
    "amlodipine", "metoprolol", "levothyroxine", "albuterol"
]


# ============================================================
# FDA DATABASE TOOL  (upgraded – per-drug retrieval)
# ============================================================

SIMILARITY_THRESHOLD = 0.95   # same value V2 uses (distance)


def _intent_keywords(lower_query: str) -> List[str]:
    """Extract clinical-intent keywords from a lowercased query string."""
    keywords = []
    if any(w in lower_query for w in ["interaction", "interact", "safe", "together", "combine", "mix"]):
        keywords.append("drug interaction")
    if any(w in lower_query for w in ["contraindication", "contraindications", "avoid", "cannot", "should not"]):
        keywords.append("contraindication")
    if any(w in lower_query for w in ["side effect", "side effects", "adverse", "reaction", "reactions"]):
        keywords.append("adverse reactions side effects")
    if any(w in lower_query for w in ["dose", "dosage", "how much", "mg"]):
        keywords.append("dosage administration")
    if any(w in lower_query for w in ["warning", "warnings", "danger"]):
        keywords.append("warnings")
    return keywords


def check_fda_database(query: str) -> Dict[str, Any]:
    """
    Returns a structured dict the agent can reason over.
    Key fields:
      - has_usable_evidence: bool
      - retrieval_distance: float | None
      - evidence_text: str          (human-readable for the prompt)
      - chunks: list[str]           (raw)

    FIX: each named drug is queried SEPARATELY (matches the V2 fix in
    app/main_demo.py's triage_and_retrieve_node). A single combined query
    like "warfarin ibuprofen drug interaction" makes both drugs compete for
    the same top-N vector search slots — if one drug has stronger/more
    generic matches (e.g. an OTC ibuprofen dosage chunk), it can crowd out
    the other drug's actually-relevant chunk (e.g. warfarin's record noting
    ibuprofen as an interaction risk) before reranking ever sees it.
    Querying per-drug guarantees each one gets an uncontested retrieval
    pass; the threshold + reranker below still decide what's usable.
    """
    try:
        lower_query = query.lower()
        found_drugs = [drug for drug in TARGET_DRUGS if drug in lower_query]
        intent_string = " ".join(_intent_keywords(lower_query))

        usable = []
        all_distances = []

        if found_drugs:
            for drug in found_drugs:
                drug_query = f"{drug} {intent_string}".strip()
                db_results = collection.query(
                    query_texts=[drug_query],
                    n_results=5,
                    include=["documents", "distances"]
                )
                documents = db_results.get("documents", [[]])[0]
                distances = db_results.get("distances", [[]])[0]

                if not documents:
                    continue

                pairs = [[drug_query, doc] for doc in documents]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(scores, documents, distances), key=lambda x: x[0], reverse=True)

                for score, doc, dist in ranked:
                    all_distances.append(dist)
                    if dist < SIMILARITY_THRESHOLD:
                        usable.append((score, doc, dist, drug))
        else:
            search_query = f"{query} {intent_string}".strip()
            db_results = collection.query(
                query_texts=[search_query],
                n_results=8,
                include=["documents", "distances"]
            )
            documents = db_results.get("documents", [[]])[0]
            distances = db_results.get("distances", [[]])[0]

            if documents:
                pairs = [[search_query, doc] for doc in documents]
                scores = reranker.predict(pairs)
                ranked = sorted(zip(scores, documents, distances), key=lambda x: x[0], reverse=True)
                for score, doc, dist in ranked:
                    all_distances.append(dist)
                    if dist < SIMILARITY_THRESHOLD:
                        usable.append((score, doc, dist, None))

        if not usable:
            return {
                "has_usable_evidence": False,
                "retrieval_distance": min(all_distances) if all_distances else None,
                "evidence_text": "No sufficiently relevant FDA evidence found for this query.",
                "chunks": []
            }

        # Keep the strongest evidence overall, capped at 5 for prompt size
        usable = sorted(usable, key=lambda x: x[0], reverse=True)[:5]
        best_distance = min(d for _, _, d, _ in usable)

        evidence_lines = []
        for rank, (score, doc, dist, drug) in enumerate(usable, 1):
            label = f" [{drug}]" if drug else ""
            evidence_lines.append(
                f"Evidence {rank}{label} (reranker={score:.4f}, distance={dist:.4f}):\n{doc}"
            )

        return {
            "has_usable_evidence": True,
            "retrieval_distance": best_distance,
            "evidence_text": (
                f"FDA database search for: {query}\n\n"
                + "\n\n".join(evidence_lines)
            ),
            "chunks": [doc for _, doc, _, _ in usable]
        }

    except Exception as e:
        sys_logger.error(f"FDA tool error: {str(e)}")
        return {
            "has_usable_evidence": False,
            "retrieval_distance": None,
            "evidence_text": "FDA database lookup failed.",
            "chunks": []
        }


# ============================================================
# LANGGRAPH AGENT STATE
# ============================================================

class AgentState(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]] = []
    fda_evidence: str = ""                  # human-readable text for prompt
    has_usable_evidence: bool = False       # NEW
    retrieval_distance: Optional[float] = None  # NEW
    final_response: str = ""
    tool_called: bool = False
    risk_level: str = "LOW"


# ============================================================
# AGENT NODES
# ============================================================

AGENT_SYSTEM_PROMPT = """You are a professional Pharmacist AI Assistant.
You have access to an FDA drug database tool.

CRITICAL RULES (never break):
0. If the query is NOT about a drug or medication (e.g. general conversation,
   weather, poems, unrelated topics), rules 1-4 below do not apply — just
   respond normally and briefly, then still follow rule 5 and rule 6.
1. For any drug-related query: ALWAYS call the FDA tool first.
2. Answer ONLY from the FDA evidence that is provided.
3. For a drug-related query only: if has_usable_evidence is False, or the
   evidence does not explicitly discuss the asked interaction / topic, you
   MUST reply with exactly:
   "I don't have sufficient FDA data for this query."
4. Do NOT infer an interaction just because two drug names appear in the question.
5. At the very end of your response (after the clinical text) emit one machine-readable line:
   RISK_LEVEL: LOW
   or
   RISK_LEVEL: MODERATE
   or
   RISK_LEVEL: HIGH
   This line will be stripped before showing the answer to the user.
6. Always end the visible answer with:
   "This is AI assistance. Final decision should be made by a licensed pharmacist or doctor."
"""


def tool_node(state: AgentState) -> Dict[str, Any]:
    """Execute FDA database tool and store structured result."""
    user_message = state.messages[-1]["content"] if state.messages else ""
    result = check_fda_database(user_message)

    return {
        "fda_evidence": result["evidence_text"],
        "has_usable_evidence": result["has_usable_evidence"],
        "retrieval_distance": result["retrieval_distance"],
        "tool_called": True
    }


def agent_node(state: AgentState) -> Dict[str, Any]:
    """Gemini decides whether to call FDA tool or respond."""
    user_message = state.messages[-1]["content"] if state.messages else ""

    lower_query = user_message.lower()
    drug_keywords = TARGET_DRUGS + [
        "drug", "medication", "medicine", "dose",
        "interaction", "side effect", "contraindication"
    ]
    needs_fda = any(word in lower_query for word in drug_keywords)

    # First pass → request the tool
    if needs_fda and not state.tool_called:
        return {"fda_evidence": "TOOL_NEEDED", "tool_called": False}

    # Second pass → we already have evidence (or the tool said none)
    context = state.fda_evidence if state.fda_evidence and state.fda_evidence != "TOOL_NEEDED" else "No FDA evidence retrieved."

    if state.tool_called:
        # Clinical query: the FDA tool ran, so give the model the retrieval
        # signal and evidence to ground its answer in.
        signal = (
            f"RETRIEVAL CONFIDENCE SIGNAL: "
            f"has_usable_evidence={state.has_usable_evidence}, "
            f"best_distance={state.retrieval_distance}"
        )

        prompt = f"""{AGENT_SYSTEM_PROMPT}

{signal}

FDA EVIDENCE:
{context}

USER QUERY: {user_message}

Respond with a clear clinical answer and the RISK_LEVEL line."""
    else:
        # Non-clinical query: the FDA tool never ran. Don't show the model
        # any "has_usable_evidence=False" / "No FDA evidence retrieved"
        # framing at all — that framing alone was enough to pull Gemini
        # toward the FDA-refusal pattern even with rule 0 telling it not
        # to, e.g. on "write a poem about the ocean." Give it a plain
        # prompt instead so there's nothing FDA-shaped to latch onto.
        prompt = f"""{AGENT_SYSTEM_PROMPT}

USER QUERY: {user_message}

This is not a drug-related question. Respond normally per rule 0, then end with the RISK_LEVEL line."""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip()
    except Exception as e:
        sys_logger.critical(f"Gemini error: {str(e)}")
        raw = "Sorry, I am temporarily unable to respond. Please try again.\nRISK_LEVEL: LOW"

    # Parse RISK_LEVEL marker and clean the visible answer
    risk_match = re.search(r"RISK_LEVEL:\s*(LOW|MODERATE|HIGH)", raw, re.IGNORECASE)
    risk = risk_match.group(1).upper() if risk_match else "LOW"

    # Remove the marker from what the user sees
    clean = re.sub(r"\s*RISK_LEVEL:\s*(LOW|MODERATE|HIGH)\s*", "", raw, flags=re.IGNORECASE).strip()

    # Hard guardrail: FDA tool ran but found no usable evidence → never claim
    # HIGH risk + force the "insufficient data" message. Gated on
    # tool_called as well as has_usable_evidence: for a non-clinical query
    # (e.g. "write a poem about the ocean") the tool never runs at all, so
    # has_usable_evidence is False by default — that must NOT be mistaken
    # for "FDA data was insufficient", or every out-of-scope question
    # incorrectly gets the FDA-shortfall message instead of a normal reply.
    if state.tool_called and not state.has_usable_evidence:
        if "sufficient FDA data" not in clean.lower():
            clean = "I don't have sufficient FDA data for this query.\n\nThis is AI assistance. Final decision should be made by a licensed pharmacist or doctor."
        if risk == "HIGH":
            risk = "MODERATE"          # or "LOW" / "UNKNOWN" – match your clinical policy

    return {
        "final_response": clean,
        "risk_level": risk
    }


def should_call_tool(state: AgentState) -> str:
    """Routing function — decides next node."""
    if state.fda_evidence == "TOOL_NEEDED":
        return "tool"
    return "end"


# ============================================================
# LANGGRAPH GRAPH
# ============================================================

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_call_tool,
    {"tool": "tool", "end": END}
)
workflow.add_edge("tool", "agent")

cdss_agent = workflow.compile(checkpointer=MemorySaver())

# ============================================================
# API SECURITY
# ============================================================

API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def validate_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("CDSS_API_KEY")
    if not expected_key:
        raise RuntimeError("CDSS_API_KEY environment variable must be set")
    if not api_key or str(api_key).strip() != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-KEY header"
        )
    return api_key


# ============================================================
# FASTAPI ENDPOINT
# ============================================================

class AgentChatRequest(BaseModel):
    session_id: str
    message: str

    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message too long')
        return v.strip()


@app.post("/chat", dependencies=[Depends(validate_api_key)])
async def agent_chat(payload: AgentChatRequest):
    request_start = time.time()

    config = {"configurable": {"thread_id": payload.session_id}}
    initial_state = {
        "session_id": payload.session_id,
        "messages": [{"role": "user", "content": payload.message}]
    }

    output = cdss_agent.invoke(initial_state, config)

    # Now correctly reflects usable evidence, not just “tool ran”
    return {
        "session_id": payload.session_id,
        "response": output.get("final_response", "Unable to process request."),
        "fda_evidence_used": output.get("has_usable_evidence", False),   # ← fixed
        "tool_called": output.get("tool_called", False),
        "risk_level": output.get("risk_level", "LOW"),
        "retrieval_distance": output.get("retrieval_distance"),
        "latency_ms": round((time.time() - request_start) * 1000)
    }