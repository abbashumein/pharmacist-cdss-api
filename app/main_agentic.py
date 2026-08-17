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


def rewrite_clinical_query(query: str) -> str:
    """Rewrite a pharmacist query into focused FDA retrieval terms."""
    lower_query = query.lower()

    found_drugs = [
        drug for drug in TARGET_DRUGS
        if drug in lower_query
    ]

    intent_keywords = []

    if any(word in lower_query for word in [
        "interaction", "interact", "safe", "together", "combine", "mix"
    ]):
        intent_keywords.append("drug interaction")

    if any(word in lower_query for word in [
        "contraindication", "contraindications", "avoid",
        "cannot", "should not"
    ]):
        intent_keywords.append("contraindication")

    if any(word in lower_query for word in [
        "side effect", "side effects", "adverse",
        "reaction", "reactions"
    ]):
        intent_keywords.append("adverse reactions side effects")

    if any(word in lower_query for word in [
        "dose", "dosage", "how much", "mg"
    ]):
        intent_keywords.append("dosage administration")

    if any(word in lower_query for word in [
        "warning", "warnings", "danger"
    ]):
        intent_keywords.append("warnings")

    if found_drugs:
        return " ".join(found_drugs + intent_keywords)

    return query


def check_fda_database(query: str) -> str:
    """
    Agent tool for searching the FDA clinical knowledge corpus.

    Pipeline:
        query rewriting -> ChromaDB retrieval -> CrossEncoder reranking
    """
    try:
        search_query = rewrite_clinical_query(query)

        fresh_collection = chroma_client.get_or_create_collection(name="langchain")
        db_results = collection.query(
            query_texts=[search_query],
            n_results=5,
            include=["documents", "distances"]
        )

        documents = db_results.get("documents", [[]])[0]
        distances = db_results.get("distances", [[]])[0]

        if not documents:
            return "No relevant FDA evidence was found."

        # Rerank retrieved evidence using the Phase 5 CrossEncoder.
        pairs = [[search_query, document] for document in documents]
        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(scores, documents, distances),
            key=lambda item: item[0],
            reverse=True
        )

        evidence = []

        for rank, (score, document, distance) in enumerate(ranked, start=1):
            evidence.append(
                f"Evidence {rank} "
                f"(reranker_score={score:.4f}, "
                f"vector_distance={distance:.4f}):\n"
                f"{document}"
            )

        return (
            f"FDA database search query: {search_query}\n\n"
            + "\n\n".join(evidence)
        )

    except Exception as e:
        sys_logger.error(f"FDA tool error: {str(e)}")
        return "FDA database lookup failed."


# ============================================================
# LANGGRAPH AGENT STATE
# ============================================================

class AgentState(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]] = []
    fda_evidence: str = ""
    final_response: str = ""
    tool_called: bool = False
    risk_level: str = "LOW"


# ============================================================
# AGENT NODES
# ============================================================

AGENT_SYSTEM_PROMPT = """You are a professional Pharmacist AI Assistant.
You have access to an FDA drug database tool.

RULES:
- For any drug-related query: ALWAYS call check_fda_database first
- For out-of-scope queries (weather, stocks, poems): respond directly without calling the tool
- Answer ONLY using FDA evidence retrieved by the tool
- If FDA evidence is insufficient, say "I don't have sufficient FDA data for this query"
- Always end with: "This is AI assistance. Final decision should be made by a licensed pharmacist or doctor."
"""


def agent_node(state: AgentState) -> Dict[str, Any]:
    """Gemini decides whether to call FDA tool or respond directly."""
    user_message = state.messages[-1]["content"] if state.messages else ""

    # Check if query needs FDA tool
    lower_query = user_message.lower()
    drug_keywords = TARGET_DRUGS + ["drug", "medication", "medicine", "dose",
                                    "interaction", "side effect", "contraindication"]
    needs_fda = any(word in lower_query for word in drug_keywords)

    if needs_fda and not state.tool_called:
        # Signal tool call needed
        return {"fda_evidence": "TOOL_NEEDED", "tool_called": False}

    # Generate final response using FDA evidence
    context = state.fda_evidence if state.fda_evidence and state.fda_evidence != "TOOL_NEEDED" else "No FDA evidence retrieved."

    prompt = f"""{AGENT_SYSTEM_PROMPT}

FDA EVIDENCE:
{context}

USER QUERY: {user_message}

Respond with a clear clinical answer."""

    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        final = response.text.strip()
    except Exception as e:
        sys_logger.critical(f"Gemini error: {str(e)}")
        final = "Sorry, I am temporarily unable to respond. Please try again."

    # Determine risk level
    risk = "HIGH" if any(w in final.lower() for w in ["dangerous", "fatal", "lethal", "emergency"]) else \
        "MODERATE" if any(w in final.lower() for w in ["caution", "warning", "risk", "avoid"]) else "LOW"

    return {"final_response": final, "risk_level": risk}


def tool_node(state: AgentState) -> Dict[str, Any]:
    """Execute FDA database tool."""
    user_message = state.messages[-1]["content"] if state.messages else ""
    evidence = check_fda_database(user_message)
    return {"fda_evidence": evidence, "tool_called": True}


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

    fda_evidence = output.get("fda_evidence", "")
    return {
        "session_id": payload.session_id,
        "response": output.get("final_response", "Unable to process request."),
        "fda_evidence_used": bool(fda_evidence) and fda_evidence != "TOOL_NEEDED",
        "tool_called": output.get("tool_called", False),
        "risk_level": output.get("risk_level", "LOW"),
        "latency_ms": round((time.time() - request_start) * 1000)
    }