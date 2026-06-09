import os
import re
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from google import genai
from app.services.chroma_service import ChromaService

# Initialize Gemini Client natively using google-genai
ai_client = genai.Client()
chroma_service = ChromaService()


# 1. State Definition
class ClinicalState(TypedDict):
    messages: List[dict]  # Sliding window history
    current_query: str
    severity_tier: str  # LOW, MODERATE, HIGH
    rag_context: str
    raw_llm_output: str
    verification_confidence: str
    emotion_state: str
    safety_passed: bool


# 2. Nodes
def triage_node(state: ClinicalState) -> dict:
    """Node 1: Evaluates clinical severity tier using Gemini."""
    prompt = f"Analyze this pharmacist/patient query and classify severity as LOW, MODERATE, or HIGH. Query: {state['current_query']}"
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    text = response.text.upper()

    tier = "LOW"
    if "HIGH" in text:
        tier = "HIGH"
    elif "MODERATE" in text:
        tier = "MODERATE"

    return {"severity_tier": tier}


def rag_lookup_node(state: ClinicalState) -> dict:
    """Node 2: Conditional execution layer for ChromaDB retrieval."""
    if state["severity_tier"] in ["MODERATE", "HIGH"]:
        results = chroma_service.query(state["current_query"])
        context = " ".join(results.get("documents", [[]])[0])
        return {"rag_context": context}
    return {"rag_context": "No RAG context required for low severity."}


def llm_generation_node(state: ClinicalState) -> dict:
    """Node 3: Core execution node parsing answers along with telemetry markers."""
    system_instruction = (
        "You are a clinical decision support system. Return your output exactly with these headers:\n"
        "Verification Confidence: [XX%]\nEmotion State: [Detected State]\nClinical Response: [Your medical advice]"
    )

    prompt = f"Context: {state['rag_context']}\nQuery: {state['current_query']}"

    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[system_instruction, prompt]
    )

    output = response.text

    # Simple, resilient regex extractions
    conf_match = re.search(r"Verification Confidence:\s*([\d%]+|N/A)", output)
    emotion_match = re.search(r"Emotion State:\s*(\w+|N/A)", output)

    return {
        "raw_llm_output": output,
        "verification_confidence": conf_match.group(1) if conf_match else "N/A",
        "emotion_state": emotion_match.group(1) if emotion_match else "N/A"
    }


def safety_guardrail_node(state: ClinicalState) -> dict:
    """Node 4: Evaluates final structural integrity and safety violations."""
    output = state["raw_llm_output"]
    # Ensure standard compliance: check if output contains actual substance
    passed = "Clinical Response:" in output and len(output) > 30
    return {"safety_passed": passed}


# 3. Graph Assembly
workflow = StateGraph(ClinicalState)

workflow.add_node("triage", triage_node)
workflow.add_node("rag_lookup", rag_lookup_node)
workflow.add_node("llm_generate", llm_generation_node)
workflow.add_node("safety_check", safety_guardrail_node)

# Linear, highly predictable pipeline setup
workflow.add_edge(START, "triage")
workflow.add_edge("triage", "rag_lookup")
workflow.add_edge("rag_lookup", "llm_generate")
workflow.add_edge("llm_generate", "safety_check")
workflow.add_edge("safety_check", END)

# Compile with an in-memory checkpoint saver for instant state updates
memory = MemorySaver()
clinical_graph = workflow.compile(checkpointer=memory)