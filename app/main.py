from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from app.core.config import settings
import logging
import time
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END

# MLOps Telemetry Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LangGraph-CDSS: %(message)s")
logger = logging.getLogger("cdss.graph")

app = FastAPI(title="Pharmacist CDSS - LangGraph Stateful AI Agent")
client = genai.Client(api_key=settings.GEMINI_API_KEY)

# --- LANGGRAPH STATE DEFINITION ---
class AgentState(TypedDict):
    user_message: str
    current_medication: str
    session_id: str
    detected_emotions: List[str]
    retrieved_context: List[str]
    agent_decision: str
    final_response: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    medication: str = "None"

# Mock Enterprise Data Core
CLINICAL_KNOWLEDGE_BASE = {
    "depression": "NICE Guideline NG222: For severe clinical presentations of depression or hopelessness, do not offer routine self-care matching. Immediate referral to specialized crisis teams is mandatory.",
    "insomnia": "PHQ-9 Index & Sleep Guidelines: High-severity single-point depressive thresholds accompanied by chronic insomnia require active safety oversight.",
    "anxiety": "NICE Guideline CG113: Generalized anxiety disorder presentations should be evaluated for functional impairment. Moderate to severe cases require stepped-care medical interventions."
}

DRUG_INTERACTIONS = {
    ("sertraline", "tramadol"): "CRITICAL INTERACTION: Concomitant use increases the risk of Serotonin Syndrome. Monitor for autonomic instability, neuromuscular changes, and altered mental state.",
    ("ibuprofen", "warfarin"): "HIGH RISK: Concurrent administration significantly increases gastrointestinal bleeding risks. Avoid combination if possible."
}

# --- GRAPH NODE 1: LOCAL EMOTION CLASSIFIER ---
def nlp_emotion_node(state: AgentState) -> AgentState:
    logger.info("🟢 LangGraph Node Run: [nlp_emotion_node]")
    msg_lower = state["user_message"].lower()
    
    # Simulating our local fine-tuned PyTorch DistilBERT model inference pipeline
    emotions = ["monitored_via_agent"]
    if any(w in msg_lower for w in ["hopeless", "depressed", "sad"]):
        emotions.append("severe_sadness")
    if any(w in msg_lower for w in ["anxious", "panic", "worry", "racing"]):
        emotions.append("anxiety")
        
    state["detected_emotions"] = emotions
    return state

# --- GRAPH NODE 2: INTENT ROUTER (LLM REASONING) ---
def llm_router_node(state: AgentState) -> AgentState:
    logger.info("doc Node Run: [llm_router_node]")
    
    prompt = f"""Analyze the pharmacist prompt below. Determine if you need to run specific database queries:
    1. If the message mentions mental health symptoms (anxiety, depression, sleep, panic), answer with 'ROUTE_TO_GAMES'.
    2. If the message mentions specific drugs or prescriptions, answer with 'ROUTE_TO_DRUGS'.
    3. If the message is a simple greeting or general conversation, answer with 'ROUTE_TO_FINAL'.
    
    Pharmacist Input: {state['user_message']}
    Current Medication: {state['current_medication']}
    
    Answer strictly with just one option string: ROUTE_TO_GAMES or ROUTE_TO_DRUGS or ROUTE_TO_FINAL."""
    
    response = client.models.generate_content(model=settings.GEMINI_MODEL, contents=prompt)
    decision = response.text.strip()
    
    # Fallback guardrail
    if "ROUTE_TO_DRUGS" in decision:
        state["agent_decision"] = "drugs"
    elif "ROUTE_TO_GAMES" in decision:
        state["agent_decision"] = "guidelines"
    else:
        state["agent_decision"] = "final"
        
    logger.info(f"🎯 LangGraph Router Decision: Determined path -> {state['agent_decision']}")
    return state

# --- GRAPH NODE 3: DYNAMIC CHROMADB RAG TOOL ---
def rag_retrieval_node(state: AgentState) -> AgentState:
    logger.info("🔧 LangGraph Node Run: [rag_retrieval_node]")
    msg_lower = state["user_message"].lower()
    found_chunks = []
    
    for key, text in CLINICAL_KNOWLEDGE_BASE.items():
        if key in msg_lower or (key == "depression" and "sad" in msg_lower):
            found_chunks.append(text)
            
    if not found_chunks:
        found_chunks.append("General Guidance: Monitor symptom baseline and maintain a clinical observation log.")
        
    state["retrieved_context"] = found_chunks
    return state

# --- GRAPH NODE 4: MOLECULAR DRUG INTERACTION TOOL ---
def drug_interaction_node(state: AgentState) -> AgentState:
    logger.info("🔧 LangGraph Node Run: [drug_interaction_node]")
    msg_lower = state["user_message"].lower()
    med_lower = state["current_medication"].lower()
    
    interaction_alert = "No severe drug-drug interaction matches registered in current configuration layout rules."
    
    for pair, alert in DRUG_INTERACTIONS.items():
        if pair[0] in msg_lower and pair[1] in med_lower:
            interaction_alert = alert
        elif pair[1] in msg_lower and pair[0] in med_lower:
            interaction_alert = alert
            
    state["retrieved_context"] = [interaction_alert]
    return state

# --- GRAPH NODE 5: RESPONSE SYNTHESIZER AND FORMATTER ---
def response_generator_node(state: AgentState) -> AgentState:
    logger.info("🏁 LangGraph Node Run: [response_generator_node]")
    
    system_prompt = """You are a clinical decision support assistant for pharmacists.
    Synthesize the retrieved database context and the conversation history to generate a structured clinical response.
    Never diagnose. Recommendation for GP referral is mandatory for moderate/severe presentation flags.
    You must include the [CLINICAL_SUMMARY] tag structure exactly at the end of your response.
    
    [CLINICAL_SUMMARY]
    severity: <none|minimal|mild|moderate|severe>
    key_symptoms: ["symptom1", "symptom2"]
    recommended_action: "<what pharmacist should do>"
    follow_up_question: "<single question to ask patient>"
    [/CLINICAL_SUMMARY]"""
    
    context_str = " / ".join(state["retrieved_context"]) if state["retrieved_context"] else "No auxiliary context fetched."
    emotions_str = ", ".join(state["detected_emotions"])
    
    prompt = f"""Context blocks retrieved from databases: {context_str}
    Detected NLP Emotions: {emotions_str}
    Pharmacist Input: {state['user_message']}
    Medication profile: {state['current_medication']}
    
    Synthesize and output the full clinical response followed by the [CLINICAL_SUMMARY] structure."""
    
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt)
    )
    
    state["final_response"] = response.text
    return state

# --- CONDITIONAL ROUTER EDGE LOGIC ---
def route_decision_edge(state: AgentState):
    if state["agent_decision"] == "drugs":
        return "route_to_drugs"
    elif state["agent_decision"] == "guidelines":
        return "route_to_guidelines"
    return "route_to_final"

# --- COMPILING THE LANGGRAPH STATE MACHINE ---
workflow = StateGraph(AgentState)

# Add our custom processing nodes to the graph
workflow.add_node("nlp_emotion", nlp_emotion_node)
workflow.add_node("llm_router", llm_router_node)
workflow.add_node("rag_retrieval", rag_retrieval_node)
workflow.add_node("drug_interaction", drug_interaction_node)
workflow.add_node("response_generator", response_generator_node)

# Establish the execution routes
workflow.set_entry_point("nlp_emotion")
workflow.add_edge("nlp_emotion", "llm_router")

# Add the conditional routing decision point
workflow.add_conditional_edges(
    "llm_router",
    route_decision_edge,
    {
        "route_to_drugs": "drug_interaction",
        "route_to_guidelines": "rag_retrieval",
        "route_to_final": "response_generator"
    }
)

# Connect intermediate execution steps back to final compiler
workflow.add_edge("drug_interaction", "response_generator")
workflow.add_edge("rag_retrieval", "response_generator")
workflow.add_edge("response_generator", END)

# Compile graph state machine
app_agent = workflow.compile()

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    start_time = time.time()
    
    # Initialize the centralized graph memory state payload
    initial_state: AgentState = {
        "user_message": payload.message,
        "current_medication": payload.medication,
        "session_id": payload.session_id,
        "detected_emotions": [],
        "retrieved_context": [],
        "agent_decision": "",
        "final_response": ""
    }
    
    try:
        logger.info(f"🚀 Triggering LangGraph Workflow Engine for Session: {payload.session_id}")
        
        # Execute the compiled graph synchronously over state turns
        final_output = app_agent.invoke(initial_state)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"🏁 LangGraph Execution Stream Complete | Combined Latency: {latency:.2f}ms")
        
        return {
            "session_id": payload.session_id,
            "clinical_guidance": final_output["final_response"],
            "detected_emotions": final_output["detected_emotions"],
            "retrieved_database_context": final_output["retrieved_context"]
        }
    except Exception as e:
        logger.error(f"❌ LangGraph Runtime Crash: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
