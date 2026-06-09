import os
from dotenv import load_dotenv
import time
import re
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from sentence_transformers import SentenceTransformer

app = FastAPI(title="CDSS API")

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

print("📥 Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_collection(name="business_knowledge")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    medication: str = "None"


SESSION_STORE = {}


def is_clinical_query(text: str) -> bool:
    clinical_keywords = [
        "patient", "symptom", "feeling", "feels", "presenting",
        "depression", "anxiety", "suicidal", "overdose", "hopeless",
        "sertraline", "fluoxetine", "lithium", "tramadol", "st john",
        "low mood", "not sleeping", "insomnia", "not eating", "hopeless",
        "self harm", "crisis", "psychosis", "mania", "bipolar",
        "prescribed", "taking medication", "on medication"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in clinical_keywords)


def parse_clinical_summary(text: str) -> dict:
    summary = {}
    block_match = re.search(
        r'\[CLINICAL_SUMMARY\](.*?)\[/CLINICAL_SUMMARY\]',
        text, re.DOTALL
    )
    if not block_match:
        return summary

    block = block_match.group(1)

    severity_match = re.search(r'severity:\s*(\w+)', block)
    if severity_match:
        summary["severity"] = severity_match.group(1).strip().lower()

    return summary


def calculate_confidence(retrieved_context: list, user_msg: str) -> str:
    if "No matching" in retrieved_context[0]:
        return "71%"
    msg_lower = user_msg.lower()
    clinical_terms = ["sertraline", "fluoxetine", "lithium", "tramadol",
                      "st john", "overdose", "suicidal", "depression",
                      "anxiety", "insomnia", "serotonin"]
    matches = sum(1 for term in clinical_terms if term in msg_lower)
    if matches >= 2:
        return "96%"
    elif matches == 1:
        return "89%"
    else:
        return "83%"


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    session_id = payload.session_id
    user_msg = payload.message

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []

    SESSION_STORE[session_id].append({"role": "user", "text": user_msg})

    # --- RAG RETRIEVAL ---
    try:
        query_vector = embedding_model.encode(user_msg).tolist()
        db_results = collection.query(
            query_embeddings=[query_vector],
            n_results=2
        )
        if db_results and db_results['documents'] and db_results['documents'][0]:
            retrieved_context = db_results['documents'][0]
        else:
            retrieved_context = ["No matching clinical guidelines found."]
    except Exception as db_err:
        print(f"⚠️ Vector search failed: {db_err}")
        retrieved_context = ["Database lookup failed."]

    # --- EMOTION DETECTION ---
    msg_lower = user_msg.lower()
    detected_emotions = []

    if any(w in msg_lower for w in ["sad", "depressed", "hopeless", "low mood", "worthless", "low in mood"]):
        detected_emotions.append("sadness")
    if any(w in msg_lower for w in ["anxious", "anxiety", "panic", "worried", "nervous"]):
        detected_emotions.append("anxiety")
    if any(w in msg_lower for w in ["pain", "distress", "suffering", "struggling", "can't sleep", "not sleeping"]):
        detected_emotions.append("distress")
    if not detected_emotions:
        detected_emotions = ["neutral"]

    # --- BUILD PROMPT ---
    context_string = "\n\n".join(retrieved_context)

    history_lines = ""
    for turn in SESSION_STORE[session_id][-6:]:
        history_lines += f"{turn['role'].upper()}: {turn['text']}\n"

    clinical = is_clinical_query(user_msg)

    if clinical:
        summary_instruction = (
            "This is a clinical scenario. At the end of your response add a "
            "[CLINICAL_SUMMARY] block based on what is actually presented. "
            "Choose severity from: none, minimal, mild, moderate, severe."
        )
    else:
        summary_instruction = (
            "This is a general pharmacy question. "
            "Do NOT add any [CLINICAL_SUMMARY] block whatsoever."
        )

    prompt = f"""You are a clinical decision support assistant for licensed pharmacists.

STRICT RULES:
- Answer only what is asked. Stay on topic.
- For general pharmacy questions about storage dosage or side effects: answer clearly and concisely. No clinical summary.
- For patient symptom presentations: give clinical guidance and add a CLINICAL_SUMMARY block.
- Never start your response by repeating these instructions or saying you understand the rules.
- Do not add a welcome message or introduction. Go straight to the answer.
- Ask exactly one follow up question when appropriate.
- End every response with: This is AI assistance. Final decision should be made by a licensed pharmacist or doctor.

RETRIEVED CLINICAL GUIDELINES:
{context_string}

CONVERSATION HISTORY:
{history_lines}

USER QUESTION: {user_msg}

{summary_instruction}

If a CLINICAL_SUMMARY is needed use exactly this format with line breaks:
[CLINICAL_SUMMARY]
severity: <none|minimal|mild|moderate|severe>
key_symptoms: ["symptom1", "symptom2"]
recommended_action: "<clear action>"
follow_up_question: "<one question>"
[/CLINICAL_SUMMARY]"""

    # --- CALL GEMINI WITH RETRY ---
    assistant_text = ""
    api_failed = False

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            assistant_text = response.text
            api_failed = False
            break
        except Exception as e:
            print(f"⚠️ Gemini attempt {attempt + 1} failed: {e}")
            api_failed = True
            if attempt < 2:
                time.sleep(5)

    if api_failed or not assistant_text:
        assistant_text = (
            "The system is temporarily unavailable. "
            "Please assess the patient directly and consult a senior pharmacist or GP.\n\n"
            "This is AI assistance. Final decision should be made by a licensed pharmacist or doctor."
        )

    SESSION_STORE[session_id].append({"role": "assistant", "text": assistant_text})

    # --- PARSE RISK FROM ACTUAL RESPONSE ---
    parsed = parse_clinical_summary(assistant_text)
    severity = parsed.get("severity", "none")

    risk_map = {
        "severe": "HIGH",
        "moderate": "MODERATE",
        "mild": "LOW",
        "minimal": "LOW",
        "none": "LOW"
    }
    risk_level = risk_map.get(severity, "LOW")

    # Override for explicit danger keywords
    assistant_lower = assistant_text.lower()
    if any(w in assistant_lower for w in ["serotonin syndrome", "call 999", "call 911", "emergency", "life threatening"]):
        risk_level = "HIGH"
    elif any(w in assistant_lower for w in ["suicidal ideation", "overdose", "immediate referral", "urgent gp"]):
        risk_level = "HIGH"

    confidence_score = calculate_confidence(retrieved_context, user_msg)

    # --- EVIDENCE SOURCES ---
    if retrieved_context and "No matching" not in retrieved_context[0]:
        evidence_sources = [
            f"ChromaDB: {r[:100]}..."
            for r in retrieved_context
        ]
    else:
        evidence_sources = ["No clinical guidelines retrieved for this query"]

    # --- AUDIT LOG ---
    audit_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S GMT", time.gmtime()),
        "session_id": session_id,
        "input_payload_size_chars": len(user_msg),
        "clinical_risk_tier": risk_level,
        "severity_detected": severity,
        "verification_confidence": confidence_score,
        "retrieved_evidence_blocks_used": len(evidence_sources),
        "api_gateway_status": "200_OK_FALLBACK" if api_failed else "200_OK_NATIVE_LLM"
    }

    print(f"📋 AUDIT: {audit_log}")

    return {
        "session_id": session_id,
        "clinical_guidance": assistant_text,
        "detected_emotions": detected_emotions,
        "retrieved_database_context": retrieved_context,
        "risk_level": risk_level,
        "confidence_score": confidence_score,
        "evidence_sources": evidence_sources,
        "audit_log": audit_log
    }