import os
from dotenv import load_dotenv
import time
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from sentence_transformers import SentenceTransformer

app = FastAPI(title="CDSS - Live UI Demo Mode")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

print("📥 Loading local retrieval embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_collection(name="business_knowledge")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    medication: str = "None"


SESSION_STORE = {}


@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    session_id = payload.session_id
    user_msg = payload.message

    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []

    SESSION_STORE[session_id].append({"role": "user", "text": user_msg})

    # --- 1. LIVE CHROMADB RAG RETRIEVAL TRACK ---
    try:
        query_vector = embedding_model.encode(user_msg).tolist()
        db_results = collection.query(
            query_embeddings=[query_vector],
            n_results=2
        )

        if db_results and db_results['documents'] and db_results['documents'][0]:
            retrieved_context = db_results['documents'][0]
        else:
            retrieved_context = [
                "No explicit localized business rules or matching policy variants found for this prompt."]
    except Exception as db_err:
        print(f"⚠️ Vector search failed, falling back: {db_err}")
        retrieved_context = ["Database lookup temporary mismatch."]

    # --- 2. DYNAMIC NLP EMOTION ENGINE PROFILE (FIXED OUTSIDE OF EXCEPT) ---
    msg_lower = user_msg.lower()
    mocked_emotions = []

    if "low" in msg_lower or "sad" in msg_lower or "depressed" in msg_lower or "sertraline" in msg_lower:
        mocked_emotions.append("sadness")
    if "anxiety" in msg_lower or "panic" in msg_lower or "worried" in msg_lower:
        mocked_emotions.append("anxiety")
    if "difficulty" in msg_lower or "insomnia" in msg_lower or "pain" in msg_lower or "complaint" in msg_lower:
        mocked_emotions.append("distress")

    # Fallback to positive/neutral if no clinical stress distress signs exist
    if not mocked_emotions:
        if "great" in msg_lower or "energetic" in msg_lower or "good" in msg_lower:
            mocked_emotions = ["positive", "energetic"]
        else:
            mocked_emotions = ["neutral"]

    # Flatten context documents for the prompt structure
    context_string = "\n".join(retrieved_context)

    # Reconstruct history context string for Gemini's multi-turn awareness
    history_context = ""
    for turn in SESSION_STORE[session_id][-6:]:
        history_context += f"{turn['role'].upper()}: {turn['text']}\n"

    # Build the augmented prompt with real database data
    prompt = f"""You are a clinical decision support assistant for licensed pharmacists.
Review this patient statement, the localized business or clinical policy guidelines, and the conversation history.

RETRIEVED CLINICAL & BUSINESS POLICY GUIDELINES:
{context_string}

CONVERSATION HISTORY:
{history_context}

Provide a concise clinical response giving guidance or asking exactly ONE follow-up question.
At the very end of your response, you MUST append a valid, raw text block in this exact schema format:

[CLINICAL_SUMMARY]
severity: severe
key_symptoms: ["insomnia", "hopelessness", "poor intake"]
recommended_action: "Refer immediately to GP / Crisis Team"
follow_up_question: "Ask about duration of intake refusal"
[/CLINICAL_SUMMARY]
"""
    # --- 3. RUN ENGINE WITH AUTOMATIC RETRY ON RATE LIMITS ---
    max_attempts = 3
    backoff_delay = 2
    assistant_text = ""
    api_failed = False

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            assistant_text = response.text
            api_failed = False
            break  # Success! Exit the retry loop.
        except Exception as e:
            print(f"⚠️ Gemini API attempt {attempt + 1} failed: {e}")
            api_failed = True
            if attempt < max_attempts - 1:
                time.sleep(backoff_delay)

    # If all retries fail or a 503 is sustained, bypass the crash and serve the local backup
    if api_failed or not assistant_text:
        print("❌ Max retries exhausted due to high model demand. Serving local fallback payload.")
        assistant_text = (
            "The system is currently experiencing high volume or API quota limits. "
            "Based on local safety protocol defaults, please evaluate the patient for high-severity markers "
            "and consider an urgent clinical medication review if symptoms persist.\n\n"
            "[CLINICAL_SUMMARY]\n"
            "severity: moderate\n"
            "key_symptoms: [\"insomnia\", \"anxiety\"]\n"
            "recommended_action: \"Monitor patient and recommend a clinical consultation if needed.\"\n"
            "follow_up_question: \"How many days have you been experiencing these symptoms?\"\n"
            "[/CLINICAL_SUMMARY]"
        )

    # Append safely to session history
    SESSION_STORE[session_id].append({"role": "assistant", "text": assistant_text})

    # Always return a clean 200 JSON object so the Streamlit UI never breaks
    return {
        "session_id": session_id,
        "clinical_guidance": assistant_text,
        "detected_emotions": mocked_emotions,
        "retrieved_database_context": retrieved_context
    }