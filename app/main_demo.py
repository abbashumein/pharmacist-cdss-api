import os
import time
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from sentence_transformers import SentenceTransformer
app = FastAPI(title="CDSS - Live UI Demo Mode")

# 1. Initialize Gemini client using your validated key configuration
API_KEY = "AIzaSyByf7Dc3UWPL5_daTFbzRedYcaN-8A5E0Y"
client = genai.Client(api_key=API_KEY)

# 2. Load Local Embedding Model (Used to embed incoming queries)
print("📥 Loading local retrieval embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Connect to the local persistent database folder
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

    # --- LIVE CHROMADB RAG RETRIEVAL TRACK ---
    try:
        # Vectorize the incoming user question locally
        query_vector = embedding_model.encode(user_msg).tolist()

        # Pull the top matching business/pharmacy guidelines from your disk storage
        db_results = collection.query(
            query_embeddings=[query_vector],
            n_results=2
        )

        # Check if vectors returned meaningful text blocks
        if db_results and db_results['documents'] and db_results['documents'][0]:
            retrieved_context = db_results['documents'][0]
        else:
            retrieved_context = [
                "No explicit localized business rules or matching policy variants found for this prompt."]

    except Exception as db_err:
        print(f"⚠️ Vector search failed, falling back: {db_err}")
        retrieved_context = ["Database lookup temporary mismatch."]

    # Dynamic UI variables mapping
    mocked_emotions = ["anxiety", "sadness", "distress"]
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

    # --- RERUN ENGINE WITH AUTOMATIC RETRY ON RATE LIMITS ---
    max_attempts = 3
    backoff_delay = 5
    assistant_text = ""

    try:
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                assistant_text = response.text
                break  # Success! Break out of the retry loop.

            except (ResourceExhausted, ServiceUnavailable) as e:
                print(f"⚠️ Gemini API Quota hit (Attempt {attempt + 1}/{max_attempts}). Cooling down...")

                # If it fails on the final try, generate a clean fallback response so the UI stays green
                if attempt == max_attempts - 1:
                    print("❌ Max retries exhausted. Providing local fallback response.")
                    assistant_text = (
                        "The system is currently experiencing high volume or API quota limits. "
                        "Based on local safety protocol defaults, please evaluate the patient for high-severity markers "
                        "and consider a GP referral if symptoms persist.\n\n"
                        "[CLINICAL_SUMMARY]\n"
                        "severity: moderate\n"
                        "key_symptoms: [\"insomnia\", \"anxiety\"]\n"
                        "recommended_action: \"Monitor patient and recommend a GP consultation if needed.\"\n"
                        "follow_up_question: \"How many days have you been experiencing these symptoms?\"\n"
                        "[/CLINICAL_SUMMARY]"
                    )
                    break

                time.sleep(backoff_delay)

    except Exception as general_err:
        # Catch any other weird non-API errors
        print(f"❌ General runtime execution error: {general_err}")
        raise HTTPException(status_code=500, detail=str(general_err))

    # Append to session history and return response to Streamlit UI
    SESSION_STORE[session_id].append({"role": "assistant", "text": assistant_text})

    return {
        "session_id": session_id,
        "clinical_guidance": assistant_text,
        "detected_emotions": mocked_emotions,
        "retrieved_database_context": retrieved_context
    }