import time
import logging
from google import genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable  # <-- Added for error handling
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical decision support assistant for licensed pharmacists.
Your role is to surface relevant clinical guidelines and ask structured follow-up questions.
You never diagnose. You always recommend GP referral for moderate or severe presentations.
Ask exactly one follow-up question per response. Summarise before asking.
Format your response with a [CLINICAL_SUMMARY] block at the end."""

CLINICAL_SUMMARY_FORMAT = """[CLINICAL_SUMMARY]
severity: <none|minimal|mild|moderate|severe>
key_symptoms: ["symptom1", "symptom2"]
recommended_action: "<what pharmacist should do>"
follow_up_question: "<single question to ask patient>"
[/CLINICAL_SUMMARY]"""


class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_MODEL
        logger.info("GeminiService ready")

    def generate(self, patient_text: str, emotion_labels: list[str], retrieved_chunks: list[str],
                 chat_history: list[dict]) -> str:
        context = "\n\n---\n\n".join(retrieved_chunks)
        history_text = ""
        for turn in chat_history[-10:]:
            role = turn["role"].upper()
            history_text += f"{role}: {turn['text']}\n"

        prompt = f"""{SYSTEM_PROMPT}

RETRIEVED CLINICAL GUIDELINES:
{context}

EMOTION ANALYSIS RESULTS (from fine-tuned DistilBERT):
Detected indicators: {', '.join(emotion_labels)}

CONVERSATION HISTORY:
{history_text}

PHARMACIST: {patient_text}

Provide your clinical response followed by {CLINICAL_SUMMARY_FORMAT}"""

        # 🔄 Automatic Retry Strategy for Rate Limits (429 / 503)
        max_attempts = 3
        backoff_delay = 5  # Seconds to wait before retrying

        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt)
                return response.text

            except (ResourceExhausted, ServiceUnavailable) as e:
                logger.warning(
                    f"⚠️ Gemini API Quota/Spike encountered (Attempt {attempt + 1}/{max_attempts}). "
                    f"Error: {e.message if hasattr(e, 'message') else str(e)}"
                )

                # If this was our final try, raise the exception up so the application handles it transparently
                if attempt == max_attempts - 1:
                    logger.error("❌ Exceeded max retry attempts for Gemini API content generation.")
                    raise e

                # Wait for the free tier counter window to clear down before looping back
                logger.info(f"⏳ Cooling down for {backoff_delay} seconds before retrying...")
                time.sleep(backoff_delay)