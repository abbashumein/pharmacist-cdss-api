from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.ml_service import MLService
from app.services.rag_service import RagService
from app.services.gemini_service import GeminiService
import os

app = FastAPI(title="Clinical Support API")
EMOTION_LABELS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "./weights/mental_health_distilbert.pt")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

ml_service = MLService(weights_path=WEIGHTS_PATH, labels=EMOTION_LABELS)
rag_service = RagService()
gemini_service = GeminiService(api_key=GEMINI_KEY)

class PatientRequest(BaseModel):
    text: str
    medication: str

@app.post("/predict")
async def process_clinical_check(data: PatientRequest):
    try:
        detected_emotions = ml_service.predict(data.text)
        drug_context = rag_service.search_warning(data.medication)
        clinical_guidance = gemini_service.generate_analysis(patient_text=data.text, emotions=detected_emotions, medication=data.medication, warning_context=drug_context)
        return {"status": "success", "detected_emotions": detected_emotions, "retrieved_database_context": drug_context, "clinical_guidance": clinical_guidance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))