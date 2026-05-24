from google import genai

class GeminiService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    def generate_analysis(self, patient_text: str, emotions: list, medication: str, warning_context: str) -> str:
        prompt = f"You are an expert clinical pharmacy advisor. Patient text: '{patient_text}' Emotions: {emotions} Medication: {medication} Warning profile: {warning_context}. Provide a concise, 2-bullet point dashboard note checking clinical risk intersection and conversational actions."
        response = self.client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text