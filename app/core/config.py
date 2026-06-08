from pydantic_settings import BaseSettings

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    MODEL_PATH: str = "weights/sentiment_model.pt"
    CHROMA_DB_PATH: str = "chroma_storage"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # FIX: Explicitly allow the model_name field coming from your local .env
    model_name: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"
        # FIX: Change from "forbid" to "ignore" so extra environment variables don't crash the pipeline
        extra = "ignore"


settings = Settings()