from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.db = Chroma(
            persist_directory=settings.CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )
        count = self.db._collection.count()
        logger.info(f"RAGService ready — {count} vectors loaded from {settings.CHROMA_DB_PATH}")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        results = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
