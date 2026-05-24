from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RagService:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.pharma_knowledge_base = [
            "Xanax (Alprazolam): High danger of severe central nervous system depression. Never mix with alcohol or opioids. May amplify existing depressive tendencies or cause severe sedation.",
            "Panadol (Paracetamol): Generally safe, but acute overdose causes hepatotoxicity and irreversible liver failure. Maximum adult dose is 4g per day.",
            "Prozac (Fluoxetine): Selective Serotonin Reuptake Inhibitor (SSRI). Can trigger sudden severe anxiety or suicidal ideation during first two weeks of treatment. Monitor patient carefully.",
            "Lipitor (Atorvastatin): Used for high cholesterol. Can rarely cause rhabdomyolysis (severe muscle breakdown). Patient should report unexplained muscle pain immediately."
        ]
        self.vector_db = Chroma.from_texts(texts=self.pharma_knowledge_base, embedding=self.embeddings, persist_directory=persist_directory)
        
    def search_warning(self, medication_name: str) -> str:
        results = self.vector_db.similarity_search(medication_name, k=1)
        return results[0].page_content if results else "No specific warning context found." 