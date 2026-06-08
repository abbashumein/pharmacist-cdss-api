import json
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def run_etl_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.join(base_dir, "data", "phq9_data.jsonl")
    target_db_dir = os.path.join(base_dir, "chroma_db")
    
    if os.path.exists(target_db_dir):
        shutil.rmtree(target_db_dir)
        
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    processed_documents = []
    
    with open(source_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            rich_text = f"Source: {record['source']}\nText: {record['core_text']}\nAction: {record['action_rule']}"
            processed_documents.append(Document(page_content=rich_text, metadata={"id": record["diagnostic_id"]}))
            
    Chroma.from_documents(documents=processed_documents, embedding=embedding_engine, persist_directory=target_db_dir)
    print("✅ Vector index compiled successfully.")

if __name__ == "__main__":
    run_etl_pipeline()
