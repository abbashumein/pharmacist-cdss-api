import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize Local Embedding Model (Downloads once, runs locally FREE forever)
print("📥 Loading local embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize Persistent ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Clear out the old collection so the 4 old vectors don't mess up your new data
try:
    chroma_client.delete_collection(name="business_knowledge")
    print("🧹 Cleared old database collection to start fresh...")
except Exception:
    pass

# Create or get collection configured for local embeddings
collection = chroma_client.get_or_create_collection(
    name="business_knowledge",
    metadata={"hnsw:space": "cosine"}
)


def get_local_embedding(text: str) -> list:
    """Generates text embeddings locally on your CPU/GPU instantly."""
    try:
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"❌ Local embedding generation failed: {e}")
        return []


def run_ingestion():
    # UPDATED PATH: Points directly to your 160 clinical jsonl rows
    jsonl_path = "data/clinical_records.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"❌ Error: Could not find data file at '{jsonl_path}'! Run your generation script first.")
        return

    print(f"🚀 Starting local JSONL text ingestion pipeline on '{jsonl_path}'...")

    doc_id_counter = 0

    # Open and process the JSONL line by line
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if line.strip():
                # Parse the raw line string into a Python dictionary
                record = json.loads(line)
                chunk = record["text"]
                metadata_info = record.get("metadata", {})

                # Add extra tracking keys to your existing metadata format
                metadata_info["source"] = "clinical_records.jsonl"
                metadata_info["chunk_index"] = index

                # Generate your vectors locally
                vector = get_local_embedding(chunk)
                if not vector:
                    continue

                doc_id = f"doc_clinical_{doc_id_counter}"
                doc_id_counter += 1

                # Save straight to local SSD using your original structure
                collection.add(
                    ids=[doc_id],
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[metadata_info]
                )

    print(f"✅ Ingestion complete! Successfully stored {doc_id_counter} vectors locally in './chroma_storage'.")


if __name__ == "__main__":
    run_ingestion()