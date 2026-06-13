import json
import zipfile
import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Configuration & Local Paths
# Make sure the filename below matches the exact name of the zip file you downloaded!
ZIP_FILE_PATH = "drug-label-0001-of-0013.json.zip"

print("📥 Initializing local vector models and ChromaDB clients...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = chroma_client.get_collection(name="langchain")
except Exception:
    collection = chroma_client.create_collection(name="langchain")

if not os.path.exists(ZIP_FILE_PATH):
    raise FileNotFoundError(f"❌ Could not find '{ZIP_FILE_PATH}' in your current directory. "
                            f"Please move the downloaded zip file into: {os.getcwd()}")

# 2. Extract and Parse Locally
print(f"📦 Opening local archive: {ZIP_FILE_PATH}...")
with zipfile.ZipFile(ZIP_FILE_PATH) as z:
    json_filename = z.namelist()[0]
    print(f"⚡ Unpacking and reading internal data structure: {json_filename}...")
    with z.open(json_filename) as f:
        fda_data = json.load(f)

records = fda_data.get("results", [])
# We limit to the first 1,000 records for your portfolio demo.
# This gives you an amazing, deep dataset without blowing up execution time!
records_to_process = records[:1000]
print(
    f"🔍 Successfully parsed {len(records)} entries. Processing a targeted slice of {len(records_to_process)} records...")

# 3. Batch Vectorization & Ingestion
documents = []
embeddings = []
ids = []
batch_size = 50  # Smaller batch size to prevent CPU choking

print(f"🚀 Vectorizing clinical rules in streaming batches of {batch_size}...")

for idx, drug in enumerate(records_to_process):
    openfda_metadata = drug.get("openfda", {})
    generic_names = openfda_metadata.get("generic_name", [])
    brand_names = openfda_metadata.get("brand_name", [])

    if not generic_names:
        continue

    generic_name = generic_names[0]
    brand_name = brand_names[0] if brand_names else "Generic Version Available"

    interactions = drug.get("drug_interactions", ["No specific interaction records provided on standard label."])[0]
    contraindications = drug.get("contraindications", ["No specific acute contraindications listed."])[0]

    # We truncate incredibly long clinical text blocks to keep the embeddings tight and accurate
    text_chunk = (
        f"Drug Name: {generic_name} ({brand_name}) | "
        f"FDA Interactions Field: {interactions[:400]}... | "
        f"Clinical Contraindications: {contraindications[:400]}..."
    )

    # Generate vectors locally for free
    vector = embedding_model.encode(text_chunk).tolist()

    documents.append(text_chunk)
    embeddings.append(vector)
    ids.append(f"fda_label_chunk_{idx}")

    # Commit to ChromaDB in chunks
    if len(documents) == batch_size or idx == len(records_to_process) - 1:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
        print(f"✅ Indexed through database record entry: {idx + 1}/{len(records_to_process)}")
        documents, embeddings, ids = [], [], []

print("\n🎉 Success! Database upgraded with hundreds of real-world FDA interaction rules.")