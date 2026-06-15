import json
import zipfile
import os
import chromadb

# 1. Configuration & Local Paths
ZIP_FILE_PATH = "drug-label-0001-of-0013.json.zip"

print("📥 Initializing ChromaDB client...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection(name="langchain")
    print("🧹 Deleted old collection")
except Exception:
    pass

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
records_to_process = records[:1000]
print(f"🔍 Successfully parsed {len(records)} entries. Processing {len(records_to_process)} records...")

# 3. Ingestion without manual embeddings — ChromaDB handles it
documents = []
ids = []
batch_size = 50

print(f"🚀 Ingesting clinical rules in batches of {batch_size}...")

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

    text_chunk = (
        f"Drug Name: {generic_name} ({brand_name}) | "
        f"FDA Interactions Field: {interactions[:400]}... | "
        f"Clinical Contraindications: {contraindications[:400]}..."
    )

    documents.append(text_chunk)
    ids.append(f"fda_label_chunk_{idx}")

    if len(documents) == batch_size or idx == len(records_to_process) - 1:
        collection.add(
            documents=documents,
            ids=ids
        )
        print(f"✅ Indexed through database record entry: {idx + 1}/{len(records_to_process)}")
        documents, ids = [], []

print(f"\n🎉 Success! Total records: {collection.count()}")