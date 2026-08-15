import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("langchain")

queries = [
    "warfarin drug interactions",
    "ibuprofen side effects",
    "metformin contraindications",
    "aspirin drug interactions",
]

for query in queries:
    print("\n" + "=" * 70)
    print("QUERY:", query)

    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]

    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    print("\nRERANKED RESULTS:")

    for i, (score, doc) in enumerate(ranked, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(doc[:300])