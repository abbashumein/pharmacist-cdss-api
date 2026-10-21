# Engineering Decisions — Pharmacist CDSS

Three versions live in this repo:
- **V1 (legacy, `app/main.py`)**: emotion model + guideline RAG. Kept for reference.
- **V2 (`app/main_demo.py`)**: LangGraph RAG pipeline over FDA drug labels. This is what the Docker image serves.
- **V3 (`app/main_agentic.py`)**: LangGraph agent where Gemini decides when to call the FDA tool.

## V2 / V3 decisions

### Local all-MiniLM-L6-v2 embeddings instead of a cloud embedding API
The Gemini embedding free tier ran out during ingest (about 1,000 requests/day). A local model has no quota and
no network call. Trade-off: MiniLM reads only about 256 tokens, so long label sections are cut when embedded.

### Per-drug retrieval, then CrossEncoder rerank
One combined query ("warfarin ibuprofen interaction") let one drug's chunks crowd out the other's before
reranking. Each named drug is now queried on its own (top 5), then reranked. A distance threshold (0.95)
removes chunks that are not relevant. The rerank query includes the resolved drug name, otherwise a drug that
was only described ("blood pressure medicine") loses the rerank.

### Refusal is enforced in code, not only in the prompt
If the tool ran and found no usable evidence, the answer is forced to "I don't have sufficient FDA data" and
a HIGH risk is lowered. A prompt rule alone can be ignored by the model.

### Gemini function calling for routing (V3), keyword fallback if the call fails
The model decides whether `fda_lookup` is needed. If the routing call itself errors, a keyword check is used so
clinical questions are not silently answered without evidence. Cost: 2 Gemini calls per clinical query.

### Fixed list of 12 drugs (known limitation)
Per-drug retrieval and the routing tool use a fixed `TARGET_DRUGS` list. This was a shortcut. The DB has 834
labels, so this limits coverage. A better design reads drug names from the DB metadata.

### Session memory: in-memory `MemorySaver`, per-turn fields reset
History is kept per `session_id` in memory only (lost on restart). Because the saver keeps the whole state,
every per-turn field is reset on each request (`build_turn_state`). Without that, turn 2 reused turn 1's
evidence. Covered by `tests/test_agentic_session.py`.

### Sync `def` endpoints
`/chat` does blocking work (Gemini, CrossEncoder, ChromaDB). As `async def` it would freeze the event loop for
all users; as plain `def`, FastAPI runs it in a thread pool.

### V2 builds the DB only when it is empty
Startup used to delete and rebuild the collection every time. Now a populated DB is reused; `POST /admin/ingest`
forces a rebuild. After a rebuild, restart V3 so it picks up the new collection.

### ChromaDB (embedded) instead of a hosted vector DB
About 834 label chunks fit easily in an embedded store: no extra service, no cost, persisted to disk. For
millions of chunks this would change (ANN index tuning, filtering, a hosted or sharded store).

### Gemini 2.5 Flash
Low latency and a free tier. The free tier (about 20 requests/day) limits how large an eval can be, so the eval
sleeps between calls and retries on 429 using the server's suggested delay.
