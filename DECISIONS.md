# Engineering Decisions — Clinical Decision Support System (CDSS) API

## Why DistilBERT over a larger model?
DistilBERT is roughly 40% smaller and 60% faster than a baseline BERT-base model while retaining up to 97% of its language understanding performance. For real-time clinical support tools, keeping inference latency to a minimum is a major priority. Fine-tuning a local DistilBERT instance ensures domain-specific emotion detection runs rapidly on consumer hardware without introducing external API network bottlenecks.

## Why ChromaDB over Cloud Vector Providers (Pinecone/Weaviate)?
For a highly specialized domain knowledge base with under 500 clinical guideline records, an embedded database like ChromaDB is the optimal lightweight choice. It runs directly within the application context, requires no external network configuration, eliminates third-party platform costs, and automatically persists data seamlessly to local disk storage. 

## Why a Custom In-Memory Session Store?
To support multi-turn conversational state tracing, we leverage an in-memory dictionary keyed by unique `session_id` tokens. This layout successfully isolates the session management logic within the API definition layer. For production scales, this store can easily be swapped out for a high-availability Redis cache adapter without requiring modifications to the core ML or RAG service abstractions.

## Why Gemini 2.5 Flash over Other Models?
Gemini 2.5 Flash delivers low latency and a strong free tier quota. It excels at adhering strictly to programmatic system prompts, processing multi-turn historical contexts, and cleanly injecting localized context chunks parsed from our RAG semantic pipeline.

## Why Matched-Care Architecture (NICE NG222) over Legacy Stepped Models?
Using outdated guidelines in active clinical support tools poses serious safety risks. This system aligns directly with current NICE clinical standards (NG222), which prioritize matching interventions directly to patient symptom severity profiles and personal preferences right at the first touchpoint, rather than enforcing rigid, incremental escalation steps.
