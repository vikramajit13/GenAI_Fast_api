RAG (Retrieval-Augmented Generation) pipeline using FastAPI and Ollama.


![alt text](image.png)



 System Architecture
1. Data Ingestion Pipeline (/ingestion)
The ingestion layer transforms raw text into structured, searchable vector and lexical data.
Smart Chunking:
Strategy: Splits text based on logical boundaries like line breaks and paragraphs to maintain structural integrity.
Context Retention: Enforces a minimum target of 800 characters per chunk. If a paragraph is too small, it is merged with the next to ensure the LLM has enough local context.
Sliding Window: Implements sentence overlap between chunks to prevent information loss at the boundaries.
Vectorization:
Model: Uses the all-MiniLM-L6-v2 embedding model (384 dimensions).
Storage: Each record stores the document_id, the full chunk_text, and the generated embedding vector in a PostgreSQL database.
2. Query Orchestration (/answer)
When a user submits a query, the Planner/Orchestrator manages the retrieval strategy.
Query Analysis:
Direct Retrieval: Used for clear, well-formed queries.
Query Rewriting: If the initial query is ambiguous, an LLM rewrites it to expand synonyms or clarify intent before searching.
Hybrid Retrieval Layer:
Semantic Search: Utilizes pgvector for cosine similarity on embeddings.
Lexical Search: Utilizes Postgres Full-Text Search (FTS) for keyword matching.
RRF Fusion: Combines scores from both methods using Reciprocal Rank Fusion (RRF) to produce a single, unified ranked list of top-K candidates.
3. Post-Retrieval Refinement
Once the top-K chunks are retrieved, they undergo deep analysis to filter noise.
Cross-Encoder Reranking:
The system passes the query and retrieved chunks through a Cross-Encoder.
Unlike bi-encoders (which compare vectors), the cross-encoder processes the query and chunk together, capturing deep semantic interactions to re-sort results by true relevance.
Sentence Evidence Selection:
To avoid "stuffing" the LLM with 800-character chunks, the system selects only the most relevant sentences from each chunk.
Selection is based on cosine similarity between the query embedding and individual sentence embeddings, followed by text normalization.
This ensures the LLM receives only the high-signal "evidence" required to answer. 
4. Safety & Generation
Guardrails:
Input Validation: Rejects empty or nonsensical queries.
Score Thresholding: Discards sentences or chunks that fall below a minimum relevance score.
Length Control: Truncates excessively long sentences to prevent context window overflow.
LLM Invocation:
Provider: Local inference via Ollama.
Prompting: A structured template combining a System Prompt (instructions on how to answer) and the User Prompt (populated with the extracted sentence evidence). 
🛠️ Technical Stack
API Framework: FastAPI
Database: PostgreSQL with pgvector extension
Embeddings: all-MiniLM-L6-v2 (via Sentence-Transformers)
LLM Runtime: Ollama (e.g., Llama 3 or Mistral)
Language: Python 3.10