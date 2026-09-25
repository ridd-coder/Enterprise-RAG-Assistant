# Architecture — Enterprise RAG Knowledge Assistant

## Overview

This document describes the system architecture of the Enterprise RAG Knowledge Assistant.

---

## Ingestion Architecture

```mermaid
flowchart LR
    A[PDF Upload] --> B[PyMuPDF Extractor]
    B --> C{Valid PDF?}
    C -->|No| D[CorruptedDocumentError]
    C -->|Yes| E[Page Content Array]
    E --> F[RecursiveCharacterTextSplitter]
    F --> G[DocumentChunk array\nwith metadata]
    G --> H[OpenAI Embeddings API\nbatch mode]
    H --> I[Qdrant Upsert\nupsert idempotent]
```

### Key design decisions:
- **SHA-256 document ID**: prevents duplicate ingestion
- **Page-level chunking**: each chunk knows its exact page
- **Chunk ID**: `{doc_id}_p{page}_c{chunk_idx}` → deterministic Qdrant point UUID (v5)
- **Batch embeddings**: reduces API calls and latency

---

## Embedding Architecture

The `EmbeddingService` is the sole place that calls OpenAI for embeddings.

- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Batch size**: 32 (configurable via `EMBEDDING_BATCH_SIZE`)
- **Retry**: tenacity with exponential backoff, up to 5 attempts
- **Swap**: change `OPENAI_EMBEDDING_MODEL` without touching any other code

---

## Vector Search

```mermaid
flowchart TD
    A[User Question] --> B[EmbeddingService.embed_query]
    B --> C[Qdrant cosine similarity search]
    C --> D{score >= MIN_RELEVANCE_SCORE?}
    D -->|No| E[Filter out]
    D -->|Yes| F[Top-K chunks]
    F --> G[RetrievalContext]
```

- **Distance metric**: Cosine similarity
- **Default top-k**: 5 (configurable via `TOP_K`)
- **Over-fetch**: retrieves `top_k * 2`, then filters by threshold
- **Filtering**: optional `document_ids` filter for scoped retrieval

---

## RAG Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant API as FastAPI
    participant R as Retriever
    participant V as Qdrant
    participant L as LLM Generator
    participant O as OpenAI

    U->>API: POST /api/chat {question}
    API->>R: retrieve(question, top_k)
    R->>O: embed_query(question)
    O-->>R: query_vector
    R->>V: search(query_vector, top_k*2)
    V-->>R: raw_results
    R-->>API: RetrievalContext (filtered)

    alt No context (score < threshold)
        API-->>U: "I could not find sufficient information..."
    else Context available
        API->>L: generate(question, chunks, history)
        L->>O: chat.completions.create(messages)
        O-->>L: completion
        L-->>API: GenerationResult
        API-->>U: ChatResponse {answer, sources, latency}
    end
```

---

## LLM Generation

The system prompt enforces three critical rules:

1. **Context-only**: "Answer ONLY using information from the provided context."
2. **Explicit refusal**: "If the context does not contain enough information... say so exactly."
3. **Source citation**: "Always cite your sources using [Source: filename, Page N]"

---

## Source Citation

Each chunk carries `page_number` from the original PDF. After generation:
- Unique `(document_id, page_number)` pairs are deduplicated
- `SourceReference` objects are returned in the API response
- The frontend displays source cards with filename and page number

---

## Evaluation

```mermaid
flowchart LR
    A[evaluation_dataset.json] --> B[RAGEvaluator]
    B --> C[RAGPipeline.query for each Q]
    C --> D[Compute metrics]
    D --> E[EvaluationReport]
    E --> F[Console output + JSON file]
```

Metrics computed:
- **Retrieval Hit Rate**: at least one expected source retrieved
- **Answer Correctness**: keyword overlap proxy
- **Citation Accuracy**: correct source cited
- **Avg Latency**: measured wall-clock time

---

## Deployment Architecture

```
Internet
    │
    ▼
nginx (port 80/443)
    │
    ├── /api/* → FastAPI (port 8000)
    │               │
    │               ├── Qdrant (port 6333)
    │               └── PostgreSQL (port 5432)
    │
    └── /* → React SPA (static files)
```

Docker Compose services:
- `api`: FastAPI with Uvicorn
- `qdrant`: Qdrant vector database with persistent volume
- `postgres`: PostgreSQL for document metadata
- `frontend`: Nginx serving built React app
