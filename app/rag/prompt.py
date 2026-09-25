"""
app/rag/prompt.py

Centralised prompt templates for the RAG system.

Keeping prompts separate from generation logic allows:
- Version control of prompt changes
- A/B testing
- Easy swap of prompting strategy
"""

from string import Template
from typing import List

from app.rag.vector_store import SearchResult

# ======================================================================
# System prompt — enforces grounding and source citation
# ======================================================================

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Your sole purpose is to answer questions \
based on the documents provided to you in the context below.

STRICT RULES you must follow:
1. Answer ONLY using information from the provided context. Do not use prior knowledge.
2. If the context does not contain enough information to answer the question, respond exactly with:
   "I could not find sufficient information in the uploaded documents to answer this question."
3. Never fabricate facts, statistics, dates, names, or policies.
4. Always cite your sources using the format: [Source: <filename>, Page <page_number>]
5. If multiple documents support the answer, cite all relevant sources.
6. Be concise and professional. This is an enterprise tool.
7. If the question is ambiguous, ask for clarification rather than guessing.
"""

# ======================================================================
# Context builder
# ======================================================================

def build_context_block(chunks: List[SearchResult]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Example output:
        [1] Source: leave_policy.pdf, Page 4
        Employees are entitled to 20 days of annual leave...

        [2] Source: employee_handbook.pdf, Page 12
        Leave requests must be submitted 2 weeks in advance...
    """
    if not chunks:
        return "No relevant context was retrieved from the document store."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.filename}, Page {chunk.page_number}\n{chunk.text}"
        )

    return "\n\n".join(parts)


def build_user_message(
    question: str,
    context: str,
    conversation_history: List[dict] | None = None,
) -> str:
    """
    Build the full user message including context and the question.
    """
    return (
        f"Context from uploaded documents:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Question: {question}\n\n"
        f"Please answer based only on the context above. "
        f"Cite sources using [Source: filename, Page N] format."
    )


def build_messages(
    question: str,
    chunks: List[SearchResult],
    conversation_history: List[dict] | None = None,
) -> List[dict]:
    """
    Assemble the complete messages list for the OpenAI chat API.

    Includes:
    - System prompt
    - (Optional) trimmed conversation history
    - Current user question with context
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject conversation history (last N turns to avoid token overflow)
    if conversation_history:
        # Keep only the last 6 messages (3 turns) for context management
        recent = conversation_history[-6:]
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

    context = build_context_block(chunks)
    user_message = build_user_message(question, context)
    messages.append({"role": "user", "content": user_message})

    return messages
