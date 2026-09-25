"""
app/evaluation/metrics.py

Individual evaluation metric functions.

All metrics are deterministic — no LLM calls in metrics computation.
(LLM-as-judge can be added as an optional advanced feature.)
"""

from typing import List


def compute_retrieval_hit(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> bool:
    """
    Retrieval Hit: Did we retrieve at least one expected source document?

    Uses case-insensitive filename matching.
    """
    retrieved_lower = {s.lower() for s in retrieved_sources}
    return any(e.lower() in retrieved_lower for e in expected_sources)


def compute_answer_correctness(
    actual_answer: str,
    expected_answer: str,
    threshold: float = 0.5,
) -> bool:
    """
    Answer Correctness: Keyword-overlap proxy.

    Checks what fraction of meaningful words in the expected answer
    appear in the actual answer.

    Note: A more robust implementation would use semantic similarity
    (e.g., cosine similarity between embeddings). This is a fast,
    interpretable baseline.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "of",
        "to", "for", "and", "or", "that", "it", "this", "with",
        "be", "as", "at", "by", "from", "on", "not",
    }

    expected_words = {
        w.lower().strip(".,;:!?\"'")
        for w in expected_answer.split()
        if w.lower() not in stopwords and len(w) > 2
    }

    if not expected_words:
        return True  # Nothing meaningful to compare

    actual_lower = actual_answer.lower()
    matched = sum(1 for w in expected_words if w in actual_lower)
    overlap = matched / len(expected_words)

    return overlap >= threshold


def compute_citation_accuracy(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> bool:
    """
    Citation Accuracy: Did we cite at least one expected source?

    Same as retrieval hit but from the citation perspective.
    """
    return compute_retrieval_hit(retrieved_sources, expected_sources)


def compute_precision_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> float:
    """
    Precision@K: Fraction of retrieved sources that are relevant.
    """
    if not retrieved_sources:
        return 0.0

    expected_lower = {e.lower() for e in expected_sources}
    relevant = sum(1 for s in retrieved_sources if s.lower() in expected_lower)
    return relevant / len(retrieved_sources)


def compute_recall_at_k(
    retrieved_sources: List[str],
    expected_sources: List[str],
) -> float:
    """
    Recall@K: Fraction of expected sources that were retrieved.
    """
    if not expected_sources:
        return 1.0

    retrieved_lower = {s.lower() for s in retrieved_sources}
    found = sum(1 for e in expected_sources if e.lower() in retrieved_lower)
    return found / len(expected_sources)
