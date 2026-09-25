"""tests/unit/test_metrics.py

Unit tests for evaluation metric functions.
"""

from app.evaluation.metrics import (
    compute_answer_correctness,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_retrieval_hit,
)


class TestRetrievalHit:
    def test_hit_exact_match(self):
        assert compute_retrieval_hit(["leave_policy.pdf"], ["leave_policy.pdf"]) is True

    def test_hit_case_insensitive(self):
        assert compute_retrieval_hit(["Leave_Policy.PDF"], ["leave_policy.pdf"]) is True

    def test_miss_no_match(self):
        assert compute_retrieval_hit(["handbook.pdf"], ["leave_policy.pdf"]) is False

    def test_empty_retrieved(self):
        assert compute_retrieval_hit([], ["leave_policy.pdf"]) is False

    def test_partial_match_counts(self):
        retrieved = ["handbook.pdf", "leave_policy.pdf"]
        assert compute_retrieval_hit(retrieved, ["leave_policy.pdf"]) is True


class TestAnswerCorrectness:
    def test_exact_keyword_present(self):
        assert compute_answer_correctness("Employees get 20 days leave.", "20 days") is True

    def test_completely_wrong_answer(self):
        # No keyword overlap at all
        result = compute_answer_correctness("XYZ", "annual leave twenty days entitlement")
        assert result is False

    def test_empty_expected(self):
        # Empty expected → nothing to check → True
        assert compute_answer_correctness("any answer", "") is True

    def test_no_context_answer_fails(self):
        no_context = "I could not find sufficient information"
        assert compute_answer_correctness(no_context, "20 days annual leave") is False


class TestPrecisionRecall:
    def test_precision_full(self):
        assert compute_precision_at_k(["leave_policy.pdf"], ["leave_policy.pdf"]) == 1.0

    def test_precision_half(self):
        result = compute_precision_at_k(
            ["leave_policy.pdf", "unrelated.pdf"],
            ["leave_policy.pdf"],
        )
        assert result == 0.5

    def test_recall_full(self):
        assert compute_recall_at_k(["leave_policy.pdf"], ["leave_policy.pdf"]) == 1.0

    def test_recall_zero(self):
        assert compute_recall_at_k(["unrelated.pdf"], ["leave_policy.pdf"]) == 0.0

    def test_empty_retrieved_precision(self):
        assert compute_precision_at_k([], ["leave_policy.pdf"]) == 0.0

    def test_empty_expected_recall(self):
        assert compute_recall_at_k(["anything.pdf"], []) == 1.0
