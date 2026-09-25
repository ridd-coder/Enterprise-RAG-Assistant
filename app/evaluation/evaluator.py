"""
app/evaluation/evaluator.py

RAG system evaluation framework.

Evaluates:
1. Retrieval Hit Rate — did we retrieve the expected source?
2. Answer Correctness — does the answer contain the expected keywords?
3. Citation Accuracy — did we cite the correct source document?
4. Hallucination proxy — did we refuse to answer when context was missing?
5. Latency — average response time
"""

import json
import time
from pathlib import Path

from app.core.logging import get_logger
from app.evaluation.metrics import (
    compute_answer_correctness,
    compute_citation_accuracy,
    compute_retrieval_hit,
)
from app.models.schemas import (
    EvaluationQuestion,
    EvaluationReport,
    EvaluationResult,
)
from app.rag.pipeline import RAGPipeline

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Runs the evaluation dataset against the RAG pipeline and computes metrics.
    """

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def load_dataset(self, dataset_path: Path) -> list[EvaluationQuestion]:
        """Load evaluation questions from a JSON file."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path, encoding="utf-8") as f:
            raw = json.load(f)

        questions = [EvaluationQuestion(**item) for item in raw]
        logger.info("eval_dataset_loaded", count=len(questions))
        return questions

    def run(
        self,
        dataset_path: Path,
        output_path: Path | None = None,
    ) -> EvaluationReport:
        """
        Execute a full evaluation run.

        Args:
            dataset_path: Path to evaluation_dataset.json.
            output_path: If set, write the report JSON to this path.

        Returns:
            EvaluationReport with all metrics.
        """
        questions = self.load_dataset(dataset_path)
        results: list[EvaluationResult] = []
        total_latency = 0

        for i, q in enumerate(questions):
            logger.info(
                "evaluating_question",
                index=i + 1,
                total=len(questions),
                question=q.question[:60],
            )

            try:
                start = time.monotonic()
                response = self._pipeline.query(question=q.question)
                latency_ms = int((time.monotonic() - start) * 1000)
                total_latency += latency_ms

                retrieved_sources = list({s.filename for s in response.sources})

                retrieval_hit = compute_retrieval_hit(
                    retrieved_sources=retrieved_sources,
                    expected_sources=q.expected_sources,
                )
                answer_correct = compute_answer_correctness(
                    actual_answer=response.answer,
                    expected_answer=q.expected_answer,
                )
                citation_correct = compute_citation_accuracy(
                    retrieved_sources=retrieved_sources,
                    expected_sources=q.expected_sources,
                )

                results.append(
                    EvaluationResult(
                        question=q.question,
                        expected_answer=q.expected_answer,
                        actual_answer=response.answer,
                        retrieval_hit=retrieval_hit,
                        answer_correct=answer_correct,
                        citation_correct=citation_correct,
                        latency_ms=latency_ms,
                        retrieved_sources=retrieved_sources,
                        expected_sources=q.expected_sources,
                    )
                )

            except Exception as exc:
                logger.error(
                    "eval_question_failed",
                    question=q.question[:60],
                    error=str(exc),
                )
                results.append(
                    EvaluationResult(
                        question=q.question,
                        expected_answer=q.expected_answer,
                        actual_answer="ERROR",
                        retrieval_hit=False,
                        answer_correct=False,
                        citation_correct=False,
                        latency_ms=0,
                        retrieved_sources=[],
                        expected_sources=q.expected_sources,
                    )
                )

        # Aggregate metrics
        n = len(results)
        retrieval_hit_rate = sum(1 for r in results if r.retrieval_hit) / n if n else 0
        answer_correctness = sum(1 for r in results if r.answer_correct) / n if n else 0
        citation_accuracy = sum(1 for r in results if r.citation_correct) / n if n else 0
        avg_latency = total_latency / n if n else 0

        report = EvaluationReport(
            total_questions=n,
            retrieval_hit_rate=round(retrieval_hit_rate, 4),
            answer_correctness=round(answer_correctness, 4),
            citation_accuracy=round(citation_accuracy, 4),
            avg_latency_ms=round(avg_latency, 2),
            results=results,
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))
            logger.info("eval_report_saved", path=str(output_path))

        return report
