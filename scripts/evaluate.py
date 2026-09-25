#!/usr/bin/env python3
"""
scripts/evaluate.py

CLI script to run the RAG evaluation suite and print a report.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --dataset app/evaluation/evaluation_dataset.json
    python scripts/evaluate.py --output evaluation_results.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import configure_logging
from app.evaluation.evaluator import RAGEvaluator
from app.rag.pipeline import RAGPipeline

configure_logging()


def print_report(report) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 52)
    print("  Enterprise RAG — Evaluation Results")
    print("=" * 52)
    print(f"  Questions evaluated : {report.total_questions}")
    print(f"  Retrieval Hit Rate  : {report.retrieval_hit_rate * 100:.1f}%")
    print(f"  Answer Correctness  : {report.answer_correctness * 100:.1f}%")
    print(f"  Citation Accuracy   : {report.citation_accuracy * 100:.1f}%")
    print(f"  Avg Latency         : {report.avg_latency_ms:.0f} ms")
    print("=" * 52 + "\n")

    print("Per-question results:")
    print("-" * 52)
    for r in report.results:
        status = "✓" if r.answer_correct else "✗"
        print(f"  {status}  {r.question[:60]}")
        if not r.retrieval_hit:
            print("     ⚠ No expected source retrieved")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("app/evaluation/evaluation_dataset.json"),
        help="Path to evaluation_dataset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save full results JSON",
    )
    args = parser.parse_args()

    print("\n🧠 Enterprise RAG Knowledge Assistant — Evaluation\n")

    pipeline = RAGPipeline()
    evaluator = RAGEvaluator(pipeline)
    report = evaluator.run(dataset_path=args.dataset, output_path=args.output)
    print_report(report)

    if args.output:
        print(f"Full results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
