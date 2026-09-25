#!/usr/bin/env python3
"""
scripts/ingest.py

CLI script to batch-ingest PDF documents into the knowledge base.

Usage:
    python scripts/ingest.py --dir data/documents/
    python scripts/ingest.py --file data/documents/leave_policy.pdf
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import configure_logging, get_logger
from app.core.security import sanitise_filename
from app.rag.pipeline import RAGPipeline

configure_logging()
logger = get_logger("ingest")


def ingest_file(pipeline: RAGPipeline, file_path: Path) -> bool:
    """Ingest a single PDF file. Returns True on success."""
    logger.info("ingesting", file=str(file_path))

    content = file_path.read_bytes()
    safe_name = sanitise_filename(file_path.name)

    result = pipeline.ingest_document(content=content, filename=safe_name)

    print(
        f"  ✓ {result.filename} — {result.page_count} pages, "
        f"{result.chunk_count} chunks ({result.document_id[:8]}…)"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG knowledge base")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Directory of PDFs to ingest")
    group.add_argument("--file", type=Path, help="Single PDF to ingest")
    args = parser.parse_args()

    print("\n🧠 Enterprise RAG Knowledge Assistant — Document Ingestion\n")

    pipeline = RAGPipeline()

    if args.file:
        files = [args.file]
    else:
        files = list(args.dir.glob("*.pdf")) + list(args.dir.glob("*.PDF"))
        if not files:
            print(f"  No PDF files found in {args.dir}")
            return

    print(f"Found {len(files)} file(s) to ingest…\n")

    success = 0
    for f in files:
        try:
            ingest_file(pipeline, f)
            success += 1
        except Exception as e:
            print(f"  ✗ {f.name} — {e}")

    print(f"\n✅ Done: {success}/{len(files)} documents ingested successfully.\n")


if __name__ == "__main__":
    main()
