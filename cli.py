"""Command-line interface for StudyMate AI.

Examples:
    python cli.py --file sample_notes.md --question "What is overfitting?"
    python cli.py --file notes.pdf            # interactive mode
    python cli.py --file notes.txt --retriever tfidf -q "Define recall"
"""

from __future__ import annotations

import argparse
import sys

from answer import answer_question
from ingest import ingest
from retrieve import build_retriever


def ask(retriever, question: str, k: int = 3) -> None:
    results = retriever.search(question, k=k)
    out = answer_question(question, results)
    print(f"\nQ: {question}")
    print(f"A ({out['method']}): {out['answer']}")
    print("Sources:")
    for res in out["sources"]:
        print(f"  - chunk {res.chunk.index} "
              f"(score {res.score:.3f}, {res.chunk.source}): {res.chunk.short()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="StudyMate AI — ask your notes.")
    parser.add_argument("--file", required=True,
                        help="Study document: .pdf, .txt or .md")
    parser.add_argument("--question", "-q", action="append", default=[],
                        help="Question to ask (repeatable). "
                             "Omit for interactive mode.")
    parser.add_argument("--retriever", default="auto",
                        choices=["auto", "embeddings", "tfidf"],
                        help="Retrieval backend (default: auto)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Chunks to retrieve per question (default: 3)")
    args = parser.parse_args()

    chunks = ingest(args.file)
    if not chunks:
        print("No readable text found in that document.", file=sys.stderr)
        return 1
    retriever = build_retriever(args.retriever).fit(chunks)
    print(f"Loaded {len(chunks)} chunks from {args.file} "
          f"(retriever: {retriever.name})")

    if args.question:
        for q in args.question:
            ask(retriever, q, k=args.top_k)
    else:
        print("Type a question, or 'quit' to exit.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                ask(retriever, q, k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
