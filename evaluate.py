"""Retrieval evaluation for StudyMate AI.

Runs a set of questions against the sample notes and checks whether the
chunk(s) containing the expected keywords are retrieved. Reports top-1
and top-3 hit rates — the real numbers quoted in the README come from
running this script.

    python evaluate.py                  # auto retriever
    python evaluate.py --retriever tfidf
    python evaluate.py --retriever embeddings
"""

from __future__ import annotations

import argparse

from ingest import ingest
from retrieve import build_retriever

# (question, keywords that appear in the chunk holding the answer)
CASES = [
    ("What is machine learning?",
     ["explicitly programmed", "learns patterns"]),
    ("What is the difference between supervised and unsupervised learning?",
     ["labels", "no labels"]),
    ("What is reinforcement learning?",
     ["rewards or penalties", "agent"]),
    ("What is the difference between classification and regression?",
     ["continuous number", "discrete category"]),
    ("Why do we split data into training, validation and test sets?",
     ["test set", "hyperparameters"]),
    ("What is overfitting?",
     ["memorising noise", "poorly on new data"]),
    ("What is the F1 score?",
     ["harmonic mean", "precision and recall"]),
    ("Why is recall important in cancer screening?",
     ["recall", "missing a real case"]),
    ("How does gradient descent work?",
     ["learning rate", "loss function"]),
    ("What is backpropagation in a neural network?",
     ["chain rule", "backpropagation"]),
    ("How are decision trees evaluated for a split?",
     ["entropy", "Gini"]),
    ("What is feature scaling in preprocessing?",
     ["standardisation", "min-max"]),
]


# Harder probe: questions deliberately worded WITHOUT the notes' vocabulary.
PARAPHRASE_CASES = [
    ("What happens when a model just memorizes the examples instead of "
     "really learning?",
     ["memorising noise", "overfitting"]),
    ("How can we tell whether the model will do well on brand new data it "
     "has never seen?",
     ["test set", "unseen data"]),
    ("Which single number balances false alarms against missed positive "
     "cases?",
     ["harmonic mean", "F1 score"]),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="sample_notes.md")
    parser.add_argument("--retriever", default="auto",
                        choices=["auto", "embeddings", "tfidf"])
    parser.add_argument("--paraphrase", action="store_true",
                        help="run the harder paraphrased-question probe "
                             "instead of the main question set")
    args = parser.parse_args()
    cases = PARAPHRASE_CASES if args.paraphrase else CASES

    chunks = ingest(args.file)
    retriever = build_retriever(args.retriever).fit(chunks)
    print(f"Retriever: {retriever.name} | chunks: {len(chunks)} | "
          f"questions: {len(cases)}\n")

    top1 = top3 = 0
    for question, keywords in cases:
        results = retriever.search(question, k=3)
        texts1 = results[0].chunk.text.lower()
        texts3 = " ".join(r.chunk.text.lower() for r in results)
        hit1 = any(kw.lower() in texts1 for kw in keywords)
        hit3 = any(kw.lower() in texts3 for kw in keywords)
        top1 += hit1
        top3 += hit3
        print(f"[{'HIT ' if hit1 else 'MISS'}] top1={hit1!s:5} top3={hit3!s:5} "
              f"best_chunk={results[0].chunk.index} "
              f"score={results[0].score:.3f} :: {question}")

    n = len(cases)
    print(f"\nTop-1 hit rate: {top1}/{n} = {top1 / n:.1%}")
    print(f"Top-3 hit rate: {top3}/{n} = {top3 / n:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
