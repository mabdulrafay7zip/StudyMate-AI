"""Answer composition for StudyMate AI.

The always-works core is *extractive*: the answer is built only from
sentences in the retrieved chunks, and the chunks used are returned as
sources so the student can verify every claim against their own notes.

Optional: if a local Ollama server is running, the retrieved chunks are
handed to it to phrase a fluent answer. Ollama is never required — when
it is absent (or errors), the extractive answer is used.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from ingest import split_sentences
from retrieve import SearchResult

_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "how", "does", "do",
    "are", "was", "were", "it", "to", "and", "or", "for", "with", "by",
    "between", "difference", "explain", "define", "mean", "meant", "can",
    "you", "we", "i", "this", "that", "these", "those", "when", "why",
    "which", "who", "whom", "should", "would", "could", "about", "work",
    "works", "used", "use", "using",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def extractive_answer(question: str, results: list[SearchResult],
                      max_sentences: int = 3) -> str:
    """Compose an answer from the most question-relevant retrieved sentences."""
    q_words = _content_words(question)
    scored: list[tuple[float, int, str]] = []
    seen: set[str] = set()
    for rank, res in enumerate(results):
        for sent in split_sentences(res.chunk.text):
            if sent in seen or len(sent) < 25:
                continue  # skip duplicates and heading-like fragments
            seen.add(sent)
            s_words = _content_words(sent)
            overlap = len(q_words & s_words)
            # small bonus for sentences from higher-ranked chunks
            score = overlap + (0.25 if rank == 0 else 0.0)
            scored.append((score, res.chunk.index, sent))
    if not scored:
        return "I could not find anything about that in your document."
    scored.sort(key=lambda t: (-t[0], t[1]))
    # only sentences that actually share a content word with the question
    best = [s for s in scored if s[0] >= 1.0][:max_sentences]
    if not best:
        # question shares no vocabulary with the notes; quote the start of
        # the best-matching chunk rather than inventing an answer
        top_chunk_sentences = split_sentences(results[0].chunk.text)
        return " ".join(top_chunk_sentences[:2])
    # present chosen sentences in document order for readability
    best.sort(key=lambda t: t[1])
    return " ".join(t[2] for t in best)


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_available(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{_ollama_base()}/api/tags",
                                   timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def ollama_answer(question: str, results: list[SearchResult],
                  timeout: float = 60.0) -> str | None:
    """Ask a local Ollama model to phrase an answer from the chunks only."""
    try:
        with urllib.request.urlopen(f"{_ollama_base()}/api/tags",
                                   timeout=5) as resp:
            models = [m["name"] for m in json.load(resp).get("models", [])]
        if not models:
            return None
        model = os.environ.get("OLLAMA_MODEL") or models[0]
        context = "\n\n".join(r.chunk.text for r in results)
        prompt = (
            "You are a study assistant. Answer the question using ONLY the "
            "notes below. If the notes do not contain the answer, say so.\n\n"
            f"Notes:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        payload = json.dumps(
            {"model": model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{_ollama_base()}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = json.load(resp).get("response", "").strip()
        return text or None
    except Exception:
        return None


def answer_question(question: str, results: list[SearchResult],
                    use_ollama: bool = True) -> dict:
    """Return {'answer', 'sources', 'method'} for a question.

    'method' is 'ollama' when a local Ollama model phrased the answer and
    'extractive' otherwise. 'sources' are the retrieved chunks the answer
    is grounded in, best first.
    """
    if not results:
        return {"answer": "No document has been loaded yet.",
                "sources": [], "method": "none"}
    answer, method = None, "extractive"
    if use_ollama and ollama_available():
        answer = ollama_answer(question, results)
        if answer:
            method = "ollama"
    if answer is None:
        answer = extractive_answer(question, results)
    return {"answer": answer, "sources": results, "method": method}
