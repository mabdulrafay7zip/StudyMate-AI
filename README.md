# 📚 StudyMate AI — Ask Your Own Notes

A **Retrieval-Augmented Generation (RAG)** study assistant. Upload your study
material (PDF or `.txt` / `.md` notes), ask questions in plain words, and get
answers built **only from your document** — with the exact source chunks shown
under every answer, so you can verify each claim against your own notes.

No paid APIs. No API keys. Everything runs locally and for free.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## How RAG works here

```
  your notes (PDF / TXT / MD)
            │
            ▼
   ┌─────────────────┐     chunks never cross a heading,
   │  ingest.py      │     ~700 chars each, with overlap
   │  chunking       │
   └────────┬────────┘
            ▼
   ┌─────────────────┐     embeddings (all-MiniLM-L6-v2)
   │  retrieve.py    │     — or TF-IDF if embeddings are
   │  vector search  │       unavailable (automatic fallback)
   └────────┬────────┘
            │  top-3 chunks for your question
            ▼
   ┌─────────────────┐     extractive answer composed from the
   │  answer.py      │     chunks (always works) — or phrased by
   │  grounded reply │     a local Ollama model if one is running
   └────────┬────────┘
            ▼
   answer + the source chunks it came from ✔ verify it yourself
```

The core design rule: **the answer may only say what your document says.**
The default answerer is extractive — it composes the reply from sentences in
the retrieved chunks — so it cannot hallucinate facts that are not in your
notes. If a local [Ollama](https://ollama.com) server happens to be running,
StudyMate will optionally let it phrase the answer from the same chunks;
Ollama is never required.

## Setup

```bash
pip install -r requirements.txt
```

This installs the core (scikit-learn, pypdf, Streamlit) and
sentence-transformers for semantic retrieval. **If sentence-transformers
cannot be installed or the model cannot be downloaded, nothing breaks** —
the app automatically falls back to TF-IDF retrieval.

## Run it

**Web app (Streamlit):**

```bash
streamlit run app.py
```

**Command line:**

```bash
# one-shot questions
python cli.py --file sample_notes.md --question "What is overfitting?"

# interactive mode
python cli.py --file my_notes.pdf

# force a retriever
python cli.py --file notes.txt --retriever tfidf -q "Define recall"
```

A sample document, `sample_notes.md` (original notes on *Introduction to
Machine Learning*), is included so you can try it immediately.

## Evaluation — real, measured results

`evaluate.py` runs fixed questions against `sample_notes.md` (12 chunks) and
checks whether the chunk containing the answer is retrieved. Every number
below is the actual output of these commands — reproduce them yourself:

```bash
python evaluate.py --retriever tfidf
python evaluate.py --retriever embeddings
python evaluate.py --retriever tfidf      --paraphrase
python evaluate.py --retriever embeddings --paraphrase
```

**Main question set — 12 questions** (asked in words close to the notes):

| Retriever | Top-1 hit rate | Top-3 hit rate |
|---|---|---|
| TF-IDF | 12/12 = **100%** | 12/12 = **100%** |
| Embeddings (all-MiniLM-L6-v2) | 12/12 = **100%** | 12/12 = **100%** |

**Paraphrase probe — 3 questions** (deliberately worded *without* the notes'
vocabulary, e.g. *"What happens when a model just memorizes the examples
instead of really learning?"*):

| Retriever | Top-1 hit rate | Top-3 hit rate |
|---|---|---|
| TF-IDF | 1/3 = 33.3% | 2/3 = 66.7% |
| Embeddings (all-MiniLM-L6-v2) | 1/3 = 33.3% | 2/3 = 66.7% |

**Honest reading of these numbers:** on well-structured notes, when questions
use vocabulary similar to the text, both retrievers find the right chunk
every time. Heavily paraphrased questions are genuinely hard for both on a
small document — retrieval quality is the ceiling of any RAG system, and
these 12 chunks are an easy test set, not proof of perfection. Treat the
main-set 100% as "works correctly on clean notes", and the paraphrase probe
as the known weak spot (bigger documents and more chunks per topic are where
embeddings usually pull ahead — that remains to be measured, not claimed).

Also verified end-to-end: the CLI answers questions against a **PDF**
version of the sample notes (12 chunks, correct source citations), and the
Streamlit app starts and serves cleanly.

## Project structure

```
studymate-ai/
├── app.py            # Streamlit web app (upload → ask → answer + sources)
├── cli.py            # command-line interface (one-shot + interactive)
├── ingest.py         # PDF/TXT/MD loading + heading-aware chunking
├── retrieve.py       # embeddings retriever + TF-IDF retriever (one interface)
├── answer.py         # extractive answers + optional Ollama phrasing
├── evaluate.py       # retrieval evaluation (the numbers above)
├── sample_notes.md   # sample study document to try immediately
├── requirements.txt
└── LICENSE           # MIT
```

## Limitations (honest)

- **Answers are extractive by default** — stitched from your notes' own
  sentences, so they can read a little plainly and never explain beyond the
  text. The optional Ollama mode phrases things more fluently but is only as
  good as your local model.
- **Retrieval is the ceiling.** If the right chunk is not in the top-3, the
  answer will be incomplete — the paraphrase probe above shows this is a
  real effect, not a theoretical one.
- **Scanned/image PDFs** yield little or no text (no OCR yet), and tables
  or equations extracted from PDFs can come out garbled.
- Chunking is heading-aware for Markdown-style notes; very unstructured
  documents fall back to plain sentence packing.
- One document at a time; no multi-file library or chat memory yet.

## Optional: Ollama answers

If you run [Ollama](https://ollama.com) locally (`ollama serve` with any
chat model pulled), StudyMate detects it automatically and uses it to phrase
answers from the retrieved chunks. Set `OLLAMA_MODEL` to pick a model and
`OLLAMA_HOST` if it is not on `http://localhost:11434`. With no Ollama
server, the extractive answerer is used — nothing else changes.

## License

MIT — see [LICENSE](LICENSE).

Built by **Muhammad Abdul Rafay** — BS Artificial Intelligence, Air University
Islamabad · GitHub: [mabdulrafay7zip](https://github.com/mabdulrafay7zip)
