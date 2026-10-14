"""StudyMate AI — Streamlit app.

Run:  streamlit run app.py

Upload your study material (PDF / TXT / MD), then ask questions.
Answers come only from your document, with the source chunks shown.
"""

from __future__ import annotations

import streamlit as st

from answer import answer_question
from ingest import chunk_text, load_document_bytes
from retrieve import build_retriever

st.set_page_config(page_title="StudyMate AI", page_icon="📚")
st.title("📚 StudyMate AI")
st.caption("Ask questions about your own study material. "
           "Answers are grounded in your document — sources always shown.")

uploaded = st.file_uploader("Upload your notes (.pdf, .txt, .md)",
                            type=["pdf", "txt", "md"])

if uploaded is not None:
    try:
        text = load_document_bytes(uploaded.name, uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    chunks = chunk_text(text, source=uploaded.name)
    if not chunks:
        st.error("No readable text found in that file.")
        st.stop()

    if st.session_state.get("doc_name") != uploaded.name:
        with st.spinner("Indexing your document…"):
            st.session_state.retriever = build_retriever("auto").fit(chunks)
            st.session_state.doc_name = uploaded.name
            st.session_state.chunks = chunks

    retriever = st.session_state.retriever
    st.success(f"Indexed {len(chunks)} chunks "
               f"(retriever: {retriever.name}). Ask away!")

    question = st.text_input("Your question")
    if question:
        results = retriever.search(question, k=3)
        out = answer_question(question, results)
        st.markdown(f"### Answer\n{out['answer']}")
        st.caption(f"Answer method: {out['method']}")
        with st.expander("📖 Sources — the chunks this answer is grounded in"):
            for res in out["sources"]:
                st.markdown(
                    f"**Chunk {res.chunk.index}** · score {res.score:.3f}\n\n"
                    f"> {res.chunk.text}"
                )
else:
    st.info("Upload a document to get started. "
            "Try the included `sample_notes.md`.")
