"""
app.py
Main Streamlit application for ContraRAG.
"""

import streamlit as st
import anthropic
import os
from dotenv import load_dotenv

from contrarag.ingestion import ingest_source
from contrarag.embedder import get_embedder
from contrarag.vector_store import VectorStore
from contrarag.retriever import Retriever
from contrarag.detector import ContradictionDetector
from utils.formatting import format_results_for_display, get_confidence_color

load_dotenv()

st.set_page_config(
    page_title="ContraRAG",
    page_icon="⚡",
    layout="wide",
)


def init_session_state():
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "source_id_to_name" not in st.session_state:
        st.session_state.source_id_to_name = {}
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()


def reset_session():
    st.session_state.vector_store.clear_all()
    st.session_state.sources = []
    st.session_state.source_id_to_name = {}


def add_source(source, source_id: str, display_name: str):
    embedder = get_embedder()
    vector_store = st.session_state.vector_store

    with st.spinner(f"Ingesting {display_name}..."):
        chunks = ingest_source(source, source_id)
        if not chunks:
            st.error(f"No text could be extracted from {display_name}.")
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)
        vector_store.add_chunks(chunks, embeddings)

        st.session_state.sources.append(
            {
                "id": source_id,
                "name": display_name,
                "chunk_count": len(chunks),
            }
        )
        st.session_state.source_id_to_name[source_id] = display_name

    st.success(f"Added {display_name} ({len(chunks)} chunks)")


init_session_state()

with st.sidebar:
    st.title("⚡ ContraRAG")
    st.caption("Feed it conflicting sources. Watch it find the fights.")
    st.divider()
    st.markdown("""
**How to use:**
1. Add 2–5 documents in the **Add Sources** tab
2. Go to the **Analyze** tab
3. Ask a question about the topic
4. ContraRAG surfaces where your sources agree and contradict each other
""")
    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_session()
        st.rerun()

tab1, tab2 = st.tabs(["📥 Add Sources", "🔍 Analyze"])

with tab1:
    st.header("Add Sources")
    st.caption("Add 2 to 5 documents on the same topic.")

    if st.session_state.sources:
        st.subheader("Loaded Sources")
        for source in st.session_state.sources:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"📄 **{source['name']}** — {source['chunk_count']} chunks"
                )
            with col2:
                st.caption(f"ID: {source['id'][:8]}...")

    st.divider()

    if len(st.session_state.sources) >= 5:
        st.warning("Maximum of 5 sources reached. Reset the session to start over.")
    else:
        col_pdf, col_url = st.columns(2)

        with col_pdf:
            st.subheader("Upload PDF")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
            )
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    already_loaded = [
                        s["name"] for s in st.session_state.sources
                    ]
                    if uploaded_file.name in already_loaded:
                        st.info(f"{uploaded_file.name} already loaded.")
                        continue
                    if len(st.session_state.sources) >= 5:
                        st.warning("Max 5 sources reached.")
                        break
                    source_id = f"pdf_{uploaded_file.name.replace(' ', '_')}"
                    add_source(uploaded_file, source_id, uploaded_file.name)

        with col_url:
            st.subheader("Add URL")
            url_input = st.text_input(
                "Enter a URL",
                placeholder="https://example.com/article",
                key="url_input",
            )
            if st.button("Add URL", use_container_width=True):
                if not url_input.startswith("http"):
                    st.error("Please enter a valid URL starting with http.")
                else:
                    already_loaded = [s["name"] for s in st.session_state.sources]
                    if url_input in already_loaded:
                        st.info("This URL is already loaded.")
                    elif len(st.session_state.sources) >= 5:
                        st.warning("Max 5 sources reached.")
                    else:
                        source_id = f"url_{url_input.replace('https://', '').replace('http://', '').replace('/', '_')[:40]}"
                        add_source(url_input, source_id, url_input)

with tab2:
    st.header("Analyze Sources")

    if len(st.session_state.sources) < 2:
        st.info("Add at least 2 sources in the Add Sources tab to begin analysis.")
    else:
        st.caption(
            f"Analyzing across {len(st.session_state.sources)} sources: "
            + ", ".join(s["name"] for s in st.session_state.sources)
        )

        query = st.text_input(
            "What do you want to analyze?",
            placeholder="e.g. What are the causes of inflation?",
        )

        if st.button("⚡ Analyze", use_container_width=True, type="primary"):
            if not query.strip():
                st.error("Please enter a question.")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    st.error("ANTHROPIC_API_KEY not found. Add it to your .env file.")
                else:
                    with st.spinner("Retrieving relevant chunks from all sources..."):
                        embedder = get_embedder()
                        vector_store = st.session_state.vector_store
                        retriever = Retriever(vector_store, embedder)
                        source_chunks = retriever.retrieve_all_sources(query)

                    with st.spinner("Claude is analyzing contradictions..."):
                        client = anthropic.Anthropic(api_key=api_key)
                        detector = ContradictionDetector(client)
                        raw_results = detector.detect(query, source_chunks)

                    if "error" in raw_results:
                        st.error(f"Analysis failed: {raw_results['error']}")
                        if raw_results.get("raw"):
                            st.code(raw_results["raw"])
                    else:
                        results = format_results_for_display(
                            raw_results,
                            st.session_state.source_id_to_name,
                        )

                        confidence = results.get("confidence", "low")
                        color = get_confidence_color(confidence)
                        st.markdown(
                            f"**Confidence:** "
                            f"<span style='color:{color};font-weight:600'>"
                            f"{confidence.upper()}</span>",
                            unsafe_allow_html=True,
                        )

                        st.divider()

                        with st.expander("✅ Points of Agreement", expanded=True):
                            agreements = results.get("agreements", [])
                            if agreements:
                                for point in agreements:
                                    st.markdown(f"- {point}")
                            else:
                                st.write("No clear points of agreement found.")

                        with st.expander("⚡ Contradictions Found", expanded=True):
                            contradictions = results.get("contradictions", [])
                            if contradictions:
                                for contradiction in contradictions:
                                    st.subheader(contradiction.get("topic", ""))
                                    positions = contradiction.get("positions", {})
                                    cols = st.columns(len(positions))
                                    for col, (source_name, position) in zip(
                                        cols, positions.items()
                                    ):
                                        with col:
                                            st.markdown(f"**{source_name}**")
                                            st.write(position)
                            else:
                                st.write("No contradictions detected.")

                        with st.expander("📄 Source Summaries", expanded=False):
                            summaries = results.get("source_summaries", {})
                            for source_name, summary in summaries.items():
                                st.markdown(f"**{source_name}**")
                                st.write(summary)
                                st.divider()

                        with st.expander("🔍 Raw JSON Output", expanded=False):
                            st.json(raw_results)
