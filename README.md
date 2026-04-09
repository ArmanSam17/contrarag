# ⚡ ContraRAG

> **Feed it conflicting sources. Watch it find the fights.**

---

## What Makes It Different

Standard RAG pipelines retrieve relevant chunks from multiple documents and blend them into a single synthesized answer. That approach is great for question answering — but it buries disagreement. When two sources contradict each other, a typical RAG response will either pick one, average them out, or quietly omit the conflict entirely.

**ContraRAG treats contradiction as the primary signal.**

Instead of merging sources into one answer, it maps where they agree and where they diverge — surfacing the tension rather than suppressing it. Each source is kept isolated through retrieval, then Claude analyzes them side by side to identify specific points of conflict.

**Who this is for:**
- **Researchers** comparing papers, reports, or studies on the same topic
- **Analysts** auditing conflicting market reports or policy documents
- **Journalists** cross-referencing sources before publishing
- **Students** stress-testing arguments across multiple texts

---

## What You Get

After loading 2–5 sources and asking a question, ContraRAG returns four structured outputs:

| Output | Description |
|---|---|
| **Points of Agreement** | Claims or positions that are consistent across all sources |
| **Contradictions Found** | Specific topics where sources disagree, with each source's position shown side by side |
| **Source Summaries** | One-sentence summary of each source's overall stance on your query |
| **Confidence Rating** | High / Medium / Low — how directly the sources address the question |

---

## Architecture

```
User Input (PDF or URL)
        │
        ▼
Ingestion ──────────── PyMuPDF (PDF text extraction)
                        BeautifulSoup (URL scraping)
        │
        ▼
Chunking ───────────── LangChain RecursiveCharacterTextSplitter
        │               (500 char chunks, 50 char overlap)
        ▼
Embedding ──────────── sentence-transformers (all-MiniLM-L6-v2)
                        Runs locally, no API call
        │
        ▼
Vector Storage ─────── ChromaDB (in-memory)
                        One collection per source document
        │
        ▼
Per-Source Retrieval ── Top-5 chunks retrieved independently per source
        │               (sources never mixed at retrieval time)
        ▼
Contradiction Detection  Claude claude-sonnet-4-5
                        Structured JSON prompt → agreements, contradictions,
                        summaries, confidence
        │
        ▼
Streamlit Display ───── Formatted side-by-side contradiction view
```

**Why per-source collections matter:**
If all chunks from all documents were stored in a single vector collection, retrieval would return whichever chunks score highest across all sources — likely over-representing the dominant source and silently dropping minority views. By keeping one ChromaDB collection per source, ContraRAG guarantees that every loaded document contributes equally to the analysis, regardless of length or similarity score.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM | [Anthropic Claude](https://anthropic.com) (`claude-sonnet-4-5`) |
| PDF Parsing | [PyMuPDF](https://pymupdf.readthedocs.io) (`fitz`) |
| URL Scraping | [Requests](https://requests.readthedocs.io) + [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) |
| Text Chunking | [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) |
| Embeddings | [sentence-transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`, local) |
| Vector Store | [ChromaDB](https://www.trychroma.com) (in-memory) |
| Config | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/contrarag.git
cd contrarag
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Anthropic API key**
```bash
cp .env.example .env
# Open .env and replace your_key_here with your actual key
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
contrarag/
├── app.py                        # Streamlit app — UI, session state, tab layout
├── contrarag/
│   ├── __init__.py               # Package exports for all core modules
│   ├── ingestion.py              # PDF parsing, URL scraping, text chunking
│   ├── embedder.py               # sentence-transformers wrapper + singleton
│   ├── vector_store.py           # ChromaDB client — one collection per source
│   ├── retriever.py              # Per-source top-k chunk retrieval
│   ├── detector.py               # Claude API call — contradiction detection
│   └── prompts.py                # Prompt templates for Claude
├── utils/
│   ├── __init__.py               # Utils package init
│   └── formatting.py             # Label remapping + confidence color helper
├── tests/
│   ├── test_ingestion.py         # Tests for PDF/URL parsing and chunking
│   └── test_detector.py          # Tests for contradiction detection output
├── requirements.txt              # All Python dependencies
├── .env.example                  # API key template
├── .gitignore                    # Excludes .env, caches, build artifacts
└── README.md                     # This file
```

---

## Portfolio Notes

*For recruiters and hiring managers — five deliberate design decisions worth noting:*

**1. Per-source ChromaDB collections instead of a shared index**
The standard approach is to dump everything into one vector store and retrieve globally. ContraRAG uses isolated collections per source so that retrieval is guaranteed to return context from every document, regardless of embedding similarity rankings. This prevents dominant sources from crowding out minority views — a subtle but important correctness property for a contradiction-detection use case.

**2. Structured JSON output from Claude instead of free text**
Rather than asking Claude to write a report, the prompt instructs it to return a strict JSON schema with typed fields (`agreements`, `contradictions`, `source_summaries`, `confidence`). This makes the output programmatically reliable and directly renderable by the Streamlit UI without any post-processing parsing logic. The detector also includes a markdown-fence stripping step to handle Claude occasionally wrapping JSON in code blocks.

**3. Local embeddings with sentence-transformers**
Embeddings are generated locally using `all-MiniLM-L6-v2` rather than through an external API. This eliminates a second API dependency, avoids per-token embedding costs, removes a latency bottleneck for multi-source ingestion, and means the app works offline after the model is cached. The singleton pattern in `embedder.py` ensures the model is loaded once per session.

**4. Separation of concerns across modules**
Each module has a single responsibility: ingestion handles parsing, embedder handles vectors, vector_store handles persistence, retriever handles query-time lookup, detector handles LLM calls, and prompts holds all prompt strings. This means each component is independently testable and replaceable — swap ChromaDB for Pinecone, or `claude-sonnet-4-5` for a different model, without touching anything else.

**5. Silent source detection instead of hallucination**
The prompt explicitly instructs Claude to write "Silent on this topic" if a source doesn't address a particular contradiction, rather than inferring or inventing a position. This is a deliberate anti-hallucination measure — in a tool designed to surface factual disagreements, a fabricated position is worse than no position.

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
