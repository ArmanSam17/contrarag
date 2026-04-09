# ⚡ ContraRAG

> Feed it conflicting sources. Watch it find the fights.

Standard RAG pipelines retrieve relevant chunks and synthesize them into a single answer. That works well for lookup — but it quietly buries disagreement. When two sources contradict each other, most pipelines average them out or pick the dominant one. ContraRAG does the opposite: it keeps sources isolated through retrieval, then uses Claude to map exactly where they agree and where they conflict. Contradiction is the output, not a side effect to be smoothed over.

---

## Features

- **Points of Agreement** — claims that hold consistently across all loaded sources
- **Contradictions Found** — specific topics where sources diverge, with each position shown side by side
- **Source Summaries** — one-sentence read on each source's overall stance toward the query
- **Confidence Rating** — high / medium / low based on how directly the sources address the question

---

## Architecture

```
User Input (PDF or URL)
        │
        ▼
Ingestion ──────────── PyMuPDF (PDF) / BeautifulSoup (URL)
        │
        ▼
Chunking ───────────── LangChain RecursiveCharacterTextSplitter
        │               500 char chunks, 50 char overlap
        ▼
Embedding ──────────── sentence-transformers all-MiniLM-L6-v2
                        Runs locally — no API call
        │
        ▼
Vector Storage ─────── ChromaDB in-memory
                        One collection per source document
        │
        ▼
Per-Source Retrieval ── Top-5 chunks pulled independently per source
        │
        ▼
Contradiction Detection  Claude claude-sonnet-4-5
                        Returns structured JSON — agreements,
                        contradictions, summaries, confidence
        │
        ▼
Streamlit Display ───── Side-by-side contradiction view
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Anthropic Claude (`claude-sonnet-4-5`) |
| PDF Parsing | PyMuPDF (`fitz`) |
| URL Scraping | Requests + BeautifulSoup4 |
| Text Chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (in-memory) |
| Config | python-dotenv |

---

## Setup

```bash
git clone https://github.com/ArmanSam17/contrarag.git
cd contrarag

pip install -r requirements.txt

cp .env.example .env
# Add your Anthropic API key to .env

streamlit run app.py
```

---

## Project Structure

```
contrarag/
├── app.py                        # Streamlit app — UI, session state, tab layout
├── contrarag/
│   ├── __init__.py               # Package exports
│   ├── ingestion.py              # PDF parsing, URL scraping, text chunking
│   ├── embedder.py               # sentence-transformers wrapper + singleton
│   ├── vector_store.py           # ChromaDB — one collection per source
│   ├── retriever.py              # Per-source top-k chunk retrieval
│   ├── detector.py               # Claude API call — contradiction detection
│   └── prompts.py                # Prompt templates
├── utils/
│   └── formatting.py             # Label remapping + confidence color helper
├── tests/
│   ├── test_ingestion.py
│   └── test_detector.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Design Decisions

**Per-source ChromaDB collections**
The obvious approach is one shared vector store for everything. The problem is that retrieval then returns whichever chunks score highest globally — which tends to over-represent longer or more embeddings-dense documents and silently drop context from shorter ones. One collection per source means retrieval is always balanced: every loaded document contributes equally to the analysis regardless of length or similarity distribution.

**Structured JSON output from Claude**
The prompt asks Claude to return a strict JSON schema rather than prose. This makes the output directly renderable by Streamlit without any parsing layer between the API response and the UI. The detector also strips markdown fences before parsing, since Claude occasionally wraps JSON in code blocks even when instructed not to.

**Local embeddings**
Embeddings run locally via `sentence-transformers` rather than through an external API. This removes a second network dependency, eliminates per-token embedding costs, and means ingestion stays fast even with several documents loaded. The model loads once and is reused across all sources via a module-level singleton in `embedder.py`.

**Silent source detection**
The prompt instructs Claude to write "Silent on this topic" when a source doesn't address a particular point, rather than inferring a position. For a tool built around surfacing factual disagreement, a fabricated stance is strictly worse than an honest gap — so the prompt makes that explicit.
