"""
ingestion.py
Handles PDF parsing, URL scraping, and text chunking for ContraRAG.
"""

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any


def extract_text_from_pdf(file_obj: Any) -> str:
    """
    Extract text from an uploaded PDF file object.

    Args:
        file_obj: A file-like object (e.g. from Streamlit file_uploader).

    Returns:
        A single string containing all extracted text from the PDF.
    """
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_url(url: str) -> str:
    """
    Fetch and extract visible text from a URL.

    Args:
        url: The URL to scrape.

    Returns:
        A string of visible text extracted from the page.

    Raises:
        ValueError: If the request fails or returns a non-200 status.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        raise ValueError(
            f"Connection timed out while trying to reach: {url}. "
            "The site may be slow or unreachable — try again or use a different source."
        )
    if response.status_code == 403:
        raise ValueError(
            f"Access denied (403): {url} blocks automated access. "
            "Try a different URL or paste the text manually."
        )
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch URL (status {response.status_code}): {url}"
        )
    soup = BeautifulSoup(response.text, "html.parser")
    tags = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
    text = "\n".join(tag.get_text(strip=True) for tag in tags)
    return text


def chunk_text(
    text: str,
    source_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text: The raw text to split.
        source_id: Identifier for the source document.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        A list of dicts, each with keys: text, source_id, chunk_index.
    """
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source_id": source_id, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]


def ingest_source(source: Any, source_id: str) -> list[dict]:
    """
    Ingest a source (PDF file object or URL string) into chunks.

    Args:
        source: Either a URL string (starting with http) or a PDF file object.
        source_id: Unique identifier for this source.

    Returns:
        A list of chunk dicts with keys: text, source_id, chunk_index.
    """
    if isinstance(source, str) and source.startswith("http"):
        text = extract_text_from_url(source)
    else:
        text = extract_text_from_pdf(source)
    return chunk_text(text, source_id)
