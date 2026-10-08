"""Shared policy-document loading, chunking, indexing, retrieval and RAG answering."""

import re

import numpy as np

from .client import get_client
from .config import CHAT_MODEL, EMBEDDING_MODEL
from .documents import POLICY_DOCUMENTS, applicable_documents, load_pdf_text
from .vector_store import MiniVectorStore


def load_documents() -> dict[str, str]:
    """The text of every policy PDF, keyed by document code, e.g. "DPAM01-E4"."""
    return {doc_code: load_pdf_text(doc_code) for doc_code in POLICY_DOCUMENTS}


def embed(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts. Returns one vector per input, preserving order."""
    response = get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [np.array(item.embedding) for item in response.data]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100, separator: str | None = None) -> list[str]:
    """Split text into overlapping fixed-size character windows, or, given a separator, at every occurrence of it."""
    if separator is not None:
        # Keep the separator with the piece it starts, e.g. "\nArt. " gives pieces starting "Art. 3 Eligibility".
        pieces = text.split(separator)
        pieces = pieces[:1] + [separator + p for p in pieces[1:]]
        return [p.strip() for p in pieces if p.strip()]
    if not 0 <= overlap < chunk_size:
        raise ValueError(f"overlap must be at least 0 and smaller than chunk_size, got overlap={overlap}, chunk_size={chunk_size}.")
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


# A heading starts a line and its title a capital letter: "Art. 35c LCA/VVG" inside an article is a cross-reference.
ARTICLE_HEADING = re.compile(r"(?m)^Art\. \d+[a-z]? +[A-Z][^\n]*")


def split_on_articles(text: str, header_re=ARTICLE_HEADING) -> list[str]:
    """Split at article headings, keeping each heading with its article.

    Pass another compiled pattern to examine or demonstrate alternate heading rules.
    """
    starts = [m.start() for m in header_re.finditer(text)]
    return [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]


def split_structured(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split on articles, then window long articles, repeating the heading on every piece."""
    pieces = []
    for article in split_on_articles(text):
        heading, _, body = article.partition("\n")
        if len(article) <= max_chars:
            pieces.append(article)
            continue
        for window in chunk_text(body, max_chars - len(heading) - 1, overlap):
            pieces.append(f"{heading}\n{window}")
    return pieces


def build_store(documents: dict[str, str], splitter, metadata=lambda doc_code: {}) -> MiniVectorStore:
    """Split every document with `splitter` and index the pieces.

    Each piece's metadata is {"source": doc_code, **metadata(doc_code)}.
    """
    pieces = [(doc_code, piece) for doc_code, text in documents.items() for piece in splitter(text)]
    store = MiniVectorStore()
    store.add(
        embeddings=embed([piece for _, piece in pieces]),
        texts=[piece for _, piece in pieces],
        metadata=[{"source": doc_code, **metadata(doc_code)} for doc_code, _ in pieces],
    )
    return store


def retrieve(store: MiniVectorStore, question: str, k: int = 3, keep=None) -> list[dict]:
    """The k pieces most similar to the question; `keep` is an optional filter on their metadata."""
    return store.query(embed([question])[0], k=k, keep=keep)


def search_policy(
    store: MiniVectorStore,
    product: str,
    treatment_date: str,
    query: str,
    k: int = 3,
) -> list[dict]:
    """Search only the policy editions that apply to `product` on `treatment_date`.

    Each result has `source`, `article` and `text`, matching the policy-search tool used in M04.
    """
    from datetime import date

    date.fromisoformat(treatment_date)
    product_name = next(
        (meta["product"] for meta in POLICY_DOCUMENTS.values()
         if meta["product"].casefold() == product.casefold()),
        None,
    )
    doc_codes = applicable_documents(product_name, treatment_date) if product_name else set()
    if not any(POLICY_DOCUMENTS[code]["product"] == product_name for code in doc_codes):
        raise ValueError(f"No terms and conditions for {product} on {treatment_date}.")

    hits = retrieve(
        store,
        query,
        k=k,
        keep=lambda metadata: metadata["source"] in doc_codes,
    )
    return [
        {
            "source": hit["metadata"]["source"],
            "article": hit["text"].splitlines()[0],
            "text": hit["text"],
        }
        for hit in hits
    ]


def show(results: list[dict], preview: int = 200) -> None:
    """Print each result's rank, score, source and first line, then (if preview) the start of its text."""
    for rank, r in enumerate(results, start=1):
        first_line = r["text"].splitlines()[0]
        print(f"[{rank}] {r['score']:.4f}  {r['metadata']['source']:10s} {first_line}")
        if preview:
            print(f"    {' '.join(r['text'].split())[:preview]}...")


def answer(store: MiniVectorStore, question: str, k: int = 3, instruction: str = "", keep=None,
           label=lambda m: m["source"]) -> str:
    """Retrieve the top k pieces and answer the question from them only.

    instruction: extra lines for the prompt. label: how each extract is labelled, from its metadata.
    """
    results = retrieve(store, question, k=k, keep=keep)
    context = "\n\n".join(f"[{label(r['metadata'])}]\n{r['text']}" for r in results)

    prompt = f"""Answer the question using only the extracts below.
{instruction}

EXTRACTS
{context}

QUESTION
{question}
"""
    return get_client().responses.create(model=CHAT_MODEL, input=prompt).output_text
