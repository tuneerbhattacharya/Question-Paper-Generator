"""Load a PDF, chunk it, and build a FAISS index."""
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qpg import config


def load_pdf(path: str):
    pages = PyPDFLoader(path).load()
    for p in pages:                                   # PyPDFLoader pages are 0-based
        p.metadata["page"] = int(p.metadata.get("page", 0)) + 1
    return pages


def build_index(pages) -> Tuple[FAISS, int]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,
                                              chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    index = FAISS.from_documents(chunks, OpenAIEmbeddings(model=config.EMBEDDING_MODEL))
    return index, len(chunks)


def make_search(index: FAISS):
    """MMR search returns relevant but diverse chunks (fewer near-duplicates)."""
    def search(query: str, k: int) -> List[dict]:
        docs = index.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)
        return [{"text": d.page_content, "page": d.metadata.get("page", 0)} for d in docs]
    return search


def document_sample(pages, max_chars: int = 6000) -> str:
    """Opening text of the document (spread across pages) for the planner."""
    per_page = max(400, max_chars // max(1, min(len(pages), 15)))
    text = "\n".join(f"[page {p.metadata['page']}] {p.page_content[:per_page]}" for p in pages[:15])
    return text[:max_chars]
