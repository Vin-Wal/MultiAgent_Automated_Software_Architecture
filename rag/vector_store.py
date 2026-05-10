"""
ChromaDB vector store management for the four agent corpora.

Each agent has a dedicated ChromaDB collection. Collections are populated
once and reused on subsequent runs (``index_corpus`` skips already-indexed
collections unless ``force_reindex=True``).

Collections:
    requirements  — IEEE 29148, EARS syntax, SRS templates
    architecture  — AWS Well-Architected, microservices patterns
    data_modeler  — database design, indexing strategies, CAP theorem
    critic        — NIST CSF 2.0, OWASP Top 10, STRIDE
"""
import logging
from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from fastembed import TextEmbedding
from tqdm import tqdm

from config import cfg
from rag.loaders import load_document, SUPPORTED_EXTENSIONS
from rag.chunker import chunk_text

log = logging.getLogger(__name__)

_embedder = TextEmbedding(cfg.EMBEDDING_MODEL)


class _LocalEmbeddingFn(EmbeddingFunction):
    """ChromaDB-compatible wrapper around the local fastembed model."""

    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in _embedder.embed(list(input))]


_chroma = chromadb.PersistentClient(path=cfg.CHROMA_DIR)
_embedding_fn = _LocalEmbeddingFn()

AGENT_CORPORA: dict[str, Path] = {
    "architecture": cfg.DOCS_ROOT / "architecture_docs_processed",
    "data_modeler": cfg.DOCS_ROOT / "datamodeler_docs" / "corpus",
    "requirements": cfg.DOCS_ROOT / "requirements_docs_processed",
    "critic":       cfg.DOCS_ROOT / "critic_docs",
}


def get_collection(name: str) -> chromadb.Collection:
    """Return (or create) a ChromaDB collection by name."""
    return _chroma.get_or_create_collection(name=name, embedding_function=_embedding_fn)


def index_corpus(
    corpus_dir: Path,
    collection_name: str,
    force_reindex: bool = False,
) -> chromadb.Collection:
    """
    Index all supported documents in ``corpus_dir`` into a ChromaDB collection.

    Skips indexing if the collection is non-empty and ``force_reindex`` is
    ``False``. Documents are chunked with :func:`rag.chunker.chunk_text` and
    upserted in batches of 500 to avoid memory pressure.

    Args:
        corpus_dir: Directory containing source documents (searched recursively).
        collection_name: Name of the ChromaDB collection to populate.
        force_reindex: If ``True``, re-index even when the collection already
            contains documents.

    Returns:
        The populated ChromaDB collection.
    """
    collection = get_collection(collection_name)

    if collection.count() > 0 and not force_reindex:
        log.info("[%s] Already indexed (%d chunks).", collection_name, collection.count())
        return collection

    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    log.info("[%s] Indexing %d files ...", collection_name, len(files))

    batch_docs, batch_ids, batch_metas = [], [], []
    doc_counter = 0

    for path in tqdm(files, desc=collection_name):
        text = load_document(path)
        if not text or not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            batch_docs.append(chunk)
            batch_ids.append(f"{path.stem}__{doc_counter}__{i}")
            batch_metas.append({
                "source": str(path.relative_to(corpus_dir)),
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        doc_counter += 1

        if len(batch_docs) >= 500:
            collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
            batch_docs, batch_ids, batch_metas = [], [], []

    if batch_docs:
        collection.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)

    log.info("[%s] Done — %d total chunks.", collection_name, collection.count())
    return collection


def build_all_collections(force_reindex: bool = False) -> dict[str, chromadb.Collection]:
    """
    Index all four agent corpora and return a dict of collections.

    Args:
        force_reindex: Passed through to :func:`index_corpus`.

    Returns:
        Mapping from collection name to ChromaDB collection object.
    """
    return {
        name: index_corpus(path, name, force_reindex=force_reindex)
        for name, path in AGENT_CORPORA.items()
    }
