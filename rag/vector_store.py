from pathlib import Path

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from fastembed import TextEmbedding
from tqdm import tqdm

from config import cfg
from rag.loaders import load_document, SUPPORTED_EXTENSIONS
from rag.chunker import chunk_text

_embedder = TextEmbedding(cfg.EMBEDDING_MODEL)


class _LocalEmbeddingFn(EmbeddingFunction):
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
    return _chroma.get_or_create_collection(name=name, embedding_function=_embedding_fn)


def index_corpus(
    corpus_dir: Path,
    collection_name: str,
    force_reindex: bool = False,
) -> chromadb.Collection:
    collection = get_collection(collection_name)

    if collection.count() > 0 and not force_reindex:
        print(f"[{collection_name}] Already indexed ({collection.count()} chunks).")
        return collection

    files = sorted(
        p for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    print(f"[{collection_name}] Indexing {len(files)} files ...")

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

    print(f"[{collection_name}] Done — {collection.count()} total chunks.")
    return collection


def build_all_collections(force_reindex: bool = False) -> dict[str, chromadb.Collection]:
    return {
        name: index_corpus(path, name, force_reindex=force_reindex)
        for name, path in AGENT_CORPORA.items()
    }
