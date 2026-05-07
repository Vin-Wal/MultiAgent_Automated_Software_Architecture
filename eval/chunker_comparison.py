"""
Chunker comparison: Semantic chunker vs fixed-size chunker.

What it tests
-------------
The pipeline uses a custom semantic chunker (rag/chunker.py) that merges
paragraphs by cosine similarity (threshold 0.72) rather than splitting at
a fixed character count.  This module runs a controlled experiment:

  1. Index the same four corpora using a naive fixed-size chunker
     (800 chars, no semantic merging) into separate "_fixed" ChromaDB
     collections.
  2. Run the same 24 in-domain queries from eval/retrieval.py against
     both sets of collections.
  3. Judge each chunk with the same LLM relevance judge.
  4. Compare P@5 between semantic and fixed-size chunkers.

Expected result
---------------
Semantic chunker P@5 > fixed-size P@5 — because semantic chunks respect
topic boundaries and keep related sentences together, reducing the chance
that the relevant answer is split across multiple chunks.

If the gap is small or reversed, we learn that the simpler baseline is
sufficient for this corpus — still a valid finding.

Cost estimate
-------------
Same 24 queries × 5 chunks × ~1 LLM call each ≈ 120 calls ≈ $0.01
"""
from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import chromadb

from agents.base import call_llm
from rag.vector_store import AGENT_CORPORA, build_all_collections
from eval.retrieval import TEST_QUERIES, _judge_chunk, QueryResult

FIXED_CHUNK_SIZE = 800

_FIXED_CLIENT: chromadb.Client | None = None


def _get_fixed_client() -> chromadb.Client:
    global _FIXED_CLIENT
    if _FIXED_CLIENT is None:
        _FIXED_CLIENT = chromadb.Client()
    return _FIXED_CLIENT


def _chunk_fixed(text: str, size: int = FIXED_CHUNK_SIZE) -> list[str]:
    """Simple fixed-size character chunker — no semantic merging."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            for sep in (". ", ".\n", "\n\n", "\n"):
                pos = text.rfind(sep, start, end)
                if pos > start + size // 2:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def build_fixed_collections() -> dict:
    """
    Index all four corpora using the fixed-size chunker.
    Returns a dict mapping collection_name -> chromadb Collection.
    Uses an in-memory client (separate from the semantic collections).
    """
    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    client = chromadb.Client()
    collections = {}

    for coll_name, corpus_path in AGENT_CORPORA.items():
        path = Path(corpus_path)
        if not path.exists():
            continue

        coll = client.get_or_create_collection(f"{coll_name}_fixed")
        existing = coll.count()
        if existing > 0:
            collections[coll_name] = coll
            continue

        texts = [f.read_text(encoding="utf-8", errors="replace")
                 for f in sorted(path.glob("**/*.txt"))
                 if f.is_file()]
        if not texts:
            texts = [f.read_text(encoding="utf-8", errors="replace")
                     for f in sorted(path.glob("**/*.md"))
                     if f.is_file()]

        chunks: list[str] = []
        for t in texts:
            chunks.extend(_chunk_fixed(t))

        if not chunks:
            continue

        ids  = [f"{coll_name}_fixed_{i}" for i in range(len(chunks))]
        embs = list(model.embed(chunks, batch_size=64))
        emb_lists = [e.tolist() for e in embs]

        for i in range(0, len(chunks), 500):
            coll.add(
                documents  = chunks[i:i+500],
                embeddings = emb_lists[i:i+500],
                ids        = ids[i:i+500],
            )

        collections[coll_name] = coll
        print(f"  [fixed] indexed {len(chunks)} chunks for '{coll_name}'")

    return collections


@dataclass
class ChunkerComparisonResult:
    collection:        str
    query_description: str
    k:                 int
    semantic_precision: float
    fixed_precision:    float

    @property
    def delta(self) -> float:
        return round(self.semantic_precision - self.fixed_precision, 4)


def _evaluate_single(collection, query: str, k: int) -> float:
    """Return LLM-judged P@k for one (collection, query) pair."""
    result = collection.query(query_texts=[query], n_results=k)
    docs = result["documents"][0]
    if not docs:
        return 0.0
    relevance = [False] * len(docs)
    with ThreadPoolExecutor(max_workers=len(docs)) as pool:
        futures = {pool.submit(_judge_chunk, query, doc): i for i, doc in enumerate(docs)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                relevance[idx] = future.result()
            except Exception:
                relevance[idx] = False
    return round(sum(relevance) / len(docs), 4)


def run_chunker_comparison(
    semantic_collections: dict,
    k: int = 5,
) -> list[ChunkerComparisonResult]:
    """
    Run the 24 in-domain queries against both chunker types and return
    a result per query.
    """
    fixed_collections = build_fixed_collections()
    in_domain_queries = [q for q in TEST_QUERIES if q["kind"] == "in_domain"]

    results: list[ChunkerComparisonResult] = []
    total = len([q for q in in_domain_queries if q["collection"] in semantic_collections])
    done  = 0

    for q in in_domain_queries:
        coll_name = q["collection"]
        if coll_name not in semantic_collections:
            continue

        sem_prec   = _evaluate_single(semantic_collections[coll_name], q["query"], k)
        fixed_prec = _evaluate_single(fixed_collections.get(coll_name,
                                        semantic_collections[coll_name]), q["query"], k)

        results.append(ChunkerComparisonResult(
            collection         = coll_name,
            query_description  = q["description"],
            k                  = k,
            semantic_precision = sem_prec,
            fixed_precision    = fixed_prec,
        ))
        done += 1
        print(f"  chunker comparison: {done}/{total}", end="\r", flush=True)

    print()
    return results


def mean_precision_by_chunker(
    results: list[ChunkerComparisonResult],
) -> tuple[float, float]:
    """Returns (mean_semantic_P@k, mean_fixed_P@k) across all results."""
    if not results:
        return 0.0, 0.0
    sem   = sum(r.semantic_precision for r in results) / len(results)
    fixed = sum(r.fixed_precision    for r in results) / len(results)
    return round(sem, 4), round(fixed, 4)
