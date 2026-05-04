import re
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from config import cfg

MERGE_THRESHOLD = 0.72
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = cfg.CHUNK_SIZE


@lru_cache(maxsize=1)
def _get_embedder() -> TextEmbedding:
    return TextEmbedding(cfg.EMBEDDING_MODEL)


def _embed(texts: list[str]) -> np.ndarray:
    return np.array(list(_get_embedder().embed(texts)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-10))


def _structural_split(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]

    result: list[str] = []
    pending = ""

    for block in raw:
        if re.match(r"^#{1,6}\s", block):
            pending = (pending + "\n\n" + block).strip() if pending else block
        else:
            full = (pending + "\n\n" + block).strip() if pending else block
            result.append(full)
            pending = ""

    if pending:
        if result:
            result[-1] = result[-1] + "\n\n" + pending
        else:
            result.append(pending)

    return result


def _sentence_split(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if current and len(current) + 1 + len(sent) > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]


def chunk_text(
    text: str,
    merge_threshold: float = MERGE_THRESHOLD,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    candidates = _structural_split(text)
    if not candidates:
        return []

    embeddings = _embed(candidates)

    chunks: list[str] = []
    cur_text = candidates[0]
    cur_emb = embeddings[0]

    for i in range(1, len(candidates)):
        para = candidates[i]
        para_emb = embeddings[i]
        sim = _cosine(cur_emb, para_emb)
        merged_size = len(cur_text) + 2 + len(para)

        if sim >= merge_threshold and merged_size <= max_chars:
            cur_text = cur_text + "\n\n" + para
            cur_emb = (cur_emb + para_emb) / 2.0
        else:
            chunks.append(cur_text)
            cur_text = para
            cur_emb = para_emb

    chunks.append(cur_text)

    result: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            result.extend(_sentence_split(chunk, max_chars))
        elif len(chunk) >= min_chars:
            result.append(chunk)
        elif result:
            result[-1] = result[-1] + "\n\n" + chunk

    return [c.strip() for c in result if c.strip()]
