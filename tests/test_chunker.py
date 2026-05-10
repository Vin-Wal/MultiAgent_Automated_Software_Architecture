"""
Unit tests for rag/chunker.py.

Tests cover the three chunking passes independently as well as the full
chunk_text pipeline. Embedding calls are patched out so tests run without
a GPU or a downloaded model.
"""
import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Minimal stubs so chunker.py can be imported without loading fastembed / cfg
# ---------------------------------------------------------------------------

def _make_stubs():
    # stub config
    config_mod = types.ModuleType("config")
    config_mod.cfg = MagicMock(
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        CHUNK_SIZE=800,
    )
    sys.modules.setdefault("config", config_mod)

    # stub fastembed
    fe_mod = types.ModuleType("fastembed")
    fe_mod.TextEmbedding = MagicMock()
    sys.modules.setdefault("fastembed", fe_mod)


_make_stubs()

from rag.chunker import (  # noqa: E402 — must come after stubs
    _structural_split,
    _sentence_split,
    chunk_text,
    MERGE_THRESHOLD,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
)


class TestStructuralSplit(unittest.TestCase):
    """Pass 1: paragraph splitting."""

    def test_basic_split(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        parts = _structural_split(text)
        self.assertEqual(len(parts), 3)

    def test_heading_attached_to_next_block(self):
        text = "## Section Title\n\nThis is the body text."
        parts = _structural_split(text)
        # heading should be merged into the block that follows it
        self.assertEqual(len(parts), 1)
        self.assertIn("Section Title", parts[0])
        self.assertIn("body text", parts[0])

    def test_empty_input(self):
        self.assertEqual(_structural_split(""), [])
        self.assertEqual(_structural_split("   \n\n   "), [])

    def test_single_paragraph(self):
        text = "Just one paragraph with no blank lines."
        parts = _structural_split(text)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], text)

    def test_windows_line_endings_normalised(self):
        text = "Para one.\r\n\r\nPara two."
        parts = _structural_split(text)
        self.assertEqual(len(parts), 2)


class TestSentenceSplit(unittest.TestCase):
    """Pass 3: overflow splitting at sentence boundaries."""

    def test_splits_long_text(self):
        # 5 sentences, each ~50 chars — max_chars=100 should produce splits
        text = "Sentence one here. Sentence two here. Sentence three. Sentence four here. Sentence five here."
        parts = _sentence_split(text, max_chars=100)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), 150)  # allow one long sentence

    def test_short_text_unchanged(self):
        text = "Short sentence."
        parts = _sentence_split(text, max_chars=800)
        self.assertEqual(parts, [text])

    def test_no_sentences_falls_back(self):
        # no sentence-ending punctuation
        text = "x" * 200
        parts = _sentence_split(text, max_chars=800)
        self.assertEqual(len(parts), 1)


class TestChunkText(unittest.TestCase):
    """Full pipeline with embedding mocked."""

    def _run_with_embeddings(self, paragraphs: list[str], sim_matrix: np.ndarray) -> list[str]:
        """
        Run chunk_text with controlled cosine similarities.

        sim_matrix[i] is the embedding for paragraph i.
        Setting them to unit vectors makes cosine = dot product.
        """
        with patch("rag.chunker._embed") as mock_embed:
            mock_embed.return_value = sim_matrix
            return chunk_text("\n\n".join(paragraphs))

    def test_similar_paragraphs_merged(self):
        paras = ["Short para A.", "Short para B."]
        # identical embeddings → cosine = 1.0 → should merge
        embs = np.array([[1.0, 0.0], [1.0, 0.0]])
        result = self._run_with_embeddings(paras, embs)
        self.assertEqual(len(result), 1)
        self.assertIn("Short para A", result[0])
        self.assertIn("Short para B", result[0])

    def test_dissimilar_paragraphs_not_merged(self):
        paras = ["Short para A.", "Short para B."]
        # orthogonal embeddings → cosine = 0.0 → should not merge
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = self._run_with_embeddings(paras, embs)
        self.assertEqual(len(result), 2)

    def test_overflow_chunk_split(self):
        # one very long paragraph that exceeds MAX_CHUNK_CHARS
        long_para = ("This is a sentence. " * 60).strip()
        embs = np.array([[1.0, 0.0]])
        with patch("rag.chunker._embed") as mock_embed:
            mock_embed.return_value = embs
            result = chunk_text(long_para)
        for chunk in result:
            self.assertLessEqual(len(chunk), MAX_CHUNK_CHARS + 50)

    def test_empty_input_returns_empty(self):
        with patch("rag.chunker._embed") as mock_embed:
            mock_embed.return_value = np.array([]).reshape(0, 2)
            result = chunk_text("")
        self.assertEqual(result, [])

    def test_min_chunk_size_respected(self):
        # very short paragraph should be absorbed, not returned alone
        paras = ["Tiny.", "A much longer paragraph that actually has content worth keeping."]
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = self._run_with_embeddings(paras, embs)
        for chunk in result:
            self.assertGreaterEqual(len(chunk), MIN_CHUNK_CHARS)


class TestChunkingConstants(unittest.TestCase):
    def test_threshold_in_range(self):
        self.assertGreater(MERGE_THRESHOLD, 0.0)
        self.assertLess(MERGE_THRESHOLD, 1.0)

    def test_min_less_than_max(self):
        self.assertLess(MIN_CHUNK_CHARS, MAX_CHUNK_CHARS)


if __name__ == "__main__":
    unittest.main()
