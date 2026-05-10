# Troubleshooting

Common problems and how to fix them.

---

## First-run embedding download is slow

The first time you run the pipeline, fastembed downloads `BAAI/bge-small-en-v1.5` (~130 MB) to `~/.cache/fastembed/`. This is a one-time cost. Subsequent runs load the model from disk in a few seconds.

If the download stalls, check your internet connection or set `HF_ENDPOINT` to a regional mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## `chromadb.errors.UniqueConstraintError` on startup

Happens when two processes try to write to the same ChromaDB directory at the same time, or when a previous run was killed mid-index.

**Fix:** delete the Chroma data directory and re-index:

```bash
rm -rf chroma_data/   # or whatever CHROMA_DIR is set to in .env
python run.py
```

---

## LLM returns `400 context_length_exceeded`

The architecture or data model prompt is hitting the model's context window. This can happen with very large corpora or when RAG retrieves many long chunks.

Options:
- Lower `MAX_TOKENS` in `.env`
- Reduce `TOP_K` in the agent's `_rag_query` call
- Switch to a model with a larger context window (`gpt-4o` instead of `gpt-4o-mini`)

---

## `DiagramAgent` produces no images

The diagram agent calls `mmdc` (Mermaid CLI) as a subprocess. If the images are missing:

1. Check that `mmdc` is installed: `mmdc --version`
2. Install it if missing: `npm install -g @mermaid-js/mermaid-cli`
3. On headless servers, Mermaid needs Chromium. Pass `--no-sandbox` via the `MMDC_ARGS` environment variable, or install `chromium-browser`.

If the agent XML contains invalid Mermaid syntax, the LLM output is logged to the console. You can copy the block into [mermaid.live](https://mermaid.live) to debug the syntax.

---

## Critic score is always 0 or the pipeline exits early

The critic agent extracts a numeric score from its own output using a regex. If the LLM changes the format (e.g., writes "Score: N/10" instead of the expected pattern), extraction fails and the score defaults to 0.

Check `agents/critic_agent.py` → `extract_score()` and confirm the regex matches the actual output format. You can add a `print(raw_critique)` temporarily to inspect.

---

## RAG retrieval returns empty or irrelevant chunks

1. **Collection not indexed** — run `python -c "from rag.vector_store import build_all_collections; build_all_collections()"` to confirm the collections are populated.
2. **Wrong `DOCS_ROOT`** — verify the path in `.env` points to the folder containing `requirements_docs_processed/`, `architecture_docs_processed/`, etc.
3. **Force reindex** — if you updated the corpus, pass `--force-reindex` to `run.py` or set `force_reindex=True` in the agent constructor.

---

## `ModuleNotFoundError: No module named 'config'`

Run all scripts from the repo root, not from a subdirectory:

```bash
cd MultiAgent_Automated_Software_Architecture
python run.py          # correct
python ../run.py       # wrong — config.py won't be on sys.path
```

---

## Tests fail with `ImportError`

The test suite stubs out `fastembed`, `chromadb`, `openai`, and `tenacity` so the tests run without any model downloads. If you add a new import to `agents/base.py` or the chunker, add a corresponding stub in the test file's `_make_stubs()` function.

Run tests from the repo root:

```bash
python -m pytest tests/ -v
```

---

## Streamlit UI shows a blank page

- Make sure you are on Python 3.10+ (`python --version`)
- Confirm Streamlit is installed: `pip install streamlit>=1.35.0`
- Check the terminal for import errors — often a missing `.env` file or an unset `LLM_API_KEY`
