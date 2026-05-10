# Multi-Agent Automated Software Architecture

Takes a plain-text system description and produces a complete architecture package: SRS, architecture document, data model, security critique, and diagrams.

## What it does

Five specialized agents run in sequence. A critic scores the output and, if the score is below 7/10, the architecture and data model agents run a second time with the critique injected as context.

```
Requirements → Architecture → Data Modeler → Critic
                                                │
                                         score ≥ 7 ──→ Diagrams → output/
                                         score < 7 ──→ round 2  → Diagrams → output/
```

Each agent pulls relevant chunks from a local vector store (ChromaDB + BAAI/bge-small-en-v1.5) so its output is grounded in real standards — IEEE 29148, EARS syntax, AWS Well-Architected, NIST CSF 2.0, OWASP Top 10.

## Output

Each run writes a timestamped folder under `output/`:

```
output/20260503_120000/
  srs.md               — Software Requirements Specification
  architecture.xml     — Architecture document with design decisions
  data_model.xml       — Data model with DDL schemas
  critique.md          — Security and architecture review
  diagrams/
    architecture.png   — Component diagram
    sequence.png       — Sequence diagram
    er.png             — Entity-relationship diagram
  DEVELOPER_BRIEF.md   — Developer handoff document
```

## Requirements

- Python 3.10+
- Node.js 16+ with `mmdc` for diagram rendering: `npm install -g @mermaid-js/mermaid-cli`
- An OpenAI API key (or any OpenAI-compatible endpoint)

## Setup

```bash
git clone https://github.com/Vin-Wal/MultiAgent_Automated_Software_Architecture.git
cd MultiAgent_Automated_Software_Architecture

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env — set LLM_API_KEY and DOCS_ROOT at minimum
```

`.env` variables:

| Variable | Description |
|---|---|
| `LLM_API_KEY` | OpenAI API key |
| `LLM_BASE_URL` | API base URL (default: `https://api.openai.com/v1`) |
| `LLM_MODEL` | Model name (default: `gpt-4o-mini`) |
| `DOCS_ROOT` | Absolute path to the corpus document folder |

## Running

```bash
# Default scenario
.venv/bin/python run.py

# Custom scenario
.venv/bin/python run.py --scenario "Build a healthcare records system with HIPAA compliance"

# Skip RAG (no vector store lookup)
.venv/bin/python run.py --no-rag

# Force corpus re-index after updating documents
.venv/bin/python run.py --force-reindex
```

## Streamlit UI

```bash
.venv/bin/streamlit run app.py
```

Tabs show each pipeline output. The sidebar lets you re-run with a modified prompt without re-indexing.

## Corpus

Four ChromaDB collections, one per agent:

| Collection | Folder | Contents |
|---|---|---|
| `requirements` | `requirements_docs_processed/` | IEEE 29148, EARS syntax, SRS templates |
| `architecture` | `architecture_docs_processed/` | AWS Well-Architected, microservices patterns |
| `data_modeler` | `datamodeler_docs/corpus/` | Database design, indexing, CAP theorem |
| `critic` | `critic_docs/` | NIST CSF 2.0, OWASP Top 10, STRIDE |

To process raw PDFs into the corpus, run `process_docs.py` first (requires Python 3.12 with `docling`).

Collections are indexed once and reused on subsequent runs. Delete the `chroma_data/` directory if you need to start fresh.

## Evaluation

The `eval/` module measures output quality:

**Automated (no LLM cost)**
- EARS compliance rate
- NFR measurability
- Section completeness
- Decision quality (rationale + trade-offs present)

**LLM-judged**
- Retrieval precision P@K
- Faithfulness (claim verification against the corpus)
- Answer relevance (user brief features covered in SRS)
- LLM-as-judge rubric (5 dimensions, max 12 points)

```bash
python -m eval.run_eval --scenarios 80

# Use cached pipeline outputs, skip re-running agents
python -m eval.run_eval --scenarios 80 --skip-pipeline

# Retrieval metrics only
python -m eval.run_eval --retrieval-only
```

Results and plots go to `eval_output/`.

## Results (n=20)

| Metric | RAG | No-RAG | Monolithic |
|---|---|---|---|
| Structural (automated) | 0.91 | 0.93 | 0.36 |
| Faithfulness | 0.42 | 0.38 | — |
| Answer relevance | 0.58 | 0.57 | — |
| LLM judge (/12) | 11.1 | 11.1 | 11.3 |

Multi-agent vs monolithic on structural metrics: 0.91 vs 0.36. The LLM judge scores all three conditions similarly because it evaluates prose quality — not format compliance or diagram presence, which is where the monolithic baseline falls short.

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover the semantic chunker and structural metrics. They run without any model downloads (fastembed and chromadb are stubbed).

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common errors.
