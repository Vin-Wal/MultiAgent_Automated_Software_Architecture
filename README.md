# Multi-Agent Automated Software Architecture

A pipeline that turns a plain-text system description into a full architecture package: SRS, architecture document, data model, security critique, and diagrams.

## How it works

Five agents run in sequence with a critic feedback loop:

```
Requirements → Architecture → Data Modeler → Critic
                                                ↓
                                         score ≥ 7 → Diagram → Developer Brief
                                         score < 7 → round 2 → Diagram → Developer Brief
```

Each agent retrieves relevant chunks from a local knowledge base (RAG) to ground its output in established standards and patterns.

## Output

Every run produces a timestamped folder under `output/`:

```
output/20260503_120000/
  srs.md               — Software Requirements Specification
  architecture.xml     — Architecture document
  data_model.xml       — Data model with DDL schemas
  critique.md          — Security and architecture review
  diagrams/
    architecture.png   — Component diagram
    sequence.png       — Sequence diagram
    er.png             — Entity-relationship diagram
  DEVELOPER_BRIEF.md   — Full developer brief
```

## Prerequisites

- Python 3.10+
- Node.js 16+ with mmdc: `npm install -g @mermaid-js/mermaid-cli`
- An OpenAI API key (or any OpenAI-compatible endpoint)

## Setup

```bash
git clone https://github.com/Vin-Wal/MultiAgent_Automated_Software_Architecture.git
cd MultiAgent_Automated_Software_Architecture

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `LLM_API_KEY` | Your OpenAI API key |
| `LLM_BASE_URL` | API base URL (default: `https://api.openai.com/v1`) |
| `LLM_MODEL` | Model name (default: `gpt-4o-mini`) |
| `DOCS_ROOT` | Absolute path to the corpus document folder |

## Running

```bash
# Default scenario (ride-sharing platform)
.venv/bin/python run.py

# Custom scenario
.venv/bin/python run.py --scenario "Build a healthcare records system with HIPAA compliance"

# Without RAG
.venv/bin/python run.py --no-rag
```

## Corpus

The RAG pipeline reads from four collections under `DOCS_ROOT`:

| Collection | Folder | Contents |
|---|---|---|
| `requirements` | `requirements_docs_processed/` | IEEE 29148, EARS syntax, SRS templates |
| `architecture` | `architecture_docs_processed/` | AWS Well-Architected, microservices patterns |
| `data_modeler` | `datamodeler_docs/corpus/` | Database design, indexing, CAP theorem |
| `critic` | `critic_docs/` | NIST CSF 2.0, OWASP Top 10, STRIDE |

To process raw PDFs, run `process_docs.py` first (requires Python 3.12 with docling).

## Streamlit UI

```bash
.venv/bin/streamlit run app.py
```

Tabs for each pipeline output, inline diagrams, and a sidebar re-prompt for iterating on results.

## Evaluation

The `eval/` module measures output quality across five dimensions:

**Automated (no LLM cost)**
- EARS compliance rate
- NFR measurability
- Section completeness
- Decision quality
- Diagram presence

**LLM-judged**
- Retrieval precision P@K (in-domain vs hard-negative queries)
- Faithfulness (RAGAS-style claim verification against the corpus)
- Answer relevance (fraction of user brief features addressed in SRS)
- LLM-as-judge (5-dimension rubric, max 12 points, blind scoring)

```bash
# Full suite on 80 scenarios
python -m eval.run_eval --scenarios 80

# Skip pipeline re-runs, use cached outputs
python -m eval.run_eval --scenarios 80 --skip-pipeline

# Retrieval evaluation only
python -m eval.run_eval --retrieval-only
```

Results and plots are written to `eval_output/`.

## Key findings (n=20)

| Metric | RAG | No-RAG | Monolithic |
|---|---|---|---|
| Structural (automated) | 0.91 | 0.93 | 0.36 |
| Faithfulness | 0.42 | 0.38 | — |
| Answer relevance | 0.58 | 0.57 | — |
| LLM judge (/12) | 11.1 | 11.1 | 11.3 |

Multi-agent vs monolithic: 0.91 vs 0.36 on structural metrics. The LLM judge scores all three conditions similarly because it evaluates prose quality, not format compliance or diagram presence — the dimensions where monolithic fails.
