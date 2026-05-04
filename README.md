# Multi-Agent Automated Software Architecture

A pipeline that takes a plain-text system description and produces a full architecture package: SRS, architecture document, data model, security critique, diagrams, and a developer brief.

## How it works

Five agents run in sequence with a 2-round critic feedback loop:

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
- An OpenAI API key (or any OpenAI-compatible API)

## Setup

```bash
git clone https://github.com/yassinejebbouri/MultiAgent_Automated_Software_Architecture.git
cd MultiAgent_Automated_Software_Architecture

# Create virtual environment and install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

Edit `.env` and set:
- `LLM_API_KEY` — your OpenAI API key
- `DOCS_ROOT` — absolute path to the folder containing the corpus documents

## Running

```bash
# Default scenario (ride-sharing platform)
.venv/bin/python run.py

# Custom scenario
.venv/bin/python run.py --scenario "Build a healthcare records system with HIPAA compliance"

# Without RAG (faster, less grounded)
.venv/bin/python run.py --no-rag

# Compare RAG vs no-RAG side by side
.venv/bin/python run.py --compare
```

## Corpus documents

The RAG pipeline reads from four corpora under `DOCS_ROOT`:

| Collection | Folder | Contents |
|---|---|---|
| requirements | `requirements_docs_processed/` | IEEE 29148, EARS syntax, SRS templates |
| architecture | `architecture_docs_processed/` | AWS Well-Architected, microservices patterns |
| data_modeler | `datamodeler_docs/corpus/` | Database design, indexing, CAP theorem |
| critic | `critic_docs/` | NIST CSF 2.0, OWASP Top 10, STRIDE |

If you have raw PDFs to process, run `process_docs.py` first (requires Python 3.12 with docling).

## Configuration

All settings can be overridden via environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | API key (required) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `MAX_TOKENS` | `8192` | Max tokens per LLM call |
| `DOCS_ROOT` | `../genai_final_project/agent_docs` | Path to corpus documents |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (fastembed) |
