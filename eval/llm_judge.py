"""
LLM-as-judge evaluation.

Scores each pipeline output against a structured rubric (5 dimensions, max 12).
RAG context markers are stripped before scoring so the judge is blind to the mode.
"""
import json
import re

from agents.base import call_llm


# ── rubric ────────────────────────────────────────────────────────────────────

RUBRIC: list[dict] = [
    {
        "id":          "completeness",
        "name":        "Component Completeness",
        "description": (
            "Every feature mentioned in the requirements maps to at least one named "
            "architectural component or data entity. Nothing is silently dropped."
        ),
        "max_score": 3,
    },
    {
        "id":          "specificity",
        "name":        "Specificity & Concreteness",
        "description": (
            "Technology choices name real, specific tools (e.g. PostgreSQL, Kafka, Redis) "
            "with a one-line justification. No vague placeholders like 'a database' or "
            "'some caching layer'."
        ),
        "max_score": 3,
    },
    {
        "id":          "consistency",
        "name":        "Cross-Agent Consistency",
        "description": (
            "The data model entities align with the architectural components. "
            "No component is named in the architecture but absent from the data model, "
            "and vice versa. No contradictions between outputs."
        ),
        "max_score": 2,
    },
    {
        "id":          "security",
        "name":        "Security Coverage",
        "description": (
            "The architecture and data model explicitly address authentication, "
            "authorisation, encryption at rest and in transit, and at least two "
            "OWASP Top 10 risks. Generic statements like 'use HTTPS' without detail "
            "do not count."
        ),
        "max_score": 2,
    },
    {
        "id":          "nfr_traceability",
        "name":        "NFR Traceability",
        "description": (
            "Each non-functional requirement from the SRS (performance, reliability, "
            "scalability, compliance, etc.) has a concrete, measurable implementation "
            "strategy in the architecture or data model."
        ),
        "max_score": 2,
    },
]

MAX_SCORE: int = sum(d["max_score"] for d in RUBRIC)   # 12


# ── system prompt ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a senior software architect performing a blind evaluation of an automated
architecture pipeline output. You do NOT know whether retrieval-augmented generation
was used — evaluate solely on the content quality.

Score each rubric dimension independently. Be critical: only award full marks when
the evidence is clear and specific in the text. Partial marks for partial evidence.

Return ONLY a valid JSON object with exactly this structure — no markdown fences,
no prose, no extra keys:

{
  "scores": {
    "completeness":      <integer 0-3>,
    "specificity":       <integer 0-3>,
    "consistency":       <integer 0-2>,
    "security":          <integer 0-2>,
    "nfr_traceability":  <integer 0-2>
  },
  "justifications": {
    "completeness":      "<one sentence>",
    "specificity":       "<one sentence>",
    "consistency":       "<one sentence>",
    "security":          "<one sentence>",
    "nfr_traceability":  "<one sentence>"
  }
}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _blind(text: str) -> str:
    """Strip RAG context markers so the judge cannot identify the mode."""
    text = re.sub(r"<retrieved_context>.*?</retrieved_context>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[Chunk \d+ \|[^\]]+\]", "", text)
    return text.strip()


def _build_prompt(requirements: str, architecture: str, data_model: str) -> str:
    rubric_lines = "\n".join(
        f"  {d['name']} (0–{d['max_score']}): {d['description']}"
        for d in RUBRIC
    )
    return f"""\
## Rubric
{rubric_lines}

## Requirements (truncated to 1 500 chars)
{_blind(requirements)[:1500]}

## Architecture (truncated to 2 000 chars)
{_blind(architecture)[:2000]}

## Data Model (truncated to 1 500 chars)
{_blind(data_model)[:1500]}

Score the outputs using the rubric. Output ONLY the JSON object described above.
"""


def _parse_scores(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract just the JSON blob
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ── public API ────────────────────────────────────────────────────────────────

def judge_output(
    requirements: str,
    architecture: str,
    data_model:   str,
) -> dict:
    """
    Score one pipeline run against the rubric.

    Returns a dict with keys:
      scores          – {dimension_id: int}
      justifications  – {dimension_id: str}
      total           – int (sum of scores)
      max             – int (MAX_SCORE = 12)
      error           – str | None  (set if JSON parsing failed)
    """
    prompt = _build_prompt(requirements, architecture, data_model)
    raw    = call_llm(_JUDGE_SYSTEM, prompt, max_tokens=800)
    parsed = _parse_scores(raw)

    if parsed is None:
        return {
            "scores":         {d["id"]: 0 for d in RUBRIC},
            "justifications": {d["id"]: "parse error" for d in RUBRIC},
            "total": 0,
            "max":   MAX_SCORE,
            "error": raw[:300],
        }

    scores = parsed.get("scores", {})
    # Clamp to valid range
    for d in RUBRIC:
        scores[d["id"]] = max(0, min(d["max_score"], int(scores.get(d["id"], 0))))

    return {
        "scores":         scores,
        "justifications": parsed.get("justifications", {}),
        "total":          sum(scores[d["id"]] for d in RUBRIC),
        "max":            MAX_SCORE,
        "error":          None,
    }
