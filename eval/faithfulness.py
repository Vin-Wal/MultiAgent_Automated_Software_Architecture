"""
Faithfulness metric, adapted from RAGAS (Es et al., 2023).

Extracts atomic claims from a generated document, retrieves supporting chunks
from the expert corpus, then verifies each claim with a batched LLM call.
Score = supported_claims / total_claims.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from agents.base import call_llm

MAX_CLAIMS = 15
VERIFY_K   = 5

# ── prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are analysing a technical document. Extract up to 15 atomic, factual claims.

Each claim must be:
- Exactly one sentence
- Specific (names a real technology, pattern, metric, or design decision)
- A positive assertion — not "the system avoids X" or "no X is used"

Output ONLY a numbered list, one claim per line, no headings or commentary.

Example output:
1. The system uses PostgreSQL for storing transactional trip and payment records.
2. Kafka is used as the event streaming platform for real-time GPS location updates.
3. Redis provides session caching with a 15-minute TTL.
"""

_VERIFY_SYSTEM = """\
For each numbered claim and its associated retrieved context below, answer YES if
the claim is directly supported or can be reasonably inferred from the context,
or NO if the context does not support it.

Output ONLY a numbered list of YES or NO answers — one per line, no explanations.

Example output:
1. YES
2. NO
3. YES
"""


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class FaithfulnessResult:
    output_label:     str           # e.g. "architecture", "srs"
    total_claims:     int   = 0
    supported_claims: int   = 0
    score:            float = 0.0
    claims:           list[str]  = field(default_factory=list)
    verdicts:         list[bool] = field(default_factory=list)
    error:            str | None = None

    def to_dict(self) -> dict:
        return {
            "label":            self.output_label,
            "total_claims":     self.total_claims,
            "supported_claims": self.supported_claims,
            "score":            self.score,
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_claims(text: str) -> list[str]:
    raw = call_llm(_EXTRACT_SYSTEM, f"Document:\n{text[:3000]}", max_tokens=600)
    claims = []
    for line in raw.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if m and len(m.group(1).strip()) > 15:
            claims.append(m.group(1).strip())
    return claims[:MAX_CLAIMS]


def _retrieve_context(collection, claim: str) -> str:
    result = collection.query(query_texts=[claim], n_results=VERIFY_K)
    docs   = result["documents"][0]
    return "\n---\n".join(d[:350] for d in docs)


def _verify_batch(claims: list[str], contexts: list[str]) -> list[bool]:
    if not claims:
        return []
    blocks = [
        f"Claim {i}: {c}\nContext {i}:\n{ctx}"
        for i, (c, ctx) in enumerate(zip(claims, contexts), 1)
    ]
    raw = call_llm(_VERIFY_SYSTEM, "\n\n".join(blocks), max_tokens=len(claims) * 12)

    verdicts: list[bool] = []
    for line in raw.splitlines():
        m = re.match(r"^\d+[\.\):]?\s*(YES|NO)", line.strip(), re.IGNORECASE)
        if m:
            verdicts.append(m.group(1).upper() == "YES")

    while len(verdicts) < len(claims):
        verdicts.append(False)
    return verdicts[:len(claims)]


# ── public API ────────────────────────────────────────────────────────────────

def compute_faithfulness(
    output_text:  str,
    collection,           # chromadb Collection for the agent that produced the output
    output_label: str = "output",
) -> FaithfulnessResult:
    """
    Parameters
    ----------
    output_text  : the generated document (architecture XML, SRS markdown, etc.)
    collection   : the corpus collection to verify claims against
    output_label : short name used in reports ("architecture", "srs", "data_model")
    """
    try:
        claims = _extract_claims(output_text)
    except Exception as e:
        return FaithfulnessResult(output_label, error=str(e))

    if not claims:
        return FaithfulnessResult(output_label, error="no claims extracted")

    # Retrieve context for each claim in parallel
    contexts = [""] * len(claims)
    with ThreadPoolExecutor(max_workers=min(len(claims), 8)) as pool:
        futures = {
            pool.submit(_retrieve_context, collection, claim): i
            for i, claim in enumerate(claims)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                contexts[i] = future.result()
            except Exception:
                contexts[i] = ""

    try:
        verdicts = _verify_batch(claims, contexts)
    except Exception as e:
        return FaithfulnessResult(output_label, total_claims=len(claims), error=str(e))

    supported = sum(verdicts)
    return FaithfulnessResult(
        output_label     = output_label,
        total_claims     = len(claims),
        supported_claims = supported,
        score            = round(supported / len(claims), 4),
        claims           = claims,
        verdicts         = verdicts,
    )


def compute_pipeline_faithfulness(
    architecture: str,
    srs:          str,
    data_model:   str,
    collections:  dict,
) -> dict[str, FaithfulnessResult]:
    """
    Compute faithfulness for all three main pipeline outputs.
    Returns dict keyed by output label.
    """
    tasks = {
        "architecture": (architecture, "architecture"),
        "srs":          (srs,          "requirements"),
        "data_model":   (data_model,   "data_modeler"),
    }
    results: dict[str, FaithfulnessResult] = {}
    for label, (text, coll_name) in tasks.items():
        if coll_name in collections:
            results[label] = compute_faithfulness(text, collections[coll_name], label)
    return results


def mean_faithfulness(results: dict[str, FaithfulnessResult]) -> float:
    scores = [r.score for r in results.values() if r.error is None and r.total_claims > 0]
    return round(sum(scores) / len(scores), 4) if scores else 0.0
