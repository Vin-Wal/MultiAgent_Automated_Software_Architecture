"""
Answer relevance metric, adapted from RAGAS (Es et al., 2023).

Extracts key features from the user brief, then checks what fraction are
explicitly addressed in the generated SRS. Score = addressed / total.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.base import call_llm

MAX_FEATURES = 15

_EXTRACT_SYSTEM = """\
You are analysing a user brief for a software system.
Extract up to 15 specific features or constraints that the system must satisfy.

Each feature must be:
- Exactly one short phrase or sentence
- Specific (a number, a standard name, a named capability)
- A positive requirement — what the system MUST do or support

Output ONLY a numbered list, one feature per line, no headings or commentary.

Example output:
1. Support 500,000 concurrent users
2. PCI-DSS compliance for payment processing
3. Real-time GPS location tracking with sub-second updates
4. Surge pricing algorithm adjusting rates based on demand
"""

_VERIFY_SYSTEM = """\
For each numbered feature and the SRS document below, answer YES if the feature
is explicitly addressed or clearly implied in the SRS, or NO if it is absent.

Output ONLY a numbered list of YES or NO — one per line, no explanations.

Example output:
1. YES
2. NO
3. YES
"""


@dataclass
class AnswerRelevanceResult:
    scenario_label:     str
    total_features:     int   = 0
    addressed_features: int   = 0
    score:              float = 0.0
    features:           list[str]  = field(default_factory=list)
    verdicts:           list[bool] = field(default_factory=list)
    error:              str | None = None

    def to_dict(self) -> dict:
        return {
            "label":              self.scenario_label,
            "total_features":     self.total_features,
            "addressed_features": self.addressed_features,
            "score":              self.score,
        }


def _extract_features(prompt: str) -> list[str]:
    raw = call_llm(_EXTRACT_SYSTEM, f"User brief:\n{prompt[:2000]}", max_tokens=500)
    features = []
    for line in raw.splitlines():
        m = re.match(r"^\d+[\.\)]\s*(.+)", line.strip())
        if m and len(m.group(1).strip()) > 5:
            features.append(m.group(1).strip())
    return features[:MAX_FEATURES]


def _verify_coverage(features: list[str], srs_text: str) -> list[bool]:
    if not features:
        return []
    blocks = "\n".join(f"{i}. {f}" for i, f in enumerate(features, 1))
    prompt = f"Features to verify:\n{blocks}\n\nSRS document (first 3000 chars):\n{srs_text[:3000]}"
    raw = call_llm(_VERIFY_SYSTEM, prompt, max_tokens=len(features) * 10)

    verdicts: list[bool] = []
    for line in raw.splitlines():
        m = re.match(r"^\d+[\.\):]?\s*(YES|NO)", line.strip(), re.IGNORECASE)
        if m:
            verdicts.append(m.group(1).upper() == "YES")

    while len(verdicts) < len(features):
        verdicts.append(False)
    return verdicts[:len(features)]


def compute_answer_relevance(
    user_prompt:    str,
    srs_text:       str,
    scenario_label: str = "scenario",
) -> AnswerRelevanceResult:
    """
    Parameters
    ----------
    user_prompt    : the original user brief / system description
    srs_text       : the generated SRS / requirements document
    scenario_label : short name used in reports
    """
    try:
        features = _extract_features(user_prompt)
    except Exception as e:
        return AnswerRelevanceResult(scenario_label, error=str(e))

    if not features:
        return AnswerRelevanceResult(scenario_label, error="no features extracted")

    try:
        verdicts = _verify_coverage(features, srs_text)
    except Exception as e:
        return AnswerRelevanceResult(
            scenario_label, total_features=len(features), error=str(e)
        )

    addressed = sum(verdicts)
    return AnswerRelevanceResult(
        scenario_label     = scenario_label,
        total_features     = len(features),
        addressed_features = addressed,
        score              = round(addressed / len(features), 4),
        features           = features,
        verdicts           = verdicts,
    )


def mean_answer_relevance(results: list[AnswerRelevanceResult]) -> float:
    scores = [r.score for r in results if r.error is None and r.total_features > 0]
    return round(sum(scores) / len(scores), 4) if scores else 0.0
