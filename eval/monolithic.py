"""
Single-prompt monolithic baseline.

Produces all four artefacts (SRS, architecture, data model, security review)
in one LLM call with no RAG and no critique loop. Used to isolate the effect
of pipeline structure on output quality.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.base import call_llm

_MONOLITHIC_SYSTEM = """\
You are a senior software architect. Given a user brief, produce a complete
software design package in exactly four sections separated by === SECTION ===.

Section order and format:

=== SECTION: REQUIREMENTS ===
<SRS in Markdown with sections: Overview, Functional Requirements (EARS syntax),
Non-Functional Requirements (measurable), Constraints, Stakeholders>

=== SECTION: ARCHITECTURE ===
<XML architecture document with root <architecture>, containing:
  <components> with each <component name="..." type="..."><responsibilities/></component>
  <decisions> with each <decision id="..." title="..."><rationale/><tradeoffs/></decision>
  <nfr_strategy> addressing performance, security, scalability
  <data_flow> describing how data moves between components>

=== SECTION: DATA_MODEL ===
<XML data model with root <data_model>, containing:
  <store name="..." type="..." technology="...">
    <entities> with <entity name="..."><fields/></entity>
    <justification/>
  </store> for each data store>

=== SECTION: SECURITY_REVIEW ===
<Security critique in Markdown covering OWASP Top 10 risks, STRIDE threats,
mitigations, and a risk rating (LOW/MEDIUM/HIGH/CRITICAL) per component.>

Be specific. Use real technology names. Be thorough but concise.
"""

_SEP = re.compile(r"===\s*SECTION[:\s]+(\w+)\s*===", re.IGNORECASE)


@dataclass
class MonolithicResult:
    requirements: str = ""
    architecture: str = ""
    data_model:   str = ""
    critique:     str = ""
    raw:          str = ""
    error:        str | None = None


def run_monolithic(user_prompt: str) -> MonolithicResult:
    """
    Run the single-agent monolithic baseline for one scenario.

    Returns a MonolithicResult whose fields match PipelineResult's fields
    so the same structural metrics and judge can be applied.
    """
    try:
        raw = call_llm(_MONOLITHIC_SYSTEM, user_prompt, max_tokens=4096)
    except Exception as e:
        return MonolithicResult(error=str(e))

    # Parse sections
    sections: dict[str, str] = {}
    parts = _SEP.split(raw)
    # parts is [pre-text, key1, body1, key2, body2, ...]
    it = iter(parts)
    next(it)  # skip pre-text
    try:
        while True:
            key  = next(it).strip().upper()
            body = next(it).strip()
            sections[key] = body
    except StopIteration:
        pass

    return MonolithicResult(
        requirements = sections.get("REQUIREMENTS", ""),
        architecture = sections.get("ARCHITECTURE", ""),
        data_model   = sections.get("DATA_MODEL",   ""),
        critique     = sections.get("SECURITY_REVIEW", ""),
        raw          = raw,
    )
