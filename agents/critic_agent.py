import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.vector_store import index_corpus, AGENT_CORPORA
from agents.base import call_llm, rag_block, prior_outputs_block

_SYSTEM = """\
You are a senior security architect and principal technical reviewer.
Your job is to critically evaluate the architecture and data model against security
best practices, architectural patterns, and quality standards.

Using NIST CSF 2.0, OWASP Top 10, STRIDE threat modeling, and architecture patterns
from the retrieved context, perform a comprehensive review.

Produce a structured Architecture Review with these sections:

## 1. OWASP TOP 10 ANALYSIS
For each applicable category: risk level, specific finding, recommended mitigation.

## 2. NIST CSF 2.0 COMPLIANCE
Assess each of the 6 functions (Govern, Identify, Protect, Detect, Respond, Recover):
**FUNCTION NAME — Assessment Level**
- Strengths: what is already addressed
- Gaps: what is missing
- Recommendations: specific actions

## 3. STRIDE THREAT ANALYSIS
Identify concrete threats: Spoofing, Tampering, Repudiation,
Information Disclosure, Denial of Service, Elevation of Privilege.

## 4. RISK ASSESSMENT (NIST SP 800-30)
Top 5 risks — for each: likelihood (1–5), impact (1–5), risk score, recommended control.

## 5. ARCHITECTURE WEAKNESSES
Single points of failure, bottlenecks, missing components, data consistency risks.

## 6. RECOMMENDATIONS
- CRITICAL: must fix before deployment
- HIGH: fix in next iteration
- MEDIUM: address in roadmap
- LOW: nice to have

## 7. OVERALL SCORE
OVERALL SCORE: X/10
Where 1–3=major flaws, 4–6=significant gaps, 7–8=good with minor improvements, 9–10=excellent
Justify the score in 2–3 sentences.

Ensure all markdown tables have proper pipe separators with each row on its own line.
"""


def _rag_query(collection, requirements: str, architecture: str, top_k: int = 5) -> str:
    brief = requirements[:200].replace("\n", " ")
    arch_brief = architecture[:200].replace("\n", " ")

    queries = [
        "security vulnerabilities risk assessment OWASP NIST threat modeling",
        f"OWASP Top 10 mitigations authentication authorization: {brief}",
        f"NIST CSF 2.0 compliance controls architecture: {arch_brief}",
        "STRIDE threat modeling data integrity confidentiality availability",
        "security architecture patterns zero trust defense in depth",
    ]

    def _query(q):
        return collection.query(query_texts=[q], n_results=top_k)

    seen: set[str] = set()
    chunks: list[tuple[str, dict]] = []

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        for result in pool.map(_query, queries):
            for doc, meta, chunk_id in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["ids"][0],
            ):
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    chunks.append((doc, meta))

    parts = []
    for i, (doc, meta) in enumerate(chunks, 1):
        source = Path(meta["source"]).name
        parts.append(f"[Chunk {i} | {source}]\n{doc}")
    return "\n\n".join(parts)


def extract_action_items(critique: str) -> str:
    sec6_match = re.search(
        r"##\s*6\.?\s*RECOMMENDATIONS(.*?)(?=^##\s*7\.|\Z)",
        critique, re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    section = sec6_match.group(1) if sec6_match else critique

    items: list[str] = []
    for level in ("CRITICAL", "HIGH"):
        block_match = re.search(
            rf"-\s*{level}[:\s]*(.*?)(?=-\s*(?:CRITICAL|HIGH|MEDIUM|LOW)|$)",
            section, re.DOTALL | re.IGNORECASE,
        )
        if block_match:
            block = block_match.group(1).strip()
            bullets = [
                line.strip().lstrip("-•* ").strip()
                for line in block.splitlines()
                if line.strip() and re.match(r"^\s*[-•*]", line)
            ]
            if bullets:
                items.append(f"**{level}**")
                items.extend(f"- {b}" for b in bullets)

    if items:
        return "\n".join(items)
    return section.strip()[:800]


def _extract_score(text: str) -> int:
    patterns = [
        r"OVERALL SCORE[:\s]+(\d+)\s*/\s*10",
        r"SCORE[:\s]+(\d+)\s*/\s*10",
        r"(\d+)\s*/\s*10",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    return 5


class CriticAgent:
    COLLECTION = "critic"
    _SYSTEM = _SYSTEM

    def __init__(self, force_reindex: bool = False):
        self._collection = index_corpus(
            AGENT_CORPORA[self.COLLECTION],
            self.COLLECTION,
            force_reindex=force_reindex,
        )

    def _build_prompt(
        self,
        requirements_output: str,
        arch_output: str,
        dm_output: str,
        use_rag: bool = True,
        prior_critique: str = "",
    ) -> str:
        context_block = ""
        if use_rag:
            raw_context = _rag_query(self._collection, requirements_output, arch_output)
            context_block = rag_block(raw_context)

        comparison_block = ""
        if prior_critique:
            comparison_block = textwrap.dedent(f"""
                <round_2_instructions>
                This is ROUND 2. The architecture and data model were revised specifically to
                address the CRITICAL and HIGH findings listed below. Your job now is to:

                1. For each item below, explicitly label it RESOLVED or STILL OPEN.
                2. A RESOLVED critical MUST increase the score — each resolved critical is worth
                   +0.5 to +1.0 points. Be generous when evidence of the fix is present.
                3. Do NOT re-penalize already-resolved findings.
                4. If 3+ criticals are resolved, the score MUST be at least 7/10.
                5. Justify the new score by listing what was fixed vs what remains open.

                Round-1 action items that were sent to the agents for fixing:
                {prior_critique}
                </round_2_instructions>
            """).strip()

        return textwrap.dedent(f"""\
            {context_block}

            {prior_outputs_block({
                "requirements": requirements_output[:1500],
                "architecture": arch_output[:2500],
                "data_model":   dm_output[:2500],
            })}

            {comparison_block}

            Using the retrieved security standards and the agent outputs above, \
produce the comprehensive Architecture Review. Follow the schema exactly.
        """).strip()

    def run(
        self,
        requirements_output: str,
        arch_output: str,
        dm_output: str,
        use_rag: bool = True,
        prior_critique: str = "",
    ) -> tuple[str, int]:
        prompt   = self._build_prompt(requirements_output, arch_output, dm_output, use_rag, prior_critique)
        critique = call_llm(_SYSTEM, prompt, max_tokens=3000)
        return critique, _extract_score(critique)
