import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.vector_store import index_corpus, AGENT_CORPORA
from agents.base import call_llm, rag_block, prior_outputs_block

_SYSTEM = """\
You are a senior software architect. Given structured requirements and retrieved expert \
context, produce a comprehensive, concrete architecture document.

Rules:
- Cite retrieved sources for every significant decision.
- Be specific: name real services, protocols, and patterns — never vague placeholders.
- Every component must have a clear owner responsibility and a justified technology choice.
- Design decisions must explain WHY, not just WHAT was chosen.

Output MUST be valid XML matching this schema exactly — no prose outside the tags:

<architecture_document>

  <overview>
    High-level description of the system, the chosen architectural style \
(e.g. event-driven microservices, layered monolith, CQRS+ES), and the two or three \
guiding principles that shape every decision below.
  </overview>

  <components>
    <component>
      <name>Service or system name</name>
      <responsibility>Single-sentence description of what it owns</responsibility>
      <technology>Specific technology and the one-line justification for choosing it</technology>
    </component>
  </components>

  <data_flow>
    Numbered, step-by-step narrative of how a request or event travels end-to-end \
through the system — from the entry point to persistence and any async side-effects.
  </data_flow>

  <design_decisions>
    <decision>
      <title>Short name of the decision</title>
      <chosen>The option selected</chosen>
      <rationale>Why, grounded in retrieved patterns or stated NFRs</rationale>
      <trade_offs>What is given up or made harder by this choice</trade_offs>
    </decision>
  </design_decisions>

  <nfr_strategy>
    One paragraph each on how the architecture addresses: scalability, reliability, \
security, performance, and cost efficiency.
  </nfr_strategy>

</architecture_document>
"""


def _multi_query_rag(collection, requirements: str, top_k: int = 4) -> str:
    brief = requirements[:300].replace("\n", " ")
    queries = [
        brief,
        f"architectural patterns scalability reliability for: {brief}",
        f"security compliance data protection architecture: {brief}",
        f"service integration event-driven data flow patterns: {brief}",
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


class ArchitectureAgent:
    COLLECTION = "architecture"

    def __init__(self, force_reindex: bool = False):
        self._collection = index_corpus(
            AGENT_CORPORA[self.COLLECTION],
            self.COLLECTION,
            force_reindex=force_reindex,
        )

    def run(
        self,
        requirements_output: str,
        use_rag: bool = True,
        prior_critique: str = "",
    ) -> str:
        context_block = ""
        if use_rag:
            raw_context = _multi_query_rag(self._collection, requirements_output)
            context_block = rag_block(raw_context)

        critique_block = ""
        if prior_critique:
            critique_block = (
                "\n\n<revision_instructions>\n"
                "This is ROUND 2. The previous architecture scored below 7/10. "
                "You MUST implement ALL of the following improvements. "
                "Do not omit any — they are non-negotiable:\n\n"
                f"{prior_critique}\n\n"
                "For each CRITICAL item: add a dedicated <component> or <decision> that explicitly addresses it. "
                "For each HIGH item: reflect it in the <nfr_strategy> or relevant <component>. "
                "The revised architecture must be meaningfully more secure and robust than round 1."
                "\n</revision_instructions>"
            )

        user_prompt = textwrap.dedent(f"""\
            {context_block}

            {prior_outputs_block({"requirements": requirements_output})}
            {critique_block}

            Using the retrieved architectural patterns and the requirements above, \
produce the architecture document. Follow the XML schema exactly.
        """).strip()

        return call_llm(_SYSTEM, user_prompt)
