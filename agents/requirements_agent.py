import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.vector_store import index_corpus, AGENT_CORPORA
from agents.base import call_llm, rag_block

_SYSTEM = """\
You are a senior requirements engineer with deep expertise in IEEE 29148, \
EARS (Easy Approach to Requirements Syntax), and ISO 25010 quality characteristics.

Using the standards, templates, and examples from the retrieved context, \
produce a complete and professional Software Requirements Specification (SRS).

Think step by step internally but do NOT show your thinking process.
Output ONLY the final SRS document — no preamble, no step-by-step breakdown.

Begin the document with:
# Software Requirements Specification (SRS)
**Version:** 1.0

Then produce a complete SRS with the following sections:

## 1. PROJECT OVERVIEW
- 1.1 Purpose: what problem this system solves
- 1.2 Scope: what is included and excluded in this release
- 1.3 Intended Users: list each user type with their primary needs

## 2. FUNCTIONAL REQUIREMENTS
Use EARS syntax strictly for every requirement:
  "The <system/actor> shall <action> when/while/if <condition>"
Every FR must follow this exact pattern — no exceptions.
Number each: FR-001, FR-002, ...
Each major feature must have its own dedicated subsection.
Derive implicit requirements the user did not mention but are clearly needed.

## 3. NON-FUNCTIONAL REQUIREMENTS
Cover ALL ISO 25010 quality characteristics:
  Performance, Security, Usability, Reliability, Maintainability, Portability, Scalability
Number each: NFR-001, NFR-002, ...
Make each measurable with specific metrics (numbers, percentages, time limits).
Never use vague terms like "minimal", "fast", "easy", "sufficient".

## 4. EXTERNAL INTERFACES
- 4.1 User Interfaces — high-level description only
- 4.2 Software Interfaces (APIs, third-party integrations)
- 4.3 Hardware Interfaces

## 5. CONSTRAINTS & ASSUMPTIONS
- 5.1 Constraints (regulatory, technical, budget)
- 5.2 Assumptions
"""


def _rag_query(collection, user_input: str, top_k: int = 5) -> str:
    brief = user_input[:300].replace("\n", " ")
    queries = [
        brief,
        f"IEEE 29148 requirements specification standards templates: {brief}",
        f"EARS syntax functional requirements examples: {brief}",
        f"ISO 25010 non-functional quality characteristics: {brief}",
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


class RequirementsAgent:
    COLLECTION = "requirements"
    _SYSTEM = _SYSTEM

    def __init__(self, force_reindex: bool = False):
        self._collection = index_corpus(
            AGENT_CORPORA[self.COLLECTION],
            self.COLLECTION,
            force_reindex=force_reindex,
        )

    def _build_prompt(self, user_input: str, use_rag: bool = True) -> str:
        context_block = ""
        if use_rag:
            raw_context = _rag_query(self._collection, user_input)
            context_block = rag_block(raw_context)
        return textwrap.dedent(f"""\
            {context_block}

            <user_input>
            {user_input}
            </user_input>

            Using the retrieved standards and the user input above, \
produce the complete SRS document. Follow the schema exactly.
        """).strip()

    def run(self, user_input: str, use_rag: bool = True) -> str:
        return call_llm(_SYSTEM, self._build_prompt(user_input, use_rag), max_tokens=3000)
