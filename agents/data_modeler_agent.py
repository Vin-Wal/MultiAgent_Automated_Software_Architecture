import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.vector_store import index_corpus, AGENT_CORPORA
from agents.base import call_llm, rag_block, prior_outputs_block

_SYSTEM = """\
You are a senior data architect. Given structured requirements and an architecture document, \
produce a precise, implementable data persistence strategy grounded in retrieved best practices.

Rules:
- Read the architecture document carefully — use the exact storage technologies it specifies.
- Every schema must have correct data types, PRIMARY KEY, NOT NULL, and FOREIGN KEY constraints.
- Every table must have an INDEXES section listing each index with its purpose.
- The ER diagram must reflect the actual entities and relationships, not generic placeholders.

Output MUST be valid XML matching this schema exactly:

<data_model_document>

  <storage_strategy>
    For each storage technology used (RDBMS, document store, cache, event store, object store):
    - What data lives there and why
    - Read/write access pattern it serves
    - Justification from retrieved best practices
  </storage_strategy>

  <er_diagram>
    ```mermaid
    erDiagram
        ENTITY_A {
            type field_name PK
            type field_name
        }
        ENTITY_A ||--o{ ENTITY_B : "relationship label"
    ```
  </er_diagram>

  <schemas>
    <schema>
      <store>Name of the storage technology (e.g. PostgreSQL, Redis, Cassandra)</store>
      <entities>
        <entity>
          <name>Table or collection name</name>
          <ddl>
            Full CREATE TABLE / collection schema definition with all constraints
          </ddl>
          <indexes>
            CREATE INDEX statements or index descriptions with purpose noted
          </indexes>
        </entity>
      </entities>
    </schema>
  </schemas>

  <normalization_notes>
    Normal form achieved for each relational schema.
    Explicit rationale for any intentional denormalization.
  </normalization_notes>

  <trade_offs>
    CAP theorem positioning for each store.
    Scaling approach (vertical vs horizontal, sharding strategy if applicable).
    Known limitations and how they are mitigated.
  </trade_offs>

</data_model_document>
"""


def _extract_technologies(arch_xml: str) -> list[str]:
    matches = re.findall(r"<technology>([^<]+)</technology>", arch_xml)
    techs = []
    for m in matches:
        first = re.split(r"[,\-–]", m)[0].strip()
        if first:
            techs.append(first)
    return list(dict.fromkeys(techs))


def _multi_query_rag(collection, requirements: str, arch_xml: str, top_k: int = 4) -> str:
    brief_req = requirements[:250].replace("\n", " ")
    techs = _extract_technologies(arch_xml)
    tech_str = ", ".join(techs[:5]) if techs else "relational and NoSQL databases"

    queries = [
        f"schema design data modeling best practices for {tech_str}",
        f"entity relationships normalization indexing for: {brief_req}",
        f"database indexing query optimization strategies {tech_str}",
        f"caching event sourcing multi-tenancy data patterns: {brief_req}",
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


class DataModelerAgent:
    COLLECTION = "data_modeler"

    def __init__(self, force_reindex: bool = False):
        self._collection = index_corpus(
            AGENT_CORPORA[self.COLLECTION],
            self.COLLECTION,
            force_reindex=force_reindex,
        )

    def run(
        self,
        requirements_output: str,
        arch_output: str,
        use_rag: bool = True,
        prior_critique: str = "",
    ) -> str:
        context_block = ""
        if use_rag:
            raw_context = _multi_query_rag(
                self._collection, requirements_output, arch_output
            )
            context_block = rag_block(raw_context)

        critique_block = ""
        if prior_critique:
            critique_block = (
                "\n\n<revision_instructions>\n"
                "This is ROUND 2. The previous data model scored below 7/10. "
                "You MUST implement ALL of the following data-layer improvements. "
                "Do not omit any — they are non-negotiable:\n\n"
                f"{prior_critique}\n\n"
                "For each CRITICAL item: add encryption, access controls, or schema constraints directly in the DDL. "
                "For each HIGH item: add indexes, audit columns, or additional schema notes. "
                "The revised data model must be meaningfully more secure than round 1."
                "\n</revision_instructions>"
            )

        user_prompt = textwrap.dedent(f"""\
            {context_block}

            {prior_outputs_block({
                "requirements": requirements_output,
                "architecture": arch_output,
            })}
            {critique_block}

            Using the retrieved database best practices and the prior agent outputs above, \
produce the data model document. Follow the XML schema exactly.
        """).strip()

        return call_llm(_SYSTEM, user_prompt)
