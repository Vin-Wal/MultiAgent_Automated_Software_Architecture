import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.vector_store import index_corpus, AGENT_CORPORA
from agents.base import call_llm, rag_block, prior_outputs_block

_SYSTEM = """\
You are a senior data architect. Given structured requirements and an architecture document, \
produce the BEST persistence strategy — not the most complex one.

CRITICAL RULE — justify every database:
- Start with the simplest possible solution (often a single well-chosen database).
- Only add a second database if there is a concrete, unavoidable reason that the primary
  store cannot handle the workload (e.g., adding Redis ONLY if sub-millisecond cache is
  truly required; adding a vector DB ONLY if semantic search is a core feature).
- NEVER add a database "because it's common" or "it could help". Each store must earn
  its place with a specific bottleneck or capability gap it solves.
- If the architecture document names specific technologies, use those EXACT names.
- 1-3 stores is normal. More than 4 is unusual and requires strong justification in
  each <why_chosen> field.

When a store IS justified, model it in its native format:
  * Relational (PostgreSQL, MySQL): full SQL CREATE TABLE with PK, NOT NULL, FK + CREATE INDEX
  * Document (MongoDB, Firestore, DynamoDB): JSON Schema or collection definition
  * Graph (Neo4j, Neptune): Cypher node/relationship definitions + UNIQUE constraints
  * Vector (Pinecone, Weaviate, Qdrant, pgvector): collection config — dimension, metric, metadata
  * Key-value / Cache (Redis, Memcached): key naming patterns, value types, TTL policy
  * Event / Stream (Kafka, Kinesis, Pulsar): topic definition with event payload schema
  * Object store (S3, GCS, Azure Blob): bucket layout with key prefixes and object schema

- Include <er_diagram> ONLY when at least one relational schema is present. Omit entirely otherwise.
- The ER diagram must reflect actual relational entities only — no placeholders.

Output MUST be valid XML matching this schema exactly:

<data_model_document>

  <storage_strategy>
    <strategy>
      <store>Technology name (e.g. PostgreSQL)</store>
      <data_lives>What specific data is stored here</data_lives>
      <why_chosen>Why this technology was selected over alternatives</why_chosen>
      <access_patterns>The read/write patterns this store serves</access_patterns>
    </strategy>
    <!-- one <strategy> block per storage technology — ALL are used simultaneously, each for a different concern -->
  </storage_strategy>

  <er_diagram>
    <!-- Include ONLY if relational schemas are present -->
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
      <store>Technology name (e.g. PostgreSQL, MongoDB, Neo4j, Redis, Pinecone, Kafka)</store>
      <type>relational | document | graph | vector | keyvalue | eventstream | objectstore</type>
      <entities>
        <entity>
          <name>Table / collection / node-label / topic / index name</name>
          <ddl>
            Schema definition in the appropriate format for this technology.
            SQL for relational, JSON Schema for document, Cypher for graph,
            JSON config for vector, key-pattern spec for cache, Avro/JSON for events.
          </ddl>
          <indexes>
            Index or access-pattern definitions with their purpose noted.
          </indexes>
        </entity>
      </entities>
    </schema>
  </schemas>

  <normalization_notes>
    Relational: normal form achieved and any intentional denormalization rationale.
    Non-relational: embedding vs referencing decisions, sharding keys, partition strategy.
  </normalization_notes>

  <trade_offs>
    CAP theorem position for each store.
    Scaling approach and known limitations with mitigations.
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
        f"schema design best practices {tech_str}",
        f"database selection NoSQL graph vector relational tradeoffs: {brief_req}",
        f"data modeling normalization indexing access patterns {tech_str}",
        f"CQRS event sourcing multi-tenancy graph vector cache patterns: {brief_req}",
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

        return textwrap.dedent(f"""\
            {context_block}

            {prior_outputs_block({
                "requirements": requirements_output,
                "architecture": arch_output,
            })}
            {critique_block}

            Using the retrieved database best practices and the prior agent outputs above, \
produce the data model document. Follow the XML schema exactly.
        """).strip()

    def run(
        self,
        requirements_output: str,
        arch_output: str,
        use_rag: bool = True,
        prior_critique: str = "",
    ) -> str:
        return call_llm(
            _SYSTEM,
            self._build_prompt(requirements_output, arch_output, use_rag, prior_critique),
            max_tokens=4000,
        )
