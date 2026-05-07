"""
Retrieval precision evaluation.

Measures P@K using an LLM binary relevance judge (RELEVANT / IRRELEVANT).
Two query types: in-domain (sent to the correct collection) and hard-negative
(sent to a wrong collection) to test semantic separation between corpora.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from agents.base import call_llm


# ── LLM relevance judge ───────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are evaluating whether a retrieved text chunk is relevant to a search query.
Answer with exactly one word: RELEVANT or IRRELEVANT.
A chunk is RELEVANT if it directly addresses the query topic and would help
answer it. It is IRRELEVANT if it discusses unrelated subjects, even if it
shares some words with the query.
"""


def _judge_chunk(query: str, chunk: str) -> bool:
    """Return True if the LLM considers this chunk relevant to the query."""
    prompt = (
        f"Query: {query}\n\n"
        f"Chunk (first 400 chars): {chunk[:400]}\n\n"
        "Is this chunk relevant to the query? Answer RELEVANT or IRRELEVANT."
    )
    raw = call_llm(_JUDGE_SYSTEM, prompt, max_tokens=5).strip().upper()
    return raw.startswith("RELEVANT")


# ── query bank ────────────────────────────────────────────────────────────────

TEST_QUERIES: list[dict] = [

    # ── requirements — in-domain (6) ─────────────────────────────────────────
    {
        "collection": "requirements", "kind": "in_domain",
        "description": "EARS conditional pattern",
        "query": (
            "Show me the EARS pattern template for writing a functional requirement "
            "that activates under a specific condition, including the when/while/if clause."
        ),
    },
    {
        "collection": "requirements", "kind": "in_domain",
        "description": "IEEE 29148 SRS structure",
        "query": (
            "What sections and subsections does the IEEE 29148 standard mandate "
            "for a Software Requirements Specification document?"
        ),
    },
    {
        "collection": "requirements", "kind": "in_domain",
        "description": "Measurable NFR examples",
        "query": (
            "Give examples of non-functional requirements written with specific, "
            "measurable acceptance criteria including numbers, percentages, and time limits."
        ),
    },
    {
        "collection": "requirements", "kind": "in_domain",
        "description": "ISO 25010 quality model",
        "query": (
            "What quality characteristics does ISO 25010 define for software product "
            "quality and how should they be covered in an SRS?"
        ),
    },
    {
        "collection": "requirements", "kind": "in_domain",
        "description": "Stakeholder and user roles",
        "query": (
            "How should an SRS document identify and describe user roles, personas, "
            "and their primary interaction goals with the system?"
        ),
    },
    {
        "collection": "requirements", "kind": "in_domain",
        "description": "Requirement traceability",
        "query": (
            "What is requirements traceability and how do numbering schemes like FR-001 "
            "support forward and backward traceability across design documents?"
        ),
    },

    # ── architecture — in-domain (6) ─────────────────────────────────────────
    {
        "collection": "architecture", "kind": "in_domain",
        "description": "Microservices decomposition",
        "query": (
            "What principles guide the decomposition of a monolith into microservices "
            "and what are the key trade-offs compared to a modular monolith?"
        ),
    },
    {
        "collection": "architecture", "kind": "in_domain",
        "description": "AWS reliability pillar",
        "query": (
            "How does the AWS Well-Architected Framework define the reliability pillar "
            "and what design patterns does it recommend for fault tolerance?"
        ),
    },
    {
        "collection": "architecture", "kind": "in_domain",
        "description": "Event-driven vs request-response",
        "query": (
            "When should an architect choose event-driven asynchronous messaging over "
            "synchronous request-response and what are the consistency trade-offs?"
        ),
    },
    {
        "collection": "architecture", "kind": "in_domain",
        "description": "CQRS pattern",
        "query": (
            "Explain the CQRS pattern — how it separates read and write models and "
            "in which scenarios does the added complexity pay off?"
        ),
    },
    {
        "collection": "architecture", "kind": "in_domain",
        "description": "Horizontal scalability design",
        "query": (
            "What architectural decisions enable horizontal scalability in a cloud-native "
            "application, including stateless services and load balancing strategies?"
        ),
    },
    {
        "collection": "architecture", "kind": "in_domain",
        "description": "API gateway responsibilities",
        "query": (
            "What responsibilities does an API gateway handle in a microservices system "
            "and how does it differ from a reverse proxy or a service mesh?"
        ),
    },

    # ── data_modeler — in-domain (6) ─────────────────────────────────────────
    {
        "collection": "data_modeler", "kind": "in_domain",
        "description": "Relational vs document DB",
        "query": (
            "Under what data access patterns and consistency requirements should a "
            "designer choose MongoDB over PostgreSQL or vice versa?"
        ),
    },
    {
        "collection": "data_modeler", "kind": "in_domain",
        "description": "3NF normalization",
        "query": (
            "What is third normal form in relational database design and what types "
            "of update and deletion anomalies does it eliminate?"
        ),
    },
    {
        "collection": "data_modeler", "kind": "in_domain",
        "description": "Indexing strategies",
        "query": (
            "What indexing strategies — covering indexes, composite indexes, partial "
            "indexes — improve read query performance for high-traffic workloads?"
        ),
    },
    {
        "collection": "data_modeler", "kind": "in_domain",
        "description": "CAP theorem implications",
        "query": (
            "How does the CAP theorem constrain distributed database design decisions "
            "and what does it mean in practice to favour AP over CP?"
        ),
    },
    {
        "collection": "data_modeler", "kind": "in_domain",
        "description": "Embedding vs referencing",
        "query": (
            "In a document database like MongoDB, when should you embed related data "
            "inside a single document versus storing it as a separate reference?"
        ),
    },
    {
        "collection": "data_modeler", "kind": "in_domain",
        "description": "Multi-tenant schema patterns",
        "query": (
            "What are the three main schema strategies for multi-tenant SaaS — shared "
            "schema, shared database, isolated database — and their trade-offs?"
        ),
    },

    # ── critic — in-domain (6) ────────────────────────────────────────────────
    {
        "collection": "critic", "kind": "in_domain",
        "description": "OWASP Top 10 overview",
        "query": (
            "What are the OWASP Top 10 most critical web application security risks "
            "and what category tops the current edition of the list?"
        ),
    },
    {
        "collection": "critic", "kind": "in_domain",
        "description": "NIST CSF Protect function",
        "query": (
            "What controls and subcategories fall under the Protect function in NIST "
            "Cybersecurity Framework 2.0 and how should organisations implement them?"
        ),
    },
    {
        "collection": "critic", "kind": "in_domain",
        "description": "STRIDE threat modeling",
        "query": (
            "Walk me through applying STRIDE threat modeling to an API endpoint — "
            "what threat does each letter represent and what control addresses it?"
        ),
    },
    {
        "collection": "critic", "kind": "in_domain",
        "description": "Broken authentication mitigations",
        "query": (
            "What mitigations does OWASP recommend for broken authentication, "
            "including session management, MFA, and secure credential storage?"
        ),
    },
    {
        "collection": "critic", "kind": "in_domain",
        "description": "NIST SP 800-30 risk scoring",
        "query": (
            "How does NIST SP 800-30 define likelihood and impact in a risk assessment "
            "and how is the overall risk level derived from those two values?"
        ),
    },
    {
        "collection": "critic", "kind": "in_domain",
        "description": "Defense-in-depth layers",
        "query": (
            "What is the defense-in-depth security principle and what layers — "
            "perimeter, network, host, application, data — should be present in a cloud system?"
        ),
    },

    # ── hard negatives (8): wrong collection, expect LOW precision ────────────
    {
        "collection": "requirements", "kind": "hard_neg",
        "description": "HN: Kafka partitioning → req",
        "query": (
            "How does Apache Kafka handle message partitioning, replication factor, "
            "and consumer group offset management for high-throughput event streaming?"
        ),
    },
    {
        "collection": "requirements", "kind": "hard_neg",
        "description": "HN: SQL injection → req",
        "query": (
            "Explain SQL injection attack vectors and the parameterised query approach "
            "that prevents them in a REST API backend."
        ),
    },
    {
        "collection": "architecture", "kind": "hard_neg",
        "description": "HN: 3NF steps → arch",
        "query": (
            "Describe the step-by-step process to normalise a relational schema to "
            "third normal form and list the functional dependencies that must be removed."
        ),
    },
    {
        "collection": "architecture", "kind": "hard_neg",
        "description": "HN: STRIDE controls → arch",
        "query": (
            "List the six STRIDE threat categories and describe the specific security "
            "control that mitigates each one in a web application."
        ),
    },
    {
        "collection": "data_modeler", "kind": "hard_neg",
        "description": "HN: NIST CSF functions → dm",
        "query": (
            "What are the six core functions of NIST CSF 2.0 — Govern, Identify, "
            "Protect, Detect, Respond, Recover — and what does each one cover?"
        ),
    },
    {
        "collection": "data_modeler", "kind": "hard_neg",
        "description": "HN: microservice comms → dm",
        "query": (
            "What communication patterns — synchronous REST, async events, gRPC — "
            "should microservices use and under what latency and coupling requirements?"
        ),
    },
    {
        "collection": "critic", "kind": "hard_neg",
        "description": "HN: EARS syntax → critic",
        "query": (
            "What is the EARS syntax template for writing a functional requirement "
            "that specifies system behaviour triggered by a user event?"
        ),
    },
    {
        "collection": "critic", "kind": "hard_neg",
        "description": "HN: CAP theorem → critic",
        "query": (
            "Explain how the CAP theorem forces a trade-off between consistency and "
            "availability during a network partition in a distributed database."
        ),
    },
]


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    description:      str
    collection:       str
    kind:             str    # "in_domain" or "hard_neg"
    k:                int
    precision:        float
    chunks_retrieved: int
    relevant_chunks:  int


# ── core evaluation ───────────────────────────────────────────────────────────

def _evaluate_query(
    collection,
    query:       str,
    k:           int,
    description: str,
    coll_name:   str,
    kind:        str,
) -> QueryResult:
    result = collection.query(query_texts=[query], n_results=k)
    docs   = result["documents"][0]

    # Judge all chunks for this query in parallel
    relevance: list[bool] = [False] * len(docs)
    with ThreadPoolExecutor(max_workers=len(docs)) as pool:
        futures = {
            pool.submit(_judge_chunk, query, doc): i
            for i, doc in enumerate(docs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                relevance[idx] = future.result()
            except Exception:
                relevance[idx] = False

    relevant = sum(relevance)
    return QueryResult(
        description      = description,
        collection       = coll_name,
        kind             = kind,
        k                = k,
        precision        = round(relevant / len(docs), 4) if docs else 0.0,
        chunks_retrieved = len(docs),
        relevant_chunks  = relevant,
    )


def run_retrieval_eval(
    collections: dict,
    k_values:    list[int] = [3, 5],
) -> dict[int, list[QueryResult]]:
    """
    Run retrieval precision evaluation for all queries and k values.

    Parameters
    ----------
    collections : dict mapping collection_name -> chromadb Collection object
    k_values    : list of k values to evaluate

    Returns
    -------
    dict[k -> list[QueryResult]] — results include both in_domain and hard_neg rows.
    Use mean_precision(..., kind="in_domain") and mean_precision(..., kind="hard_neg")
    to report them separately.
    """
    results: dict[int, list[QueryResult]] = {k: [] for k in k_values}
    total = len([q for q in TEST_QUERIES if q["collection"] in collections]) * len(k_values)
    done  = 0

    for k in k_values:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(
                    _evaluate_query,
                    collections[tq["collection"]],
                    tq["query"],
                    k,
                    tq["description"],
                    tq["collection"],
                    tq["kind"],
                ): tq
                for tq in TEST_QUERIES
                if tq["collection"] in collections
            }
            for future in as_completed(futures):
                try:
                    results[k].append(future.result())
                except Exception as e:
                    tq = futures[future]
                    print(f"\n  [retrieval] error on '{tq['description']}': {e}")
                done += 1
                print(f"  retrieval: {done}/{total} queries judged", end="\r", flush=True)

    print()
    return results


# ── aggregation helpers ───────────────────────────────────────────────────────

def mean_precision(
    results: dict[int, list[QueryResult]],
    k:       int,
    kind:    str | None = None,
) -> float:
    rows = results.get(k, [])
    if kind:
        rows = [r for r in rows if r.kind == kind]
    return round(sum(r.precision for r in rows) / len(rows), 4) if rows else 0.0


def per_collection_mean(
    results: dict[int, list[QueryResult]],
    k:       int,
    kind:    str | None = None,
) -> dict[str, float]:
    rows = results.get(k, [])
    if kind:
        rows = [r for r in rows if r.kind == kind]
    by_coll: dict[str, list[float]] = {}
    for r in rows:
        by_coll.setdefault(r.collection, []).append(r.precision)
    return {c: round(sum(v) / len(v), 4) for c, v in by_coll.items()}
