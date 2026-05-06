import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base import call_llm

# ── system prompts with verified few-shot examples ─────────────────────────────

_ARCH_SYSTEM = """\
You are a software diagramming expert. Convert an architecture document into a \
Mermaid flowchart.

Rules — follow exactly, violations cause parse errors:
- Output ONLY a fenced Mermaid block — no prose.
- Use graph TD.
- Node IDs: alphanumeric only, no spaces or special chars (e.g. ApiGw, AuthSvc).
- Node labels: always double-quoted rectangles — ApiGw["API Gateway"].
- Arrow syntax: --> (sync), -.-> (async), ==> (persistence write).
- Arrow labels: pipe syntax with double quotes — A -->|"label"| B.
  ONLY add a label when it adds meaning. NEVER write -->|""| — empty labels cause parse errors.
- Subgraph names: always double-quoted — subgraph "Back-End".
- NEVER use the word "end" as a node ID or label — it breaks the parser.
- NEVER start a node ID with lowercase "o" or "x" — reserved edge markers.
- NEVER use %% comments.
- Keep to 20 nodes or fewer.

EXAMPLE of correct output:
```mermaid
graph TD
  subgraph "Client"
    Browser["Browser"]
  end
  subgraph "Back-End"
    ApiGw["API Gateway"]
    AuthSvc["Auth Service"]
    OrderSvc["Order Service"]
  end
  subgraph "Storage"
    PgDb[("PostgreSQL")]
    Cache[("Redis")]
  end
  Browser -->|"HTTPS"| ApiGw
  ApiGw -->|"verify token"| AuthSvc
  ApiGw -->|"route"| OrderSvc
  OrderSvc ==> PgDb
  OrderSvc -.-> Cache
```

Respond with exactly:
```mermaid
graph TD
  ...
```
"""

_SEQ_SYSTEM = """\
You are a software diagramming expert. Convert an architecture document's data flow \
into a Mermaid sequence diagram.

Rules — follow exactly, violations cause parse errors:
- Output ONLY a fenced Mermaid block — no prose.
- Use sequenceDiagram.
- PARTICIPANT ORDER IS CRITICAL: declare participants left-to-right in the order they
  appear in the flow. The end-user or client MUST be declared FIRST (leftmost).
  Internal services follow in the order they are called. Backend stores last.
  Example order: User → API Gateway → Auth Service → Order Service → Database
- Declare every participant: participant A as "Label"
- Valid arrows (ONLY these four):
    ->>   solid arrowhead — synchronous call
    -->>  dotted arrowhead — response / async message
    -x    solid with X — fire-and-forget (no response expected)
    --x   dotted with X — lost/dropped message
- NEVER use -x>>, ->, -->, --, or any other variant.
  In particular "--" with no ">" is INVALID and will crash the parser.
- Message text: plain words and spaces ONLY.
  NEVER put < > ( ) [ ] { } # ; in message text — they break the parser.
- NEVER use "end" as a participant name — it is a reserved keyword.
- activate/deactivate around long-running operations.
- 20 messages maximum.

EXAMPLE of correct output (User declared first, services follow in call order):
```mermaid
sequenceDiagram
  participant U as "User"
  participant G as "API Gateway"
  participant A as "Auth Service"
  participant D as "Database"
  U ->> G: POST login
  activate G
  G ->> A: verify credentials
  activate A
  A -->> G: token issued
  deactivate A
  G -->> U: 200 token
  deactivate G
  U ->> G: GET orders
  G ->> D: query orders
  D -->> G: rows
  G -->> U: 200 orders
```

Respond with exactly:
```mermaid
sequenceDiagram
  ...
```
"""

_FIX_SYSTEM = """\
You are a Mermaid diagram syntax expert. The diagram below failed to render with the \
error shown. Fix ONLY the syntax — preserve all content and structure.
Common fixes needed:
- Replace empty pipe labels -->|""| with plain arrows -->
- Replace -x>> with -x
- Remove < > ( ) [ ] { } from sequence message text
- Rename any node ID that is the reserved word "end"
- Ensure erDiagram closing } is on its own line

Output ONLY the corrected fenced Mermaid block, no prose.
"""

_ER_FALLBACK_SYSTEM = """\
You are a software diagramming expert. Convert a data model document into a \
Mermaid entity-relationship diagram.

Rules — follow exactly, violations cause parse errors:
- Output ONLY a fenced Mermaid block — no prose.
- Use erDiagram.
- Relationship syntax: EntityA ||--o{ EntityB : "label"
  Valid cardinality: |o  ||  }o  }|   Line types: -- (solid)  .. (dotted)
- Entity block:
    ENTITY_NAME {
        type fieldName PK
        type fieldName FK
        type fieldName
    }
  Field types must start with a letter. NO SQL keywords (NOT NULL, UNIQUE, DEFAULT…).
- Names: alphanumeric and underscores ONLY.
- Closing } MUST be on its own line — never on the same line as a relationship.
- Only include relational SQL entities.

EXAMPLE of correct output:
```mermaid
erDiagram
  USER {
    uuid id PK
    string email
    string name
    timestamp created_at
  }
  ORDER {
    uuid id PK
    uuid user_id FK
    string status
    numeric total
  }
  USER ||--o{ ORDER : "places"
```

Respond with exactly:
```mermaid
erDiagram
  ...
```
"""

_GRAPH_SCHEMA_SYSTEM = """\
You are a software diagramming expert. Visualise the graph database schema as a \
Mermaid flowchart showing node labels and relationship types.

Rules — follow exactly:
- Output ONLY a fenced Mermaid block — no prose.
- Use flowchart LR.
- Node labels: stadium shape — NodeId(["NodeType\\nkey: type"])
- Relationships: NodeA -->|"REL_TYPE"| NodeB
- Node IDs: alphanumeric only. Never start with lowercase o or x.
- Labels: always double-quoted. Never include ( ) [ ] inside a label.
- NEVER write -->|""| — omit the pipe entirely if no label is needed.
- NEVER use "end" as a node ID.
- Max 15 nodes.

EXAMPLE of correct output:
```mermaid
flowchart LR
  User(["User\\nid: uuid, email: string"])
  Post(["Post\\nid: uuid, title: string"])
  Tag(["Tag\\nid: uuid, name: string"])
  User -->|"WROTE"| Post
  User -->|"FOLLOWS"| User
  Post -->|"HAS_TAG"| Tag
```

Respond with exactly:
```mermaid
flowchart LR
  ...
```
"""

_DOCUMENT_SCHEMA_SYSTEM = """\
You are a software diagramming expert. Visualise the document database schema as a \
Mermaid class diagram showing collections and their fields.

Rules — follow exactly:
- Output ONLY a fenced Mermaid block — no prose.
- Use classDiagram.
- Class names: alphanumeric and underscores ONLY — no spaces, hyphens, or special chars.
- Field syntax: +type fieldName  (e.g. +String email, +ObjectId userId)
- Use stereotypes <<Document>>, <<Embedded>>, <<Reference>> where helpful.
- Relationships:
    CollA --> CollB : "references"
    CollA --* CollB : "embeds"
- Cardinality in quotes: CollA "1" --> "0..*" CollB : "has"
- Max 8 classes.

EXAMPLE of correct output:
```mermaid
classDiagram
  class User {
    <<Document>>
    +ObjectId _id
    +String email
    +String name
    +Array addresses
  }
  class Order {
    <<Document>>
    +ObjectId _id
    +ObjectId userId
    +Array items
    +Number total
  }
  class Address {
    <<Embedded>>
    +String street
    +String city
  }
  User "1" --> "0..*" Order : "has"
  User --* Address : "embeds"
```

Respond with exactly:
```mermaid
classDiagram
  ...
```
"""

_EVENT_TOPOLOGY_SYSTEM = """\
You are a software diagramming expert. Visualise the event streaming topology as a \
Mermaid flowchart showing producers, topics, and consumers.

Rules — follow exactly:
- Output ONLY a fenced Mermaid block — no prose.
- Use graph LR.
- Producers: rounded rectangle — ProdId(["Producer Name"])
- Topics: subroutine shape — TopicId[["topic-name"]]
- Consumers: rectangle — ConsId["Consumer Name"]
- Arrows with event-name labels: A -->|"EventName"| B
- Node IDs: alphanumeric only. Never start with lowercase o or x.
- NEVER write -->|""| — omit the pipe if there is no label.
- NEVER use "end" as a node ID.
- Max 20 nodes.

EXAMPLE of correct output:
```mermaid
graph LR
  OrderSvc(["Order Service"])
  PaySvc(["Payment Service"])
  OrderTopic[["orders"]]
  PayTopic[["payments"]]
  NotifySvc["Notification Service"]
  Analytics["Analytics Service"]
  OrderSvc -->|"OrderPlaced"| OrderTopic
  PaySvc -->|"PaymentProcessed"| PayTopic
  OrderTopic -->|"consume"| NotifySvc
  OrderTopic -->|"consume"| Analytics
  PayTopic -->|"consume"| NotifySvc
```

Respond with exactly:
```mermaid
graph LR
  ...
```
"""

# ── sanitizers ──────────────────────────────────────────────────────────────────

_ER_CONSTRAINT_WORDS = re.compile(
    r"\b(NOT\s+NULL|NULL|UNIQUE|DEFAULT\s+\S+|CHECK\s*\([^)]*\)"
    r"|REFERENCES\s+\S+|ON\s+DELETE\s+\w+|ON\s+UPDATE\s+\w+)\b",
    re.IGNORECASE,
)

_SCHEMA_TYPE_TRIGGERS: dict[str, list[str]] = {
    "graph":       ["neo4j", "neptune", "janusgraph", "tigergraph"],
    "document":    ["mongodb", "dynamodb", "firestore", "couchdb", "cosmosdb", "documentdb"],
    "eventstream": ["kafka", "kinesis", "pulsar", "rabbitmq", "eventbridge", "event hub"],
}


def _extract_mermaid_definition(text: str) -> str:
    match = re.search(r"```mermaid\s*\n(.*?)(?:\n```|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _sanitize_er(definition: str) -> str:
    definition = re.sub(
        r"\}\s*([A-Za-z_]\w*\s*[|}{o][^\n]+)",
        lambda m: "}\n" + m.group(1),
        definition,
    )
    lines = []
    for line in definition.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("erDiagram", "}", "{")) and ":" not in stripped:
            line = _ER_CONSTRAINT_WORDS.sub("", line).rstrip()
        lines.append(line)
    return "\n".join(lines)


def _detect_schema_types(dm_output: str) -> set[str]:
    found: set[str] = set()
    tags = re.findall(r"<type>\s*([^<]+)\s*</type>", dm_output, re.IGNORECASE)
    for t in tags:
        if t.strip().lower() in _SCHEMA_TYPE_TRIGGERS:
            found.add(t.strip().lower())
    stores = re.findall(r"<store>\s*([^<]+)\s*</store>", dm_output, re.IGNORECASE)
    for store in stores:
        s = store.lower()
        for stype, keywords in _SCHEMA_TYPE_TRIGGERS.items():
            if any(k in s for k in keywords):
                found.add(stype)
    return found


def _has_relational_schema(dm_output: str) -> bool:
    types = re.findall(r"<type>\s*([^<]+)\s*</type>", dm_output, re.IGNORECASE)
    if types:
        return any(t.strip().lower() == "relational" for t in types)
    return bool(re.search(r"CREATE\s+TABLE", dm_output, re.IGNORECASE))


def _extract_er_from_dm(dm_output: str) -> str:
    if not _has_relational_schema(dm_output):
        return ""
    match = re.search(r"```mermaid\s*\n(erDiagram.*?)(?:\n```|$)", dm_output, re.DOTALL)
    if match:
        return _sanitize_er(match.group(1).strip())
    er_raw = call_llm(_ER_FALLBACK_SYSTEM, dm_output)
    return _sanitize_er(_extract_mermaid_definition(er_raw))


_SCHEMA_DIAGRAM_SYSTEMS: dict[str, tuple[str, str]] = {
    "graph":       (_GRAPH_SCHEMA_SYSTEM,    "Graph schema"),
    "document":    (_DOCUMENT_SCHEMA_SYSTEM, "Document schema"),
    "eventstream": (_EVENT_TOPOLOGY_SYSTEM,  "Event topology"),
}

# ── degenerate diagram detection ────────────────────────────────────────────────

def _is_degenerate(definition: str) -> bool:
    """
    Returns True when the diagram is syntactically valid but semantically broken.
    Uses diagram-type-specific checks so sequence diagrams are not mis-flagged.
    """
    if not definition or not definition.strip():
        return True
    first = definition.strip().splitlines()[0].lower()

    if "sequencediagram" in first:
        # Sequence diagrams are valid if they have at least one participant declaration
        return not bool(re.search(r"^\s*participant\s+", definition, re.MULTILINE | re.IGNORECASE))

    if "erdiagram" in first:
        # ER diagrams are valid if they have at least one entity block
        return not bool(re.search(r"^\s*\w+\s*\{", definition, re.MULTILINE))

    if "classdiagram" in first:
        return not bool(re.search(r"^\s*class\s+\w+", definition, re.MULTILINE | re.IGNORECASE))

    # Flowcharts (graph TD / flowchart LR etc.):
    # The A→B→C→Z chain: many arrows with bare short IDs and no quoted labels
    lines = [l.strip() for l in definition.splitlines() if l.strip()]
    labeled    = sum(1 for l in lines if re.search(r'\["', l))
    bare_arrow = sum(1 for l in lines if re.match(r'^[A-Za-z]\w{0,2}\s*(-->|-\.->|==>)', l))
    if bare_arrow >= 6 and labeled < 3:
        return True
    return False


_REGEN_SUFFIX = (
    "\n\nIMPORTANT: Every node MUST have a descriptive quoted label, e.g. "
    'ApiGw["API Gateway"]. Do NOT use bare single-letter IDs. '
    "Do NOT use the word 'end' as a node ID."
)

# ── agent ───────────────────────────────────────────────────────────────────────

class DiagramAgent:

    def run(self, arch_output: str, dm_output: str) -> str:
        arch_prompt = textwrap.dedent(f"""\
            Convert this architecture document into a Mermaid flowchart.
            Follow the rules and match the style of the example exactly.

            {arch_output}
        """).strip()

        seq_prompt = textwrap.dedent(f"""\
            Convert the data_flow section of this architecture document into a \
Mermaid sequence diagram showing the primary end-to-end request flow.
            Follow the rules and match the style of the example exactly.

            {arch_output}
        """).strip()

        schema_types = _detect_schema_types(dm_output)

        tasks: dict[str, tuple[str, str]] = {
            "architecture": (_ARCH_SYSTEM, arch_prompt),
            "sequence":     (_SEQ_SYSTEM,  seq_prompt),
        }
        for stype in schema_types:
            sys_prompt, _ = _SCHEMA_DIAGRAM_SYSTEMS[stype]
            tasks[stype] = (sys_prompt, dm_output)

        results: dict[str, str] = {}
        max_workers = len(tasks) + 1
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(call_llm, sys_p, usr_p): name
                for name, (sys_p, usr_p) in tasks.items()
            }
            er_future = pool.submit(_extract_er_from_dm, dm_output)
            futures[er_future] = "er"

            for future in as_completed(futures):
                name = futures[future]
                raw = future.result()
                results[name] = (
                    _extract_mermaid_definition(raw) if name != "er" else raw
                )

        # ── degenerate check + one retry ────────────────────────────────────
        retries: dict[str, tuple[str, str]] = {}
        for name, defn in results.items():
            if name in tasks and _is_degenerate(defn):
                sys_p, usr_p = tasks[name]
                retries[name] = (sys_p, usr_p + _REGEN_SUFFIX)

        if retries:
            with ThreadPoolExecutor(max_workers=len(retries)) as pool:
                retry_futures = {
                    pool.submit(call_llm, s, u): n
                    for n, (s, u) in retries.items()
                }
                for f in as_completed(retry_futures):
                    n = retry_futures[f]
                    results[n] = _extract_mermaid_definition(f.result())

        def _wrap(definition: str) -> str:
            return f"```mermaid\n{definition}\n```"

        # ── only emit diagrams with actual content ────────────────────────
        # Determine which schema-specific diagram types to include
        applicable = list(schema_types)       # e.g. ["graph"], ["document"], etc.
        if _has_relational_schema(dm_output):
            applicable.append("er")

        parts = ["<diagrams>\n\n"]
        for name in ["architecture", "sequence"] + applicable:
            defn = results.get(name, "")
            if not defn or _is_degenerate(defn):
                continue                       # skip empty / still-broken diagrams
            parts.append(
                f'  <diagram type="{name}">\n{_wrap(defn)}\n  </diagram>\n\n'
            )
        parts.append("</diagrams>")
        return "".join(parts)

    @staticmethod
    def extract_mermaid(raw: str, diagram_type: str = "architecture") -> str:
        pattern = rf'<diagram type="{diagram_type}">\s*```mermaid\s*\n(.*?)\n```'
        match = re.search(pattern, raw, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def available_types(raw: str) -> list[str]:
        return re.findall(r'<diagram type="([^"]+)">', raw)
