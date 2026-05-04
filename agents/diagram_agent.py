import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.base import call_llm

_ARCH_SYSTEM = """\
You are a software diagramming expert. Convert an architecture document into a \
Mermaid component diagram.

Rules:
- Output ONLY a fenced Mermaid block — no prose, no explanation.
- Use graph TD (top-down flow).
- Every node ID must be a plain alphanumeric token (no spaces, no special chars): A, B, SvcAuth, etc.
- Every node label MUST be double-quoted: A["Service Name"]
- Subgraph names must be plain words or double-quoted: subgraph Backend or subgraph "Back-End"
- Arrow labels must be double-quoted: A -->|"label"| B  (pipe syntax, not square brackets)
- NEVER use parentheses () or square brackets [] inside a quoted label — replace with plain text.
- NEVER use %% comments — they can break some parsers.
- Use --> for synchronous calls, -.-> for async/event-driven, ==> for persistence writes.
- Keep the diagram to 20 nodes or fewer to avoid parse timeouts.
- The diagram must render without modification in Mermaid v10+.

Respond with exactly:
```mermaid
graph TD
  ...
```
"""

_SEQ_SYSTEM = """\
You are a software diagramming expert. Convert an architecture document's data flow \
into a Mermaid sequence diagram showing the primary happy-path request end-to-end.

Rules:
- Output ONLY a fenced Mermaid block — no prose, no explanation.
- Use sequenceDiagram.
- Every participant must be declared with a short alias and a quoted label.
- Use ->> for synchronous messages, -->> for responses, -x for async/fire-and-forget.
- Add activate/deactivate around long-running operations.
- Cover the full flow from the first actor to the last persistence step.

Respond with exactly:
```mermaid
sequenceDiagram
  ...
```
"""

_FIX_SYSTEM = """\
You are a Mermaid diagram syntax expert. The diagram below has a parse error.
Fix ONLY the syntax — do not change the diagram content or structure.
Output ONLY the corrected fenced Mermaid block, no prose.
"""

_ER_FALLBACK_SYSTEM = """\
You are a software diagramming expert. Convert a data model document into a \
Mermaid erDiagram showing all entities and their relationships.

Rules:
- Output ONLY a fenced Mermaid block — no prose.
- Use erDiagram syntax.
- Include field names and types for each entity.
- Mark primary keys with PK and foreign keys with FK.
- Do NOT include SQL constraint keywords (NOT NULL, UNIQUE, DEFAULT, etc.) in field definitions.

Respond with exactly:
```mermaid
erDiagram
  ...
```
"""

_ER_CONSTRAINT_WORDS = re.compile(
    r"\b(NOT\s+NULL|NULL|UNIQUE|DEFAULT\s+\S+|CHECK\s*\([^)]*\)|REFERENCES\s+\S+|ON\s+DELETE\s+\w+|ON\s+UPDATE\s+\w+)\b",
    re.IGNORECASE,
)


def _extract_mermaid_definition(text: str) -> str:
    match = re.search(r"```mermaid\s*\n(.*?)(?:\n```|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _sanitize_er(definition: str) -> str:
    lines = []
    for line in definition.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("erDiagram", "}", "{")) and ":" not in stripped:
            line = _ER_CONSTRAINT_WORDS.sub("", line).rstrip()
        lines.append(line)
    return "\n".join(lines)


def _extract_er_from_dm(dm_output: str) -> str:
    match = re.search(r"```mermaid\s*\n(erDiagram.*?)(?:\n```|$)", dm_output, re.DOTALL)
    if match:
        return _sanitize_er(match.group(1).strip())
    er_raw = call_llm(_ER_FALLBACK_SYSTEM, dm_output)
    return _sanitize_er(_extract_mermaid_definition(er_raw))


class DiagramAgent:

    def run(self, arch_output: str, dm_output: str) -> str:
        arch_prompt = textwrap.dedent(f"""\
            Convert this architecture document into a Mermaid component diagram.
            Follow the output rules exactly.

            {arch_output}
        """).strip()

        seq_prompt = textwrap.dedent(f"""\
            Convert the data_flow section of this architecture document into a \
Mermaid sequence diagram showing the primary end-to-end request flow.
            Follow the output rules exactly.

            {arch_output}
        """).strip()

        tasks = {
            "architecture": (_ARCH_SYSTEM, arch_prompt),
            "sequence":     (_SEQ_SYSTEM,  seq_prompt),
        }

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(call_llm, sys, usr): name
                for name, (sys, usr) in tasks.items()
            }
            er_future = pool.submit(_extract_er_from_dm, dm_output)
            futures[er_future] = "er"

            for future in as_completed(futures):
                name = futures[future]
                raw = future.result()
                results[name] = (
                    _extract_mermaid_definition(raw) if name != "er" else raw
                )

        def _wrap(definition: str) -> str:
            return f"```mermaid\n{definition}\n```"

        return (
            "<diagrams>\n\n"
            f"  <diagram type=\"architecture\">\n{_wrap(results['architecture'])}\n  </diagram>\n\n"
            f"  <diagram type=\"sequence\">\n{_wrap(results['sequence'])}\n  </diagram>\n\n"
            f"  <diagram type=\"er\">\n{_wrap(results['er'])}\n  </diagram>\n\n"
            "</diagrams>"
        )

    @staticmethod
    def extract_mermaid(raw: str, diagram_type: str = "architecture") -> str:
        pattern = rf'<diagram type="{diagram_type}">\s*```mermaid\s*\n(.*?)\n```'
        match = re.search(pattern, raw, re.DOTALL)
        return match.group(1).strip() if match else ""
