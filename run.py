import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from pipeline import Pipeline, PipelineResult
from agents.diagram_agent import DiagramAgent, _FIX_SYSTEM, _extract_mermaid_definition
from agents.base import call_llm

DEFAULT_INPUT = (
    "Build a real-time ride-sharing platform for 500k concurrent users. "
    "Features: rider/driver matching in under 3 seconds, live GPS tracking, "
    "dynamic surge pricing, payment processing (cards + wallets), trip history, "
    "and driver earnings dashboard. "
    "NFRs: 99.99% uptime, p99 match latency < 3s, PCI-DSS for payments, "
    "GDPR compliance, multi-region active-active deployment."
)

_BRIEF_SYSTEM = """\
You are a senior software architect writing a concise developer briefing document.
Given the full outputs of a multi-agent design pipeline, produce a structured
DEVELOPER_BRIEF.md that gives a developer everything they need before writing
the first line of code.

Output ONLY valid markdown — no preamble, no meta-commentary.

Structure the document exactly as:

# Developer Implementation Brief

## 1. System Overview
One paragraph describing the system purpose, scale, and core constraints.

## 2. Technology Stack
A markdown table with columns: Layer | Technology | Reason

## 3. Architecture Summary
3-5 bullet points describing the key architectural decisions and the style chosen.

## 4. Data Model Highlights
2-4 bullet points on storage choices, key entities, and any important schema decisions.

## 5. Key Design Decisions
For each major decision: Decision | Choice | Why | Trade-off (4-column table).

## 6. Security Checklist
Bullet list of all CRITICAL and HIGH security controls that MUST be implemented,
derived from the critic review. Mark each as [ ] (checkbox).

## 7. Implementation Best Practices
10 concrete, numbered implementation notes specific to this system.
Each note must be actionable (e.g. "Use UUIDs not auto-increment IDs for ride records
because they will be exposed in APIs and need to be unguessable").
No generic advice — every note must reference this system's specific stack and domain.

## 8. Suggested Implementation Order
A numbered list of the recommended build sequence (what to build first through last),
with a one-sentence reason for each step.

## 9. Key Risks & Mitigations
Top 5 risks with their mitigation strategy as a table.
"""

def _mmdc_run(definition: str, output_path: Path) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile(suffix=".mmd", mode="w", delete=False) as f:
        f.write(definition)
        tmp = Path(f.name)
    result = subprocess.run(
        ["mmdc", "-i", str(tmp), "-o", str(output_path), "-b", "white"],
        capture_output=True, text=True,
    )
    tmp.unlink(missing_ok=True)
    return result


def render_mermaid(definition: str, output_path: Path, diagram_type: str = "") -> bool:
    mmd_path = output_path.with_suffix(".mmd")
    mmd_path.write_text(definition, encoding="utf-8")

    result = _mmdc_run(definition, output_path)
    if result.returncode == 0:
        return True

    print(f"\n    syntax error — asking LLM to fix ...")
    fix_prompt = (
        f"Diagram type: {diagram_type}\n\n"
        f"Parse error:\n{result.stderr.strip()}\n\n"
        f"Broken diagram:\n```mermaid\n{definition}\n```"
    )
    fixed_raw = call_llm(_FIX_SYSTEM, fix_prompt)
    fixed_def = _extract_mermaid_definition(fixed_raw)
    mmd_path.write_text(fixed_def, encoding="utf-8")

    result2 = _mmdc_run(fixed_def, output_path)
    if result2.returncode != 0:
        print(f"    [mmdc error after fix] {result2.stderr[:200]}")
        return False
    print(f"    fixed and rendered.", end=" ")
    return True


def generate_developer_brief(result: PipelineResult, out_dir: Path) -> Path:
    user_prompt = f"""\
<srs>
{result.requirements[:3000]}
</srs>

<architecture>
{result.architecture[:3000]}
</architecture>

<data_model>
{result.data_model[:2000]}
</data_model>

<critique_summary>
{result.critique[:2000]}
</critique_summary>

<pipeline_metadata>
Critic score: {result.score}/10
Rounds run: {result.rounds}
</pipeline_metadata>

Using all of the above, produce the DEVELOPER_BRIEF.md document.
"""
    print("  generating Developer Brief ...", end=" ", flush=True)
    brief = call_llm(_BRIEF_SYSTEM, user_prompt)
    path  = out_dir / "DEVELOPER_BRIEF.md"
    path.write_text(brief, encoding="utf-8")
    print(f"done  →  {path.name}  ({len(brief):,} chars)")
    return path


def run_full(user_input: str, use_rag: bool, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diagrams"
    diag_dir.mkdir(exist_ok=True)

    label = "with RAG" if use_rag else "without RAG"
    bar   = "━" * 55

    print(f"\n{bar}")
    print(f"  Pipeline  [{label}]")
    print(f"  Output → {out_dir}")
    print(bar)

    pipeline = Pipeline()
    result   = pipeline.run(user_input, use_rag=use_rag)

    if result.errors:
        print("\n  Errors:")
        for k, v in result.errors.items():
            print(f"    [{k}] {v}")

    print(f"\n  Critic score: {result.score}/10  |  Rounds: {result.rounds}")

    print(f"\n{bar}")
    print("  Saving outputs")
    print(bar)
    (out_dir / "srs.md").write_text(result.requirements, encoding="utf-8")
    print(f"  srs.md              ({len(result.requirements):,} chars)")
    (out_dir / "architecture.xml").write_text(result.architecture, encoding="utf-8")
    print(f"  architecture.xml    ({len(result.architecture):,} chars)")
    (out_dir / "data_model.xml").write_text(result.data_model, encoding="utf-8")
    print(f"  data_model.xml      ({len(result.data_model):,} chars)")
    (out_dir / "critique.md").write_text(result.critique, encoding="utf-8")
    print(f"  critique.md         ({len(result.critique):,} chars)")

    print(f"\n{bar}")
    print("  Rendering diagrams")
    print(bar)
    for dtype, filename in [("architecture", "architecture.png"),
                             ("sequence",     "sequence.png"),
                             ("er",           "er.png")]:
        definition = DiagramAgent.extract_mermaid(result.diagrams, dtype)
        out_path   = diag_dir / filename
        print(f"  {dtype:<14}", end=" ", flush=True)
        if not definition:
            print("SKIPPED (empty)")
            continue
        ok = render_mermaid(definition, out_path, dtype)
        print(f"→  {out_path.name}" if ok else "FAILED")

    print(f"\n{bar}")
    print("  Generating Developer Brief")
    print(bar)
    generate_developer_brief(result, out_dir)

    print(f"\n{bar}")
    print(f"  Done.  Score: {result.score}/10  |  Rounds: {result.rounds}")
    print(f"  Output: {out_dir.resolve()}/")
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    user_input = args.scenario or DEFAULT_INPUT
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out   = Path("output")

    if args.compare:
        run_full(user_input, use_rag=True,  out_dir=base_out / f"{ts}_rag")
        run_full(user_input, use_rag=False, out_dir=base_out / f"{ts}_no_rag")
    else:
        run_full(user_input, use_rag=not args.no_rag, out_dir=base_out / ts)


if __name__ == "__main__":
    main()
