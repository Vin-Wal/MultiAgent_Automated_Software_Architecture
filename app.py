import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from agents.requirements_agent import RequirementsAgent
from agents.architecture_agent import ArchitectureAgent
from agents.data_modeler_agent import DataModelerAgent
from agents.critic_agent import CriticAgent, extract_action_items
from agents.diagram_agent import DiagramAgent, _sanitize_er
from agents.base import call_llm_stream
from pipeline import PipelineResult

PASS_THRESHOLD = 7

st.set_page_config(
    page_title="Architecture Pipeline",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
  /* ── global chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  section[data-testid="stSidebar"] { display: none; }
  .block-container { padding: 0; max-width: 820px; margin: 0 auto; }

  /* ── landing ── */
  .landing-wrap {
    min-height: 88vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 2rem 1rem;
  }
  .landing-logo {
    font-size: 2.6rem; font-weight: 800; color: #0f172a;
    letter-spacing: -0.04em; margin-bottom: 0.4rem; text-align: center;
  }
  .landing-sub {
    color: #64748b; font-size: 1rem; margin-bottom: 2.2rem;
    text-align: center; max-width: 480px;
  }

  /* ── chat-style input box ── */
  .input-shell {
    width: 100%; background: #fff; border: 1.5px solid #e2e8f0;
    border-radius: 16px; padding: 1rem 1.2rem 0.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 1rem;
  }
  .stTextArea textarea {
    border: none !important; box-shadow: none !important;
    background: transparent !important; font-size: 0.96rem;
    color: #0f172a; resize: none; outline: none;
    padding: 0 !important;
  }
  .stTextArea textarea:focus { border: none !important; box-shadow: none !important; }
  [data-testid="stTextAreaRootElement"] { border: none; box-shadow: none; }

  /* ── bottom bar (send row) ── */
  .send-row {
    display: flex; align-items: center; gap: 0.8rem;
    margin-top: 0.4rem;
  }

  /* ── feature pills ── */
  .feature-row {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    justify-content: center; margin-top: 2rem;
  }
  .fpill {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 9999px; padding: 0.3rem 0.85rem;
    font-size: 0.78rem; color: #475569; white-space: nowrap;
  }

  /* ── results header ── */
  .result-header {
    display: flex; align-items: center; gap: 1.2rem;
    padding: 1.4rem 0 1rem; border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1rem;
  }
  .score-ring {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; width: 72px; height: 72px;
    border-radius: 50%; border: 5px solid; flex-shrink: 0;
  }
  .score-num  { font-size: 1.8rem; font-weight: 800; line-height: 1; }
  .score-den  { font-size: 0.7rem; color: #94a3b8; }
  .c-red    { border-color: #fca5a5; color: #dc2626; }
  .c-orange { border-color: #fdba74; color: #ea580c; }
  .c-indigo { border-color: #a5b4fc; color: #4f46e5; }
  .c-green  { border-color: #86efac; color: #16a34a; }

  .result-meta h3   { margin: 0 0 0.25rem; font-size: 1.1rem; color: #0f172a; }
  .result-meta p    { margin: 0; font-size: 0.83rem; color: #64748b; }

  /* ── tabs ── */
  [data-testid="stTabs"] { margin-top: 0.5rem; }
  [data-testid="stTabs"] button {
    font-size: 0.83rem; font-weight: 500; color: #64748b;
    padding: 0.45rem 0.8rem; border-radius: 0;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #4f46e5; border-bottom: 2px solid #4f46e5; font-weight: 600;
  }

  /* ── section label ── */
  .slabel {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94a3b8; margin: 1.4rem 0 0.5rem;
  }

  /* ── architecture cards (shared style for all sections) ── */
  .arch-card {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem 1.1rem; margin-bottom: 0.6rem; background: #fff;
  }
  .arch-card-title {
    font-weight: 700; color: #1e293b; font-size: 0.92rem; margin-bottom: 0.25rem;
  }
  .arch-card-badge {
    display: inline-block; background: #ede9fe; color: #5b21b6;
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 9999px; letter-spacing: 0.03em; margin-bottom: 0.4rem;
  }
  .arch-card-body {
    color: #475569; font-size: 0.84rem; line-height: 1.6;
  }
  .arch-card-why {
    color: #64748b; font-size: 0.78rem; margin-top: 0.5rem;
    padding-top: 0.5rem; border-top: 1px solid #f1f5f9;
  }
  .arch-card-why b { color: #475569; }
  /* keep old .comp alias so overview tab still works */
  .comp { border: 1px solid #e2e8f0; border-left: 3px solid #4f46e5;
    border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.55rem; background:#fff; }
  .comp-name { font-weight: 600; color: #1e293b; font-size: 0.92rem; }
  .comp-tech { color: #4f46e5; font-size: 0.78rem; margin-top: 0.1rem; }
  .comp-resp { color: #475569; font-size: 0.86rem; margin-top: 0.28rem; }

  /* ── callout ── */
  .callout {
    background: #f0f4ff; border: 1px solid #c7d2fe;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    color: #3730a3; font-size: 0.91rem; line-height: 1.65;
  }

  /* ── diagram card ── */
  .diag-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  }
  .diag-title {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 0.7rem;
  }

  /* ── new prompt bar (bottom) ── */
  .reprompt-bar {
    border-top: 1px solid #e2e8f0; padding: 1.2rem 0 2rem;
    margin-top: 2rem;
  }
</style>
""", unsafe_allow_html=True)


# ── mermaid → PNG renderer ──────────────────────────────────────────────────────

_MMDC     = "/usr/local/bin/mmdc"
_NODE_BIN = "/usr/local/bin"

_MMDC_CONFIG = """\
{
  "theme": "base",
  "themeVariables": {
    "primaryColor": "#ede9fe",
    "primaryTextColor": "#1e1b4b",
    "primaryBorderColor": "#4f46e5",
    "lineColor": "#94a3b8",
    "background": "#ffffff",
    "nodeBorder": "#4f46e5",
    "edgeLabelBackground": "#f8fafc",
    "fontFamily": "ui-sans-serif, system-ui, sans-serif",
    "fontSize": "14px"
  }
}
"""


def _clean_mermaid(definition: str) -> str:
    definition = re.sub(r"%%.*$", "", definition, flags=re.MULTILINE)
    lines = [line.rstrip() for line in definition.splitlines() if line.strip()]
    return "\n".join(lines)


# ── per-type sanitizers (applied before every mmdc call) ────────────────────────

# Sequence: fix invalid arrows, strip forbidden chars from message text
_SEQ_INVALID_ARROWS = re.compile(r"-x>>|-x>")            # -x>> / -x> → -x
_SEQ_BARE_ARROW     = re.compile(r"(?<![<\-])->(?![>)])")  # bare -> → ->>
# Bare "--" between two participants (no arrowhead) e.g. "K -- P: msg" → "K -->> P: msg"
_SEQ_BARE_DOUBLE_DASH = re.compile(r"(\w+)\s+--\s+(\w+)\s*:")
_SEQ_MSG_STRIP      = re.compile(r"[<>()\[\]{}#;]")
_SEQ_MSG_LINE       = re.compile(
    r"^(\s*\w[\w ]*(?:->>|-->>|-x|--x)\s*\w[\w ]*\s*:)\s*(.+)$"
)


_SEQ_PARTICIPANT_RE = re.compile(r"^\s*participant\s+(\w+)", re.IGNORECASE)
_SEQ_USER_ALIASES   = {"user", "client", "browser", "mobile", "app", "frontend",
                       "u", "c", "usr", "cli"}

def _sanitize_sequence(definition: str) -> str:
    lines = []
    for line in definition.splitlines():
        line = _SEQ_INVALID_ARROWS.sub("-x", line)
        line = _SEQ_BARE_DOUBLE_DASH.sub(r"\1 -->> \2:", line)  # K -- P: → K -->> P:
        line = _SEQ_BARE_ARROW.sub("->>", line)
        line = re.sub(r"\bend\b", "End", line)
        m = _SEQ_MSG_LINE.match(line)
        if m:
            prefix, msg = m.group(1), m.group(2)
            msg = _SEQ_MSG_STRIP.sub("", msg).strip()
            line = f"{prefix} {msg}"
        lines.append(line)

    # Ensure the user/client participant is declared first
    participant_indices = [
        i for i, l in enumerate(lines)
        if _SEQ_PARTICIPANT_RE.match(l)
    ]
    if len(participant_indices) > 1:
        first_idx = participant_indices[0]
        for idx in participant_indices[1:]:
            alias = _SEQ_PARTICIPANT_RE.match(lines[idx]).group(1).lower()
            label = lines[idx].lower()
            if any(kw in alias or kw in label for kw in _SEQ_USER_ALIASES):
                # Move this participant line to be the first participant
                p_line = lines.pop(idx)
                lines.insert(first_idx, p_line)
                break

    return "\n".join(lines)


# Matches -->|""| or -->|''|  (empty pipe label)
_FLOW_EMPTY_LABEL = re.compile(r'(\-\->|==\>|\-\.\->)\|["\']["\']?\|')
# Matches -->|==>| or -->|-.->|  (another arrow type used as a pipe label)
_FLOW_ARROW_AS_LABEL = re.compile(r'(\-\->|==\>|\-\.\->)\|([=\-\.>]+)\|')


def _sanitize_flowchart(definition: str) -> str:
    lines = []
    for line in definition.splitlines():
        stripped = line.strip()
        # Remove empty pipe labels:  -->|""| Node  →  --> Node
        line = _FLOW_EMPTY_LABEL.sub(lambda m: m.group(1), line)
        # Fix arrow-as-label:  -->|==>|  →  ==>   and  -->|-.-|  →  -.->
        line = _FLOW_ARROW_AS_LABEL.sub(lambda m: m.group(2) if m.group(2).endswith(">") else m.group(1), line)
        # Standalone "end" on its own line closes a subgraph — leave it alone
        if stripped.lower() != "end":
            line = re.sub(r"\bend\b", "End", line)
        lines.append(line)
    return "\n".join(lines)


# classDiagram: remove spaces from class names, strip special chars
_CLASS_NAME_RE = re.compile(r"\bclass\s+([\w\s\-]+?)\s*\{")


def _sanitize_classdiagram(definition: str) -> str:
    def _fix_name(m: re.Match) -> str:
        raw = m.group(1)
        clean = re.sub(r"[^A-Za-z0-9_]", "_", raw).strip("_")
        return f"class {clean} {{"
    return _CLASS_NAME_RE.sub(_fix_name, definition)


_SANITIZERS: dict[str, list] = {
    "sequencediagram": [_sanitize_sequence],
    "graph":           [_sanitize_flowchart],
    "flowchart":       [_sanitize_flowchart],
    "classdiagram":    [_sanitize_classdiagram],
}


def _auto_sanitize(definition: str) -> str:
    first = definition.lstrip().split("\n")[0].lower().replace(" ", "")
    for key, fns in _SANITIZERS.items():
        if first.startswith(key):
            for fn in fns:
                definition = fn(definition)
            break
    return definition


def _mmdc_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = _NODE_BIN + ":" + env.get("PATH", "")
    return env


def _render_to_png(definition: str, width: int = 1200) -> tuple[bytes | None, str]:
    """Render a Mermaid definition to PNG bytes via mmdc.
    Returns (png_bytes, error_message). png_bytes is None on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "diagram.mmd").write_text(definition, encoding="utf-8")
        (tmp / "config.json").write_text(_MMDC_CONFIG, encoding="utf-8")
        out_png = tmp / "diagram.png"
        try:
            result = subprocess.run(
                [
                    _MMDC,
                    "-i", str(tmp / "diagram.mmd"),
                    "-o", str(out_png),
                    "--configFile", str(tmp / "config.json"),
                    "-w", str(width),
                    "--backgroundColor", "white",
                ],
                capture_output=True, text=True, timeout=30,
                env=_mmdc_env(),
            )
            if out_png.exists():
                return out_png.read_bytes(), ""
            err = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr or result.stdout or "").strip()
            return None, err
        except subprocess.TimeoutExpired:
            return None, "mmdc timed out"
        except FileNotFoundError:
            return None, f"mmdc not found at {_MMDC}"


@st.dialog("Diagram", width="large")
def _diagram_fullscreen(definition: str, title: str) -> None:
    st.markdown(f"**{title}**")
    png, _ = _render_to_png(definition, width=1600)
    if png:
        st.image(png, use_container_width=True)
    else:
        st.code(definition, language="text")


def _llm_fix_diagram(definition: str, error: str) -> str:
    from agents.diagram_agent import _FIX_SYSTEM, _extract_mermaid_definition
    from agents.base import call_llm
    prompt = (
        f"Parse error:\n{error}\n\n"
        f"Broken diagram:\n```mermaid\n{definition}\n```"
    )
    fixed_raw = call_llm(_FIX_SYSTEM, prompt, max_tokens=1500)
    return _extract_mermaid_definition(fixed_raw)


def render_mermaid(definition: str, title: str = "Diagram") -> None:
    definition = _auto_sanitize(_clean_mermaid(definition))
    if not definition:
        st.info("No diagram definition available.")
        return

    # Pass 1: sanitizers only
    png, err = _render_to_png(definition)

    # Pass 2: all sanitizers applied regardless of type
    if not png:
        definition = _sanitize_sequence(_sanitize_flowchart(_sanitize_classdiagram(definition)))
        png, err = _render_to_png(definition)

    # Pass 3: LLM self-correction using the exact mmdc error message
    if not png and err:
        with st.spinner("Fixing diagram syntax …"):
            definition = _llm_fix_diagram(definition, err)
        definition = _auto_sanitize(_clean_mermaid(definition))
        png, err = _render_to_png(definition)

    if png:
        st.image(png, use_container_width=True)
        if st.button("⛶  Full screen", key=f"fs_{hash(definition) & 0xFFFFFF}"):
            _diagram_fullscreen(definition, title)
    else:
        st.warning("Diagram render failed after 3 attempts.")
        with st.expander("View diagram definition"):
            st.code(definition, language="text")


# ── agents ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base ...")
def load_agents():
    return {
        "requirements": RequirementsAgent(),
        "architecture": ArchitectureAgent(),
        "data_modeler": DataModelerAgent(),
        "critic":       CriticAgent(),
        "diagram":      DiagramAgent(),
    }


# ── xml parsers ─────────────────────────────────────────────────────────────────

def parse_arch(xml_str: str) -> dict | None:
    if "<overview>" not in xml_str and "<component>" not in xml_str:
        return None

    components = []
    for blk in re.findall(r"<component>(.*?)</component>", xml_str, re.DOTALL):
        name = _rx("name", blk)
        if name:
            components.append({
                "name":           name,
                "responsibility": _rx("responsibility", blk),
                "technology":     _rx("technology",     blk),
            })

    decisions = []
    for blk in re.findall(r"<decision>(.*?)</decision>", xml_str, re.DOTALL):
        title = _rx("title", blk)
        if title:
            decisions.append({
                "Decision":  title,
                "Choice":    _rx("chosen",     blk),
                "Rationale": _rx("rationale",  blk),
                "Trade-off": _rx("trade_offs", blk),
            })

    return {
        "overview":     _rx("overview",     xml_str),
        "data_flow":    _rx("data_flow",    xml_str),
        "nfr_strategy": _rx("nfr_strategy", xml_str),
        "components":   components,
        "decisions":    decisions,
    }


def _rx(tag: str, text: str, default: str = "") -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else default


def parse_dm(xml_str: str) -> dict | None:
    if "<storage_strategy>" not in xml_str and "<schema>" not in xml_str:
        return None

    er_m = re.search(r"```mermaid\s*\n(erDiagram.*?)```", xml_str, re.DOTALL)

    schemas = []
    for schema_block in re.findall(r"<schema>(.*?)</schema>", xml_str, re.DOTALL):
        store = _rx("store", schema_block)
        stype = _rx("type", schema_block, "relational").lower()
        entities = []
        for ent_block in re.findall(r"<entity>(.*?)</entity>", schema_block, re.DOTALL):
            entities.append({
                "name":    _rx("name",    ent_block),
                "ddl":     _rx("ddl",     ent_block),
                "indexes": _rx("indexes", ent_block),
            })
        if store:
            schemas.append({"store": store, "type": stype, "entities": entities})

    # Parse storage_strategy — supports structured sub-tags or free-text <strategy> blocks
    raw_strategy = _rx("storage_strategy", xml_str)
    strategy_entries = []
    for blk in re.findall(r"<strategy>(.*?)</strategy>", raw_strategy, re.DOTALL):
        store = _rx("store", blk)
        if not store:
            # Free-text block: first word/phrase before "is"/"are" is the store name
            m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9\s\-\.]{1,30?}?)\s+(?:is|are)\b", blk.strip())
            store = m.group(1).strip() if m else blk.strip().split()[0]
        desc = re.sub(r"<[^>]+>", " ", blk).strip()
        desc = re.sub(r"\s{2,}", " ", desc)
        strategy_entries.append({
            "store":           store,
            "data_lives":      _rx("data_lives",     blk) or _rx("what",   blk),
            "why_chosen":      _rx("why_chosen",      blk) or _rx("why",    blk),
            "access_patterns": _rx("access_patterns", blk) or _rx("access", blk),
            "description":     desc,
        })
    strategy_text = raw_strategy if not strategy_entries else ""

    return {
        "storage_strategy":    strategy_text,
        "strategy_entries":    strategy_entries,
        "normalization_notes": _rx("normalization_notes",  xml_str),
        "trade_offs":          _rx("trade_offs",           xml_str),
        "er":                  er_m.group(1).strip() if er_m else "",
        "schemas":             schemas,
    }


def _score_cls(s: int) -> str:
    if s <= 5:  return "c-red"
    if s == 6:  return "c-orange"
    if s <= 8:  return "c-indigo"
    return "c-green"


# ── pipeline runner ─────────────────────────────────────────────────────────────

def _stream_agent(label: str, stream_fn, bar: st.delta_generator.DeltaGenerator,
                  pct: int, caption_el: st.delta_generator.DeltaGenerator) -> str:
    caption_el.caption(label)
    bar.progress(pct)
    output_el = st.empty()
    text = ""
    for token in stream_fn():
        text += token
        output_el.markdown(text + "▌")
    output_el.empty()
    return text


def run_pipeline(scenario: str, use_rag: bool) -> PipelineResult:
    agents = load_agents()
    result = PipelineResult()
    bar    = st.progress(0)
    cap    = st.empty()

    def stream(label, pct, fn):
        return _stream_agent(label, fn, bar, pct, cap)

    # Requirements
    req_agent = agents["requirements"]
    req_prompt = req_agent._build_prompt(scenario, use_rag)
    result.requirements = stream("Generating requirements ...", 10,
        lambda: call_llm_stream(req_agent._SYSTEM, req_prompt, max_tokens=3000))

    # Architecture round 1
    arch_agent = agents["architecture"]
    arch_prompt = arch_agent._build_prompt(result.requirements, use_rag)
    result.architecture = stream("Designing architecture ...", 30,
        lambda: call_llm_stream(arch_agent._SYSTEM, arch_prompt, max_tokens=3000))

    # Data model round 1
    dm_agent = agents["data_modeler"]
    dm_prompt = dm_agent._build_prompt(result.requirements, result.architecture, use_rag)
    result.data_model = stream("Modelling data ...", 50,
        lambda: call_llm_stream(dm_agent._SYSTEM, dm_prompt, max_tokens=4000))

    # Critic round 1
    critic_agent = agents["critic"]
    crit_prompt = critic_agent._build_prompt(
        result.requirements, result.architecture, result.data_model, use_rag
    )
    critique_raw = stream("Running security review ...", 65,
        lambda: call_llm_stream(critic_agent._SYSTEM, crit_prompt, max_tokens=3000))
    from agents.critic_agent import _extract_score
    result.critique = critique_raw
    result.score    = _extract_score(critique_raw)
    result.rounds   = 1

    # Round 2 if needed
    if result.score < PASS_THRESHOLD:
        cap.caption(f"Score {result.score}/10 — running improvement round ...")
        items = extract_action_items(result.critique)

        arch_prompt2 = arch_agent._build_prompt(result.requirements, use_rag, prior_critique=items)
        result.architecture = stream("Improving architecture ...", 72,
            lambda: call_llm_stream(arch_agent._SYSTEM, arch_prompt2, max_tokens=3000))

        dm_prompt2 = dm_agent._build_prompt(
            result.requirements, result.architecture, use_rag, prior_critique=items
        )
        result.data_model = stream("Improving data model ...", 80,
            lambda: call_llm_stream(dm_agent._SYSTEM, dm_prompt2, max_tokens=4000))

        crit_prompt2 = critic_agent._build_prompt(
            result.requirements, result.architecture, result.data_model,
            use_rag, prior_critique=items,
        )
        critique2_raw = stream("Re-scoring ...", 88,
            lambda: call_llm_stream(critic_agent._SYSTEM, crit_prompt2, max_tokens=3000))
        result.critique = critique2_raw
        result.score    = _extract_score(critique2_raw)
        result.rounds   = 2

    # Diagrams
    cap.caption("Generating diagrams ...")
    bar.progress(95)
    result.diagrams = agents["diagram"].run(result.architecture, result.data_model)

    bar.progress(100)
    bar.empty()
    cap.empty()
    return result


# ── tab renderers ───────────────────────────────────────────────────────────────

def tab_overview(r: PipelineResult) -> None:
    cls  = _score_cls(r.score)
    arch = parse_arch(r.architecture)

    st.markdown(
        f'<div class="result-header">'
        f'<div class="score-ring {cls}">'
        f'<span class="score-num">{r.score}</span>'
        f'<span class="score-den">/10</span></div>'
        f'<div class="result-meta">'
        f'<h3>Architecture Review Complete</h3>'
        f'<p>{r.rounds} critic round(s) &nbsp;·&nbsp; '
        f'SRS {len(r.requirements):,} chars &nbsp;·&nbsp; '
        f'Architecture {len(r.architecture):,} chars</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if arch and arch["overview"]:
        st.markdown('<div class="slabel">System Overview</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(arch["overview"])

    if arch and arch["components"]:
        st.markdown('<div class="slabel">Key Components</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for i, c in enumerate(arch["components"]):
            with (col1 if i % 2 == 0 else col2):
                st.markdown(
                    f'<div class="comp"><div class="comp-name">{c["name"]}</div>'
                    f'<div class="comp-tech">⚡ {c["technology"]}</div>'
                    f'<div class="comp-resp">{c["responsibility"]}</div></div>',
                    unsafe_allow_html=True,
                )


def tab_requirements(r: PipelineResult) -> None:
    st.markdown(r.requirements)


def _prose_to_md(text: str) -> str:
    """
    Convert plain numbered prose to a proper markdown numbered list.
    Handles both multi-line and single-line formats like '1. Step one 2. Step two'.
    """
    text = text.strip()
    # If the text has inline numbers (e.g. "1. ... 2. ..."), split on them first
    if re.search(r"\s+\d+\.\s", text):
        text = re.sub(r"\s+(\d+\.)\s", r"\n\1 ", text)
    lines = text.splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        out.append(line)
    return "\n".join(out)


_NFR_ICONS: dict[str, str] = {
    "scalability":     "Scalability",
    "reliability":     "Reliability",
    "security":        "Security",
    "performance":     "Performance",
    "cost_efficiency": "Cost Efficiency",
    "maintainability": "Maintainability",
    "availability":    "Availability",
    "observability":   "Observability",
}

# Strip RAG chunk citations like "(source: [Chunk 5])" or "[5]" from display text
_CHUNK_REF = re.compile(r"\s*[\(\[]?source[:\s]*\[?Chunk\s*\d+\]?\)?|\[\d+\]", re.IGNORECASE)

def _clean(text: str) -> str:
    return _CHUNK_REF.sub("", text).strip().rstrip(".")


def tab_architecture(r: PipelineResult) -> None:
    arch = parse_arch(r.architecture)
    if arch is None:
        stripped = re.sub(r"<[^>]+>", "\n", r.architecture).strip()
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        st.markdown(stripped)
        with st.expander("Raw XML"):
            st.code(r.architecture, language="xml")
        return

    # ── Architecture diagram ──────────────────────────────────────────────────
    arch_def = DiagramAgent.extract_mermaid(r.diagrams, "architecture")
    if arch_def:
        st.markdown('<p class="slabel">Architecture diagram</p>', unsafe_allow_html=True)
        render_mermaid(arch_def, title="Architecture")

    # ── Overview ──────────────────────────────────────────────────────────────
    if arch["overview"]:
        st.markdown('<p class="slabel">Overview</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="arch-card">'
            f'<div class="arch-card-body" style="font-size:0.91rem;color:#334155">'
            f'{_clean(arch["overview"])}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Components ────────────────────────────────────────────────────────────
    if arch["components"]:
        st.markdown('<p class="slabel">Components</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for i, c in enumerate(arch["components"]):
            tech = _clean(c["technology"])
            resp = _clean(c["responsibility"])
            card = (
                f'<div class="arch-card">'
                f'<div class="arch-card-title">{c["name"]}</div>'
                + (f'<span class="arch-card-badge">{tech}</span>' if tech else "")
                + f'<div class="arch-card-body">{resp}</div>'
                f'</div>'
            )
            (col1 if i % 2 == 0 else col2).markdown(card, unsafe_allow_html=True)

    # ── Request flow + sequence diagram ──────────────────────────────────────
    if arch["data_flow"]:
        st.markdown('<p class="slabel">Request flow</p>', unsafe_allow_html=True)
        flow_text = _prose_to_md(_clean(arch["data_flow"]))
        with st.container(border=True):
            st.markdown(flow_text)
    seq_def = DiagramAgent.extract_mermaid(r.diagrams, "sequence")
    if seq_def:
        st.markdown('<p class="slabel">Sequence diagram</p>', unsafe_allow_html=True)
        render_mermaid(seq_def, title="Sequence")

    # ── Design decisions ──────────────────────────────────────────────────────
    if arch["decisions"]:
        st.markdown('<p class="slabel">Design decisions</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for i, d in enumerate(arch["decisions"]):
            rationale = _clean(d["Rationale"])
            tradeoff  = _clean(d["Trade-off"])
            card = (
                f'<div class="arch-card">'
                f'<div class="arch-card-title">{d["Decision"]}</div>'
                + (f'<span class="arch-card-badge">{d["Choice"]}</span>' if d["Choice"] else "")
                + f'<div class="arch-card-body">{rationale}</div>'
                + (f'<div class="arch-card-why"><b>Trade-off:</b> {tradeoff}</div>' if tradeoff else "")
                + '</div>'
            )
            (col1 if i % 2 == 0 else col2).markdown(card, unsafe_allow_html=True)

    # ── Non-functional requirements ───────────────────────────────────────────
    if arch["nfr_strategy"]:
        st.markdown('<p class="slabel">Non-functional requirements</p>', unsafe_allow_html=True)
        raw_nfr = arch["nfr_strategy"]
        nfr_entries = re.findall(r"<(\w+)>(.*?)</\1>", raw_nfr, re.DOTALL)
        if nfr_entries:
            col1, col2 = st.columns(2)
            for i, (tag, body) in enumerate(nfr_entries):
                label = _NFR_ICONS.get(tag, tag.replace("_", " ").title())
                text  = _clean(re.sub(r"\s+", " ", body))
                card  = (
                    f'<div class="arch-card">'
                    f'<div class="arch-card-title">{label}</div>'
                    f'<div class="arch-card-body">{text}</div>'
                    f'</div>'
                )
                (col1 if i % 2 == 0 else col2).markdown(card, unsafe_allow_html=True)
        else:
            cleaned = _clean(re.sub(r"<[^>]+>", " ", raw_nfr))
            st.markdown(
                f'<div class="arch-card"><div class="arch-card-body">{cleaned}</div></div>',
                unsafe_allow_html=True,
            )


_SCHEMA_TYPE_META: dict[str, tuple[str, str, str]] = {
    # type          → (entity label,    code language, badge colour)
    "relational":   ("Table",           "sql",          "#4f46e5"),
    "document":     ("Collection",      "json",         "#0891b2"),
    "graph":        ("Node / Edge",     "text",         "#7c3aed"),
    "vector":       ("Index",           "json",         "#be185d"),
    "keyvalue":     ("Key pattern",     "text",         "#b45309"),
    "eventstream":  ("Topic",           "json",         "#065f46"),
    "objectstore":  ("Bucket / Prefix", "text",         "#1e40af"),
}

_TYPE_LABELS: dict[str, str] = {
    "relational":  "Relational",
    "document":    "Document",
    "graph":       "Graph",
    "vector":      "Vector",
    "keyvalue":    "Key-Value / Cache",
    "eventstream": "Event Stream",
    "objectstore": "Object Store",
}


def _type_badge(schema_type: str) -> str:
    _, _, colour = _SCHEMA_TYPE_META.get(schema_type, ("Entity", "text", "#64748b"))
    label = _TYPE_LABELS.get(schema_type, schema_type.title())
    return (
        f'<span style="background:{colour};color:#fff;font-size:0.7rem;'
        f'font-weight:600;padding:2px 8px;border-radius:9999px;'
        f'letter-spacing:0.04em;vertical-align:middle">{label}</span>'
    )


_SCHEMA_TO_DIAGRAM: dict[str, str] = {
    "relational":  "er",
    "document":    "document",
    "graph":       "graph",
    "eventstream": "eventstream",
}

_DIAGRAM_TITLES: dict[str, str] = {
    "architecture": "Architecture diagram",
    "sequence":     "Sequence diagram",
    "er":           "Entity relationship",
    "graph":        "Graph schema",
    "document":     "Document schema",
    "eventstream":  "Event topology",
}


def tab_data_model(r: PipelineResult) -> None:
    dm = parse_dm(r.data_model)
    if dm is None:
        stripped = re.sub(r"<[^>]+>", "\n", r.data_model).strip()
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        st.markdown(stripped)
        with st.expander("Raw XML"):
            st.code(r.data_model, language="xml")
        return

    # ── Storage strategy ──────────────────────────────────────────────────────
    _STORE_TYPE_KEYWORDS: dict[str, list[str]] = {
        "relational":  ["postgres", "mysql", "sqlite", "aurora", "cockroach", "tidb"],
        "document":    ["mongo", "firestore", "dynamo", "cosmos", "couch", "documentdb"],
        "graph":       ["neo4j", "neptune", "janusgraph", "tigergraph"],
        "vector":      ["pinecone", "weaviate", "qdrant", "pgvector", "milvus", "chroma"],
        "keyvalue":    ["redis", "memcached", "elasticache", "dragonfly"],
        "eventstream": ["kafka", "kinesis", "pulsar", "rabbitmq", "eventbridge"],
        "objectstore": ["s3", "gcs", "blob", "minio", "r2"],
    }

    def _infer_store_type(store_name: str) -> str:
        sl = store_name.lower()
        for stype, kws in _STORE_TYPE_KEYWORDS.items():
            if any(k in sl for k in kws):
                return stype
        return ""

    entries = dm.get("strategy_entries", [])
    if entries:
        st.markdown('<p class="slabel">Storage strategy</p>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#64748b;font-size:0.82rem;margin:-8px 0 14px">'
            'All stores below are used simultaneously — each handles a different concern.</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(entries), 3))
        for i, entry in enumerate(entries):
            stype_hint = _infer_store_type(entry["store"])
            badge = _type_badge(stype_hint) if stype_hint else ""
            body_parts = []
            if entry.get("data_lives"):
                body_parts.append(
                    f'<div style="margin-bottom:5px"><span style="color:#cbd5e1;font-weight:600">'
                    f'What lives here</span><br>'
                    f'<span style="color:#94a3b8">{entry["data_lives"]}</span></div>'
                )
            if entry.get("why_chosen"):
                body_parts.append(
                    f'<div style="margin-bottom:5px"><span style="color:#cbd5e1;font-weight:600">'
                    f'Why chosen</span><br>'
                    f'<span style="color:#94a3b8">{entry["why_chosen"]}</span></div>'
                )
            if entry.get("access_patterns"):
                body_parts.append(
                    f'<div><span style="color:#cbd5e1;font-weight:600">'
                    f'Access patterns</span><br>'
                    f'<span style="color:#94a3b8">{entry["access_patterns"]}</span></div>'
                )
            if not body_parts and entry.get("description"):
                body_parts.append(
                    f'<div style="color:#94a3b8;font-size:0.82rem">{entry["description"]}</div>'
                )
            with cols[i % 3]:
                st.markdown(
                    f'<div style="border:1px solid #334155;border-radius:10px;'
                    f'padding:14px 16px;margin-bottom:10px;background:#0f172a;height:100%">'
                    f'<div style="font-weight:700;font-size:0.95rem;margin-bottom:10px">'
                    f'{entry["store"]}&nbsp;&nbsp;{badge}</div>'
                    f'<div style="font-size:0.8rem">{"".join(body_parts)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    elif dm["storage_strategy"]:
        st.markdown('<p class="slabel">Storage strategy</p>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(dm["storage_strategy"])

    # ── Data diagrams (rendered inline, grouped by type) ─────────────────────
    rendered_diag_types: set[str] = set()
    if dm["schemas"]:
        for schema in dm["schemas"]:
            stype = schema.get("type", "relational")
            diag_type = _SCHEMA_TO_DIAGRAM.get(stype)
            if diag_type and diag_type not in rendered_diag_types:
                defn = DiagramAgent.extract_mermaid(r.diagrams, diag_type)
                if defn:
                    title = _DIAGRAM_TITLES.get(diag_type, diag_type.title())
                    st.markdown(f'<p class="slabel">{title}</p>', unsafe_allow_html=True)
                    render_mermaid(defn, title=title)
                    rendered_diag_types.add(diag_type)

    # ── Schemas ───────────────────────────────────────────────────────────────
    if dm["schemas"]:
        st.markdown('<p class="slabel">Schemas</p>', unsafe_allow_html=True)

        # Build a lookup of store→description from strategy entries
        strategy_desc: dict[str, str] = {}
        for e in dm.get("strategy_entries", []):
            dl = e.get("data_lives") or e.get("description", "")
            if dl:
                strategy_desc[e["store"].lower()] = dl

        for schema in dm["schemas"]:
            stype = schema.get("type", "relational")
            entity_label, lang, _ = _SCHEMA_TYPE_META.get(stype, ("Entity", "text", "#64748b"))
            type_label = _TYPE_LABELS.get(stype, stype.title())
            expander_label = f"{schema['store']}  ·  {type_label}  ·  {len(schema['entities'])} {entity_label.lower()}(s)"
            with st.expander(expander_label, expanded=False):
                st.markdown(
                    _type_badge(stype)
                    + f'&nbsp; <span style="color:#94a3b8;font-size:0.82rem">'
                    + f'{len(schema["entities"])} {entity_label.lower()}(s)</span>',
                    unsafe_allow_html=True,
                )
                desc = strategy_desc.get(schema["store"].lower(), "")
                if desc:
                    st.caption(desc[:200])
                st.markdown("")
                for entity in schema["entities"]:
                    st.markdown(f"##### `{entity['name']}`")
                    if entity["ddl"]:
                        st.code(entity["ddl"].strip(), language=lang)
                    if entity["indexes"]:
                        st.caption("Indexes / access patterns")
                        idx_lang = "sql" if stype == "relational" else lang
                        st.code(entity["indexes"].strip(), language=idx_lang)
                    st.divider()

    # ── Notes & trade-offs ────────────────────────────────────────────────────
    if dm["normalization_notes"] or dm["trade_offs"]:
        c1, c2 = st.columns(2)
        if dm["normalization_notes"]:
            with c1.expander("Normalization / partition notes"):
                st.markdown(dm["normalization_notes"])
        if dm["trade_offs"]:
            with c2.expander("CAP theorem & trade-offs"):
                st.markdown(dm["trade_offs"])


def tab_security(r: PipelineResult) -> None:
    st.markdown(r.critique)


def tab_diagrams(r: PipelineResult) -> None:
    available = DiagramAgent.available_types(r.diagrams)

    for dtype in available:
        title = _DIAGRAM_TITLES.get(dtype, dtype.replace("_", " ").title())
        st.markdown(
            f'<div class="diag-card"><div class="diag-title">{title}</div></div>',
            unsafe_allow_html=True,
        )
        definition = DiagramAgent.extract_mermaid(r.diagrams, dtype)
        if definition:
            render_mermaid(definition, title=title)
        elif dtype == "er":
            st.info(
                "No relational schemas — see the **Data Model** tab for schema definitions."
            )
        else:
            st.info("Diagram could not be generated.")


def tab_brief(r: PipelineResult) -> None:
    if "brief" not in st.session_state:
        from run import generate_developer_brief
        with st.spinner("Generating developer brief ..."):
            with tempfile.TemporaryDirectory() as tmp:
                path = generate_developer_brief(r, Path(tmp))
                st.session_state.brief = path.read_text(encoding="utf-8")

    brief = st.session_state.brief
    st.download_button(
        "⬇  Download DEVELOPER_BRIEF.md",
        data=brief, file_name="DEVELOPER_BRIEF.md", mime="text/markdown",
    )
    st.divider()
    st.markdown(brief)


# ── landing page ────────────────────────────────────────────────────────────────

def landing_page() -> None:
    st.markdown("""
    <div class="landing-wrap">
      <div class="landing-logo">⚙️ Architecture Pipeline</div>
      <div class="landing-sub">
        Describe any system. Get a complete architecture package in minutes.
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([0.05, 0.9, 0.05])
    with col:
        scenario = st.text_area(
            "prompt",
            placeholder="Describe the system you want to design — e.g. \"Build a real-time ride-sharing platform for 500k users...\"",
            height=130,
            label_visibility="collapsed",
            key="landing_input",
        )

        c1, c2, c3 = st.columns([2, 1, 1])
        use_rag = c2.toggle("RAG", value=True, help="Retrieve from knowledge base")
        run     = c3.button("Generate →", type="primary", use_container_width=True)

    st.markdown("""
    <div class="feature-row">
      <span class="fpill">SRS · IEEE 29148</span>
      <span class="fpill">Architecture decisions</span>
      <span class="fpill">Data model + DDL</span>
      <span class="fpill">OWASP · NIST CSF 2.0 · STRIDE</span>
      <span class="fpill">3 rendered diagrams</span>
      <span class="fpill">Developer brief</span>
    </div>
    """, unsafe_allow_html=True)

    if run and scenario.strip():
        st.session_state.pop("brief", None)
        st.session_state.scenario = scenario.strip()
        st.session_state.use_rag  = use_rag
        st.session_state.running  = True
        st.rerun()


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── trigger run ──
    if st.session_state.get("running"):
        st.session_state.running = False
        scenario = st.session_state.scenario
        use_rag  = st.session_state.use_rag

        st.markdown(f"**Designing:** {scenario[:120]}{'...' if len(scenario) > 120 else ''}")
        st.session_state.result = run_pipeline(scenario, use_rag)
        st.rerun()

    # ── landing ──
    if "result" not in st.session_state:
        landing_page()
        return

    # ── results ──
    result = st.session_state.result
    tabs = st.tabs(["Overview", "Requirements", "Architecture",
                    "Data Model", "Security", "Dev Brief"])

    # ── sidebar: design a new system ──
    with st.sidebar:
        st.markdown("### Design a new system")
        new_scenario = st.text_area(
            "new_prompt", placeholder="Describe a new system to design...",
            height=160, label_visibility="collapsed", key="reprompt",
        )
        use_rag2 = st.toggle("RAG", value=True, key="rag2")
        if st.button("Generate →", type="primary", use_container_width=True, key="rerun_btn"):
            if new_scenario.strip():
                st.session_state.pop("brief", None)
                st.session_state.pop("result", None)
                st.session_state.scenario = new_scenario.strip()
                st.session_state.use_rag  = use_rag2
                st.session_state.running  = True
                st.rerun()
        st.divider()
        st.caption(f"Current: {st.session_state.get('scenario', '')[:80]}...")

    with tabs[0]: tab_overview(result)
    with tabs[1]: tab_requirements(result)
    with tabs[2]: tab_architecture(result)
    with tabs[3]: tab_data_model(result)
    with tabs[4]: tab_security(result)
    with tabs[5]: tab_brief(result)


if __name__ == "__main__":
    main()
