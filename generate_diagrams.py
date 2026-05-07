"""
Generate architecture diagrams for the report.
Produces three PNGs in ./diagrams/:
  1. data_pipeline.png   — corpus ingestion & semantic chunking
  2. rag_retrieval.png   — runtime RAG retrieval flow
  3. agent_pipeline.png  — multi-agent loop with what is passed between agents

Run:
    python generate_diagrams.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT = Path("diagrams")
OUT.mkdir(exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
C = dict(
    blue    = "#3B82F6",
    indigo  = "#4F46E5",
    green   = "#10B981",
    amber   = "#F59E0B",
    red     = "#EF4444",
    purple  = "#8B5CF6",
    slate   = "#64748B",
    bg      = "#F8FAFC",
    white   = "#FFFFFF",
    dark    = "#1E293B",
    lblue   = "#DBEAFE",
    lgreen  = "#D1FAE5",
    lamber  = "#FEF3C7",
    lpurple = "#EDE9FE",
    lred    = "#FEE2E2",
    lgray   = "#F1F5F9",
)


def _box(ax, x, y, w, h, text, color, textcolor=C["white"], fontsize=9,
         radius=0.04, bold=False, subtext=None, subtextsize=7.5):
    """Draw a rounded rectangle with centered label (and optional subtext)."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         linewidth=1.2, edgecolor=color,
                         facecolor=color, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x, y + 0.012, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight,
                zorder=4, linespacing=1.3)
        ax.text(x, y - h*0.22, subtext, ha="center", va="center",
                fontsize=subtextsize, color=textcolor, alpha=0.88,
                zorder=4, style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight,
                zorder=4, linespacing=1.3)


def _lightbox(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.5,
              textcolor=C["dark"], bold=False, subtext=None, subtextsize=7):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.01,rounding_size=0.03",
                         linewidth=1.4, edgecolor=edgecolor,
                         facecolor=facecolor, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x, y + h*0.12, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)
        ax.text(x, y - h*0.2, subtext, ha="center", va="center",
                fontsize=subtextsize, color=C["slate"], zorder=4,
                style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color=C["slate"], label="", lw=1.5,
           labelside="top", fontsize=7.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dy = 0.022 if labelside == "top" else -0.022
        ax.text(mx, my + dy, label, ha="center", va="center",
                fontsize=fontsize, color=color,
                bbox=dict(fc="white", ec="none", pad=1.5))


def _section_bg(ax, x, y, w, h, color, title, fontsize=8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.005,rounding_size=0.03",
                          linewidth=1, edgecolor=color,
                          facecolor=color + "18", zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.012, y + h - 0.018, title,
            ha="left", va="top", fontsize=fontsize,
            color=color, fontweight="bold", zorder=2)


def _formula_box(ax, x, y, formula, fontsize=8.5):
    ax.text(x, y, formula, ha="center", va="center",
            fontsize=fontsize, color=C["dark"],
            fontfamily="monospace",
            bbox=dict(fc="#F8FAFC", ec=C["slate"], lw=0.8,
                      pad=4, boxstyle="round,pad=0.3"),
            zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1 — Corpus ingestion & semantic chunking
# ══════════════════════════════════════════════════════════════════════════════

def diagram_data_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle("Stage 1 — Corpus Ingestion & Semantic Chunking",
                 fontsize=13, fontweight="bold", color=C["dark"], y=0.97)

    # ── Stage A background: ingestion ──────────────────────────────────────
    _section_bg(ax, 0.01, 0.05, 0.30, 0.85, C["blue"], "A  Knowledge Base")
    _section_bg(ax, 0.33, 0.05, 0.37, 0.85, C["indigo"], "B  Semantic Chunker")
    _section_bg(ax, 0.72, 0.05, 0.27, 0.85, C["green"], "C  Vector Store")

    # ── A: four corpus boxes ───────────────────────────────────────────────
    corpora = [
        ("requirements",  "IEEE 29148 · EARS · ISO 25010", C["blue"]),
        ("architecture",  "AWS WAF · Microservices · CQRS", C["indigo"]),
        ("data_modeler",  "CAP theorem · 3NF · DB patterns", C["purple"]),
        ("critic",        "OWASP · NIST CSF · STRIDE",       C["red"]),
    ]
    ys = [0.78, 0.58, 0.38, 0.18]
    for (name, sub, col), y in zip(corpora, ys):
        _lightbox(ax, 0.16, y, 0.26, 0.14,
                  f"'{name}' corpus", C["white"], col,
                  fontsize=8, bold=True, textcolor=col,
                  subtext=sub, subtextsize=6.8)

    # ── B: chunker steps ───────────────────────────────────────────────────
    steps = [
        (0.515, 0.78, "Pass 1 — Structural Split",
         "Split at paragraph boundaries (\\n\\n)"),
        (0.515, 0.555, "Pass 2 — Cosine Merge",
         "Merge adjacent paragraphs if sim ≥ τ"),
        (0.515, 0.33, "Pass 3 — Overflow Split",
         "Split chunks > 800 chars at sentence boundary"),
    ]
    for (bx, by, title, sub) in steps:
        _lightbox(ax, bx, by, 0.32, 0.13,
                  title, C["lpurple"], C["indigo"],
                  fontsize=8, bold=True, textcolor=C["indigo"],
                  subtext=sub, subtextsize=7)

    # Formulas between passes
    _formula_box(ax, 0.515, 0.665,
                 "sim(pᵢ, pᵢ₊₁) = (eᵢ · eᵢ₊₁) / (‖eᵢ‖ ‖eᵢ₊₁‖)  ≥  τ = 0.72")
    _formula_box(ax, 0.515, 0.44,
                 "ē_merged = (ē_cur + ē_new) / 2        max = 800 chars")

    # arrows between passes
    _arrow(ax, 0.515, 0.715, 0.515, 0.695, C["indigo"])
    _arrow(ax, 0.515, 0.618, 0.515, 0.598, C["indigo"])
    _arrow(ax, 0.515, 0.492, 0.515, 0.468, C["indigo"])
    _arrow(ax, 0.515, 0.265, 0.515, 0.245, C["indigo"])

    # embedding label below pass 3
    ax.text(0.515, 0.19, "BAAI/bge-small-en-v1.5\n384-dim embeddings",
            ha="center", va="center", fontsize=8, color=C["indigo"],
            style="italic")

    # ── C: ChromaDB ────────────────────────────────────────────────────────
    _box(ax, 0.855, 0.58, 0.22, 0.22,
         "ChromaDB\nVector Store", C["green"],
         fontsize=9, bold=True)
    ax.text(0.855, 0.43, "4 collections\n~500 chunks each\nHNSW index",
            ha="center", va="center", fontsize=7.5, color=C["green"],
            style="italic")

    # arrows A → B
    for y in ys:
        _arrow(ax, 0.295, y, 0.35, 0.58, C["blue"], lw=1.2)

    # arrow B → C
    _arrow(ax, 0.68, 0.58, 0.735, 0.58, C["indigo"], label="chunks + embeddings", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "data_pipeline.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2 — Runtime RAG retrieval
# ══════════════════════════════════════════════════════════════════════════════

def diagram_rag_retrieval():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle("Stage 2 — Runtime RAG Retrieval (per agent)",
                 fontsize=13, fontweight="bold", color=C["dark"], y=0.97)

    # User brief
    _box(ax, 0.08, 0.5, 0.13, 0.18, "User Brief\n+\nPrior Outputs",
         C["slate"], fontsize=8.5, bold=True)

    # 4 query box
    _section_bg(ax, 0.185, 0.18, 0.19, 0.64, C["blue"], "4 Parallel Queries")
    qcolors = [C["blue"], C["indigo"], C["purple"], C["slate"]]
    qlabels = ["Broad query", "Domain-focused", "Constraint-driven", "Context-grounded"]
    qys = [0.76, 0.60, 0.44, 0.28]
    for col, lbl, qy in zip(qcolors, qlabels, qys):
        _lightbox(ax, 0.285, qy, 0.155, 0.10, lbl,
                  C["lblue"], col, fontsize=7.5, textcolor=col, bold=True)

    _arrow(ax, 0.145, 0.5, 0.205, 0.5, C["slate"], lw=1.5)

    # Embedding model
    _box(ax, 0.44, 0.5, 0.11, 0.18,
         "Embedding\nModel\nbge-small", C["indigo"], fontsize=8, bold=True)
    for qy in qys:
        _arrow(ax, 0.363, qy, 0.41, 0.5, C["blue"], lw=1)

    # Formula
    _formula_box(ax, 0.44, 0.23,
                 "q_emb = embed(query)  ∈ ℝ³⁸⁴")

    # Vector DB
    _box(ax, 0.60, 0.5, 0.11, 0.18,
         "ChromaDB\nHNSW\nSearch", C["green"], fontsize=8, bold=True)
    _arrow(ax, 0.495, 0.5, 0.54, 0.5, C["indigo"],
           label="query vector", fontsize=7)

    _formula_box(ax, 0.60, 0.23,
                 "score(q,c) = (q·c)/(‖q‖‖c‖)   →   top-K chunks")

    # Dedup box
    _lightbox(ax, 0.755, 0.5, 0.13, 0.18,
              "Deduplicate\nby chunk ID",
              C["lamber"], C["amber"], fontsize=8,
              textcolor=C["amber"], bold=True,
              subtext="4×K → unique set", subtextsize=7)
    _arrow(ax, 0.655, 0.5, 0.69, 0.5, C["green"],
           label="top-K × 4", fontsize=7)
    _arrow(ax, 0.82, 0.5, 0.855, 0.5, C["amber"], lw=1.5)

    # Agent prompt
    _box(ax, 0.925, 0.5, 0.13, 0.30,
         "Agent\nPrompt", C["blue"], fontsize=8.5, bold=True,
         subtext="<retrieved_context>\n..chunks..\n</retrieved_context>",
         subtextsize=6.5)

    ax.text(0.5, 0.08,
            "Each agent independently queries its own collection. "
            "4 query strategies maximise recall across diverse phrasings.",
            ha="center", va="center", fontsize=8, color=C["slate"],
            style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "rag_retrieval.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3 — Multi-agent pipeline with critique loop
# ══════════════════════════════════════════════════════════════════════════════

def diagram_agent_pipeline():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle("Stage 3 — Multi-Agent Pipeline & Critique Loop",
                 fontsize=13, fontweight="bold", color=C["dark"], y=0.98)

    # ── input ──────────────────────────────────────────────────────────────
    _box(ax, 0.06, 0.72, 0.09, 0.16,
         "User\nBrief", C["slate"], fontsize=9, bold=True)

    # ── 5 agents ───────────────────────────────────────────────────────────
    agents = [
        (0.21, 0.72, "Requirements\nAgent",   C["blue"],   "requirements\ncorpus"),
        (0.37, 0.72, "Architecture\nAgent",   C["indigo"], "architecture\ncorpus"),
        (0.53, 0.72, "DataModeler\nAgent",    C["purple"], "data_modeler\ncorpus"),
        (0.69, 0.72, "Diagram\nAgent",        C["amber"],  None),
        (0.85, 0.72, "Critic\nAgent",         C["red"],    "critic\ncorpus"),
    ]
    for (ax_, ay, name, col, corpus) in agents:
        _box(ax, ax_, ay, 0.12, 0.16, name, col, fontsize=8.5, bold=True)
        if corpus:
            _lightbox(ax, ax_, ay - 0.20, 0.11, 0.09,
                      corpus, C["lgray"], col,
                      fontsize=7, textcolor=col, bold=False)
            _arrow(ax, ax_, ay - 0.155, ax_, ay - 0.082, col, lw=1)

    # RAG label
    ax.text(0.5, 0.46, "↑  RAG retrieval (4 parallel queries per agent)  ↑",
            ha="center", va="center", fontsize=7.5, color=C["slate"],
            style="italic")

    # arrows between agents — top row
    arrow_labels = [
        "SRS.md",
        "SRS.md\n+ Arch.xml",
        "SRS.md + Arch.xml\n+ DataModel.xml",
        "All 3 docs",
    ]
    xs = [0.155, 0.31, 0.465, 0.625, 0.79]
    for i, lbl in enumerate(arrow_labels):
        _arrow(ax, xs[i]+0.005, 0.72, xs[i+1]-0.005, 0.72,
               C["dark"], label=lbl, fontsize=6.5, lw=1.8)

    # user brief → requirements
    _arrow(ax, 0.105, 0.72, 0.15, 0.72, C["slate"], lw=1.8)

    # ── outputs ────────────────────────────────────────────────────────────
    outputs = [
        (0.21, "SRS\n(Markdown)"),
        (0.37, "Architecture\n(XML)"),
        (0.53, "Data Model\n(XML)"),
        (0.69, "Diagrams\n(XML)"),
        (0.85, "Critique\n(Markdown)\n+ Score /10"),
    ]
    out_colors = [C["blue"], C["indigo"], C["purple"], C["amber"], C["red"]]
    for (ox, olbl), col in zip(outputs, out_colors):
        _lightbox(ax, ox, 0.895, 0.12, 0.12, olbl,
                  C["white"], col, fontsize=7.5, textcolor=col, bold=True)
        _arrow(ax, ox, 0.80, ox, 0.835, col, lw=1.2)

    # ── critique loop ──────────────────────────────────────────────────────
    # score < 7 box
    _section_bg(ax, 0.10, 0.02, 0.82, 0.36, C["red"], "  Critique Loop  (Round 2 if score < θ = 7 / 10)")

    _lightbox(ax, 0.85, 0.19, 0.14, 0.12,
              "score < 7?", C["lred"], C["red"],
              fontsize=8.5, textcolor=C["red"], bold=True)

    _arrow(ax, 0.85, 0.64, 0.85, 0.25, C["red"], label="score", lw=1.5)

    # extract action items
    _lightbox(ax, 0.63, 0.19, 0.16, 0.12,
              "Extract\nAction Items",
              C["lamber"], C["amber"],
              fontsize=8, textcolor=C["amber"], bold=True)
    _arrow(ax, 0.775, 0.19, 0.71, 0.19, C["red"], label="YES", fontsize=7.5)

    # reinject arrows
    _lightbox(ax, 0.40, 0.19, 0.16, 0.12,
              "Reinject as\nprior_critique",
              C["lpurple"], C["indigo"],
              fontsize=8, textcolor=C["indigo"], bold=True)
    _arrow(ax, 0.55, 0.19, 0.48, 0.19, C["amber"], lw=1.5)

    # arch + datamodeler re-run
    _lightbox(ax, 0.22, 0.19, 0.13, 0.12,
              "Arch &\nDataModeler\nRe-run",
              C["lblue"], C["blue"],
              fontsize=7.5, textcolor=C["blue"], bold=True)
    _arrow(ax, 0.32, 0.19, 0.285, 0.19, C["indigo"], lw=1.5)

    # back up to agent row
    _arrow(ax, 0.22, 0.25, 0.37, 0.64, C["blue"], lw=1.2,
           label="Round 2", fontsize=7)
    _arrow(ax, 0.22, 0.25, 0.53, 0.64, C["blue"], lw=1.2)

    # NO → final output
    _lightbox(ax, 0.85, 0.065, 0.13, 0.09,
              "✓  Final Output\n(score ≥ 7)",
              C["lgreen"], C["green"],
              fontsize=8, textcolor=C["green"], bold=True)
    _arrow(ax, 0.85, 0.13, 0.85, 0.11, C["green"], label="NO", fontsize=7.5)

    # max rounds note
    ax.text(0.50, 0.045,
            "Maximum 2 rounds. Final artefacts: SRS.md  +  Architecture.xml  "
            "+  DataModel.xml  +  Diagrams.xml  +  Critique.md",
            ha="center", va="center", fontsize=7.5,
            color=C["dark"], style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUT / "agent_pipeline.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating diagrams ...")
    diagram_data_pipeline()
    diagram_rag_retrieval()
    diagram_agent_pipeline()
    print(f"\nAll diagrams saved to ./{OUT}/")
