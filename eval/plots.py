"""
All evaluation plots. Each function saves a PNG to `out_dir` and prints the path.
Uses the Agg backend so no display is required (safe for headless servers).
"""
from pathlib import Path
from math import pi
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.structural import StructuralReport
from eval.retrieval import QueryResult
from eval.llm_judge import RUBRIC

if TYPE_CHECKING:
    from eval.answer_relevance import AnswerRelevanceResult
    from eval.chunker_comparison import ChunkerComparisonResult
    from eval.faithfulness import FaithfulnessResult

# ── colour palette ────────────────────────────────────────────────────────────
RAG_COLOR   = "#4f46e5"   # indigo
NORAG_COLOR = "#94a3b8"   # slate
PASS_COLOR  = "#dc2626"   # red (threshold lines)

COLLECTION_COLORS = {
    "requirements": "#7c3aed",
    "architecture": "#2563eb",
    "data_modeler": "#059669",
    "critic":       "#dc2626",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ── 1. Structural comparison ──────────────────────────────────────────────────

MONO_COLOR = "#f59e0b"   # amber — monolithic baseline


def plot_structural_comparison(
    rag_reports:   list[StructuralReport],
    norag_reports: list[StructuralReport],
    labels:        list[str],
    out_dir:       Path,
    mono_reports:  list[StructuralReport] | None = None,
) -> None:
    """
    2×3 box-plot grid. Each subplot = one structural metric.
    Each box shows the distribution across all scenarios for that mode.
    Works cleanly regardless of how many scenarios there are.
    """
    has_mono = bool(mono_reports)

    metric_fns = [
        ("EARS Compliance Rate",     lambda r: r.srs.ears_rate),
        ("NFR Measurability Rate",   lambda r: r.srs.nfr_measurable_rate),
        ("SRS Section Completeness", lambda r: r.srs.section_completeness),
        ("Arch Decision Quality",    lambda r: r.arch.decision_quality),
        ("Component Count (norm·6)", lambda r: min(r.arch.component_count / 6, 1.0)),
        ("Overall Structural Score", lambda r: r.summary_score()),
    ]

    series = [("RAG", rag_reports, RAG_COLOR), ("No-RAG", norag_reports, NORAG_COLOR)]
    if has_mono:
        series.append(("Monolithic", mono_reports, MONO_COLOR))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (title, fn) in zip(axes, metric_fns):
        data   = [([fn(r) for r in reps] if reps else [0]) for _, reps, _ in series]
        colors = [col for _, _, col in series]
        xlbls  = [lbl for lbl, _, _ in series]

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="white", linewidth=2))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)
        for element in ["whiskers", "caps", "fliers"]:
            for item in bp[element]:
                item.set(color="#555555", linewidth=1)

        # Annotate medians
        for i, d in enumerate(data):
            med = float(np.median(d))
            ax.text(i + 1, med + 0.03, f"{med:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(range(1, len(series) + 1))
        ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Score (0–1)", fontsize=8)
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)

    title_str = "Structural Metrics: RAG vs No-RAG vs Monolithic" if has_mono else "Structural Metrics: RAG vs No-RAG"
    n = len(rag_reports)
    fig.suptitle(f"{title_str}  (n={n} scenarios)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "structural_comparison.png")


# ── 2. Retrieval precision ────────────────────────────────────────────────────

def plot_retrieval_precision(
    retrieval_results: dict[int, list[QueryResult]],
    out_dir:           Path,
) -> None:
    """
    Two-panel figure:
      Top panel    — in-domain queries (expect high precision)
      Bottom panel — hard-negative queries (expect low precision)
    Each panel shows P@3 and P@5 per query, with collection separators.
    """
    if not retrieval_results:
        return

    k_values = sorted(retrieval_results.keys())
    palette  = [RAG_COLOR, "#06b6d4"]

    for kind, title_suffix, fname_suffix in [
        ("in_domain", "In-Domain Queries (expect HIGH precision)",  "indomain"),
        ("hard_neg",  "Hard-Negative Queries (expect LOW precision)", "hardneg"),
    ]:
        base_rows = [r for r in retrieval_results[k_values[0]] if r.kind == kind]
        if not base_rows:
            continue

        # Sort by collection so separator lines work
        base_rows = sorted(base_rows, key=lambda r: r.collection)
        descs       = [r.description for r in base_rows]
        colls       = [r.collection  for r in base_rows]

        # Build a lookup: description -> precision for each k
        prec_by_k: dict[int, dict[str, float]] = {}
        for k in k_values:
            prec_by_k[k] = {
                r.description: r.precision
                for r in retrieval_results[k] if r.kind == kind
            }

        x     = np.arange(len(descs))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(12, len(descs) * 0.9), 6))

        for i, k in enumerate(k_values):
            precisions = [prec_by_k[k].get(d, 0.0) for d in descs]
            bars = ax.bar(x + (i - len(k_values) / 2 + 0.5) * width,
                          precisions, width, label=f"P@{k}",
                          color=palette[i], alpha=0.85)
            for bar, val in zip(bars, precisions):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        # Collection separators and labels
        prev, coll_starts = None, {}
        for i, coll in enumerate(colls):
            if coll not in coll_starts:
                coll_starts[coll] = i
            if prev and coll != prev:
                ax.axvline(i - 0.5, color="gray", linewidth=1, linestyle=":", alpha=0.5)
            prev = coll
        for coll, start in coll_starts.items():
            ax.text(start, 1.14, coll.replace("_", " ").title(),
                    fontsize=8, color=COLLECTION_COLORS.get(coll, "gray"),
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(descs, rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, 1.3)
        ax.set_ylabel("LLM-judged Precision", fontsize=10)
        ax.set_title(f"Retrieval Precision — {title_suffix}", fontsize=12, fontweight="bold")
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        _save(fig, out_dir / f"retrieval_precision_{fname_suffix}.png")


def plot_retrieval_by_collection(
    retrieval_results: dict[int, list[QueryResult]],
    out_dir:           Path,
) -> None:
    """
    Side-by-side bars: in-domain vs hard-negative mean P@5 per collection.
    The gap between the two bars shows how well the collections are discriminated.
    """
    if not retrieval_results:
        return

    k = max(retrieval_results.keys())   # use the larger k
    coll_names = sorted({r.collection for r in retrieval_results[k]})

    in_means  = []
    neg_means = []
    for coll in coll_names:
        in_rows  = [r.precision for r in retrieval_results[k]
                    if r.collection == coll and r.kind == "in_domain"]
        neg_rows = [r.precision for r in retrieval_results[k]
                    if r.collection == coll and r.kind == "hard_neg"]
        in_means.append(sum(in_rows)  / len(in_rows)  if in_rows  else 0.0)
        neg_means.append(sum(neg_rows) / len(neg_rows) if neg_rows else 0.0)

    x     = np.arange(len(coll_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_in  = ax.bar(x - width / 2, in_means,  width, label="In-domain",    color=RAG_COLOR,   alpha=0.85)
    bars_neg = ax.bar(x + width / 2, neg_means, width, label="Hard-negative", color=PASS_COLOR,  alpha=0.65)

    for bars, vals in [(bars_in, in_means), (bars_neg, neg_means)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in coll_names], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(f"Mean P@{k} (LLM-judged)", fontsize=10)
    ax.set_title(
        f"In-Domain vs Hard-Negative Precision per Collection (P@{k})\n"
        "Large gap = collections are semantically distinct ✓",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir / "retrieval_by_collection.png")


# ── 3. Critic scores ──────────────────────────────────────────────────────────

def plot_critic_scores(
    rag_scores:   list[int],
    norag_scores: list[int],
    labels:       list[str],
    out_dir:      Path,
) -> None:
    """Bar chart of internal CriticAgent scores with pass-threshold line."""
    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - width / 2, rag_scores,   width, label="RAG",    color=RAG_COLOR,   alpha=0.85)
    bars_n = ax.bar(x + width / 2, norag_scores, width, label="No-RAG", color=NORAG_COLOR, alpha=0.85)

    for bars in [bars_r, bars_n]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(7, color=PASS_COLOR, linewidth=1.5, linestyle="--", label="Pass threshold (7)")
    ax.set_ylim(0, 11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Critic Score (0–10)")
    ax.set_title("Internal Critic Scores: RAG vs No-RAG", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir / "critic_scores.png")


# ── 4. LLM judge – total scores ───────────────────────────────────────────────

def plot_judge_scores(
    judge_rag:   list[dict],
    judge_norag: list[dict],
    labels:      list[str],
    out_dir:     Path,
) -> None:
    """Bar chart of total LLM judge scores per scenario."""
    if not judge_rag:
        return

    max_score = judge_rag[0]["max"]
    x         = np.arange(len(labels))
    width     = 0.35

    rag_totals   = [r["total"] for r in judge_rag]
    norag_totals = [r["total"] for r in judge_norag]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - width / 2, rag_totals,   width, label="RAG",    color=RAG_COLOR,   alpha=0.85)
    bars_n = ax.bar(x + width / 2, norag_totals, width, label="No-RAG", color=NORAG_COLOR, alpha=0.85)

    for bars in [bars_r, bars_n]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{bar.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, max_score + 2)
    ax.axhline(max_score, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel(f"Judge Score (0–{max_score})")
    ax.set_title("LLM-as-Judge: Total Score by Scenario", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir / "judge_scores.png")


# ── 5. Radar chart ────────────────────────────────────────────────────────────

def plot_radar(
    judge_rag:   list[dict],
    judge_norag: list[dict],
    out_dir:     Path,
    judge_mono:  list[dict] | None = None,
) -> None:
    """
    Radar/spider chart: mean normalised score per rubric dimension,
    averaged across all scenarios. Shows RAG / No-RAG / Monolithic.
    """
    if not judge_rag:
        return

    dim_ids    = [d["id"]        for d in RUBRIC]
    dim_labels = [d["name"]      for d in RUBRIC]
    max_scores = [d["max_score"] for d in RUBRIC]
    N          = len(RUBRIC)

    def _mean_norm(results: list[dict]) -> list[float]:
        if not results:
            return [0.0] * N
        return [
            sum(r["scores"].get(did, 0) for r in results) / (len(results) * mx)
            for did, mx in zip(dim_ids, max_scores)
        ]

    angles = [n / N * 2 * pi for n in range(N)] + [0]

    def _close(vals): return vals + vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    rag_vals = _mean_norm(judge_rag)
    ax.plot(angles, _close(rag_vals),   "o-",  linewidth=2,  color=RAG_COLOR,   label="RAG (pipeline + retrieval)")
    ax.fill(angles, _close(rag_vals),          alpha=0.18,   color=RAG_COLOR)

    norag_vals = _mean_norm(judge_norag)
    ax.plot(angles, _close(norag_vals), "o--", linewidth=1.5, color=NORAG_COLOR, label="No-RAG (pipeline only)")
    ax.fill(angles, _close(norag_vals),        alpha=0.10,   color=NORAG_COLOR)

    if judge_mono:
        mono_vals = _mean_norm(judge_mono)
        ax.plot(angles, _close(mono_vals), "s:",  linewidth=1.5, color=MONO_COLOR,  label="Monolithic (single prompt)")
        ax.fill(angles, _close(mono_vals),        alpha=0.08,   color=MONO_COLOR)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], size=7)
    title = "LLM-as-Judge: RAG vs No-RAG vs Monolithic" if judge_mono else "LLM-as-Judge: RAG vs No-RAG"
    ax.set_title(
        f"{title}\n(mean normalised score per dimension)",
        size=12, fontweight="bold", pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15))

    plt.tight_layout()
    _save(fig, out_dir / "radar_chart.png")


# ── 6. Summary heat-map ───────────────────────────────────────────────────────

def plot_summary_heatmap(
    rag_reports:   list[StructuralReport],
    norag_reports: list[StructuralReport],
    labels:        list[str],
    out_dir:       Path,
    mono_reports:  list[StructuralReport] | None = None,
) -> None:
    """
    Compact aggregate heatmap: rows = metrics, columns = modes (3 max).
    Each cell shows mean ± std across all scenarios — stays readable at any scale.
    """
    metric_names = [
        "EARS rate", "NFR measurable", "Section complete",
        "Decision quality", "Components (norm)", "Diagram present", "Overall",
    ]

    metric_fns = [
        lambda r: r.srs.ears_rate,
        lambda r: r.srs.nfr_measurable_rate,
        lambda r: r.srs.section_completeness,
        lambda r: r.arch.decision_quality,
        lambda r: min(r.arch.component_count / 6, 1.0),
        lambda r: float(r.diagrams.arch_present),
        lambda r: r.summary_score(),
    ]

    modes = [("RAG", rag_reports), ("No-RAG", norag_reports)]
    if mono_reports:
        modes.append(("Monolithic", mono_reports))

    n_metrics = len(metric_names)
    n_modes   = len(modes)

    means = np.zeros((n_metrics, n_modes))
    stds  = np.zeros((n_metrics, n_modes))

    for j, (_, reports) in enumerate(modes):
        for i, fn in enumerate(metric_fns):
            vals = [fn(r) for r in reports]
            means[i, j] = np.mean(vals)
            stds[i, j]  = np.std(vals)

    fig, ax = plt.subplots(figsize=(max(5, n_modes * 2.5), 5))
    im = ax.imshow(means, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    col_labels = [m[0] for m in modes]
    ax.set_xticks(range(n_modes))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metric_names, fontsize=10)

    for i in range(n_metrics):
        for j in range(n_modes):
            val = means[i, j]
            std = stds[i, j]
            txt_color = "white" if val < 0.25 or val > 0.80 else "black"
            ax.text(j, i, f"{val:.2f}\n±{std:.2f}",
                    ha="center", va="center", fontsize=9,
                    color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    n = len(rag_reports)
    title = "Structural Metrics — Mean ± Std Across All Scenarios"
    if mono_reports:
        title += "\nRAG vs No-RAG vs Monolithic"
    ax.set_title(f"{title}  (n={n})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "heatmap.png")


# ── 7. Faithfulness ───────────────────────────────────────────────────────────

def plot_faithfulness(
    rag_results:   dict,   # dict[str, FaithfulnessResult]
    norag_results: dict,
    out_dir:       Path,
) -> None:
    """Bar chart comparing faithfulness scores RAG vs No-RAG per output type."""
    labels = sorted(set(rag_results) | set(norag_results))
    if not labels:
        return

    rag_scores   = [rag_results[l].score   if l in rag_results   else 0.0 for l in labels]
    norag_scores = [norag_results[l].score if l in norag_results else 0.0 for l in labels]

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - width / 2, rag_scores,   width, label="RAG",    color=RAG_COLOR,   alpha=0.85)
    bars_n = ax.bar(x + width / 2, norag_scores, width, label="No-RAG", color=NORAG_COLOR, alpha=0.85)

    for bars in [bars_r, bars_n]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Faithfulness Score (0–1)")
    ax.set_title(
        "RAGAS Faithfulness: RAG vs No-RAG\n"
        "(fraction of claims supported by expert corpus)",
        fontsize=12, fontweight="bold",
    )
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _save(fig, out_dir / "faithfulness.png")


# ── 8. Answer Relevance ───────────────────────────────────────────────────────

def plot_answer_relevance(
    rag_results:   list,   # list[AnswerRelevanceResult]
    norag_results: list,
    scenario_labels: list[str],
    out_dir:         Path,
) -> None:
    """Bar chart comparing answer relevance scores RAG vs No-RAG per scenario."""
    if not rag_results:
        return

    rag_scores   = [r.score for r in rag_results]
    norag_scores = [r.score for r in norag_results]

    x     = np.arange(len(scenario_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.bar(x - width / 2, rag_scores,   width, label="RAG",    color=RAG_COLOR,   alpha=0.85)
    bars_n = ax.bar(x + width / 2, norag_scores, width, label="No-RAG", color=NORAG_COLOR, alpha=0.85)

    for bars in [bars_r, bars_n]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.2)
    ax.axhline(0.90, color="#16a34a", linewidth=1.2, linestyle="--", alpha=0.5, label="Excellent (0.90)")
    ax.axhline(0.75, color="#ca8a04", linewidth=1.2, linestyle=":",  alpha=0.5, label="Acceptable (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=10)
    ax.set_ylabel("Answer Relevance (0–1)")
    ax.set_title(
        "Answer Relevance: RAG vs No-RAG\n"
        "(fraction of prompt requirements addressed in SRS)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, out_dir / "answer_relevance.png")


# ── 9. Chunker comparison ─────────────────────────────────────────────────────

def plot_chunker_comparison(
    results: list,   # list[ChunkerComparisonResult]
    out_dir: Path,
) -> None:
    """Scatter/bar comparing semantic vs fixed-size chunker P@k per query."""
    if not results:
        return

    descriptions = [r.query_description for r in results]
    sem_vals     = [r.semantic_precision for r in results]
    fix_vals     = [r.fixed_precision    for r in results]
    x = np.arange(len(descriptions))
    k = results[0].k if results else 5

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(results) * 0.8), 10))

    # Top: side-by-side bars per query
    ax = axes[0]
    width = 0.35
    bars_s = ax.bar(x - width / 2, sem_vals, width, label="Semantic",   color=RAG_COLOR,   alpha=0.85)
    bars_f = ax.bar(x + width / 2, fix_vals, width, label="Fixed-size", color=NORAG_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(descriptions, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(f"P@{k}")
    ax.set_title(f"Chunker Comparison: Semantic vs Fixed-size (P@{k})", fontsize=12, fontweight="bold")
    ax.legend()

    # Bottom: delta per query (positive = semantic wins)
    ax2 = axes[1]
    deltas = [r.delta for r in results]
    colors = [RAG_COLOR if d >= 0 else PASS_COLOR for d in deltas]
    ax2.bar(x, deltas, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(descriptions, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("Δ P@k  (Semantic − Fixed)")
    ax2.set_title("Per-query delta (blue = semantic wins, red = fixed wins)", fontsize=10)

    # Summary text
    sem_mean = sum(sem_vals) / len(sem_vals)
    fix_mean = sum(fix_vals) / len(fix_vals)
    ax2.text(0.98, 0.95,
             f"Mean: Semantic={sem_mean:.3f}  Fixed={fix_mean:.3f}  Δ={sem_mean - fix_mean:+.3f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    _save(fig, out_dir / "chunker_comparison.png")


# ── 10. Monolithic baseline comparison ───────────────────────────────────────

def plot_monolithic_comparison(
    pipeline_struct:    list[StructuralReport],   # RAG pipeline results
    monolithic_struct:  list[StructuralReport],
    scenario_labels:    list[str],
    judge_pipeline:     list[dict],
    judge_monolithic:   list[dict],
    out_dir:            Path,
) -> None:
    """
    Side-by-side comparison of multi-agent pipeline vs monolithic baseline
    across structural score and judge total.
    """
    x     = np.arange(len(scenario_labels))
    width = 0.35
    MONO_COLOR = "#f59e0b"  # amber

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: structural scores
    ax = axes[0]
    p_scores = [r.summary_score() for r in pipeline_struct]
    m_scores = [r.summary_score() for r in monolithic_struct]
    ax.bar(x - width / 2, p_scores, width, label="Multi-agent (RAG)", color=RAG_COLOR,  alpha=0.85)
    ax.bar(x + width / 2, m_scores, width, label="Monolithic",        color=MONO_COLOR, alpha=0.85)
    for bars, vals in [(ax.patches[:len(x)], p_scores), (ax.patches[len(x):], m_scores)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Structural Score (0–1)")
    ax.set_title("Structural Quality: Pipeline vs Monolithic", fontsize=11, fontweight="bold")
    ax.legend()

    # Right: judge scores
    ax2 = axes[1]
    if judge_pipeline and judge_monolithic:
        pj = [r["total"] for r in judge_pipeline]
        mj = [r["total"] for r in judge_monolithic]
        max_score = judge_pipeline[0]["max"]
        ax2.bar(x - width / 2, pj, width, label="Multi-agent (RAG)", color=RAG_COLOR,  alpha=0.85)
        ax2.bar(x + width / 2, mj, width, label="Monolithic",        color=MONO_COLOR, alpha=0.85)
        for vals, offset in [(pj, -width / 2), (mj, width / 2)]:
            for i, val in enumerate(vals):
                ax2.text(i + offset, val + 0.1, f"{val}", ha="center", va="bottom", fontsize=9)
        ax2.set_ylim(0, max_score + 2)
        ax2.set_ylabel(f"Judge Score (0–{max_score})")
        ax2.set_title("LLM Judge: Pipeline vs Monolithic", fontsize=11, fontweight="bold")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Judge scores not available",
                 ha="center", va="center", transform=ax2.transAxes)

    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels, rotation=10)

    fig.suptitle("RQ3: Multi-agent Pipeline vs Monolithic Baseline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "monolithic_comparison.png")
