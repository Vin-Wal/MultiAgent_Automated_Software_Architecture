"""
Evaluation orchestrator.

Runs all evaluations and produces plots + JSON result files:

  1. Retrieval precision  — no pipeline run needed, uses the indexed ChromaDB
  2. Chunker comparison   — semantic vs fixed-size chunker (optional)
  3. Pipeline runs        — RAG vs No-RAG vs Monolithic, N scenarios (parallelised)
  4. Structural metrics   — parses pipeline output text, no LLM calls (all scenarios)
  5. Faithfulness         — RAGAS-style, sampled subset
  6. Answer relevance     — fraction of prompt requirements in SRS, all scenarios
  7. LLM-as-judge         — sampled subset

Usage
-----
# Generate 80 scenarios then run everything
python -m eval.run_eval --scenarios 80

# Use cached scenarios + cached pipeline outputs (re-run metrics only)
python -m eval.run_eval --scenarios 80 --skip-pipeline

# Retrieval only (no pipeline needed)
python -m eval.run_eval --retrieval-only

# Skip LLM judge (saves API calls)
python -m eval.run_eval --scenarios 80 --no-judge

# Skip monolithic baseline
python -m eval.run_eval --scenarios 80 --no-monolithic

# Custom output directory
python -m eval.run_eval --scenarios 80 --out-dir my_eval_run
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BAR = "━" * 58

# How many scenarios to send to the LLM judge / faithfulness (expensive)
JUDGE_SAMPLE    = 20
FAITH_SAMPLE    = 20


# ── lazy imports ──────────────────────────────────────────────────────────────

def _import_all():
    global Pipeline, PipelineResult, score_all, run_retrieval_eval
    global mean_precision, per_collection_mean, judge_output
    global plots, RUBRIC, SCENARIOS, build_all_collections
    global compute_faithfulness, mean_faithfulness
    global compute_answer_relevance, mean_answer_relevance
    global run_chunker_comparison, mean_precision_by_chunker
    global run_monolithic, load_or_generate

    from pipeline import Pipeline, PipelineResult                    # noqa: F401
    from eval.structural import score_all                            # noqa: F401
    from eval.retrieval import (                                     # noqa: F401
        run_retrieval_eval, mean_precision, per_collection_mean,
    )
    from eval.llm_judge import judge_output, RUBRIC                  # noqa: F401
    from eval import plots                                           # noqa: F401
    from eval.scenarios import SCENARIOS                             # noqa: F401
    from rag.vector_store import build_all_collections               # noqa: F401
    from eval.faithfulness import compute_faithfulness, mean_faithfulness  # noqa: F401
    from eval.answer_relevance import (                              # noqa: F401
        compute_answer_relevance, mean_answer_relevance,
    )
    from eval.chunker_comparison import (                            # noqa: F401
        run_chunker_comparison, mean_precision_by_chunker,
    )
    from eval.monolithic import run_monolithic                       # noqa: F401
    from eval.scenario_generator import load_or_generate             # noqa: F401


# ── pipeline helpers ──────────────────────────────────────────────────────────

def _run_one(scenario: dict, out_dir: Path, run_mono: bool):
    """Run RAG, no-RAG, and optionally monolithic for one scenario."""
    pipeline = Pipeline()
    r_rag   = pipeline.run(scenario["input"], use_rag=True)
    r_norag = pipeline.run(scenario["input"], use_rag=False)
    _save_result(r_rag,   out_dir / scenario["name"] / "rag")
    _save_result(r_norag, out_dir / scenario["name"] / "norag")

    r_mono = None
    if run_mono:
        r_mono = run_monolithic(scenario["input"])
        _save_mono(r_mono, out_dir / scenario["name"] / "monolithic")

    return scenario["name"], r_rag, r_norag, r_mono


def _save_result(result, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "requirements.md").write_text(result.requirements,  encoding="utf-8")
    (folder / "architecture.xml").write_text(result.architecture, encoding="utf-8")
    (folder / "data_model.xml").write_text(result.data_model,     encoding="utf-8")
    (folder / "critique.md").write_text(result.critique,          encoding="utf-8")
    (folder / "diagrams.xml").write_text(result.diagrams,         encoding="utf-8")
    meta = {"score": result.score, "rounds": result.rounds, "errors": result.errors}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_mono(result, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "requirements.md").write_text(result.requirements, encoding="utf-8")
    (folder / "architecture.xml").write_text(result.architecture, encoding="utf-8")
    (folder / "data_model.xml").write_text(result.data_model,    encoding="utf-8")
    (folder / "critique.md").write_text(result.critique,         encoding="utf-8")
    if result.error:
        (folder / "error.txt").write_text(result.error, encoding="utf-8")


def _load_result(folder: Path):
    result = PipelineResult(
        requirements = (folder / "requirements.md").read_text(encoding="utf-8"),
        architecture = (folder / "architecture.xml").read_text(encoding="utf-8"),
        data_model   = (folder / "data_model.xml").read_text(encoding="utf-8"),
        critique     = (folder / "critique.md").read_text(encoding="utf-8"),
        diagrams     = (folder / "diagrams.xml").read_text(encoding="utf-8"),
    )
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    result.score  = meta["score"]
    result.rounds = meta["rounds"]
    return result


def _load_mono(folder: Path):
    from eval.monolithic import MonolithicResult
    if not folder.exists():
        return None
    return MonolithicResult(
        requirements = (folder / "requirements.md").read_text(encoding="utf-8") if (folder / "requirements.md").exists() else "",
        architecture = (folder / "architecture.xml").read_text(encoding="utf-8") if (folder / "architecture.xml").exists() else "",
        data_model   = (folder / "data_model.xml").read_text(encoding="utf-8")  if (folder / "data_model.xml").exists()  else "",
        critique     = (folder / "critique.md").read_text(encoding="utf-8")     if (folder / "critique.md").exists()     else "",
    )


def _load_one(scenario: dict, out_dir: Path, run_mono: bool):
    base    = out_dir / scenario["name"]
    r_rag   = _load_result(base / "rag")
    r_norag = _load_result(base / "norag")
    r_mono  = _load_mono(base / "monolithic") if run_mono else None
    return scenario["name"], r_rag, r_norag, r_mono


# ── parallel runner ───────────────────────────────────────────────────────────

def _run_all_scenarios(
    scenarios: list[dict],
    out_dir:   Path,
    skip:      bool,
    run_mono:  bool,
    workers:   int = 4,
) -> tuple[list, list, list]:
    """
    Run (or load) all scenarios in parallel.
    Returns (rag_results, norag_results, mono_results) aligned with scenarios.
    """
    rag_results   = [None] * len(scenarios)
    norag_results = [None] * len(scenarios)
    mono_results  = [None] * len(scenarios)
    done = 0

    fn = _load_one if skip else _run_one

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fn, s, out_dir, run_mono): i
            for i, s in enumerate(scenarios)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                name, r_rag, r_norag, r_mono = future.result()
                rag_results[idx]   = r_rag
                norag_results[idx] = r_norag
                mono_results[idx]  = r_mono
            except Exception as e:
                print(f"\n  [error] {scenarios[idx]['name']}: {e}")
            done += 1
            print(f"  pipeline: {done}/{len(scenarios)} scenarios done", end="\r", flush=True)

    print()
    return rag_results, norag_results, mono_results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full evaluation suite for the architecture pipeline."
    )
    parser.add_argument(
        "--scenarios", type=int, default=0,
        help="Number of auto-generated scenarios to use (0 = use the 3 built-in scenarios).",
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Load cached pipeline outputs instead of running the pipeline.",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip the LLM-as-judge step.",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only run retrieval precision — no pipeline required.",
    )
    parser.add_argument(
        "--chunker-comparison", action="store_true",
        help="Run semantic vs fixed-size chunker comparison.",
    )
    parser.add_argument(
        "--no-monolithic", action="store_true",
        help="Skip the monolithic single-prompt baseline.",
    )
    parser.add_argument(
        "--no-faithfulness", action="store_true",
        help="Skip faithfulness metric.",
    )
    parser.add_argument(
        "--no-answer-relevance", action="store_true",
        help="Skip answer relevance metric.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for pipeline runs (default: 4).",
    )
    parser.add_argument(
        "--judge-sample", type=int, default=JUDGE_SAMPLE,
        help=f"How many scenarios to sample for LLM judge (default: {JUDGE_SAMPLE}).",
    )
    parser.add_argument(
        "--out-dir", default="eval_output",
        help="Directory for outputs and plots (default: eval_output).",
    )
    args = parser.parse_args()

    _import_all()

    out_dir   = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    run_mono = not args.no_monolithic

    # ── load scenarios ────────────────────────────────────────────────────────
    if args.scenarios > 0:
        print(f"\n{BAR}")
        print(f"  Loading/generating {args.scenarios} scenarios ...")
        print(BAR)
        scenarios = load_or_generate(args.scenarios)
    else:
        scenarios = list(SCENARIOS)
        print(f"\n  Using {len(scenarios)} built-in scenarios.")

    labels = [s["label"] for s in scenarios]

    # ── step 1: retrieval precision ───────────────────────────────────────────
    print(f"\n{BAR}")
    print("  Step 1 — Retrieval Precision  (24 in-domain + 8 hard-negative queries)")
    print(BAR)

    collections = build_all_collections()
    ret_results = run_retrieval_eval(collections, k_values=[3, 5])

    p3_in  = mean_precision(ret_results, 3, kind="in_domain")
    p5_in  = mean_precision(ret_results, 5, kind="in_domain")
    p3_neg = mean_precision(ret_results, 3, kind="hard_neg")
    p5_neg = mean_precision(ret_results, 5, kind="hard_neg")

    print(f"  In-domain   P@3={p3_in:.3f}  P@5={p5_in:.3f}")
    print(f"  Hard-neg    P@3={p3_neg:.3f}  P@5={p5_neg:.3f}")
    print(f"  Gap (in - neg): P@3={p3_in - p3_neg:+.3f}  P@5={p5_in - p5_neg:+.3f}")

    by_coll_in  = per_collection_mean(ret_results, 5, kind="in_domain")
    by_coll_neg = per_collection_mean(ret_results, 5, kind="hard_neg")
    print(f"\n  Per-collection in-domain P@5:")
    for coll in sorted(by_coll_in):
        print(f"    {coll:<20} in-domain={by_coll_in[coll]:.3f}  hard-neg={by_coll_neg.get(coll, 0.0):.3f}")

    plots.plot_retrieval_precision(ret_results, plots_dir)
    plots.plot_retrieval_by_collection(ret_results, plots_dir)

    retrieval_export = {
        str(k): [
            {"description": r.description, "collection": r.collection,
             "kind": r.kind, "precision": r.precision,
             "retrieved": r.chunks_retrieved, "relevant": r.relevant_chunks}
            for r in rows
        ]
        for k, rows in ret_results.items()
    }
    (out_dir / "retrieval_results.json").write_text(
        json.dumps(retrieval_export, indent=2), encoding="utf-8"
    )
    summary["retrieval"] = {
        "in_domain_p3": p3_in, "in_domain_p5": p5_in,
        "hard_neg_p3": p3_neg, "hard_neg_p5": p5_neg,
        "gap_p5": round(p5_in - p5_neg, 4),
        "by_collection_in_p5": by_coll_in,
        "by_collection_neg_p5": by_coll_neg,
    }

    # ── step 1b: chunker comparison ───────────────────────────────────────────
    if args.chunker_comparison:
        print(f"\n{BAR}")
        print("  Step 1b — Chunker Comparison")
        print(BAR)
        chunker_results = run_chunker_comparison(collections, k=5)
        sem_mean, fix_mean = mean_precision_by_chunker(chunker_results)
        print(f"  Semantic={sem_mean:.3f}  Fixed={fix_mean:.3f}  Δ={sem_mean - fix_mean:+.3f}")
        plots.plot_chunker_comparison(chunker_results, plots_dir)
        summary["chunker_comparison"] = {
            "semantic_mean_p5": sem_mean, "fixed_mean_p5": fix_mean,
            "delta": round(sem_mean - fix_mean, 4),
        }

    if args.retrieval_only:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  --retrieval-only done. Results → {out_dir.resolve()}/")
        return

    # ── step 2: pipeline runs (parallelised) ──────────────────────────────────
    print(f"\n{BAR}")
    print(f"  Step 2 — Pipeline Runs ({len(scenarios)} scenarios × 3 modes, {args.workers} workers)")
    print(f"  {'Loading from cache' if args.skip_pipeline else 'Running pipeline — this will take a while'}")
    print(BAR)

    rag_results, norag_results, mono_results = _run_all_scenarios(
        scenarios, out_dir, skip=args.skip_pipeline,
        run_mono=run_mono, workers=args.workers,
    )

    # Filter out any None results (failed scenarios)
    valid = [
        (s, r, n, m)
        for s, r, n, m in zip(scenarios, rag_results, norag_results, mono_results)
        if r is not None and n is not None
    ]
    if len(valid) < len(scenarios):
        print(f"  Warning: {len(scenarios) - len(valid)} scenarios failed and will be skipped.")

    scenarios_ok   = [v[0] for v in valid]
    rag_results    = [v[1] for v in valid]
    norag_results  = [v[2] for v in valid]
    mono_results   = [v[3] for v in valid]
    labels         = [s["label"] for s in scenarios_ok]

    avg_rag   = sum(r.score for r in rag_results)   / len(rag_results)
    avg_norag = sum(r.score for r in norag_results) / len(norag_results)
    print(f"  Mean critic score — RAG={avg_rag:.2f}/10  no-RAG={avg_norag:.2f}/10")
    if run_mono:
        valid_mono = [m for m in mono_results if m and not m.error]
        print(f"  Monolithic: {len(valid_mono)}/{len(scenarios_ok)} completed without error")

    summary["critic_scores"] = {
        "rag_mean":   round(avg_rag,   2),
        "norag_mean": round(avg_norag, 2),
        "n_scenarios": len(scenarios_ok),
    }

    # ── step 3: structural metrics (all scenarios) ────────────────────────────
    print(f"\n{BAR}")
    print(f"  Step 3 — Structural Metrics ({len(scenarios_ok)} scenarios)")
    print(BAR)

    rag_struct   = [score_all(r.requirements, r.architecture, r.data_model, r.diagrams)
                    for r in rag_results]
    norag_struct = [score_all(r.requirements, r.architecture, r.data_model, r.diagrams)
                    for r in norag_results]
    mono_struct  = [score_all(m.requirements, m.architecture, m.data_model, "")
                    for m in mono_results if m] if run_mono and mono_results else []

    rag_mean_struct   = sum(r.summary_score() for r in rag_struct)   / len(rag_struct)
    norag_mean_struct = sum(r.summary_score() for r in norag_struct) / len(norag_struct)
    print(f"  RAG structural mean    = {rag_mean_struct:.3f}")
    print(f"  No-RAG structural mean = {norag_mean_struct:.3f}")
    if mono_struct:
        mono_mean_struct = sum(r.summary_score() for r in mono_struct) / len(mono_struct)
        print(f"  Monolithic struct mean = {mono_mean_struct:.3f}")

    structural_rows = []
    for i, s in enumerate(scenarios_ok):
        for tag, reps in [("rag", rag_struct), ("norag", norag_struct)]:
            row = {"scenario": s["name"], "mode": tag}
            row.update(reps[i].to_dict())
            structural_rows.append(row)
        if mono_struct and i < len(mono_struct):
            row = {"scenario": s["name"], "mode": "monolithic"}
            row.update(mono_struct[i].to_dict())
            structural_rows.append(row)

    (out_dir / "structural_results.json").write_text(
        json.dumps(structural_rows, indent=2), encoding="utf-8"
    )

    # For plots, sample up to 20 scenarios to keep charts readable
    plot_n = min(20, len(scenarios_ok))
    step   = max(1, len(scenarios_ok) // plot_n)
    plot_idx = list(range(0, len(scenarios_ok), step))[:plot_n]

    plots.plot_structural_comparison(
        [rag_struct[i]   for i in plot_idx],
        [norag_struct[i] for i in plot_idx],
        [labels[i]       for i in plot_idx],
        plots_dir,
        mono_reports=[mono_struct[i] for i in plot_idx] if mono_struct else None,
    )
    plots.plot_summary_heatmap(
        [rag_struct[i]   for i in plot_idx],
        [norag_struct[i] for i in plot_idx],
        [labels[i]       for i in plot_idx],
        plots_dir,
        mono_reports=[mono_struct[i] for i in plot_idx] if mono_struct else None,
    )

    summary["structural"] = {
        "n_scenarios":  len(scenarios_ok),
        "rag_mean":     round(rag_mean_struct,   4),
        "norag_mean":   round(norag_mean_struct, 4),
    }
    if mono_struct:
        summary["structural"]["mono_mean"] = round(mono_mean_struct, 4)

    # ── step 4: faithfulness (sampled) ────────────────────────────────────────
    if not args.no_faithfulness:
        sample_idx = random.sample(range(len(scenarios_ok)),
                                   min(FAITH_SAMPLE, len(scenarios_ok)))
        print(f"\n{BAR}")
        print(f"  Step 4 — Faithfulness (sampled {len(sample_idx)} scenarios)")
        print(BAR)

        faith_rag_scores, faith_norag_scores = [], []
        for i in sample_idx:
            s = scenarios_ok[i]
            for label, text, coll_name, store in [
                ("architecture", rag_results[i].architecture,   "architecture", faith_rag_scores),
                ("srs",          rag_results[i].requirements,   "requirements", faith_rag_scores),
            ]:
                if coll_name in collections:
                    r = compute_faithfulness(text, collections[coll_name], label)
                    store.append(r.score)
            for label, text, coll_name, store in [
                ("architecture", norag_results[i].architecture, "architecture", faith_norag_scores),
                ("srs",          norag_results[i].requirements, "requirements", faith_norag_scores),
            ]:
                if coll_name in collections:
                    r = compute_faithfulness(text, collections[coll_name], label)
                    store.append(r.score)

        faith_rag_mean   = sum(faith_rag_scores)   / len(faith_rag_scores)   if faith_rag_scores   else 0.0
        faith_norag_mean = sum(faith_norag_scores) / len(faith_norag_scores) if faith_norag_scores else 0.0
        print(f"  Faithfulness — RAG={faith_rag_mean:.3f}  no-RAG={faith_norag_mean:.3f}")
        summary["faithfulness"] = {
            "sample_size": len(sample_idx),
            "rag_mean":    round(faith_rag_mean,   4),
            "norag_mean":  round(faith_norag_mean, 4),
        }

    # ── step 5: answer relevance (all scenarios, cheap) ───────────────────────
    if not args.no_answer_relevance:
        print(f"\n{BAR}")
        print(f"  Step 5 — Answer Relevance ({len(scenarios_ok)} scenarios)")
        print(BAR)

        ar_rag, ar_norag = [], []
        for i, s in enumerate(scenarios_ok):
            r_ar = compute_answer_relevance(s["input"], rag_results[i].requirements,   s["label"])
            n_ar = compute_answer_relevance(s["input"], norag_results[i].requirements, s["label"])
            ar_rag.append(r_ar)
            ar_norag.append(n_ar)
            print(f"  [{i+1}/{len(scenarios_ok)}] {s['label'][:30]:<30} RAG={r_ar.score:.2f}  no-RAG={n_ar.score:.2f}")

        plots.plot_answer_relevance(
            ar_rag[:plot_n], ar_norag[:plot_n],
            [s["label"] for s in scenarios_ok[:plot_n]],
            plots_dir,
        )
        summary["answer_relevance"] = {
            "n_scenarios": len(scenarios_ok),
            "rag_mean":    round(mean_answer_relevance(ar_rag),   4),
            "norag_mean":  round(mean_answer_relevance(ar_norag), 4),
        }

    # ── step 6: LLM judge (sampled) ───────────────────────────────────────────
    if not args.no_judge:
        n_judge   = min(args.judge_sample, len(scenarios_ok))
        judge_idx = random.sample(range(len(scenarios_ok)), n_judge)

        print(f"\n{BAR}")
        print(f"  Step 6 — LLM-as-Judge (sampled {n_judge} scenarios)")
        print(BAR)

        judge_rag, judge_norag, judge_mono = [], [], []

        for i in judge_idx:
            s = scenarios_ok[i]
            print(f"  [{s['label'][:25]}] judging ...", end=" ", flush=True)
            jr = judge_output(rag_results[i].requirements,
                              rag_results[i].architecture,
                              rag_results[i].data_model)
            jn = judge_output(norag_results[i].requirements,
                              norag_results[i].architecture,
                              norag_results[i].data_model)
            judge_rag.append(jr)
            judge_norag.append(jn)
            print(f"RAG={jr['total']}/{jr['max']}  no-RAG={jn['total']}/{jn['max']}", end="")

            if run_mono and mono_results[i] and not mono_results[i].error:
                jm = judge_output(mono_results[i].requirements,
                                  mono_results[i].architecture,
                                  mono_results[i].data_model)
                judge_mono.append(jm)
                print(f"  mono={jm['total']}/{jm['max']}", end="")
            print()

        (out_dir / "judge_results.json").write_text(
            json.dumps({"rag": judge_rag, "norag": judge_norag, "monolithic": judge_mono},
                       indent=2),
            encoding="utf-8",
        )

        sample_labels = [scenarios_ok[i]["label"] for i in judge_idx]
        plots.plot_judge_scores(judge_rag, judge_norag, sample_labels, plots_dir)
        plots.plot_radar(judge_rag, judge_norag, plots_dir,
                         judge_mono=judge_mono or None)

        summary["judge"] = {
            "sample_size": n_judge,
            "rag_mean":    round(sum(r["total"] for r in judge_rag)   / len(judge_rag),   2),
            "norag_mean":  round(sum(r["total"] for r in judge_norag) / len(judge_norag), 2),
            "max":         judge_rag[0]["max"],
        }
        if judge_mono:
            summary["judge"]["mono_mean"] = round(
                sum(r["total"] for r in judge_mono) / len(judge_mono), 2
            )

    # ── final summary ─────────────────────────────────────────────────────────
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{BAR}")
    print("  Evaluation Complete")
    print(BAR)
    r = summary["retrieval"]
    print(f"  Retrieval    in-domain P@5={r['in_domain_p5']:.3f}  "
          f"hard-neg P@5={r['hard_neg_p5']:.3f}  gap={r['gap_p5']:+.3f}")
    if "structural" in summary:
        s = summary["structural"]
        mono_s = f"  mono={s['mono_mean']:.3f}" if "mono_mean" in s else ""
        print(f"  Structural   RAG={s['rag_mean']:.3f}  no-RAG={s['norag_mean']:.3f}{mono_s}"
              f"  (n={s['n_scenarios']})")
    if "faithfulness" in summary:
        f = summary["faithfulness"]
        print(f"  Faithfulness RAG={f['rag_mean']:.3f}  no-RAG={f['norag_mean']:.3f}"
              f"  (sample={f['sample_size']})")
    if "answer_relevance" in summary:
        a = summary["answer_relevance"]
        print(f"  Ans.Rel.     RAG={a['rag_mean']:.3f}  no-RAG={a['norag_mean']:.3f}"
              f"  (n={a['n_scenarios']})")
    if "judge" in summary:
        j = summary["judge"]
        mono_j = f"  mono={j['mono_mean']:.1f}" if "mono_mean" in j else ""
        print(f"  Judge        RAG={j['rag_mean']:.1f}/{j['max']}  "
              f"no-RAG={j['norag_mean']:.1f}/{j['max']}{mono_j}"
              f"  (sample={j['sample_size']})")
    print(f"\n  Results → {out_dir.resolve()}/")
    print(f"  Plots   → {plots_dir.resolve()}/")
    print(BAR)


if __name__ == "__main__":
    main()
