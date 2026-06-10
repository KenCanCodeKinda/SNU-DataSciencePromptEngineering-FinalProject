"""Rebuild the full score history on the official /100 scale from run traces.

Every run under runs/ writes a `system_finish` record into its llm_run_trace.jsonl
containing the evaluator summary. This script recovers each run's /100 score from
that record so the README's score history is reproducible from the artifacts on
disk (no re-running of paid LLM evals required).

Two groups, because the scorer changed mid-project:
  * CURRENT  - trace stored student_view.official_score_100 (the live 45/5/15/25/10
               scale; directly comparable to the headline 96.23).
  * OLD      - trace predates that scorer; only the obsolete 40/30/20/10
               student_overall_score is available, and `exactish` was never stored,
               so the current /100 cannot be reconstructed. Shown on its own scale.

Usage:  python _score_history.py            # plain table
        python _score_history.py --md       # GitHub-markdown tables (for the README)
"""
import json
import glob
import os
import sys

MD = "--md" in sys.argv
cur, old = [], []
for trace in glob.glob("runs/*/llm_run_trace.jsonl"):
    run = os.path.basename(os.path.dirname(trace))
    with open(trace, encoding="utf-8") as f:
        for line in f:
            if '"system_finish"' not in line:
                continue
            r = json.loads(line)
            if r.get("event") != "system_finish":
                continue
            s = r.get("summary") or {}
            sv = s.get("student_view") or {}
            rec = dict(
                run=run, system=r.get("system", "?"),
                ep=sv.get("episodes", s.get("episodes", 0)),
                cost=sv.get("total_cost_usd", 0.0),
                dq=s.get("mean_decision_quality", 0.0),
                hard=s.get("mean_hard_constraint_rate", 0.0),
                stale=s.get("mean_stale_doc_retirement_rate", 0.0),
                spoken=s.get("mean_spoken_rule_compliance_rate", 0.0),
            )
            if "official_score_100" in sv:
                rec["score"] = sv["official_score_100"]
                cur.append(rec)
            elif "student_overall_score" in sv:
                rec["score"] = sv["student_overall_score"]
                old.append(rec)
            break


def emit(title, rows, label):
    rows.sort(key=lambda x: x["score"], reverse=True)
    if MD:
        print(f"\n#### {title}\n")
        print(f"| Run | Episodes | {label} | decision_quality | hard | stale_doc | spoken_rule | cost_usd |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for x in rows:
            print(f"| `{x['run']}` | {x['ep']} | {x['score']:.2f} | {x['dq']:.3f} | "
                  f"{x['hard']:.3f} | {x['stale']:.3f} | {x['spoken']:.3f} | {x['cost']:.6f} |")
    else:
        print(f"\n=== {title} ===")
        h = f"{'run':<22}{'ep':>4}{label:>9}{'dq':>7}{'hard':>7}{'stale':>7}{'spoken':>8}{'cost':>10}"
        print(h); print("-" * len(h))
        for x in rows:
            print(f"{x['run']:<22}{x['ep']:>4}{x['score']:>9.2f}{x['dq']:>7.3f}"
                  f"{x['hard']:>7.3f}{x['stale']:>7.3f}{x['spoken']:>8.3f}{x['cost']:>10.6f}")


emit("Current scorer — official /100 (directly comparable)", cur, "/100")
emit("Earlier runs — old 40/30/20/10 rubric /100 (NOT comparable)", old, "old/100")
