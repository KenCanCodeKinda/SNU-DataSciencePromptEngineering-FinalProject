# Dynamic Travel Replanning

SNU *Data Science & Prompt Engineering* final project. An LLM agent (`student_solver.py`) picks a
flight + hotel + restaurant + activity bundle for a multi-turn travel-planning task, honoring hard
constraints, soft preferences, and a memory-hygiene contract while keeping cost low.

## Two architectures, one toggle

`student_solver.py` ships with two interchangeable architectures, selected by the
`STUDENT_AGENT_MODE` environment variable (it can't be a budget knob — the harness rejects unknown
`--set` keys). Default is `single`.

| Mode | Pipeline | When |
|---|---|---|
| `single` (default) | **gather (LLM, tools)** → deterministic memory sweep → deterministic feasibility-first select → grounded rationale | Cheapest, highest feasibility, lowest variance |
| `multi` | **gather (LLM, tools)** → memory sweep → **decide (LLM, no tools)** → deterministic verify/repair | Lets the model express soft-preference judgment, with a deterministic feasibility guardrail |

```bash
STUDENT_AGENT_MODE=single python run_llm_baselines.py --config llm_eval_config.json \
  --systems student_solver --skip-hidden --skip-ablations --limit-public 20 --output-dir runs/single

STUDENT_AGENT_MODE=multi  python run_llm_baselines.py --config llm_eval_config.json \
  --systems student_solver --skip-hidden --skip-ablations --limit-public 20 --output-dir runs/multi
```

Both modes share one tool session so the trace accumulates correctly, derive all constraints from the
user turns (never from hidden metadata), and guarantee the memory/rejected tools fire.

## Run it

```bash
# deps (conda or venv both fine)
pip install -r requirements.txt
cp .env.sample .env                       # add OPENAI_API_KEY

# smoke (3 episodes)
STUDENT_AGENT_MODE=single python run_llm_baselines.py --config llm_eval_config.json \
  --systems student_solver --skip-hidden --skip-ablations --limit-public 3 --output-dir runs/smoke
```

> We invoke `run_llm_baselines.py` directly rather than `run_student.py`: the wrapper defaults to
> `llm_eval_config_student.json`, which only exists in a packaged staff release. In this dev checkout
> use `--config llm_eval_config.json` as shown.

Summary lands at `runs/<dir>/llm_eval_summary_v2.md`; per-call trace at `runs/<dir>/llm_run_trace.jsonl`.

## Scoring (v4.8 — one official /100)

The framework now reports a single 100-point official score:

| Group | Points | Components |
|---|---:|---|
| Hard constraints | 45 | `hard_constraint_rate` |
| Bundle coherence | 5 | `bundle_coherence_rate` |
| Soft preference fit | 15 | `semantic_fit` (10) + `exactish` (5) |
| Replanning behavior | 25 | `update_handling` (15) + `stale_doc_retirement` (5) + `rejected_option_memory` (5) |
| Cost efficiency | 10 | `0.03 / total_cost`, halved if `decision_quality < 0.35` |

**Hidden-eval reality (the key design driver).** In official grading the evaluator is passed the live
tool trace and **rebuilds `memory_report` from what the session actually retrieved** (`docs_seen`,
`retrieved_keys_seen`, `rejected_option_notes_seen`) plus the rationale text — it ignores the
self-reported dict (`evaluator._trace_grounded_memory_report`). So the replanning bucket is won by
*actually firing* `search_memory(include_stale=true)` (surfacing `stale:*` docs) and
`get_rejected_options`, not by filling a dict. Hidden episodes also strip `scenario_state`, so the
solver infers all constraints from the user turns via `_episode_state(...)`.

## Current score (20 public episodes)

| | Hard /45 | Bundle /5 | Soft /15 | Replanning /25 | Efficiency /10 | **Official /100** | $/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`single`** | 37.33 | 5.00 | 12.13 | 21.24 | 8.25 | **83.95** | $0.0018 |
| `multi` | 36.20 | 5.00 | 12.31 | 21.24 | 6.47 | 81.23 | $0.0023 |

Single mode wins on both score and cost: the deterministic picker keeps feasibility high, and the
extra `multi` decide call doesn't buy enough soft-fit to offset its efficiency cost — consistent with
the course's caution against reaching for multi-agent by default. `stale_doc_retirement` and
`rejected_option_memory` are near-perfect in both modes thanks to the deterministic memory sweep.

## Files

- `student_solver.py` — submission entrypoint (`solve_episode(runtime)`); both architectures + toggle
- `student_custom_tools_template.py` — turn-inference (`_episode_state`), rerankers, memory derivers
- `llm_eval_config_student.json` / `llm_eval_config.json` — student-tunable budgets
- `dynamic_travel_replanning/` — simulator data + evaluator (do not edit)
- `runtime_api.py`, `llm_runner.py`, `llm_tools.py`, `llm_agents.py`, `run_llm_baselines.py`, `budget_knobs.py` — staff harness (do not edit)

See `CLAUDE.md` for the longer design notes and `STUDENT_EVALUATION.md` for the scoring rubric.
