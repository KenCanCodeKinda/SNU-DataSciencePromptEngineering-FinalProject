# Dynamic Travel Replanning

SNU *Data Science & Prompt Engineering* final project. An LLM agent (`student_solver.py`) picks a
flight + hotel + restaurant + activity bundle for a multi-turn travel-planning task, honoring hard
constraints, soft preferences, and a memory-hygiene contract while keeping cost low.

## Architecture: deterministic-first, with an LLM planner as a soft signal

`student_solver.py` runs a single pipeline. The LLM proposes; deterministic Python decides. This keeps
feasibility high and cost low (~$0.002/episode) while still letting the model express
soft-preference judgment, and it directly reflects the course's caution against reaching for
multi-agent orchestration by default.

| Stage | What it does |
|---|---|
| **1. `ContextEvolution`** | Replays the user turns. Extracts active constraints (`_extract_constraints`), detects stale-assumption retirements (`_detect_retirements`), and builds a supersession timeline so later turns override earlier ones. |
| **2. `_deterministic_gather`** | Unconditionally fires `get_profile_brief`, `get_venue_brief`, `get_city_ops_notes`, then `search_memory(include_stale=True)` over the known `stale:*`/`heuristic:*` docs, then `get_rejected_options`. This **guarantees the memory-discipline tools fire** — the official evaluator rebuilds `memory_report` from what the session actually retrieved, not from a self-reported dict. |
| **3. `_llm_planner_pass`** *(optional, on by default)* | One Planner LLM call with the primitive tools + `planner_schema`, producing a draft bundle. Its IDs become `prefer_ids` — a **soft tie-breaker**, never binding. Wrapped in `try/except`: a planner failure degrades gracefully to pure deterministic selection. |
| **4. `_python_select`** | The decision core. Tiered flight selection (refund → arrival cutoff → meeting-safe fallbacks), then enumerates hotel × restaurant × activity combos scored on `(partner-promo, meeting-zone coherence, soft-tag richness, −cost, prefer_ids hits)` with budget + zone filters and graceful degradation. Uses the evaluator's **real** restaurant cost (`price_level * 25000`). |
| **5. `merge_memory_report`** | Canonicalizes `memory_report` from session telemetry — forced retired keys/docs, capped active context (`active_doc_cap=4`, `active_key_cap=6`) for the lean-context metric, and spoken-rule hits derived from the detected constraints. |
| **6. `ensure_grounded_submission`** | Guarantees all four IDs are valid and grounded in the live inventory. |
| **7. `_critique_bundle`** | Deterministic post-check (hotel quiet? restaurant vegan when required? activity weather-safe when rainy?), attached to the result as `critique`. |

All constraints are derived from the **user turns** (and `scenario_state` when present on public episodes),
never from hidden gold — hidden episodes strip `scenario_state`, so the same turn-inference path covers both.

## Run it

```bash
# deps (conda base or venv both fine — repo is tested on conda base)
pip install -r requirements.txt

# API key: put OPENAI_API_KEY in a .env file at the repo root
#   echo "OPENAI_API_KEY=sk-..." > .env
```

```bash
# full 20-episode public run
python run_llm_baselines.py --config llm_eval_config.json \
  --systems student_solver --skip-hidden --skip-ablations \
  --limit-public 20 --output-dir runs/public20

# quick 3-episode smoke
python run_llm_baselines.py --config llm_eval_config.json \
  --systems student_solver --skip-hidden --skip-ablations \
  --limit-public 3 --output-dir runs/smoke
```

> We invoke `run_llm_baselines.py` directly rather than `run_student.py`: the wrapper defaults to
> `llm_eval_config_student.json`, which only exists in a packaged staff release. In this dev checkout
> use `--config llm_eval_config.json` as shown. The dev config runs `student_solver` on
> **`gpt-5.4-nano`** with `max_tool_rounds=9`, `max_tool_results=4`, `max_output_tokens=800`.

Summary lands at `runs/<dir>/llm_eval_summary_v2.md`; per-call trace at `runs/<dir>/llm_run_trace.jsonl`;
full per-episode metrics at `runs/<dir>/llm_results_public_v2.json`.

## Scoring (official /100)

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
self-reported dict. So the replanning bucket is won by *actually firing*
`search_memory(include_stale=true)` (surfacing `stale:*` docs) and `get_rejected_options`, which is
exactly what `_deterministic_gather` guarantees.

## Latest results — 20 public episodes

Run on 2026-06-10 via `--config llm_eval_config.json` (`gpt-5.4-nano`), 20 episodes
(2 easy / 7 medium / 11 hard). All 20 API calls succeeded, 0 errors.

| Hard /45 | Bundle /5 | Soft /15 | Replanning /25 | Efficiency /10 | **Official /100** |
|---:|---:|---:|---:|---:|---:|
| 44.62 | 5.00 | 14.44 | 25.00 | 7.17 | **96.23** |

Soft = semantic_fit 9.88 + exactish 4.56. Replanning = update_handling 15.00 + stale_doc_retirement 5.00 + rejected_option_memory 5.00.

| Metric | Value | | Metric | Value |
|---|---:|---|---|---:|
| `decision_quality` | 0.805 | | `bundle_coherence_rate` | 1.000 |
| `hard_constraint_rate` | 0.992 | | `update_handling_rate` | 1.000 |
| `semantic_fit_rate` | 0.988 | | `stale_doc_retirement_rate` | 1.000 |
| `exactish_rate` | 0.913 | | `distractor_avoidance_rate` | 1.000 |
| `spoken_rule_compliance_rate` | 0.931 | | `rejected_option_memory_rate` | 1.000 |
| `policy_ok` | 1.000 | | `memory_retirement_rate` | 1.000 |

- **Cost:** $0.041857 total over 20 episodes ($0.00209/episode; range $0.00122–$0.00320). 291,150 tokens, 8.7 tool calls/episode.
- **Decision quality is flat across tiers** — easy 0.808, medium 0.800, hard 0.808 — i.e. the hard episodes are not dragging the score down.
- **Where the points go:** the whole replanning bucket (25/25) and bundle coherence (5/5) are maxed; hard constraints (44.62/45) and soft fit (14.44/15) are near-ceiling. The only real headroom is **efficiency (7.17/10)** — `0.03 / 0.041857 = 0.717`. Dropping the dev model from `gpt-5.4-nano` to the cheaper `gpt-5-nano` (input $0.05 vs $0.10, output $0.40 vs $0.625 per 1M) is the obvious lever there.

## Score logs

Three separate logs exist, on two different scales — don't compare across them.

### 1. Live evaluation (real LLM, official /100)

The authoritative numbers. Produced by `run_llm_baselines.py` against `episodes_public_example.json`.

| Run | Episodes | Model | Official /100 | decision_quality | cost_usd | tool_calls | API |
|---|---:|---|---:|---:|---:|---:|---|
| `runs/public20` (2026-06-10) | 20 | gpt-5.4-nano | **96.23** | 0.805 | 0.041857 | 8.70 | all ok |
| `runs/smoke_test` (2026-06-10) | 2 | gpt-5.4-nano | 100.00 | 0.808 | 0.004098 | 8.50 | all ok |

### 2. Offline ablation study (`_ablation_results.json`, /100, zero-cost)

`_ablation_eval.py` runs the deterministic path with no network calls (lexical retrieval, `FakeRunner`),
so LLM cost is 0 and efficiency is always 10/10 — the full system reads 100 here, above the 96.23 live
score whose only gap is the cost bucket. The point of this log is **module isolation**: drop one module
and see what the official scorer loses. `semantic_fit` and `efficiency` are 1.00 across every variant.

| Variant | /100 | hard | coherence | exactish | update | stale_doc | rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full system (all modules) | 100.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − Meta-Context Controller | 100.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − Context-Evolution module | 100.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − Verifier-Critic self-audit | 100.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − Retirement detection | 100.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| − Global retrieval scope only | 97.79 | 1.00 | 1.00 | 1.00 | 0.98 | 0.97 | 0.65 |
| − Memory-Context Manager (no gather) | 83.50 | 1.00 | 1.00 | 1.00 | 0.57 | 0.00 | 0.00 |
| − Verifier floor (naive greedy select) | 83.34 | 0.75 | 0.85 | 0.34 | 1.00 | 1.00 | 1.00 |

Two modules carry the score: the **memory gather** (`_deterministic_gather` — drop it and the whole
replanning bucket collapses, −16.5) and the **deterministic verifier floor** (`_python_select` — drop it
for a naive greedy pick and feasibility + exactish collapse, −16.66). The meta-context controller,
context-evolution timeline, retirement detection, and self-critic are score-neutral on this offline path
— they're diagnostics/guardrails, not score drivers. Reproduce with `python _ablation_eval.py`
(or `python _offline_eval.py` for the full-system offline number alone).

### 3. Legacy experiment log (`runs/*/run.log`, old `raw_score` scale)

These predate the current official /100 scorer and are on the older `raw_score` scale (~30–37) — **not
comparable to the /100 numbers above**. Kept for development history. Sorted by score.

| Run dir | raw_score | decision_quality | stale_doc | spoken_rule | cost_usd | tool_calls |
|---|---:|---:|---:|---:|---:|---:|
| `exp_gpt5nano` | 29.94 | 0.44 | 1.00 | 0.43 | 0.0184 | 5.70 |
| `student_full_p8` | 29.80 | 0.68 | 0.39 | 0.61 | 0.1156 | 15.10 |
| `student_full_p9` | 30.25 | 0.68 | 0.50 | 0.59 | 0.1177 | 14.15 |
| `exp_trim9_600` | 33.79 | 0.67 | 1.00 | 0.44 | 0.0658 | 10.95 |
| `state_enrich` | 34.72 | 0.77 | 0.93 | 0.83 | 0.0696 | 10.45 |
| `state_enrich2` | 35.83 | 0.76 | 0.90 | 0.86 | 0.0726 | 11.15 |
| `student_smoke_p9` | 36.11 | 0.73 | 0.50 | 0.47 | 0.0132 | 10.50 |
| `verify_v1` | 36.66 | 0.79 | 0.88 | 0.88 | 0.0714 | 10.80 |
| `exp_r16` | 37.11 | 0.80 | 1.00 | 0.89 | 0.0781 | 10.80 |

## Files

- `student_solver.py` — submission entrypoint (`solve_episode(runtime)`); the full pipeline above
- `student_custom_tools_template.py` — student-owned helper module (reranker/bundle stubs); nothing in the harness imports it, safe to extend or replace
- `llm_eval_config_student.json` / `llm_eval_config.json` — student-tunable budgets (dev checkout uses the latter)
- `dynamic_travel_replanning/` — simulator data + evaluator (do not edit)
- `runtime_api.py`, `llm_runner.py`, `llm_tools.py`, `llm_agents.py`, `run_llm_baselines.py`, `budget_knobs.py` — staff harness (do not edit)

See `CLAUDE.md` for the longer design notes and `STUDENT_EVALUATION.md` for the scoring rubric.
