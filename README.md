# Dynamic Travel Replanning

SNU *Data Science & Prompt Engineering* final project. An LLM agent (`student_solver.py`) picks a flight + hotel + restaurant + activity bundle for a multi-turn travel-planning task. Scored on feasibility (40%), preference fit (30%), memory discipline (20%), and cost efficiency (10%).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env                       # add OPENAI_API_KEY

# smoke test (2 episodes)
python run_student.py --solver student_solver --limit-public 2 \
  --output-dir runs/smoke

# full public set (20 episodes)
python run_student.py --solver student_solver --output-dir runs/full
```

Summary lands at `runs/<dir>/llm_eval_summary_v2.md`. Per-call trace at `runs/<dir>/trace.jsonl`.

## Current score

20 public episodes, single trial (`runs/readme_full`):

| Bucket | Weight | Score | Weighted |
|---|---|---:|---:|
| Feasibility | 40% | 0.924 | 36.97 |
| Preference fit | 30% | 0.526 | 15.77 |
| Memory | 20% | 0.537 | 10.74 |
| Efficiency | 10% | 0.853 | 8.53 |
| **Official /100** | | | **79.42** |

Total cost for the 20-episode run: **$0.035** (~$0.0018/episode).

## Files

- `student_solver.py` — submission entrypoint (`solve_episode(runtime)`)
- `student_custom_tools_template.py` — deterministic state→memory derivers
- `run_student.py` — wrapper that calls the staff harness
- `llm_eval_config_student.json` — student-tunable budgets
- `dynamic_travel_replanning/` — simulator data + evaluator (do not edit)
- `runtime_api.py`, `llm_runner.py`, `llm_tools.py`, `llm_agents.py`, `run_llm_baselines.py`, `budget_knobs.py` — staff harness (do not edit)

See `CLAUDE.md` for the longer design notes and `STUDENT_EVALUATION.md` for the scoring rubric.
