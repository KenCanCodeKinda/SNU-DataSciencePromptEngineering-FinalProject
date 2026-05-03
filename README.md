# Dynamic Travel Replanning

Final project for SNU's *Data Science & Prompt Engineering* course. We built an LLM agent (`student_solver.py`) that picks a flight + hotel + restaurant + activity bundle for a multi-turn travel-planning task while honoring shifting constraints, spoken rules, and a strict memory-hygiene contract. The agent is scored on four buckets: feasibility (40%), preference fit (30%), memory discipline (20%), and cost efficiency (10%).

## Best run — `student_full_p7` (20 public episodes, 0 failures)

**Overall score: 71.04 / 100** at **$0.0057 per episode**.

| Bucket | Weight | Mean | Weighted |
|---|---|---:|---:|
| Feasibility | 40% | 0.836 | 33.4 |
| Preference fit | 30% | 0.761 | 22.8 |
| Adaptation / memory | 20% | 0.606 | 12.1 |
| Efficiency | 10% | 0.263 | 2.6 |
| **Total** | 100% |  | **71.04** |

### Per-metric highlights

| Metric | Score |
|---|---:|
| policy_ok | 1.000 |
| distractor_avoidance | 1.000 |
| semantic_fit | 0.918 |
| bundle_coherence | 0.850 |
| active_context_hygiene | 0.792 |
| update_handling | 0.696 |
| decision_quality | 0.710 |
| hard_constraint_rate | 0.658 |
| spoken_rule_compliance | 0.657 |
| rejected_option_memory | 0.617 |
| memory_retrieval | 0.609 |
| stale_doc_retirement | 0.421 |
| memory_retirement | 0.375 |
| distributed_context | 0.338 |

### Cost vs. course baselines

| System | Score | Cost / 50-ep set |
|---|---:|---:|
| Course baseline | 29.82 | $0.355 |
| Memory Single | 32.34 | $0.298 |
| Multi-Agent System | 33.11 | $0.584 |
| **Ours (p7, 20-ep)** | **31.18 raw_score** | **$0.114** |

On the public set, our solver matches "Memory Single" on score while running roughly **5× cheaper**.

## Run history

| Run | Episodes | Failures | raw_score | Notes |
|---|---|---|---:|---|
| `student_smoke_p1` | 2 | 1/2 | — | Verifier crash surfaced |
| `student_smoke_p1b` | 2 | 0/2 | 38.78 | Verifier try/except added |
| `student_full_p3` | 20 | 4/20 | 25.48 | Round-overflow surfaced |
| `student_full_p3_retry` | 4 | 0/4 | ~32 | Fallback fix |
| `student_full_p4` | 20 | 0/20 | 26.33 | Memory-fidelity overhaul — regressed, rolled back |
| `student_full_p5` | 20 | 0/20 | 29.73 | Surgical rollback, partial recovery |
| `student_full_p6` | 20 | 0/20 | 30.99 | Verifier rules 6 + 7 added |
| `student_full_p7` | 20 | 0/20 | **31.18** | Verifier rule 5 + spoken-rule canonical-vocab check — **best confirmed** |
| `student_full_p8` | 20 | 0/20 | 29.80 | Planner-side retirement cue — regressed, rolled back |
| `student_full_p9` | 20 | 0/20 | 30.24 | Verifier-side rule 8 — regressed, rolled back |
