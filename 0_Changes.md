# Dynamic Travel Replanning — Change Log

Last update: 2026-05-01. Working from `_Context/` lecture notes (W3 prompt engineering, W7 advanced RAG, W8 context engineering, W9 hallucination mitigation) applied to [student_solver.py](student_solver.py).

## TL;DR

- [student_solver.py](student_solver.py) was a stub (`raise NotImplementedError`). Now implements Phases 1–3 of the plan in [/Users/kenkim/.claude/plans/lets-take-some-information-frolicking-cerf.md](../../.claude/plans/lets-take-some-information-frolicking-cerf.md).
- **Best confirmed run (p7, 20-ep public):** 0/20 failures, raw_score **31.18**, official `student_overall_score` **71.04/100**, **$0.0057/ep** (20% cheaper than CLAUDE.md reference baseline).
- Strengths: `policy_ok=1.0`, `distractor_avoidance=1.0`, `semantic_fit=0.92`. Weaknesses: `distributed_context_rate=0.34`, `memory_retirement_rate=0.37`, `stale_doc_retirement=0.42`, `spoken_rule_compliance=0.66` (target ≥0.70).
- **Most recent change (p7, 2026-05-01):** Priority 1 applied — verifier rule 5 augmented with canonical-vocabulary check for `spoken_rule_hits`. Net +0.84 on student_overall_score, spoken_rule +0.026 (partial — didn't hit ≥0.70 target). Kept.

---

## Files changed

### [student_solver.py](student_solver.py) — full rewrite (was a stub)

Single file, single function `solve_episode(runtime)`. Drives a tool-using planner directly through `runtime.runner.run_tool_agent_json` and `runtime.runner.create_json_response` — bypasses [llm_agents.py:608](llm_agents.py#L608) `run_single_baseline` and [llm_agents.py:636](llm_agents.py#L636) `run_memory_single` because both call into [llm_agents.py:559](llm_agents.py#L559) `run_single_tool_agent`, which references the undefined symbol `ensure_grounded_submission` at [llm_agents.py:593](llm_agents.py#L593) (see "Dev-checkout caveats" below).

Components:
- [student_solver.py:18-69](student_solver.py#L18-L69) `STUDENT_PLANNER_INSTRUCTIONS` — Phase 1 (W8 P/M split + W3 decomposition) + Phase 2 (hygiene preflight, faithfulness rule). Appends `MEMORY_REPORT_GUIDANCE`.
- [student_solver.py:72-87](student_solver.py#L72-L87) `STUDENT_VERIFIER_INSTRUCTIONS` — Phase 3 verifier system prompt (W9 chain-of-verification).
- [student_solver.py:90-119](student_solver.py#L90-L119) `_run_planner` — wraps `runner.run_tool_agent_json` so the same call shape is reusable for the retry round.
- [student_solver.py:122-189](student_solver.py#L122-L189) `_verify_and_maybe_retry` — runs verifier via `create_json_response`. If `approve=False`, reissues the planner once with verifier hints injected (max retry rounds = `max_tool_rounds-5`). Verifier `usage` is merged via `combine_usages` so cost accounting matches.
- [student_solver.py:192-211](student_solver.py#L192-L211) `_fallback_result` — synthesizes a minimal valid result (None IDs, empty memory_report) when the planner exhausts its tool-round budget. Session telemetry still populates `memory_report` via `merge_memory_report`, so the memory bucket earns partial credit instead of zeroing.
- [student_solver.py:214-237](student_solver.py#L214-L237) `solve_episode` — opens session with role `single_memory` (unlocks rich-context tools at [llm_tools.py:76-83](llm_tools.py#L76-L83)), runs planner → verifier → optional revision → returns `tool_result(...)`. Caps: `active_doc_cap=3`, `active_key_cap=5` (mirror `run_memory_single`).

Two robustness fixes added during smoke testing:
1. **Verifier `create_json_response` wrapped in try/except** ([student_solver.py:131-149](student_solver.py#L131-L149)). Without this, an empty-output verifier response on a hard episode killed the entire episode (zero across all four buckets).
2. **Primary planner `_run_planner` wrapped in try/except RuntimeError** ([student_solver.py:226-229](student_solver.py#L226-L229)). Without this, "Exceeded max_tool_rounds" zeroed 4/20 episodes on the first full run.

### [student_custom_tools_template.py](student_custom_tools_template.py) — Phase 4 stubs filled

Pure-Python sort-by-tag-overlap rerankers ([student_custom_tools_template.py:13-39](student_custom_tools_template.py#L13-L39)). **Not invoked** by `solve_episode` — Phase 4 is gated OFF per the plan, only enable if Phase 3 plateaus.

### [llm_eval_config.json](llm_eval_config.json) — added empty `ablations` block

```json
"ablations": { "systems": {}, "public_sample_trip_ids": [] }
```

Required by [run_llm_baselines.py:328](run_llm_baselines.py#L328) `build_ablation_systems`, which is called unconditionally before the `--skip-ablations` flag is honored. The dev checkout omits this block; staff release script presumably ships it.

### [.env](.env) — created from `api.txt`

Mode 600. Loaded by [llm_runner.py:641](llm_runner.py#L641).

---

## Memory-fidelity prompt overhaul — REGRESSED, surgically rolled back

Applied 2026-04-30 to address the three weak metrics (`distributed_context`, `memory_retirement`, `stale_doc_retirement`). Five planner-side edits + 2 verifier-side edits + 1 plumbing edit.

**Result on `runs/student_full_p4` (20 public, max_tool_rounds=16): regressed almost every metric.** Compared to `p3+retry`:

| metric | prev | overhaul | delta |
|---|---:|---:|---:|
| decision_quality | 0.6820 | 0.5628 | -0.1192 |
| spoken_rule_compliance | 0.7121 | 0.2928 | **-0.4194** |
| stale_doc_retirement | 0.4890 | 0.3627 | -0.1263 |
| memory_retirement_rate | 0.4444 | 0.3373 | -0.1071 |
| bundle_coherence_rate | 0.7500 | 0.6000 | -0.1500 |
| hard_constraint_rate | 0.6266 | 0.5849 | -0.0417 |
| distributed_context_rate | 0.3320 | 0.3466 | +0.0146 |
| active_context_hygiene | 0.7981 | 0.8090 | +0.0109 |
| **raw_score** | ~32 | **26.33** | regressed |
| tool_calls / ep | 12.75 | 17.95 | +41% |
| cost_usd / ep | $0.00577 | $0.00612 | +6% |

**Diagnosis.** Initially suspected the worked example's literal IDs (`HT205`, `stale:budget_cap_archive`) were leaking into submissions. Verified with grep: prior baseline (p3+retry) ALSO has 20/20 episodes carrying those tokens — they originate in the staff's [llm_agents.py:14 MEMORY_REPORT_GUIDANCE](llm_agents.py#L14) which I append to my prompt. So the example was NOT the smoking gun.

**Most likely cause.** Planner prompt grew 5.6k → 7.9k chars (+40%) and tool_calls jumped 41% — the model became less efficient and over-prescriptive instructions made it conflate constraint enumeration with `retired` listing, and skip filling `spoken_rule_hits` correctly. gpt-5.4-nano probably can't absorb that much rule-stacking without losing the thread on the actual decision.

**Surgical rollback applied.** Reverted the four planner-side additions (worked example/shape guide, step-1 enumeration mandate, lean-context tightening, echo-back check). Kept the two verifier-side additions (rules 6+7 doc-ID checks) and the verifier-revision `retired_docs` hint propagation, since those only fire on revision and don't bloat every planner call. Planner prompt now ~5.9k chars (close to original 5.6k); verifier 2.1k.

**Lesson logged.** For this codebase: prompt edits to gpt-5.4-nano should be small and incremental. Big batches regress unpredictably. Test each edit alone, not bundled.

## Run history

| Run | Episodes | Failures | raw_score | Notes |
|---|---|---|---:|---|
| `runs/student_smoke_p1` | 2 public | 1/2 (verifier crash) | — | Surfaced verifier-call-itself-failing case |
| `runs/student_smoke_p1b` | 2 public | 0/2 | 38.78 | After verifier try/except fix |
| `runs/student_full_p3` | 20 public | 4/20 (round overflow) | 25.48 | Surfaced primary planner round-overflow case |
| `runs/student_full_p3_retry` | 4 reran | 0/4 | ~32 combined | After fallback fix + `max_tool_rounds=14` |
| `runs/student_full_p4` | 20 public | 0/20 | 26.33 | **REGRESSED** — memory-fidelity overhaul (rolled back) |
| `runs/student_full_p5` | 20 public | 0/20 | 29.73 | After surgical rollback + `max_tool_rounds=16`. Partial recovery. |
| `runs/student_full_p6` | 20 public | 0/20 | **30.99** | `max_tool_rounds=14` + verifier rules 6+7 retained. Big lift on feasibility (`bundle_coherence` 0.75→0.85, `hard_constraint` 0.63→0.68, `semantic_fit` 0.87→0.90) at slight cost to memory bucket (`memory_retirement` 0.44→0.36, `stale_doc_retirement` 0.49→0.42). |
| `runs/student_smoke_p7` | 2 public | 0/2 | 37.67 | Smoke after Priority 1 (verifier rule 5 + spoken-rule canonical-vocab check). Green. |
| `runs/student_full_p7` | 20 public | 0/20 | **31.18** | Priority 1 (isolated). `spoken_rule` 0.631 → 0.657 (+0.026, partial — target was ≥0.70). `student_overall_score` 70.2 → **71.04** (+0.84). Kept. |
| `runs/student_full_p8` | 20 public | 0/20 | 29.80 | Priority 2 (isolated): planner rule 4 retirement-cue addition. **REGRESSED** — `bundle_coherence` 0.85 → 0.70 (-0.15), `student_overall_score` 71.04 → 67.66 (-3.38). Target metric `memory_retirement` only +0.014; sibling `stale_doc_retirement` -0.032. Rolled back. |
| `runs/student_smoke_p9` | 2 public | 0/2 | 36.11 | Smoke after Priority 2 deferred verifier-side rule 8 (retirement surface-phrase cues, verifier-only). Green. |
| `runs/student_full_p9` | 20 public | 0/20 | 30.24 | Priority 2 deferred verifier-side rule 8 (isolated). **REGRESSED** — target metrics improved (`memory_retirement` 0.375 → 0.456 +0.081, `stale_doc_retirement` 0.421 → 0.495 +0.074, `adaptation_memory` bucket +2.58), but `bundle_coherence` 0.85 → 0.75 (-0.10), `spoken_rule` 0.66 → 0.59 (-0.07), `semantic_fit` 0.92 → 0.88 (-0.04), `decision_quality` 0.71 → 0.68. `student_overall_score` 71.04 → 68.65 (-2.39). Rolled back. |

## Current performance (best confirmed: `runs/student_full_p7`)

### Configuration

- **Planner prompt:** original phrasing, no worked example / enumeration mandate / echo-back. ~5.9k chars.
- **Verifier prompt:** 5 base rules (rule 5 now also checks `spoken_rule_hits` canonical vocab — added in p7) + rule 6 (every `old_*` retired key needs paired `stale:*` doc id) + rule 7 (`docs_retrieved` includes canonical doc IDs queried, `active_docs ≤ 3`, empty `ignored_distractors` flagged). ~2.2k chars.
- **Verifier-revision plumbing:** `retired_docs` hint propagated into revision instructions.
- **Robustness:** `_fallback_result` for round-overflow + verifier `create_json_response` wrapped in try/except.
- **Knobs:** `max_tool_rounds=14`, `max_output_tokens=900`.

### Per-bucket score (40/30/20/10 student weighting, official `student_view`)

| Bucket | Weight | Mean | Weighted | vs p6 |
|---|---|---:|---:|---:|
| Feasibility | 40% | 0.8361 | **33.4** | -0.2 |
| Preference fit | 30% | 0.7614 | 22.8 | +0.4 |
| Adaptation / memory | 20% | 0.6059 | 12.1 | +0.5 |
| Efficiency | 10% | 0.2631 | 2.6 | +0.2 |
| **student_overall_score** | 100% |  | **71.04** | **+0.84** |

### Per-metric (p7, 20-ep public)

| Metric | p6 | p7 | Δ | Note |
|---|---:|---:|---:|---|
| policy_ok | 1.0000 | **1.0000** | 0 | maxed |
| distractor_avoidance_rate | 1.0000 | **1.0000** | 0 | maxed |
| semantic_fit_rate | 0.8993 | **0.9175** | +0.018 | strong |
| bundle_coherence_rate | 0.8500 | 0.8500 | 0 | strong |
| active_context_hygiene_rate | 0.7981 | 0.7922 | -0.006 | OK |
| update_handling_rate | 0.7150 | 0.6960 | -0.019 | OK |
| decision_quality | 0.7081 | 0.7097 | +0.002 | OK |
| hard_constraint_rate | 0.6764 | 0.6583 | -0.018 | OK |
| spoken_rule_compliance | 0.6311 | **0.6571** | **+0.026** | partial — target ≥0.70 |
| rejected_option_memory_rate | 0.6500 | 0.6167 | -0.033 | mid |
| memory_retrieval_rate | 0.6066 | 0.6087 | +0.002 | mid |
| stale_doc_retirement_rate | 0.4216 | 0.4211 | -0.001 | **weak — Priority 2 target** |
| memory_retirement_rate | 0.3631 | 0.3748 | +0.012 | **weak — Priority 2 target** |
| distributed_context_rate | 0.3466 | 0.3380 | -0.009 | weak |

### Cost & throughput

- **Cost:** $0.1140 total, $0.00570/episode, 13.4 mean tool calls/ep.
- **vs CLAUDE.md baselines** (50-ep public+hidden, not apples-to-apples):
  - Baseline 29.82 / $0.355
  - Memory Single 32.34 / $0.298
  - MAS 33.11 / $0.584
  - Our p7 31.18 / $0.114 (20-ep) — at or above Memory Single on the public set, **~5× cheaper**.

### Per-difficulty `decision_quality`

| Tier | Episodes | Mean dq (p7) | vs p6 |
|---|---|---:|---:|
| easy | 2 | 0.798 | -0.002 |
| medium | 7 | 0.720 | +0.034 |
| hard | 11 | 0.687 | -0.016 |

Medium tier picked up most of the gain in p7; hard regressed slightly.

## Lessons logged (cross-session reference)

1. **Prompt edits to gpt-5.4-nano should be small and incremental.** Bundling 5 prompt edits caused a ~6-point raw_score regression (p4) that took 2 more runs to surface and partially undo. Test each edit in isolation.
2. **Worked examples with concrete IDs are not the smoking gun for ID-leakage.** The staff's `MEMORY_REPORT_GUIDANCE` (which we must include) already contains canonical example IDs like `HT205` — they appear in 20/20 submissions of every run. The regression came from prompt bloat, not example contamination.
3. **Verifier-side prompt additions are safer than planner-side.** Rules added to the verifier only fire on revision, so they don't bloat every planner call. Rules 6+7 (verifier doc-ID checks) survived the rollback and netted a marginal improvement. Reconfirmed p8: a single planner-side sentence (retirement cues) regressed `student_overall_score` -3.38 with bundle_coherence taking a -0.15 hit, while a parallel verifier-side edit in p7 was +0.84.
4. **`max_tool_rounds=14` is the sweet spot** in this benchmark. 10 fails on hard episodes (round overflow), 16 increases tool churn without a quality lift.
5. **Cue lists in the planner over-trigger.** Adding surface-phrase cues for retirement (p8) inflated `tool_calls` +13% and disrupted bundle coherence — the model treated the cues as a checklist to satisfy rather than a signal to weigh. Future "make the model do X more" ideas should land on the verifier (where they only fire on revision), not the planner.
6. **Verifier-side ≠ free either.** p9 moved the same retirement cue list to the verifier (rule 8). Target metrics improved as predicted (`memory_retirement` +0.081, `stale_doc_retirement` +0.074, adaptation bucket +2.58), but the verifier's revision suggestions pushed the planner into over-eager retirement on revision, costing `bundle_coherence` -0.10 and `spoken_rule` -0.07 — net `student_overall_score` -2.39. Verifier additions are *safer* than planner additions (p9's -2.39 < p8's -3.38), not *safe*. The gold's retirement targets need deeper inference than any surface-phrase rule can supply on either side; this 0.37→0.46 ceiling is likely structural for `gpt-5.4-nano`.
7. **Windows-only setup notes.** `Path.read_text()` in `run_llm_baselines.py` defaults to `cp949` on Korean Windows and crashes on UTF-8 JSON. Run with `PYTHONUTF8=1` set in the environment. Staff `run.sh` is macOS-only — invoke `python run_llm_baselines.py …` directly on Windows. (Caveat applies to the dev checkout only; staff would presumably handle this in their packaged release.)

---

## Dev-checkout caveats (bugs in staff code we worked around)

Both surfaced by exploration / smoke testing. Either can break the staff baseline; neither breaks our solver.

1. **`ensure_grounded_submission` undefined.** Referenced at [llm_agents.py:593](llm_agents.py#L593), defined nowhere in the repo. Means `run_single_baseline` and `run_memory_single` crash at the end of every episode in this dev checkout. Our solver bypasses both and uses `runner.run_tool_agent_json` + `tool_result` directly.
2. **`config["ablations"]` missing.** [run_llm_baselines.py:328](run_llm_baselines.py#L328) accesses it unconditionally, before honoring `--skip-ablations`. We patched [llm_eval_config.json](llm_eval_config.json) with an empty block — see "Files changed" above.

If/when staff ships a fixed [llm_agents.py](llm_agents.py), `student_solver` should still work because it doesn't depend on `run_single_tool_agent`. If staff's eval config has its own `ablations` block, our additive change is harmless.

---

## Suggested next changes (prioritized, post-iteration)

The big planner-prompt overhaul didn't work. Smaller, surgical changes did. Going forward, prefer one-edit-per-run experiments.

### Priority 1 — Recover `spoken_rule_compliance` to ≥0.70 — **APPLIED & KEPT (partial)**

Dropped 0.71 → 0.63 between p3+retry and p6. Most likely cause: verifier rules 6+7 push the model to spend output tokens on doc-id pairs at the expense of populating `spoken_rule_hits` carefully.

**Edit applied 2026-05-01 (p7):** rule 5 of `STUDENT_VERIFIER_INSTRUCTIONS` ([student_solver.py:77-80](student_solver.py#L77-L80)) now appends: "`spoken_rule_hits` must use the canonical bucket vocabulary (`must_remember`, `forbidden`, `one_off_only`, `retire`, `do_not_reconsider`, `keep_context_lean`) — flag any episode where the planner left this empty but the user's turns clearly contain a spoken rule."

Verifier-side only (lesson 3: safer than planner-side; only fires on revision). Result: spoken_rule 0.631 → 0.657 (+0.026, partial — target was ≥0.70 not hit). `student_overall_score` 70.2 → 71.04. Net win, kept.

### Priority 2 — Improve `memory_retirement_rate` (0.37) and `stale_doc_retirement_rate` (0.42) — **BOTH ATTEMPTS ROLLED BACK (p8 planner-side, p9 verifier-side)**

**Attempt 1 — planner-side (p8, 2026-05-01):** added retirement-cue sentence to rule 4 of `STUDENT_PLANNER_INSTRUCTIONS`: "If a user turn says 'no longer', 'doesn't apply anymore', 'forget that', 'this trip is different', 'used to', or similar, treat the corresponding earlier preference as retired."

**Result: regressed.** `student_overall_score` 71.04 → 67.66 (-3.38). `bundle_coherence` 0.85 → 0.70 (-0.15) was the worst hit; `spoken_rule` -0.04, `rejected_option_memory` -0.05, `tool_calls` +13%. The target metric `memory_retirement` improved only +0.014; the sibling `stale_doc_retirement` actually regressed -0.032. Rolled back.

**Attempt 2 — verifier-side (p9, 2026-05-02):** moved the same surface-phrase cue list to a new rule 8 of `STUDENT_VERIFIER_INSTRUCTIONS`: "If a user turn explicitly says 'no longer' / 'doesn't apply anymore' / 'forget that' / 'this trip is different' / 'used to' or similar near a constraint, and `memory_report.retired` does NOT contain the matching `old_*` key, list the missing retirement in `issues` and propose the canonical key via `retire`." Hypothesis (lesson 3): verifier-side fires only on revision, so it shouldn't bloat planner tokens.

**Result: also regressed, less severely.** `student_overall_score` 71.04 → **68.65** (-2.39). The hypothesis was *partially* validated — target metrics did improve as designed (`memory_retirement` 0.375 → 0.456 +0.081; `stale_doc_retirement` 0.421 → 0.495 +0.074; `adaptation_memory` bucket +2.58, biggest single bucket gain in any run). But the verifier's revision suggestions over-corrected: the planner re-emitted with looser bundle selection, costing `bundle_coherence` 0.85 → 0.75 (-0.10), `spoken_rule` 0.66 → 0.59 (-0.07), `semantic_fit` 0.92 → 0.88 (-0.04), `decision_quality` 0.71 → 0.68 (-0.03). Feasibility -3.73 + preference_fit -4.42 outweighed adaptation +2.58. Rolled back.

**Diagnosis (consolidated across p8 & p9).** Both attempts targeted the same retirement metrics with the same surface-phrase vocabulary; both regressed; verifier-side was less bad (-2.39 vs -3.38) but still net-negative. The retirement metric ceiling at ~0.45 appears to be structural for `gpt-5.4-nano` on this benchmark — gold retirements often hinge on deeper inference (e.g. "the meeting got moved to Friday" implies retiring an earlier zone preference, no surface cue) that no rule list can encode. Pushing the model harder on this dimension trades feasibility/preference-fit for adaptation. **Recommendation: stop optimizing this metric directly. Accept ~0.45 ceiling and look elsewhere for gains.**

**Next idea (revised, deferred):** if Priority 2 is revisited, do NOT use surface-phrase cue lists on either side. A more promising direction would be a small Python-side verifier helper that scans turn text for retirement-implying patterns AND cross-checks against the actual `gold` retired keys observed in p7+ runs to learn which patterns correlate — feed only high-precision matches to the model. That's algorithmic preprocessing, not prompting, and lives outside the LLM call path.

### Priority 3 — Phase 4 rerankers (defer)

Wired but not invoked ([student_custom_tools_template.py:13-39](student_custom_tools_template.py#L13-L39)). Risk noted in plan: changes search-result ordering mid-conversation, may confuse positional references. Defer until Priorities 1–2 land.

### Anti-priorities (do NOT redo)

- Bundling multiple prompt edits — caused 6-point regression in one run.
- Adding a worked example with concrete IDs to the planner prompt — leaks no worse than baseline but adds ~2k chars of overhead that hurt overall behavior.
- `max_tool_rounds=16` — adds tool churn without quality lift.
- Echo-back / step-1 enumeration mandates in the planner — over-constrains gpt-5.4-nano.

---

## How to reproduce

Conda env: `mldl_mac` at `/opt/anaconda3/envs/mldl_mac` (override with `PY=...`).

Use the [run.sh](run.sh) wrapper — it shows a tqdm progress bar and prints the headline summary at the end:

```bash
./run.sh <name>          # 20-ep public benchmark → runs/<name>/
./run.sh <name> 2        # 2-ep smoke
./run.sh <name> 20 16    # 20-ep, max_tool_rounds=16
```

The wrapper uses the canonical knobs (`max_tool_rounds=14`, `max_output_tokens=900`) and writes full stdout to `runs/<name>/run.log`.

Or call the harness directly:

```bash
/opt/anaconda3/envs/mldl_mac/bin/python run_llm_baselines.py \
  --config llm_eval_config.json --systems student_solver \
  --skip-hidden --skip-ablations \
  --set student_solver.max_tool_rounds=14 \
  --set student_solver.max_output_tokens=900 \
  --limit-public 20 --output-dir runs/<name>
```

Tail an in-flight run from another terminal: `python watch_progress.py runs/<name> --total 20`.

Rerun only failed episodes from a previous run:

```bash
... --rerun-failed-from runs/<previous-dir> --output-dir runs/<new-name>
```

Inspect: `runs/<name>/llm_eval_summary_v2.md` for the headline table; `runs/<name>/llm_run_trace.jsonl` for per-call events; `runs/<name>/llm_results_public_v2.json` for per-episode metrics.
