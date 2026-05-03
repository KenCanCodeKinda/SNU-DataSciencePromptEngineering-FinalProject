# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Final project for "Dynamic Travel Replanning" — a benchmark where an LLM agent picks a flight/hotel/restaurant/activity bundle for a trip while honoring constraints, spoken rules, and a memory hygiene contract. The student deliverable is `student_solver.py`. Staff rerun the same `solve_episode(runtime)` function on hidden episodes using the official runtime + evaluator.

**Do not edit:** anything in `dynamic_travel_replanning/` (simulator data + evaluator), `runtime_api.py`, `llm_runner.py`, `llm_tools.py`, `llm_agents.py`, `run_llm_baselines.py`, `budget_knobs.py`. Staff reruns your code against their own copies of these.
**Edit freely:** `student_solver.py`, `student_custom_tools_template.py`, plus any new modules you add and import from `student_solver.py`.

Other docs in the repo: `README_STARTER.md` (quick start), `STUDENT_EVALUATION.md` (score buckets in plain English), `STUDENT_TOOLING.md` (which tools you may extend), `llm_prompts_baseline.md` (baseline prompt text).

## Setup and commands

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env                                  # then put OPENAI_API_KEY in .env
```

Smoke-test the included baseline, then your solver:

```bash
python run_student.py --solver student_solver_example --limit-public 2 --output-dir runs/example_smoke
python run_student.py --solver student_solver         --limit-public 3 --output-dir runs/my_solver_smoke
```

`run_student.py` is a thin wrapper that shells out to `run_llm_baselines.py` with `--skip-hidden --skip-ablations`. Use `run_llm_baselines.py` directly only when you need ablations or hidden-set support.

> **Heads-up:** `run_student.py` defaults to `--config llm_eval_config_student.json`, which is only generated when staff packages the release (see `build_student_release.py`). In this dev checkout only `llm_eval_config.json` exists, so the wrapper will `FileNotFoundError`. Either copy/symlink `llm_eval_config.json` to that name, or bypass the wrapper:
> ```bash
> python run_llm_baselines.py --config llm_eval_config.json \
>   --systems student_solver --skip-hidden --skip-ablations \
>   --limit-public 2 --output-dir runs/my_solver_smoke
> ```

Run a specific episode by trip id: `--public-trip-ids TRIP_ID1,TRIP_ID2`. Run failed episodes from a previous run: `--rerun-failed-from runs/prev_dir`.

There is no test suite, linter, or build script — the canonical "is it working" signal is `python run_student.py --solver student_solver --limit-public 2` exiting 0 with non-zero `decision_quality` in the printed summary.

## Budget knobs (the only config you should change)

Hard caps live in `llm_eval_config.json` under `student_tunable_budgets.student_solver` and are enforced by `budget_knobs.validate_system_budget_caps`. Override at the CLI:

```bash
python run_student.py --solver student_solver \
  --set student_solver.max_tool_rounds=12 \
  --set student_solver.max_tool_results=5 \
  --set student_solver.max_output_tokens=1200
```

Allowed knobs for `student_solver`: `max_tool_rounds` (1–16), `max_tool_results` (1–8), `max_output_tokens` (300–1500). Any other field, or a value outside the range, is rejected before the run starts.

Allowed models (enforced by `LLMRunner._assert_generation_model`): `gpt-4o-mini`, `gpt-5-nano`, `gpt-5-mini`, `gpt-5.4-nano`. Embeddings: `text-embedding-3-small`. Pricing for cost tracking is in the same config under `pricing_usd_per_1m_tokens`.

## How the harness invokes your solver

`run_llm_baselines._run_dynamic_solver` imports the module named by `solver_module` (e.g. `student_solver`), calls `solve_episode(runtime)`, and expects a dict with **at least** `submission` and `usage`. Optional keys: `response_ids`, `tool_trace`, `retrieval`, `api_status`. The runtime (`runtime_api.StudentRuntime`) gives you:

- `runtime.runner` — the `LLMRunner` (OpenAI client wrapper). **All model calls must go through this**, otherwise usage/cost won't be measured by the staff wrapper. Key methods: `create_json_response`, `run_tool_agent_json` (the tool-calling loop), `embed_texts`, `combine_usages`, `empty_usage`.
- `runtime.toolbox` / `runtime.new_session(...)` — builds a `TravelToolSession` that exposes the primitive tools as OpenAI function specs (`session.tool_specs(primitive_only=True)`) and dispatches them (`session.dispatch`). The session also tracks `docs_seen`, `rejected_memory_seen`, and `retrieved_keys_seen`, which feed the memory-discipline metrics.
- `runtime.episode` — the public episode dict (no `gold` field).
- `runtime.system_config` — the resolved per-system config including any `--set` overrides.

The minimal pattern is what `student_solver_example.solve_episode` does: delegate to `llm_agents.run_single_baseline`, which builds a session, calls `runner.run_tool_agent_json` with `final_decision_schema()`, and packages the result via `tool_result(...)` (which merges the model's memory report with what the session actually saw).

Per-run output lands under `--output-dir` (default `runs/student_run`): the results JSON, a summary markdown, and `trace.jsonl` — one record per LLM call, tool call, and lifecycle event. The trace is your primary debugging surface; read it before changing prompts.

## Submission shape (must match the evaluator)

`submission` must be a dict matching `llm_agents.final_decision_schema()` plus `usage`/`debug`. Concretely:

```python
{
  "flight_id": "FL...",  "hotel_id": "HT...",
  "restaurant_id": "RS...", "activity_id": "AC...",
  "memory_report": { ... matches memory_report_schema() ... },
  "notes": "...",
  "usage": runtime.runner.combine_usages(...),  # added in run_llm_baselines.attach_debug
  "debug": { "tool_call_count": int, "tool_trace": [...] }
}
```

The `memory_report` is **not optional** — most of the adaptation/hygiene score depends on it. Keys must be the short benchmark identifiers (snake_case context keys, `stale:doc_id` for retired docs, `reason_key:OPTION_ID` for rejected notes). See `MEMORY_REPORT_GUIDANCE` in `llm_agents.py` for the canonical vocabulary, and `merge_memory_report` for how a model report is canonicalized + augmented with what the session actually retrieved. Inventing freeform strings will fail the evaluator's overlap checks.

## Scoring (what to optimize)

`dynamic_travel_replanning/evaluator.evaluate_episode` returns per-episode metrics. `summarize_rows_student` rolls them into four buckets:

- **feasibility (40%)** — `hard_constraint_rate`, `policy_ok`, `bundle_coherence_rate`. Depends on the chosen IDs satisfying the gold's `required_hard` constraints (under-budget, quiet hotel, weather-safe activity, zone coherence, refund safety, bundle dependency validity, dietary support).
- **preference fit (30%)** — `decision_quality` (a weighted combination — see `evaluator.py:419`), `semantic_fit_rate`, `spoken_rule_compliance_rate`. Driven by chosen items' `semantic_tags` overlapping the gold `soft_tags` and by `memory_report.spoken_rule_hits` matching `gold.required_spoken_rules`.
- **adaptation / memory (20%)** — eight memory metrics: retrieval, retirement, distributed-context (docs retrieved), stale-doc retirement, distractor avoidance, rejected-option memory, active-context hygiene, update-handling. All require the right IDs/keys in `memory_report`.
- **efficiency (10%)** — capped at `0.03 / total_cost_usd`, halved if `decision_quality < 0.35` (cheap-and-wrong is penalized).

The MD-rendered `raw_score_for_ranking` is a different weighting (15/14/11/5) used for staff comparisons, not the student-facing 40/30/20/10. Optimize the bucket means, not raw_score.

## Architecture map

- `runtime_api.py` — `StudentRuntime` dataclass; the only public seam between student code and the harness.
- `llm_runner.py` — `LLMRunner`. Wraps `client.responses.create` with retries, JSON-schema strict mode, malformed-JSON repair, and a tool-calling loop (`run_tool_agent_json`) that drives `previous_response_id` between rounds. Records `ModelUsage` per call.
- `llm_tools.py` — `TravelToolbox` (per-process) and `TravelToolSession` (per-episode). Function specs for `search_flights/hotels/restaurants/activities`, `search_memory`, `get_rejected_options`, `get_profile_brief`, `get_venue_brief`, `get_city_ops_notes`, `get_policy`, plus richer-context tools that are gated by role and `primitive_tools_only`. The session also normalizes context keys via `_CONTEXT_KEY_ALIASES` — student tools should produce keys that survive that mapping.
- `llm_agents.py` — schemas (`final_decision_schema`, `memory_report_schema`, `context_pack_schema`, `verifier_schema`), context-key/rejected-note canonicalization, `merge_memory_report` (the bridge between model output and session telemetry), and the two reference systems `run_single_baseline` / `run_memory_single`.
- `retrieval.py` — `RetrievalCorpus` for memory-doc retrieval (lexical and embedding modes; embeddings cached on disk under `dynamic_travel_replanning/.cache`).
- `dynamic_travel_replanning/rtl_semantic_env.py` — deterministic JSON-backed environment; loads inventories and tag-decorates rows on retrieval.
- `dynamic_travel_replanning/evaluator.py` — `evaluate_episode` and `summarize_rows`/`summarize_rows_student`. Read these before tuning anything; the metric definitions are the source of truth.
- `run_llm_baselines.py` — the eval orchestrator. Concurrency, hidden-set sampling, ablations, output JSON, summary markdown. `_run_dynamic_solver` is the dispatch point for `student_solver`.
- `budget_knobs.py` — parses `--set` overrides and validates them against `student_tunable_budgets`. The harness aborts before calling your solver if a knob is out of range.
- `trace_logger.py` — JSONL trace writer; one record per LLM call, tool call, and lifecycle event. Outputs land at `<output-dir>/trace.jsonl`.
- `student_solver.py` — your submission entrypoint; just defines `solve_episode(runtime)`.
- `student_custom_tools_template.py` — student-owned helper module with placeholder `rerank_hotels` / `rerank_restaurants` / `choose_bundle` stubs. Intended home for rerankers, fallback search, bundle scoring. Safe to rename/replace; nothing in the harness imports it.

## Things that will break submissions silently

- Calling `OpenAI()` directly instead of going through `runtime.runner` — usage stays at zero, the cost-aware bucket reads as suspiciously cheap, and the staff wrapper's accounting won't match your local numbers.
- Producing freeform strings in `memory_report` arrays. The evaluator only counts items that normalize to the canonical keys in `_CONTEXT_KEY_ALIASES` / `_REJECTED_REASON_ALIASES` / `_RETIRED_DOC_BY_KEY`.
- Returning a submission with valid IDs but no `memory_report` — feasibility may pass, but the 20% adaptation bucket goes to zero.
- Hard-coding model names or knob values that conflict with the cap ranges in `llm_eval_config.json`. The staff config is authoritative; treat your local edits as suggestions.

---

# Final Project 정리: Dynamic Travel Replanning

## 한 줄 요약
**바뀌는 제약 조건 속에서 여행 계획을 다시 짜는 LLM 시스템을 만들어라.** 핵심은 "여행 계획" 자체가 아니라 **변화하는 컨텍스트와 메모리를 잘 관리하는 것**.

## 무엇을 만드는가

LLM 기반 시스템 하나 + 다음 도구/데이터에 접근:
- **검색 도구**: `search_flights`, `search_hotels`, `search_restaurants`, `search_activities`
- **컨텍스트 도구**: `search_memory`, `get_rejected_options`, `get_profile_brief`, `get_venue_brief`, `get_city_ops_notes`, `get_policy`
- 위 도구를 그대로 써도 되고, 래퍼(예: `semantic_hotel_reranker`)로 개선해도 됨

## 무엇을 푸는가 (Task의 본질)

멀티턴 에피소드. 사용자가 **이전 발화를 무효화하는 새 제약**을 계속 던짐. 시스템은:
1. 관련 메모리/프로필만 골라 가져오기 (전부 가져오면 감점)
2. **이번 trip에만 적용되는 override** 인식 (예: "이번 출장은 공항 접근성이 더 중요")
3. **stale assumption 폐기** ("예전 예산 가정은 무효")
4. 거절된 옵션 다시 꺼내지 않기
5. 최종적으로 valid ID로 grounded된 plan 제출

## 채점 (반드시 외워둬야 함)

| 비중 | 항목 | 핵심 |
|---|---|---|
| 40% | Feasibility | hard constraint, policy 위반 없음, 미팅 시간 안 늦음 |
| 30% | Preference fit | spoken-rule 준수, 의사결정 품질 |
| 20% | Memory discipline | 업데이트 반영, retire, distractor 회피, **lean context** |
| 10% | Efficiency bonus | 토큰 비용 (단, 나쁜 plan을 구제하진 못함) |

평가는 **public 20 + hidden 30** = 총 50 episode. 공개 20개로만 개발 가능. 최종 grading은 TA가 official runtime wrapper로 재실행.

## 일정 (놓치면 안 됨)
- **2026-05-07 / 05-12**: 초기 status check (팀별 20분 F2F)
- **2026-06-01 ~ 06-05**: Deep Dive — **개인 기여도** 검증 (1시간)
- **2026-06-09 / 06-11**: 최종 발표 (942-302)

---

## 멘토로서 짚고 갈 포인트 (중요)

이 부분이 과제 설명보다 더 중요할 수 있음.

**1. Test Results 그래프를 다시 봐.**
- Baseline: 29.82점 / $0.355
- Memory Single: 32.34점 / $0.298 ← **점수 더 높고 비용 더 쌈**
- MAS: 33.11점 / $0.584 ← 점수 0.77 더 오르려고 비용 2배

조교가 친절해서 보여준 게 아니야. **"MAS 무지성으로 가지 마라"**는 신호임. 점수 차이의 cost-effectiveness가 처참함. Efficiency 10%까지 고려하면 MAS의 우위는 더 줄어들 거야. 많은 팀이 화려한 multi-agent architecture로 갔다가 점수도 비용도 손해 볼 것임.

**2. 진짜 차별화 포인트는 "메모리 규율(20%)"이다.**
40%는 어차피 다들 어느 정도 채움(valid한 booking ID 뽑는 거). 30%도 평탄. **20% 메모리 점수에서 갈린다.** "lean active context", "distractor avoidance", "stale retirement" — 이 세 가지를 명시적으로 처리하는 모듈이 핵심. 단순히 RAG 갖다 붙이지 말고, **retire 정책**을 명시적으로 설계해라.

**3. Hidden task 비중을 봐.**
Public 20개 중 Hard가 11개(55%), Hidden 30개 중 Hard가 13개(43%). **Hidden이 약간 더 쉽지 않다** — 그저 양이 많을 뿐. Public 20개에 overfitting하지 않도록 주의. Public 점수는 신호일 뿐임.

**4. "Deep Dive"가 따로 있다.**
개인 기여도를 따로 본다는 건 **누가 뭘 했는지 코드/커밋 단위로 추적**한다는 뜻. 한 명이 다 짜는 팀은 나머지가 망함. 역할 분담 + 커밋 로그 관리 지금부터 시작해야 함.

**5. 베이스라인 모델이 `gpt-5.4-nano`라는 점.**
조교 베이스라인이 nano급이라는 건, 더 큰 모델로 갈아끼우면 점수는 오르겠지만 efficiency 점수가 깎인다는 의미. **모델 키우기로 승부 보려는 전략은 함정**일 수 있음. 같은 모델 안에서 retrieval/verification 품질로 이기는 게 정공법.

---

## 다음으로 결정해야 할 것

지금 시점에서 너희 팀이 가장 먼저 정해야 하는 건 단순함 vs 복잡성의 tradeoff야. 어떤 방향이 너희 상황에 맞는지 — 팀 사이즈, 코딩 분담 가능성, 마감까지 시간 — 알려주면 더 구체적인 전략 짜줄 수 있어.
