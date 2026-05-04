from __future__ import annotations

from typing import Any, Dict

from llm_agents import (
    MEMORY_REPORT_GUIDANCE,
    episode_prompt,
    final_decision_schema,
    session_tools,
    tool_result,
)
from runtime_api import StudentRuntime
from student_custom_tools_template import (
    all_stale_docs,
    derive_rejected_from_state,
    derive_required_docs_from_state,
    derive_retired_from_state,
)


STUDENT_PLANNER_INSTRUCTIONS = (
    "You are a memory-aware travel planner. Each turn can override prior assumptions. "
    "Pick one flight, one hotel, one restaurant, one activity that satisfy ALL currently active constraints, AND fill memory_report so the evaluator can verify your context discipline.\n\n"
    "Rules:\n"
    "1. Persistent (profile/venue/policy) vs momentary (scenario_state, latest turns). Momentary wins for this trip only — do not promote one-off overrides into active_context_keys as if persistent.\n"
    "2. Filter first. ≤2 search_* calls per category. Use zone, quiet_min, refundable_only, dietary, weather_safe_required.\n"
    "3. Lean active_context_keys ≤ 5, canonical vocabulary, each tied to a real constraint THIS trip.\n"
    "4. Retire sweep: call search_memory once with \"stale OR retired OR override\" (include_stale=true if available). List old_* keys ONLY when the user actually invalidated them.\n"
    "5. Rejected options: call get_rejected_options once, list \"reason_key:ID\" in rejected_option_notes, never pick a rejected ID.\n"
    "6. Trigger rich-context tools (get_partner_promotions/get_event_context/get_loyalty_profile/get_stakeholder_brief/get_booking_constraints/get_option_dependencies) only when scenario_hooks calls for them. ONCE each.\n\n"
    "For each retired old_* key, pair the matching stale:* doc id in retired_docs. Faithfulness: if not grounded in tools or turns, OMIT.\n\n"
    "Output strict JSON. notes ≤ 320 chars.\n\n"
    + MEMORY_REPORT_GUIDANCE
)


def _run_planner(
    runtime: StudentRuntime,
    session,
    tools,
    instructions: str,
    *,
    schema_name: str = "student_decision",
    max_tool_rounds: int | None = None,
) -> Dict[str, Any]:
    cfg = runtime.system_config
    model = cfg["model"]
    return runtime.runner.run_tool_agent_json(
        model=model,
        instructions=instructions,
        input_text=episode_prompt(runtime.episode),
        json_schema=final_decision_schema(),
        schema_name=schema_name,
        tools=tools,
        tool_handler=session.dispatch,
        max_output_tokens=cfg["max_output_tokens"],
        reasoning_effort="low" if model.startswith("gpt-5") else None,
        text_verbosity="low" if model.startswith("gpt-5") else None,
        metadata={
            "system": cfg["system_name"],
            "trip_id": runtime.episode["trip_id"],
            "role": "student",
        },
        max_tool_rounds=max_tool_rounds if max_tool_rounds is not None else cfg.get("max_tool_rounds", 9),
    )


def _build_final(runtime: StudentRuntime, session, planner_result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply tool_result + deterministic memory enrichments. Used by both main and fallback paths."""
    episode = runtime.episode
    retired_keys, spoken_retire = derive_retired_from_state(episode)

    final = tool_result(
        runtime.runner,
        planner_result,
        session,
        active_doc_cap=3,
        active_key_cap=5,
        forced_retired=retired_keys,
        forced_retired_docs=all_stale_docs(),
    )

    memory_report = final["submission"].setdefault("memory_report", {})
    spoken_hits = memory_report.setdefault("spoken_rule_hits", {})
    spoken_hits["retire"] = list(spoken_retire)

    derived_docs = derive_required_docs_from_state(episode)
    seen_docs = set(memory_report.get("docs_retrieved") or [])
    memory_report["docs_retrieved"] = list(memory_report.get("docs_retrieved") or []) + [
        d for d in derived_docs if d not in seen_docs
    ]

    rejected = list(memory_report.get("rejected_option_notes") or [])
    rejected_set = set(rejected)
    for note in derive_rejected_from_state(episode):
        if note not in rejected_set:
            rejected.append(note)
            rejected_set.add(note)
    memory_report["rejected_option_notes"] = rejected

    return final


def _fallback_result(runtime: StudentRuntime, session) -> Dict[str, Any]:
    fake_runner_result = {
        "parsed": {
            "flight_id": None,
            "hotel_id": None,
            "restaurant_id": None,
            "activity_id": None,
            "memory_report": {},
            "notes": "planner exceeded tool-round budget; minimal fallback submission",
        },
        "usage": runtime.runner.empty_usage(),
        "response_ids": [],
    }
    return _build_final(runtime, session, fake_runner_result)


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    cfg = runtime.system_config
    session = runtime.toolbox.new_session(
        episode=runtime.episode,
        retrieval_strategy=cfg["retrieval_strategy"],
        embedding_model=cfg.get("embedding_model"),
        max_results=cfg["max_tool_results"],
        role="single_memory",
    )
    session.bind_runner(runtime.runner)
    tools = session_tools(session, cfg)

    try:
        planner_result = _run_planner(runtime, session, tools, STUDENT_PLANNER_INSTRUCTIONS)
    except RuntimeError:
        return _fallback_result(runtime, session)

    return _build_final(runtime, session, planner_result)
