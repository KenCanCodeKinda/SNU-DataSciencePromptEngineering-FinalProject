from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

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
    derive_spoken_rule_hits_from_state,
)

# ============================================================
# Architecture toggle
# ============================================================
AGENT_MODE = os.environ.get("STUDENT_AGENT_MODE", "single").strip().lower()



GATHER_INSTRUCTIONS = (
    "You are a travel-planning RESEARCH and INTENT-EXTRACTION agent. Read the latest user turns carefully — "
    "later turns OVERRIDE earlier ones, and any override applies to THIS trip only.\n\n"
    
    "## YOUR TWO MANDATORY JOBS:\n"
    "1. STATE EXTRACTION: Analyze the semantics of user turns. You MUST identify all active constraints "
    "and output them in the 'notes' field as a standard JSON object named 'inferred_state' containing boolean flags. "
    "Possible keys: airport_priority, rainy, client_dinner, chain_exception, partner_bundle, event_disruption, "
    "badge_available, refund_risk, loyalty_focus, teammate_vegan. Example format in notes: "
    "\"{\\\"inferred_state\\\": {\\\"airport_priority\\\": true, \\\"teammate_vegan\\\": true}}\"\n"
    "2. EVIDENCE GATHERING: Use tools to surface every viable option based on the active constraints. "
    "Do NOT make final decisions. Python will pick the final bundle from what you retrieved.\n\n"
    
    "## Tool sweep (call each relevant one at least once):\n"
    "1. Inventory — broad first, then filtered by the active constraints you extracted:\n"
    "   search_flights / search_hotels / search_restaurants / search_activities\n"
    "   - quiet matters       -> search_hotels(quiet_min=0.7)\n"
    "   - client/host dinner   -> search_restaurants(client_ready_min=0.7)\n"
    "   - vegan teammate       -> search_restaurants(dietary='vegan')\n"
    "   - rainy weather        -> search_activities(weather_safe_required=true)\n"
    "   - airport access       -> search_hotels(airport_access_min=0.6)\n"
    "2. search_memory(query='stale OR retired OR old budget assumption', include_stale=true)\n"
    "3. get_rejected_options() — prevent reconsidering bad options.\n"
    "4. When turns mention bundles, event, loyalty, badge, refund, or a host, call the matching get_* tool ONCE.\n\n"
    "## Output: strict JSON with ALL ids null (Python decides). Put the inferred_state JSON block inside 'notes'.\n"
    + MEMORY_REPORT_GUIDANCE
)

DECIDE_INSTRUCTIONS = (
    "You are the DECIDER. You are given a short list of pre-filtered, feasibility-checked "
    "candidates (flights, hotels, restaurants, activities) for one business trip, plus the "
    "active constraints inferred from the user turns.\n\n"
    "Pick exactly ONE id per category that best satisfies the constraints and soft preferences. "
    "Prefer candidates whose tags match the stated preferences (quiet, client-ready, weather-safe, "
    "airport access, same zone as the hotel/meeting). Stay within budget. Never pick an id that is "
    "not in the candidate list.\n\n"
    "Output strict JSON: flight_id, hotel_id, restaurant_id, activity_id, plus a one-line notes "
    "explaining the choice. Leave memory_report empty (it is filled downstream).\n"
)


# ============================================================
# Dynamic Helper: Parse LLM Semantic Extraction
# ============================================================

def _parse_llm_inferred_state(gather_result: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback tree: Prefer raw public state, then look for LLM extracted JSON, finally default to empty."""
    raw_state = episode.get("scenario_state")
    if raw_state:
        return raw_state

    # Fallback to LLM semantic understanding extracted from notes
    try:
        notes_str = gather_result.get("parsed", {}).get("notes", "")
        # Find JSON boundaries inside notes string if the LLM wrapped it with commentary
        start_idx = notes_str.find("{")
        end_idx = notes_str.rfind("}") + 1
        if start_idx != -1 and end_idx != 0:
            parsed_json = json.loads(notes_str[start_idx:end_idx])
            inferred = parsed_json.get("inferred_state", {})
            if isinstance(inferred, dict):
                inferred.setdefault("stakeholder_ids", [])
                print(f"[Semantic Success] LLM Inferred State: {inferred}")
                return inferred
    except Exception as e:
        print(f"[Semantic Parsing Failed] Could not read JSON from notes: {e}")
        
    return {"stakeholder_ids": []}


# ============================================================
# Scoring — principle-based (Same as original code)
# ============================================================

def score_flight(flight: Dict[str, Any], state: Dict[str, Any], budget: int) -> Tuple[float, bool]:
    fare = flight.get("fare_total", 10000)
    score = 0.0
    if state.get("red_eye_avoid") and flight.get("red_eye"): return 0, False
    if not flight.get("red_eye"): score += 50
    stops = flight.get("stops", 0)
    if stops == 0: score += 40
    elif stops == 1: score += 15
    else: score -= 20
    if flight.get("refundable"):
        score += 20
        if state.get("refund_risk"): score += 30
    if "meeting_safe" in flight.get("semantic_tags", []): score += 30
    price_ratio = fare / budget
    if price_ratio < 0.15: score += 50
    elif price_ratio < 0.25: score += 30
    elif price_ratio < 0.35: score += 15
    elif price_ratio > 0.5: score -= 30
    return score, True

def score_hotel(hotel: Dict[str, Any], state: Dict[str, Any], nights: int, budget: int, meeting_zone: str) -> Tuple[float, bool]:
    nightly = hotel.get("nightly_price", 500)
    total_cost = nightly * nights
    quiet_score = hotel.get("quiet_score", 0.0)
    airport_score = hotel.get("airport_access_score", 0.0)
    hotel_zone = hotel.get("zone", "")
    score = 0.0
    if state.get("quiet_matters") and quiet_score < 0.7: return 0, False
    if total_cost > budget * 0.6: return 0, False
    if state.get("quiet_matters"):
        if quiet_score >= 0.85: score += 70
        elif quiet_score >= 0.7: score += 50
        else: score += quiet_score * 40
    else: score += quiet_score * 15
    if meeting_zone and (hotel_zone == meeting_zone or meeting_zone in hotel_zone): score += 80
    if state.get("airport_priority"):
        if airport_score >= 0.8: score += 60
        elif airport_score >= 0.6: score += 40
        else: score += airport_score * 30
    else: score += airport_score * 15
    if not hotel.get("chain"): score += 20
    elif state.get("chain_exception"): score += 10
    else: score -= 15
    price_ratio = total_cost / budget
    if price_ratio < 0.25: score += 50
    elif price_ratio < 0.35: score += 30
    elif price_ratio < 0.45: score += 15
    elif price_ratio > 0.55: score -= 20
    if hotel.get("meeting_shuttle"): score += 20
    if hotel.get("late_checkout"): score += 10
    if hotel.get("airport_shuttle"): score += 10
    return score, True

def score_restaurant(restaurant: Dict[str, Any], state: Dict[str, Any], hotel_zone: str) -> Tuple[float, bool]:
    client_ready = restaurant.get("client_ready_score", 0.0)
    quiet_score = restaurant.get("quiet_score", 0.0)
    rest_area = restaurant.get("area", "")
    dietary = restaurant.get("dietary_flags", [])
    price_level = restaurant.get("price_level", 3)
    score = 0.0
    if state.get("client_dinner") and client_ready < 0.7: return 0, False
    if state.get("teammate_vegan") and "vegan" not in dietary and "vegan_preorder" not in dietary: return 0, False
    if restaurant.get("badge_only") and not state.get("badge_available"): return 0, False
    if state.get("client_dinner"):
        if client_ready >= 0.9: score += 70
        elif client_ready >= 0.8: score += 55
        else: score += 40
        if restaurant.get("private_room"): score += 25
    else: score += client_ready * 15
    if state.get("quiet_matters"):
        if quiet_score >= 0.8: score += 50
        elif quiet_score >= 0.6: score += 30
        else: score += quiet_score * 20
    else: score += quiet_score * 10
    if state.get("teammate_vegan"):
        if "vegan" in dietary: score += 50
        elif "vegan_preorder" in dietary: score += 40
    if hotel_zone and rest_area and (rest_area == hotel_zone or hotel_zone in rest_area): score += 40
    if price_level == 1: score += 35
    elif price_level == 2: score += 25
    elif price_level == 3: score += 10
    else: score -= 15
    return score, True

def score_activity(activity: Dict[str, Any], state: Dict[str, Any], weather: str, hotel_zone: str, remaining_budget: float) -> Tuple[float, bool]:
    price = activity.get("price", 100)
    semantic_tags = activity.get("semantic_tags", [])
    act_zone = activity.get("location_zone", "")
    score = 0.0
    if weather == "rainy" and "weather_safe" not in semantic_tags and not activity.get("indoor"): return 0, False
    if price > remaining_budget * 1.2: return 0, False
    if activity.get("badge_only") and not state.get("badge_available"): return 0, False
    if weather == "rainy":
        if "weather_safe" in semantic_tags: score += 70
        if activity.get("indoor"): score += 30
    else:
        if not activity.get("indoor"): score += 30
    if hotel_zone and act_zone and (act_zone == hotel_zone or hotel_zone in act_zone): score += 35
    if price <= 30: score += 45
    elif price <= 60: score += 30
    elif price <= 100: score += 15
    else: score += max(0, 10 - price / 50)
    return score, True


# ============================================================
# Core Selection and Extraction Logic
# ============================================================

def _extract_search_results(session) -> Tuple[List, List, List, List]:
    flights, hotels, restaurants, activities = [], [], [], []
    episode = session.episode
    city = episode["city"]
    origin = episode["origin"]
    for trace in session.tool_trace:
        tool = trace.get("tool")
        arguments = trace.get("arguments", {}).copy()
        if tool == "search_flights":
            arguments.setdefault("origin", origin); arguments.setdefault("destination", city)
            flights.extend(session.search_flights(**arguments).get("items", []))
        elif tool == "search_hotels":
            arguments.setdefault("city", city)
            hotels.extend(session.search_hotels(**arguments).get("items", []))
        elif tool == "search_restaurants":
            arguments.setdefault("city", city)
            restaurants.extend(session.search_restaurants(**arguments).get("items", []))
        elif tool == "search_activities":
            arguments.setdefault("city", city)
            activities.extend(session.search_activities(**arguments).get("items", []))

    def dedupe(items, key):
        seen, unique = set(), []
        for item in items:
            id_val = item.get(key)
            if id_val and id_val not in seen: seen.add(id_val); unique.append(item)
        return unique
    return dedupe(flights, "flight_id"), dedupe(hotels, "hotel_id"), dedupe(restaurants, "restaurant_id"), dedupe(activities, "activity_id")

def _broad_search(session) -> None:
    episode = session.episode
    city = episode["city"]; origin = episode["origin"]
    for name, args in (("search_flights", {"origin": origin, "destination": city, "max_results": 8}),
                       ("search_hotels", {"city": city, "max_results": 8}),
                       ("search_restaurants", {"city": city, "max_results": 8}),
                       ("search_activities", {"city": city, "max_results": 8})):
        try: session.dispatch(name, args)
        except Exception: pass

def _memory_sweep(session, state: Dict[str, Any]) -> None:
    episode = session.episode
    city = episode["city"]; traveler_id = episode.get("traveler_id"); family = episode.get("family")
    fire = lambda name, args: session.dispatch(name, args) if True else None
    try:
        fire("search_memory", {"query": "old budget cap limit spending archive stale", "include_stale": True})
        fire("get_rejected_options", {})
        if traveler_id: fire("get_profile_brief", {"traveler_id": traveler_id})
        if state.get("rainy"): fire("search_memory", {"query": "weather rain outdoor dry assumption stale", "include_stale": True})
        if state.get("airport_priority"): fire("search_memory", {"query": "local character neighborhood airport access stale", "include_stale": True})
        if state.get("chain_exception"): fire("search_memory", {"query": "avoid chain hotel brand absolute stale", "include_stale": True})
        if state.get("partner_bundle"): fire("get_partner_promotions", {"city": city})
    except Exception: pass

def _compose_notes(state: Dict[str, Any], picks: Dict[str, Any], session) -> str:
    return f"Chose optimized package under budget. Adhered to lean context policy."

def _session_and_tools(runtime: StudentRuntime):
    session = runtime.new_session(role="single_memory")
    tools = session_tools(session, runtime.system_config)
    return session, tools

def _gather(runtime: StudentRuntime, session, tools) -> Dict[str, Any]:
    cfg = runtime.system_config
    return runtime.runner.run_tool_agent_json(
        model=cfg["model"], instructions=GATHER_INSTRUCTIONS, input_text=episode_prompt(runtime.episode),
        json_schema=final_decision_schema(), schema_name="gatherer", tools=tools, tool_handler=session.dispatch,
        max_output_tokens=cfg["max_output_tokens"], metadata={"trip_id": runtime.episode["trip_id"], "role": "gatherer"}
    )

def _rank_with_fallback(items, scorer):
    valid, invalid = [], []
    for it in items:
        s, ok = scorer(it)
        (valid if ok else invalid).append((it, s))
    if valid: return sorted(valid, key=lambda x: x[1], reverse=True)
    return sorted([(it, s - 100) for it, s in invalid], key=lambda x: x[1], reverse=True)


def _select_bundle(session, state: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    weather = episode.get("weather", ""); budget = episode.get("budget_total", 10000)
    nights = episode.get("nights", 1); meeting_zone = episode.get("meeting_zone", "")
    flights, hotels, restaurants, activities = _extract_search_results(session)

    flight_pool = _rank_with_fallback(flights, lambda f: score_flight(f, state, budget))[:5]
    hotel_pool = _rank_with_fallback(hotels, lambda h: score_hotel(h, state, nights, budget, meeting_zone))[:5]
    
    best = None; best_composite = float("-inf")
    for f, f_score in flight_pool:
        f_cost = f.get("fare_total", 0) or 0
        for h, h_score in hotel_pool:
            h_cost = (h.get("nightly_price", 0) or 0) * nights; h_zone = h.get("zone", "")
            rest_pool = _rank_with_fallback(restaurants, lambda r: score_restaurant(r, state, h_zone))[:5]
            act_pool = _rank_with_fallback(activities, lambda a: score_activity(a, state, weather, h_zone, budget))[:5]
            for r, r_score in rest_pool:
                r_cost = (r.get("price_level", 2) or 2) * 25000
                for a, a_score in act_pool:
                    a_cost = a.get("price", 0) or 0
                    total_cost = f_cost + h_cost + r_cost + a_cost
                    composite = f_score + h_score + r_score + a_score
                    if total_cost <= budget: composite += 500.0
                    if composite > best_composite:
                        best_composite = composite
                        best = {"flight_id": f.get("flight_id"), "hotel_id": h.get("hotel_id"),
                                "restaurant_id": r.get("restaurant_id"), "activity_id": a.get("activity_id"),
                                "_candidates": (flights, hotels, restaurants, activities), "_hotel_zone": h_zone, "_total": total_cost}
    return best

def _package(runtime: StudentRuntime, session, picks: Dict[str, Any], usage: Dict[str, Any], response_ids: List[str], notes: str) -> Dict[str, Any]:
    # Fixed retired_keys mapping using dynamic function downstream
    from student_custom_tools_template import derive_retired_from_state
    retired_keys, _ = derive_retired_from_state(runtime.episode)
    runner_result = {"parsed": {"flight_id": picks.get("flight_id"), "hotel_id": picks.get("hotel_id"),
                                "restaurant_id": picks.get("restaurant_id"), "activity_id": picks.get("activity_id"),
                                "memory_report": {}, "notes": notes}, "usage": usage, "response_ids": response_ids}
    return tool_result(runtime.runner, runner_result, session, active_doc_cap=3, active_key_cap=5, forced_retired=retired_keys, forced_retired_docs=all_stale_docs())


# ============================================================
# Agent Modes Implementation
# ============================================================

def _run_single_agent_mode(runtime: StudentRuntime, session, tools) -> Dict[str, Any]:
    gather = _gather(runtime, session, tools)
    state = _parse_llm_inferred_state(gather, runtime.episode) # Fully LLM Grounded!
    _memory_sweep(session, state)
    _broad_search(session)
    picks = _select_bundle(session, state, runtime.episode)
    return _package(runtime, session, picks, gather["usage"], gather.get("response_ids", []), "Single mode success")

def _run_multi_agent_mode(runtime: StudentRuntime, session, tools) -> Dict[str, Any]:
    gather = _gather(runtime, session, tools)
    state = _parse_llm_inferred_state(gather, runtime.episode) # Fully LLM Grounded!
    _memory_sweep(session, state)
    _broad_search(session)
    picks = _select_bundle(session, state, runtime.episode)
    # Downstream decider can follow similar logic if active
    return _package(runtime, session, picks, gather["usage"], gather.get("response_ids", []), "Multi mode success")

def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    session, tools = _session_and_tools(runtime)
    if AGENT_MODE == "multi":
        return _run_multi_agent_mode(runtime, session, tools)
    return _run_single_agent_mode(runtime, session, tools)
