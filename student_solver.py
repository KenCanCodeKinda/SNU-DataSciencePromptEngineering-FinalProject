
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set

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
# 可泛化的数据收集指令
# ============================================================

DATA_COLLECTION_INSTRUCTIONS = (
    "You are a data collector. Search for ALL travel options.\n\n"
    
    "## SEARCH STRATEGY:\n"
    "1. Search broadly first, then narrow down:\n"
    "   search_hotels(city, max_results=20)\n"
    "   search_flights(origin, destination, max_results=20)\n"
    "   search_restaurants(city, max_results=20)\n"
    "   search_activities(city, max_results=20)\n\n"
    
    "2. Then apply specific filters based on constraints:\n"
    "   - If quiet_matters: search_hotels(city, quiet_min=0.7)\n"
    "   - If client_dinner: search_restaurants(city, client_ready_min=0.7)\n"
    "   - If teammate_vegan: search_restaurants(city, dietary='vegan')\n"
    "   - If rainy: search_activities(city, weather_safe_required=True)\n"
    "   - If airport_priority: search_hotels(city, airport_access_min=0.6)\n\n"
    
    "3. Memory management:\n"
    "   search_memory(query='stale OR retired', include_stale=true)\n"
    "   get_rejected_options()\n\n"
    
    "## OUTPUT:\n"
    "Return JSON with null IDs, all results in memory_report.\n"
    + MEMORY_REPORT_GUIDANCE
)


# ============================================================
# 评分函数 - 基于原则，不基于特定 ID
# ============================================================

def score_flight(flight: Dict[str, Any], state: Dict[str, Any], budget: int) -> Tuple[float, bool]:
    """评分航班 - 基于通用原则"""
    fare = flight.get("fare_total", 10000)
    score = 0.0
    
    # 硬约束
    if state.get("red_eye_avoid") and flight.get("red_eye"):
        return 0, False
    
    # 质量评分（通用）
    if not flight.get("red_eye"):
        score += 50
    
    # 直飞更好
    stops = flight.get("stops", 0)
    if stops == 0:
        score += 40
    elif stops == 1:
        score += 15
    else:
        score -= 20
    

    if flight.get("refundable"):
        score += 20
    
  
    if "meeting_safe" in flight.get("semantic_tags", []):
        score += 30
    
    # 价格评分（比例化，不依赖具体数字）
    price_ratio = fare / budget
    if price_ratio < 0.15:
        score += 50
    elif price_ratio < 0.25:
        score += 30
    elif price_ratio < 0.35:
        score += 15
    elif price_ratio > 0.5:
        score -= 30
    
    return score, True


def score_hotel(hotel: Dict[str, Any], state: Dict[str, Any], nights: int, 
                budget: int, meeting_zone: str) -> Tuple[float, bool]:
    """评分酒店 - 基于通用原则"""
    nightly = hotel.get("nightly_price", 500)
    total_cost = nightly * nights
    quiet_score = hotel.get("quiet_score", 0.0)
    airport_score = hotel.get("airport_access_score", 0.0)
    hotel_zone = hotel.get("zone", "")
    
    score = 0.0
    
    # 硬约束
    if state.get("quiet_matters") and quiet_score < 0.7:
        return 0, False
    
    # 预算硬约束
    if total_cost > budget * 0.6:
        return 0, False
    
    # 安静度（如果要求）
    if state.get("quiet_matters"):
        if quiet_score >= 0.85:
            score += 70
        elif quiet_score >= 0.7:
            score += 50
        else:
            score += quiet_score * 40
    else:
        score += quiet_score * 15
    
    # 区域匹配（通用原则：靠近会议区更好）
    if meeting_zone:
        if hotel_zone == meeting_zone:
            score += 80
        elif hotel_zone and meeting_zone in hotel_zone:
            score += 40
    
    # 机场接入
    if state.get("airport_priority"):
        if airport_score >= 0.8:
            score += 60
        elif airport_score >= 0.6:
            score += 40
        else:
            score += airport_score * 30
    else:
        score += airport_score * 15
    
    # 连锁酒店偏好
    if not hotel.get("chain"):
        score += 20
    elif state.get("chain_exception"):
        score += 10
    else:
        score -= 15
    
    # 价格评分
    price_ratio = total_cost / budget
    if price_ratio < 0.25:
        score += 50
    elif price_ratio < 0.35:
        score += 30
    elif price_ratio < 0.45:
        score += 15
    elif price_ratio > 0.55:
        score -= 20
    
    # 便利设施加分（通用好品质指标）
    if hotel.get("meeting_shuttle"):
        score += 20
    if hotel.get("late_checkout"):
        score += 10
    if hotel.get("airport_shuttle"):
        score += 10
    
    return score, True


def score_restaurant(restaurant: Dict[str, Any], state: Dict[str, Any], 
                     hotel_zone: str) -> Tuple[float, bool]:
    """评分餐厅 - 基于通用原则"""
    client_ready = restaurant.get("client_ready_score", 0.0)
    quiet_score = restaurant.get("quiet_score", 0.0)
    rest_area = restaurant.get("area", "")
    dietary = restaurant.get("dietary_flags", [])
    price_level = restaurant.get("price_level", 3)
    
    score = 0.0
    
    # 硬约束
    if state.get("client_dinner") and client_ready < 0.7:
        return 0, False
    
    if state.get("teammate_vegan"):
        if "vegan" not in dietary and "vegan_preorder" not in dietary:
            return 0, False
    
    # client_ready（如果要求）
    if state.get("client_dinner"):
        if client_ready >= 0.9:
            score += 70
        elif client_ready >= 0.8:
            score += 55
        elif client_ready >= 0.7:
            score += 40
        if restaurant.get("private_room"):
            score += 25
    else:
        score += client_ready * 15
    
    # 安静度
    if state.get("quiet_matters"):
        if quiet_score >= 0.8:
            score += 50
        elif quiet_score >= 0.6:
            score += 30
        else:
            score += quiet_score * 20
    else:
        score += quiet_score * 10
    
    # 素食支持
    if state.get("teammate_vegan"):
        if "vegan" in dietary:
            score += 50
        elif "vegan_preorder" in dietary:
            score += 40
        elif "vegetarian" in dietary:
            score += 20
    
    # 区域匹配（与酒店同区域更方便）
    if hotel_zone and rest_area:
        if rest_area == hotel_zone:
            score += 40
        elif hotel_zone in rest_area or rest_area in hotel_zone:
            score += 20
    
    # 价格（低价格更好）
    if price_level == 1:
        score += 35
    elif price_level == 2:
        score += 25
    elif price_level == 3:
        score += 10
    elif price_level >= 4:
        score -= 15
    
    # badge 检查
    if restaurant.get("badge_only") and not state.get("badge_available"):
        return 0, False
    
    return score, True


def score_activity(activity: Dict[str, Any], state: Dict[str, Any], 
                   weather: str, hotel_zone: str, remaining_budget: float) -> Tuple[float, bool]:
    """评分活动 - 基于通用原则"""
    price = activity.get("price", 100)
    semantic_tags = activity.get("semantic_tags", [])
    act_zone = activity.get("location_zone", "")
    
    score = 0.0
    
    # 硬约束
    if weather == "rainy":
        if "weather_safe" not in semantic_tags:
            if not activity.get("indoor"):
                return 0, False
    
    if price > remaining_budget * 1.2:
        return 0, False
    
    if activity.get("badge_only") and not state.get("badge_available"):
        return 0, False
    
    # 天气安全（下雨时重要）
    if weather == "rainy":
        if "weather_safe" in semantic_tags:
            score += 70
        if activity.get("indoor"):
            score += 30
        if not activity.get("indoor") and "weather_safe" not in semantic_tags:
            score -= 50
    else:
        # 晴天时户外更好
        if not activity.get("indoor"):
            score += 30
        if "weather_safe" in semantic_tags:
            score += 15
    
    # 区域匹配
    if hotel_zone and act_zone:
        if act_zone == hotel_zone:
            score += 35
        elif hotel_zone in act_zone or act_zone in hotel_zone:
            score += 15
    
    # 价格
    if price <= 30:
        score += 45
    elif price <= 60:
        score += 30
    elif price <= 100:
        score += 15
    else:
        score += max(0, 10 - price / 50)
    
    return score, True



def select_best_flight(flights: List[Dict[str, Any]], state: Dict[str, Any], 
                       budget: int) -> Tuple[Optional[str], float]:
    """选择最佳航班"""
    if not flights:
        return None, 0
    
    scored = [(f, score_flight(f, state, budget)[0]) for f in flights]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    best = scored[0][0]
    print(f"[Select] Flight: {best.get('flight_id')} (score={scored[0][1]:.0f})")
    
    return best.get("flight_id"), best.get("fare_total", 0)


def select_best_hotel(hotels: List[Dict[str, Any]], state: Dict[str, Any], 
                      nights: int, budget: int, meeting_zone: str,
                      remaining_budget: float) -> Tuple[Optional[str], float, str]:
    """选择最佳酒店"""
    if not hotels:
        return None, 0, ""
    
    scored = []
    for h in hotels:
        s, valid = score_hotel(h, state, nights, budget, meeting_zone)
        if valid:
            scored.append((h, s))
    
    if not scored:
        # 放宽约束
        for h in hotels:
            s, _ = score_hotel(h, state, nights, budget, meeting_zone)
            scored.append((h, s - 100))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0][0]
    total_cost = best.get("nightly_price", 0) * nights
    zone = best.get("zone", "")
    
    print(f"[Select] Hotel: {best.get('hotel_id')} (price=${best.get('nightly_price')}, score={scored[0][1]:.0f})")
    
    return best.get("hotel_id"), total_cost, zone


def select_best_restaurant(restaurants: List[Dict[str, Any]], state: Dict[str, Any],
                           hotel_zone: str) -> Tuple[Optional[str], float]:
    """选择最佳餐厅"""
    if not restaurants:
        return None, 0
    
    scored = [(r, score_restaurant(r, state, hotel_zone)[0]) for r in restaurants]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    best = scored[0][0]
    price_level = best.get("price_level", 2)
    est_cost = [25, 45, 75, 120][min(price_level - 1, 3)]
    
    print(f"[Select] Restaurant: {best.get('restaurant_id')} (score={scored[0][1]:.0f})")
    
    return best.get("restaurant_id"), est_cost


def select_best_activity(activities: List[Dict[str, Any]], state: Dict[str, Any],
                         weather: str, hotel_zone: str, remaining_budget: float) -> Tuple[Optional[str], float]:
    """选择最佳活动"""
    if not activities:
        return None, 0
    
    scored = [(a, score_activity(a, state, weather, hotel_zone, remaining_budget)[0]) for a in activities]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    best = scored[0][0]
    price = best.get("price", 0)
    
    print(f"[Select] Activity: {best.get('activity_id')} (price=${price}, score={scored[0][1]:.0f})")
    
    return best.get("activity_id"), price


# ============================================================
# 数据提取和主流程（与之前相同）
# ============================================================

def _extract_search_results(session) -> Tuple[List, List, List, List]:
    """从 session 历史提取搜索结果"""
    flights = []
    hotels = []
    restaurants = []
    activities = []
    episode = session.episode
    city = episode["city"]
    origin = episode["origin"]
    
    for trace in session.tool_trace:
        tool = trace.get("tool")
        arguments = trace.get("arguments", {}).copy()
        
        if tool == "search_flights":
            if "origin" not in arguments:
                arguments["origin"] = origin
            if "destination" not in arguments:
                arguments["destination"] = city
            result = session.search_flights(**arguments)
            flights.extend(result.get("items", []))
            
        elif tool == "search_hotels":
            if "city" not in arguments:
                arguments["city"] = city
            result = session.search_hotels(**arguments)
            hotels.extend(result.get("items", []))
            
        elif tool == "search_restaurants":
            if "city" not in arguments:
                arguments["city"] = city
            result = session.search_restaurants(**arguments)
            restaurants.extend(result.get("items", []))
            
        elif tool == "search_activities":
            if "city" not in arguments:
                arguments["city"] = city
            result = session.search_activities(**arguments)
            activities.extend(result.get("items", []))
    
    # 去重
    def dedupe(items, key):
        seen = set()
        unique = []
        for item in items:
            id_val = item.get(key)
            if id_val and id_val not in seen:
                seen.add(id_val)
                unique.append(item)
        return unique
    
    return (dedupe(flights, "flight_id"),
            dedupe(hotels, "hotel_id"),
            dedupe(restaurants, "restaurant_id"),
            dedupe(activities, "activity_id"))


def _collect_data(runtime: StudentRuntime, session, tools) -> Dict[str, Any]:
    """收集数据"""
    cfg = runtime.system_config
    model = cfg["model"]
    
    return runtime.runner.run_tool_agent_json(
        model=model,
        instructions=DATA_COLLECTION_INSTRUCTIONS,
        input_text=episode_prompt(runtime.episode),
        json_schema=final_decision_schema(),
        schema_name="data_collector",
        tools=tools,
        tool_handler=session.dispatch,
        max_output_tokens=cfg["max_output_tokens"],
        reasoning_effort="low" if model.startswith("gpt-5") else None,
        text_verbosity="low" if model.startswith("gpt-5") else None,
        metadata={
            "system": cfg["system_name"],
            "trip_id": runtime.episode["trip_id"],
            "role": "collector",
        },
        max_tool_rounds=cfg.get("max_tool_rounds", 12),
    )


def _python_select(runtime: StudentRuntime, session, collector_result: Dict[str, Any]) -> Dict[str, Any]:
    """Python 选择最优选项"""
    episode = runtime.episode
    state = episode.get("scenario_state", {})
    weather = episode.get("weather", "")
    budget = episode.get("budget_total", 10000)
    nights = episode.get("nights", 1)
    meeting_zone = episode.get("meeting_zone", "")
    
    flights, hotels, restaurants, activities = _extract_search_results(session)
    
    print(f"\n[Budget] ${budget}, Nights={nights}, Meeting zone={meeting_zone}")
    print(f"[Found] {len(flights)} flights, {len(hotels)} hotels, "
          f"{len(restaurants)} restaurants, {len(activities)} activities")
    
    remaining = budget
    
    flight_id, flight_cost = select_best_flight(flights, state, budget)
    remaining -= flight_cost
    
    hotel_id, hotel_cost, hotel_zone = select_best_hotel(
        hotels, state, nights, budget, meeting_zone, remaining
    )
    remaining -= hotel_cost
    
    restaurant_id, restaurant_cost = select_best_restaurant(restaurants, state, hotel_zone)
    remaining -= restaurant_cost
    
    activity_id, activity_cost = select_best_activity(
        activities, state, weather, hotel_zone, remaining
    )
    
    total = flight_cost + hotel_cost + restaurant_cost + activity_cost
    print(f"\n[Total] ${total:.0f} / ${budget}")
    
    memory_report = collector_result.get("parsed", {}).get("memory_report", {})
    
    return {
        "parsed": {
            "flight_id": flight_id,
            "hotel_id": hotel_id,
            "restaurant_id": restaurant_id,
            "activity_id": activity_id,
            "memory_report": memory_report,
            "notes": f"Total=${total:.0f}",
        },
        "usage": collector_result["usage"],
        "response_ids": collector_result.get("response_ids", []),
    }


def _build_final(runtime: StudentRuntime, session, selection_result: Dict[str, Any]) -> Dict[str, Any]:
    """构建最终结果"""
    episode = runtime.episode
    retired_keys, _ = derive_retired_from_state(episode)
    
    final = tool_result(
        runtime.runner,
        selection_result,
        session,
        active_doc_cap=3,
        active_key_cap=5,
        forced_retired=retired_keys,
        forced_retired_docs=all_stale_docs(),
    )
    
    memory_report = final["submission"].setdefault("memory_report", {})
    memory_report["spoken_rule_hits"] = derive_spoken_rule_hits_from_state(episode)
    
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
    
    print(f"\n[FINAL] {episode['trip_id']}")
    print(f"  F:{final['submission'].get('flight_id')} "
          f"H:{final['submission'].get('hotel_id')} "
          f"R:{final['submission'].get('restaurant_id')} "
          f"A:{final['submission'].get('activity_id')}")
    
    return final


def _fallback_result(runtime: StudentRuntime, session) -> Dict[str, Any]:
    """回退结果"""
    fake_result = {
        "parsed": {
            "flight_id": None,
            "hotel_id": None,
            "restaurant_id": None,
            "activity_id": None,
            "memory_report": {},
            "notes": "fallback",
        },
        "usage": runtime.runner.empty_usage(),
        "response_ids": [],
    }
    return _build_final(runtime, session, fake_result)


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """主入口"""
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
    
    state = runtime.episode.get("scenario_state", {})
    active = [k for k, v in state.items() if v]
    print(f"\n{'='*60}")
    print(f"[START] {runtime.episode['trip_id']}")
    print(f"[START] Active: {active}")
    print(f"[START] Weather: {runtime.episode.get('weather')}")
    print(f"[START] Meeting zone: {runtime.episode.get('meeting_zone')}")
    print(f"[START] Budget: ${runtime.episode.get('budget_total')}")
    print(f"{'='*60}")
    
    try:
        collector_result = _collect_data(runtime, session, tools)
        selection_result = _python_select(runtime, session, collector_result)
        return _build_final(runtime, session, selection_result)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return _fallback_result(runtime, session)

# from __future__ import annotations

# from typing import Any, Dict

# from llm_agents import (
#     MEMORY_REPORT_GUIDANCE,
#     episode_prompt,
#     final_decision_schema,
#     session_tools,
#     tool_result,
# )
# from runtime_api import StudentRuntime
# from student_custom_tools_template import (
#     all_stale_docs,
#     derive_rejected_from_state,
#     derive_required_docs_from_state,
#     derive_retired_from_state,
#     derive_spoken_rule_hits_from_state,
# )


# STUDENT_PLANNER_INSTRUCTIONS = (
#     "You are a memory-aware travel planner. Each turn can override prior assumptions. "
#     "Pick one flight, one hotel, one restaurant, one activity that satisfy ALL currently active constraints, AND fill memory_report so the evaluator can verify your context discipline.\n\n"
#     "Rules:\n"
#     "1. Persistent (profile/venue/policy) vs momentary (scenario_state, latest turns). Momentary wins for this trip only — do not promote one-off overrides into active_context_keys as if persistent.\n"
#     "2. Filter first. ≤2 search_* calls per category. Use zone, quiet_min, refundable_only, dietary, weather_safe_required.\n"
#     "3. Lean active_context_keys ≤ 5, canonical vocabulary, each tied to a real constraint THIS trip.\n"
#     "4. Retire sweep: call search_memory once with \"stale OR retired OR override\" (include_stale=true if available). List old_* keys ONLY when the user actually invalidated them.\n"
#     "5. Rejected options: call get_rejected_options once, list \"reason_key:ID\" in rejected_option_notes, never pick a rejected ID.\n"
#     "6. Trigger rich-context tools (get_partner_promotions/get_event_context/get_loyalty_profile/get_stakeholder_brief/get_booking_constraints/get_option_dependencies) only when scenario_hooks calls for them. ONCE each.\n\n"
#     "For each retired old_* key, pair the matching stale:* doc id in retired_docs. Faithfulness: if not grounded in tools or turns, OMIT.\n\n"
#     "Output strict JSON. notes ≤ 320 chars.\n\n"
#     + MEMORY_REPORT_GUIDANCE
# )


# def _run_planner(
#     runtime: StudentRuntime,
#     session,
#     tools,
#     instructions: str,
#     *,
#     schema_name: str = "student_decision",
#     max_tool_rounds: int | None = None,
# ) -> Dict[str, Any]:
#     cfg = runtime.system_config
#     model = cfg["model"]
#     return runtime.runner.run_tool_agent_json(
#         model=model,
#         instructions=instructions,
#         input_text=episode_prompt(runtime.episode),
#         json_schema=final_decision_schema(),
#         schema_name=schema_name,
#         tools=tools,
#         tool_handler=session.dispatch,
#         max_output_tokens=cfg["max_output_tokens"],
#         reasoning_effort="low" if model.startswith("gpt-5") else None,
#         text_verbosity="low" if model.startswith("gpt-5") else None,
#         metadata={
#             "system": cfg["system_name"],
#             "trip_id": runtime.episode["trip_id"],
#             "role": "student",
#         },
#         max_tool_rounds=max_tool_rounds if max_tool_rounds is not None else cfg.get("max_tool_rounds", 9),
#     )


# def _build_final(runtime: StudentRuntime, session, planner_result: Dict[str, Any]) -> Dict[str, Any]:
#     """Apply tool_result + deterministic memory enrichments. Used by both main and fallback paths."""
#     episode = runtime.episode
#     retired_keys, _ = derive_retired_from_state(episode)

#     final = tool_result(
#         runtime.runner,
#         planner_result,
#         session,
#         active_doc_cap=3,
#         active_key_cap=5,
#         forced_retired=retired_keys,
#         forced_retired_docs=all_stale_docs(),
#     )

#     memory_report = final["submission"].setdefault("memory_report", {})
#     memory_report["spoken_rule_hits"] = derive_spoken_rule_hits_from_state(episode)

#     derived_docs = derive_required_docs_from_state(episode)
#     seen_docs = set(memory_report.get("docs_retrieved") or [])
#     memory_report["docs_retrieved"] = list(memory_report.get("docs_retrieved") or []) + [
#         d for d in derived_docs if d not in seen_docs
#     ]

#     rejected = list(memory_report.get("rejected_option_notes") or [])
#     rejected_set = set(rejected)
#     for note in derive_rejected_from_state(episode):
#         if note not in rejected_set:
#             rejected.append(note)
#             rejected_set.add(note)
#     memory_report["rejected_option_notes"] = rejected

#     return final


# def _fallback_result(runtime: StudentRuntime, session) -> Dict[str, Any]:
#     fake_runner_result = {
#         "parsed": {
#             "flight_id": None,
#             "hotel_id": None,
#             "restaurant_id": None,
#             "activity_id": None,
#             "memory_report": {},
#             "notes": "planner exceeded tool-round budget; minimal fallback submission",
#         },
#         "usage": runtime.runner.empty_usage(),
#         "response_ids": [],
#     }
#     return _build_final(runtime, session, fake_runner_result)


# def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
#     cfg = runtime.system_config
#     session = runtime.toolbox.new_session(
#         episode=runtime.episode,
#         retrieval_strategy=cfg["retrieval_strategy"],
#         embedding_model=cfg.get("embedding_model"),
#         max_results=cfg["max_tool_results"],
#         role="single_memory",
#     )
#     session.bind_runner(runtime.runner)
#     tools = session_tools(session, cfg)

#     try:
#         planner_result = _run_planner(runtime, session, tools, STUDENT_PLANNER_INSTRUCTIONS)
#     except RuntimeError:
#         return _fallback_result(runtime, session)

#     return _build_final(runtime, session, planner_result)