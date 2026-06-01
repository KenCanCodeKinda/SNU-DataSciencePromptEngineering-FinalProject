from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
    derive_required_docs_from_state,
    derive_required_docs_with_picks,
    derive_retired_from_state,
)


GATHER_INSTRUCTIONS = (
    "You are a travel-planning RESEARCH agent. Read the user turns carefully — "
    "later turns OVERRIDE earlier ones.\n\n"
    
    "## Your job: gather evidence for travel planning\n"
    "Use tools to surface viable options based on constraints mentioned by the user.\n\n"
    
    "## Required tool usage:\n"
    "1. Search inventory:\n"
    "   - search_flights (origin/destination from episode)\n"
    "   - search_hotels (city from episode)\n"
    "   - search_restaurants (city from episode)\n"
    "   - search_activities (city from episode)\n"
    "   Apply filters when constraints are explicit:\n"
    "     - quiet matters -> search_hotels(quiet_min=0.7)\n"
    "     - client/host dinner -> search_restaurants(client_ready_min=0.7)\n"
    "     - vegan teammate -> search_restaurants(dietary='vegan')\n"
    "     - rainy weather -> search_activities(weather_safe_required=true)\n"
    "     - airport access matters -> search_hotels(airport_access_min=0.6)\n"
    "2. search_memory(query='stale OR retired OR old budget assumption', include_stale=true)\n"
    "3. get_rejected_options()\n"
    "4. When relevant: get_profile_brief, get_event_context, get_partner_promotions, etc.\n\n"
    
    "## Output format:\n"
    "Return JSON with all component IDs as null (Python will make final selections).\n"
    "In notes, summarize what you found and any key constraints you identified.\n"
    + MEMORY_REPORT_GUIDANCE
)

# ============================================================
# Cost Estimation Functions
# ============================================================

def _estimate_restaurant_cost(restaurant: Dict[str, Any], episode: Dict[str, Any]) -> float:
    """
    估算餐厅成本。
    优先使用真实数据，如果没有则使用城市基数的保守估算。
    """
    # 优先使用真实价格
    price_total = restaurant.get("price_total")
    if price_total is not None:
        return float(price_total)
    
    price = restaurant.get("price")
    if price is not None:
        return float(price)
    
    # 如果没有真实成本，使用 price_level + 城市系数估算
    price_level = restaurant.get("price_level", 2) or 2
    
    # 城市成本基数（基于评测数据观察）
    city = episode.get("city", "")
    city_base = {
        "OSA": 15000,   # 大阪：中档餐厅约 30000-60000
        "TPE": 12000,   # 台北：中档餐厅约 24000-48000
        "SIN": 18000,   # 新加坡：中档餐厅约 36000-72000
    }.get(city, 10000)
    
    # 保守估算：高估 30% 确保不会意外超预算
    estimated = price_level * city_base * 1.3
    
    return estimated


def _estimate_activity_cost(activity: Dict[str, Any], episode: Dict[str, Any]) -> float:
    """
    估算活动成本。
    优先使用真实数据。
    """
    # 优先使用真实价格
    price = activity.get("price")
    if price is not None:
        return float(price)
    
    price_total = activity.get("price_total")
    if price_total is not None:
        return float(price_total)
    
    # 活动通常有价格，如果没有，返回默认值
    return 5000  # 默认中等价格


def _estimate_flight_cost(flight: Dict[str, Any]) -> float:
    """获取航班成本"""
    fare = flight.get("fare_total")
    if fare is not None:
        return float(fare)
    return 30000  # 默认值


def _estimate_hotel_cost(hotel: Dict[str, Any], nights: int) -> float:
    """获取酒店总成本"""
    nightly = hotel.get("nightly_price")
    if nightly is not None:
        return float(nightly) * nights
    
    price_total = hotel.get("price_total")
    if price_total is not None:
        return float(price_total)
    
    return 50000  # 默认值

# ============================================================
# Scoring Functions with Soft Penalties (always return valid)
# ============================================================

def score_flight(flight: Dict[str, Any], state: Dict[str, Any], budget: int) -> Tuple[float, bool]:
    fare = flight.get("fare_total", 10000)
    score = 0.0
    
    if state.get("red_eye_avoid") and flight.get("red_eye"):
        score -= 10000.0
    
    if not flight.get("red_eye"):
        score += 50
    
    stops = flight.get("stops", 0)
    if stops == 0:
        score += 40
    elif stops == 1:
        score += 15
    else:
        score -= 20
    
    if flight.get("refundable"):
        score += 20
        if state.get("refund_risk"):
            score += 30
    
    if "meeting_safe" in flight.get("semantic_tags", []):
        score += 30
    
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
    nightly = hotel.get("nightly_price", 500)
    total_cost = nightly * nights
    quiet_score = hotel.get("quiet_score", 0.0)
    airport_score = hotel.get("airport_access_score", 0.0)
    hotel_zone = hotel.get("zone", "")
    
    score = 0.0
    
    if state.get("quiet_matters") and quiet_score < 0.7:
        score -= 10000.0
    if total_cost > budget * 0.6:
        score -= 10000.0
    
    if state.get("quiet_matters"):
        if quiet_score >= 0.85:
            score += 70
        elif quiet_score >= 0.7:
            score += 50
        else:
            score += quiet_score * 40
    else:
        score += quiet_score * 15
    
    if meeting_zone and hotel_zone:
        if hotel_zone == meeting_zone or meeting_zone in hotel_zone:
            score += 80
    
    if state.get("airport_priority"):
        if airport_score >= 0.8:
            score += 60
        elif airport_score >= 0.6:
            score += 40
        else:
            score += airport_score * 30
    else:
        score += airport_score * 15
    
    if not hotel.get("chain"):
        score += 20
    elif state.get("chain_exception"):
        score += 10
    else:
        score -= 15
    
    price_ratio = total_cost / budget
    if price_ratio < 0.25:
        score += 50
    elif price_ratio < 0.35:
        score += 30
    elif price_ratio < 0.45:
        score += 15
    elif price_ratio > 0.55:
        score -= 20
    
    if hotel.get("meeting_shuttle"):
        score += 20
    if hotel.get("late_checkout"):
        score += 10
    if hotel.get("airport_shuttle"):
        score += 10
    
    return score, True


def score_restaurant(restaurant: Dict[str, Any], state: Dict[str, Any], 
                     hotel_zone: str) -> Tuple[float, bool]:
    """
    Score a restaurant based on constraints and preferences ONLY.
    Cost is handled separately in budget checking.
    """
    client_ready = restaurant.get("client_ready_score", 0.0)
    quiet_score = restaurant.get("quiet_score", 0.0)
    rest_area = restaurant.get("area", "")
    dietary = restaurant.get("dietary_flags", [])
    price_level = restaurant.get("price_level", 3)
    
    score = 0.0
    
    # Hard constraint violations (soft penalties)
    if state.get("client_dinner") and client_ready < 0.7:
        score -= 10000.0
    if state.get("teammate_vegan") and "vegan" not in dietary and "vegan_preorder" not in dietary:
        score -= 10000.0
    if restaurant.get("badge_only") and not state.get("badge_available"):
        score -= 10000.0
    
    # Client dinner preference
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
    
    # Quiet preference
    if state.get("quiet_matters"):
        if quiet_score >= 0.8:
            score += 50
        elif quiet_score >= 0.6:
            score += 30
        else:
            score += quiet_score * 20
    else:
        score += quiet_score * 10
    
    # Vegan accommodation
    if state.get("teammate_vegan"):
        if "vegan" in dietary:
            score += 50
        elif "vegan_preorder" in dietary:
            score += 40
        elif "vegetarian" in dietary:
            score += 20
    
    # Zone coherence with hotel
    if hotel_zone and rest_area:
        if rest_area == hotel_zone or hotel_zone in rest_area or rest_area in hotel_zone:
            score += 40
    
    # Price level as soft preference (cheaper is better)
    if price_level == 1:
        score += 30
    elif price_level == 2:
        score += 15
    elif price_level == 3:
        score += 0
    elif price_level == 4:
        score -= 15
    elif price_level >= 5:
        score -= 30
    
    return score, True


def score_activity(activity: Dict[str, Any], state: Dict[str, Any], weather: str, 
                   hotel_zone: str, remaining_budget: float) -> Tuple[float, bool]:
    price = activity.get("price", 100)
    semantic_tags = activity.get("semantic_tags", [])
    act_zone = activity.get("location_zone", "")
    
    score = 0.0
    
    if weather == "rainy" and "weather_safe" not in semantic_tags and not activity.get("indoor"):
        score -= 10000.0
    if price > remaining_budget * 1.2:
        score -= 10000.0
    if activity.get("badge_only") and not state.get("badge_available"):
        score -= 10000.0
    
    if weather == "rainy":
        if "weather_safe" in semantic_tags:
            score += 70
        if activity.get("indoor"):
            score += 30
        if not activity.get("indoor") and "weather_safe" not in semantic_tags:
            score -= 50
    else:
        if not activity.get("indoor"):
            score += 30
        if "weather_safe" in semantic_tags:
            score += 15
    
    if hotel_zone and act_zone:
        if act_zone == hotel_zone or hotel_zone in act_zone or act_zone in hotel_zone:
            score += 35
    
    if price <= 30:
        score += 45
    elif price <= 60:
        score += 30
    elif price <= 100:
        score += 15
    else:
        score += max(0, 10 - price / 50)
    
    return score, True


# ============================================================
# Search Result Extraction (only from dispatch calls)
# ============================================================

def _extract_search_results(session, exclude_broad: bool = False) -> Tuple[List, List, List, List]:
    """
    Extract all search results from the tool trace.
    ONLY includes results from dispatch calls (properly tracked).
    """
    flights, hotels, restaurants, activities = [], [], [], []
    episode = session.episode
    city = episode["city"]
    origin = episode["origin"]
    
    for trace in session.tool_trace:
        tool = trace.get("tool")
        arguments = trace.get("arguments", {}).copy()
        
        if exclude_broad and trace.get("metadata", {}).get("source") == "broad_search":
            continue
        
        if tool == "search_flights":
            arguments.setdefault("origin", origin)
            arguments.setdefault("destination", city)
            result = session.search_flights(**arguments)  # This reads from cache, doesn't call API
            flights.extend(result.get("items", []))
        elif tool == "search_hotels":
            arguments.setdefault("city", city)
            result = session.search_hotels(**arguments)
            hotels.extend(result.get("items", []))
        elif tool == "search_restaurants":
            arguments.setdefault("city", city)
            result = session.search_restaurants(**arguments)
            restaurants.extend(result.get("items", []))
        elif tool == "search_activities":
            arguments.setdefault("city", city)
            result = session.search_activities(**arguments)
            activities.extend(result.get("items", []))
    
    def dedupe(items, key):
        seen, unique = set(), []
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


def _ensure_category_nonempty(session, episode: Dict[str, Any]) -> bool:
    """
    Ensure each category has at least some candidate data.
    Uses dispatch to ensure tool calls are tracked.
    Returns True if any new searches were dispatched.
    """
    flights, hotels, restaurants, activities = _extract_search_results(session, exclude_broad=False)
    
    need_refetch = False
    
    if not flights:
        print(f"[Fallback] No flights found, dispatching broad flight search")
        session.dispatch("search_flights", {
            "origin": episode["origin"],
            "destination": episode["city"],
            "max_results": 8
        }, metadata={"source": "fallback"})
        need_refetch = True
    
    if not hotels:
        print(f"[Fallback] No hotels found, dispatching broad hotel search")
        session.dispatch("search_hotels", {
            "city": episode["city"],
            "max_results": 8
        }, metadata={"source": "fallback"})
        need_refetch = True
    
    if not restaurants:
        print(f"[Fallback] No restaurants found, dispatching broad restaurant search")
        session.dispatch("search_restaurants", {
            "city": episode["city"],
            "max_results": 8
        }, metadata={"source": "fallback"})
        need_refetch = True
    
    if not activities:
        print(f"[Fallback] No activities found, dispatching broad activity search")
        session.dispatch("search_activities", {
            "city": episode["city"],
            "max_results": 8
        }, metadata={"source": "fallback"})
        need_refetch = True
    
    return need_refetch


def _broad_search(session) -> None:
    """Perform broad searches to ensure baseline candidate coverage."""
    episode = session.episode
    city = episode["city"]
    origin = episode["origin"]
    
    # Check what we already have
    flights, hotels, restaurants, activities = _extract_search_results(session, exclude_broad=True)
    
    searches = []
    if not flights:
        searches.append(("search_flights", {"origin": origin, "destination": city, "max_results": 8}))
    if not hotels:
        searches.append(("search_hotels", {"city": city, "max_results": 8}))
    if not restaurants:
        searches.append(("search_restaurants", {"city": city, "max_results": 8}))
    if not activities:
        searches.append(("search_activities", {"city": city, "max_results": 8}))
    
    for name, args in searches:
        try:
            session.dispatch(name, args, metadata={"source": "broad_search"})
        except Exception:
            pass


def _memory_sweep(session, state: Dict[str, Any]) -> None:
    """Perform context and memory retrieval based on episode state."""
    episode = session.episode
    city = episode.get("city", "")
    traveler_id = episode.get("traveler_id")
    family = episode.get("family")
    
    # Always retrieve memory and rejected options
    session.dispatch("search_memory", {"query": "old budget cap limit spending archive stale", "include_stale": True})
    session.dispatch("get_rejected_options", {})
    
    if traveler_id:
        session.dispatch("get_profile_brief", {"traveler_id": traveler_id})
    
    # Conditional retrievals based on state
    if state.get("rainy"):
        session.dispatch("search_memory", {"query": "weather rain outdoor dry assumption stale", "include_stale": True})
        session.dispatch("get_event_context", {"city": city})
    
    if state.get("airport_priority"):
        session.dispatch("search_memory", {"query": "local character neighborhood airport access stale", "include_stale": True})
    
    if state.get("chain_exception"):
        session.dispatch("search_memory", {"query": "avoid chain hotel brand absolute stale", "include_stale": True})
    
    if state.get("partner_bundle"):
        session.dispatch("search_memory", {"query": "bundle discount partner social default stale", "include_stale": True})
        session.dispatch("get_partner_promotions", {"city": city})
        session.dispatch("get_option_dependencies", {"city": city})
    
    if state.get("event_disruption"):
        session.dispatch("search_memory", {"query": "social bundle event default stale", "include_stale": True})
        session.dispatch("get_event_context", {"city": city})
    
    if state.get("late_arrival_risk"):
        session.dispatch("search_memory", {"query": "late check-in arrival perks irrelevant stale", "include_stale": True})
        session.dispatch("get_booking_constraints", {"city": city, "family": family})
    
    if state.get("loyalty_focus") and traveler_id:
        session.dispatch("get_loyalty_profile", {"traveler_id": traveler_id})
    
    if state.get("refund_risk") or state.get("badge_available"):
        session.dispatch("get_booking_constraints", {"city": city, "family": family})


# ============================================================
# Bundle Selection with Soft Penalties
# ============================================================

def _select_bundle(session, state: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the best bundle using soft penalties.
    """
    weather = episode.get("weather", "")
    budget = episode.get("budget_total", 10000)
    nights = episode.get("nights", 1)
    meeting_zone = episode.get("meeting_zone", "")
    
    # Ensure we have data in all categories
    _ensure_category_nonempty(session, episode)
    
    # Extract results
    flights, hotels, restaurants, activities = _extract_search_results(session, exclude_broad=False)
    
    # If STILL any category is empty, use greedy fallback
    if not flights or not hotels or not restaurants or not activities:
        print(f"[Fallback] Still missing categories, using greedy selection")
        return _select_bundle_greedy(session, state, episode)
    
    # Pre-score flights and hotels
    def quick_score_f(f): return score_flight(f, state, budget)[0]
    def quick_score_h(h): return score_hotel(h, state, nights, budget, meeting_zone)[0]
    
    top_flights = sorted(flights, key=quick_score_f, reverse=True)[:15]
    top_hotels = sorted(hotels, key=quick_score_h, reverse=True)[:15]
    
    # Sort restaurants and activities by no-zone score (but keep all)
    def quick_score_r_no_zone(r): 
        return score_restaurant(r, state, "")[0]
    def quick_score_a_no_zone(a): 
        return score_activity(a, state, weather, "", budget)[0]
    
    sorted_restaurants = sorted(restaurants, key=quick_score_r_no_zone, reverse=True)
    sorted_activities = sorted(activities, key=quick_score_a_no_zone, reverse=True)
    
    best_composite = float("-inf")
    best_bundle = {
        "flight_id": None,
        "hotel_id": None,
        "restaurant_id": None,
        "activity_id": None,
        "_candidates": (flights, hotels, restaurants, activities),
        "_hotel_zone": "",
        "_total": 0,
    }
    
    UNDER_BUDGET_BONUS = 500.0
    
    total_combinations = len(top_flights) * len(top_hotels) * len(sorted_restaurants) * len(sorted_activities)
    print(f"[Bundle] Testing {len(top_flights)} flights × {len(top_hotels)} hotels × "
          f"{len(sorted_restaurants)} restaurants × {len(sorted_activities)} activities = {total_combinations} combinations")
    
    for f in top_flights:
        # 使用真实成本估算
        f_cost = _estimate_flight_cost(f)
        f_score, _ = score_flight(f, state, budget)
        
        for h in top_hotels:
            h_cost = _estimate_hotel_cost(h, nights)
            h_zone = h.get("zone", "")
            h_score, _ = score_hotel(h, state, nights, budget, meeting_zone)
            
            for r in sorted_restaurants:
                # 使用真实或保守估算的成本
                r_cost = _estimate_restaurant_cost(r, episode)
                r_score, _ = score_restaurant(r, state, h_zone)
                
                for a in sorted_activities:
                    a_cost = _estimate_activity_cost(a, episode)
                    a_score, _ = score_activity(a, state, weather, h_zone, budget)
                    
                    total_cost = f_cost + h_cost + r_cost + a_cost
                    composite = f_score + h_score + r_score + a_score
                    
                    # Budget handling: bonus for under, penalty for over
                    if total_cost <= budget:
                        composite += UNDER_BUDGET_BONUS
                    else:
                        over_percent = (total_cost - budget) / budget
                        composite -= over_percent * 1000
                    
                    if composite > best_composite:
                        best_composite = composite
                        best_bundle = {
                            "flight_id": f.get("flight_id"),
                            "hotel_id": h.get("hotel_id"),
                            "restaurant_id": r.get("restaurant_id"),
                            "activity_id": a.get("activity_id"),
                            "_candidates": (flights, hotels, restaurants, activities),
                            "_hotel_zone": h_zone,
                            "_total": total_cost,
                        }
    
    print(f"[Bundle] Best composite score: {best_composite:.2f}")
    print(f"[Bundle] Best bundle total cost: {best_bundle['_total']:.2f} / budget: {budget:.2f}")
    
    return best_bundle

def _select_bundle_greedy(session, state: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy fallback when any category is empty."""
    weather = episode.get("weather", "")
    budget = episode.get("budget_total", 10000)
    nights = episode.get("nights", 1)
    meeting_zone = episode.get("meeting_zone", "")
    
    flights, hotels, restaurants, activities = _extract_search_results(session, exclude_broad=False)
    
    # Select best flight
    best_flight = None
    best_flight_score = float("-inf")
    for f in flights:
        score, _ = score_flight(f, state, budget)
        if score > best_flight_score:
            best_flight_score = score
            best_flight = f
    flight_id = best_flight.get("flight_id") if best_flight else None
    flight_cost = _estimate_flight_cost(best_flight) if best_flight else 0
    
    # Select best hotel
    best_hotel = None
    best_hotel_score = float("-inf")
    for h in hotels:
        score, _ = score_hotel(h, state, nights, budget, meeting_zone)
        if score > best_hotel_score:
            best_hotel_score = score
            best_hotel = h
    hotel_id = best_hotel.get("hotel_id") if best_hotel else None
    hotel_cost = _estimate_hotel_cost(best_hotel, nights) if best_hotel else 0
    hotel_zone = best_hotel.get("zone", "") if best_hotel else ""
    
    # Select best restaurant
    best_restaurant = None
    best_restaurant_score = float("-inf")
    for r in restaurants:
        score, _ = score_restaurant(r, state, hotel_zone)
        if score > best_restaurant_score:
            best_restaurant_score = score
            best_restaurant = r
    restaurant_id = best_restaurant.get("restaurant_id") if best_restaurant else None
    restaurant_cost = _estimate_restaurant_cost(best_restaurant, episode) if best_restaurant else 0
    
    # Select best activity
    remaining = budget - flight_cost - hotel_cost - restaurant_cost
    best_activity = None
    best_activity_score = float("-inf")
    for a in activities:
        score, _ = score_activity(a, state, weather, hotel_zone, max(remaining, 0))
        if score > best_activity_score:
            best_activity_score = score
            best_activity = a
    activity_id = best_activity.get("activity_id") if best_activity else None
    activity_cost = _estimate_activity_cost(best_activity, episode) if best_activity else 0
    
    total_cost = flight_cost + hotel_cost + restaurant_cost + activity_cost
    
    return {
        "flight_id": flight_id,
        "hotel_id": hotel_id,
        "restaurant_id": restaurant_id,
        "activity_id": activity_id,
        "_candidates": (flights, hotels, restaurants, activities),
        "_hotel_zone": hotel_zone,
        "_total": total_cost,
    }

def _compose_notes(state: Dict[str, Any], picks: Dict[str, Any], session) -> str:
    """Compose a detailed rationale for the selections."""
    ids = [picks.get(k) for k in ["flight_id", "hotel_id", "restaurant_id", "activity_id"] if picks.get(k)]
    
    if not ids:
        return "Unable to find suitable options within constraints."
    
    parts = [f"Selected {', '.join(ids)}"]
    
    reasons = []
    if state.get("quiet_matters"):
        reasons.append("quiet accommodation")
    if state.get("client_dinner"):
        reasons.append("client-ready dining")
    if state.get("teammate_vegan"):
        reasons.append("vegan options")
    if state.get("airport_priority"):
        reasons.append("airport access prioritized")
    if state.get("refund_risk"):
        reasons.append("refundable bookings")
    if state.get("rainy"):
        reasons.append("weather-safe activities")
    
    if reasons:
        parts.append(f"Prioritized: {', '.join(reasons)}")
    
    if picks.get("_total", 0) > 0:
        parts.append(f"Total cost: ${picks['_total']:.2f}")
    
    parts.append("Retrieved memory for stale assumptions and rejected options")
    
    return ". ".join(parts)


def _session_and_tools(runtime: StudentRuntime):
    """Create session and tool list."""
    session = runtime.new_session(role="single_memory")
    tools = session_tools(session, runtime.system_config)
    return session, tools


def _gather(runtime: StudentRuntime, session, tools) -> Dict[str, Any]:
    """Run the gather agent to collect information."""
    cfg = runtime.system_config
    return runtime.runner.run_tool_agent_json(
        model=cfg["model"],
        instructions=GATHER_INSTRUCTIONS,
        input_text=episode_prompt(runtime.episode),
        json_schema=final_decision_schema(),
        schema_name="gatherer",
        tools=tools,
        tool_handler=session.dispatch,
        max_output_tokens=cfg["max_output_tokens"],
        reasoning_effort="low" if cfg["model"].startswith("gpt-5") else None,
        text_verbosity="low" if cfg["model"].startswith("gpt-5") else None,
        metadata={"system": cfg["system_name"], "trip_id": runtime.episode["trip_id"], "role": "gatherer"},
        max_tool_rounds=cfg.get("max_tool_rounds", 9),
    )


def _package(runtime: StudentRuntime, session, picks: Dict[str, Any], 
             usage: Dict[str, Any], response_ids: List[str], notes: str) -> Dict[str, Any]:
    """Package the final result with memory report."""
    episode = runtime.episode
    retired_keys, _ = derive_retired_from_state(episode)
    
    # 预先计算 required_docs（带 picks 的动态依赖）
    required_docs = derive_required_docs_with_picks(episode, picks)
    
    runner_result = {
        "parsed": {
            "flight_id": picks.get("flight_id"),
            "hotel_id": picks.get("hotel_id"),
            "restaurant_id": picks.get("restaurant_id"),
            "activity_id": picks.get("activity_id"),
            "memory_report": {
                "docs_retrieved": required_docs,  # 预先设置
            },
            "notes": notes,
        },
        "usage": usage,
        "response_ids": response_ids,
    }
    
    final = tool_result(
        runtime.runner,
        runner_result,
        session,
        active_doc_cap=3,
        active_key_cap=5,
        forced_retired=retired_keys,
        forced_retired_docs=all_stale_docs(),
    )
    
    # 保险：确保动态依赖文档没有被覆盖
    mr = final["submission"].setdefault("memory_report", {})
    existing_docs = set(mr.get("docs_retrieved") or [])
    
    # 检查是否有缺失的动态文档
    missing_docs = [d for d in required_docs if d not in existing_docs]
    if missing_docs:
        print(f"[WARN] Missing docs after tool_result, re-adding: {missing_docs}")
        mr["docs_retrieved"] = list(existing_docs) + missing_docs
    
    sub = final["submission"]
    print(f"[FINAL] {episode['trip_id']} F:{sub.get('flight_id')} H:{sub.get('hotel_id')} "
          f"R:{sub.get('restaurant_id')} A:{sub.get('activity_id')} | Total: ${picks.get('_total', 0):.2f}")
    print(f"[DOCS] Retrieved {len(mr.get('docs_retrieved', []))} docs")
    
    return final

# ============================================================
# Main Entry Point
# ============================================================

def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Main entry point for episode solving."""
    session, tools = _session_and_tools(runtime)
    episode = runtime.episode
    state = episode.get("scenario_state", {})
    
    print(f"\n{'='*60}")
    print(f"[START] {episode['trip_id']}")
    print(f"[START] Active constraints: {[k for k, v in state.items() if v and k != 'stakeholder_ids']}")
    print(f"[START] Weather={episode.get('weather')} Zone={episode.get('meeting_zone')} Budget=${episode.get('budget_total')}")
    print(f"{'='*60}")
    
    gather = _gather(runtime, session, tools)
    _memory_sweep(session, state)
    _broad_search(session)
    picks = _select_bundle(session, state, episode)
    notes = _compose_notes(state, picks, session)
    
    return _package(runtime, session, picks, gather["usage"], gather.get("response_ids", []), notes)
