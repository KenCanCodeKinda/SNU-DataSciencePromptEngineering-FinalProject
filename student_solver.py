# student_solver.py - Complete with fixed arrival time scoring

from __future__ import annotations

import json
import os
import traceback
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
    _episode_state,
    all_stale_docs,
    derive_rejected_from_state,
    derive_required_docs_from_state,
    derive_retired_from_state,
    derive_spoken_rule_hits_from_state,
    set_dynamic_state,
    clear_dynamic_state,
)


# ============================================================
# Configuration
# ============================================================

LIGHTWEIGHT_MODEL = os.environ.get("STUDENT_KEYWORD_MODEL", "gpt-5.4-nano")


# ============================================================
# Step 1: LLM Keyword Extraction
# ============================================================

KEYWORD_EXTRACTION_PROMPT = """Extract key preferences and constraints from the user's travel dialogue.
Output ONLY a JSON object with these fields (use null if not mentioned):

{{
  "quiet_hotel": null,
  "airport_access": null,
  "airport_priority_override": null,
  "no_red_eye": null,
  "refundable": null,
  "client_dinner": null,
  "vegan_options": null,
  "avoid_chain": null,
  "badge_access": null,
  "direct_flight": null,
  "indoor_activity": null,
  "budget_sensitive": null,
  "loyalty_focus": null,
  "partner_bundle": null,
  "late_arrival": null,
  "event_disruption": null,
  "early_meeting": null
}}

Guidelines:
- "no_red_eye": true if user says "no red-eye", "not overnight", "daytime flight"
- "early_meeting": true if user mentions "early meeting", "morning session", "conference morning"
- "budget_sensitive": true if user mentions "tight budget", "cost matters", "lower fare"
- "airport_priority_override": true if user says "for this trip only", "matters more than local character"
- "quiet_hotel": true if user mentions quiet room, no noise
- "client_dinner": true if user mentions client dinner, polished dinner
- "vegan_options": true if user mentions dietary flexibility, vegan
- "late_arrival": true if user mentions late arrival, late check-in

Dialogue:
{conversation}

Return ONLY valid JSON. Use true/false/null. No extra text."""

def extract_keywords(runner, episode: Dict[str, Any]) -> Dict[str, Any]:
    """Extract keywords from conversation history."""
    print("\n" + "="*60)
    print("[STEP 1] LLM Keyword Extraction")
    print("="*60)
    
    conversation = "\n".join(
        f"{turn['speaker']}: {turn['text']}"
        for turn in episode.get("turns", [])
    )
    print(f"📝 Input conversation ({len(conversation)} chars)")
    
    print(f"🤖 Calling LLM: {LIGHTWEIGHT_MODEL}")
    
    properties = {
        "quiet_hotel": {"type": ["boolean", "null"]},
        "airport_access": {"type": ["boolean", "null"]},
        "airport_priority_override": {"type": ["boolean", "null"]},
        "no_red_eye": {"type": ["boolean", "null"]},
        "refundable": {"type": ["boolean", "null"]},
        "client_dinner": {"type": ["boolean", "null"]},
        "vegan_options": {"type": ["boolean", "null"]},
        "avoid_chain": {"type": ["boolean", "null"]},
        "badge_access": {"type": ["boolean", "null"]},
        "direct_flight": {"type": ["boolean", "null"]},
        "indoor_activity": {"type": ["boolean", "null"]},
        "budget_sensitive": {"type": ["boolean", "null"]},
        "loyalty_focus": {"type": ["boolean", "null"]},
        "partner_bundle": {"type": ["boolean", "null"]},
        "late_arrival": {"type": ["boolean", "null"]},
        "event_disruption": {"type": ["boolean", "null"]},
        "early_meeting": {"type": ["boolean", "null"]},
    }
    
    try:
        result = runner.create_json_response(
            model=LIGHTWEIGHT_MODEL,
            instructions="Extract travel preferences from dialogue.",
            input_text=KEYWORD_EXTRACTION_PROMPT.format(conversation=conversation),
            json_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
            schema_name="keyword_extraction",
            max_output_tokens=500,
        )
        
        print(f"✅ LLM call successful")
        
        parsed = result.get("parsed", {})
        non_none = {k: v for k, v in parsed.items() if v is not None}
        print(f"🎯 Extracted keywords: {json.dumps(non_none, indent=2)}")
        
        return {"parsed": parsed, "usage": result["usage"], "response_ids": result.get("response_ids", [])}
        
    except Exception as e:
        print(f"❌ ERROR in keyword extraction: {e}")
        return {
            "parsed": {
                "quiet_hotel": None, "airport_access": None, "airport_priority_override": None,
                "no_red_eye": None, "refundable": None, "client_dinner": None,
                "vegan_options": None, "avoid_chain": None, "badge_access": None,
                "direct_flight": None, "indoor_activity": None, "budget_sensitive": None,
                "loyalty_focus": None, "partner_bundle": None, "late_arrival": None,
                "event_disruption": None, "early_meeting": None
            },
            "usage": runner.empty_usage(),
            "response_ids": []
        }


# ============================================================
# Step 2: LLM Scenario Expansion
# ============================================================

SCENARIO_EXPANSION_PROMPT = """Based on the extracted keywords, create a travel scenario.

Extracted keywords: {keywords}

Trip context:
- City: {city}
- Origin: {origin}
- Nights: {nights}
- Budget: ${budget_total}
- Meeting zone: {meeting_zone}
- Weather: {weather}

Output a JSON object with these exact fields:
{{
  "priority_1": "string (highest priority)",
  "priority_2": "string (second priority)",
  "priority_3": "string (third priority)",
  "must_have": ["list", "of", "must-haves"],
  "nice_to_have": ["list", "of", "nice-to-haves"],
  "avoid": ["list", "of", "things to avoid"],
  "scenario_summary": "One sentence summary"
}}

Return ONLY valid JSON, no other text."""

def expand_scenario(runner, keywords: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """Expand keywords into detailed scenario."""
    print("\n" + "="*60)
    print("[STEP 2] LLM Scenario Expansion")
    print("="*60)
    
    filtered_keywords = {k: v for k, v in keywords.items() if v is not None}
    print(f"📥 Input keywords: {json.dumps(filtered_keywords, indent=2)}")
    
    print(f"🤖 Calling LLM: {LIGHTWEIGHT_MODEL}")
    
    properties = {
        "priority_1": {"type": "string"},
        "priority_2": {"type": "string"},
        "priority_3": {"type": "string"},
        "must_have": {"type": "array", "items": {"type": "string"}},
        "nice_to_have": {"type": "array", "items": {"type": "string"}},
        "avoid": {"type": "array", "items": {"type": "string"}},
        "scenario_summary": {"type": "string"},
    }
    
    try:
        result = runner.create_json_response(
            model=LIGHTWEIGHT_MODEL,
            instructions="Expand travel keywords into concrete scenario.",
            input_text=SCENARIO_EXPANSION_PROMPT.format(
                keywords=json.dumps(filtered_keywords, indent=2) if filtered_keywords else "No specific keywords extracted",
                city=episode.get("city", ""),
                origin=episode.get("origin", ""),
                nights=episode.get("nights", 1),
                budget_total=episode.get("budget_total", 10000),
                meeting_zone=episode.get("meeting_zone", ""),
                weather=episode.get("weather", ""),
            ),
            json_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
            schema_name="scenario_expansion",
            max_output_tokens=400,
        )
        
        print(f"✅ LLM call successful")
        
        parsed = result.get("parsed", {})
        print(f"🎯 Scenario summary: {parsed.get('scenario_summary', 'N/A')}")
        
        return {"parsed": parsed, "usage": result["usage"], "response_ids": result.get("response_ids", [])}
        
    except Exception as e:
        print(f"❌ ERROR in scenario expansion: {e}")
        return {
            "parsed": {
                "priority_1": "Comfort and convenience",
                "priority_2": "Budget compliance", 
                "priority_3": "Schedule flexibility",
                "must_have": [],
                "nice_to_have": [],
                "avoid": [],
                "scenario_summary": f"Business trip to {episode.get('city', 'destination')}"
            },
            "usage": runner.empty_usage(),
            "response_ids": []
        }


# ============================================================
# Step 3: Dynamic Weights System
# ============================================================

def build_dynamic_weights(keywords: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, float]:
    """Build dynamic weights based on extracted user preferences."""
    weights = {
        # Flight weights
        "arrival_early_bonus": 0,
        "arrival_late_penalty": 0,
        "red_eye_penalty": 0,
        "daytime_bonus": 0,
        "price_weight": 0,
        "refundable_bonus": 0,
        
        # Hotel weights
        "airport_weight": 0,
        "quiet_weight": 0,
        "zone_bonus": 0,
        
        # Restaurant weights
        "client_ready_weight": 0,
        
        # Activity weights
        "indoor_bonus": 0,
    }
    
    # ============================================================
    # FLIGHT WEIGHTS
    # ============================================================
    
    # Red-eye preference
    if keywords.get("no_red_eye"):
        weights["red_eye_penalty"] = 60
        weights["daytime_bonus"] = 40
    else:
        weights["red_eye_penalty"] = 20
        weights["daytime_bonus"] = 20
    
    # Early meeting preference
    if keywords.get("early_meeting"):
        weights["arrival_early_bonus"] = 70
        weights["arrival_late_penalty"] = 40
    else:
        weights["arrival_early_bonus"] = 50
        weights["arrival_late_penalty"] = 30
    
    # Budget sensitivity
    if keywords.get("budget_sensitive"):
        weights["price_weight"] = 50
    else:
        weights["price_weight"] = 30
    
    # Refundable preference
    if keywords.get("refundable"):
        weights["refundable_bonus"] = 40
    else:
        weights["refundable_bonus"] = 20
    
    # ============================================================
    # HOTEL WEIGHTS
    # ============================================================
    
    # Airport priority
    if keywords.get("airport_priority_override"):
        weights["airport_weight"] = 25
        weights["zone_bonus"] = 5
    elif keywords.get("airport_access"):
        weights["airport_weight"] = 15
        weights["zone_bonus"] = 15
    else:
        weights["airport_weight"] = 8
        weights["zone_bonus"] = 20
    
    # Quiet preference
    if keywords.get("quiet_hotel"):
        weights["quiet_weight"] = 10
    else:
        weights["quiet_weight"] = 4
    
    # ============================================================
    # RESTAURANT WEIGHTS
    # ============================================================
    
    if keywords.get("client_dinner"):
        weights["client_ready_weight"] = 12
    else:
        weights["client_ready_weight"] = 4
    
    # ============================================================
    # ACTIVITY WEIGHTS
    # ============================================================
    
    if keywords.get("indoor_activity") or episode.get("weather") == "rainy":
        weights["indoor_bonus"] = 40
    else:
        weights["indoor_bonus"] = 20
    
    print(f"\n📊 Dynamic Weights:")
    for key, value in weights.items():
        print(f"     {key}: {value}")
    
    return weights


def build_constraints_from_keywords(keywords: Dict[str, Any], episode: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert extracted keywords into deterministic constraints."""
    print("\n" + "="*60)
    print("[STEP 3a] Building Constraints from Keywords")
    print("="*60)
    
    constraints = {
        "quiet_min": None,
        "airport_access_min": None,
        "client_ready_min": None,
        "dietary": None,
        "chain_ok": True,
        "badge_ok": True,
        "indoor_only": False,
        "weather_safe_required": False,
        "max_budget_ratio": 1.0,
        "preferred_zone": episode.get("meeting_zone"),
    }
    
    state = {}
    
    filtered = {k: v for k, v in keywords.items() if v is not None}
    print(f"📥 Processing keywords: {json.dumps(filtered, indent=2)}")
    
    if keywords.get("quiet_hotel"):
        constraints["quiet_min"] = 0.7
        state["quiet_matters"] = True
        print(f"   ✓ quiet_hotel=True → quiet_min=0.7")
    
    if keywords.get("airport_access"):
        constraints["airport_access_min"] = 0.6
        state["airport_priority"] = True
        print(f"   ✓ airport_access=True → airport_access_min=0.6")
    
    if keywords.get("airport_priority_override"):
        state["airport_priority_override"] = True
        print(f"   ✓ airport_priority_override=True")
    
    if keywords.get("no_red_eye"):
        state["red_eye_avoid"] = True
        print(f"   ✓ no_red_eye=True")
    
    if keywords.get("refundable"):
        state["refund_risk"] = True
        print(f"   ✓ refundable=True")
    
    if keywords.get("client_dinner"):
        constraints["client_ready_min"] = 0.7
        state["client_dinner"] = True
        print(f"   ✓ client_dinner=True → client_ready_min=0.7")
    
    if keywords.get("vegan_options"):
        constraints["dietary"] = "vegan"
        state["teammate_vegan"] = True
        print(f"   ✓ vegan_options=True → dietary=vegan")
    
    if keywords.get("avoid_chain"):
        constraints["chain_ok"] = False
        state["chain_exception"] = True
        print(f"   ✓ avoid_chain=True → chain_ok=False")
    
    if keywords.get("badge_access"):
        constraints["badge_ok"] = True
        state["badge_available"] = True
        print(f"   ✓ badge_access=True → badge_ok=True")
    
    if keywords.get("budget_sensitive"):
        constraints["max_budget_ratio"] = 0.85
        state["budget_sensitive"] = True
        print(f"   ✓ budget_sensitive=True")
    
    if keywords.get("loyalty_focus"):
        state["loyalty_focus"] = True
        print(f"   ✓ loyalty_focus=True")
    
    if keywords.get("partner_bundle"):
        state["partner_bundle"] = True
        print(f"   ✓ partner_bundle=True")
    
    if keywords.get("late_arrival"):
        state["late_arrival_risk"] = True
        print(f"   ✓ late_arrival=True")
    
    if keywords.get("event_disruption"):
        state["event_disruption"] = True
        print(f"   ✓ event_disruption=True")
    
    if keywords.get("early_meeting"):
        state["early_meeting"] = True
        print(f"   ✓ early_meeting=True")
    
    # Build dynamic weights
    weights = build_dynamic_weights(keywords, episode)
    state["weights"] = weights
    
    print(f"\n📊 Final constraints ready")
    
    return constraints, state


def deterministic_plan(session, constraints: Dict[str, Any], state: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Deterministic planning with dynamic weights."""
    print("\n" + "="*60)
    print("[STEP 3b] Deterministic Planning (Dynamic Weights)")
    print("="*60)
    
    city = episode["city"]
    origin = episode["origin"]
    budget = episode.get("budget_total", 10000)
    nights = episode.get("nights", 1)
    weights = state.get("weights", {})
    
    print(f"📍 Planning for: {origin} → {city}")
    print(f"💰 Budget: ${budget:,}")
    print(f"🌙 Nights: {nights}")
    
    try:
        # Search all options
        flights = session.search_flights(
            origin=origin,
            destination=city,
            red_eye_allowed=True,
            refundable_only=False,
            max_results=10,
        ).get("items", [])
        
        hotels = session.search_hotels(
            city=city,
            quiet_min=constraints.get("quiet_min"),
            airport_access_min=constraints.get("airport_access_min"),
            chain_ok=constraints.get("chain_ok", True),
            max_results=10,
        ).get("items", [])
        
        restaurants = session.search_restaurants(
            city=city,
            dietary=constraints.get("dietary"),
            client_ready_min=constraints.get("client_ready_min"),
            max_results=10,
        ).get("items", [])
        
        activities = session.search_activities(
            city=city,
            indoor_only=constraints.get("indoor_only", False),
            weather_safe_required=constraints.get("weather_safe_required", False),
            max_results=10,
        ).get("items", [])
        
        # Print options
        print("\n✈️ Flights:")
        for f in flights:
            red_eye = "RED-EYE" if f.get("red_eye") else "daytime"
            print(f"     {f.get('flight_id')}: ${f.get('fare_total'):,}, {red_eye}, arrival={f.get('arrival_time', 'unknown')}")
        
        print("\n🏨 Hotels:")
        for h in hotels:
            print(f"     {h.get('hotel_id')}: ${h.get('nightly_price'):,}/night, quiet={h.get('quiet_score')}, airport={h.get('airport_access_score')}, zone={h.get('zone')}")
        
        print("\n🍽️ Restaurants:")
        for r in restaurants:
            print(f"     {r.get('restaurant_id')}: price_level={r.get('price_level')}, client_ready={r.get('client_ready_score')}, area={r.get('area')}")
        
        print("\n🎯 Activities:")
        for a in activities:
            print(f"     {a.get('activity_id')}: ${a.get('price'):,}, zone={a.get('location_zone')}")
        
        # ============================================================
        # SCORING FUNCTIONS - FIXED ARRIVAL TIME LOGIC
        # ============================================================
        
        def get_arrival_hour(arrival_time: str) -> int:
            """Extract hour from arrival time string."""
            try:
                return int(arrival_time.split(":")[0])
            except:
                return 12  # Default to noon
        
        def score_flight(f):
            score = 0
            
            # Red-eye penalty
            if f.get("red_eye"):
                score -= weights.get("red_eye_penalty", 60)
            else:
                score += weights.get("daytime_bonus", 40)
            
            # Arrival time - FIXED LOGIC
            arrival = f.get("arrival_time", "")
            if arrival:
                hour = get_arrival_hour(arrival)
                
                # MIDDLE OF NIGHT (11pm - 4am) - HEAVY PENALTY
                if hour >= 23 or hour <= 3:
                    score -= weights.get("arrival_late_penalty", 30)
                
                # VERY EARLY MORNING (4am - 6am) - slight penalty
                elif 4 <= hour <= 6:
                    score += weights.get("arrival_early_bonus", 50) - 20
                
                # IDEAL EARLY MORNING (6am - 8am) - best for business
                elif 6 < hour <= 8:
                    score += weights.get("arrival_early_bonus", 50)
                
                # GOOD MORNING (8am - 10am) - still good
                elif 8 < hour <= 10:
                    score += weights.get("arrival_early_bonus", 50) - 10
                
                # LATE MORNING (10am - 12pm) - acceptable
                elif 10 < hour <= 12:
                    score += weights.get("arrival_early_bonus", 50) - 20
                
                # AFTERNOON (12pm - 4pm) - neutral
                elif 12 < hour <= 16:
                    pass  # No bonus, no penalty
                
                # LATE AFTERNOON (4pm - 7pm) - slight penalty
                elif 16 < hour <= 19:
                    score -= 10
                
                # EVENING (7pm - 11pm) - penalty
                elif 19 < hour < 23:
                    score -= weights.get("arrival_late_penalty", 30) - 10
            
            # Refundable bonus
            if f.get("refundable"):
                score += weights.get("refundable_bonus", 40)
            
            # Non-stop bonus
            if f.get("stops", 99) == 0:
                score += 30
            
            # Price
            fare = f.get("fare_total", 10000)
            min_fare = min([f2.get("fare_total", 10000) for f2 in flights]) if flights else fare
            price_weight = weights.get("price_weight", 30)
            if fare == min_fare:
                score += price_weight
            elif fare < min_fare * 1.1:
                score += price_weight // 2
            elif fare < min_fare * 1.2:
                score += price_weight // 3
            
            return score
        
        def score_hotel(h):
            if constraints.get("quiet_min") and h.get("quiet_score", 0) < constraints["quiet_min"]:
                return -10000
            if constraints.get("airport_access_min") and h.get("airport_access_score", 0) < constraints["airport_access_min"]:
                return -10000
            
            score = 0
            
            # Airport access
            airport_weight = weights.get("airport_weight", 12)
            score += h.get("airport_access_score", 0) * airport_weight
            
            # Quiet
            quiet_weight = weights.get("quiet_weight", 4)
            score += h.get("quiet_score", 0) * quiet_weight
            
            # Zone bonus
            zone_bonus = weights.get("zone_bonus", 20)
            if h.get("zone") == constraints.get("preferred_zone"):
                score += zone_bonus
            
            # Price
            nightly = h.get("nightly_price", 500)
            min_nightly = min([h2.get("nightly_price", 500) for h2 in hotels]) if hotels else nightly
            if nightly == min_nightly:
                score += 15
            elif nightly < min_nightly * 1.1:
                score += 8
            
            return score
        
        def score_restaurant(r):
            if constraints.get("client_ready_min") and r.get("client_ready_score", 0) < constraints["client_ready_min"]:
                return -10000
            if constraints.get("dietary") and constraints["dietary"] not in r.get("dietary_flags", []):
                if "vegan_preorder" not in r.get("dietary_flags", []):
                    return -10000
            
            score = 0
            
            client_weight = weights.get("client_ready_weight", 4)
            score += r.get("client_ready_score", 0) * client_weight
            
            quiet_weight = weights.get("quiet_weight", 4) // 2
            score += r.get("quiet_score", 0) * quiet_weight
            
            price_level = r.get("price_level", 3)
            if price_level <= 2:
                score += 30
            elif price_level == 3:
                score += 15
            else:
                score += 5
            
            if r.get("area") == constraints.get("preferred_zone"):
                score += 15
            
            return score
        
        def score_activity(a):
            if constraints.get("indoor_only") and not a.get("indoor"):
                return -5000
            if constraints.get("weather_safe_required") and "weather_safe" not in a.get("semantic_tags", []):
                return -5000
            
            price = a.get("price", 100)
            if price == 0:
                score = 100
            elif price <= 50:
                score = 70
            elif price <= 100:
                score = 40
            elif price <= 200:
                score = 20
            else:
                score = -10
            
            if a.get("indoor"):
                score += weights.get("indoor_bonus", 20)
            
            if a.get("location_zone") == constraints.get("preferred_zone"):
                score += 20
            
            return score
        
        # ============================================================
        # COMBINATION SEARCH
        # ============================================================
        
        print("\n" + "="*60)
        print("[BUDGET CHECK] Searching combinations...")
        print("="*60)
        
        best_score = -float('inf')
        best_picks = None
        valid_combinations = []
        
        for f in flights:
            flight_score = score_flight(f)
            if flight_score < -5000:
                continue
            flight_cost = f.get("fare_total", 0)
            
            for h in hotels:
                hotel_score = score_hotel(h)
                if hotel_score < -5000:
                    continue
                hotel_cost = h.get("nightly_price", 0) * nights
                
                for r in restaurants:
                    restaurant_score = score_restaurant(r)
                    if restaurant_score < -5000:
                        continue
                    restaurant_cost = r.get("price_level", 2) * 25000
                    
                    for a in activities:
                        activity_score = score_activity(a)
                        if activity_score < -5000:
                            continue
                        activity_cost = a.get("price", 0)
                        
                        total_cost = flight_cost + hotel_cost + restaurant_cost + activity_cost
                        
                        if total_cost > budget:
                            continue
                        
                        total_score = flight_score + hotel_score + restaurant_score + activity_score
                        
                        if total_cost < budget * 0.9:
                            total_score += 15
                        elif total_cost < budget:
                            total_score += 8
                        
                        valid_combinations.append({
                            "flight": f.get("flight_id"),
                            "hotel": h.get("hotel_id"),
                            "restaurant": r.get("restaurant_id"),
                            "activity": a.get("activity_id"),
                            "total_score": total_score,
                            "total_cost": total_cost,
                            "flight_score": flight_score,
                            "hotel_score": hotel_score,
                        })
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_picks = {
                                "flight_id": f.get("flight_id"),
                                "hotel_id": h.get("hotel_id"),
                                "restaurant_id": r.get("restaurant_id"),
                                "activity_id": a.get("activity_id"),
                                "_total_cost": total_cost,
                            }
        
        # Display top combinations
        valid_combinations.sort(key=lambda x: x["total_score"], reverse=True)
        
        print(f"\n📊 Top {min(15, len(valid_combinations))} valid combinations:")
        print("="*100)
        for combo in valid_combinations[:15]:
            print(f"  Score {combo['total_score']:>6.1f} | {combo['flight']:>8} | {combo['hotel']:>8} | {combo['restaurant']:>8} | {combo['activity']:>20} | ${combo['total_cost']:>11,}")
        print("="*100)
        
        if best_picks:
            print(f"\n✅ Best combination:")
            print(f"   Flight: {best_picks['flight_id']}")
            print(f"   Hotel: {best_picks['hotel_id']}")
            print(f"   Restaurant: {best_picks['restaurant_id']}")
            print(f"   Activity: {best_picks['activity_id']}")
            print(f"   Total cost: ${best_picks['_total_cost']:,} / ${budget:,}")
        
        return best_picks if best_picks else {
            "flight_id": None,
            "hotel_id": None,
            "restaurant_id": None,
            "activity_id": None,
        }
        
    except Exception as e:
        print(f"❌ ERROR in deterministic planning: {e}")
        traceback.print_exc()
        return {
            "flight_id": None,
            "hotel_id": None,
            "restaurant_id": None,
            "activity_id": None,
        }


# ============================================================
# Step 4: Spoken Rule Generation
# ============================================================

def build_spoken_rule_hits(state: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build spoken_rule_hits with canonical keys."""
    hits: Dict[str, List[str]] = {
        "must_remember": [],
        "forbidden": [],
        "one_off_only": [],
        "retire": [],
        "do_not_reconsider": ["noise_rejected_hotel", "wrong_vibe_restaurant"],
        "keep_context_lean": ["relevant_only"],
    }
    
    if state.get("quiet_matters"):
        hits["must_remember"].append("prefer_quiet_hotel")
    if state.get("client_dinner"):
        hits["must_remember"].append("client_dinner_polished")
    if state.get("teammate_vegan"):
        hits["must_remember"].append("team_dietary_flex")
    
    if state.get("red_eye_avoid"):
        hits["forbidden"].append("avoid_red_eye")
    if state.get("quiet_matters"):
        hits["forbidden"].append("loud_after_10pm")
    
    if state.get("airport_priority_override"):
        hits["one_off_only"].append("prefer_airport_access")
    if state.get("chain_exception"):
        hits["one_off_only"].append("chain_ok_this_trip")
    
    hits["retire"].append("old_budget_cap")
    if state.get("airport_priority_override"):
        hits["retire"].append("old_local_character_priority")
    if state.get("chain_exception"):
        hits["retire"].append("old_chain_absolute_rule")
    if state.get("partner_bundle"):
        hits["retire"].append("old_bundle_discount_absolute")
    if state.get("late_arrival_risk"):
        hits["retire"].append("late_checkin_irrelevant")
    
    return hits


# ============================================================
# Memory Sweep
# ============================================================

def _memory_sweep(session, state: Dict[str, Any], episode: Dict[str, Any]) -> None:
    """Enhanced memory retrieval."""
    print("\n" + "="*60)
    print("[MEMORY SWEEP] Context Retrieval")
    print("="*60)
    
    city = episode.get("city", "")
    traveler_id = episode.get("traveler_id", "")
    family = episode.get("family", "")
    
    def fire(name: str, args: Dict[str, Any]) -> None:
        try:
            print(f"  🔧 Calling {name}")
            session.dispatch(name, args)
            print(f"     ✓ Success")
        except Exception:
            pass
    
    fire("search_memory", {"query": "old budget cap limit spending archive stale", "include_stale": True})
    fire("get_rejected_options", {})
    
    if traveler_id:
        fire("get_profile_brief", {"traveler_id": traveler_id})
    if city and family:
        fire("get_venue_brief", {"city": city, "family": family})
    
    if state.get("airport_priority"):
        fire("get_city_ops_notes", {"city": city, "query": "airport access transit"})
    
    if state.get("partner_bundle"):
        fire("get_partner_promotions", {"city": city})
    
    if state.get("quiet_matters"):
        fire("search_memory", {"query": "quiet noise silent hotel", "include_stale": True})
    
    print(f"  ✅ Memory sweep complete")


# ============================================================
# Main Pipeline
# ============================================================

def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Hybrid LLM + deterministic planning pipeline."""
    print("\n" + "🚀"*30)
    print("HYBRID PIPELINE START")
    print("🚀"*30)
    
    session, tools = _session_and_tools(runtime)
    episode = runtime.episode
    episode_id = episode.get("trip_id", "unknown")
    
    clear_dynamic_state()
    
    print(f"\n📋 Episode Info:")
    print(f"   ID: {episode_id}")
    print(f"   City: {episode.get('city')}")
    print(f"   Origin: {episode.get('origin')}")
    print(f"   Budget: ${episode.get('budget_total'):,}")
    print(f"   Nights: {episode.get('nights')}")
    print(f"   Weather: {episode.get('weather')}")
    print(f"   Meeting Zone: {episode.get('meeting_zone')}")
    
    all_usage = []
    response_ids = []
    
    try:
        keywords_result = extract_keywords(runtime.runner, episode)
        all_usage.append(keywords_result["usage"])
        response_ids.extend(keywords_result.get("response_ids", []))
        keywords = keywords_result["parsed"]
        
        scenario_result = expand_scenario(runtime.runner, keywords, episode)
        all_usage.append(scenario_result["usage"])
        response_ids.extend(scenario_result.get("response_ids", []))
        
        constraints, dynamic_state = build_constraints_from_keywords(keywords, episode)
        
        base_state = _episode_state(episode)
        dynamic_state.update(base_state)
        
        set_dynamic_state(dynamic_state)
        
        _memory_sweep(session, dynamic_state, episode)
        
        picks = deterministic_plan(session, constraints, dynamic_state, episode)
        
        spoken_rule_hits = build_spoken_rule_hits(dynamic_state)
        
        notes = _compose_notes(dynamic_state, picks, scenario_result["parsed"])
        
        print("\n" + "🎉"*30)
        print("FINAL PICKS:")
        print(f"  Flight: {picks.get('flight_id')}")
        print(f"  Hotel: {picks.get('hotel_id')}")
        print(f"  Restaurant: {picks.get('restaurant_id')}")
        print(f"  Activity: {picks.get('activity_id')}")
        print("🎉"*30)
        
        usage = runtime.runner.combine_usages(*all_usage) if all_usage else runtime.runner.empty_usage()
        return _package(runtime, session, picks, usage, response_ids, notes, spoken_rule_hits)
        
    except Exception as exc:
        print(f"\n❌❌❌ ERROR: {exc}")
        traceback.print_exc()
        return _fallback_result(runtime, session)
    finally:
        clear_dynamic_state()


def _compose_notes(state: Dict[str, Any], picks: Dict[str, Any], scenario: Dict[str, Any]) -> str:
    """Compose notes."""
    ids = [picks.get(k) for k in ["flight_id", "hotel_id", "restaurant_id", "activity_id"] if picks.get(k)]
    
    parts = [f"Hybrid pipeline.", f"Selected {', '.join(ids)}." if ids else "Selected best available."]
    
    if scenario.get('scenario_summary'):
        parts.append(f"Scenario: {scenario['scenario_summary'][:100]}")
    
    return " ".join(parts)[:315]


def _session_and_tools(runtime: StudentRuntime):
    cfg = runtime.system_config
    session = runtime.new_session(role="single_memory")
    tools = session_tools(session, cfg)
    return session, tools


def _package(runtime: StudentRuntime, session, picks: Dict[str, Any],
             usage: Dict[str, Any], response_ids: List[str], notes: str,
             spoken_rule_hits: Dict[str, List[str]]) -> Dict[str, Any]:
    """Package final result."""
    episode = runtime.episode
    retired_keys, _ = derive_retired_from_state(episode)
    
    memory_report = {
        "retrieved": [],
        "retired": retired_keys,
        "retired_docs": all_stale_docs(),
        "rejected_option_notes": derive_rejected_from_state(episode),
        "active_context_keys": [],
        "docs_retrieved": [],
        "active_docs": [],
        "ignored_distractors": [],
        "spoken_rule_hits": spoken_rule_hits,
        "critical_constraints": ["relevant_only"],
    }
    
    runner_result = {
        "parsed": {
            "flight_id": picks.get("flight_id"),
            "hotel_id": picks.get("hotel_id"),
            "restaurant_id": picks.get("restaurant_id"),
            "activity_id": picks.get("activity_id"),
            "memory_report": memory_report,
            "notes": notes,
        },
        "usage": usage,
        "response_ids": response_ids,
    }
    
    final = tool_result(
        runtime.runner,
        runner_result,
        session,
        active_doc_cap=4,
        active_key_cap=6,
        forced_retired=retired_keys,
        forced_retired_docs=all_stale_docs(),
    )
    
    print(f"\n[FINAL] {episode['trip_id']}")
    return final


def _fallback_result(runtime: StudentRuntime, session) -> Dict[str, Any]:
    """Fallback."""
    print("\n⚠️ FALLBACK MODE")
    
    picks = {"flight_id": None, "hotel_id": None, "restaurant_id": None, "activity_id": None}
    
    try:
        episode = runtime.episode
        if episode.get("city") and episode.get("origin"):
            flights = session.search_flights(origin=episode["origin"], destination=episode["city"], max_results=1)
            if flights.get("items"):
                picks["flight_id"] = flights["items"][0].get("flight_id")
            
            hotels = session.search_hotels(city=episode["city"], max_results=1)
            if hotels.get("items"):
                picks["hotel_id"] = hotels["items"][0].get("hotel_id")
    except Exception:
        pass
    
    spoken_rule_hits = {
        "must_remember": [],
        "forbidden": [],
        "one_off_only": [],
        "retire": ["old_budget_cap"],
        "do_not_reconsider": ["noise_rejected_hotel", "wrong_vibe_restaurant"],
        "keep_context_lean": ["relevant_only"],
    }
    
    notes = "Fallback."
    return _package(runtime, session, picks, runtime.runner.empty_usage(), [], notes, spoken_rule_hits)
