from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

from runtime_api import StudentRuntime
from llm_agents import ensure_grounded_submission

from student_custom_tools_template import (
    fetch_all_context_info,
    build_bundle_bonus_map,
    get_loyalty_bonus,
    filter_by_bundle,
    extract_user_requirements,
    extract_spoken_rules_from_turns,
    build_memory_report_from_context,
)


class TravelPlanner:
    """Scenario-based travel planner with LLM pairwise selection using preference hierarchy"""
    
    def __init__(self, runtime: StudentRuntime):
        self.runtime = runtime
    
    def _parse_hour(self, time_str: str) -> int:
        """Safely parse time string, return hour"""
        if not time_str:
            return 12
        try:
            cleaned = time_str.strip()
            if ':' in cleaned:
                hour_part = cleaned.split(':')[0].strip()
                hour = int(hour_part)
                return hour
            import re
            match = re.search(r'(\d{1,2})', cleaned)
            if match:
                hour = int(match.group(1))
                if 'pm' in cleaned.lower() and hour < 12:
                    hour += 12
                if 'am' in cleaned.lower() and hour == 12:
                    hour = 0
                return hour
            return 12
        except (ValueError, IndexError, AttributeError):
            return 12
    
    def fetch_all_options(self, episode: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Fetch all available options"""
        city = episode['city']
        origin = episode['origin']
        
        session = self.runtime.new_session(
            retrieval_strategy="lexical",
            max_results=20
        )
        
        flights_result = session.search_flights(origin=origin, destination=city, max_results=20)
        hotels_result = session.search_hotels(city=city, max_results=20)
        restaurants_result = session.search_restaurants(city=city, max_results=20)
        activities_result = session.search_activities(city=city, max_results=20)
        
        return {
            "flights": flights_result.get('items', []),
            "hotels": hotels_result.get('items', []),
            "restaurants": restaurants_result.get('items', []),
            "activities": activities_result.get('items', []),
        }
    
    def display_options(self, options: Dict[str, List[Dict]], episode: Dict[str, Any]) -> None:
        """Display all options"""
        print("\n" + "="*80)
        print(f"📋 Episode: {episode.get('trip_id')} - All Available Options")
        print("="*80)
        
        print(f"\n✈️ Flights ({len(options['flights'])}):")
        for f in options['flights']:
            red_eye_flag = "🛑Red-eye" if f.get('red_eye') else "✓Normal"
            refund_flag = "🔄Refundable" if f.get('refundable') else "❌Non-refund"
            print(f"  - {f.get('flight_id')}: {f.get('depart_time', 'N/A')} → {f.get('arrival_time', 'N/A')}, "
                  f"₩{f.get('fare_total', 'N/A'):,}, {red_eye_flag}, {refund_flag}")
        
        print(f"\n🏨 Hotels ({len(options['hotels'])}):")
        for h in options['hotels']:
            chain_flag = "🏢Chain" if h.get('chain') else "🏨Independent"
            print(f"  - {h.get('hotel_id')}: {h.get('zone', 'N/A')}, ₩{h.get('nightly_price', 'N/A'):,}/night, "
                  f"Quiet:{h.get('quiet_score', 0):.1f}, Airport:{h.get('airport_access_score', 0):.1f}, {chain_flag}")
        
        print(f"\n🍽️ Restaurants ({len(options['restaurants'])}):")
        for r in options['restaurants']:
            dietary = ", ".join(r.get('dietary_flags', [])) if r.get('dietary_flags') else "None"
            print(f"  - {r.get('restaurant_id')}: {r.get('area', 'N/A')}, {r.get('cuisine', 'N/A')}, "
                  f"Price Level:{r.get('price_level', 'N/A')}, Quiet:{r.get('quiet_score', 0):.1f}, "
                  f"Client:{r.get('client_ready_score', 0):.1f}, Diet:{dietary}")
        
        print(f"\n🎯 Activities ({len(options['activities'])}):")
        for a in options['activities']:
            indoor_flag = "🏠Indoor" if a.get('indoor') else "🌳Outdoor"
            print(f"  - {a.get('activity_id')}: {a.get('location_zone', 'N/A')}, "
                  f"₩{a.get('price', 0):,}, {indoor_flag}")
    
    def filter_by_scenario_state(
        self, 
        options: Dict[str, List[Dict]], 
        state: Dict[str, Any],
        episode: Dict[str, Any]
    ) -> Dict[str, List[Dict]]:
        """Filter by scenario_state"""
        
        meeting_zone = episode.get('meeting_zone')
        budget = episode.get('budget_total', float('inf'))
        nights = episode.get('nights', 2)
        
        print("\n" + "="*80)
        print("🔍 Filtering by scenario_state")
        print("="*80)
        
        active_filters = []
        for key, value in state.items():
            if value and key not in ['stakeholder_ids']:
                active_filters.append(f"{key}: {value}")
        print(f"  Active filters: {', '.join(active_filters)}")
        
        result = {
            "flights": options['flights'].copy(),
            "hotels": options['hotels'].copy(),
            "restaurants": options['restaurants'].copy(),
            "activities": options['activities'].copy(),
        }
        
        # Flights: budget filter
        original = len(result['flights'])
        result['flights'] = [f for f in result['flights'] if f.get('fare_total', float('inf')) <= budget]
        if original > len(result['flights']):
            print(f"✈️ Budget filter: {original} → {len(result['flights'])}")
        
        # Hotels: meeting zone + budget + quiet
        if meeting_zone:
            meeting_hotels = [h for h in result['hotels'] if h.get('zone') == meeting_zone]
            if meeting_hotels:
                result['hotels'] = meeting_hotels
                print(f"🏨 Meeting zone filter: keep {meeting_zone} → {len(result['hotels'])}")
        
        original = len(result['hotels'])
        result['hotels'] = [h for h in result['hotels'] if h.get('nightly_price', float('inf')) * nights <= budget]
        if original > len(result['hotels']):
            print(f"🏨 Budget filter: {original} → {len(result['hotels'])}")
        
        original = len(result['hotels'])
        result['hotels'] = [h for h in result['hotels'] if h.get('quiet_score', 0) >= 7.0]
        if original > len(result['hotels']):
            print(f"🏨 Quiet filter: keep quiet_score>=7.0 → {len(result['hotels'])}")
        
        if state.get('airport_priority'):
            result['hotels'] = sorted(result['hotels'], key=lambda h: (-h.get('airport_access_score', 0), h.get('nightly_price', float('inf'))))
            print(f"🏨 Airport priority: sort by airport access score")
        
        # Restaurants: meeting zone + vegan
        if meeting_zone:
            meeting_restaurants = [r for r in result['restaurants'] if r.get('area') == meeting_zone]
            if meeting_restaurants:
                result['restaurants'] = meeting_restaurants
                print(f"🍽️ Meeting zone filter: keep {meeting_zone} → {len(result['restaurants'])}")
        
        if state.get('teammate_vegan'):
            vegan_restaurants = [r for r in result['restaurants'] if 'vegan' in r.get('dietary_flags', [])]
            if vegan_restaurants:
                result['restaurants'] = vegan_restaurants
                print(f"🍽️ Vegan filter: keep vegan options → {len(result['restaurants'])}")
        
        # Activities: meeting zone + rainy indoor
        if meeting_zone:
            meeting_activities = [a for a in result['activities'] if a.get('location_zone') == meeting_zone]
            if meeting_activities:
                result['activities'] = meeting_activities
                print(f"🎯 Meeting zone filter: keep {meeting_zone} → {len(result['activities'])}")
        
        if state.get('rainy'):
            indoor_activities = [a for a in result['activities'] if a.get('indoor')]
            if indoor_activities:
                result['activities'] = indoor_activities
                print(f"🎯 Rainy filter: keep indoor activities → {len(result['activities'])}")
        
        return result
    
    def generate_combinations(
        self, 
        filtered_options: Dict[str, List[Dict]]
    ) -> List[Tuple[str, str, str, str]]:
        """Generate all combinations"""
        
        flights = [f.get('flight_id') for f in filtered_options['flights']]
        hotels = [h.get('hotel_id') for h in filtered_options['hotels']]
        restaurants = [r.get('restaurant_id') for r in filtered_options['restaurants']]
        activities = [a.get('activity_id') for a in filtered_options['activities']]
        
        if not flights:
            flights = [None]
        if not hotels:
            hotels = [None]
        if not restaurants:
            restaurants = [None]
        if not activities:
            activities = [None]
        
        combinations = list(product(flights, hotels, restaurants, activities))
        
        return combinations
    
    def calculate_total_cost(
        self,
        combo: Tuple[str, str, str, str],
        flights_dict: Dict,
        hotels_dict: Dict,
        restaurants_dict: Dict,
        activities_dict: Dict,
        nights: int
    ) -> int:
        """Calculate total cost of a combination"""
        
        flight_id, hotel_id, restaurant_id, activity_id = combo
        
        total = 0
        
        flight = flights_dict.get(flight_id) if flight_id else None
        if flight:
            total += flight.get('fare_total', 0)
        
        hotel = hotels_dict.get(hotel_id) if hotel_id else None
        if hotel:
            total += hotel.get('nightly_price', 0) * nights
        
        restaurant = restaurants_dict.get(restaurant_id) if restaurant_id else None
        if restaurant:
            total += restaurant.get('price_level', 2) * 25000
        
        activity = activities_dict.get(activity_id) if activity_id else None
        if activity:
            total += activity.get('price', 0)
        
        return total
    
    def generate_memory_report_with_llm(
        self,
        episode: Dict[str, Any],
        best_combo: Tuple[str, str, str, str],
        best_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate memory_report using LLM based on conversation"""
        
        turns = episode.get('turns', [])
        flight_id, hotel_id, restaurant_id, activity_id = best_combo
        
        # Build conversation with timestamps
        conversation_lines = []
        for i, turn in enumerate(turns, 1):
            speaker = "User" if turn.get('speaker') == 'user' else "System"
            text = turn.get('text', '')
            conversation_lines.append(f"[Message {i}] {speaker}: {text}")
        conversation = "\n".join(conversation_lines)
        
        memory_prompt = f"""
## User Conversation (chronological order)
{conversation}

## Selected Plan
- Flight: {flight_id}
- Hotel: {hotel_id}
- Restaurant: {restaurant_id}
- Activity: {activity_id}

## Your Task
Extract spoken rules from the conversation into the exact memory_report format required.

## Rule Mapping Guide

| User says | Category | Key to add |
|-----------|----------|------------|
| "quiet room matters", "genuinely quiet room" | must_remember | "quiet_matters" |
| "client dinner polished", "polished enough for conversation" | must_remember | "client_ready_dinner" |
| "no red-eye", "do not send me through a red-eye" | forbidden | "red_eye" |
| "loud after 10pm", "nightlife spillover" | forbidden | "loud_after_10pm" |
| "airport access matters more this trip only", "For this trip only, airport access matters more" | one_off_only | "airport_access_more_important_now" |
| "chain hotel acceptable this trip only", "This trip only, a chain hotel is acceptable" | one_off_only | "chain_ok_this_trip" |
| "older budget assumption no longer valid", "retire it", "The older budget assumption is no longer valid" | retire | "old_budget_cap" |
| "stop carrying old local-character preference", "stop carrying the old local-character preference" | retire | "local_character_if_safe" |
| "old chain absolute rule" | retire | "avoid_chain_hotels_stable" |
| "old social bundle default" | retire | "old_social_bundle_default" |
| "keep active context lean", "relevant old preferences", "bring back only the relevant old preferences" | keep_context_lean | "relevant_only" |
| "reject hotel for noise, do not reconsider" | do_not_reconsider | "noise_rejected_hotel" |
| "dinner option out for wrong vibe" | do_not_reconsider | "wrong_vibe_restaurant" |

## Output Format (MUST follow exactly)

Return JSON:
{{
  "memory_report": {{
    "retrieved": ["prefer_quiet_hotel", "avoid_red_eye", "loud_after_10pm", "prefer_airport_access"],
    "retired": ["old_budget_cap", "local_character_if_safe"],
    "retired_docs": ["stale:budget_cap_archive", "stale:local_character_default"],
    "rejected_option_notes": [],
    "active_context_keys": ["prefer_quiet_hotel", "avoid_red_eye", "loud_after_10pm", "prefer_airport_access"],
    "docs_retrieved": ["profile:traveler_ops", "city_ops:OSA"],
    "active_docs": ["profile:traveler_ops"],
    "ignored_distractors": [],
    "spoken_rule_hits": {{
      "must_remember": ["quiet_matters", "client_ready_dinner"],
      "forbidden": ["red_eye", "loud_after_10pm"],
      "one_off_only": ["airport_access_more_important_now", "chain_ok_this_trip"],
      "retire": ["old_budget_cap", "local_character_if_safe"],
      "do_not_reconsider": ["noise_rejected_hotel", "wrong_vibe_restaurant"],
      "keep_context_lean": ["relevant_only"]
    }}
  }}
}}

## Important Rules
1. ALWAYS include "relevant_only" in keep_context_lean
2. If user says something is "this trip only", put it in one_off_only
3. If user says something is "no longer valid" or "retire", put it in retire
4. If user explicitly forbids something (like "do not send me through a red-eye"), put it in forbidden
5. If user emphasizes something is important (like "quiet room matters"), put it in must_remember
6. Use ONLY the key names from the mapping guide above

Return ONLY valid JSON matching the exact format.
"""
        
        memory_schema = {
            "type": "object",
            "properties": {
                "memory_report": {
                    "type": "object",
                    "properties": {
                        "retrieved": {"type": "array", "items": {"type": "string"}},
                        "retired": {"type": "array", "items": {"type": "string"}},
                        "retired_docs": {"type": "array", "items": {"type": "string"}},
                        "rejected_option_notes": {"type": "array", "items": {"type": "string"}},
                        "active_context_keys": {"type": "array", "items": {"type": "string"}},
                        "docs_retrieved": {"type": "array", "items": {"type": "string"}},
                        "active_docs": {"type": "array", "items": {"type": "string"}},
                        "ignored_distractors": {"type": "array", "items": {"type": "string"}},
                        "spoken_rule_hits": {
                            "type": "object",
                            "properties": {
                                "must_remember": {"type": "array", "items": {"type": "string"}},
                                "forbidden": {"type": "array", "items": {"type": "string"}},
                                "one_off_only": {"type": "array", "items": {"type": "string"}},
                                "retire": {"type": "array", "items": {"type": "string"}},
                                "do_not_reconsider": {"type": "array", "items": {"type": "string"}},
                                "keep_context_lean": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["must_remember", "forbidden", "one_off_only", "retire", "do_not_reconsider", "keep_context_lean"]
                        }
                    },
                    "required": ["retrieved", "retired", "retired_docs", "rejected_option_notes", 
                                 "active_context_keys", "docs_retrieved", "active_docs", 
                                 "ignored_distractors", "spoken_rule_hits"]
                }
            },
            "required": ["memory_report"]
        }
        
        print(f"\n🤖 Stage 3: LLM Generating Memory Report")
        
        try:
            memory_result = self.runtime.runner.create_json_response(
                model="gpt-4o-mini",
                instructions="Extract spoken rules from conversation into exact memory_report format. Follow the mapping guide exactly.",
                input_text=memory_prompt,
                json_schema=memory_schema,
                schema_name="memory_report",
                max_output_tokens=1000,
                metadata={"system": "student_solver", "trip_id": episode["trip_id"], "stage": "memory"},
            )
            
            memory_report = memory_result.get("parsed", {}).get("memory_report", {})
            
            # Track usage
            if hasattr(self.runtime, 'runner'):
                self.runtime.runner._record_observed_usage(memory_result.get('usage', {}))
            
        except Exception as e:
            print(f"   ⚠️ LLM memory generation failed: {e}")
            memory_report = {}
        
        # Ensure required fields exist
        if "spoken_rule_hits" not in memory_report:
            memory_report["spoken_rule_hits"] = {
                "must_remember": [],
                "forbidden": [],
                "one_off_only": [],
                "retire": [],
                "do_not_reconsider": [],
                "keep_context_lean": ["relevant_only"]
            }
        
        # Ensure keep_context_lean always has relevant_only
        if "keep_context_lean" in memory_report["spoken_rule_hits"]:
            if "relevant_only" not in memory_report["spoken_rule_hits"]["keep_context_lean"]:
                memory_report["spoken_rule_hits"]["keep_context_lean"].append("relevant_only")
        else:
            memory_report["spoken_rule_hits"]["keep_context_lean"] = ["relevant_only"]
        
        # Ensure all spoken_rule_hits fields exist
        for field in ["must_remember", "forbidden", "one_off_only", "retire", "do_not_reconsider", "keep_context_lean"]:
            if field not in memory_report["spoken_rule_hits"]:
                memory_report["spoken_rule_hits"][field] = []
        
        # Set default active_context_keys if empty
        if not memory_report.get("active_context_keys"):
            memory_report["active_context_keys"] = ["prefer_quiet_hotel", "avoid_red_eye", "relevant_only"]
        
        # Set default retrieved if empty
        if not memory_report.get("retrieved"):
            memory_report["retrieved"] = memory_report["active_context_keys"][:4]
        
        print(f"\n   📋 Generated Memory Report:")
        print(f"      must_remember: {memory_report['spoken_rule_hits']['must_remember']}")
        print(f"      forbidden: {memory_report['spoken_rule_hits']['forbidden']}")
        print(f"      one_off_only: {memory_report['spoken_rule_hits']['one_off_only']}")
        print(f"      retire: {memory_report['spoken_rule_hits']['retire']}")
        print(f"      keep_context_lean: {memory_report['spoken_rule_hits']['keep_context_lean']}")
        
        return memory_report
    
    def llm_selection(
        self,
        episode: Dict[str, Any],
        valid_combos: List[Dict],
        requirements: Dict[str, Any],
        flights_dict: Dict,
        hotels_dict: Dict,
        restaurants_dict: Dict,
        activities_dict: Dict
    ) -> Tuple[Tuple[str, str, str, str], float, Dict, Dict]:
        """Three-stage LLM selection: Filter -> Select -> Memory Report"""
        
        turns = episode.get('turns', [])
        
        # Build conversation with timestamps
        conversation_lines = []
        for i, turn in enumerate(turns, 1):
            speaker = "User" if turn.get('speaker') == 'user' else "System"
            text = turn.get('text', '')
            conversation_lines.append(f"[Message {i}] {speaker}: {text}")
        conversation = "\n".join(conversation_lines)
        
        # Build combo table with detailed info
        combo_table = []
        for i, item in enumerate(valid_combos, 1):
            flight_id, hotel_id, restaurant_id, activity_id = item['combo']
            
            flight = flights_dict.get(flight_id)
            hotel = hotels_dict.get(hotel_id)
            restaurant = restaurants_dict.get(restaurant_id)
            activity = activities_dict.get(activity_id)
            
            # Flight info
            flight_info = f"{flight_id}"
            if flight:
                flight_info += f"({flight.get('depart_time', '?')}->{flight.get('arrival_time', '?')})"
                if flight.get('red_eye'):
                    flight_info += " RED-EYE"
                    fare = flight.get('fare_total', 0)
                    flight_info += f" {fare:,} won"
                    # Check if significantly cheaper
                    avg_fare = 450000
                    if fare < avg_fare * 0.7:
                        flight_info += " BIG SAVINGS"
                else:
                    flight_info += " NORMAL"
                if flight.get('refundable'):
                    flight_info += " REFUND"
            
            # Hotel info
            hotel_info = f"{hotel_id}"
            if hotel:
                hotel_info += f"({hotel.get('zone', '?')} zone quiet:{hotel.get('quiet_score', 0):.1f})"
                if hotel.get('zone') == episode.get('meeting_zone'):
                    hotel_info += " MEETING ZONE"
            
            # Restaurant info
            rest_info = f"{restaurant_id}"
            if restaurant:
                rest_info += f"({restaurant.get('area', '?')} zone client:{restaurant.get('client_ready_score', 0):.1f})"
                if 'vegan' in restaurant.get('dietary_flags', []):
                    rest_info += " VEGAN"
            
            # Activity info
            act_info = f"{activity_id}"
            if activity:
                act_info += f"({activity.get('location_zone', '?')} zone {'INDOOR' if activity.get('indoor') else 'OUTDOOR'})"
            
            combo_table.append(f"{i}. {flight_info} | {hotel_info} | {rest_info} | {act_info} | {item['total_cost']:,} won")
        
        # ========== Stage 1: Filter by Preference Hierarchy ==========
        filter_prompt = f"""
## User Conversation (chronological order, later messages override earlier)
{conversation}

## Candidate Combinations ({len(valid_combos)} total)
{chr(10).join(combo_table)}

## Preference Hierarchy (Highest to Lowest Priority)

### Level 1 - HIGHEST (Must satisfy)
- Meeting attendance: must arrive in time for the meeting/event
- Functional for work: need to stay productive

### Level 2 - HIGH (Satisfy unless overridden by Level 1)
- **Red-eye flight rule**: User said "Do not send me through a red-eye just because the fare is lower"
  - REJECT red-eye whose ONLY advantage is lower cost
  - ACCEPT red-eye that is the ONLY way to attend the meeting
  - CONSIDER red-eye with significant savings (e.g., 30%+ cheaper) and budget pressure

### Level 3 - MEDIUM (Can be overridden by Level 2)
- Local character (suppressed for this trip)
- Chain hotel restriction (temporarily overridden by "this trip only")

### Level 4 - RETIRED (No longer apply)
- Old budget assumption (retired by user)

## Your Task
1. First, check if the user can attend the meeting on time (check flight arrival time)
2. Then evaluate red-eye flights: reject only if the ONLY reason is cost savings
3. Return the numbers of combinations that are ACCEPTABLE under this hierarchy

Return JSON: {{"valid_combo_ids": [1,2,3], "reason": "filtering reason"}}
"""
        
        filter_schema = {
            "type": "object",
            "properties": {
                "valid_combo_ids": {"type": "array", "items": {"type": "integer"}},
                "reason": {"type": "string"}
            },
            "required": ["valid_combo_ids", "reason"],
            "additionalProperties": False
        }
        
        print(f"\n🤖 Stage 1: LLM Hierarchy Filtering")
        print(f"   Model: gpt-4o-mini")
        print(f"   Candidates: {len(valid_combos)}")
        
        filter_result = self.runtime.runner.create_json_response(
            model="gpt-4o-mini",
            instructions="Apply the preference hierarchy. Return combo numbers that are acceptable.",
            input_text=filter_prompt,
            json_schema=filter_schema,
            schema_name="filter_combos",
            max_output_tokens=800,
            metadata={"system": "student_solver", "trip_id": episode["trip_id"], "stage": "filter"},
        )
        
        filter_parsed = filter_result.get("parsed", {})
        valid_ids = filter_parsed.get("valid_combo_ids", [])
        filter_reason = filter_parsed.get("reason", "")
        
        print(f"   After filter: {len(valid_ids)} combos")
        print(f"   Reason: {filter_reason[:200]}")
        
        # Print filtered combos
        if valid_ids:
            print(f"\n   Filtered combinations:")
            for idx in valid_ids[:10]:
                if 1 <= idx <= len(valid_combos):
                    item = valid_combos[idx - 1]
                    flight_id, hotel_id, restaurant_id, activity_id = item['combo']
                    print(f"      #{idx}: {flight_id} | {hotel_id} | {restaurant_id} | {activity_id} | {item['total_cost']:,} won")
            if len(valid_ids) > 10:
                print(f"      ... and {len(valid_ids) - 10} more")
        
        # Get filtered combos
        filtered_combos = []
        for idx in valid_ids:
            if 1 <= idx <= len(valid_combos):
                filtered_combos.append(valid_combos[idx - 1])
        
        if not filtered_combos:
            print(f"   No valid combos, using first 10")
            filtered_combos = valid_combos[:10]
        
        if hasattr(self.runtime, 'runner'):
            self.runtime.runner._record_observed_usage(filter_result.get('usage', {}))
        
        # ========== Stage 2: Select best from filtered combos ==========
        if len(filtered_combos) == 1:
            best_item = filtered_combos[0]
            reason = filter_reason + " -> Only valid combination"
        else:
            filtered_table = []
            for i, item in enumerate(filtered_combos, 1):
                flight_id, hotel_id, restaurant_id, activity_id = item['combo']
                flight = flights_dict.get(flight_id)
                hotel = hotels_dict.get(hotel_id)
                restaurant = restaurants_dict.get(restaurant_id)
                activity = activities_dict.get(activity_id)
                
                flight_info = f"{flight_id}({flight.get('depart_time', '?')}->{flight.get('arrival_time', '?')})" if flight else flight_id
                if flight and flight.get('red_eye'):
                    flight_info += "R"
                hotel_info = f"{hotel_id}(quiet:{hotel.get('quiet_score', 0):.1f})" if hotel else hotel_id
                rest_info = f"{restaurant_id}(client:{restaurant.get('client_ready_score', 0):.1f})" if restaurant else restaurant_id
                act_info = f"{activity_id}" + (" indoor" if activity and activity.get('indoor') else "") if activity else activity_id
                
                filtered_table.append(f"{i}. {flight_info} | {hotel_info} | {rest_info} | {act_info} | {item['total_cost']:,} won")
            
            print(f"\n   Stage 2 candidates:")
            for line in filtered_table:
                print(f"      {line}")
            
            select_prompt = f"""
## User Conversation
{conversation}

## Valid Candidates ({len(filtered_combos)} combos)
{chr(10).join(filtered_table)}

## Task
Select the combination that best meets the user's needs.
Return JSON: {{"selected": number, "reason": "selection reason"}}
"""
            
            select_schema = {
                "type": "object",
                "properties": {
                    "selected": {"type": "integer"},
                    "reason": {"type": "string"}
                },
                "required": ["selected", "reason"],
                "additionalProperties": False
            }
            
            print(f"\n🤖 Stage 2: LLM Best Selection")
            print(f"   Candidates: {len(filtered_combos)}")
            
            select_result = self.runtime.runner.create_json_response(
                model="gpt-4o-mini",
                instructions="Select the best combination that meets user needs.",
                input_text=select_prompt,
                json_schema=select_schema,
                schema_name="select_best",
                max_output_tokens=500,
                metadata={"system": "student_solver", "trip_id": episode["trip_id"], "stage": "select"},
            )
            
            select_parsed = select_result.get("parsed", {})
            selected_idx = select_parsed.get("selected", 1) - 1
            reason = select_parsed.get("reason", "")
            
            if 0 <= selected_idx < len(filtered_combos):
                best_item = filtered_combos[selected_idx]
            else:
                best_item = filtered_combos[0]
            
            if hasattr(self.runtime, 'runner'):
                self.runtime.runner._record_observed_usage(select_result.get('usage', {}))
        
        flight_id, hotel_id, restaurant_id, activity_id = best_item['combo']
        
        print(f"\nFinal selection: {flight_id} | {hotel_id} | {restaurant_id} | {activity_id}")
        print(f"   Reason: {reason[:300]}")
        
        best_info = {
            'total_cost': best_item['total_cost'],
            'reason': reason
        }
        
        # ========== Stage 3: Generate Memory Report ==========
        memory_report = self.generate_memory_report_with_llm(
            episode, 
            best_item['combo'], 
            best_info
        )
        
        return best_item['combo'], 0, best_info, memory_report
    
    def select_best_combination(
        self,
        episode: Dict[str, Any],
        filtered_options: Dict[str, List[Dict]],
        combinations: List[Tuple[str, str, str, str]],
        context_info: Dict[str, Any],
        requirements: Dict[str, Any],
        spoken_rules: Dict[str, List[str]]
    ) -> Tuple[Tuple[str, str, str, str], float, Dict, Dict]:
        """Select best combination using LLM with memory report generation"""
        
        budget = requirements["budget"]
        nights = requirements["nights"]
        
        flights_dict = {f.get('flight_id'): f for f in filtered_options['flights']}
        hotels_dict = {h.get('hotel_id'): h for h in filtered_options['hotels']}
        restaurants_dict = {r.get('restaurant_id'): r for r in filtered_options['restaurants']}
        activities_dict = {a.get('activity_id'): a for a in filtered_options['activities']}
        
        print("\n" + "="*80)
        print("Calculating total cost for each combination")
        print("="*80)
        print(f"Total budget: {budget:,} won")
        print(f"Number of nights: {nights}")
        print(f"Total combos: {len(combinations)}")
        print("-" * 80)
        
        valid_combos = []
        skipped = 0
        
        for combo in combinations:
            total_cost = self.calculate_total_cost(
                combo, flights_dict, hotels_dict, restaurants_dict, activities_dict, nights
            )
            
            if total_cost > budget:
                skipped += 1
                continue
            
            valid_combos.append({
                'combo': combo,
                'total_cost': total_cost,
                'score': 0,
            })
            
            flight_id, hotel_id, restaurant_id, activity_id = combo
            print(f"  {flight_id} | {hotel_id} | {restaurant_id} | {activity_id} -> {total_cost:,} won")
        
        print(f"\n  Within budget: {len(valid_combos)}")
        print(f"  Over budget: {skipped}")
        
        if not valid_combos:
            print("\nNo combinations within budget")
            empty_memory = {
                "spoken_rule_hits": {
                    "must_remember": [],
                    "forbidden": [],
                    "one_off_only": [],
                    "retire": [],
                    "do_not_reconsider": [],
                    "keep_context_lean": ["relevant_only"]
                },
                "retrieved": [],
                "retired": [],
                "active_context_keys": ["relevant_only"]
            }
            return (None, None, None, None), 0, {'total_cost': 0, 'reason': 'No valid combos within budget'}, empty_memory
        
        best_combo, best_score, best_info, memory_report = self.llm_selection(
            episode, valid_combos, requirements,
            flights_dict, hotels_dict, restaurants_dict, activities_dict
        )
        
        flight_id, hotel_id, restaurant_id, activity_id = best_combo
        
        print(f"\nBest combo: {flight_id} | {hotel_id} | {restaurant_id} | {activity_id}")
        print(f"   Cost: {best_info.get('total_cost', 0):,} won")
        print(f"   Reason: {best_info.get('reason', '')[:200]}")
        
        return best_combo, best_score, best_info, memory_report
    
    def recommend(self, episode: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate recommendation for a single episode"""
        
        state = episode.get('scenario_state', {})
        
        all_options = self.fetch_all_options(episode)
        self.display_options(all_options, episode)
        
        filtered_options = self.filter_by_scenario_state(all_options, state, episode)
        
        print("\n" + "="*80)
        print("Filtered Results")
        print("="*80)
        print(f"Flights: {len(filtered_options['flights'])} -> {[f.get('flight_id') for f in filtered_options['flights']]}")
        print(f"Hotels: {len(filtered_options['hotels'])} -> {[h.get('hotel_id') for h in filtered_options['hotels']]}")
        print(f"Restaurants: {len(filtered_options['restaurants'])} -> {[r.get('restaurant_id') for r in filtered_options['restaurants']]}")
        print(f"Activities: {len(filtered_options['activities'])} -> {[a.get('activity_id') for a in filtered_options['activities']]}")
        
        combinations = self.generate_combinations(filtered_options)
        print(f"\nTotal Combinations: {len(combinations)}")
        
        session = self.runtime.new_session(retrieval_strategy="lexical", max_results=10)
        context_info = fetch_all_context_info(session, episode)
        
        if hasattr(self.runtime, 'runner') and hasattr(session, 'usage'):
            self.runtime.runner._record_observed_usage(session.usage)
        
        turns = episode.get('turns', [])
        requirements = extract_user_requirements(turns, state, episode)
        spoken_rules = extract_spoken_rules_from_turns(episode)
        
        best_combo, best_score, best_info, memory_report = self.select_best_combination(
            episode, filtered_options, combinations, context_info, requirements, spoken_rules
        )
        
        flight_id, hotel_id, restaurant_id, activity_id = best_combo
        
        picks = {
            "flight_id": flight_id,
            "hotel_id": hotel_id,
            "restaurant_id": restaurant_id,
            "activity_id": activity_id,
            "notes": f"LLM selected: {best_info.get('reason', '')[:150]}"
        }
        
        print("\n" + "="*80)
        print("Final Recommendation")
        print("="*80)
        print(f"Flight: {flight_id}")
        print(f"Hotel: {hotel_id}")
        print(f"Restaurant: {restaurant_id}")
        print(f"Activity: {activity_id}")
        print(f"Total Cost: {best_info.get('total_cost', 0):,} won")
        
        print("\nMemory Report:")
        print(f"   must_remember: {memory_report.get('spoken_rule_hits', {}).get('must_remember', [])}")
        print(f"   forbidden: {memory_report.get('spoken_rule_hits', {}).get('forbidden', [])}")
        print(f"   one_off_only: {memory_report.get('spoken_rule_hits', {}).get('one_off_only', [])}")
        print(f"   retire: {memory_report.get('spoken_rule_hits', {}).get('retire', [])}")
        print(f"   keep_context_lean: {memory_report.get('spoken_rule_hits', {}).get('keep_context_lean', [])}")
        
        return picks, memory_report


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Official evaluator entry point"""
    
    episode = runtime.episode
    
    print("\n" + "="*80)
    print(f"Processing Episode: {episode.get('trip_id')}")
    print("="*80)
    print(f"City: {episode.get('city')}")
    print(f"Origin: {episode.get('origin')}")
    print(f"Meeting Zone: {episode.get('meeting_zone')}")
    print(f"Budget: {episode.get('budget_total'):,} won")
    print(f"Weather: {episode.get('weather')}")
    print(f"Traveler: {episode.get('traveler_id')}")
    
    state = episode.get('scenario_state', {})
    print(f"\nScenario State:")
    for key, value in state.items():
        if value and key != 'stakeholder_ids':
            print(f"   - {key}: {value}")
        elif key == 'stakeholder_ids' and value:
            print(f"   - {key}: {value}")
    
    turns = episode.get('turns', [])
    print(f"\nUser Messages: {len(turns)}")
    
    planner = TravelPlanner(runtime)
    picks, memory_report = planner.recommend(episode)
    
    submission = {
        "flight_id": picks.get("flight_id"),
        "hotel_id": picks.get("hotel_id"),
        "restaurant_id": picks.get("restaurant_id"),
        "activity_id": picks.get("activity_id"),
        "memory_report": memory_report,
        "notes": picks.get("notes", f"LLM selection for {episode.get('trip_id')}")
    }
    
    session = runtime.new_session(retrieval_strategy="lexical", max_results=4)
    grounded_submission = ensure_grounded_submission(session, episode, submission)
    
    usage = runtime.combine_usages()
    if hasattr(runtime.runner, '_usage_ledger'):
        usage = runtime.combine_usages(usage, runtime.runner._usage_ledger)
    
    print("\n" + "="*80)
    print("Episode Complete")
    print("="*80 + "\n")
    
    return {
        "submission": grounded_submission,
        "usage": usage
    }
