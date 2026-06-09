from __future__ import annotations

import re
import json
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
    """Scenario-based travel planner with LLM + CoT understanding user preferences"""
    
    def __init__(self, runtime: StudentRuntime):
        self.runtime = runtime
    
    def _parse_hour(self, time_str: str) -> int:
        """Safely parse time string, return hour (24-hour format)"""
        if not time_str:
            return 12
        try:
            cleaned = time_str.strip().lower()
            if ':' in cleaned:
                hour_part = cleaned.split(':')[0].strip()
                hour = int(hour_part)
                if 'pm' in cleaned and hour < 12:
                    hour += 12
                if 'am' in cleaned and hour == 12:
                    hour = 0
                return hour
            match = re.search(r'(\d{1,2})', cleaned)
            if match:
                hour = int(match.group(1))
                if 'pm' in cleaned and hour < 12:
                    hour += 12
                if 'am' in cleaned and hour == 12:
                    hour = 0
                return hour
            return 12
        except (ValueError, IndexError, AttributeError):
            return 12
    
    def _can_attend_meeting(self, flight: Dict, meeting_time: str = "09:00") -> tuple[bool, str]:
        """Determine if flight allows attending meeting on Day 2 at meeting_time"""
        if not flight:
            return False, "No flight"
        
        flight_id = flight.get('flight_id', '')
        arrival_time = flight.get('arrival_time', '')
        arrival_day = flight.get('arrival_day', flight.get('day', 'day1'))
        
        if arrival_day == 'day1' or 'day1' in str(flight_id).lower():
            return True, "Arrives Day 1"
        
        if arrival_day == 'day2' or 'day2' in str(flight_id).lower():
            arr_hour = self._parse_hour(arrival_time)
            meeting_hour = self._parse_hour(meeting_time)
            if arr_hour < meeting_hour:
                return True, f"Arrives Day 2 at {arrival_time} (before meeting)"
            else:
                return False, f"Arrives Day 2 at {arrival_time} (misses meeting)"
        
        return True, "Assuming Day 1 arrival"
    
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
            day_flag = f.get('arrival_day', f.get('day', 'day1'))
            print(f"  - {f.get('flight_id')}: {f.get('depart_time', 'N/A')} → {f.get('arrival_time', 'N/A')} ({day_flag}), "
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
    
    def llm_select_with_cot(
        self,
        episode: Dict[str, Any],
        valid_combos: List[Dict],
        flights_dict: Dict,
        hotels_dict: Dict,
        restaurants_dict: Dict,
        activities_dict: Dict
    ) -> Tuple[Tuple[str, str, str, str], Dict, Dict]:
        """
        LLM + Chain of Thought selection - compares ALL options and picks the best
        """
        
        turns = episode.get('turns', [])
        meeting_time = episode.get('meeting_time', '09:00')
        meeting_zone = episode.get('meeting_zone', '')
        budget_total = episode.get('budget_total', 0)
        weather = episode.get('weather', '')
        
        # Build conversation
        conversation_lines = []
        for i, turn in enumerate(turns, 1):
            speaker = "Traveler" if turn.get('speaker') == 'user' else "Assistant"
            text = turn.get('text', '')
            conversation_lines.append(f"Message {i} - {speaker}: {text}")
        conversation = "\n".join(conversation_lines)
        
        # Build options summary
        flights_summary = []
        for f_id, flight in flights_dict.items():
            fare = flight.get('fare_total', 0)
            arr_time = flight.get('arrival_time', '?')
            arr_day = flight.get('arrival_day', flight.get('day', 'day1'))
            red_eye = "⚠️RED-EYE" if flight.get('red_eye') else "Normal"
            refund = "Refundable" if flight.get('refundable') else "Non-refund"
            can_attend, _ = self._can_attend_meeting(flight, meeting_time)
            attend = "✅CAN MEET" if can_attend else "❌MISSES MEETING"
            flights_summary.append(f"  {f_id}: Arr {arr_time} ({arr_day}), ₩{fare:,}, {red_eye}, {refund}, {attend}")
        
        hotels_summary = []
        for h_id, hotel in hotels_dict.items():
            price = hotel.get('nightly_price', 0)
            zone = hotel.get('zone', '?')
            quiet = hotel.get('quiet_score', 0)
            airport = hotel.get('airport_access_score', 0)
            chain = "Chain" if hotel.get('chain') else "Independent"
            zone_match = "🎯MEETING ZONE" if zone == meeting_zone else f"Zone:{zone}"
            hotels_summary.append(f"  {h_id}: ₩{price}/night, Quiet:{quiet:.1f}, Airport:{airport:.1f}, {chain}, {zone_match}")
        
        restaurants_summary = []
        for r_id, restaurant in restaurants_dict.items():
            area = restaurant.get('area', '?')
            client_score = restaurant.get('client_ready_score', 0)
            quiet_score = restaurant.get('quiet_score', 0)
            dietary = restaurant.get('dietary_flags', [])
            diet_str = ", ".join(dietary) if dietary else "None"
            restaurants_summary.append(f"  {r_id}: {area}, Client-ready:{client_score:.1f}, Quiet:{quiet_score:.1f}, Diet:[{diet_str}]")
        
        activities_summary = []
        for a_id, activity in activities_dict.items():
            zone = activity.get('location_zone', '?')
            price = activity.get('price', 0)
            indoor = "🏠Indoor" if activity.get('indoor') else "🌳Outdoor"
            zone_match = "🎯MEETING ZONE" if zone == meeting_zone else f"Zone:{zone}"
            activities_summary.append(f"  {a_id}: {zone_match}, ₩{price}, {indoor}")
        
        # Build combo table (all within budget that can attend meeting)
        combo_table = []
        for i, item in enumerate(valid_combos, 1):
            flight_id, hotel_id, restaurant_id, activity_id = item['combo']
            flight = flights_dict.get(flight_id)
            hotel = hotels_dict.get(hotel_id)
            restaurant = restaurants_dict.get(restaurant_id)
            activity = activities_dict.get(activity_id)
            
            # Score indicators
            indicators = []
            if flight and flight.get('red_eye'):
                indicators.append("⚠️RED-EYE")
            if flight and flight.get('refundable'):
                indicators.append("🔄REFUND")
            if hotel and hotel.get('zone') == meeting_zone:
                indicators.append("🎯HOTEL_ZONE")
            if hotel and hotel.get('quiet_score', 0) >= 8.5:
                indicators.append("🤫QUIET")
            if restaurant and restaurant.get('client_ready_score', 0) >= 8.0:
                indicators.append("🍽️CLIENT_READY")
            if activity and activity.get('indoor') and weather == 'rainy':
                indicators.append("🏠INDOOR_GOOD")
            
            indicator_str = " ".join(indicators) if indicators else ""
            
            combo_table.append(
                f"{i}. {flight_id} | {hotel_id} | {restaurant_id} | {activity_id} | ₩{item['total_cost']:,} {indicator_str}"
            )
        
        # CoT prompt - forces comparison
        cot_prompt = f"""You are a travel planner. Read the conversation and COMPARE ALL combinations to select the BEST one.

## TRIP INFO
- Destination: {episode.get('city')}
- Meeting: Day 2 at {meeting_time} in {meeting_zone} zone
- 2 nights (arrive Day 1, meeting Day 2, depart Day 3)
- Budget: ₩{budget_total:,}
- Weather: {weather}

## CONVERSATION (read carefully to understand what user wants)
{conversation}

## AVAILABLE OPTIONS

FLIGHTS:
{chr(10).join(flights_summary)}

HOTELS:
{chr(10).join(hotels_summary)}

RESTAURANTS:
{chr(10).join(restaurants_summary)}

ACTIVITIES:
{chr(10).join(activities_summary)}

## ALL CANDIDATE COMBINATIONS ({len(combo_table)} total)
{chr(10).join(combo_table)}

## YOUR TASK: COMPARE ALL combinations and select the BEST

Follow these steps EXACTLY:

### STEP 1: Extract user preferences from conversation
List what user explicitly says they want:
- Flight preferences:
- Hotel preferences:
- Restaurant preferences:
- Activity preferences:
- Dealbreakers (things user says NO to):
- "This trip only" overrides:

### STEP 2: Score each combination (1-10 for each category)
For each combination, rate:
- Flight score (arrival time, no red-eye, refundable)
- Hotel score (quiet, meeting zone, airport access)
- Restaurant score (client-ready, dietary needs)
- Activity score (weather-appropriate, zone)
- Cost score (within budget, good value)

### STEP 3: Compare top 3-5 combinations
Which combinations have the highest scores? Why?

### STEP 4: Select the absolute best
Which combination # best satisfies ALL the user's preferences?

## OUTPUT FORMAT
Return ONLY valid JSON:
{{
  "selected": number,
  "reason": "Why this is best compared to others (mention trade-offs)",
  "flight_pref": "Summary of what user wants for flights",
  "hotel_pref": "Summary of what user wants for hotels", 
  "restaurant_pref": "Summary of what user wants for restaurants",
  "activity_pref": "Summary of what user wants for activities"
}}

IMPORTANT:
- You MUST compare ALL combinations, not just the first one
- The "selected" number MUST be between 1 and {len(combo_table)}
- Your reason MUST explain why this is better than alternatives
- If there are trade-offs, explain why this choice is optimal

Do not include any text before or after the JSON."""
        
        print(f"\n🤖 LLM + CoT Selection (Comparing all {len(combo_table)} combinations)")
        print(f"   Conversation messages: {len(turns)}")
        
        try:
            response = self.runtime.runner.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a travel planning assistant. You MUST compare ALL options and select the BEST one. Output ONLY valid JSON."},
                    {"role": "user", "content": cot_prompt}
                ],
                max_tokens=3000,
                temperature=0.3,
            )
            
            response_text = response.choices[0].message.content
            
            print(f"\n   Raw response preview: {response_text[:400]}...")
            
            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*"selected"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                # Try to find any JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    parsed = {}
            
            selected_num = parsed.get("selected", 1)
            reasoning = parsed.get("reason", "LLM selected after comparing all options")
            
            # Validate selected number
            if not (1 <= selected_num <= len(combo_table)):
                print(f"   ⚠️ Invalid selection {selected_num}, using best based on cost")
                # Fallback: pick cheapest that meets basic criteria
                selected_num = 1
                for i, item in enumerate(valid_combos, 1):
                    flight_id = item['combo'][0]
                    flight = flights_dict.get(flight_id)
                    if flight and not flight.get('red_eye'):
                        selected_num = i
                        reasoning = "Fallback: selected non-red-eye flight with reasonable cost"
                        break
                    if item['total_cost'] < valid_combos[0]['total_cost']:
                        selected_num = i
                if selected_num == 1:
                    reasoning = "Fallback: selected first combination"
            
            best_item = valid_combos[selected_num - 1]
            
            # Track usage
            if hasattr(self.runtime, 'runner') and hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                self.runtime.runner._record_observed_usage(usage)
                
        except Exception as e:
            print(f"   ⚠️ LLM selection failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: select combination with highest quiet + client scores
            best_item = None
            best_score = -1
            for item in valid_combos:
                flight_id, hotel_id, restaurant_id, activity_id = item['combo']
                hotel = hotels_dict.get(hotel_id)
                restaurant = restaurants_dict.get(restaurant_id)
                score = 0
                if hotel:
                    score += hotel.get('quiet_score', 0)
                if restaurant:
                    score += restaurant.get('client_ready_score', 0)
                if score > best_score:
                    best_score = score
                    best_item = item
            if best_item is None:
                best_item = valid_combos[0] if valid_combos else None
            reasoning = "Fallback: selected combination with highest quiet + client-ready scores"
        
        if best_item is None:
            empty_memory = {
                "spoken_rule_hits": {"must_remember": [], "forbidden": [], "keep_context_lean": ["relevant_only"]}
            }
            return (None, None, None, None), {'total_cost': 0, 'reason': 'No valid combos'}, empty_memory
        
        flight_id, hotel_id, restaurant_id, activity_id = best_item['combo']
        
        print(f"\n✅ Selected: {flight_id} | {hotel_id} | {restaurant_id} | {activity_id}")
        print(f"   Cost: ₩{best_item['total_cost']:,}")
        print(f"   Reason: {reasoning[:300]}")
        
        best_info = {
            'total_cost': best_item['total_cost'],
            'reason': reasoning
        }
        
        # Simple memory report
        memory_report = {
            "retrieved": [],
            "retired": [],
            "active_context_keys": ["relevant_only"],
            "spoken_rule_hits": {
                "must_remember": [],
                "forbidden": [],
                "one_off_only": [],
                "retire": [],
                "keep_context_lean": ["relevant_only"]
            }
        }
        
        return best_item['combo'], best_info, memory_report
    
    def select_best_combination(
        self,
        episode: Dict[str, Any],
        filtered_options: Dict[str, List[Dict]],
        combinations: List[Tuple[str, str, str, str]],
        context_info: Dict[str, Any],
        requirements: Dict[str, Any],
        spoken_rules: Dict[str, List[str]]
    ) -> Tuple[Tuple[str, str, str, str], Dict, Dict]:
        """Select best combination using LLM + CoT based on user preferences"""
        
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
        
        valid_combos = []
        for combo in combinations:
            total_cost = self.calculate_total_cost(
                combo, flights_dict, hotels_dict, restaurants_dict, activities_dict, nights
            )
            
            if total_cost <= budget:
                valid_combos.append({
                    'combo': combo,
                    'total_cost': total_cost,
                })
        
        print(f"  Within budget: {len(valid_combos)}")
        
        if not valid_combos:
            print("\n⚠️ No combinations within budget")
            empty_memory = {
                "spoken_rule_hits": {"must_remember": [], "forbidden": [], "keep_context_lean": ["relevant_only"]}
            }
            return (None, None, None, None), {'total_cost': 0, 'reason': 'No valid combos within budget'}, empty_memory
        
        # Filter by meeting attendance
        meeting_time = episode.get('meeting_time', '09:00')
        meeting_combos = []
        for item in valid_combos:
            flight_id = item['combo'][0]
            flight = flights_dict.get(flight_id)
            can_attend, _ = self._can_attend_meeting(flight, meeting_time)
            if can_attend:
                meeting_combos.append(item)
        
        print(f"  Can attend meeting: {len(meeting_combos)}")
        
        if not meeting_combos:
            print("⚠️ No flights can attend meeting! Using all combos.")
            meeting_combos = valid_combos
        
        # Use LLM + CoT to select (compares all)
        best_combo, best_info, memory_report = self.llm_select_with_cot(
            episode, meeting_combos,
            flights_dict, hotels_dict, restaurants_dict, activities_dict
        )
        
        return best_combo, best_info, memory_report
    
    def recommend(self, episode: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate recommendation for a single episode"""
        
        state = episode.get('scenario_state', {})
        
        if 'meeting_time' not in episode:
            episode['meeting_time'] = '09:00'
        
        all_options = self.fetch_all_options(episode)
        self.display_options(all_options, episode)
        
        filtered_options = self.filter_by_scenario_state(all_options, state, episode)
        
        print("\n" + "="*80)
        print("Filtered Results")
        print("="*80)
        print(f"Flights: {len(filtered_options['flights'])}")
        print(f"Hotels: {len(filtered_options['hotels'])}")
        print(f"Restaurants: {len(filtered_options['restaurants'])}")
        print(f"Activities: {len(filtered_options['activities'])}")
        
        combinations = self.generate_combinations(filtered_options)
        print(f"Total Combinations: {len(combinations)}")
        
        session = self.runtime.new_session(retrieval_strategy="lexical", max_results=10)
        context_info = fetch_all_context_info(session, episode)
        
        if hasattr(self.runtime, 'runner') and hasattr(session, 'usage'):
            self.runtime.runner._record_observed_usage(session.usage)
        
        turns = episode.get('turns', [])
        requirements = extract_user_requirements(turns, state, episode)
        spoken_rules = extract_spoken_rules_from_turns(episode)
        
        best_combo, best_info, memory_report = self.select_best_combination(
            episode, filtered_options, combinations, context_info, requirements, spoken_rules
        )
        
        flight_id, hotel_id, restaurant_id, activity_id = best_combo
        
        picks = {
            "flight_id": flight_id,
            "hotel_id": hotel_id,
            "restaurant_id": restaurant_id,
            "activity_id": activity_id,
            "notes": best_info.get('reason', 'LLM+CoT selection')[:200]
        }
        
        print("\n" + "="*80)
        print("Final Recommendation")
        print("="*80)
        print(f"Flight: {flight_id}")
        print(f"Hotel: {hotel_id}")
        print(f"Restaurant: {restaurant_id}")
        print(f"Activity: {activity_id}")
        print(f"Total Cost: {best_info.get('total_cost', 0):,} won")
        print(f"\nReason: {best_info.get('reason', '')[:300]}")
        
        return picks, memory_report


def solve_episode(runtime: StudentRuntime) -> Dict[str, Any]:
    """Official evaluator entry point"""
    
    episode = runtime.episode
    
    print("\n" + "="*80)
    print(f"Processing Episode: {episode.get('trip_id')}")
    print("="*80)
    print(f"City: {episode.get('city')}")
    print(f"Origin: {episode.get('origin')}")
    print(f"Meeting: {episode.get('meeting_zone')} at {episode.get('meeting_time', '09:00')}")
    print(f"Budget: {episode.get('budget_total'):,} won")
    print(f"Weather: {episode.get('weather')}")
    
    state = episode.get('scenario_state', {})
    print(f"\nScenario State:")
    for key, value in state.items():
        if value and key != 'stakeholder_ids':
            print(f"   - {key}: {value}")
    
    turns = episode.get('turns', [])
    print(f"\nConversation: {len(turns)} messages")
    
    planner = TravelPlanner(runtime)
    picks, memory_report = planner.recommend(episode)
    
    submission = {
        "flight_id": picks.get("flight_id"),
        "hotel_id": picks.get("hotel_id"),
        "restaurant_id": picks.get("restaurant_id"),
        "activity_id": picks.get("activity_id"),
        "memory_report": memory_report,
        "notes": picks.get("notes", f"LLM+CoT selection")
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
