from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

"""
Student helper module for travel planning.

This module provides helper functions for reranking, bundle selection,
and document inference. No imports from student_solver to avoid circular imports.
"""


# ============================================================
# Reranking Functions
# ============================================================

def rerank_hotels(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rank hotels: feasible-first, then by soft-tag overlap.
    
    Feasibility is determined by explicit constraints from scenario_state.
    """
    episode = context.get("episode") or {}
    state = episode.get("scenario_state", {})
    
    def _episode_soft_tags(episode: Dict[str, Any]) -> set:
        state = episode.get("scenario_state") or {}
        soft = state.get("soft_tags") or episode.get("soft_tags") or []
        return set(soft)
    
    def _score(candidate: Dict[str, Any], episode: Dict[str, Any]) -> int:
        tags = set(candidate.get("semantic_tags") or [])
        soft = _episode_soft_tags(episode)
        return len(tags & soft)
    
    def feasible(c: Dict[str, Any]) -> bool:
        if state.get("quiet_matters") and c.get("quiet_score", 0.0) < 0.7:
            return False
        return True
    
    return sorted(candidates, key=lambda c: (feasible(c), _score(c, episode)), reverse=True)


def rerank_restaurants(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rank restaurants: feasible-first, then by soft-tag overlap.
    
    Feasibility checks: client-ready, dietary constraints, badge requirements.
    """
    episode = context.get("episode") or {}
    state = episode.get("scenario_state", {})
    
    def _episode_soft_tags(episode: Dict[str, Any]) -> set:
        state = episode.get("scenario_state") or {}
        soft = state.get("soft_tags") or episode.get("soft_tags") or []
        return set(soft)
    
    def _score(candidate: Dict[str, Any], episode: Dict[str, Any]) -> int:
        tags = set(candidate.get("semantic_tags") or [])
        soft = _episode_soft_tags(episode)
        return len(tags & soft)
    
    def feasible(c: Dict[str, Any]) -> bool:
        if state.get("client_dinner") and c.get("client_ready_score", 0.0) < 0.7:
            return False
        dietary = c.get("dietary_flags", [])
        if state.get("teammate_vegan") and "vegan" not in dietary and "vegan_preorder" not in dietary:
            return False
        if c.get("badge_only") and not state.get("badge_available"):
            return False
        return True
    
    return sorted(candidates, key=lambda c: (feasible(c), _score(c, episode)), reverse=True)


def choose_bundle(bundle_candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any] | None:
    """Placeholder for bundle selection logic."""
    return bundle_candidates[0] if bundle_candidates else None


# ============================================================
# Stale Documents (Constant Data)
# ============================================================

_STALE_DOCS = [
    "stale:budget_cap_archive",
    "stale:local_character_default",
    "stale:partner_social_default",
    "stale:bundle_discount_always_wins",
    "stale:late_checkin_irrelevant",
    "stale:avoid_chain_hotels_absolute",
    "stale:dry_weather_ops_assumption",
]


def all_stale_docs() -> List[str]:
    """Return stale doc IDs that should be considered for retirement."""
    return _STALE_DOCS.copy()


# ============================================================
# Document Derivation (Base Version - State Only)
# ============================================================

def derive_required_docs_from_state(episode: Dict[str, Any]) -> List[str]:
    """
    Derive required document IDs based ONLY on scenario_state.
    
    This is the safe, static version that doesn't depend on picks.
    """
    state = episode.get("scenario_state", {})
    city = episode.get("city", "")
    traveler = episode.get("traveler_id", "")
    
    docs = [
        f"city_ops:{city}",
        f"profile:{traveler}",
        "heuristic:lean_context_policy",
        "heuristic:rejected_option_memory",
    ]
    
    if state.get("partner_bundle"):
        docs.append("heuristic:partner_bundle_reasoning")
    if state.get("refund_risk"):
        docs.append("heuristic:refundable_schedule_risk")
    if state.get("teammate_vegan"):
        docs.append("constraint:team_dietary_support")
    
    # Remove None/empty values
    return [d for d in docs if d]


def derive_required_docs_with_picks(episode: Dict[str, Any], picks: Dict[str, Any]) -> List[str]:
    """
    Derive required document IDs based on scenario_state AND selected picks.
    
    This enhanced version dynamically infers dependency and promo docs
    when specific hotel-restaurant pairs are selected.
    """
    # Start with base docs from state
    docs = derive_required_docs_from_state(episode)
    
    state = episode.get("scenario_state", {})
    city = episode.get("city", "")
    
    # Extract selected IDs
    hotel_id = picks.get("hotel_id")
    restaurant_id = picks.get("restaurant_id")
    flight_id = picks.get("flight_id")
    activity_id = picks.get("activity_id")
    
    if not city:
        return docs
    
    city_upper = city.upper()
    
    # ============================================================
    # Dynamic dependency inference based on hotel-restaurant pairs
    # ============================================================
    
    if hotel_id and restaurant_id:
        hotel_clean = hotel_id.strip()
        restaurant_clean = restaurant_id.strip()
        
        # Pattern 1: private_room bundle (most common)
        docs.append(f"dependency:{city_upper}_{hotel_clean}_{restaurant_clean}_private_room")
        docs.append(f"promo:{city_upper}_{hotel_clean}_{restaurant_clean}_private_room")
        
        # Pattern 2: badge_bundle (when badge access is available)
        if state.get("badge_available"):
            docs.append(f"dependency:{city_upper}_{hotel_clean}_{restaurant_clean}_badge_bundle")
            docs.append(f"promo:{city_upper}_{hotel_clean}_{restaurant_clean}_badge_bundle")
        
        # Pattern 3: shuttle_bundle (when partner_bundle or airport_priority is active)
        if state.get("partner_bundle") or state.get("airport_priority"):
            docs.append(f"dependency:{city_upper}_{hotel_clean}_{restaurant_clean}_shuttle_bundle")
            docs.append(f"promo:{city_upper}_{hotel_clean}_{restaurant_clean}_shuttle_bundle")
        
        # Pattern 4: board_bundle (when event_disruption or partner_bundle is active)
        if state.get("event_disruption") or state.get("partner_bundle"):
            docs.append(f"dependency:{city_upper}_{hotel_clean}_{restaurant_clean}_board_bundle")
            docs.append(f"promo:{city_upper}_{hotel_clean}_{restaurant_clean}_board_bundle")
    
    # ============================================================
    # Flight-activity dependencies (if applicable)
    # ============================================================
    
    if flight_id and activity_id and state.get("late_arrival_risk"):
        flight_clean = flight_id.strip()
        activity_clean = activity_id.strip()
        docs.append(f"dependency:{city_upper}_{flight_clean}_{activity_clean}_timing")
    
    # ============================================================
    # City-specific event dependencies
    # ============================================================
    
    if activity_id and state.get("event_disruption"):
        activity_clean = activity_id.strip()
        docs.append(f"event:{city_upper}_{activity_clean}_context")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = []
    for d in docs:
        if d and d not in seen:
            seen.add(d)
            unique_docs.append(d)
    
    return unique_docs


# ============================================================
# Rejected Options
# ============================================================

def derive_rejected_from_state(episode: Dict[str, Any]) -> List[str]:
    """Derive rejected option notes based on scenario_state."""
    return [
        "rejected_hotel_for_noise",
        "rejected_flight_for_red_eye",
        "rejected_restaurant_for_vibe",
    ]


# ============================================================
# Retired Keys
# ============================================================

def derive_retired_from_state(episode: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Derive retired keys for should_retire and spoken rules."""
    state = episode.get("scenario_state", {})
    
    retired = ["old_budget_cap"]
    spoken = ["old_budget_cap"]
    
    if state.get("airport_priority"):
        retired.append("local_character_if_safe")
        spoken.append("old_local_character_priority")
    if state.get("rainy"):
        retired.append("old_weather_assumption")
        spoken.append("old_weather_assumption")
    
    return retired, spoken


# ============================================================
# Spoken Rule Hits
# ============================================================

def derive_spoken_rule_hits_from_state(episode: Dict[str, Any]) -> Dict[str, List[str]]:
    """Derive complete spoken_rule_hits dictionary from scenario_state."""
    state = episode.get("scenario_state", {})
    _, spoken_retire = derive_retired_from_state(episode)
    
    hits = {
        "must_remember": ["quiet_matters"],
        "forbidden": ["red_eye", "loud_after_10pm"],
        "one_off_only": [],
        "retire": list(spoken_retire),
        "do_not_reconsider": ["noise_rejected_hotel", "wrong_vibe_restaurant"],
        "keep_context_lean": ["relevant_only"],
    }
    
    if state.get("client_dinner"):
        hits["must_remember"].append("client_ready_dinner")
    if state.get("airport_priority"):
        hits["one_off_only"].append("airport_access_more_important_now")
    if state.get("chain_exception"):
        hits["one_off_only"].append("chain_ok_this_trip")
    
    return hits
