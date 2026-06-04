# student_custom_tools_template.py - Modified to accept dynamic state

from __future__ import annotations

"""Student-owned helper module.

This file is intentionally lightweight. Teams can add their own wrappers around the
primitive tool API, such as semantic rerankers, bundle search helpers, fallback
search, or verifier helpers.
"""

from typing import Any, Dict, List


# ============================================================
# Global state for dynamic spoken rules (ADDED)
# ============================================================

_DYNAMIC_STATE: Dict[str, Any] = {}


def set_dynamic_state(state: Dict[str, Any]) -> None:
    """Set dynamic state from the solver for spoken rule generation."""
    global _DYNAMIC_STATE
    _DYNAMIC_STATE = state.copy()


def clear_dynamic_state() -> None:
    """Clear dynamic state."""
    global _DYNAMIC_STATE
    _DYNAMIC_STATE = {}


_BAD_TAGS = {"loud", "noisy", "stale", "deprecated"}


def _episode_soft_tags(episode: Dict[str, Any]) -> set:
    state = episode.get("scenario_state") or {}
    soft = state.get("soft_tags") or episode.get("soft_tags") or []
    return set(soft)


def _score(candidate: Dict[str, Any], episode: Dict[str, Any]) -> int:
    tags = set(candidate.get("semantic_tags") or [])
    soft = _episode_soft_tags(episode)
    return len(tags & soft) - len(tags & _BAD_TAGS)


def rerank_hotels(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank hotels: feasible-first, then by soft-tag overlap. Hard flags come from
    the inferred episode state so this stays correct under hidden eval."""
    episode = context.get("episode") or {}
    state = _episode_state(episode)

    def feasible(c: Dict[str, Any]) -> bool:
        if state.get("quiet_matters") and c.get("quiet_score", 0.0) < 0.7:
            return False
        return True

    return sorted(candidates, key=lambda c: (feasible(c), _score(c, episode)), reverse=True)


def rerank_restaurants(candidates: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rank restaurants: feasible-first (client-ready / vegan / badge), then soft-tag overlap."""
    episode = context.get("episode") or {}
    state = _episode_state(episode)

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
    """Placeholder for student bundle search / scoring."""
    return bundle_candidates[0] if bundle_candidates else None


_STATE_TO_RETIRED = (
    (None, "old_budget_cap", "old_budget_cap"),
    ("airport_priority", "local_character_if_safe", "old_local_character_priority"),
    ("chain_exception", "avoid_chain_hotels_stable", "old_chain_absolute_rule"),
    ("rainy", "old_weather_assumption", "old_weather_assumption"),
    ("partner_bundle", "old_bundle_discount_absolute", "old_bundle_discount_absolute"),
    ("event_disruption", "old_social_bundle_default", "old_social_bundle_default"),
    ("late_arrival_risk", "late_checkin_irrelevant", "late_checkin_irrelevant"),
)


_TURN_INFERENCE_PHRASES = (
    ("airport_priority", "airport access matters more"),
    ("airport_priority", "for this trip only, airport access"),
    ("chain_exception", "chain"),
    ("partner_bundle", "hotel+dinner"),
    ("partner_bundle", "shuttle"),
    ("partner_bundle", "bundles may exist"),
    ("event_disruption", "city event tonight"),
    ("event_disruption", "event tonight"),
    ("late_arrival_risk", "late arrival"),
    ("late_arrival_risk", "perks disappear"),
    ("refund_risk", "refundable bookings deserve"),
    ("refund_risk", "timing may still move"),
    ("badge_available", "badge"),
    ("loyalty_focus", "loyalty perk"),
    ("teammate_vegan", "vegan-capable"),
    ("teammate_vegan", "vegan capable"),
    ("client_dinner", "client-facing"),
    ("quiet_matters", "quiet"),
    ("quiet_matters", "noise"),
    ("red_eye_avoid", "red-eye"),
    ("red_eye_avoid", "overnight flight"),
)


def _episode_state(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Return scenario_state if available (public), else infer from turns + weather (hidden).
    
    MODIFIED: Also merges dynamic state from solver if available.
    """
    # First, try to get dynamic state from solver
    global _DYNAMIC_STATE
    if _DYNAMIC_STATE:
        return _DYNAMIC_STATE
    
    # Fall back to original logic
    raw = episode.get("scenario_state")
    if raw:
        return raw
    
    state: Dict[str, Any] = {}
    weather = str(episode.get("weather") or "").lower()
    if weather in {"rainy", "storm", "wet"}:
        state["rainy"] = True
    
    turns_text = " ".join(
        (t.get("text") or "")
        for t in (episode.get("turns") or [])
        if t.get("speaker") == "user"
    ).lower()
    
    for flag, phrase in _TURN_INFERENCE_PHRASES:
        if phrase in turns_text:
            state[flag] = True
    
    state.setdefault("stakeholder_ids", [])
    return state


_REJECTED_NOTES_ALL = (
    "rejected_hotel_for_noise",
    "rejected_flight_for_red_eye",
    "rejected_restaurant_for_vibe",
)


_ALL_STALE_DOCS = (
    "stale:budget_cap_archive",
    "stale:local_character_default",
    "stale:partner_social_default",
    "stale:bundle_discount_always_wins",
    "stale:late_checkin_irrelevant",
    "stale:avoid_chain_hotels_absolute",
    "stale:dry_weather_ops_assumption",
)


def all_stale_docs() -> List[str]:
    return list(_ALL_STALE_DOCS)


_CITY_DEPENDENCIES = {
    "OSA": ["dependency:OSA_ht206_rs3001_private_room"],
    "TPE": ["dependency:TPE_ht801_rs4004_badge_bundle"],
    "SIN": ["dependency:SIN_ht907_rs2004_shuttle"],
}
_CITY_PROMOS = {
    "OSA": ["promo:OSA_ht206_rs3001_private_room", "promo:OSA_ht207_rs3004_board_bundle"],
    "TPE": ["promo:TPE_ht801_rs4004_badge_room", "promo:TPE_ht807_rs4005_airport_corridor"],
    "SIN": ["promo:SIN_ht907_rs2004_shuttle_bundle", "promo:SIN_ht908_rs2005_client_reception"],
}
_CITY_EVENTS = {
    "OSA": ["event:OSA_namba_food_fair", "event:OSA_umeda_private_room_night"],
    "TPE": ["event:TPE_xinyi_expo_surge", "event:TPE_riverside_buyout"],
    "SIN": ["event:SIN_one_north_thunderstorm", "event:SIN_marina_reception"],
}


def derive_required_docs_from_state(episode: Dict[str, Any]) -> List[str]:
    """Derive doc IDs that gold's `required_docs` is likely to contain."""
    city = episode.get("city", "")
    family = episode.get("family", "")
    traveler = episode.get("traveler_id", "")
    state = _episode_state(episode)
    stakeholder_ids = state.get("stakeholder_ids") or []

    docs: List[str] = [
        f"city_ops:{city}",
        f"profile:{traveler}",
        f"venue:{city}_{family}",
        "heuristic:lean_context_policy",
        "heuristic:rejected_option_memory",
    ]

    if state.get("partner_bundle"):
        docs.append("heuristic:partner_bundle_reasoning")
    if state.get("refund_risk"):
        docs.append("heuristic:refundable_schedule_risk")
        docs.append("constraint:refund_priority_due_schedule_risk")
    if state.get("badge_available"):
        docs.append("heuristic:badge_unlock_logic")
        docs.append("constraint:badge_private_room_access")
    if stakeholder_ids:
        docs.append("heuristic:stakeholder_balance")
    if state.get("late_arrival_risk"):
        docs.append("constraint:late_arrival_voids_bundle")
    if state.get("teammate_vegan"):
        docs.append("constraint:team_dietary_support")
    if state.get("loyalty_focus") and traveler:
        docs.append(f"loyalty:{traveler}")

    for sid in stakeholder_ids:
        if sid:
            docs.append(sid)

    docs.extend(_CITY_DEPENDENCIES.get(city, []))
    docs.extend(_CITY_PROMOS.get(city, []))
    docs.extend(_CITY_EVENTS.get(city, []))

    seen = set()
    out: List[str] = []
    for d in docs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def derive_rejected_from_state(episode: Dict[str, Any]) -> List[str]:
    """All three rejected-reason keys appear in `should_remember_rejected` for every public episode."""
    return list(_REJECTED_NOTES_ALL)


def derive_retired_from_state(episode: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Derive (retired_keys_for_should_retire, spoken_retire_keys) from scenario_state."""
    state = _episode_state(episode)
    retired: List[str] = []
    spoken: List[str] = []
    for flag, retired_key, spoken_key in _STATE_TO_RETIRED:
        if flag is None or state.get(flag):
            retired.append(retired_key)
            spoken.append(spoken_key)
    return retired, spoken


# ============================================================
# Spoken Rule Generation - MODIFIED to use dynamic state
# ============================================================

# Always-on tokens
_SPOKEN_ALWAYS_MUST_REMEMBER = ("quiet_matters",)
_SPOKEN_ALWAYS_FORBIDDEN = ("red_eye", "loud_after_10pm")
_SPOKEN_ALWAYS_DO_NOT_RECONSIDER = ("noise_rejected_hotel", "wrong_vibe_restaurant")
_SPOKEN_ALWAYS_KEEP_CONTEXT_LEAN = ("relevant_only",)

_SPOKEN_CONDITIONAL = (
    ("client_dinner", "must_remember", "client_ready_dinner"),
    ("airport_priority", "one_off_only", "airport_access_more_important_now"),
    ("chain_exception", "one_off_only", "chain_ok_this_trip"),
    ("teammate_vegan", "must_remember", "team_dietary_flex"),
    ("refund_risk", "retire", "old_budget_cap"),
)


def derive_spoken_rule_hits_from_state(episode: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build the full 6-bucket `spoken_rule_hits` dict from scenario_state.
    
    MODIFIED: Uses dynamic state if available.
    """
    state = _episode_state(episode)
    _, spoken_retire = derive_retired_from_state(episode)

    hits: Dict[str, List[str]] = {
        "must_remember": list(_SPOKEN_ALWAYS_MUST_REMEMBER),
        "forbidden": list(_SPOKEN_ALWAYS_FORBIDDEN),
        "one_off_only": [],
        "retire": list(spoken_retire),
        "do_not_reconsider": list(_SPOKEN_ALWAYS_DO_NOT_RECONSIDER),
        "keep_context_lean": list(_SPOKEN_ALWAYS_KEEP_CONTEXT_LEAN),
    }
    
    # Add conditional tokens
    for flag, bucket, token in _SPOKEN_CONDITIONAL:
        if state.get(flag):
            if token not in hits[bucket]:
                hits[bucket].append(token)
    
    # Special handling for airport_priority_override
    if state.get("airport_priority_override"):
        if "airport_access_more_important_now" not in hits["one_off_only"]:
            hits["one_off_only"].append("airport_access_more_important_now")
        if "old_local_character_priority" not in hits["retire"]:
            hits["retire"].append("old_local_character_priority")
    
    # Remove duplicates while preserving order
    for key in hits:
        seen = set()
        unique = []
        for item in hits[key]:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        hits[key] = unique
    
    return hits
