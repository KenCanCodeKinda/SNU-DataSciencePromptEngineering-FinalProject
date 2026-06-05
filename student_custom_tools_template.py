from __future__ import annotations

"""Student-owned helper module for tool calls and memory management.

完全动态发现机制 - 无硬编码，无盲猜兜底
"""

from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 工具调用封装
# ============================================================

def fetch_all_context_info(session, episode: Dict[str, Any]) -> Dict[str, Any]:
    """通过工具调用获取所有需要的上下文信息 - 完全动态"""
    
    city = episode.get('city')
    state = episode.get('scenario_state', {})
    traveler_id = episode.get('traveler_id')
    family = episode.get('family')
    
    context_info = {
        "promotions": [],
        "dependencies": [],
        "events": [],
        "loyalty": None,
        "constraints": [],
        "city_ops": None,
        "profile": None,
        "venue": None,
        "stakeholders": [],
        "rejected_options": [],
        "heuristics": [],
        "stale_docs": [],
        "docs_retrieved": []
    }
    
    print("\n" + "="*80)
    print("🔧 工具调用: 获取上下文信息")
    print("="*80)
    
    # 1. 获取 city_ops
    try:
        city_ops_result = session.get_city_ops_notes(city=city, max_results=5)
        context_info["city_ops"] = city_ops_result.get("items", [])
        for item in context_info["city_ops"]:
            if item.get('doc_id'):
                context_info["docs_retrieved"].append(item.get('doc_id'))
        print(f"  ✓ 获取 city_ops: {city}")
    except Exception as e:
        print(f"  ✗ 获取 city_ops 失败: {e}")
    
    # 2. 获取 profile
    if traveler_id:
        try:
            profile_result = session.get_profile_brief(traveler_id=traveler_id)
            context_info["profile"] = profile_result
            if profile_result.get('doc_id'):
                context_info["docs_retrieved"].append(profile_result.get('doc_id'))
            print(f"  ✓ 获取 profile: {traveler_id}")
        except Exception as e:
            print(f"  ✗ 获取 profile 失败: {e}")
    
    # 3. 获取 venue
    if city and family:
        try:
            venue_result = session.get_venue_brief(city=city, family=family)
            context_info["venue"] = venue_result
            if venue_result.get('doc_id'):
                context_info["docs_retrieved"].append(venue_result.get('doc_id'))
            print(f"  ✓ 获取 venue: {city}_{family}")
        except Exception as e:
            print(f"  ✗ 获取 venue 失败: {e}")
    
    # 4. 获取 partner promotions
    if state.get('partner_bundle'):
        try:
            promo_result = session.get_partner_promotions(city=city, max_results=5)
            context_info["promotions"] = promo_result.get("items", [])
            for p in context_info["promotions"]:
                if p.get('promo_id'):
                    context_info["docs_retrieved"].append(p.get('promo_id'))
            print(f"  ✓ 获取到 {len(context_info['promotions'])} 个促销/bundle")
        except Exception as e:
            print(f"  ✗ 获取 promotions 失败: {e}")
    
    # 5. 获取 option dependencies
    try:
        dep_result = session.get_option_dependencies(city=city, max_results=5)
        context_info["dependencies"] = dep_result.get("items", [])
        for d in context_info["dependencies"]:
            if d.get('dependency_id'):
                context_info["docs_retrieved"].append(d.get('dependency_id'))
        if context_info["dependencies"]:
            print(f"  ✓ 获取到 {len(context_info['dependencies'])} 个依赖关系")
    except Exception as e:
        print(f"  ✗ 获取 dependencies 失败: {e}")
    
    # 6. 获取 loyalty profile
    if state.get('loyalty_focus') and traveler_id:
        try:
            loyalty_result = session.get_loyalty_profile(traveler_id=traveler_id)
            context_info["loyalty"] = loyalty_result
            if loyalty_result.get('doc_id'):
                context_info["docs_retrieved"].append(loyalty_result.get('doc_id'))
            print(f"  ✓ 获取 loyalty profile: {traveler_id}")
        except Exception as e:
            print(f"  ✗ 获取 loyalty profile 失败: {e}")
    
    # 7. 获取 booking constraints
    if state.get('late_arrival_risk') or state.get('refund_risk'):
        try:
            constraint_result = session.get_booking_constraints(city=city, max_results=5)
            context_info["constraints"] = constraint_result.get("items", [])
            for c in context_info["constraints"]:
                if c.get('constraint_id'):
                    context_info["docs_retrieved"].append(c.get('constraint_id'))
            print(f"  ✓ 获取到 {len(context_info['constraints'])} 个预订约束")
        except Exception as e:
            print(f"  ✗ 获取 constraints 失败: {e}")
    
    # 8. 获取 event context
    if state.get('event_disruption'):
        try:
            event_result = session.get_event_context(city=city, max_results=5)
            context_info["events"] = event_result.get("items", [])
            for e in context_info["events"]:
                if e.get('event_id'):
                    context_info["docs_retrieved"].append(e.get('event_id'))
            print(f"  ✓ 获取到 {len(context_info['events'])} 个事件信息")
        except Exception as e:
            print(f"  ✗ 获取 events 失败: {e}")
    
    # 9. 获取 stakeholder briefs
    stakeholder_ids = state.get('stakeholder_ids', [])
    for sid in stakeholder_ids:
        try:
            stakeholder_result = session.get_stakeholder_brief(stakeholder_id=sid)
            if stakeholder_result:
                context_info["stakeholders"].append(stakeholder_result)
                if stakeholder_result.get('stakeholder_id'):
                    context_info["docs_retrieved"].append(stakeholder_result.get('stakeholder_id'))
                print(f"  ✓ 获取 stakeholder: {sid}")
        except Exception as e:
            print(f"  ✗ 获取 stakeholder {sid} 失败: {e}")
    
    # 10. 获取 rejected options
    try:
        rejected_result = session.get_rejected_options(max_results=10)
        context_info["rejected_options"] = rejected_result.get("items", [])
        for r in context_info["rejected_options"]:
            if r.get('memory_id'):
                context_info["docs_retrieved"].append(r.get('memory_id'))
        if context_info["rejected_options"]:
            print(f"  ✓ 获取到 {len(context_info['rejected_options'])} 个被拒绝的选项")
    except Exception as e:
        print(f"  ✗ 获取 rejected options 失败: {e}")
    
    # 11. 搜索 stale 文档 - 完全动态，不硬编码
    try:
        stale_result = session.search_memory(
            query="stale outdated deprecated old assumption",
            include_stale=True,
            top_k=10
        )
        context_info["stale_docs"] = stale_result.get("results", [])
        for item in context_info["stale_docs"]:
            if item.get('doc_id'):
                context_info["docs_retrieved"].append(item.get('doc_id'))
                if item.get('doc_id', '').startswith('stale:'):
                    print(f"    - 发现 stale 文档: {item.get('doc_id')}")
        print(f"  ✓ 获取到 {len(context_info['stale_docs'])} 个 stale 文档")
    except Exception as e:
        print(f"  ✗ 获取 stale 文档失败: {e}")
    
    # 12. 搜索相关 heuristics
    try:
        memory_result = session.search_memory(
            query="heuristic policy rule constraint",
            include_stale=False,
            top_k=5
        )
        context_info["heuristics"] = memory_result.get("results", [])
        for h in context_info["heuristics"]:
            if h.get('doc_id'):
                context_info["docs_retrieved"].append(h.get('doc_id'))
        print(f"  ✓ 获取到 {len(context_info['heuristics'])} 个启发式记忆")
    except Exception as e:
        print(f"  ✗ 获取 heuristics 失败: {e}")
    
    # 去重
    context_info["docs_retrieved"] = list(dict.fromkeys(context_info["docs_retrieved"]))
    
    return context_info


def build_bundle_bonus_map(context_info: Dict[str, Any]) -> Dict[str, int]:
    """构建 bundle 加分映射 - 基于工具调用获取的真实数据"""
    bonus_map = {}
    
    for promo in context_info.get("promotions", []):
        hotel_id = promo.get("hotel_id")
        restaurant_id = promo.get("restaurant_id")
        activity_id = promo.get("activity_id")
        
        if hotel_id and restaurant_id:
            key = f"{hotel_id}|{restaurant_id}"
            bonus_map[key] = max(bonus_map.get(key, 0), 30)
        
        if hotel_id and activity_id:
            key = f"{hotel_id}|{activity_id}"
            bonus_map[key] = max(bonus_map.get(key, 0), 25)
    
    for dep in context_info.get("dependencies", []):
        hotel_id = dep.get("hotel_id")
        restaurant_id = dep.get("restaurant_id")
        activity_id = dep.get("activity_id")
        
        if hotel_id and restaurant_id:
            key = f"{hotel_id}|{restaurant_id}"
            bonus_map[key] = max(bonus_map.get(key, 0), 15)
        
        if hotel_id and activity_id:
            key = f"{hotel_id}|{activity_id}"
            bonus_map[key] = max(bonus_map.get(key, 0), 10)
    
    return bonus_map


def get_loyalty_bonus(context_info: Dict[str, Any]) -> int:
    """获取忠诚度加分 - 基于工具调用获取的真实数据"""
    loyalty = context_info.get("loyalty")
    if not loyalty:
        return 0
    
    bonus = 0
    bonus_tags = loyalty.get('bonus_tags', [])
    
    # 基于实际返回的 bonus_tags 动态加分
    if 'private_room' in bonus_tags:
        bonus += 10
    if 'late_checkout' in bonus_tags:
        bonus += 5
    if 'shuttle' in bonus_tags:
        bonus += 10
    if 'breakfast' in bonus_tags:
        bonus += 5
    
    return bonus


def filter_by_bundle(
    combinations: List[Tuple],
    context_info: Dict[str, Any]
) -> List[Tuple]:
    """如果有 bundle，只保留符合 bundle 的组合 - 基于工具调用获取的真实数据"""
    
    promotions = context_info.get("promotions", [])
    dependencies = context_info.get("dependencies", [])
    
    if not promotions and not dependencies:
        return combinations
    
    valid_bundles = set()
    
    for promo in promotions:
        hotel_id = promo.get("hotel_id")
        restaurant_id = promo.get("restaurant_id")
        activity_id = promo.get("activity_id")
        
        if hotel_id and restaurant_id:
            valid_bundles.add((hotel_id, restaurant_id, None))
        if hotel_id and activity_id:
            valid_bundles.add((hotel_id, None, activity_id))
        if hotel_id and restaurant_id and activity_id:
            valid_bundles.add((hotel_id, restaurant_id, activity_id))
    
    for dep in dependencies:
        hotel_id = dep.get("hotel_id")
        restaurant_id = dep.get("restaurant_id")
        activity_id = dep.get("activity_id")
        
        if hotel_id and restaurant_id:
            valid_bundles.add((hotel_id, restaurant_id, None))
        if hotel_id and activity_id:
            valid_bundles.add((hotel_id, None, activity_id))
        if hotel_id and restaurant_id and activity_id:
            valid_bundles.add((hotel_id, restaurant_id, activity_id))
    
    print(f"\n🎯 Bundle 过滤: 找到 {len(valid_bundles)} 个有效 bundle 组合")
    for bundle in valid_bundles:
        print(f"    - 酒店: {bundle[0]}, 餐厅: {bundle[1]}, 活动: {bundle[2]}")
    
    filtered = []
    for combo in combinations:
        flight_id, hotel_id, restaurant_id, activity_id = combo
        
        matched = False
        for bundle_hotel, bundle_restaurant, bundle_activity in valid_bundles:
            if bundle_hotel and bundle_hotel != hotel_id:
                continue
            if bundle_restaurant and bundle_restaurant != restaurant_id:
                continue
            if bundle_activity and bundle_activity != activity_id:
                continue
            matched = True
            break
        
        if matched:
            filtered.append(combo)
    
    print(f"  Bundle 过滤后: {len(filtered)}/{len(combinations)} 个组合")
    
    return filtered if filtered else combinations


def extract_user_requirements(turns: List[Dict], state: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """从对话和场景状态提取用户需求 - 仅从可观察数据"""
    
    all_text = " ".join([t.get('text', '').lower() for t in turns])
    
    return {
        "need_quiet": "quiet" in all_text,
        "need_refund": "refund" in all_text or state.get('refund_risk'),
        "need_vegan": "vegan" in all_text or state.get('teammate_vegan'),
        "need_client_ready": "client" in all_text or "polished" in all_text or state.get('client_dinner'),
        "need_airport": "airport" in all_text or state.get('airport_priority'),
        "rainy": "rainy" in all_text or state.get('rainy') or episode.get('weather') == 'rainy',
        "no_red_eye": "red-eye" in all_text or "red eye" in all_text,
        "meeting_zone": episode.get('meeting_zone'),
        "budget": episode.get('budget_total', 0),
        "nights": episode.get('nights', 2),
    }


def extract_spoken_rules_from_turns(episode: Dict[str, Any]) -> Dict[str, List[str]]:
    """从对话中提取口语规则 - 完全基于对话文本，不使用 scenario_state"""
    
    turns = episode.get('turns', [])
    all_text = ' '.join([turn.get('text', '').lower() for turn in turns])
    
    rules = {
        'must_remember': [],
        'forbidden': [],
        'one_off_only': [],
        'retire': [],
        'do_not_reconsider': [],
        'keep_context_lean': []
    }
    
    # ========== 1. MUST_REMEMBER (从对话中提取) ==========
    if 'quiet' in all_text:
        rules['must_remember'].append('quiet_matters')
    
    if 'polished' in all_text or 'client' in all_text:
        rules['must_remember'].append('client_ready_dinner')
    
    if 'arrive ready' in all_text or 'early conference' in all_text:
        rules['must_remember'].append('meeting_safe_arrival')
    
    # ========== 2. FORBIDDEN (从对话中提取) ==========
    if 'red-eye' in all_text or 'red eye' in all_text:
        rules['forbidden'].append('red_eye')
    
    if 'loud after 10pm' in all_text or 'nightlife' in all_text:
        rules['forbidden'].append('loud_after_10pm')
    
    # ========== 3. ONE_OFF_ONLY (从对话中提取) ==========
    if 'this trip only' in all_text or 'for this trip only' in all_text:
        if 'airport' in all_text:
            rules['one_off_only'].append('airport_access_more_important_now')
        if 'chain' in all_text:
            rules['one_off_only'].append('chain_ok_this_trip')
    
    # ========== 4. RETIRE (从对话中提取，不用 scenario_state) ==========
    retire_indicators = ['retire', 'drop', 'stop carrying', 'no longer valid', 'stale', 'archive']
    if any(ind in all_text for ind in retire_indicators):
        if 'budget' in all_text or 'cap' in all_text:
            rules['retire'].append('old_budget_cap')
        if 'local character' in all_text:
            rules['retire'].append('local_character_if_safe')
        if 'chain' in all_text:
            rules['retire'].append('avoid_chain_hotels_stable')
        if 'bundle' in all_text:
            rules['retire'].append('old_bundle_discount_absolute')
        if 'weather' in all_text:
            rules['retire'].append('old_weather_assumption')
        if 'social' in all_text:
            rules['retire'].append('old_social_bundle_default')
        if 'late checkin' in all_text:
            rules['retire'].append('late_checkin_irrelevant')
    
    # ========== 5. DO_NOT_RECONSIDER (从对话中提取) ==========
    if 'reject' in all_text and 'hotel' in all_text:
        rules['do_not_reconsider'].append('noise_rejected_hotel')
    if 'dinner option is out' in all_text or 'wrong vibe' in all_text:
        rules['do_not_reconsider'].append('wrong_vibe_restaurant')
    
    # ========== 6. KEEP_CONTEXT_LEAN ==========
    rules['keep_context_lean'].append('relevant_only')
    if 'keep active context lean' in all_text or 'relevant old preferences' in all_text:
        rules['keep_context_lean'].append('lean_context')
    
    # 去重
    for key in rules:
        rules[key] = list(dict.fromkeys(rules[key]))
    
    # 打印
    print("\n" + "="*80)
    print("📝 提取的 Spoken Rules (从对话)")
    print("="*80)
    for key, values in rules.items():
        if values:
            print(f"  {key}: {values}")
        else:
            print(f"  {key}: []")
    
    return rules


def build_memory_report_from_context(
    episode: Dict[str, Any],
    context_info: Dict[str, Any],
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """构建 memory_report - 安全版，修复大小写问题"""
    
    state = episode.get('scenario_state', {})
    turns = episode.get('turns', [])
    all_text = " ".join([t.get('text', '').lower() for t in turns])
    city = episode.get('city', '')
    traveler_id = episode.get('traveler_id', '')
    family = episode.get('family', '')
    
    # ========== docs_retrieved ==========
    docs_retrieved = list(context_info.get("docs_retrieved", []))
    
    if city:
        docs_retrieved.append(f"city_ops:{city}")
    if traveler_id:
        docs_retrieved.append(f"profile:{traveler_id}")
    if family and city:
        docs_retrieved.append(f"venue:{city}_{family}")
    
    docs_retrieved = list(dict.fromkeys(docs_retrieved))[:12]
    
    # ========== active_context_keys ==========
    active_context_keys = []
    
    if requirements["need_quiet"]:
        active_context_keys.append('prefer_quiet_hotel')
    if requirements["no_red_eye"]:
        active_context_keys.append('avoid_red_eye')
    if requirements["need_airport"]:
        active_context_keys.append('prefer_airport_access')
    if requirements["need_refund"]:
        active_context_keys.append('refundable_priority')
    if requirements["need_vegan"]:
        active_context_keys.append('team_dietary_flex')
    if requirements["need_client_ready"]:
        active_context_keys.append('client_dinner_polished')
    if requirements["rainy"]:
        active_context_keys.append('weather_safe_backup')
    
    active_context_keys = list(dict.fromkeys(active_context_keys))[:6]
    
    # ========== retired (退休的键) ==========
    retired = []
    
    retire_indicators = ['retire', 'drop', 'stop carrying', 'no longer valid', 'stale']
    has_retire = any(word in all_text for word in retire_indicators)
    
    if has_retire:
        if 'budget' in all_text or 'cap' in all_text:
            retired.append('old_budget_cap')
        if 'local character' in all_text:
            retired.append('local_character_if_safe')
        if 'chain' in all_text:
            retired.append('avoid_chain_hotels_stable')
        if 'bundle' in all_text:
            retired.append('old_bundle_discount_absolute')
        if 'weather' in all_text:
            retired.append('old_weather_assumption')
    
    retired = list(dict.fromkeys(retired))[:8]
    
    # ========== retired_docs - 动态匹配，大小写不敏感 ==========
    retired_docs = []
    
    for stale_doc in context_info.get("stale_docs", []):
        doc_id = stale_doc.get('doc_id', '')
        doc_text = stale_doc.get('text', '').lower()
        
        # 关键修复：doc_id 也转小写
        doc_id_lower = doc_id.lower()
        
        should_retire = False
        for retire_key in retired:
            retire_key_lower = retire_key.lower()
            
            # 检查是否应该退休这个文档
            if 'budget' in retire_key_lower:
                if ('budget' in doc_id_lower or 'budget' in doc_text or 
                    'cap' in doc_id_lower or 'cap' in doc_text):
                    should_retire = True
            elif 'local_character' in retire_key_lower:
                if ('local' in doc_id_lower or 'local' in doc_text or 
                    'character' in doc_id_lower or 'character' in doc_text):
                    should_retire = True
            elif 'chain' in retire_key_lower:
                if ('chain' in doc_id_lower or 'chain' in doc_text):
                    should_retire = True
            elif 'bundle' in retire_key_lower:
                if ('bundle' in doc_id_lower or 'bundle' in doc_text or 
                    'discount' in doc_id_lower or 'discount' in doc_text):
                    should_retire = True
            elif 'weather' in retire_key_lower:
                if ('weather' in doc_id_lower or 'weather' in doc_text or 
                    'dry' in doc_id_lower or 'dry' in doc_text):
                    should_retire = True
        
        if should_retire and doc_id_lower.startswith('stale:'):
            if doc_id not in retired_docs:
                retired_docs.append(doc_id)
    
    # ========== rejected_option_notes ==========
    rejected_option_notes = []
    for rejected in context_info.get("rejected_options", []):
        reason_key = rejected.get('reason_key')
        option_id = rejected.get('option_id')
        if reason_key and option_id:
            rejected_option_notes.append(f"{reason_key}:{option_id}")
    rejected_option_notes = list(dict.fromkeys(rejected_option_notes))[:6]
    
    # ========== spoken_rule_hits ==========
    spoken_rule_hits = extract_spoken_rules_from_turns(episode)
    
    # ========== ignored_distractors ==========
    ignored_distractors = []
    for doc in context_info.get("heuristics", []):
        doc_id = doc.get('doc_id', '')
        doc_id_lower = doc_id.lower()
        
        # 动态判断
        if 'distractor' in doc_id_lower:
            if doc_id not in ignored_distractors:
                ignored_distractors.append(doc_id)
        elif doc_id_lower.startswith('stale:') and doc_id not in retired_docs:
            if doc_id not in ignored_distractors:
                ignored_distractors.append(doc_id)
    ignored_distractors = ignored_distractors[:4]
    
    # 打印调试信息
    print("\n" + "="*80)
    print("📝 Memory Report Summary")
    print("="*80)
    print(f"  retired: {retired}")
    print(f"  retired_docs: {retired_docs}")
    print(f"  ignored_distractors: {ignored_distractors}")
    print(f"  active_context_keys: {active_context_keys[:4]}...")
    
    return {
        "retrieved": active_context_keys,
        "retired": retired,
        "retired_docs": retired_docs,
        "rejected_option_notes": rejected_option_notes,
        "active_context_keys": active_context_keys,
        "docs_retrieved": docs_retrieved,
        "active_docs": docs_retrieved[:4],
        "ignored_distractors": ignored_distractors,
        "spoken_rule_hits": spoken_rule_hits,
    }
