# Dynamic Travel Replanning — Student Starter Pack

This release gives you everything you need to develop on the **public** benchmark:
- public episodes
- deterministic simulator data
- transparent evaluator
- official runtime wrapper for model usage / cost logging
- a blank submission file and a working baseline example

## What you should edit

Start with these files:
- `student_solver.py` — your submission entrypoint
- `student_solver_example.py` — working baseline example
- `run_student.py` — simplest way to run your solver locally
- `STUDENT_EVALUATION.md` — what gets scored
- `STUDENT_TOOLING.md` — what tools you may use and extend

You do **not** need to edit the simulator data files.

## Recommended first run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env
# add OPENAI_API_KEY to .env
cp llm_eval_config.json llm_eval_config_student.json
# modify your configs

python run_student.py --solver student_solver_example --limit-public 2 --output-dir runs/example_smoke
```

Then implement `solve_episode(runtime)` in `student_solver.py` and try:

```bash
python run_student.py --solver student_solver --limit-public 3 --output-dir runs/my_solver_smoke
```

## Advanced usage

If you want more control after the basic workflow works, you can still call `run_llm_baselines.py` directly. Most students should start with `run_student.py`.

```bash
python run_llm_baselines.py \
  --config llm_eval_config_student.json \
  --systems student_solver \
  --skip-hidden \
  --skip-ablations \
  --output-dir runs/my_solver_public
```

## Budget knobs

The assignment expects you to tune a bounded search / reasoning budget. Use `--set` to change allowed knobs.

```bash
python run_student.py \
  --solver student_solver \
  --set student_solver.max_tool_rounds=12 \
  --set student_solver.max_output_tokens=1200
```

Hard caps remain enforced by the official config.

## Submission

Submit your `student_solver.py`, `llm_eval_config_student.json` plus any helper Python modules you import from it. Staff will rerun your code with the official runtime wrapper on hidden episodes.



##logic：
## 🎯 底层逻辑和流程说明

### 整体架构

```
solve_episode(runtime) 
    ↓
TravelPlanner.recommend(episode)
    ↓
1. 获取所有选项 → 2. 场景过滤 → 3. 生成组合 → 4. 获取上下文 → 5. 提取需求 → 6. Bundle过滤 → 7. 评分排名 → 8. 输出结果
```

---

## 📋 详细流程

### 第1步：获取所有可用选项 (`fetch_all_options`)

```python
all_options = {
    "flights": search_flights(origin, city),      # 所有航班
    "hotels": search_hotels(city),                # 所有酒店
    "restaurants": search_restaurants(city),      # 所有餐厅
    "activities": search_activities(city)         # 所有活动
}
```

**显示给用户**：打印所有选项，包括价格、时间、安静分、客户分等属性。

---

### 第2步：根据 scenario_state 过滤 (`filter_by_scenario_state`)

根据 `episode["scenario_state"]` 中的字段过滤选项：

| 字段 | 作用 |
|------|------|
| `meeting_zone` | 只保留会议区的酒店/餐厅/活动 |
| `budget` | 只保留预算内的选项 |
| `airport_priority` | 酒店按机场访问分排序 |
| `teammate_vegan` | 只保留有素食的餐厅 |
| `rainy` | 只保留室内活动 |

**输出**：过滤后的选项列表（减少组合数量）。

---

### 第3步：生成所有组合 (`generate_combinations`)

```python
combinations = product(flights, hotels, restaurants, activities)
# 例如: 7×3×2×2 = 84 个组合
```

每个组合是一个元组：`(flight_id, hotel_id, restaurant_id, activity_id)`

---

### 第4步：获取上下文信息 (`fetch_all_context_info`)

**通过工具调用**获取额外信息，用于评分：

| 工具 | 获取的信息 |
|------|-----------|
| `get_city_ops_notes` | 城市运营信息 |
| `get_profile_brief` | 用户档案 |
| `get_venue_brief` | 场地信息 |
| `get_partner_promotions` | 促销/bundle |
| `get_option_dependencies` | 依赖关系 |
| `get_loyalty_profile` | 忠诚度信息 |
| `get_booking_constraints` | 预订约束 |
| `get_event_context` | 事件信息 |
| `get_stakeholder_brief` | 利益相关者信息 |
| `get_rejected_options` | 被拒绝的选项 |
| `search_memory` | stale 文档、启发式规则 |

**输出**：`context_info` 包含所有获取的信息和 `docs_retrieved`。

---

### 第5步：提取用户需求 (`extract_user_requirements`)

从 `turns` 对话中提取用户偏好：

```python
requirements = {
    "need_quiet": "quiet" in all_text,
    "need_refund": "refund" in all_text or state.get('refund_risk'),
    "need_vegan": "vegan" in all_text or state.get('teammate_vegan'),
    "need_client_ready": "client" in all_text or "polished" in all_text,
    "need_airport": "airport" in all_text or state.get('airport_priority'),
    "rainy": "rainy" in all_text or state.get('rainy') or weather == 'rainy',
    "no_red_eye": "red-eye" in all_text,
    "meeting_zone": episode.get('meeting_zone'),
    "budget": episode.get('budget_total'),
    "nights": episode.get('nights'),
}
```

---

### 第6步：Bundle 过滤 (`filter_by_bundle`)

如果有 bundle 促销或依赖关系，**只保留符合 bundle 的组合**：

```python
# 从 promotions 和 dependencies 提取有效 bundle
valid_bundles = {(HT907, RS2004, None), (HT907, None, ACT304), ...}

# 只保留匹配这些 bundle 的组合
filtered = [combo for combo in combinations if matches_bundle(combo)]
```

---

### 第7步：评分排名 (`select_best_combination`)

对每个预算内的组合进行**详细评分**：

#### 评分维度（总分 ~100+）

| 类别 | 权重 | 评分项 |
|------|------|--------|
| **航班** | 40分 | 非红眼(+10)、可退款(+25)、早班机(+15)、直飞(+10)、价格便宜(+15) |
| **酒店** | 35分 | 安静分(×2.5)、会议区(+15)、机场分(×1.2) |
| **餐厅** | 25分 | 安静分、素食(+20)、客户分(×1.5)、会议区(+5) |
| **活动** | 20分 | 室内(+15)、会议区(+8)、免费(+5) |
| **Bundle** | 加分 | 餐厅 bundle (+30)、活动 bundle (+25)、依赖关系 (+15) |
| **忠诚度** | 加分 | private_room(+10)、shuttle(+10)、late_checkout(+5) |
| **预算** | 加分 | 预算利用率高(+10)、总花费低(+5) |

**排序**：按得分降序，得分相同按花费升序（越便宜越好）。

---

### 第8步：构建 memory_report (`build_memory_report_from_context`)

构建评估器需要的 memory_report，包含：

| 字段 | 作用 | 影响指标 |
|------|------|----------|
| `docs_retrieved` | 检索的文档 ID | distributed_context |
| `retired` | 退休的键 | spoken_rule |
| `retired_docs` | 退休的 stale 文档 | **stale_doc_retirement** |
| `active_context_keys` | 活跃的上下文 | distributed_context |
| `spoken_rule_hits` | 口语规则 | **spoken_rule** |
| `rejected_option_notes` | 被拒绝的选项 | distributed_context |

#### spoken_rule_hits 的 6 个 bucket：

| Bucket | 说明 | 示例 |
|--------|------|------|
| `must_remember` | 必须记住的规则 | `quiet_matters`, `client_ready_dinner` |
| `forbidden` | 禁止的规则 | `red_eye`, `loud_after_10pm` |
| `one_off_only` | 本次例外 | `airport_access_more_important_now`, `chain_ok_this_trip` |
| `retire` | 退休的旧规则 | `old_budget_cap`, `local_character_if_safe` |
| `do_not_reconsider` | 不再考虑 | `noise_rejected_hotel`, `wrong_vibe_restaurant` |
| `keep_context_lean` | 保持简洁 | `relevant_only` |

---

### 第9步：输出最终推荐

```python
submission = {
    "flight_id": best_flight,
    "hotel_id": best_hotel,
    "restaurant_id": best_restaurant,
    "activity_id": best_activity,
    "memory_report": {...},
    "notes": "..."
}
```

返回给评估器。

---

## 📊 指标影响对照表

| 指标 | 受哪些字段影响 | 当前值 | 目标 |
|------|---------------|--------|------|
| **decision_quality** | 选择的组合是否正确 | 0.6733 | 0.8+ |
| **distributed_context** | `docs_retrieved`, `active_context_keys` | 0.5385 | 0.7+ |
| **stale_doc_retirement** | `retired_docs` | **1.0000** ✅ | 满分 |
| **distractor_avoidance** | `ignored_distractors` | **1.0000** ✅ | 满分 |
| **spoken_rule** | `spoken_rule_hits` | 0.1667 | 0.5+ |

---

## 🔧 关键函数映射

| 函数 | 文件 | 作用 |
|------|------|------|
| `fetch_all_context_info` | student_custom_tools_template.py | 工具调用获取上下文 |
| `extract_spoken_rules_from_turns` | student_custom_tools_template.py | 从对话提取口语规则 |
| `build_memory_report_from_context` | student_custom_tools_template.py | 构建 memory_report |
| `score_combination` | student_solver.py | 评分单个组合 |
| `select_best_combination` | student_solver.py | 排序选最佳 |

这就是完整的底层逻辑和流程！
