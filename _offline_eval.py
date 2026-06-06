"""Offline harness: mirror the official deterministic grading path.

Strips gold + scenario_state from each public episode (simulating hidden eval),
runs the real student_solver gather + selection on a lexical session (zero API
cost), then scores with the real evaluator against the full gold.  Not part of
the submission.
"""
import json, copy
from pathlib import Path

from dynamic_travel_replanning.rtl_semantic_env import RTLSemanticEnv
from dynamic_travel_replanning import evaluator as EV
from llm_tools import TravelToolbox
import student_solver as S

DATA = Path("dynamic_travel_replanning")
env = RTLSemanticEnv(DATA)
toolbox = TravelToolbox(DATA)
episodes = json.loads((DATA / "episodes_public_example.json").read_text())


class FakeRunner:
    """No-op runner: lexical retrieval never touches the network, so usage is 0."""

    def empty_usage(self):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}

    def combine_usages(self, *usages):
        out = self.empty_usage()
        for u in usages:
            if not u:
                continue
            for k in out:
                out[k] = out[k] + u.get(k, 0)
        return out

    def usage_summary(self):
        return self.empty_usage()

    def trace(self, *a, **k):
        pass


runner = FakeRunner()
rows = []
for ep in episodes:
    gold = ep["gold"]
    hidden = copy.deepcopy(ep)
    hidden.pop("gold", None)
    hidden.pop("scenario_state", None)
    hidden.pop("scenario_hooks", None)

    constraints = S._extract_constraints(hidden)
    retired_keys, retired_docs = S._detect_retirements(hidden)

    session = toolbox.new_session(
        episode=hidden, retrieval_strategy="lexical", embedding_model=None,
        max_results=8, role="single_memory",
    )
    session.bind_runner(runner)
    S._deterministic_gather(session, hidden)

    rejected_ids = set()
    for note in session.rejected_notes_seen:
        parts = (note or "").split(":")
        if len(parts) >= 2:
            rejected_ids.add(parts[-1])

    py = S._python_select(session, hidden, constraints, rejected_ids)
    notes = S._build_notes(hidden, py, constraints, retired_keys)
    submission = {
        "flight_id": py["flight_id"],
        "hotel_id": py["hotel_id"],
        "restaurant_id": py["restaurant_id"],
        "activity_id": py["activity_id"],
        "notes": notes,
        "usage": runner.usage_summary(),  # official path overrides with runner ledger (= 0)
    }
    trace = session.summary()
    row = EV.evaluate_episode(env, ep, submission, gold, trace)
    rows.append(row)

summary = EV._summary_bucket_means(rows)
print("=== OFFLINE (real deterministic path) ===")
print("official_score_100:", round(summary["official_score_100"], 2))
for k in ["hard_constraint_rate", "bundle_coherence_rate", "semantic_fit_rate",
          "exactish_rate", "update_handling_rate", "stale_doc_retirement_rate",
          "rejected_option_memory_rate", "efficiency_score"]:
    vals = [r.get(k) for r in rows if k in r]
    if vals:
        print(f"  mean {k}: {sum(vals)/len(vals):.4f}")

print("\n=== per-episode ===")
for ep, r in zip(episodes, rows):
    flag = "" if r["hard_constraint_rate"] >= 0.999 else "  <-- HARD LOSS"
    print(f"  {ep['trip_id']}: hard={r['hard_constraint_rate']:.3f} "
          f"coh={r['bundle_coherence_rate']:.0f} sem={r['semantic_fit_rate']:.2f} "
          f"exact={r['exactish_rate']:.2f} upd={r['update_handling_rate']:.3f} "
          f"stale={r['stale_doc_retirement_rate']:.2f} rej={r['rejected_option_memory_rate']:.2f}{flag}")
