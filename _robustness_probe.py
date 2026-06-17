"""Generalization probe (zero API cost).

The deterministic path scores a perfect 100 on the 20 public episodes — but those
are the episodes the heuristics were hand-tuned on, so 100 is an overfit ceiling,
not evidence of hidden-set robustness. This probe applies realistic *rephrasings*
of the rtl7 register to the user turns (the hidden set is the same family but will
not use identical wording), re-runs the real deterministic path with gold +
scenario_state stripped (mirroring hidden eval), and reports the official /100.

A drop under a scheme = a place where extraction is keyed on a specific surface
form. Caveat: the rephrasings are author-chosen, so "passes the probe" is a
robustness signal, not a confirmed hidden-set gain. Not part of the submission.
"""
import json, copy, re
from pathlib import Path

from dynamic_travel_replanning.rtl_semantic_env import RTLSemanticEnv
from dynamic_travel_replanning import evaluator as EV
from llm_tools import TravelToolbox
import student_solver as S

DATA = Path("dynamic_travel_replanning")
env = RTLSemanticEnv(DATA)
toolbox = TravelToolbox(DATA)
episodes = json.loads((DATA / "episodes_public_example.json").read_text())


def _sub(text, pairs):
    for a, b in pairs:
        text = re.sub(a, b, text, flags=re.IGNORECASE)
    return text


# Realistic same-register rephrasings a hidden episode might use.
SCHEMES = {
    "baseline": [],
    "vegan->plant_based": [(r"\bvegan\b", "plant-based")],
    "redeye->overnight": [(r"red-eye", "overnight flight"), (r"red eye", "overnight flight")],
    "quiet->low_noise": [(r"\bquiet\b", "low-noise")],
    "all_combined": [
        (r"\bvegan\b", "plant-based"),
        (r"red-eye", "overnight flight"),
        (r"red eye", "overnight flight"),
        (r"\bquiet\b", "low-noise"),
    ],
}


class FakeRunner:
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


def run_scheme(pairs):
    rows = []
    for ep in episodes:
        gold = ep["gold"]
        hidden = copy.deepcopy(ep)
        hidden.pop("gold", None)
        hidden.pop("scenario_state", None)
        hidden.pop("scenario_hooks", None)
        for t in hidden.get("turns", []):
            t["text"] = _sub(t.get("text", ""), pairs)

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
            "flight_id": py["flight_id"], "hotel_id": py["hotel_id"],
            "restaurant_id": py["restaurant_id"], "activity_id": py["activity_id"],
            "notes": notes, "usage": runner.usage_summary(),
        }
        trace = session.summary()
        rows.append(EV.evaluate_episode(env, ep, submission, gold, trace))
    summary = EV._summary_bucket_means(rows)
    hard = sum(r["hard_constraint_rate"] for r in rows) / len(rows)
    losses = [ep["trip_id"] for ep, r in zip(episodes, rows) if r["hard_constraint_rate"] < 0.999]
    return summary["official_score_100"], hard, losses


print("=== ROBUSTNESS PROBE (deterministic path, gold+scenario_state stripped) ===")
print(f"{'scheme':22} {'score/100':>10} {'hard_rate':>10}  hard_losses")
for name, pairs in SCHEMES.items():
    score, hard, losses = run_scheme(pairs)
    print(f"{name:22} {score:>10.2f} {hard:>10.4f}  {losses}")
