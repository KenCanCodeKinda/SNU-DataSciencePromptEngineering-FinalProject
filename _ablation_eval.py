"""Ablation harness: toggle each framework module and measure score + behavior.

Runs the real deterministic path on the zero-cost lexical session (no API), the
same way _offline_eval.py does, but parameterizes each module so we can quantify
what removing it costs — either in the official /100 score or in the auditable
behavioral diagnostics (evolution events, retrieval coverage, critic findings).

Not part of the submission.
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


# ── parameterized module variants ─────────────────────────────────────────────

def gather_full(session, episode, *, global_scope=True):
    """Memory-Context Manager: surface docs + rejected options via official tools."""
    traveler = episode.get("traveler_id", "") or ""
    city = episode.get("city", "") or ""
    family = episode.get("family", "") or ""
    if traveler:
        try: session.get_profile_brief(traveler)
        except Exception: pass
    if city and family:
        try: session.get_venue_brief(city, family)
        except Exception: pass
    if city:
        try: session.get_city_ops_notes(city)
        except Exception: pass
    for doc_id in S._GATHER_STALE_DOCS + S._GATHER_HEURISTIC_DOCS:
        keywords = doc_id.split(":", 1)[-1].replace("_", " ")
        query = f"{doc_id} {keywords} stale retired no longer"
        try:
            if global_scope:
                try: session.search_memory(query=query, include_stale=True, top_k=8, scope="global")
                except TypeError: session.search_memory(query=query, include_stale=True, top_k=8)
            else:
                session.search_memory(query=query, include_stale=True, top_k=8)
        except Exception: pass
    rejected_query = f"{city} {family} rejected hotel flight restaurant red-eye noise vibe"
    try:
        if global_scope:
            try: session.get_rejected_options(query=rejected_query, max_results=8, scope="global")
            except TypeError: session.get_rejected_options(query=rejected_query, max_results=8)
        else:
            session.get_rejected_options(query=rejected_query, max_results=8)
    except Exception: pass


def naive_select(session, episode):
    """Ablated Verifier: greedy first/cheapest pick with NO constraint filtering."""
    city = episode.get("city", "")
    origin = episode.get("origin", "")

    def items(fn, *a, **k):
        try:
            r = fn(*a, **k)
            if isinstance(r, dict): return list(r.get("items", []) or [])
            if isinstance(r, list): return list(r)
        except Exception:
            return []
        return []

    flights = items(session.search_flights, origin, city, max_results=8) if origin and city else []
    hotels = items(session.search_hotels, city, max_results=8) if city else []
    rests = items(session.search_restaurants, city, max_results=8) if city else []
    acts = items(session.search_activities, city, max_results=8) if city else []
    flight = min(flights, key=lambda f: float(f.get("fare_total", 0))) if flights else None
    return {
        "flight_id": flight.get("flight_id") if flight else None,
        "hotel_id": hotels[0].get("hotel_id") if hotels else None,
        "restaurant_id": rests[0].get("restaurant_id") if rests else None,
        "activity_id": acts[0].get("activity_id") if acts else None,
    }


def run_variant(name, *, gather=True, global_scope=True, retirements=True,
                verifier=True, evolution=True, critic=True, meta=True):
    rows = []
    diag = {"evolution_events": 0, "supersessions": 0, "critic_findings": 0,
            "docs_seen": 0, "rejected_seen": 0, "meta_confidence_sum": 0.0,
            "meta_count": 0}
    for ep in episodes:
        gold = ep["gold"]
        hidden = copy.deepcopy(ep)
        hidden.pop("gold", None)
        hidden.pop("scenario_state", None)
        hidden.pop("scenario_hooks", None)

        # Context-Evolution module (belief state + timeline)
        if evolution:
            evo = S.ContextEvolution(hidden)
            constraints = evo.constraints
            retired_keys = evo.retired_keys if retirements else []
            retired_docs = evo.retired_docs if retirements else []
            diag["evolution_events"] += evo.summary()["evolution_events"]
            diag["supersessions"] += evo.summary()["supersessions"]
        else:
            constraints = S._extract_constraints(hidden)
            rk, rd = S._detect_retirements(hidden)
            retired_keys = rk if retirements else []
            retired_docs = rd if retirements else []

        session = toolbox.new_session(
            episode=hidden, retrieval_strategy="lexical", embedding_model=None,
            max_results=8, role="single_memory",
        )
        session.bind_runner(runner)

        if gather:
            gather_full(session, hidden, global_scope=global_scope)

        rejected_ids = set()
        for note in session.rejected_notes_seen:
            parts = (note or "").split(":")
            if len(parts) >= 2:
                rejected_ids.add(parts[-1])

        if verifier:
            py = S._python_select(session, hidden, constraints, rejected_ids)
        else:
            py = naive_select(session, hidden)

        notes = S._build_notes(hidden, py, constraints, retired_keys)
        submission = {
            "flight_id": py["flight_id"], "hotel_id": py["hotel_id"],
            "restaurant_id": py["restaurant_id"], "activity_id": py["activity_id"],
            "notes": notes, "usage": runner.usage_summary(),
        }
        # memory_report carries forced retirements (drives stale/update credit)
        submission["memory_report"] = S.merge_memory_report(
            {}, session, active_doc_cap=4, active_key_cap=6,
            forced_retired=retired_keys or None,
            forced_retired_docs=retired_docs or None,
        )

        if critic:
            crit = S._critique_bundle(session, hidden, constraints, submission)
            diag["critic_findings"] += len(crit["findings"])
        if meta and evolution:
            mc = S._build_meta_context(evo, constraints, llm_engaged=False,
                                       retrieval_scope="global" if global_scope else "episode")
            diag["meta_confidence_sum"] += mc["interpretation_confidence"]
            diag["meta_count"] += 1

        trace = session.summary()
        diag["docs_seen"] += len(trace.get("docs_seen", []) or [])
        diag["rejected_seen"] += len(trace.get("rejected_memory_seen", []) or [])
        row = EV.evaluate_episode(env, ep, submission, gold, trace)
        rows.append(row)

    summary = EV._summary_bucket_means(rows)
    comp = {}
    for k in ["hard_constraint_rate", "bundle_coherence_rate", "semantic_fit_rate",
              "exactish_rate", "update_handling_rate", "stale_doc_retirement_rate",
              "rejected_option_memory_rate", "efficiency_score"]:
        if k in summary:
            comp[k] = summary[k]
        else:
            vals = [r.get(k) for r in rows if k in r]
            comp[k] = sum(vals) / len(vals) if vals else 0.0
    return {
        "name": name,
        "score": round(summary["official_score_100"], 2),
        "components": comp,
        "diag": diag,
    }


VARIANTS = [
    ("Full system (all 5 modules)", {}),
    ("− Meta-Context Controller", {"meta": False}),
    ("− Context-Evolution module", {"evolution": False}),
    ("− Verifier-Critic self-audit", {"critic": False}),
    ("− Memory-Context Manager (no gather)", {"gather": False}),
    ("− Global retrieval scope only", {"global_scope": False}),
    ("− Retirement detection", {"retirements": False}),
    ("− Verifier floor (naive greedy select)", {"verifier": False}),
]

results = [run_variant(name, **flags) for name, flags in VARIANTS]

print("=== ABLATION (real deterministic path, zero API cost) ===\n")
hdr = f"{'variant':<42} {'score':>7} {'hard':>6} {'upd':>6} {'stale':>6} {'rej':>6}  {'evo':>4} {'crit':>5} {'docs':>5}"
print(hdr)
print("-" * len(hdr))
for r in results:
    c = r["components"]; d = r["diag"]
    print(f"{r['name']:<42} {r['score']:>7.2f} "
          f"{c['hard_constraint_rate']:>6.3f} {c['update_handling_rate']:>6.3f} "
          f"{c['stale_doc_retirement_rate']:>6.3f} {c['rejected_option_memory_rate']:>6.3f}  "
          f"{d['evolution_events']:>4} {d['critic_findings']:>5} {d['docs_seen']:>5}")

print("\n=== full-system component detail ===")
for k, v in results[0]["components"].items():
    print(f"  {k:<32} {v:.4f}")
d0 = results[0]["diag"]
print(f"\n  total evolution_events : {d0['evolution_events']}")
print(f"  total supersessions    : {d0['supersessions']}")
print(f"  total critic_findings  : {d0['critic_findings']}")
print(f"  total docs_seen        : {d0['docs_seen']}")
print(f"  total rejected_seen    : {d0['rejected_seen']}")
if d0["meta_count"]:
    print(f"  mean meta confidence   : {d0['meta_confidence_sum']/d0['meta_count']:.3f}")

# emit machine-readable table for the deck
out = Path("_ablation_results.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}")
