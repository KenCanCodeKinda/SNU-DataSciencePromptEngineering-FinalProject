"""Tail an llm_run_trace.jsonl and show a tqdm progress bar.

Usage: python watch_progress.py runs/<dir> [--total 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm


def count_event(path: Path, name: str) -> int:
    n = 0
    with path.open() as f:
        for line in f:
            try:
                if json.loads(line).get("event") == name:
                    n += 1
            except Exception:
                pass
    return n


def run_finished(path: Path) -> bool:
    with path.open() as f:
        for line in f:
            try:
                if json.loads(line).get("event") == "run_finish":
                    return True
            except Exception:
                pass
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="runs/<name> directory containing llm_run_trace.jsonl")
    ap.add_argument("--total", type=int, default=20, help="Expected episode count (default 20)")
    ap.add_argument("--poll", type=float, default=2.0, help="Polling interval seconds")
    args = ap.parse_args()

    trace = Path(args.run_dir) / "llm_run_trace.jsonl"
    while not trace.exists():
        time.sleep(args.poll)

    with tqdm(total=args.total, unit="ep", desc=Path(args.run_dir).name) as bar:
        finished = 0
        while True:
            success = count_event(trace, "episode_success")
            failure = count_event(trace, "episode_failure")
            done = success + failure
            if done > finished:
                bar.update(done - finished)
                finished = done
            bar.set_postfix(ok=success, fail=failure)
            if run_finished(trace) or finished >= args.total:
                break
            time.sleep(args.poll)
        bar.update(args.total - finished)


if __name__ == "__main__":
    sys.exit(main())
