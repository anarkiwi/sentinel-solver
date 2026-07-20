#!/usr/bin/env python3
"""Run a sim player against the REAL game in VICE, recording an AVI.

The live-execution machinery lives in :mod:`driver.live_player`; this is the
shared runner.  ``--player {greedy,astar}`` selects ``LiveGreedy``/``LiveAStar``
(``python -m driver.play_player 0`` or ``... 42 --player astar``).
"""

import argparse
import json
import os
import time

from driver import core
from driver.live_player import LiveAStar, LiveGreedy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_player(args, session, log, result):
    if args.player == "astar":
        return LiveAStar(
            session,
            log,
            result,
            node_budget=args.node_budget,
            time_budget=args.time_budget,
            weight=args.weight,
        )
    return LiveGreedy(session, log, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("landscape", nargs="?", type=int, default=0)
    parser.add_argument("--player", choices=("greedy", "astar"), default="greedy")
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--node-budget", type=int, default=200000)
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--weight", type=float, default=1.4)
    parser.add_argument("--video", default=None)
    args = parser.parse_args(argv)
    digits = f"{args.landscape:04d}"
    video = args.video or f"player_ls{args.landscape}_win.avi"
    tap = os.path.join(ROOT, "sentinel-gold.tap")
    renders = os.path.join(ROOT, "renders")
    result = {
        "landscape": args.landscape,
        "player": args.player,
        "actions": [],
        "energy_curve": [],
    }

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def play_fn(session):
        lp = _make_player(args, session, log, result)
        won = lp.run(max_actions=args.max_actions)
        result["won"] = bool(won and lp.ex.won())
        result["trace"] = [list(r) for r in lp.trace]
        result["final_energy"] = lp.st.energy
        log(f"play loop done: won={result['won']} actions={len(lp.trace)}")

    core.boot_and_play(tap, renders, digits, video, log, play_fn, result)
    ok, size, frames, msg = core.validate_avi(result.get("video", ""))
    result["avi"] = {"ok": ok, "bytes": size, "frames": frames, "msg": msg}
    out_path = os.path.join(ROOT, "out", f"play_player_{digits}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    log(
        f"RESULT: won={result.get('won')} avi={msg} ({frames} frames, {size} bytes) "
        f"-> {result.get('video')}; log {out_path}"
    )
    return 0 if result.get("won") and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
