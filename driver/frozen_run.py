#!/usr/bin/env python3
"""Fidelity discriminator: run a live player with the enemy AI frozen.

RTS-stubs update_enemies ($16B5) in live VICE memory after landscape entry, so no
enemy rotates, senses or drains. A win when frozen but a loss when live isolates
the residual to frame-cost fidelity rather than search/energy management.
"""

import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from driver import core  # noqa: E402
from driver.play_player import _make_player  # noqa: E402
from sentinel import enemies  # noqa: E402

UPDATE_ENEMIES = 0x16B5
RTS = 0x60


def freeze_sim_enemies():
    """No-op the sim's enemy advance so the model matches the RTS-stubbed live game.

    Freezing only the live side leaves the planner still predicting rotations, so it
    gates on a threat that cannot occur; both sides must be frozen for the run to
    isolate search capability from enemy-phase fidelity.
    """
    enemies.advance_frames = lambda *a, **k: None
    enemies.advance_frame = lambda *a, **k: None
    enemies.step = lambda *a, **k: None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("landscape", type=int)
    p.add_argument("--player", choices=("greedy", "astar"), default="greedy")
    p.add_argument("--max-actions", type=int, default=120)
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--node-budget", type=int, default=200000)
    p.add_argument("--time-budget", type=float, default=30.0)
    p.add_argument("--weight", type=float, default=1.4)
    args = p.parse_args(argv)

    digits = f"{args.landscape:04d}"
    tag = "live" if args.no_freeze else "frozen"
    result = {
        "landscape": args.landscape,
        "player": args.player,
        "frozen": not args.no_freeze,
        "actions": [],
        "energy_curve": [],
    }

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def play_fn(session):
        bm = session.bm
        if not args.no_freeze:
            with bm.halted():
                before = bm.mem_get(UPDATE_ENEMIES, UPDATE_ENEMIES)[0]
                bm.mem_set(UPDATE_ENEMIES, bytes([RTS]))
                after = bm.mem_get(UPDATE_ENEMIES, UPDATE_ENEMIES)[0]
            log(f"FREEZE ${UPDATE_ENEMIES:04X}: {before:02X} -> {after:02X}")
            if after != RTS:
                raise RuntimeError("freeze poke did not stick")
            freeze_sim_enemies()
            log("FREEZE sim: enemies.advance_frames/advance_frame/step -> no-op")
        lp = _make_player(args, session, log, result)
        won = lp.run(max_actions=args.max_actions)
        result["won"] = bool(won and lp.ex.won())
        result["final_energy"] = lp.st.energy
        result["trace"] = [list(r) for r in lp.trace]
        log(f"play loop done: won={result['won']} actions={len(lp.trace)}")

    os.environ["NO_RECORD"] = "1"
    core.boot_and_play(
        os.path.join(ROOT, "sentinel-gold.tap"),
        os.path.join(ROOT, "renders"),
        digits,
        f"frozen_ls{args.landscape}.avi",
        log,
        play_fn,
        result,
    )
    out = os.path.join(ROOT, "out", f"{tag}_{args.player}_{digits}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    log(
        f"RESULT {tag} {args.player} ls{digits}: won={result.get('won')} "
        f"energy={result.get('final_energy')} actions={len(result.get('trace') or [])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
