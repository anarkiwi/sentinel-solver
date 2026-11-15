#!/usr/bin/env python3
"""Run a sim player against the REAL game in VICE, recording an AVI, or with ``--emu``
against the CSDb release on vagrant-moses (no AVI).

The live-execution machinery lives in :mod:`driver.live_player`; this is the
shared runner.  ``--player {greedy,phase,search}`` selects ``LiveGreedy``/``LivePhase``
or :class:`driver.search_player.SearchPlayer`, which plans on an emulated session and,
live, plays each move on VICE too (``live_audit``).
"""

import argparse
import json
import os
import time

from driver import core, sentinel_state as gs
from driver.emu import EmuMon
from driver.emu_phase import EmuPhase
from driver.live_player import LiveGreedy, LivePhase
from driver.search_player import PRESETS, SearchPlayer
from sentinel import emu, memmap as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_player(args, session, log, result):
    if args.player == "phase":
        cls = EmuPhase if args.emu else LivePhase
        return cls(session, log, result, ticks_dir=args.ticks)
    return LiveGreedy(session, log, result, ticks_dir=args.ticks)


def emu_session(number, log):
    """A :class:`core.GameSession` in play on the emulated release, entered as
    :func:`core.boot_and_play` enters VICE (no AVI)."""
    landscape = core.landscape_from_digits(f"{number:04d}")
    log(f"EMU replanning mode: landscape {landscape}")
    bm = EmuMon(emu.enter(number))
    entry_match = gs.verify_entry(bm, landscape, log)
    state0 = gs.read_game_state(gs.ViceSource(bm))
    return core.GameSession(bm, landscape, time.time(), entry_match, state0, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "landscape",
        nargs="?",
        type=int,
        default=0,
        help="the landscape number you TYPE",
    )
    parser.add_argument(
        "--player", choices=("greedy", "phase", "search"), default="phase"
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="wide")
    for knob in ("width", "depth", "branch", "commit"):
        parser.add_argument(f"--{knob}", type=int, help=f"search {knob} (--preset)")
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--video", default=None)
    parser.add_argument(
        "--emu",
        action="store_true",
        help="play the CSDb release on vagrant-moses (sentinel.emu) instead of VICE",
    )
    parser.add_argument("--out", default=None, help="result JSON path")
    parser.add_argument(
        "--ticks",
        default=None,
        help="directory to save every observed tick's live image (driver.replay_tick)",
    )
    args = parser.parse_args(argv)
    digits = f"{args.landscape:04d}"
    win = "search" if args.player == "search" else "win"
    video = args.video or f"player_ls{args.landscape}_{win}.avi"
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

    def search_fn(session):
        knobs = {k: getattr(args, k) or v for k, v in PRESETS[args.preset].items()}
        live = None if args.emu else session.bm
        plan = session if args.emu else emu_session(args.landscape, log)
        sp = SearchPlayer(plan.bm, log=log, live=live, **knobs)
        result["search"] = dict(knobs)
        result["trace"], result["lookahead_audit"] = sp.trace, sp.audit
        if live is not None:
            result["live_audit"] = sp.live_audit
        try:
            result["won"] = sp.run(args.max_actions)
        finally:
            result["final_energy"] = sp.trace[-1][3] if sp.trace else None
            result["search"]["stats"] = sp.stats
            if live is not None:
                result["divergence"] = sp.diverged
                with live.halted():
                    done = live.mem_get(mm.LANDSCAPE_COMPLETE, mm.LANDSCAPE_COMPLETE)
                result["landscape_done"] = done[0]
                result["won"] = bool(result.get("won") and done[0] & 0x40)
                agree = sum(r[2] == r[3] for r in sp.live_audit)
                log(f"live audit: {agree}/{len(sp.live_audit)} rows agree")
        log(f"play loop done: won={result['won']} actions={len(sp.trace)}")

    def play_fn(session):
        if args.player == "search":
            search_fn(session)
            return
        lp = _make_player(args, session, log, result)
        won = lp.run(max_actions=args.max_actions, progress=True)
        result["won"] = bool(won and lp.ex.won())
        result["trace"] = [list(r) for r in lp.trace]
        result["final_energy"] = lp.st.energy
        result["pan_passes"] = lp.kbd.passes
        log(f"play loop done: won={result['won']} actions={len(lp.trace)}")

    if args.emu:
        t_start = time.time()
        play_fn(emu_session(args.landscape, log))
        result["wall_seconds"] = round(time.time() - t_start, 1)
        ok, size, frames, msg = True, 0, 0, "emu"
    else:
        core.boot_and_play(tap, renders, digits, video, log, play_fn, result)
        ok, size, frames, msg = core.validate_avi(result.get("video", ""))
    result["avi"] = {"ok": ok, "bytes": size, "frames": frames, "msg": msg}
    suffix = ("_emu" if args.emu else "") + (
        "_search" if args.player == "search" else ""
    )
    out_path = args.out or os.path.join(
        ROOT, "out", f"play_player_{digits}{suffix}.json"
    )
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    log(
        f"RESULT: won={result.get('won')} avi={msg} ({frames} frames, {size} bytes) "
        f"-> {result.get('video')}; log {out_path}"
    )
    return 0 if result.get("won") and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
