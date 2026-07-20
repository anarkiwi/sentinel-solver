#!/usr/bin/env python3
"""Replay a recorded HUMAN win line in VICE, capturing TRUE enemy phase per step.

Before each recorded action fires, reads live memory for every enemy's facing +
cooldowns (the ground truth the fixtures omit), then drives the human's recorded view
and fires via ``perform_step``.  Emits derived, non-copyrighted ``<fixture>_truth.json``.
"""

import argparse
import json
import os
import time

from driver import core, sentinel_execute as sx
from driver.live_player import MeasuringKbdDriver, drive_transfer_aim
from sentinel import enemies, los, memmap as mm
from sentinel.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIX_DIR = os.path.join(ROOT, "sentinel", "tests", "fixtures", "human_wins")


def _signed(b):
    return b - 256 if b >= 128 else b


def _enemy_truth(mem):
    """Per living enemy: facing + rotation/drain/update cooldowns, read from a live
    64 KB image (ROTATION_SPEED_TABLE $9D37 lives above the 4 KB play region)."""
    st = State.from_mem(mem)
    out = []
    for e in enemies.enemy_slots(st):
        out.append(
            {
                "slot": int(e),
                "type": mm.TYPES[st.obj_type[e]],
                "tile": [int(st.obj_x[e]), int(st.obj_y[e])],
                "h_angle": int(mem[mm.OBJECTS_H_ANGLE + e]),
                "v_angle": int(mem[mm.OBJECTS_V_ANGLE + e]),
                "rot_step": _signed(mem[mm.ROTATION_SPEED_TABLE + e]),
                "rot_cooldown": int(mem[mm.ENEMIES_ROTATION_COOLDOWN + e]),
                "drain_cooldown": int(mem[mm.ENEMIES_DRAINING_COOLDOWN + e]),
                "update_cooldown": int(mem[mm.ENEMIES_UPDATE_COOLDOWN + e]),
            }
        )
    return out


def _player_truth(mem):
    """Player stance/energy read straight from the live object arrays."""
    ps = mem[mm.PLAYER_OBJECT]
    return {
        "slot": int(ps),
        "x": int(mem[mm.OBJECTS_X + ps]),
        "y": int(mem[mm.OBJECTS_Y + ps]),
        "z": int(mem[mm.OBJECTS_Z_HEIGHT + ps]),
        "zf": int(mem[mm.OBJECTS_Z_FRACTION + ps]),
        "hang": int(mem[mm.OBJECTS_H_ANGLE + ps]),
        "vang": int(mem[mm.OBJECTS_V_ANGLE + ps]),
        "energy": int(mem[mm.PLAYER_ENERGY] & 0x3F),
    }


def _snap_view(mem, tile):
    """A keyboard view (lattice h/v + cursor) landing on ``tile`` with LOS, from the
    sim's full-band oracle over the live image -- reproduces the human's aim on the
    SAME tile without the recorded raw ``h_angle`` (a transient off-lattice read)."""
    return los.landable_view_targeted(State.from_mem(mem), tuple(tile))


def replay(session, events, log, result):
    """Replay the recorded human events in order, capturing true enemy phase per
    step.  Divergent steps are logged and skipped; the trace is the deliverable."""
    bm = session.bm
    bm.auto_resume = False
    ex = sx.Executor(bm, log)
    kbd = MeasuringKbdDriver(bm, log, quantized=True, ex=ex)
    steps = []
    reproduced = 0
    diverged_since = False

    for i, ev in enumerate(events):
        verb, otype = ev["verb"], ev["otype"]
        tile = tuple(ev["target"])
        pl = ev["player"]
        recorded = {
            "hang": pl["hang"],
            "vang": pl["vang"],
            "cursor": list(ev["cursor"]),
        }

        with bm.halted():
            mem = core.live_image(bm)
        view = _snap_view(mem, tile)
        rec = {
            "i": i,
            "verb": verb,
            "otype": otype,
            "otype_name": mm.TYPES.get(otype, f"?{otype}"),
            "target": list(tile),
            "recorded_view": recorded,
            "snapped_view": view,
            "player": _player_truth(mem),
            "enemies": _enemy_truth(mem),
        }

        stp = {"verb": verb, "otype": otype, "target": list(tile), "view": view}
        try:
            if view is None:
                outcome = "no_view"
            elif verb == "transfer" and not drive_transfer_aim(kbd, tile, view, log):
                outcome = "aim_miss"
            else:
                outcome = sx.perform_step(ex, kbd, f"h{i}", stp, log, result)
        except Exception as e:  # a monitor drop mid-step: record and continue
            outcome = f"error:{type(e).__name__}"
            log(f"[h{i}] {verb} {tile}: replay error {e}")
            core.reconnect(bm, log)

        ok = outcome in ("ok", "diverge", "best_effort_miss")
        if not ok:
            diverged_since = True
        rec["replay"] = {
            "outcome": outcome,
            "matched_recording": ok,
            "diverged_since": diverged_since,
        }
        reproduced += int(ok)
        steps.append(rec)
        log(
            f"[h{i}] {verb:8} {tile} otype={otype}: outcome={outcome} "
            f"(reproduced {reproduced}/{i + 1})"
        )
        with bm.halted():
            won = bm.mem_get(mm.LANDSCAPE_COMPLETE, mm.LANDSCAPE_COMPLETE)[0] & 0x40
        if won:
            log(f"[h{i}] landscape WON (bit6 set); replay complete")
            result["won_at_step"] = i
            break

    result["truth_steps"] = steps
    result["reproduced"] = reproduced
    result["n_events"] = len(events)
    return steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture", nargs="?", default="ls42.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    fixture_path = os.path.join(_FIX_DIR, args.fixture)
    with open(fixture_path, encoding="utf-8") as fh:
        data = json.load(fh)
    events = data["events"]
    entered_code = data["entered_code"]
    seed = data["landscape"]
    digits = f"{entered_code:04d}"
    out_path = args.out or os.path.join(_FIX_DIR, args.fixture[:-5] + "_truth.json")

    result = {
        "fixture": args.fixture,
        "landscape": seed,
        "entered_code": entered_code,
        "source": "live asid-vice replay of the recorded human ACTION line",
        "actions": [],
        "energy_curve": [],
    }

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def play_fn(session):
        log(f"entry match vs generate({seed}): {session.entry_match}")
        replay(session, events, log, result)

    tap = os.path.join(ROOT, "sentinel-gold.tap")
    renders = os.path.join(ROOT, "renders")
    os.environ.setdefault("NO_RECORD", "1")  # the phase trace is the goal, not a video
    core.boot_and_play(
        tap, renders, digits, f"replay_{args.fixture}.avi", log, play_fn, result
    )

    truth = {
        "fixture": args.fixture,
        "landscape": seed,
        "entered_code": entered_code,
        "source": result["source"],
        "reproduced": result.get("reproduced", 0),
        "n_events": result.get("n_events", len(events)),
        "won_at_step": result.get("won_at_step"),
        "divergence": result.get("divergence"),
        "steps": result.get("truth_steps", []),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1)
    log(
        f"TRUTH: reproduced {truth['reproduced']}/{truth['n_events']} steps "
        f"-> {out_path} (divergence={truth['divergence']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
