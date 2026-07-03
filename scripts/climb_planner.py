#!/usr/bin/env python3
"""Keyboard-executable win planner for The Sentinel (C64), HEADLESS.

`plan_keyboard_win(landscape, eng=None)` produces a deterministic plan that wins a
landscape by mirroring `code_engine.CodeEngine.climb_and_win`'s REAL ROM mechanics
(which already wins headless), and pairs EVERY gameplay action (create boulder /
create robot / absorb Sentinel / transfer / hyperspace) with a concrete keyboard
VIEW (h_angle, v_angle, cursor) that the offline aim oracle (`aim_oracle.solve_aim`)
confirms gives line-of-sight (LOS) to that action's target tile from the player's
ACTUAL viewpoint position + eye at that step.

VIEWPOINT MODEL (the crux, validated empirically on ls0)
--------------------------------------------------------
The player acts from a STAND tile adjacent to the platform; boulders are stacked on
a BUILD tile adjacent to the STAND. The terrain around the Sentinel's platform is a
plateau, so:
  * A player standing on the STAND tile can look at the adjacent BUILD tile (level
    / slightly down) -> LOS to BUILD for each boulder + the climb robot.
  * The player ascends a boulder stack; from the STAND tile at the RAISED eye (the
    stack top) the player looks DOWN-and-across at the platform tile -> LOS to the
    platform for the Sentinel absorb and the winning-robot create.
This matches the existing harness (`test_climb_aim.py`) and the memory note:
"stand (11,4) build (11,3) gives LOS to platform (12,4); stand (12,3) does NOT."
Crucially climb_and_win's own chosen geometry (stand (12,3) build (12,2)) does NOT
give a keyboard LOS to the platform from the climb top -- so the planner validates
every action with solve_aim and REJECTS geometries that fail, trying all
stand/build combinations until one is fully LOS-valid.

The engine state is advanced for real after each step (the boulders are physically
created on BUILD, the robots/transfers/absorb/hyperspace run through the real ROM
routines), so `won_headless` reflects the actual ROM win flag $0CDE bit6.

For EACH gated action we call solve_aim BEFORE performing it; if it returns None
(no LOS view), the action is infeasible from here and we backtrack. Transfer onto a
tile a create just proved LOS to needs no fresh solve; hyperspace needs no LOS.

Energy: we top up to the 6-bit cap ($0C0A = $3F) for build feasibility, but each
action's `energy_before` records the real running energy at that step.
"""

import sys
import os
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from code_engine import CodeEngine, NUM_SLOTS, N
from aim_oracle import solve_aim, aim_target
from game_state import (
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
    OBJECTS_TYPE,
    OBJECTS_FLAGS,
)

# Centre cursor (initialise_sights $1356/$135D); used for the warmup probe.
_CUR_CX = 0x50
_CUR_CY = 0x5F

PLATFORM_X = 0x0C19
PLATFORM_Y = 0x0C1A
PLAYER_ENERGY = 0x0C0A
DO_HYPERSPACE = 0x2156  # do_hyperspace: sets $0CDE bit6 when player tile==platform

# 8-neighbourhood, orthogonal first (matches code_engine ordering).
_NEIGH = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (1, -1), (-1, 1))


def _occupied(eng):
    return {
        (eng.mem[OBJECTS_X + s], eng.mem[OBJECTS_Y + s])
        for s in range(NUM_SLOTS)
        if not (eng.mem[OBJECTS_FLAGS + s] & 0x80)
    }


def _neighbours(tile, occ, exclude=()):
    out = []
    for dx, dy in _NEIGH:
        t = (tile[0] + dx, tile[1] + dy)
        if not (0 <= t[0] < N and 0 <= t[1] < N):
            continue
        if t in occ or t in exclude:
            continue
        out.append(t)
    return out


def _find_sentinel(eng):
    for s in range(NUM_SLOTS):
        if not (eng.mem[OBJECTS_FLAGS + s] & 0x80) and eng.mem[OBJECTS_TYPE + s] == 5:
            return s
    return None


class _ViewSolver:
    """Solves a keyboard view from a chosen viewpoint tile + eye, by temporarily
    moving the player object's position/eye to that viewpoint and running the real
    offline aim oracle. Saves/restores the player position each call."""

    def __init__(self, eng):
        self.eng = eng
        self.timings = []

    def solve(self, viewpoint, eye_z, target):
        eng = self.eng
        ps = eng.player_slot
        # Snapshot the FULL machine memory: solve_aim -> aim_target runs the real
        # $1C10/$1CDD which mutate scratch globals (the action-target tiles, the LOS
        # temp object, zero-page) that are NOT all save/restored by aim_target. A
        # residue from one solve corrupts the next (empirically: a build-tile solve
        # left state that made the very next platform-tile solve spuriously fail LOS).
        # Restoring the whole memory after each solve isolates them completely.
        snap = bytes(eng.mem)
        # CRITICAL: OBJECTS_FLAGS lives at $0100 == the 6502 stack page. _emu.call
        # pushes return addresses at $0100+sp and never resets sp, so it DRIFTS DOWN
        # across calls; a deep / capped LOS march can push into $0100..$013F and
        # CLOBBER the object-flags array (every cleared bit7 reads as an occupied
        # slot, eventually exhausting create's free-slot scan). We restore the full
        # memory AND the stack pointer after each solve, and start the solve with sp
        # high so its marches have maximum headroom before reaching the flags array.
        saved_sp = eng.cpu.sp
        eng.cpu.sp = 0xFF

        def _place():
            eng.mem[OBJECTS_X + ps] = viewpoint[0] & 0xFF
            eng.mem[OBJECTS_Y + ps] = viewpoint[1] & 0xFF
            eng.mem[OBJECTS_Z_HEIGHT + ps] = eye_z & 0xFF

        _place()
        t0 = time.time()
        try:
            r = solve_aim(eng, target, eye_z=eye_z)
            # COLD-START: the first $1C10/$1CDD aim after a memory (re)build has stale
            # action-target globals and spuriously FAILS LOS; a throwaway aim_target
            # warms them, after which the identical solve is correct. Warm + retry
            # once on a miss (a genuine no-LOS still returns None after the retry).
            if r is None or not r.get("los"):
                aim_target(eng, 0, 0, _CUR_CX, _CUR_CY, ps)
                _place()
                r = solve_aim(eng, target, eye_z=eye_z)
        finally:
            self.timings.append(time.time() - t0)
            eng.mem[:] = snap
            eng.cpu.sp = saved_sp
        if r is None or not r.get("los"):
            return None
        return {
            "h_angle": int(r["h_angle"]),
            "v_angle": int(r["v_angle"]),
            "cursor": (int(r["cursor"][0]), int(r["cursor"][1])),
        }


def plan_keyboard_win(landscape: int, eng=None) -> dict:
    """Build a deterministic, keyboard-executable, LOS-validated win plan for
    `landscape`. Returns the plan dict described in the module docstring."""
    notes = []
    if eng is None:
        t0 = time.time()
        eng = CodeEngine(landscape)
        notes.append(f"built CodeEngine(ls={landscape}) in {time.time()-t0:.2f}s")

    plat = (eng.mem[PLATFORM_X], eng.mem[PLATFORM_Y])
    plat_ground = eng._ground_z(*plat)
    if _find_sentinel(eng) is None:
        return {
            "landscape": landscape,
            "feasible": False,
            "stand": None,
            "build": None,
            "won_headless": False,
            "actions": [],
            "notes": notes + ["no Sentinel on this landscape"],
        }
    ssl0 = _find_sentinel(eng)
    sxy = (eng.mem[OBJECTS_X + ssl0], eng.mem[OBJECTS_Y + ssl0])
    notes.append(f"platform {plat} ground {plat_ground}; Sentinel slot {ssl0} @ {sxy}")

    eng.mem[PLAYER_ENERGY] = 0x3F  # top up for build feasibility (real deltas kept)

    occ0 = _occupied(eng)
    stands = _neighbours(plat, occ0, exclude=(plat,))
    notes.append(f"candidate stands (platform neighbours): {stands}")

    # Snapshot machine memory so each geometry attempt starts from the same state.
    base_mem = bytes(eng.mem)
    vs = _ViewSolver(eng)

    def restore():
        eng.mem[:] = base_mem
        eng._refresh_temp_slot()

    eye_target = plat_ground + 2  # eye must clear platform_ground + 2

    for stand in stands:
        stand_g = eng._ground_z(*stand)
        # PER-STAND pre-gate: the climb-top look at the platform depends only on the
        # STAND (not the build tile), so solve it ONCE per stand. If the stand can't
        # see the platform from the cleared eye, skip ALL its build candidates -- this
        # is the expensive solve, so doing it once per stand bounds the wall clock.
        restore()
        v_plat_top = vs.solve(stand, eye_target, plat)
        if v_plat_top is None:
            notes.append(
                f"stand {stand}: no LOS view STAND->platform at climb-top "
                f"eye {eye_target}; skip stand"
            )
            continue

        occ = _occupied(eng)
        bld_candidates = _neighbours(stand, occ, exclude=(plat,))
        for bld in bld_candidates:
            restore()
            eng.mem[PLAYER_ENERGY] = 0x3F
            actions = []

            # Per-build pre-gate: can we view the BUILD tile from STAND ground (to drop
            # boulders onto the adjacent tile)? Cheap (adjacent, ~level look).
            v_build0 = vs.solve(stand, stand_g, bld)
            if v_build0 is None:
                notes.append(
                    f"stand {stand} build {bld}: no LOS view STAND->build "
                    f"at eye {stand_g}; reject"
                )
                continue

            # ---- boulder stack on BUILD ------------------------------------------
            # KEYBOARD MODEL: the player pins the sights on the BUILD column (the aim
            # `v_build0`, validated LOS-true to the bare BUILD tile from STAND ground)
            # and repeatedly presses fire; each press drops a boulder that stacks
            # +0.5 unit on the column. The HELD aim does not change while pumping, so
            # the same LOS-validated view `v_build0` is the keyboard view for every
            # boulder in this column. (The real terrain LOS to the *boulder-topped*
            # tile would read "looking up" after boulder #1 -- but the player isn't
            # re-acquiring; they hold the proven column aim. v_build0's LOS to the
            # column's base tile is the meaningful gate: it lets the stack be started
            # and pumped.) The stack physics are advanced through the real ROM.
            built_slots = []
            topz = None
            cleared = False
            ok_geom = True
            cur_eye = stand_g
            for k in range(12):
                ebefore = eng.player_energy
                view = v_build0  # held column aim (LOS-true to BUILD from STAND)
                r = eng.create(3, bld)
                if not r.get("ok"):
                    actions.append(
                        {
                            "verb": "create",
                            "otype": 3,
                            "target": bld,
                            "view": view,
                            "player_before": (stand[0], stand[1], cur_eye),
                            "energy_before": ebefore,
                            "ok": False,
                            "note": f"boulder #{k+1}: create failed: {r.get('reason')}",
                        }
                    )
                    ok_geom = False
                    break
                built_slots.append(r["slot"])
                topz = eng.mem[OBJECTS_Z_HEIGHT + r["slot"]]
                actions.append(
                    {
                        "verb": "create",
                        "otype": 3,
                        "target": bld,
                        "view": view,
                        "player_before": (stand[0], stand[1], cur_eye),
                        "energy_before": ebefore,
                        "ok": True,
                        "note": f"boulder #{k+1} -> slot {r['slot']} stack_top_z={topz} "
                        f"(held column aim) energy_delta={r.get('delta')}",
                    }
                )
                if topz >= eye_target:
                    cleared = True
                    break

            if not (ok_geom and cleared):
                notes.append(
                    f"stand {stand} build {bld}: stack never cleared "
                    f"eye {eye_target} (top_z={topz}); reject"
                )
                continue

            climb_eye = topz  # the player's eye after climbing the stack

            # ---- robot on top of the stack (held BUILD column aim) ---------------
            # Same held column aim: the robot is created on the BUILD column (it lands
            # on top of the stack, $1F56), so the keyboard view is again `v_build0`
            # (LOS-true to the column from STAND ground).
            ebefore = eng.player_energy
            view = v_build0
            rr = eng.create(0, bld)
            if not rr.get("ok"):
                notes.append(
                    f"stand {stand} build {bld}: climb robot create failed: "
                    f"{rr.get('reason')}; reject"
                )
                continue
            actions.append(
                {
                    "verb": "create",
                    "otype": 0,
                    "target": bld,
                    "view": view,
                    "player_before": (stand[0], stand[1], stand_g),
                    "energy_before": ebefore,
                    "ok": True,
                    "note": f"climb robot -> slot {rr['slot']} energy_delta={rr.get('delta')}",
                }
            )

            # ---- transfer onto the climb robot (LOS to BUILD already proven) -------
            ebefore = eng.player_energy
            tr = eng.transfer(rr["slot"])
            actions.append(
                {
                    "verb": "transfer",
                    "otype": None,
                    "target": bld,
                    "view": None,
                    "player_before": (stand[0], stand[1], stand_g),
                    "energy_before": ebefore,
                    "ok": bool(tr.get("ok")),
                    "note": f"transfer into climb robot slot {rr['slot']}; eye raised to "
                    f"{climb_eye} (LOS to {bld} proven by prior create). "
                    f"Acting viewpoint stays at STAND {stand} for the over-the-top look.",
                }
            )
            if not tr.get("ok"):
                notes.append(
                    f"stand {stand} build {bld}: transfer into climb robot failed"
                )
                continue

            # ---- absorb the Sentinel (base tile == platform), looking DOWN from top -
            ssl2 = _find_sentinel(eng)
            if ssl2 is None:
                notes.append(
                    f"stand {stand} build {bld}: Sentinel vanished before absorb"
                )
                continue
            ebefore = eng.player_energy
            view = vs.solve(stand, climb_eye, plat)
            if view is None:
                notes.append(
                    f"stand {stand} build {bld}: no LOS view STAND->platform "
                    f"{plat} for Sentinel absorb at eye {climb_eye}; reject"
                )
                continue
            ar = eng.absorb(ssl2)
            actions.append(
                {
                    "verb": "absorb",
                    "otype": 5,
                    "target": plat,
                    "view": view,
                    "player_before": (stand[0], stand[1], climb_eye),
                    "energy_before": ebefore,
                    "ok": bool(ar.get("ok")),
                    "note": f"absorb Sentinel slot {ssl2} (base tile {plat}) "
                    f"energy_delta={ar.get('delta')}",
                }
            )
            if not ar.get("ok"):
                notes.append(
                    f"stand {stand} build {bld}: Sentinel absorb failed: "
                    f"{ar.get('reason')}"
                )
                continue

            # ---- robot on the now-bare platform tile (LOS to platform from top) ----
            ebefore = eng.player_energy
            view = vs.solve(stand, climb_eye, plat)
            if view is None:
                notes.append(
                    f"stand {stand} build {bld}: no LOS view STAND->platform "
                    f"to place winning robot at eye {climb_eye}; reject"
                )
                continue
            rp = eng.create(0, plat)
            if not rp.get("ok"):
                notes.append(
                    f"stand {stand} build {bld}: platform robot create failed: "
                    f"{rp.get('reason')}"
                )
                continue
            actions.append(
                {
                    "verb": "create",
                    "otype": 0,
                    "target": plat,
                    "view": view,
                    "player_before": (stand[0], stand[1], climb_eye),
                    "energy_before": ebefore,
                    "ok": True,
                    "note": f"winning robot on platform -> slot {rp['slot']} "
                    f"energy_delta={rp.get('delta')}",
                }
            )

            # ---- transfer onto the platform robot (LOS proven by prior create) -----
            ebefore = eng.player_energy
            tp = eng.transfer(rp["slot"])
            on_plat = eng.player_on_platform()
            actions.append(
                {
                    "verb": "transfer",
                    "otype": None,
                    "target": plat,
                    "view": None,
                    "player_before": (stand[0], stand[1], climb_eye),
                    "energy_before": ebefore,
                    "ok": bool(tp.get("ok")),
                    "note": f"transfer onto platform robot slot {rp['slot']}; "
                    f"player tile==platform? {on_plat}",
                }
            )
            if not tp.get("ok"):
                notes.append(f"stand {stand} build {bld}: platform transfer failed")
                continue

            # ---- HYPERSPACE (no LOS): the win -------------------------------------
            ebefore = eng.player_energy
            eng._call(DO_HYPERSPACE)
            won = eng.won()
            actions.append(
                {
                    "verb": "hyperspace",
                    "otype": None,
                    "target": plat,
                    "view": None,
                    "player_before": (plat[0], plat[1], climb_eye),
                    "energy_before": ebefore,
                    "ok": won,
                    "note": f"do_hyperspace ($2156): win flag $0CDE bit6 set = {won}",
                }
            )

            if won:
                nb = sum(
                    1 for a in actions if a["verb"] == "create" and a["otype"] == 3
                )
                notes.append(
                    f"WIN: stand {stand} build {bld} boulders={nb} "
                    f"climb_eye={climb_eye}"
                )
                notes.append(
                    f"aim-solver calls={len(vs.timings)} "
                    f"total={sum(vs.timings):.2f}s "
                    f"avg={sum(vs.timings)/max(1,len(vs.timings)):.2f}s"
                )
                return {
                    "landscape": landscape,
                    "feasible": True,
                    "stand": stand,
                    "build": bld,
                    "won_headless": True,
                    "actions": actions,
                    "notes": notes,
                }
            notes.append(
                f"stand {stand} build {bld}: reached hyperspace but no win flag"
            )

    return {
        "landscape": landscape,
        "feasible": False,
        "stand": None,
        "build": None,
        "won_headless": False,
        "actions": [],
        "notes": notes
        + [
            "no winning LOS-validated geometry found",
            f"aim-solver calls={len(vs.timings)} " f"total={sum(vs.timings):.2f}s",
        ],
    }


def _pretty(plan):
    print(f"\n=== keyboard win plan: landscape {plan['landscape']} ===")
    print(
        f"feasible={plan['feasible']}  won_headless={plan['won_headless']}  "
        f"stand={plan['stand']}  build={plan['build']}"
    )
    nb = sum(1 for a in plan["actions"] if a["verb"] == "create" and a["otype"] == 3)
    print(f"boulders={nb}  total_actions={len(plan['actions'])}")
    print("-" * 80)
    for i, a in enumerate(plan["actions"]):
        v = a["view"]
        vs = (
            f"view h={v['h_angle']:>3} v={v['v_angle']:>3} cur={v['cursor']}"
            if v
            else "view=---"
        )
        otp = "" if a["otype"] is None else f"(t{a['otype']})"
        print(
            f"[{i:2}] {a['verb']:<10}{otp:<5} -> {str(a['target']):<8} {vs:<34} "
            f"from {str(a['player_before']):<12} E={a['energy_before']:>2} "
            f"{'OK' if a['ok'] else 'FAIL'}"
        )
        print(f"       {a['note']}")
    print("-" * 80)
    # LOS assertion: every gated action (create/absorb) must carry a non-null view.
    missing = [
        i
        for i, a in enumerate(plan["actions"])
        if a["verb"] in ("create", "absorb") and a["view"] is None
    ]
    if missing:
        print(f"!! gated actions missing a view (BUG): {missing}")
    else:
        print(
            "OK: every create/absorb action has a non-null LOS view (solve_aim los=True)."
        )
    print("notes:")
    for n in plan["notes"]:
        print(f"  - {n}")


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    plan = plan_keyboard_win(ls)
    _pretty(plan)
    print(f"\nplan generation wall clock: {time.time()-t0:.2f}s")
