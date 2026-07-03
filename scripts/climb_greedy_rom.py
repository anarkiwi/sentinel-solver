#!/usr/bin/env python3
"""Greedy height-first climb, EXECUTED ON THE REAL ROM (code_engine) so every move is
ROM-legal by construction -- the native-model version (climb_greedy.py) diverges from
the ROM over far-boulder builds and produces plans the real gate rejects.

Strategy (user's, verbatim): gain height as quickly as possible (MAX height gain per
transfer is the primary key), using the square FURTHEST from the current position,
avoiding the map CENTRE (least enemy-observable), using however many boulders make
sense. Boulders may be built on ANY line-of-sight tile, not just adjacent ones.

Per step: native visibility_sweep on the LIVE ROM state ranks candidates by an estimated
resulting eye (furthest / edge tie-breaks); we then ACTUALLY perform the top candidate
through the real action gate (create_via_gate / transfer), falling back to the next
candidate if the gate refuses -- so the recorded plan always replays through $0CDE.
Energy is recovered by re-absorbing the shell left on the departed tile.
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import code_engine, native_game
import validate_kbd_plan as VKP
from native_game import visibility_sweep, terrain_z, cheb
from game_state import (
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
    OBJECTS_TYPE,
    OBJECTS_FLAGS,
    PLAYER_ENERGY,
)

PLATFORM_X, PLATFORM_Y = 0x0C19, 0x0C1A  # native_game PLAT_X/PLAT_Y

N = 32
CENTRE = (N - 1) / 2.0


def edge_dist(t):
    return abs(t[0] - CENTRE) + abs(t[1] - CENTRE)


def _snap(eng):
    c = eng.cpu
    return (bytes(eng.mem), c.pc, c.a, c.x, c.y, c.sp, c.p)


def _restore(eng, s):
    eng.mem[:] = s[0]
    c = eng.cpu
    c.pc, c.a, c.x, c.y, c.sp, c.p = s[1:]


def _player(eng):
    ps = eng.player_slot
    return (eng.mem[OBJECTS_X + ps], eng.mem[OBJECTS_Y + ps]), eng.mem[
        OBJECTS_Z_HEIGHT + ps
    ]


def _est_eye(eng, T2, use_b):
    """Native estimate of the resulting eye for ranking (the ROM gives the truth)."""
    tz = terrain_z(eng.mem, *T2)
    if tz is None:
        return None
    return tz + (1 if use_b else 0)


def _try_step(eng, T2, use_b):
    """Perform a boulder-step or hop on the REAL ROM. Returns True on success (object(s)
    built on T2 and the player transferred onto the synthoid there), else restores."""
    s = _snap(eng)
    ps = eng.player_slot
    ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
    if use_b:
        vb = VKP.native_view_for(eng, T2) or native_game.centre_view_for(
            eng.mem, T2, ps, ez
        )
        if vb is None or not eng.create_via_gate(3, T2, vb).get("ok"):
            _restore(eng, s)
            return False
    ps = eng.player_slot
    ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
    vs = native_game.centre_view_for(eng.mem, T2, ps, ez) or VKP.native_view_for(
        eng, T2
    )
    # NO emulated _centre_aim_search fallback in the loop -- it runs ~900 emulated builds
    # per call (emulation in the planning loop). The native centre view is bit-exact and
    # succeeds first-try; if it doesn't, just reject this candidate and move on (fast).
    if vs is None or not eng.create_via_gate(0, T2, vs).get("ok"):
        _restore(eng, s)
        return False
    slot = VKP._slot_on_tile(eng, T2, want_type=0)
    if slot is None:
        _restore(eng, s)
        return False
    eng.transfer(slot)
    return True


def _reabsorb_shell(eng, prev_tile):
    """Recover energy: re-absorb the synthoid shell on the departed tile if visible."""
    slot = VKP._slot_on_tile(eng, prev_tile, want_type=0, top=False)
    if slot is None:
        return
    ps = eng.player_slot
    ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
    view = native_game.centre_view_for(eng.mem, prev_tile, ps, ez)
    if view is not None:
        VKP._absorb_via_aim(eng, view)


def plan_greedy_rom(landscape, verbose=True, max_steps=80, topk=8):
    t0 = time.time()
    eng = code_engine.CodeEngine(landscape)
    eng.mem[PLAYER_ENERGY] = 0x3F
    log = lambda *a: verbose and print(*a)
    plat = (eng.mem[PLATFORM_X], eng.mem[PLATFORM_Y])
    pg = terrain_z(eng.mem, *plat)
    plat_ground = pg if pg is not None else 8
    target_z = plat_ground + 1
    sentinel_slot = next(
        (
            s
            for s in range(64)
            if not (eng.mem[OBJECTS_FLAGS + s] & 0x80)
            and eng.mem[OBJECTS_TYPE + s] == 5
        ),
        None,
    )
    steps = []
    cur, eye = _player(eng)
    log(
        f"greedy-ROM ls{landscape}: start {cur} eye {eye} plat {plat} "
        f"plat_ground {plat_ground} target_z {target_z} energy {eng.player_energy}"
    )

    visited = set()
    peak, no_gain = eye, 0
    for step in range(max_steps):
        cur, eye = _player(eng)
        if eye > plat_ground and cheb(cur, plat) <= 1:
            log(f"  reached platform approach: {cur} eye {eye}")
            break
        if eye > peak:
            peak, no_gain = eye, 0
        else:
            no_gain += 1
        if no_gain > 14:
            log(f"  no height progress in 14 steps (peak {peak}); stop")
            break

        sweep = visibility_sweep(eng.mem, eng.player_slot, eye, max_steps=320)
        cands = []
        for T2 in sweep:
            if T2 == cur or terrain_z(eng.mem, *T2) is None:
                continue
            for use_b in (True, False):
                e = _est_eye(eng, T2, use_b)
                if e is not None:
                    cands.append((T2, use_b, e))
        # THE STRATEGY: prefer a strict height GAIN (max gain, then furthest, then edge).
        # If none is available HERE, REPOSITION to the furthest unvisited square at the
        # same height (a lateral boulder-step) -- "use the squares furthest from me" --
        # so a new, higher foothold comes into view from there (this is what climbs out
        # of a same-terrain basin; requiring an immediate gain stalls).
        gain = [c for c in cands if c[2] > eye]
        if gain:
            ranked = sorted(
                gain,
                key=lambda c: (c[2], cheb(c[0], cur), edge_dist(c[0])),
                reverse=True,
            )
        else:
            reposition = [
                c for c in cands if c[2] >= eye and (c[0], c[1]) not in visited
            ]
            ranked = sorted(
                reposition,
                key=lambda c: (cheb(c[0], cur), edge_dist(c[0]), c[2]),
                reverse=True,
            )

        moved = False
        for T2, use_b, _est in ranked[:topk]:
            if (T2, use_b) in visited:
                continue
            prev = cur
            if _try_step(eng, T2, use_b):
                visited.add((T2, use_b))
                steps.append(
                    {"verb": "create", "otype": 3 if use_b else 0, "target": list(T2)}
                )
                if use_b:
                    steps.append({"verb": "create", "otype": 0, "target": list(T2)})
                steps.append({"verb": "transfer", "otype": None, "target": list(T2)})
                _reabsorb_shell(eng, prev)
                cur, eye = _player(eng)
                log(
                    f"  [{step:2}] {'step' if use_b else 'hop '} -> {T2} eye {eye} "
                    f"(d={cheb(cur, plat)}) edge={edge_dist(T2):.0f} energy {eng.player_energy}"
                )
                moved = True
                break
        if not moved:
            log(
                f"  step {step}: no ROM-legal height move from {cur} eye {eye} "
                f"(d={cheb(cur, plat)}); stop"
            )
            break

    # endgame: absorb the Sentinel, synthoid on the platform, hyperspace (win).
    won = eng.won()
    cur, eye = _player(eng)
    if (
        not won
        and eye > plat_ground
        and cheb(cur, plat) <= 1
        and sentinel_slot is not None
    ):
        ps = eng.player_slot
        ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
        sv = native_game.centre_view_for(eng.mem, plat, ps, ez)
        if sv is not None and VKP._absorb_via_aim(eng, sv).get("ok"):
            steps.append({"verb": "absorb", "otype": 5, "target": list(plat)})
            log(f"  absorbed Sentinel, energy {eng.player_energy}")
        ps = eng.player_slot
        ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
        pv = VKP.native_view_for(eng, plat) or native_game.centre_view_for(
            eng.mem, plat, ps, ez
        )
        if pv is not None and eng.create_via_gate(0, plat, pv).get("ok"):
            slot = VKP._slot_on_tile(eng, plat, want_type=0)
            if slot is not None:
                eng.transfer(slot)
                steps.append({"verb": "create", "otype": 0, "target": list(plat)})
                steps.append({"verb": "transfer", "otype": None, "target": list(plat)})
    if not eng.won():
        eng.mem[0x0C61] = 0x22
        eng.mem[0x006E] = eng.player_slot
        eng._call(0x1B18)
    won = eng.won()
    log(
        f"=== greedy-ROM {'WON' if won else 'INCOMPLETE'} in {time.time()-t0:.2f}s, "
        f"{len(steps)} steps, $0CDE={won}, energy {eng.player_energy} ==="
    )
    return won, steps


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    won, steps = plan_greedy_rom(ls)
    json.dump(
        {"landscape": ls, "won": won, "steps": steps},
        open(f"out/kbd_greedy_{ls:04d}.json", "w"),
        indent=0,
    )
    print(f"FINAL won={won} steps={len(steps)}")
    sys.exit(0 if won else 1)
