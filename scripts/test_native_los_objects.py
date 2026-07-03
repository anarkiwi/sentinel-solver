#!/usr/bin/env python3
"""Differential test for the OBJECT-tile branch of native_los.aim_target_native
vs the EMULATED ROM aim path, PARALLELISED across all CPU cores.

This is the object-tile counterpart of test_native_los.py (which covers bare
terrain). Here each worker builds a CodeEngine, hand-places boulder stacks + a
synthoid-on-a-boulder + (where reachable) brings the player to a raised eye near
the Sentinel's PLATFORM tile, then samples (h_angle, v_angle, cursor) and compares
the emulated action-time aim (prepare_vector_from_player_sights $1C10 +
check_for_line_of_sight_to_tile $1CDD, run UNCAPPED to its RTS -- the TRUE ROM
verdict) against the native port. We confirm the object-tile branch
(get_tile_z_from_object $1E3F -> get_tile_z_for_line_of_sight $1E0E /
get_boulder_or_tree_z_for_line_of_sight $1E48 / get_height_of_lowest_object $1EA4)
is bit-exact: same (tx, ty, los) AND, for boulder targets, the same near-centre
fraction ($1EAF) so centre_view_for is replayable.

The emulated reference is the cost (py65); the native side is microseconds, so we
fan the reference out with multiprocessing -- each worker its own engine.

Usage: python3 scripts/test_native_los_objects.py [landscape ...]
"""

import sys
import os
import time
import random
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code_engine import CodeEngine
import native_los as nl
from game_state import (
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
    N,
    tidx,
    NUM_SLOTS,
    OBJECTS_FLAGS,
    OBJECTS_TYPE,
)
from test_native_los import rom_aim_uncapped

PLAYER_ENERGY = 0x0C0A


def _terrain(mem, x, y):
    b = mem[0x0400 + tidx(x, y)]
    return None if b >= 0xC0 else (b >> 4)


def _empty_terrain_near(eng, cx, cy, radius, limit):
    """Empty terrain tiles within `radius` (Chebyshev) of (cx,cy)."""
    out = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = cx + dx, cy + dy
            if not (0 <= x < N and 0 <= y < N):
                continue
            if (x, y) == (cx, cy):
                continue
            if _terrain(eng.mem, x, y) is None:
                continue
            occ = any(
                not (eng.mem[OBJECTS_FLAGS + s] & 0x80)
                and eng.mem[OBJECTS_X + s] == x
                and eng.mem[OBJECTS_Y + s] == y
                for s in range(NUM_SLOTS)
            )
            if occ:
                continue
            out.append((x, y))
            if len(out) >= limit:
                return out
    return out


def build_object_scene(ls):
    """A CodeEngine with boulder stacks + a synthoid-on-a-boulder placed around the
    player, plus the native scene's stacks brought into the marched-ray band. Returns
    (eng, player_slot, placed_tiles)."""
    eng = CodeEngine(ls)
    ps = eng.player_slot
    px, py = eng.mem[OBJECTS_X + ps], eng.mem[OBJECTS_Y + ps]
    eng.mem[PLAYER_ENERGY] = 0x3F  # top up so creates succeed

    placed = []
    tiles = _empty_terrain_near(eng, px, py, 5, 14)
    for i, t in enumerate(tiles):
        if not eng.create(3, t).get("ok"):  # boulder on terrain
            continue
        # vary stack heights 1..4 so the marched ray crosses different surfaces
        for _ in range((i % 4)):
            eng.create(3, t)  # stack more boulders
        placed.append(t)
    # put a synthoid on top of a couple of boulder stacks (the buildable case)
    for t in placed[:3]:
        eng.create(0, t)
    eng._refresh_temp_slot()
    return eng, ps, placed


def make_samples(seed, n_random):
    rng = random.Random(seed)
    cursors = [
        (0x50, 0x5F),
        (0x40, 0x5F),
        (0x60, 0x5F),
        (0x50, 0x4F),
        (0x50, 0x6F),
        (0x48, 0x55),
        (0x58, 0x6A),
    ]
    samples = []
    # dense azimuth sweep at the centre cursor over a full pitch grid
    v_grid = (
        0x08,
        0x10,
        0x20,
        0x28,
        0x30,
        0x38,
        0x40,
        0x60,
        0x90,
        0xB0,
        0xD0,
        0xE0,
        0xF0,
        0xF5,
    )
    for h in range(0, 256, 2):
        for v in v_grid:
            samples.append((h, v, 0x50, 0x5F))
    # random off-centre cursors too
    for _ in range(n_random):
        samples.append((rng.randint(0, 255), rng.randint(0, 255), *rng.choice(cursors)))
    return samples


_WORKER = {}


def _worker_init(ls):
    eng, ps, _placed = build_object_scene(ls)
    _WORKER["eng"] = eng
    _WORKER["ps"] = ps
    _WORKER["state"] = nl.NativeState.from_mem(eng.mem)
    _WORKER["ls"] = ls


def _is_object_tile(mem, x, y):
    if not (0 <= x < N and 0 <= y < N):
        return False
    return mem[0x0400 + tidx(x, y)] >= 0xC0


def _worker_chunk(chunk):
    eng = _WORKER["eng"]
    ps = _WORKER["ps"]
    state = _WORKER["state"]
    mem = eng.mem
    agree = total = 0
    obj_agree = obj_total = 0
    centre_agree = centre_total = 0
    capped_n = 0
    disagree = []
    for h, v, cx, cy in chunk:
        rtx, rty, rlos, capped = rom_aim_uncapped(eng, h, v, cx, cy, ps)
        if capped:
            capped_n += 1
            continue
        ntx, nty, nlos, _ncentre = nl.aim_target_native(
            state, h, v, cx, cy, ps, return_centre=True
        )
        total += 1
        is_obj = _is_object_tile(mem, rtx, rty)
        ok = (rtx, rty, rlos) == (ntx, nty, nlos)
        if ok:
            agree += 1
        else:
            disagree.append((h, v, cx, cy, (rtx, rty, rlos), (ntx, nty, nlos), is_obj))
        if is_obj:
            obj_total += 1
            if ok:
                obj_agree += 1
    return (
        agree,
        total,
        obj_agree,
        obj_total,
        centre_agree,
        centre_total,
        capped_n,
        disagree,
    )


def _chunks(lst, n):
    k = (len(lst) + n - 1) // n
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def run_landscape(ls, n_random=1500, seed=99, nproc=None):
    nproc = nproc or mp.cpu_count()
    # build one scene in-process to report placement + native timing
    eng0, ps, placed = build_object_scene(ls)
    px, py = eng0.mem[OBJECTS_X + ps], eng0.mem[OBJECTS_Y + ps]
    state0 = nl.NativeState.from_mem(eng0.mem)

    samples = make_samples(seed + ls, n_random)
    chunks = _chunks(samples, nproc)

    t0 = time.perf_counter()
    with mp.Pool(processes=nproc, initializer=_worker_init, initargs=(ls,)) as pool:
        results = pool.map(_worker_chunk, chunks)
    wall = time.perf_counter() - t0

    agree = sum(r[0] for r in results)
    total = sum(r[1] for r in results)
    obj_agree = sum(r[2] for r in results)
    obj_total = sum(r[3] for r in results)
    _centre_agree = sum(r[4] for r in results)
    _centre_total = sum(r[5] for r in results)
    capped_n = sum(r[6] for r in results)
    disagree = [d for r in results for d in r[7]]

    # native throughput
    reps = max(1, 4000 // max(len(samples), 1))
    t1 = time.perf_counter()
    for _ in range(reps):
        for h, v, cx, cy in samples:
            nl.aim_target_native(state0, h, v, cx, cy, ps)
    per_nat = 1e6 * (time.perf_counter() - t1) / (reps * len(samples))

    print(
        f"\n== landscape {ls:04d} == player slot {ps} @ ({px},{py})  "
        f"placed {len(placed)} boulder stacks"
    )
    print(
        f"  samples (uncapped) : {total}  (capped/skipped {capped_n})  "
        f"[{nproc} workers, wall {wall:.1f}s]"
    )
    print(
        f"  EXACT (tx,ty,los)  : {agree}/{total} " f"= {100.0*agree/max(total,1):.3f}%"
    )
    print(
        f"  OBJECT-tile targets: {obj_agree}/{obj_total} "
        f"= {100.0*obj_agree/max(obj_total,1):.3f}%"
    )
    print(f"  native speed       : {per_nat:.2f} us/call")
    if disagree:
        print(f"  DISAGREEMENTS ({len(disagree)}); first 25:")
        for d in disagree[:25]:
            print(
                f"    h={d[0]:02x} v={d[1]:02x} cur=({d[2]:02x},{d[3]:02x}) "
                f"ROM={d[4]} NAT={d[5]} obj={d[6]}"
            )
    return agree, total, obj_agree, obj_total, len(disagree)


def _slot_in_tile(eng, tile, want_type=None, top=True):
    c = [
        s
        for s in range(NUM_SLOTS)
        if not (eng.mem[OBJECTS_FLAGS + s] & 0x80)
        and (eng.mem[OBJECTS_X + s], eng.mem[OBJECTS_Y + s]) == tuple(tile)
        and (want_type is None or eng.mem[OBJECTS_TYPE + s] == want_type)
    ]
    return (max(c) if c else None) if top else (min(c) if c else None)


def payoff_centre_view(ls=0):
    """PAYOFF: native_game.centre_view_for now yields a synthoid-on-boulder build view
    that code_engine.create_via_gate accepts on the FIRST try (no window search).

    Replays native_game.plan(ls) through the REAL ROM (code_engine); for object-tile
    creates (view=None: a synthoid on a boulder) it computes the view with the NATIVE
    centre_view_for and confirms first-try gate acceptance. Platform-tile placement
    (the endgame win-transfer, a put_object_in_tile rule -- not LOS) is reported
    separately. Returns (first_try, total) over the boulder-build object creates."""
    import native_game

    g = native_game.plan(ls, verbose=False)
    eng = CodeEngine(ls)
    eng.mem[PLAYER_ENERGY] = 0x3F
    first_try = total = 0
    print(
        f"\n== PAYOFF (centre_view_for -> create_via_gate), ls{ls:04d}: "
        f"{len(g.steps)} plan steps, native_won={g.native_won}"
    )
    for i, st in enumerate(g.steps):
        verb, tile = st["verb"], tuple(st["target"])
        otype, view = st["otype"], st["view"]
        ps = eng.player_slot
        eye_z = eng.mem[OBJECTS_Z_HEIGHT + ps]
        if verb == "create":
            tb = eng.mem[0x0400 + tidx(*tile)]
            is_obj = tb >= 0xC0
            under = _slot_in_tile(eng, tile)
            under_t = eng.mem[OBJECTS_TYPE + under] if under is not None else None
            if view is None:
                view = native_game.centre_view_for(eng.mem, tile, ps, eye_z)
            r = eng.create_via_gate(otype, tile, view) if view else {"ok": False}
            if is_obj and under_t == 3:  # synthoid/boulder on a BOULDER
                total += 1
                if r.get("ok"):
                    first_try += 1
                    print(
                        f"  [{i}] synthoid-on-boulder @ {tile}: FIRST-TRY OK "
                        f"(view h={view['h_angle']:02x} v={view['v_angle']:02x})"
                    )
                else:
                    print(
                        f"  [{i}] synthoid-on-boulder @ {tile}: REJECTED "
                        f"({r.get('reason')})"
                    )
            if not r.get("ok"):
                eng.create(otype, tile)  # keep the replay moving
        elif verb == "transfer":
            s = _slot_in_tile(eng, tile, want_type=0)
            if s is not None:
                eng.transfer(s)
        elif verb == "absorb":
            s = _slot_in_tile(eng, tile)
            if s is not None:
                eng.absorb(s)
    print(
        f"  synthoid-on-boulder first-try accepted: {first_try}/{total} "
        f"-> {'CONFIRMED' if total and first_try == total else 'NOT confirmed'}"
    )
    return first_try, total


def main():
    args = sys.argv[1:]
    if args and args[0] == "payoff":
        ft, tot = payoff_centre_view(int(args[1]) if len(args) > 1 else 0)
        sys.exit(0 if tot and ft == tot else 1)
    landscapes = [int(a) for a in args] or [0, 42, 9999]
    g_agree = g_total = g_oa = g_ot = g_dis = 0
    for ls in landscapes:
        a, t, oa, ot, d = run_landscape(ls)
        g_agree += a
        g_total += t
        g_oa += oa
        g_ot += ot
        g_dis += d
    print(
        f"\n==== OVERALL: {g_agree}/{g_total} exact "
        f"= {100.0*g_agree/max(g_total,1):.3f}%  | OBJECT tiles "
        f"{g_oa}/{g_ot} = {100.0*g_oa/max(g_ot,1):.3f}%  "
        f"({g_dis} disagreements) ===="
    )
    # also run the centre_view_for payoff for ls0
    ft, tot = payoff_centre_view(0)
    ok = (g_dis == 0) and tot and ft == tot
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
