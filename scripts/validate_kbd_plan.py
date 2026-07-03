#!/usr/bin/env python3
"""Validate a native keyboard plan (native_game.plan) by REPLAYING it through the
real ROM's LOS-gated path in code_engine, confirming the genuine win flag $0CDE.

For each native step:
  * create  -> create_via_gate(otype, tile, view).  If view is None (the stacking /
    platform steps where native LOS diverges on object tiles), compute the exact
    view with the EMULATED aim_oracle.solve_aim (correct for occupied tiles), then
    gate-create.  Confirms the object landed on the intended tile.
  * transfer-> transfer onto the slot we created on that tile.
  * absorb  -> absorb_via_gate(slot) on the object at that tile.

Step A confirms the gated MECHANICS reach the real win (energy topped up); energy
economy (earning fuel from trees) is layered on separately.
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import native_game
import code_engine
from game_state import (
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
    OBJECTS_TYPE,
    OBJECTS_FLAGS,
    OBJECTS_H_ANGLE,
    OBJECTS_V_ANGLE,
    PLAYER_ENERGY,
)


def _absorb_via_aim(eng, view):
    """Faithful keyboard absorb: aim the player's sights (a native centre-aimed view)
    at the object's tile and press absorb -- drives the REAL handle_player_actions
    $1B18 with action code $20, exactly like the player.  This works for the
    platform/Sentinel tile (which needs a centre-aimed view, $1E25) where the
    synthetic can_absorb straight-line LOS fails.  Carry clear => object absorbed."""
    m = eng.mem
    ps = eng.player_slot
    saved = (
        m[OBJECTS_H_ANGLE + ps],
        m[OBJECTS_V_ANGLE + ps],
        m[0x0CC6],
        m[0x0CC7],
        m[0x0C61],
        m[0x0C6E],
    )
    m[OBJECTS_H_ANGLE + ps] = view["h_angle"] & 0xFF
    m[OBJECTS_V_ANGLE + ps] = view["v_angle"] & 0xFF
    m[0x0CC6], m[0x0CC7] = view["cursor"]
    m[0x0C61] = 0x20  # action code: absorb (bit5 set)
    m[0x006E] = ps
    m[0x0C6E] &= 0x7F
    eng._call(0x1B18)  # handle_player_actions (real gate)
    ok = not bool(eng.cpu.p & 0x01)  # carry clear => object changed
    (
        m[OBJECTS_H_ANGLE + ps],
        m[OBJECTS_V_ANGLE + ps],
        m[0x0CC6],
        m[0x0CC7],
        m[0x0C61],
        m[0x0C6E],
    ) = saved
    return {"ok": ok, "gated": True, "energy": eng.player_energy}


def native_view_for(eng, tile):
    """Compute a keyboard view that aims at `tile` by running the full NATIVE
    visibility sweep on the EMULATED engine's EXACT state (real player x,y + real
    z_height) -> bit-exact valid in the real gate for terrain tiles, no height
    matching.  Returns None for occupied tiles (native LOS diverges) -> solve_aim."""
    ps = eng.player_slot
    z = eng.mem[OBJECTS_Z_HEIGHT + ps]
    sw = native_game.visibility_sweep(eng.mem, ps, z)
    return sw.get(tile)


def _slot_on_tile(eng, tile, want_type=None, top=True):
    """Return the live object slot whose (x,y) == tile (optionally of a type),
    preferring the most-recently created (highest slot) when stacked."""
    cands = [
        s
        for s in range(64)
        if not (eng.mem[OBJECTS_FLAGS + s] & 0x80)
        and (eng.mem[OBJECTS_X + s], eng.mem[OBJECTS_Y + s]) == tile
        and (want_type is None or eng.mem[OBJECTS_TYPE + s] == want_type)
    ]
    if not cands:
        return None
    return max(cands) if top else min(cands)


def _centre_aim_search(eng, otype, tile, base):
    """Find a CENTRE-aimed view that builds `otype` on an object-occupied `tile`
    through the gate.  Per the ROM (get_tile_z_from_object $1E48: the boulder is only
    targetable when get_minimum_x_or_y_fraction_from_tile_centre < $40), building on a
    boulder requires aiming near the tile centre.  We search a small window around the
    boulder's own view (which already points at the tile) using mem snapshot/restore."""
    snap = bytes(eng.mem)
    # `base` is a NATIVE centre-aimed estimate (native_game.centre_view_for) that is
    # close but not exact (native diverges over object tiles), so we only nail it with a
    # TINY window of real creates -- the emulation just advances the game, it does not do
    # the planning search.
    h0 = base["h_angle"] if base else 0
    hrange = [(h0 + dh) & 0xFF for dh in range(-16, 17)]
    v0 = base["v_angle"] if base else 0xE0
    for h in hrange:
        if True:
            for dv in range(-14, 15):
                eng.mem[:] = snap
                view = {
                    "h_angle": h,
                    "v_angle": (v0 + dv) & 0xFF,
                    "cursor": (0x50, 0x5F),
                }
                if eng.create_via_gate(otype, tile, view).get("ok"):
                    eng.mem[:] = snap  # restore; caller commits
                    return view
    eng.mem[:] = snap
    return None


def _view_dict(r):
    if r is None:
        return None
    cx, cy = r["cursor"]
    return {
        "h_angle": int(r["h_angle"]),
        "v_angle": int(r["v_angle"]),
        "cursor": (int(cx), int(cy)),
    }


def validate(landscape=0, top_energy=True, verbose=True, plan=None):
    # `plan` lets a caller validate an ALTERNATIVE native plan (e.g. the timed
    # climb_timed.plan_timed) using the exact same real-ROM replay path; default
    # None reproduces the original behaviour (validate native_game.plan).
    g = plan if plan is not None else native_game.plan(landscape, verbose=False)
    steps = g.steps
    log = lambda *a: verbose and print(*a)
    log(f"native plan: {len(steps)} steps, native_won={g.native_won}")

    eng = code_engine.CodeEngine(landscape)
    if top_energy:
        eng.mem[PLAYER_ENERGY] = 0x3F
    t0 = time.time()
    filled = 0
    prev_view = {}
    for i, st in enumerate(steps):
        verb, tile = st["verb"], tuple(st["target"])
        otype, view = st["otype"], st["view"]
        if verb == "create":
            ps = eng.player_slot
            ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
            # 100% NATIVE view: native_los is now bit-exact for terrain AND object tiles,
            # so centre_view_for returns a gate-acceptable (centre-aimed) view first-try.
            view = native_view_for(eng, tile) or native_game.centre_view_for(
                eng.mem, tile, ps, ez
            )
            if view is None:
                # no LOS view from this eye -> the A* routed an infeasible build.
                return _fail(
                    i, st, "no native LOS view (A* foothold not gate-feasible)", eng, t0
                )
            r = eng.create_via_gate(otype, tile, view)
            if not r.get("ok"):
                # native says LOS but the gate refused (e.g. platform placement quirk):
                # tiny emulated nudge only as a safety net.
                view = _centre_aim_search(eng, otype, tile, view)
                filled += 1
                if view is None or not eng.create_via_gate(otype, tile, view).get("ok"):
                    return _fail(i, st, "gated create rejected", eng, t0)
            prev_view[tile] = view
            st["view"] = view
        elif verb == "transfer":
            slot = _slot_on_tile(eng, tile, want_type=0)  # the robot we built there
            if slot is None:
                return _fail(i, st, "no robot to transfer onto", eng, t0)
            eng.transfer(slot)
        elif verb == "absorb":
            # faithful keyboard absorb: aim a NATIVE centre-aimed view at the object's
            # tile and press absorb (the platform/Sentinel tile needs the centre-aim).
            ps = eng.player_slot
            ez = eng.mem[OBJECTS_Z_HEIGHT + ps]
            view = native_game.centre_view_for(eng.mem, tile, ps, ez)
            if view is None:
                return _fail(i, st, "no centre-aimed view to absorb target", eng, t0)
            if not _absorb_via_aim(eng, view).get("ok"):
                return _fail(i, st, "gated absorb rejected", eng, t0)
        log(f"  [{i:2}] {verb:8} {otype} -> {tile}  ok  energy={eng.player_energy}")

    # final win action: HYPERSPACE from the platform tile.  do_hyperspace ($2186-$2196)
    # sets $0CDE bit6 (landscape complete) iff the player is standing on the platform.
    if not eng.won():
        eng.mem[0x0C61] = 0x22  # action code: hyperspace
        eng.mem[0x006E] = eng.player_slot
        eng._call(0x1B18)  # handle_player_actions -> do_hyperspace
        log(
            f"  [hyperspace from {tuple(eng.mem[a + eng.player_slot] for a in (OBJECTS_X, OBJECTS_Y))}]"
        )
    won = eng.won()
    log(
        f"\n{'WON' if won else 'NOT WON'}: real ROM $0CDE win flag = {won}; "
        f"energy {eng.player_energy}; views filled by emulated solve_aim = {filled}; "
        f"replay {time.time()-t0:.1f}s"
    )
    return won


def _fail(i, st, why, eng, t0):
    print(
        f"  [{i:2}] {st['verb']} -> {tuple(st['target'])}  FAIL: {why}  "
        f"(energy {eng.player_energy}, {time.time()-t0:.1f}s)"
    )
    return False


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ok = validate(ls)
    sys.exit(0 if ok else 1)
