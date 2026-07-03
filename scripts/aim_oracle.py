#!/usr/bin/env python3
"""Offline aim oracle for The Sentinel: given the player's view state in a
CodeEngine machine, run the REAL prepare_vector_from_player_sights ($1C10) +
check_for_line_of_sight_to_tile ($1CDD) to compute the action target tile for a
candidate (h_angle, v_angle, sights cursor). Then search the discrete view space
for the candidate that hits a desired tile with LOS.

This is the faithful action-time path (handle_player_actions $1B40-$1B46): it
reads $0CC6/$0CC7 (cursor) and objects_h_angle/objects_v_angle[player], so unlike
the cold $1C54 LOS-probe stub it matches the live target exactly.
"""

import sys, os, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_state import (
    OBJECTS_H_ANGLE,
    OBJECTS_V_ANGLE,
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
)

PREP_SIGHTS = 0x1C10
CHECK_LOS = 0x1CDD
A_SIGHTS_X = 0x0CC6
A_SIGHTS_Y = 0x0CC7
A_OBSERVER = 0x006E
A_DO_LOS = 0x0C6E  # do_line_of_sight_checks; top bit cleared at $1B40
A_TGT_X = 0x0024
A_TGT_Y = 0x0026
A_0C56 = 0x0C56
A_0CDD = 0x0CDD
A_C58 = 0x0C58  # targeted object

# cursor centre values (initialise_sights $1356/$135D)
CUR_CX = 0x50
CUR_CY = 0x5F


def aim_target(eng, h_angle, v_angle, cur_x, cur_y, player_slot=None):
    """Compute the action target tile + LOS for a candidate view state, using the
    REAL $1C10 + $1CDD. Returns (tx, ty, los_ok). Saves/restores mutated memory."""
    m = eng.mem
    ps = eng.player_slot if player_slot is None else player_slot
    sH = m[OBJECTS_H_ANGLE + ps]
    sV = m[OBJECTS_V_ANGLE + ps]
    sCX = m[A_SIGHTS_X]
    sCY = m[A_SIGHTS_Y]
    sOB = m[A_OBSERVER]
    sDO = m[A_DO_LOS]
    sTX = m[A_TGT_X]
    sTY = m[A_TGT_Y]
    s56 = m[A_0C56]
    sDD = m[A_0CDD]
    s58 = m[A_C58]
    try:
        m[OBJECTS_H_ANGLE + ps] = h_angle & 0xFF
        m[OBJECTS_V_ANGLE + ps] = v_angle & 0xFF
        m[A_SIGHTS_X] = cur_x & 0xFF
        m[A_SIGHTS_Y] = cur_y & 0xFF
        m[A_OBSERVER] = ps
        m[A_DO_LOS] &= 0x7F  # $1B40 LSR $0C6E (not considering a robot)
        m[A_C58] = 0xFF  # no specific targeted object
        eng._call(PREP_SIGHTS, x=ps)  # $1B29 LDX $006E before $1C10
        # cap the LOS march: a real on-board hit settles in <12k instrs; a
        # near-horizontal ray sweeping to the board edge burns 100k-400k. We bound
        # it (an over-cap ray is treated as no-LOS -- it walked off the board).
        import _emu

        eng.cpu.a = m[OBJECTS_V_ANGLE + ps]  # benign; check uses ZP
        n = _emu.call(eng.cpu, eng.mem, CHECK_LOS, state=eng.state, maxins=20000)
        eng.state["stop"] = False
        eng.instructions += n
        capped = n >= 20000
        los_ok = (not capped) and ((eng.cpu.p & 0x01) == 0)
        tx, ty = m[A_TGT_X], m[A_TGT_Y]
    finally:
        m[OBJECTS_H_ANGLE + ps] = sH
        m[OBJECTS_V_ANGLE + ps] = sV
        m[A_SIGHTS_X] = sCX
        m[A_SIGHTS_Y] = sCY
        m[A_OBSERVER] = sOB
        m[A_DO_LOS] = sDO
        m[A_TGT_X] = sTX
        m[A_TGT_Y] = sTY
        m[A_0C56] = s56
        m[A_0CDD] = sDD
        m[A_C58] = s58
    return tx, ty, los_ok


def solve_aim(
    eng,
    target_tile,
    player_slot=None,
    eye_z=None,
    h_window=40,
    h_step=2,
    require_los=True,
    budget=400,
):
    """Search (h_angle, v_angle, cursor) for a candidate whose REAL action target
    == the desired tile with LOS. Returns a dict {h_angle, v_angle, cursor, los,
    target} or None.

    Strategy: analytic atan2 estimate of the azimuth, hill-climb h in a window,
    scan v over the full circle at step 4 (v dominates pitch). Cursor pinned at
    centre primarily, with a few off-centre fine options. eye_z overrides the
    player's z_height for the duration (model a raised eye after a climb)."""
    tx0, ty0 = target_tile
    ps = eng.player_slot if player_slot is None else player_slot
    m = eng.mem
    px, py = m[OBJECTS_X + ps], m[OBJECTS_Y + ps]
    saved_z = m[OBJECTS_Z_HEIGHT + ps]
    if eye_z is not None:
        m[OBJECTS_Z_HEIGHT + ps] = eye_z & 0xFF
    try:
        dx, dy = tx0 - px, ty0 - py
        est = int(round((math.atan2(dy, dx) / (2 * math.pi)) * 256)) & 0xFF
        curs = [
            (CUR_CX, CUR_CY),
            (0x40, CUR_CY),
            (0x60, CUR_CY),
            (CUR_CX, 0x4F),
            (CUR_CX, 0x6F),
        ]
        best = None
        calls = [0]

        def consider(h, v, cx, cy):
            calls[0] += 1
            tx, ty, los = aim_target(eng, h, v, cx, cy, ps)
            if (tx, ty) == (tx0, ty0):
                cand = {
                    "h_angle": h & 0xFF,
                    "v_angle": v & 0xFF,
                    "cursor": (cx, cy),
                    "los": los,
                    "target": (tx, ty),
                }
                if los:
                    return cand, True
                nonlocal best
                if best is None:
                    best = cand
            return None, False

        # v: looking DOWN at ground targets uses v in the upper band (~$90..$FF and
        # wrap $00..$30). Order the down-looking band first; it both finds solutions
        # faster and avoids the slow near-horizontal sweeps (v~$30..$70).
        v_order = (
            list(range(0x90, 0x100, 4))
            + list(range(0x00, 0x34, 4))
            + list(range(0x70, 0x90, 4))
            + list(range(0x34, 0x70, 4))
        )
        for v in v_order:
            if calls[0] > budget:
                break
            for off in range(-h_window, h_window + 1, h_step):
                h = (est + off) & 0xFF
                cand, done = consider(h, v, CUR_CX, CUR_CY)
                if done:
                    return cand
                if calls[0] > budget:
                    break
        # fine cursor pass around the best (no-LOS) hit, varying cursor
        if best is not None and require_los:
            bh, bv = best["h_angle"], best["v_angle"]
            for dv in range(-4, 5):
                for cx, cy in curs:
                    cand, done = consider(bh, (bv + dv) & 0xFF, cx, cy)
                    if done:
                        return cand
        return (
            best
            if (best and not require_los)
            else (best if best and best["los"] else (best if not require_los else None))
        )
    finally:
        m[OBJECTS_Z_HEIGHT + ps] = saved_z
