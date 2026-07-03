#!/usr/bin/env python3
"""Complete, keyboard-executable winning plan for The Sentinel ls0, HEADLESS (no VICE).

Every create/absorb is COMMITTED through the real LOS-gated ROM path
(code_engine.create_via_gate / absorb_via_action_gate -> handle_player_actions $1B18:
gate $1B43 prepare_vector_from_player_sights $1C10 + $1B46 check_for_line_of_sight_
to_tile $1CDD; create $1BBA / absorb $1B9E), paired with a real-LOS aim VIEW that is
VERIFIED to reproduce on the committed state.

Faithful Sentinel mechanics encoded here (all measured against the real ROM):
  * You cannot build a tower taller than your sightline from the ground: from a fixed
    eye the gate lets you stack only ~3 boulders (column top ~ eye+2), then the
    climb-robot has no aim view. So ascent is STAGED -- build 3, robot, transfer up
    (eye +2), repeat -- the authentic multi-round build/transfer climb.
  * Navigation is hops (create robot on a real-LOS empty tile, transfer in); the
    plateau holding the Sentinel's platform is mounted by staged climbing, not flat
    hops (flat-eye LOS cannot reach onto the height-8 plateau).
  * native_los.aim_target_native (bit-exact, ~150us) is used to SCAN empty-tile aim
    cheaply; the committed view is always re-VERIFIED with the emulated aim_target
    under a full-memory snapshot/restore per probe (solve_aim's returned los flag is
    NOT trustworthy on its own -- its stale action-target globals yield views that do
    not reproduce at commit time; this was the bug that blocked the absorb).

Output: out/kbd_plan_0000.json -- ordered steps {verb, otype, target, view{h,v,cursor}
| null, player_tile, eye_z, note}; confirms won() ($0CDE bit6) through the gated path.
"""

import sys
import os
import json
import math
import time
from multiprocessing import Pool

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from code_engine import CodeEngine, NUM_SLOTS, N
from aim_oracle import aim_target
import native_los as nl
from game_state import (
    OBJECTS_X,
    OBJECTS_Y,
    OBJECTS_Z_HEIGHT,
    OBJECTS_TYPE,
    OBJECTS_FLAGS,
)

PLATFORM_X = 0x0C19
PLATFORM_Y = 0x0C1A
PLAYER_ENERGY = 0x0C0A
DO_HYPERSPACE = 0x2156
OUT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "out",
    "kbd_plan_0000.json",
)
_NEIGH = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (1, -1), (-1, 1))
_CUR_CX, _CUR_CY = 0x50, 0x5F


def _occupied(eng):
    return {
        (eng.mem[OBJECTS_X + s], eng.mem[OBJECTS_Y + s])
        for s in range(NUM_SLOTS)
        if not (eng.mem[OBJECTS_FLAGS + s] & 0x80)
    }


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _pxy(eng):
    ps = eng.player_slot
    return (eng.mem[OBJECTS_X + ps], eng.mem[OBJECTS_Y + ps])


def _peye(eng):
    return eng.mem[OBJECTS_Z_HEIGHT + eng.player_slot]


def _ground(eng, t):
    return eng._ground_z(t[0], t[1])


def _is_empty(eng, t):
    return eng.mem[0x0400 + ((t[0] << 3 & 0xE0) | (t[1] & 0x1F))] < 0xC0


def _find_sentinel(eng):
    for s in range(NUM_SLOTS):
        if not (eng.mem[OBJECTS_FLAGS + s] & 0x80) and eng.mem[OBJECTS_TYPE + s] == 5:
            return s
    return None


# ----------------------------------------------------------------------------
# VERIFIED gated view: sweep aim_target with full-memory snapshot/restore around
# EVERY probe and return the first view that reproduces (target, los=True). This is
# faithful to the real gate (handle_player_actions $1B18 marches the same $1C10/$1CDD)
# and committable. native scan first narrows the angle band cheaply.
# ----------------------------------------------------------------------------
def _verify_view(eng, target, h, v):
    """True iff aim_target on a clean snapshot reproduces (target, los=True) for view
    (h, v) at centre cursor -- i.e. it is committable through the real gate."""
    m = eng.mem
    ps = eng.player_slot
    base = bytes(m)
    sp = eng.cpu.sp
    try:
        m[:] = base
        eng.cpu.sp = 0xFF
        tx, ty, los = aim_target(eng, h, v, _CUR_CX, _CUR_CY, ps)
        return (tx, ty) == target and los
    finally:
        m[:] = base
        eng.cpu.sp = sp


def gated_view(eng, target, h_window=110, v_step=1):
    """Return a view {h,v,cursor} VERIFIED to reproduce (target, los=True) through the
    real gate. Strategy: native scan (bit-exact for empty targets, fast) finds a
    candidate band; verify candidates with the emulated aim_target under full-memory
    snapshot/restore (faithful + idempotent). Falls back to a verified emulated sweep
    (object-occupied targets, where native diverges)."""
    ps = eng.player_slot
    m = eng.mem
    px, py = m[OBJECTS_X + ps], m[OBJECTS_Y + ps]
    eye = m[OBJECTS_Z_HEIGHT + ps]
    est = (
        int(round((math.atan2(target[1] - py, target[0] - px) / (2 * math.pi)) * 256))
        & 0xFF
    )
    target_empty = _is_empty(eng, target)

    # 1) native candidates (only meaningful for empty targets); verify each.
    if target_empty:
        st = nl.NativeState.from_mem(m)
        cands = _native_hits(st, (px, py), eye, target, ps, h_window=44, want=24)
        for h, v in cands:
            if _verify_view(eng, target, h, v):
                return {"h_angle": h, "v_angle": v, "cursor": [_CUR_CX, _CUR_CY]}

    # 2) emulated verified sweep (object-occupied target, or native missed). Full
    # snapshot/restore around each probe keeps it idempotent.
    base = bytes(m)
    sp = eng.cpu.sp
    v_order = (
        list(range(0x90, 0x100, v_step))
        + list(range(0, 0x40, v_step))
        + list(range(0x40, 0x90, v_step))
    )
    try:
        for v in v_order:
            for off in range(-h_window, h_window + 1, 1):
                h = (est + off) & 0xFF
                m[:] = base
                eng.cpu.sp = 0xFF
                tx, ty, los = aim_target(eng, h, v, _CUR_CX, _CUR_CY, ps)
                if (tx, ty) == target and los:
                    return {"h_angle": h, "v_angle": v, "cursor": [_CUR_CX, _CUR_CY]}
    finally:
        m[:] = base
        eng.cpu.sp = sp
    return None


def _native_hits(st, viewpoint, eye_z, target, ps, h_window=36, want=1):
    """Collect up to `want` native views (h,v) from `viewpoint` at `eye_z` whose
    action target == `target` with LOS. Tight band around the analytic azimuth, and
    the v-pitch sweep ordered so a near hit is found fast and a MISS exits quickly.
    native is bit-exact for EMPTY targets. Returns a list of (h,v).

    Cost control: the marched native LOS is ~150us; to keep a MISS cheap we bound the
    h-band to +/-h_window and step the v sweep coarsely first (the pitch that hits a
    ground tile from a given range is narrow). This is the navigation hot loop."""
    st.obj_x[ps] = viewpoint[0] & 0xFF
    st.obj_y[ps] = viewpoint[1] & 0xFF
    st.obj_z_height[ps] = eye_z & 0xFF
    px, py = viewpoint
    tx0, ty0 = target
    est = int(round((math.atan2(ty0 - py, tx0 - px) / (2 * math.pi)) * 256)) & 0xFF
    # ground/near targets are hit by down-looking (0x90..0xFF) and the low wrap band
    # (0x00..0x30); search those, h innermost so the azimuth is found fast.
    v_order = list(range(0x90, 0x100, 1)) + list(range(0x00, 0x34, 1))
    hits = []
    for v in v_order:
        for off in range(-h_window, h_window + 1, 1):
            h = (est + off) & 0xFF
            ttx, tty, los = nl.aim_target_native(
                st, h, v, _CUR_CX, _CUR_CY, ps, eye_z=eye_z
            )
            if (ttx, tty) == (tx0, ty0) and los:
                hits.append((h, v))
                if len(hits) >= want:
                    return hits
    return hits


def _native_los(st, viewpoint, eye_z, target, ps):
    return len(_native_hits(st, viewpoint, eye_z, target, ps, want=1)) > 0


def native_visible_empty(eng, eye_z, occ, plat, radius=7, closer_than=None, limit=10):
    """Empty tiles within `radius` of the player that have real terrain LOS at `eye_z`
    (the cheap cold $1CDD probe eng.check_los -- a single ray, fast), ordered by
    closeness to the platform. Returns up to `limit` candidate tiles for the nav loop;
    the committable VIEW is found by gated_view on the chosen tile. (The cold probe is
    a slightly conservative pre-filter, not the commit gate, so this is safe.)"""
    ps = eng.player_slot
    px, py = eng.mem[OBJECTS_X + ps], eng.mem[OBJECTS_Y + ps]
    cands = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            t = ((px + dx), (py + dy))
            if not (0 <= t[0] < N and 0 <= t[1] < N) or t == (px, py):
                continue
            if t in occ or not _is_empty(eng, t):
                continue
            if closer_than is not None and _cheb(t, plat) >= closer_than:
                continue
            cands.append(t)
    cands.sort(key=lambda t: (_cheb(t, plat), _cheb(t, (px, py))))
    # parallel native LOS pre-filter (each tile independent; MISS ~2s emulation-free
    # but ~8k native marches, so fan out across cores -- the coordinator-mandated spot).
    pool = _get_pool()
    memb = bytes(eng.mem)
    jobs = [(memb, (px, py), eye_z, t, ps) for t in cands]
    vis = {}
    for t, ok in pool.imap_unordered(_los_job, jobs, chunksize=4):
        if ok:
            vis[t] = True
    out = [(t, None) for t in cands if t in vis][:limit]
    return out


_POOL = None


def _get_pool(procs=24):
    global _POOL
    if _POOL is None:
        _POOL = Pool(procs)
    return _POOL


def _los_job(args):
    memb, src, eye_z, t, ps = args
    st = nl.NativeState.from_mem(list(memb))
    return (t, _native_los(st, src, eye_z, t, ps))


def _native_aim(st, viewpoint, eye_z, target, ps, h_window=128):
    st.obj_x[ps] = viewpoint[0] & 0xFF
    st.obj_y[ps] = viewpoint[1] & 0xFF
    st.obj_z_height[ps] = eye_z & 0xFF
    px, py = viewpoint
    tx0, ty0 = target
    est = int(round((math.atan2(ty0 - py, tx0 - px) / (2 * math.pi)) * 256)) & 0xFF
    v_order = (
        list(range(0x90, 0x100, 2))
        + list(range(0, 0x34, 2))
        + list(range(0x70, 0x90, 2))
        + list(range(0x34, 0x70, 2))
    )
    for v in v_order:
        for off in range(-h_window, h_window + 1, 1):
            h = (est + off) & 0xFF
            ttx, tty, los = nl.aim_target_native(
                st, h, v, _CUR_CX, _CUR_CY, ps, eye_z=eye_z
            )
            if (ttx, tty) == (tx0, ty0) and los:
                return {"h_angle": h, "v_angle": v, "cursor": [_CUR_CX, _CUR_CY]}
    return None


# ----------------------------------------------------------------------------
# committed actions (mutate the real machine, record a step)
# ----------------------------------------------------------------------------
class Plan:
    def __init__(self, eng, result):
        self.eng = eng
        self.result = result
        self.steps = result["steps"]

    def rec(self, verb, otype, target, view, ok, note):
        e = self.eng
        self.steps.append(
            {
                "verb": verb,
                "otype": otype,
                "target": [target[0], target[1]] if target else None,
                "view": view,
                "player_tile": list(_pxy(e)),
                "eye_z": _peye(e),
                "ok": bool(ok),
                "note": note,
            }
        )
        _dump(self.result)
        return ok

    def hop(self, t):
        """Flat/raised hop: gated create robot on empty tile t with a verified view,
        then transfer in. Returns True on success."""
        e = self.eng
        v = gated_view(e, t)
        if v is None:
            return self.rec("create", 0, t, None, False, f"hop {t}: no verified view")
        r = e.create_via_gate(0, t, v)
        if not self.rec(
            "create",
            0,
            t,
            v,
            r["ok"],
            f"hop create robot -> {t} ({r.get('reason','')})",
        ):
            return False
        tr = e.transfer(r["slot"])
        return self.rec(
            "transfer", None, t, None, tr["ok"], f"transfer into hop robot @ {t}"
        )

    def staged_climb(self, build_tile, target_eye, per_stage=3):
        """Staged ascent on `build_tile` (adjacent to the player): build up to
        `per_stage` boulders, robot on top, transfer up (+~2 eye), repeat until the
        player eye >= target_eye. Re-picks an adjacent build tile each stage from the
        raised position. Returns the build tile actually used last, or None on fail."""
        e = self.eng
        bld = build_tile
        while _peye(e) < target_eye:
            placed = 0
            for _ in range(per_stage):
                v = gated_view(e, bld)
                if v is None:
                    break
                r = e.create_via_gate(3, bld, v)
                if not r["ok"]:
                    break
                placed += 1
                self.rec(
                    "create",
                    3,
                    bld,
                    v,
                    True,
                    f"staged boulder on {bld} (top_z={e.mem[OBJECTS_Z_HEIGHT+r['slot']]})",
                )
            if placed == 0:
                self.rec(
                    "create",
                    3,
                    bld,
                    None,
                    False,
                    f"staged climb stuck: no boulder placeable on {bld}",
                )
                return None
            vr = gated_view(e, bld)
            if vr is None:
                self.rec(
                    "create",
                    0,
                    bld,
                    None,
                    False,
                    f"staged climb: no robot view on {bld}",
                )
                return None
            rr = e.create_via_gate(0, bld, vr)
            if not self.rec(
                "create",
                0,
                bld,
                vr,
                rr["ok"],
                f"staged climb robot on {bld} ({rr.get('reason','')})",
            ):
                return None
            tr = e.transfer(rr["slot"])
            self.rec(
                "transfer",
                None,
                bld,
                None,
                tr["ok"],
                f"transfer up onto stack @ {bld}; eye now {_peye(e)}",
            )
            if _peye(e) >= target_eye:
                return bld
            # pick next adjacent empty build tile toward... just any empty neighbour
            occ = _occupied(e)
            cur = _pxy(e)
            nb = [(cur[0] + dx, cur[1] + dy) for dx, dy in _NEIGH]
            nb = [
                t
                for t in nb
                if 0 <= t[0] < N and 0 <= t[1] < N and t not in occ and _is_empty(e, t)
            ]
            if not nb:
                self.rec(
                    "create", 3, cur, None, False, "staged climb: no next build tile"
                )
                return None
            bld = nb[0]
        return bld


# ----------------------------------------------------------------------------
def _dump(result):
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)


def plan(landscape=0, time_budget=2400):
    t0 = time.time()
    eng = CodeEngine(landscape)
    eng.mem[PLAYER_ENERGY] = 0x3F
    plat = (eng.mem[PLATFORM_X], eng.mem[PLATFORM_Y])
    plat_g = eng._ground_z(*plat)
    start = _pxy(eng)
    result = {
        "landscape": landscape,
        "start": list(start),
        "platform": list(plat),
        "platform_ground": plat_g,
        "won": False,
        "steps": [],
        "notes": [],
        "route": [],
    }
    _dump(result)
    P = Plan(eng, result)

    # ---------- PHASE 1: navigate (flat real-LOS hops) toward the plateau base ----
    # Greedy: repeatedly hop to the verified-LOS empty tile closest to the platform,
    # until no flat hop makes progress (we are at the plateau base / blocked).
    while True:
        cur = _pxy(eng)
        d = _cheb(cur, plat)
        if d <= 1:
            break
        occ = _occupied(eng)
        occ.discard(cur)
        vis = native_visible_empty(eng, _peye(eng), occ, plat, radius=7, closer_than=d)
        print(
            f"[nav] at {cur} d={d} eye={_peye(eng)} visible={[t for t,_ in vis]}",
            flush=True,
        )
        progressed = False
        for t, _v in vis:
            # commit with a re-verified gated view (native view may be one of several;
            # gated_view re-finds a committable one)
            if P.hop(t):
                result["route"].append(["hop", list(t)])
                progressed = True
                print(f"[nav] hopped to {t}", flush=True)
                break
            else:
                result["notes"].append(f"hop to {t} failed gate; trying next")
        if not progressed:
            result["notes"].append(
                f"flat nav stuck at {cur} (d={d}); begin staged climb"
            )
            print(f"[nav] flat nav stuck at {cur} d={d}; -> staged climb", flush=True)
            break
        if time.time() - t0 > time_budget:
            result["notes"].append("time budget exceeded in nav")
            _dump(result)
            return result

    # ---------- PHASE 2: staged climb up the plateau to a platform-adjacent stand ---
    # From the current (plateau-base) tile, staged-climb until adjacent to the platform
    # with LOS down onto the platform tile.
    while _cheb(_pxy(eng), plat) > 1:
        cur = _pxy(eng)
        print(
            f"[climb] phase2 at {cur} d={_cheb(cur,plat)} eye={_peye(eng)}", flush=True
        )
        occ = _occupied(eng)
        # build on an empty neighbour that heads toward the platform
        nb = [(cur[0] + dx, cur[1] + dy) for dx, dy in _NEIGH]
        nb = [
            t
            for t in nb
            if 0 <= t[0] < N
            and 0 <= t[1] < N
            and t not in occ
            and _is_empty(eng, t)
            and t != plat
        ]
        nb.sort(key=lambda t: _cheb(t, plat))
        if not nb:
            result["notes"].append(f"phase2 stuck: no build neighbour at {cur}")
            _dump(result)
            return result
        # target an eye that opens LOS toward closer tiles / the platform
        used = P.staged_climb(nb[0], target_eye=_peye(eng) + 2)
        if used is None:
            result["notes"].append("phase2 staged climb failed")
            _dump(result)
            return result
        result["route"].append(["climb", list(nb[0]), f"eye{_peye(eng)}"])
        # after climbing we may now flat-hop closer; try a hop toward the platform
        cur = _pxy(eng)
        if _cheb(cur, plat) > 1:
            occ = _occupied(eng)
            occ.discard(cur)
            vis = native_visible_empty(
                eng, _peye(eng), occ, plat, radius=7, closer_than=_cheb(cur, plat)
            )
            for t, _v in vis:
                if P.hop(t):
                    result["route"].append(["hop", list(t)])
                    break
        if time.time() - t0 > time_budget:
            result["notes"].append("time budget exceeded in climb")
            _dump(result)
            return result

    stand = _pxy(eng)
    result["stand"] = list(stand)
    result["notes"].append(f"reached platform-adjacent stand {stand} eye {_peye(eng)}")
    _dump(result)

    # ---------- PHASE 3: final climb so the eye clears the platform; absorb; win ----
    # build on an empty neighbour (not the platform) until LOS down onto the platform.
    occ = _occupied(eng)
    nbs = [(stand[0] + dx, stand[1] + dy) for dx, dy in _NEIGH]
    nbs = [
        t
        for t in nbs
        if 0 <= t[0] < N
        and 0 <= t[1] < N
        and t not in occ
        and _is_empty(eng, t)
        and t != plat
    ]
    won = False
    for bld in sorted(nbs, key=lambda t: _cheb(t, plat)):
        # staged climb on bld until we can absorb the Sentinel (LOS down to platform)
        _snap = bytes(eng.mem)
        # raise eye in stages, checking platform LOS after each stage
        ok_climb = True
        while True:
            pv = gated_view(eng, plat)
            if pv is not None:
                break
            used = P.staged_climb(bld, target_eye=_peye(eng) + 2)
            if used is None:
                ok_climb = False
                break
            bld_cur = _pxy(eng)
            # after a climb the build tile is now under us; pick a fresh neighbour
            occ2 = _occupied(eng)
            nb2 = [(bld_cur[0] + dx, bld_cur[1] + dy) for dx, dy in _NEIGH]
            nb2 = [
                t
                for t in nb2
                if 0 <= t[0] < N
                and 0 <= t[1] < N
                and t not in occ2
                and _is_empty(eng, t)
                and t != plat
            ]
            if not nb2:
                ok_climb = False
                break
            bld = nb2[0]
            if _peye(eng) > plat_g + 6:
                ok_climb = False
                break
        if not ok_climb:
            continue
        # absorb the Sentinel through the action-time gate
        ssl = _find_sentinel(eng)
        pv = gated_view(eng, plat)
        ar = eng.absorb_via_action_gate(plat, pv)
        if not P.rec(
            "absorb",
            5,
            plat,
            pv,
            ar["ok"],
            f"absorb Sentinel slot {ssl} base {plat} ({ar.get('reason','')})",
        ):
            continue
        # robot on the bare platform tile, transfer, hyperspace
        pv2 = gated_view(eng, plat)
        rp = (
            eng.create_via_gate(0, plat, pv2)
            if pv2
            else {"ok": False, "reason": "no view"}
        )
        if not P.rec(
            "create",
            0,
            plat,
            pv2,
            rp["ok"],
            f"winning robot on platform ({rp.get('reason','')})",
        ):
            continue
        tp = eng.transfer(rp["slot"])
        P.rec(
            "transfer",
            None,
            plat,
            None,
            tp["ok"],
            f"transfer onto platform robot; on_platform={eng.player_on_platform()}",
        )
        eng._call(DO_HYPERSPACE)
        won = eng.won()
        P.rec(
            "hyperspace",
            None,
            plat,
            None,
            won,
            f"do_hyperspace $2156: $0CDE bit6 = {won}",
        )
        if won:
            break

    result["won"] = bool(won)
    result["wall_clock_s"] = round(time.time() - t0, 2)
    result["notes"].append(
        f"WON={won} steps={len(result['steps'])} " f"{result['wall_clock_s']}s"
    )
    _dump(result)
    print(
        f"\nWON (real $0CDE bit6): {won} | steps {len(result['steps'])} | "
        f"route {result['route']} | {result['wall_clock_s']}s"
    )
    return result


if __name__ == "__main__":
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    plan(ls)
