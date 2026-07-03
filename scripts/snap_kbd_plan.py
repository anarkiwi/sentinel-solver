#!/usr/bin/env python3
"""Generate the ls timed plan NATIVELY and snap each aim onto the KEYBOARD-REACHABLE
grid, proving the whole thing with native_los only -- NO py65/code_engine.

Why native-only (task #23): native_los is bit-exact vs the real ROM for terrain AND
object tiles (1404/1404 LOS cases), so every plan gate -- create lands on the
LOS-marched tile ($1B46/$1F16), absorb needs LOS + look-down, energy/slots/stacking,
the rotating-enemy timing (climb_timed), and the hyperspace win ($0CDE on the
platform tile) -- is computable in milliseconds. The VICE recording run is the
authoritative real-ROM execution that proves the win, so pre-validating in py65 would
emulate the whole solve twice (and code_engine does not even model enemy rotation,
ls42's hard part). We therefore snap + dry-run entirely in the native forward model.

HOW: we run climb_timed.plan_timed (the native, enemy-rotation-timed planner) but
WRAP native_game.Game.create / .absorb so that, at the instant each action fires (the
board state in g.mem is exactly what the real VICE action will see), we re-snap the
planner's aim VIEW onto the keyboard grid and ASSERT via aim_target_native (bit-exact
$1C10/$1CDD) that the snapped view's ray still hits the intended TARGET TILE with LOS.
A create/absorb with view=None (the climb-robot / platform-robot built blind onto the
boulder we are standing under) needs no aim and is left null. The planner's own win
check (player ends on the platform tile, hyperspace $0CDE) is the native win.

Keyboard aim grid (calibrated live in VICE, ROM pan_viewpoint $10B7):
  * S / D pan objects_h_angle -8 / +8 per press   -> h reachable = {h_now + 8k mod 256}.
  * COMMA / L pan objects_v_angle -4 / +4 per press -> v reachable = {v_now + 4j}
    (parity invariant under panning).
  * U u-turn EORs objects_h_angle with $80 (a multiple of 8; already in the h grid).
  * SPACE toggles the sights; with sights ON, S/D/L/COMMA move the sights CURSOR
    ($0CC6/$0CC7) +-5px, adding a SUB-pan offset to the effective aim
    (prepare_vector_from_player_sights $1C10: h += cur_x>>3, v += (cur_y-5)>>4). The
    cursor reaches tiles whose native view is off the coarse +-8/+-4 pan grid.

The pan/cursor grid is searched relative to the player's CURRENT objects_h_angle /
objects_v_angle at the moment of each action (the open-loop VICE replay pans from the
live angle, which we read back). Output: out/kbd_snapped_<ls>.json with each step's
snapped view (or null), the per-step pre-action obj_h/obj_v, and the dry-run result.
"""

import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import native_game as NG
from native_game import Game
from native_los import NativeState, aim_target_native
import climb_timed

CUR_CX, CUR_CY = 0x50, 0x5F  # sights centre ($1356)
H_STEP, V_STEP = 8, 4  # objects_h_angle/objects_v_angle pan step (live)
CUR_STEP = 5  # sights cursor px per key (live)
CUR_X_MIN, CUR_X_MAX = 0x10, 0x90
CUR_Y_MIN, CUR_Y_MAX = 0x20, 0xA0


def _cursor_grid(centre, step, lo, hi):
    return sorted({c for n in range(-12, 13) if lo <= (c := centre + step * n) <= hi})


CUR_X_GRID = _cursor_grid(CUR_CX, CUR_STEP, CUR_X_MIN, CUR_X_MAX)
CUR_Y_GRID = _cursor_grid(CUR_CY, CUR_STEP, CUR_Y_MIN, CUR_Y_MAX)


def _player_angles(g):
    return g.mem[0x09C0 + g.player], g.mem[0x0140 + g.player]


def snap_view(g, tile, want_centre):
    """Search the keyboard grid (pan + sights cursor) for a view whose NATIVE
    action-time ray hits `tile` with LOS from the player's CURRENT eye/angle. The pan
    grid is relative to the player's current objects_h_angle/objects_v_angle (what the
    keyboard pans from). Returns (view, info) or (None, info)."""
    ps = g.player
    ez = int(g.eye)
    h0, v0 = _player_angles(g)
    st = NativeState.from_mem(g.mem)
    hbase, vbase = h0 % H_STEP, v0 % V_STEP
    hgrid = [(hbase + H_STEP * k) & 0xFF for k in range(256 // H_STEP)]
    # v is clamped to the keyboard pan band [$CD..$35] ($1149) and parity-locked to
    # v0 % 4; a snapped out-of-band view makes vice_record_plan.pan_v_to press against
    # the clamp and abort. Keep only the reachable lattice.
    vgrid = [
        v
        for v in (list(range(0xCD, 0x100, V_STEP)) + list(range(0x01, 0x36, V_STEP)))
        if v % V_STEP == vbase
    ]

    def best_over(hl, vl, cxl, cyl):
        best = None
        for h in hl:
            for v in vl:
                for cx in cxl:
                    for cy in cyl:
                        tx, ty, los, centre = aim_target_native(
                            st,
                            h,
                            v,
                            cx,
                            cy,
                            ps,
                            eye_z=ez,
                            max_steps=600,
                            return_centre=True,
                        )
                        if (tx, ty) != tile or not los:
                            continue
                        key = (centre if want_centre else 0,)
                        if best is None or key < best[0]:
                            best = (
                                key,
                                {"h_angle": h, "v_angle": v, "cursor": [cx, cy]},
                            )
        return best

    # phase 1: centre cursor, full pan grid (the common case).
    best = best_over(hgrid, vgrid, [CUR_CX], [CUR_CY])
    if best is not None:
        return best[1], {"centre": best[0][0], "cursor_used": False}
    # phase 2: off-grid rescue -- open the +-5px cursor around the native seed view.
    seed = NG.centre_view_for(g.mem, tile, ps, ez) or NG.visibility_sweep(
        g.mem, ps, ez
    ).get(tile)
    if seed is not None:
        sh, sv = seed["h_angle"], seed["v_angle"]
        hloc = [h for h in hgrid if abs(((h - sh + 128) % 256) - 128) <= 16]
        vloc = [v for v in vgrid if abs(((v - sv + 128) % 256) - 128) <= 16]
        best = best_over(hloc or hgrid, vloc or vgrid, CUR_X_GRID, CUR_Y_GRID)
        if best is not None:
            return best[1], {"centre": best[0][0], "cursor_used": True}
    return None, {"reason": "no keyboard aim (pan+cursor) hits tile with LOS"}


def _assert_hits(g, view, tile):
    st = NativeState.from_mem(g.mem)
    tx, ty, los = aim_target_native(
        st,
        view["h_angle"],
        view["v_angle"],
        view["cursor"][0],
        view["cursor"][1],
        g.player,
        eye_z=int(g.eye),
        max_steps=2000,
    )
    return (tx, ty) == tile and los, (tx, ty), los


class _Snap:
    """A lightweight stand-in for the live Game at one action: just the fields
    snap_view/_assert_hits read (mem snapshot, player slot, eye, col membership).
    Built from a plain dict so it survives pickling regardless of module name."""

    __slots__ = ("mem", "player", "eye", "col")

    def __init__(self, d):
        self.mem = d["mem"]
        self.player = d["player"]
        self.eye = d["eye"]
        self.col = d["col"]


def _snap_dict(g):
    # native_game.Game tracks the eye as a float but does NOT write the player's
    # z_height/z_fraction ($0940/$0A00+slot) back into mem, and its float-eye fraction
    # diverges from the ROM's standing surface (Game.create gives a synthoid +0.5 but
    # the ROM eye on a synthoid-on-terrain is +0.875 = frac $E0; on a boulder it is
    # +0.375 = frac $60). The LOS march (check_for_line_of_sight_to_tile $1CDD seeds the
    # eye from OBJ_Z + OBJ_ZF) is sensitive to this ~1-unit difference -- it decides
    # whether a far hop tile is visible. So we bake the ROM-correct standing surface
    # for the player into the snapshot: z_height = floor(terrain under the player), and
    # z_frac = $60 if the player stands on a boulder column, else $E0 (synthoid on
    # terrain). This matches native_game._move_placed's measured ROM surfaces and
    # code_engine's live eye (verified vs the real gate at the hop tiles).
    mem = bytearray(g.mem)
    ps = g.player
    px, py = mem[NG.OBJ_X + ps], mem[NG.OBJ_Y + ps]
    tz = NG.terrain_z(mem, px, py)
    # is there a boulder/object column on the player's tile (eye raised on a stack)?
    on_stack = (px, py) in g.col and g.col[(px, py)] > (
        tz if tz is not None else 0
    ) + 0.9
    if tz is None:  # player tile is object-occupied; use int eye
        mem[NG.OBJ_Z + ps] = int(g.eye) & 0xFF
        mem[NG.OBJ_ZF + ps] = 0x60 if on_stack else 0xE0
    else:
        mem[NG.OBJ_Z + ps] = (int(g.eye) if on_stack else tz) & 0xFF
        mem[NG.OBJ_ZF + ps] = 0x60 if on_stack else 0xE0
    return {"mem": bytes(mem), "player": ps, "eye": g.eye, "col": dict(g.col)}


def _plan_with_snapshots(landscape):
    """Run the native enemy-timed planner ONCE, capturing a board snapshot at the
    instant of every create/absorb/transfer (when g.mem == what the VICE action sees).
    Snapping is done AFTER, off the snapshots, so the slow A* runs only once."""
    rows = []  # (verb, otype, tile, view_or_None, note, snapshot, h_before, v_before)
    orig_create, orig_absorb, orig_transfer = Game.create, Game.absorb, Game.transfer

    def wc(self, otype, tile, view, note=""):
        h, v = _player_angles(self)
        rows.append(
            ("create", otype, tuple(tile), view, note, _snap_dict(self), int(h), int(v))
        )
        return orig_create(self, otype, tile, view, note)

    def wa(self, slot, view, note=""):
        h, v = _player_angles(self)
        tile = (self.mem[NG.OBJ_X + slot], self.mem[NG.OBJ_Y + slot])
        rows.append(
            (
                "absorb",
                int(self.mem[NG.OBJ_TYPE + slot]),
                tile,
                view,
                note,
                _snap_dict(self),
                int(h),
                int(v),
            )
        )
        return orig_absorb(self, slot, view, note)

    def wt(self, slot, note=""):
        h, v = _player_angles(self)
        tile = (self.mem[NG.OBJ_X + slot], self.mem[NG.OBJ_Y + slot])
        rows.append(("transfer", None, tile, None, note, None, int(h), int(v)))
        return orig_transfer(self, slot, note)

    Game.create, Game.absorb, Game.transfer = wc, wa, wt
    try:
        g = climb_timed.plan_timed(landscape, verbose=False)
    finally:
        Game.create, Game.absorb, Game.transfer = (
            orig_create,
            orig_absorb,
            orig_transfer,
        )
    return g, rows


def _cached_plan(landscape):
    """Cache the (deterministic) native plan + per-step board snapshots so the slow
    enemy-timed A* runs ONCE; later snaps are instant. Cache key = landscape + the
    planner source mtime (invalidate if the planner changes)."""
    import pickle, hashlib

    src = [
        os.path.join("scripts", f)
        for f in (
            "climb_timed.py",
            "native_game.py",
            "native_los.py",
            "enemy_dynamics.py",
            "snap_kbd_plan.py",
        )
    ]
    sig = hashlib.md5(
        ("|".join(str(os.path.getmtime(s)) for s in src if os.path.exists(s))).encode()
    ).hexdigest()[:8]
    cache = f"out/.plan_cache_{landscape:04d}_{sig}.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    g, rows = _plan_with_snapshots(landscape)
    # g holds an unpicklable closure-free Game; keep only what we need.
    slim = {
        "player_xy": g.player_xy(),
        "plat": g.plat,
        "native_won": bool(g.native_won),
    }
    with open(cache, "wb") as f:
        pickle.dump((slim, rows), f)
    return slim, rows


def snap_and_dryrun(landscape, log=print):
    snapped, rescued, errors = [], [], []
    t0 = time.time()
    g, rows = _cached_plan(landscape)  # g is a slim dict {player_xy, plat, native_won}
    plan_s = time.time() - t0

    # snap each captured action off its board snapshot (fast: no A*).
    t1 = time.time()
    for i, (verb, otype, tile, view, note, snapd, hb, vb) in enumerate(rows):
        rec = {
            "verb": verb,
            "otype": otype,
            "target": list(tile),
            "view": None,
            "note": note,
            "obj_h_before": hb,
            "obj_v_before": vb,
        }
        if verb in ("create", "absorb") and view is not None:
            snap = _Snap(snapd)
            want_centre = (verb == "absorb") or (tile in snap.col) or (otype in (0, 3))
            sv, info = snap_view(snap, tile, want_centre)
            if sv is None:
                errors.append((i, verb, tile, info.get("reason")))
            else:
                ok, hit, los = _assert_hits(snap, sv, tile)
                if not ok:
                    errors.append((i, verb + "-assert", tile, f"hits {hit} los={los}"))
                else:
                    rec["view"] = sv
                    if info.get("cursor_used"):
                        rescued.append(i)
        snapped.append(rec)
    snap_s = time.time() - t1
    elapsed = plan_s + snap_s

    g0 = Game(landscape)
    h0_origin, v0_origin = _player_angles(g0)
    on_plat = g["player_xy"] == g["plat"]
    won = bool(g["native_won"]) and on_plat
    n_aims = sum(1 for s in snapped if s["view"] is not None)
    log(
        f"ls{landscape}: native dry-run via plan_timed -> native_won={g['native_won']} "
        f"player ends {g['player_xy']} platform {g['plat']} on_platform={on_plat}"
    )
    log(
        f"  grid origin (spawn) h0=${h0_origin:02x} (h%8={h0_origin%8}) "
        f"v0=${v0_origin:02x} (v%4={v0_origin%4}); steps={len(snapped)} aimed={n_aims} "
        f"cursor-rescued={len(rescued)} {rescued}"
    )
    if errors:
        log("  SNAP/ASSERT ERRORS:")
        for e in errors:
            log(f"    step {e[0]} {e[1]} tile {e[2]}: {e[3]}")
    log(
        f"  native compute: plan_timed {plan_s:.1f}s + snap {snap_s*1000:.0f}ms "
        f"= {elapsed:.1f}s"
    )
    for i, s in enumerate(snapped):
        v = s["view"]
        vs = (
            f"h=${v['h_angle']:02x} v=${v['v_angle']:02x} cur={v['cursor']}"
            if v
            else "(no aim)"
        )
        log(f"  [{i:2}] {s['verb']:8} otype={s['otype']} -> {tuple(s['target'])} {vs}")
    result = {
        "landscape": landscape,
        "won": won and not errors,
        "native_won": bool(g["native_won"]),
        "on_platform": on_plat,
        "h0": int(h0_origin),
        "v0": int(v0_origin),
        "v_step": V_STEP,
        "h_step": H_STEP,
        "cursor_rescued_steps": rescued,
        "snap_errors": errors,
        "native_compute_ms": round(elapsed * 1000, 1),
        "steps": snapped,
    }
    return result


def main():
    ls = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    res = snap_and_dryrun(ls)
    outp = f"out/kbd_snapped_{ls:04d}.json"
    with open(outp, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {outp}: won={res['won']} errors={len(res['snap_errors'])}")
    return 0 if res["won"] else 1


if __name__ == "__main__":
    sys.exit(main())
