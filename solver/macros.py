#!/usr/bin/env python3
"""Macro expanders for the A* climb planner (T2.x).

A macro takes a search :class:`~solver.search_node.Node` and returns the child
``Node``s reachable by one high-level move.  T2.1 implements the CLIMB macro: a
keyboard LOS sweep supplies buildable footholds; each buildable tile yields a hop
(a lone synthoid) and, when it climbs, a boulder-step (batched boulders capped by
a synthoid).  Every child is priced by :mod:`solver.cost`, feasibility-gated up
front, and survivability-gated over the drained window via the true transition.

Refuel (T2.2, :func:`expand_refuel`) and endgame (T2.3, :func:`endgame_child`)
macros are implemented alongside the climb expander.
"""

import os
from typing import Optional

from solver import plan_game, cost, launch
from solver.search_node import Node
from solver.search_node import energy as node_energy
from sentinel import los, actions, actioncost, aim
from sentinel import memmap as mm
from sentinel.state import State

# Fuel types the refuel macro will absorb (below-eye, in-LOS): synthoid, sentry,
# tree, boulder.  Meanies/Sentinel/platform are excluded.
FUEL_TYPES = (mm.T_ROBOT, mm.T_SENTRY, mm.T_TREE, mm.T_BOULDER)

# Energy kept in reserve above a build's up-front cost (env-overridable, ROM-tuned):
# live enemies drain during aiming, so planned refunds silently shrink -- the reserve
# absorbs that loss when sizing a boulder batch.
RESERVE = int(os.environ.get("CLIMB_RESERVE", "2"))
# Largest boulder batch a single dwell will stack (build-cap + affordability aside).
MAX_BATCH = 4


def _top_type(g, tile):
    """Type of the TOPMOST live object on `tile` (highest z_height+z_frac), or None if
    the tile is bare terrain.  Gates boulder placement: a boulder may sit on bare terrain
    or on another boulder (type 3), never on a synthoid (0) / tree (2)."""
    top, topz = None, -1
    for s in range(64):
        if g.mem[mm.OBJECTS_FLAGS + s] & 0x80:
            continue
        if (g.mem[mm.OBJECTS_X + s], g.mem[mm.OBJECTS_Y + s]) == tile:
            z = g.mem[mm.OBJECTS_Z_HEIGHT + s] * 256 + g.mem[mm.OBJECTS_Z_FRACTION + s]
            if z > topz:
                topz, top = z, g.mem[mm.OBJECTS_TYPE + s]
    return top


def _boulder_batch(g, tile):
    """(n_boulders, capped-synthoid eye) for stacking boulders on `tile` in one dwell.
    The ROM build-height gate caps a build on an ALREADY-occupied column at top <= eye +
    ROBOT_EYE_FUDGE; the first boulder on bare terrain is uncapped.  Stack while the next
    boulder AND its capping synthoid still fit under the cap, then bound by affordability.
    Returns (0, None) if not even one boulder-step is feasible here."""
    base = g.top_of(tile)
    if base is None:
        return 0, None
    first = tile not in g.col
    cap = g.eye + plan_game.ROBOT_EYE_FUDGE
    top = base + (0.875 if first else 0.5)  # first boulder
    if top + 0.5 > cap + 1e-9:  # no room for even the capping synthoid
        return 0, None
    n_slack = 1
    while (top + 0.5) + 0.5 <= cap + 1e-9:
        top += 0.5
        n_slack += 1
    # affordability: a batch costs 2*n + 3 (+ RESERVE) up front; keep the tallest batch
    # the current energy can pay for (>=1 so an unaffordable tile is still offered and
    # dropped by can_create, not silently mis-sized).
    n_afford = int((g.energy - RESERVE - 3) // 2)
    n = max(1, min(n_slack, n_afford))
    return n, base + (0.875 if first else 0.5) + 0.5 * (n - 1) + 0.5


def _foothold_eye(g, tile, use_b, n_b):
    """TRUE resulting eye (float) of moving onto `tile`, using create's exact ROM height
    increments: first object on terrain = +0.875 (boulder) then synthoid +0.5; stacking
    onto an existing column (tile already in g.col) = +0.5 each.  A boulder-step batches
    `n_b` boulders then a capping synthoid.  Captures the fractional staircase gain a flat
    terrain+1 model misses.  None if the tile has no buildable top."""
    cur = g.top_of(tile)
    if cur is None:
        return None
    first = tile not in g.col
    if use_b:
        return cur + (0.875 if first else 0.5) + 0.5 * (max(1, n_b) - 1) + 0.5
    return cur + (0.875 if first else 0.5)  # synthoid on terrain/column


def _full_top(g, slot):
    """Full top height (z_height + z_frac/256) of the object in `slot`."""
    return g.state.obj_z_height[slot] + g.state.obj_z_frac[slot] / 256.0


def _boulder_centre_feasible(g, tile):
    """Simulate landing a boulder at `tile` (a byte-level mutation mirroring create's own
    boulder placement, on a throwaway mem copy) and check the on-boulder synthoid then has
    a keyboard aim onto it from the PLAYER's TRUE eye (``aim.propose`` -- the single
    landable proposer, gated by the ROM action LOS at fire time).  The pre-build sweep's
    estimate is against the BARE surface, so a tile can look ok yet have no valid on-boulder
    aim once the boulder is real -- catch it here.
    """
    if not g.free:
        return False
    mem = bytearray(g.mem)
    slot = g.free[-1]
    tb = mem[mm.TILES_TABLE + mm.tidx(*tile)]
    if tb >= mm.OBJECT_TILE:  # stacking a boulder onto an existing boulder/platform
        below = tb & 0x3F
        if mem[mm.OBJECTS_TYPE + below] not in (mm.T_BOULDER, mm.T_PLATFORM):
            return False
        t = mem[mm.OBJECTS_Z_FRACTION + below] + 0x80
        zf = t & 0xFF
        z = mem[mm.OBJECTS_Z_HEIGHT + below] + (t >> 8)
        mem[mm.OBJECTS_FLAGS + slot] = 0x40 | below
    else:  # bare terrain
        zf = 0xE0
        z = tb >> 4
        mem[mm.OBJECTS_FLAGS + slot] = 0x00
    mem[mm.TILES_TABLE + mm.tidx(*tile)] = mm.OBJECT_TILE | slot
    mem[mm.OBJECTS_X + slot] = tile[0]
    mem[mm.OBJECTS_Y + slot] = tile[1]
    mem[mm.OBJECTS_Z_HEIGHT + slot] = z & 0xFF
    mem[mm.OBJECTS_Z_FRACTION + slot] = zf & 0xFF
    mem[mm.OBJECTS_TYPE + slot] = mm.T_BOULDER
    return aim.propose(State(mem), tile, eye_z=None, player=g.player) is not None


def _buildable(g, tile, use_b):
    """Legality gate 1: can_create ($1F38 energy/free-slot/stackability) plus, for a
    boulder-step, the on-boulder centre-aim feasibility.  LOS is supplied by the sweep.
    """
    otype = mm.T_BOULDER if use_b else mm.T_ROBOT
    if not actions.can_create(g.state, otype, tuple(tile)):
        return False
    if use_b and not _boulder_centre_feasible(g, tile):
        return False
    return True


def _foothold_options(g, tile, centres):
    """The (use_boulder, n_boulders) build options on `tile`: a hop, then a bounded
    boulder batch (capped at MAX_BATCH) when a boulder-step is feasible.  Gates a
    boulder-topped column on a keyboard centre-aim (tile-centre fraction < $40)."""
    top = _top_type(g, tile)
    centre_ok = (top != mm.T_BOULDER) or centres.get(tile, 0xFF) < 0x40
    if not centre_ok:
        return []
    free = len(g.free)
    opts = []
    if free >= 1:
        opts.append((False, 0))  # hop: lone synthoid
    n, _ = _boulder_batch(g, tile)
    if n >= 1 and free >= 2:  # boulder-step needs boulder + synthoid slots
        opts.append((True, min(n, MAX_BATCH)))
    return opts


def _apply_climb(n, t2, use_b, n_b, view, gaze) -> Optional[Node]:
    """Concrete climb macro body on a cloned PlanGame: build, transfer, then advance the
    world and run the T1.2 survivability test.  Returns the child Node, or None if a
    legality/survivability gate rejects the move."""
    del gaze  # T2.1 prices the true transition via cost.survivable, not the gaze forecast
    g = n.g.clone()
    window, end_h, end_v = cost.move_rounds(g, t2, use_b, n_b, view, n.vh, n.vv)
    if not _buildable(g, t2, use_b):  # legality gate 1
        return None
    prev_slot, prev_tile = g.player, g.player_xy()
    if use_b:
        for i in range(max(1, n_b)):
            b = g.create(mm.T_BOULDER, t2, view if i == 0 else None, "climb boulder")
            if (
                i == 0 and b is None
            ):  # gated aim has no real-eye LOS -> not buildable here
                return None
        s = g.create(mm.T_ROBOT, t2, None, "climb synthoid")
    else:
        s = g.create(mm.T_ROBOT, t2, view, "hop synthoid")
    if s is None:
        return None
    g.transfer(s, "step")
    # look-back reabsorb of the departed shell if it is now below eye and in LOS.  The
    # aim is proposed at the TRUE eye (aim.propose == the single player-aim proposer);
    # absorb re-gates it and no-ops on a reject, so this stays best-effort.
    rv = aim.propose(g.state, prev_tile, eye_z=None, player=g.player)
    if (
        rv is not None
        and g.state.obj_type[prev_slot] == mm.T_ROBOT
        and _full_top(g, prev_slot) <= g.eye + 1e-9
    ):
        g.absorb(prev_slot, rv, "reabsorb prior shell")
    # legality gate 2: managed-exposure survivability over the whole window.
    ok, _ea = cost.survivable(g, prev_tile, window)
    if not ok:
        return None
    return Node(
        g=g,
        t=n.t + int(round(window)),
        vh=end_h,
        vv=end_v,
        cost=n.cost + window,
        parent=n,
        macro={
            "kind": "climb",
            "tile": list(t2),
            "use_boulder": use_b,
            "n_boulders": n_b,
        },
    )


def expand_climb(n, gaze, beam=None, horizon=None):
    """Climb-macro expander: every buildable foothold in the keyboard LOS sweep, as a hop
    and (if it climbs) a boulder-step, priced and survivability-gated.  Non-regressing
    (resulting eye >= current eye) children only.

    ``beam``/``horizon`` enable lazy, beam-priority materialization: candidates are ranked
    by the planner's beam key ``(-resulting_eye, window_cost)`` from cheap geometry
    (:func:`_foothold_eye`, :func:`cost.move_rounds`, no aim-LOS), and the expensive
    ``_apply_climb`` runs ONLY until ``beam`` children pass ``energy>0 ^ t<horizon``.  That
    key equals the planner's beam key (``_apply_climb`` leaves eye ``he``, cost ``window``),
    so the survivors are identical to full materialization.  ``beam=None`` materializes all.
    """
    g = n.g
    cur = g.player_xy()
    views, centres = los.landable_sweep_with_centres(
        g.state, g.player, int(g.eye), max_steps=6000
    )
    # (-he, window, t2, use_b, n_b, view): the planner beam key + build args
    cands = []
    for t2, view in views.items():
        if t2 == cur:
            continue
        if plan_game.terrain_z(g.mem, *t2) is None and t2 not in g.col:
            continue  # object tile that isn't our own boulder column
        top = _top_type(g, t2)
        if top is not None and top != mm.T_BOULDER:  # synthoid/tree on top -> nothing
            continue
        for use_b, n_b in _foothold_options(g, t2, centres):
            he = _foothold_eye(g, t2, use_b, n_b)
            if he is None or he < g.eye - 1e-9:  # non-regression
                continue
            window, _eh, _ev = cost.move_rounds(g, t2, use_b, n_b, view, n.vh, n.vv)
            cands.append((-he, window, t2, use_b, n_b, view))
    cands.sort(key=lambda c: (c[0], c[1]))
    out = []
    kept = 0
    for _nhe, _w, t2, use_b, n_b, view in cands:
        child = _apply_climb(n, t2, use_b, n_b, view, gaze)
        if child is None:
            continue
        out.append(child)
        if (
            beam is not None
            and node_energy(child) > 0
            and (horizon is None or child.t < horizon)
        ):
            kept += 1
            if kept >= beam:
                break
    return out


def _top_fuel_slot(g, tile):
    """The topmost live object slot on `tile` if it is below-eye, absorbable fuel;
    else None.  Picks the highest-z object, then gates it on FUEL_TYPES, full top
    ``<= g.eye`` (below the player's eye), and ``actions.can_absorb``."""
    best, bz = None, -1.0
    for s in range(mm.NUM_SLOTS):
        if g.state.obj_flags[s] & 0x80:
            continue
        if (g.state.obj_x[s], g.state.obj_y[s]) != tile:
            continue
        z = _full_top(g, s)
        if z > bz:
            bz, best = z, s
    if best is None:
        return None
    if g.state.obj_type[best] not in FUEL_TYPES:
        return None
    if _full_top(g, best) > g.eye + 1e-9:  # not below eye
        return None
    if not actions.can_absorb(g.state, best):
        return None
    return best


def _apply_refuel(n, tile, slot, view) -> Optional[Node]:
    """Concrete refuel macro body: price the aim+absorb window, apply the absorb on
    a cloned PlanGame, then survivability-gate it over the drained window.  Returns
    the child Node, or None if the window is not survivable or energy did not rise."""
    g = n.g.clone()
    from_tile = g.player_xy()
    e0 = g.energy
    window = cost.aim_rounds(n.vh, n.vv, view) + actioncost.SETTLE["absorb"]
    if not g.absorb(slot, view, "refuel"):
        return None  # ROM action LOS gate rejected the aim -> the refuel did not happen
    ok, ea = cost.survivable(g, from_tile, window)
    if not ok or ea <= e0:  # must survive and net an energy gain
        return None
    end_h = view.get("h_angle") if view else n.vh
    end_v = view.get("v_angle") if view else n.vv
    return Node(
        g=g,
        t=n.t + int(round(window)),
        vh=end_h if end_h is not None else n.vh,
        vv=end_v if end_v is not None else n.vv,
        cost=n.cost + window,
        parent=n,
        macro={"kind": "refuel", "tile": list(tile)},
    )


def expand_refuel(n, gaze, need=cost.NEXT_COST_FLOOR + 2):
    """Refuel-macro expander (T2.2): only when energy-constrained, absorb below-eye
    in-LOS fuel to buy the next climb.  Deferred while ``energy >= need``.

    Candidates are the topmost absorbable fuel object on each keyboard-LOS sweep
    tile.  "Broadly visible" fuel -- fuel whose tile is ALSO seen from a raised
    observer eye (``eye + ROBOT_EYE_FUDGE``), i.e. exposed on open high ground -- is
    deferred while any less-visible source exists, and taken only as a last resort
    (the sole affordable source).  Each accepted candidate is priced (aim + absorb
    SETTLE), applied, and survivability-gated; only survivable, energy-raising
    children are emitted."""
    del gaze  # priced via the true transition (cost.survivable), not the forecast
    g = n.g
    if g.energy >= need:  # defer: only refuel when energy-blocked
        return []
    # Landable (aim-consistent) sweep at the player's TRUE eye supplies the fuel
    # candidates + their views; a raised-eye landable sweep classifies "broadly
    # visible" (open high-ground) fuel to defer -- the same landable oracle the
    # climb macro uses for the player's aim.
    sw, _ = los.landable_sweep_with_centres(g.state, g.player, eye_z=None)
    hi, _ = los.landable_sweep_with_centres(
        g.state, g.player, eye_z=int(g.eye) + plan_game.ROBOT_EYE_FUDGE
    )
    cur = g.player_xy()
    cands = []  # (tile, slot, view, broadly_visible)
    for tile, view in sw.items():
        if tile == cur:
            continue
        slot = _top_fuel_slot(g, tile)
        if slot is not None:
            cands.append((tile, slot, view, tile in hi))
    only_broad = bool(cands) and all(c[3] for c in cands)
    out = []
    for tile, slot, view, broad in cands:
        if broad and not only_broad:  # defer broadly-visible fuel
            continue
        child = _apply_refuel(n, tile, slot, view)
        if child is not None:
            out.append(child)
    return out


def endgame_child(n, plat, plat_ground) -> Optional[Node]:
    """Endgame macro (T2.3): from a launch-ready node, drive-through absorb the
    Sentinel then build+transfer onto the platform for the win.  Returns the winning
    child Node, or None if the aim/feasibility/terminal gate fails.

    The win shot is the player looking DOWN onto the platform, so the precondition is a
    real KEYBOARD-aimable down-look view (``los.landable_view`` with the body-pitch band),
    NOT mere geometric LOS: the sim absorbs by slot regardless of aim, but the live driver
    aims BY this view, so a geometric-only launch fires blind and misses live.
    The resolved view is attached to every platform-targeting step so the plan is aim-exact.
    """
    if not launch.endgame_ready(n.g, plat, plat_ground):
        return None
    plat = tuple(plat)
    g = n.g.clone()
    # Resolve the launch view at the player's TRUE eye (eye_z=None reads the real
    # z_height + z_frac -- what the ROM aim uses at fire time). Ceiling the eye here
    # picked a cursor that only clears terrain at the next integer height, so the
    # live fire from the real (lower) eye had no LOS and missed the Sentinel.
    view = los.landable_view(g.state, plat, g.player, eye_z=None, v_band=True)
    if (
        view is None
    ):  # no keyboard aim lands on the platform from here -- not launchable
        return None
    view = {**view, "cursor": list(view["cursor"])}
    sent = g.state.slot_of_type(mm.T_SENTINEL)
    if sent is not None and not g.absorb(sent, view, "absorb Sentinel"):
        return None  # ROM action LOS gate: no real-eye LOS to the platform -> not launchable
    if not g.feasible(mm.T_ROBOT, plat):
        return None
    s = g.create(mm.T_ROBOT, plat, view, "platform synthoid")
    if s is None:
        return None
    g.transfer(s, "hyperspace onto platform (WIN)")
    if actions.on_platform(g.state) and not actions.player_dead(g.state):
        return Node(
            g=g,
            t=n.t,
            vh=n.vh,
            vv=n.vv,
            cost=n.cost,
            parent=n,
            macro={"kind": "endgame"},
        )
    return None
