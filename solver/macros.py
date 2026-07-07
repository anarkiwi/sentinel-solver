#!/usr/bin/env python3
"""Macro expanders for the A* climb planner (T2.x).

A macro takes a search :class:`~solver.search_node.Node` and returns the child
``Node``s reachable by one high-level move.  T2.1 implements the CLIMB macro: a
keyboard LOS sweep supplies buildable footholds; each buildable tile yields a hop
(a lone synthoid) and, when it climbs, a boulder-step (batched boulders capped by
a synthoid).  Every child is priced by :mod:`solver.cost`, feasibility-gated up
front, and survivability-gated over the drained window via the true transition.

The climb mechanics (`_top_type`, `_boulder_batch`, `_foothold_eye`,
`_boulder_centre_feasible`) are ported from the validated ``solver.climb_search``.
Refuel (T2.2) and endgame (T2.3) macros are separate later tasks; this module
leaves room for them but implements only the climb expander.
"""

import os
from typing import Optional

from solver import plan_game, cost
from solver.search_node import Node
from sentinel import los, actions
from sentinel import memmap as mm

# Energy kept in reserve above a build's up-front cost (env-overridable, ROM-tuned;
# ported from climb_search): live enemies drain during aiming, so planned refunds
# silently shrink -- the reserve absorbs that loss when sizing a boulder batch.
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


def _boulder_centre_feasible(g, tile, max_steps=2000):
    """Simulate landing a boulder at `tile` (a byte-level mutation mirroring create's own
    boulder placement, on a throwaway mem copy) and check the on-boulder synthoid then has
    a keyboard-reachable centre-aim ($1E48 <$40) from the PLAYER's current eye.  The
    pre-build sweep's centre estimate is against the BARE surface, so a tile can look
    centre-ok yet have no valid on-boulder aim once the boulder is real -- catch it here.
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
    return (
        plan_game.centre_view_for(mem, tile, g.player, int(g.eye), max_steps=max_steps)
        is not None
    )


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
    """Concrete climb macro body (mirrors climb_search._apply on a cloned PlanGame, then
    advances the world and runs the T1.2 survivability test).  Returns the child Node, or
    None if a legality/survivability gate rejects the move."""
    del gaze  # T2.1 prices the true transition via cost.survivable, not the gaze forecast
    g = n.g.clone()
    window, end_h, end_v = cost.move_rounds(g, t2, use_b, n_b, view, n.vh, n.vv)
    if not _buildable(g, t2, use_b):  # legality gate 1
        return None
    prev_slot, prev_tile = g.player, g.player_xy()
    if use_b:
        for i in range(max(1, n_b)):
            g.create(mm.T_BOULDER, t2, view if i == 0 else None, "climb boulder")
        s = g.create(mm.T_ROBOT, t2, None, "climb synthoid")
    else:
        s = g.create(mm.T_ROBOT, t2, view, "hop synthoid")
    if s is None:
        return None
    g.transfer(s, "step")
    # look-back reabsorb of the departed shell if it is now below eye and in LOS.
    sw = plan_game.visibility_sweep(g.mem, g.player, int(g.eye), max_steps=200)
    if (
        prev_tile in sw
        and g.state.obj_type[prev_slot] == mm.T_ROBOT
        and _full_top(g, prev_slot) <= g.eye + 1e-9
    ):
        g.absorb(prev_slot, sw[prev_tile], "reabsorb prior shell")
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


def expand_climb(n, gaze):
    """Climb-macro expander: every buildable foothold in the keyboard LOS sweep, as a hop
    and (if it climbs) a boulder-step, priced and survivability-gated.  Non-regressing
    (resulting eye >= current eye) children only."""
    g = n.g
    cur = g.player_xy()
    views, centres = los.landable_sweep_with_centres(
        g.state, g.player, int(g.eye), max_steps=6000
    )
    out = []
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
            child = _apply_climb(n, t2, use_b, n_b, view, gaze)
            if child is not None:
                out.append(child)
    return out
