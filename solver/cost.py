"""Move cost model: elapsed enemy-rounds per aim/action, derived purely from the
keyboard-aim geometry (``sentinel.aimcost``) and the per-verb settle floors
(``sentinel.actioncost``).

Mirrors the verified ``climb_search._move_cost`` / ``_pan_rounds`` accounting but with
no climb-search coupling: a climb macro's cost is the aim pan onto the build tile, the
per-verb SETTLE floor of every fired verb (a stacked create adds ``STACK_CREATE``), and
the return-pan that swings the view back to reabsorb the departed shell. The same figures
the tick-accurate runner advances the world by (``run_plan_simulated.execute_step``), so
the planner forecasts enemy rotation/drain over exactly the window an action really costs.
"""

from sentinel import aimcost as ac, actioncost

ROUNDS_PER_H_STEP = 16.0
ROUNDS_PER_V_STEP = 8.0
ROUNDS_PER_UTURN = 16.0


def aim_rounds(h0, v0, view):
    """Enemy rounds to pan the view from heading ``(h0, v0)`` onto ``view``'s aim: the
    U-turn-aware bearing pan plus the pitch pan, weighted by the per-axis scroll cadence.
    0 when the view is empty or carries no bearing."""
    if not view or view.get("h_angle") is None:
        return 0.0
    r = ac.bearing_rounds(h0, view["h_angle"], ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
    if view.get("v_angle") is not None and v0 is not None:
        r += ac.v_steps(v0, view["v_angle"]) * ROUNDS_PER_V_STEP
    return r


def move_rounds(g, t2, use_boulder, n_boulders, view, vh, vv):
    """``(rounds, end_h, end_v)`` for a climb macro: aim + build + transfer + look-back
    reabsorb. Prices each fired verb by ``actioncost.SETTLE``; a stacked create adds
    ``actioncost.STACK_CREATE``. ``end_h``/``end_v`` are the ending view heading so the
    next move's pan chains from where this one left the view."""
    prev = g.player_xy()
    r = aim_rounds(vh, vv, view)
    if use_boulder:
        n = max(1, n_boulders)
        r += ROUNDS_PER_H_STEP + ROUNDS_PER_V_STEP  # recentre on-boulder synthoid
        first_stacked = t2 in g.col
        settle = actioncost.SETTLE["create"] + (
            actioncost.STACK_CREATE if first_stacked else 0.0
        )
        # boulders 2..n and the capping synthoid all stack -> STACK_CREATE each.
        settle += n * (actioncost.SETTLE["create"] + actioncost.STACK_CREATE)
    else:
        settle = actioncost.SETTLE["create"]
    settle += actioncost.SETTLE["transfer"]
    end_h, end_v = (view.get("h_angle"), view.get("v_angle")) if view else (vh, vv)
    back_h = ac.bearing_to(t2[0], t2[1], prev[0], prev[1])  # look back at departed tile
    if back_h is not None and end_h is not None:
        r += ac.bearing_rounds(end_h, back_h, ROUNDS_PER_H_STEP, ROUNDS_PER_UTURN)
        end_h = back_h
        settle += actioncost.SETTLE["absorb"]  # reabsorb-shell confirm
    return r + settle, end_h, end_v
