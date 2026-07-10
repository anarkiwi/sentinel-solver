"""The ONE player-action aiming/LOS layer, shared by the planner (``solver``), the
simulated runner (``scripts/run_plan_simulated``) and the live driver (``driver``).

Every player create/absorb -- in every consumer -- resolves its aim through the two
primitives here, so the three can never diverge on *how* a tile is aimed or *whether*
it is visible.  Both are the ROM's own action path (``handle_player_actions`` $1B40):

    gate(state, view, tile)   $1B46 check_for_line_of_sight_to_tile: does the sights
                              ``view`` (h_angle, v_angle, cursor) reach ``tile`` with
                              clear line of sight, at the player's TRUE eye?  This is
                              ``los.aim_target`` (bit-exact vs the 6502).  A player
                              action is valid ONLY when this holds -- it is the gate
                              the sim, planner and driver all apply before firing.

    propose(state, tile)      find a keyboard-lattice view whose ``gate`` holds for
                              ``tile`` (``los.landable_view``), or None.  The single
                              buildability proposer for every consumer (planner, sim
                              runner, live driver).

The eye is the player's TRUE eye (``eye_z=None`` reads the object's real
``z_height`` + ``z_frac``, exactly what the ROM aim uses at fire time).  A ceiled or
otherwise rounded eye fabricates line of sight the real eye does not have, which is
how an unfaithful aim slipped past the (previously LOS-blind) sim.
"""

from sentinel import los


def resolve(state, view, eye_z=None, player=None):
    """Raw ROM aim: march the sights ``view`` and return ``(tile, los)`` -- the tile
    the ray reaches and whether line of sight is clear -- at the player's true eye.
    ``los.aim_target``, the port of prepare_vector_from_player_sights $1C10 +
    check_for_line_of_sight_to_tile $1CDD ($1B40-$1B46)."""
    p = state.player if player is None else player
    tx, ty, seen = los.aim_target(
        state,
        view["h_angle"],
        view["v_angle"],
        view["cursor"][0],
        view["cursor"][1],
        p,
        eye_z=eye_z,
    )
    return (tx, ty), bool(seen)


def gate(state, view, tile, eye_z=None, player=None):
    """True iff the sights ``view`` reaches ``tile`` with line of sight at the true
    eye -- the ROM's action-time LOS gate ($1B46).  A player create/absorb is valid
    only when this holds; ``view`` None (no aim resolved) is never valid."""
    if view is None:
        return False
    hit, seen = resolve(state, view, eye_z=eye_z, player=player)
    return seen and hit == tuple(tile)


def propose(state, tile, eye_z=None, player=None, v_band=True):
    """A keyboard-lattice view whose :func:`gate` holds for ``tile`` at the true eye,
    or None when no keyboard aim lands on it.  ``los.landable_view`` (the sights-cursor
    sweep) is the single proposer for every consumer.  The returned cursor is a fresh
    list so callers may mutate it."""
    p = state.player if player is None else player
    view = los.landable_view(state, tuple(tile), p, eye_z=eye_z, v_band=v_band)
    if view is not None:
        view = {**view, "cursor": list(view["cursor"])}
    return view
