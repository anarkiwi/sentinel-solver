"""Scalar references for the view pricing: one aim's frames term by term, and boards."""

from sentinel import actions, aimcost, memmap as mm, pancost, projector, settlecost
from sentinel import views as vw
from sentinel.game import Game


def pan_frames(st, h0, v0, h1, v1, observer):
    """Frames a pan from ``(h0, v0)`` to ``(h1, v1)`` spends: per notch its redraw plus
    its queued scroll steps."""
    skey = projector.scene_key(st)
    return sum(
        pancost.notch_frames(st, d, ph, pv, observer, skey)
        + vw.SCROLL[0 if d < pancost.V_UP else 1]
        for d, ph, pv in pancost.notch_plots(h0, v0, h1, v1)
    )


def aim_frames(st, view, aim_from, observer, cursor_from=None):
    """Frames of a keyboard aim to ``view`` from facing ``aim_from``: sights toggle (none
    when ``cursor_from`` reuses a lit cursor), u-turns, pan, cursor drive and the tap,
    summed in that order as :func:`views.aim_total` sums them."""
    h0, v0 = int(aim_from[0]), int(aim_from[1])
    nu = aimcost.h_press_count(h0, view["h_angle"])[0]
    toggle = 0 if cursor_from is not None else vw.TOGGLE_FRAMES
    start = vw.SIGHTS_CENTRE if cursor_from is None else cursor_from
    cur = max(abs(view["cursor"][i] - start[i]) for i in range(2))
    cur = cur + vw.CURSOR_RAMP if cur else 0.0
    pan = pan_frames(st, h0, v0, view["h_angle"], view["v_angle"], observer)
    uturn = settlecost.uturn_settle_frames(st, observer) if nu else 0.0
    return sum((toggle, nu * uturn, pan, cur + vw.TAP_FRAMES))


def board(number, boulders=0):
    """Landscape ``number`` at entry with ``boulders`` boulders placed on the first free
    tiles of the player's row, the facing turned a notch per boulder."""
    st = Game.typed(number).state
    me = st.player
    x0, y = st.player_xy()
    placed = 0
    for x in sorted(range(mm.N), key=lambda x: abs(x - x0)):
        if placed == boulders:
            break
        if x != x0 and actions.create(st, mm.T_BOULDER, (x, y)) is not None:
            placed += 1
            st.obj_h_angle[me] = (st.obj_h_angle[me] + aimcost.AZIMUTH_STEP) & 0xFF
    return st
