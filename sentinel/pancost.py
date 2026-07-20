"""Per-notch pan redraw cost, derived from pan_viewpoint ($10B7).

One keyboard notch clears the strip buffer, adds its $9925 delta, runs ONE plot_world
($2625) at that INTERMEDIATE angle (not the destination), then fixes the residue and
queues the notch's scroll steps. See docs/render_cost.md for the derivation.
"""

import os

from sentinel import aimcost, projector

# $9925 delta added before the plot, indexed by the $0008 direction ($10E9/$1130 fix the residue after it): 0 = bearing +8, 1 = bearing -8, 2 = pitch +4, 3 = pitch -4.
PAN_DELTA = (0x14, 0xF8, 0x04, 0xF4)
H_RIGHT, H_LEFT, V_UP, V_DOWN = 0, 1, 2, 3
# $2993 buffer mode per direction: $994F A=#$02 for a bearing notch, $9939 A=#$00 for a pitch one.
PAN_MODE = (2, 2, 0, 0)

# Strip clears: $3912 stores 24 bytes/iteration over 64 iterations, $38AD 32 over 40, each called twice (odd then even X), at 5 cycles a store plus 7 for the loop tail.
_CLEAR_CYCLES_H = 2 * 64 * (24 * 5 + 7)
_CLEAR_CYCLES_V = 2 * 40 * (32 * 5 + 7)
CLEAR_FRAMES = (
    _CLEAR_CYCLES_H / projector.FRAME_CYCLES,
    _CLEAR_CYCLES_V / projector.FRAME_CYCLES,
)

_CACHE_MAX = int(os.environ.get("PAN_CACHE_MAX", "20000"))
_NOTCH_CACHE = {}


def notch_plots(h0, v0, h1, v1):
    """The ($0008 direction, plot h, plot v) of every notch a keyboard aim from
    (``h0``, ``v0``) to (``h1``, ``v1``) animates, in the order the executor keys them:
    u-turn-aware bearing steps (``coarse_h``) then pitch (``coarse_v``). A u-turn
    contributes none -- $1B2F is an EOR $80 with no scroll and no replot."""
    out = []
    nu, ns = aimcost.h_press_count(h0, h1)
    h = (h0 ^ 0x80) if nu else h0
    step = 1 if ((h1 - h) & 0xFF) < 0x80 else -1
    for _ in range(ns):
        d = H_RIGHT if step > 0 else H_LEFT
        out.append((d, (h + PAN_DELTA[d]) & 0xFF, v0))
        h = (h + step * aimcost.AZIMUTH_STEP) & 0xFF
    v = v0
    vstep = 1 if ((v1 - v) & 0xFF) < 0x80 else -1
    for _ in range(aimcost.v_steps(v0, v1)):
        d = V_UP if vstep > 0 else V_DOWN
        out.append((d, h, (v + PAN_DELTA[d]) & 0xFF))
        v = (v + vstep * aimcost.PITCH_STEP) & 0xFF
    return out


def notch_frames(state, direction, plot_h, plot_v, observer=None):
    """Frames one pan notch costs: the strip clear plus the ONE plot_world it runs at
    the intermediate angle, through that direction's $2993 buffer mode. Excludes the
    queued scroll steps, which the caller charges as H_SCROLL/V_SCROLL."""
    obs = state.player if observer is None else observer
    key = (projector.scene_key(state), obs, direction, plot_h, plot_v)

    def make():
        mode = PAN_MODE[direction]
        cost = projector.render_cost(
            state, {"h_angle": plot_h, "v_angle": plot_v}, obs, mode
        )
        return cost + CLEAR_FRAMES[0 if mode else 1]

    return projector.memo(_NOTCH_CACHE, key, _CACHE_MAX, make)


def pan_frames(state, h0, v0, h1, v1, scroll, observer=None):
    """Total frames a keyboard aim from (``h0``, ``v0``) to (``h1``, ``v1``) spends
    panning: per notch, its own redraw plus ``scroll[axis]`` queued scroll steps."""
    return sum(
        notch_frames(state, d, ph, pv, observer) + scroll[0 if d < V_UP else 1]
        for d, ph, pv in notch_plots(h0, v0, h1, v1)
    )
