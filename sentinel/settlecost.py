"""Per-verb settle frames, composed from the ROM's own settle paths.

Create/absorb settle through $12D0 -> $1B18 -> $12EC ($1F9F strip replot with the
$86A5 dither) and the pass tail; a transfer or u-turn starts its tune ($1B84), sets
$0C63 and settles through the $357D redraw with the tune wait overlapped.
"""

from sentinel import los
from sentinel import memmap as mm
from sentinel import passcost, projector, relative, writeweight
from sentinel.badline import PAL_FRAME_CYCLES

SETTLE_BODY_FLAT = 356  # $95E9..RTI, $0CE4 bit7 set, $130C Bresenham no-carry
SETTLE_BODY_CARRY = 704  # same, carry frame: the $131E cooldown walk + $1635
_BRES = mm.COOLDOWN_BRESENHAM_STEP / 256.0
SETTLE_BODY = SETTLE_BODY_FLAT + _BRES * (SETTLE_BODY_CARRY - SETTLE_BODY_FLAT)
SETTLE_STEAL = 1069.0  # store-heavy plot frames, fixtures/live_badline.json machine
SETTLE_FG = PAL_FRAME_CYCLES - passcost.SHORT_IRQ_FRAME - SETTLE_BODY - SETTLE_STEAL

BRACKET_FRAMES = 3.0  # tap_action: idle scan + press scan + the $130B=2 scan re-arm

REDRAW_HEAD = 13_249  # $357D..$35BA: $351C, $8643, the $3592 zeroing, sprites off
TAB_3700 = 1_648_000  # $35BD..$35C0 per-position tables; scene band 1.631-1.666M
FILL_1090 = 108_935  # $35C0..$35C3, constant
STATUS_9508 = 1_946  # $35C6..$35C9
STATUS_98B2 = 130_553  # $35C9..$35CC
BUF_WAIT_HEAD = 75  # $35CC..$35D5: $2993 A=#$02 and the wait entry
EXIT_TAIL = 13_014  # $35DD..$363D: $114D, $33ED, $106C back to the loop
SFX_HEAD = 283  # $12EC..$12F9: the #$02 $3470 sfx
PASS_TAIL = 2_040  # $1F93..$1289: $3525, $9508, $1302, $127C re-entry
HEAD_BASE = 4_600  # $12D0 dispatch + handler bookkeeping, less the fire ray

TUNE_FRAMES = {0x19: 96.0, 0x28: 24.0}  # $AB50 tune-table note-holds, decoded


def fire_ray_cycles(state, view):
    """$1B40-$1B46: the sights ray's own $1C10 + $1CDD march cycles at fire time."""
    if not view:
        return 0.0
    vec = los.prepare_vector_from_player_sights(
        state,
        view["h_angle"],
        view["v_angle"],
        view["cursor"][0],
        view["cursor"][1],
        state.player,
    )
    do_los = state.mem[0x0C6E] & 0x7F
    _tx, _ty, _ok, cyc = los.check_for_line_of_sight_to_tile(
        vec, state, state.player, do_los
    )
    return float(writeweight.cycles(int(cyc)))


def _frames(cycles):
    return cycles / SETTLE_FG


def viewpoint_settle_frames(state, observer=None, view=None, tune=0x19, eye_view=None):
    """Transfer/u-turn settle: the $357D redraw from ``observer``'s own facing
    (or ``eye_view``), overlapped with ``tune`` (started at $1B84, waited at $35D5)."""
    obs = state.player if observer is None else observer
    if eye_view is None:
        eye_view = {
            "h_angle": int(state.obj_h_angle[obs]),
            "v_angle": int(state.obj_v_angle[obs]),
        }
    from sentinel import occlusioncost

    fg = (
        HEAD_BASE
        + fire_ray_cycles(state, view)
        + REDRAW_HEAD
        + occlusioncost.occlusion_cycles(state, obs)
        + TAB_3700
        + FILL_1090
        + projector.FRAME_CYCLES * projector.render_cost(state, eye_view, obs)
        + STATUS_9508
        + STATUS_98B2
        + BUF_WAIT_HEAD
        + EXIT_TAIL
    )
    return BRACKET_FRAMES + max(_frames(fg), TUNE_FRAMES.get(tune, 96.0))


def uturn_settle_frames(state, observer=None):
    """$1B2F: bearing ^ $80, tune #$28, then the $357D redraw with $245B/$3700
    SKIPPED ($1B2F's ASL leaves $0C51 bit7 set, so $35B5 branches to $35C0);
    no fire ray ($1B18 dispatches code $23 straight to $1B2F)."""
    obs = state.player if observer is None else observer
    flipped = {
        "h_angle": (int(state.obj_h_angle[obs]) ^ 0x80) & 0xFF,
        "v_angle": int(state.obj_v_angle[obs]),
    }
    fg = (
        HEAD_BASE
        + REDRAW_HEAD
        + FILL_1090
        + projector.FRAME_CYCLES * projector.render_cost(state, flipped, obs)
        + STATUS_9508
        + STATUS_98B2
        + BUF_WAIT_HEAD
        + EXIT_TAIL
    )
    return BRACKET_FRAMES + max(_frames(fg), TUNE_FRAMES[0x28])


def action_settle_frames(state, slot, view=None, dither_chain=None, plot_state=None):
    """Create/absorb settle: the $12EC dither replot of ``slot`` plus the pass tail.

    ``slot`` is the created object, or the absorbed one still holding its coords with
    flags $80; ``plot_state`` is the scene the strip plots -- for an absorb, the state
    with the object REMOVED (an $80-flagged object is being erased, not drawn)."""
    from sentinel import dithercost

    visible, left, columns, span, span_cyc = relative.object_screen_span(state, slot)
    strip = 0.0
    dither = 0.0
    if visible:
        strip = projector.FRAME_CYCLES * projector.strip_replot_frames(
            state if plot_state is None else plot_state, slot, left, columns, span
        )
        if dither_chain is None:
            dither = dithercost.dither_cycles_mean(span)
        else:
            dither, _ = dithercost.dither_cycles(span, dither_chain)
    fg = (
        HEAD_BASE
        + fire_ray_cycles(state, view)
        + SFX_HEAD
        + float(writeweight.cycles(int(span_cyc)))
        + strip
        + dither
        + STATUS_9508
        + PASS_TAIL
    )
    return BRACKET_FRAMES + _frames(fg)
