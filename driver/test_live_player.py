"""Live-execution parity for the composed player: the stale-step gate must make
progress, and the aim charge must agree with the executor's REUSE decision."""

import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver import kbd_aim, live_player  # noqa: E402
from sentinel.astar_player import (  # noqa: E402
    AStarPlayer,
    GATE_BODY,
    GATE_TILE,
    PlanStep,
)
from sentinel.game import Game  # noqa: E402
from sentinel.playerbase import TAP_FRAMES  # noqa: E402

_LANDSCAPE = 42
_VIEW = {"h_angle": 0x60, "v_angle": 0x35, "cursor": [80, 95]}


def _step(verb, tile, gate):
    """A plan step carrying the gate rule the search would have recorded."""
    return PlanStep(verb, tile, 310.0, gate, math.inf, math.inf)


def _live(player, name):
    """Bind a LiveMixin method onto a plain sim player (no VICE session)."""
    setattr(
        player, name, types.MethodType(getattr(live_player.LiveMixin, name), player)
    )


class _FakeKbd:
    def __init__(self, bearing=None):
        self._bearing = bearing

    def committed_bearing(self):
        return self._bearing


def _aim_player(bearing, sights_on=True, cursor=(80, 95)):
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game)
    player.kbd = _FakeKbd(bearing)
    st = player.st
    st.mem[kbd_aim.A_SFLAG] = 0x80 if sights_on else 0
    st.mem[kbd_aim.A_CX], st.mem[kbd_aim.A_CY] = cursor
    st.obj_h_angle[st.player] = _VIEW["h_angle"]
    st.obj_v_angle[st.player] = _VIEW["v_angle"]
    live_player.LiveMixin._sync_aim_state(player)
    return player


def test_live_reuse_bearing_is_charged_zero_aim():
    """A step the executor treats as a bearing REUSE (sights live, committed bearing
    == the view's) charges no transfer aim, and only the action latch for an
    absorb whose cursor is already parked -- the model's aim state IS the driver's."""
    player = _aim_player((_VIEW["h_angle"], _VIEW["v_angle"]))
    assert player.last_bearing == (_VIEW["h_angle"], _VIEW["v_angle"])
    assert player._step_aim_frames("transfer", _VIEW) == 0.0
    assert player._aim_frames(_VIEW) == TAP_FRAMES  # no toggle, no pan, cursor parked


def test_uncommitted_or_sights_off_bearing_is_charged_a_full_aim():
    """No committed bearing (or sights off, whose OFF->ON toggle re-centres the
    cursor) is exactly the executor's re-drive: the full aim is charged."""
    for kwargs in ({"bearing": None}, {"bearing": (0x60, 0x35), "sights_on": False}):
        player = _aim_player(**kwargs)
        assert player.last_bearing is None
        assert player._step_aim_frames("transfer", _VIEW) > 0.0


def test_repeated_stale_verdict_terminates_instead_of_livelocking():
    """A step blocked ONLY by the margin re-gates identically forever if the stale
    path just re-plans (the search is a fixpoint on an unchanged board).  The repeat
    must wait -- advancing the world -- and the step then proceeds on the raw budget."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game, time_budget=0.01, node_budget=1)
    step = _step("boulder", (9, 8), GATE_TILE)
    player.plan = [step]
    player._pi = 0
    player._search = lambda margin_k=None: [step]  # fixpoint: same head every time
    player._view_for = lambda tile: _VIEW
    player._step_aim_frames = lambda verb, view: 100.0
    player._settle = lambda verb, view=None, observer=None: 210.0
    # margin-only block: >= the 310 raw budget, < budget + margin, whatever sigma is
    budget = 310.0
    player._player_window = lambda exclude=None: budget + 0.5 * player._margin(0)
    # a build re-gates on the TILE window the plan gated it with, not the body's
    player._gaze_window = lambda tile, exposed=None: budget + 0.5 * player._margin(0)
    player.live_log = lambda msg: None
    _live(player, "_plan_step_stale")
    waits, fired = [], []
    player._wait = lambda: waits.append(len(fired))
    player._fire = lambda verb, tile, view: fired.append((verb, tuple(tile))) or True

    for _ in range(4):
        player._tick()
        if fired:
            break
    assert waits, "the repeated stale verdict never advanced the world"
    assert fired == [(step.verb, step.tile)], "the stale step never progressed"


def test_stale_step_still_blocked_on_the_raw_budget_keeps_waiting():
    """The release is margin-only: a step the raw budget does not cover stays stale
    however often it repeats, and each repeat advances the world."""
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game, time_budget=0.01, node_budget=1)
    step = _step("boulder", (9, 8), GATE_TILE)
    player.plan = [step]
    player._search = lambda margin_k=None: [step]
    player._view_for = lambda tile: _VIEW
    player._step_aim_frames = lambda verb, view: 100.0
    player._settle = lambda verb, view=None, observer=None: 210.0
    player._player_window = lambda exclude=None: 10.0
    player._gaze_window = lambda tile, exposed=None: 10.0
    player.live_log = lambda msg: None
    _live(player, "_plan_step_stale")
    waits, fired = [], []
    player._wait = lambda: waits.append(1)
    player._fire = lambda verb, tile, view: fired.append(verb) or True

    for _ in range(4):
        player._tick()
    assert not fired
    assert len(waits) >= 2  # every repeat spends real world time, never a spin


def test_stale_gate_uses_the_window_the_plan_gated_the_step_with():
    """A build is re-gated on its TARGET TILE's window (what ``_pick_hop``/``_hop_exec``
    gated it with), an absorb on the player's own body window (what ``_c_absorb`` and
    ``_reclaim_one`` gated it with).  Re-deriving a stricter body-window rule for every
    verb refuses steps the plan never promised, and ``_search`` re-derives the same head,
    so the refusal repeats until the ladder concedes a hyperspace -- the ls42 live loss.
    """
    game = Game.new(_LANDSCAPE)
    player = AStarPlayer(game, time_budget=0.01, node_budget=1)
    player._view_for = lambda tile: _VIEW
    player._step_aim_frames = lambda verb, view: 100.0
    player._settle = lambda verb, view=None, observer=None: 100.0
    player.live_log = lambda msg: None
    _live(player, "_plan_step_stale")
    wide, narrow = 100000.0, 1.0
    player._player_window = lambda exclude=None: narrow
    player._gaze_window = lambda tile, exposed=None: wide
    build = _step("boulder", (9, 8), GATE_TILE)
    xfer = _step("transfer", (9, 8), GATE_TILE)
    strike = _step("absorb", (9, 8), GATE_BODY)
    assert not player._plan_step_stale(build, _VIEW)
    assert not player._plan_step_stale(xfer, _VIEW)
    assert player._plan_step_stale(strike, _VIEW)
    player._player_window = lambda exclude=None: wide
    player._gaze_window = lambda tile, exposed=None: narrow
    assert player._plan_step_stale(build, _VIEW)
    assert not player._plan_step_stale(strike, _VIEW)
