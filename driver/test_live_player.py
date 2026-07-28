"""Live-execution parity for the composed player: the aim charge must agree with the
executor's REUSE decision."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver import kbd_aim, live_player  # noqa: E402
from sentinel.phase_player import PhasePlayer  # noqa: E402
from sentinel.game import Game  # noqa: E402
from sentinel.playerbase import TAP_FRAMES  # noqa: E402

_LS42 = 42  # landscape 42, a substrate board for these aim/exec unit tests
_VIEW = {"h_angle": 0x60, "v_angle": 0x35, "cursor": [80, 95]}


class _FakeKbd:
    def __init__(self, bearing=None):
        self._bearing = bearing

    def committed_bearing(self):
        return self._bearing


def _aim_player(bearing, sights_on=True, cursor=(80, 95)):
    game = Game.typed(_LS42)
    player = PhasePlayer(game)
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
