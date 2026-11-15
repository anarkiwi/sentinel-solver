"""Live tick capture and replay: a saved tick rebuilds the player the live one was, and
the pan-pass timer splits a pass at its ROM boundaries in raster-resolved frames."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driver import live_player, replay_tick  # noqa: E402
from sentinel.game import Game  # noqa: E402
from sentinel.phase_player import PhasePlayer  # noqa: E402

_LS = 42


def _saved(tmp_path, bearing=(0x60, 0xF5), cursor=(70, 90)):
    player = PhasePlayer(Game.typed(_LS))
    player.last_bearing, player.cursor = bearing, list(cursor)
    live_player.save_tick(str(tmp_path), 3, player)
    return player, os.path.join(str(tmp_path), "t0003.npz")


def test_tick_round_trip_restores_image_and_aim_state(tmp_path):
    player, path = _saved(tmp_path)
    back = live_player.load_tick(path)
    assert bytes(back.st.mem) == bytes(player.st.mem)
    assert back.cursor == [70, 90] and back.last_bearing == (0x60, 0xF5)
    _player, path = _saved(tmp_path, bearing=None)
    assert live_player.load_tick(path).last_bearing is None


def test_image_diff_lists_exactly_the_changed_bytes(tmp_path):
    _player, path = _saved(tmp_path)
    with np.load(path) as z:
        mem = np.asarray(z["mem"]).copy()
    assert not replay_tick.image_diff(mem, _LS)
    old = int(mem[0x0C7C])
    mem[0x0C7C] ^= 0xFF
    assert replay_tick.image_diff(mem, _LS) == [(0x0C7C, old ^ 0xFF, old)]


def test_first_action_matches_a_replayed_tick(tmp_path):
    _player, path = _saved(tmp_path, bearing=None, cursor=(80, 95))
    _p, offset, act = replay_tick.first_action((path, 0, 8))
    assert offset == 0 and act is not None
    played = replay_tick.replay(path, ticks=8, verbose=False)
    _f, verb, tile, *_ = played.trace[0]
    assert act == (verb, tuple(tile))


class _Ex:
    def __init__(self, bm):
        self.bm = bm

    def frames(self):
        return self.bm.frame

    def rd(self, _addr):
        return 0x80


class _PassBM:
    """A monitor whose run_until_pc jumps a fixed number of raster lines per stop."""

    STEPS = {
        live_player.PC_PASS: 3 * 312 + 100,
        live_player.PC_PAN_CALL: 156,
        0x365D: 312 * 20,
    }

    def __init__(self):
        self.line = 0x105  # $D011 bit7 set: lines 256..311
        self.frame = 0

    def run_until_pc(self, pc, timeout=None):  # pylint: disable=unused-argument
        total = self.line + self.STEPS[pc]
        self.frame += total // 312
        self.line = total % 312

    def advance_instructions(self, _n):
        pass

    def mem_get(self, start, end, bank=0):  # pylint: disable=unused-argument
        assert (start, end, bank) == (0xD011, 0xD012, "io")
        return bytes(((self.line >> 1) & 0x80 | 0x1B, self.line & 0xFF))


def test_pass_timer_splits_scroll_update_and_pan_in_raster_frames():
    bm = _PassBM()
    kbd = live_player.MeasuringKbdDriver.__new__(live_player.MeasuringKbdDriver)
    kbd.bm, kbd._ex, kbd.passes, kbd._label = bm, _Ex(bm), [], "p7"
    kbd._run_pass()
    label, frozen, scroll, update, pan = kbd.passes[0]
    assert (label, frozen) == ("p7", 1)
    assert (scroll, update, pan) == (round(3 + 100 / 312, 3), 0.5, 20.0)


def test_cli_replays_jitters_and_diffs_saved_ticks(tmp_path, capsys):
    _saved(tmp_path, bearing=None, cursor=(80, 95))
    d = str(tmp_path)
    assert replay_tick.main([d, "--only", "3", "--diff", str(_LS), "--ticks", "1"]) == 0
    assert "== t0003.npz" in capsys.readouterr().out
    assert (
        replay_tick.main([d, "--jitter", "0", "60", "--ticks", "8", "--workers", "1"])
        == 0
    )
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("t0003.npz ") and line.split()[1] in ("STABLE", "FLIPS")
