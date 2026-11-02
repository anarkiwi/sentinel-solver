"""The beam player: selection, candidate generation, op building, the pluggable objective,
and CPU-versus-CUDA identity of a played line."""

import numpy as np
import pytest

from sentinel import actions, beam_player as bp, engine, memmap as mm, rows_jit
from sentinel.game import Game
from sentinel.playerbase import _Views
from sentinel.tests.test_kern import SIM, gpu


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("SENTINEL_BACKEND", "cpu")


def test_keep_best_dedupes_on_identity_and_keeps_the_best_of_each():
    identity = np.array([[1, 0], [1, 0], [2, 0], [2, 0], [3, 0]])
    keys = [np.array([5, 9, 7, 7, 1])]
    frames = np.array([10, 10, 30, 20, 5])
    assert bp.keep_best(identity, keys, frames, 10).tolist() == [1, 3, 4]


def test_keep_best_orders_lexicographically_then_by_frames_then_position():
    identity = np.arange(6)[:, None]
    primary = np.array([1, 2, 2, 2, 2, 0])
    second = np.array([0, 0, 5, 5, 5, 9])
    frames = np.array([0, 4, 9, 3, 3, 0])
    got = bp.keep_best(identity, [primary, second], frames, 10)
    assert got.tolist() == [3, 4, 2, 1, 0, 5]
    assert bp.keep_best(identity, [primary, second], frames, 2).tolist() == [3, 4]


def test_keep_best_is_empty_safe_and_width_bounded():
    assert (
        bp.keep_best(
            np.zeros((0, 2), dtype=np.int64), [np.zeros(0)], np.zeros(0), 3
        ).size
        == 0
    )
    ident = np.arange(5)[:, None]
    assert len(bp.keep_best(ident, [np.arange(5)], np.zeros(5), 3)) == 3


def test_the_objective_is_lexicographic_on_the_phase_players_terms():
    fields = {
        "alive": np.array([1, 0, 1]),
        "won": np.array([0, 0, 0]),
        "enemies": np.array([2, 1, 1]),
        "eye": np.array([9, 9, 3]),
        "energy": np.array([4, 4, 4]),
    }
    alive, keys = bp.lexicographic(fields)
    assert alive.tolist() == [True, False, True]
    live = np.flatnonzero(alive)
    got = bp.keep_best(
        np.arange(3)[live, None], [np.asarray(k)[live] for k in keys], np.zeros(2), 3
    )
    assert live[got].tolist() == [2, 0]


def _board(code):
    st = Game.typed(code).state
    return st, bp.Board.of_states([st])


def test_board_columns_match_the_state():
    st, board = _board(335)
    assert board.player[0] == st.player and board.energy[0] == st.energy
    assert board.eye[0] == st.eye_z()
    slots = [s for s in range(mm.NUM_SLOTS) if board.occupied[0, s]]
    assert slots == st.occupied_slots()
    assert board.fuel().shape == (1, mm.N, mm.N) and board.fuel().max() > 0


def test_moves_are_gated_like_the_phase_player_and_always_include_a_wait():
    st = Game.typed(42).state
    board = bp.Board.of_states([st])
    landed = _Views(st).landed()
    got = bp.moves(board, landed[None], 2)
    kind, tile, k, slot = (got[f] for f in ("kind", "tile", "k", "slot"))
    assert kind[-1] == bp.WAIT and (kind == bp.WAIT).sum() == 1
    assert (got["parent"] == 0).all()
    assert len(kind) == len(tile) == len(k) == len(slot)
    assert (np.bincount(kind, minlength=4)[:3] <= 2).all()
    hop = kind == bp.HOP
    assert hop.any() and (k[hop] <= bp.MAX_BOULDERS).all()
    here = board.x[0, board.player[0]] * mm.N + board.y[0, board.player[0]]
    assert tile[kind == bp.WAIT][0] == here and here not in tile[~(kind == bp.WAIT)]
    for t in tile[kind == bp.ABSORB]:
        assert landed[t // mm.N, t % mm.N]
    st.obj_flags[actions.SENTINEL_SLOT] = 0x80
    kind = bp.moves(bp.Board.of_states([st]), landed[None], 2)["kind"]
    assert not (kind == bp.ABSORB).any()


def test_a_branch_cap_of_zero_leaves_only_the_wait():
    st = Game.typed(42).state
    landed = _Views(st).landed()[None]
    assert bp.moves(bp.Board.of_states([st]), landed, 0)["kind"].tolist() == [bp.WAIT]


def test_build_ops_lays_out_each_move_class():
    kind = np.array([bp.HOP, bp.HOP, bp.ABSORB, bp.MOUNT, bp.WAIT])
    tile = np.array([70, 71, 5, 9, 0])
    k = np.array([2, 0, 0, 0, 0])
    slot = np.array([-1, -1, 12, 7, -1])
    view = np.array([[10, 11], [20, 21], [30, 31], [-1, -1], [-1, -1]])
    ops = bp.build_ops(kind, tile, k, slot, view)
    face = rows_jit.FACE
    assert ops[0].tolist()[:5] == [
        [face, 10, 11],
        [rows_jit.CREATE, mm.T_BOULDER, 70],
        [rows_jit.CREATE, mm.T_BOULDER, 70],
        [rows_jit.CREATE, mm.T_ROBOT, 70],
        [rows_jit.TRANSFER, rows_jit.LAST, 0],
    ]
    assert ops[1].tolist()[:3] == [
        [face, 20, 21],
        [rows_jit.CREATE, mm.T_ROBOT, 71],
        [rows_jit.TRANSFER, rows_jit.LAST, 0],
    ]
    assert ops[2].tolist()[:2] == [[face, 30, 31], [rows_jit.ABSORB, 12, 0]]
    assert ops[3, 0].tolist() == [rows_jit.TRANSFER, 7, 0]
    assert (ops[4, :, 0] == rows_jit.NOOP).all() and (
        ops[3, 1:, 0] == rows_jit.NOOP
    ).all()


def test_a_face_op_turns_the_player_and_nothing_else(new_game):
    st = new_game(42).state
    pool = engine.Pool.from_states([st])
    kids, ok = pool.expand([0], engine.pad_ops([[(rows_jit.FACE, 0x44, 0xF1)]]))
    kid = pool.state(kids[0])
    assert ok[0] and kid.obj_h_angle[kid.player] == 0x44
    assert kid.obj_v_angle[kid.player] == 0xF1
    kid.obj_h_angle[kid.player] = st.obj_h_angle[st.player]
    kid.obj_v_angle[kid.player] = st.obj_v_angle[st.player]
    assert bytes(kid.mem) == bytes(st.mem)


def _play(code, **kw):
    game = Game.typed(code)
    player = bp.BeamPlayer(game, **kw)
    return player.run(), player, game


def test_the_beam_is_deterministic_and_reports_a_well_formed_trace():
    one = _play(0, width=3, depth=4, branch=2)
    two = _play(0, width=3, depth=4, branch=2)
    assert one[1].trace == two[1].trace and one[1].frames == two[1].frames
    assert bytes(one[2].state.mem) == bytes(two[2].state.mem)
    trace = one[1].trace
    assert len(trace) == 4 and all(len(t) == 5 for t in trace)
    assert {t[1] for t in trace} <= set(bp.KINDS)
    assert [t[0] for t in trace] == sorted(t[0] for t in trace)


def test_a_pluggable_objective_changes_the_line():
    def broke(fields):
        return fields["alive"] == 1, (-fields["energy"],)

    default = _play(0, width=2, depth=2, branch=2)[1].trace
    plugged = _play(0, width=2, depth=2, branch=2, objective=broke)[1].trace
    assert default != plugged


def test_a_row_that_cannot_be_planned_is_reported(monkeypatch):
    def boom(*_args):
        raise ValueError("boom")

    monkeypatch.setattr(bp, "plan", boom)
    with pytest.raises(ValueError, match="boom"):
        _play(0, width=2, depth=2, branch=1)


def test_a_beam_with_no_surviving_child_stops_on_its_best_row():
    def never(fields):
        return fields["alive"] == 2, (fields["energy"],)

    won, player, _game = _play(0, width=2, depth=3, branch=1, objective=never)
    assert not won and not player.trace


@gpu
@pytest.mark.skipif(SIM, reason="played lines need whole-board runs")
def test_cpu_and_cuda_play_the_same_line(monkeypatch):
    got = []
    for backend in ("cpu", "cuda"):
        monkeypatch.setenv("SENTINEL_BACKEND", backend)
        won, player, game = _play(0, width=4, depth=5, branch=2)
        got.append((won, player.trace, player.frames, bytes(game.state.mem)))
    assert got[0] == got[1]


def test_main_prints_the_line(capsys):
    assert bp.main(["0", "--width", "2", "--depth", "1", "--branch", "1"]) == 1
    assert "landscape 0: lost" in capsys.readouterr().out
