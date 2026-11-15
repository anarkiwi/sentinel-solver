"""Move generation: selection order, board columns, the gates on each move class, the
endgame on the platform, and each move's view against the per-board view lookup."""

import numpy as np
import pytest

from sentinel import actions, memmap as mm, moves as bp
from sentinel.game import Game
from sentinel.terrain import set_tile_byte
from sentinel.views import _Views


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
    empty = bp.keep_best(np.zeros((0, 2), dtype=np.int64), [np.zeros(0)], [], 3)
    assert empty.size == 0
    ident = np.arange(5)[:, None]
    assert len(bp.keep_best(ident, [np.arange(5)], np.zeros(5), 3)) == 3


def test_the_objective_ranks_won_enemies_home_unseen_eye_then_energy():
    fields = {
        "alive": np.array([1, 0, 1, 1, 1, 1]),
        "won": np.array([0, 0, 0, 1, 0, 0]),
        "enemies": np.array([2, 1, 1, 3, 1, 1]),
        "home": np.array([0, 0, 0, 0, 0, 1]),
        "seen": np.array([0, 0, 1, 1, 0, 1]),
        "eye": np.array([9, 9, 9, 0, 3, 0]),
        "energy": np.array([4, 4, 4, 0, 4, 0]),
    }
    alive, keys = bp.lexicographic(fields)
    assert alive.tolist() == [True, False, True, True, True, True]
    live = np.flatnonzero(alive)
    got = bp.keep_best(
        np.arange(5)[:, None], [np.asarray(k)[live] for k in keys], np.zeros(5), 5
    )
    assert live[got].tolist() == [3, 5, 4, 2, 0]


def _moves(st, branch=2, landed=None):
    landed = _Views(st).landed() if landed is None else landed
    return bp.moves(bp.Board.of_states([st]), landed[None], branch)


def test_board_columns_match_the_state():
    st = Game.typed(335).state
    board = bp.Board.of_states([st])
    assert board.player[0] == st.player and board.energy[0] == st.energy
    assert board.eye[0] == st.eye_z()
    assert [s for s in range(mm.NUM_SLOTS) if board.occupied[0, s]] == (
        st.occupied_slots()
    )
    assert board.fuel().shape == (1, mm.N, mm.N) and board.fuel().max() > 0


def test_each_class_is_capped_by_branch_and_one_wait_stands_on_the_player():
    st = Game.typed(42).state
    landed = _Views(st).landed()
    got = _moves(st, landed=landed)
    kind, tile, k = got["kind"], got["tile"], got["k"]
    assert (kind == bp.WAIT).sum() == 1 and (got["parent"] == 0).all()
    assert (np.bincount(kind, minlength=5)[:3] <= 2).all()
    hop = kind == bp.HOP
    assert hop.any() and (k[hop] <= bp.MAX_BOULDERS).all()
    here = st.player_xy()[0] * mm.N + st.player_xy()[1]
    assert tile[kind == bp.WAIT][0] == here and here not in tile[kind < bp.WAIT]
    assert all(landed[t // mm.N, t % mm.N] for t in tile[kind == bp.ABSORB])
    assert _moves(st, 0)["kind"].tolist() == [bp.WAIT, bp.RELOCATE]


def test_a_relocation_needs_the_toll_and_nothing_else_while_the_sentinel_stands():
    st = Game.typed(42).state
    nothing = np.zeros((mm.N, mm.N), dtype=bool)
    assert (_moves(st)["kind"] == bp.RELOCATE).sum() == 1
    assert (_moves(st, landed=nothing)["kind"] == bp.RELOCATE).sum() == 1
    st.energy = bp.ROBOT_E - 1
    assert not (_moves(st)["kind"] == bp.RELOCATE).any()


def _finished(code):
    """Landscape ``code`` with the Sentinel absorbed off its platform."""
    st = Game.typed(code).state
    flags = st.obj_flags[actions.SENTINEL_SLOT]
    set_tile_byte(st, *st.platform_xy, mm.OBJECT_TILE | (flags & 0x3F))
    st.obj_flags[actions.SENTINEL_SLOT] = 0x80
    st.energy = 2 * bp.ROBOT_E
    return st


def test_with_the_sentinel_gone_nothing_is_absorbed_and_the_platform_takes_a_hop():
    st = _finished(0)
    st.obj_z_height[st.player] = 15
    plat = st.platform_xy[0] * mm.N + st.platform_xy[1]
    got = _moves(st, 1, np.ones((mm.N, mm.N), dtype=bool))
    hop = got["kind"] == bp.HOP
    assert list(zip(got["tile"][hop].tolist(), got["k"][hop].tolist())) == [(plat, 0)]
    assert not (got["kind"] == bp.ABSORB).any()
    assert not (got["kind"] == bp.RELOCATE).any()


def test_with_the_sentinel_gone_the_relocation_is_offered_only_from_the_platform():
    st = _finished(0)
    me = st.player
    st.obj_x[me], st.obj_y[me] = st.platform_xy
    assert (_moves(st)["kind"] == bp.RELOCATE).sum() == 1


@pytest.mark.parametrize("number", [42, 335])
def test_each_aimed_move_takes_the_view_its_board_lookup_gives(number):
    st = Game.typed(number).state
    mems = np.frombuffer(st.mem, dtype=np.uint8)[None]
    cols = bp.views([st], mems, bp.Board.of_states([st]), 2)
    views = _Views(st)
    kinds = cols["kind"].tolist()
    assert kinds.count(bp.WAIT) == 1 and len(kinds) > 2
    for i, kind in enumerate(kinds):
        if kind in (bp.WAIT, bp.RELOCATE):
            assert cols["view"][i].tolist() == [-1] * 4
            continue
        t = int(cols["tile"][i])
        view = views.get((t // mm.N, t % mm.N), band=True)
        assert cols["view"][i].tolist() == [view["h_angle"], view["v_angle"]] + list(
            view["cursor"]
        )


def test_a_hops_later_actions_aim_at_the_stack_from_the_previous_facing():
    st = Game.typed(42).state
    mems = np.frombuffer(st.mem, dtype=np.uint8)[None]
    cols = bp.views([st], mems, bp.Board.of_states([st]), 3)
    steps = bp.action_views([st], cols)
    for i, kind in enumerate(cols["kind"].tolist()):
        if kind != bp.HOP:
            assert steps[i] == (tuple(cols["view"][i].tolist()),)
            continue
        assert steps[i] is None or len(steps[i]) == int(cols["k"][i]) + 2
    hops = [s for s, kind in zip(steps, cols["kind"]) if kind == bp.HOP and s]
    assert any(len(s) > 1 for s in hops)
