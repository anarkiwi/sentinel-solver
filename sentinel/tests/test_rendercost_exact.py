"""The exact render-cost backend reproduces golden_render_cost.json by construction
(it runs the same emulated plot_world), memoizes per board+view fingerprint, and stays
behind RENDER_COST_BACKEND=exact so the default proxy and ROM-free operation hold.
"""

import json
import os

import pytest

from sentinel import landscape, occlusioncost, projector, rendercost_exact, settlecost
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_render_cost.json")


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.oracle
def test_exact_backend_matches_golden():
    """render_cost_exact == golden_render_cost.json cycles/19656 for every sweep view
    (exact, by construction: it runs the same emulated plot_world as the generator)."""
    rendercost_exact.reset()
    for key, rec in _golden().items():
        ls, h, v = (int(x) for x in key.split(","))
        got = rendercost_exact.render_cost_exact(landscape.generate(ls), h, v)
        want = rec["cycles"] / projector.FRAME_CYCLES
        assert got == pytest.approx(want, abs=1e-9), key


@pytest.mark.oracle
def test_memoization_hits_and_bounds():
    """A repeated (board, view) is a cache hit; the cache is LRU-bounded."""
    rendercost_exact.reset()
    st = landscape.generate(66)
    a = rendercost_exact.render_cost_exact(st, 0x60, 0x10)
    b = rendercost_exact.render_cost_exact(st.clone(), 0x60, 0x10)
    assert a == b
    assert rendercost_exact._STATS == {"hits": 1, "misses": 1}


@pytest.mark.oracle
def test_threaded_batch_equals_serial():
    """render_costs_exact over threads gives each view's serial cost, one prepare per board."""
    st = landscape.generate(66)
    views = [(h, v) for h in range(0, 256, 32) for v in (0x00, 0x10)]
    rendercost_exact.reset()
    serial = [rendercost_exact.render_cost_exact(st, h, v) for h, v in views]
    rendercost_exact.reset()
    assert rendercost_exact.render_costs_exact(st, views, workers=8) == serial
    assert rendercost_exact._PREPARED_STATS["misses"] == 1


@pytest.mark.oracle
def test_wired_backend_replaces_proxy():
    """With RENDER_COST_BACKEND=exact, projector.render_cost returns the exact cost;
    the default proxy differs, confirming the selection hook is live."""
    st = landscape.generate(66)
    view = {"h_angle": 0x60, "v_angle": 0x10}
    proxy = projector.render_cost(st, view)
    os.environ["RENDER_COST_BACKEND"] = "exact"
    try:
        rendercost_exact.reset()
        exact = projector.render_cost(st, view)
    finally:
        del os.environ["RENDER_COST_BACKEND"]
    want = _golden()["66,96,16"]["cycles"] / projector.FRAME_CYCLES
    assert exact == pytest.approx(want, abs=1e-9)
    assert exact != proxy  # the proxy is an approximation, not the exact cost


def test_fingerprint_separates_board_and_view():
    """The memo key changes with the render-relevant board and with the view."""
    st = landscape.generate(0)
    base = rendercost_exact._key(st, 0, 0)
    assert rendercost_exact._key(st, 0x10, 0) != base  # view h
    assert rendercost_exact._key(st, 0, 0x10) != base  # view v
    st2 = st.clone()
    st2.mem[0x0400] ^= 0xFF  # perturb a terrain tile
    assert rendercost_exact._key(st2, 0, 0) != base


def test_default_backend_is_proxy():
    """No env selection => render_cost uses the pure proxy (project_scene), never the ROM."""
    os.environ.pop("RENDER_COST_BACKEND", None)
    st = landscape.generate(0)
    assert projector._exact_render_cost(st, 0, 0, None) is None
    view = {"h_angle": 0, "v_angle": 0}
    tiles, _n, exam, fill, walk = projector.project_scene(st, 0, 0)
    setup = projector._setup(st, 0, 0, st.player)
    got, exact = projector.fill_frames(st, setup, fill, projector.PLAY_MODE, st.player)
    base = 0.0 if exact else projector._inview_object_base(st, tiles)
    expect = (exam + walk + got + base) / projector.FRAME_CYCLES
    assert projector.render_cost(st, view) == pytest.approx(expect)


def test_rom_absent_falls_back_to_proxy(monkeypatch, capsys):
    """RENDER_COST_BACKEND=exact with the ROM absent warns once and uses the proxy --
    the ROM-free player must never crash on the exact backend."""
    monkeypatch.setattr(oracle, "available", lambda: False)
    monkeypatch.setenv("RENDER_COST_BACKEND", "exact")
    monkeypatch.setattr(rendercost_exact, "_WARNED", [False])
    st = landscape.generate(0)
    assert projector.render_cost(st, {"h_angle": 0, "v_angle": 0}) > 0
    assert "using proxy" in capsys.readouterr().out
    assert rendercost_exact.backend() is None


@pytest.mark.oracle
def test_another_slots_view_is_the_roms_redraw_from_it(monkeypatch):
    """A non-player observer is priced as the ROM's redraw with $000B := observer."""
    monkeypatch.setenv("RENDER_COST_BACKEND", "exact")
    rendercost_exact.reset()
    st = landscape.generate(66)
    obs = next(s for s in range(64) if s != st.player and st.obj_flags[s] < 0x80)
    got = projector._exact_render_cost(st, 0x60, 0x10, obs)
    as_player = st.clone()
    as_player.player = obs
    assert got == rendercost_exact.render_cost_exact(as_player, 0x60, 0x10)
    assert rendercost_exact.viewer(st, st.player) is st


@pytest.mark.oracle
@pytest.mark.parametrize("ls", [0, 335, 9795])
def test_prologue_occlusion_is_the_occlusion_model(ls):
    """The ROM's own $245B cycles (the prologue the exact backend runs) equal the
    occlusioncost model's; $3700 is board-dependent, inside its band."""
    st = landscape.generate(ls)
    occlusion, tables = rendercost_exact.prologue_cycles(st)
    assert occlusion == occlusioncost.occlusion_cycles(st)
    assert 1_500_000 < tables < settlecost.TAB_3700


@pytest.mark.oracle
def test_settle_prices_the_prologue_on_the_rom(monkeypatch):
    """Under the exact backend the viewpoint settle takes $245B + $3700 from the ROM."""
    st = landscape.generate(335)
    proxy = settlecost._prologue_cycles(st, st.player)
    monkeypatch.setenv("RENDER_COST_BACKEND", "exact")
    exact = settlecost._prologue_cycles(st, st.player)
    assert exact == sum(rendercost_exact.prologue_cycles(st)) != proxy
