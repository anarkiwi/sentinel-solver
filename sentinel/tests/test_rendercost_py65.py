"""The optional py65 EXACT render-cost backend: it reproduces golden_render_cost.json
by construction (it runs the same emulated plot_world), memoizes per render-relevant
board+view fingerprint, and stays behind RENDER_COST_BACKEND=py65 so the default proxy
path and ROM-free operation are untouched.
"""

import json
import os

import pytest

from sentinel import landscape, projector, rendercost_py65
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_render_cost.json")


def _golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.oracle
def test_exact_backend_matches_golden():
    """render_cost_exact == golden_render_cost.json cycles/19656 for every sweep view
    (exact, by construction: it runs the same emulated plot_world as the generator)."""
    rendercost_py65.reset()
    for key, rec in _golden().items():
        ls, h, v = (int(x) for x in key.split(","))
        got = rendercost_py65.render_cost_exact(landscape.generate(ls), h, v)
        want = rec["cycles"] / projector.FRAME_CYCLES
        assert got == pytest.approx(want, abs=1e-9), key


@pytest.mark.oracle
def test_memoization_hits_and_bounds():
    """A repeated (board, view) is a cache hit; the cache is LRU-bounded."""
    rendercost_py65.reset()
    st = landscape.generate(66)
    a = rendercost_py65.render_cost_exact(st, 0x60, 0x10)
    b = rendercost_py65.render_cost_exact(st.clone(), 0x60, 0x10)
    assert a == b
    assert rendercost_py65._STATS == {"hits": 1, "misses": 1}


@pytest.mark.oracle
def test_wired_backend_replaces_proxy():
    """With RENDER_COST_BACKEND=py65, projector.render_cost returns the exact cost;
    the default proxy differs, confirming the selection hook is live."""
    st = landscape.generate(66)
    view = {"h_angle": 0x60, "v_angle": 0x10}
    proxy = projector.render_cost(st, view)
    os.environ["RENDER_COST_BACKEND"] = "py65"
    try:
        rendercost_py65.reset()
        exact = projector.render_cost(st, view)
    finally:
        del os.environ["RENDER_COST_BACKEND"]
    want = _golden()["66,96,16"]["cycles"] / projector.FRAME_CYCLES
    assert exact == pytest.approx(want, abs=1e-9)
    assert exact != proxy  # the proxy is an approximation, not the exact cost


def test_fingerprint_separates_board_and_view():
    """The memo key changes with the render-relevant board and with the view."""
    st = landscape.generate(0)
    base = rendercost_py65._key(st, 0, 0)
    assert rendercost_py65._key(st, 0x10, 0) != base  # view h
    assert rendercost_py65._key(st, 0, 0x10) != base  # view v
    st2 = st.clone()
    st2.mem[0x0400] ^= 0xFF  # perturb a terrain tile
    assert rendercost_py65._key(st2, 0, 0) != base


def test_default_backend_is_proxy():
    """No env selection => render_cost uses the pure proxy (project_scene), never py65."""
    os.environ.pop("RENDER_COST_BACKEND", None)
    st = landscape.generate(0)
    assert projector._exact_render_cost(st, 0, 0, None) is None
    view = {"h_angle": 0, "v_angle": 0}
    tiles, n = projector.project_scene(st, 0, 0)
    area = sum(
        projector.PER_SCANLINE * t["h"] + projector.PER_PIXEL * t["h"] * t["w"]
        for t in tiles
    )
    base = projector._terrain_poly_base(tiles) + projector._inview_object_base(
        st, tiles
    )
    expect = (n * projector.C_EXAMINE + area + base) / projector.FRAME_CYCLES
    assert projector.render_cost(st, view) == pytest.approx(expect)


def test_rom_absent_falls_back_to_proxy(monkeypatch, capsys):
    """RENDER_COST_BACKEND=py65 with the ROM absent warns once and uses the proxy --
    the ROM-free player must never crash on the exact backend."""
    monkeypatch.setattr(oracle, "available", lambda: False)
    monkeypatch.setenv("RENDER_COST_BACKEND", "py65")
    projector._EXACT_WARNED[0] = False
    st = landscape.generate(0)
    view = {"h_angle": 0, "v_angle": 0}
    got = projector.render_cost(st, view)
    assert got > 0  # proxy value, not a crash
    assert "using proxy" in capsys.readouterr().out
    # observer != player also declines the exact backend (player-view only)
    monkeypatch.setattr(oracle, "available", lambda: True)
    assert projector._exact_render_cost(st, 0, 0, (st.player + 1) & 0x3F) is None
