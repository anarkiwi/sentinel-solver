"""The numba projector twins reproduce the pure-Python references exactly.

:mod:`sentinel.projector`'s ``_occlusion_visible_py`` and ``_project_scene_py`` are the
bit-exact references (locked to the ROM by ``test_render_cost.py`` and
``test_projector.py``); this pins :mod:`sentinel.projector_jit` to them.
"""

import numpy as np
import pytest

from sentinel import (
    actions,
    landscape,
    memmap as mm,
    pancost,
    projector,
    projector_jit,
    terrain,
)
from sentinel.tests import test_projector

# The boards the golden projector/render-cost tests already exercise.
LANDSCAPES = (0, 42, 66, 335, 777, 2024)


def _mem(state):
    return np.frombuffer(state.mem, dtype=np.uint8)


def _observers(state, limit=4):
    """The player plus the first occupied slots at distinct positions."""
    seen = {(state.obj_x[state.player], state.obj_y[state.player])}
    obs = [state.player]
    for slot in range(mm.NUM_SLOTS):
        if len(obs) >= limit:
            break
        if slot == state.player or state.is_empty(slot):
            continue
        xy = (state.obj_x[slot], state.obj_y[slot])
        if xy in seen:
            continue
        seen.add(xy)
        obs.append(slot)
    return obs


def _assert_same(state, observer):
    ref = projector._occlusion_visible_py(state, observer)
    got = projector_jit.occlusion_table(_mem(state), observer)
    mism = [
        (x, y) for y in range(mm.N) for x in range(mm.N) if bool(got[y, x]) != ref[y][x]
    ]
    assert not mism, f"observer {observer}: {len(mism)} mismatches {mism[:8]}"


@pytest.mark.parametrize("ls", LANDSCAPES)
def test_jit_matches_python_per_observer(ls):
    """Every board, several viewpoint objects: the table is observer-dependent."""
    state = landscape.generate(ls)
    observers = _observers(state)
    assert len(observers) > 1  # more than the player alone
    for observer in observers:
        _assert_same(state, observer)


def _stacked(state):
    """Build boulder/synthoid stacks around the player; returns the object tiles made."""
    px, py = state.obj_x[state.player], state.obj_y[state.player]
    made = []
    for dx, dy in ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (-1, 0), (0, -1), (2, 2)):
        x, y = px + dx, py + dy
        if not 0 <= x < mm.N or not 0 <= y < mm.N:
            continue
        state.energy = 0xFF
        actions.create(state, mm.T_BOULDER, (x, y))
        actions.create(state, mm.T_BOULDER, (x, y))
        actions.create(state, mm.T_ROBOT, (x, y))
        if terrain.tile_byte(state, x, y) >= mm.OBJECT_TILE:
            made.append((x, y))
    return made


@pytest.mark.parametrize("ls", LANDSCAPES)
def test_jit_matches_python_with_object_tiles(ls):
    """Object tiles (byte >= $C0) drive the flags-chain walk in the tz build."""
    state = landscape.generate(ls)
    tiles = _stacked(state)
    assert tiles, "fixture produced no object tile"
    deep = [
        (x, y)
        for x, y in tiles
        if state.obj_flags[terrain.tile_byte(state, x, y) & 0x3F] >= 0x40
    ]
    assert deep, "fixture produced no stacked object (flags chain never walked)"
    for observer in _observers(state):
        _assert_same(state, observer)


def test_dispatcher_matches_reference_with_and_without_jit(monkeypatch):
    """The dispatcher returns the reference answer on both paths, as a list-of-lists of
    bools (consumers index it and compare it whole)."""
    state = landscape.generate(42)
    ref = projector._occlusion_visible_py(state)
    jit = projector._occlusion_visible(state)
    monkeypatch.setattr(projector, "_HAVE_JIT", False)
    fallback = projector._occlusion_visible(state)
    assert jit == ref
    assert fallback == ref
    assert isinstance(jit[0][0], bool)


# Angles chosen so view_angle = (h+$20)&$FF lands in each of the four $2665 quadrants.
QUADRANT_ANGLES = (0x00, 0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xC0, 0xD0, 0xF0)
PITCHES = (0x00, 0x04, 0x08, 0xF0, 0xF8)
MODES = (projector.PLAY_MODE,) + tuple(sorted(set(pancost.PAN_MODE)))


def _views(state):
    """(setup, key) for every quadrant x pitch x buffer mode."""
    for h in QUADRANT_ANGLES:
        for v in PITCHES:
            for mode in MODES:
                setup = projector._setup(state, h, v, state.player, mode)
                yield setup, (h, v, mode, setup["quadrant"])


def _scene_cases():
    """(state, label) over the golden VIEWS boards, plain and with object stacks."""
    for ls in LANDSCAPES:
        yield landscape.generate(ls), f"ls{ls}"
    for ls in LANDSCAPES[:3]:
        state = landscape.generate(ls)
        tiles = _stacked(state)
        assert tiles, f"ls{ls}: fixture produced no object tile"
        yield state, f"ls{ls}+objects"


@pytest.mark.parametrize("ls", LANDSCAPES)
def test_project_scene_jit_matches_python(ls):
    """Full tiles list (every key) and n_examine, over the golden boards and views."""
    state = landscape.generate(ls)
    quadrants = set()
    for setup, key in _views(state):
        quadrants.add(setup["quadrant"])
        ref = projector._project_scene_py(state, setup, state.player)
        got = projector._project_scene_jit(state, setup, state.player)
        assert got == ref, f"{ls} {key}"
    assert quadrants == {0, 1, 2, 3}
    for ls_g, h, v in test_projector.VIEWS:
        if ls_g != ls:
            continue
        setup = projector._setup(state, h, v, state.player)
        assert projector._project_scene_jit(
            state, setup, state.player
        ) == projector._project_scene_py(state, setup, state.player)


def test_project_scene_jit_matches_python_with_object_tiles():
    """Object tiles ($C0+) and stacked flags chains drive _tile_height and the
    $291B object-tile bypass of the occlusion gate."""
    state = landscape.generate(42)
    made = _stacked(state)
    assert made, "fixture produced no object tile"
    seen_object_tile = False
    for setup, key in _views(state):
        ref = projector._project_scene_py(state, setup, state.player)
        assert projector._project_scene_jit(state, setup, state.player) == ref, key
        seen_object_tile |= any(t["tile_byte"] >= mm.OBJECT_TILE for t in ref[0])
    assert seen_object_tile, "no plotted object tile in any view"


@pytest.mark.parametrize("ls", LANDSCAPES)
def test_n_examine_is_identical(ls):
    """The byte-exact $2845 call count (cache hits included) drives the cost model."""
    state = landscape.generate(ls)
    for setup, key in _views(state):
        ref = projector._scan_visible(state, setup)[0]
        got = projector._project_scene_jit(state, setup, state.player)[1]
        assert got == ref, f"{ls} {key}: {got} != {ref}"
        assert isinstance(got, int)


def test_project_scene_jit_matches_python_per_observer():
    """The observer selects both the $2625 setup and the $245B table."""
    state = landscape.generate(777)
    observers = _observers(state)
    assert len(observers) > 1
    for observer in observers:
        for h in (0x00, 0x60, 0xB0):
            setup = projector._setup(state, h, 0x04, observer)
            assert projector._project_scene_jit(
                state, setup, observer
            ) == projector._project_scene_py(state, setup, observer)


def test_project_scene_dispatcher_matches_without_jit(monkeypatch):
    """project_scene keeps its (tiles, n_examine) contract on both paths."""
    state = landscape.generate(0)
    jit = projector.project_scene(state, 0x30, 0x08)
    monkeypatch.setattr(projector, "_HAVE_JIT", False)
    projector._OCCLUSION_CACHE.clear()
    fallback = projector.project_scene(state, 0x30, 0x08)
    assert jit == fallback
    tile = jit[0][0]
    assert isinstance(jit[1], int) and isinstance(tile["h"], int)
    assert isinstance(tile["w"], float) and isinstance(tile["tile"], tuple)


def test_env_constants_stay_tunable(monkeypatch):
    """_SCREEN_H/_W_SCALE/_W_SCREEN/_ROW_HINT reach the kernel as arguments."""
    state = landscape.generate(42)
    setup = projector._setup(state, 0x50, 0x04, state.player)
    base = projector._project_scene_jit(state, setup, state.player)
    monkeypatch.setattr(projector, "_W_SCREEN", 1)
    monkeypatch.setattr(projector, "_SCREEN_H", 4)
    tuned = projector._project_scene_jit(state, setup, state.player)
    assert tuned != base
    assert all(t["w"] <= 1 and t["h"] <= 4 for t in tuned[0])
    assert tuned == projector._project_scene_py(state, setup, state.player)
    monkeypatch.setattr(projector, "_ROW_HINT", 0x11)
    hinted = projector._project_scene_jit(state, setup, state.player)
    assert hinted == projector._project_scene_py(state, setup, state.player)
