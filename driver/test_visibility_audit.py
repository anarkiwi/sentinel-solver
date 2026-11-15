"""The visibility audit: stacks, observers moved onto the entry RAM, and the tallies of
a board with one boulder built."""

import json

import numpy as np
import pytest

from driver import visibility_audit as va
from sentinel import actions, emu, memmap as mm, terrain
from sentinel.state import State

oracle = pytest.mark.oracle
KNOBS = {"width": 2, "depth": 2, "branch": 1, "commit": 1}


@pytest.fixture(scope="module", name="entry")
def _entry():
    return bytes(emu.enter(0).ram)


def _bare(st):
    """A bare, flat tile no enemy stands next to."""
    for x in range(2, va.TILES - 2):
        for y in range(2, va.TILES - 2):
            b = terrain.tile_byte(st, x, y)
            if b < mm.OBJECT_TILE and not b & 0x0F:
                return x, y
    raise AssertionError("no bare flat tile")


def _with_boulder(entry):
    st = State.from_mem(entry)
    tile = _bare(st)
    st.energy = 20
    assert actions.create(st, mm.T_BOULDER, tile) is not None
    return st, tile


@oracle
def test_a_stack_reads_top_down_and_a_bare_tile_its_byte(entry):
    st, tile = _with_boulder(entry)
    assert isinstance(va.stack(State.from_mem(entry), *tile), int)
    (top,) = va.stack(st, *tile)
    assert top[0] == mm.T_BOULDER
    me = State.from_mem(entry)
    assert va.stack(me, *me.tile_of(me.player))[0][0] == mm.T_ROBOT


@oracle
def test_observers_move_onto_the_entry_ram(entry):
    now = State.from_mem(entry)
    me, slot = now.player, now.free_slots()[0]
    for row in va.OBSERVER_ROWS:
        now.mem[row + slot] = now.mem[row + me]
    now.obj_x[slot] = (now.obj_x[me] + 3) % va.TILES
    now.player = slot
    then = va.observed(entry, now)
    assert then.player == slot
    assert then.tile_of(slot) == now.tile_of(slot)
    assert then.obj_flags[slot] == now.obj_flags[slot]
    assert (
        bytes(then.mem[mm.TILES_TABLE : mm.TILES_TABLE + 0x400])
        == entry[mm.TILES_TABLE : mm.TILES_TABLE + 0x400]
    )


@oracle
def test_the_entry_table_answers_its_own_ram(entry):
    t = va.compare(entry, entry)
    assert t["changed"] == 0 and t["unchanged"] == va.TILES**2
    assert t["enemy_diff"] == t["player_diff"] == 0 and not t["diff_tiles"]
    assert 0 < t["enemy"] <= t["unchanged"]


@oracle
def test_a_built_boulder_changes_one_tile_and_is_never_queried(entry):
    st, tile = _with_boulder(entry)
    t = va.compare(entry, bytes(st.mem))
    assert (t["changed"], t["unchanged"]) == (1, va.TILES**2 - 1)
    assert tuple(tile) not in map(tuple, t["diff_tiles"])
    assert len(t["diff_tiles"]) <= t["enemy_diff"] + t["player_diff"]


@oracle
def test_a_game_is_cached_and_summarised(tmp_path, monkeypatch):
    entry, rams, won = va.snapshots(0, max_actions=1, **KNOBS)
    assert len(rams) == 1 and rams[0] != entry and not won
    calls = []

    def once(landscape, **knobs):
        calls.append((landscape, knobs))
        return entry, rams, won

    monkeypatch.setattr(va, "snapshots", once)
    first = va.audit(0, KNOBS, str(tmp_path))
    assert va.audit(0, KNOBS, str(tmp_path)) == first and len(calls) == 1
    assert first["moves"][0]["changed"] >= 1
    line = va.summary(first)
    assert line.startswith("ls0") and f"/{va.TILES**2}" in line
    assert va.main(["0", "--out-dir", str(tmp_path)]) == 0
    with open(tmp_path / "visibility_audit.json", encoding="utf-8") as fh:
        assert json.load(fh)[0]["landscape"] == 0
    assert np.load(tmp_path / "snap_0000.npz")["rams"].shape == (1, 0x10000)
