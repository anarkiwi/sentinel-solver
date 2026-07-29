"""The landscape atlas: the state cache is exact and generator-keyed, metrics are not.

The contract under test is the split -- a cached entry restores the generated board
byte for byte, a generator edit invalidates it, a metric edit cannot, and measuring a
cached range never generates.
"""

import json

import pytest

from sentinel import atlas, landscape, memmap as mm, statecache
from sentinel.game import Game


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test gets an empty cache root and a freshly computed signature."""
    monkeypatch.setenv(statecache.ROOT_ENV, str(tmp_path / "atlas"))
    statecache.signature.cache_clear()
    yield tmp_path
    statecache.signature.cache_clear()


def test_a_cached_entry_restores_the_generated_image_byte_for_byte():
    fresh = statecache.generate(42)
    state, hit = statecache.state_for(42)
    assert hit is False
    again, hit = statecache.state_for(42)
    assert hit is True
    assert bytes(again.mem) == bytes(state.mem) == bytes(fresh.mem)
    assert again.player_xy() == Game.typed(42).player_xy()


def test_a_restored_state_is_an_independent_mutable_board():
    first, _ = statecache.state_for(0)
    first.mem[mm.PLAYER_ENERGY] = 33
    second, hit = statecache.state_for(0)
    assert hit is True
    assert second.energy == 10


def test_regen_bypasses_the_cache_and_rewrites_it():
    statecache.state_for(0)
    with open(statecache.entry_path(0), "wb") as handle:
        handle.write(b"corrupt")
    assert statecache.load(0) is None
    state, hit = statecache.state_for(0)
    assert hit is False and state.energy == 10
    assert statecache.load(0) is not None
    _, hit = statecache.state_for(0, regen=True)
    assert hit is False


def test_a_generator_change_invalidates_but_a_metric_change_cannot(
    tmp_path, monkeypatch
):
    """The signature digests the generator sources only."""
    assert "atlas.py" not in statecache.SOURCES
    assert set(statecache.SOURCES) >= {"landscape.py", "prng.py"}
    monkeypatch.setattr(statecache, "_HERE", str(tmp_path))
    monkeypatch.setattr(statecache, "SOURCES", ("gen.py",))
    source = tmp_path / "gen.py"

    def sign(text):
        source.write_text(text, encoding="utf-8")
        statecache.signature.cache_clear()
        return statecache.signature()

    before = sign("z = 1\n")
    assert sign("z = 1\n") == before
    after = sign("z = 2\n")
    assert after != before


def test_an_entry_written_under_one_signature_is_not_read_under_another(monkeypatch):
    statecache.state_for(7)
    old = statecache.entry_path(7)
    monkeypatch.setattr(statecache, "CACHE_VERSION", statecache.CACHE_VERSION + 1)
    statecache.signature.cache_clear()
    assert statecache.entry_path(7) != old
    assert statecache.load(7) is None


def test_metrics_run_off_the_cache_without_generating(monkeypatch):
    """Adding a metric must never cost a regeneration: warm rows call no generator."""
    atlas.row_for(42)

    def forbidden(_code):
        raise AssertionError("generated a board that was already cached")

    monkeypatch.setattr(statecache, "generate", forbidden)
    monkeypatch.setattr(landscape, "generate", forbidden)
    assert atlas.row_for(42)["enemies"] == 2


def test_a_new_metric_needs_no_regeneration(monkeypatch):
    """Registering a metric after the board was cached still measures it."""
    board, hit = atlas.board_for(110)
    assert hit is False
    monkeypatch.setitem(atlas.METRICS, "tallest_enemy", lambda b: int(b.oz.max()))
    monkeypatch.setattr(statecache, "generate", lambda _c: pytest.fail("regenerated"))
    row = atlas.row_for(110, ["tallest_enemy"])
    assert row["tallest_enemy"] == int(board.oz.max())


@pytest.mark.parametrize("code,enemies", [(0, 1), (42, 2), (110, 3), (335, 7)])
def test_enemy_count_matches_the_model(code, enemies):
    row = atlas.row_for(code)
    game = Game.typed(code)
    assert row["enemies"] == enemies == len(game.enemy_slots())
    assert row["start_energy"] == game.energy
    assert tuple(row["start_tile"]) == game.player_xy()
    assert row["seed"] == landscape.seed_for(code)


def test_pinned_metrics_on_landscape_0():
    """Landscape 0000 is the ROM's fixed board: one Sentinel, player at (8, 17)."""
    row = atlas.row_for(0)
    assert row["enemy_list"] == [
        {"slot": 0, "type": "SENTINEL", "x": 12, "y": 4, "z": 9}
    ]
    assert row["start_tile"] == [8, 17]
    assert (row["start_z"], row["start_eye"], row["start_energy"]) == (5, 5.875, 10)
    assert (row["relief"], row["flat_tiles"]) == (5, 525)
    assert row["landscape_energy"] == 20
    assert row["roughness"] == pytest.approx(0.2767, abs=1e-4)


def test_landscape_energy_is_the_absorbable_pool_excluding_the_player():
    for code in (0, 42, 335):
        state = Game.typed(code).state
        expected = sum(
            mm.ENERGY_IN_OBJECTS[state.obj_type[s]]
            for s in state.occupied_slots()
            if s != state.player
        )
        assert atlas.row_for(code)["landscape_energy"] == expected


def test_heights_resolve_object_tiles_to_their_stack_floor():
    from sentinel import terrain

    board, _ = atlas.board_for(42)
    for x in range(mm.N):
        for y in range(mm.N):
            height, slope = terrain.resolve_ground(board.state, x, y)
            assert (board.heights[x, y], board.slopes[x, y]) == (height, slope)


def test_scan_is_order_preserving_and_parallel_safe():
    codes = [0, 42, 110, 335, 7]
    rows = atlas.scan(codes, ["enemies"], jobs=3)
    assert [r["code"] for r in rows] == codes
    assert rows == atlas.scan(codes, ["enemies"], jobs=1)


def test_metric_selection_limits_the_row():
    row = atlas.row_for(0, ["enemies", "relief"])
    assert list(row) == ["code", "enemies", "relief"]


def test_distance_is_zero_against_itself_and_grows_with_difference():
    ref = atlas.row_for(0)
    assert atlas.distance(ref, ref) == 0.0
    near = dict(ref, roughness=ref["roughness"] * 1.05)
    far = dict(ref, roughness=ref["roughness"] * 2.0)
    assert atlas.distance(ref, near) < atlas.distance(ref, far)


def test_cli_json_output_carries_the_signature(capsys):
    assert atlas.main(["--codes", "0,42", "--format", "json", "--jobs", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["signature"] == statecache.signature()
    assert [r["code"] for r in out["rows"]] == [0, 42]


def test_cli_table_and_like_and_bad_metric(capsys):
    assert atlas.main(["--start", "0", "--stop", "6", "--jobs", "1"]) == 0
    table = capsys.readouterr().out
    assert table.splitlines()[0].split()[:2] == ["code", "seed"]
    args = ["--start", "0", "--stop", "8", "--like", "3", "--top", "2", "--jobs", "1"]
    assert atlas.main(args) == 0
    out = capsys.readouterr().out
    assert "enemy_list" in out and "distance" in out
    with pytest.raises(SystemExit):
        atlas.main(["--codes", "0", "--metrics", "nope"])


def test_codes_outside_the_keypad_range_are_rejected():
    for code in (-1, statecache.MAX_CODE + 1):
        with pytest.raises(ValueError):
            statecache.valid_code(code)
    assert statecache.valid_code(statecache.MAX_CODE) == 9999
