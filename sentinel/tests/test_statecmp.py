"""Tests for the shared sim/emulator state comparator."""

from sentinel import landscape, memmap as mm, statecmp as sc


def test_schema_addresses_unique_and_bounded():
    addrs = [a for a, _, _ in sc.FIELDS]
    assert len(set(addrs)) == len(addrs)
    assert all(0 <= a <= sc.MAX_ADDR for a in addrs)
    assert sc.MAX_ADDR == mm.COOLDOWN_BRESENHAM


def test_every_tier_present():
    tiers = {t for _, _, t in sc.FIELDS}
    assert tiers == set(sc.TIERS)


def test_identical_images_have_no_divergence():
    st = landscape.generate(53)
    assert not sc.diff(st.mem, st.mem)


def test_diff_decodes_object_and_scalar_fields():
    st = landscape.generate(53)
    a = bytearray(st.mem)
    b = bytearray(st.mem)
    b[mm.OBJECTS_X + 5] = (a[mm.OBJECTS_X + 5] + 3) & 0xFF
    b[mm.PLAYER_ENERGY] = (a[mm.PLAYER_ENERGY] + 1) & 0xFF
    divs = {d.label: d for d in sc.diff(a, b)}
    assert "obj[5].x" in divs and divs["obj[5].x"].tier == sc.CORE
    assert divs["obj[5].x"].b == (a[mm.OBJECTS_X + 5] + 3) & 0xFF
    assert "player_energy" in divs


def test_tier_filter_and_grouping():
    st = landscape.generate(53)
    a = bytearray(st.mem)
    b = bytearray(st.mem)
    b[mm.CURSOR] = (a[mm.CURSOR] + 1) & 0xFF  # SWEEP
    b[mm.PLAYER_ENERGY] = (a[mm.PLAYER_ENERGY] + 1) & 0xFF  # CORE
    core_only = sc.diff(a, b, tiers={sc.CORE})
    assert [d.label for d in core_only] == ["player_energy"]
    grouped = sc.by_tier(sc.diff(a, b))
    assert [d.label for d in grouped[sc.SWEEP]] == ["cursor"]
    assert [d.label for d in grouped[sc.CORE]] == ["player_energy"]


def test_tiles_cover_full_board():
    tile_labels = {lbl for _, lbl, _ in sc.FIELDS if lbl.startswith("tile[")}
    assert len(tile_labels) == mm.N * mm.N
