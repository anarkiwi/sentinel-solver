"""The isometric diagram is well-formed SVG over the real generated board."""

import xml.etree.ElementTree as ET

from sentinel import isoview, memmap as mm, terrain
from sentinel.game import Game

NS = "{http://www.w3.org/2000/svg}"


def _svg(**kw):
    return ET.fromstring(isoview.diagram(Game.typed(335).state, **kw))


def test_projection_is_isometric():
    """Screen x tracks x-y, screen y tracks x+y, and height lifts the point."""
    org = (0.0, 0.0)
    assert isoview.project(0, 0, 0, org) == (0.0, 0.0)
    assert isoview.project(1, 1, 0, org)[0] == 0.0
    assert isoview.project(1, 0, 0, org)[0] > 0 > isoview.project(0, 1, 0, org)[0]
    assert isoview.project(0, 0, 3, org)[1] < isoview.project(0, 0, 0, org)[1]


def test_flat_tile_corners_are_level():
    """A flat tile draws as a level quad; a sloped one reads its neighbours."""
    st = Game.typed(335).state
    flat = sloped = 0
    for x in range(32):
        for y in range(32):
            corners = isoview._corners(st, x, y)
            z, slope = terrain.resolve_ground(st, x, y)
            if slope:
                sloped += 1
            else:
                flat += 1
                assert corners == (z, z, z, z)
    assert flat and sloped


def test_diagram_draws_every_tile_and_object():
    """One quad per tile plus one base diamond per object, and the platform ring."""
    st = Game.typed(335).state
    root = _svg()
    polys = root.findall(f".//{NS}polygon")
    assert len(polys) >= 32 * 32 + len(st.occupied_slots())
    text = " ".join(t.text or "" for t in root.findall(f".//{NS}text"))
    assert f"PLATFORM {st.platform_xy[0]},{st.platform_xy[1]}" in text
    assert "YOU" in text and f"E={st.energy}" in text


def test_annotations_render_arrows_and_panel():
    """Each action becomes one arc plus a numbered badge and a panel line."""
    acts = [
        {
            "kind": "human",
            "verb": "absorb",
            "otype_name": "TREE",
            "from": (5, 5),
            "to": (7, 9),
            "energy": 4,
        },
        {
            "kind": "planner",
            "verb": "create",
            "otype_name": None,
            "from": (5, 5),
            "to": (6, 6),
            "energy": 2,
        },
    ]
    root = _svg(title="t", subtitle=["s"], acts=acts, notes=["n"])
    assert len(root.findall(f"{NS}path")) == len(acts)  # markers live under defs
    text = " ".join(t.text or "" for t in root.findall(f".//{NS}text"))
    for token in ("t", "s", "n", "absorb", "create", "(7, 9)", mm.TYPES[mm.T_TREE]):
        assert token in text


def test_cost_layer_loops_cleanly():
    """Every animation track is a well-formed loop: values and keyTimes agree, and
    keyTimes run monotonically from 0 to at most 1 (SMIL drops a track otherwise)."""
    st = Game.typed(335).state
    measured = {"captions": [("a", "b", "c")] * len(isoview.BEATS)}
    root = _svg(overlay=isoview.cost_layer(st, measured, [(3, 4), (9, 9)]))
    tracks = root.findall(f".//{NS}animate")
    assert len(tracks) > 50
    for a in tracks:
        vals = a.get("values").split(";")
        keys = [float(k) for k in a.get("keyTimes").split(";")]
        assert len(vals) == len(keys) and keys == sorted(keys)
        assert keys[0] == 0.0 and keys[-1] <= 1.0
        assert a.get("dur") == f"{isoview.LOOP}s"


def test_cli_writes_svg(tmp_path):
    out = tmp_path / "ls0.svg"
    assert isoview.main(["0", "--out", str(out)]) == 0
    assert ET.parse(out).getroot().tag == f"{NS}svg"
