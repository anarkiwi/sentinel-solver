"""Permanent human-win fixtures exercising the sim's keyboard-aim / buildable path.

Three recorded HUMAN wins (landscapes seed 0 / 66 / 821; entered codes 0 / 42 /
335) were distilled to compact, non-copyrighted STATE fixtures
(``tests/fixtures/human_wins/ls*.json`` -- object coordinates/types/heights,
player position, aim angles; NO raw ``mem`` and NO terrain -- see
``_extract.py``).  The terrain is regenerated at test time with the audited
byte-exact :func:`sentinel.landscape.generate`.

For every player ACTION these tests reconstruct the exact PRE-action ``State`` and
check the sim against ground truth:

* create   -> :func:`sentinel.actions.can_create` is True                 (GREEN)
* absorb   -> the target tile holds an absorbable object                   (GREEN)
* transfer -> the target tile holds a robot to transfer into              (GREEN)

and a buildability validation:

* the keyboard-aim landability oracle (:func:`sentinel.los.landable_view` /
  ``landable_views``) must contain every tile the human built or absorbed on.  The
  oracle enumerates the sights cursor at the ROM's true 1px resolution ($9965/$9994
  move +/-1px; each 1px step a distinct ray via prepare_vector_from_player_sights
  $1C10) over a 64px window bit-equivalent to the full cursor range, so every GENUINE
  player build/absorb is landable (the old 9px notch grid false-negatived far/
  adjacent tiles).  The extractor (``_extract.py``) now drops Sentinel-SPAWNED trees
  (an enemy discharging absorbed energy plants a TREE at a RANDOM tile, ROM
  ``consider_discharging_enemy_energy $1A5D``, NOT gated by the player's sights), so
  every event in the fixtures is a real player action and this oracle covers them all
  with NO gap.
"""

import json
import os

import pytest

from sentinel import actions, landscape, los, memmap as mm
from sentinel.state import State
from sentinel.terrain import set_tile_byte, top_object

_FIX_DIR = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "human_wins")
FIXTURES = ("ls0.json", "ls42.json", "ls335.json")

# ---------------------------------------------------------------------------
# fixture loading + PRE-action State reconstruction
# ---------------------------------------------------------------------------
_CACHE = {}
_BASE = {}


def _load(name):
    if name not in _CACHE:
        with open(os.path.join(_FIX_DIR, name), encoding="utf-8") as fh:
            _CACHE[name] = json.load(fh)
    return _CACHE[name]


def _base_mem(seed):
    """A generated board with all objects STRIPPED back to bare terrain (object
    tiles reverted, every slot empty) -- the canvas the fixture objects overlay.
    Terrain heights/slopes and the memmap scalars (platform xy, prng...) stay."""
    if seed not in _BASE:
        gen = landscape.generate(seed)
        for s in range(mm.NUM_SLOTS):  # objects sit only on flat tiles -> h<<4 reverts
            if not (gen.obj_flags[s] & 0x80) and gen.obj_flags[s] < 0x40:
                set_tile_byte(
                    gen, gen.obj_x[s], gen.obj_y[s], (gen.obj_z_height[s] << 4) & 0xFF
                )
        for s in range(mm.NUM_SLOTS):
            gen.obj_flags[s] = 0x80
        _BASE[seed] = bytes(gen.mem)
    return bytearray(_BASE[seed])


def state_from_event(ev, seed):
    """Reconstruct the exact PRE-action :class:`State`: generated terrain + the
    fixture's occupied objects, the player slot/energy set, aim angles applied."""
    st = State(_base_mem(seed))
    objs = ev["objects"]
    below = {o[6] & 0x3F for o in objs if o[6] >= 0x40}
    for slot, x, y, zh, zf, otype, flags in objs:
        st.obj_x[slot] = x
        st.obj_y[slot] = y
        st.obj_z_height[slot] = zh
        st.obj_z_frac[slot] = zf
        st.obj_type[slot] = otype
        st.obj_flags[slot] = flags
    for slot, x, y, zh, zf, otype, flags in objs:
        if slot not in below:  # topmost object owns the tile byte
            set_tile_byte(st, x, y, mm.OBJECT_TILE | slot)
    pl = ev["player"]
    st.player = pl["slot"]
    st.energy = ev["energy"]
    st.obj_h_angle[pl["slot"]] = pl["hang"]
    st.obj_v_angle[pl["slot"]] = pl["vang"]
    return st


def _verb_params(verbs):
    """(name, idx) params for every event whose verb is in `verbs`."""
    out = []
    for name in FIXTURES:
        for i, ev in enumerate(_load(name)["events"]):
            if ev["verb"] in verbs:
                out.append(pytest.param(name, i, id=f"{name[:-5]}-ev{i}-{ev['verb']}"))
    return out


def _landable_params():
    """(name, idx) for every create/absorb event (all are real player actions now
    that _extract.py drops Sentinel-spawned trees, so every one must be landable)."""
    out = []
    for name in FIXTURES:
        for i, ev in enumerate(_load(name)["events"]):
            if ev["verb"] == "transfer":
                continue
            out.append(pytest.param(name, i, id=f"{name[:-5]}-ev{i}-{ev['verb']}"))
    return out


# ---------------------------------------------------------------------------
# fixture integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_reconstructs(name):
    """Every event reconstructs a PRE-action State whose player is a live object,
    proving the recovered generate() seed + distilled objects are self-consistent."""
    data = _load(name)
    assert data["n_events"] == len(data["events"]) > 0
    for ev in data["events"]:
        st = state_from_event(ev, data["landscape"])
        slot = ev["player"]["slot"]
        assert not st.is_empty(slot)
        assert st.player == slot


# ---------------------------------------------------------------------------
# GREEN: per-action mechanics against the real human pre-state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,idx", _verb_params(("create",)))
def test_can_create(name, idx):
    ev = _load(name)["events"][idx]
    st = state_from_event(ev, _load(name)["landscape"])
    assert actions.can_create(
        st, ev["otype"], tuple(ev["target"])
    ), f"{name} ev{idx}: human create of {ev['otype']} at {ev['target']} infeasible"


@pytest.mark.parametrize("name,idx", _verb_params(("absorb",)))
def test_absorb_target_is_absorbable(name, idx):
    ev = _load(name)["events"][idx]
    st = state_from_event(ev, _load(name)["landscape"])
    tx, ty = ev["target"]
    top = top_object(st, tx, ty)
    assert top is not None and actions.can_absorb(
        st, top
    ), f"{name} ev{idx}: no absorbable object at human absorb target {ev['target']}"


@pytest.mark.parametrize("name,idx", _verb_params(("transfer",)))
def test_transfer_target_is_robot(name, idx):
    ev = _load(name)["events"][idx]
    st = state_from_event(ev, _load(name)["landscape"])
    tx, ty = ev["target"]
    top = top_object(st, tx, ty)
    assert (
        top is not None and st.obj_type[top] == mm.T_ROBOT
    ), f"{name} ev{idx}: transfer target {ev['target']} is not a robot"


# ---------------------------------------------------------------------------
# buildability oracle exhaustiveness -- every player build/absorb tile is landable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,idx", _landable_params())
def test_build_tile_is_aim_landable(name, idx):
    """Every tile the human built/absorbed on must be reachable by SOME keyboard aim
    (``landable_view(v_band=True)``, the targeted single-tile form -- it sweeps the cheap
    $F5 plane first and only falls to the full pitch band when needed, so each event is one
    cheap march in the common case rather than a full-board sweep)."""
    ev = _load(name)["events"][idx]
    st = state_from_event(ev, _load(name)["landscape"])
    pl = ev["player"]
    target = tuple(ev["target"])
    view = los.landable_view(st, target, pl["slot"], eye_z=pl["z"], v_band=True)
    assert view is not None, (
        f"{name} ev{idx} {ev['verb']} target {target} from ({pl['x']},{pl['y']}) "
        f"eye_z={pl['z']} is NOT aim-landable"
    )
