"""The recorded enemy clock threads log -> fixture -> offline audit.

``_extract`` carries a ``watch_play/3`` log's pre-action ``enemies`` clock into each
fixture event as ``enemy_clock``; ``human_audit._load_truth`` then sources true enemy
phase from the fixture itself, so an offline sim replay needs no live VICE ``_truth``.
"""

import base64
import json

from sentinel import memmap as mm
from sentinel.tests import human_audit
from sentinel.tests.fixtures.human_wins import _extract


def _img_with(objs):
    """A bare seed-0 board (all slots empty) with ``objs`` = {slot: (x, y, type)}."""
    mem = bytearray(_extract._base_mem(0))
    for slot, (x, y, otype) in objs.items():
        mem[mm.OBJECTS_FLAGS + slot] = 0x00  # occupied, on ground
        mem[mm.OBJECTS_X + slot] = x
        mem[mm.OBJECTS_Y + slot] = y
        mem[mm.OBJECTS_TYPE + slot] = otype
    return mem


def _record(mem, clock=None, bracket=None, energy=10):
    rec = {
        "player": {"slot": 62, "x": 5, "y": 5, "z": 3, "zf": 0, "hang": 0, "vang": 0},
        "energy": energy,
        "do_los": 0,
        "cursor": [40, 40],
        "mem": base64.b64encode(bytes(mem[:0x0D00])).decode("ascii"),
    }
    if clock is not None:
        rec["enemies"] = clock
        rec["cooldown_bresenham"] = 137
        rec["cooldown_gate"] = 2
    if bracket:
        rec["bracket"] = bracket
    return rec


def _write_log(tmp_path, initial, bracket):
    path = tmp_path / "play.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": "watch_play/3"}) + "\n")
        fh.write(json.dumps(initial) + "\n")
        fh.write(json.dumps(bracket) + "\n")
    return str(path)


_CLOCK = [
    {
        "slot": 0,
        "type": "SENTRY",
        "tile": [9, 9],
        "h_angle": 128,
        "v_angle": 245,
        "rot_step": 20,
        "rot_cooldown": 0,
        "drain_cooldown": 0,
        "update_cooldown": 7,
    }
]


def test_extract_threads_enemy_clock(tmp_path):
    """A player create bracketed against the pre-state yields an event carrying the
    pre-action clock verbatim, plus the cooldown accumulator."""
    pre = _img_with({62: (5, 5, mm.T_ROBOT)})  # no enemy -> the create is kept
    post = _img_with({62: (5, 5, mm.T_ROBOT), 5: (7, 7, mm.T_BOULDER)})
    log = _write_log(
        tmp_path,
        _record(pre, clock=_CLOCK),
        # the boulder is PAID FOR: _extract._paid_for drops a create that cost nothing
        _record(post, _CLOCK, "post", energy=10 - mm.ENERGY_IN_OBJECTS[mm.T_BOULDER]),
    )

    data = _extract.extract(log, entered_code=0, seed=0)
    assert data["n_events"] == 1
    ev = data["events"][0]
    assert ev["verb"] == "create" and ev["otype"] == mm.T_BOULDER
    assert ev["enemy_clock"] == _CLOCK
    assert ev["cooldown_bresenham"] == 137


def test_extract_omits_clock_for_legacy_logs(tmp_path):
    """A watch_play/2 log (no ``enemies``) still extracts, without the new fields."""
    pre = _img_with({62: (5, 5, mm.T_ROBOT)})
    post = _img_with({62: (5, 5, mm.T_ROBOT), 5: (7, 7, mm.T_BOULDER)})
    log = _write_log(
        tmp_path,
        _record(pre),
        _record(post, bracket="post", energy=10 - mm.ENERGY_IN_OBJECTS[mm.T_BOULDER]),
    )

    ev = _extract.extract(log, entered_code=0, seed=0)["events"][0]
    assert "enemy_clock" not in ev and "cooldown_bresenham" not in ev


def test_load_truth_falls_back_to_fixture_clock(monkeypatch):
    """With no live ``_truth.json``, the audit sources true phase from the fixture."""
    fake = {"landscape": 0, "events": [{"enemy_clock": _CLOCK}, {}]}
    monkeypatch.setattr(human_audit, "_load", lambda name: fake)
    truth = human_audit._load_truth("ls_absent.json")
    assert truth == {0: {0: _CLOCK[0]}}
