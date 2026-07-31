"""The $8533 object emulation against the real 6502, and its ROM-free fallback.

``golden_object_cost.json`` holds, per (landscape, view, slot), the ROM's exact
plot_object cycle count and its transformed vertex plottables (no ROM bytes: the
plottables are projections of the board, and the count is a cycle total).
"""

import json
import os

import numpy as np
import pytest

from sentinel import landscape, objmodel, projector, rendercost
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_object_cost.json")
VIEWS = [(0, 0x00, 0x00), (42, 0xC0, 0x00), (335, 0x00, 0x00), (2024, 0x10, 0x00)]

pytestmark = pytest.mark.skipif(
    not objmodel.available(), reason="game image absent: the object model is ROM-gated"
)


def _slots(state):
    return [i for i in range(64) if state.obj_flags[i] < 0x80 and i != state.player][:5]


def _rom_plot_object(img, h, v, target):
    """Run the real $8533 headless and return (cycles, the vertex plottables)."""
    cpu, mem, state = oracle.machine_from_image(img)
    player = mem[0x000B]
    mem[0x006E] = player
    mem[0x09C0 + player] = h
    mem[0x0140 + player] = v
    for addr in (0x001F, 0x005E, 0x0C78, 0x0C1B, 0x0CDE):
        mem[addr] = 0
    mem[0x0CCE] = 0x80
    mem[0x352C] = 0x60
    mem[0x0051], mem[0x0052] = 0xF0, 0x30
    oracle.call(cpu, mem, 0x2993, a=0, state=state)
    state["stop"] = False
    oracle.call(cpu, mem, 0x245B, state=state)
    ret = 0xFFF0
    mem[ret] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.y = target
    cpu.pc = 0x8533
    c0 = cpu.processorCycles
    steps = 0
    while cpu.pc != ret and steps < 5_000_000:
        cpu.step()
        steps += 1
    verts = [
        [
            mem[0x0BA0 + 0x40 + i],
            mem[0xA800 + 0x40 + i],
            mem[0x0A80 + 0x40 + i],
            mem[0x0AE0 + 0x40 + i],
        ]
        for i in range(objmodel.MAX_VERTICES)
    ]
    return cpu.processorCycles - c0 + 6, verts


def _model_plot_object(state, h, v, target):
    from sentinel import objectcost

    zp = objectcost.scratch_zp()
    work = objectcost.workspace()
    left, right = rendercost.edge_tables()
    bufs, _sect = rendercost.buffers(projector.PLAY_MODE)
    cost, _s = objectcost.object_cycles(
        np.frombuffer(state.mem, dtype=np.uint8),
        zp,
        objectcost.model_arrays(),
        state.player,
        target,
        h,
        v,
        0,
        bufs,
        rendercost.SCREEN_TOP,
        rendercost.SCREEN_BOTTOM,
        left,
        right,
        work,
    )
    return cost, work[0]


def _build_golden():
    out = {}
    for ls, h, v in VIEWS:
        img = bytes(oracle.generate(ls))
        state = landscape.generate(ls)
        for target in _slots(state):
            cost, verts = _rom_plot_object(img, h, v, target)
            out[f"{ls},{h},{v},{target}"] = {"cycles": cost, "vertices": verts}
    return out


def _check(data):
    tabs = objmodel.tables()
    ratios = []
    for key, rec in sorted(data.items()):
        ls, h, v, target = (int(x) for x in key.split(","))
        state = landscape.generate(ls)
        cost, vxy = _model_plot_object(state, h, v, target)
        nv = int(
            tabs["vlast"][state.obj_type[target]]
            - tabs["vfirst"][state.obj_type[target]]
        )
        for i in range(nv):
            got = [int(x) for x in vxy[i, :4]]
            assert got == rec["vertices"][i], f"{key} vertex {i}: {got}"
        ratios.append(cost / rec["cycles"])
        assert 0.9 <= cost / rec["cycles"] <= 1.1, f"{key}: {cost} vs {rec['cycles']}"
    assert len(ratios) >= 10
    ratios.sort()
    assert 0.93 <= ratios[len(ratios) // 2] <= 1.05


@pytest.mark.oracle
def test_regenerate_object_cost_golden():
    """Drive the real $8533 across VIEWS, dump the golden, and confirm the emulation
    reproduces every transformed vertex byte for byte."""
    data = _build_golden()
    with open(GOLDEN, "w") as handle:
        json.dump(data, handle, separators=(",", ":"), sort_keys=True)
    _check(data)


def test_object_transform_matches_the_rom():
    with open(GOLDEN) as handle:
        data = json.load(handle)
    assert data
    _check(data)


def test_model_tables_match_the_documented_engine_facts():
    """The per-type vertex and polygon counts are what architecture.md records, so a
    mis-read table is caught before it silently reprices every object."""
    tabs = objmodel.tables()
    want = ((29, 27), (22, 25), (17, 15), (8, 10), (18, 25), (30, 35), (12, 11), (8, 4))
    got = tuple(
        (
            int(tabs["vlast"][i] - tabs["vfirst"][i]),
            int(tabs["plast"][i] - tabs["pfirst"][i]),
        )
        for i in range(8)
    )
    assert got == want


def test_object_term_falls_back_to_the_floor_without_the_image(monkeypatch):
    """No game image => the object term is _inview_object_base's floor, not a crash."""
    state = landscape.generate(42)
    view = {"h_angle": 0xC0, "v_angle": 0x00}
    exact = projector.render_cost(state, view)
    monkeypatch.setattr(projector, "_OBJECT_MODEL_CACHE", [None])
    projector._COST_CACHE.clear()
    floor = projector.render_cost(state, view)
    projector._COST_CACHE.clear()
    assert 0 < floor < exact
