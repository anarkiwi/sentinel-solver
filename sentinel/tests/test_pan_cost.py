"""pancost reproduces one pan_viewpoint notch, cycle-counted in py65.

``golden_pan_cost.json`` holds the exact $10B7 cycle count per (landscape, bearing,
pitch, direction), split into strip-clear / examine / rest, plus the $2845 and $2A24
counts (no ROM bytes). The non-oracle tests check selection exactness and cost.
"""

import json
import math
import os
import statistics

import pytest

from sentinel import landscape, pancost, projector
from sentinel.tests import oracle

GOLDEN = os.path.join(os.path.dirname(__file__), "golden_pan_cost.json")

LANDSCAPES = (0, 42, 335)
PITCHES = (0x00, 0x04, 0xF8)
BEARINGS = tuple(range(0, 256, 32))

_RET = 0xFFF0
_EXAMINE = ((0x2845, 0x295C), (0x9287, 0x93AC), (0x0D4A, 0x0F49), (0x0F4A, 0x1000))


def _in_examine_tree(pc):
    return any(lo <= pc <= hi for lo, hi in _EXAMINE)


def _measure_notch(cpu, mem):
    """Run one pan_viewpoint ($10B7) headless and split its cycles into the strip
    clear ($3912/$38AD), the $2845 examine call-tree and everything else."""
    mem[_RET] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (_RET - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (_RET - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = 0x10B7
    c0 = cpu.processorCycles
    n_examine = n_filled = examine_cycles = clear_cycles = steps = 0
    in_clear = False
    clear_sp = 0
    while cpu.pc != _RET and steps < 20_000_000:
        pc = cpu.pc
        if pc == 0x2845:
            n_examine += 1
        elif pc == 0x2A24 and mem[0x0180 + cpu.x]:
            n_filled += 1
        if pc in (0x3912, 0x38AD) and not in_clear:
            in_clear, clear_sp = True, cpu.sp
        c1 = cpu.processorCycles
        cpu.step()
        d = cpu.processorCycles - c1
        if _in_examine_tree(pc):
            examine_cycles += d
        if in_clear:
            clear_cycles += d
            if cpu.sp > clear_sp:
                in_clear = False
        steps += 1
    return {
        "cycles": cpu.processorCycles - c0,
        "clear_cycles": clear_cycles,
        "examine_cycles": examine_cycles,
        "n_examine": n_examine,
        "n_filled": n_filled,
    }


def _build_golden():
    """{ 'ls,h,v,direction': {cycles, clear/examine cycles, counts} } over the sweep."""
    out = {}
    for ls in LANDSCAPES:
        cpu, mem, state = oracle.generate_machine(ls)
        player = mem[0x000B]
        mem[0x006E] = player
        for addr in (0x001F, 0x005E, 0x0C78, 0x0C1B, 0x0CDE):
            mem[addr] = 0
        mem[0x0CCE] = 0x80  # skip the raytracer's secret-code check
        mem[0x352C] = 0x60  # stub update_sound (foreground-only cost)
        oracle.call(cpu, mem, 0x2993, a=projector.PLAY_MODE, state=state)
        state["stop"] = False
        oracle.call(cpu, mem, 0x245B, state=state)  # occlusion table, view-independent
        for v in PITCHES:
            for h in BEARINGS:
                for d in range(4):
                    mem[0x09C0 + player] = h
                    mem[0x0140 + player] = v
                    mem[0x0008] = d
                    mem[0x0C48] = 0  # furthest-row hint, fresh as the model assumes
                    state["stop"] = False
                    out[f"{ls},{h},{v},{d}"] = _measure_notch(cpu, mem)
    return out


def _plot_angles(h, v, d):
    """The angle the notch's plot_world actually runs at ($9925 delta, pre-fix-up)."""
    if d < pancost.V_UP:
        return (h + pancost.PAN_DELTA[d]) & 0xFF, v
    return h, (v + pancost.PAN_DELTA[d]) & 0xFF


def _rows():
    with open(GOLDEN) as fh:
        data = json.load(fh)
    assert data
    for key, rec in sorted(data.items()):
        ls, h, v, d = (int(x) for x in key.split(","))
        yield key, landscape.generate(ls), h, v, d, rec


@pytest.mark.oracle
def test_regenerate_pan_cost_golden():
    """Cycle-count pan_viewpoint across the sweep in py65 and dump the golden.
    Writes via rename so a concurrent xdist worker never reads a half-written file."""
    tmp = f"{GOLDEN}.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(_build_golden(), fh, separators=(",", ":"), sort_keys=True)
    os.replace(tmp, GOLDEN)
    test_pan_notch_selects_the_same_tiles_as_the_rom()


def test_pan_notch_selects_the_same_tiles_as_the_rom():
    """The exact half of the model: each notch plots at the INTERMEDIATE angle through
    its own $2993 window, so examined and filled tile counts match the 6502
    byte-for-byte -- including the mode-2 window that culls tiles the play buffer keeps.
    """
    for key, state, h, v, d, rec in _rows():
        ph, pv = _plot_angles(h, v, d)
        tiles, n_examine = projector.project_scene(
            state, ph, pv, None, pancost.PAN_MODE[d]
        )
        assert n_examine == rec["n_examine"], f"{key} examines {n_examine}"
        assert len(tiles) == rec["n_filled"], f"{key} fills {len(tiles)}"


def test_horizontal_pan_window_is_not_the_play_window():
    """Guards the finding the exactness above rests on: mode 2 really selects a
    different tile set, so a play-window pan model would be wrong, not merely equal."""
    rows = [r for r in _rows() if r[4] < pancost.V_UP]
    differ = 0
    for _key, state, h, v, d, _rec in rows:
        ph, pv = _plot_angles(h, v, d)
        strip = projector.project_scene(state, ph, pv, None, pancost.PAN_MODE[d])[1]
        play = projector.project_scene(state, ph, pv, None, projector.PLAY_MODE)[1]
        differ += strip != play
    assert differ > 0.5 * len(rows)


def test_pan_notch_cost_matches_the_measured_plot():
    """Whole-notch accuracy. The residual is the fill proxy's (docs/render_cost.md
    gap 3), not the notch model's -- tile selection is exact -- so this bracket is the
    same class as ``test_render_cost_matches_golden``'s."""
    errors = []
    for key, state, h, v, d, rec in _rows():
        ph, pv = _plot_angles(h, v, d)
        want_clear = rec["clear_cycles"] / projector.FRAME_CYCLES
        got_clear = pancost.CLEAR_FRAMES[0 if pancost.PAN_MODE[d] else 1]
        assert got_clear == pytest.approx(want_clear, abs=1.0), f"{key} clear"
        errors.append(
            pancost.notch_frames(state, d, ph, pv)
            - rec["cycles"] / projector.FRAME_CYCLES
        )
    rms = math.sqrt(statistics.fmean(e * e for e in errors))
    median = statistics.median(abs(e) for e in errors)
    bias = statistics.fmean(errors)
    assert rms < 9.0, f"per-notch rms {rms:.1f} f"
    assert median < 6.0, f"per-notch median |error| {median:.1f} f"
    assert abs(bias) < 3.0, f"per-notch bias {bias:+.1f} f"


def test_derived_notch_beats_the_flat_base_it_replaced():
    """The reason this model exists: a flat per-notch base cannot span a redraw that
    swings by an order of magnitude across a board, whatever value it takes. The
    derived model must beat the BEST such constant, not merely the old 34."""
    meas, pred = [], []
    for _key, state, h, v, d, rec in _rows():
        ph, pv = _plot_angles(h, v, d)
        meas.append(rec["cycles"] / projector.FRAME_CYCLES)
        pred.append(pancost.notch_frames(state, d, ph, pv))
    assert max(meas) > 10 * min(meas), "sweep does not span the swing it claims to"
    best_flat = statistics.fmean(meas)  # the rms-optimal flat constant
    flat_rms = math.sqrt(statistics.fmean((best_flat - m) ** 2 for m in meas))
    model_rms = math.sqrt(statistics.fmean((p - m) ** 2 for p, m in zip(pred, meas)))
    assert model_rms < 0.5 * flat_rms, f"model {model_rms:.1f} vs flat {flat_rms:.1f} f"
