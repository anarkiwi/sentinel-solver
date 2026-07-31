"""A cost term's cycle offset names an instruction, and that instruction's write run.

The opcode table is checked against every instruction the live captures timed, and the
walk against the cycle counts ``passcost`` already claims for the same ROM runs.
"""

import json
import os

import pytest

from sentinel import badline, passcost, writemap, writeweight

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "live_badline.json")
IMG = os.path.join(os.path.dirname(__file__), "..", "..", "out", "sentinel_stage2.bin")


def _live():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["boards"]


@pytest.fixture(name="image", scope="module")
def _image():
    with open(IMG, "rb") as fh:
        return fh.read()


def test_the_derived_write_cycles_reproduce_the_proven_hand_table():
    """Store, read-modify-write and push cycles fall out of the addressing mode, so
    the opcode table needs no hand entry -- and none of them runs three long."""
    for op, cycles in badline.WRITE_CYCLES.items():
        assert writemap.OP_WRITE_CYCLES[op] == cycles, hex(op)
    assert all(len(c) <= 2 for c in writemap.OP_WRITE_CYCLES)
    assert writemap.write_run(0x2E, 4) == 2  # $31D9 ROL abs, the LFSR's 41s


def test_the_opcode_cost_table_prices_every_live_instruction():
    """Each capture times every opcode class it executed by its own minimum delta; the
    table must equal all of them, which is what caught jennings' $CE DEC abs."""
    for name, board in _live().items():
        assert board["op_cycle_mismatch"] == {}, name
    assert writemap.OP_CYCLES[0xCE] == writemap.DEC_ABS_CYCLES


@pytest.mark.oracle
def test_the_walk_reproduces_the_cycle_counts_passcost_claims(image):
    """Three terms counted off the disassembly, re-walked over the image: the PRNG's
    eight LFSR rounds, the meanie's four writes and the tile-address formula."""
    for pc, want, branches in (
        (0x31CA, passcost.PRND, [True] * 7 + [False]),
        (0x1289, passcost.PASS_HEAD, [False, True]),
        (0x1973, passcost.MEANIE_INIT, []),
        (0x2BA8, passcost.TILE_ADDR - passcost.TILE_Z_CALL, []),
    ):
        run = writemap.walk(image, pc, want, branches)
        assert run[-1][0] + writemap.OP_CYCLES[run[-1][2]] == want, hex(pc)


@pytest.mark.oracle
def test_the_static_walk_alone_resolves_most_windows(image):
    """With no branch record at all -- every conditional falling through -- walking
    from the model's own anchor already names the instruction at 95% of live BA
    windows; the rest are branches the charging term itself decides."""
    assert image[0x31CA] == 0x8C  # $31CA STY abs, the PRNG entry
    windows = sum(b["windows"] for b in _live().values())
    decoded = sum(b["statically_decoded"] for b in _live().values())
    assert decoded > 0.94 * windows, (decoded, windows)


@pytest.mark.oracle
def test_a_branch_the_term_charges_decides_the_ambiguous_window(image):
    """$0D03's shift-add rounds branch on the multiplier's bits -- what MUL8_BIT
    charges -- and that decision is exactly what splits the one ambiguous offset."""
    assert writemap.steal_at(image, 0x0D05, 57, []) == passcost.BADLINE_STEAL
    assert writemap.steal_at(image, 0x0D05, 57, [True]) == badline.MIN_STEAL


@pytest.mark.oracle
def test_the_shipped_write_map_is_what_the_image_generates(image):
    """``sentinel.writeruns`` is generated, so it must equal a fresh walk of the ROM."""
    from driver import writeruns as gen  # pylint: disable=import-outside-toplevel
    from sentinel import writeruns  # pylint: disable=import-outside-toplevel

    with open(os.path.join(os.path.dirname(__file__), "..", "passcost.py")) as fh:
        rows = gen.table(image, fh.read())
    assert tuple(a for a, _, _ in rows) == writeruns.ANCHOR
    assert tuple(c for _, c, _ in rows) == writeruns.LENGTH
    assert tuple(w for _, _, w in rows) == writeruns.RUNS
    assert writeruns.RUN[writeruns.START[0x31CA] + 26] == 2  # $31D9 ROL abs


def _flat_lap(image, neg):
    """``(cycles, write weight)`` of the $1CE8..$1D18 sub-step, walked over the ROM.

    The branch record is the march's own: per axis the $1CCA BPL (taken when that
    component is positive) and the $1CD9 loop-back, then both edge tests, the object
    test and the slope test falling through, and the $1D18 BMI closing the loop.
    """
    branches = []
    for axis in range(3):
        branches += [axis >= neg, axis != 2]
    branches += [False, False, False, False, True]
    run = writemap.walk(image, 0x1CE8, 4000, branches)
    end = next(offset for offset, pc, _ in run[1:] if pc == 0x1CE8)
    weight = 0
    for offset, _pc, op in run:
        cycles = [c for c in writemap.OP_WRITE_CYCLES[op] if offset + c < end]
        weight += len(cycles) * (len(cycles) + 1) // 2
    return end, weight


@pytest.mark.oracle
def test_the_march_laps_write_weight_is_its_own_instructions(image):
    """A window inside a march is priced by the lap's write weight, and that weight is
    the lap's instruction sequence: for all four component-sign patterns the terms
    ``writeweight`` composes are the ROM walk's own cycles and write cycles."""
    for neg in range(4):
        cycles, weight = _flat_lap(image, neg)
        assert (
            cycles
            == sum(getattr(passcost, name) for name in writeweight.FLAT_LAP)
            + neg * passcost.ADD_VECTOR_NEG
        ), neg
        assert (
            weight
            == sum(writeweight.WEIGHT.get(name, 0) for name in writeweight.FLAT_LAP)
            + neg * writeweight.WEIGHT["ADD_VECTOR_NEG"]
        ), neg
    assert _flat_lap(image, 0)[1] == 30 and _flat_lap(image, 3)[1] == 39
