"""A cost term's cycle offset names an instruction, and that instruction's write run.

The opcode table is checked against every instruction the live captures timed, and the
walk against the cycle counts ``passcost`` already claims for the same ROM runs.
"""

import json
import os

import pytest

from sentinel import badline, passcost, writemap

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
