"""$86A5 dissolve-loop pricing: dithercost against the real ROM loop."""

import pytest

from sentinel import dithercost
from sentinel.tests import oracle

SPANS = (1, 2, 4, 6, 12, 15, 20, 25)
CHAINS = (
    dithercost.DEFAULT_CHAIN,
    (0x37, 0x12, 0x00),
    (0xFF, 0xFF, 0xFF),
    (0x00, 0x80, 0x41),
)
LAPS = (dithercost.LAPS_PLAY, dithercost.LAPS_SETTLE)
WRAP_CASES = (
    (25, (0xC5, 0x9A, 0x03), 25, 7, (30, 0)),
    (6, (0xFF, 0xFF, 0xFF), 25, 0, (17, 9)),
    (2, (0x00, 0x80, 0x41), 40, 1, (39, 24)),
)


@pytest.fixture(scope="module", name="machine")
def _machine():
    cpu, mem, _ = oracle.fresh_machine()
    with open(oracle.IMG, "rb") as f:
        img = f.read()
    mem[0xFFF1:0x10000] = img[0xFFF1:]
    return cpu, mem


def _run_rom(machine, span, chain, laps, d9d=0, org=(0, 0)):
    """One real $86A5 call: (cycles, chain_out, d9d_out)."""
    cpu, mem = machine
    mem[dithercost.SPAN_ADDR] = span
    mem[dithercost.LAPS_ADDR] = laps
    for addr, val in zip(dithercost.CHAIN_ADDRS, chain):
        mem[addr] = val
    mem[dithercost.D9D_ADDR] = d9d
    for addr, val in zip(dithercost.ORG_ADDRS, org):
        mem[addr] = val
    for addr in (0x0CDF, 0x0C73, 0x0C1E):
        mem[addr] = 0
    ret = 0xFFF0
    mem[ret] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = dithercost.DITHER_ENTRY
    start = cpu.processorCycles
    steps = 0
    while cpu.pc != ret and steps < 2_000_000:
        cpu.step()
        steps += 1
    assert cpu.pc == ret
    return (
        cpu.processorCycles - start,
        dithercost.chain_from_mem(mem),
        mem[dithercost.D9D_ADDR],
    )


@pytest.mark.oracle
def test_image_carries_default_chain(machine):
    _, mem = machine
    assert dithercost.chain_from_mem(mem) == dithercost.DEFAULT_CHAIN
    assert mem[dithercost.D9D_ADDR] == 0


@pytest.mark.oracle
@pytest.mark.parametrize("chain", CHAINS, ids=lambda c: "%02x%02x%02x" % c[::-1])
@pytest.mark.parametrize("laps", LAPS)
@pytest.mark.parametrize("span", SPANS)
def test_dither_cycles_exact(machine, span, laps, chain):
    got = _run_rom(machine, span, chain, laps)
    assert got == dithercost.dither_call(span, chain, laps)
    assert got[:2] == dithercost.dither_cycles(span, chain, laps)


@pytest.mark.oracle
@pytest.mark.parametrize("case", WRAP_CASES, ids=("colwrap", "rowwrap", "bothwrap"))
def test_dither_cycles_exact_with_origin(machine, case):
    span, chain, laps, d9d, org = case
    got = _run_rom(machine, span, chain, laps, d9d, org)
    assert got == dithercost.dither_call(span, chain, laps, d9d, org)


@pytest.mark.oracle
def test_chain_threads_across_calls(machine):
    """Two back-to-back calls: the second is exact iff chain+$9D are threaded."""
    chain, d9d = dithercost.DEFAULT_CHAIN, 0
    for span, laps in ((25, 25), (6, 40)):
        got = _run_rom(machine, span, chain, laps, d9d)
        want = dithercost.dither_call(span, chain, laps, d9d)
        assert got == want
        _, chain, d9d = want


def test_mask_matches_shift_loop():
    """span_mask reproduces the $86A5..$86B9 shift count into $15C4."""
    table = [0xFF >> i for i in range(8)]
    for span in range(2, 257):
        acc, y = span - 1, -1
        while acc < 0x100:
            acc <<= 1
            y += 1
        assert dithercost.span_mask(span) == table[y]
        assert dithercost.header_cycles(span) == 23 + 7 * (y + 1)
    assert dithercost.span_mask(1) == 0
    assert dithercost.header_cycles(1) == 19


def test_structural_constants():
    assert dithercost.STEPS_PER_LAP == 96
    assert dithercost.LAPS_PLAY == 25
    assert dithercost.LAPS_SETTLE == 40
    fixed = dithercost.LAP_HEAD + 96 * dithercost.STEP_BASE + 95 * dithercost.STEP_TAIL
    closed = 19 + 25 * fixed + 24 * dithercost.LAP_TAIL + dithercost.CALL_TAIL
    cycles, chain_out = dithercost.dither_cycles(1, (0, 0, 0))
    assert (cycles, chain_out) == (closed, (0, 0, 0))
    assert dithercost.dither_cycles_mean(1) == closed


def test_chain_out_closed_form():
    chain = (0x37, 0x12, 0x00)
    for laps in LAPS:
        _, chain_out = dithercost.dither_cycles(25, chain, laps)
        x = chain[0] | (chain[1] << 8) | (chain[2] << 16)
        x = x * pow(5, 96 * laps, 1 << 24) % (1 << 24)
        assert chain_out == (x & 0xFF, (x >> 8) & 0xFF, x >> 16)


def test_mean_matches_model_average():
    """The derived P(out) = (2^L - span)/2^L predicts the model's own average."""
    for span in (6, 25):
        total = 0
        n = 64
        for i in range(n):
            x = (i * 0x9E3779B9 + 1) % (1 << 24)
            chain = (x & 0xFF, (x >> 8) & 0xFF, x >> 16)
            total += dithercost.dither_cycles(span, chain)[0]
        mean = dithercost.dither_cycles_mean(span)
        assert abs(total / n - mean) / mean < 2e-4
