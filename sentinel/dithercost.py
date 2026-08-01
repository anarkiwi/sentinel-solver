"""Exact cycle cost of the $86A5 create/absorb dissolve (dither) loop.

One call runs ``laps`` outer laps of 96 chain steps; each step advances the
3-byte pointer chain ($87B0 operand, $A3, $A4) by x5 mod 2^24 and plots one
dither pixel.  The chain and $9D persist across calls and are threaded here.
"""

DITHER_ENTRY = 0x86A5
CHAIN_ADDRS = (0x87B0, 0x00A3, 0x00A4)
D9D_ADDR = 0x009D
ORG_ADDRS = (0x0096, 0x0097)
SPAN_ADDR = 0x211B
LAPS_ADDR = 0x2099
STEPS_PER_LAP = 0x60  # $86BC LDA #$60
LAPS_PLAY = 0x19  # $1FA4 LDA #$19
LAPS_SETTLE = 0x28  # $2051 LDA #$28 when $0C4E bit7 set
SCREEN_COLS = 0x28  # $873A CMP #$28
SCREEN_ROWS = 0x19  # $8751 CMP #$19

CHAIN_MULT = 5
CHAIN_MOD = 1 << 24

DEFAULT_CHAIN = (0x01, 0x00, 0x00)  # the stage2 image's seed (oracle-verified)

STEP_BASE = 391  # $86C1..$8795 with BCC $8703 taken and no wrap penalties
OUT_OF_SPAN_EXTRA = 3  # BCC not taken + SBC $0CD3
WRAP_EXTRA = 1  # BCC not taken + SBC #imm, per screen-edge wrap
STEP_TAIL = 18  # BIT $0C1E, BMI, DEC $0CD2, BNE taken, JMP $86C1
LAP_HEAD = 6  # LDA #$60, STA $0CD2
LAP_TAIL = 60  # BNE fallthrough, JSR $352C (35), DEC $2099, BEQ, JMP $86BC
CALL_TAIL = 64  # as LAP_TAIL but BEQ $87AF taken + RTS


def span_mask(span):
    """The $0CCF mask $86A5..$86B9 builds: 0 for span 1, else 2^bits(span-1)-1."""
    if span == 1:
        return 0
    return (1 << (span - 1).bit_length()) - 1


def header_cycles(span):
    """$86A5..$86B9: 19 for span 1, else 23 + 7 shifts of the $15C4 index loop."""
    if span == 1:
        return 19
    return 23 + 7 * (9 - (span - 1).bit_length())


def chain_from_mem(mem):
    """The live 3-byte chain state ($87B0 operand, $A3, $A4)."""
    return tuple(mem[a] for a in CHAIN_ADDRS)


def dither_call(span, chain=DEFAULT_CHAIN, laps=LAPS_PLAY, d9d=0, org=(0, 0)):
    """One $86A5 call: exact (cycles, chain_out, d9d_out).

    ``d9d`` is the incoming $9D (written only by this loop, so threadable like
    the chain); ``org`` is ($96, $97), the strip origin column/row the caller
    stores at $2041/$2045.  Cycles cover $86A5 through RTS, JSR excluded.
    """
    x = chain[0] | (chain[1] << 8) | (chain[2] << 16)
    mask = span_mask(span)
    o96, o97 = org
    d = d9d
    cycles = header_cycles(span)
    for lap in range(laps):
        cycles += LAP_HEAD + STEPS_PER_LAP * STEP_BASE + (STEPS_PER_LAP - 1) * STEP_TAIL
        for _ in range(STEPS_PER_LAP):
            x = (x * CHAIN_MULT) % CHAIN_MOD
            a3, a4 = (x >> 8) & 0xFF, x >> 16
            m = (a4 + d) & mask
            if m >= span:
                m -= span
                cycles += OUT_OF_SPAN_EXTRA
            d = m
            if (m + o96) & 0xFF >= SCREEN_COLS:
                cycles += WRAP_EXTRA
            t = (a3 >> 1) & 0x1F
            if ((((t >> 1) + t) >> 1) + o97) & 0xFF >= SCREEN_ROWS:
                cycles += WRAP_EXTRA
        cycles += LAP_TAIL if lap < laps - 1 else CALL_TAIL
    return cycles, (x & 0xFF, (x >> 8) & 0xFF, x >> 16), d


def dither_cycles(span, chain=DEFAULT_CHAIN, laps=LAPS_PLAY, d9d=0, org=(0, 0)):
    """Exact cycle count of one $86A5 call and the outgoing chain state."""
    cycles, chain_out, _ = dither_call(span, chain, laps, d9d, org)
    return cycles, chain_out


def dither_cycles_mean(span, laps=LAPS_PLAY):
    """Expected $86A5 cycles over a uniform chain, for callers with no image.

    Each step's masked value is uniform on [0, mask] (mask+1 divides 256), so
    P(out-of-span) = (mask+1-span)/(mask+1).  Assumes org=(0,0) and
    span <= SCREEN_COLS, where neither screen-edge wrap can fire.
    """
    mask = span_mask(span)
    p_out = (mask + 1 - span) / (mask + 1)
    per_lap = (
        LAP_HEAD
        + STEPS_PER_LAP * (STEP_BASE + OUT_OF_SPAN_EXTRA * p_out)
        + (STEPS_PER_LAP - 1) * STEP_TAIL
    )
    return header_cycles(span) + laps * per_lap + (laps - 1) * LAP_TAIL + CALL_TAIL
