"""Exact CPU-cycle model of the occlusion raytrace $245B in the play posture.

Prices populate_tile_visibility_bit_table from the board bytes alone: fixed
scaffolding + per-tile read/delta/normalise/march terms, no emulator.  Posture:
$0CDE=0, $0CCE bit7 set, $0CDF=$0C73=0 ($352C idles at 29 cycles).
"""

import numpy as np

from sentinel import terrain

N = 32
CALL = 6  # JSR
RTS = 6

ENTRY = 13  # $245B LDA abs 4 + BMI 2 + LDA# 2 + STA zp 3 + LDX# 2
ZERO_FILL = 128 * 10 - 1  # $2466 STA abs,X 5 + DEX 2 + BPL 3, last BPL 2

# $25C4 pass 1 terms (fill (z<<1)|slope per tile via $1DF9):
READER_HEAD = 18  # $25C4..$25D0
READER_ROW = 8  # $25D2 LDA# 2 + STA $24 3 + BNE 3
READER_ROW_TAIL = 13  # $25E7 DEC $71 5 + DEC $26 5 + BPL 3 (last 2)
READER_TILE = 25  # $25DB JSR 6 + LDY 3 + ROL 2 + STA (zp),Y 6 + DEC 5 + BPL 3
TILE_Z_TERRAIN = 76  # JSR $2BA8 6+34 + $1DFC..$1E0D terrain read 36
TILE_Z_OBJ_BASE = 59  # 6+34+5+2 + $1E00 BCS 3 + $1EA9 exit 2 + LDA 4 + RTS 6 - 3
TILE_Z_OBJ_LEVEL = 19  # $1E3F AND/TAY/BIT/BPL/LDA/CMP 16 + BCS-loop 3

# $25ED pass 2 terms (min-of-4-corners >> 1):
P2_HEAD = 7  # $25ED LDA# 2 + STA $72 3 + LDX# 2
P2_ROW = 14  # $25F3 TXA/CLC/ADC/STA/STA/LDY#
P2_ROW_TAIL = 6  # $2621 DEX 2 + BPL 4 (page-crossed; last 2)
P2_FLAT = 22  # LDA (zp),Y 5 + LSR 2 + BCC 3 + STA (zp),Y 6 + DEY 2 + BPL 4
P2_SLOPE = 54  # non-flat body $25FD..$261F, corner branches apart
P2_KEEP = 3  # corner CMP's BCC taken (running min kept)
P2_TAKE = 7  # BCC 2 + LDA (zp),Y 5 (corner replaces the running min)

# $24E2 row-march terms:
ROW_HEAD = 15  # $24E2..$24EB
TILE_HEAD = 120  # JSR+$352C 35 + $24F0..$2500 26 + JSR+$1ECC 59
DELTA_HEAD = 2  # $2503 LDX#
DELTA_COMP = 28  # $2505..$2512 per component
DELTA_POS = 3  # $2514 BPL taken
DELTA_NEG = 22  # BPL 2 + $2516..$251F negate 20
DELTA_CMP = 3  # $2521 CMP $74
DELTA_STORE = 5  # BCC 2 + STA $74 3
DELTA_SKIP = 3  # BCC taken
DELTA_NEXT = 5  # DEX 2 + BPL 3 (last component: BPL 2)
CHEAP_OUT = 12  # $252A LDA/ASL/ASL/CMP 9 + BCC taken 3
NOT_CHEAP = 11  # same with BCC untaken
NORM_LAP = 40  # $2532..$2541: six zp shifts 30 + LSR $17 5 + ASL 2 + BCC 3
PRE_MARCH = 23  # $2543 10 + $254A LDA/BMI 7 + $2576 LDX/LDY 6
LAP_FULL = 79  # $257A..$25AA: adds 72 + BCC 2 + DEX 2 + BNE 3
LAP_OCCL = 77  # adds 72 + BCC taken 3 + $25AF SEC 2
LAP_EXHAUST = 83  # adds 72 + BCC 2 + DEX 2 + BNE 2 + $25AC CLC 2 + BCC 3
TILE_TAIL = 27  # $25B0..$25C0 + JMP (last tile: BMI taken + RTS = 31)
TILE_TAIL_LAST = 31

# $2478 row/merge loop terms:
FIRST_ROW = 5  # $246F LDA# 2 + STA $1A 3
ROW_TOGGLE = 8  # $2478 LDA $05 3 + EOR 2 + STA 3
MERGE_SETUP = 12  # $2481 LDX $0B 3 + LDA abs,X 4 + STA 3 + LDX# 2
MERGE_TAIL = 87  # $249A..$24D3 fixed part incl DEX 2 + BPL 3 (last col: 86)
MERGE_SLOPED = 16  # $248A..$2491 with BCS taken (bit0 set)
MERGE_BELOW = 21  # bit0 clear, z < eye: CMP + BCC taken
MERGE_EQ = 23  # z == eye: BCC 2 + BEQ taken 3
MERGE_ABOVE = 24  # z > eye: BEQ 2 + INY 2
DEC_ROW = 5  # DEC $1A
ROW_LOOP = 3  # $24D7 BPL taken (last 2)

_RAMP = np.arange(256, dtype=np.int64)


def _reader(state):
    """$25C4 pass 1: returns (tz[y][x] = (z<<1)|slope, cycles)."""
    tz = np.zeros((N, N), np.int64)
    cyc = READER_HEAD + N * (READER_ROW + READER_ROW_TAIL + N * READER_TILE) - N - 1
    for y in range(N):
        for x in range(N):
            b = terrain.tile_byte(state, x, y)
            if b < 0xC0:
                z, slope = b >> 4, 1 if b & 0x0F else 0
                cyc += TILE_Z_TERRAIN
            else:
                slot, depth = b & 0x3F, 1
                for _ in range(63):
                    if state.obj_flags[slot] < 0x40:
                        break
                    slot = state.obj_flags[slot] & 0x3F
                    depth += 1
                z, slope = state.obj_z_height[slot], 0
                cyc += TILE_Z_OBJ_BASE + TILE_Z_OBJ_LEVEL * depth
            tz[y, x] = ((z << 1) | slope) & 0xFF
    return tz, cyc


def _corner_min(tz):
    """$25ED pass 2: returns (mz[0:31,0:31], cycles)."""
    b = tz[:31, :31]
    flat = (b & 1) == 0
    c1, c2, c3 = tz[:31, 1:32], tz[1:32, 1:32], tz[1:32, :31]
    take1 = b >= c1
    a = np.where(take1, c1, b)
    take2 = a >= c2
    a = np.where(take2, c2, a)
    take3 = a >= c3
    a = np.where(take3, c3, a)
    mz = np.where(flat, b >> 1, a >> 1)
    corner = sum(np.where(t, P2_TAKE, P2_KEEP) for t in (take1, take2, take3))
    col = np.where(flat, P2_FLAT, P2_SLOPE + corner)
    cyc = int(col.sum()) + 31 * (P2_ROW + P2_ROW_TAIL - 2) - 2 + P2_HEAD + RTS
    return mz, cyc


def _overlay(state, tz, mz):
    """Memory as the march's CMP ($72),Y sees it: board RAM + the built tables.

    Pages $40-$59 are fresh RAM (zero), $5A-$5F carry the ROM identity-ramp
    pages outside the freshly written rows, $3E80 was just zero-filled."""
    ov = np.frombuffer(bytes(state.mem), dtype=np.uint8).astype(np.int64)
    ov[0x4000:0x5A00] = 0
    ov[0x5A00:0x6000] = np.tile(_RAMP, 6)
    ov[0x3E80:0x3F00] = 0
    for y in range(N):
        ov[0x4000 + (y << 8) : 0x4000 + (y << 8) + N] = tz[y]
        if y < 31:
            ov[0x4020 + (y << 8) : 0x4020 + (y << 8) + 31] = mz[y]
    return ov


def _tile_terms(tz, objx, objy, objz, zfrac):
    """$2503..$2541 per tile: delta cycles, cheap mask, shift count, shifted
    16-bit deltas + sign bytes, march lap budget."""
    tx, ty = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64))
    tgt = (tx, tz >> 1, ty)
    olo = (0x80, zfrac, 0x80)
    ohi = (objx, objz, objy)
    lo, hi, ext = {}, {}, {}
    cost = np.full((N, N), DELTA_HEAD, np.int64)
    maxd = np.zeros((N, N), np.int64)
    for k in (2, 1, 0):
        lo[k] = (0 - olo[k]) & 0xFF
        borrow = 0 if olo[k] == 0 else 1
        hi[k] = (tgt[k] - ohi[k] - borrow) & 0xFF
        neg = (hi[k] & 0x80) != 0
        ext[k] = np.where(neg, 0xFF, 0)
        absd = np.where(neg, (0 - hi[k] - (1 if lo[k] else 0)) & 0xFF, hi[k])
        store = absd >= maxd
        maxd = np.where(store, absd, maxd)
        cost += (
            DELTA_COMP
            + np.where(neg, DELTA_NEG, DELTA_POS)
            + DELTA_CMP
            + np.where(store, DELTA_STORE, DELTA_SKIP)
            + (DELTA_NEXT if k else DELTA_NEXT - 1)
        )
    a0 = (maxd << 2) & 0xFF
    cheap = a0 < 6
    bl = np.zeros((N, N), np.int64)
    for s in range(8):
        bl = np.where(a0 >= (1 << s), s + 1, bl)
    shifts = 9 - bl  # $2532 loop laps: ASL A until the carry emerges
    cost += np.where(cheap, CHEAP_OUT, NOT_CHEAP + NORM_LAP * shifts - 1 + PRE_MARCH)
    laps = np.where(cheap, 0, 0xFF >> shifts)
    sh = {}
    for k in (2, 1, 0):
        v = (((hi[k] << 8) | lo[k]) << shifts) & 0xFFFF
        sh[k] = (v & 0xFF, v >> 8)
    return cost, cheap, sh, ext, laps


def _march(ov, cheap, sh, ext, laps, objx, objy, objz, zfrac):
    """$2576 march, all 1024 rays in lockstep: per-lap cost incl the CMP ($72),Y
    page-cross penalty, exit by occlusion (BCC) or lap-budget exhaustion."""
    active = ~cheap.ravel()
    lapsf = laps.ravel()
    e0, e1, e2 = (ext[k].ravel() for k in (0, 1, 2))
    s0hi, s2hi = sh[0][1].ravel(), sh[2][1].ravel()
    s1lo, s1hi = sh[1][0].ravel(), sh[1][1].ravel()
    n = active.size
    xlo = np.full(n, 0x80, np.int64)
    xin = np.full(n, objx, np.int64)
    ylo = np.full(n, 0x80, np.int64)
    yrow = np.full(n, (objy + 0x40) & 0xFF, np.int64)
    zlo = np.zeros(n, np.int64)
    zmid = np.full(n, zfrac, np.int64)
    zhi = np.full(n, objz, np.int64)
    cyc = np.zeros(n, np.int64)
    lap = 0
    while active.any():
        lap += 1
        t = xlo + s0hi
        xlo = np.where(active, t & 0xFF, xlo)
        xin = np.where(active, (xin + e0 + (t >> 8)) & 0xFF, xin)
        t = ylo + s2hi
        ylo = np.where(active, t & 0xFF, ylo)
        yrow = np.where(active, (yrow + e2 + (t >> 8)) & 0xFF, yrow)
        t = zlo + s1lo
        t2 = zmid + s1hi + (t >> 8)
        zlo = np.where(active, t & 0xFF, zlo)
        zmid = np.where(active, t2 & 0xFF, zmid)
        zhi = np.where(active, (zhi + e1 + (t2 >> 8)) & 0xFF, zhi)
        addr = ((yrow << 8) + 0x20 + xin) & 0xFFFF
        occl = zhi < ov[addr]
        exhausted = lapsf == lap
        add = np.where(occl, LAP_OCCL, np.where(exhausted, LAP_EXHAUST, LAP_FULL))
        cyc += np.where(active, add + (xin >= 0xE0), 0)
        active &= ~(occl | exhausted)
    return cyc.reshape((N, N))


def _merge_cost(tz, objz):
    """$248A per-row merge over columns 30..0 (rows 30..0 only)."""
    b = tz[:31, :31]
    z = b >> 1
    head = np.where(
        (b & 1) != 0,
        MERGE_SLOPED,
        np.where(z < objz, MERGE_BELOW, np.where(z == objz, MERGE_EQ, MERGE_ABOVE)),
    )
    return (head + MERGE_TAIL).sum(axis=1) - 1  # last column's BPL untaken


def occlusion_cycles(state, observer=None):
    """CPU cycles $245B spends for this board and observer (default: player)."""
    obs = state.player if observer is None else observer
    objx, objy = state.obj_x[obs], state.obj_y[obs]
    objz, zfrac = state.obj_z_height[obs], state.obj_z_frac[obs]
    tz, reader_cyc = _reader(state)
    mz, corner_cyc = _corner_min(tz)
    ov = _overlay(state, tz, mz)
    tile_cyc, cheap, sh, ext, laps = _tile_terms(tz, objx, objy, objz, zfrac)
    march_cyc = _march(ov, cheap, sh, ext, laps, objx, objy, objz, zfrac)
    per_tile = tile_cyc + march_cyc + TILE_HEAD + TILE_TAIL
    rows = per_tile.sum(axis=1) + (TILE_TAIL_LAST - TILE_TAIL) + ROW_HEAD
    merge_rows = _merge_cost(tz, objz)
    total = ENTRY + ZERO_FILL + CALL + reader_cyc + corner_cyc
    total += FIRST_ROW + CALL + int(rows[31]) + DEC_ROW
    total += (ROW_TOGGLE + CALL + MERGE_SETUP + DEC_ROW + ROW_LOOP) * 31
    total += int(rows[:31].sum()) + int(merge_rows.sum())
    total += RTS - 1  # last $24D7 BPL untaken
    return total
