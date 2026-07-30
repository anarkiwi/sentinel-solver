"""Cycle cost of one play-loop pass ($1289), counted from the ROM.

The loop never counts frames: the raster IRQ pre-empts it and the foreground spends
what is left, so passes per frame = (FRAME_CYCLES - IRQ_CYCLES) / pass cost, and the
pass cost is a property of the state.  Per-term arithmetic: docs/architecture.md.
"""

from sentinel import memmap as mm

PAL_FRAME_CYCLES = 19656  # PAL 6569: 312 raster lines x 63 cycles
IRQ_CYCLES = 4142  # $9630 + VIC-II DMA steal; the complement of the foreground
IRQ_SPRITES = 1490  # $1635 loses its $963A fast exit once $0C04 != 0
FOREGROUND_CYCLES = PAL_FRAME_CYCLES - IRQ_CYCLES

LOOP_PASS = 142  # $1289..$12C7 in-play straight line, less the $16B5/$191F bodies

EXPOSURE_FIXED = 50  # $191F prologue 7 + epilogue 37 + RTS 6
EXPOSURE_EMPTY = 12  # $1925 LDA 4 + BMI 3 + DEX 2 + BPL 3
EXPOSURE_OTHER = 24  # + LDA type 4 + CMP 2 + BEQ 2 + CMP 2 + BNE 3
EXPOSURE_SENTRY = 30  # + LDA targeted 4 + CMP $0B 3 + BNE 3
EXPOSURE_SENTINEL = 33  # type 5 takes the longer compare chain
EXPOSURE_TARGETS_PLAYER = 6  # $193A falls through: LDA $0C20,X 4 + BEQ 3 - 1
EXPOSURE_DRAINING = 10  # $193F falls through: LDY 2 + LDA 4 + STA 3 + BMI 2 - 1
EXPOSURE_LAST = 1  # X == 0 leaves via BPL not taken

UPDATE_NOT_ENEMY = 22  # $16B5 TSX 2 + STX 4 + LDX 3 + LDA 4 + CMP/BEQ/CMP/BNE 9
UPDATE_DISPATCH_SENTRY = 29  # ... + BEQ 3 + STA 4 + LDA flags 4 + BPL 3
UPDATE_DISPATCH_SENTINEL = 32  # ... with the longer type compare chain
UPDATE_GATE_CLOSED = 9  # $16E6 LDA $0C30,X 4 + CMP 2 + BCS 3
UPDATE_ABSORBED = 8  # $16CC BMI 2 for BPL 3 + JSR $1A5D 6 + BCS 3
PRND = 427  # $31CA: 8 rounds of the 40-bit LFSR, 51 each
UPDATE_TAIL = 453  # $16D6 JSR 6 + prnd + DEC 5 + BPL 3 + LDA 3 + STA 3 + RTS 6
UPDATE_TAIL_WRAP = 457  # cursor 0: BPL not taken 2 + LDA 2 + STA 3
CONSIDER_ENTRY = 30  # $16E6..$16F6 gate open 21 + $16F7 LDA 4 + BPL 2 + JMP 3
CONSIDER_PREAMBLE = 36  # $1773..$17B2 around the discharge and considering flags

DISCHARGE_NONE = 18  # $1A5D LDX 3 + SEC 2 + LDA 4 + BEQ 3 + RTS 6
DISCHARGE_FIXED = 100  # $1A5D success 62 + create_object $211D + $1238 entry
DISCHARGE_TRY = 966  # $1238 loop body 76..98 + two $1272 draws (445 + 440n each)

SEE_SLOT_EMPTY = 40  # $1887 exit at $1893
SEE_SLOT_WRONG_TYPE = 49  # $1887 exit at $189D
SEE_GEOMETRY = 1128  # $1887 prologue 37 + $8401 + the $18CA FOV compare/exit 85
SEE_PROBE = 210  # $18E6 body 82 + $1CDD/$1ECC march entry 74 + $933D/$1C54
SEE_STEP = 700  # one $1CE8 tile step: 306 flat, + the $1D46 slope/object sub-path

SCAN_SLOT = 22  # $17BA the per-slot loop body around a visibility call
SCAN_FIXED = 12  # a 64-slot scan's entry/exit


def exposure_cycles(mem):
    """$191F for the current board: the 8-slot walk plus prologue/epilogue.

    X counts 7 down to 0 and stops early only when an enemy targeting the player has
    exposure bit 7 set ($1948 BMI $194D)."""
    player = mem[mm.PLAYER_OBJECT]
    total = EXPOSURE_FIXED
    for x in range(7, -1, -1):
        last = EXPOSURE_LAST if x == 0 else 0
        if mem[mm.OBJECTS_FLAGS + x] & 0x80:
            total += EXPOSURE_EMPTY - last
            continue
        otype = mem[mm.OBJECTS_TYPE + x]
        if otype == mm.T_SENTRY:
            slot = EXPOSURE_SENTRY
        elif otype == mm.T_SENTINEL:
            slot = EXPOSURE_SENTINEL
        else:
            total += EXPOSURE_OTHER - last
            continue
        if mem[mm.ENEMIES_TARGETED_OBJECT + x] != player:
            total += slot - last
            continue
        slot += EXPOSURE_TARGETS_PLAYER
        if mem[mm.ENEMIES_DRAINING_COOLDOWN + x] == 0:
            total += slot - last
            continue
        slot += EXPOSURE_DRAINING
        if mem[mm.ENEMIES_TARGETED_OBJECT_EXPOSURE + x] & 0x80:
            return total + slot + 1  # the walk stops here
        total += slot - last
    return total


def see_cycles(steps, probes, in_slot, wrong_type):
    """$1887 for one visibility query: a reject path, else the bearing chain plus one
    march per probe, linear in the sub-steps the marches took."""
    if not in_slot:
        return SEE_SLOT_WRONG_TYPE if wrong_type else SEE_SLOT_EMPTY
    return SEE_GEOMETRY + probes * SEE_PROBE + steps * SEE_STEP


def idle_pass_cycles(mem):
    """One pass with every enemy's $16E9 gate closed, averaged over the 8 cursor
    positions: the loop's fastest cadence on this board."""
    total = 0
    for x in range(8):
        otype = mem[mm.OBJECTS_TYPE + x]
        if otype == mm.T_SENTINEL:
            total += UPDATE_DISPATCH_SENTINEL + UPDATE_GATE_CLOSED
        elif otype == mm.T_SENTRY:
            total += UPDATE_DISPATCH_SENTRY + UPDATE_GATE_CLOSED
        else:
            total += UPDATE_NOT_ENEMY
        total += UPDATE_TAIL_WRAP if x == 0 else UPDATE_TAIL
    return total / 8.0 + LOOP_PASS + exposure_cycles(mem)
