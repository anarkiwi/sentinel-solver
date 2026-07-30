"""Cycle cost of one play-loop pass ($1289), counted from the ROM.

The loop never counts frames: the raster IRQ pre-empts it and the foreground spends
what is left, so passes per frame = (FRAME_CYCLES - IRQ_CYCLES) / pass cost, and the
pass cost is a property of the state.  Per-term arithmetic: docs/architecture.md.
"""

from sentinel import memmap as mm

PAL_FRAME_CYCLES = 19656  # PAL 6569: 312 raster lines x 63 cycles
BADLINE_STEAL = 43  # a $30..$F7 line whose low 3 bits are YSCROLL ($D011 & 7 = 3)
BADLINES_PER_FRAME = 25  # raster 51..243 step 8; $D015 = 0, so there is no sprite term
SHORT_IRQ = 119  # $95E9 split chain at raster 53/93/133/173: 7 entry + 112 body
SHORT_IRQS_PER_FRAME = 4  # the $9589 table 35 D5 AD 85 5D, less the $9593 full entry
IRQ_BODY = 2491  # the $9630 body: $95E9 202 + $119F 2156 + $1635/$FFC2/$FFC5 126 + 7
IRQ_CYCLES = 4042  # 1075 + 476 + 2491: every FIXED cycle a frame denies the play loop
IRQ_SPRITES = 1490  # $1635 loses its $963A fast exit once $0C04 != 0
FOREGROUND_CYCLES = PAL_FRAME_CYCLES - IRQ_CYCLES  # less this frame's own $130C

COOLDOWN_TICK_NO_CARRY = 21  # $130C LDA/CLC/ADC/STA 12 + $1315 BCC taken 3 + RTS 6
COOLDOWN_TICK_GATE = 33  # + $1317 LDA 4 + BNE 3 + $1331 DEC 6 + RTS 6 - the taken BCC 1
COOLDOWN_TICK_WALK = 33  # the $131C walk's own entry 22 + $132B reload 6 + RTS 6 - 1
COOLDOWN_TICK_BYTE_STICK = 14  # $131E LDA 4 + CMP 2 + BCC taken 3 + DEX 2 + BPL 3
COOLDOWN_TICK_BYTE_DEC = 20  # + $1325 DEC 7, less the taken BCC 1

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

MARCH_STEP = 314  # $1CE8..$1D18: JSR $1CBB 6 + edge tests 20 + $1CFB 17 + $1DF9 + 23
MARCH_OBJECT = 24  # $1E00 BCS: the $1E3F object-stack surface above the flat check
MARCH_SLOPE = 581  # $1D0B BCS $1D46: check_sloping_tile instead of check_flat_tile
# Measured, NOT split: the $1D46 corner path (nibble 4/12) is 332 and the quad path 579,
# but pricing them apart makes relative.can_see_object and enemies_jit._can_see_object
# disagree by 104 cycles on one long march -- see docs/open_items.md item 8.

SCAN_SLOT = 27  # $17B2 LDA 2 + JSR 6 + the $17B7 gates 11 + $17CA DEY/BPL 5
SCAN_FIXED = 12  # a 64-slot scan's entry/exit

# find_drainable_boulder_or_tree $1AB0 walks its own loop, not $17B2's.
TILE_SCAN_FIXED = 10  # $1AB0 LDX 2 + $1AF2 SEC/RTS 8
TILE_SCAN_EMPTY = 12  # $1AB2 LDA 4 + BMI 3 + $1AEF DEX/BPL 5
TILE_SCAN_OTHER = 24  # + the $1AB7 flag and $1ABB type compares 12
TILE_SCAN_STACKED = 11  # flags >= $40 leaves at $1AB9 BCS
TILE_SCAN_LOOSE = 18  # a lone boulder falls through the $1ABE type compare instead
TILE_SCAN_TILE = 61  # $1AC2..$1AD1 the tile fetch, its $2BA8 lookup 34 included
TILE_SCAN_NO_TILE = 8  # $1AD3 BCC taken + $1AEF DEX/BPL
TILE_SCAN_TOP = 12  # $1AD3 BCC 2 + $1AD5..$1ADB the top object's type read 10
TILE_SCAN_WRONG_TOP = 12  # $1ADD..$1AE1 the two compares 7 + $1AEF DEX/BPL 5
TILE_SCAN_SEE = 9  # $1AE1 BEQ 3 + the $1AE3 JSR 6 (a boulder top pays 3 more)
TILE_SCAN_NEXT = 11  # $1AE6 LDA/BPL 6 + $1AEF DEX/BPL 5 when the see was not full
TILE_SCAN_HIT = 8  # $1AE8 BPL not taken 2 + $1AEA STY/CLC 6

# consider_creating_meanie $197D walks the search counter, not a slot index.
MEANIE_SCAN_SLOT = (
    26  # $198F LDX 3 + LDY 4 + BNE 3 + $19A1 DEC 7 + DEY 2 + LDA 4 + BMI 3
)
MEANIE_SCAN_OTHER = 34  # + the $19AA tree-type compare 9 - the taken BMI 1
MEANIE_SCAN_DX = 24  # $19B1..$19C5 the 10-tile x test, its $19BE sign fixup included
MEANIE_SCAN_DY = 18  # $19C7..$19D7 the same in y, off the already-loaded index
MEANIE_SCAN_SEE = 8  # $19D9 LDA 2 + the $19DB JSR 6
MEANIE_SCAN_DONE = 33  # $198F..$1994 9 + $1996 INC/LDA/STA 16 + SEC/RTS 8

ROTATE_GATE = 12  # $17F9 LDX 3 + LDA 4 + CMP 2 + BCC 3; 14 when the gate holds
ROTATE = 454  # $1805..$1884: $1AF4 31 + $1973 32 + $3470 323 + $187B 18 + 50 straight
ROTATE_REDRAW = 1723  # $1F9F update_object_on_screen; MEASURED 1576..1843, 16 rotations
MEANIE_ROTATE = 444  # $1728..$1884, the meanie's own turn: $1AF4 31 + $3470 323 + 90


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


def see_cycles(march_cycles, probes, in_slot, wrong_type):
    """$1887 for one visibility query: a reject path, else the bearing chain plus one
    march per probe, each march carrying the cycles its own sub-steps cost."""
    if not in_slot:
        return SEE_SLOT_WRONG_TYPE if wrong_type else SEE_SLOT_EMPTY
    return SEE_GEOMETRY + probes * SEE_PROBE + march_cycles


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
