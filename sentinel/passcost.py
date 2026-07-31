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

PASS_HEAD = 25  # $1289..$129F: LDA 4 + BMI 2 + LSR 6 + LDA 4 + BPL 3 + the JSR 6
PASS_TAIL = 117  # $12A2..$12C7 48 + the $34BA/$352C/$347D bodies 13 + 29 + 27
LOOP_PASS = PASS_HEAD + PASS_TAIL  # 142: the whole straight line, for the idle rate

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
UPDATE_PRND = 433  # $16D6 JSR 6 + prnd: spent BEFORE the stream advances
UPDATE_CURSOR = 20  # $16D9 DEC 5 + BPL 3 + $16E1 LDA/STA 6 + RTS 6
UPDATE_CURSOR_WRAP = 24  # cursor 0: BPL not taken 2 + LDA 2 + STA 3
UPDATE_TAIL = UPDATE_PRND + UPDATE_CURSOR  # 453, the whole $16D6..$16E5
UPDATE_TAIL_WRAP = UPDATE_PRND + UPDATE_CURSOR_WRAP  # 457
CONSIDER_ENTRY = 30  # $16E6..$16F6 gate open 21 + $16F7 LDA 4 + BPL 2 + JMP 3
CONSIDER_PREAMBLE = 36  # $1773..$17B2 around the discharge and considering flags

DISCHARGE_NONE = 18  # $1A5D LDX 3 + SEC 2 + LDA 4 + BEQ 3 + RTS 6
DISCHARGE_FIXED = 100  # $1A5D success 62 + create_object $211D + $1238 entry
DISCHARGE_TRY = 966  # $1238 loop body 76..98 + two $1272 draws (445 + 440n each)

SEE_SLOT_EMPTY = 40  # $1887 exit at $1893
SEE_SLOT_WRONG_TYPE = 49  # $1887 exit at $189D
SEE_GEOMETRY = 1128  # $1887 prologue 37 + $8401 + the $18CA FOV compare/exit 85
SEE_PROBE = 719  # $18E6 82 + $933D 627 + the $1CDD/$1ECC entry 59 -- NOT $1C54,
# which prepare_vector_from_angle prices from its own shift-adds and branches.

MUL8 = 102  # $0D03 JSR 6 + STA/LDA 5 + the 8 unrolled shift-add rounds 91
MUL8_LOW = 99  # the $0D05 entry, $0074 already stored
MUL8_BIT = 4  # per set bit of the multiplier: BCC not taken 2 + CLC 2 + ADC 3, less 3
MUL_DBL_BYTE = 230  # $0F4A..$0F61 with both $0D03 calls, BCC taken
MUL_DBL_BYTE_CARRY = 4  # $0F5D BCC not taken 2 + $0F5F INC $0075 5, less 3
MUL_DBL_BYTE_CALL = 6  # $0EAB/$0EDC JSR $0F4A; $0F3E falls in with no JSR
MUL_PI = 22  # $0F3E..$0F49, the two ASL/ROL and the $C9 setup
SIN_COS = 42  # $0E75 JSR 6 + $0E75..$0E8C 36, the $0F3E call apart
SIN_COS_PI_CALL = 6  # $0E7B JSR $0F3E
SIN_COS_QUAD = 9  # $0E8F BVC not taken 2 + INX 2 + DEC $0060 5, less the taken 3
SIN_COS_LOW = 5  # $0E94 CMP #$7A 2 + BCC taken 3: the $0EA1 interpolation
SIN_COS_HIGH = 7  # ... BCC not taken 2 + $0E98 BCS taken 3: the $0ECB clamp
SIN_COS_ONE = 49  # $0EA1 LDA 2 + $0EA9 STA $0076 3 + $0EAE..$0EC8 44
SIN_COS_TWO = 60  # $0ECB..$0EFA, the $0F4A call apart, BCC taken
SIN_COS_TWO_CLAMP = 8  # $0EF1 BCC not taken 2 + $0EF3..$0EF8 9, less 3
SIN_COS_NEXT = 36  # $0EFD BEQ not taken 5 + $0F01..$0F16 31: the second quadrant
SIN_COS_DONE = 6  # $0EFD CPX 3 + BEQ taken 3
SIN_COS_SIGNS = 30  # $0F19..$0F3A, both BPLs taken
SIN_COS_SIGN = 9  # per sign set: BPL not taken 2 + LDA/ORA/STA 10, less 3
PROC_SC = 64  # $1C9D JSR 6 + the 58-cycle divide-by-16, BCC taken
PROC_SC_INVERT = 29  # BCC not taken 2 + $1009 JSR 6 + 24, less the taken 3
MUL_DBL_DBL = 415  # $0F9E JSR 6 + head 11 + the three $0D03 blocks 383 + tail 15
MUL_DBL_DBL_NEG_Y = 25  # $0FA0 BPL not taken: negate y and flip $0067
MUL_DBL_DBL_ODD = 7  # $0FB9 BEQ not taken: flip $0067 on x's bit 0
MUL_DBL_DBL_CARRY = 4  # per $0FD3/$0FE5/$0FFF carry: INC, less the taken branch
MUL_DBL_DBL_NEG = 17  # $1007 BPL not taken + the $1009 invert
PREP_VEC = 135  # $1C54's own line 39 + the two $1C7D set_vector bodies 96

# One $1CE8 sub-step from its own branches; a taken branch is 3, 4 when it crosses a
ADD_VECTOR = 163  # page ($1CF1/$1CF9->$1D44, $1D18/$1D40->$1CE8). $1CE8 JSR 6 + 157
ADD_VECTOR_NEG = 4  # per component whose high byte is negative: $1CCC DEC $0074
STEP_EDGE = 10  # $1CEB/$1CF3 LDA/STA/CMP 8 + the BCS not taken 2, per axis
STEP_EDGE_EXIT = 2  # off the board: BCS taken 4, page crossed, instead of 2
STEP_SETUP = 17  # $1CFB..$1D07: the $0060/$000C/$0079/$0C67 reset
TILE_Z_CALL = 6  # JSR $1DF9 at $1D08 and at $1D4E/$1D55/$1D5C
TILE_ADDR = 40  # $1DF9 JSR $2BA8 6 + calculate_tile_address 34
TILE_Z_READ = 9  # $1DFC LDA ($5E),Y 5 + CMP #$C0 2 + $1E00 BCS not taken 2
TILE_Z_FLAT = 27  # $1E02..$1E0D: the nibble/height split and RTS
TILE_Z_OBJ = 1  # $1E00 BCS taken 3 instead of 2
LEAVE_SET = 8  # $1D44 SEC 2 + RTS 6, every blocked exit
FLAT_BRANCH = 2  # $1D0B BCS $1D46 not taken
FLAT_DIFF = 18  # $1D0D..$1D16, before the $1D18 BMI
FLAT_BELOW = 4  # $1D18 BMI $1CE8 taken, page crossed: keep marching
FLAT_ABOVE = 5  # BMI not taken 2 + $1D1A BNE $1D44 taken 3
FLAT_TOL = 8  # BNE not taken 2 + $1D1C LDA $0079 / CMP $000C 6
FLAT_TOL_HIT = 3  # $1D20 BCS $1D44 taken
FLAT_BIT60 = 5  # BCS not taken 2 + $1D22 BIT $0060 3
FLAT_BIT60_HIT = 3  # $1D24 BVS $1D44 taken
FLAT_ANGLE = 10  # BVS not taken 2 + $1D26 LDA $0C6E / ORA $0C67 8
FLAT_ANGLE_SKIP = 3  # $1D2C BMI $1D32 taken
FLAT_LOOKUP = 5  # BMI not taken 2 + $1D2E LDA $0030 3
FLAT_LOOKING_UP = 3  # $1D30 BPL $1D44 taken
FLAT_SAME = 10  # $1D32 LDX $6E / LDA $0024 / CMP $0900,X
FLAT_SAME_X_DIFF = 11  # $1D39 BNE $1D42 taken 3 + $1D42 CLC 2 + RTS 6
FLAT_SAME_Y = 9  # BNE not taken 2 + $1D3B LDA $0026 / CMP $0980,X 7
FLAT_SAME_HIT = 4  # $1D40 BEQ $1CE8 taken, page crossed: keep marching
FLAT_SAME_Y_DIFF = 10  # BEQ not taken 2 + $1D42 CLC 2 + RTS 6
SLOPE_BRANCH = 3  # $1D0B BCS $1D46 taken
SLOPE_HEAD = 87  # $1D46..$1D68 with $2BA8, the three $1DF9 corner reads apart
SLOPE_NIB_4 = 5  # $1D6A CMP #$04 2 + BEQ $1D72 taken 3
SLOPE_NIB_12 = 8  # ... BEQ not taken 2 + CMP #$0C 2 + BNE not taken 2
SLOPE_NIB_QUAD = 9  # ... BNE $1D8A taken 3
SLOPE_EDGE_LDA = 3  # $1D72 LDA $003B
SLOPE_EDGE_MISS = 5  # $1D74.. CMP zp 3 + BCS not taken 2, per corner above the ray
SLOPE_EDGE_HIT = 9  # CMP 3 + BCS taken 3 + $1D87 JMP $1CE8 3
SLOPE_EDGE_BLOCK = 3  # all four below: $1D84 JMP $1D44 3
SLOPE_Q_C2 = 7  # $1D8A LSR 2 + BCC $1D9C taken 3 + $1D9C LSR 2
SLOPE_Q_C1 = 15  # LSR 2 + BCC not taken 2 + LSR 2 + BCS taken 3 + $1D95 ADC/AND/JMP 7
SLOPE_Q_EDGE = 9  # ... BCS not taken 2 + $1D90 AND #$01 2 + JMP $1DAF 3
SLOPE_Q_CORNER = 25  # $1D9D..$1DAC, the $1DA2 BCC taken
SLOPE_Q_CORNER_EOR = 1  # $1DA2 BCC not taken 2 + EOR #$FF 2, less 3
SLOPE_Q_TAIL = 198  # $1DAF..$1DEE, every branch taken, the $0D03 body included
SLOPE_Q_USE_Y = 2  # $1DB3 BCS not taken 2 + $1DB5 LDY $0039 3, less 3
SLOPE_Q_INVERT = 1  # $1DB9 BCC not taken 2 + $1DBB EOR #$FF 2, less 3
SLOPE_Q_ABS = 5  # $1DC9 BPL not taken 2 + $1DCB EOR/CLC/ADC 6, less 3
SLOPE_Q_NEG = 17  # $1007 BPL not taken 2 + the $1009 invert 18, less 3
SLOPE_Q_BLOCK = -1  # $1DE9 BPL not taken 2 + JMP $1D44 3, against the taken 3 + 3
# $1E00 BCS get_tile_z_from_object, charged per stack level by the branch it takes.
OBJ_HEAD_GHOL = 10  # $1E3F AND/TAY/BIT 7 + $1E44 BPL taken 3
OBJ_HEAD_LOS = 12  # ... $1E44 BPL not taken 2 + $1E46 BMI taken 3
OBJ_TARGET_HIT = 11  # $1E0E CPY 3 + BNE not taken 2 + $1E13 ROR $0C56 6
OBJ_TARGET_MISS = 6  # $1E0E CPY 3 + $1E11 BNE taken 3
OBJ_TYPE_BOULDER = 9  # $1E16 LDA 4 + CMP #3 2 + BEQ taken 3
OBJ_TYPE_TREE = 13  # + $1E1D CMP #2 2 + BEQ taken 3, less the untaken BEQ 1
OBJ_TYPE_OTHER = 17  # + $1E21 CMP #6 2 + $1E23 BNE taken 3, less 1
OBJ_TYPE_PLATFORM = 16  # ... $1E23 BNE not taken 2
MINXY = 44  # $1EAF JSR 6 + the 38-cycle straight line, both BPLs and the BCS taken
MINXY_ABS = 1  # $1EB4/$1EBF BPL not taken 2 + EOR #$FF 2, less the taken branch 3
MINXY_Y_WINS = 2  # $1EC5 BCS not taken 2 + $1EC7 LDA $0074 3, less 3
OBJ_BT_SKIP = 5  # $1E4B CMP #$40 2 + BCS taken 3
OBJ_BT_TYPE = 4  # $1E4B CMP #$40 2 + BCS not taken 2
OBJ_BT_TREE = 9  # + $1E4F LDA 4 + CMP #2 2 + BEQ taken 3
OBJ_BT_BOULDER = 41  # + $1E56..$1E68 the near-centre boulder's own RTS 33
OBJ_PLAT_SKIP = 5  # $1E28 CMP #$64 2 + BCS taken 3
OBJ_PLAT_RTS = 34  # $1E28 2 + BCS not taken 2 + $1E2C..$1E3E 30
OBJ_TREE_BELOW = 41  # $1E69..$1E81 BMI taken: the tree is under the ray
OBJ_TREE_HIGH = 52  # + $1E83 LSR/ROR/LSR 9 + $1E87 BNE taken 3, less 1
OBJ_TREE_NEAR = 62  # + $1E89 LDA/ROR/CMP 8 + $1E8E BCC taken 3, less 1
OBJ_TREE_TARGETED = 68  # + $1E90 BIT $0C56 4 + BMI taken 3, less 1
OBJ_TREE_SEEN = 75  # + $1E95 SEC 2 + ROR $0CDD 6, less the taken BMI 1
OBJ_SKIP_TREE = 9  # $1E99 LDA 4 + CMP #2 2 + BEQ taken 3
OBJ_SKIP_OTHER = 13  # + $1EA0 LDA #$C0 2 + STA $0060 3, less 1
OBJ_GHOL_LOOP = 9  # $1EA4 LDA 4 + CMP #$40 2 + BCS taken 3: another stack level
OBJ_GHOL_RTS = 18  # + $1EAB LDA 4 + RTS 6, less 1

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
