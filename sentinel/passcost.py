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
CONSIDER_ENTRY = 21  # $16E6 gate open 8 + $16ED reload 7 + $16F2 FOV width 6
CONSIDER_MEANIE = 7  # $16F7 LDA $0CA0,X 4 + $16FA BPL $16FF taken 3: owns a meanie
CONSIDER_NO_MEANIE = 9  # ... BPL not taken 2 + $16FC JMP $1773 3
DISCHARGE_CALL = 9  # $1773 STX $6E 3 + $1775 JSR $1A5D 6
DISCHARGED = 5  # $1778 BCS not taken 2 + $177A JMP $1876 3: this update is spent
NO_DISCHARGE = 10  # ... BCS taken 3 + $177D LDX $90 3 + $177F LDA $0CB8,X 4
HUNT_CLEAR = 3  # $1782 BPL $1795 taken: not mid meanie-hunt
HUNT_CALL = 8  # ... not taken 2 + $1784 JSR $1AB0 6
HUNT_MISS = 13  # $1787 LDX 3 + $1789 BCS taken 3 + $1792 LSR $0CB8,X 7: flag decays
HUNT_HIT = 15  # ... BCS nt 2 + $178B LDA 2 + STA $0C80,X 5 + $1790 BNE $17EA 3
HELD_NONE = 7  # $1795 LDA $0C20,X 4 + $1798 BEQ $17AC taken 3: nothing held
HELD_CALL = 18  # ... BEQ nt 2 + $179A LDY 4 + $179D LDA 2 + $179F JSR $1887 6
HELD_LOST = 11  # $17A2 LDA $14 3 + $17A4 BEQ taken 3 + $17A9 STA $0C20,X 5
HELD_KEPT = 8  # ... BEQ not taken 2 + $17A6 JMP $1825 3
SCAN_INIT = 7  # $17AC LDA 2 + STA $0F 3 + $17B0 LDY #$3F 2

REDUCE_HEAD = 7  # $1A08 LDX $0C58 4 + $1A0B CPX $0B 3, as its caller sees it
REDUCE_KILL = 27  # $1A0D BNE nt 2 + LDA 4 + $1A12 BEQ $1A00 taken 3 + $1A00 9 + $1AF9 9
REDUCE_PLAYER = 35  # ... BEQ nt 2 + $1A14 8 + JSR $9508 6 + LDA 2 + JSR $3470 6 + 5
REDUCE_OBJECT = 46  # $1A0D BNE taken 3 + TXA 2 + JSR $1AF4 6 + $1AF4 31 + LDA type 4
REDUCE_ROBOT = 24  # $1A2D BNE nt 2 + $1A2F..$1A38 15 + $1A4B STA 5 + $1A4E CLC 2
REDUCE_TREE = 18  # ... BNE taken 3 + CMP 2 + BNE nt 2 + JSR $1EEF 6 + JMP 3 + CLC 2
REDUCE_BOULDER = 23  # ... BNE $1A44 taken 3 + $1A44..$1A4B 13 + CLC 2
REDUCE_BANK = 29  # $1A4F PHP 3 + LDY 3 + LDA 4 + CLC/ADC 4 + STA 5 + PLP 4 + RTS 6
REMOVE_STACKED = 86  # $1EEF for an object standing on another: the flags go back
REMOVE_GROUND = 94  # ... on bare ground: the tile byte is rebuilt from the z nibble
TUNE = 323  # $3470 start_tune, the sound a drain makes; also inside ROTATE

# plot_status_bar $9508 pads to fixed columns, so its cost is a function of the energy.
STATUS_CHAR = 38  # $9579 JSR 6 + PHA/LDX/CLC/ADC/STA/INC/PLA/RTS 32
STATUS_HEAD = 51  # $9508 LDA/STA 6 + one character 38 + $9510 LDA/STA 7
STATUS_BLOCK = 95  # one $9515/$952C lap: subtract, then plot two characters
STATUS_BLOCK_DONE = 8  # its LDA/CMP and the BCC out of the lap
STATUS_UNIT_NONE = 5  # $9543 CMP #$01 2 + BCC $9551 taken 3: nothing left over
STATUS_UNIT = 86  # ... not taken 2 + $9547..$954E: the odd unit's two characters
STATUS_PAD = 49  # $9551/$9562 one padding character; the last lap's BCC is 1 shorter
STATUS_MID = 40  # $955D the bar's own separator character
STATUS_TAIL = 86  # $956E..$9578 the last two characters and the RTS
STATUS_PAD_END = 0x1D  # $9559: the first pad run fills up to column 29
STATUS_PAD_LAPS = 8  # $956A: the second run fills $1E..$25 whatever the energy

DISCHARGE_NONE = 18  # $1A5D LDX 3 + SEC 2 + LDA 4 + BEQ 3 + RTS 6
DISCHARGE_FIXED = 100  # $1A5D success 62 + create_object $211D + $1238 entry
DISCHARGE_TRY = 966  # $1238 loop body 76..98 + two $1272 draws (445 + 440n each)

SEE_SLOT_EMPTY = 40  # $1887 exit at $1893
SEE_SLOT_WRONG_TYPE = 49  # $1887 exit at $189D
SEE_PROLOGUE = 37  # $1887..$189D both compares falling through + the $189F JSR $8401
SEE_FOV = 54  # $18A2..$18C7; $0F3D & $0F is 0, so the $18AD BEQ is never taken
SEE_FOV_REJECT = 18  # $18CA BCS $1917 taken 4, page crossed, + $1917..$191D 14
SEE_FOV_PASS = 22  # $18CA BCS not taken 2 + $18CC..$18D8 20
SEE_ROBOT = 19  # $18DA BNE not taken 2 + the $18DC head-probe setup 17
SEE_NOT_ROBOT = 4  # $18DA BNE $1904 taken, page crossed
SEE_PROBE_FIXED = 56  # $18E6 JSR 6 + $18E9 15 + the $1C54/$1CDD JSRs 12 + $18F9 23
SEE_BASE = 26  # $1904..$1912: the base point and the $1E probe counter
SEE_BASE_AGAIN = 4  # $1914 BNE $18E6 taken, page crossed
SEE_BASE_LAST = 2  # $1914 BNE not taken
SEE_TAIL = 16  # $1916 CLC 2 + $1917..$191D 14
MARCH_ENTRY = 74  # $1CDD's own 15 + JSR $1ECC 6 + get_object_details 53

# $8401 calculate_object_relative_angles_and_distance, and the trig beneath it.
REL_XY = 66  # $85C4 JSR 6 + 60, both component deltas non-negative
REL_XY_ABS = 6  # per negative delta: $85D5/$85EB BPL not taken 2 + the negate 7, -3
REL_Z = 36  # $85F5 JSR 6 + the 30-cycle straight line
REL_ANGLES = 119  # $8401's own line 82 + $843E BNE taken 3 + the $8460 stores 34;
ANG_CMP = 6  # $9287 LDA $85 / CMP $83
ANG_X_LARGER = 3  # $928B BCC $9295 taken
ANG_Y_LARGER = 5  # ... not taken 2 + $928D BNE $92A8 taken 3
ANG_EQ = 10  # ... BNE not taken 2 + $928F LDA $82 / CMP $80 6
ANG_EQ_Y = 3  # $9293 BCS $92A8 taken
ANG_EQ_X = 2  # $9293 BCS not taken
ANG_MIN_Y = 27  # $9295..$92A5: min = y, max = x, and the JMP into the shift loop
ANG_MIN_X = 27  # $92A8..$92B8: min = x, max = y, through the $92B8 ORA $82
ANG_ZERO = 18  # $92BA BEQ $9280 taken 3 + the $9280 zero-angle exit 15
ANG_NONZERO = 8  # ... not taken 2 + $92BC LDA $85 3 + JMP $9303 3
SCALE_SHIFT = 9  # $92C5/$9303 ASL/ROL 7 + the loop's final BCC not taken 2
SCALE_LOOP = 20  # $92C1 per loop-back: $92C8 BCC taken 3 + 10 + the next ASL/ROL 7
SCALE_LOOP_Y = 21  # $92FF: its $9306 BCC crosses a page, so the loop-back is 4
SCALE_TAIL = 33  # $92CA/$9308 to the JSR $0D4A: back off the shift, set the divisor
ANG_SIGN = 6  # $92DE/$931C LDA $86 / EOR $88
ANG_SIGN_KEEP = 3  # the $92E2/$9320 branch that skips the negate
ANG_SIGN_NEGATE = 20  # ... not taken 2 + the $92E4/$9322 16-bit negate 18
ANG_QUAD = 5  # $92F1/$932F LDA #imm 2 + BIT 3
ANG_QUAD_LOW = 3  # $92F5/$9333 BPL taken: keep the first quadrant constant
ANG_QUAD_HIGH = 4  # ... not taken 2 + the second LDA #imm 2
ANG_QUAD_TAIL = 14  # CLC / ADC $8B / STA $8B / RTS

# $0D4A divide_and_arctan: three 16-bit rounds, six 8-bit, a half, then the lookup.
DIV_FULL_SHIFT = 7  # $0D4A/$0D68/$0D86 ASL-or-ROL $74 5 + ROL A 2
DIV_FULL_CARRY = 3  # $0D4D BCS taken: the compare is skipped, a is over
DIV_FULL_UNDER = 8  # BCS not taken 2 + CMP $76 3 + $0D51 BCC taken 3
DIV_FULL_OVER = 10  # ... BCC not taken 2 + $0D53 BNE taken 3
DIV_FULL_EQ = 15  # ... BNE not taken 2 + $0D55 LDY $74 / CPY $77 6
DIV_FULL_EQ_UNDER = 3  # $0D59 BCC taken
DIV_FULL_EQ_OVER = 2  # $0D59 BCC not taken
DIV_SUB = 20  # $0D5B..$0D67: the 16-bit subtract and the SEC that carries the bit
DIV_SKIP_TEST = 6  # $0DA4 PHP 3 + CMP $76 3
DIV_SKIP = 28  # $0DA7 BEQ $0E10 taken 4, page crossed, + the $0E10 short exit 24
DIV_NO_SKIP = 2  # $0DA7 BEQ not taken
DIV_BYTE_SHIFT = 7  # $0DA9.. ASL-or-ROL $74 5 + ROL A 2
DIV_BYTE_CARRY = 8  # BCS taken 3 + SBC $76 / SEC 5
DIV_BYTE_UNDER = 8  # BCS not taken 2 + CMP $76 3 + BCC taken 3
DIV_BYTE_OVER = 12  # ... BCC not taken 2 + SBC $76 / SEC 5
DIV_LAST_SHIFT = 7  # $0DF1 ROR $78 5 + ROL A 2
DIV_LAST_CARRY = 3  # $0DF4 BCS $0DF8 taken
DIV_LAST_CMP = 5  # ... not taken 2 + CMP $76 3
DIV_LAST_TAIL = 8  # $0DF8 ROR $78 5 + $0DFA LDA $74 3
DIV_PLP = 4  # $0DFC PLP: the round-3 carry comes back
DIV_ROUND3_UNDER = 7  # $0DFD BCC $0E01 taken 4, page crossed, + $0E01 BCC taken 3
DIV_ROUND3_OVER = 4  # ... not taken 2 + $0DFF ADC #$1f 2
DIV_NO_OVERFLOW = 3  # $0E01 BCC $0E1F taken
DIV_OVERFLOW = 23  # ... not taken 2 + the $0E03 45-degree clamp and RTS 21
DIV_ARCTAN = 22  # $0E1F TAY/STA + the two table reads + $0E2C BIT $78
DIV_ARCTAN_DONE = 13  # $0E2E BMI nt 2 + $0E30 BVS nt 2 + JMP $0E74 3 + RTS 6
DIV_DELTA_BR = 3  # $0E2E BMI $0E35 taken: round 10 was over
DIV_HALF_BR = 5  # ... not taken 2 + $0E30 BVS $0E50 taken 3
DIV_DELTA = 22  # $0E35..$0E42: arctan[Y] - arctan[Y+1], then BIT $78
DIV_DELTA_KEEP = 3  # $0E44 BVC $0E49 taken
DIV_DELTA_INVERT = 32  # ... not taken 2 + JSR $1009 6 + the invert 24
DIV_DELTA_HALVE = 15  # $0E49..$0E4E: the sign-preserving >>1
DIV_NEXT = 25  # $0E50..$0E5F: angle += arctan[Y+1], then BIT $78
DIV_ADD_SKIP = 3  # $0E61 BPL $0E70 taken
DIV_ADD_HALF = 22  # ... not taken 2 + $0E63..$0E6E add the half-delta 20
DIV_AVERAGE = 16  # $0E70 LSR $8B / ROR $8A / RTS
DIV_TABLE_CROSS = 1  # $3B01,Y crosses a page at Y $FF, $3C02,Y at Y $FE

VANG_HEAD = 5  # $933D STA $86 3 + TAY 2
VANG_POS = 3  # $9340 BPL $934D taken
VANG_NEG = 17  # ... not taken 2 + $9342..$934B: negate the relative z 15
VANG_SETUP = 26  # $934D..$935B: hypotenuse into y, zero the y sign, JSR $9287
VANG_SHIFT = 52  # $935E..$9377: subtract the observer's v_angle, PHP, >>4, PLP
VANG_SIGN_POS = 3  # $9378 BPL $937C taken
VANG_SIGN_NEG = 4  # ... not taken 2 + $937A ORA #$f0 2
VANG_TAIL = 9  # $937C STA $8D 3 + RTS 6

HYP_HEAD = 38  # $937F..$9395: the $3D02 coefficient read and the JSR $0F4A
HYP_TAIL = 40  # $9398..$93AC: halve, add the max component, restore Y, RTS

MUL8 = 102  # $0D03 JSR 6 + STA/LDA 5 + the 8 unrolled shift-add rounds 91
MUL8_LOW = 99  # the $0D05 entry, $0074 already stored
MUL8_BIT = 4  # per set bit of the multiplier: BCC not taken 2 + CLC 2 + ADC 3, less 3
MUL_DBL_BYTE = 230  # $0F4A..$0F61 with both $0D03 calls, BCC taken
MUL_DBL_BYTE_CARRY = 4  # $0F5D BCC not taken 2 + $0F5F INC $0075 5, less 3
MUL_DBL_BYTE_CALL = 6  # $0EAB/$0EDC JSR $0F4A; $0F3E falls in with no JSR
MUL_PI = 22  # $0F3E..$0F49, the two ASL/ROL and the $C9 setup
SIN_COS = 42  # $0E75 JSR 6 + $0E75..$0E8C 36, the $0E7B JSR $0F3E included
SIN_COS_NOQUAD = 3  # $0E8F BVC $0E94 taken
SIN_COS_QUAD = 9  # ... not taken 2 + $0E91 INX 2 + $0E92 DEC $0060 5
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
PREP_VEC = 141  # $1C54's own line 39 + $1C76 JSR 6 + the two $1C7D bodies 96

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
FLAT_TOL = 10  # $1D18 BMI nt 2 + $1D1A BNE nt 2 + $1D1C LDA $0079 / CMP $000C 6
FLAT_TOL_HIT = 3  # $1D20 BCS $1D44 taken
FLAT_BIT60 = 5  # BCS not taken 2 + $1D22 BIT $0060 3
FLAT_BIT60_HIT = 3  # $1D24 BVS $1D44 taken
FLAT_ANGLE = 10  # BVS not taken 2 + $1D26 LDA $0C6E / ORA $0C67 8
FLAT_ANGLE_SKIP = 3  # $1D2C BMI $1D32 taken
FLAT_LOOKUP = 7  # BMI not taken 2 + $1D2E LDA $0030 3 + $1D30 BPL not taken 2
FLAT_LOOKING_UP = 1  # $1D30 BPL $1D44 taken 3 instead of the not-taken 2
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
SLOPE_Q_C1 = 16  # LSR 2 + BCC not taken 2 + LSR 2 + BCS taken 3 + $1D95 ADC/AND/JMP 7
SLOPE_Q_EDGE = 13  # $1D8A LSR/BCC/LSR 6 + BCS nt 2 + $1D90 AND 2 + JMP 3
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
OBJ_TARGET_HIT = 12  # $1E0E CPY abs 4 + BNE not taken 2 + $1E13 ROR $0C56 6
OBJ_TARGET_MISS = 7  # $1E0E CPY abs 4 + $1E11 BNE taken 3
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

# find_drainable_robot_loop $17B2, one slot, by the branch it takes; the $1887 apart.
SCAN_SLOT_HIDDEN = 22  # $17B2 2 + JSR 6 + $17B7 6 + $17BC BNE taken 3 + DEY/BPL 5
SCAN_SLOT_UNSEEN = 27  # ... BNE nt 2 + $17BE LDA $14 3 + $17C0 BEQ taken 3 + 5
SCAN_SLOT_FULL = 25  # ... BEQ nt 2 + $17C2 BMI $1825 taken 4, page crossed
SCAN_SLOT_OTHER = 34  # ... BMI nt 2 + $17C4 CPY $0B 3 + $17C6 BNE taken 3 + 5
SCAN_SLOT_PARTIAL = 36  # ... BNE nt 2 + $17C8 STY $0F 3 + DEY/BPL 5
SCAN_LAST = 1  # slot 0 leaves by $17CB BPL not taken, one cycle short
SCAN_END = 6  # $17CD LDY $0F 3 + $17CF BMI $17E0 taken 3: no head-only player
SCAN_END_PARTIAL = 11  # ... BMI nt 2 + $17D1 TYA 2 + $17D2 CMP $0C90,X 4
PARTIAL_KNOWN = 3  # $17D5 BEQ $17E0 taken: this player already failed a hunt
PARTIAL_ARM = 49  # ... nt 2 + $17D7 JSR 6 + $1973 32 + LDA 2 + STA 3 + BNE 4
MEANIE_INIT = 32  # $1973..$1985: the four writes and the RTS
TREE_CALL = 13  # $17E0 LDA 2 + $17E2 STA $0C20,X 5 + $17E5 JSR $1AB0 6
TREE_NONE = 3  # $17E8 BCS $17F9 taken: nothing on a stack to drain
TREE_HIT = 2  # ... BCS not taken

# find_drainable_boulder_or_tree $1AB0 walks its own loop, not $17B2's.
TILE_SCAN_ENTRY = 2  # $1AB0 LDX #$3F
TILE_SCAN_EXHAUSTED = 8  # $1AF2 SEC 2 + RTS 6
TILE_SCAN_LAST = 1  # slot 0 leaves by $1AF0 BPL not taken, one cycle short
TILE_SCAN_EMPTY = 12  # $1AB2 LDA 4 + BMI 3 + $1AEF DEX/BPL 5
TILE_SCAN_OTHER = 24  # + the $1AB7 flag and $1ABB type compares 12
TILE_SCAN_STACKED = 11  # flags >= $40 leaves at $1AB9 BCS
TILE_SCAN_LOOSE = 18  # a lone boulder falls through the $1ABE type compare instead
TILE_SCAN_TILE = 61  # $1AC2..$1AD1 the tile fetch, its $2BA8 lookup 34 included
TILE_SCAN_NO_TILE = 8  # $1AD3 BCC taken + $1AEF DEX/BPL
TILE_SCAN_TOP = 12  # $1AD3 BCC 2 + $1AD5..$1ADB the top object's type read 10
TILE_SCAN_WRONG_TOP = 12  # $1ADD..$1AE1 the two compares 7 + $1AEF DEX/BPL 5
TILE_SCAN_SEE = 9  # a tree top: $1ADD BEQ $1AE3 taken 3 + the $1AE3 JSR 6
TILE_SCAN_SEE_BOULDER = 12  # ... BEQ nt 2 + $1ADF CMP #3 2 + $1AE1 BNE nt 2 + JSR 6
TILE_SCAN_NEXT = 11  # $1AE6 LDA/BPL 6 + $1AEF DEX/BPL 5 when the see was not full
TILE_SCAN_HIT = 17  # $1AE6 LDA $14 3 + BPL nt 2 + $1AEA STY/CLC 6 + RTS 6

# consider_creating_meanie $197D walks the search counter, not a slot index.
MEANIE_SCAN_SLOT = (
    26  # $198F LDX 3 + LDY 4 + BNE 3 + $19A1 DEC 7 + DEY 2 + LDA 4 + BMI 3
)
MEANIE_SCAN_OTHER = 34  # + the $19AA tree-type compare 9 - the taken BMI 1
MEANIE_SCAN_DX = 24  # $19B1..$19C5 the 10-tile x test, its $19BE sign fixup included
MEANIE_SCAN_DY = 18  # $19C7..$19D7 the same in y, off the already-loaded index
MEANIE_SCAN_SEE = 8  # $19D9 LDA 2 + the $19DB JSR 6
MEANIE_SCAN_DONE = 33  # $198F..$1994 9 + $1996 INC/LDA/STA 16 + SEC/RTS 8

DRAIN_CALL = 6  # $17EA JSR $1A08 reduce_object_energy
DRAIN_TAIL = 15  # $17ED BCS nt 2 + $17EF LDY 3 + LDA 2 + STA $0C30,Y 5 + JMP 3
DRAIN_TAIL_PLAYER = 7  # $17ED BCS $1802 taken 4, page crossed, + JMP $16D6 3
BODY_TAIL = 24  # $1876..$1884: the $0C6D flag, the JSR $1F9F and the JMP $16D6

# target_object $1825, entered from $17A6, $17C2, $17DE.
TARGET_HEAD = 21  # $1825..$1831: record the target and its exposure, read the timer
TARGET_FIRST = 12  # $1833 BCS nt 2 + $1835 LDA 2 + STA $0C20,X 5 + JMP 3: arm it
TARGET_WAIT = 9  # $1833 BCS taken 3 + $183D BNE taken 3 + $183A JMP $16D6 3
TARGET_DUE = 8  # ... $183D BNE not taken 2 + $183F LDA $14 3: the timer is up
TARGET_MEANIE = 3  # $1841 BPL $1852 taken: a head-only player buys a meanie hunt
TARGET_DRAIN = 18  # ... nt 2 + $1843 JSR $1A08 6 + $1846 LDY 3 + LDA 2 + STA 5
TARGET_DRAIN_OBJ = 5  # $184D BCS not taken 2 + $184F JMP $1876 3
TARGET_DRAIN_PLAYER = 6  # ... BCS $1884 taken 3 + JMP $16D6 3

ROTATE_GATE = 12  # $17F9 LDX 3 + LDA $0C28,X 4 + CMP 2 + $1800 BCC $1805 taken 3
ROTATE_GATE_HELD = 14  # ... BCC not taken 2 + $1802 JMP $16D6 3: too soon to turn
ROTATE = 454  # $1805..$1884: $1AF4 31 + $1973 32 + $3470 323 + $187B 18 + 50 straight
# $1F9F update_object_on_screen: no screen span, no replot -- see relative.py $209B.
REDRAW_CALL = 6  # $1F9F JSR $209B
REDRAW_NONE = 23  # $1FA2 BCS $1F93 taken 3 + the $1F93 flag reset and RTS 20
REDRAW_PLOT_ENTRY = 2  # $1FA2 BCS not taken; $1FA4 on is the strip replot, unpriced
SPAN_HEAD = 6  # $209B LDY $91 3 + CPY $0B 3
SPAN_PLAYER = 12  # $209F BEQ $2110 taken 4, page crossed, + $2110 SEC/RTS 8
SPAN_ANGLES = 8  # ... BEQ not taken 2 + $20A1 JSR $8401 6
SPAN_SIZE = 33  # $20A4..$20BB: the $2112 half-angle, the $0CD4 floor, the JSR $933D
SPAN_SIZE_FLOOR = 3  # $0CD4 is larger: BCS not taken 2 + $20B1 LDA $0CD4 4, less 3
SPAN_LEFT = 12  # $20BE..$20C4: bearing - half-angle, low byte
SPAN_LEFT_HI = 7  # $20C6 LDA $0C57 4 + SBC $8B 3
SPAN_LEFT_NEG = 7  # $20CB BPL nt 2 + LDA #0 2 + $20CF BEQ $20D8 taken 3: clip to 0
SPAN_LEFT_POS = 12  # ... BPL taken 3 + $20D1 ASL $74 / ROL A / CMP #$28 9
SPAN_OFF_RIGHT = 12  # $20D6 BCS $2110 taken 4, page crossed, + SEC/RTS 8
SPAN_LEFT_ONSCREEN = 2  # ... BCS not taken
SPAN_LEFT_OK = 8  # $20D8/$20DB: the left column into $0C62 and $211C
SPAN_RIGHT = 12  # $20DE..$20E4: bearing + half-angle, low byte
SPAN_RIGHT_HI = 7  # $20E6 LDA $0C57 4 + ADC $8B 3
SPAN_BEHIND = 12  # $20EB BMI $2110 taken 4, page crossed, + SEC/RTS 8
SPAN_RIGHT_OK = 11  # ... BMI not taken 2 + $20ED ASL $74 / ROL A / CMP #$28 9
SPAN_WIDTH_CLIP = 1  # $20F2 BCC not taken 2 + $20F4 LDA #$27 2, less the taken 3
SPAN_WIDTH = 21  # $20F2 BCC taken 3 + $20F6..$2100: right + 1 - left into $0C6A/$211B
SPAN_ZERO_WIDTH = 11  # $2103 BEQ $2110 taken 3 + SEC/RTS 8
SPAN_VISIBLE = 19  # ... BEQ nt 2 + $2105 CMP/BCC 5 + $210B STA $0C69 4 + CLC/RTS 8
SPAN_WIDTH_CAP = 1  # $2107 BCC not taken 2 + $2109 LDA #$14 2, less the taken 3
MEANIE_ROTATE = 444  # $1728..$1884, the meanie's own turn: $1AF4 31 + $3470 323 + 90

# $2845 check_if_tile_is_on_screen_and_calculate_screen_coordinates, by its branches; the trig it calls ($9287, $937F, $933D) is counted by relative.py off its own data.
EXAM_CALL = 6  # the JSR, from $26D6/$270A/$271E/$2738/$274C and the $27D7 loops
EXAM_HEAD = 44  # $2845..$2861: the slot index, the $007F reset and the column delta
EXAM_COL_POS = 3  # $2863 BPL $2870 taken: the column delta is already positive
EXAM_COL_NEG = 17  # ... not taken 2 + the $2865 16-bit negate 15
EXAM_ROW = 19  # $2870..$287B: the row delta against the observer's row
EXAM_ROW_POS = 3  # $287D BPL $288A taken
EXAM_ROW_NEG = 17  # ... not taken 2 + the $287F negate 15
EXAM_ANGLE = 9  # $288A STA $85 3 + $288C JSR $9287 6
EXAM_STORE = 36  # $288F..$28A0 the two 16-bit stores and the JSR $937F 33 + $28A3 BIT 3
EXAM_QUAD_NORTH = 13  # $28A5 BMI nt 2 + $28A7 BVS nt 2 + $28A9..$28AD 9
EXAM_QUAD_EAST = 20  # ... BVS taken 3 + $28B0..$28B8 15
EXAM_QUAD_SOUTH = 26  # $28A5 BMI taken 3 + $28BB BVS nt 2 + $28BD..$28C9 21
EXAM_QUAD_WEST = 18  # ... BVS taken 3 + $28CC..$28D2 12, falling into $28D4
EXAM_ADDR = 47  # $28D4..$28F0 calculate_tile_address through the CMP #$c0
EXAM_OBJECT = 2  # $28F2 BCC not taken: the tile holds an object stack
EXAM_OBJECT_STEP = 13  # $28F4 one level whose flags say another is below it
EXAM_OBJECT_BOTTOM = 22  # the bottommost level 12 + $28FE its z_height and the JMP 10
EXAM_GROUND = 37  # $28F2 BCC taken 4, page crossed, + $2906..$2919 the $3E80 bit 33
EXAM_VISIBLE = 3  # $2919 BNE taken: the raytrace bit is set
EXAM_HIDDEN = 7  # ... not taken 2 + $291B STA $0180,X 5: the plot byte is zeroed
EXAM_VANGLE = 53  # $291E..$293F: relative z, JSR $933D, both stores, the $0007 CMP
EXAM_OFF_LEFT = 3  # $2941 BCC leave taken: beyond the left edge of the buffer
EXAM_ON_LEFT = 2  # ... not taken
EXAM_NO_FRACTION = 3  # $2943 BNE taken: the high byte alone settles the left edge
EXAM_FRACTION = 9  # ... not taken 2 + $2945 LDA $0BA0,Y 4 + CMP $0028 3
EXAM_FRACTION_LEFT = 3  # $294A BCC leave taken
EXAM_FRACTION_OK = 6  # ... not taken 2 + $294C LDA $A800,Y 4
EXAM_RIGHT = 8  # $294F ROR $007F 5 + CMP $0012 3
EXAM_OFF_RIGHT = 3  # $2953 BCC leave taken
EXAM_ON_RIGHT = 7  # ... not taken 2 + $2955 INC $007F 5
EXAM_TAIL = 14  # $2957 LDY $0F 3 + LDA $7F 3 + CLC 2 + RTS 6


def status_bar_cycles(energy):
    """$9508 plot_status_bar: 15-blocks, 3-blocks, the odd unit, then the padding."""
    fifteens, rest = divmod(energy, 15)
    threes, unit = divmod(rest, 3)
    total = STATUS_HEAD + (fifteens + threes) * STATUS_BLOCK + 2 * STATUS_BLOCK_DONE
    total += STATUS_UNIT_NONE if unit == 0 else STATUS_UNIT
    chars = 1 + 2 * (fifteens + threes + (1 if unit else 0))
    total += STATUS_PAD * max(1, STATUS_PAD_END - chars) - 1
    return total + STATUS_MID + STATUS_PAD * STATUS_PAD_LAPS - 1 + STATUS_TAIL


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


# plot_world's fill sequence block by block off the disassembly: the $2A24 tile dispatch, $2D6C prepare_polygon, $2EE4 the edge DDA and $22AA span_fill (sentinel/rendercost.py).
TILE_ENTRY = 25  # $2A15..$2A1F the slot arithmetic 19 + BMI nt 2 + $2A24 LDA $0180,X 4
TILE_SOUND = 35  # $2A12 JSR 6 + update_sound $352C's 29-cycle body
TILE_EMPTY = 9  # $2A27 BEQ taken 3 + $2ACB RTS 6: the slot is zero, nothing plotted
TILE_KEPT = 4  # ... BEQ not taken 2 + $2A29 CMP #$C0 2
TILE_OBJECT = 18  # $2A2B BCC nt 2 + PHA 3 + JSR 6 + PLA 4 + JMP $21AE 3
TILE_GROUND = 5  # $2A2B BCC taken 3 + $2A35 AND #$0F 2
TILE_FLAT = 3  # $2A37 BEQ plot_checkerboard_tile taken
TILE_EDGE_NIB = 25  # ... nt 2 + CMP #$0C 2 + BEQ 3 + $2A41..$2A49 the $0045 flag 18
TILE_SLOPE_CALC = 25  # ... both compares failing 8 + BNE 3 + $2A69..$2A74 16, -2
TILE_SLOPE_QUAD = 3  # $2A74 BEQ plot_quadrilateral_tile taken
TILE_SLOPE_TRI = 28  # ... not taken 2 + $2A76..$2A86 the $2D39 flag and the colour 26
TILE_SLOPE_TRI_ALT = 1  # $2A86 BCS not taken 2 + $2A88 ORA #$10 2, less the taken 3
CHECKER_HEAD = 10  # $2A4B LDX #0 2 + $2A4D the tile-parity EOR 8
CHECKER_EVEN = 3  # $2A53 BEQ plot_quadrilateral_tile taken
CHECKER_ODD = 13  # ... not taken 2 + $2A55 LDX #8 2 + the $0C75 prnd copy 9
QUAD_TILE = 15  # $2A5D the colour, $003B = 0 and the JMP into plot_polygon
TWO_TRI_FIRST = 23  # $2A8A..$2A96: $0034, $003B = $80, the colour and the JSR
TWO_TRI_SECOND = 22  # $2A99..$2AA7: the other colour and $003B |= $40
PP_HEAD = 6  # $2AA9 LDY $0010 3 + CPY #$02 3
PP_WIDE = 8  # $2AAD BCS not taken 2 + $2AAF JSR prepare_polygon 6
PP_TALL = 9  # ... BCS taken 3 + $2AC3 JSR prepare_polygon 6
PP_NOTHING = 3  # $2AB2/$2AC6 BCS taken: prepare_polygon found nothing to fill
PP_SPAN = 8  # $2AB2 BCS not taken 2 + $2AB4 JSR span_fill 6
PP_CLIP_TEST = (
    12  # $2AB7 LDY 3 + LDA $2C,Y 4 + CMP #$01 2 + BEQ 3: the section is clean
)
PP_CLIPPED = 2  # ... BEQ not taken: the polygon ran off an edge, so the other section
PP_RTS = 6  # $2ACB RTS
BUFVARS = 75  # $2AC0 JSR 6 + $298D's own 7 + initialise_buffer_variables $2993 62
PREP_HEAD = 9  # $2D6C LDA $25 3 + ORA $05 3 + BIT $3B 3
PREP_QUAD = 40  # $2D72/$2D74 both branches falling through 4 + $2D76..$2D8E 36
PREP_TRI_HEAD = 6  # $2D72 BMI taken 3 + $2D49 LDY $0045 3
PREP_TRI_ONE = 3  # $2D4B BVC is_triangle_one taken
PREP_TRI_TWO = 12  # ... not taken 2 + $2D4D..$2D53 the +$21 and the two INYs 10
PREP_TRI_TAIL = 37  # $2D54..$2D6A the four vertex stores 28 + BNE 3 + $2D8E 6
PREP_DATA_HEAD = 8  # $2DC9 LDA #0 2 + STA $6C 3 + LDY $17 3
PREP_VERTEX = 61  # $2DCF one vertex inside the inner area, through the $2DF0 BPL
PREP_VERTEX_LAST = 60  # ... the $2DF0 BPL not taken
PREP_VERTEX_OUT = 31  # $2DCF..$2DE1 the sum 28 + the BCS into $2D93 3
PREP_WIDE_HEAD = 8  # $2D93 LDA #$C0 2 + STA $6C 3 + LDY $17 3
PREP_WIDE_VERTEX = 74  # $2D99 one vertex, its screen_x high byte non-negative
PREP_WIDE_LAST = 73  # ... the $2DC4 BPL not taken
PREP_WIDE_NEG = 1  # $2DBC BCC not taken 2 + $2DBE ORA #$F8 2, less the taken 3
PREP_WIDE_JMP = 3  # $2DC6 JMP process_lines
LINES_HEAD = 23  # $2DF2..$2E02: the row floor/ceiling, $001E and $007F
LINE_HEAD = 45  # $2E04..$2E20: both vertices, the 16-bit y delta and the BPL
LINE_DOWN = 3  # $2E20 BPL skip_inversion taken: the line already slopes downwards
LINE_UP = 37  # ... not taken 2 + $2E22..$2E37 swap the ends and negate the delta 35
LINE_MID = 6  # $2E39 STA $0076 3 + $2E3B BIT $006C 3
LINE_INNER = 3  # $2E3D BVC line_is_entirely_within_inner_area taken
LINE_OUTER = 12  # ... not taken 2 + $2E3F the two screen_x high bytes 8 + BEQ nt 2
LINE_OUTER_INNER = 13  # ... with the $2E45 BEQ taken instead
LINE_OUTER_STEEP = 9  # $2E47 LDA 3 + BNE taken 3 + $2E4F JMP process_line 3
LINE_OUTER_FLAT = 11  # ... BNE nt 2 + $2E4B LDA 3 + BEQ nt 3 + $2E4F JMP 3
LINE_OUTER_POINT = 8  # ... with the $2E4D BEQ consider_next_line taken
LINE_STEEP = 23  # $2E52 LDA 3 + BEQ nt 2 + $2E56 zero both screen_x highs 12 + JMP 6
LINE_SHALLOW = 6  # $2E52 LDA 3 + $2E54 BEQ process_horizontal_line taken 3
LINE_IS_POINT = 6  # $2E61 LDA $000C 3 + $2E63 BEQ line_is_a_point taken 3
LINE_RASTERISE = 56  # ... BEQ nt 2 + $2E65..$2E89 the eight copies and the JSR 54
POINT_LINE = 20  # $2ECC..$2EDD: the two clamps, INC $001E and the JMP, both BCC taken
POINT_LINE_FLOOR = 1  # $2ED1 BCC not taken 2 + $2ED3 STA $0031 3, less the taken 3 + 1
POINT_LINE_CEIL = 1  # $2ED7 BCS not taken 2 + $2ED9 STA $0030 3, less the taken 3 + 1
LINE_NEXT = 13  # $2E8C LDY 3 + INY 2 + CPY $17 3 + BEQ nt 2 + $2E93 JMP 3
LINE_LAST = 11  # ... $2E91 BEQ check_if_polygon_is_point taken 3
LINES_TAIL = 6  # $2E96 LDA $001E 3 + CMP $0017 3
LINES_NOT_POINT = 3  # $2E9A BNE polygon_isn't_point taken
LINES_ALL_POINTS = 13  # ... nt 2 + $2E9C LDA $0AE0,X 4 + BNE nt 2 + LDY 4 + $2EA4 CPY 3
LINES_POINT_OFF = 3  # $2E9F/$2EA6/$2EAA leaving for polygon_isn't_point
LINES_POINT_ROW = 31  # $2EAC..$2EBC: the single row, its two edges and $007F = 0
LINES_DONE = 15  # $2EBE LDA 3 + BNE nt 2 + $2EC2 LDA/CMP 6 + BCC nt 2 + $2EC8 CLC 2
LINES_NOTHING = 8  # leaving by $2EC0/$2EC6 for $2ECA SEC 2 + RTS 6, the tests apart
LINES_RTS = 6  # $2EC9 RTS
EDGE_CALL = 6  # $2E89/$3083/$30B7 JSR rasterise_polygon_edge
EDGE_HEAD = 6  # $2EE4 LDA $003F 3 + $2EE6 BMI 3
EDGE_ABOVE = 9  # $2EE8 BNE leave taken 3 + $2EE3 RTS 6
EDGE_BOTTOM_TEST = 13  # $2EEA LDA 3 + CMP $51 3 + BCS nt 2 + CMP $04 3 + BCS nt 2
EDGE_BOTTOM_OFF = 12  # $2EEC CMP $51 3 + BCS taken 3 + RTS 6, the $2EEA LDA apart
EDGE_BOTTOM_KEEP = 3  # $2EF2 BCS not_new_bottom_row taken
EDGE_BOTTOM_SET = 9  # $2EF4 CMP $52 3 + BCS taken 3 + $2EFA STA $04 3
EDGE_BOTTOM_CLIP = 8  # ... BCS not taken 2 + $2EF8 LDA $52 3 + $2EFA STA $04 3
EDGE_BOTTOM_LOW = 9  # $2EE6 BMI taken 3 + $2EF8 LDA 3 + $2EFA STA 3
EDGE_TOP_TEST = 13  # $2EFC LDA $3E 3 + BMI nt 2 + BNE nt 2 + $2F02 LDA $1A 3 + CMP 3
EDGE_TOP_OFF = 9  # $2EFE BMI leave taken 3 + RTS 6
EDGE_TOP_BELOW = 9  # $2F06 BCC leave taken 3 + RTS 6
EDGE_TOP_KEEP = 3  # $2F0A BCC not_new_top_row taken
EDGE_TOP_SET = 9  # $2F0C CMP $51 3 + BCC taken 3 + $2F15 STA $06 3
EDGE_TOP_CLIP = 12  # ... BCC not taken 2 + $2F10 LDA/SEC/SBC 7 + $2F15 STA $06 3
EDGE_TOP_HIGH = 13  # $2F00 BNE taken 3 + $2F10 LDA/SEC/SBC 7 + STA 3
EDGE_WIDE_TEST = 8  # $2F17 LDA $41 3 + ORA $42 3 + BNE not taken 2
EDGE_WIDE = 12  # ... BNE taken 3 + $2EE0 JMP 3 + the $30BD LDA/SEC 6
EDGE_DX = 15  # $2F1D LDA/SEC/SBC 8 + BCS taken 3 + $2F2F STA $0D 3 + LDX 2, -1
EDGE_DX_NEG = 15  # ... BCS nt 2 + $2F24 EOR/CLC/ADC 6 + STA 3 + LDX 2 + BNE 3, -1
EDGE_SLOPE = 14  # $2F33 LDY $0D 3 + CPY $0C 3 + LDY $1A 3 + LDA $02 3 + the BCS 2
STEEP_SETUP = 34  # $2F3D..$2F55: three self-modifies 12, the DDA seed 16 and the JMP 6
STEEP_ROW = 23  # $2F58 ADC/BCC 6 + $2F5F STX/DEC/BEQ 12 + $2F67 DEY/BNE 5
STEEP_COLUMN = 1  # $2F58 BCC not taken 2 + SBC $0C 3 + INX/DEX 2, less the taken 3
STEEP_LAST = 4  # the last row leaves by $2F68 BNE nt 2 + $2F6A STY $7F 3 + RTS 6, -5
STEEP_BOTTOM = 5  # $2F65 BEQ taken 3 + $2F6D LDY/BEQ 5 + $2F6A 9, less the row tail 12
SHALLOW_SETUP = 55  # $2F71..$2F9E: the four self-modifies 27, the DDA seed 22 and 6
SHALLOW_STEP = 13  # $2FA1 INX/DEX 2 + ADC $0C 3 + BCC 3 + $2FB0 DEY/BNE 5
SHALLOW_STORE = 4  # $2FAD STX abs, on the columns the branch offset does not skip
SHALLOW_ROW = 8  # $2FA4 BCC nt 2 + SBC 3 + DEC $2FAE 6 + BEQ nt 2, less the taken 3
SHALLOW_LAST = 7  # the last column leaves by $2FB1 BNE nt 2 + $2FB3 JMP/STY/RTS 12, -7
SHALLOW_BOTTOM = 5  # $2FAB BEQ taken 3 + $2F6D LDY/BEQ 5 + $2F6A 9, less the tail 12
OUTSIDE_ENTRY = 20  # $2FB6/$2FDC INC 6 + LDX/STX abs 8 + LDX $18 3 + the JMP 3
OUTSIDE_STEEP_ROW = 8  # $2FCB DEC abs 6 + $2FCE BEQ not taken 2
OUTSIDE_STEEP_NEXT = 5  # $2FD0 DEY 2 + $2FD1 BNE taken 3
OUTSIDE_STEEP_ACC = 6  # $2FC4 ADC $0D 3 + $2FC6 BCC taken 3
OUTSIDE_STEEP_COLUMN = 4  # ... BCC nt 2 + $2FC8 SBC 3 + $2FCA INX/DEX 2, less the 3
OUTSIDE_STEEP_END = 13  # $2FD0 DEY 2 + BNE nt 2 + $2FD3 JMP 3 + $2F6C RTS 6
OUTSIDE_STEEP_BACK = 10  # $2FCE BEQ taken 3, less nt 2, + $2FD6 DEC 6 + $2FD9 JMP 3
STEEP_ENTER = 5  # $2FD9 lands on the storing loop's $2F67 DEY 2 + BNE taken 3
STEEP_ENTER_END = 13  # ... BNE not taken 2 instead + $2F6A STY $7F 3 + RTS 6
OUTSIDE_SHALLOW_STEP = 13  # $2FF6 DEY/BNE 5 + $2FEA INX/DEX 2 + ADC $0C 3 + BCC taken 3
OUTSIDE_SHALLOW_ROW = 10  # ... BCC nt 2 + SBC 3 + $2FF1 DEC 6 + BEQ nt 2, less the 3
OUTSIDE_SHALLOW_END = 13  # $2FF6 DEY 2 + BNE nt 2 + $2FF9 JMP 3 + $2F6C RTS 6
OUTSIDE_SHALLOW_BACK = 10  # $2FF4 BEQ taken 3, less nt 2, + $2FFC DEC 6 + JMP 3
SECTION_HEAD = 36  # $3002 STX/LDA/STA 8 + the 16-bit x delta 22 + $3019 JSR $1007 6
SECTION_INVERT = 24  # invert_A_and_a_fraction_if_negative $1007, either way
SECTION_TEST = 9  # $301C STA $75 3 + ORA $76 3 + $3020 BEQ skip_calculating 3
SECTION_HALVE = 26  # $3022 one halving lap: both 16-bit shifts, the ROL $40 and the BNE
SECTION_SCALE = 16  # $3030..$303A the two overflow tests, both BEQ not taken
SECTION_SEED = 51  # $303C..$305F the sign restore, the start-vertex copies and the BEQ
SECTION_STEP = 63  # $3061..$3083 walk one section's end back along the line + the JSR
SECTION_LOOP = 8  # $3086 DEC $0040 5 + BNE process_sections_loop 3
SECTION_END = 61  # $308A..$30B7 the final section's own vertex copies and the JSR
SECTION_TAIL = 3  # $30BA JMP consider_next_line
WIDE_HEAD = 14  # $30BF..$30C8 the 16-bit x delta 12 + the BPL 2, the $30BD LDA apart
WIDE_DIR = 16  # $30CA/$30DA the step direction and $30E0 STY/STA 6
WIDE_SLOPE = 14  # $30E4 LDY $0D 3 + CPY $0C 3 + LDY $1A 3 + LDA $02 3 + the BCS 2
WIDE_STEEP_SETUP = 55  # $30EE..$3110: the self-modifies, the DDA seed and both JMPs
WIDE_STEEP_ROW = 20  # $311F the store 4 + DEC $3120 6 + BEQ nt 2 + $3127 DEC/BNE 8
WIDE_STEEP_STEP = 6  # $3113 ADC $000D 3 + $3115 BCC set_row taken 3
WIDE_STEEP_COLUMN = 11  # ... nt 2 + SBC 3 + INX/DEX 2 + CPX $0076 3 + CLC/BEQ nt 4, -3
WIDE_STEEP_AREA_H = 15  # $311D BEQ taken 3 + $3140 INC/DEC $0041 5 + JSR 6 + JMP 3, -2
WIDE_STEEP_AREA_V = (
    14  # $3125 BEQ taken 3 + $312E DEC $3E 5 + BPL 3 + BNE 3, less the 0
)
WIDE_STEEP_AREA_BACK = 12  # $3135 BNE nt 2 + $3137 DEC 6 + JSR 6 + JMP 3, less the 5
WIDE_SHALLOW_SETUP = 55  # $3148..$316A, the same shape as the steep setup
WIDE_SHALLOW_STEP = (
    18  # $316D INX/DEX 2 + CPX/CLC/BEQ nt 7 + $317E store 4 + DEC/BNE 8, -3
)
WIDE_SHALLOW_ROW = (
    8  # $3175 BCC nt 2 + SBC 3 + DEC $317F 6 + BEQ nt 2, less the taken 3
)
WIDE_SHALLOW_AREA_H = 15  # $3171 BEQ taken 3 + $319A INC/DEC 5 + JSR 6 + JMP 3, -2
WIDE_SHALLOW_AREA_V = 14  # $317C BEQ taken 3 + $3188 DEC 5 + BPL 3 + BNE 3
WIDE_SHALLOW_AREA_BACK = 12  # $318F BNE nt 2 + DEC 6 + JSR 6 + JMP 3, less the 5
WIDE_AREA_CALL = 6  # $310D/$3167/$313A/$3142/$3194/$319C JSR $31A4
WIDE_AREA_OUT = 33  # $31A4 with $003E non-zero: nothing to plot in this area
WIDE_AREA = 40  # ... $0041 zero: the line's own x is the edge
WIDE_AREA_LEFT = 45  # ... $0041 negative: clip to column 0
WIDE_AREA_RIGHT = 43  # ... $0041 positive: clip to column $FF
WIDE_LEAVE = 9  # $312B/$3185 JMP leave 3 + $2F6C RTS 6
SPAN_CALL = 6  # $2AB4/$2AC8 JSR span_fill
SPAN_ENTRY = 25  # $22AA the two flags 7 + $22B0 the middle row 10 + $22B7 LDA/CMP 8
SPAN_BACKFACE = 9  # $22BD BCC leave taken 3 + $2310 RTS 6
SPAN_ADDRESS = 55  # $22BF..$22DE the buffer address for the polygon's first row
SPAN_COLOURS = 24  # $22E0..$2306 with $0C7A bit 7 clear, the $22E5 BPL taken
SPAN_SUPPRESSED = 45  # ... bit 7 set ($8538, a distant object): $22E7..$22F1 21 more
SPAN_START = 12  # $2308 LDY $06 3 + STY $1A 3 + CPY $04 3 + the BCS 3
SPAN_START_EMPTY = 8  # ... BCS not taken 2 + $2310 RTS 6, less the taken 3
SPAN_ROW_TEST = 12  # $2377 LDA $AE00,Y 4 + CMP $AD00,Y 4 + BCS nt 2 + $237F TAX 2
SPAN_ROW_EMPTY = 9  # $237D BCC leave taken 3 + $2310 RTS 6
SPAN_LEFT_EDGE = 12  # $2380 SBC $35 3 + BCS nt 2 + CMP $61 3 + BCC nt 2 + $2388 ASL 2
SPAN_OFF_LEFT = 3  # $2382 BCC set_polygon_extends_to_left taken
SPAN_OFF_RIGHT_ROW = 3  # $2384 BCS plot_row_clipped_to_right taken
SPAN_EXTENDS = 8  # $232B/$2331 LDA #0 2 + STA 3 + BEQ move_to_next_row 3
SPAN_RIGHT_PIXEL = 4  # $2389 AND #$F8 2 + $238B TAY 2
SPAN_PLOT_RIGHT = 26  # $238C..$239C the byte's pixel, the read-modify-write and the STY
SPAN_CLIP_RIGHT = 11  # $2337 LDA $61 3 + ASL 2 + STA $56 3 + STA $2C 3 + the BNE 3
SPAN_READ_LEFT = 12  # $239E LDY $1A 3 + LDA $AD00,Y 4 + TAX 2 + $23A4 CMP $36 3
SPAN_LEFT_TEST = 9  # $23A6 BCS nt 2 + SEC 2 + SBC $35 3 + BCC nt 2 + $23AD ASL 2, -2
SPAN_SAME_BYTE = 39  # $23B1 CPY/BCS 6 + $2355..$236F the two pixels merged 33
SPAN_PLOT_LEFT = 34  # $23B1 CPY/BCS 5 + $23B5..$23CF the pixel and the middle base 29
SPAN_CLIP_LEFT = 27  # $2340..$2353: the base two pixels left of the buffer and the BNE
SPAN_MIDDLE = 13  # $23D0..$23DA the unrolled-loop length and the always-taken BCC
SPAN_BYTE = 8  # one $23DC..$2456 LDY #imm 2 + STA ($70),Y 6: four filled pixels
SPAN_NEXT = 17  # $2458 JMP 3 + $2311 LDY/CPY/BEQ nt 8 + $2317 TYA/AND/BNE 6
SPAN_NEXT_GROUP = 21  # ... $231A BNE not taken 2 + $231C the buffer step 18 + the BNE 3
SPAN_LAST = 12  # $2311 LDY $1A 3 + CPY $04 3 + BEQ taken 3 + RTS 6, the $2458 JMP apart
SPAN_ROW_STEP = 10  # $2372 INC $0072 5 + $2374 DEY 2 + STY $1A 3

# plot_stack_of_objects $21AE, plot_object $8533 and its $8475 transform loop, block by block (sentinel/objectcost.py); the trig and the $0D03 multiplies are counted from their own data.
STACK_HEAD = 8  # $21AE AND #$3F 2 + STA $0C6C 4 + LDX #0 2
STACK_LEVEL = 15  # $21B5 one level of the height walk, its $21BE BCS taken
STACK_LAST = 14  # ... the bottommost level, BCS not taken
STACK_COUNT = 6  # $21C0 DEX 2 + STX $0C6B 4
STACK_ONE = 3  # $21C4 BEQ plot_objects_loop taken: one object in the tile
STACK_MANY = 2  # ... not taken
STACK_BELOW_TEST = (
    33  # $21C6..$21DB the 16-bit z compare with the player, the BMI apart
)
STACK_ABOVE = 3  # $21D7 BMI plot_objects_above_player taken
STACK_LEVEL_TYPE = (
    11  # $21DD LDA type 4 + CMP #6 2 + BEQ 3, an object level with the eye
)
STACK_BELOW = (
    21  # $21E4 JSR 6 + LDY 4 + DEC 6 + BMI nt 2 + LDX 4 + BPL 3, the JSR apart
)
STACK_RESCAN = 13  # $21F4 one lap: LDA 4 + AND 2 + TAY 2 + DEX 2 + BNE 3
STACK_ABOVE_HEAD = 4  # $21FF LDY $0C6C
STACK_PLOT = 17  # $2202 LDA 4 + AND 2 + TAY 2 + DEC 6 + BPL 3, the JSR apart
STACK_RTS = 6  # $2210 RTS

# plot_world's walk: the $2625 prologue, the $26DE row loop, the $27D7 row scan and the $295D plot loop, each block by its own branches (sentinel/projector.py).
WALK_SETUP = 172  # $2625..$26C7 looking north with a non-negative v_angle
WALK_SETUP_QUAD = (0, 5, 9, 3)  # $2684's four view cases, north/east/south/west
WALK_SETUP_PITCH = 1  # $263C BPL not taken 2 + $263E ORA #$F0 2, less the taken 3
WALK_ROWS = 17  # $26C9..$26D6: row $1F, the $0C48 hint into $0032 and $0005 = 0
WALK_HINT = 7  # $26D9 LDA $0032 3 + STA $0C48 4
ROW_HEAD = 25  # $26DE the $0005 flip 8 + $26E4 the $0037/$0038 copy 12 + $26EF DEC 5
ROW_LAST = 11  # $26F1 BMI leave_with_carry_clear taken 3 + $26FC CLC 2 + RTS 6
ROW_MORE = 8  # ... not taken 2 + $26F3 LDY $0026 3 + CPY $001D 3
ROW_NEAR = 3  # $26F7 BNE move_to_nearer_row taken
ROW_OBSERVER = 5  # ... not taken 2 + $26F9 JMP consider_plotting_observer_row 3
ROW_SCAN = 6  # $26D6/$26FE JSR find_visible_extent_of_row_of_tiles
ROW_START_SAME = 9  # $2701 LDY 3 + CPY $0037 3 + $2705 BEQ consider_end_of_row 3
ROW_START_AFTER = 4  # ... BEQ not taken 2 + $2707 BCC not taken 2
ROW_START_AFTER_TAIL = 3  # $2711 BEQ consider_end_of_row
ROW_START_BEFORE = 19  # $2707 BCC taken 3 + $2713..$271B the bank flip and INC $0026 16
ROW_START_BEFORE_TAIL = 16  # $2725 STY 3 + DEC $0026 5 + the bank flip back 8
ROW_EXTRA = (
    8  # one extra tile: DEY/INY 2 + CPY 3 + BNE taken 3, the JSR being the examine's
)
ROW_EXTRA_LAST = 7  # ... the BNE not taken
ROW_END_SAME = 9  # $272F LDY 3 + CPY $0038 3 + $2733 BEQ plot_row 3
ROW_END_BEFORE = 4  # ... BEQ not taken 2 + $2735 BCS not taken 2
ROW_END_BEFORE_TAIL = 3  # $273F BEQ plot_row
ROW_END_AFTER = 19  # $2735 BCS taken 3 + $2741..$2749 16
ROW_END_AFTER_TAIL = 16  # $2753 STY 3 + DEC $0026 5 + the bank flip back 8
ROW_PLOT = 16  # $275D JSR $295D 6 + BIT $0C1B 4 + BPL taken 3 + $276A JMP 3
OBS_ROW_HEAD = 8  # $276F LDY $0037 3 + INY 2 + CPY $0003 3
OBS_ROW_START = 9  # $2774 BNE nt 2 + $2776 STY $0038 3 + JMP plot_observer_row 3, +1
OBS_ROW_TEST = 13  # ... BNE taken 3 + $277B LDY 3 + DEY/DEY 4 + CPY $0003 3
OBS_ROW_END = 8  # $2781 BNE not taken 2 + $2783 INY 2 + STY $0037 3, +1
OBS_ROW_SKIP = 3  # $2781 BNE skip_plotting_observer_row taken
OBS_ROW_PLOT = (
    24  # $2786 the two examine calls' own LDYs 6 + JSR $295D 6, the JSRs apart
)
OBS_TILE = (
    19  # $2793 LDA/STA $0005 5 + INC $0026 5 + LDY $0003 3, the examine's JSR apart
)
OBS_TILE_TEST = 6  # $279E LDA $0AE0,Y 4 + CMP #$02 2
OBS_TILE_OFF = 11  # $27A3 BCS taken 3 + $27D1 CLC 2 + RTS 6
OBS_TILE_ON = (
    72  # ... not taken 2 + $27A5..$27CC forcing the tile's four corners 64 + 6
)
OBS_TILE_TAIL = 8  # $27D1 CLC 2 + RTS 6
SCAN_HEAD = 3  # $27D7 LDY $0032, the $27D9 JSR being the examine's own
SCAN_OFF = 3  # $27DC BEQ find_first_visible_tile_at_start_loop taken
SCAN_VISIBLE = 4  # ... not taken 2 + $27DE CMP #$80 2
SCAN_CROPPED = 3  # $27E0 BEQ tile_is_cropped_to_right taken
SCAN_WHOLE = 2  # ... not taken
SCAN_INC = 16  # $2836..$2840 increase_tile_column, falling into $2845
SCAN_INC_END = 24  # ... $2839 BEQ taken: the JSR 6 + the head 10 + $282C SEC/RTS 8
SCAN_DEC = 10  # $282E LDY 3 + BEQ nt 2 + DEY 2 + JMP $2845 3
SCAN_DEC_END = 20  # ... BEQ taken: the JSR 6 + LDY 3 + BEQ 3 + $282C 8
SCAN_END_HEAD = 6  # $27E2 LDA $0024 3 + STA $0032 3
SCAN_END_WHOLE = 7  # $27E9 BCS nt 2 + CMP #$81 2 + $27ED BEQ find_end_of_row_loop 3
SCAN_END_CROP = 9  # ... BEQ nt 2 + CMP #$80 2 + $27F1 BEQ reached_end_of_row 3, +2
SCAN_END_GAP = 8  # ... $27F1 BEQ not taken 2: fall into find_first_visible_tile_at_end
SCAN_END_STOP = 3  # $27E9 BCS reached_end_of_row taken
SCAN_GAP_LOOP = 5  # $27F6 BCS nt 2 + $27F8 BEQ find_first_visible_tile_at_end_loop 3
SCAN_GAP_HIT = 4  # ... BEQ not taken
SCAN_GAP_STOP = 3  # $27F6 BCS reached_end_of_row taken
SCAN_END_TAIL = 12  # $27FA LDA $0024 3 + STA $0033 3 + RTS 6
SCAN_CROP_HEAD = 6  # $27FF LDA $0024 3 + STA $0033 3
SCAN_CROP_MORE = 4  # $2806 BCS nt 2 + $2808 CMP #$80 2
SCAN_CROP_AGAIN = 3  # $280A BEQ tile_is_cropped_to_right taken
SCAN_CROP_EXIT = 9  # ... nt 2 + CMP #0 2 + JMP $2825 3 + the $2825 BEQ not taken 2
SCAN_CROP_LEFT = 10  # ... with the $2825 BEQ into the start-of-row loop taken 3
SCAN_CROP_STOP = 3  # $2806 BCS reached_start_of_row taken
SCAN_START_LOOP = (
    5  # $2814 BCS nt 2 + $2816 BEQ find_first_visible_tile_at_start_loop 3
)
SCAN_START_HIT = 4  # ... BEQ not taken
SCAN_START_STOP = 3  # $2814 BCS reached_end_of_row taken
SCAN_START_SETUP = 12  # $2818 the $0033 store 6 + the $0032 hint back into $0024 6
SCAN_LEFT_LOOP = (
    5  # $2823 BCS nt 2 + $2825 BEQ find_first_visible_tile_at_start_of_row 3
)
SCAN_LEFT_HIT = 4  # ... BEQ not taken
SCAN_LEFT_STOP = 3  # $2823 BCS reached_start_of_row taken
SCAN_LEFT_TAIL = 12  # $2827 LDA $0024 3 + STA $0032 3 + RTS 6
PLOT_ROW_HEAD = 6  # $295D LDA $0037 3 + STA $0025 3
PLOT_ROW_FRONT = 27  # $2961 one ascending lap, both compares, the JSR and the $296C INC
PLOT_ROW_TURN = 14  # $2961 CMP 3 + BCS nt 2 + CMP $0003 3 + BCS taken 3 + $2973 LDA 3
PLOT_ROW_DONE = 12  # $2961 CMP $0038 3 + BCS leave taken 3 + $298C RTS 6
PLOT_ROW_BACK = 31  # $2975 one descending lap, its three compares and the JSR
PLOT_ROW_BACK_END = (
    26  # ... $2980 CMP $0003 3 + BCC leave taken 3 + RTS 6, less the JSR 9
)
PLOT_ROW_BACK_LOW = 21  # ... $297C CMP $0037 3 + BCC leave taken 3 + RTS 6 instead
PLOT_ROW_BACK_EMPTY = 13  # $2975 SEC 2 + SBC 2 + $2978 BMI leave taken 3 + RTS 6

SC8_CALL = 6  # $848F JSR calculate_sine_and_cosine $0F70
SC8_POS = 3  # $0F70 BPL positive_sine taken
SC8_NEG = 4  # ... not taken 2 + $0F72 EOR #$40 2
SC8_BODY = 20  # $0F74..$0F80: the sign byte, the 9-bit shift and the table index
SC8_NOCLAMP = 3  # $0F81 BPL skip_ceiling taken
SC8_CLAMP = 4  # ... not taken 2 + $0F83 LDA #$7F 2
SC8_TABLE = 13  # $0F85 TAY 2 + the two $AC80 reads 8 + $0F8C BIT $0067 3
SC8_SAME = 16  # $0F8E BMI nt 2 + BVS nt 2 + $0F92 STA/STX/RTS 12
SC8_SWAP = 17  # ... $0F90 BVS taken 3 instead
SC8_NEG_SAME = 18  # $0F8E BMI taken 3 + $0F97 BVS taken 3 + $0F92 12
SC8_NEG_SWAP = 17  # ... $0F97 BVS not taken 2 + $0F99 12

OBJ_CALL = 6  # $21E4/$2202 JSR plot_object $8533
OBJ_RELATIVE = 6  # $8533 JSR calculate_object_relative_angles_and_distance $8401
OBJ_DISTANCE = 11  # $8536 LDA $7D 3 + CMP #$0F 2 + ROR $0C7A 6: distant => no edges
OBJ_TRANSFORM = 28  # $853D JSR 6 + $8475..$8483 the vertex bounds 22
OBJ_VERTEX_ANGLE = 15  # $8485..$848C the vertex's apparent angle
OBJ_VERTEX_COS = 13  # $8492..$8499 the radius and the cosine, the $0D03 JSR apart
OBJ_VERTEX_COS_SIGN = 8  # $849E STA 3 + LDA #0 2 + BIT $0067 3
OBJ_COS_POSITIVE = 3  # $84A4 BVC cosine_is_positive taken
OBJ_COS_NEGATIVE = 32  # ... not taken 2 + JSR $1009 invert_A_and_a_fraction 30
OBJ_VERTEX_Y = 25  # $84A9..$84B8 the radial component, the $84BA BPL apart
OBJ_Y_POSITIVE = 3  # $84BA BPL y_is_positive taken
OBJ_Y_NEGATIVE = 17  # ... not taken 2 + the $84BC 16-bit negate 15
OBJ_VERTEX_X = 33  # $84C7..$84DD the tangential component and the JSR $9287
OBJ_VERTEX_SCREEN_X = 35  # $84E0..$84F3 both 16-bit stores and the JSR $937F
OBJ_VERTEX_Z = 14  # $84F6..$84FE the doubled height, the $8500 BCC apart
OBJ_Z_POSITIVE = 3  # $8500 BCC z_is_positive taken
OBJ_Z_NEGATIVE = 32  # ... not taken 2 + JSR $1009 30
OBJ_VERTEX_VANGLE = 31  # $8505..$8516 the observer-relative z and the JSR $933D
OBJ_VERTEX_STORE = 19  # $8519..$8524 the two screen_y stores
OBJ_VERTEX_NEXT = 21  # $8525..$852F INC/INC/LDY/CPY 16 + BEQ nt 2 + JMP 3
OBJ_VERTEX_LAST = 25  # ... $852D BEQ taken 3 + $8532 RTS 6 instead of the JMP
OBJ_PASS_HEAD = 14  # $8540..$8548 $003B = $40, $0053 = 0 and $0C47 = 0
OBJ_SHAPE = 7  # $854B LDX 3 + LDA $9CB6,X 4
OBJ_CONVEX = 3  # $8550 BEQ plot_object_pass_loop taken: one pass covers the object
OBJ_CONCAVE = 4  # ... not taken 2 + $8552 LSR A 2
OBJ_MEANIE = 10  # $8553 BCS nt 2 + $8555 LDA $005A 3 + ADC #$C0 2 + JMP 3
OBJ_UPRIGHT = 7  # $8553 BCS taken 3 + $855C LDA $0C5C 4
OBJ_UPRIGHT_Z = 5  # $855F BNE plot_object_in_one_pass_if_negative taken 3 + EOR #$80 2
OBJ_UPRIGHT_LOW = 6  # ... not taken 2 + $8561 LDA $0C5B 4
OBJ_UPRIGHT_FLAT = 3  # $8564 BEQ plot_object_pass_loop taken
OBJ_UPRIGHT_HALVE = 6  # ... not taken 2 + $8566 LSR A 2 + $8567 EOR #$80 2
OBJ_ONE_PASS = 3  # $8569 BPL plot_object_pass_loop taken
OBJ_TWO_PASS = 7  # ... not taken 2 + $856B LDA #2 2 + STA $0053 3
OBJ_POLY_HEAD = 14  # $856F..$8576 the polygon bounds for this pass
OBJ_POLY_TEST = 13  # $8579 STY 3 + LDA $A1A0,Y 4 + LDX $0053 3 + $8580 BEQ 3
OBJ_POLY_PASS = 9  # ... BEQ not taken 2 + $8582 EOR $0C47 4 + $8585 BMI 3
OBJ_POLY_SETUP = 41  # $8587..$85A2 the colour, $0017, the vertex pointer and the JSR
OBJ_POLY_NEXT = 8  # $85A4 INY 2 + CPY $004F 3 + BCC plot_object_polygons_loop 3
OBJ_POLY_DONE = 12  # ... BCC not taken 2 + $85A9 DEC $0053 5 + BMI/BEQ 3, +2
OBJ_PASS_AGAIN = 11  # $85AB BEQ nt 2 + $85AF LDA #$80 2 + STA $0C47 4 + BMI 3
OBJ_TAIL = 25  # $85B6..$85C3 reset $003C, clear $0C7A bit 7, restore Y and RTS
