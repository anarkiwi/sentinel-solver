"""Cycle cost of one play-loop pass ($1289), counted from the ROM.

The loop never counts frames: the raster IRQ pre-empts it and the foreground spends
what is left, so passes per frame = (FRAME_CYCLES - IRQ_CYCLES) / pass cost, and the
pass cost is a property of the state.  Per-term arithmetic: docs/architecture.md.
"""

LINE_CYCLES = 63  # PAL 6569 cycles per raster line
PAL_FRAME_CYCLES = 312 * LINE_CYCLES  # PAL 6569: 312 raster lines
SHORT_IRQ = 119  # $95E9 split chain at raster 53/93/133/173: 7 entry + 112 body
SHORT_IRQ_WRAP = 1  # the one entry a frame whose $9603 BPL wraps the split index to 4

SHORT_IRQS_PER_FRAME = 4  # the $9589 table 35 D5 AD 85 5D, less the $9593 full entry
SHORT_IRQ_FRAME = SHORT_IRQS_PER_FRAME * SHORT_IRQ + SHORT_IRQ_WRAP  # 477, exact live
IRQ_BODY = 2385  # 7 entry + $95E9 81 + $9630..$969A 2275 + the RTI tail 22: counted


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


# $1F9F update_object_on_screen: no screen span, no replot -- see relative.py $209B.
REDRAW_CALL = 6  # $1F9F JSR $209B
REDRAW_NONE = 23  # $1FA2 BCS $1F93 taken 3 + the $1F93 flag reset and RTS 20
REDRAW_PLOT_ENTRY = 2  # $1FA2 BCS not taken; $1FA4 on is projector.strip_replot_frames
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

# $283D picks the examine off $9AF6, the play-display flag: $3577 sets it $80 for the whole play loop and $117E (reset_game_state) clears it for generation and the preview; $9A51 is its only writer.
EXAM_ENTRY_TRIG = 7  # $283D BIT $9AF6 4 + $2840 BPL $2845 taken 3
EXAM_ENTRY_TABLE = 9  # ... BPL not taken 2 + $2842 JMP $37F2 3

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

# $37F2, the play path's examine: the same plottables read off the per-position projection tables $3700 builds at ($A8)/($AA)/($AC)/($AE), less the view's $001F/$00B2/$00B0/$00B1 camera reference. No trig, and no object-stack walk.
TAB_HEAD = 17  # $37F2..$37FB: the slot index and the $001C view-case BIT
TAB_QUAD_NORTH = 13  # $37FD BMI nt 2 + $37FF BVS nt 2 + $3801..$3805 9
TAB_QUAD_EAST = 20  # ... BVS taken 3 + $3808..$3810 15
TAB_QUAD_SOUTH = (
    25  # $37FD BMI taken 4, page crossed, + $3813 BVS nt 2 + $3815..$3820 19
)
TAB_QUAD_WEST = 17  # ... BVS taken 3 + $3823..$3828 10, falling into $382A
TAB_ADDR = 123  # $382A..$3874: the tile address and the four table pointers 55, the four reads less the reference 53, the $0180 store and the CMP #$C0 15
TAB_OBJECT = 3  # $3876 BCS taken: an object tile skips the raytrace bit
TAB_GROUND = 24  # ... not taken 2 + $3878..$3882 the $3E80/$24DA bit 22
TAB_VISIBLE = 3  # $3885 BNE taken: the raytrace bit is set
TAB_HIDDEN = 7  # ... not taken 2 + $3887 STA $0180,X 5: the plot byte is zeroed
TAB_SCREEN = 9  # $388A LDA #0 2 + LDY $A800,X 4 + CPY $0007 3
TAB_OFF_LEFT = 3  # $3891 BCC leave taken: beyond the left edge of the buffer
TAB_ON_LEFT = 2  # ... not taken
TAB_NO_FRACTION = 3  # $3893 BNE taken: the high byte alone settles the left edge
TAB_FRACTION = 9  # ... not taken 2 + $3895 LDY $0BA0,X 4 + CPY $0028 3
TAB_FRACTION_LEFT = 3  # $389A BCC leave taken
TAB_FRACTION_OK = 6  # ... not taken 2 + $389C LDY $A800,X 4
TAB_RIGHT = 5  # $389F ROR A 2 + CPY $0012 3
TAB_OFF_RIGHT = 3  # $38A2 BCC leave taken
TAB_ON_RIGHT = 4  # ... not taken 2 + $38A4 ADC #$00 2
TAB_TAIL = 16  # $38A6 LDY $0F 3 + STA $7F 3 + TAX 2 + CLC 2 + RTS 6


# plot_world's fill sequence block by block off the disassembly: the $2A24 tile dispatch, $2D6C prepare_polygon, $2EE4 the edge DDA and $22AA span_fill (sentinel/rendercost.py).
TILE_ENTRY = 25  # $2A15..$2A1F the slot arithmetic 19 + BMI nt 2 + $2A24 LDA $0180,X 4
TILE_SOUND = 35  # $2A12 JSR 6 + update_sound $352C's 29-cycle body
TILE_EMPTY = 10  # $2A27 BEQ taken 4, page crossed, + $29F8 RTS 6: the slot is zero
TILE_KEPT = 4  # ... BEQ not taken 2 + $2A29 CMP #$C0 2
TILE_OBJECT = 18  # $2A2B BCC nt 2 + PHA 3 + JSR 6 + PLA 4 + JMP $21AE 3
TILE_GROUND = 5  # $2A2B BCC taken 3 + $2A35 AND #$0F 2
TILE_FLAT = 3  # $2A37 BEQ plot_checkerboard_tile taken
TILE_EDGE_NIB = 25  # ... nt 2 + CMP #$0C 2 + BEQ 3 + $2A41..$2A49 the $0045 flag 18
TILE_EDGE_NIB_LOW = 3  # ... a $04 nibble instead: BEQ nt 2 + CMP #$04 2 + BNE nt 2, -3
TILE_SLOPE_CALC = 26  # ... both compares failing 8 + BNE taken 3 + $2A69..$2A72 15
TILE_SLOPE_QUAD = 3  # $2A74 BEQ plot_quadrilateral_tile taken
TILE_SLOPE_TRI = 29  # ... not taken 2 + $2A76..$2A86 the $2D39 flag and the colour 27
TILE_SLOPE_TRI_ALT = 1  # $2A86 BCS not taken 2 + $2A88 ORA #$10 2, less the taken 3
CHECKER_HEAD = 10  # $2A4B LDX #0 2 + $2A4D the tile-parity EOR 8
CHECKER_EVEN = 3  # $2A53 BEQ plot_quadrilateral_tile taken
CHECKER_ODD = 13  # ... not taken 2 + $2A55 LDX #8 2 + the $0C75 prnd copy 9
QUAD_TILE = 15  # $2A5D the colour, $003B = 0 and the JMP into plot_polygon
TWO_TRI_FIRST = 23  # $2A8A..$2A96: $0034, $003B = $80, the colour and the JSR
TWO_TRI_SECOND = 22  # $2A99..$2AA7: the other colour and $003B |= $40
PP_HEAD = 5  # $2AA9 LDY $0010 3 + CPY #$02 2
PP_WIDE = 8  # $2AAD BCS not taken 2 + $2AAF JSR prepare_polygon 6
PP_TALL = 9  # ... BCS taken 3 + $2AC3 JSR prepare_polygon 6
PP_NOTHING = 3  # $2AB2/$2AC6 BCS taken: prepare_polygon found nothing to fill
PP_SPAN = 2  # $2AB2/$2AC6 BCS not taken; the JSR is SPAN_CALL
PP_SECOND = 6  # $2AC3 JSR prepare_polygon, the second section's own
PP_CLIP_TEST = 9  # $2AB7 LDY 3 + LDA $2C,Y 4 + CMP #$01 2
PP_CLEAN = 3  # $2ABE BEQ taken: the section is clean, so no second pass
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
LINES_HEAD = 24  # $2DF2..$2E02: the row floor/ceiling, $001E and $007F
LINE_HEAD = 45  # $2E04..$2E20: both vertices, the 16-bit y delta and the BPL
LINE_DOWN = 3  # $2E20 BPL skip_inversion taken: the line already slopes downwards
LINE_UP = 37  # ... not taken 2 + $2E22..$2E37 swap the ends and negate the delta 35
LINE_MID = 6  # $2E39 STA $0076 3 + $2E3B BIT $006C 3
LINE_INNER = 3  # $2E3D BVC line_is_entirely_within_inner_area taken
LINE_OUTER = 12  # ... not taken 2 + $2E3F the two screen_x high bytes 8 + BEQ nt 2
LINE_OUTER_INNER = 13  # ... with the $2E45 BEQ taken instead
LINE_OUTER_STEEP = 9  # $2E47 LDA 3 + BNE taken 3 + $2E4F JMP process_line 3
LINE_OUTER_FLAT = 13  # ... BNE nt 2 + $2E4B LDA 3 + BEQ nt 2 + $2E4F JMP 3
LINE_OUTER_POINT = 11  # ... with the $2E4D BEQ consider_next_line taken 3
LINE_STEEP = 20  # $2E52 LDA 3 + BEQ nt 2 + $2E56 zero both screen_x highs 12 + JMP 3
LINE_SHALLOW = 6  # $2E52 LDA 3 + $2E54 BEQ process_horizontal_line taken 3
LINE_IS_POINT = 6  # $2E61 LDA $000C 3 + $2E63 BEQ line_is_a_point taken 3
LINE_RASTERISE = 55  # ... BEQ nt 2 + $2E65..$2E87 the eight copies 50; JSR: EDGE_CALL
POINT_LINE = 24  # $2ECC..$2EDD: the two clamps, INC $001E and the JMP, both taken
POINT_LINE_FLOOR = 2  # $2ED1 BCC not taken 2 + $2ED3 STA $0031 3, less the taken 3
POINT_LINE_CEIL = 2  # $2ED7 BCS not taken 2 + $2ED9 STA $0030 3, less the taken 3
LINE_NEXT = 13  # $2E8C LDY 3 + INY 2 + CPY $17 3 + BEQ nt 2 + $2E93 JMP 3
LINE_LAST = 11  # ... $2E91 BEQ check_if_polygon_is_point taken 3
LINES_TAIL = 6  # $2E96 LDA $001E 3 + CMP $0017 3
LINES_NOT_POINT = 3  # $2E9A BNE polygon_isn't_point taken
LINES_ALL_POINTS = 15  # ... nt 2 + $2E9C LDA $0AE0,X 4 + BNE nt 2 + LDY 4 + $2EA4 CPY 3
LINES_POINT_OFF = 3  # $2E9F/$2EA6/$2EAA leaving for polygon_isn't_point
LINES_POINT_ROW = 34  # $2EA6 BCC nt/CPY/BCS nt 7 + $2EAC..$2EBC the single row 27
LINES_DONE = 15  # $2EBE LDA 3 + BNE nt 2 + $2EC2 LDA/CMP 6 + BCC nt 2 + $2EC8 CLC 2
LINES_NOTHING = 8  # leaving by $2EC0/$2EC6 for $2ECA SEC 2 + RTS 6, the tests apart
LINES_RTS = 6  # $2EC9 RTS
EDGE_CALL = 6  # $2E89/$3083/$30B7 JSR rasterise_polygon_edge
EDGE_HEAD = 3  # $2EE4 LDA $003F
EDGE_ABOVE = 11  # $2EE6 BMI nt 2 + $2EE8 BNE leave taken 3 + $2EE3 RTS 6
EDGE_BOTTOM_TEST = 10  # $2EE6 BMI nt 2 + BNE nt 2 + $2EEA LDA $16 3 + CMP $51 3
EDGE_BOTTOM_OFF = 9  # $2EEE BCS leave taken 3 + $2EE3 RTS 6
EDGE_BOTTOM_KEEP = 8  # $2EEE BCS nt 2 + $2EF0 CMP $04 3 + BCS not_new_bottom taken 3
EDGE_BOTTOM_SET = 16  # ... $2EF2 BCS nt 2 + $2EF4 CMP $52 3 + BCS taken 3 + STA $04 3
EDGE_BOTTOM_CLIP = 18  # ... $2EF6 BCS nt 2 + $2EF8 LDA $52 3 + $2EFA STA $04 3
EDGE_BOTTOM_LOW = 9  # $2EE6 BMI taken 3 + $2EF8 LDA $52 3 + $2EFA STA $04 3
EDGE_TOP_TEST = 13  # $2EFC LDA $3E 3 + BMI nt 2 + BNE nt 2 + $2F02 LDA $1A 3 + CMP 3
EDGE_TOP_OFF = 13  # $2EFC LDA 3 + $2EFE BMI leave taken 4, page crossed, + RTS 6
EDGE_TOP_BELOW = 10  # $2F06 BCC leave taken 4, page crossed, + $2EE3 RTS 6
EDGE_TOP_KEEP = 8  # $2F06 BCC nt 2 + $2F08 CMP $06 3 + BCC not_new_top_row taken 3
EDGE_TOP_SET = 16  # ... $2F0A BCC nt 2 + $2F0C CMP $51 3 + BCC taken 3 + STA $06 3
EDGE_TOP_CLIP = 22  # ... $2F0E BCC nt 2 + $2F10 LDA/SEC/SBC 7 + $2F15 STA $06 3
EDGE_TOP_HIGH = 18  # $2EFC LDA 3 + BMI nt 2 + $2F00 BNE taken 3 + $2F10 7 + STA 3
EDGE_WIDE_TEST = 8  # $2F17 LDA $41 3 + ORA $42 3 + BNE not taken 2
EDGE_WIDE = 7  # ... BNE taken 4, page crossed, + $2EE0 JMP process_wide_line 3
EDGE_DX = 16  # $2F1D LDA/SEC/SBC 8 + BCS taken 3 + $2F2F STA $0D 3 + LDX #$CA 2
EDGE_DX_NEG = 24  # ... BCS nt 2 + $2F24 EOR/CLC/ADC 6 + STA 3 + LDX #$E8 2 + BNE 3
EDGE_SLOPE = 12  # $2F33 LDY $0D 3 + CPY $0C 3 + LDY $1A 3 + LDA $02 3
EDGE_INNER_ENTRY = 8  # $2F51/$2F9A BNE nt 2 + LDX $18 3 + the JMP into the store 3
STEEP_SETUP = 31  # $2F3B BCS nt 2 + $2F3D..$2F4F three self-modifies 12, the seed 17
STEEP_ROW = 17  # $2F5F STX 4 + DEC abs 6 + BEQ nt 2 + $2F67 DEY 2 + BNE taken 3
STEEP_ACC = 6  # $2F58 ADC $000D 3 + $2F5A BCC set_row taken 3
STEEP_COLUMN = 4  # $2F5A BCC not taken 2 + SBC $0C 3 + INX/DEX 2, less the taken 3
STEEP_LAST = 8  # the last row leaves by $2F68 BNE nt 2 + $2F6A STY/RTS 9, less the 3
STEEP_BOTTOM = 10  # $2F65 BEQ taken 3 + $2F6D LDY/BEQ 5 + $2F6A 9, less the tail 7
SHALLOW_SETUP = 48  # $2F3B BCS taken 3 + $2F71..$2F98 the self-modifies and the seed
SHALLOW_SETUP_ALT = 1  # ... the longer arm of the $2F7C/$2F80 store-op tree
SHALLOW_STEP = 13  # $2FA1 INX/DEX 2 + ADC $0C 3 + BCC 3 + $2FB0 DEY/BNE 5
SHALLOW_STORE = 4  # $2FAD STX abs, on the columns the branch offset does not skip
SHALLOW_ROW = 10  # $2FA4 BCC nt 2 + SBC 3 + DEC $2FAE 6 + BEQ nt 2, less the taken 3
SHALLOW_LAST = 16  # $2FB0 DEY 2 + BNE nt 2 + $2FB3 JMP 3 + $2F6A STY $7F 3 + RTS 6
SHALLOW_BOTTOM = 10  # $2FAB BEQ taken 3 + $2F6D LDY/BEQ 5 + $2F6A 9, less the tail 7
OUTSIDE_ENTRY = 23  # $2F51/$2F9A BNE taken 3 + $2FB6/$2FDC INC 6 + LDX/STX 8 + 6
OUTSIDE_STEEP_ROW = 8  # $2FCB DEC abs 6 + $2FCE BEQ not taken 2
OUTSIDE_STEEP_NEXT = 5  # $2FD0 DEY 2 + $2FD1 BNE taken 3
OUTSIDE_STEEP_ACC = 6  # $2FC4 ADC $0D 3 + $2FC6 BCC taken 3
OUTSIDE_STEEP_COLUMN = 4  # ... BCC nt 2 + $2FC8 SBC 3 + $2FCA INX/DEX 2, less the 3
OUTSIDE_STEEP_END = 13  # $2FD0 DEY 2 + BNE nt 2 + $2FD3 JMP 3 + $2F6C RTS 6
OUTSIDE_STEEP_BACK = 10  # $2FCE BEQ taken 3, less nt 2, + $2FD6 DEC 6 + $2FD9 JMP 3
STEEP_ENTER = 5  # $2FD9 lands on the storing loop's $2F67 DEY 2 + BNE taken 3
STEEP_ENTER_END = 11  # ... BNE not taken 2 instead + $2F6A STY $7F 3 + RTS 6
OUTSIDE_SHALLOW_STEP = 13  # $2FF6 DEY/BNE 5 + $2FEA INX/DEX 2 + ADC $0C 3 + BCC taken 3
OUTSIDE_SHALLOW_ROW = 10  # ... BCC nt 2 + SBC 3 + $2FF1 DEC 6 + BEQ nt 2, less the 3
OUTSIDE_SHALLOW_END = 13  # $2FF6 DEY 2 + BNE nt 2 + $2FF9 JMP 3 + $2F6C RTS 6
OUTSIDE_SHALLOW_BACK = 10  # $2FF4 BEQ taken 3, less nt 2, + $2FFC DEC 6 + JMP 3
SECTION_HEAD = 38  # $3002 STX/LDA/STA 8 + the 16-bit x delta 24 + $3019 JSR $1007 6
SECTION_INVERT_POS = 9  # invert_A_and_a_fraction_if_negative $1007: BPL taken 3 + RTS 6
SECTION_INVERT_NEG = 26  # ... BPL nt 2 + $1009..$1014 the 16-bit negate 18 + RTS 6
SECTION_TEST = 8  # $301C STA $75 3 + ORA $76 3 + $3020 BEQ nt 2
SECTION_ZERO = 1  # ... BEQ skip_calculating taken 3 instead
SECTION_HALVE = (
    31  # $3022 one halving lap: both 16-bit shifts 20, SEC/ROL/LSR 9, BNE nt 2
)
SECTION_HALVE_AGAIN = 1  # ... its $302E BNE taken 3 when another lap follows
SECTION_SCALE = 14  # $3030..$303A the two overflow tests, both BEQ not taken
SECTION_GUARD_LO = 8  # ... $0C is $FF: LDX/CPX 5 + BEQ taken 3
SECTION_GUARD_HI = 15  # ... $74 is: the first test 7 + LDX/CPX 5 + BEQ taken 3
SECTION_SEED = 54  # $303C..$305F the sign restore, the start-vertex copies, the BEQ nt
SECTION_ONE = 1  # ... $305F BEQ the_last_section taken 3 instead: no section to walk
SECTION_STEP = 57  # $3061..$3081 walk one section's end back along the line 51 + JSR 6
SECTION_LOOP = 7  # $3086 DEC $0040 5 + $3088 BNE not taken 2
SECTION_LOOP_AGAIN = 1  # ... BNE process_sections_loop taken 3 instead
SECTION_END = 72  # $308A..$30B5 the final section's own vertex copies 66 + the JSR 6
SECTION_TAIL = 3  # $30BA JMP consider_next_line
WIDE_HEAD = 17  # $30BD LDA $18 3 + $30BF..$30C6 the 16-bit x delta 14
WIDE_DIR = 15  # $30C8 BPL taken 3 + $30DA LDX/LDA/LDY 6 + $30E0 STY/STA 6
WIDE_DIR_NEG = 27  # ... BPL nt 2 + $30CA the negate and the other step ops 19 + 6
WIDE_SLOPE = 12  # $30E4 LDY $0D 3 + CPY $0C 3 + LDY $1A 3 + LDA $02 3
WIDE_STEEP_SETUP = 49  # $30EC BCS nt 2 + $30EE..$3110 the self-modifies and the seed 47
WIDE_AREA_INY = (
    1  # $30F6/$3150 BEQ nt 2 + INY 2, less the taken 3: a line above the area
)
WIDE_STEEP_ROW = 12  # $311F the store 4 + $3122 DEC $3120 6 + $3125 BEQ not taken 2
WIDE_SHALLOW_STORE = 4  # $317E the shallow loop's own store
WIDE_TAIL = 8  # $3127/$3181 DEC $0075 5 + $3129/$3183 BNE taken 3
WIDE_STEEP_STEP = 6  # $3113 ADC $000D 3 + $3115 BCC set_row taken 3
WIDE_STEEP_COLUMN = 11  # ... nt 2 + SBC 3 + INX/DEX 2 + CPX $0076 3 + CLC/BEQ nt 4, -3
WIDE_STEEP_AREA_H = 9  # $311D BEQ taken 3, less nt 2, + $3140 INC/DEC $0041 5 + JMP 3
WIDE_STEEP_AREA_V = 6  # $3125 BEQ taken 3, less nt 2, + $312E DEC $003E 5
WIDE_AREA_ON = (
    6  # ... $3130 BPL taken 3 + $3135/$318F BNE taken 3: the area still has rows
)
WIDE_STEEP_AREA_BACK = (
    14  # $3130 BPL taken 3 + $3135 BNE nt 2 + $3137 DEC abs 6 + JMP 3
)
WIDE_SHALLOW_SETUP = 51  # $30EC BCS taken 4, page crossed, + $3148..$316A's own 47
WIDE_SHALLOW_STEP = 15  # $316D INX/DEX 2 + CPX/CLC/BEQ nt 7 + $3173 ADC 3 + BCC taken 3
WIDE_SHALLOW_ROW = (
    10  # $3175 BCC nt 2 + SBC 3 + DEC $317F 6 + BEQ nt 2, less the taken 3
)
WIDE_SHALLOW_AREA_H = 9  # $3171 BEQ taken 3, less nt 2, + $319A INC/DEC 5 + JMP 3
WIDE_SHALLOW_AREA_V = 6  # $317C BEQ taken 3, less nt 2, + $3188 DEC $003E 5
WIDE_SHALLOW_AREA_BACK = 14  # $318A BPL taken 3 + $318F BNE nt 2 + $3191 DEC 6 + JMP 3
WIDE_AREA_CALL = 6  # $310D/$3167/$313A/$3142/$3194/$319C JSR $31A4
WIDE_AREA_OUT = 31  # $31A4 with $003E non-zero: nothing to plot in this area
WIDE_AREA = 40  # ... $0041 zero: the line's own x is the edge
WIDE_AREA_LEFT = 45  # ... $0041 negative: clip to column 0
WIDE_AREA_RIGHT = 43  # ... $0041 positive: clip to column $FF
WIDE_LEAVE = 16  # $3127/$3181 DEC 5 + BNE nt 2 + $312B/$3185 JMP 3 + $2F6C RTS 6
WIDE_LEAVE_AREA = 11  # $3130/$318A BPL nt 2 + $3132/$318C JMP 3 + $2F6C RTS 6
SPAN_CALL = 6  # $2AB4/$2AC8 JSR span_fill
SPAN_ENTRY = 28  # $22AA the two flags 8 + $22B0 the middle row 12 + $22B7 LDA/CMP 8
SPAN_BACKFACE = 10  # $22BD BCC leave taken 4, page crossed, + $2310 RTS 6
SPAN_ADDRESS = 50  # $22BD BCC nt 2 + $22BF..$22DE the first row's buffer address 48
SPAN_COLOURS = 39  # $22E0..$2306 with $0C7A bit 7 clear, the $22E5 BPL taken
SPAN_SUPPRESSED = 55  # ... bit 7 set ($8538, a distant object): $22E7..$22F1 17 more
SPAN_START = 12  # $2308 LDY $06 3 + STY $1A 3 + CPY $04 3 + the BCS 3
SPAN_START_EMPTY = 5  # ... BCS not taken 2 + $2310 RTS 6, less the taken 3
SPAN_ROW_TEST = 12  # $2377 LDA $AE00,Y 4 + CMP $AD00,Y 4 + BCS nt 2 + $237F TAX 2
SPAN_ROW_EMPTY = 5  # $237D BCC leave taken 3 + $2310 RTS 6, less the row test's tail 4
SPAN_LEFT_EDGE = 12  # $2380 SBC $35 3 + BCS nt 2 + CMP $61 3 + BCC nt 2 + $2388 ASL 2
SPAN_OFF_LEFT = 6  # $2380 SBC $35 3 + $2382 BCC set_polygon_extends_to_left taken 3
SPAN_OFF_RIGHT_ROW = 11  # $2380 SBC/BCC nt 5 + CMP $61 3 + BCS clipped_to_right taken 3
SPAN_EXTENDS = 8  # $232B/$2331 LDA #0 2 + STA 3 + BEQ move_to_next_row 3
SPAN_RIGHT_PIXEL = 4  # $2389 AND #$F8 2 + $238B TAY 2
SPAN_PLOT_RIGHT = 31  # $238C..$239C the byte's pixel, the read-modify-write and the STY
SPAN_CLIP_RIGHT = 14  # $2337 LDA $61 3 + ASL 2 + STA $56 3 + STA $2C 3 + the BNE 3
SPAN_READ_LEFT = 12  # $239E LDY $1A 3 + LDA $AD00,Y 4 + TAX 2 + $23A4 CMP $36 3
SPAN_LEFT_OFF = 3  # $23A6 BCS set_polygon_extends_to_right taken
SPAN_LEFT_TEST = (
    18  # $23A6 BCS nt 2 + SEC/SBC 5 + BCC nt 2 + $23AD ASL/AND/TAY 6 + CPY 3
)
SPAN_SAME_BYTE = 48  # $23B3 BCS taken 3 + $2355..$236F the two pixels merged 42 + JMP 3
SPAN_PLOT_LEFT = 47  # $23B3 BCS nt 2 + $23B5..$23CF the pixel and the middle base 45
SPAN_CLIP_LEFT = 38  # $23A6 BCS nt/SEC/SBC/BCC taken 10 + $2340..$2353 the clipped base
SPAN_MIDDLE = 22  # $23D0..$23D9 16 + the BCC into the run 3 + $2458 JMP move_to_next 3
SPAN_MIDDLE_FAR = 1  # ... the $23DA BCC crosses into $24xx for fewer than 23 bytes
SPAN_BYTE = 8  # one $23DC..$2456 LDY #imm 2 + STA ($70),Y 6: four filled pixels
SPAN_NEXT = 14  # $2311 LDY/CPY/BEQ nt 8 + $2317 TYA/AND/BNE nt 6; the JMP is the row's
SPAN_NEXT_GROUP = 26  # ... $231C the buffer step 18 + BNE taken 3 + $2374 DEY/STY 5
SPAN_LAST = 15  # $2311 LDY $1A 3 + CPY $04 3 + BEQ taken 3 + $2310 RTS 6
SPAN_ROW_STEP = 11  # $231A BNE taken 3, less nt 2, + $2372 INC 5 + $2374 DEY/STY 5

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
SCAN_INC = 9  # $2836..$283B increase_tile_column, falling into $283D's own entry
SCAN_INC_END = 24  # ... $2839 BEQ taken: the JSR 6 + the head 10 + $282C SEC/RTS 8
SCAN_DEC = 10  # $282E LDY 3 + BEQ nt 2 + DEY 2 + JMP $283D 3
SCAN_DEC_END = 20  # ... BEQ taken: the JSR 6 + LDY 3 + BEQ 3 + $282C 8
SCAN_END_HEAD = 6  # $27E2 LDA $0024 3 + STA $0032 3
SCAN_END_WHOLE = 7  # $27E9 BCS nt 2 + CMP #$81 2 + $27ED BEQ find_end_of_row_loop 3
SCAN_END_CROP = 9  # ... BEQ nt 2 + CMP #$80 2 + $27F1 BEQ reached_end_of_row 3, +2
SCAN_END_GAP = 8  # ... $27F1 BEQ not taken 2: fall into find_first_visible_tile_at_end
SCAN_END_STOP = 3  # $27E9 BCS reached_end_of_row taken
SCAN_GAP_LOOP = 5  # $27F6 BCS nt 2 + $27F8 BEQ find_first_visible_tile_at_end_loop 3
SCAN_GAP_HIT = 4  # ... BEQ not taken
SCAN_GAP_STOP = 3  # $27F6 BCS reached_end_of_row taken
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
