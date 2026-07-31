"""Cycle cost of one play-loop pass ($1289), counted from the ROM.

The loop never counts frames: the raster IRQ pre-empts it and the foreground spends
what is left, so passes per frame = (FRAME_CYCLES - IRQ_CYCLES) / pass cost, and the
pass cost is a property of the state.  Per-term arithmetic: docs/architecture.md.
"""

from sentinel import memmap as mm

PAL_FRAME_CYCLES = 19656  # PAL 6569: 312 raster lines x 63 cycles
BADLINE_STEAL = 43  # 40 VIC c-accesses + the 3-cycle AEC lag: the MAXIMUM, not a mode
BADLINES_PER_FRAME = 25  # raster 51..243 step 8; $D015 = 0, so there is no sprite term
BADLINE_FRAME = 1071  # THE ONE FIT LEFT: sentinel.badline, open_items.md 8
SHORT_IRQ = 119  # $95E9 split chain at raster 53/93/133/173: 7 entry + 112 body
SHORT_IRQ_WRAP = 1  # the one entry a frame whose $9603 BPL wraps the split index to 4
SHORT_IRQS_PER_FRAME = 4  # the $9589 table 35 D5 AD 85 5D, less the $9593 full entry
SHORT_IRQ_FRAME = SHORT_IRQS_PER_FRAME * SHORT_IRQ + SHORT_IRQ_WRAP  # 477, exact live
IRQ_BODY = 2385  # 7 entry + $95E9 81 + $9630..$969A 2275 + the RTI tail 22: counted
IRQ_GATE_SHUT = 7  # $9659 LDA $0CE5 4 + $965C BMI $9669 taken 3: no clock, no $1635
IRQ_GATE_OPEN = 43  # ... nt 2 + $965E LDA/BMI 6 + JSR $130C 6 + the $1635 call 25
IRQ_CYCLES = BADLINE_FRAME + SHORT_IRQ_FRAME + IRQ_BODY  # 3933, the frame's fixed part
IRQ_SPRITES = 1490  # $1635 past its $0C04 exit; live it took the 25-cycle one, 500/500
FOREGROUND_CYCLES = PAL_FRAME_CYCLES - IRQ_CYCLES  # less the $9659 gate and the tick

COOLDOWN_TICK_NO_CARRY = 21  # $130C LDA/CLC/ADC/STA 12 + $1315 BCC taken 3 + RTS 6
COOLDOWN_TICK_GATE = 33  # + $1317 LDA 4 + BNE 3 + $1331 DEC 6 + RTS 6 - the taken BCC 1
COOLDOWN_TICK_WALK = 33  # the $131C walk's own entry 22 + $132B reload 6 + RTS 6 - 1
COOLDOWN_TICK_BYTE_STICK = 14  # $131E LDA 4 + CMP 2 + BCC taken 3 + DEX 2 + BPL 3
COOLDOWN_TICK_BYTE_DEC = 20  # + $1325 DEC 7, less the taken BCC 1
COOLDOWN_TICK_LAST = 1  # $0C20 leaves the walk by $1329 BPL not taken, one cycle short

# $8ED1 note tick, one lap a voice, from its own branches ($8E86 then $8E92).
SOUND_TICK_FIXED = 17  # $963D JSR 6 + the $FFC2 JMP 3 + $8ED1 LDX 2 + $8F0B RTS 6
SOUND_VOICE_READ = 4  # $8ED3 LDA $8E86,X
SOUND_VOICE_OFF = 6  # $8ED6 BEQ nt 2 + $8ED8 BMI $8F08 taken 4, page crossed
SOUND_VOICE_SPENT = 3  # $8ED6 BEQ $8EEE taken: this note's timer is already out
SOUND_VOICE_TICK = 11  # ... nt 2 + $8ED8 BMI nt 2 + $8EDA DEC $8E86,X 7
SOUND_VOICE_MORE = 4  # $8EDD BNE $8F08 taken 4, page crossed: the note plays on
SOUND_VOICE_NOTE = 24  # ... nt 2 + $8EDF..$8EEB: the $D404 write and the gate reload
SOUND_GATE_READ = 4  # $8EEE LDA $8E92,X
SOUND_GATE_OFF = 4  # $8EF1 BEQ $8F08 taken 4, page crossed
SOUND_GATE_TICK = 9  # ... nt 2 + $8EF3 DEC $8E92,X 7
SOUND_GATE_MORE = 4  # $8EF6 BNE $8F08 taken 4, page crossed
SOUND_GATE_END = 25  # ... nt 2 + $8EF8..$8F05: silence the voice, $8E96,X = $80
SOUND_VOICE_NEXT = 6  # $8F08 DEX 2 + $8F09 BPL $8ED3 taken 4, page crossed
SOUND_VOICE_LAST = 4  # ... not taken 2: voice 0 leaves the lap
SOUND_TICK_IDLE = 63  # all three voices idle -- what IRQ_BODY used to fold in

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
TUNE_ROTATE = 323  # $3470 tune 0 (and the meanie's 1): $FFF1 is JMP $8D81, in RAM
TUNE_DRAIN = 431  # $1A1D LDA #$05: the drain's tune walks a longer $AC28 descriptor

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
DISCHARGE_CREATE = 19  # ... BEQ nt 2 + $1A65 LDA #$02 2 + JSR $211D 6, less the BEQ 1
DISCHARGE_PLACE = 10  # $1A6A LDA $0C06 4 + $1A6D JSR $1238 6
DISCHARGE_ABANDON = 9  # $1A70 BCS $1A80 taken 3 + RTS 6: no tile took the tree
DISCHARGE_DONE = 48  # ... nt 2 + $1A72 TXA/JSR $1B00 8 + $1B00 15 + $1A76..$1A80 23
CREATE_HEAD = 6  # $211D STA $0C61 4 + $2120 LDX #$3F 2
CREATE_SLOT = 11  # per occupied slot: LDA 4 + BMI nt 2 + DEX 2 + BPL taken 3
CREATE_LAST = 1  # slot 0 leaves by $2128 BPL not taken
CREATE_HIT = 27  # LDA 4 + BMI taken 3 + $212C STX/LDA/STA 12 + CLC 2 + RTS 6
CREATE_NONE = 8  # $212A SEC 2 + RTS 6, every slot full
PLACE_HEAD = 8  # $1238 STA $06 3 + LDA #$00 2 + STA $15 3
PLACE_LAP = 73  # $123E DEC/BNE 8 + the two $1272 calls 18 + $2BA8 40 + the read 7
PLACE_WRAP = 11  # $1240 BNE nt 2 + $1242 INC $06 5 + LDA 3 + CMP 2, less the BNE 1
PLACE_GIVE_UP = 28  # the wrap that lifts the ceiling past 12: $1270 SEC/RTS included
PLACE_OCCUPIED = 3  # $125B BCS $123E taken: the tile already holds an object
PLACE_NOT_FLAT = 7  # ... nt 2 + AND #$0F 2 + $125F BNE $123E taken 3
PLACE_TOO_HIGH = 25  # ... BNE nt 2 + the z nibble 13 + $1269 BCS $123E taken 3
PLACE_HIT = 38  # ... BCS nt 2 + $126B JSR $1F16 6 + $126E CLC 2 + RTS 6
PUT_IN_TILE = 569  # $1F16 onto a bare flat tile, its $2BA8 and its facing draw included
DRAW = 445  # $1272 JSR $31CA 6 + prnd 427 + AND/CMP/BCS 6 + RTS 6
DRAW_REJECT = 440  # per draw masking to $1F: the $1279 BCS loops back

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
