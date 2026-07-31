"""What a cost term's own writes refund, when the term is too long to place.

A window ``d`` cycles into a term refunds the consecutive write cycles at ``d``, so
over the term that is its instruction sequence alone: 1 per one-cycle write, 3 per two
(an RMW's dummy-plus-real pair or a JSR's pushes), 0 otherwise.  The $1CDD march, the
$8401 trig chain beneath it and the $95E9 raster-IRQ body each carry their own.
"""

import types

from sentinel import passcost

SHIFT = 32  # a term's weight rides above its cycles; both fit an int64 with room over
_HALF = 1 << (SHIFT - 1)


def _w(singles, pairs=0):
    """The weight of a run driving ``singles`` one-cycle writes and ``pairs`` two."""
    return singles + 3 * pairs


# Every write the $1CDD ray-march's terms drive; a term absent here drives none.
WEIGHT = {
    "ADD_VECTOR": _w(12, 1),  # $1CE8 JSR + per axis STA $74/$34,X/$37,X/$3A,X
    "ADD_VECTOR_NEG": _w(0, 1),  # $1CCC DEC $0074
    "STEP_EDGE": _w(1),  # $1CED STA $24 / $1CF5 STA $26
    "STEP_EDGE_EXIT": _w(0),  # the BCS taken instead: that axis' STA never ran
    "STEP_SETUP": _w(4),  # $1CFD/$1CFF/$1D03 STA zp + $1D05 STA $0C67
    "TILE_Z_CALL": _w(0, 1),  # JSR $1DF9
    "TILE_ADDR": _w(1, 1),  # $1DF9 JSR $2BA8 + $2BB9 STA $5F
    "TILE_Z_FLAT": _w(1),  # $1E02 PHA
    "FLAT_DIFF": _w(1),  # $1D13 STA $0079
    "SLOPE_HEAD": _w(6, 6),  # $1D46..$1D63: five STA, LSR/INC/INC/DEC/DEC, JSR + $2BB9
    "SLOPE_Q_CORNER": _w(1),  # $1D9D STA $0078
    "SLOPE_Q_TAIL": _w(6, 11),  # $1DBD..$1DDE, both JSRs, and $0D03's STA/LSR/8x ROR
    "SLOPE_Q_NEG": _w(2),  # $1009/$1010 STA zp
    "OBJ_TARGET_HIT": _w(0, 1),  # $1E13 ROR $0C56
    "MINXY": _w(2),  # $1EB8/$1EC9 STA $0074
    "OBJ_BT_BOULDER": _w(1, 1),  # $1E57 ROR $0C67 + $1E60 STA $0079
    "OBJ_PLAT_RTS": _w(2),  # $1E2E STA $000C + $1E36 STA $0079
    "OBJ_TREE_BELOW": _w(3),  # $1E6F/$1E7C STA $0075 + $1E76 PHA
    "OBJ_TREE_HIGH": _w(3, 1),  # ... + $1E84 ROR $0075
    "OBJ_TREE_NEAR": _w(3, 1),  # ... the $1E89 compare writes nothing
    "OBJ_TREE_TARGETED": _w(3, 1),  # ... nor the $1E90 BIT
    "OBJ_TREE_SEEN": _w(3, 2),  # ... + $1E96 ROR $0CDD
    "OBJ_SKIP_OTHER": _w(1),  # $1EA2 STA $0060
    "MARCH_ENTRY": _w(9, 3),  # $1CDF/$1CE2 LSR abs, $1CE5 JSR, $1ECC's nine stores
    # The $8401 chain beneath a $1887 query: $85C4/$85F5, $9287, $0D4A, $933D, $937F.
    "REL_XY": _w(6, 1),  # $85C4 JSR + $85C6/$85D3/$85DC/$85E0/$85E9/$85F2 STA zp
    "REL_Z": _w(2, 1),  # $85F5 JSR + $85FC/$8604 STA zp
    "REL_ANGLES": _w(10, 2),  # $8401/$8406/$841A/$8425/$842F/$8436 + $8460's four
    "ANG_MIN_Y": _w(4),  # $9297/$929B/$929F/$92A3 STA zp
    "ANG_MIN_X": _w(4),  # $92AA/$92AE/$92B2/$92B6 STA zp
    "ANG_ZERO": _w(3),  # $9280/$9282/$9284 STA zp
    "SCALE_SHIFT": _w(0, 1),  # $92C5/$9303 ASL zp
    "SCALE_LOOP": _w(0, 3),  # $92C1/$92C3 ROL zp and the next $92C5 ASL zp
    "SCALE_LOOP_Y": _w(0, 3),  # $92FF/$9301/$9303, the same three
    "SCALE_TAIL": _w(3, 2),  # $92CB ROR zp, three STA zp, the $0D4A JSR
    "ANG_SIGN_NEGATE": _w(2),  # $92E9/$92EF STA zp
    "ANG_QUAD_TAIL": _w(1),  # $92FC/$933A STA $8B
    "DIV_FULL_SHIFT": _w(0, 1),  # $0D4A/$0D68/$0D86 ASL-or-ROL $74
    "DIV_SUB": _w(2),  # $0D5B/$0D61 STA zp
    "DIV_SKIP_TEST": _w(1),  # $0DA4 PHP
    "DIV_SKIP": _w(1, 2),  # $0E12 STA $78 + $0E14/$0E17 ROR $74
    "DIV_BYTE_SHIFT": _w(0, 1),  # $0DA9.. ASL-or-ROL $74
    "DIV_LAST_SHIFT": _w(0, 1),  # $0DF1 ROR $78
    "DIV_LAST_TAIL": _w(0, 1),  # $0DF8 ROR $78
    "DIV_OVERFLOW": _w(3),  # $0E05/$0E09/$0E0D STA zp
    "DIV_ARCTAN": _w(3),  # $0E20/$0E25/$0E2A STA zp
    "DIV_DELTA": _w(1),  # $0E3B STA $74
    "DIV_DELTA_INVERT": _w(2, 1),  # $0E46 JSR + $1009/$1010 STA zp
    "DIV_DELTA_HALVE": _w(1, 2),  # $0E49 STA $75 + $0E4C/$0E4E ROR zp
    "DIV_NEXT": _w(2),  # $0E56/$0E5D STA zp
    "DIV_ADD_HALF": _w(2),  # $0E68/$0E6E STA zp
    "DIV_AVERAGE": _w(0, 2),  # $0E70 LSR $8B + $0E72 ROR $8A
    "VANG_HEAD": _w(1),  # $933D STA $86
    "VANG_NEG": _w(1),  # $9347 STA $80
    "VANG_SETUP": _w(4, 1),  # $934D/$9351/$9355/$9359 STA zp + the $9287 JSR
    "VANG_SHIFT": _w(2, 4),  # $9363 STA $50, $936A PHP, four $936C.. ROR $50
    "VANG_TAIL": _w(1),  # $937C STA $8D
    "HYP_HEAD": _w(4, 1),  # $937F STY abs, three STA zp, the $0F4A JSR
    "HYP_TAIL": _w(2, 2),  # $9398/$939A LSR/ROR zp + $93A1/$93A7 STA zp
}

# The $95E9 raster-IRQ body's own writes; the $8ED1 tick and $9659 gate are charged apart.
IRQ_BODY_WRITES = {
    "$95E9": _w(8),  # three PHA, $95F2 STA $D019, $9607/$960D/$9615/$961B
    "$9630": _w(0, 2),  # DEC $0CDF then, $9633 BPL falling through, INC $0CDF
    "$963A": _w(1, 1),  # JSR $FFC5 + $8F0E STA $8E85; three idle voices write none
    "$9678": _w(0, 1),  # JSR $119F
    "$119F": _w(5, 3),  # $11A1/$11D6/$11E2/$11EC/$121F + its three $0F62/$1363 JSRs
    "$1363": _w(4, 15),  # $1367 STA $0CE8,X four times + the $1370 JSR fifteen
    "$0F62": _w(17, 17),  # per scan call: $0F63 PHA and the $FFF4 JSR
    "$8CF9": _w(51),  # per scan: $8CFC/$8D01/$8D0D strobe the CIA1 keyboard matrix
}
WEIGHT["IRQ_BODY"] = sum(IRQ_BODY_WRITES.values())


def pack(name):
    """``passcost.<name>`` with the write weight of its own run packed above it."""
    return getattr(passcost, name) + (WEIGHT.get(name, 0) << SHIFT)


def weight(total):
    """The write weight of a packed cost total."""
    return (int(total) + _HALF) >> SHIFT


def cycles(total):
    """The 6502 cycles of a packed cost total."""
    return int(total) - (weight(total) << SHIFT)


def namespace(names):
    """A ``passcost``-shaped namespace of the named terms, each packed."""
    return types.SimpleNamespace(**{name: pack(name) for name in names})


MARCH_TERMS = (
    "ADD_VECTOR",
    "ADD_VECTOR_NEG",
    "STEP_EDGE",
    "STEP_EDGE_EXIT",
    "STEP_SETUP",
    "TILE_Z_CALL",
    "TILE_ADDR",
    "TILE_Z_READ",
    "TILE_Z_FLAT",
    "TILE_Z_OBJ",
    "LEAVE_SET",
    "FLAT_BRANCH",
    "FLAT_DIFF",
    "FLAT_BELOW",
    "FLAT_ABOVE",
    "FLAT_TOL",
    "FLAT_TOL_HIT",
    "FLAT_BIT60",
    "FLAT_BIT60_HIT",
    "FLAT_ANGLE",
    "FLAT_ANGLE_SKIP",
    "FLAT_LOOKUP",
    "FLAT_LOOKING_UP",
    "FLAT_SAME",
    "FLAT_SAME_X_DIFF",
    "FLAT_SAME_Y",
    "FLAT_SAME_HIT",
    "FLAT_SAME_Y_DIFF",
    "SLOPE_BRANCH",
    "SLOPE_HEAD",
    "SLOPE_NIB_4",
    "SLOPE_NIB_12",
    "SLOPE_NIB_QUAD",
    "SLOPE_EDGE_LDA",
    "SLOPE_EDGE_MISS",
    "SLOPE_EDGE_HIT",
    "SLOPE_EDGE_BLOCK",
    "SLOPE_Q_C2",
    "SLOPE_Q_C1",
    "SLOPE_Q_EDGE",
    "SLOPE_Q_CORNER",
    "SLOPE_Q_CORNER_EOR",
    "SLOPE_Q_TAIL",
    "SLOPE_Q_USE_Y",
    "SLOPE_Q_INVERT",
    "SLOPE_Q_ABS",
    "SLOPE_Q_NEG",
    "SLOPE_Q_BLOCK",
    "OBJ_HEAD_GHOL",
    "OBJ_HEAD_LOS",
    "OBJ_TARGET_HIT",
    "OBJ_TARGET_MISS",
    "OBJ_TYPE_BOULDER",
    "OBJ_TYPE_TREE",
    "OBJ_TYPE_OTHER",
    "OBJ_TYPE_PLATFORM",
    "MINXY",
    "MINXY_ABS",
    "MINXY_Y_WINS",
    "OBJ_BT_SKIP",
    "OBJ_BT_TYPE",
    "OBJ_BT_TREE",
    "OBJ_BT_BOULDER",
    "OBJ_PLAT_SKIP",
    "OBJ_PLAT_RTS",
    "OBJ_TREE_BELOW",
    "OBJ_TREE_HIGH",
    "OBJ_TREE_NEAR",
    "OBJ_TREE_TARGETED",
    "OBJ_TREE_SEEN",
    "OBJ_SKIP_TREE",
    "OBJ_SKIP_OTHER",
    "OBJ_GHOL_LOOP",
    "OBJ_GHOL_RTS",
    "MARCH_ENTRY",
)

MARCH = namespace(MARCH_TERMS)

# The $8401 bearing/distance chain a $1887 query runs before and between its marches.
TRIG_TERMS = (
    "REL_XY",
    "REL_XY_ABS",
    "REL_Z",
    "REL_ANGLES",
    "ANG_CMP",
    "ANG_X_LARGER",
    "ANG_Y_LARGER",
    "ANG_EQ",
    "ANG_EQ_Y",
    "ANG_EQ_X",
    "ANG_MIN_Y",
    "ANG_MIN_X",
    "ANG_ZERO",
    "ANG_NONZERO",
    "SCALE_SHIFT",
    "SCALE_LOOP",
    "SCALE_LOOP_Y",
    "SCALE_TAIL",
    "ANG_SIGN",
    "ANG_SIGN_KEEP",
    "ANG_SIGN_NEGATE",
    "ANG_QUAD",
    "ANG_QUAD_LOW",
    "ANG_QUAD_HIGH",
    "ANG_QUAD_TAIL",
    "DIV_FULL_SHIFT",
    "DIV_FULL_CARRY",
    "DIV_FULL_UNDER",
    "DIV_FULL_OVER",
    "DIV_FULL_EQ",
    "DIV_FULL_EQ_UNDER",
    "DIV_FULL_EQ_OVER",
    "DIV_SUB",
    "DIV_SKIP_TEST",
    "DIV_SKIP",
    "DIV_NO_SKIP",
    "DIV_BYTE_SHIFT",
    "DIV_BYTE_CARRY",
    "DIV_BYTE_UNDER",
    "DIV_BYTE_OVER",
    "DIV_LAST_SHIFT",
    "DIV_LAST_CARRY",
    "DIV_LAST_CMP",
    "DIV_LAST_TAIL",
    "DIV_PLP",
    "DIV_ROUND3_UNDER",
    "DIV_ROUND3_OVER",
    "DIV_NO_OVERFLOW",
    "DIV_OVERFLOW",
    "DIV_ARCTAN",
    "DIV_ARCTAN_DONE",
    "DIV_DELTA_BR",
    "DIV_HALF_BR",
    "DIV_DELTA",
    "DIV_DELTA_KEEP",
    "DIV_DELTA_INVERT",
    "DIV_DELTA_HALVE",
    "DIV_NEXT",
    "DIV_ADD_SKIP",
    "DIV_ADD_HALF",
    "DIV_AVERAGE",
    "DIV_TABLE_CROSS",
    "VANG_HEAD",
    "VANG_POS",
    "VANG_NEG",
    "VANG_SETUP",
    "VANG_SHIFT",
    "VANG_SIGN_POS",
    "VANG_SIGN_NEG",
    "VANG_TAIL",
    "HYP_HEAD",
    "HYP_TAIL",
)

TRIG = namespace(TRIG_TERMS)

# The flat sub-step: $1CE8 JSR $1CBB through the $1D18 BMI that closes the loop.
FLAT_LAP = (
    "ADD_VECTOR",
    "STEP_EDGE",
    "STEP_EDGE",
    "STEP_SETUP",
    "TILE_Z_CALL",
    "TILE_ADDR",
    "TILE_Z_READ",
    "TILE_Z_FLAT",
    "FLAT_BRANCH",
    "FLAT_DIFF",
    "FLAT_BELOW",
)
