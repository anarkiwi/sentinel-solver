"""What a cost term's own writes refund, when the term is too long to place.

A window ``d`` cycles into a term refunds the consecutive write cycles at ``d``, so
over the term that is its instruction sequence alone: 1 per one-cycle write, 3 per two
(an RMW's dummy-plus-real pair or a JSR's pushes), 0 otherwise.
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
}


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
