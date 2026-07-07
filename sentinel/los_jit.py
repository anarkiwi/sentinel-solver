"""Numba fast-march for the line-of-sight ray (:func:`sentinel.los.march`).

A bit-exact numba transcription of the HOT part of the ROM ray-march
(check_for_line_of_sight_to_tile $1CDD): the per-sub-step ``add_vector``, the
board-edge test, and the flat-tile / sloping-tile surface comparison, all over a
raw ``numpy.uint8`` view of the 64 KB memory image.

It reproduces exactly the pure-Python march in :mod:`sentinel.los` for every tile
EXCEPT a *primary object tile* (the ray's current tile byte >= $C0): that case
walks the recursive object stack (get_tile_z_from_object $1E3F), which stays in
tested Python.  On such a tile :func:`march` returns status ``OBJECT`` with the
full ray position so the caller can resolve it and resume the march.

Return status:
  * ``LOS_CLEAR`` (1) -- the ray reached a visible tile (ROM carry clear).
  * ``BLOCKED``   (0) -- tile above the ray / board edge / max_steps (no LOS).
  * ``OBJECT``    (2) -- current tile is an object tile; resolve in Python.

Every integer is kept in a Python-int-width (int64) register and masked with
``& 0xFF`` exactly where the 6502 truncates, so uint8 wrap never corrupts a byte.
"""

from numba import njit

LOS_CLEAR = 1
BLOCKED = 0
OBJECT = 2

# Object-array bases in the 64 KB image (sentinel.memmap), inlined so the njit
# code needs no Python object: OBJECTS_FLAGS $0100, OBJECTS_Z_HEIGHT $0940,
# OBJECTS_Z_FRACTION $0A00, OBJECTS_TYPE $0A40.
_OFLAGS = 0x0100
_OZHEIGHT = 0x0940
_OZFRAC = 0x0A00
_OTYPE = 0x0A40


@njit(cache=True)
def _tile_byte(mem, x, y):
    """terrain.tile_byte: the raw tiles_table byte via the ROM address arithmetic
    ``page=(x&3)+4``, ``lo=((x<<3)&0xE0)|(y&0x1F)``."""
    xx = x & 0xFF
    yy = y & 0xFF
    lo = ((xx << 3) & 0xE0) | (yy & 0x1F)
    page = (xx & 3) + 4
    return int(mem[page * 256 + lo])


@njit(cache=True)
def _corner_z(mem, x, y):
    """los._slope_corner_z: corner height at (x,y) -- bare-terrain nibble, or the
    bottommost stacked object's z_height for an object tile."""
    b = _tile_byte(mem, x, y)
    if b >= 0xC0:
        o = b & 0x3F
        for _ in range(64):
            f = int(mem[_OFLAGS + o])
            if f < 0x40:
                break
            o = f & 0x3F
        return int(mem[_OZHEIGHT + o])
    return (b >> 4) & 0x0F


@njit(cache=True)
def _edge(y):
    """The $1DF1-$1DF8 edge table."""
    if y == 0:
        return 0x00
    elif y == 1:
        return 0x03
    elif y == 2:
        return 0x01
    elif y == 3:
        return 0x00
    elif y == 4:
        return 0x01
    elif y == 5:
        return 0x02
    elif y == 6:
        return 0x02
    return 0x03  # y == 7


@njit(cache=True)
def _corner_at(i, p73, p74, p75, p76):
    """The zero-page corner square $73..$77 == [p73,p74,p75,p76,p73], indexed."""
    if i == 0:
        return p73
    elif i == 1:
        return p74
    elif i == 2:
        return p75
    elif i == 3:
        return p76
    return p73  # i == 4


@njit(cache=True)
def _slope_quad(nib, p73, p74, p75, p76, px_sub, py_sub, pz_sub, pz_whole):
    """los._slope_corner_or_quad $1D8A-$1DEE, opcode-faithful.  Returns 1 (blocked,
    ray below slope -> tile hit) or 0 (loop, ray above slope -> keep marching)."""
    A = nib & 0xFF
    C = A & 1
    A >>= 1
    if C == 0:
        C = A & 1
        A >>= 1
        s78 = A
        C = A & 1
        A >>= 1
        A = px_sub & 0xFF
        if C != 0:
            A ^= 0xFF
        C = 1 if A >= (py_sub & 0xFF) else 0
        A = ((s78 << 1) | C) & 0xFF
        A = _edge(A)
    else:
        C = A & 1
        A >>= 1
        if C == 1:
            A = (A + 1 + C) & 0xFF
            A &= 0x03
            s78 = A
            C = A & 1
            A >>= 1
            A = px_sub & 0xFF
            if C != 0:
                A ^= 0xFF
            C = 1 if A >= (py_sub & 0xFF) else 0
            A = ((s78 << 1) | C) & 0xFF
            A = _edge(A)
        else:
            A = A & 1

    # use_edge_for_slope $1DAF
    X = A & 0xFF
    C = A & 1
    A >>= 1
    yreg = px_sub & 0xFF
    if C == 0:
        yreg = py_sub & 0xFF
    C = A & 1
    A >>= 1
    A = yreg
    if C != 0:
        A ^= 0xFF
    s02 = A & 0xFF

    s78 = _corner_at(X & 3, p73, p74, p75, p76)
    a = (
        _corner_at((X & 3) + 1, p73, p74, p75, p76)
        - _corner_at(X & 3, p73, p74, p75, p76)
    ) & 0x1FF
    res = a & 0xFF
    neg = (res & 0x80) != 0
    if neg:
        res = ((res ^ 0xFF) + 1) & 0xFF
    s75 = res
    prod = (s02 & 0xFF) * (s75 & 0xFF)
    prod_h = (prod >> 8) & 0xFF
    prod_lo = prod & 0xFF
    if neg:
        val = ((prod_h & 0xFF) << 8) | (prod_lo & 0xFF)
        negv = (-val) & 0xFFFF
        prod_h = (negv >> 8) & 0xFF
        prod_lo = negv & 0xFF
    s75b = (prod_h + s78) & 0xFF
    lo = (pz_sub & 0xFF) - prod_lo
    borrow = 1 if lo < 0 else 0
    hi8 = ((pz_whole & 0xFF) - s75b - borrow) & 0xFF
    if hi8 & 0x80:
        return 1  # blocked
    return 0  # loop


@njit(cache=True)
def _check_slope(mem, x, y, z00, px_sub, py_sub, pz_sub, pz_whole):
    """los._check_sloping_tile $1D46.  Returns 1 (blocked) or 0 (loop)."""
    p73 = z00 & 0xFF
    p76 = _corner_z(mem, x + 1, y)
    p75 = _corner_z(mem, x + 1, y + 1)
    p74 = _corner_z(mem, x, y + 1)
    nib = _tile_byte(mem, x, y) & 0x0F
    if nib == 0x04 or nib == 0x0C:
        b = pz_whole & 0xFF
        if b >= p73 or b >= p74 or b >= p75 or b >= p76:
            return 0  # loop
        return 1  # blocked
    return _slope_quad(nib, p73, p74, p75, p76, px_sub, py_sub, pz_sub, pz_whole)


@njit(cache=True)
def _min_xy(px_sub, py_sub):
    """los._get_min_xy_fraction $1EAF: min tile-centre fraction of x/y (the exact
    6502 form, not plain abs)."""
    ax = (px_sub - 0x80) & 0xFF
    if ax & 0x80:
        ax ^= 0xFF
    t74 = ax & 0xFF
    ay = (py_sub - 0x80) & 0xFF
    if ay & 0x80:
        ay ^= 0xFF
    if ay >= t74:
        t74 = ay
    return t74 & 0xFF


@njit(cache=True)
def _is_tree_cdd(mem, Y, pz_sub, pz_whole, px_sub, py_sub, c56, cdd):
    """los._is_tree $1E69: the enemy-can-see-a-tree marker ($0CDD).  Works in a
    scratch byte, not $0079, so it only (maybe) sets $0CDD; returns the new cdd."""
    zf = int(mem[_OZFRAC + Y])
    t = zf - (pz_sub & 0xFF)
    s75 = t & 0xFF
    borrow = 1 if t < 0 else 0
    saved_hi = (int(mem[_OZHEIGHT + Y]) - (pz_whole & 0xFF) - borrow) & 0xFF
    t2 = s75 + 0xE0
    s75 = t2 & 0xFF
    carry = 1 if t2 > 0xFF else 0
    a = (saved_hi + carry) & 0xFF
    if a & 0x80:
        return cdd
    c = a & 1
    a >>= 1
    s75 = ((s75 >> 1) | (c << 7)) & 0xFF
    c = a & 1
    a >>= 1
    if a != 0:
        return cdd
    a = ((s75 >> 1) | (c << 7)) & 0xFF
    if a < _min_xy(px_sub, py_sub):
        return cdd
    if c56 & 0x80:
        return cdd
    return ((cdd >> 1) | 0x80) & 0xFF


@njit(cache=True)
def _object_surface(mem, raw0, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd):
    """los._get_tile_z_from_object $1E3F and its helpers, flattened into one bounded
    iterative walk of the object stack.  Returns the 7-tuple
    ``(z, s79, c0c, c67, c56, cdd, s60)`` -- the object surface the flat-tile check
    then compares against, plus the $0C56/$0CDD trackers (threaded in/out)."""
    s60 = 0x80
    s79 = 0
    c0c = 0x80
    c67 = 0
    raw = raw0 & 0xFF
    for _ in range(80):
        Y = raw & 0x3F
        do_ghol = False
        if (s60 & 0x80) == 0:
            # $1E44 BPL get_height_of_lowest_object
            do_ghol = True
        else:
            # get_tile_z_for_line_of_sight $1E0E
            if Y == (c58 & 0xFF):
                c56 = ((c56 >> 1) | 0x80) & 0xFF
            otype = int(mem[_OTYPE + Y])
            if otype == 3 or otype == 2:
                # get_boulder_or_tree_z_for_line_of_sight $1E48
                go_skip = False
                if _min_xy(px_sub, py_sub) >= 0x40:
                    go_skip = True
                elif otype == 2:  # is_tree $1E69
                    cdd = _is_tree_cdd(
                        mem, Y, pz_sub, pz_whole, px_sub, py_sub, c56, cdd
                    )
                    go_skip = True
                else:
                    # boulder near-centre $1E56: targetable, RTS with z
                    c67 = ((c67 >> 1) | 0x80) & 0xFF
                    t = int(mem[_OZFRAC + Y]) - 0x60
                    s79 = t & 0xFF
                    borrow = 1 if t < 0 else 0
                    z = (int(mem[_OZHEIGHT + Y]) - borrow) & 0xFF
                    return (z, s79, c0c, c67, c56, cdd, s60)
                if go_skip:
                    # skip_targeting_object $1E99, then fall into ghol
                    if int(mem[_OTYPE + Y]) != 2:
                        s60 = 0xC0
                    do_ghol = True
            elif otype != 6:
                # $1E23 BNE ghol (robot/sentry/enemy)
                do_ghol = True
            else:
                # platform (type 6) $1E25
                if _min_xy(px_sub, py_sub) >= 0x64:
                    if int(mem[_OTYPE + Y]) != 2:
                        s60 = 0xC0
                    do_ghol = True
                else:
                    c0c = 0x10
                    t = int(mem[_OZFRAC + Y]) + 0x20
                    s79 = t & 0xFF
                    carry = 1 if t > 0xFF else 0
                    z = (int(mem[_OZHEIGHT + Y]) + carry) & 0xFF
                    return (z, s79, c0c, c67, c56, cdd, s60)
        if do_ghol:
            # get_height_of_lowest_object $1EA4: stacked -> recurse on the object
            # beneath (raw = flags); else the bottom object's z_height.
            flags = int(mem[_OFLAGS + Y])
            if flags >= 0x40:
                raw = flags
                continue
            return (int(mem[_OZHEIGHT + Y]), s79, c0c, c67, c56, cdd, s60)
    # safety: corrupt/deep stack
    return (int(mem[_OZHEIGHT + (raw & 0x3F)]), s79, c0c, c67, c56, cdd, s60)


@njit(cache=True)
def march(
    mem,
    ax_lo,
    ax_hi,
    az_lo,
    az_hi,
    ay_lo,
    ay_hi,
    s30,
    px_frac,
    px_sub,
    px_whole,
    pz_frac,
    pz_sub,
    pz_whole,
    py_frac,
    py_sub,
    py_whole,
    ox,
    oy,
    c6e,
    c58,
    c56,
    cdd,
    max_steps,
):
    """March the ray from the given position for at most ``max_steps`` sub-steps.

    Fully self-contained: flat, sloping AND object tiles are resolved in numba, so
    the march never bails back to Python.  Returns the 15-tuple::

        (status, tx, ty,
         px_frac, px_sub, px_whole, pz_frac, pz_sub, pz_whole,
         py_frac, py_sub, py_whole, c56, cdd, steps_used)

    ``s30`` is the ray's vector_z high byte (the looking-up sign).  ``c6e`` is the
    do_line_of_sight_checks byte ($0C6E); bit7 waives the looking-up rejection.
    ``c58`` is the targeted-object slot; ``c56``/``cdd`` are the $0C56/$0CDD
    trackers (seeded/LSR'd by the caller, returned so the caller can persist them)."""
    ty = 0
    tx = 0
    steps = 0
    status = BLOCKED
    while steps < max_steps:
        steps += 1
        # add_vector $1CBB: signed 24-bit step on x, z, y (order irrelevant --
        # the axes are independent).
        t = (px_frac & 0xFF) + (ax_lo & 0xFF)
        px_frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        ext = -1 if (ax_hi & 0x80) else 0
        t2 = (px_sub & 0xFF) + (ax_hi & 0xFF) + carry
        px_sub = t2 & 0xFF
        px_whole = (px_whole + ext + (1 if t2 > 0xFF else 0)) & 0xFF

        t = (pz_frac & 0xFF) + (az_lo & 0xFF)
        pz_frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        ext = -1 if (az_hi & 0x80) else 0
        t2 = (pz_sub & 0xFF) + (az_hi & 0xFF) + carry
        pz_sub = t2 & 0xFF
        pz_whole = (pz_whole + ext + (1 if t2 > 0xFF else 0)) & 0xFF

        t = (py_frac & 0xFF) + (ay_lo & 0xFF)
        py_frac = t & 0xFF
        carry = 1 if t > 0xFF else 0
        ext = -1 if (ay_hi & 0x80) else 0
        t2 = (py_sub & 0xFF) + (ay_hi & 0xFF) + carry
        py_sub = t2 & 0xFF
        py_whole = (py_whole + ext + (1 if t2 > 0xFF else 0)) & 0xFF

        tx = px_whole & 0xFF
        if tx >= 0x1F:
            status = BLOCKED
            break
        ty = py_whole & 0xFF
        if ty >= 0x1F:
            status = BLOCKED
            break

        b = _tile_byte(mem, tx, ty)
        if b >= 0xC0:
            # object tile: resolve the stack surface ($1E3F) then the general
            # (object-aware) check_flat_tile $1D0D.
            z, s79, c0c, c67, c56, cdd, s60 = _object_surface(
                mem, b, px_sub, py_sub, pz_sub, pz_whole, c58, c56, cdd
            )
            t = (s79 & 0xFF) - (pz_sub & 0xFF)
            borrow = 1 if t < 0 else 0
            s79 = t & 0xFF
            d = ((z & 0xFF) - (pz_whole & 0xFF) - borrow) & 0xFF
            if d & 0x80:
                continue
            if d != 0:
                status = BLOCKED
                break
            if s79 >= (c0c & 0xFF):
                status = BLOCKED
                break
            if s60 & 0x40:
                status = BLOCKED
                break
            if ((c6e | c67) & 0x80) == 0:
                if (s30 & 0x80) == 0:  # looking up -> rejected
                    status = BLOCKED
                    break
            if tx == (ox & 0xFF) and ty == (oy & 0xFF):
                continue
            status = LOS_CLEAR
            break

        slope = b & 0x0F
        z = (b >> 4) & 0x0F
        if slope == 0:
            # check_flat_tile $1D0D, fast path (s79=0, tolerance $000C=$80,
            # $0060 bit6 clear, $0C67 clear).
            s79 = (0 - (pz_sub & 0xFF)) & 0xFF
            borrow = 1 if pz_sub & 0xFF else 0
            d = ((z & 0xFF) - (pz_whole & 0xFF) - borrow) & 0xFF
            if d & 0x80:
                continue  # tile below the ray -> keep marching
            if d != 0:
                status = BLOCKED
                break
            if s79 >= 0x80:
                status = BLOCKED
                break
            if (c6e & 0x80) == 0:
                if (s30 & 0x80) == 0:  # looking up -> rejected
                    status = BLOCKED
                    break
            if tx == (ox & 0xFF) and ty == (oy & 0xFF):
                continue  # same tile as the observer -> keep going
            status = LOS_CLEAR
            break
        else:
            if _check_slope(mem, tx, ty, z, px_sub, py_sub, pz_sub, pz_whole) == 0:
                continue  # ray above the slope -> keep marching
            status = BLOCKED
            break
    return (
        status,
        tx,
        ty,
        px_frac,
        px_sub,
        px_whole,
        pz_frac,
        pz_sub,
        pz_whole,
        py_frac,
        py_sub,
        py_whole,
        c56,
        cdd,
        steps,
    )
