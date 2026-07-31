"""plot_stack_of_objects $21AE and plot_object $8533, emulated and cycle-counted.

`$8475` transforms each model vertex into the plottables the terrain fill already uses,
then `$856F` walks the object's polygons through the same `plot_polygon $2AA9`. Needs
the model geometry, so it runs only where :mod:`sentinel.objmodel` finds the game image.
"""

import numpy as np
from numba import njit

from sentinel import memmap as mm, passcost, rendercost
from sentinel.enemies_jit import (
    ZP_HI,
    _calc_angle,
    _calc_hypotenuse,
    _relative_angles,
    _vertical_angle,
)
from sentinel.los_jit import _bits, _vmul8

_OFLAGS = mm.OBJECTS_FLAGS
_OTYPE = mm.OBJECTS_TYPE
_OHANG = mm.OBJECTS_H_ANGLE
_OZH = mm.OBJECTS_Z_HEIGHT
_OZF = mm.OBJECTS_Z_FRACTION
_NUM_SLOTS = mm.NUM_SLOTS
_T_PLATFORM = 6  # $21E0 CMP #$06: a platform level with the eye is plotted from above

_MUL8 = passcost.MUL8
_MUL8_BIT = passcost.MUL8_BIT
_PREP_OBJECT = passcost.PREP_HEAD + 5  # $2D72 BMI nt 2 + $2D74 BVS process_polygon 3


@njit(cache=True)
def _sin_cos(sine, angle, frac):
    """calculate_sine_and_cosine $0F70: |sin| and |cos| of the 9-bit (angle.frac) out of
    the $AC80 quarter-turn table.  Returns (sin, cos, sign byte $0067, cycles)."""
    cyc = passcost.SC8_CALL
    a = angle & 0xFF
    if a & 0x80:
        a ^= 0x40
        cyc += passcost.SC8_NEG
    else:
        cyc += passcost.SC8_POS
    signs = a
    cyc += passcost.SC8_BODY
    x = ((a << 1) | (frac >> 7)) & 0x7F
    y = (x ^ 0x7F) + 1
    if y & 0x80:
        y = 0x7F
        cyc += passcost.SC8_CLAMP
    else:
        cyc += passcost.SC8_NOCLAMP
    cyc += passcost.SC8_TABLE
    tx, ty = sine[x], sine[y]
    if signs & 0x80:
        if signs & 0x40:
            return tx, ty, signs, cyc + passcost.SC8_NEG_SAME
        return ty, tx, signs, cyc + passcost.SC8_NEG_SWAP
    if signs & 0x40:
        return ty, tx, signs, cyc + passcost.SC8_SWAP
    return tx, ty, signs, cyc + passcost.SC8_SAME


@njit(cache=True)
def _transform(
    zp,
    vangle,
    vradius,
    vheight,
    sine,
    vxy,
    first,
    last,
    c59,
    c57,
    c5b,
    c5c,
    c5d,
    c5e,
    rot_lo,
    rot_hi,
    v_angle,
):
    """$8475: project the object's vertices into ``vxy``, returning the cycles."""
    cyc = passcost.OBJ_TRANSFORM
    for i in range(last - first):
        v = first + i
        cyc += passcost.OBJ_VERTEX_ANGLE
        s, c, signs, scyc = _sin_cos(sine, rot_hi + vangle[v], rot_lo)
        cyc += scyc
        radius = vradius[v]
        cyc += passcost.OBJ_VERTEX_COS + _MUL8 + _MUL8_BIT * _bits(c)
        # $849E keeps only the product's HIGH byte, as the low byte of a 16-bit value.
        lo, hi = _vmul8(c, radius)[0], 0
        cyc += passcost.OBJ_VERTEX_COS_SIGN
        if signs & 0x40:  # $84A4: the cosine is negative, so negate the product
            cyc += passcost.OBJ_COS_NEGATIVE
            val = (-lo) & 0xFFFF
            hi, lo = (val >> 8) & 0xFF, val & 0xFF
        else:
            cyc += passcost.OBJ_COS_POSITIVE
        cyc += passcost.OBJ_VERTEX_Y
        t = c5d + lo
        y_lo = t & 0xFF
        y_hi = (c5e + hi + (1 if t > 0xFF else 0)) & 0xFF
        if y_hi & 0x80:
            cyc += passcost.OBJ_Y_NEGATIVE
            val = (-((y_hi << 8) | y_lo)) & 0xFFFF
            y_hi, y_lo = val >> 8, val & 0xFF
        else:
            cyc += passcost.OBJ_Y_POSITIVE
        cyc += passcost.OBJ_VERTEX_X + _MUL8 + _MUL8_BIT * _bits(s)
        x_hi, _x_lo = _vmul8(s, radius)
        for k in range(ZP_HI):
            zp[k] = 0
        zp[0x80] = x_hi
        zp[0x83] = 0
        zp[0x82] = y_lo
        zp[0x85] = y_hi
        zp[0x86] = signs
        zp[0x88] = 0
        cyc += _calc_angle(zp)
        cyc += passcost.OBJ_VERTEX_SCREEN_X
        t = zp[0x8A] + c59
        vxy[i, 0] = t & 0xFF
        vxy[i, 1] = (zp[0x8B] + c57 + (1 if t > 0xFF else 0)) & 0xFF
        cyc += _calc_hypotenuse(zp)
        cyc += passcost.OBJ_VERTEX_Z
        z = (vheight[v] << 1) & 0xFF
        if vheight[v] & 0x80:  # $8500: bit 7 doubled out into the carry means negative
            cyc += passcost.OBJ_Z_NEGATIVE
            z_hi = 0xFF if z else 0x00
            z = (-z) & 0xFF
        else:
            cyc += passcost.OBJ_Z_POSITIVE
            z_hi = 0
        cyc += passcost.OBJ_VERTEX_VANGLE
        t = c5b + z
        zp[0x80] = t & 0xFF
        rel_hi = (c5c + z_hi + (1 if t > 0xFF else 0)) & 0xFF
        sy_hi, vcyc = _vertical_angle(zp, rel_hi, v_angle)
        cyc += vcyc
        vxy[i, 2] = zp[0x50]
        vxy[i, 3] = sy_hi
        cyc += passcost.OBJ_VERTEX_STORE
        cyc += (
            passcost.OBJ_VERTEX_LAST
            if i == last - first - 1
            else passcost.OBJ_VERTEX_NEXT
        )
    return cyc


@njit(cache=True)
def _passes(concave, rot_hi, c5b, c5c):
    """$854D: how many passes a concave object needs, and the cycles the test costs."""
    cyc = passcost.OBJ_SHAPE
    if concave == 0:
        return 1, cyc + passcost.OBJ_CONVEX
    cyc += passcost.OBJ_CONCAVE
    if concave & 1:  # $8553: a robot, sentry or sentinel stands upright
        cyc += passcost.OBJ_UPRIGHT
        if c5c:
            cyc += passcost.OBJ_UPRIGHT_Z
            a = (c5c ^ 0x80) & 0xFF
        else:
            cyc += passcost.OBJ_UPRIGHT_LOW
            if c5b == 0:
                return 1, cyc + passcost.OBJ_UPRIGHT_FLAT
            cyc += passcost.OBJ_UPRIGHT_HALVE
            a = ((c5b >> 1) ^ 0x80) & 0xFF
    else:  # $8555 a meanie, whose own bearing decides
        cyc += passcost.OBJ_MEANIE
        a = (rot_hi + 0xC0) & 0xFF
    if a & 0x80:
        return 2, cyc + passcost.OBJ_TWO_PASS
    return 1, cyc + passcost.OBJ_ONE_PASS


@njit(cache=True)
def object_cycles(
    mem,
    zp,
    model,
    observer,
    target,
    h_angle,
    v_angle,
    sect,
    bufs,
    top,
    bot,
    left,
    right,
    work,
):
    """plot_object $8533 for ``target`` seen from ``observer``: the vertex transform,
    then every polygon through plot_polygon.  Returns (cycles, $0010)."""
    (
        vfirst,
        vlast,
        pfirst,
        plast,
        concave,
        vangle,
        vheight,
        vradius,
        pcolour,
        pnverts,
        plist,
        sine,
    ) = model
    vxy, sxb, sxh, vlist, rows, flags = work
    cyc = passcost.OBJ_CALL + passcost.OBJ_RELATIVE
    _c57, ang_lo, ang_hi, z_lo, z_hi, rcyc = _relative_angles(mem, zp, observer, target)
    cyc += rcyc
    # $841F/$8423 against the view being priced, which is not the image's own bearing.
    c57 = (ang_hi - h_angle + 0x0A) & 0xFF
    c59 = ang_lo  # $8415: $001F is 0 in play
    c5d, c5e = zp[0x7C], zp[0x7D]
    rot_lo = (-ang_lo) & 0xFF  # $842A the object's apparent rotation to the origin
    rot_hi = (int(mem[_OHANG + target]) - ang_hi - (1 if ang_lo else 0)) & 0xFF
    otype = int(mem[_OTYPE + target])
    cyc += passcost.OBJ_DISTANCE
    suppress = (
        1 if c5e >= 0x0F else 0
    )  # $8538 CMP #$0F: a distant object loses its edges
    cyc += _transform(
        zp,
        vangle,
        vradius,
        vheight,
        sine,
        vxy,
        vfirst[otype],
        vlast[otype],
        c59,
        c57,
        z_lo,
        z_hi,
        c5d,
        c5e,
        rot_lo,
        rot_hi,
        v_angle,
    )
    cyc += passcost.OBJ_PASS_HEAD
    npass, pcyc = _passes(concave[otype], rot_hi, z_lo, z_hi)
    cyc += pcyc
    skip_flag = 0
    for p in range(npass):
        cyc += passcost.OBJ_POLY_HEAD
        for poly in range(pfirst[otype], plast[otype]):
            cyc += passcost.OBJ_POLY_TEST
            if npass > 1:
                cyc += passcost.OBJ_POLY_PASS
                if (pcolour[poly] ^ skip_flag) & 0x80:  # $8585: not this pass
                    cyc += passcost.OBJ_POLY_NEXT
                    continue
            nv = pnverts[poly]
            for k in range(nv + 1):
                vlist[k] = plist[poly, k]
            cyc += passcost.OBJ_POLY_SETUP
            c, sect = rendercost._plot_polygon(
                vxy,
                vlist,
                nv,
                _PREP_OBJECT,
                sect,
                bufs,
                top,
                bot,
                left,
                right,
                sxb,
                sxh,
                rows,
                flags,
                suppress,
            )
            cyc += c + passcost.OBJ_POLY_NEXT
        cyc += passcost.OBJ_POLY_DONE
        if p == 0 and npass > 1:
            cyc += passcost.OBJ_PASS_AGAIN
            skip_flag = 0x80
    return cyc + passcost.OBJ_TAIL, sect


@njit(cache=True)
def stack_cycles(
    mem,
    zp,
    model,
    observer,
    player,
    h_angle,
    v_angle,
    tile_byte,
    sect,
    bufs,
    top,
    bot,
    left,
    right,
    work,
):
    """plot_stack_of_objects $21AE: the tile's whole object stack, bottom-up for the
    levels at or below the eye and then top-down for the rest."""
    cyc = passcost.STACK_HEAD
    top_slot = tile_byte & 0x3F
    slots = np.zeros(_NUM_SLOTS, dtype=np.int64)
    n = 0
    slot = top_slot
    for _ in range(_NUM_SLOTS):  # $21B5 the height walk, top slot downwards
        slots[n] = slot
        n += 1
        flags = int(mem[_OFLAGS + slot])
        if flags < 0x40:
            cyc += passcost.STACK_LAST
            break
        cyc += passcost.STACK_LEVEL
        slot = flags & 0x3F
    cyc += passcost.STACK_COUNT
    left_to_plot = n - 1
    if left_to_plot == 0:
        cyc += passcost.STACK_ONE
    else:
        cyc += passcost.STACK_MANY
    below = 0
    pzf, pzh = int(mem[_OZF + player]), int(mem[_OZH + player])
    while left_to_plot > 0:  # $21C6 the objects at or below the player, bottom-up
        slot = slots[n - 1 - below]
        cyc += passcost.STACK_BELOW_TEST
        d = pzf - int(mem[_OZF + slot])
        dh = (pzh - int(mem[_OZH + slot]) - (1 if d < 0 else 0)) & 0xFF
        if dh & 0x80:
            cyc += passcost.STACK_ABOVE
            break
        if dh == 0 and (d & 0xFF) == 0:
            cyc += passcost.STACK_LEVEL_TYPE
            if int(mem[_OTYPE + slot]) == _T_PLATFORM:
                break
        c, sect = object_cycles(
            mem,
            zp,
            model,
            observer,
            slot,
            h_angle,
            v_angle,
            sect,
            bufs,
            top,
            bot,
            left,
            right,
            work,
        )
        cyc += c + passcost.STACK_BELOW
        below += 1
        left_to_plot -= 1
        cyc += passcost.STACK_RESCAN * below
    cyc += passcost.STACK_ABOVE_HEAD
    for k in range(left_to_plot + 1):  # $21FF the rest, top-down
        c, sect = object_cycles(
            mem,
            zp,
            model,
            observer,
            slots[k],
            h_angle,
            v_angle,
            sect,
            bufs,
            top,
            bot,
            left,
            right,
            work,
        )
        cyc += c + passcost.STACK_PLOT
    return cyc + passcost.STACK_RTS, sect


@njit(cache=True)
def pass_cycles(
    mem,
    zp,
    model,
    tiles,
    ntiles,
    s4b,
    s66,
    sect0,
    bufs,
    top,
    bot,
    observer,
    player,
    h_angle,
    v_angle,
    tile_work,
    obj_work,
    left,
    right,
):
    """One plot_world pass with the object stacks emulated as well as the terrain.

    The same $AD00/$AE00 tables and the same $0010 thread through both, in the order
    plot_tile draws them: a tile's checkerboard first, then $21AE's whole stack.
    """
    cyc = 0
    sect = sect0
    for i in range(ntiles):
        c, sect = rendercost.tile_cycles(
            tiles, i, s4b, s66, sect, bufs, top, bot, left, right, tile_work
        )
        cyc += c
        tb = tiles[i, rendercost.T_BYTE]
        if tb >= 0xC0:  # $2A32 JMP plot_stack_of_objects
            c, sect = stack_cycles(
                mem,
                zp,
                model,
                observer,
                player,
                h_angle,
                v_angle,
                tb,
                sect,
                bufs,
                top,
                bot,
                left,
                right,
                obj_work,
            )
            cyc += c
    return cyc


def model_arrays():
    """:mod:`sentinel.objmodel`'s tables as the tuple ``object_cycles`` unpacks."""
    from sentinel import objmodel

    t = objmodel.tables()
    if t is None:
        return None
    return (
        t["vfirst"],
        t["vlast"],
        t["pfirst"],
        t["plast"],
        t["concave"],
        t["vangle"],
        t["vheight"],
        t["vradius"],
        t["pcolour"],
        t["pnverts"],
        t["plist"],
        t["sine"],
    )


def workspace():
    """Scratch the object emulation reuses across a whole plot_world pass."""
    from sentinel import objmodel

    n = objmodel.MAX_VERTICES + 2
    return (
        np.zeros((n, 4), dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        np.zeros(5, dtype=np.int64),
        np.zeros(rendercost.R_N, dtype=np.int64),
        np.zeros(rendercost.F_N, dtype=np.int64),
    )


def scratch_zp():
    """The zero-page window the trig ports write through."""
    return np.zeros(ZP_HI, dtype=np.int64)
