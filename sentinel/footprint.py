"""The static write footprint of the modelled routines: every address of the 64 KB image
that the spec kernels and actions can ever write, so a pool row stores only those bytes.

The footprint is read off the sources, not observed.  Each store into the image -- a
``mem[...] =``, a ``_wr`` or tile write, an object-array or player-scalar assignment --
contributes the interval its address expression can take: constants and module integers
are exact, ``+ - * & | << >>`` combine intervals, and anything else (a slot, a tile
coordinate, a table read) is a byte, as it is for the 6502's own index registers.
"""

import ast
import functools
import importlib
import inspect

import numpy as np

from sentinel import memmap as mm

MODULES = (
    "sentinel.enemies_jit",
    "sentinel.los_jit",
    "sentinel.badline_jit",
    "sentinel.actions_jit",
    "sentinel.projector",
)
IMAGE = 1 << 16
BYTE = (0, 0xFF)
TILE_MAP = (0x0400, 0x07FF)
_PRIMITIVES = ("_wr", "_set_tile_byte")
_ARRAYS = {
    "obj_x": mm.OBJECTS_X,
    "obj_y": mm.OBJECTS_Y,
    "obj_z_height": mm.OBJECTS_Z_HEIGHT,
    "obj_z_frac": mm.OBJECTS_Z_FRACTION,
    "obj_h_angle": mm.OBJECTS_H_ANGLE,
    "obj_v_angle": mm.OBJECTS_V_ANGLE,
    "obj_flags": mm.OBJECTS_FLAGS,
    "obj_type": mm.OBJECTS_TYPE,
}
_SCALARS = {"energy": mm.PLAYER_ENERGY, "player": mm.PLAYER_OBJECT}


def _is_mem(node):
    """Whether ``node`` names the image: ``mem`` or ``<anything>.mem``."""
    return (isinstance(node, ast.Name) and node.id == "mem") or (
        isinstance(node, ast.Attribute) and node.attr == "mem"
    )


def _resolve(node, scope):
    """The value a dotted name denotes in ``scope``, or None."""
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    base = _resolve(node.value, scope) if isinstance(node, ast.Attribute) else None
    return getattr(base, node.attr, None) if base is not None else None


def _span(node, scope):
    """The ``(lo, hi)`` interval of integer expression ``node``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value, node.value
    if isinstance(node, ast.Name) and node.id in scope.get("__loops__", ()):
        if node.id not in scope:
            return scope["__loops__"][node.id]
    if isinstance(node, (ast.Name, ast.Attribute)):
        val = _resolve(node, scope)
        return (int(val),) * 2 if isinstance(val, (int, np.integer)) else BYTE
    if isinstance(node, ast.Call) and len(node.args) == 1:
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name.startswith(("int", "uint")):
            return _span(node.args[0], scope)
    if isinstance(node, ast.BinOp):
        (alo, ahi), (blo, bhi) = _span(node.left, scope), _span(node.right, scope)
        op = type(node.op)
        if op is ast.Add:
            return alo + blo, ahi + bhi
        if op is ast.Sub:
            return alo - bhi, ahi - blo
        if op is ast.Mult and min(alo, blo) >= 0:
            return alo * blo, ahi * bhi
        if op is ast.BitAnd and blo == bhi >= 0:
            return 0, min(ahi, bhi) if alo >= 0 else bhi
        if op is ast.BitOr and min(alo, blo) >= 0:
            return max(alo, blo), (1 << max(ahi, bhi).bit_length()) - 1
        if op is ast.LShift and blo == bhi >= 0 and alo >= 0:
            return alo << blo, ahi << bhi
        if op is ast.RShift and blo == bhi >= 0 and alo >= 0:
            return alo >> blo, ahi >> bhi
    return BYTE


def _clip(lo, hi):
    return max(0, lo), min(IMAGE - 1, hi)


def _loop_bounds(fn, scope):
    """``{name: (lo, hi)}`` of the ``for name in range(...)`` variables of function ``fn``
    that are not its parameters, the union over the loops that bind each."""
    params = {a.arg for a in fn.args.args}
    bounds = {}
    for node in ast.walk(fn):
        it = getattr(node, "iter", None)
        if not (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id not in params
            and isinstance(it, ast.Call)
            and getattr(it.func, "id", "") == "range"
        ):
            continue
        args = [_span(a, scope) for a in it.args]
        step = args[2][0] if len(args) == 3 else 1
        start, stop = (args[0], args[1]) if len(args) > 1 else ((0, 0), args[0])
        iv = (start[0], stop[1] - 1) if step > 0 else (stop[0] + 1, start[1])
        old = bounds.get(node.target.id, iv)
        bounds[node.target.id] = (min(old[0], iv[0]), max(old[1], iv[1]))
    return bounds


def _stores(fn, scope):
    """The intervals the stores inside function ``fn`` can reach."""
    out = []
    scope = {**scope, "__loops__": _loop_bounds(fn, scope)}
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for tgt in targets:
            if isinstance(tgt, ast.Subscript) and _is_mem(tgt.value):
                out.append(_span(tgt.slice, scope))
            elif isinstance(tgt, ast.Subscript) and isinstance(
                tgt.value, ast.Attribute
            ):
                base = _ARRAYS.get(tgt.value.attr)
                if base is not None:
                    lo, hi = _span(tgt.slice, scope)
                    out.append((base + lo, base + hi))
            elif isinstance(tgt, ast.Attribute) and tgt.attr in _SCALARS:
                out.append((_SCALARS[tgt.attr],) * 2)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_wr" and len(node.args) >= 2:
                out.append(_span(node.args[1], scope))
            elif node.func.id == "_set_tile_byte":
                out.append(TILE_MAP)
    return [_clip(*iv) for iv in out]


def _writes(tree, scope):
    """The intervals the stores of module ``tree`` can reach: those of every function but
    the store primitives themselves, which their call sites account for."""
    return [
        iv
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef) and fn.name not in _PRIMITIVES
        for iv in _stores(fn, scope)
    ]


@functools.cache
def mask():
    """``(65536,)`` bool: the addresses the modelled routines can write."""
    out = np.zeros(IMAGE, dtype=bool)
    for name in MODULES:
        module = importlib.import_module(name)
        tree = ast.parse(inspect.getsource(module))
        for lo, hi in _writes(tree, vars(module)):
            out[lo : hi + 1] = True
    return out


@functools.cache
def region():
    """``(addrs, amap)``: the ascending written addresses and, per address of the image,
    its index in ``addrs`` (-1 for a byte only ever read)."""
    addrs = np.flatnonzero(mask()).astype(np.int32)
    amap = np.full(IMAGE, -1, dtype=np.int32)
    amap[addrs] = np.arange(addrs.size, dtype=np.int32)
    return addrs, amap
