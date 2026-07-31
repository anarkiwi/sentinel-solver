"""The object model tables ($9CA0-$A420) read out of the local game image.

The geometry `plot_object $8533` draws is data, not code, so it cannot be carried in
this repo; it is read at import from the same gitignored ``out/sentinel_stage2.bin``
the oracle tests use, and :func:`available` is False when that image is absent.
"""

import os

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG = os.path.join(_ROOT, "out", "sentinel_stage2.bin")

VERTEX_BOUNDS = 0x9CA0  # objects_first_vertex; last_vertex is the next entry
POLYGON_BOUNDS = 0x9CAB  # objects_first_polygon / objects_last_polygon, likewise
CONCAVE = 0x9CB6  # $854D: 0 = one pass, else the two-pass orientation rule
V_ANGLE = 0x9DE0  # vertices_angular_coordinate
V_HEIGHT = 0x9F20  # vertices_height, bit 0 the sign after the $84FB doubling
V_RADIUS = 0xA060  # vertices_radial_coordinate
P_COLOUR = 0xA1A0  # polygons_type_and_colour; bits 0-1 the vertex count - 3
P_LIST_LO = 0xA2E0  # polygon_vertex_list_lo / _hi -> the $003C vertex index list
P_LIST_HI = 0xA420
SIN_TABLE = 0xAC80  # $0F86/$0F89 |sin| and |cos| by quarter-turn index
N_TYPES = 8
MAX_VERTICES = 30  # the largest objects_last_vertex - objects_first_vertex

_TABLES = [None]


def available():
    """True when the game image the model tables live in is present."""
    return os.path.exists(IMG)


def tables():
    """The model tables as numpy arrays, or None when the image is absent.

    Keys: vfirst/vlast/pfirst/plast/concave per type, vangle/vheight/vradius per vertex,
    pcolour/plist/pnverts per polygon, and the $AC80 sine table.
    """
    if _TABLES[0] is None:
        _TABLES[0] = _load()
    return _TABLES[0]


def _load():
    if not available():
        return None
    with open(IMG, "rb") as handle:
        img = handle.read()
    raw = np.frombuffer(img, dtype=np.uint8)
    vb = raw[VERTEX_BOUNDS : VERTEX_BOUNDS + N_TYPES + 1].astype(np.int64)
    pb = raw[POLYGON_BOUNDS : POLYGON_BOUNDS + N_TYPES + 1].astype(np.int64)
    npoly = int(pb[N_TYPES])
    nvert = int(vb[N_TYPES])
    pcolour = raw[P_COLOUR : P_COLOUR + npoly].astype(np.int64)
    pnverts = (pcolour & 3) + 3  # $858E: 0 = triangle, 1 = quadrilateral
    plist = np.zeros((npoly, 5), dtype=np.int64)
    for p in range(npoly):
        base = (int(raw[P_LIST_HI + p]) << 8) | int(raw[P_LIST_LO + p])
        for k in range(int(pnverts[p]) + 1):
            plist[p, k] = int(raw[base + k]) - 0x40  # $8477: slot $40 is vertex 0
    return {
        "vfirst": vb[:N_TYPES],
        "vlast": vb[1:],
        "pfirst": pb[:N_TYPES],
        "plast": pb[1:],
        "concave": raw[CONCAVE : CONCAVE + N_TYPES].astype(np.int64),
        "vangle": raw[V_ANGLE : V_ANGLE + nvert].astype(np.int64),
        "vheight": raw[V_HEIGHT : V_HEIGHT + nvert].astype(np.int64),
        "vradius": raw[V_RADIUS : V_RADIUS + nvert].astype(np.int64),
        "pcolour": pcolour,
        "pnverts": pnverts.astype(np.int64),
        "plist": plist,
        "sine": raw[SIN_TABLE : SIN_TABLE + 128].astype(np.int64),
    }
