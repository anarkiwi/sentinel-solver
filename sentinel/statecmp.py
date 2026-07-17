"""A shared, high-precision state comparator for the sim and the emulator.

The sim and the live VICE game hold play state in a 64 KB image at the same RAM
addresses, so one schema decodes either and :func:`diff` compares two by address.
A field's *tier* is CORE (frame-faithful), SWEEP (cursor/PRNG) or SCRATCH (LOS).
"""

import collections

from sentinel import memmap as mm

CORE = "core"
SWEEP = "sweep"
SCRATCH = "scratch"
TIERS = (CORE, SWEEP, SCRATCH)

_OBJECT_ARRAYS = (
    (mm.OBJECTS_FLAGS, "flags"),
    (mm.OBJECTS_V_ANGLE, "v_angle"),
    (mm.OBJECTS_X, "x"),
    (mm.OBJECTS_Z_HEIGHT, "z_height"),
    (mm.OBJECTS_Y, "y"),
    (mm.OBJECTS_H_ANGLE, "h_angle"),
    (mm.OBJECTS_Z_FRACTION, "z_frac"),
    (mm.OBJECTS_TYPE, "type"),
)

_ENEMY_ARRAYS = (
    (mm.ENEMIES_DRAINING_COOLDOWN, "draining_cd"),
    (mm.ENEMIES_ROTATION_COOLDOWN, "rotation_cd"),
    (mm.ENEMIES_UPDATE_COOLDOWN, "update_cd"),
    (mm.ENEMIES_MEANIE_SEARCH_OBJECT, "meanie_search"),
    (mm.ENEMIES_ENERGY_TO_DISCHARGE, "energy_discharge"),
    (mm.ENEMIES_FAILED_MEANIE_MEMORY, "failed_meanie"),
    (mm.ENEMIES_MEANIE_ATTEMPT_SCANS, "meanie_scans"),
    (mm.ENEMIES_MEANIE_OBJECT, "meanie_obj"),
    (mm.ENEMIES_TARGETED_OBJECT, "targeted_obj"),
    (mm.ENEMIES_TARGETED_OBJECT_EXPOSURE, "targeted_exposure"),
    (mm.ENEMIES_CONSIDERING_MEANIE, "considering_meanie"),
)

_CORE_SCALARS = (
    (mm.PLAYER_OBJECT, "player_object"),
    (mm.PLAYER_ENERGY, "player_energy"),
    (mm.MAX_ENEMIES, "max_enemies"),
    (mm.VERTICAL_SCALE, "vertical_scale"),
    (mm.ENEMY_BELOW_Z, "enemy_below_z"),
    (mm.PLATFORM_X, "platform_x"),
    (mm.PLATFORM_Y, "platform_y"),
    (mm.PLAYER_DIED_BY_DRAINING, "died_by_draining"),
    (mm.COOLDOWN_GATE, "cooldown_gate"),
    (mm.COOLDOWN_BRESENHAM, "cooldown_bresenham"),
    (mm.LANDSCAPE_COMPLETE, "landscape_complete"),
    (mm.PLAYER_NOT_ACTED, "player_not_acted"),
)

# Rewritten by the ROM per-scan LOS march; the sim treats them as queries.
_SCRATCH_SCALARS = (
    (mm.OBJECT_EXPOSURE, "object_exposure"),
    (mm.TARGETED_OBJECT_IN_LOS, "targeted_in_los"),
    (mm.FOV_RELATIVE_H_ANGLE, "fov_relative_h"),
    (mm.TARGETED_OBJECT_SLOT, "targeted_slot"),
    (mm.FOV_WIDTH, "fov_width"),
    (mm.TREE_IN_LOS, "tree_in_los"),
    (mm.TREE_IN_LOS_TO_HEAD, "tree_in_los_head"),
)


def _build_fields():
    """The full ordered (addr, label, tier) schema, arrays expanded per index."""
    fields = []
    for base, name in _OBJECT_ARRAYS:
        for s in range(mm.NUM_SLOTS):
            fields.append((base + s, f"obj[{s}].{name}", CORE))
    for base, name in _ENEMY_ARRAYS:
        for e in range(8):
            fields.append((base + e, f"enemy[{e}].{name}", CORE))
    for x in range(mm.N):
        for y in range(mm.N):
            fields.append((mm.TILES_TABLE + mm.tidx(x, y), f"tile[{x},{y}]", CORE))
    for addr, name in _CORE_SCALARS:
        fields.append((addr, name, CORE))
    fields.append((mm.CURSOR, "cursor", SWEEP))
    for i in range(5):
        fields.append((mm.PRND_STATE + i, f"prng[{i}]", SWEEP))
    for addr, name in _SCRATCH_SCALARS:
        fields.append((addr, name, SCRATCH))
    return tuple(fields)


FIELDS = _build_fields()
MAX_ADDR = max(a for a, _, _ in FIELDS)  # highest address the diff reads ($1335)

Divergence = collections.namedtuple("Divergence", "addr label tier a b")


def diff(img_a, img_b, tiers=None):
    """Schema fields whose byte differs between two images, in canonical order.

    ``tiers`` optionally restricts to a set of tiers; ``a``/``b`` are the first
    and second image's bytes at each divergent address."""
    out = []
    for addr, label, tier in FIELDS:
        if tiers is not None and tier not in tiers:
            continue
        va, vb = img_a[addr], img_b[addr]
        if va != vb:
            out.append(Divergence(addr, label, tier, va, vb))
    return out


def by_tier(divs):
    """Group a divergence list into ``{tier: [Divergence, ...]}``."""
    out = {t: [] for t in TIERS}
    for d in divs:
        out[d.tier].append(d)
    return out


def format_divergence(d, a_name="A", b_name="B"):
    """A one-line ``label @ $addr: A=.. B=.. d=..`` rendering of one divergence."""
    delta = (d.b - d.a) & 0xFF
    return (
        f"{d.label:<28} @ ${d.addr:04X}: "
        f"{a_name}={d.a:>3} (${d.a:02X})  {b_name}={d.b:>3} (${d.b:02X})  d={delta:+d}"
    )
