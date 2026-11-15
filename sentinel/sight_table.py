"""Enemy sight of a body on every tile and stack height, from one board's entry RAM.

``seen(tile, k)`` is the ROM's check ($1887) of a robot on ``tile`` above ``k`` boulders
built on its entry stack, asked once per key: ``{(enemy slot, full)}``, or None where
nothing can stand.
"""

import numpy as np

from sentinel import actions, exposure_jit, memmap as mm, terrain, threat
from sentinel.state import State

ENEMIES = (mm.T_SENTRY, mm.T_SENTINEL)


def stack(st, x, y):
    """The tile's bare terrain byte, or its object stack top down as (type, z, frac)."""
    b = terrain.tile_byte(st, x, y)
    out = []
    while b >= mm.OBJECT_TILE:
        s = b & 0x3F
        out.append((st.obj_type[s], st.obj_z_height[s], st.obj_z_frac[s]))
        flags = st.obj_flags[s]
        b = mm.OBJECT_TILE | (flags & 0x3F) if flags & 0x40 else 0
    return tuple(out) or b


def enemies(st):
    """Occupied sentry and Sentinel slots."""
    return {s for s in st.occupied_slots() if st.obj_type[s] in ENEMIES}


class SightTable:
    """Memoized entry-RAM sight answers; ``fresh(st, tile)`` says whether ``st``'s stack
    on ``tile`` is still entry's, the condition for an answer to hold."""

    def __init__(self, entry):
        self.entry = bytes(entry)
        self.base = State.from_mem(self.entry)
        self._seen = {}

    def fresh(self, st, tile):
        """Whether ``tile`` holds in ``st`` exactly what it held at entry."""
        return stack(st, *tile) == stack(self.base, *tile)

    def seen(self, tile, k=0):
        """``{(enemy, full)}`` with sight of a robot on ``tile`` over ``k`` new boulders."""
        key = (tuple(tile), int(k))
        if key not in self._seen:
            self._seen[key] = self._ask(*key)
        return self._seen[key]

    def _ask(self, tile, k):
        st = self.base.clone()
        for _ in range(k):
            st.energy = 0x3F
            if actions.create(st, mm.T_BOULDER, tile) is None:
                return None
        free = st.free_slots()
        if not free or not threat._place_phantom(  # pylint: disable=protected-access
            st, tile, free[-1]
        ):
            return None
        rows = exposure_jit.exposures_of(
            np.frombuffer(st.mem, dtype=np.uint8), free[-1]
        )
        return frozenset((int(e), bool(f)) for e, _a, f in rows.tolist())
