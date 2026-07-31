"""Key numba's on-disk cache to the cost constants and kernel sources it inlines.

``@njit(cache=True)`` invalidates on the DEFINING file's source stamp, so editing
:mod:`sentinel.passcost` leaves a stale compilation charging the old cycles, and a kernel
reached only through another module's dispatcher can miss its own edit as well.
"""

import hashlib
import os

from sentinel import passcost, writeruns

# Modules whose source the jit kernels are compiled from; an edit to any must be a miss.
KERNEL_SOURCES = (
    "rendercost.py",
    "objectcost.py",
    "projector_jit.py",
    "enemies_jit.py",
    "los_jit.py",
    "badline_jit.py",  # inlined into enemies_jit, so its own stamp is not consulted
)


def _digest():
    """A stable hash of the cost constants and of every kernel module's source."""
    items = sorted(
        (n, v)
        for n, v in vars(passcost).items()
        if not n.startswith("_") and isinstance(v, (int, float))
    )
    dig = hashlib.blake2b(repr(items).encode(), digest_size=8)
    dig.update(writeruns.RUN.tobytes())  # the badline clock's tables inline too
    dig.update(repr(writeruns.ANCHOR).encode())
    here = os.path.dirname(os.path.abspath(__file__))
    for name in KERNEL_SOURCES:
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                dig.update(handle.read())
    return dig.hexdigest()


def install():
    """Point NUMBA_CACHE_DIR at a source-keyed directory, unless already set."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return os.environ["NUMBA_CACHE_DIR"]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
    path = os.path.join(root, f"numba-{_digest()}")
    os.makedirs(path, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = path
    return path
