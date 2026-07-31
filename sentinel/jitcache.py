"""Key numba's on-disk cache to the cost constants the jit twins inline.

``@njit(cache=True)`` invalidates on the DEFINING file's source stamp, so editing
:mod:`sentinel.passcost` alone leaves a stale compilation charging the old cycles.
A digest of those constants in ``NUMBA_CACHE_DIR`` makes a constant change a new cache.
"""

import hashlib
import os

from sentinel import passcost, writeruns


def _digest():
    """A stable hash of every module-level number the jit twins bind as a global."""
    items = sorted(
        (n, v)
        for n, v in vars(passcost).items()
        if not n.startswith("_") and isinstance(v, (int, float))
    )
    hasher = hashlib.blake2b(repr(items).encode(), digest_size=8)
    hasher.update(writeruns.RUN.tobytes())  # the badline clock's tables inline too
    hasher.update(repr(writeruns.ANCHOR).encode())
    return hasher.hexdigest()


def install():
    """Point NUMBA_CACHE_DIR at a constants-keyed directory, unless already set."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return os.environ["NUMBA_CACHE_DIR"]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__pycache__")
    path = os.path.join(root, f"numba-{_digest()}")
    os.makedirs(path, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = path
    return path
