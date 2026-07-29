"""Atlas layer 1: the generated board, cached as its 64 KB memory image.

An entry is that image zlib-compressed (~1.2 KB) at ``out/atlas/<signature>/ls<code>.z``,
loaded in ~0.3 ms instead of the ~17 ms to generate it.  The signature digests the
GENERATOR sources only: editing generation invalidates entries, editing a metric cannot.
"""

import functools
import hashlib
import os
import zlib

from sentinel import memmap as mm
from sentinel.game import Game
from sentinel.state import State

CACHE_VERSION = 1  # bump to invalidate independently of the sources
_HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("landscape.py", "prng.py", "state.py", "memmap.py", "game.py")
DEFAULT_ROOT = os.path.join(os.path.dirname(_HERE), "out", "atlas")
ROOT_ENV = "SENTINEL_ATLAS_CACHE"

# A landscape code is the four digits a player types on the keypad.
MIN_CODE = 0
MAX_CODE = 9999


@functools.lru_cache(maxsize=1)
def signature():
    """12 hex chars of sha256 over ``CACHE_VERSION`` + the generator sources."""
    digest = hashlib.sha256(str(CACHE_VERSION).encode())
    for name in SOURCES:
        with open(os.path.join(_HERE, name), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def root():
    """The cache root: ``$SENTINEL_ATLAS_CACHE``, else ``out/atlas``.  Read per call so
    it survives any process start method."""
    return os.environ.get(ROOT_ENV) or DEFAULT_ROOT


def cache_dir():
    """The directory holding every entry valid for the current generator."""
    return os.path.join(root(), signature())


def entry_path(code):
    return os.path.join(cache_dir(), f"ls{code:04d}.z")


def valid_code(code):
    """``code`` as an int, checked against the keypad range 0..9999."""
    code = int(code)
    if not MIN_CODE <= code <= MAX_CODE:
        raise ValueError(f"landscape code {code} outside {MIN_CODE}..{MAX_CODE}")
    return code


def generate(code):
    """A freshly generated board at the ROM's at-entry state, bypassing the cache."""
    return Game.typed(valid_code(code)).state


def load(code):
    """The cached :class:`~sentinel.state.State` for ``code``, or None on a miss."""
    try:
        with open(entry_path(code), "rb") as handle:
            raw = zlib.decompress(handle.read())
    except (OSError, zlib.error):
        return None
    return State(bytearray(raw)) if len(raw) == mm.MEM_SIZE else None


def store(code, state):
    """Write ``state``'s image as this code's entry, by atomic rename."""
    path = entry_path(code)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(zlib.compress(bytes(state.mem), 6))
    os.replace(tmp, path)


def state_for(code, regen=False):
    """``(State, hit)`` for ``code``: the cached image unless ``regen``, else generated
    and cached."""
    code = valid_code(code)
    if not regen:
        state = load(code)
        if state is not None:
            return state, True
    state = generate(code)
    store(code, state)
    return state, False
