"""A self-contained py65 harness that runs the real 6502 game code as a parity
oracle for the simulator's tests.

It needs the game memory image at ``out/sentinel_stage2.bin`` (copyrighted, not
distributed, gitignored); oracle-marked tests auto-skip when it is absent
(see ``conftest.py``). This module imports nothing from ``scripts/`` -- it is the
package's own, minimal re-home of the emulator harness used to freeze the golden
fixtures.
"""

import os

from py65.devices.mpu6502 import MPU
from py65.memory import ObservableMemory

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG = os.path.join(_ROOT, "out", "sentinel_stage2.bin")

SEED = 0x33ED  # seed_prnd_from_landscape_number (X=lo, Y=hi)
RESET = 0x1149  # reset_game_state
GENERATE = 0x2ACC  # generate_landscape (terrain + provisional objects)
GENERATE_END = 0x2B21  # terrain-build end, before the preview render tail
INIT_ENEMIES = 0x1420  # place Sentinel + sentries
INIT_PLAYER = 0x1450  # place player + trees

# memory regions present in the image (everything else = fresh, zeroed RAM).
LOADED = [
    (0x0200, 0x023F),
    (0x02A7, 0x0303),
    (0x033C, 0x03FB),
    (0x0D00, 0x3FFF),
    (0x5A00, 0xB5FF),
    (0xBF00, 0xC1FF),
]

_KERNAL_RTS = (
    0xFF81,
    0xFF90,
    0xFFBA,
    0xFFBD,
    0xFFD2,
    0xFFD5,
    0xFFE1,
    0xFFE4,
    0xFFE7,
    0xFF9F,
    0xFFCC,
)


def available():
    """True when the ROM image fixture is present."""
    return os.path.exists(IMG)


def fresh_machine():
    """A machine with the image loaded and the uninitialised RAM zeroed."""
    if not available():
        raise FileNotFoundError(f"{IMG} missing: place the game memory image there")
    with open(IMG, "rb") as f:
        img = f.read()
    mem = bytearray(0x10000)
    for a, b in LOADED:
        mem[a : b + 1] = img[a : b + 1]
    state = {"stop": False, "raster": 0}
    m = ObservableMemory()

    def rd(addr):
        if 0xDC00 <= addr <= 0xDC01:  # keyboard scan == play loop running
            state["stop"] = True
            return 0xFF
        if addr == 0xD012:  # raster: advance so raster-waits complete
            state["raster"] = (state["raster"] + 1) & 0xFF
            return state["raster"]
        return mem[addr]

    def wr(addr, v):
        if 0xE000 <= addr <= 0xFFFF:  # first render-buffer write == generation done
            state["stop"] = True
        mem[addr] = v & 0xFF

    m.subscribe_to_write(range(0x10000), wr)
    m.subscribe_to_read(range(0x10000), rd)
    cpu = MPU(memory=m)
    for a in _KERNAL_RTS:
        mem[a] = 0x60  # RTS at KERNAL entry points so stray JSRs return cleanly
    return cpu, mem, state


def call(cpu, mem, addr, a=0, x=0, y=0, maxins=40_000_000, state=None, stop_pc=None):
    """JSR-style call: run from `addr` until it returns to a guard, reaches
    `stop_pc`, or the keyboard is scanned (frame complete)."""
    ret = 0xFFF0
    mem[ret] = 0x60
    cpu.a, cpu.x, cpu.y = a & 0xFF, x & 0xFF, y & 0xFF
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = addr
    n = 0
    while n < maxins:
        if cpu.pc == ret or (stop_pc is not None and cpu.pc == stop_pc):
            break
        cpu.step()
        n += 1
        if state is not None and state["stop"]:
            break
    return n


def generate(landscape):
    """Run the real play-setup sequence for `landscape` and return the memory
    image: seed -> generate terrain -> place enemies -> place player + trees."""
    cpu, mem, state = fresh_machine()
    lo, hi = landscape & 0xFF, (landscape >> 8) & 0xFF
    call(cpu, mem, RESET, state=state)
    state["stop"] = False
    call(cpu, mem, SEED, x=lo, y=hi, state=state)
    state["stop"] = False
    call(cpu, mem, GENERATE, state=state, stop_pc=GENERATE_END)
    state["stop"] = False
    call(cpu, mem, INIT_ENEMIES, state=state)
    state["stop"] = False
    call(cpu, mem, INIT_PLAYER, state=state)
    return mem
