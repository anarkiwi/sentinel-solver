"""Run the actual C64 Sentinel machine code in a 6502 emulator (py65) to generate
a landscape deterministically — "the same algorithm" by literally executing it.

Loads the game memory image (out/sentinel_stage2.bin), zeroes the uninitialised
RAM the game assumes is clear, seeds the PRNG from a landscape number via the real
seed routine ($33ED), then calls the real generator ($2ACC) and returns the machine
state for inspection/rendering.
"""

import os

from py65.devices.mpu6502 import MPU
from py65.memory import ObservableMemory

IMG = "out/sentinel_stage2.bin"

SEED = 0x33ED  # seed_prnd_from_landscape_number (X=lo, Y=hi)
RESET = 0x1149  # reset_game_state
GENERATE = 0x2ACC  # generate_landscape (terrain + provisional)
INIT_ENEMIES = (
    0x1420  # use_enemies_based_on_landscape_number (places Sentinel/sentries)
)
INIT_PLAYER = 0x1450  # initialise_player_and_trees

# generate_landscape ($2ACC) finishes its TERRAIN work at the RTS $2B21 ("Leaves
# to &5f7e to_preview_screen"); it then TAIL-CALLS into the preview render path.
# When called JSR-style the control flow does NOT return to our guard at $2B21 --
# it falls through into rendering, which consumes EXTRA prnd calls (object random
# rotations $1F83, secret-code table draws $14B6, ...). Stopping there desyncs the
# PRNG from the real PLAY path, so the subsequent enemy/player/tree placement draws
# a DIFFERENT tree count + positions (15 trees vs the live ROM's 16). We therefore
# stop generate at $2B21 -- exactly where play_setup's `JSR generate_landscape`
# logically returns -- so the prnd stream entering initialise_player_and_trees
# matches the live game. (Verified: ls0 now yields the live 16-tree set.)
GENERATE_END = 0x2B21  # logical end of generate_landscape terrain build

# memory regions actually present in the tape image (everything else = fresh RAM = 0)
LOADED = [
    (0x0200, 0x023F),
    (0x02A7, 0x0303),
    (0x033C, 0x03FB),
    (0x0D00, 0x3FFF),
    (0x5A00, 0xB5FF),
    (0xBF00, 0xC1FF),
]


def fresh_machine():
    if not os.path.exists(IMG):
        raise FileNotFoundError(
            f"{IMG} missing: place the game memory image there (not distributed)"
        )
    img = open(IMG, "rb").read()
    mem = bytearray(0x10000)
    for a, b in LOADED:
        mem[a : b + 1] = img[a : b + 1]
    state = {"stop": False, "raster": 0}
    m = ObservableMemory()

    def rd(addr):
        if 0xDC00 <= addr <= 0xDC01:  # keyboard scan = play loop running
            state["stop"] = True
            return 0xFF
        if addr == 0xD012:  # raster: advance so raster-waits complete
            state["raster"] = (state["raster"] + 1) & 0xFF
            return state["raster"]
        return mem[addr]

    def wr(addr, v):
        if 0xE000 <= addr <= 0xFFFF:  # first render-buffer write = gen complete
            state["stop"] = True
        mem[addr] = v & 0xFF

    m.subscribe_to_write(range(0x10000), wr)
    m.subscribe_to_read(range(0x10000), rd)
    cpu = MPU(memory=m)
    # put RTS ($60) at common KERNAL entry points so any stray JSR returns cleanly
    for a in (
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
    ):
        mem[a] = 0x60
    return cpu, mem, state


def call(cpu, mem, addr, a=0, x=0, y=0, maxins=40_000_000, state=None, stop_pc=None):
    """JSR-style call: push a sentinel return address, run until it returns
    (RTS to the guard) or until the keyboard is scanned (frame complete).

    `stop_pc` (optional): also stop when the CPU reaches this address. Used to halt
    generate_landscape at its terrain-build end ($2B21) before it tail-calls into
    preview rendering (which would consume extra prnd and desync the PRNG -- see
    GENERATE_END). When stopped via stop_pc, `state["stop"]` is left False so the
    caller can continue the sequence cleanly."""
    RET = 0xFFF0
    mem[RET] = 0x60  # RTS guard
    cpu.a, cpu.x, cpu.y = a & 0xFF, x & 0xFF, y & 0xFF
    sp = cpu.sp
    mem[0x0100 + sp] = (RET - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (RET - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = addr
    n = 0
    while n < maxins:
        if cpu.pc == RET or (stop_pc is not None and cpu.pc == stop_pc):
            break
        cpu.step()
        n += 1
        if state is not None and state["stop"]:
            break
    return n


def generate(landscape):
    """Generate landscape `landscape` by running the real C64 code, reproducing the
    game's play-setup sequence ($1A97): seed -> generate terrain -> place enemies ->
    place player + trees. $2ACC runs into the (non-terminating) preview render, so it
    is stopped at the first render-buffer write — terrain + provisional objects are
    done by then — and the enemy/player/tree placement routines are called explicitly.
    Returns (mem, instr_count)."""
    cpu, mem, state = fresh_machine()
    lo, hi = landscape & 0xFF, (landscape >> 8) & 0xFF
    call(cpu, mem, RESET, state=state)
    state["stop"] = False
    call(cpu, mem, SEED, x=lo, y=hi, state=state)
    state["stop"] = False
    # Stop generate at its terrain-build end ($2B21), NOT at the first $E000 render
    # write -- the render tail consumes extra prnd and desyncs the tree placement
    # (15 trees instead of the live 16). See GENERATE_END.
    ins = call(cpu, mem, GENERATE, state=state, stop_pc=GENERATE_END)
    state["stop"] = False
    call(cpu, mem, INIT_ENEMIES, state=state)  # Sentinel + sentries on flat tiles
    state["stop"] = False
    call(cpu, mem, INIT_PLAYER, state=state)  # player + trees
    return mem, ins
