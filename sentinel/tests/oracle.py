"""A self-contained py65 harness that runs the real 6502 game code as a parity
oracle for the simulator's tests.

It needs the game memory image at ``out/sentinel_stage2.bin`` (copyrighted, not
distributed, gitignored); oracle-marked tests auto-skip when it is absent
(see ``conftest.py``). This is the package's own, self-contained emulator harness
used to generate the golden fixtures.
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


def _wrap(mem):
    """Wrap a 64 KB image in an ObservableMemory play machine: the keyboard/raster
    read hooks, the render-buffer stop-write hook and the KERNAL RTS stubs."""
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


def _rom_image():
    if not available():
        raise FileNotFoundError(f"{IMG} missing: place the game memory image there")
    with open(IMG, "rb") as f:
        return f.read()


def fresh_machine():
    """A machine with the image loaded and the uninitialised RAM zeroed."""
    img = _rom_image()
    mem = bytearray(0x10000)
    for a, b in LOADED:
        mem[a : b + 1] = img[a : b + 1]
    return _wrap(mem)


def machine_from_image(src):
    """A play machine whose board is ``src`` (a live 64 KB sim image, e.g.
    ``State.mem``) with the ROM code/tables overlaid from the game image; the
    board lives in low RAM outside the ROM ``LOADED`` regions, so it survives."""
    img = _rom_image()
    mem = bytearray(src)
    for a, b in LOADED:
        mem[a : b + 1] = img[a : b + 1]
    return _wrap(mem)


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


PLOT_WORLD = 0x2625  # plot_world
FRAME_CYCLES = 19656.0  # PAL frame


def render_frame_cost(cpu, mem, state, h_angle, v_angle, maxins=20_000_000):
    """Run the real plot_world ($2625) once headless with the raytraced occlusion
    table ($245B) active, from the player at (h_angle, v_angle); return the exact
    frame cost = plot_world CPU-cycle delta / 19656. Mirrors the golden setup."""
    player = mem[0x000B]
    mem[0x006E] = player
    mem[0x09C0 + player] = h_angle & 0xFF  # objects_h_angle
    mem[0x0140 + player] = v_angle & 0xFF  # objects_v_angle
    for addr in (0x001F, 0x005E, 0x0C78, 0x0C1B, 0x0CDE):
        mem[addr] = 0
    mem[0x0CCE] = 0x80  # skip secret-code check in the raytracer
    mem[0x352C] = 0x60  # stub update_sound (foreground-only cost)
    mem[0x0051], mem[0x0052] = 0xF0, 0x30  # play-view raster clip window ($994b/$994d)
    call(cpu, mem, 0x2993, a=0, state=state)  # initialise_buffer_variables
    state["stop"] = False
    call(cpu, mem, 0x245B, state=state)  # populate raytraced occlusion table
    ret = 0xFFF0
    mem[ret] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (ret - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (ret - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF
    cpu.pc = PLOT_WORLD
    c0 = cpu.processorCycles
    steps = 0
    while cpu.pc != ret and steps < maxins:
        cpu.step()
        steps += 1
    return (cpu.processorCycles - c0) / FRAME_CYCLES


def generate_machine(landscape):
    """Run the real play-setup sequence for `landscape` and return the live
    (cpu, mem, state): seed -> terrain -> place enemies -> place player + trees."""
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
    return cpu, mem, state


def generate(landscape):
    """The `generate_machine` memory image alone (the play-setup result)."""
    return generate_machine(landscape)[1]


# --- round-by-round enemy driver -------------------------------------------
TICK_COOLDOWNS = 0x1317  # update_enemy_cooldowns
UPDATE_ENEMIES = 0x16B5  # update_enemies
# Rendering/sound entry points reached from update_enemies -- stubbed with RTS so
# the enemy dynamics run headless (they don't change the modelled game state):
# update_object_on_screen, play_sound, plot_status_bar, start_tune,
# set_busy_plotting, flush_buffer, start_tune_and_set_viewpoint_has_changed.
RENDER_STUBS = (0x1F9F, 0x3470, 0x9508, 0x888F, 0x1214, 0x3527, 0x1B84)
WORLD_BUSY_PLOTTING = (
    0x0C1F  # bit7 clear => check_if_object_can_be_updated allows updates
)


def prime_enemy_driver(cpu, mem, state):
    """Stub rendering/sound and prime the cursor/gate so update_enemies can be
    stepped one game round at a time, matching the pure-sim ``enemies.step``."""
    for addr in RENDER_STUBS:
        mem[addr] = 0x60  # RTS
    mem[WORLD_BUSY_PLOTTING] = 0x00
    mem[0x0090] = 7  # cursor
    mem[0x0C50] = 0  # cooldown gate
    del cpu, state


def step_enemy_round(cpu, mem, state):
    """One game round on the real 6502: update_enemy_cooldowns + update_enemies."""
    state["stop"] = False
    call(cpu, mem, TICK_COOLDOWNS, state=state)
    state["stop"] = False
    call(cpu, mem, UPDATE_ENEMIES, state=state)
