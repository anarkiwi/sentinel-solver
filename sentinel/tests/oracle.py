"""A self-contained jennings harness that runs the real 6502 game code as a parity
oracle for the simulator's tests and the exact render-cost backend.

jennings executes the NMOS 6510 illegal opcodes, so every entry point must be a real
instruction boundary. The machine is jennings' native core over a ``NativeMemory``
that shares the caller's ``bytearray``: Python reads and writes of ``mem`` touch RAM
directly (no hooks), while the CPU's own accesses fire the play-machine watches, and
whole calls run in :meth:`~jennings.devices.mpu6502.NativeMPU.run_until` with the GIL
released.

It needs the game memory image at ``out/sentinel_stage2.bin`` (copyrighted, not
distributed, gitignored); oracle-marked tests auto-skip when it is absent
(see ``conftest.py``). This is the package's own, self-contained emulator harness
used to generate the golden fixtures.
"""

import functools
import os
from typing import NamedTuple

import numba
import numpy as np
from jennings.devices.mpu6502 import NativeMPU
from jennings.memory import NativeMemory

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG = os.path.join(_ROOT, "out", "sentinel_stage2.bin")

SEED = 0x33ED  # seed_prnd_from_landscape_number (X=lo, Y=hi)
RESET = 0x114D  # reset_game_state (the JSR target at $101F/$1034/$1074/$35F4)
GENERATE = 0x2ACC  # generate_landscape (terrain + provisional objects)
GENERATE_END = 0x2B21  # terrain-build end, before the preview render tail
INIT_ENEMIES = 0x1420  # place Sentinel + sentries
INIT_PLAYER = 0x1450  # place player + trees
PLAY_DISPLAY_FLAG = 0x9AF6  # $9A51's byte, $80 for the whole play loop ($3577)

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
    """Wrap a whole-address-space ``bytearray`` in a native play machine: the keyboard/raster read
    watches and the KERNAL RTS stubs. :func:`call` given ``state`` also arms the
    render-buffer stop-write watch; the watches end its run on first firing."""
    state = {"stop": False, "raster": 0}
    for a in _KERNAL_RTS:
        mem[a] = 0x60  # RTS at KERNAL entry points so stray JSRs return cleanly
    cpu = NativeMPU(memory=NativeMemory(mem, share=True))
    cpu.halt = False

    def flag(*_):
        state["stop"] = True
        if cpu.halt:
            cpu.stop = True

    def rd(addr):
        if addr == 0xD012:  # raster: advance so raster-waits complete
            state["raster"] = (state["raster"] + 1) & 0xFF
            return state["raster"]
        flag()  # keyboard scan == play loop running
        return 0xFF

    cpu.memory.subscribe_to_read((0xD012, 0xDC00, 0xDC01), rd)
    cpu.on_buffer_write = flag  # first render-buffer write == generation done
    return cpu, mem, state


def wrap_image(image):
    """A play machine over ``image`` exactly as given, with no ROM overlay.

    For resuming a snapshot of an already-set-up machine, where an overlay would undo
    the setup: :func:`machine_from_image` rewrites every LOADED region."""
    return _wrap(bytearray(image))


def _rom_image():
    if not available():
        raise FileNotFoundError(
            f"{IMG} missing: run `python -m sentinel.tests.csdb` to build it"
        )
    return _read(IMG, os.stat(IMG).st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _read(path, _mtime):
    with open(path, "rb") as f:
        return f.read()


def fresh_machine():
    """A machine with the image loaded and the uninitialised RAM zeroed."""
    img = _rom_image()
    mem = bytearray(0x10000)
    for a, b in LOADED:
        mem[a : b + 1] = img[a : b + 1]
    return _wrap(mem)


def machine_from_image(src):
    """A play machine whose board is ``src`` (a live whole-address-space sim image, e.g.
    ``State.mem``) with the ROM code/tables overlaid from the game image; the
    board lives in low RAM outside the ROM ``LOADED`` regions, so it survives.

    Per-enemy state INSIDE a LOADED region does not: ``ROTATION_SPEED_TABLE`` (`$9D37`)
    and ``COOLDOWN_BRESENHAM`` (`$1335`) are overwritten by the image, so a caller seeding
    a recorded clock must rewrite them after this returns or the ROM rotates by the
    wrong step and every comparison is silently invalid.  ``PLAY_DISPLAY_FLAG`` is one
    of those, and is restored here because the image was dumped before play began.
    """
    img = _rom_image()
    mem = bytearray(src)
    for a, b in LOADED:
        mem[a : b + 1] = img[a : b + 1]
    mem[PLAY_DISPLAY_FLAG] = 0x80
    return _wrap(mem)


RET = 0xFFF0  # the guard every harness call returns to
MAX_CYCLES = 320_000_000  # runaway cap for one call


def push_return(cpu, mem):
    """Point the stack at the :data:`RET` guard, as a JSR from it would."""
    mem[RET] = 0x60
    sp = cpu.sp
    mem[0x0100 + sp] = (RET - 1) >> 8
    mem[0x0100 + ((sp - 1) & 0xFF)] = (RET - 1) & 0xFF
    cpu.sp = (sp - 2) & 0xFF


def enter(cpu, mem, addr, a=0, x=0, y=0):
    """JSR-style entry at ``addr`` with the given registers, returning to :data:`RET`."""
    cpu.a, cpu.x, cpu.y = a & 0xFF, x & 0xFF, y & 0xFF
    push_return(cpu, mem)
    cpu.pc = addr
    cpu.stop = False


def call(cpu, mem, addr, a=0, x=0, y=0, maxins=None, state=None, stop_pc=None):
    """JSR-style call: run from ``addr`` until it returns to the guard, reaches
    ``stop_pc``, or (given ``state``) the keyboard is scanned or the render buffer
    written. ``maxins`` caps the instruction count exactly. Returns instructions run."""
    enter(cpu, mem, addr, a, x, y)
    stops = [RET] if stop_pc is None else [RET, stop_pc]
    s0 = cpu.steps
    cpu.halt = state is not None
    if cpu.halt:
        cpu.memory.observe_data_access(
            on_write=cpu.on_buffer_write, writes=range(0xE000, 0x10000)
        )
    try:
        if maxins is None:
            cpu.run_until(max_cycles=MAX_CYCLES, stop_pcs=stops)
        else:
            stop_set = set(stops)
            while cpu.steps - s0 < maxins and cpu.pc not in stop_set and not cpu.stop:
                cpu.step()
    finally:
        cpu.halt = False
        cpu.memory.observe_data_access(None, None)
    return cpu.steps - s0


def cycles(cpu, mem, addr, a=0, x=0, y=0):
    """Run ``addr`` to its return and give its CPU-cycle cost (watches never stop it)."""
    enter(cpu, mem, addr, a, x, y)
    c0 = cpu.processorCycles
    cpu.run_until(max_cycles=MAX_CYCLES, stop_pcs=[RET])
    assert cpu.pc == RET, f"${addr:04X} did not return"
    return cpu.processorCycles - c0


class Trace(NamedTuple):
    """Every instruction of a run: ``pc``/``cycles`` at its start, registers after it,
    ``sp0`` the SP before the first and ``end`` the cycle count after the last."""

    pc: np.ndarray
    cycles: np.ndarray
    a: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sp: np.ndarray
    p: np.ndarray
    sp0: int
    end: int

    @property
    def sp_before(self):
        return np.concatenate(([self.sp0], self.sp[:-1])).astype(np.uint8)

    @property
    def x_before(self):
        return np.concatenate(([0], self.x[:-1])).astype(np.uint8)

    @property
    def cycles_after(self):
        return np.append(self.cycles[1:], np.uint64(self.end))

    def hits(self):
        """Executions per PC, indexed by address."""
        return np.bincount(self.pc, minlength=0x10000)

    def subtree(self, entries):
        """(cycles, entries) of the call-trees entered at any PC in ``entries``."""
        mark = np.zeros(0x10000, np.bool_)
        mark[list(entries)] = True
        cyc = self.cycles.astype(np.int64, copy=False)
        return _subtree(self.pc, cyc, self.sp, self.sp0, int(self.end), mark)


@numba.njit(cache=True)
def _subtree(pc, cyc, sp, sp0, end, mark):
    """A call-tree opens at a marked PC (not already inside one) with the SP it saw,
    and closes after the instruction that lifts SP above that; its cycles run from
    the opening instruction's start to the closing one's end."""
    total = 0
    count = 0
    inside = -1
    c0 = 0
    n = len(pc)
    before = sp0
    for i in range(n):
        if inside < 0 and mark[pc[i]]:
            inside, c0 = before, cyc[i]
            count += 1
        before = sp[i]
        if 0 <= inside < before:
            total += (cyc[i + 1] if i + 1 < n else end) - c0
            inside = -1
    if inside >= 0:
        total += end - c0
    return total, count


def run_traced(cpu, probes=None, stops=(RET,)):
    """Run to any of ``stops`` recording every instruction as a :class:`Trace`.

    ``probes`` maps a PC to ``fn(cpu)``, called as the CPU reaches it, before the
    instruction executes -- for sampling memory the trace cannot reconstruct."""
    probes = probes or {}
    halt = list(stops) + list(probes)
    sp0 = cpu.sp
    cols = [[] for _ in range(7)]
    while True:
        _, tr = cpu.run_until(max_cycles=MAX_CYCLES, stop_pcs=halt, regs=True)
        for col, src in zip(cols, (tr.pc, tr.cycles, tr.a, tr.x, tr.y, tr.sp, tr.p)):
            col.append(np.asarray(src).copy())
        probe = probes.get(cpu.pc)
        if probe is None:
            break
        probe(cpu)
        row = (cpu.pc, cpu.processorCycles)
        cpu.step()
        row += (cpu.a, cpu.x, cpu.y, cpu.sp, cpu.p)
        for col, val, ref in zip(cols, row, cols):
            col.append(np.array([val], dtype=ref[0].dtype))
    arrays = [np.concatenate(c) for c in cols]
    return Trace(*arrays, sp0=sp0, end=cpu.processorCycles)


PLOT_WORLD = 0x2625  # plot_world
FRAME_CYCLES = 19656.0  # PAL frame


def prepare_render_context(cpu, mem, buffer_mode=0):
    """Put the machine in the state ``play_landscape_loop $357D`` leaves before its
    ``$2625``, and run that entry's own prologue in the ROM's order ($35BA-$35C0):
    ``$245B`` occlusion, ``$3700`` projection tables, ``$2993`` buffer variables.
    The raytrace leaves $0035/$0036 as $1ECC/$2597 march state, so $2993 comes last.
    Each returns to the guard (no watch is armed); gives their :class:`Prologue` cycles.
    """
    for addr in (0x001F, 0x005E, 0x0C78, 0x0C1B, 0x0CDE):
        mem[addr] = 0
    mem[0x0CCE] = 0x80  # skip secret-code check in the raytracer
    mem[0x0CDF], mem[0x0C73] = 0, 0  # no tune: $352C takes its 29-cycle play path,
    # which a live $1FFC image confirms (both bytes 0 on every captured ls9795 pass)
    mem[0x0051], mem[0x0052] = 0xF0, 0x30  # play-view raster clip window ($994b/$994d)
    mem[PLAY_DISPLAY_FLAG] = 0x80  # $3577: $283D sends every examine to $37F2
    occlusion = cycles(cpu, mem, 0x245B)
    tables = cycles(cpu, mem, 0x3700)
    cycles(cpu, mem, 0x2993, a=buffer_mode)
    return occlusion, tables


class Prepared(NamedTuple):
    """A board after the $357D prologue, with the CPU cycles of its $245B and $3700."""

    image: bytes
    occlusion_cycles: int
    table_cycles: int


def render_frame_cost(cpu, mem, h_angle, v_angle):
    """Run the real plot_world ($2625) once headless with the raytraced occlusion
    table ($245B) active, from the player at (h_angle, v_angle); return the exact
    frame cost = plot_world CPU-cycle delta / 19656. Mirrors the golden setup."""
    player = mem[0x000B]
    mem[0x006E] = player
    mem[0x09C0 + player] = h_angle & 0xFF  # objects_h_angle
    mem[0x0140 + player] = v_angle & 0xFF  # objects_v_angle
    prepare_render_context(cpu, mem)
    return cycles(cpu, mem, PLOT_WORLD) / FRAME_CYCLES


def prepared_image(src, buffer_mode=0):
    """``src`` (a live sim image) after the $357D render prologue, as a :class:`Prepared`.

    $245B, $3700 and $2993 read no view angle, so one prepare serves every view of
    the board through :func:`plot_world_frames` and every strip through
    :func:`strip_frames`."""
    cpu, mem, _ = machine_from_image(src)
    mem[0x006E] = mem[0x000B]
    occlusion, tables = prepare_render_context(cpu, mem, buffer_mode)
    return Prepared(bytes(mem), occlusion, tables)


def plot_world_frames(prepared, h_angle, v_angle):
    """Exact plot_world frame cost on a :func:`prepared_image` at (h_angle, v_angle)."""
    cpu, mem, _ = wrap_image(prepared.image)
    player = mem[0x000B]
    mem[0x09C0 + player] = h_angle & 0xFF
    mem[0x0140 + player] = v_angle & 0xFF
    return cycles(cpu, mem, PLOT_WORLD) / FRAME_CYCLES


UPDATE_OBJECT_ON_SCREEN = 0x1F9F


REPLOT_LINE = 0x1FA4  # $1FA2 BCS not taken: the strip replot proper, past $209B


def strip_frames(prepared, target, from_pc=REPLOT_LINE):
    """Exact $1F9F frame cost of ``target`` on a :func:`prepared_image`, counted from
    the first arrival at ``from_pc``."""
    cpu, mem, _ = wrap_image(prepared.image)
    return _update_object(cpu, mem, target, from_pc)


def _update_object(cpu, mem, target, from_pc=None, trace=False):
    mem[0x0091] = target
    enter(cpu, mem, UPDATE_OBJECT_ON_SCREEN)
    c0 = cpu.processorCycles
    starts = {}

    def start(c):
        starts.setdefault(from_pc, c.processorCycles)

    probes = None if from_pc is None else {from_pc: start}
    tr = run_traced(cpu, probes) if trace else None
    if not trace:
        stops = [RET] if from_pc is None else [RET, from_pc]
        cpu.run_until(max_cycles=MAX_CYCLES, stop_pcs=stops)
        if cpu.pc != RET:
            start(cpu)
            cpu.run_until(max_cycles=MAX_CYCLES, stop_pcs=[RET])
    assert cpu.pc == RET, "$1F9F did not return"
    frames = (cpu.processorCycles - starts.get(from_pc, c0)) / FRAME_CYCLES
    return (frames, tr) if trace else frames


def update_object_cost(cpu, mem, target, trace=False, from_pc=None):
    """Run the real $1F9F once headless on ``target`` and return its frame cost.

    Same render context as :func:`render_frame_cost`; $1FC2 re-points the camera at
    the object's strip, so the view is the ROM's. ``from_pc`` starts counting there
    (``REPLOT_LINE`` drops $209B). With ``trace``, returns ``(frames, Trace)`` of the
    whole run."""
    mem[0x006E] = mem[0x000B]
    prepare_render_context(cpu, mem)
    return _update_object(cpu, mem, target, from_pc, trace)


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
    mem[PLAY_DISPLAY_FLAG] = 0x80  # $3577, as the ROM leaves it entering $357D
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
