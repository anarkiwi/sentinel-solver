"""Build the oracle's game image from The Sentinel's CSDb release (id 233853).

The D64 is fetched once into ``out/csdb/`` (SHA-256 pinned) and its PRG booted headless
on jennings' native core in two stages: :func:`decrunch` runs the release's loader until
the game's own code is entered, then :func:`title` runs the game itself -- its `$8900`
machine setup (the RAM vector page under the KERNAL), its `$3F96` random table and its
title screen, under a PAL raster interrupt -- to the title's keyboard wait. That image
becomes ``out/sentinel_stage2.bin``. Run as ``python -m sentinel.tests.csdb``.
"""

import argparse
import hashlib
import os
import re
import urllib.request
from typing import NamedTuple

from d64 import DiskImage
from jennings.devices.mpu6502 import NativeMPU
from jennings.memory import NativeMemory

from sentinel import badline
from sentinel.tests import oracle

RELEASE = "https://csdb.dk/release/?id=233853"
D64_URL = "https://csdb.dk/getinternalfile.php/247149/Sentinel.d64"
D64_SHA256 = "352690551128b45799399b81bbe4038dfeae39bb12b32471790adea968d73314"
CACHE = os.path.join(os.path.dirname(oracle.IMG), "csdb")
D64 = os.path.join(CACHE, "Sentinel.d64")

SIG_ADDR = 0x35A4  # play_landscape: LDA player_object / STA, once the game is resident
SIG_BYTES = bytes((0xA5, 0x0B, 0x85))
BASIC_START = 0x0801
KEY_SPACE = 0xEF  # $DC01 with SPACE down: the intro's $1138 wait
SLICE_CYCLES = 1_000_000
MAX_CYCLES = 200_000_000
FRAME_LINES = badline.PAL_FRAME_CYCLES // badline.LINE_CYCLES
VIC_CONTROL_1, VIC_RASTER, VIC_IRQ, VIC_IRQ_ENABLE = 0xD011, 0xD012, 0xD019, 0xD01A
CIA1_PRA, CIA1_PRB = 0xDC00, 0xDC01  # joystick 2 / keyboard columns, keyboard rows


class Entry(NamedTuple):
    """The machine as the loader hands it to the game: RAM and the CPU registers."""

    mem: bytearray
    pc: int
    a: int
    x: int
    y: int
    p: int
    sp: int


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url=D64_URL, path=D64, sha256=D64_SHA256, log=print):
    """The release's D64 bytes, downloaded once into the cache and checked."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "sentinel-solver"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if _sha256(data) != sha256:
            raise RuntimeError(f"{url}: sha256 {_sha256(data)}, want {sha256}")
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    with open(path, "rb") as f:
        data = f.read()
    if _sha256(data) != sha256:
        raise RuntimeError(f"{path}: sha256 mismatch; delete it to re-download")
    return data


def prg(path=D64):
    """The disk's one PRG file, load address first."""
    with DiskImage(path) as image:
        files = [e for e in image.iterdir() if e.entry.file_type == "PRG"]
        if len(files) != 1:
            raise RuntimeError(f"{path}: {len(files)} PRG files, want 1")
        with files[0].open() as f:
            return f.read()


def _sys_target(mem):
    """The ``SYS`` address of the BASIC stub at $0801."""
    m = re.search(rb"\x9e\s*([0-9]+)", bytes(mem[BASIC_START : BASIC_START + 32]))
    if m is None:
        raise RuntimeError("no SYS in the BASIC program at $0801")
    return int(m.group(1))


def _io(port):
    """The 6510 port at ``$01`` banks the I/O page in: CHAREN with LORAM or HIRAM."""
    return bool(port & 3) and bool(port & 4)


def raster_due(mem, now):
    """The first cycle at or after ``now`` that starts the raster line latched in
    ``$D011``/``$D012`` (written to RAM), or None for a line the PAL beam never reaches.
    """
    line = mem[VIC_RASTER] | (mem[VIC_CONTROL_1] & 0x80) << 1
    if line >= FRAME_LINES:
        return None
    start = now + -now % badline.LINE_CYCLES
    lines = (line - start // badline.LINE_CYCLES) % FRAME_LINES
    return start + lines * badline.LINE_CYCLES


def decrunch(program, log=print):
    """Run the loader of ``program`` (load address + bytes) to the game's own code.

    KERNAL calls return at once, a jump into BASIC is the CLR + RUN of the program
    then at $0801, and I/O reads honour the $01 banking the decrunchers rely on."""
    load = program[0] | program[1] << 8
    end = load + len(program) - 2
    mem = bytearray(0x10000)
    mem[load:end] = program[2:]
    mem[0x0001] = 0x37
    mem[0x002B], mem[0x002C] = BASIC_START & 0xFF, BASIC_START >> 8
    for zp in (0x2D, 0x2F, 0x31, 0xAE):  # VARTAB/ARYTAB/STREND, LOAD's end address
        mem[zp], mem[zp + 1] = end & 0xFF, end >> 8
    mem[0x0033], mem[0x0034] = mem[0x0037], mem[0x0038] = 0x00, 0xA0
    cpu = NativeMPU(memory=NativeMemory(mem, share=True))
    cpu.pc, cpu.sp = _sys_target(mem), 0xF6
    raster = [0]

    def resident():
        return mem[SIG_ADDR : SIG_ADDR + len(SIG_BYTES)] == SIG_BYTES

    def rd(addr):
        if not _io(mem[0x0001]):
            return None
        if addr == VIC_RASTER:
            raster[0] = (raster[0] + 1) & 0xFF
            return raster[0]
        return 0xFF if addr == CIA1_PRA or resident() else KEY_SPACE

    def bank(_addr, _value):
        cpu.stop = True

    cpu.memory.subscribe_to_read((VIC_RASTER, CIA1_PRA, CIA1_PRB), rd)
    cpu.memory.subscribe_to_write((0x0001,), bank)
    game = [(a, b + 1) for a, b in oracle.LOADED if b - a > 0x1000]
    while cpu.processorCycles < MAX_CYCLES:
        port = mem[0x0001]
        roms = ([(0xE000, 0x10000)] if port & 2 else []) + (
            [(0xA000, 0xC000)] if port & 3 == 3 else []
        )
        stops = roms + (game if resident() else [])
        cpu.stop = False
        reason = cpu.run_until(
            max_cycles=SLICE_CYCLES, stop_ranges=stops, stop_on_jam=True
        )
        pc = cpu.pc
        if reason == "jam":
            raise RuntimeError(f"JAM at ${pc:04X}")
        if reason != "pc" or not any(a <= pc < b for a, b in stops):
            continue
        if resident() and any(a <= pc < b for a, b in game):
            log(f"game entered at ${pc:04X} after {cpu.processorCycles} cycles")
            return Entry(mem, pc, cpu.a, cpu.x, cpu.y, cpu.p, cpu.sp)
        if pc >= 0xE000:
            sp = cpu.sp
            ret = mem[0x0100 + ((sp + 1) & 0xFF)] | mem[0x0100 + ((sp + 2) & 0xFF)] << 8
            cpu.sp, cpu.pc = (sp + 2) & 0xFF, (ret + 1) & 0xFFFF
        else:
            cpu.sp, cpu.pc = 0xF6, _sys_target(mem)
    raise RuntimeError(f"game not entered in {MAX_CYCLES} cycles")


def title(entry, log=print):
    """Run the game from ``entry`` to its first keyboard scan: ``(image, cycles, IRQs)``.

    The I/O page follows the ``$01`` banking with the beam at the cycle count, and the
    VIC raises its raster interrupt as the PAL machine does: the line latched in
    ``$D011``/``$D012`` sets ``$D019`` bit 0, which with ``$D01A`` bit 0 holds the IRQ
    line until the handler acknowledges it. No key or joystick is ever down."""
    mem = bytearray(entry.mem)
    memory = NativeMemory(mem, share=True)
    cpu = NativeMPU(memory=memory)
    cpu.pc, cpu.a, cpu.x, cpu.y, cpu.p, cpu.sp = entry[1:]
    memory.raster_model()
    memory.io = _io(mem[0x0001])
    latch, scan, irqs = [0], [], 0

    def rd(addr):
        if not memory.io:
            return None
        if addr == VIC_IRQ:
            return 0x70 | latch[0] | (0x80 if latch[0] & mem[VIC_IRQ_ENABLE] else 0)
        if addr == CIA1_PRB:
            scan.append(cpu.pc)
            cpu.stop = True
        return 0xFF

    def wr(addr, value):
        if addr == 0x0001:
            memory.io = _io(value)
        elif memory.io and addr == VIC_IRQ:
            latch[0] &= ~value
        elif memory.io:
            cpu.stop = True  # the raster latch moved

    memory.subscribe_to_read((VIC_IRQ, CIA1_PRA, CIA1_PRB), rd)
    memory.subscribe_to_write((0x0001, VIC_CONTROL_1, VIC_RASTER, VIC_IRQ), wr)
    due = raster_due(mem, 0)
    while not scan:
        now = cpu.processorCycles
        if now > MAX_CYCLES:
            raise RuntimeError(f"no keyboard scan in {MAX_CYCLES} cycles")
        if due is not None and now >= due:
            latch[0] |= 1
        due = raster_due(mem, now + (due == now))
        asserted = latch[0] & mem[VIC_IRQ_ENABLE] & 1
        if asserted and not cpu.p & cpu.INTERRUPT:
            cpu.irq()
            irqs += 1
            continue
        cpu.stop = False
        reason = cpu.run_until(
            max_cycles=SLICE_CYCLES if due is None else due - now,
            stop_on_jam=True,
            stop_on_irq=bool(asserted),
        )
        if reason == "jam":
            raise RuntimeError(f"JAM at ${cpu.pc:04X}")
    log(f"keyboard scanned at ${scan[0]:04X} after {cpu.processorCycles} cycles")
    return mem, cpu.processorCycles, irqs


def boot(program, log=print):
    """Boot ``program`` (load address + bytes) to the title's keyboard wait."""
    return title(decrunch(program, log=log), log=log)[0]


def build(path=oracle.IMG, log=print):
    """Fetch, boot and write the oracle image to ``path``; returns ``path``."""
    fetch(log=log)
    image = boot(prg(), log=log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(image)
    os.replace(path + ".part", path)
    log(f"wrote {path} (from {RELEASE})")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=oracle.IMG)
    parser.add_argument("--force", action="store_true", help="rebuild if present")
    args = parser.parse_args(argv)
    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} present; --force to rebuild")
        return 0
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
