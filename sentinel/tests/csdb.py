"""Build the oracle's game image from The Sentinel's CSDb release (id 233853).

The D64 is fetched once into ``out/csdb/`` (SHA-256 pinned) and its PRG booted headless
on jennings' native core until the game's own code runs; that 64 KB image becomes
``out/sentinel_stage2.bin``. Run as ``python -m sentinel.tests.csdb [--force]``.
"""

import argparse
import hashlib
import os
import re
import urllib.request

from d64 import DiskImage
from jennings.devices.mpu6502 import NativeMPU
from jennings.memory import NativeMemory

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


def boot(program, log=print):
    """Boot ``program`` (load address + bytes) to the game's own code; the 64 KB image.

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
        if not mem[0x0001] & 3 or not mem[0x0001] & 4:
            return None
        if addr == 0xD012:
            raster[0] = (raster[0] + 1) & 0xFF
            return raster[0]
        return 0xFF if addr == 0xDC00 or resident() else KEY_SPACE

    def bank(_addr, _value):
        cpu.stop = True

    cpu.memory.subscribe_to_read((0xD012, 0xDC00, 0xDC01), rd)
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
            return mem
        if pc >= 0xE000:
            sp = cpu.sp
            ret = mem[0x0100 + ((sp + 1) & 0xFF)] | mem[0x0100 + ((sp + 2) & 0xFF)] << 8
            cpu.sp, cpu.pc = (sp + 2) & 0xFF, (ret + 1) & 0xFFFF
        else:
            cpu.sp, cpu.pc = 0xF6, _sys_target(mem)
    raise RuntimeError(f"game not entered in {MAX_CYCLES} cycles")


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
