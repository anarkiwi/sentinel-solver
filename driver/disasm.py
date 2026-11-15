#!/usr/bin/env python3
"""Disassemble ranges of the game's in-play image (jennings' NMOS decoder).

``python -m driver.disasm 1214 1236 127c 1310 [--landscape N]``: each pair of hex
addresses is a range, listed as the CPU sees memory once landscape ``N`` is in play.
"""

import argparse

from jennings.devices.mpu6502 import MPU
from jennings.disassembler import Disassembler

from sentinel import emu


def listing(ram, start, end):
    """``[(address, text)]`` of the instructions from ``start`` up to ``end``."""
    mpu = MPU()
    mpu.memory[: len(ram)] = list(ram)
    dis = Disassembler(mpu)
    out, pc = [], start
    while pc < end:
        n, text = dis.instruction_at(pc)
        out.append((pc, text))
        pc += n
    return out


def main(argv=None):
    """Print each range given as hex address pairs."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bounds", nargs="+", type=lambda s: int(s, 16))
    ap.add_argument("--landscape", type=int, default=0)
    args = ap.parse_args(argv)
    if len(args.bounds) % 2:
        ap.error("addresses come in start/end pairs")
    ram = bytes(emu.enter(args.landscape).ram)
    pairs = zip(args.bounds[::2], args.bounds[1::2])
    for start, end in pairs:
        for pc, text in listing(ram, start, end):
            print(f"{pc:04X}  {text}")
        print()


if __name__ == "__main__":
    main()
