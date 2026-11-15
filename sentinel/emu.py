"""The Sentinel's CSDb release running on vagrant-moses, a cycle-exact C64.

:func:`boot` RUNs the release on a PAL machine with the real ROMs to the title's
keyboard wait (its RAM becomes the oracle image when absent); :func:`enter` types a
landscape with :func:`driver.core.navigate` as the live VICE run does. Both return
clones of a cached template.
"""

import functools
import os

from vagrant_moses import roms
from vagrant_moses.machine import Machine

from sentinel.tests import csdb, oracle

ROMS = ("kernal", "basic", "chargen")
READY = 0xA480  # BASIC's main loop: READY. printed, a line about to be input
GAME_ENTRY = 0x3F00  # the loader's jump into the resident game
TITLE_SCAN = 0x8D17  # the title's first keyboard read
SPACE = (7, 4)  # the release's intro waits for SPACE before decrunching the game
KEYBOARD_BUFFER, KEYBOARD_COUNT = 0x0277, 0x00C6
BOOT_TIMEOUT = 60.0  # hang guard on each boot leg, in PAL seconds


def _quiet(_msg):
    pass


@functools.cache
def _title():
    from driver.emu import EmuMon

    csdb.fetch(log=_quiet)
    m = Machine(roms=roms.load_roms(ROMS))
    m.reset(m[0xFFFC] | m[0xFFFD] << 8)
    bm = EmuMon(m)
    bm.run_until_pc(READY, timeout=BOOT_TIMEOUT)
    program = csdb.prg()
    load = program[0] | program[1] << 8
    end = load + len(program) - 2
    m.ram[load:end] = program[2:]
    for zp in (0x2D, 0x2F, 0x31, 0xAE):  # VARTAB/ARYTAB/STREND, LOAD's end address
        m.ram[zp], m.ram[zp + 1] = end & 0xFF, end >> 8
    run = b"RUN\r"
    m.ram[KEYBOARD_BUFFER : KEYBOARD_BUFFER + len(run)] = run
    m.ram[KEYBOARD_COUNT] = len(run)
    m.set_key(*SPACE, True)
    bm.run_until_pc(GAME_ENTRY, timeout=BOOT_TIMEOUT)
    m.release_all()
    bm.run_until_pc(TITLE_SCAN, timeout=BOOT_TIMEOUT)
    if bytes(m.ram[csdb.SIG_ADDR : csdb.SIG_ADDR + 3]) != csdb.SIG_BYTES:
        raise RuntimeError("title reached without the game resident")
    if not os.path.exists(oracle.IMG):
        csdb.write(bytes(m.ram), log=_quiet)
    return m


def boot():
    """A machine at the title's keyboard wait, about to execute ``TITLE_SCAN``."""
    return _title().clone()


@functools.cache
def _in_play(number):
    from driver import core, sentinel_state as gs
    from driver.emu import EmuMon

    m = boot()
    bm = EmuMon(m)
    digits = f"{number:04d}"
    core.navigate(bm, digits, log=_quiet)
    matched, total = gs.verify_entry(bm, core.landscape_from_digits(digits), _quiet)
    live = len(gs.read_game_state(gs.ViceSource(bm)).objects)
    if not matched == total == live:
        raise RuntimeError(f"entry {number}: {matched}/{total} objects, live {live}")
    return m


def enter(number):
    """A machine in play on the landscape a player types as ``number``, its board
    verified object for object against the model's generator."""
    return _in_play(number).clone()
