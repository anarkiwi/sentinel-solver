#!/usr/bin/env python3
"""The seed's own error: --follow must start the sim at the machine's exact cycle."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import instrument  # noqa: E402
from sentinel import enemies  # noqa: E402

# a $9630 marker inside the $1CDD march: resumable only to the $17B2 scan's head
INEXACT = ((enemies.PHASE_BODY, enemies.BODY_SCAN, 61, -1, None), 0x1D16, 0x17B7)
# one on the $1289 straight line: the cycles already spent are counted
EXACT = ((enemies.PHASE_HEAD, enemies.BODY_ENTRY, 0, -1, 25), 0x1294, 0x1294)


class StubEmu:
    """An EmuClock over a scripted sequence of $9630 positions."""

    def __init__(self, positions):
        self.positions = list(positions)
        self.frame = 0

    def full_image(self):
        return bytearray(0x10000)

    def position(self, _image):
        return self.positions[min(self.frame, len(self.positions) - 1)]

    def step_frame(self):
        self.frame += 1


def test_the_seed_waits_for_a_marker_whose_cycle_it_can_count():
    emu = StubEmu([INEXACT, INEXACT, EXACT])
    _image, seed = instrument._seed(emu)
    assert seed.exact and seed.waited == 2 and emu.frame == 2
    assert seed.resume[4] == 25 and "+/-0 cycles" in seed.note()
    assert seed.caught == (0x1D16, 0x17B7)  # where the frame that asked for it was


def test_an_uncountable_run_reports_its_uncertainty_rather_than_absorbing_it():
    emu = StubEmu([INEXACT])
    _image, seed = instrument._seed(emu, tries=3)
    assert not seed.exact and seed.waited == 3
    note = seed.note()
    assert "unbounded" in note and "$17B7" in note and "$1D16" in note


def test_the_control_seeds_wherever_the_marker_falls():
    emu = StubEmu([INEXACT, EXACT])
    _image, seed = instrument._seed(emu, exact=False)
    assert not seed.exact and seed.waited == 0 and emu.frame == 0


def test_stack_position_names_the_pc_the_raster_interrupted():
    """$95E9 pushes Y, X, A, P, PCL, PCH; SP+5/6 is the interrupted PC."""
    mem = bytearray(0x10000)
    page = bytearray(0x100)
    sp = 0xF0
    page[sp + 5], page[sp + 6] = 0x94, 0x12  # interrupted at $1294, on the pass head
    resume, pc, addr = enemies.stack_position(mem, sp, page)
    assert (pc, addr) == (0x1294, 0x1294)
    assert resume == enemies.resume_from_stack(mem, sp, page)
    assert resume[0] == enemies.PHASE_HEAD and resume[4] == enemies._HEAD_SPENT[0x1294]
