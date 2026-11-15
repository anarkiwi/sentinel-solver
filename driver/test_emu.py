"""EmuMon keeps VICE's monitor semantics on a vagrant-moses machine: stops land before
the target executes, steps owe their boundary's checkpoint test to the next resume,
budgets expire like VICE waits, and the banks read as VICE's do. A bare machine runs a
few hand-assembled instructions, so no game image is needed."""

import pytest
from vagrant_moses.machine import Machine
from vice_driver.binmon import TAP_MODE_FIXED, BinmonError

from driver import emu
from sentinel.passcost import PAL_FRAME_CYCLES

LOOP, INC, JMP = 0x1000, 0x1000, 0x1002  # INC $20 / JMP $1000
COUNTER = 0x20


@pytest.fixture(name="bm")
def _bm():
    m = Machine()
    m.ram[LOOP : LOOP + 5] = bytes((0xE6, COUNTER, 0x4C, LOOP & 0xFF, LOOP >> 8))
    m.reset(LOOP)
    bm = emu.EmuMon(m)
    bm.run_until_pc(LOOP, timeout=0.01)
    return bm


def test_run_until_pc_stops_before_the_target_executes(bm):
    assert bm.m.ram[COUNTER] == 0 and bm.m.mpu.pc == INC
    bm.run_until_pc(JMP, timeout=0.01)
    assert (bm.m.mpu.pc, bm.m.ram[COUNTER]) == (JMP, 1)
    cycle = bm.m.cycle
    bm.run_until_pc(JMP, timeout=0.01)
    assert bm.m.cycle == cycle  # already there: nothing runs
    bm.run_until_pc(INC, timeout=0.01)
    assert (bm.m.mpu.pc, bm.m.ram[COUNTER]) == (INC, 1)


def test_advance_executes_exactly_n_instructions(bm):
    bm.advance_instructions(3)
    assert (bm.m.mpu.pc, bm.m.ram[COUNTER]) == (JMP, 2)
    with pytest.raises(NotImplementedError):
        bm.advance_instructions(1, step_over_subroutines=True)


def test_silent_checkpoint_counts_every_execution(bm):
    cp = bm.checkpoint_set(INC, stop_when_hit=False, silent=True).checknum
    for _ in range(5):
        bm.run_until_pc(JMP, timeout=0.01)
        bm.run_until_pc(INC, timeout=0.01)
    assert bm.checkpoint_get(cp).hit_count == 5 and bm.m.ram[COUNTER] == 5
    bm.checkpoint_toggle(cp, False)
    bm.run_until_pc(JMP, timeout=0.01)
    assert bm.checkpoint_get(cp).hit_count == 5


def test_a_step_owes_its_boundary_test_to_the_next_resume(bm):
    """VICE tests the step count before the checkpoints, so a step that ends on a
    checkpointed pc counts the hit only when the CPU resumes; a stop checkpoint there
    then stops at once, nothing executed."""
    count = bm.checkpoint_set(JMP, silent=True).checknum
    bm.advance_instructions(1)
    assert bm.m.mpu.pc == JMP and bm.checkpoint_get(count).hit_count == 0
    stop = bm.checkpoint_set(JMP, stop_when_hit=True).checknum
    cycle = bm.m.cycle
    bm.wait_for_checkpoint(stop, timeout=0.01)
    assert bm.m.cycle == cycle and bm.checkpoint_get(count).hit_count == 1
    bm.advance_instructions(1)
    bm.wait_for_checkpoint(stop, timeout=0.01)
    assert bm.checkpoint_get(count).hit_count == 2 and bm.m.ram[COUNTER] == 2


def test_a_wait_stopped_by_another_checkpoint_raises(bm):
    other = bm.checkpoint_set(JMP, stop_when_hit=True).checknum
    with pytest.raises(BinmonError):
        bm.run_until_pc(0x2000, timeout=0.01)
    assert bm.m.mpu.pc == JMP
    bm.checkpoint_toggle(other, False)
    assert bm.m.watch == frozenset()


def test_an_unreached_pc_spends_the_budget_and_raises(bm):
    cycle = bm.m.cycle
    with pytest.raises(BinmonError, match="timed out"):
        bm.run_until_pc(0x2000, timeout=0.001)
    assert bm.m.cycle - cycle == emu.cycles(0.001)


def test_a_jam_raises(bm):
    bm.m.ram[JMP] = 0x02
    with pytest.raises(BinmonError, match="JAM"):
        bm.run_until_pc(0x2000, timeout=0.01)


def test_unsupported_monitor_calls_refuse():
    bm = emu.EmuMon(Machine())
    for call in (
        lambda: bm.run_until_pc(LOOP, 1.0, condition="A == $00"),
        lambda: bm.checkpoint_set(LOOP, temporary=True),
        lambda: bm.keymatrix_tap([(0, 1)], mode=0),
    ):
        with pytest.raises(NotImplementedError):
            call()
    with pytest.raises(BinmonError):
        bm.mem_get(1, 0)
    with pytest.raises(BinmonError):
        bm.mem_get(0, 1, bank="cart")


def test_banks_read_as_vices(bm):
    m = bm.m
    m.ram[0xE000] = 0x5A
    kernal = bm.mem_get(0xE000, 0xE000)[0]
    assert kernal != 0x5A and bm.mem_get(0xE000, 0xE000, bank="ram") == b"\x5a"
    bm.mem_set(0xD020, b"\x07")
    assert m.vic.regs[0x20] == 0x07 and bm.mem_get(0xD020, 0xD020)[0] & 0x0F == 7
    ctrl, line = bm.mem_get(0xD011, 0xD012, bank="io")
    assert ((ctrl & 0x80) << 1 | line) == m.vic.raster_y
    assert bm.mem_get(0xD011, 0xD012) == bytes((ctrl, line))
    image = bm.mem_get(0x0000, 0xFFFF)
    assert len(image) == 0x10000 and image[LOOP : LOOP + 2] == bytes((0xE6, COUNTER))
    bm.mem_set(0xE000, b"\x11")
    assert m.ram[0xE000] == 0x11  # a write under a ROM lands in RAM


def test_registers_report_the_stopped_instruction(bm):
    regs = bm.registers_get()
    assert regs[3] == INC and regs[53] == bm.m.vic.raster_y


def test_keymatrix_set_and_release(bm):
    bm.keymatrix_set([(1, 2, 1), (7, 4, 1), (-3, 0, 1)])
    assert bm.m.keymatrix[1] == 0x04 and bm.m.keymatrix[7] == 0x10
    bm.keymatrix_set([(1, 2, 0)])
    assert bm.m.keymatrix[1] == 0
    bm.keymatrix_release_all()
    assert bm.m.keymatrix == bytes(8)


def test_a_fixed_tap_releases_after_its_frames(bm):
    bm.keymatrix_tap([(7, 4)], mode=TAP_MODE_FIXED, frames=2)
    release = bm._tap[0]
    assert release == bm.m.cycle - 1 + 2 * PAL_FRAME_CYCLES
    bm.keymatrix_tap([(1, 2)], mode=TAP_MODE_FIXED, frames=2)
    assert bm.m.keymatrix[7] == 0 and bm.m.keymatrix[1] == 0x04
    with pytest.raises(BinmonError):
        bm.run_until_pc(0x2000, timeout=release / 2 / emu.vmm.PAL.clock_hz)
    assert bm.m.keymatrix == bytes(8) and bm._tap is None


def test_lifecycle_calls_are_noops(bm):
    bm.connect()
    bm.exit()
    bm.close()
    bm.auto_resume = True
    with bm.halted():
        assert bm.auto_resume is False
    assert bm.auto_resume is True


def test_a_clone_runs_identically_and_independently(bm):
    """A clone carries the checkpoints and their counts, then diverges only by what
    is done to it: the original does not see the clone's run."""
    cp = bm.checkpoint_set(INC, silent=True).checknum
    bm.run_until_pc(JMP, timeout=0.01)
    twin = bm.clone()
    for mon in (bm, twin):
        for _ in range(3):
            mon.run_until_pc(INC, timeout=0.01)
            mon.run_until_pc(JMP, timeout=0.01)
    assert twin.m.cycle == bm.m.cycle and twin.m.ram[COUNTER] == bm.m.ram[COUNTER]
    assert twin.checkpoint_get(cp).hit_count == bm.checkpoint_get(cp).hit_count == 3
    twin.run_until_pc(INC, timeout=0.01)
    twin.checkpoint_toggle(cp, False)
    assert twin.m.cycle > bm.m.cycle and bm.checkpoint_get(cp).enabled
    assert bm.checkpoint_get(cp).hit_count == 3
