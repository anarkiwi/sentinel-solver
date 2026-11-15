"""The CSDb release on vagrant-moses boots to the title, enters the model's board by
keyboard and plays under the unchanged live driver: frames count the $9630 marker and
keys reach the game's own scan."""

import json

import pytest

from driver import clock, emu as demu, kbd_aim, play_player, sentinel_state as gs
from driver.core import landscape_from_digits
from sentinel import emu
from sentinel.passcost import LINE_CYCLES, PAL_FRAME_CYCLES
from sentinel.prng import Prng

pytestmark = pytest.mark.oracle


def _quiet(_msg):
    pass


def test_boot_runs_the_games_own_setup():
    """$8900 builds the RAM vector page ($3470's JMP $FFF1 lands on $8D81), $3F08 points
    the IRQ at $95E9, and $3F96 seeds the LFSR with ones, discards 256 draws and fills
    $0200 then $02FF downwards."""
    m = emu.boot()
    assert m.mpu.pc == emu.TITLE_SCAN
    ram = bytes(m.ram)
    assert ram[0xFFF1:0xFFF4] == bytes((0x4C, 0x81, 0x8D))
    assert ram[0xFFFE:0x10000] == bytes((0xE9, 0x95))
    rng = Prng([1] * 5)
    for _ in range(256):
        rng.next()
    table = bytearray(0x100)
    for x in range(0x100, 0, -1):
        table[x & 0xFF] = rng.next()
    assert ram[0x0200:0x0300] == table


def test_boot_fits_its_hang_guard():
    assert emu.boot().cycle < demu.cycles(emu.BOOT_TIMEOUT)


def test_boot_and_enter_are_deterministic():
    a, b = emu.boot(), emu.boot()
    assert (a.cycle, bytes(a.ram)) == (b.cycle, bytes(b.ram))
    a, b = emu.enter(42), emu.enter(42)
    assert (a.cycle, a.mpu.pc, bytes(a.ram)) == (b.cycle, b.mpu.pc, bytes(b.ram))
    a.run(a.cycle + PAL_FRAME_CYCLES)
    assert bytes(a.ram) != bytes(b.ram) and bytes(emu.enter(42).ram) == bytes(b.ram)


@pytest.mark.parametrize("number", (0, 42, 1442))
def test_enter_lands_in_play_on_the_models_board(number):
    bm = demu.EmuMon(emu.enter(number))
    assert bm.m.mpu.pc == clock.FRAME_PC
    assert not bm.mem_get(kbd_aim.A_PLOT, kbd_aim.A_PLOT)[0] & 0x80
    landscape = landscape_from_digits(f"{number:04d}")
    matched, total = gs.verify_entry(bm, landscape, _quiet)
    assert matched == total == len(gs.read_game_state(gs.ViceSource(bm)).objects)


def test_frames_count_the_frame_marker():
    bm = demu.EmuMon(emu.enter(0))
    f0, c0 = clock.frames(bm), bm.m.cycle
    assert clock.run_frames(bm, 50) == 50 == clock.frames(bm) - f0
    assert abs(bm.m.cycle - c0 - 50 * PAL_FRAME_CYCLES) < LINE_CYCLES


def test_keys_reach_the_games_scan():
    bm = demu.EmuMon(emu.enter(0))
    kbd = kbd_aim.KbdDriver(bm, _quiet, quantized=True)
    on = kbd.sights_live_on()
    assert kbd.sights_set(not on) and kbd.sights_live_on() is not on
    assert kbd.sights_set(on) and kbd.sights_live_on() is on
    h = kbd.hang()
    assert kbd.coarse_h(h + 8) == "ok" and kbd.hang() == (h + 8) & 0xFF


def test_play_player_runs_the_live_stack_on_the_emulator(tmp_path):
    out = tmp_path / "run.json"
    argv = ["0", "--emu", "--preset", "narrow", "--max-actions", "1", "--out"]
    play_player.main(argv + [str(out)])
    result = json.loads(out.read_text())
    assert result["avi"]["msg"] == "emu" and len(result["trace"]) == 1
    assert result["enemies"] == 1 and result["frames"] == result["trace"][0][0]
    assert result["lookahead_audit"][0][3] == result["lookahead_audit"][0][4]


def test_peeking_the_io_page_leaves_the_machine_alone():
    looked, alone = demu.EmuMon(emu.enter(0)), demu.EmuMon(emu.enter(0))
    for _ in range(20):
        looked.mem_get(0x0000, 0xFFFF)
        looked.mem_get(0xD011, 0xD012, bank="io")
        clock.run_frames(looked, 1)
    clock.run_frames(alone, 20)
    assert looked.m.cycle == alone.m.cycle
    assert looked.mem_get(0, 0xFFFF) == alone.mem_get(0, 0xFFFF)
