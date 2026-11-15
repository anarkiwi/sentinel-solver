"""The disassembly tool lists the in-play image as the CPU decodes it."""

from driver import disasm
from sentinel import emu


def test_a_listing_decodes_the_rom_and_steps_by_instruction():
    ram = bytes(emu.enter(0).ram)
    rows = disasm.listing(ram, 0x1214, 0x1223)
    assert rows[0] == (0x1214, "LDA #$80")
    assert rows[1] == (0x1216, "STA $0ce4")
    assert rows[-1] == (0x1222, "RTS")


def test_main_prints_each_range(capsys):
    disasm.main(["1b1f", "1b28", "--landscape", "0"])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "1B1F  JSR $2156" and out[-2] == "1B27  SEC"
