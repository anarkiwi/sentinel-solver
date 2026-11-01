"""The CSDb release boots headless to a game image whose own generator builds the
model's board, and the download cache refuses a file that is not the pinned release."""

import io
import os

import pytest

from sentinel import badline, landscape
from sentinel.prng import Prng
from sentinel.tests import csdb, oracle

needs_d64 = pytest.mark.skipif(
    not os.path.exists(csdb.D64), reason="CSDb release not cached"
)
URL = "https://csdb.invalid/x.d64"


def _quiet(_msg):
    pass


@pytest.fixture(name="entry", scope="module")
def _entry():
    return csdb.decrunch(csdb.prg(), log=_quiet)


def test_sys_target_reads_the_basic_stub():
    mem = bytearray(0x10000)
    stub = b"\x1c\x08\xbb\x0b\x9e2075 HOK 1988\x00\x00\x00"
    mem[0x0801 : 0x0801 + len(stub)] = stub
    assert csdb._sys_target(mem) == 2075
    with pytest.raises(RuntimeError):
        csdb._sys_target(bytearray(0x10000))


def test_fetch_checks_the_cached_file(tmp_path):
    path = tmp_path / "x.d64"
    path.write_bytes(b"release")
    assert csdb.fetch(path=str(path), sha256=csdb._sha256(b"release")) == b"release"
    with pytest.raises(RuntimeError):
        csdb.fetch(path=str(path), sha256=csdb._sha256(b"other"))


def _latched(line):
    mem = bytearray(0x10000)
    mem[csdb.VIC_RASTER], mem[csdb.VIC_CONTROL_1] = line & 0xFF, (line >> 8) << 7
    return mem


def test_the_raster_interrupt_falls_on_the_latched_lines_start():
    line, frame = badline.LINE_CYCLES, badline.PAL_FRAME_CYCLES
    mem = _latched(badline.RASTER_IRQ_LINE)
    due = badline.RASTER_IRQ_LINE * line
    assert csdb.raster_due(mem, 0) == csdb.raster_due(mem, due) == due
    assert csdb.raster_due(mem, due + 1) == due + frame
    assert csdb.raster_due(_latched(0x137), 5 * frame) == 5 * frame + 0x137 * line
    assert csdb.raster_due(_latched(0), frame - 1) == frame
    assert csdb.raster_due(_latched(csdb.FRAME_LINES), 0) is None


@needs_d64
def test_the_title_stage_runs_the_games_own_setup(entry):
    """$8900 builds the RAM vector page ($3470's JMP $FFF1 lands on $8D81), $3F08 points
    the IRQ at $95E9, and $3F96 seeds the LFSR with ones, discards 256 draws and fills
    $0200 then $02FF downwards; one raster IRQ is taken per PAL frame on the way."""
    image, cycles, irqs = csdb.title(entry, log=_quiet)
    assert image[0xFFF1:0xFFF4] == bytes((0x4C, 0x81, 0x8D))
    assert image[0xFFFE:0x10000] == bytes((0xE9, 0x95))
    rng = Prng([1] * 5)
    for _ in range(256):
        rng.next()
    table = bytearray(0x100)
    for x in range(0x100, 0, -1):
        table[x & 0xFF] = rng.next()
    assert image[0x0200:0x0300] == table
    assert abs(irqs - cycles / badline.PAL_FRAME_CYCLES) <= 1


@needs_d64
def test_the_title_image_is_independent_of_the_raster_phase(entry):
    """The loader leaves the raster latch undefined; every LOADED byte the oracle reads
    comes out the same whichever line the first interrupts fall on."""

    def loaded(line):
        mem = bytearray(entry.mem)
        mem[csdb.VIC_RASTER] = line & 0xFF
        mem[csdb.VIC_CONTROL_1] = mem[csdb.VIC_CONTROL_1] & 0x7F | (line >> 8) << 7
        image = csdb.title(entry._replace(mem=mem), log=_quiet)[0]
        return [bytes(image[a : b + 1]) for a, b in oracle.LOADED + [(0xE000, 0xFFFF)]]

    assert loaded(0) == loaded(badline.RASTER_IRQ_LINE) == loaded(0x137)


@needs_d64
def test_the_booted_release_generates_the_models_board(tmp_path, monkeypatch):
    image = csdb.boot(csdb.prg(), log=_quiet)
    assert image[csdb.SIG_ADDR : csdb.SIG_ADDR + 3] == csdb.SIG_BYTES
    path = tmp_path / "stage2.bin"
    path.write_bytes(image)
    monkeypatch.setattr(oracle, "IMG", str(path))
    for number in (0, 0x0042, 0x0335):
        rom = bytes(oracle.generate(number)[0x0400:0x0800])
        assert rom == bytes(landscape.generate(number).mem[0x0400:0x0800]), number


def test_fetch_downloads_once_and_refuses_a_wrong_file(tmp_path, monkeypatch):
    served = []

    def urlopen(req, timeout):
        served.append(req.full_url)
        return io.BytesIO(b"release")

    monkeypatch.setattr(csdb.urllib.request, "urlopen", urlopen)
    path = str(tmp_path / "cache" / "x.d64")
    want = csdb._sha256(b"release")
    for _ in range(2):
        assert csdb.fetch(URL, path, want, log=_quiet) == b"release"
    assert served == [URL]
    with pytest.raises(RuntimeError):
        csdb.fetch(URL, str(tmp_path / "y.d64"), csdb._sha256(b"other"), log=_quiet)
    assert not os.path.exists(tmp_path / "y.d64")


@needs_d64
def test_main_builds_the_image_once(tmp_path, capsys):
    path = tmp_path / "stage2.bin"
    assert csdb.main(["--out", str(path)]) == 0
    built = path.read_bytes()
    assert built[csdb.SIG_ADDR : csdb.SIG_ADDR + 3] == csdb.SIG_BYTES
    path.write_bytes(b"kept")
    assert csdb.main(["--out", str(path)]) == 0
    assert path.read_bytes() == b"kept" and "present" in capsys.readouterr().out
    assert csdb.main(["--out", str(path), "--force"]) == 0
    assert path.read_bytes() == built
