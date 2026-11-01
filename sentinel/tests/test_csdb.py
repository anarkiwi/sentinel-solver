"""The CSDb release boots headless to a game image whose own generator builds the
model's board, and the download cache refuses a file that is not the pinned release."""

import os

import pytest

from sentinel import landscape
from sentinel.tests import csdb, oracle

needs_d64 = pytest.mark.skipif(
    not os.path.exists(csdb.D64), reason="CSDb release not cached"
)


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


@needs_d64
def test_the_booted_release_generates_the_models_board(tmp_path, monkeypatch):
    image = csdb.boot(csdb.prg(), log=lambda _msg: None)
    assert image[csdb.SIG_ADDR : csdb.SIG_ADDR + 3] == csdb.SIG_BYTES
    path = tmp_path / "stage2.bin"
    path.write_bytes(image)
    monkeypatch.setattr(oracle, "IMG", str(path))
    for number in (0, 0x0042, 0x0335):
        rom = bytes(oracle.generate(number)[0x0400:0x0800])
        assert rom == bytes(landscape.generate(number).mem[0x0400:0x0800]), number
