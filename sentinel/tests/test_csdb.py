"""The CSDb release boots on vagrant-moses to a game image whose own generator builds
the model's board, and the download cache refuses a file that is not the pinned
release."""

import io
import os

import pytest

from sentinel import emu, landscape
from sentinel.tests import csdb, oracle

needs_d64 = pytest.mark.skipif(
    not os.path.exists(csdb.D64), reason="CSDb release not cached"
)
URL = "https://csdb.invalid/x.d64"


def _quiet(_msg):
    pass


def test_fetch_checks_the_cached_file(tmp_path):
    path = tmp_path / "x.d64"
    path.write_bytes(b"release")
    assert csdb.fetch(path=str(path), sha256=csdb._sha256(b"release")) == b"release"
    with pytest.raises(RuntimeError):
        csdb.fetch(path=str(path), sha256=csdb._sha256(b"other"))


@needs_d64
def test_the_booted_release_generates_the_models_board(tmp_path, monkeypatch):
    image = bytes(emu.boot().ram)
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
