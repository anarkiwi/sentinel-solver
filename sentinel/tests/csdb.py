"""Build the oracle's game image from The Sentinel's CSDb release (id 233853).

The D64 is fetched once into ``out/csdb/`` (SHA-256 pinned) and its PRG RUN on
vagrant-moses (:func:`sentinel.emu.boot`) to the title's keyboard wait; that machine's
RAM becomes ``out/sentinel_stage2.bin``. Run as ``python -m sentinel.tests.csdb``.
"""

import argparse
import hashlib
import os
import urllib.request

from d64 import DiskImage

from sentinel.tests import oracle

RELEASE = "https://csdb.dk/release/?id=233853"
D64_URL = "https://csdb.dk/getinternalfile.php/247149/Sentinel.d64"
D64_SHA256 = "352690551128b45799399b81bbe4038dfeae39bb12b32471790adea968d73314"
CACHE = os.path.join(os.path.dirname(oracle.IMG), "csdb")
D64 = os.path.join(CACHE, "Sentinel.d64")

SIG_ADDR = 0x35A4  # play_landscape: LDA player_object / STA, once the game is resident
SIG_BYTES = bytes((0xA5, 0x0B, 0x85))


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


def build(path=oracle.IMG, log=print):
    """Fetch, boot and write the oracle image to ``path``; returns ``path``."""
    from sentinel import emu

    image = bytes(emu.boot().ram)
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
