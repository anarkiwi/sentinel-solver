#!/usr/bin/env python3
"""Dump the loaded game's 64 KB memory image to ``out/sentinel_stage2.bin``.

Boots ``sentinel-gold.tap`` in asid-vice to the title screen (:mod:`driver.boot`) and
reads RAM halted (:func:`driver.core.live_image`). That image is the ``oracle``-marked
tests' fixture: copyrighted and gitignored, so it is regenerated, never distributed.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import boot  # noqa: E402
from driver.core import live_image  # noqa: E402

IMG = os.path.join(boot.ROOT, "out", "sentinel_stage2.bin")


VERIFY_LANDSCAPE = 0


def verify(path, landscape_number=VERIFY_LANDSCAPE):
    """Tiles matched of 1024: run the ROM's own generator from the image at ``path`` and
    compare against the golden-validated model. The $35A4 load signature only proves code
    is resident; a half-loaded or title-dirtied image still generates a different board.
    """
    from sentinel import landscape  # noqa: PLC0415 - optional, oracle-only dependency
    from sentinel.tests import oracle  # noqa: PLC0415

    oracle.IMG = path
    rom = bytes(oracle.generate(landscape_number)[0x0400:0x0800])
    ref = bytes(landscape.generate(landscape_number).mem[0x0400:0x0800])
    return sum(1 for a, b in zip(rom, ref) if a == b)


def dump(path=IMG, log=print):
    """Boot the tape to the title screen and write the loaded 64 KB image to ``path``."""
    container, bm = boot.boot_loaded(log=log)
    try:
        img = live_image(bm)
    finally:
        try:
            container.stop()
        except Exception:  # pragma: no cover - teardown is best-effort
            pass
    if len(img) != 0x10000:
        raise RuntimeError(f"short image: {len(img)} bytes")
    sig = bytes(img[boot.SIG_ADDR : boot.SIG_ADDR + len(boot.SIG_BYTES)])
    if sig != boot.SIG_BYTES:
        raise RuntimeError(
            f"game not resident: ${boot.SIG_ADDR:04x} = {sig.hex()}, "
            f"want {boot.SIG_BYTES.hex()}"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".candidate"
    with open(tmp, "wb") as f:
        f.write(img)
    matched = verify(tmp)
    if matched != 1024:
        raise RuntimeError(
            f"image rejected: ROM generation matches the model on {matched}/1024 tiles. "
            f"Candidate left at {tmp} for inspection."
        )
    os.replace(tmp, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=IMG)
    parser.add_argument("--force", action="store_true", help="re-dump if present")
    args = parser.parse_args(argv)
    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} present; --force to regenerate")
        return 0
    print(f"wrote {dump(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
