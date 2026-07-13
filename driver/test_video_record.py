#!/usr/bin/env python3
"""Standalone test: prove asid-vice can produce a real, playable video file.

Boots anarkiwi/asid-vice:latest with ViceContainer (warp, silent), connects BinMon,
starts native VICE video recording (ZMBV-in-AVI via the new VIDEO_RECORD
binmon opcode 0x79), lets the default C64 boot animation run for a few
seconds of real (non-warp) emulation, stops recording, closes the container,
and asserts the resulting .avi file exists, is non-trivial in size and has a
valid RIFF/AVI header.

The recording is driven by VICE's native screenshot/movie recorder, so the
file is a genuine frame-by-frame capture of the emulated VIC-II output --
NOT a stitched-together animated GIF.

Run:
    python3 driver/test_video_record.py
"""

import os
import struct
import sys
import time

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vice-driver")),
)
from vice_driver import BinMon, ViceContainer, DiskMount  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))
os.makedirs(OUT, exist_ok=True)

# Host port for binmon. Deliberately not 6502 so this test does not collide
# with a separately-running asid-vice container on the default port.
HOST_PORT = 6510

# Container-side directory the host renders/ dir is mounted into.
CONTAINER_RENDERS = "/renders"
VIDEO_NAME = "test_video_record.avi"


def validate_avi(path: str) -> tuple[int, int]:
    """Return (size, n_frames). Raises AssertionError if the file is not a
    plausible RIFF/AVI. Walks the 'movi' list chunk-by-chunk (respecting
    each chunk's length + word padding) and counts '##dc'/'##db' video
    frames, stopping at the idx1 index so index entries are not counted."""
    size = os.path.getsize(path)
    assert size > 4096, f"video file suspiciously small: {size} bytes"
    with open(path, "rb") as f:
        data = f.read()
    # RIFF container: 'RIFF' <u32 size> 'AVI '
    assert data[0:4] == b"RIFF", f"not a RIFF file: {data[0:4]!r}"
    assert data[8:12] == b"AVI ", f"RIFF form is not AVI: {data[8:12]!r}"
    riff_size = struct.unpack("<I", data[4:8])[0]
    assert riff_size > 0, "RIFF size field is zero (file not finalized?)"

    movi = data.find(b"movi")
    assert movi != -1, "no 'movi' list found (file not finalized?)"
    n_frames = 0
    p = movi + 4
    while p + 8 <= len(data):
        cid = data[p : p + 4]
        sz = struct.unpack("<I", data[p + 4 : p + 8])[0]
        if cid == b"idx1":
            break
        if cid[2:4] in (b"dc", b"db"):
            n_frames += 1
        p += 8 + sz + (sz & 1)
    return size, n_frames


def main() -> int:
    host_video = os.path.join(OUT, VIDEO_NAME)
    container_video = f"{CONTAINER_RENDERS}/{VIDEO_NAME}"
    # Clean any stale artifact so a non-trivial size proves *this* run wrote it.
    if os.path.exists(host_video):
        os.remove(host_video)

    container = ViceContainer(
        binmon_port=HOST_PORT,
        warp=True,
        silent=True,
        mounts=[DiskMount(OUT, CONTAINER_RENDERS, read_only=False)],
    )
    with container:
        bm = BinMon("127.0.0.1", HOST_PORT)
        bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
        bm.exit()  # resume CPU

        # Let the machine settle on the boot screen briefly under warp.
        time.sleep(2)

        print(f"starting recording -> {container_video}", flush=True)
        # video_record() forces warp OFF so frames are actually encoded.
        bm.video_record(container_video)

        # Run in real time for a few seconds so we capture a meaningful
        # number of frames (~50/s PAL). The CPU keeps running because
        # BinMon auto-resumes after the command.
        time.sleep(5)

        print("stopping recording (finalize file)", flush=True)
        bm.video_stop()  # finalize + restore warp
        # Give the close/flush a moment to land on the mounted volume.
        time.sleep(1)
        bm.close()

    assert os.path.exists(host_video), f"video file was not created: {host_video}"
    size, n_frames = validate_avi(host_video)
    print(f"OK: {host_video}")
    print(f"    size      = {size} bytes ({size / 1024:.1f} KiB)")
    print(f"    container = RIFF/AVI (valid header)")
    print(f"    frames    = {n_frames} video frames")
    assert n_frames > 0, "no video frames found in AVI"
    return 0


if __name__ == "__main__":
    sys.exit(main())
