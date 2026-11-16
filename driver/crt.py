#!/usr/bin/env python3
"""VICE's CRT-filtered frames at 720p, without the redraw's blank screen.

asid-vice's rendered recording (``VIDEO_RECORD`` action 2) encodes every emulated frame
after VICE's render filter. ``$8643`` blanks the viewport by giving all its colour
sources one colour, ``$D021`` last; a write there pauses or resumes the recording.
"""

from vice_driver.binmon import CHECK_STORE, OPCODE

HEIGHT = 720
ARGS = ["-VICIIfilter", "1", "-VICIIaspectmode", "2"]  # CRT filter, true pixel aspect
BACKGROUND = 0xD021  # then $D022/$D023: the multicolour bitmap's shared colours
START, PAUSE, RESUME = 2, 3, 4


def start(bm, path, height=HEIGHT):
    """Record every frame VICE renders to the container-side MP4 ``path``, scaled to
    ``height`` lines at the canvas's pixel aspect (stop with ``bm.video_stop()``)."""
    name = path.encode("ascii")
    body = bytes([START]) + height.to_bytes(2, "little") + bytes([len(name)]) + name
    bm.call(OPCODE.VIDEO_RECORD, body)


def blanked(colours):
    """Whether the VIC's ``$D021``-``$D023`` colours are one: a viewport of one colour."""
    return len({c & 0x0F for c in colours}) == 1


def background(bm):
    """Pause the recording while the viewport is blanked, resume once it is not."""
    colours = bm.mem_get(BACKGROUND, BACKGROUND + 2, bank="io")
    bm.call(OPCODE.VIDEO_RECORD, bytes([PAUSE if blanked(colours) else RESUME]))


WATCH = {(CHECK_STORE, BACKGROUND): background}
