#!/usr/bin/env python3
"""Load Sentinel landscape 0000, enter first-person, and survey a full rotation
(plus look-up) capturing VICE's rendered frames so we can see every object's
true shape. Frames -> renders/survey_*.png
"""

import os, sys, time

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vice-driver")),
)
from vice_driver import BinMon, DiskMount, ViceContainer, keys
from vice_driver.binmon import TAP_MODE_FIXED
from vice_driver.display import parse_display_response, parse_palette_response

TAP = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sentinel-gold.tap")
)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renders"))


def main():
    container = ViceContainer(
        autostart="/work/sentinel.tap",
        mounts=[DiskMount(TAP, "/work/sentinel.tap", read_only=True)],
        warp=True,
        silent=True,
    )
    with container:
        bm = BinMon("127.0.0.1", 6502)
        bm.connect(timeout=20.0, attempts=200, retry_delay=0.5)
        bm.exit()
        pal = parse_palette_response(bm.palette_get())

        def grab(tag):
            snap = parse_display_response(bm.display_get())
            p = os.path.join(OUT, f"survey_{tag}.png")
            snap.save_png(p, pal)
            print(f"  {tag}", flush=True)

        def tap(*names, frames=22):
            bm.keymatrix_tap(
                [keys.lookup(n) for n in names], mode=TAP_MODE_FIXED, frames=frames
            )
            time.sleep(0.45)

        time.sleep(48)  # turbo load
        for _ in range(3):
            tap("SPACE", frames=30)
            time.sleep(1.5)
        for ch in keys.text_to_chords("0000"):
            bm.keymatrix_tap(
                [keys.lookup(n) for n in ch], mode=TAP_MODE_FIXED, frames=25
            )
            time.sleep(0.4)
        tap("RETURN", frames=30)
        time.sleep(7)
        grab("preview")  # isometric preview
        tap("SPACE", frames=30)
        time.sleep(3)  # enter first-person
        grab("fp_start")

        # full rotation: ~24 right pans, grab every 2
        for i in range(24):
            tap("D", frames=12)
            if i % 2 == 0:
                grab(f"rot_{i:02d}")
        # look up to see the Sentinel on its tower
        for i in range(4):
            tap("W", frames=12)
        grab("up")
        for i in range(8):
            tap("D", frames=12)
            if i % 2 == 0:
                grab(f"up_rot_{i:02d}")
        bm.close()


if __name__ == "__main__":
    main()
