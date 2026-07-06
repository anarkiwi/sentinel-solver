#!/usr/bin/env python3
"""Drive the real Sentinel game in asid-vice (VICE) and grab VICE's rendered
display, to see exactly how the game draws its objects.

Boots the tape, waits for the loader, enters a landscape, and saves PNG frames
of VICE's own framebuffer at each step (renders/vice_*.png).
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
os.makedirs(OUT, exist_ok=True)


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
        bm.exit()  # resume CPU

        pal = parse_palette_response(bm.palette_get())

        def grab(tag):
            snap = parse_display_response(bm.display_get())
            p = os.path.join(OUT, f"vice_{tag}.png")
            w, h = snap.save_png(p, pal)
            print(f"  grab {tag}: {w}x{h} -> {p}", flush=True)

        def tap_names(*names, frames=20):
            # FIXED hold: survive the game's flush-then-read keyboard handling
            bm.keymatrix_tap(
                [keys.lookup(n) for n in names], mode=TAP_MODE_FIXED, frames=frames
            )
            time.sleep(0.5)

        def tap_text(t, frames=20):
            for chord in keys.text_to_chords(t):
                bm.keymatrix_tap(
                    [keys.lookup(n) for n in chord], mode=TAP_MODE_FIXED, frames=frames
                )
                time.sleep(0.5)

        # let the turbo loader run (warp); grab periodically to see progress
        for i in range(5):
            time.sleep(10)
            grab(f"load_{i}")

        # title screen -> press a key (hold long); confirm we reach the prompt
        for _attempt in range(3):
            tap_names("SPACE", frames=30)
            time.sleep(2)
        grab("after_key")

        # landscape number 0000 (no secret code needed for 0000)
        tap_text("0000")
        time.sleep(0.5)
        grab("typed_0000")
        tap_names("RETURN", frames=30)
        time.sleep(8)
        grab("entered_0000")

        # in game: look around to see objects
        time.sleep(3)
        grab("game_0")
        for k in "DDDDDD":  # pan right (rotate view)
            tap_names(k, frames=14)
            time.sleep(0.5)
        grab("game_pan_right")
        for k in "SSSSSS":  # pan left back and beyond
            tap_names(k, frames=14)
            time.sleep(0.5)
        grab("game_pan_left")
        bm.close()


if __name__ == "__main__":
    main()
