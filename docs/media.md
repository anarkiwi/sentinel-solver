# Recordings: AVI to APNG

`driver.play_player` records the live run through VICE's ZMBV gfxoutput driver
(`core.boot_and_play`, via `BinMon.video_record`): lossless, indexed, 384×272, PAL
50.125 fps, written to `renders/<name>.avi`. A won landscape is tens of thousands of
frames, so `driver.avi2apng` turns one into an APNG small enough to embed.

```bash
python -m driver.avi2apng renders/player_ls340_win.avi -o docs/media/ls340.png --fps 12
```

| flag | default | effect |
|---|---|---|
| `--fps` | 12 | decimate to this rate; `0` keeps every frame |
| `--scale` | 1 | integer nearest-neighbour upscale |
| `--hud-rows` | 6 | rows of status strip a blanked viewport keeps drawing |
| `--min-blank-run` | 1 | blank runs shorter than this are kept |
| `--loops` | 0 | `0` loops forever |
| `--no-crop` | off | keep the border instead of the picture |
| `--max-frames` | 0 | stop decoding after N source frames |

## Dropping the blanks

The game blanks the 3D viewport for the whole of a **hyperspace** and a **transfer**
— a flat fill under the status strip it keeps drawing — so those frames carry no
information and dominate the running time.

A frame is blank when its play area (the picture, less the top `--hud-rows`) is a
**single colour**. That test needs no threshold. Measured over
`renders/player_ls340_win.avi` (5321 frames), the play area is either one flat colour
or carries at least 215 edge transitions, with nothing in between: 1393 frames
(26.2%) are blank, in runs matching the driver's own frame audit — the 349-frame run
at frame 582 is the transfer the log prices at 367 frames.

Dominance fractions do *not* separate the two: gameplay reaches 0.9847 of one colour
and blanks start at 0.9851. Uniformity does, exactly.

## Staying small

Three properties of the recording carry the size reduction, all lossless:

* the source is `pal8`, so the APNG is indexed — no quantisation, no colour loss;
* consecutive frames mostly repeat, so a repeat becomes delay on its predecessor
  rather than another frame;
* what changes is local, so each frame stores only its changed rectangle
  (`fcTL` offset + `fdAT`, dispose `NONE`, blend `SOURCE`).

The picture is cropped to the pixels that ever change across the run, which recovers
the C64 border: 384×272 → 320×200.

The writer is stdlib `zlib` plus `numpy` — rows carry the usual adaptive PNG filter,
picked per row by minimum sum of absolute differences. Only decoding needs PyAV.
`driver/test_avi2apng.py` asserts the round trip is pixel-exact, including a
random-noise case that forces every filter, Paeth included.
