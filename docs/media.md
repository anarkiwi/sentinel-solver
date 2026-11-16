# Recordings: AVI to APNG

`driver.play_player` records the live run through VICE's ZMBV gfxoutput driver
(`core.boot_and_play`, via `BinMon.video_record`): lossless, indexed, 384×272, PAL
50.125 fps, written to `renders/<name>.avi`. A won landscape is tens of thousands of
frames, so `driver.avi2apng` turns one into an APNG small enough to embed.

```bash
python -m driver.play_player 9716             # live search player -> renders/player_ls9716_search.avi
python -m driver.avi2apng renders/player_ls9716_search.avi \
    -o docs/media/ls9716_search_win.png --fps 12 --speed 2
```

`docs/media/lsN_search_win.png` are the search player's live wins, one per row of
`docs/results/live.json`.

## CRT at 720p

The ZMBV driver encodes VICE's palette-indexed screen before rendering, so it carries no
CRT emulation. `--crt` uses asid-vice's rendered recording instead (`driver.crt`):

```bash
python -m driver.play_player 9795 --crt       # -> renders/player_ls9795_search_crt.mp4
```

| | how |
|---|---|
| frames | `VIDEO_RECORD` action 2: every emulated frame after VICE's render filter (`-VICIIfilter 1`), scaled to 720 lines at the true pixel aspect (952×720 for the PAL canvas), H.264 at the machine's refresh rate; a halted machine records nothing, so search pauses cost no frames |
| blanks | a redraw blanks the viewport by giving all its colour sources one colour (`$8643` with A = 6 at `$358D`: `$D022`/`$D023`, screen RAM, colour RAM, `$D021` last); a store watchpoint on `$D021` (`prog.WATCH`) pauses the recording while `$D021`–`$D023` are one colour and resumes once the landscape's scheme is back |

## Dropping the blanks

The game blanks the 3D viewport for the whole of a **hyperspace** and a **transfer**
— a flat fill under the status strip it keeps drawing — so those frames carry no
information and dominate the running time.

A frame is blank when its play area (the picture, less the top `--hud-rows`) is a
**single colour**. That test needs no threshold: a drawn viewport always carries edges,
and a blanked one carries none. A dominance fraction would not separate the two, since a
drawn landscape can be almost entirely one colour; uniformity does, exactly.

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
