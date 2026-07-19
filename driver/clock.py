#!/usr/bin/env python3
"""Machine-side clocks for the live driver: waits measured in EMULATED frames or
instructions, never in host seconds.

A host sleep's emulated length depends on warp -- the same delay spans several times
as many frames with warp on (``NO_RECORD=1``) as with recording forcing warp off.
"""

FRAME_PC = 0x9630  # once-per-frame raster-IRQ top marker; MEASURED live to recur in
# every phase the driver touches -- code-entry menu, generation, preview and play.
_FRAME_CP_ATTR = "sentinel_frame_cp"
_STOP_CP_ATTR = "sentinel_frame_stop_cp"


def frame_checkpoint(bm):
    """checknum of a silent, non-stopping $9630 checkpoint, cached on the BinMon so
    every caller shares ONE counter."""
    cp = getattr(bm, _FRAME_CP_ATTR, None)
    if cp is None:
        cp = bm.checkpoint_set(FRAME_PC, stop_when_hit=False, silent=True).checknum
        setattr(bm, _FRAME_CP_ATTR, cp)
    return cp


def stop_checkpoint(bm):
    """checknum of a PERSISTENT stop-on-hit $9630 checkpoint, installed once and
    reused. Installing per call is delivered to a running machine and only serviced
    at the next vsync poll, so it lands on a host-timed frame boundary and can miss
    the frame it was meant to catch (measured: 37 misses per 1655 stops)."""
    cp = getattr(bm, _STOP_CP_ATTR, None)
    if cp is None:
        with bm.halted():
            cp = bm.checkpoint_set(FRAME_PC, stop_when_hit=True, silent=False).checknum
        setattr(bm, _STOP_CP_ATTR, cp)
    return cp


def frames(bm):
    """Wrap-free elapsed video-frame count (u32 hit_count of the $9630 checkpoint) --
    unlike the $1335 delta ((d*5)&0xFF), which aliases every 256 frames.

    Read HALTED: under auto_resume every monitor call is followed by an EXIT, so an
    unguarded read resumes the CPU and the observation itself costs a frame.
    """
    with bm.halted():
        return bm.checkpoint_get(frame_checkpoint(bm)).hit_count


def run_frames(bm, n, timeout=6.0):
    """Advance the running game EXACTLY ``n`` video frames, leaving the CPU HALTED.

    Steps the $9630 marker, which recurs every frame unconditionally, so a marker
    that stops is a stalled emulator and RAISES rather than being waited out.
    Returns the measured frame delta, equal to ``n`` by construction."""
    n = int(n)
    if n <= 0:
        return 0
    with bm.halted():
        f0 = frames(bm)
        cp = stop_checkpoint(bm)
        bm.checkpoint_toggle(cp, True)  # enabled only here: a live stop at $9630 would
        # otherwise pre-empt every other run_until_pc wait (e.g. the gated input scan)
        for _ in range(n):
            bm.advance_instructions(
                1
            )  # step off the marker so the hit is the NEXT frame
            bm.wait_for_checkpoint(cp, timeout=timeout)
        bm.checkpoint_toggle(cp, False)
        got = frames(bm) - f0
    if got < n:
        raise RuntimeError(
            f"run_frames: measured {got} of {n} frames at ${FRAME_PC:04x} (stalled?)"
        )
    return got
