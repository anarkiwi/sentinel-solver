#!/usr/bin/env python3
"""Frame-locked divergence instrument: race the sim against the real game.

Seeds the sim from the emulator's own 64 KB image (byte-identical start), unfreezes
the enemy clock on both, advances ONE video frame on each in lockstep, and diffs the
shared schema (:mod:`sentinel.statecmp`) per frame. Run: ``python -m driver.instrument 335``.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sentinel import enemies, memmap as mm, statecmp as sc
from sentinel.state import State
from driver import clock, core

RENDERS = os.path.join(core.boot.ROOT, "renders")
FRAME_PC = (
    clock.FRAME_PC
)  # once-per-frame raster-IRQ top marker (the frame-step anchor)


class SimClock:
    """The standalone sim as a one-frame-per-tick clock over a 64 KB image."""

    def __init__(self, image, plotting=False):
        self.state = State.from_mem(image)
        self.plotting = plotting

    def image(self):
        return self.state.mem

    def step_frame(self):
        enemies.advance_frame(self.state, plotting=self.plotting)

    def poke(self, addr, val):
        self.state.mem[addr] = val & 0xFF


class EmuClock:
    """The live VICE game as a one-frame-per-tick clock (CPU driven while halted)."""

    def __init__(self, bm):
        self.bm = bm

    def full_image(self):
        return core.live_image(self.bm)

    def image(self):
        return self.bm.mem_get(0x0000, sc.MAX_ADDR)

    def sync_to_frame(self):
        self.bm.run_until_pc(FRAME_PC, timeout=6.0)

    def step_frame(self):
        self.bm.advance_instructions(1)  # step off the marker
        self.bm.run_until_pc(FRAME_PC, timeout=6.0)

    def poke(self, addr, val):
        self.bm.mem_set(addr, bytes([val & 0xFF]))


def _unfreeze(img):
    """The $0CE5-cleared byte that starts the cooldown clock (player has acted)."""
    return img[mm.PLAYER_NOT_ACTED] & 0x7F


def race(bm, max_frames, follow=False, log=print):
    """Frame-lock the sim against the live game and collect divergences.

    Returns ``{first, core_events, resyncs, frames}``. In ``follow`` mode a CORE
    divergence reseeds the sim from live memory and the race continues; else it
    stops at the first. ``a`` in each Divergence is the emulator, ``b`` the sim."""
    emu = EmuClock(bm)
    first = {}
    core_events = []
    resyncs = 0
    frames_run = 0
    with bm.halted():
        emu.sync_to_frame()
        seed = emu.full_image()
        sim = SimClock(seed)
        unfrozen = _unfreeze(seed)
        emu.poke(mm.PLAYER_NOT_ACTED, unfrozen)
        sim.poke(mm.PLAYER_NOT_ACTED, unfrozen)
        log(
            f"[instrument] seeded; energy={seed[mm.PLAYER_ENERGY]} "
            f"player_slot={seed[mm.PLAYER_OBJECT]} not_acted->${unfrozen:02X} "
            f"follow={follow}"
        )
        seg_start = 0
        for f in range(1, max_frames + 1):
            frames_run = f
            emu.step_frame()
            sim.step_frame()
            grouped = sc.by_tier(sc.diff(emu.image(), sim.image()))
            for tier, tier_divs in grouped.items():
                if tier_divs and tier not in first:
                    first[tier] = (f, tier_divs)
                    log(
                        f"[instrument] first {tier.upper()} divergence at frame {f} "
                        f"({len(tier_divs)} field(s))"
                    )
            core_divs = grouped[sc.CORE]
            if core_divs:
                core_events.append((f, f - seg_start, core_divs))
                if not follow:
                    break
                sim = SimClock(emu.full_image())  # resync from live truth
                resyncs += 1
                seg_start = f
                if resyncs <= 15:
                    log(f"[instrument] CORE divergence at frame {f}; resynced")
    return {
        "first": first,
        "core_events": core_events,
        "resyncs": resyncs,
        "frames": frames_run,
    }


def report(result, max_frames, follow=False, log=print):
    """Print the per-tier first-divergence summary, plus the follow-mode sequence."""
    first = result["first"]
    log("\n================ DIVERGENCE REPORT (emu=A, sim=B) ================")
    for tier in sc.TIERS:
        if tier not in first:
            log(f"[{tier.upper():7}] no divergence within {max_frames} frames")
            continue
        frame, divs = first[tier]
        log(f"[{tier.upper():7}] first at frame {frame}: {len(divs)} field(s)")
        for d in divs[:24]:
            log("    " + sc.format_divergence(d, "emu", "sim"))
        if len(divs) > 24:
            log(f"    ... (+{len(divs) - 24} more)")
    if not follow:
        return
    events = result["core_events"]
    log(
        f"\n---- follow: {len(events)} CORE event(s), {result['resyncs']} resync(s) "
        f"over {result['frames']} frames (each resynced from live truth) ----"
    )
    if not events:
        return
    gaps = sorted(e[1] for e in events)
    log(
        f"  frames between divergences: min={gaps[0]} "
        f"median={gaps[len(gaps) // 2]} max={gaps[-1]}"
    )
    for f, gap, divs in events[:12]:
        labels = ", ".join(sorted({d.label for d in divs}))
        log(f"  frame {f:>4} (+{gap:>3}): {labels}")
    if len(events) > 12:
        log(f"  ... (+{len(events) - 12} more events)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="frame-locked sim-vs-emu divergence")
    ap.add_argument(
        "landscape",
        nargs="?",
        default="335",
        help="landscape number to type (e.g. 335 -> types 0335)",
    )
    ap.add_argument("--frames", type=int, default=1200, help="max frames to race")
    ap.add_argument(
        "--follow",
        action="store_true",
        help="on a CORE divergence, resync the sim from live memory and keep racing",
    )
    args = ap.parse_args(argv)
    os.environ.setdefault("NO_RECORD", "1")

    drv = core.SentinelDriver.boot(record_mount=RENDERS)
    result = {}
    try:
        try:
            drv.bm.resource_set_int("WarpMode", 1)  # container already boots warp
        except Exception as e:  # pylint: disable=broad-except
            print(f"[instrument] warp resource set skipped: {e}")
        digits = args.landscape.zfill(4)  # e.g. "335" -> "0335"
        drv.enter_landscape(int(digits, 16))
        result = race(drv.bm, args.frames, follow=args.follow)
        report(result, args.frames, follow=args.follow)
    finally:
        drv.close()
    return 0 if result.get("first") else 1


if __name__ == "__main__":
    raise SystemExit(main())
