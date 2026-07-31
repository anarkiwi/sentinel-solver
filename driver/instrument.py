#!/usr/bin/env python3
"""Frame-locked divergence instrument: race the sim against the real game.

Seeds the sim from the emulator's own image at the machine's own cycle, advances ONE
video frame on each in lockstep and diffs :mod:`sentinel.statecmp` per frame.
Run: ``python -m driver.instrument 335``.
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


REG_SP = 4  # registers_get id of the stack pointer


class SimClock:
    """The standalone sim as a one-frame-per-tick clock over a 64 KB image."""

    def __init__(self, image, resume=None):
        self.state = State.from_mem(image)
        if resume is not None:  # the machine was mid-pass; occupy the same position
            (
                self.state.pass_phase,
                self.state.body_stage,
                self.state.body_index,
                self.state.body_partial,
                residual,
            ) = resume
            self.state.cycle_residual = residual or 0
        self.plotting = False

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

    def position(self, image):
        """(resume, interrupted PC, play-loop address) where $9630 caught the loop."""
        sp = self.bm.registers_get()[REG_SP]
        page = self.bm.mem_get(0x0100, 0x01FF)
        return enemies.stack_position(image, sp, page)


SEED_TRIES = 60  # default wait; race() instead gives a seed the rest of its own budget


class Seed:
    """One seeding of the sim, and the cycle uncertainty it carries.

    ``exact``: the position was counted off a ROM straight line, so the sim starts at
    the machine's own cycle.  Else it starts at ``addr``, the head of the segment the
    machine is inside, and that segment's spent cycles are the seed's own error."""

    def __init__(self, resume, waited, pc, addr, caught):
        self.resume, self.waited, self.pc, self.addr = resume, waited, pc, addr
        self.caught = caught  # (pc, loop addr) at the frame that asked for this seed
        self.exact = resume[4] is not None

    def note(self):
        if self.exact:
            return f"seeded exact (+/-0 cycles) at ${self.pc:04X}"
        return (
            f"seeded at ${self.addr:04X} head, machine at ${self.pc:04X}: "
            f"+/- unbounded (no countable marker in {self.waited} frames)"
        )


def _seed(emu, exact=True, tries=SEED_TRIES):
    """Step frames until the marker catches the loop at the machine's exact cycle.

    A position off the ROM's straight lines resumes only to its segment's head, which
    injects that segment's spent cycles; waiting for a countable one is the only
    seeding with no error of its own."""
    waited, caught = 0, None
    while True:
        image = emu.full_image()
        resume, pc, addr = emu.position(image)
        caught = caught or (pc, addr)
        if resume[4] is not None or not exact or waited >= tries:
            return image, Seed(resume, waited, pc, addr, caught)
        emu.step_frame()
        waited += 1


def _unfreeze(img):
    """The $0CE5-cleared byte that starts the cooldown clock (player has acted)."""
    return img[mm.PLAYER_NOT_ACTED] & 0x7F


def race(bm, max_frames, follow=False, log=print, exact_seed=True):
    """Frame-lock the sim against the live game and collect divergences.

    ``max_frames`` bounds the MACHINE frames the race spans, seeding waits included.
    ``exact_seed`` False is the control: it reseeds wherever the marker falls.
    ``a`` in each Divergence is the emulator, ``b`` the sim."""
    emu = EmuClock(bm)
    first = {}
    core_events = []
    seeds = []
    raced = 0
    with bm.halted():
        emu.sync_to_frame()
        seed, first_seed = _seed(emu, exact_seed, max_frames)
        seeds.append(first_seed)
        frames_run = first_seed.waited
        sim = SimClock(seed, first_seed.resume)
        unfrozen = _unfreeze(seed)
        emu.poke(mm.PLAYER_NOT_ACTED, unfrozen)
        sim.poke(mm.PLAYER_NOT_ACTED, unfrozen)
        log(
            f"[instrument] seeded; energy={seed[mm.PLAYER_ENERGY]} "
            f"player_slot={seed[mm.PLAYER_OBJECT]} not_acted->${unfrozen:02X} "
            f"follow={follow}; {seeds[0].note()}"
        )
        seg_start = frames_run
        while frames_run < max_frames:
            emu.step_frame()
            sim.step_frame()
            frames_run += 1
            raced += 1
            f = frames_run
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
                event = {
                    "frame": f,
                    "gap": f - seg_start,
                    "divs": core_divs,
                    "seed": seeds[-1],
                    "caught": None,
                }
                core_events.append(event)
                if not follow:
                    break
                live, nxt = _seed(emu, exact_seed, max_frames - frames_run)
                event["caught"] = nxt.caught
                frames_run += nxt.waited
                sim = SimClock(live, nxt.resume)
                seeds.append(nxt)
                seg_start = frames_run
                if len(seeds) <= 16:
                    log(f"[instrument] CORE at frame {f}; {seeds[-1].note()}")
    return {
        "first": first,
        "core_events": core_events,
        "seeds": seeds,
        "resyncs": len(seeds) - 1,
        "frames": frames_run,
        "raced": raced,
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
    seeds = result["seeds"]
    exact = sum(1 for s in seeds if s.exact)
    waited = sum(s.waited for s in seeds)
    log(
        f"\n---- follow: {len(events)} CORE event(s), {result['resyncs']} resync(s) "
        f"over {result['frames']} machine frames ({result['raced']} raced, "
        f"{waited} spent waiting for a countable marker) ----"
    )
    log(
        f"  seeds at the machine's exact cycle: {exact}/{len(seeds)} "
        f"({100.0 * exact / len(seeds):.1f}%); "
        f"longest wait {max(s.waited for s in seeds)} frame(s)"
    )
    for s in seeds:
        if not s.exact:
            log(f"    inexact seed: {s.note()}")
    if not events:
        return
    gaps = sorted(e["gap"] for e in events)
    log(
        f"  frames between divergences: min={gaps[0]} "
        f"median={gaps[len(gaps) // 2]} max={gaps[-1]}"
    )
    log("  where the raster caught the machine at each event (pc -> loop address):")
    for key, n in _histogram(e["caught"] for e in events if e["caught"]):
        log(f"    ${key[0]:04X} -> ${key[1]:04X} x{n}")
    log("  the diverging fields:")
    for label, n in _histogram(d.label for e in events for d in e["divs"]):
        log(f"    {label} x{n}")
    for e in events[:24]:
        labels = ", ".join(sorted({d.label for d in e["divs"]}))
        caught = f" pc=${e['caught'][0]:04X}" if e["caught"] else ""
        log(
            f"  frame {e['frame']:>4} (+{e['gap']:>3}){caught}: {labels}"
            f"  [{e['seed'].note()}]"
        )
    if len(events) > 24:
        log(f"  ... (+{len(events) - 24} more events)")


def _histogram(values):
    """``(value, count)`` descending by count; addresses stay raw, named by no table."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


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
    ap.add_argument(
        "--seed-any",
        action="store_true",
        help="control: reseed wherever the marker falls, segment head and all",
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
        result = race(
            drv.bm, args.frames, follow=args.follow, exact_seed=not args.seed_any
        )
        report(result, args.frames, follow=args.follow)
    finally:
        drv.close()
    return 0 if result.get("first") else 1


if __name__ == "__main__":
    raise SystemExit(main())
