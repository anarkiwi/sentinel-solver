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

from sentinel import enemies, memmap as mm, projector, statecmp as sc
from sentinel.state import State
from driver import clock, core

RENDERS = os.path.join(core.boot.ROOT, "renders")
FRAME_PC = (
    clock.FRAME_PC
)  # once-per-frame raster-IRQ top marker (the frame-step anchor)
REPLOT_RETURN = 0x1FFF  # $1FFC JSR $2625, update_object_on_screen's strip replot
_ROW_LOOP = range(0x26D9, 0x26F0)  # $26EF DEC $26 unspent: the next row is $0026 - 1
_PLOT_ROW_RETURNS = (0x2760, 0x2793)  # the two JSR $295D, so $0025 names a live tile
_PLOT_SETUP = range(
    0x2625, 0x26D6
)  # $26C9 has not seeded $0026: the whole pass is owed
REG_SP = 4  # registers_get id of the stack pointer
CAMERA_FRACTION = 0x1F  # $1FD0 the fine angle, zeroed again at $2003


def stack_frames(page, sp):
    """The interrupted PC and every return address under the $95E9 frame.

    $95E9 pushes Y, X and A over the hardware P/PCH/PCL, so the foreground's PC is at
    SP+5/6 and its JSR chain runs up the page from SP+7 as (lo, hi) of return-1."""
    out = [page[(sp + 5) & 0xFF] | (page[(sp + 6) & 0xFF] << 8)]
    for a in range((sp + 7) & 0xFF, 0xFF, 2):
        out.append(((page[a] | (page[a + 1] << 8)) + 1) & 0xFFFF)
    return out


def replot_debt(state, frames):
    """The cycles an interrupted $1FFC JSR $2625 still owes, 0 when there is none.

    plot_world's progress is its own zero page: $0026 is the row it has walked down
    to and $0025 the column $295D is plotting, which :func:`projector.replot_owed`
    turns into the unspent tail of the very pass the model would otherwise recharge."""
    if REPLOT_RETURN not in frames:
        return 0
    inside = frames[: frames.index(REPLOT_RETURN)]
    if any(a in _PLOT_SETUP for a in inside):
        row = 0xFF
    else:
        row = state.mem[projector.PLOT_ROW] - any(a in _ROW_LOOP for a in inside)
    return projector.replot_owed(
        state,
        row,
        state.mem[projector.PLOT_COLUMN],
        any(a in _PLOT_ROW_RETURNS for a in inside),
    )


class SimClock:
    """The standalone sim as a one-frame-per-tick clock over a 64 KB image."""

    def __init__(self, image, resume=None):
        self.state = State.from_mem(image)
        self.plotting = False
        self.replot_tail = None
        if resume is not None:
            self.seat(resume)

    def seat(self, resume):
        """Occupy the sub-pass position and cycle the machine itself is at."""
        (
            self.state.pass_phase,
            self.state.body_stage,
            self.state.body_index,
            self.state.body_partial,
            residual,
        ) = resume
        self.state.cycle_residual = residual or 0

    def image(self):
        return self.state.mem

    def step_frame(self):
        enemies.advance_frame(self.state, plotting=self.plotting)
        # $2005: the frame the replot debt clears is the frame $2625 returns in.
        if self.replot_tail and self.state.body_stage != enemies.BODY_DONE:
            self.replot_tail()
            self.replot_tail = None

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

    def frames_on_stack(self):
        """:func:`stack_frames` for the foreground this $9630 halt interrupted."""
        sp = self.bm.registers_get()[REG_SP] & 0xFF
        return stack_frames(self.bm.mem_get(0x0100, 0x01FF), sp)

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

    def __init__(self, resume, waited, pc, addr, caught, debt=0):
        self.resume, self.waited, self.pc, self.addr = resume, waited, pc, addr
        self.caught = caught  # (pc, loop addr) at the frame that asked for this seed
        self.debt = debt  # the unspent $1FFC replot this position carries, if any
        self.exact = resume[4] is not None

    def note(self):
        owed = (
            f", owing {self.debt / projector.FRAME_CYCLES:.1f}f of replot"
            if self.debt
            else ""
        )
        if self.exact:
            return f"seeded exact (+/-0 cycles) at ${self.pc:04X}{owed}"
        return (
            f"seeded at ${self.addr:04X} head, machine at ${self.pc:04X}: "
            f"+/- unbounded (no countable marker in {self.waited} frames){owed}"
        )


def _replot_resume(state, emu):
    """The resume a halt inside the $1FFC replot occupies, its cycle debt and its tail.

    plot_world's own $0025/$0026 price the unspent tail, so the position is countable
    where :func:`enemies.stack_position` has no straight line: the body is done and the
    prnd is not ($1884 JMP $16D6), and $1FFF..$2008 restores the camera on the way."""
    debt = replot_debt(state, emu.frames_on_stack())
    if not debt:
        return None, 0, None
    resume = (enemies.PHASE_BODY, enemies.BODY_DONE, 0, -1, -int(round(debt)))
    return resume, debt, lambda: _restore_camera(state)


def _seed(emu, exact=True, tries=SEED_TRIES):
    """Step frames until the marker catches the loop at the machine's exact cycle.

    A position off the ROM's straight lines resumes only to its segment's head, which
    injects that segment's spent cycles; the $1FFC replot is the one such position
    :func:`_replot_resume` can count, so it never has to be waited out."""
    waited, caught = 0, None
    while True:
        image = emu.full_image()
        resume, pc, addr = emu.position(image)
        caught = caught or (pc, addr)
        sim = SimClock(image)
        replot, debt, tail = _replot_resume(sim.state, emu)
        resume = replot or resume
        if resume[4] is not None or not exact or waited >= tries:
            sim.seat(resume)
            sim.replot_tail = tail
            return sim, Seed(resume, waited, pc, addr, caught, debt)
        emu.step_frame()
        waited += 1


def _restore_camera(state):
    """$2003/$2008: undo the $1FC2 strip shift of the camera's own bearing."""
    state.mem[CAMERA_FRACTION] = 0
    saved = state.mem[projector.CAMERA_SAVED]
    state.obj_h_angle[state.mem[projector.CAMERA_OBJECT]] = saved


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
        sim, first_seed = _seed(emu, exact_seed, max_frames)
        seed = sim.image()
        seeds.append(first_seed)
        frames_run = first_seed.waited
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
                sim, nxt = _seed(emu, exact_seed, max_frames - frames_run)
                event["caught"] = nxt.caught
                frames_run += nxt.waited
                seeds.append(nxt)
                seg_start = frames_run
                if len(seeds) <= 16:
                    log(f"[instrument] CORE at frame {f}; {nxt.note()}")
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
