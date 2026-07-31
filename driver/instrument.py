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
_SP_REGISTER = 4  # VICE binary-monitor register id
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

    def __init__(self, image):
        self.state = State.from_mem(image)
        self.plotting = False
        self.replot_tail = None

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
        sp = self.bm.registers_get()[_SP_REGISTER] & 0xFF
        return stack_frames(self.bm.mem_get(0x0100, 0x01FF), sp)

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


def seed_sim(emu, image):
    """A sim clock at the machine's position, owing what the machine still owes.

    A halt inside the $1FFC replot resumes at $1884's JMP $16D6 -- the body is done,
    the prnd is not -- carrying the unspent replot as this frame's cycle debt and the
    $1FFF..$2008 camera restore as a write the frame that debt clears makes."""
    sim = SimClock(image)
    st = sim.state
    debt = replot_debt(st, emu.frames_on_stack())
    if debt:
        st.cycle_residual -= int(round(debt))
        st.pass_phase = enemies.PHASE_BODY
        st.body_stage = enemies.BODY_DONE
        sim.replot_tail = lambda: _restore_camera(st)
    return sim, debt


def _restore_camera(state):
    """$2003/$2008: undo the $1FC2 strip shift of the camera's own bearing."""
    state.mem[CAMERA_FRACTION] = 0
    saved = state.mem[projector.CAMERA_SAVED]
    state.obj_h_angle[state.mem[projector.CAMERA_OBJECT]] = saved


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
        sim, _debt = seed_sim(emu, seed)
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
                sim, debt = seed_sim(emu, emu.full_image())  # resync from live truth
                resyncs += 1
                seg_start = f
                if resyncs <= 15:
                    owed = (
                        f" owing {debt / projector.FRAME_CYCLES:.1f}f of replot"
                        if debt
                        else ""
                    )
                    log(f"[instrument] CORE divergence at frame {f}; resynced{owed}")
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
