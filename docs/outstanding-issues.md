# Outstanding issues

State after the on_platform win-gate, LOS/gaze safety, and aim-readback fixes:
the offline planner produces a real ls0 win, and the live driver executes aims
faithfully (no false aim-misses). Live ls0 still LOSES, for the reasons below.

## 1. The planner routes into exposure and flails to death

**Symptom (live ls0).** The planner climbs to launch height (eye ~8.9), then,
unable to find a feasible next foothold, goes "stuck" and pan-spins the view to
bank fuel from nearby objects **while exposed**, and is drained to death
(`refuel drained player to death at (2,10) (seen while banking fuel); stuck`). It
also abandons the departed climb-stack shell instead of reabsorbing it (wasted
energy the sim can reclaim -- see `scratchpad/repro_reabsorb.py`, which confirms
the whole synthoid+boulders stack absorbs cleanly).

**Root.** "Stuck" is treated as "keep refuelling in place" instead of a terminal
state, and step/refuel selection does not hard-avoid dwelling where the Sentinel
can see the player (violates Strategy 1 -- exposure is a timed loss -- and
Strategy 2 -- safety is line-of-sight, not distance).

**Proposed fix.**
- When stuck (no LOS-safe foothold that gains launch height), do **not** linger
  refuelling on exposed tiles: fire the endgame from current LOS if reachable,
  else retreat to a tile the Sentinel cannot see at any rotation
  (`threat.ticks_until_seen == horizon`); never pan-spin in view.
- Always reabsorb the departed climb-stack shell before moving on -- it is free,
  on-lattice, predicted-LOS energy.
- Gate every foothold and refuel on the gaze forecast so an exposed dwell is
  never chosen when a safe one exists.

## 2. Wasted aiming through views known to have no line of sight

**Symptom.** The driver pitches the sights down to "stare at its feet" and pans
in circles on aims that have no LOS to any useful target -- pure wasted time,
during which the Sentinel keeps draining.

**Key point.** The view→LOS map is **fully predictable offline**. The sim
(`sentinel.los` / `snap_keyboard_view`) computes exactly which `(bearing, pitch,
cursor)` views have LOS to which tiles *before any key is pressed*; the fired tile
is a deterministic function of `objects_h_angle`, `objects_v_angle` and the cursor
(disasm `INPUT.md` §4). So there is never a reason to drive the sights to a
pitch/bearing already known to have no LOS -- the outcome is known in advance to
be nothing, and every keystroke of that pan is exposed time.

**Proposed fix.**
- The planner emits an aim only to a view predicted (on the resynced state) to
  have LOS to a genuinely useful target (foothold, platform, or worthwhile fuel).
  Never snap or drive to a no-LOS view.
- The driver treats "snapped view has no predicted LOS" as *skip -- do not aim*,
  rather than driving there and probing.
- Because the aim is deterministic, the minimal-cost drive to a predicted-LOS
  view is planned as one direct move, never an exploratory sweep.
