# The divergence instrument (`driver/instrument.py`)

Races the standalone simulator against the real game in VICE frame-for-frame and
reports the **first** state disagreement, decoded to a named field. It is the gate
that keeps the enemy clock exact: the bar is zero CORE divergence, and the
instrument says exactly where and when that bar is missed.

## Why a shared instrument is possible

Both worlds keep their entire play state in a 64 KB image at the **same** RAM
addresses (`sentinel/memmap.py`): the sim's `State` is a `bytearray` addressed like
the C64, and the live game is read out of VICE at those same addresses. So one
schema (`sentinel/statecmp.py`) decodes either image, and a byte-diff over that
schema is directly meaningful on both.

## The shared clock: one video frame

- **Emulator** (`EmuClock`) — `advance_instructions(1)` off the per-frame raster
  marker `$9630`, then `run_until_pc($9630)`. One `$9630`→`$9630` span is exactly one
  ROM frame (one `$9663` raster cooldown tick + the frame's `update_enemies` passes).
  Same frame anchor `kbd_aim.idle_until_rotation` uses.
- **Simulator** (`SimClock`) — `sentinel.enemies.advance_frame`: the `$130C`/`$1335`
  Bresenham cooldown tick first, then `UPDATES_PER_FRAME` (8) `update_enemies` passes,
  suppressed entirely when `plotting=True`.

Both are seeded from the **emulator's own 64 KB image** at entry, so frame 0 is
byte-identical and every later difference is purely the sim's dynamics vs the ROM's.
Seeding from the full image also gives the sim the real in-RAM tables (e.g. the
rotation-speed data at `$9D37`) and sidesteps the landscape-number/seed ambiguity.

## The schema and its tiers

`statecmp.FIELDS` expands every sim-maintained byte to one labelled address — all
eight object arrays (per slot), the eleven per-enemy phase arrays, the 32×32 tile
grid, the PRNG, both cooldown clocks and the scalar play variables. Render, sound
and screen scratch are excluded (a raw full-image diff would drown the signal).

| Tier | Fields | Meaning |
|------|--------|---------|
| `CORE` | objects, enemy cooldowns, energy, tiles, discharge/meanie arrays, `$1335`, `$0C50` | the state the sim reproduces frame-for-frame; a CORE divergence is a real model/ROM disagreement |
| `SWEEP` | cursor `$0090`, PRNG `$0C7B-$0C7F` | by-design non-goals: the PRNG drives unreadable hyperspace/meanie landing coords, and `$0090` only orders slots within a frame — the sim sweeps all 8 each frame, the ROM's foreground makes 2/3/4 passes, and the enemy's own `$16E9` update_cd gate (reload 4) is what rate-limits it |
| `SCRATCH` | `$0014`, `$0C56-$0C58`, `$0C68`, `$0C76`, `$0CDD` | LOS/targeting bytes the ROM rewrites every scan; transient |

The instrument records the first frame each tier first disagrees and stops on the
first `CORE` divergence.

## Running it

```bash
python -m driver.instrument 42 --frames 1200
```

Boots the landscape under warp with **no recording** (`NO_RECORD=1`), unfreezes the
enemy clock on both sides by clearing `$0CE5` bit7 (the "player has acted" gate),
then frame-locks and prints the per-tier first-divergence report (`emu=A`, `sim=B`).

### Follow mode (`--follow`)

A CORE divergence **reseeds the sim from live memory** (the same resync the live
driver does on a world-divergence) and the race continues to `--frames`, so the
whole *sequence* of CORE disagreements is reported rather than just the first. The
report adds the event count, the resync count, the min/median/max frame gap between
successive events, and the first dozen events with their fields.

## Status and the gate

`driver/test_enemy_sim_divergence.py::test_enemy_sim_frame_locked_to_live_ls42` is
the gate: it boots ls42, races 600 frames and asserts **no CORE divergence** — a
plain assertion, no xfail. `python -m driver.instrument 42 --frames 1200` likewise
reports no CORE divergence. See [plan_fidelity.md](plan_fidelity.md) for the
measured state and the remaining open items.

Fidelity here is binary: a sim that reproduces enemy phase 97% of the time is 0%
correct on the outcome it decides, because one rotation step of drift places a body
into a gaze the planner modelled empty. A CORE divergence is a model bug to
eliminate — never a "residual" or "readout artifact".
