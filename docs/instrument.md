# The divergence instrument (`driver/instrument.py`)

A shared, high-precision instrument that races the standalone simulator against
the real game in VICE frame-for-frame and reports the **first** state disagreement,
decoded to a named field. It exists because the simulator is known to be unfaithful
to the ROM; the instrument is the arbiter that says *exactly where and when* it
first diverges — it does not try to fix the model.

## Why a shared instrument is possible

Both worlds keep their entire play state in a 64 KB image at the **same** RAM
addresses (`sentinel/memmap.py`): the sim's `State` is a `bytearray` addressed like
the C64, and the live game is read out of VICE at those same addresses. So one
schema (`sentinel/statecmp.py`) decodes either image, and a byte-diff over that
schema is directly meaningful on both.

## The shared clock: one video frame

The instrument steps both worlds by exactly one video frame and compares:

- **Emulator** — `advance_instructions(1)` off the per-frame raster marker `$9630`,
  then `run_until_pc($9630)`. One `$9630`→`$9630` span is exactly one ROM frame
  (one `$9663` raster cooldown tick + one `update_enemies` pass). This is the same
  frame anchor `kbd_aim.idle_until_rotation` uses.
- **Simulator** — `sentinel.enemies.advance_frame`, the sim's own real-time model
  (the `$130C`/`$1335` Bresenham cooldown clock + a cursor-gated `update_enemies`).

Both are seeded from the **emulator's own 64 KB image** at entry, so frame 0 is
byte-identical and every later difference is purely the sim's dynamics vs the ROM's.
Seeding from the full image also gives the sim the real in-RAM tables (e.g. the
rotation-speed data at `$9D37`) and sidesteps the landscape-number/seed ambiguity.

## The schema and its tiers

`statecmp.FIELDS` expands every sim-maintained byte to one labelled address — all
eight object arrays (per slot), the eleven per-enemy phase arrays, the 32×32 tile
grid, the PRNG, both cooldown clocks and the scalar play variables. Render, sound
and screen scratch are excluded (a raw full-image diff would drown the signal).
Each field has a **tier**:

| Tier | Fields | Meaning |
|------|--------|---------|
| `CORE` | objects, enemy cooldowns, energy, tiles, discharge/meanie arrays, `$1335`, `$0C50` | the state the sim claims to reproduce frame-for-frame; a CORE divergence is a real model/ROM disagreement |
| `SWEEP` | cursor `$0090`, PRNG `$0C7B-$0C7F` | PRNG drives unreadable hyperspace/meanie landing coords (a by-design non-goal, § gameplay); cursor `$0090` sets enemy processing order — its divergence is the cursor-phase defect below, not an accepted artifact |
| `SCRATCH` | `$0014`, `$0C56-$0C58`, `$0C68`, `$0C76`, `$0CDD` | LOS/targeting bytes the ROM rewrites every scan; transient |

The instrument records the first frame each tier first disagrees and stops on the
first `CORE` divergence — the "very first disagreement" in the meaningful state.

## Running it

```bash
python -m driver.instrument 335 --frames 1200
```

Boots landscape 335 under warp with **no recording** (`NO_RECORD=1`), unfreezes the
enemy clock on both sides by clearing `$0CE5` bit7 (the "player has acted" gate),
then frame-locks and prints the per-tier first-divergence report (`emu=A`, `sim=B`).

### Follow mode (`--follow`)

By default the race stops at the first CORE divergence. With `--follow`, a CORE
divergence instead **reseeds the sim from live memory** (the same resync the live
driver does on a world-divergence) and the race continues to `--frames`, so the
instrument reports the whole *sequence* of CORE disagreements rather than just the
first. The report then adds the event count, the resync count, the min/median/max
frame gap between successive divergences (a small gap means the sim is persistently
off; a large one means it re-aligns between events) and the first dozen events with
their fields.

```bash
python -m driver.instrument 335 --frames 1200 --follow
```

## CORE divergence is a defect to drive to zero

The bar is **zero CORE divergence over the full race**. Fidelity is binary: a sim
that reproduces enemy phase 97% of the time is 0% correct on the outcome it decides,
because one rotation step of drift places a body into a gaze the planner modelled
empty. A CORE divergence is a model bug to eliminate — never a "residual" or
"readout artifact" to explain away.

Two mechanisms are pinned:

- **Frame→tick order skew (fixed).** `advance_frame` once ran the cooldown clock
  *before* the enemy sweep, committing a due enemy's rotation one frame early vs the
  ROM order (foreground `$1289` before raster `$9663`). Reordered to match; on ls0
  slot 0 `obj[0].h_angle`/`rotation_cd` led the ROM by one frame at frame 50, on 335
  slot 4 at frame 20 before the fix.
- **Cursor-phase sub-frame floor (ls42 first CORE, ~frame 50).** The live foreground
  sweeps the enemy cursor `$0090` at **~3.43 `update_enemies` calls/frame** (measured
  3-or-4, period-7 — not the model's 1 or full-8), so it matters *which* slot is
  processed *when*. The 3-vs-4 split each frame rides a sub-frame CPU-cycle
  accumulator that is **absent from the 64KB seed**, so a RAM-seeded sim drifts ±1
  frame on when each enemy's slot is swept — the `enemy.update_cd $0C30+N` divergence
  that cascades to `obj.h_angle`. This is a bounded ±1-frame floor, not a "residual":
  zero needs a cycle-accurate loop model plus a canonical sub-frame seed.

`driver/test_enemy_sim_divergence.py` (frame-locked, strict) is the gate: it must
reach zero CORE divergence. The PRNG `$0C7B-$0C7F` is the only by-design non-goal
(unreadable landing coords); the cursor `$0090` is not — it must track the ROM.
