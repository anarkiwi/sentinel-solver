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
| `SWEEP` | cursor `$0090`, PRNG `$0C7B-$0C7F` | the ROM advances these once per frame; the sim's `advance_frame` collapses a whole 8-slot cursor sweep into one frame, so they diverge by construction |
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

## The frame→tick cadence skew (fixed) and the residual

The first CORE divergence the instrument originally pinned was a **frame→tick
cadence skew**: `advance_frame` ran the cooldown clock *before* the enemy sweep, so
an enemy the cooldown tick made due was acted on in that *same* frame, whereas the
ROM runs the foreground sweep (`$1289`) *before* that frame's raster-IRQ cooldown
tick (`$9663`) and so defers the action to the next frame. The sim's enemy therefore
rotated one frame early — on landscape 0, slot 0's facing `obj[0].h_angle $09C0` and
`rotation_cd $0C28` led the ROM by one frame at frame 50; on 335, slot 4 (`$09C4` /
`$0C2C`) at frame 20.

The fix reorders `advance_frame` to sweep-then-cooldown (`sentinel/enemies.py`),
matching the ROM's within-frame order. The rotation now commits on the same frame the
ROM commits it, and the first CORE divergence moves well past the old point (0: 50 →
84; 335: 20 → 32).

What remains at that later frame is a **readout artifact, not a model error**:

| Tier | First frame (ls 0 / 335) | Fields |
|------|--------------------------|--------|
| `SWEEP` | 1 | `cursor $0090`, `prng[0..4] $0C7B-$0C7F` |
| `CORE` | 84 / 32 | `enemy[N].update_cd $0C30+N` (emu `$04`, sim `$01`) |
| `SCRATCH` | 51 / 21 | `fov_relative_h $0C57` |

- **SWEEP at frame 1** is by construction: the sim advances the PRNG/cursor a
  whole cursor sweep per frame while the ROM advances them one slot per frame.
- **The residual CORE is a mid-routine interrupt split.** The ROM's foreground
  `consider_enemy_state` reloads the enemy's `update_cooldown` to `$04` *early* in the
  routine and commits the rotation at its *end*. The per-frame raster marker `$9630`
  the instrument locks to falls inside that routine on the frame it fires, so a live
  read catches `update_cd` already reloaded (`$04`) while the sim — which applies the
  whole routine atomically one frame later — still holds the pre-reload `$01`. Both
  worlds reconverge on the very next frame. It is a one-byte, self-healing snapshot
  phase of an internal cooldown, cycle-timing dependent (a different boot interrupts
  at a different instruction), and carries none of the enemy's gameplay-visible facing
  state — which now tracks the ROM frame-for-frame.
