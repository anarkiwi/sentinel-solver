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

## What it finds on 335

Seeded from the live board (`energy=10`, `player_slot=62`), racing frame-for-frame:

| Tier | First frame | Fields |
|------|-------------|--------|
| `SWEEP` | 1 | `cursor $0090`, `prng[0..4] $0C7B-$0C7F` |
| `CORE` | 20 | `obj[4].h_angle $09C4` (emu 0, sim `$14`), `enemy[4].rotation_cd $0C2C` (emu 0, sim `$C8`) |
| `SCRATCH` | 20 | `fov_relative_h $0C57`, `targeted_slot $0C58` |

- **SWEEP at frame 1** is by construction: the sim advances the PRNG/cursor a
  whole cursor sweep per frame while the ROM advances them one slot per frame.
- **CORE at frame 20 is the real disagreement.** Enemy 4's rotation cooldown
  reaches 0 on both, but the **sim immediately applies the rotation** in that same
  frame — `obj[4].h_angle` steps by the `$9D37` rotation unit (+`$14`) and
  `rotation_cd` reloads to 200 (`$C8`) — whereas the **ROM has not rotated it yet**
  (`h_angle` still 0, `rotation_cd` still 0): the running game defers the rotation
  until its once-per-frame round-robin cursor lands on slot 4. The sim fires an
  enemy action the instant its cooldown expires; the ROM spreads it across the
  cursor sweep. This is the frame→tick cadence skew the model was flagged for, now
  pinned to the exact frame and byte.
