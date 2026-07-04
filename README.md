# sentinel-solver

An automated solver for **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64. It plans a winning action sequence for a landscape, validates the
plan against the real game code running in a headless 6502 emulator, then drives
the real game in [VICE](https://vice-emu.sourceforge.io/) (asid-vice) purely by
keyboard input and records the win as an AVI — verified by the game's own
landscape-complete flag (`$0CDE` bit 6).

## Gameplay model

Each landscape is a 32x32 tile board (`N=32`) with a height field of vertex
corners (`state.height[y][x]`, 0..11 nibble units) and a slope nibble per tile
that splits it into two triangular facets rather than a smooth quad
(`calculate_tile_slope` $2C7C). One **Sentinel** stands on a **platform** on
the highest tile; winning a landscape means absorbing the Sentinel and
transferring onto its platform tile (sets `$0CDE` bit 6, landscape-complete).

**Energy.** The player has one energy meter, 0..63 (`player_energy` $0C0A,
masked to 6 bits by `set_player_energy` $2148). Every object type has a fixed
energy value (`energy_in_objects` $214F): robot/sentry = 3, boulder = 2,
tree/meanie = 1, Sentinel = 4, platform = 0 (never absorbed or created).
Absorbing an object adds its value; creating one subtracts it (fails if it
would underflow). Up to 64 objects can exist at once (one 64-slot object
table).

**Actions**, all gated on line of sight to the target tile
(`check_for_line_of_sight_to_tile` $1CDD, via $1B46):
- `absorb (x,y)` — remove the object there, gain its energy.
- `create type (x,y)` — spend energy to place a tree/boulder/robot on a
  visible empty (or stackable) tile, if a slot is free.
- `transfer (x,y)` — move your point of view into one of your own robots you
  can see. This is the only way to change position; there is no "walking".
- `win (x,y)` — absorb the Sentinel, then transfer onto the platform tile it
  stood on.
- `hyperspace` — panic escape: creates a new robot on a random flat tile
  strictly below your current height and costs 3 energy (robot cost); the
  destination is PRNG-driven and not predictable offline. Hyperspacing while
  standing on the platform tile also wins.

**Line of sight** is a raytrace from the observer's eye toward the target: if
terrain rises above the ray before it reaches the target, sight is blocked. A
tile also can't be seen if its surface sits above the observer's eye (you
can't "look up" at ground level — the ROM's $1D2E check), though this is
waived when aiming at the *top* of an object such as a tree. The raytrace
must follow the true slope-facet surface, not a bilinear average across it,
or it will over-estimate what is visible.

**Climbing.** The only way to raise your eye height (and thus extend what you
can see and absorb) is to build a boulder, transfer onto it, build another
boulder from up there, and repeat — a staged climb of alternating
create+transfer. Builds are limited to a small height slack above the current
eye (roughly 2 units) before another transfer-up is required. This
create/transfer stacking, not terrain walking, is how the player eventually
gets a sight line over the landscape to the Sentinel's platform.

**Enemies.** Sentries and the Sentinel itself rotate and scan a field of view;
if an enemy's LOS reaches the player or an object, it drains 1 energy per
tick or downgrades objects (robot -> boulder -> tree) on a cooldown cycle
(`update_enemies` $16B5, `check_if_enemy_can_see_object` $1887). A tile is
"safe" from a given enemy only if the enemy's LOS to it is blocked from every
angle it could rotate to, not just its current facing.

## What's here

| Area | Files | Role |
|------|-------|------|
| Game state | `scripts/game_state.py` | read the game state from emulator memory |
| Game model | `scripts/game_model.py`, `scripts/enemy_dynamics.py` | forward simulation of rules, energy, enemy timing |
| Native LOS | `scripts/native_los.py` | fast Python port of the game's line-of-sight / sights-targeting |
| Planners | `scripts/solver*.py`, `scripts/climb_*.py`, `scripts/native_game.py` | plan a winning climb + absorb sequence |
| Engine oracle | `scripts/code_engine.py`, `scripts/validate_kbd_plan.py`, `scripts/climb_greedy_validate.py` | replay plans through the real game code headlessly (py65) |
| Live driver | `scripts/boot.py`, `scripts/kbd_aim.py`, `scripts/vice_execute.py`, `scripts/record_win_0042.py` | drive the real game by keystrokes in VICE and record video |

## Fixtures (not distributed)

The game itself is copyrighted and is **not** included. Place your own copies at:

- `sentinel-gold.tap` — C64 tape image of The Sentinel (Firebird "gold" release)
- `out/sentinel_stage2.bin` — 64KB C64 memory image of the loaded game
  (a raw dump of $0000-$FFFF taken after the game has fully loaded; the py65
  harness `scripts/_emu.py` executes the game code from it)

Both paths are gitignored. Tests that need them auto-skip when absent.

## Setup

```bash
pip install -r requirements.txt
```

The live driver additionally needs Docker and the `asid-vice:latest` image
(build from https://github.com/anarkiwi/asid-vice).

## Run

```bash
# plan (fast, deterministic; landscape "0042" is internal seed 66):
python3 -c "import sys;sys.path.insert(0,'scripts');import climb_greedy as c;\
g=c.plan_greedy(66,verbose=False,toward_plat=True);print(g.native_won,g.energy,len(g.steps))"

# validate the plan against the real game code (slow, ~5 min):
python3 scripts/_finalize66.py          # writes out/kbd_greedy_0066.json

# drive the real game to the win and record it (Docker; ~3-20 min):
python3 scripts/record_win_0042.py      # -> renders/solver_run_0042.avi
```

## Tests

```bash
pytest -n auto
```
