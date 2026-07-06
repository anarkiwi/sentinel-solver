# sentinel-solver

An automated solver for **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64. It plans a winning action sequence for a landscape on a standalone
bit-exact simulator (`sentinel/`), then drives the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) purely by keyboard input and
records the win as an AVI — verified by the game's own landscape-complete flag
(`$0CDE` bit 6). The simulator is validated byte-for-byte against the real 6502
code, frozen as golden fixtures so CI proves correctness without the ROM.

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

The code is in three cleanly separated layers, each its own top-level package:
the **simulator** (`sentinel/`) it all builds on, the **solver** (`solver/`)
that plans on the simulator, and the **driver** (`driver/`) that plays the real
game. The driver never imports the solver and the solver never imports the
driver; only the glue runners in `scripts/` wire the two together.

| Area | Files | Role |
|------|-------|------|
| Simulator | `sentinel/` | standalone, bit-exact forward model of the whole game — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator) |
| Solver | `solver/climb_greedy.py`, `solver/climb_search.py`, `solver/plan_game.py` | plan a winning climb + absorb sequence on the simulator: greedy height-first and the receding-horizon best-first lookahead (`climb_search`, the one that wins), over the `plan_game` keyboard-step adapter. Imports only `sentinel/` |
| Driver | `driver/boot.py`, `driver/kbd_aim.py`, `driver/sentinel_execute.py`, `driver/sentinel_state.py`, `driver/live_climb.py`, … | drive the real game by keystrokes in VICE, read live state (via `sentinel`) and record video. Solver-independent — executes a plan, never plans |
| Runners | `scripts/record_win_0042.py` | glue entry points that wire a solver plan into the driver |

## Simulator (`sentinel/`)

A standalone Python package that reproduces the game bit-for-bit — terrain
generation, line-of-sight, player actions, energy and enemy dynamics — with no
emulator, as the substrate for testing strategies.

```python
from sentinel import Game
g = Game.new(42)                 # generate the board from scratch
g.player_xy(), g.energy           # ((14, 27), 10)
g.step_enemies()                  # advance the world one round
g.player_sees(g.platform_xy())    # line-of-sight query
```

Every mechanic is validated byte-for-byte against the real 6502 code; the ROM is
used only as a test-time oracle, and those checks are frozen as golden fixtures so
CI proves correctness without the copyrighted image. See
[docs/simulator.md](docs/simulator.md).

## Fixtures (not distributed)

The game itself is copyrighted and is **not** included. Place your own copies at:

- `sentinel-gold.tap` — C64 tape image of The Sentinel (Firebird "gold" release),
  used only by the live driver and the video-record test
- `out/sentinel_stage2.bin` — 64KB C64 memory image of the loaded game (a raw dump
  of $0000-$FFFF after the game has loaded), used only by the `oracle`-marked
  `sentinel/` tests that regenerate the golden fixtures from the real 6502 code

Both paths are gitignored. Tests that need them auto-skip when absent; the
simulator, the planners and their tests run without either.

## Setup

```bash
pip install -r requirements.txt
```

The live driver additionally needs Docker and the `asid-vice:latest` image
(build from https://github.com/anarkiwi/asid-vice).

## Run

```bash
# plan a win with the best-first lookahead (landscape 0, depth 2; ~85s):
python3 solver/climb_search.py 0 2     # prints native_won True + the step plan

# the plan is validated by construction: it is built on the sentinel simulator,
# which is bit-exact vs the real 6502 code (golden-fixture CI, see below).

# drive the real game to the win and record it (Docker; ~3-20 min):
python3 scripts/record_win_0042.py      # -> renders/solver_run_*.avi
```

## Tests

```bash
pytest -n auto
```
