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

**Time is continuous.** There are no turns. The world clock runs the whole time
the player is doing anything — panning the sights, building, absorbing,
transferring — and while idle; enemies rotate, target and drain on that clock
regardless of what the player is doing. An action is therefore not an instant:
it spans many enemy rounds, and there is no "mid-action" pause in which the
world is frozen. A sequence like build-boulder-then-build-synthoid is a stretch
of real time in which the enemies keep scanning and draining throughout.

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
drain-cooldown period or downgrades objects (robot -> boulder -> tree) on a
cooldown cycle (`update_enemies` $16B5, `check_if_enemy_can_see_object` $1887).
A tile is "safe" from a given enemy only if the enemy's LOS to it is blocked from
every angle it could rotate to, not just its current facing.

## Strategy

These are consequences of the rules above, not new mechanics — the principles a
correct planner must follow. Enumerated so we can extend the list.

1. **Exposure is a timed resource race, not a hard veto.** Because time is
   continuous, once an enemy has line of sight the player is drained ~1 energy per
   drain-cooldown for as long as that sightline holds, and every action costs time
   during which the drain keeps running. Survival is a race between banked energy
   and the drain over the time the planned actions take: a productive sequence —
   create a boulder, create a synthoid, transfer up — *can* be completed while
   being drained, provided the player has enough energy banked to pay the drain
   across its duration. What does **not** work is trying to *recover* energy by
   absorbing low-value objects while drained: a tree gives +1, but its aim+absorb
   spans several drain periods, so chasing trees to catch up nets negative — the
   absorb rate cannot exceed a continuous drain when each absorb spends multiple
   drain intervals. You only come out ahead by absorbing *faster than you are
   drained*, in practice by absorbing the **drainer itself**: absorbing the
   Sentinel (or a sentry) removes the source and ends the drain as it lands, so an
   endgame absorb can be driven straight through a drain even from low energy. The
   other escape is to break line of sight (transfer to an unseen robot, or
   hyperspace). So budget exposure as a timed energy cost over the action window —
   spendable against a sufficient buffer, or against a move that ends the drain —
   but never treat a seen tile as free, and never fund a plan on tree-refueling
   done while being drained.

2. **Only line of sight matters — there is no "danger ring."** A tile's safety is
   determined solely by whether the Sentinel (at any facing it can rotate to) has
   line of sight to it, never by how many tiles away the plinth is. A tile can be
   far from the plinth yet in full view (deadly), or immediately adjacent to it yet
   occluded / behind the Sentinel's back (safe). Standing on a boulder right next
   to the Sentinel while it is not looking at you is perfectly fine — indeed a
   boulder adjacent to the plinth, built and occupied while the Sentinel faces
   away, is a valid *penultimate winning position*: from there you absorb the
   Sentinel and take its platform. Gate builds and footholds on an actual
   line-of-sight/gaze query to the specific tile (`sentinel/threat.py`), not on
   Chebyshev distance to the platform.

3. **Prefer high, terrain-covered edge and corner tiles.** Centrality is a
   non-signal: a single rotating enemy points at any given tile for the same ~8%
   of its rotation (its FOV cone ÷ 256) wherever that tile sits, so being "in the
   middle" does not make you more seen — only occlusion does. What genuinely
   favours an edge or corner tile is geometry:
   - **Cover is easier to hold.** Safety is "the enemy's one bearing to me is
     blocked at every facing it can reach." At a corner the map boundary is at your
     back and all terrain, occluders and enemies lie in a ≤90° quadrant in front of
     you, so a single occluding rise can make the tile permanently safe; in the
     open centre the threat bearing can come from any of 360°.
   - **Your climb stack stays hidden.** The boulders/synthoids of a climb are
     exposed objects too (seen → downgraded → meanie spawn). A corner lets the
     stack grow in the Sentinel's occlusion shadow instead of out in the open —
     this, not a distance "ring," is the real reason not to build in the Sentinel's
     face.
   - **Longest reach.** From a corner the board extends up to its full ~45-tile
     diagonal in front of you, giving the longest unobstructed sightlines — best
     for surveying, reaching a distant platform, and firing the long-range endgame
     absorb from afar. From the centre the maximum reach in any direction is only
     ~16 tiles.
   - **Fewer enemies bear on you.** On multi-sentry maps a corner tile often sits
     outside some sentries' arc entirely.

   The catch: edge tiles are often **low**, and low tiles have short,
   easily-blocked sightlines — useless for the win unless you can climb high there.
   And the farther the launch tile is from the platform, the thinner and more
   occludable the sightline to it, so a corner launch only pays off when that long
   diagonal to the plinth is genuinely clear. Net: the ideal launch tile is a
   **high, terrain-covered corner with a clear diagonal to the plinth**, scored by
   LOS and occlusion — the opposite of what a centrality or proximity heuristic
   would pick.

4. **Place and time builds against the enemy's predictable gaze precession.**
   Enemy rotation is deterministic: an idle enemy adds a fixed per-enemy step to
   its facing every ~200 ticks (`rotate_enemy` $1805, `ROTATION_SPEED_TABLE`) and
   scans a ±10-unit cone. A fixed step added mod-256 does not oscillate — it
   *precesses*, walking the compass in a knowable, periodic sequence, so for any
   tile you can compute exactly when the gaze will next fall within the cone of its
   bearing and how long the gap is until then. This converts "exposed at some
   facing" into "safe for a schedulable window": permanently-occluded tiles are
   ideal but rare, and anticipating the precession lets you safely use the many
   high footholds that are only *briefly* visible, during the long intervals the
   gaze is elsewhere. It is the lever that keeps the candidate set from collapsing
   near good high ground. Combined with strategy 1, a dwell-and-build is safe iff
   the whole action sequence — priced in ticks (`sentinel/actioncost.py`) — fits
   inside the gaze gap and is started just after the gaze sweeps past; since time
   is continuous with no mid-action pause, a build that overruns the gap is caught.
   So optimize placement to tiles whose safe window ≥ the planned build duration,
   and phase the build into that window. With multiple sentries the safe interval
   is the *intersection* of every enemy's gaze gaps, so anticipate each precession
   to find a tile-and-time where all gazes point away at once. Two schedule levers:
   an enemy locked onto a drainable target stops precessing (it rotates only while
   idle), so a decoy elsewhere can hold a gaze away — and, conversely, a careless
   exposed build can hijack a gaze onto you. Because all of this is deterministic,
   the planner must forecast the gaze schedule (`enemies.step` /
   `threat.ticks_until_seen`) rather than veto every ever-visible tile with a
   static mask.

5. **Gain height early, and batch boulders to gain it in fewer transfers.** Height
   is the win resource: your eye height sets how far you can see and absorb, you
   cannot look up (a tile above your eye is unseeable), and the endgame is a
   long-range line-of-sight shot fired from launch height — so every unit of eye
   height unlocks more of the map (fuel, covered high ground, the platform itself).
   Given continuous time, *when* and *how* you climb matters:
   - **Transfers dominate the time and the risk.** A transfer is the most
     expensive action (hyperspace-tune wait + full redraw, ~300 ticks) and is also
     the moment your position changes — a fresh landing that must be safe. So
     maximize *height gained per transfer*: the transfer count sets both the
     cumulative drain time and the number of exposed landings you must schedule.
   - **Use more than one boulder at once.** A boulder raises a tile half a unit and
     builds are capped ~2 units above your eye, so you can stack ~4 boulders on a
     target tile in a single dwell, cap it with a synthoid, and transfer once to
     gain the full ~2 units — instead of build-one/transfer/build-one, which pays a
     full transfer for every half unit. Batching amortizes the fixed per-transfer
     cost and concentrates the whole climb into one safe tile and one gaze gap
     (strategy 4) rather than four landings and four windows. The batch size is
     bounded by the build slack and by how much build time fits inside the gaze
     gap — stack as many boulders as the safe window affords, phased into it.
   - **Do it early.** The opening is often the cheapest exposure window (enemies
     not yet precessed/locked onto you, energy buffer still full to fund the
     climb), and height compounds: the sooner you see far, the sooner you can spot
     fuel, covered high ground and the endgame launch tile and plan the rest of the
     route — whereas staying low keeps you blind, reactive and forced into whatever
     few (often exposed) tiles a short sightline reaches, bleeding drain across many
     small moves. Early, batched height converts to a shorter, safer, better-
     informed remainder.

## What's here

The code is in three cleanly separated layers, each its own top-level package:
the **simulator** (`sentinel/`) it all builds on, the **solver** (`solver/`)
that plans on the simulator, and the **driver** (`driver/`) that plays the real
game. The driver never imports the solver and the solver never imports the
driver; only the glue runners in `scripts/` wire the two together.

| Area | Files | Role |
|------|-------|------|
| Simulator | `sentinel/` | standalone, bit-exact forward model of the whole game — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator) |
| Solver | `solver/climb_search.py`, `solver/plan_game.py` | plan a winning climb + absorb sequence on the simulator: a receding-horizon best-first lookahead (`climb_search`) over the `plan_game` keyboard-step adapter. Imports only `sentinel/` |
| Driver | **`driver/core.py`** (the foundation `SentinelDriver` + the shared plumbing: container/bridge-IP, monitor-resilience, full 64 KB live-image read, landscape entry by title-menu navigation, and the live sights-ray probe `probe_tile`), `driver/boot.py` (boot → title + reusable `boot.vsf`, snapshot save/load), `driver/kbd_aim.py` (aim), `driver/sentinel_state.py` (read live state → `GameState`, `verify_entry`), `driver/sentinel_execute.py` (action keys, the `Executor` accessors, and the ROM memory-delta `verify`) | boot the game, enter an arbitrary landscape, and run memory-verified operations (aim a tile, create/absorb/transfer/hyperspace). Solver-independent — executes operations, never plans |
| Runners | `scripts/run_plan_simulated.py`, `scripts/run_plan_live.py` | run the SAME bare planner loop (resync → decide → execute → replan) against, respectively, the simulator as a tick-accurate "real game" and the actual game in asid-vice. Both verify the win by the on-platform condition / ROM win flag |

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
# plan a win offline with the weighted-A* macro planner (landscape 0; ~55s):
python3 solver/astar_planner.py 0     # prints won=True + peak eye + step count

# run that planner against the simulator as a TICK-ACCURATE "real game": enemies
# advance every game round (drain/rotate/downgrade) between and during actions,
# and the loop resyncs + replans each step (~55s, no emulator):
python3 scripts/run_plan_simulated.py 0     # reports WON / lost + the climb log
```

### Live run + video record

Drive the same plan loop in the real game (asid-vice, in Docker) and record the
run as an AVI. The landscape is chosen by the 4 typed code digits (`--digits`);
code `0000` is landscape 0. Boot + landscape entry are handled by the driver,
which auto-caches a code-entry snapshot under `renders/` to skip the ~50s tape
load after the first run.

```bash
# win landscape 0 live and record it (Docker; first run ~3-20 min incl. tape load):
python3 scripts/run_plan_live.py --digits 0000 --video-name ls0_win.avi

# the recording lands at renders/ls0_win.avi (default name: solver_run_<digits>.avi).
# the script prints WIN VERIFIED PASS/FAIL ($0CDE bit6), the per-step log, and the
# AVI's validated size/frame count. exit status is 0 on a verified win.
```

Options: `--max-seconds` (wall cap, default 1500), `--max-replans` (live resync +
re-plan budget, default 4), `--video-name` (AVI basename under `renders/`).
Env: `NO_RECORD=1` skips the AVI; `BINMON_HOST=<ip>` overrides the auto-detected
container bridge IP if the binmon connection can't be reached.

View the result:

```bash
ffplay renders/ls0_win.avi          # or any AVI player; mplayer/vlc also work
```

Sharing one loop makes the two runners a controlled comparison: `run_plan_live`
(`scripts/`) is just glue — it wires the solver's decision to the driver's
booted-game session (`driver.core.boot_and_play`) and per-step keyboard executor
(`driver.sentinel_execute.perform_step`), owning no emulator driving itself.

The planner wins landscape 0 offline and against the tick-accurate simulator
(on-platform, zero-drain). Remaining work toward reliable *live* wins and other
landscapes is tracked in [docs/outstanding-issues.md](docs/outstanding-issues.md).

## Tests

```bash
pytest -n auto
```
