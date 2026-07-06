# sentinel-solver

An automated solver for **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64. It plans a winning action sequence for a landscape, validates the
plan against the real game code running in a headless 6502 emulator, then drives
the real game in [VICE](https://vice-emu.sourceforge.io/) (asid-vice) purely by
keyboard input and records the win as an AVI — verified by the game's own
landscape-complete flag (`$0CDE` bit 6).

## What's here

| Area | Files | Role |
|------|-------|------|
| Game state | `scripts/game_state.py` | read the game state from emulator memory |
| Game model | `scripts/game_model.py`, `scripts/enemy_dynamics.py` | forward simulation of rules, energy, enemy timing |
| Native LOS | `scripts/native_los.py` | fast Python port of the game's line-of-sight / sights-targeting |
| Planners | `scripts/solver*.py`, `scripts/climb_*.py`, `scripts/native_game.py` | plan a winning climb + absorb sequence |
| Engine oracle | `scripts/code_engine.py`, `scripts/validate_kbd_plan.py`, `scripts/climb_greedy_validate.py` | replay plans through the real game code headlessly (py65) |
| Live driver | `scripts/boot.py`, `scripts/kbd_aim.py`, `scripts/vice_execute.py`, `scripts/record_win_0042.py` | drive the real game by keystrokes in VICE and record video |
| Simulator | `sentinel/` | standalone, bit-exact forward model of the whole game (no emulator) |

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
