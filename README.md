# sentinel-solver

An automated solver for **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64. It plans a winning action sequence for a landscape on a standalone
bit-exact simulator (`sentinel/`), then drives the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) purely by keyboard input and
records the win as an AVI — verified by the game's own landscape-complete flag
(`$0CDE` bit 6). The simulator is validated byte-for-byte against the real 6502
code, frozen as golden fixtures so CI proves correctness without the ROM.

## Layout

Three cleanly separated layers, each its own top-level package. The driver never
imports the solver and the solver never imports the driver; only the runners in
`scripts/` wire them together.

| Area | Path | Role |
|------|------|------|
| Simulator | `sentinel/` | standalone, bit-exact forward model of the whole game — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator). See [docs/simulator.md](docs/simulator.md). |
| Solver | `solver/` | plan a winning climb + absorb sequence on the simulator: weighted-A\* macro planner (`astar_planner.py`, `macros.py`) over `plan_game.py`, with `cost.py`, `launch.py`, `gaze.py`, `search_node.py`. Imports only `sentinel/`. |
| Driver | `driver/` | boot the game, enter a landscape, and run memory-verified live keyboard operations. Solver-independent. See [docs/driver.md](docs/driver.md). |
| Runners | `scripts/` | glue that runs the plan loop against the simulator (`run_plan_simulated.py`) or the live game (`run_plan_live.py`). |
| Docs | `docs/` | rules, strategy, subsystem references (below). |

## Fixtures (not distributed)

The game is copyrighted and is **not** included. Place your own copies at:

- `sentinel-gold.tap` — C64 tape image (Firebird "gold"), used by the live driver
  and the video-record test.
- `out/sentinel_stage2.bin` — 64KB C64 memory image of the loaded game, used only
  by the `oracle`-marked `sentinel/` tests that regenerate the golden fixtures.

Both paths are gitignored; tests that need them auto-skip when absent.

## Setup

```bash
pip install -r requirements.txt
```

The live driver additionally needs Docker and the `asid-vice:latest` image
(build from https://github.com/anarkiwi/asid-vice).

## Run

```bash
# plan a win offline with the weighted-A* macro planner (landscape 0; ~55s):
python3 solver/astar_planner.py 0            # prints won=True + peak eye + step count

# run that plan against the simulator as a TICK-ACCURATE "real game" (no emulator):
python3 scripts/run_plan_simulated.py 0      # reports WON / lost + the climb log

# win landscape 0 live and record it (Docker; first run ~3-20 min incl. tape load):
python3 scripts/run_plan_live.py --digits 0000 --video-name ls0_win.avi
```

The live landscape is chosen by the 4 typed code digits (`--digits`); code `0000`
is landscape 0. The recording lands at `renders/<video-name>` (default
`solver_run_<digits>.avi`). The script prints `WIN VERIFIED PASS/FAIL` (`$0CDE`
bit 6), the per-step log, and the AVI's validated size/frame count; exit status is
0 on a verified win. Options: `--max-seconds` (wall cap, default 1500),
`--max-replans` (default 4), `--video-name`. Env: `NO_RECORD=1` skips the AVI;
`BINMON_HOST=<ip>` overrides the auto-detected container bridge IP. View with
`ffplay renders/ls0_win.avi` (or any AVI player).

The planner wins landscape 0 offline and against the tick-accurate simulator
(on-platform, zero-drain). Remaining work toward reliable *live* wins and other
landscapes is in [docs/planner.md](docs/planner.md).

## Tests

```bash
pytest -n auto
```

## Docs

- [docs/gameplay.md](docs/gameplay.md) — the game's rules and mechanics.
- [docs/strategy.md](docs/strategy.md) — how a landscape is won (the principles a planner must follow).
- [docs/simulator.md](docs/simulator.md) — the `sentinel/` simulator modules and validation.
- [docs/driver.md](docs/driver.md) — the live driver: boot/enter/record, keyboard aim → fire → verify, container plumbing.
- [docs/planner.md](docs/planner.md) — the planner: design, build plan, status/outstanding work, and the superseded design.
