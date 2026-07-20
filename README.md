# The Sentinel — bit-exact model + live driver

A ROM-faithful model of **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64, plus a live driver that plays the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) by keyboard and records an AVI.
Transition primitives are validated byte-for-byte against the real 6502 code (golden
fixtures, so CI proves them without the ROM); the enemy clock is gated frame-for-frame
against the running game by the divergence instrument.

The A\* player wins landscape 42 **live, on the real game** — 39 actions, final energy
11, verified by the ROM's own landscape-complete flag (`$0CDE` bit 6):

![A* player winning landscape 42 live in VICE](docs/media/ls42_astar_win.png)

```bash
python -m driver.play_player 42 --player astar   # live in VICE, records an AVI
python -m sentinel.astar_player 66               # offline (typed "0042" = seed 66)
```

## Layout

| Area | Path | Role |
|------|------|------|
| Model | `sentinel/` | standalone bit-exact forward model — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator). [docs/simulator.md](docs/simulator.md) |
| A\* player | `sentinel/astar_player.py` | weighted best-first search that plans a winning line and executes it. [docs/astar_player.md](docs/astar_player.md) |
| Reactive player | `sentinel/player.py` | tick-by-tick greedy player over the same `BasePlayer`. [docs/player.md](docs/player.md) |
| Driver | `driver/` | boot, enter a landscape, run memory-verified live keyboard operations (aim → fire → verify), record. Imports only `sentinel/`. [docs/driver.md](docs/driver.md) |
| Instrument | `driver/instrument.py`, `sentinel/statecmp.py` | frame-locked sim-vs-emulator divergence: seed the sim from the live image, step both one frame at a time, report the first disagreement. [docs/instrument.md](docs/instrument.md) |

## Fixtures (not distributed)

The game is copyrighted and is **not** included. Place your own copies at
`sentinel-gold.tap` (C64 tape image, used by the live driver) and
`out/sentinel_stage2.bin` (64 KB memory image of the loaded game, used only by the
`oracle`-marked tests that regenerate the goldens). Both are gitignored; tests that
need them auto-skip when absent.

## Setup and tests

```bash
pip install -r requirements.txt
pytest -n auto
```

The live driver additionally needs Docker and the `anarkiwi/asid-vice:latest` image
(build from https://github.com/anarkiwi/asid-vice).

## Docs

- [gameplay.md](docs/gameplay.md) — the game's rules and mechanics (ROM-derived spec).
- [simulator.md](docs/simulator.md) — the model's modules and golden validation.
- [astar_player.md](docs/astar_player.md) — search, candidate generators, cost model.
- [player.md](docs/player.md) — the reactive player: priorities, threat model, timing.
- [render_cost.md](docs/render_cost.md) — the `plot_world` redraw/settle frame cost.
- [driver.md](docs/driver.md) — boot/enter/record, keyboard aim → fire → verify.
- [instrument.md](docs/instrument.md) — the frame-locked divergence gate.
- [plan_fidelity.md](docs/plan_fidelity.md) — measured plan-vs-live error budget and ranked open problems.
