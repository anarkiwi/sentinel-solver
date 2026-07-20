# The Sentinel — bit-exact model + live driver

A ROM-faithful model of **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64, plus a live driver that plays the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) by keyboard input and records
it as an AVI. Transition primitives are validated byte-for-byte against the real
6502 code (golden fixtures, so CI proves them without the ROM); live frame-cadence
and aim-cost fidelity is gated frame-for-frame by the divergence instrument
([docs/instrument.md](docs/instrument.md)) — any gap is a defect to close, not
tolerated error.

## Layout

| Area | Path | Role |
|------|------|------|
| Model | `sentinel/` | standalone, bit-exact forward model of the game — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator). See [docs/simulator.md](docs/simulator.md). |
| Player | `sentinel/player.py` | reactive tick-by-tick greedy player over the model (`python -m sentinel.player 0`). See [docs/player.md](docs/player.md). |
| A* player | `sentinel/astar_player.py` | weighted best-first search player that plans a winning line, then executes it (`python -m sentinel.astar_player 66`); shares `sentinel/playerbase.py` (`BasePlayer`) with the reactive player. See [docs/astar_player.md](docs/astar_player.md). |
| Driver | `driver/` | boot the game, enter a landscape, and run memory-verified live keyboard operations (aim → fire → verify), recording an AVI. Imports only `sentinel/`. See [docs/driver.md](docs/driver.md). |
| Instrument | `driver/instrument.py` + `sentinel/statecmp.py` | frame-locked sim-vs-emulator divergence: seed the sim from the live image, step both one frame at a time, and report the first state disagreement decoded to a named field (`python -m driver.instrument 335`). See [docs/instrument.md](docs/instrument.md). |
| Docs | `docs/` | rules and subsystem references (below). |

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

The live driver additionally needs Docker and the `anarkiwi/asid-vice:latest` image
(build from https://github.com/anarkiwi/asid-vice).

## Tests

```bash
pytest -n auto
```

## Docs

- [docs/gameplay.md](docs/gameplay.md) — the game's rules and mechanics.
- [docs/simulator.md](docs/simulator.md) — the `sentinel/` model's modules and validation.
- [docs/player.md](docs/player.md) — the reactive greedy player: priorities, threat model, timing.
- [docs/astar_player.md](docs/astar_player.md) — the A* planning player: search, candidate generators, cost model, shared `BasePlayer`.
- [docs/render_cost.md](docs/render_cost.md) — the `plot_world` render/settle frame-cost model that prices redraw and transfer settle.
- [docs/driver.md](docs/driver.md) — the live driver: boot/enter/record, keyboard aim → fire → verify, container plumbing.
- [docs/instrument.md](docs/instrument.md) — the shared frame-locked divergence instrument: schema, tiers, first-disagreement report.
- [docs/plan_fidelity.md](docs/plan_fidelity.md) — current state: the clocks are exact and the A* player wins ls42 offline (live: 14 actions); measured error budget and ranked open problems.
</content>
