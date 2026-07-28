***Claude - including Fable - was unable to solve this simple game from 1986 without my help. Is what it is.***


# The Sentinel — bit-exact model + live driver

A ROM-faithful model of **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64, plus a live driver that plays the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) by keyboard and records an AVI.
Transition primitives are validated byte-for-byte against the real 6502 code (golden
fixtures, so CI proves them without the ROM); the enemy clock is gated frame-for-frame
against the running game by the divergence instrument.

The phase player wins **live, on the real game**, verified by the ROM's own
landscape-complete flag (`$0CDE` bit 6) — landscape 335 in 66 actions, final energy 25:

![the phase player winning landscape 335 live in VICE](docs/media/ls335_phase_win.png)

```bash
python -m driver.play_player 335        # live in VICE, records an AVI
python -m sentinel.phase_player 335     # offline, same board
```

| landscape | enemies | offline | live |
|---|---|---|---|
| 0 | 1 | 16 actions | — |
| 42 | 2 | 35 | — |
| 60 | 7 | 55 | — |
| 110 | 3 | 53 | — |
| 298 | 7 | 34 | — |
| 321 | 7 | 78 | — |
| 335 | 7 | 73 | **66 actions** |
| 373 | 7 | 65 | — |

A landscape is identified by one number: the one a player types on the keypad. Every
tool here — `driver.play_player`, `sentinel.phase_player`, `sentinel.player`,
`sentinel.isoview` — takes exactly that number, and `Game.typed(335)` builds the same
board offline.

## Layout

| Area | Path | Role |
|------|------|------|
| Model | `sentinel/` | standalone bit-exact forward model — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator). [docs/simulator.md](docs/simulator.md) |
| Phase player | `sentinel/phase_player.py` | primary planner: freedom first, then convert — no cost model, gaps found by forward simulation. Wins all eight measured boards. [docs/phase_player.md](docs/phase_player.md) |
| Landscape analyzer | `sentinel/landscan.py` | enemy count and terrain shape per landscape, to match one board to another. |
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
- [phase_player.md](docs/phase_player.md) — the phase player: freedom first, then convert; the planner that wins ls335 from entry.
- [fast_iteration.md](docs/fast_iteration.md) — checkpointing a stalled tick so a planner change is judged in milliseconds, not a board replay.
- [player.md](docs/player.md) — the reactive player: priorities, threat model, timing.
- [render_cost.md](docs/render_cost.md) — the `plot_world` redraw/settle frame cost.
- [driver.md](docs/driver.md) — boot/enter/record, keyboard aim → fire → verify.
- [instrument.md](docs/instrument.md) — the frame-locked divergence gate.
- [human_clock.md](docs/human_clock.md) — recovering exactly what an action cost from the recorded enemy clock, and what that grades.
- [plan_fidelity.md](docs/plan_fidelity.md) — measured plan-vs-live error budget and ranked open problems.
