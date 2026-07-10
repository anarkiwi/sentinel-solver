# The Sentinel — bit-exact model + live driver

A ROM-faithful model of **The Sentinel** (Geoff Crammond, Firebird, 1986) on the
Commodore 64, plus a live driver that plays the real game in
[VICE](https://vice-emu.sourceforge.io/) (asid-vice) by keyboard input and records
it as an AVI. The model is validated byte-for-byte against the real 6502 code,
frozen as golden fixtures so CI proves correctness without the ROM.

## Layout

| Area | Path | Role |
|------|------|------|
| Model | `sentinel/` | standalone, bit-exact forward model of the game — terrain, LOS/aim, actions, energy, enemies, landscape generation (no emulator). See [docs/simulator.md](docs/simulator.md). |
| Driver | `driver/` | boot the game, enter a landscape, and run memory-verified live keyboard operations (aim → fire → verify), recording an AVI. Imports only `sentinel/`. See [docs/driver.md](docs/driver.md). |
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

The live driver additionally needs Docker and the `asid-vice:latest` image
(build from https://github.com/anarkiwi/asid-vice).

## Tests

```bash
pytest -n auto
```

## Docs

- [docs/gameplay.md](docs/gameplay.md) — the game's rules and mechanics.
- [docs/simulator.md](docs/simulator.md) — the `sentinel/` model's modules and validation.
- [docs/driver.md](docs/driver.md) — the live driver: boot/enter/record, keyboard aim → fire → verify, container plumbing.
</content>
