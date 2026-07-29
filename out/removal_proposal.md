# Removal proposal

All line numbers are as of commit `5ed6a98`. `sentinel/playerbase.py`, `sentinel/phase_player.py`,
`sentinel/landtable.py` and `docs/` were under concurrent edit while this was written — re-run the
reachability script (see Method) before acting on items S1, S4, S5.

## Summary

Baseline: **14,043 lines of source** (`sentinel/` + `driver/`, excluding `test_*.py` and `conftest.py`)
and **8,257 lines of test** (`test_*.py`, `sentinel/tests/` helpers, `conftest.py`).

| | lines | share |
|---|---|---|
| Source proposed for removal | **1,188** | 8.5% |
| Test proposed for removal | **619** | 7.5% |
| **Total** | **1,807** | 8.1% |
| Committed JSON artifacts proposed for removal | **551 KB** | — |

Of that, **853 lines (526 source / 327 test) are low-risk and mechanical** — dead code with no caller
anywhere, orphan `main()` blocks, always-default parameters, a pytest file containing no tests, and
committed output artifacts nothing reads. The remaining ~950 lines are two judgement calls
(`isoview.py`, `ckpt.py`): working, well-tested diagnostic tools that nothing in the play path or the
correctness proof depends on.

Headline items:

1. **`sentinel/isoview.py` + its test (760 lines)** — a documented isometric SVG renderer whose only
   in-tree consumer is a `--diagram` flag on a test helper. Nothing about correctness touches it.
2. **A dead 133-line scheduling calendar in `playerbase.py`** (`_earliest_start` and its five
   exclusive callees), plus `_drain_units`/`_affords_drains`/`advance_phases` — 180 lines, zero
   references anywhere in source, tests, or docs-as-code.
3. **`driver/test_video_record.py` (119 lines)** — collected by pytest, contains **no test function**,
   and imports `vice_driver` from a sibling checkout at module scope.
4. **551 KB of committed JSON** (`human_wins/ls{0,42,335}_audit.json`, `ls110.json`) written by a
   helper's `main()` and read by nothing.
5. **`timing_registry.UNVALIDATED_PIN` (69 lines)** — verified to be exactly
   `{n for n, m in REGISTRY.items() if m["class"] == UNVALIDATED}`, computed from a dict three lines
   above it. It is a second-edit speed bump, not an independent oracle.

**The largest apparent duplication in the repo — ~1,100 lines of pure-Python/numba mirroring across
`los`/`los_jit`, `enemies`/`enemies_jit`, `projector`/`projector_jit`, `relative`/`enemies_jit` — is
explicitly NOT proposed.** See "Explicitly NOT proposed", item N1. It is the thing that proves the
model, and merging it is a rewrite, not a removal.

### Coverage

Measured now (`pytest -n auto -k "not regenerate" --cov=sentinel --cov=driver`, 804 passed, 5 skipped,
5 xfailed, 1 xpassed, 346 s wall):

| scope | stmts | cover |
|---|---|---|
| everything | 12,311 | **70.9%** |
| source, non-test | 8,892 | 62.9% |
| `sentinel/` source **excluding the three `_jit` modules** | 4,559 | **86.5%** |
| the three `_jit` modules | 1,735 | 8–22% |
| `driver/` source | 1,632 | 35.8% |
| test code itself | 3,419 | 91.9% |

Two facts the raw 70.9% hides. First, `coverage.py` cannot trace inside numba-compiled functions, so
`los_jit` (7.6%), `projector_jit` (14.3%) and `enemies_jit` (21.6%) read as untested when
`test_los_jit.py` / `test_projector_jit.py` / `test_enemies_jit.py` exercise them exhaustively — a
measurement artifact, not a gap. Second, `driver/`'s 35.8% is the VICE/Docker-gated live path, which
cannot run in CI at all. **The only figure that meets the 85% bar is the sentinel-excluding-jit one,
and no CI job enforces coverage** — `.github/workflows/ci.yml` runs `pytest -n auto` with no `--cov`
and no `--cov-fail-under`.

**Net coverage effect of this proposal: positive.** Every source item except `isoview.py` removes
statements that are currently 0–49% covered (`landtable.main` 0%, `core.main` 0%,
`sentinel_state.main` 0%, `mem_image` 0%, the numba-absent fallbacks 0%, `landscan.py` 46%).
`isoview.py` is 99% covered and removing it with its test is coverage-neutral. `test_video_record.py`
is 23% covered and its removal raises the figure. Accepting the whole proposal moves
sentinel-excluding-jit from 86.5% to roughly 88%.

---

## Ranked table

Ranked by lines-saved-per-unit-risk (low = 1, medium = 2, high = 3), not raw size.

| # | Item | Source | Test | Risk | Ratio | Justification |
|---|---|---:|---:|---|---:|---|
| S9 | `sentinel/isoview.py` + `test_isoview.py` | 662 | 98 | med | 253 | Diagnostic SVG renderer; no correctness path reaches it; outputs are gitignored |
| S1 | `playerbase.py` dead scheduling calendar + drain helpers | 180 | 0 | low | 180 | Six mutually-exclusive methods, zero references in the whole tree |
| T1 | `driver/test_video_record.py` | 0 | 119 | low | 119 | Collected by pytest, contains no `test_*` function |
| S2 | Orphan `main()` blocks (`landtable`, `core`, `sentinel_state`) | 69 | 0 | low | 69 | No doc, script, CI job or test invokes them; already 0% covered |
| T3 | `UNVALIDATED_PIN` + `test_unvalidated_debt_does_not_grow` | 0 | 69 | low | 69 | Verified identical to a comprehension over `REGISTRY` 3 lines above |
| S8 | `sentinel/landscan.py` + `test_landscan.py` | 95 | 28 | med | 61 | Zero source callers; its only CLI filter flag is permanently on |
| T4 | Strictly-subsumed tests (7 sites) | 0 | 63 | low | 63 | Each has a stronger test in the same or an adjacent file |
| T7 | `sentinel/tests/ckpt.py` + `test_ckpt.py` | 0 | 194 | med | 97 | 119-line helper whose only consumer is its own 75-line test |
| S5 | Always-default parameters and their dead branches (24 sites) | 53 | 0 | low | 35 | Every call site passes the same value, or none at all |
| S4 | Unreachable numba-absent fallbacks that are not differential references | 72 | 0 | med | 36 | 0% covered; no test forces the branch; numba is a hard requirement |
| T2 | `human_audit.main`/`artifact_path` + 551 KB of unread JSON | 0 | 26 | low | 26 | Writes four committed fixtures that no test opens |
| S6 | Dead constants and env knobs (+ registry cascade) | 6 | 18 | low | 24 | `_RU_PAN`/`_RU_STA`/`CURSOR_SLOTS`/`OBJECT`/`--same-enemies` |
| S7 | Share four duplicated helpers across the `_jit` modules | 40 | 0 | med | 20 | `_prep_vec_angle`, `_tile_byte` ×3, `_signed16` ×4, `edges` ×2 |
| T6 | Mock-symmetric driver tests (3 sites) | 0 | 18 | med | 9 | Fake writes through the same symbols production reads |
| S3 | `driver/sentinel_state.mem_image` | 11 | 0 | low | 11 | Referenced only by its own module docstring and a docs table |
| T8 | `driver/test_watch_play.py` `_live_truth` — **fix, do not delete** | 0 | −18 | low | — | Import the real `replay_human._enemy_truth`; the test gains teeth |
| | **Total** | **1,188** | **619** | | | |

---

## Per-item detail

### S1 — the dead scheduling calendar in `playerbase.py` (180 lines, low risk)

**Evidence.** Transitive reachability from every root (module-level references, all names used inside
test files, every `test_*` function, every `main`) leaves this subtree unreached:

| symbol | lines | count | only caller |
|---|---|---:|---|
| `BasePlayer._rotation_epochs` | 425–468 | 44 | `_cover_intervals:479` |
| `BasePlayer._cover_intervals` | 470–490 | 21 | `_busy_spans:498` |
| `BasePlayer._busy_spans` | 492–508 | 17 | `_gap_starts:530` |
| `BasePlayer._earliest_start` | 510–519 | 10 | **none** |
| `BasePlayer._gap_starts` | 521–536 | 16 | `_earliest_start:518` |
| `BasePlayer._verify_starts` | 538–557 | 20 | `_earliest_start:517` |
| `BasePlayer._drain_units` | 629–651 | 23 | **none** |
| `BasePlayer._affords_drains` | 653–663 | 11 | **none** |
| `BasePlayer.advance_phases` (staticmethod) | 249–255 | 7 | **none** |
| `BasePlayer._margin` | 684–688 | 5 | `_affords:717`, returns constant `0.0` |

`grep -rn` over `sentinel/ driver/ conftest.py` confirms each: `_drain_units`, `_affords_drains` and
`_earliest_start` appear exactly once each — on their own `def` line. Lines 425–557 form one
contiguous 133-line block. `_earliest_start`'s own docstring calls it *"the scheduling primitive the
gates lacked"* — built, never wired in. `_margin`'s docstring says *"a lookahead planner overrides it
with the accumulated per-step error at its own plan depth"*; the lookahead planners were
`astar_player.py` and `stance_player.py`, both deleted this week. It carries a
`# pylint: disable=unused-argument` for a `depth` parameter no caller passes.

Note `advance_phases` (249, staticmethod) is **not** `_advance_phases` (242, live, called at 833/836
and from `test_human_clock.py:179`). Do not confuse them. `_cone_onset` (406–423) is live (called at
568) and stays.

**What breaks if this is wrong.** Nothing statically. The risk is dynamic dispatch —
`phase_player.py:138,151` uses `getattr(twin, name)(...)` where `name` comes from a phase-method
table. Confirm the table's contents (`PhasePlayer` phase names) do not include any of the above; they
are all `_`-private helpers on `BasePlayer`, not phases.

**How to verify.** `pytest -n auto` green, then the eight-board run
(`python -m sentinel.phase_player <n>` for each measured board) producing identical action traces.
`sentinel/tests/test_endgame_fuzz.py` and `test_pin.py` exercise the drain/gate model that
`_drain_units` and `_affords_drains` claim to serve; if they were secretly live, those fail first.

**Concurrency.** `playerbase.py` was being edited in another worktree. Re-run the reachability script
before deleting.

---

### S9 — `sentinel/isoview.py` + `sentinel/tests/test_isoview.py` (760 lines, medium risk)

**Evidence.** 662 source lines drawing an isometric SVG of a board state (lit mesh, object glyphs,
enemy cones, numbered action arrows, side panel, an animated cost overlay). Consumers, complete:

- `sentinel/tests/test_isoview.py` (98 lines) — tests only `isoview` itself.
- `sentinel/tests/human_regress.py:335,341,355` — the `--diagram` flag of a regression CLI.
- `sentinel/isoview.py:661` — its own `python -m sentinel.isoview <n>` CLI.

Documented at `README.md:39`, `docs/architecture.md:426,959,978`. Every output it produces
(`renders/*.svg`) is gitignored (`.gitignore:8` — `/renders/`).

No golden, no oracle test, no board result and no ROM constant depends on it. It is 99% covered, but
by a test that exists solely because it exists.

**What breaks.** You lose the ability to look at a board and see why the player lost. For a project
whose open problem is a facing divergence on a seven-enemy board, that is a real cost — which is why
this is medium and not low. The counter-argument: the tool renders *the model's own belief*, not the
emulator's, so it cannot diagnose the divergence the instrument exists for.

**How to verify.** `pytest -n auto` green after also removing the `--diagram` path from
`sentinel/tests/human_regress.py` (lines 331–356) and the three README/docs references. The
eight-board results cannot move — `isoview` writes files and never reads state back.

---

### T1 — `driver/test_video_record.py` (119 lines, low risk)

**Evidence.** `pytest.ini` sets `testpaths = sentinel driver`, so the file is collected. It defines
exactly two functions, `validate_avi` (line 42) and `main` (line 72) — **no `test_*` function**. My
own scan confirms: 0 tests, 7 asserts, all inside `validate_avi`. At module scope (lines 24–31) it
does:

```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vice-driver")))
from vice_driver import BinMon, ViceContainer, DiskMount
os.makedirs(OUT, exist_ok=True)
```

— it depends on a sibling `../../vice-driver` checkout and creates a directory during collection.
Coverage records it at 23% (61 stmts, 47 missed), i.e. only the imports run. It also `time.sleep()`s,
which `driver/test_no_sleep.py` exists to forbid in the live path.

**What breaks.** Nothing in CI. It is a manual AVI-recording demo misnamed `test_*`.

**How to verify.** `pytest --collect-only driver` before and after: the collected test count is
unchanged. If the script is still wanted, rename it `driver/record_video_demo.py` rather than delete
it — that also removes it from `testpaths`.

---

### S2 — orphan `main()` blocks (69 lines, low risk)

| location | lines | evidence |
|---|---:|---|
| `sentinel/landtable.py:463–511` (+ `:513` guard) | 49 | prints a per-lattice census. No `python -m sentinel.landtable` in README, `docs/`, CI or any script. Already carries `# pragma: no cover`. |
| `driver/core.py:535–549` | 15 | its own docstring calls it a *"smoke demo: boot, enter a landscape, print state"*. No invocation anywhere. |
| `driver/sentinel_state.py:313–317` | 5 | dumps landscapes 0/42/9999. `docs/architecture.md:769` documents the *module*, never a CLI. |

All three are 0% covered.

**Kept** (do not confuse): `phase_player`, `player`, `isoview` (unless S9 accepted), `play_player`,
`instrument`, `dump_stage2`, `frozen_run` (executed by `driver/test_live_determinism.py:45`),
`human_regress`, `human_audit` and `_extract` all have a README, docs or test invocation.

**How to verify.** `pytest -n auto` green; `grep -rn "python -m sentinel.landtable\|-m driver.core\|-m driver.sentinel_state"` returns nothing.

---

### T3 — `UNVALIDATED_PIN` and its test (69 lines, low risk)

**Evidence.** Executed directly:

```
REGISTRY unvalidated: 50   UNVALIDATED_PIN: 50   equal: True
```

`sentinel/tests/timing_registry.py:360–413` is a 54-line frozenset of identifiers that is *exactly*
`{n for n, m in REGISTRY.items() if m["class"] == UNVALIDATED}`, where `REGISTRY` ends at line 357 —
three lines above. `test_unvalidated_debt_does_not_grow` (`test_timing_registry.py:73–87`, 15 lines)
asserts the two agree.

Adding an unvalidated constant therefore requires editing `REGISTRY` *and* `UNVALIDATED_PIN`, and the
test fails until you do both. It cannot detect anything the `REGISTRY` edit did not already announce
in the same diff. It is a second-edit speed bump.

**What is lost.** The ratchet against unvalidated-constant growth — which is worth keeping. Replace
the 69 lines with two:

```python
def test_unvalidated_debt_does_not_grow():
    assert sum(1 for m in tr.REGISTRY.values() if m["class"] == tr.UNVALIDATED) <= 50
```

That is a real ratchet over an independent quantity, and it does not duplicate 50 identifiers.

**How to verify.** `pytest sentinel/tests/test_timing_registry.py` green; then temporarily flip one
`REGISTRY` entry to `UNVALIDATED` and confirm the new assertion fires.

---

### S8 — `sentinel/landscan.py` + `test_landscan.py` (123 lines, medium risk)

**Evidence.** Zero source callers — the module import graph shows `srcusers=0, testusers=1`. Its
`main()` (lines 62–92) is never invoked by any doc, script or CI job; `README.md:49` and
`docs/architecture.md:426` describe the module's *role*, not a command. 46% covered (50 stmts, 27
missed — the whole `main()`).

Its only filter flag is dead by construction (`sentinel/landscan.py:68`):

```python
ap.add_argument("--same-enemies", action="store_true", default=True)
```

`store_true` with `default=True` and no `--no-` counterpart can never be false, so the guard at line
82 (`if args.same_enemies and row["enemies"] != ref["enemies"]: continue`) is unconditional. Passing
the flag is a no-op.

**What breaks.** You lose a board-matching analyser ("find me a landscape like 335"). It answers a
research question, not a correctness one.

**How to verify.** `pytest -n auto` green after removing the README and architecture table rows.

---

### T4 — strictly-subsumed tests (63 lines, low risk)

Each pair verified; the stronger test is named in every case.

| remove | lines | subsumed by | why the survivor is stronger |
|---|---:|---|---|
| `driver/test_core.py:23–29` `test_landscape_from_digits_keeps_the_high_bcd_byte` | 7 | `:32–34` `..._inverts_enter_landscape` | Its 3 assertions (`"0042"→0x42`, `"0335"→0x335`, `"2024"→0x2024`) are 3 of the survivor's 5, since `f"{0x0042:04x}" == "0042"`. Verified. |
| `sentinel/tests/test_human_fixture_hygiene.py:34–36` | 3 | `:25–31` `test_real_action_counts[ls335]` | The parametrized test asserts `{"absorb":70,"create":35,"transfer":18}` across 3 fixtures; 70+35+18 = 123, the removed test's only claim. |
| `sentinel/tests/test_landscape.py:57–60` `test_generate_is_deterministic` | 4 | `:48–55` `test_generate_matches_golden` | "Same answer twice" on a pure function. The golden pins the exact bytes for seed 42, so it fails on nondeterminism *and* on wrongness. |
| `sentinel/tests/test_prng.py:35–40` `test_shuffle_is_deterministic` | 6 | `:21–26` `test_golden_stream_matches_rom` | Same argument, `golden_prng.json`. |
| `sentinel/tests/test_relative.py:31–36` `test_divide_and_arctan_is_pure` | 6 | `:39–53` `test_relative_angles_matches_golden` | Same argument, `golden_relative.json`. |
| `sentinel/tests/test_landscape.py:76–78` `test_landscape_0_player_is_fixed` | 3 | `:48–55` golden | The golden's `SPANS` include `0x000B` (`PLAYER_OBJECT`) and `0x0900`+`0x180` (the whole object array) for key `"0"` — it pins the byte, not a derived tuple. |
| `sentinel/tests/test_los_jit.py:67–88` `test_jit_matches_python_with_objects` | 23 | `:195–205` `test_jit_matches_python_object_stacks` | 7 cases (6 objects) vs 56 cases (7 seeds × 4 scenarios × 2 `max_steps`), where `_scen_boulders` places 24 boulders in 3 concentric rings. Same `_sweep_mismatches` driver; the survivor also asserts the sweep count. |
| `sentinel/tests/test_render_cost.py:152–162` `test_examine_count_is_exact_and_scene_dependent` | 11 | `:145–150` `test_render_cost_matches_golden` | `_check()`'s first line is already `assert n_examine == rec["n_examine"]`. Only `assert len(counts) > 5` is novel — collapse this test to that one line rather than deleting it outright. |

Also dead, found by the reachability scan and confirmed by grep:
`sentinel/tests/test_enemies.py:102–104` `_drive_meanie_golden` (3 lines — the regenerator for
`golden_meanie.json`, defined and never called) and `sentinel/tests/test_los.py:18` `_SPANS = None`
(1 line, assigned and never read).

**Coverage effect.** None of these removals drops coverage of otherwise-untested source: every one
has a named stronger test covering the same statements. `test_los_jit.py:67–88` and
`test_render_cost.py:152–162` are the only ones with meaningful runtime, so removing them also
shortens the suite.

**How to verify.** Run `pytest --cov=sentinel --cov-report=term-missing` before and after; the missed
line sets for `sentinel/los.py`, `sentinel/projector.py`, `sentinel/landscape.py`, `sentinel/prng.py`
and `sentinel/relative.py` must be identical.

---

### T7 — `sentinel/tests/ckpt.py` + `test_ckpt.py` (194 lines, medium risk)

**Evidence.** `ckpt.py` (119 lines, 7 public functions: `capture`, `restore`, `verify`, `save`,
`load`, `describe`, `snapshot`) has exactly one importer in the whole tree —
`sentinel/tests/test_ckpt.py` (75 lines, 7 tests). No production module, no other test, no driver
script. `snapshot` (lines 29–45) has zero external call sites; it is reached only from `capture`.
Documented at `docs/architecture.md:1075` as a developer workflow for resuming long phase-player runs.

**What breaks.** You lose the ability to checkpoint and resume a multi-hour phase-player run. That is
a real developer convenience — hence medium, not low. But it is 194 lines of test tree that proves
only itself: no board result, no golden and no ROM constant depends on it, and `test_ckpt.py` exists
solely to test `ckpt.py`.

**How to verify.** `pytest -n auto` green after removing the `docs/architecture.md` section. No board
run changes.

---

### S5 — always-default parameters and their dead branches (53 lines, low risk)

Every call site enumerated. A parameter listed here is either never passed, or passed the identical
value at every site.

| function | parameter | call sites | dead lines |
|---|---|---|---:|
| `landtable.landable_view:365`, `landtable.landable_set:415` | `stats` | 5 sites, none pass it | 12 (`:379–380, 392–393, 403–404, 453–454`) |
| `player.Player._climb:266` | `need_progress` | `player.py:61,66,72` (default) and `:102` (explicit `True`) | 8 (`:398–403` skip + signature) |
| `driver/core._enter_play:220` | `chunk, gen_chunks, play_chunks, taps` | one site, `core.py:198`, passes none | 4 |
| `aim.resolve:26`, `aim.gate:44`, `aim.propose:54` | `player` (all three), `eye_z` (`gate`) | `aim.py:50`, `playerbase.py:845–847`, `sentinel_execute.py:143` | 6 |
| `playerbase._player_window:720` | `exclude` | `player.py:48`, `playerbase.py:716`, `test_endgame_fuzz.py:62`, `human_audit.py:149` — none pass it; the list-comp filter at `:727` is a no-op | 2 |
| `playerbase._affords:708` | `window` | one site, `playerbase.py:854` | 2 |
| `playerbase._seen_now:393` | `full_only` | one site, `human_audit.py:146` | 2 |
| `threat.player_sees_tile:60` | `eye_z` | 6 sites, none pass it | 2 |
| `phase_player._arbitrate:125` | `lookahead` | one site, not passed | 1 |
| `statecmp.format_divergence:120` | `a_name`, `b_name` defaults | both sites pass `("emu","sim")`; `"A"/"B"` never used | 2 |
| `driver/boot.save_snapshot:50` | `save_roms, save_disks, timeout` | 3 sites, none pass any; `struct.pack` at `:57` always packs 0,0 | 3 |
| `driver/boot.wait_for_load:24` | `total, poll` | `core.py:176` omits; `boot.py:236` passes literally the defaults | 2 |
| `driver/boot.boot_loaded:199` | `attempts` | forwarded from `SentinelDriver.boot(attempts=4)`; no caller overrides | 2 |
| `driver/core.robust:57` | `tries` | 5 sites, none pass it | 1 |
| `driver/clock.run_frames:49` | `timeout` | 8 sites, none pass it | 1 |
| `driver/kbd_aim._uturn:248`, `tap_action:365` | `max_passes` (both) | 6 sites, none pass it (`settle=False` **is** used at `sentinel_execute.py:431` — keep that one) | 2 |
| `driver/kbd_aim._one_scan_press:215` | `key`, `timeout` | one site, `:242`, `self._one_scan_press("SPACE")` — `key` is a constant, not a parameter | 2 |
| `driver/instrument.SimClock.__init__:29` | `plotting` | one site, `:87`, `SimClock(seed)` | 1 |

Separately, **`los.landable_view:1364`'s `v_band=False` default is the value nobody uses**: all four
call sites (`aim.py:60`, `test_landable.py:106`, `test_human_win_logs.py:199`,
`_extract.py:143`) pass `v_band=True`. Flip the default and the `not v_band` disjunct at `:1384`
becomes unreachable.

**What breaks.** Each of these is a future option nobody took. Removing `need_progress` is the only
semantic one — `need_progress=False` would enable non-progressing boulder placements the greedy player
has never made. If that mode is wanted later it is 6 lines to reinstate.

**How to verify.** `pytest -n auto` green plus the eight-board run. Every one of these is a pure
signature change with no branch flip in the taken path, so any behavioural change is a bug in the edit.

---

### S4 — unreachable numba-absent fallbacks (72 lines, medium risk)

**The distinction that matters.** Some pure-Python bodies behind `_HAVE_JIT` are *differential
references* — a test drives them directly and asserts the jit agrees. Those are load-bearing (see N1).
Others are only *fallbacks* — reached solely when numba is absent, which the supported configuration
never is (`requirements.txt` pins `numba>=0.66.0`, CI installs it).

| body | lines | count | driven by a test? | coverage |
|---|---|---:|---|---|
| `los._march_python` | 690–800 | 111 | **yes** — `test_los_jit.py:44` calls it directly | keep |
| `los.prepare_vector_from_player_sights` (via `_prepare_vector`) | 1518–1527 | 3 | **yes** — `test_los_jit.py:98–111` sets `_HAVE_JIT = False` | keep |
| `projector._occlusion_visible_py` | 280–372 | 93 | **yes** — `test_projector_jit.py:48,103` | keep |
| `projector._project_scene_py` | 435–485 | 51 | **yes** — `test_projector_jit.py:190–205` monkeypatches `_HAVE_JIT` | keep |
| `enemies.advance_frames_python` | 573–589 | 17 | **yes** — `test_enemies_jit.py:39,51,63` | keep |
| `los._landable_sweep_py` | 1310–1339 | 30 | **no** | 0% |
| `los._landable_view_py` | 1473–1500 | 28 | **no** | 0% |
| `los` guards at `:1273–1274`, `:1464–1465` | | 4 | — | 0% |
| `landtable` guards at `:381–384`, `:426–427` | | 6 | — | 0%, already `# pragma: no cover` |
| `playerbase` guards at `:91–92`, `:116–117` | | 4 | — | 0%, `:116` already `# pragma: no cover` |

The four `# pragma: no cover` markers already in the tree are an explicit admission that these
branches never execute. `test_landable.py:196,224,258` and `test_landtable.py:13–15` *skip* when
numba is absent, so the fallbacks are never compared against the jit path either — they can rot
silently and no test will notice.

**What breaks.** `los.landable_view` / `landable_views` / `landable_sweep_with_centres` would raise
`NameError` on a machine without numba instead of running slowly. Given numba is a hard requirement
this is the honest behaviour; the `try/except ImportError` at `los.py:45–51` should become a plain
import at the same time.

**How to verify.** `pytest -n auto` in the normal (numba-present) configuration must be entirely
unaffected — these lines execute zero times today, which the coverage report proves. Then confirm
`python -c "import sentinel.los"` still imports.

---

### T2 — unread committed artifacts (26 lines + 551 KB, low risk)

**Evidence.** Tracked by git (`git ls-files sentinel/tests/fixtures/human_wins/`):

| file | bytes | read by |
|---|---:|---|
| `ls335_audit.json` | 406,032 | nothing |
| `ls42_audit.json` | 72,257 | nothing |
| `ls0_audit.json` | 38,008 | nothing |
| `ls110.json` | 34,828 | nothing |

The three `_audit.json` files are written by `sentinel/tests/human_audit.py:295–320`
(`artifact_path` + `main`). `sentinel/tests/test_human_audit.py:92` calls
`human_audit.audit_fixture(name)` **live** and never opens the file. `grep -rn "_audit.json"` over
`sentinel/ driver/` returns only `human_audit.py`'s writer and four references to
`fixtures/frozen_ls42_audit.json` — a **different file** in `fixtures/`, which is live and stays.

`ls110.json` has no loader: `sentinel/test_human_win_logs.py:43` sets
`FIXTURES = ("ls0.json", "ls42.json", "ls335.json")` and `human_audit.py:15` imports that same tuple.
`ls110` appears only in `_extract.py:79` (the regeneration recipe), a prose comment in
`phase_player.py:301`, and a docstring in `test_timing_derivations.py:162`. There is no
`ls110_clock.json`, so it cannot join `test_human_clock.py`'s `LIVE_CLOCKS` either.

The `_clock.json` files (`ls0`, `ls42`, `ls335`) **are** read — `test_human_clock.py:265` builds the
filename as `f"{board}_clock.json"`. Keep all three.

**What breaks.** Nothing. If the audit artifacts are wanted for eyeballing, they are one
`python -m sentinel.tests.human_audit` away — the removal here is of the *committed copies* plus the
`main()`/`artifact_path` writer, not of `audit_fixture`.

**How to verify.** `pytest -n auto` green; `git grep -l ls110\\.json` returns only the extraction
recipe.

---

### S6 — dead constants and env knobs (24 lines, low risk)

| symbol | location | evidence |
|---|---|---|
| `_RU_STA` | `driver/kbd_aim.py:35` | reads `KBD_STA_TIMEOUT`; the variable is never used in `kbd_aim.py`. Its only references are three entries in `sentinel/tests/timing_registry.py` (`:339`, `:391`) |
| `_RU_PAN` | `driver/kbd_aim.py:34` | reads `KBD_PAN_TIMEOUT`; never used. Referenced only by `timing_registry.py:340,390,416` and `test_timing_registry.py:122,131` |
| `CURSOR_SLOTS` | `sentinel/enemies.py:554` | referenced only in a comment at `enemies.py:63` |
| `OBJECT` | `sentinel/los_jit.py:28` | zero references |
| `--same-enemies` | `sentinel/landscan.py:68,82` | `store_true` + `default=True` — permanently on (folded into S8) |

Removing `_RU_PAN` cascades: it is the sole member of
`timing_registry.KNOWN_FALSE_PROVENANCE_COMMENTS` (`:416`) and the counter-example in
`test_timing_registry.py:120–131` `test_comment_attribution_detects_both_directions`. That test's
other two assertions duplicate `test_registry_entries_are_well_formed:49` (module attribution, all 70
constants) and `test_unvalidated_debt_does_not_grow` (class), so the 12-line test goes with it — 18
test lines in total.

Related but **kept**: 23 other env knobs (`RENDER_*`, `PAN_CACHE_MAX`, `VIEW_CACHE_MAX`,
`TUNE_TRANSFER_FRAMES`, `FRAME_TICKS`, `UPDATES_PER_FRAME`, `BINMON_HOST/PORT`,
`VICE_REAP_ORPHANS`, …) are never set by any test, script, doc or CI job, and all but three are read
at import time into a module constant, so they are not runtime-tunable either. Collapsing them to
plain constants would save ~23 lines. They are documented as the tuning surface
(`docs/architecture.md:693–694`) and each is one line; the saving does not justify removing a
documented capability. Two do deserve a note: `RENDER_CACHE_MAX` is read by **both**
`sentinel/rendercost_py65.py:16` (default 8192) and `sentinel/projector.py:396` (default 20000), so a
single `export` silently retunes two unrelated caches.

**How to verify.** `pylint --rcfile=.pylintrc sentinel driver conftest.py` clean; `pytest -n auto`
green.

---

### S7 — share four duplicated helpers across the `_jit` modules (40 lines, medium risk)

These are genuinely mechanical because both copies are already `njit` and the modules already import
from each other (`enemies_jit.py:12` imports `_ARCTAN_LO/_ARCTAN_HI/_HYP` from `relative`;
`projector_jit.py:12–18` imports `_calc_angle`/`_calc_hypotenuse`/`_vertical_angle` from
`enemies_jit`).

| duplicate | locations | saving |
|---|---|---:|
| `_prep_vec_angle` | `enemies_jit.py:424–434` restates the tail of `los_jit._prep_vec:879–910`; lines 428–434 and 904–910 are the same seven statements | 11 |
| `_tile_byte` | `los_jit.py:40–47`, `enemies_jit.py:107–110`, `projector_jit.py:34–39` — three copies of `((x<<3)&0xE0) | (y&0x1F)` with page `(x&3)+4` | ~16 |
| `_signed16` | `los.py:1407–1409`, `landtable.py:66–68`, `projector.py:127–130`, `projector_jit.py:207–209` | ~12 |
| the `edges` table | `sentinel/los.py:961` **and** `:984` — `[0x00,0x03,0x01,0x00,0x01,0x02,0x02,0x03]` ($1DF1–$1DF8) written out twice inside one function | 1 |

The `edges` duplication is worth fixing for correctness reasons independent of line count: two copies
of a ROM lookup table thirteen lines apart is a drift hazard, and `los_jit.py:67–83` already factors
it into `_edge`.

**What breaks.** A numba typing error at import if a shared helper is inlined into a `parallel=True`
kernel that previously had its own copy. `los_jit.march_batch` is `parallel=True`.

**How to verify.** `pytest sentinel/tests/test_los_jit.py sentinel/tests/test_enemies_jit.py
sentinel/tests/test_projector_jit.py` — these three pin the jit paths against their Python references
byte-for-byte and would catch any behavioural change. Then time
`sentinel/tests/test_landable.py::test_batched_sweep_matches_per_probe_aim_target` to confirm no
compilation regression.

---

### T6 — mock-symmetric driver tests (18 lines, medium risk)

**Evidence.** `driver/conftest.py:23–29`'s `FakeBM.set_player` writes through the *same symbolic
constants* the production code reads:

```python
# conftest FakeBM.set_player          # core.player_tile (driver/core.py:422-425)
self.mem[core.A_SLOT] = slot          ps = bm.mem_get(A_SLOT, A_SLOT)[0]
self.mem[core.A_X + slot] = x         return bm.mem_get(A_X+ps, A_X+ps)[0], ...
```

If `A_X` were wrong, every one of these still passes.

| test | lines | what it actually verifies |
|---|---:|---|
| `driver/test_kbd_aim.py:29–37` `test_committed_bearing_lifecycle` | 10 | that `set_bearing`/`committed_bearing` round-trip a tuple through `& 0xFF`. The `fake_bm` fixture is constructed and never touched. |
| `driver/test_core.py:18–20` `test_player_tile` | 3 | the slot-indexed indirection only |
| `driver/test_kbd_aim.py:40–44` `test_cur_reads_cursor_bytes` | 5 | nothing (`d.rd(A_CX)` vs `mem[A_CX]`) |

Note `kbd_aim.A_CX/A_CY/A_SFLAG` (`driver/kbd_aim.py:52–54`) are raw literals with the comment *"no
memmap entry"* — precisely the addresses with no independent pin, and the tests that look like they
pin them do not.

**Kept from the same family:** `test_energy_is_six_bit` (`& 0x3F`) and
`test_sights_live_on_reads_flag_bit7` (`& 0x80`) each verify one real bit-mask; they are weak but not
vacuous.

**What breaks.** `driver/` coverage falls further from 35.8%. These are the only tests touching those
lines. The honest reading is that the coverage they provide is illusory — but if the goal is a
coverage number rather than a guarantee, leave them. Medium risk for that reason.

---

### S3 — `driver/sentinel_state.mem_image` (11 lines, low risk)

`driver/sentinel_state.py:117–127`. Referenced only by its own module docstring (`:20`) and a table
row in `docs/architecture.md:769`. Zero call sites in source or tests; 0% covered.

**How to verify.** `pytest -n auto` green; `grep -rn mem_image sentinel driver` returns only the
docstring.

---

### T8 — `driver/test_watch_play.py` is a fix, not a removal

The docstring claims the test pins that `watch_play._enemy_clock` decodes *"exactly the per-enemy
facing + rotation step + cooldowns that `driver.replay_human._enemy_truth` reads live"*. But
`_live_truth` is **re-implemented inside the test file** (`test_watch_play.py:15–33`) rather than
imported. It is field-for-field identical to `driver/replay_human.py:27–46` (`slot`, `type`, `tile`,
`h_angle`, `v_angle`, `rot_step`, `rot_cooldown`, `drain_cooldown`, `update_cooldown`). The assertion
at `:45` is `assert clock == _live_truth(img)`.

**If `replay_human._enemy_truth` changed tomorrow, this test would still pass** — the invariant it
advertises is not asserted. Replace the 19-line copy with
`from driver.replay_human import _enemy_truth`. Net −18 lines, and the test starts doing what it says.

---

## Explicitly NOT proposed

### N1 — the ~1,100 lines of numba/pure-Python mirroring

This is the single largest duplication in the repo and the most tempting target. It must stay.

| pair | mirrored lines | pinned by |
|---|---:|---|
| `los.py` ↔ `los_jit.py` (6502 fixed-point core, ray-vector build, object-stack walk, slope solver, the march) | ~700 | `test_los_jit.py` — 7 seeds × 32 bearings × the full `PITCH_BAND`, plus object stacks and `return_centre` |
| `enemies.py` ↔ `enemies_jit.py` (21 ROM routines) | ~420 | `test_enemies_jit.py` — full 64 KB image compared every 25 frames, 400 frames, landscapes {0, 42, 335} |
| `projector.py` ↔ `projector_jit.py` (projection walk, occlusion) | ~340 | `test_projector_jit.py` — per-observer occlusion across 6 landscapes, `project_scene` on all 4 quadrants, exact `n_examine` |
| `relative.py` ↔ `enemies_jit.py` (trig, `can_see_object`) | ~330 | `test_enemies_jit.py`, indirectly |

Three reasons it is not removable:

1. **The Python bodies are the oracle.** `test_los_jit.py`, `test_enemies_jit.py` and
   `test_projector_jit.py` are differential tests: they assert the fast path equals an *independent*
   implementation. Merge the two and the test becomes `assert f(x) == f(x)`.
2. **Merging is a rewrite, not a deletion.** The Python bodies take `State` objects, use Python lists,
   dicts, classes and `Prng`; `njit` accepts none of those. "Just decorate the reference" means
   rewriting it into numba-compatible form — at which point you have deleted the readable
   ROM-provenance version, which is this project's actual product.
3. **`los_jit.march` and `projector_jit.project_scene` have genuinely diverged in implementation**
   (per-tile caching, a closed-form intra-tile fast-forward, signed-16 pre-decode). Those are not
   mirrors at all; they are optimised rewrites that the reference exists to validate.

The one sub-case that *is* safely shareable — four small helpers already `njit` on both sides — is
proposed separately as S7.

### N2 — the golden fixtures and the `oracle`-marked tests

`golden_actions.json` (121 KB), `golden_projector.json` (336 KB), `golden_landscape.json`,
`golden_los.json`, `golden_meanie.json`, `golden_relative.json`, `golden_render_cost.json`,
`golden_pan_cost.json`, `golden_enemies.json`, `golden_prng.json` and the `@pytest.mark.oracle` tests
that regenerate them are what prove the model against the real 6502. `conftest.py:30–37` skips the
oracle tests when `out/sentinel_stage2.bin` is absent, so the goldens are the *only* proof available
in CI, where the ROM image cannot be distributed (it is copyrighted and gitignored).

I checked the overlap claim and it does not hold: `golden_render_cost.json` and
`golden_projector.json` share 10 of their keys but store disjoint schemas (`{setup, tiles}` vs
`{cycles, examine_cycles, n_examine, n_filled, object_fill_cycles, terrain_fill_cycles}`).
`golden_los.json`'s keys are a subset of `golden_landscape.json`'s but the contents are aim samples
versus generator spans. No golden is a subset of another.

**Two hygiene issues found, neither a removal.** First, `golden_los.json`, `golden_relative.json`,
`golden_prng.json` and `golden_enemies.json` (59 KB) have **no in-repo regenerator** — if they drift
they cannot be rebuilt — and `golden_meanie.json`'s regenerator
(`test_enemies.py:102–104` `_drive_meanie_golden`) is defined and never called (proposed for removal
in T4, but the honest fix is to *wire it up*). Second, the regenerate tests rewrite tracked goldens as
a side effect of a plain `pytest` run (`test_pan_cost.py:114–121`, `test_render_cost.py:136–142`,
`test_projector.py:106–115`); three abandoned `golden_pan_cost.json.<pid>` temp files are sitting in
`sentinel/tests/` right now, saved from being committed only by a `.gitignore` rule
(`sentinel/tests/*.json.[0-9]*`).

### N3 — the numba-vs-Python equivalence tests and their `_HAVE_JIT` reference branches

`test_los_jit.py` (205), `test_enemies_jit.py` (65) and `test_projector_jit.py` (222) are the tests
that make N1's duplication safe. Keep every one, and keep the five Python bodies they drive
(`_march_python`, `prepare_vector_from_player_sights`, `_occlusion_visible_py`, `_project_scene_py`,
`advance_frames_python` — 275 lines). S4 removes only the fallbacks that *no* test drives; the
distinction is the whole point of that item.

### N4 — the instrument, the divergence gates and the live-driver determinism gate

`driver/instrument.py` (194) + `sentinel/statecmp.py` (126) + `driver/test_enemy_sim_divergence.py`
(43) are the frame-locked sim-vs-emulator race. `driver/test_live_determinism.py` (94) is the
host-clock-leakage gate. All four stay. `instrument.py` is 55% covered and `statecmp.py` 96%.

### N5 — `sentinel/player.py`, the reactive greedy player (472 source + 63 test)

This looks like an obvious deletion — `PhasePlayer` is the default and wins all eight measured boards,
so a second player reads as redundant. It is not. `driver/test_live_determinism.py:8–12` states the
reason in its own docstring:

> Driven by the GREEDY player: the subject is the driver's clock discipline, and greedy decides in
> milliseconds, so a divergence here can only be the driver. The phase player fails this gate — the
> monitor socket drops across the ~25 s think gap a tie opens and the reconnect leaks frames into the
> measurement.

`test_live_determinism.py:45–56` shells out to `python -m driver.frozen_run 42 --player greedy`.
Removing `player.py` removes the only player fast enough to run the determinism gate, and
`open_items.md:29–32` records that substituting the phase player is a measured defect, not an option.
`player.py` is 72% covered.

Consequently `driver/frozen_run.py` (97), `driver/play_player.py` (75) and the
`--player {greedy,phase}` option are also load-bearing despite reading as 0%-covered orphan scripts —
`frozen_run` is the only driver CLI with an automated consumer.

### N6 — the 3,620 lines of docstring and comment in source (26% of it)

Source is 14,086 lines: 2,111 docstring, 1,509 comment, 1,638 blank, ~8,828 code. `los.py` is 41%
prose, `enemies.py` 38%, `relative.py` 39%, `memmap.py` 65%. That is a large number and it is the
wrong target. Nearly all of it is ROM provenance — the `$1825` / `$1A31` / `$1F47` addresses and the
measured frame counts that say *why* a line is what it is. For a reverse-engineering project that is
the product, not the packaging. Only 35 docstrings exceed 10 lines (586 lines total) and most of those
are module-level orientation.

If prose must be cut, cut the measured-narrative blocks that `docs/` now restates — but `docs/` was
being rewritten concurrently, so that comparison could not be made honestly here.

### N7 — `landable_views` / `landable_set` as cross-check oracles

`sentinel/landtable.py:408–460` `landable_set` (53 lines) has **no production caller** — its only
consumers are `test_landtable.py:244` and `test_landable.py:146`. So does
`los.landable_views:1233–1249`, whose only source caller is inside `landable_set`. This looked like 70
lines of dead code.

It is not: they are the exhaustive whole-board sweeps that `test_landable.py:128–151` uses to prove
the fast targeted path (`landable_view_targeted`, the one the player actually calls) returns the same
answer. Deleting them deletes the proof that the optimisation is sound. They stay.

I also considered deleting `test_landable.py:128–151` on the grounds that
`test_landtable.py:175–194` tests all 1024 tiles on a mid-game board versus `test_landable.py`'s
start board — a strictly larger input space. That is true, but `test_landtable.py` is gated by
`pytestmark = skipif(not los._HAVE_JIT)` while `test_landable.py` is not, so the two do not cover the
same configurations. Not proposed.

### N8 — `sentinel/tests/timing_registry.py`'s `discover()` and `test_validated_constants_name_a_real_test`

`discover()` (`:155–206`, 52 lines) is a real AST-plus-comment scanner that catches both module
constants and kwarg defaults, and `test_validated_constants_name_a_real_test` (`:59–70`) regex-greps
the test sources for `^\s*def <evidence>\b`, so it catches a renamed or deleted evidence test — 20
constants cite one of 5 test names. Those are the teeth of that file. Only the `UNVALIDATED_PIN`
bookkeeping (T3) and the `_RU_PAN` counter-example (S6) are proposed.

### N9 — `sentinel/tests/human_clock.rounds_between`

Reported to me as dead. It is not: `human_clock.py:88` and `:93` both call it (the second with
`strict=False`). Verified by grep. Listed here so it is not re-litigated.

### N10 — the driver's VICE scripts

`driver/watch_play.py` (257, 24% covered), `driver/replay_human.py` (216, 0%),
`driver/dump_stage2.py` (84, 0%), `driver/boot.py` (250, 67%), `driver/kbd_aim.py` (437, 39%),
`driver/sentinel_execute.py` (449, 20%) read as dead weight on a coverage report. They are the live
path against the real machine: `dump_stage2` regenerates the `oracle` fixture, `watch_play` recorded
every human-win fixture in the tree, `replay_human` produced `ls42_truth.json`. Their coverage is low
because CI has no C64.

---

## Method

**Call graph.** Built with `ast` over all 106 `.py` files in `sentinel/`, `driver/` and `conftest.py`
(worktrees under `.claude/` excluded). Two passes:

1. *Definitions* — every `FunctionDef`, `AsyncFunctionDef`, `ClassDef` and module-level `Assign`,
   recorded with file, line span, kind and dotted qualname.
2. *References* — every `Name` in `Load` context, every `Attribute` access by attribute name, every
   `ImportFrom` alias, and every string constant (to catch `getattr` dispatch). Edges run from a
   definition to the names appearing anywhere in its body.

**Reachability.** BFS from a deliberately generous root set: every name referenced at module level of
any file, every name referenced anywhere inside a test file, every `test_*` function (pytest collects
them by name), and every `main`. Anything unreached is a candidate. Raw output: 34 symbols / 321
lines, of which 15 were `__init__` methods and their cascade (constructors are called implicitly —
false positives, discarded), leaving the S1/S3 set. Every survivor was then confirmed by `grep -rn`
across `sentinel/ driver/ docs/ README.md conftest.py .github/`.

Cross-checked with a second, independent script that reports per-symbol caller sets split into
source-callers and test-callers, which is what surfaced N7 (`landable_set` reachable only from tests)
and the single-caller inventory.

**Coverage.** `COVERAGE_FILE=<scratch> pytest -n auto -q -k "not regenerate" --cov=sentinel
--cov=driver --cov-report=term-missing`. The `not regenerate` filter is necessary: the
`@pytest.mark.oracle` regenerate tests rewrite tracked golden JSON as a side effect of a normal run,
and this was a read-only analysis. That excludes 3 tests; all three assert the same thing their
golden-replay siblings assert, so the coverage figures are unaffected except in
`sentinel/tests/oracle.py`.

**Module import graph.** Separate pass resolving dotted `import`/`from` targets to files, splitting
importers into source and test, which produced the "only consumer is its own test" findings (S8, S9,
T7).

**Prose measurement.** `ast.get_docstring` line spans plus `tokenize` COMMENT tokens, per file.

### What this analysis cannot see

- **Dynamic dispatch.** `sentinel/phase_player.py:138,151` calls `getattr(twin, name)(...)` where
  `name` comes from a phase table; `driver/sentinel_state.py:153` uses `setattr(self, k, kw[k])`;
  `driver/clock.py:18,30` stash attributes on a third-party `BinMon` via `getattr`/`setattr`. I
  mitigated this by treating every string constant in the tree as a possible symbol reference, which
  is conservative — it can only cause false *negatives* (something reported live that is actually
  dead), never a false positive.
- **The `.claude/worktrees/` copies were excluded**, but `sentinel/playerbase.py`,
  `sentinel/phase_player.py`, `sentinel/landtable.py` and `docs/` were under concurrent edit by other
  agents. Line numbers in S1, S4, S5 and S7 may have moved.
- **Entry points invoked only by a human at a shell.** `python -m driver.watch_play` is how every
  human-win fixture in the tree was recorded, and nothing in the repo records that it was run. I used
  README/docs mentions as a proxy for "someone runs this", which under-counts.
- **The eight-board results were not re-run.** Every claim about what is safe to remove from the
  player path rests on static reachability plus the existing test suite, not on a fresh
  `phase_player` run per board. Any removal touching `playerbase.py` or `phase_player.py` should be
  gated on that run.
- **CI-only paths.** `driver/test_live_determinism.py` and `driver/test_enemy_sim_divergence.py` skip
  without Docker + the tape + a code-entry snapshot, so their real behaviour was inferred from source,
  not observed.
- **`vice-driver` and `jennings`** are external packages; symbols they provide are outside the graph.
