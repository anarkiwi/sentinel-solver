# Handoff — state and ranked next steps

Branch `feat/astar-macro-planner-human-audit`, work **uncommitted** in the tree
(`/scratch` is an NFS share; artifacts under `out/` persist and are gitignored).
Live tests need docker + `sentinel-gold.tap` + `renders/vice_code_entry.vsf`.

Suite re-run after the `pytest.ini` revert: **665 passed, 5 skipped, 3 xfailed** (7m43s),
so the revert cost nothing. Run it with the project venv
(`/scratch/tmp/fogbank-sentinel-solver/venv/bin/python -m pytest -q`) — the default
`PATH` python has no pytest.

New untracked files: `docs/{handoff,plan_fidelity}.md`, `driver/{clock,frozen_run,
plan_audit,test_enemy_sim_divergence,test_live_player,test_no_sleep,
test_sentinel_execute}.py`, `sentinel/tests/{test_aim_subterms,test_kbd_scan_gate}.py`
+ `fixtures/live_aim_subframes.json`, `out/FACTS.md`.

## Settled

Details in [plan_fidelity.md](plan_fidelity.md), [render_cost.md](render_cost.md),
[astar_player.md](astar_player.md), [driver.md](driver.md).

Ten defects fixed. Six were **driver** bugs injecting the enemy-phase error the cost
model was blamed for: the `$1236` sights-latch (171 f toggle), a 6.0 s wall clock
clipping the transfer-settle measurement at ~300 f, a wall-clock `_wait`, `last_bearing`
permanently `None` live, a latent `probe_tile` hang (the live player leaves the CPU
halted, so its `sleep(0.05)` could never let the plot finish and clear `$0CE4`), and a
blanket container reap that made concurrent runs impossible.

Per-step charged-vs-measured rms on ls42: **68.4 -> 49.3 f**. ls42 outcome moved
dead(E0) -> livelock -> energy-starved dead-end (alive, E2). ls42 is **not** won.

**Live play is now 100% event driven** and warp-independent — the acceptance criterion,
measured on ls335 with 4 waits each: warp-on `charged=60 measured=60` (host 0.48 s),
warp-off `charged=60 measured=60` (host 1.6 s). Identical in emulated frames, 3.3x apart
in host seconds. `driver/test_no_sleep.py` (AST + `tokenize`) fails if a `time.sleep`
returns to a live-play module; `# sleep-ok: <reason>` markers are pinned as an
exact-match set of 10 `(module, reason)` pairs, so adding one is a reviewable act.

**Disproved** (do not resurrect): "transfer settle over-charges, fix
`viewpoint_replot_frames` first". That was the 6.0 s clipping artifact; unclipped the
error has no systematic sign, and correcting the settle's viewpoint moves it *up*
(median +28 f). Ranking fixes by cumulative drift is also wrong — net drift at the
failing step was ~-17 f while the phase was ~35 f out. Per-step |error| is the metric.

Eleventh defect: `landscape_from_digits` parsed only the last two typed digits, so any
code above 0099 seeded the reference generator with a different landscape (`"0335"` ->
0x35). `verify_entry` was therefore silently dead on ls335 (`ENTRY MATCH: 0/26` in every
historical log) while ls42 passed 16/16 because 0x0042 == 0x42. Live play was unaffected
— the planner reads `ViceSource`, never `generate()`. Fixed and pinned.

Two useful negative results about the machine: `$3642`/`$363D`/`$365D` are never reached
at landscape entry (probed live, all time out), and `$9630` recurs in *every* phase —
menu, generation, preview, play — so it is a universal exact frame clock
(`driver/clock.py`).

## Next steps, ranked

### 1. SETTLED -- the both-frozen experiment says fidelity, not search
Verdict and numbers in [plan_fidelity.md](plan_fidelity.md). Frozen on both sides, ls42
is **won** in sim (36 actions, 12.1 s, no emulator) and **won live** (36 actions,
energy 11), with all 35 pre-win steps matching on `(action, tile, energy)`; step 36 is
the PRNG hyperspace landing, which differs by design and wins either way. The search is
exonerated -- the live loss is enemy-phase timing. Item 2 is now the sole critical path.

Residue from this experiment:
- ls335 is not a usable arm: A* takes 0 actions there frozen **and** unfrozen (separate
  open defect, see plan_fidelity.md's Disproved list).
- `frozen_run.py` should join `LIVE_MODULES` in `driver/test_no_sleep.py`.
- Frozen greedy on ls335 stopped at 6 actions with energy 10 (live greedy: 15 actions,
  energy 2), both losses -- unexplained, low priority.

### 2. Per-notch pan redraw model
Largest remaining frame error (ls335 p21 **-383 f**, p8 -268 f, ls42 p15 -125 f). Each
`pan_viewpoint` attempt is one `plot_world` at the *intermediate* bearing into the pan
strip buffer (`$10D7`/`$111C`) plus the notch's 16 h / 8 v scroll steps; measured 25-27 f
empty vs 73-112 f busy. `REDRAW_BASE = 34` and `STEPS_PER_EDGE = 0.02` are **fitted** and
cannot span that — price each notch with a `render_cost`-class model at that notch's
bearing. Blockers: `projector.render_cost` models the full play buffer (`$14..$8A`), not
the pan strip, and costs ~60 ms/call, so it needs a bearing-keyed memo before it can sit
in `_aim_frames`' A* inner loop. Do not retune the two constants instead.

### 3. CLOSED -- the energy-2 dead-end is a symptom
Frozen, the same `_search` wins ls42 in 36 actions from the same start state, so it is
not deficient: the live run drifts into the energy-2 position once mispriced frames shift
rotation phases. Fixing item 2 is the way to close it. Do not work it as a search defect.

### 4. Smaller items
- ls335 A* plans nothing (`1 nodes: None` -> `2 nodes: None`) from the start state at
  full energy, frozen and unfrozen. Open defect; reproduce offline by feeding a live-read
  ls335 start state to `_search`. It also voids ls335 as an experiment arm.
  CAVEAT: measured with 4 arms in parallel while orphaned VICE containers were burning
  ~2.2 cores. `_search` has a 30 s WALL-CLOCK deadline (`astar_player.py:312`), so load
  can cut the node count. A 2-node stop looks like pruning, not a timeout, but re-run on
  an idle box before spending time on it.
- `SENTINEL_STEP_SIGMA` is 68.4 in code; clean runs measure **49.3 / 46.4**. Lowering it
  does not change the ls42 outcome and the margin tests pin concrete window/budget
  numbers, so the update needs a test-pin review.
- py65 exact backend no longer covers transfer settles: `_exact_render_cost` returns
  `None` for any non-player observer, and transfer settles are now always priced from a
  non-player slot.
- `actioncost.action_rounds` has zero callers — delete rather than maintain a duplicate
  of `playerbase._settle`.
- DONE: `REDRAW_BASE` / `STEPS_PER_EDGE` comments now declare FITTED, and
  `sentinel/tests/test_timing_registry.py` fails CI on any timing constant that is
  unregistered, claims evidence it lacks, or grows the 55-entry UNVALIDATED debt set.
- `test_player_placement_invariant` (250 s) and `test_player_wins_landscape_0042` (183 s)
  far exceed the 60 s budget, and did so before this work.
- `docs/player.md` was not reviewed this pass and may carry stale timing claims.
- `_RU_COMMIT` (4 s) on `run_until_pc(PC_PAN_DONE)` aborted 1 live run in 5 (`won=None`)
  before the conversion. Confirm it can no longer fire.

## Concurrency

Multiple VICE containers run in parallel. `ViceContainer` now publishes `-p 0:6502`
(docker picks a free host port; the fixed `6502:6502` bind was the hard blocker, failing
8/8), and `boot.stale_filter()` scopes teardown to `name=^asid-vice-<own pid>-`.
`VICE_REAP_ORPHANS=1` opts back into the blanket `ancestor=` sweep.

`SIGKILL` on a run leaks its container: teardown is scoped to the owning pid, so nothing
reaps it and it burns a core under warp indefinitely. Kill with TERM and check
`docker ps` -- three orphans survived 8-10 h here (~2.2 cores) and contaminated
measurements taken during that window.

## Live determinism (open)

`driver/test_live_determinism.py` runs the same frozen ls42 landscape twice: actions
match, per-step measured frames differ by +-1 on 2-3 of 8 steps. A 1-frame shift moves
the enemy cooldown phase (`UNIT_FRAMES` 3.746), which is the ~4 f window differences seen
in `plan_audit`, which accumulate until a gate flips and trajectories split (16/18/22
steps, 0/0/5 violations across three runs). `pred` is identical across runs -- the
planner is deterministic; the live driver is not.

Fixed so far: `clock.frames` and `core.live_image` read unguarded, so observing the
machine advanced it (reproducer: `run_frames(10)` measured 11, scan windows 2 not 1);
`auto_resume` is now cleared session-wide when play starts; vice-driver v0.4.1 scopes
`run_until_pc`'s wait to its own checknum.

Ruled out: wall-clock timeouts (none fire over a full run), warp (divergence persists
with warp off), asynchronous emulator stop (two boots gave byte-identical post-hit PC
sequences, so the off-target stop is state-deterministic).

FOUND and fixed (vice-driver v0.5.0). An instrumented emulator build (`ASID_CPDBG`,
logging every checkpoint hit with checknum/stop/clk) showed consecutive stop hits 1 frame
apart 1615 times and **2 frames apart 37 times**, every checkpoint distinct, none hit
twice: the stop was exact, the INSTALL was late. `CHECKPOINT_SET` sent to a running
machine is only serviced at the next vsync poll (`monitor_check_binary`, once per frame),
so the arm landed on a host-timed frame boundary. `run_until_pc` now caches one
checkpoint per target and toggles it while halted. The +-1 divergence is gone.

Residual (different bug): ~2 runs in 9, same actions but per-step frames differ by
tens-to-hundreds (+93 / +17 / -715). That magnitude is a retry loop taking a different
path -- `_run_to_scan` passes, `sentinel_execute` `range(3)`/`range(4)` retries -- not a
frame-stepping race. `driver/test_live_determinism.py` carries it as a non-strict xfail.

Also disproved on the way: stopping on a PC not shared with the frame counter ($9640)
does not help, and the emulator's stop itself is exact (300 ms idle after a stop advances
0 frames).

`pytest.ini`'s `--dist loadgroup` and the `xdist_group("vice")` marks have been
**reverted**: they were added on a misdiagnosis — the flaky live strict-xfail was the
wall-clock `_wait`, not xdist contention, and serialising did not fix it.

## Artifacts

`out/FACTS.md` (measured ground truth), `out/rerun_final_ls42.log` (clean post-fix run),
`out/rerun_sigma49b_ls42.log` (sigma 49.3), `out/sim_frozen_astar_0042.log` +
`out/f3_frozen_astar_0042.log` (the settled both-frozen ls42 win, sim and live),
`out/f2_*_335.log` (ls335 arms: A* 0 actions both ways), `out/frozen_*_335.log` (void
earlier arms), `out/play_player_0042.json` (`exact_audit`, `aim_subframes`, `frame_audit`,
`wait_audit`).
