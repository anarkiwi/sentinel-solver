# Policy and tuning (`sentinel/policy.py`, `sentinel/tune.py`)

`Policy` is every knob that is a **choice**, in one frozen dataclass whose defaults are the
shipped player. Each field is env-overridable as `SENTINEL_<FIELD>`, so one candidate
policy is one object a worker process can be handed.

ROM-derived quantities are deliberately **not** here — `DRAIN_DELAY`, `DITHER_FRAMES`, the
`$0C20` drain rate, the `$1335` cadence. Fitting a measurement launders a modelling error
into a constant, and `sentinel/tests/timing_registry.py` enforces provenance on those
separately.

| group | fields |
|---|---|
| search shape | `weight`, `top_targets`, `top_hops`, `top_clears`, `top_relocs`, `pursue_branch`, `max_pursue`, `max_reclaim`, `strand_prune` |
| heuristic | `target_eye`, `eye_per_hop` |
| risk | `margin_k` |
| unsettled rate options | `source_gate`, `build_toll`, `liquidity_gate` |

The last group is deliberately switchable rather than asserted: what ground truth settles
is settled in code, what was guesswork is left for the tuner to decide.

## The staged objective

```bash
python -m sentinel.tune --trials 200 --workers 6 --node-budget 4000 --cap 120
```

**Stage A — cheap and dense (~1 s, no search).** Replay every recorded human WIN and count
the moves the create gate refuses. A human win means each of those moves survived in the
real game, so a rejection is a *measured false positive*. 73 creates across four logs; the
shipped policy rejects 0. Treated as a constraint, not a score.

**Stage B — expensive and sparse.** Solve a board set under a **node** budget, never a wall
clock: `_search`'s deadline makes a plan host-load dependent, which would make the objective
noisy and the fit irreproducible. `REGRESSION` boards are constraints — losing one is
infeasible, so a policy cannot buy one board by losing another. `TARGETS` are scored on
win, then frames, then expansions.

Optuna TPE drives it, with a median pruner reading the running loss after each board, and
SQLite storage so a run resumes. Each board runs in a **spawned** process (a fork aborts:
the numba LOS march leaves an OpenMP runtime in the parent) with `NUMBA_NUM_THREADS` pinned
to `cores // workers`, because `march_batch` is `parallel=True` and uncapped workers put a
24-thread machine at load 117.

## What this cannot do

Tuning constants moves along the manifold a fixed functional form allows. When the form is
missing a term no coefficient can supply it: `_pick_hop` ranks on eye gain and stack cost
alone, so no setting of them expresses "do not climb onto a plateau a Sentinel permanently
watches". The two changes that actually moved the ls335 retrograde boundary were missing
*generators* (`_c_relocate`, `_c_reach`), not mis-set constants. Fit the scalars here; find
the features with [human_regress.md](human_regress.md).
