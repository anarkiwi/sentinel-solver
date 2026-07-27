# The minimal ls335 board the stance planner loses

ls335 with all 7 enemies is the board the planners lose. Deleting enemies from it
localises the failure to **three**: the Sentinel plus sentries at `(4,18)` and
`(12,10)`. That board is small enough to re-enter directly
([fast_iteration.md](fast_iteration.md)) instead of being replayed.

Enemies are removed with `actions.remove_object` ($1EEF), the ROM's own unlink, so the
tile map is repaired exactly as an absorb leaves it. Each variant has its own
`stancegraph.signature` and cache entry. Measured with `--planner stance` on this
branch, i.e. including the `_can_leave` departure gate.

## The result

| enemies | boards | lost |
|---|---|---|
| 2 (Sentinel + 1 sentry) | 6 | **0** |
| 3 (Sentinel + 2 sentries) | 15 | **1** — `{(4,18), (12,10)}` |

```
{"keep": [1,5], "tiles": [[4,18],[12,10]], "won": false, "why": "dead",
 "actions": 35, "frames": 12035, "energy": 0, "alive": 3}
```

It dies at E=0 with **all three enemies alive**.

Minimal in both senses:

- **by count** — every 2-enemy board wins, so 3 is the floor;
- **by subset** — both proper subsets win alone: `{(4,18)}` in 50 actions (E=7),
  `{(12,10)}` in 57 (E=3). Neither sentry is individually fatal.

`(12,10)` is the stressor but not sufficient: `{(12,10), (4,24)}` wins with the
*highest* spare energy of the 15 (E=14).

## The two-sentry matrix

Sentry slots are the generated order; `1=(4,18) 2=(2,1) 3=(30,27) 4=(7,11)
5=(12,10) 6=(4,24)`.

| pair | result | | pair | result |
|---|---|---|---|---|
| 1,2 | won 52 / E11 | | 2,6 | won 52 / E9 |
| 1,3 | won 66 / E6 | | 3,4 | won 53 / E12 |
| 1,4 | won 50 / E8 | | 3,5 | won 54 / E8 |
| **1,5** | **lost — dead, E0, 3 alive** | | 3,6 | won 53 / E11 |
| 1,6 | won 52 / E12 | | 4,5 | won 56 / E5 |
| 2,3 | won 52 / E11 | | 4,6 | won 53 / E12 |
| 2,4 | won 53 / E12 | | 5,6 | won 56 / E14 |
| 2,5 | won 54 / E4 | | | |

The four surviving `(12,10)` pairs are also the four most expensive: lowest energy
margins (E=4, 8, 5) bar `5,6`, and 1191-1510 s against 144-601 s for the rest, because
the search spends hundreds of expansions per re-plan in the same region where `1,5`
dies outright.

## Where it dies

Both planners die on the **same stance**: body on `(18,24)` at eye 8.375.

| | A\* on `{(12,10)}` (2 enemies) | stance on `{(4,18),(12,10)}` (3 enemies) |
|---|---|---|
| stall stance | `(18,24)` eye 8.375 | `(18,24)` eye 8.375 |
| actions | 32 | 35 |
| frames | 13876 | 12035 |
| enemies alive at death | 2 of 2 | 3 of 3 |

The stance planner **transfers onto `(18,24)` at E=0** (`f=7874 transfer (18,24) E=0
eye=8.375`). From there the only plan any search returns is a pure scavenge —
`plan (9 nodes): [absorb x19]`, no hop and no kill — energy still falls 3 -> 2 -> 1 -> 0
because the drain outruns the reclaims, and `_wait` is billed 1 energy per
`DRAIN_DELAY` ($178C holds a visible target so the cone never rotates off) until it
dies at f=12035.

The `_pick_hop` tally at death reads "98 of 104 landable tiles killed by no viable k",
but that is **downstream of E=0, not the cause**. The cause is upstream: a hop was
committed that lands the body with nothing left to leave on. The first question for
any fix is therefore why the climb onto `(18,24)` was permitted, not why the tiles
after it were refused — the pre-hop tick is in the checkpoint corpus for exactly that.

## Determinism

Every number here is bounded by **node budget only** (`--node-budget 2000`,
`time_budget=None`). A wall-clock cut makes the search a function of host load, not of
the board: an earlier pass at `--time-budget 20` under 6-way parallelism reported
stalls on boards that win solo in 15 s. With the clock out of the loop, parallelism
changes wall time and never a verdict.

The budget is calibrated, not guessed: across every run recorded here the largest
*completed* search was 76 expansions, and the ls335 root search returns a full winning
plan at 114, so 2000 is >17x headroom over any search that has ever succeeded. It is
also why a losing board is slow — it now spends the whole budget proving it has
nothing.
