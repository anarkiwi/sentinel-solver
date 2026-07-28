# Iterating on a planner without replaying the board

Changing a generator and re-running a planner over a whole board costs **94 s** solo and
494 s under load. Almost none of that is the thing being changed: it is dozens of ticks
of already-correct play being re-derived from cold caches so the planner can arrive at
the one stance that fails.

The failure is a property of a **single state**. So re-enter that state instead of
replaying to it.

`sentinel/tests/ckpt.py` does this for both surviving planners: `PhasePlayer`
([phase_player.md](phase_player.md), the default) and `AStarPlayer`
([astar_player.md](astar_player.md)), selected by `restore(snap, cls=...)`.

## Tiers

Each tier answers a narrower question and costs 1-2 orders of magnitude less than the
next. Promote a change only when the tier below it passes. Measured on a reduced ls335
board, enemies `{(4,18), (12,10)}` plus the Sentinel:

| tier | question | measured |
|---|---|---|
| 0 filter tally | which gate kills each `(tile, k)` here? | **0-5 ms** |
| 1 probe | does the candidate generator (`_climb_candidates` / `_mount`, or A\*'s `_pick_hop` / `_expand`) yield anything here? | **140-210 ms** |
| 1b search | does one A\* `_search` from here return a plan? | 0.8-4.6 s |
| 2 resume | does the run still die from this tick on? | 0.5 s (5 acts) - 6.5 s (32 acts) |
| 3 board | does the whole board flip to a win? | 94 s |
| 4 matrix | did any other board regress? | existing `human_regress` |

Tier 0-1 is the loop you actually iterate in: a generator change is judged in **210 ms**
against the stance that defeats it, not 94 s.

## The checkpoint

The game state is one 64 KB `bytearray`, so a checkpoint is that image plus the player
scalars it does not carry. Everything else the player holds (`_view_memo`, `_cone_memo`,
`_hop_price_memo`, `_hold_memo`, the module-level `_VIEW_CACHE`) is a pure cache keyed
on a state signature and is rebuilt on demand.

Two images are stored, not one. The board the player was **constructed** on is stored
beside the live image, for any player whose caches snapshot it; restoring only the live
board would rebuild those caches from the wrong tile map. So `restore` constructs on the
start image and then overwrites the live image. A player with no such snapshot
(`PhasePlayer`) stores its live state twice.

```python
FIELDS = ("cursor", "last_bearing", "frames", "trace", "fire_reason", "_stale",
          "plan", "_pi", "expansions", "_hs_streak", "_depth", "_margin_k",
          "_on_plan", "_last_pbody", "waited")

def snapshot(player, tick=0):
    start = getattr(player, "_graph_state", player.st)
    return {"tick": tick,
            "start_mem": bytes(start.mem),
            "mem": bytes(player.st.mem),
            "fields": {n: copy.deepcopy(getattr(player, n))
                       for n in FIELDS if hasattr(player, n)}}

def restore(snap, cls=PhasePlayer, **kwargs):
    player = cls(Game(State(bytearray(snap["start_mem"]))), **kwargs)
    player.st.mem[:] = snap["mem"]
    for name, value in snap["fields"].items():
        setattr(player, name, copy.deepcopy(value))
    return player
```

One field list serves both planners: a field the player lacks is skipped, so `waited` is
carried for `PhasePlayer` and `plan`/`_pi`/`expansions` for `AStarPlayer`.

The deep copies are load-bearing on both sides, and both were caught by the fidelity
gate below rather than by inspection:

- on `snapshot`, because `trace`/`plan`/`cursor` are live mutable objects: storing
  references makes every checkpoint alias the final tick;
- on `restore`, because assigning the snapshot's own list to the player means replaying
  **mutates the checkpoint** — fatal for a corpus that is re-entered many times.

A whole run is small: 36 ticks, 88 KB compressed, ~2.4 KB a tick.

## The fidelity gate

A checkpoint is only worth having if re-entering it is indistinguishable from never
having left. The invariant:

> restoring at tick *t* and replaying to the end reproduces the run's own trace tail
> entry for entry.

```
36 checkpoints, last tick 35
{"from_tick": 30, "actions_replayed":  5, "identical": true, "wall": 0.48}
{"from_tick": 25, "actions_replayed": 10, "identical": true, "wall": 7.23}
{"from_tick": 15, "actions_replayed": 20, "identical": true, "wall": 6.67}
{"from_tick":  3, "actions_replayed": 32, "identical": true, "wall": 6.46}
FIDELITY PASS
```

The gate must assert a non-zero `actions_replayed`. Both of the bugs above first
presented as `identical: true` over an **empty** tail — a vacuous pass. This belongs in
CI on a small board: it is the test that keeps the field list honest as the player grows
state.

## The stall corpus

One checkpoint is a debugger. The set of them is a regression suite.

Keep, from every run and every board, the ticks where the candidate generator returned
nothing, plus the tick that **committed the hop** that led there. Score a change by how
many corpus stances now yield a viable child — a dense, millisecond-cheap objective in
place of one binary win/lose that costs 94 s. Boards that already win contribute their
own near-miss stances, so the corpus also catches regressions the win/lose bit hides.

## Persistent memoisation

The remaining cost after checkpointing is the LOS/aim sweep, which `playerbase` already
calls "~90% of a player's runtime" and memoises **in process**. Persist those memos to
disk under their existing keys, plus a hash of the source of the modules that compute
them, so editing pricing invalidates the cache automatically while editing an unrelated
generator keeps it warm. This is visible in the gate above: the first restore in a
process replays 5 actions in 0.48 s, and a later one replays 32 in 6.46 s, because the
process-level caches are warm by then. Persisting them buys that across processes.

## Determinism contract

Anything whose result is compared must be bounded by **node budget only**, never by
`time_budget`. A wall-clock cut makes the search a function of host load: an earlier
pass at `--time-budget 20` under 6-way parallelism reported permanent stalls on boards
that win solo in 15 s. With the clock out of the loop, parallelism changes wall time and
never a verdict, so the corpus can be scored across all cores at once.
