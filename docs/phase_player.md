# The phase player: freedom first, then convert

`sentinel/freeplayer.py`. A second planner, written from the game's own rules rather
than from a cost model, because the weighted search
([stance_planner.md](stance_planner.md)) does not convert ls335 and its failures were
never in the search — they were in what the search was asked to optimise.

It wins **all eight boards measured**, ls335 from entry in 73 actions — the board this
repo is named for losing — and ls110 in 53. See [Where it stands](#where-it-stands).

## The three facts it is built on

**Time is free; only exposure costs.** An idle frame costs nothing unless a cone is on
the body, so a body nobody can see may wait as long as it likes. The corollary is the
rule the first version got wrong: waiting is the one move that cannot help while you
ARE being drained.

**A gap does not need forecasting.** `enemies.step` is byte-exact, so instead of
predicting when a cone arrives (`_cone_onset`, and the errors that cost this repo a
documented open item), clone the state, run it forward over the action's real duration,
and look. `_drained_over` is four lines and is never wrong.

**Height buys freedom, and nothing else does.** Measured over the whole stance graph on
ls335:

| eye | tiles it can land | stances it can climb to | enemies it can strike |
|---|---|---|---|
| 4.0 | 6 | 18 | 0 |
| 7.0 | 8 | 6 | 0 |
| 8.5 | 62 | 146 | 0 |
| 9.5 | 107 | 140 | 2.91 |
| 11.5 | 153 | 13 | 4.78 |

**Nothing on the board can strike any enemy below eye 8.875**, and the body starts at
3.875. So the opening cannot be scored by progress toward a target: for the first five
units of climb there is no reachable target to progress toward. Exposure rises with
height too (`hot` 2.75 -> 4.9), so the climb is not into safety, it is into options.

## The phases

| phase | goal | ends when |
|---|---|---|
| **1a establish** | energy in hand, in fuel-rich ground | the best climb on offer is affordable |
| **1b breakout** | height, per unit of a finite purse | an enemy is landable |
| **2 convert** | absorb, Sentinel last, then the platform | hyperspace from the platform |

Splitting 1 from 2 took ls335 from unwinnable to won. Splitting 1a from 1b then halved
the opening, 107 actions to 55: the two want opposite things from the same tile — 1a wants
to stay where the trees are, 1b wants to leave for height — and a single scorer
blending them strands the body at eye 7.875 with 56 climbs available and one absorbable
object in reach.

## The rules that decide a move

Each is a fact about the ROM, not a tuned quantity.

* **Never idle under fire** — a cone on the body makes every waiting quantum a drain.
* **A hop's danger is at the destination**, and the probe must cover the *whole* hop
  including aims. Checking 420 frames instead of ~1500 is why it kept landing in live
  cones at E=1.
* **Never land on zero** — $1A00 kills on a drain arriving at zero energy.
* **Do not hop where you cannot eat** — a landing with no fuel under it is a dead end,
  whatever it gains in height.
* **The Sentinel is the point of no return** — $1B91 reads `objects_flags[0]`
  absolutely, so once its slot empties EVERY later absorb is refused. The robot and the
  3-energy hyperspace toll ($216A, which kills on underflow at $2170) must be banked
  before the strike, not gathered after it.
* **The endgame is exclusive** — with the Sentinel gone the platform is an eye gain, so
  an unguarded climb will spend the hyperspace toll on height.
* **A tie is settled by the outcome** — when the score cannot separate the candidates it
  holds no information about them, so each is played to the end and one that wins is
  kept ([the tie](#the-tie-that-decided-ls110)).

## Where it stands

There is no node budget and no wall-clock cutoff: the planner is deterministic, so a
run's outcome does not depend on host load and the times below are observation only.

| board | enemies | result | actions | energy left | wall |
|---|---|---|---|---|---|
| ls0 | 1 | won | 16 | 1 | 13 s |
| ls42 | 2 | won | 35 | 4 | 92 s |
| ls110 | 3 | **won** | 53 | 17 | 126 s |
| ls60 | 7 | won | 55 | 15 | 231 s |
| ls298 | 7 | won | 34 | 5 | 51 s |
| ls321 | 7 | won | 78 | 12 | 127 s |
| ls373 | 7 | won | 65 | 15 | 159 s |
| ls335 | 7 | won | 73 | 20 | 90 s |

**8 of 8.** ls335 and ls373 converted when the four invented `BUILD_FRAMES` durations
were replaced by each action's real aim-plus-settle (`_span`) — the planner was not
mis-reasoning about those boards, it was asking for gaps of the wrong length. ls110
converted when the tie below stopped being broken by dictionary order.

## The tie that decided ls110

The old run died at 17 actions, eye 7.375, E=0. The stall itself was forced: at the
previous tick 11 of its 12 candidates failed the arrival probe and the survivor was the
tile it died on. The decision that lost the board was four ticks earlier.

At that tick, eye 6.375 with E=10, **nine candidates scored identically** —
`(gain/cost, fuel_near) = (0.1667, 5)` for every one — so `_best_climb` returned
whichever `views.band()` yielded first. Forcing each of the nine and playing on with the
planner otherwise untouched:

| forced | outcome | | forced | outcome |
|---|---|---|---|---|
| (23,1) | **won** 73, E=11 | | (25,0) | lost — the one taken |
| (24,1) | **won** 53, E=17 | | (26,0) (26,8) (26,9) | lost |
| (25,4) | **won** 42, E=5 | | (24,11) (26,10) | lost |

Three of nine win the board outright. The score cannot see which, and neither can any
other measurement taken there: eye, energy, the stance graph's fuel, its seer count, its
landable-set size, its hop-distance to a strike, and the 4-tick fork outcome were all
computed for the nine, and **two of the three winners rank last on every one**. That is
why deepening the lookahead from 4 to 8 to 16 changed nothing: the winning hop's payoff
is 40 actions away, and no valuation at any affordable depth reaches it.

So the tie is not scored, it is **played**. `_settle_tie` forks each tied candidate,
builds it, and runs the fixed ladder to termination, keeping the first that wins. No
weight and no horizon — the only thing consulted is whether the game was won.

It is affordable because ties are rare (two of fifteen ticks on ls110) and because the
rollout is the fixed ladder, which plays a whole board in seconds. A tie costs one
rollout per candidate until one wins: ~50 s at the ls110 opening, where none does.

Two rules were tried against this trap before and are recorded as failures, because both
were attempts to score the move rather than play it:

| rule tried | result |
|---|---|
| absorb only if its value exceeds the drain over its own span | ls110 won, ls321 + ls373 lost — 6/8 |
| make `_supply`'s affordability test agree with `_best_climb`'s | ls42 + ls110 lost — 6/8 |
| route every climb through `stancegraph.route` instead | 5/8: ls42 and ls373 lost, ls110 still lost |

The first also **disproves an earlier claim in this document**, that `_refuel` harvests
"at a loss" under a cone. It cannot: idling through the same span costs the same drain
and yields nothing, so an absorb under fire is never worse than waiting.

## What is not solved

**Runtime.** 97% of it is `los._landable_batch`, the pitch-band sweep, at ~0.4 s a call,
and a tie now multiplies that by the number of candidates. Making the sweep cheap is the
enabling work for anything that wants to play more of the tree, and it is engineering
rather than strategy.
