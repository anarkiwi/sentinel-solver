# The phase player: freedom first, then convert

`sentinel/freeplayer.py`. A second planner, written from the game's own rules rather
than from a cost model, because the weighted search
([stance_planner.md](stance_planner.md)) does not convert ls335 and its failures were
never in the search — they were in what the search was asked to optimise.

In its fixed-ladder form it wins **ls335 from entry** in 55 actions — the board this
repo is named for losing. In the fork-arbiter form committed here it wins ls110 and
ls298 instead, and loses ls335. Six of eight either way; see [Where it
stands](#where-it-stands).

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

Splitting 1 from 2 took ls335 from unwinnable to won. Splitting 1a from 1b halved the
opening, 107 actions to 55: the two want opposite things from the same tile — 1a wants
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

## Where it stands

Node-budget-free; every run is bounded by wall clock only.

| board | enemies | result | actions | wall |
|---|---|---|---|---|
| ls0 | 1 | won | 16 | 26 s |
| ls42 | 2 | won | 44 | 56 s |
| ls110 | 3 | won | 65 | 131 s |
| ls60 | 7 | won | 38 | 69 s |
| ls298 | 7 | won | 30 | 70 s |
| ls321 | 7 | won | 58 | 126 s |
| ls373 | 7 | lost | 24 | 104 s |
| ls335 | 7 | lost | 32 | 76 s |

**6 of 8, and WHICH six depends on the arbitration order** — see below. With the fork
arbiter as committed, ls110 and ls298 win and ls335 does not; with the fixed ladder that
preceded it, ls335 wins in 55 actions and ls110 does not. Both configurations score 6,
which is the clearest statement of the open problem: the planner is one decision rule
short, not one board short.

The losses share a signature — `E=0, dead, most enemies alive` — the establish/breakout
boundary failing the way ls335 did before the fuel rule, not a new fault.

## What is not solved

**The arbiter.** A fixed priority ladder cannot express which of climb / harvest /
salvage is right at an ambiguous point: ordering them one way wins ls110 and loses
ls335, the other way does the reverse. The right answer is to decide by playing each
option out on a fork (`_arbitrate`), which is sound and which did fix ls110 — but an
honest rollout must rank on the same view model it plays, and that is unaffordable:
lookahead 2 costs ~52 s a board, lookahead 4 exceeds 120 s. Ranking forks on the cheap
primary sweep instead makes the arbiter decide a different game than it plays, and it
stalls at ls335's opening where that plane finds no climbs at all.

So the arbiter is blocked behind cost, and the cost is one function: **97% of runtime is
`los._landable_batch`**, the pitch-band sweep, at ~0.4 s a call. Make that cheap and the
rollout becomes affordable, the arbiter becomes honest, and the ordering problem
dissolves. That is the next thing to do, and it is engineering rather than strategy.

Second, `_refuel` has no value test: on ls373 it harvests under a cone at a loss,
taking +1 trees for 2 drains, and starves at eye 6.875 having eaten its own freshly
built boulder. An absorb should be worth what standing there costs.
