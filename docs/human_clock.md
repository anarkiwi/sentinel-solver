# The recorded clock: measuring what an action cost

A `watch_play/3` fixture event carries the whole pre-action enemy clock — the `$1335`
Bresenham accumulator, the `$0C50` gate, and every enemy's `$0C28` rotation cooldown.
That is enough to recover, exactly, how many frames the real game advanced between two
recorded actions, with no cost model in the loop.

```bash
python -m pytest sentinel/tests/test_human_clock.py -q
```

## How the span is recovered

`$130C` adds `$CD` to the accumulator each frame and runs `$1317` only on the carry
(205/256 of frames); `$1317` decrements the cooldowns only on every third carry
(`$0C50`). So:

| quantity | what it pins |
|---|---|
| `$1335` accumulator | the frame count **mod 256** (`$CD` is invertible mod 256) |
| `$0C50` gate | lifts that to **mod 768** (205 carries per 256 frames, `205 % 3 == 1`) |
| `$0C28` sawtooth | picks the multiple — 200 rounds, reloaded at `$1813`, sticking at 1 |

An enemy whose rotation cooldown reloaded or sits on the stick value cannot vote on the
round count; a span is **exact** only when every remaining voter agrees. 91 of the 155
ls335 spans are exact. `human_clock.span_frames` is closed-form, and
`test_closed_form_matches_the_stepped_clock` checks it against the stepped loop over the
whole `(accumulator, gate)` space.

## What it grades

**The cooldown clock is right.** Seeded with a recorded clock and advanced by the
measured span, `enemies.advance_frames` lands on the next event's recorded `$1335`/`$0C50`
reading on **91 of 91** exact spans.

**The action cost model is not 6% long.** `span[i] = settle(verb[i-1]) + think[i] +
aim[i]`, and the human's think time sits on the measured side, so a correct bill should
land *under* the span. Over the exact spans between genuine player actions the bill is
**0.993x** measured — near parity, which means it overcharges by whatever the think time
was. The earlier "~6% long" figure came from the quantised `h_angle` proxy (an angle
only moves when a rotation fires, so it reads 0 or 749 and nothing between); the clock
supersedes it. On **17** spans the bill exceeds the *entire* elapsed time, which is proof
of overcharge regardless of think time.

**The enemy behaviour is not.** Over the same spans the facings follow on only **67 of
91**: an enemy the ROM holds in an earlier branch of `consider_enemy_state` (`$16E6`)
rotates in ours. Rotation is the last thing that routine reaches — after discharge, a
held target, a drainable robot, and a drainable boulder/tree have all declined — so a
facing that runs ahead means we exit a branch the ROM stays in. The same defect shows up
as energy: replayed per action from ground truth, ours ends one *above* the human on
several absorbs, a drain we do not model.

Swapping our billed cost for the exactly measured span changes the per-action energy
outcome on 1 of 122 actions. The charge model is therefore **not** the lever for the
ls335 replay floor; the enemy branch is.

## The sharpest lead: `$0C30` never reaches 1 in our model

Advancing the same 91 spans and reading each enemy's `update_cooldown` (`$0C30`) against
the recorded value:

| value | 1 | 2 | 3 | 4 | >4 |
|---|---|---|---|---|---|
| recorded | **393** | 30 | 49 | 18 | 16 |
| ours | 0 | 183 | 163 | 149 | 11 |

The real `$0C30` sits on the stick value 1 in 78% of samples; ours never does, and the
two agree on 41 of 506. `_consider_enemy_state` writes `UPDATE_COOLDOWN_SCAN` (4) at the
top as soon as it passes the `$16E9` gate, so in our model an enemy that decays to 1 is
reloaded on the very next consideration. For the ROM to leave it at 1 that often, either
the enemy is considered far more rarely than `UPDATES_PER_FRAME = 8` assumes, or `$16ED`
is not written on every path out of `$16E6`. Those two are distinguishable by reading
`$16E6` — not by fitting, and not from the facings alone, which is why the existing
400-frame instrument check passes without catching it.

## Fixture hygiene

The dither loop (`$1FA4`, ~50 f) and the transfer tune wait (`$35D5`, 96 f) are hard
floors on how close two real player actions can be. **8** exact ls335 spans fall below
the floor for the action preceding them, so those bracket pairs are one action recorded
twice — a recorder artifact class beyond the two `_is_player_action` already drops
(enemy discharge trees, drain ticks minted as self-transfers). ls335 also carries 33
events of those two known classes; `human_replay` skips them.
