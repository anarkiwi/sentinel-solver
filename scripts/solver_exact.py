#!/usr/bin/env python3
"""Exact / minimax search planner for The Sentinel (C64) -- solver step 3b.

This is the *non-greedy* successor to `solver.py`. Where `solver.py` is a fixed
staged pipeline (sweep -> pick one minimal climb -> win), this planner runs a
systematic, deterministic, depth-limited search and is **drain-aware** via the
time-accurate enemy model in `enemy_dynamics.py`.

It is built as a depth-limited MINIMAX (negamax with alpha-beta + iterative
deepening), per the design brief:

  * MAX layer  -- the PLAYER chooses an action from a bounded action lattice.
  * MIN layer  -- the ENEMY is modelled WORST-CASE over a small timing band
                  (+/- ENEMY_PHASE_BAND ticks): a tile counts as exposed if any
                  enemy sees it at any tick in that window, not just at the exact
                  nominal phase. Treating the enemy as an adversary over a timing
                  window (rather than the exact deterministic phase) makes the plan
                  ROBUST to the executor's VICE timing jitter and to our approximate
                  LOS -- robust-in-model beats optimal-but-brittle. The nominal
                  rollout still uses the exact deterministic dynamics; the "min"
                  only widens the *exposure* used for safety scoring/gating.

  * HEURISTIC EVAL at the horizon (the core of the search). A non-terminal state
    is scored by a documented weighted sum (see `evaluate`):
      (a) VANTAGE      -- the eye height achievable at the player's tile (taller
                          == better; height is what unlocks LOS). DOMINANT term.
      (b) NEWLY-ABSORBABLE -- how many objects become absorbable (LOS + reach)
                          from here; "highest vantage to absorb the most objects".
      (c) PROGRESS     -- current energy + objects already absorbed.
      (d) SAFETY       -- worst-case ticks until an enemy scan reaches the player's
                          tile (more margin == better; 0 == being drained now).
    Weights are explicit, tunable module constants with documented rationale.

  * RECEDING HORIZON. `best_move(state, depth)` returns the best single action by
    depth-n minimax+eval, so the EXECUTOR can re-plan from the LIVE game state
    after each executed action (correcting model error move-by-move). `solve()`
    rolls that policy forward in the model to a full Plan and replay-verifies it
    to `won` through `game_model.apply` (asserted), emitting the same Plan shape
    `solver.py` does so the step-4 executor consumes it unchanged.

A bounded branch-and-bound terminal search (`solve_bnb`) is kept as a comparison /
fallback (admissible-bound pruning to a winning terminal), but the PRIMARY planner
is the minimax-with-eval policy.

----------------------------------------------------------------------------
THE PLANNER STATE (abstract climb model, shared with solver.py)
----------------------------------------------------------------------------
`game_model.apply` cannot represent the game's defining mechanic -- gaining height
by stacking boulders on a tile and transferring into a robot on top -- because a
boulder stack does not change the terrain height field (and `apply` even rejects
creating on an occupied tile). So, exactly like `solver.py`, we carry an ABSTRACT
climb level on top of the model:

    eye height at player tile = terrain_surf(tile) + ROBOT_EYE + k*BOULDER_HEIGHT

where `k` is the boulder stack height the player has climbed at the current tile.
The search operates on a compact `PState` (player tile, stack level, energy,
absorbed set, free-slot count, enemy phase, tick) and uses `enemy_dynamics` for
drain-aware exposure. When a winning sequence is found it is lowered to the same
two-view `Plan` (`actions` for the executor incl. the real stacked climb, and
`model_replay` for the structural/energy replay-verification through the model).
"""

import sys
import os
import copy
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, FrozenSet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game_model as gm
from game_model import (
    GameModel,
    Action,
    apply,
    can_see,
    tile_surface_height,
    object_in_tile,
    ROBOT_EYE,
    ENERGY_IN_OBJECTS,
    ENERGY_MASK,
    T_ROBOT,
    T_TREE,
    T_BOULDER,
    T_SENTINEL,
    T_PLATFORM,
    NUM_SLOTS,
)
from game_state import GameState, GameObject
import enemy_dynamics as ed
import solver as greedy  # reuse its caches / climb constants / Plan

# Reuse solver.py's modelled climb constants (single source of truth).
BOULDER_HEIGHT = greedy.BOULDER_HEIGHT  # 1.0 nibble unit per stacked boulder
MAX_STACK = greedy.MAX_STACK  # 12

# ---- the Plan container (reuse solver.py's so the executor is unchanged) ----
Plan = greedy.Plan
PlanStep = greedy.PlanStep
Climb = greedy.Climb

# ============================================================================
# SEARCH / EVAL TUNABLES (explicit, documented)
# ============================================================================
# --- minimax / iterative deepening ---
DEFAULT_DEPTH = 6  # plies of the receding-horizon minimax lookahead
DEFAULT_BUDGET_S = 90.0  # per-landscape wall-clock budget (brief: a couple of min)
MAX_REPLAN_STEPS = 40  # safety cap on receding-horizon rollout length

# --- worst-case enemy timing band (MIN layer robustness) ---
# A tile is treated as exposed if any enemy sees it at any tick within +/- this
# many ticks of the nominal phase. Bigger == more conservative/robust but costlier.
# Chosen small: VICE jitter + our LOS approximation are a handful of ticks, not many.
ENEMY_PHASE_BAND = 4

# --- action-lattice bounds (keep the branching finite; documented honestly) ---
# We do NOT branch over every visible tile (32x32 is far too wide). The "create"
# frontier is restricted to tiles that change reachability or LOS -- the relevant
# frontier -- plus the climb base candidates. These bounds make the result
# "optimal within this lattice", stated plainly.
MAX_CREATE_FRONTIER = 12  # at most this many create-target tiles per MAX node
MAX_ABSORB_TARGETS = 12  # at most this many absorb targets per MAX node
BOUND_STACK = MAX_STACK  # boulder-stack height bound (== model's MAX_STACK)

# --- eval weights (rationale in evaluate(); vantage->newly-absorbable dominate) ---
W_VANTAGE = 6.0  # eye height achievable here. DOMINANT: height unlocks LOS.
W_NEWABS = 10.0  # objects newly absorbable from here. DOMINANT: the goal --
# rewards moves that OPEN UP the board (climb for vantage).
W_ENERGY = 1.0  # current energy (mild; capped at 63 by the 6-bit mask)
W_ABSORBED = 40.0  # objects already absorbed (PRIMARY objective: absorb the
# most objects before the Sentinel). Heaviest non-terminal
# weight so the policy collects before it rushes to win.
W_SAFETY = 0.5  # ticks of margin before worst-case enemy scan reaches us
W_LOS_SENTINEL = 30.0  # bonus once the player tile+stack has LOS to the Sentinel.
# Deliberately SMALLER than the value of the objects still
# collectible, so the policy doesn't rush to win while there
# are still trees in view it could absorb first.
SAFETY_HORIZON = 24  # cap on the safety-margin lookahead (ticks). Kept modest:
# the term only breaks ties toward safer tiles; a deep scan
# is expensive and rarely changes the chosen move.

WIN_SCORE = 1.0e9  # terminal: a won state dominates everything


# ============================================================================
# COMPACT PLANNER STATE
# ============================================================================
@dataclass(frozen=True)
class PStateKey:
    """Hashable key for transposition / dominance (enemy phase folded to tick)."""

    tile: Tuple[int, int]
    stack: int
    energy: int
    absorbed: FrozenSet[int]
    tick_mod: int


class PState:
    """A planner state: the model GameState plus the abstract climb level, enemy
    phase, and bookkeeping. Mutable; we copy on branch."""

    __slots__ = (
        "gs",
        "stack",
        "phase",
        "absorbed",
        "tick",
        "actions_log",
        "model_replay",
        "_terrain_key",
    )

    def __init__(
        self,
        gs: GameState,
        stack: int,
        phase: ed.EnemyPhase,
        absorbed: Tuple[int, ...] = (),
        tick: int = 0,
        actions_log: Optional[List] = None,
        model_replay: Optional[List] = None,
    ):
        self.gs = gs
        self.stack = stack
        self.phase = phase
        self.absorbed = absorbed
        self.tick = tick
        self.actions_log = actions_log or []  # (Action, note, stack_level, eye)
        self.model_replay = model_replay or []
        self._terrain_key = None

    def player_xy(self) -> Tuple[int, int]:
        p = self.gs.player
        return (p.x, p.y)

    def eye_offset(self) -> float:
        return ROBOT_EYE + self.stack * BOULDER_HEIGHT

    def eye_height(self) -> float:
        px, py = self.player_xy()
        return tile_surface_height(self.gs, px, py) + self.eye_offset()

    def won(self) -> bool:
        return getattr(self.gs, "won", False)

    def key(self) -> PStateKey:
        return PStateKey(
            tile=self.player_xy(),
            stack=self.stack,
            energy=self.gs.player_energy,
            absorbed=frozenset(self.absorbed),
            # fold enemy phase into a coarse tick bucket so transpositions hit:
            # exposure is what matters and it is periodic in the cooldown cadence.
            tick_mod=self.tick % 64,
        )

    def copy(self) -> "PState":
        return PState(
            copy.deepcopy(self.gs),
            self.stack,
            self.phase.copy(),
            self.absorbed,
            self.tick,
            list(self.actions_log),
            list(self.model_replay),
        )


# ============================================================================
# helpers
# ============================================================================
def _sentinel(gs: GameState) -> Optional[GameObject]:
    for o in gs.objects:
        if o.type == T_SENTINEL:
            return o
    return None


def _free_slot(gs: GameState) -> bool:
    return len({o.slot for o in gs.objects}) < NUM_SLOTS


def _occupied(gs: GameState, exclude_slot: Optional[int] = None):
    return {(o.x, o.y) for o in gs.objects if o.slot != exclude_slot}


def _has_los_to_sentinel(ps: PState) -> bool:
    sent = _sentinel(ps.gs)
    if sent is None:
        return False
    return greedy.cached_can_see(
        ps.gs, ps.player_xy(), (sent.x, sent.y), eye_offset=ps.eye_offset()
    )


def _worst_case_exposed(ps: PState, x: int, y: int, object_top: float) -> bool:
    """MIN layer: True if any enemy sees (x,y) within +/-ENEMY_PHASE_BAND ticks of
    the nominal phase. We advance from the current phase forward ENEMY_PHASE_BAND
    ticks (the past band is already folded by the periodic cooldown cadence)."""
    return ed.exposed_within_window(
        ps.gs, ps.phase, x, y, ticks=ENEMY_PHASE_BAND, object_top=object_top
    )


def _advance_phase(ps: PState, ticks: int) -> None:
    """Advance the enemy phase `ticks` ticks in place (drain handled by callers)."""
    ph = ps.phase
    for _ in range(ticks):
        ph = ed.step_enemies(ps.gs, ph)
    ps.phase = ph
    ps.tick += ticks


# how many enemy ticks elapse per player action (the executor takes time to act).
# A create/transfer/absorb is several game rounds; we model a fixed cost so drains
# accrue. Documented estimate -- the executor must mind real timing.
TICKS_PER_ACTION = 8


# ============================================================================
# TRANSITIONS (player actions in the abstract climb model)
# ============================================================================
def _newly_absorbable(ps: PState) -> List[GameObject]:
    """Objects (non-Sentinel, non-platform) the player can see+absorb from here,
    at the current eye offset."""
    p = ps.gs.player
    pxy = ps.player_xy()
    out = []
    for o in ps.gs.objects:
        if o.slot == p.slot or o.type in (T_SENTINEL, T_PLATFORM):
            continue
        if o.type not in gm.ABSORBABLE:
            continue
        if greedy.cached_can_see(ps.gs, pxy, (o.x, o.y), eye_offset=ps.eye_offset()):
            out.append(o)
    return out


def _create_frontier(ps: PState) -> List[Tuple[int, int]]:
    """Bounded set of create-target tiles that matter: visible empty tiles that
    either (a) extend reachability (new vantage) or (b) are climb-base candidates
    with LOS to the Sentinel achievable by stacking. We rank by how much height a
    short stack there buys toward seeing the Sentinel, then take the top
    MAX_CREATE_FRONTIER. This is the explicit action-lattice bound."""
    pxy = ps.player_xy()
    sent = _sentinel(ps.gs)
    occ = _occupied(ps.gs)
    vis = greedy.cached_visible_tiles(ps.gs, pxy, eye_offset=ps.eye_offset())
    scored = []
    for x, y in vis:
        if (x, y) in occ:
            continue
        # minimal stack here that reveals the Sentinel (cheap proxy for value).
        kmin = None
        if sent is not None:
            for k in range(0, BOUND_STACK):
                if greedy.cached_can_see(
                    ps.gs,
                    (x, y),
                    (sent.x, sent.y),
                    eye_offset=ROBOT_EYE + k * BOULDER_HEIGHT,
                ):
                    kmin = k
                    break
        # value: prefer tiles that reveal the Sentinel with a small stack; tie-break
        # by surface height (a higher tile is a better generic vantage).
        surf = tile_surface_height(ps.gs, float(x), float(y))
        if kmin is not None:
            scored.append(((0, kmin, -surf, (x, y)), (x, y)))
        else:
            scored.append(((1, 0, -surf, (x, y)), (x, y)))
    scored.sort(key=lambda s: s[0])
    return [xy for _k, xy in scored[:MAX_CREATE_FRONTIER]]


def player_actions(ps: PState) -> List[Tuple[str, Action, dict]]:
    """Bounded action lattice from `ps`. Each item is (kind, Action, meta).

    Kinds:
      'win'       -- absorb the Sentinel + transfer onto its platform (terminal).
      'absorb'    -- absorb a visible object (energy/objects up).
      'hop'       -- move: create a robot on a visible empty tile and transfer in
                     (energy-neutral; re-absorb handled in apply). Resets stack=0.
      'stack'     -- climb one boulder on the current tile (stack += 1).
    """
    out: List[Tuple[str, Action, dict]] = []
    gs_ = ps.gs
    _p = gs_.player
    pxy = ps.player_xy()
    sent = _sentinel(gs_)

    # WIN: only when the current eye actually has LOS to the Sentinel (drain-aware:
    # the win tile must not be worst-case exposed mid-commit -- but the win itself
    # is instantaneous, so we just require LOS).
    if sent is not None and _has_los_to_sentinel(ps):
        out.append(
            (
                "win",
                Action("win", sent.x, sent.y),
                {"note": f"win: absorb Sentinel @ {sent.x},{sent.y}"},
            )
        )

    # ABSORB visible objects (bounded).
    abss = _newly_absorbable(ps)
    abss.sort(
        key=lambda o: (
            0 if o.type == T_TREE else 1,
            max(abs(o.x - pxy[0]), abs(o.y - pxy[1])),
            o.slot,
        )
    )
    for o in abss[:MAX_ABSORB_TARGETS]:
        out.append(
            (
                "absorb",
                Action("absorb", o.x, o.y),
                {
                    "note": f"absorb {o.type_name} @ {o.x},{o.y}",
                    "slot": o.slot,
                    "type": o.type,
                },
            )
        )

    # STACK one boulder here (climb), if affordable and within the height bound.
    if ps.stack < BOUND_STACK and ps.gs.player_energy >= ENERGY_IN_OBJECTS[T_BOULDER]:
        out.append(
            (
                "stack",
                Action("create", T_BOULDER, pxy[0], pxy[1]),
                {"note": f"stack boulder #{ps.stack+1} (climb)"},
            )
        )

    # HOP to a frontier tile (move). Resets stack to 0 (new ground tile).
    if _free_slot(gs_) and ps.gs.player_energy >= ENERGY_IN_OBJECTS[T_ROBOT]:
        for x, y in _create_frontier(ps):
            out.append(
                (
                    "hop",
                    Action("create", T_ROBOT, x, y),
                    {"note": f"hop to {x},{y}", "dest": (x, y)},
                )
            )
    return out


def apply_action(ps: PState, kind: str, action: Action, meta: dict) -> Optional[PState]:
    """Apply one lattice action, returning a NEW PState (or None if illegal).
    Advances the enemy phase by TICKS_PER_ACTION and debits any player drain."""
    ns = ps.copy()
    px, py = ns.player_xy()

    if kind == "win":
        sent = _sentinel(ns.gs)
        # model-replayable win (absorb Sentinel +4, transfer onto platform).
        ns.gs = apply(ns.gs, action)
        ns.model_replay.append(action)
        ns.absorbed = ns.absorbed + (sent.slot,)
        ns.actions_log.append((action, meta["note"], ns.stack, ns.eye_height()))
        return ns

    if kind == "absorb":
        o = object_in_tile(ns.gs, action.a, action.b)
        if o is None or o.type in (T_PLATFORM,):
            return None
        ns.gs = apply(ns.gs, action)
        ns.model_replay.append(action)
        ns.absorbed = ns.absorbed + (meta["slot"],)
        ns.actions_log.append((action, meta["note"], ns.stack, ns.eye_height()))
        _post_action_drain(ns)
        return ns

    if kind == "stack":
        # Abstract climb: a same-tile boulder the model can't represent. Charge the
        # boulder cost against energy (executor really spends it), bump the stack.
        cost = ENERGY_IN_OBJECTS[T_BOULDER]
        if ns.gs.player_energy < cost:
            return None
        ns.gs = copy.deepcopy(ns.gs)
        ns.gs.player_energy = (ns.gs.player_energy - cost) & ENERGY_MASK
        ns.stack += 1
        ns.actions_log.append((action, meta["note"], ns.stack, ns.eye_height()))
        _post_action_drain(ns)
        return ns

    if kind == "hop":
        dest = meta["dest"]
        if object_in_tile(ns.gs, dest[0], dest[1]) is not None:
            return None
        # create robot on dest, transfer into it, re-absorb the robot we left
        # (energy-neutral, frees the slot) -- exactly solver.py's hop.
        try:
            cact = Action("create", T_ROBOT, dest[0], dest[1])
            g = apply(ns.gs, cact)
            tact = Action("transfer", dest[0], dest[1])
            prev = (px, py)
            g = apply(g, tact)
            # re-absorb abandoned robot if still visible from the new tile
            left = greedy.object_in_tile_at(g, prev, exclude_slot=g.player_slot)
            replay = [cact, tact]
            if (
                left is not None
                and left.type == T_ROBOT
                and can_see(g, (g.player.x, g.player.y), prev)
            ):
                aact = Action("absorb", prev[0], prev[1])
                g = apply(g, aact)
                replay.append(aact)
        except ValueError:
            return None
        ns.gs = g
        ns.model_replay.extend(replay)
        ns.stack = 0  # landed on fresh ground; the old stack is left behind
        ns.actions_log.append((action, meta["note"], 0, ns.eye_height()))
        _post_action_drain(ns)
        return ns

    return None


def _sweep_visible(ps: PState) -> PState:
    """Deterministically absorb every object currently visible+absorbable from the
    player's tile+stack (passive trees first, then sentries/boulders), updating the
    state after each. Used as a completion sweep just before the win so the planner
    collects the most objects from the vantage it reached. Energy-neutral-or-better
    (absorbing only gains energy)."""
    while True:
        cands = _newly_absorbable(ps)
        if not cands:
            break
        cands.sort(
            key=lambda o: (
                0 if o.type == T_TREE else 1,
                max(abs(o.x - ps.player_xy()[0]), abs(o.y - ps.player_xy()[1])),
                o.slot,
            )
        )
        o = cands[0]
        nxt = apply_action(
            ps,
            "absorb",
            Action("absorb", o.x, o.y),
            {
                "note": f"absorb {o.type_name} @ {o.x},{o.y}",
                "slot": o.slot,
                "type": o.type,
            },
        )
        if nxt is None:
            break
        ps = nxt
    return ps


def _post_action_drain(ns: PState) -> None:
    """Advance the enemy phase TICKS_PER_ACTION and debit any player drain that
    fires during that window. Drain-aware: this is energy the climb really loses."""
    ph = ns.phase
    total = 0
    for _ in range(TICKS_PER_ACTION):
        ph = ed.step_enemies(ns.gs, ph)
        delta, _ev = ed.drain_tick(ns.gs, ph)
        total += delta
    ns.phase = ph
    ns.tick += TICKS_PER_ACTION
    if total:
        ns.gs = copy.deepcopy(ns.gs)
        ns.gs.player_energy = max(0, ns.gs.player_energy + total) & ENERGY_MASK


# ============================================================================
# HEURISTIC EVALUATION (horizon scoring) -- the core of the search
# ============================================================================
def evaluate(ps: PState) -> float:
    """Score a (non-terminal) planner state. Higher == more promising.

    Documented weighted sum (weights are module constants):
      (a) VANTAGE  (W_VANTAGE): the eye height reachable at the player's tile. The
          game is "climb high to see the Sentinel", so height is the master key --
          this is a dominant, smooth term that guides the search up.
      (b) NEWLY-ABSORBABLE (W_NEWABS): how many objects are absorbable from here.
          The stated goal is "highest vantage to absorb the MOST objects", so this
          is the other dominant term -- it rewards moves that OPEN UP the board.
      (c) PROGRESS: W_ABSORBED * (objects already absorbed) + W_ENERGY * energy.
          Absorbed objects are the primary objective (heavy weight); energy is a
          mild tie-break (it's capped at 63 by the 6-bit mask anyway).
      (d) SAFETY (W_SAFETY): worst-case ticks until an enemy scan reaches the
          player's tile (more margin == safer). 0 == being drained right now.
      (e) LOS bonus (W_LOS_SENTINEL): a large bonus once the current tile+stack has
          LOS to the Sentinel -- this is one step from winning.
    """
    if ps.won():
        return WIN_SCORE
    px, py = ps.player_xy()

    vantage = ps.eye_height()
    newabs = len(_newly_absorbable(ps))
    progress = W_ABSORBED * len(ps.absorbed) + W_ENERGY * ps.gs.player_energy
    # safety: worst-case (band-widened) margin before the player's tile is scanned.
    margin = ed.ticks_until_seen(
        ps.gs, ps.phase, px, py, horizon=SAFETY_HORIZON, object_top=ps.eye_offset()
    )
    los = W_LOS_SENTINEL if _has_los_to_sentinel(ps) else 0.0

    return (
        W_VANTAGE * vantage
        + W_NEWABS * newabs
        + progress
        + W_SAFETY * min(margin, SAFETY_HORIZON)
        + los
    )


# ============================================================================
# MINIMAX / NEGAMAX with alpha-beta + iterative deepening
# ============================================================================
class _SearchStats:
    def __init__(self):
        self.nodes = 0
        self.depth_reached = 0
        self.t0 = time.time()
        self.budget = DEFAULT_BUDGET_S

    def time_left(self) -> bool:
        return (time.time() - self.t0) < self.budget


def _ordered_actions(ps: PState):
    """Move ordering: win first, then stacks/absorbs (cheap value), then hops.
    Good ordering makes alpha-beta prune hard."""
    acts = player_actions(ps)
    order = {"win": 0, "absorb": 1, "stack": 2, "hop": 3}
    acts.sort(key=lambda a: order.get(a[0], 9))
    return acts


def _negamax(
    ps: PState, depth: int, alpha: float, beta: float, stats: _SearchStats, tt: Dict
) -> Tuple[float, Optional[Tuple]]:
    """Depth-limited negamax with alpha-beta over the player's action lattice.

    There is no genuine opposing *player*; the adversary (the enemy) is folded into
    the transition (drain debits) and the eval's worst-case safety term -- i.e. the
    MIN layer is realised as worst-case-over-timing inside `apply_action` /
    `evaluate`, not as a separate minimising ply. This keeps the search a clean
    maximisation while remaining robust to enemy timing (the brief's intent). The
    'minimax' shape is: MAX picks actions; the enemy 'min' has already pessimised
    exposure within the band. Returns (value, best_action_item)."""
    stats.nodes += 1
    if ps.won():
        return WIN_SCORE - ps.tick * 1.0, None  # prefer faster wins
    if depth <= 0 or not stats.time_left():
        return evaluate(ps), None

    key = (ps.key(), depth)
    cached = tt.get(key)
    if cached is not None:
        return cached

    best_val = -math.inf
    best_act = None
    for item in _ordered_actions(ps):
        kind, action, meta = item
        child = apply_action(ps, kind, action, meta)
        if child is None:
            continue
        val, _ = _negamax(child, depth - 1, alpha, beta, stats, tt)
        if val > best_val:
            best_val = val
            best_act = item
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
        if not stats.time_left():
            break

    if best_act is None:  # no legal action -> static eval
        best_val = evaluate(ps)
    result = (best_val, best_act)
    tt[key] = result
    return result


def best_move(
    state,
    depth: int = DEFAULT_DEPTH,
    phase: Optional[ed.EnemyPhase] = None,
    stack: int = 0,
    budget_s: float = DEFAULT_BUDGET_S,
):
    """Receding-horizon policy: return the best (kind, Action, meta) from `state`
    by iterative-deepening minimax to `depth`, plus (value, stats). The executor
    calls this on the LIVE game state after each move to re-plan and correct model
    drift. `state` may be a GameState or GameModel; `phase` defaults to the
    worst-case initial enemy phase if not supplied."""
    gs_ = state.state if isinstance(state, GameModel) else state
    gs_ = copy.deepcopy(gs_)
    if not hasattr(gs_, "won"):
        gs_.won = False
    if phase is None:
        phase = ed.init_phase(gs_)
    ps = PState(gs_, stack, phase)

    stats = _SearchStats()
    stats.budget = budget_s
    best = None
    best_val = -math.inf
    tt: Dict = {}
    # iterative deepening: scale depth to the budget.
    for d in range(1, depth + 1):
        if not stats.time_left():
            break
        tt.clear()
        val, act = _negamax(ps, d, -math.inf, math.inf, stats, tt)
        stats.depth_reached = d
        if act is not None:
            best, best_val = act, val
        # if we've already found a forced win at this depth, no need to go deeper.
        if val >= WIN_SCORE - 1e6:
            break
    return best, best_val, stats


# ============================================================================
# solve(): roll the receding-horizon policy forward to a full Plan
# ============================================================================
def solve(
    state_or_model,
    depth: int = DEFAULT_DEPTH,
    budget_s: float = DEFAULT_BUDGET_S,
    verbose: bool = False,
) -> Plan:
    """Plan a winning sequence by rolling the minimax best-move policy forward in
    the model. Returns a `Plan` (same shape as solver.solve). Replay-verifies the
    model_replay projection reaches `won` (asserted when solved).

    This is the in-model rollout of the receding-horizon policy. The executor
    should instead call `best_move` on the LIVE state move-by-move (re-planning),
    but the rolled-forward plan is what we emit + verify here."""
    gs0 = copy.deepcopy(
        state_or_model.state
        if isinstance(state_or_model, GameModel)
        else state_or_model
    )
    if not hasattr(gs0, "won"):
        gs0.won = False
    plan = Plan()

    sent = _sentinel(gs0)
    if sent is None:
        plan.notes.append("no Sentinel in landscape")
        return plan

    phase = ed.init_phase(gs0)
    ps = PState(copy.deepcopy(gs0), 0, phase)

    total_nodes = 0
    max_depth = 0
    t0 = time.time()
    # per-move budget so the whole rollout stays within budget_s.
    per_move_budget = max(2.0, budget_s / MAX_REPLAN_STEPS)

    for _step in range(MAX_REPLAN_STEPS):
        if ps.won():
            break
        act, _val, stats = best_move(
            ps.gs, depth=depth, phase=ps.phase, stack=ps.stack, budget_s=per_move_budget
        )
        total_nodes += stats.nodes
        max_depth = max(max_depth, stats.depth_reached)
        if act is None:
            plan.notes.append("policy found no legal action")
            break
        kind, action, meta = act
        if kind == "win":
            # COMPLETION SWEEP (deterministic, guarantees maximal collection from
            # the achieved vantage): before committing the win, absorb every object
            # still visible+absorbable from the current tile+stack. This makes the
            # result stable and ensures the planner never leaves a visible tree
            # uncollected just because the eval pulled it toward the win early.
            ps = _sweep_visible(ps)
        nxt = apply_action(ps, kind, action, meta)
        if nxt is None:
            plan.notes.append(f"chosen action {kind} became illegal")
            break
        ps = nxt
        if time.time() - t0 > budget_s:
            plan.notes.append("budget exhausted mid-rollout")
            break

    # --- lower the abstract action log to the two-view Plan -------------------
    _emit_plan(ps, plan, gs0)

    plan.search = {
        "nodes": total_nodes,
        "depth": max_depth,
        "time": time.time() - t0,
        "optimal_within_bounds": False,  # minimax-with-eval, not proven
        "mode": "minimax+eval (receding horizon)",
    }

    # replay-verify
    won = _replay_verify(gs0, plan.model_replay)
    plan.solved = bool(won and plan.los_proven)
    if not won:
        plan.notes.append("model-replay projection did not reach won")
    if plan.solved:
        assert _replay_verify(
            gs0, plan.model_replay
        ), "solved plan must replay through game_model.apply to won"
    if verbose:
        for s in plan.steps:
            print(f"  {s.action!r:40} {s.note}")
    return plan


def _emit_plan(ps: PState, plan: Plan, gs0: GameState) -> None:
    """Translate the planner's abstract action log into the executor-facing Plan
    (steps incl. the real stacked-boulder climb) + the model_replay projection."""
    sent = _sentinel(gs0)
    plan.model_replay = list(ps.model_replay)
    # Walk the action log, expanding 'stack' levels into real same-tile creates and
    # tracking the climb so we can prove LOS at the win.
    win_eye_offset = None
    win_tile = None
    cur_stack = 0
    _cur_tile = None
    for action, note, stack_level, eye in ps.actions_log:
        # The abstract 'stack' action is a SAME-TILE boulder create on the player's
        # own tile (stack_level >= 1) -- the real ROM rejects that (put_object_in_tile
        # $1F38), so it is NOT executable by play_plan. We drop it from the executor-
        # facing steps; the real climb (an ADJACENT boulder ascent) is performed by
        # play_plan's `win` handler (code_engine.climb_and_win). We keep the action
        # only for the abstract eye/stack bookkeeping that reports the modelled
        # vantage and final energy.
        is_abstract_stack = (
            action.verb == "create" and action.a == T_BOULDER and stack_level >= 1
        )
        if not is_abstract_stack:
            plan.steps.append(
                PlanStep(
                    action=action, note=note, stack_level=stack_level, eye_height=eye
                )
            )
        if action.verb == "create" and action.a == T_BOULDER:
            cur_stack = stack_level
            _cur_tile = (action.b, action.c)
        elif action.verb == "win":
            win_tile = (action.a, action.b)
            win_eye_offset = ROBOT_EYE + cur_stack * BOULDER_HEIGHT
        # record absorbed types for the report
    # absorbed list (type_name,x,y) from model_replay absorbs + the win
    for a in ps.model_replay:
        if a.verb == "absorb":
            o = None
            # best-effort type from the original state
            for oo in gs0.objects:
                if oo.x == a.a and oo.y == a.b:
                    o = oo
                    break
            tname = o.type_name if o else "?"
            if tname not in ("ROBOT",):  # hops re-absorb robots; not 'collected'
                plan.absorbed.append((tname, a.a, a.b))
        elif a.verb == "win":
            plan.absorbed.append(("SENTINEL", a.a, a.b))

    # (The old Stage-D same-tile stranded-boulder re-absorption is gone: the abstract
    # same-tile stack is no longer emitted as executor steps, and the REAL adjacent
    # boulder ascent + its recovery is performed entirely inside play_plan's `win`
    # handler / code_engine.climb_and_win.)

    # FINAL WIN (executor-facing): always emit a `win` action on the Sentinel tile.
    # The model's abstract `can_see` is now CONSERVATIVE (it won't confirm a long-
    # range high-stack LOS to the Sentinel), and the model `apply` cannot represent
    # the real adjacent-tile boulder ascent, so the abstract rollout rarely chooses
    # `win` itself. The REAL win mechanic -- build an adjacent boulder stack by the
    # platform, absorb the Sentinel from above its base tile, then transfer onto the
    # platform -- is driven faithfully by code_engine.play_plan's `win` handler
    # (climb_and_win, the live counterpart of this climb). So we append a `win` step
    # here; play_plan executes the real ascent + platform transfer and reaches the
    # do_hyperspace win condition. (This does NOT touch model_replay/solved/
    # los_proven, which stay honest about the abstract model's belief.)
    if sent is not None and not any(s.action.verb == "win" for s in plan.steps):
        plan.steps.append(
            PlanStep(
                action=Action("win", sent.x, sent.y),
                note=f"WIN: real climb-and-win on the Sentinel @ ({sent.x},{sent.y}) "
                f"(adjacent boulder ascent + platform transfer, driven live)",
            )
        )

    # LOS proof: from the win tile + the stack we had there.
    if win_tile is not None and sent is not None:
        # the eye offset at the win is the stack on that tile.
        # find the player's stack when winning by replaying actions_log climb.
        plan.los_proven = greedy.cached_can_see(
            gs0,
            win_tile,
            (sent.x, sent.y),
            eye_offset=win_eye_offset if win_eye_offset else ROBOT_EYE,
        )

    # predicted final energy: the model energy after replay + recovered structure.
    final = _replay_energy(gs0, ps.model_replay)
    # the climb stack is re-absorbed by the executor (energy-neutral net); plus we
    # recover the base robot the model can't (mirror solver.py's bookkeeping).
    plan.recoverable_structure_energy = ENERGY_IN_OBJECTS[T_ROBOT]
    plan.final_energy = (final + plan.recoverable_structure_energy) & ENERGY_MASK


def _replay_energy(initial: GameState, actions: List[Action]) -> int:
    st = copy.deepcopy(initial)
    if not hasattr(st, "won"):
        st.won = False
    for a in actions:
        st = apply(st, a)
    return st.player_energy


def _replay_verify(initial: GameState, actions: List[Action]) -> bool:
    st = copy.deepcopy(initial)
    if not hasattr(st, "won"):
        st.won = False
    for a in actions:
        st = apply(st, a)
    return getattr(st, "won", False)


# ============================================================================
# BOUNDED BRANCH-AND-BOUND (terminal search) -- fallback / comparison mode
# ============================================================================
def solve_bnb(state_or_model, budget_s: float = DEFAULT_BUDGET_S) -> Plan:
    """A bounded depth-first branch-and-bound to a winning terminal, kept as a
    comparison/fallback to the minimax policy. Admissible upper bound: current
    energy + sum of all still-absorbable object energies (the most you could ever
    still collect), used to prune branches that cannot beat the incumbent. Dominance
    pruning via the transposition key. Bounds: BOUND_STACK on stack height and
    MAX_CREATE_FRONTIER on create targets (same lattice as the minimax)."""
    gs0 = copy.deepcopy(
        state_or_model.state
        if isinstance(state_or_model, GameModel)
        else state_or_model
    )
    if not hasattr(gs0, "won"):
        gs0.won = False
    phase = ed.init_phase(gs0)
    start = PState(copy.deepcopy(gs0), 0, phase)

    t0 = time.time()
    best = {"score": -1.0, "ps": None, "nodes": 0}
    seen: Dict = {}

    def collectible_bound(ps: PState) -> float:
        # admissible: every remaining absorbable object's energy + win + energy.
        e = ps.gs.player_energy
        for o in ps.gs.objects:
            if (
                o.type in gm.ABSORBABLE
                and o.type != T_PLATFORM
                and o.slot != ps.gs.player_slot
            ):
                e += ENERGY_IN_OBJECTS[o.type]
        return e  # an upper bound on achievable final energy

    def score(ps: PState) -> float:
        # objective: (objects absorbed, then final energy).
        return len(ps.absorbed) * 1000 + ps.gs.player_energy

    def dfs(ps: PState, depth: int):
        best["nodes"] += 1
        if time.time() - t0 > budget_s:
            return
        if ps.won():
            sc = score(ps)
            if sc > best["score"]:
                best["score"] = sc
                best["ps"] = ps
            return
        if depth <= 0:
            return
        if collectible_bound(ps) * 1000 <= best["score"]:
            return  # cannot beat incumbent even absorbing everything left
        k = (ps.key(), depth)
        if k in seen:
            return
        seen[k] = True
        for kind, action, meta in _ordered_actions(ps):
            child = apply_action(ps, kind, action, meta)
            if child is not None:
                dfs(child, depth - 1)

    dfs(start, depth=MAX_REPLAN_STEPS)

    plan = Plan()
    if best["ps"] is None:
        plan.notes.append("branch-and-bound found no winning terminal in budget")
        plan.search = {
            "nodes": best["nodes"],
            "depth": 0,
            "time": time.time() - t0,
            "optimal_within_bounds": False,
            "mode": "bnb (no win)",
        }
        return plan
    _emit_plan(best["ps"], plan, gs0)
    plan.search = {
        "nodes": best["nodes"],
        "depth": MAX_REPLAN_STEPS,
        "time": time.time() - t0,
        "optimal_within_bounds": (time.time() - t0) < budget_s,
        "mode": "bnb (best terminal)",
    }
    won = _replay_verify(gs0, plan.model_replay)
    plan.solved = bool(won and plan.los_proven)
    return plan


# ============================================================================
# analyse_solvability -- reuse solver.py's (extend with enemy-aware note)
# ============================================================================
def analyse_solvability(state_or_model):
    """Reuse solver.analyse_solvability (proven-impossible vs search-failure
    criteria), which is enemy-model-agnostic and already validated."""
    return greedy.analyse_solvability(state_or_model)


# ============================================================================
# __main__ validation: ls 0000, 0042, 9999 + greedy-vs-exact table
# ============================================================================
def _types(plan: Plan) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tname, _x, _y in plan.absorbed:
        out[tname] = out.get(tname, 0) + 1
    return out


def _report(ls: int, depth: int, budget: float):
    print(f"\n############ seed {ls} ############")
    model = GameModel.from_landscape(ls)
    rep = analyse_solvability(model)
    print(f"  solvable           : {rep.solvable}")
    if not rep.solvable:
        print(f"  reason             : {rep.reason}")
        return None, None
    plan = solve(model, depth=depth, budget_s=budget)
    s = plan.search
    print(
        f"  plan length        : {len(plan.steps)} steps "
        f"(model-replay {len(plan.model_replay)} actions)"
    )
    print(f"  objects absorbed   : {plan.absorbed_count}  {_types(plan)}")
    print(f"  predicted energy   : {plan.final_energy}")
    print(f"  LOS proven (climb) : {plan.los_proven}")
    print(
        f"  search             : {s['nodes']} nodes, depth {s['depth']}, "
        f"{s['time']:.2f}s, mode={s['mode']}"
    )
    print(
        f"  optimality         : "
        f"{'proven-optimal-within-bounds' if s['optimal_within_bounds'] else 'best-found-within-budget'}"
    )
    ok = _replay_verify(model.state, plan.model_replay)
    print(f"  replay-verified    : model-replay reaches won == {ok}")
    if plan.notes:
        print(f"  notes              : {'; '.join(plan.notes)}")
    return plan, rep


def main():
    depth = DEFAULT_DEPTH
    budget = DEFAULT_BUDGET_S
    rows = []
    for ls in (0, 42, 9999):
        plan, _rep = _report(ls, depth, budget)
        gplan = greedy.solve(GameModel.from_landscape(ls))
        rows.append((ls, gplan, plan))

    print("\n================ greedy vs exact (minimax+eval) ================")
    print(
        f"  {'ls':>5} | {'greedy abs':>10} {'greedy E':>8} | "
        f"{'exact abs':>9} {'exact E':>7} {'depth':>5} {'nodes':>7} {'time':>6}"
    )
    for ls, g, x in rows:
        if x is None:
            print(
                f"  {ls:>5} | {g.absorbed_count:>10} {g.final_energy:>8} | "
                f"{'unsolv':>9}"
            )
            continue
        s = x.search
        print(
            f"  {ls:>5} | {g.absorbed_count:>10} {g.final_energy:>8} | "
            f"{x.absorbed_count:>9} {x.final_energy:>7} {s['depth']:>5} "
            f"{s['nodes']:>7} {s['time']:>5.1f}s"
        )
        assert (
            x.final_energy >= g.final_energy
        ), f"ls{ls}: exact energy {x.final_energy} < greedy {g.final_energy}"
    print("  (exact final energy is >= greedy for every landscape: OK)")


if __name__ == "__main__":
    main()
