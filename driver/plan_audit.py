#!/usr/bin/env python3
"""Per-step PLAN-vs-REALITY audit for the live A* planner.

Replays the plan's forward model at each fired/stale step to recover the enemy phase
and dwell windows the plan believed, beside the LIVE values read from VICE -- so a
test can assert the plan never gates a step drain-safe that is live-hot.
"""

import math
import os

from driver import core, live_player
from sentinel import actions, enemies, memmap as mm
from sentinel.playerbase import SIGHTS_CENTRE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PlanAuditAStar(live_player.LiveAStar):
    """LiveAStar that records, per executed/stale step, the plan's predicted enemy
    phase + dwell windows against the live board (into ``self.audit``)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._audit_plan = object()
        self._audit_pred = []
        self.audit = []

    def _apply(self, st, verb, tile):
        if verb in ("boulder", "robot", "create"):
            actions.create(st, mm.T_BOULDER if verb == "boulder" else mm.T_ROBOT, tile)
        elif verb in ("absorb", "transfer"):
            top = self._top(tile)
            if top is not None:
                (actions.absorb if verb == "absorb" else actions.transfer)(st, top)

    def _snap(self, st, tile, view):
        slots = enemies.enemy_slots(st)
        return {
            "faces": [int(st.obj_h_angle[e]) for e in slots],
            "rcd": [int(st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e]) for e in slots],
            "win": self._gaze_window(tile) if view is not None else math.inf,
            "pbody": self._player_window(),
        }

    def _predict_plan(self, start_st, plan):
        """Replay the plan through the sim forward model (charge advances enemies by
        each step's aim+settle, then the action lands); return the per-step phase the
        plan believed the player would face."""
        saved = (self.st, list(self.cursor), self.last_bearing)
        st = start_st.clone()
        self.st, self.cursor, self.last_bearing = st, list(SIGHTS_CENTRE), None
        preds = []
        for verb, tile in plan:
            if verb == "hyperspace":
                preds.append(None)
                continue
            view = self._view_for(tile)
            snap = self._snap(st, tile, view)
            snap["budget"] = (
                self._step_aim_frames(verb, view)
                + self._settle(verb, view, self._settle_eye(verb, tile))
                if view is not None
                else math.inf
            )
            preds.append(snap)
            self._charge(st, verb, tile)
            self._apply(st, verb, tile)
        self.st, self.cursor, self.last_bearing = saved
        return preds

    def _record(self, tag, verb, tile, view):
        if self.plan is not None and self._audit_plan is not self.plan:
            self._audit_plan = self.plan
            self._audit_pred = self._predict_plan(self.st, self.plan)
        pred = self._audit_pred[self._pi] if self._pi < len(self._audit_pred) else None
        if pred is None:
            return
        live = self._snap(self.st, tile, view)
        self.audit.append(
            {
                "tag": tag,
                "verb": verb,
                "tile": tuple(tile),
                "budget": pred["budget"],
                **{f"pred_{k}": v for k, v in pred.items() if k != "budget"},
                **{f"live_{k}": v for k, v in live.items()},
            }
        )

    def _plan_step_stale(self, verb, tile, view):
        stale = super()._plan_step_stale(verb, tile, view)
        if stale:
            self._record("STALE", verb, tile, view)
        return stale

    def _fire(self, verb, tile, view):
        self._record("FIRE", verb, tile, view)
        return super()._fire(verb, tile, view)


def run_audit(typed_digits, max_actions=120, log=print):
    """Boot VICE, run the plan-audit A* on ``typed_digits`` (e.g. ``\"0042\"``), and
    return the per-step plan-vs-live records (list of dicts)."""
    os.environ["NO_RECORD"] = "1"
    result = {
        "landscape": typed_digits,
        "player": "astar",
        "actions": [],
        "energy_curve": [],
    }
    holder = {}

    def play_fn(session):
        lp = PlanAuditAStar(
            session, log, result, node_budget=200000, time_budget=30.0, weight=1.4
        )
        lp.run(max_actions=max_actions)
        holder["audit"] = lp.audit

    core.boot_and_play(
        os.path.join(ROOT, "sentinel-gold.tap"),
        os.path.join(ROOT, "renders"),
        typed_digits,
        "player_audit.avi",
        log,
        play_fn,
        result,
    )
    return holder.get("audit", [])
