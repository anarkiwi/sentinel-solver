#!/usr/bin/env python3
"""Per-step PLAN-vs-REALITY audit for the live A* planner.

Each fired/stale step reports the budget and dwell windows the SEARCH recorded on it
(``astar_player.PlanStep``) beside the LIVE values read from VICE -- so a test can
assert the plan never gates a step drain-safe that is live-hot.
"""

import math
import os

from driver import core, live_player
from sentinel import enemies, memmap as mm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PlanAuditAStar(live_player.LiveAStar):
    """LiveAStar that records, per executed/stale step, the plan's own predicted
    dwell windows against the live board (into ``self.audit``)."""

    audit_pred = True  # this tool is the only consumer of PlanStep.pbody

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.audit = []

    def _snap(self, st, tile, view):
        slots = enemies.enemy_slots(st)
        return {
            "faces": [int(st.obj_h_angle[e]) for e in slots],
            "rcd": [int(st.mem[mm.ENEMIES_ROTATION_COOLDOWN + e]) for e in slots],
            "win": self._gaze_window(tile) if view is not None else math.inf,
            "pbody": self._player_window(),
        }

    def _plan_head(self, verb, tile):
        """The plan's pending ``PlanStep`` when it IS this action (a defensive
        ``_fire`` from ``_react``/``_defend`` is not a plan step and has no premise
        to audit)."""
        if not self.plan or self._pi >= len(self.plan):
            return None
        step = self.plan[self._pi]
        if step.verb != verb or tuple(step.tile) != tuple(tile):
            return None
        return step

    def _record(self, tag, step, view):
        live = self._snap(self.st, step.tile, view)
        self.audit.append(
            {
                "tag": tag,
                "verb": step.verb,
                "tile": tuple(step.tile),
                "budget": step.budget,
                "gate": step.gate,
                "pred_win": step.window,
                "pred_pbody": step.pbody,
                **{f"live_{k}": v for k, v in live.items()},
            }
        )

    def _plan_step_stale(self, step, view):
        stale = super()._plan_step_stale(step, view)
        if stale:
            self._record("STALE", step, view)
        return stale

    def _fire(self, verb, tile, view):
        step = self._plan_head(verb, tile)
        if step is not None:
            self._record("FIRE", step, view)
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
        lp = PlanAuditAStar(session, log, result, node_budget=200000, weight=1.4)
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
