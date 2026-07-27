"""The player's tunable POLICY: every knob that is a CHOICE, not a ROM measurement.

ROM-derived quantities ($0C20's drain rate, DITHER_FRAMES, the $1335 cadence) are
absent -- fitting a measurement launders a modelling error into a constant.  Fields
are env-overridable as ``SENTINEL_<FIELD>``: how a tuner worker gets one policy."""

import dataclasses
import os


def _coerce(kind, raw):
    return float(raw) if kind is float else int(raw)


@dataclasses.dataclass(frozen=True)
class Policy:
    """One candidate policy; the defaults ARE the shipped player."""

    weight: float = 1.4  # A* heuristic weight
    top_targets: int = 4  # enemies a node may branch a pursuit toward
    top_hops: int = 8  # ranked pedestal candidates a pursuit tries per climb
    top_clears: int = 4  # tree-blocked sites a node may branch a clearing on
    top_relocs: int = 4  # existing remote robots a node may transfer into
    pursue_branch: int = 3  # first-hop alternatives per pursuit target
    max_pursue: int = 40  # hop/reclaim steps one pursuit macro may chain
    max_reclaim: int = 8  # reclaims one macro (or strand probe) may chain
    strand_prune: int = 0  # discard landings _climb_continues calls stranded
    target_eye: float = 9.0  # eye the heuristic stops charging climb for
    eye_per_hop: float = 0.9  # eye one hop is assumed to buy
    margin_k: float = 1.0  # sigmas of step-cost headroom a gate holds back
    # the three below are UNSETTLED rate-model options: the tuner decides them
    source_gate: int = 1  # price the SOURCE stance's exposure in _pick_hop
    build_toll: int = 1  # charge a build's drains to g, as absorbs already are
    liquidity_gate: int = 1  # reclaim when a hop is unaffordable AT RATE, not a floor

    @classmethod
    def from_env(cls, env=None):
        """Read a policy from ``env``; absent keys keep the shipped default, so an
        unset environment reproduces the shipped player exactly."""
        env = os.environ if env is None else env
        vals = {}
        for f in dataclasses.fields(cls):
            raw = env.get("SENTINEL_" + f.name.upper())
            if raw is not None:
                vals[f.name] = _coerce(f.type, raw)
        return cls(**vals)

    def as_env(self):
        """The environment a worker process needs to reproduce this policy."""
        return {
            "SENTINEL_" + f.name.upper(): str(getattr(self, f.name))
            for f in dataclasses.fields(self)
        }


POLICY = Policy.from_env()
