"""Accuracy of the SETTLE timing constants against measured ground truth.

``frozen_ls42_audit.json`` is a both-frozen run (enemies RTS-stubbed live AND
no-oped in the sim), so its charged-vs-measured residuals carry NO enemy-phase
feedback: pure cost-model error, prediction vs measurement, never self-pinned.
"""

import json
import os
import statistics as st

import pytest

from sentinel import actioncost, projector

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "frozen_ls42_audit.json")

# Fixture columns (self-described by the file's *_fields keys).
LABEL, VERB, _TILE, AIM_CHARGED, AIM_MEASURED, SETTLE_CHARGED, SETTLE_MEASURED = range(
    7
)

# Bias tolerance in frames: ~5% of the ~94 f create/absorb settle, inside its ~6.6 f scatter.
BIAS_TOL_F = 5.0
# Welch t for "means separated": n=13/15, sd~6.6, so t>3 is p<0.01 two-sided.
SEPARATION_T = 3.0


def _rows():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["exact_audit"]


def _measured(verb):
    return [r[SETTLE_MEASURED] for r in _rows() if r[VERB] == verb]


def _welch_t(a, b):
    return (st.mean(a) - st.mean(b)) / (
        st.variance(a) / len(a) + st.variance(b) / len(b)
    ) ** 0.5


def test_fixture_shape():
    rows = _rows()
    assert len(rows) == 35
    assert {r[VERB] for r in rows} == {"absorb", "create", "transfer"}


def test_create_settle_prediction_is_accurate():
    """actioncost.SETTLE["create"] predicts the measured create settle to <5 f."""
    measured = _measured("create")
    bias = st.mean(measured) - actioncost.SETTLE["create"]
    assert len(measured) == 13
    assert abs(bias) < BIAS_TOL_F, f"create settle bias {bias:.2f} f"
    assert min(measured) <= actioncost.SETTLE["create"] <= max(measured)


def test_absorb_and_create_measured_settles_are_separated():
    """Measured settles differ by verb: create 96.00 f (sd 6.59, n=13) vs absorb
    85.87 f (sd 6.51, n=15)."""
    t = _welch_t(_measured("create"), _measured("absorb"))
    assert t > SEPARATION_T, f"Welch t {t:.2f}"


@pytest.mark.xfail(
    strict=True,
    reason="actioncost.SETTLE uses ONE shared dither+replot value (93.75 f) for "
    "absorb and create, but the measured settles differ: create mean 96.00 f "
    "(sd 6.59, n=13) vs absorb mean 85.87 f (sd 6.51, n=15). The shared constant "
    "fits create (bias +2.25 f) and is biased HIGH on absorb (-7.88 f). Splitting "
    "SETTLE per verb makes this pass.",
)
def test_shared_settle_fits_both_verbs():
    assert actioncost.SETTLE["absorb"] == actioncost.SETTLE["create"]
    bias = {
        v: st.mean(_measured(v)) - actioncost.SETTLE[v] for v in ("create", "absorb")
    }
    worst = max(bias, key=lambda v: abs(bias[v]))
    assert abs(bias[worst]) < BIAS_TOL_F, f"{worst} settle bias {bias[worst]:.2f} f"


def test_transfer_charged_settles_clear_the_projector_fixed_floor():
    """Every charged transfer settle exceeds the view-less fallback (tune + fixed
    foreground), i.e. the per-scene replot term is non-negative."""
    floor = projector.TUNE_TRANSFER_FRAMES + projector.SETTLE_FIXED_FRAMES
    assert actioncost.VIEWPOINT_REPLOT_FRAMES == floor
    charged = [r[SETTLE_CHARGED] for r in _rows() if r[VERB] == "transfer"]
    assert len(charged) == 7
    assert min(charged) > floor
    assert projector.REPLOT_PASSES >= 1


@pytest.mark.xfail(
    strict=True,
    reason="The per-scene transfer settle model (viewpoint_replot_frames = "
    "TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES + REPLOT_PASSES*render_cost) is "
    "biased LOW by 27.30 f with rms residual 46.8 f against measured mean "
    "376.14 f (sd 65.14, range 270..474, n=7). Open item: docs/plan_fidelity.md "
    "open problem 1 (render_cost-class scene model).",
)
def test_transfer_settle_model_is_unbiased_and_within_scatter():
    rows = [r for r in _rows() if r[VERB] == "transfer"]
    err = [r[SETTLE_MEASURED] - r[SETTLE_CHARGED] for r in rows]
    rms = (sum(e * e for e in err) / len(err)) ** 0.5
    measured_sd = st.pstdev([r[SETTLE_MEASURED] for r in rows])
    assert abs(st.mean(err)) < BIAS_TOL_F * 2, f"transfer bias {st.mean(err):.2f} f"
    # A useful scene model must beat half the raw spread it explains.
    assert rms < measured_sd / 2, f"rms {rms:.1f} f vs measured sd {measured_sd:.1f} f"


def test_transfer_aim_is_zero_measured_and_charged():
    """Transfer reuses the previous aim: aim is exactly 0 measured (and charged) on
    every transfer row."""
    rows = [r for r in _rows() if r[VERB] == "transfer"]
    assert len(rows) == 7
    assert all(r[AIM_MEASURED] == 0 for r in rows)
    assert all(r[AIM_CHARGED] == 0 for r in rows)


@pytest.mark.parametrize("verb", ["create", "absorb"])
def test_dither_dominates_the_charged_settle(verb):
    """Charged settle is dither + one post-action replot, dither-dominated, and is
    what the fixture charged."""
    assert actioncost.DITHER_FRAMES > actioncost.POST_ACTION_REPLOT_FRAMES
    charged = {r[SETTLE_CHARGED] for r in _rows() if r[VERB] == verb}
    assert len(charged) == 1
    assert charged.pop() == pytest.approx(actioncost.SETTLE[verb])
