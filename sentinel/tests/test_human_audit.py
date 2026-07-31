"""Regenerate the human-win audit and assert the invariants that hold plus pin the
current model-vs-reality disagreements, so a future threat-model fix that clears
them flips this test.  One comprehensive test per fixture keeps it xdist-safe."""

import pytest

from sentinel.tests import human_audit

# Pinned CURRENT ls335 disagreements: recording play_20260725_105258, a live WIN whose enemy clock is recorded into the fixture, so the audit runs on TRUE facings (no live _truth.json). Regenerable via ``python -m sentinel.tests.human_audit``; a model fix that clears any -> update here.
# 94 joined the breaches when the RATE reserve stopped refusing its create: a step
# the floor never let happen cannot leave a body in a cone.  On a WIN every placement
# survived, so a breach here is a model false positive, same as the rejects.
_BREACH335 = [23, 24, 28, 44, 45, 56, 95, 137]  # 52 retired by the phase-split advance
# 94 became 95 when $1F9F's on-screen strip replot ($1FFC) started stalling the clock.
# 56 joined with the derived pass cadence: the enemy clock over that span now runs at
# the board's real rate, so the model puts the human inside a cone it had missed.
# 18 joined when UTURN_FRAMES went 74 -> 77 and left again when the examine moved to the play machine's own $37F2: its budget is aim + HOP_FRAMES, and either way it is the drain window's edge that decides, not the step.
_GATE335 = [17, 23, 24, 28, 42, 44, 50, 56, 74, 76, 77, 92, 94, 95, 97, 108, 137]
# was [30, 32, 34, 50, 51, 77, 84, 92, 94, 95] against the flat survival floor; pricing
# exposure as the $0C20 RATE accepts seven of those human creates (docs/architecture.md)
_FIRE335 = [32, 34, 77]
_DRAIN335 = [
    19,
    22,
    25,
    27,
    29,
    33,
    41,
    46,
    49,
    53,
    58,
    67,
    69,
    73,
    75,
    79,
    83,
    86,
    93,
    103,
    107,
    110,
    111,
    134,
    136,
    138,
    141,
    145,
    147,
]  # noqa: E501
_TREE335 = [
    28,
    32,
    34,
    42,
    50,
    55,
    71,
    74,
    77,
    92,
    94,
    95,
    97,
    106,
    135,
    137,
    139,
    143,
    146,
]  # noqa: E501
_ENERGY335 = sorted(set(_DRAIN335) | set(_TREE335))

# ls42: with TRUE replayed facings the corrected drain model AGREES on the winning tiles (2,24)/(5,22); only the real drain at step 15 remains.
EXPECTED_CODES = {
    "ls0.json": {},
    "ls42.json": {"energy": [15]},
    "ls335.json": {
        "account_breach": _BREACH335,
        "energy": _ENERGY335,
        "fire_fail": _FIRE335,
        "gate_reject": _GATE335,
    },
}
EXPECTED_ENERGY = {
    "ls0.json": {},
    "ls42.json": {"drain": [15]},
    "ls335.json": {"drain": _DRAIN335, "tree_spawn": _TREE335},
}


@pytest.mark.parametrize("name", human_audit.FIXTURES)
def test_fixture_audit(name):
    audit = human_audit.audit_fixture(name)
    steps = audit["steps"]
    summ = audit["summary"]
    assert len(steps) == audit["n_events"] == summ["n_steps"]

    # Action-cost energy is EXACT for every genuine build: divergences are only enemy drains and mis-kept enemy tree spawns.
    assert not [s["i"] for s in steps if s["energy"]["note"] == "cost_mismatch"]

    # A keyboard aim exists for EVERY distinct human target; view-less steps are only same-tile transfers.
    noview = {s["i"] for s in steps if not s["aim"]["has_view"]}
    assert noview == set(summ["own_tile_transfers"])

    # Pinned disagreements: energy drains/tree spawns + gate/fire/breach false-positives from baseline enemy facings.
    got_energy = {
        k: v for k, v in summ["energy_notes"].items() if k in ("drain", "tree_spawn")
    }
    assert got_energy == EXPECTED_ENERGY[name]
    assert summ["disagreement_steps_by_code"] == EXPECTED_CODES[name]


def test_ls0_is_clean_baseline():
    """The trivial board is a full model-vs-human agreement across every dimension."""
    summ = human_audit.audit_fixture("ls0.json")["summary"]
    assert summ["energy_model_agree"] == summ["n_steps"] == 25
    assert summ["landable_view_agree"] == 25
    assert summ["n_steps_with_disagreement"] == 0


# Human's own winning steps at (2,24)/(5,22): create-robot, transfer, create-boulder.
_LS42_WIN_STEPS = {13: [2, 24], 14: [2, 24], 17: [5, 22]}


def test_ls42_truth_over_classifies():
    """With the TRUE replayed enemy facings (ls42_truth.json) the corrected drain
    model now AGREES with the human on every winning tile (2,24)/(5,22): no
    gate_reject, no breach.  Only FULL sight drains ($1838); these stand under
    PARTIAL sight (a boulder, or a robot the enemy half-sees) and never drained."""
    audit = human_audit.audit_fixture("ls42.json")
    assert audit["enemy_truth_steps"] == 24  # reproduced steps in the committed truth
    by_i = {s["i"]: s for s in audit["steps"]}
    for i, tile in _LS42_WIN_STEPS.items():
        s = by_i[i]
        assert s["target"] == tile
        assert s["enemy_facings_source"] == "replay_truth"
        assert s["verdict"]["gate_allow"] is True  # no longer rejects the winning move
        assert not s["verdict"]["breaches"]  # partial/boulder sight never drains
    # steps 14/17 still sit under a cone now, yet partial sight (14) / a boulder (17) are undrainable, so the human survived and the model no longer flags them.
    assert by_i[14]["exposure_target"]["seen_now"]
    assert by_i[17]["exposure_target"]["seen_now"]
    assert by_i[14]["exposure_target"]["n_full"] == 1  # only a partial cone reaches it
    assert by_i[17]["otype_name"] == "BOULDER"  # an undrainable body


def test_truth_provenance():
    """ls42 sources true facings from its live replay (24 steps); ls335 from the enemy
    clock recorded straight into the fixture (154 steps); ls0 has neither and falls
    back to baseline facings for every step."""
    assert human_audit.audit_fixture("ls42.json")["enemy_truth_steps"] == 24
    assert human_audit.audit_fixture("ls335.json")["enemy_truth_steps"] == 154
    assert human_audit.audit_fixture("ls0.json")["enemy_truth_steps"] == 0
    for s in human_audit.audit_fixture("ls0.json")["steps"]:
        assert s["enemy_facings_source"] == "generate_baseline"
