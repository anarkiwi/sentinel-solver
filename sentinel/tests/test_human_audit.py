"""Regenerate the human-win audit and assert the invariants that hold plus pin the
current model-vs-reality disagreements, so a future threat-model fix that clears
them flips this test.  One comprehensive test per fixture keeps it xdist-safe."""

import pytest

from sentinel.tests import human_audit

# Pinned CURRENT disagreements (regenerable via ``python -m sentinel.tests.human_audit``); a model fix that clears any changes the set -> update here.
_BREACH335 = [
    22,
    27,
    29,
    34,
    38,
    55,
    62,
    68,
    69,
    73,
    75,
    84,
    99,
    101,
    103,
    104,
    106,
    116,
    121,
    133,
    134,
    135,
]
_GATE335 = [
    5,
    15,
    16,
    20,
    21,
    22,
    23,
    27,
    28,
    29,
    34,
    35,
    37,
    38,
    40,
    52,
    53,
    55,
    57,
    59,
    61,
    62,
    68,
    69,
    70,
    73,
    75,
    76,
    78,
    81,
    83,
    84,
    85,
    94,
    99,
    101,
    103,
    104,
    106,
    116,
    121,
    127,
    133,
    134,
    135,
    142,
    143,
]
_FIRE335 = [6, 15, 21, 23, 28, 35, 37, 40, 85, 124, 127]
_DRAIN335 = [
    26,
    33,
    36,
    50,
    54,
    58,
    60,
    64,
    66,
    71,
    74,
    77,
    78,
    80,
    82,
    86,
    100,
    102,
    105,
    107,
    114,
    115,
    117,
    120,
    122,
    128,
]
_TREE335 = [
    23,
    35,
    40,
    52,
    55,
    57,
    59,
    62,
    68,
    69,
    76,
    79,
    81,
    85,
    106,
    119,
    121,
    124,
    127,
]
_ENERGY335 = sorted(set(_DRAIN335) | set(_TREE335))

# ls42: TRUE replayed facings applied -> winning tiles (2,24)/(5,22) flag gate_reject+breach.
EXPECTED_CODES = {
    "ls0.json": {},
    "ls42.json": {
        "account_breach": [13, 14, 17, 23],
        "energy": [15],
        "gate_reject": [13, 14, 17, 23],
    },
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
    """DECISIVE A-vs-B result: with the TRUE replayed enemy facings (ls42_truth.json),
    the model STILL flags the human's own winning tiles as seen/window=0/breach even
    though the human built/transferred there and won -- hypothesis B (over-classify),
    not A (a wrong-phase aim clock)."""
    audit = human_audit.audit_fixture("ls42.json")
    assert audit["enemy_truth_steps"] == 24  # reproduced steps in the committed truth
    by_i = {s["i"]: s for s in audit["steps"]}
    for i, tile in _LS42_WIN_STEPS.items():
        s = by_i[i]
        assert s["target"] == tile
        assert s["enemy_facings_source"] == "replay_truth"
        assert s["verdict"]["gate_allow"] is False  # rejects the human's winning move
        assert s["verdict"][
            "breaches"
        ]  # placement ends in a live cone under true phase
    # steps 14/17 land under a cone on the tile NOW, yet the human survived.
    assert by_i[14]["exposure_target"]["seen_now"]
    assert by_i[17]["exposure_target"]["seen_now"]
    assert by_i[14]["verdict"]["breaches"][0]["seen_by"][0][1] is False  # PARTIAL sight
    assert by_i[17]["verdict"]["breaches"][0]["seen_by"][0][1] is True  # FULL sight


def test_truth_provenance():
    """ls42 (24 steps) and ls335 (25) have committed replay-truth fixtures; ls0 has
    none and falls back to baseline facings for every step."""
    assert human_audit.audit_fixture("ls42.json")["enemy_truth_steps"] == 24
    assert human_audit.audit_fixture("ls335.json")["enemy_truth_steps"] == 25
    assert human_audit.audit_fixture("ls0.json")["enemy_truth_steps"] == 0
    for s in human_audit.audit_fixture("ls0.json")["steps"]:
        assert s["enemy_facings_source"] == "generate_baseline"
