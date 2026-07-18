"""Regenerate the human-win audit and assert the invariants that hold plus pin the
current model-vs-reality disagreements, so a future threat-model fix that clears
them flips this test.  One comprehensive test per fixture keeps it xdist-safe."""

import pytest

from sentinel.tests import human_audit

# Pinned CURRENT disagreements (regenerable via ``python -m sentinel.tests.human_audit``); a model fix that clears any changes the set -> update here.
_BREACH335 = [
    5,
    7,
    11,
    12,
    13,
    14,
    16,
    20,
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
    3,
    4,
    5,
    6,
    7,
    11,
    12,
    13,
    14,
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
