"""Enforce that every timing constant carries a declared provenance.

Structural ban on unvalidated timing constants: discovery must equal the registry,
claimed evidence must be a real test, and the unvalidated debt set is pinned.
"""

import re

from sentinel.tests import timing_registry as tr

FIX = (
    "classify it in sentinel/tests/timing_registry.py and add evidence "
    "(a test in test_timing_derivations.py or test_settle_accuracy.py), "
    "or it cannot be merged"
)

DISCOVERED = tr.discover()


def _test_source():
    parts = []
    for pattern in tr.TEST_SOURCE_GLOBS:
        for path in sorted(tr.ROOT.glob(pattern)):
            parts.append(path.read_text())
    return "\n".join(parts)


def test_every_timing_constant_is_registered():
    unregistered = sorted(set(DISCOVERED) - set(tr.REGISTRY))
    assert not unregistered, (
        f"unregistered timing constants: "
        f"{', '.join(f'{n} ({DISCOVERED[n]['module']})' for n in unregistered)}. "
        f"For each: {FIX}."
    )


def test_no_stale_registry_entries():
    stale = sorted(set(tr.REGISTRY) - set(DISCOVERED))
    assert not stale, (
        f"registry names no longer present in source: {', '.join(stale)}. "
        "Remove the entry, or restore the constant."
    )


def test_registry_entries_are_well_formed():
    for name, meta in tr.REGISTRY.items():
        assert meta["class"] in tr.CLASSES, f"{name}: bad class {meta['class']!r}"
        assert meta["note"], f"{name}: note is required"
        assert meta["module"] == DISCOVERED[name]["module"], (
            f"{name}: registry says {meta['module']}, "
            f"source says {DISCOVERED[name]['module']}"
        )
        if meta["class"] == tr.UNVALIDATED:
            assert meta["evidence"] is None, f"{name}: UNVALIDATED cannot cite evidence"
        else:
            assert meta["evidence"], f"{name}: {meta['class']} requires evidence"


def test_validated_constants_name_a_real_test():
    source = _test_source()
    missing = sorted(
        f"{name} -> {meta['evidence']}"
        for name, meta in tr.REGISTRY.items()
        if meta["class"] in (tr.DERIVED, tr.MEASURED)
        and not re.search(rf"^\s*def {re.escape(meta['evidence'])}\b", source, re.M)
    )
    assert not missing, (
        f"constants claim validation by a test that does not exist: "
        f"{'; '.join(missing)}. Write the test, or reclassify as UNVALIDATED."
    )


def test_unvalidated_debt_does_not_grow():
    current = frozenset(
        n for n, m in tr.REGISTRY.items() if m["class"] == tr.UNVALIDATED
    )
    added = sorted(current - tr.UNVALIDATED_PIN)
    removed = sorted(tr.UNVALIDATED_PIN - current)
    assert not added, (
        f"new unvalidated timing constants: {', '.join(added)}. "
        f"Debt may not grow: {FIX}."
    )
    assert not removed, (
        f"constants left the unvalidated set: {', '.join(removed)}. "
        "Validating one is good -- update UNVALIDATED_PIN in timing_registry.py "
        "so the smaller debt is what is pinned from now on."
    )


def _claims_validation(comment):
    lowered = comment.lower()
    return any(word in lowered for word in tr.PROVENANCE_CLAIM_WORDS)


def test_provenance_comments_are_truthful():
    claimants = {n for n, m in DISCOVERED.items() if _claims_validation(m["comment"])}
    lying = {
        n
        for n in claimants
        if tr.REGISTRY[n]["class"] == tr.UNVALIDATED
        and n not in tr.KNOWN_FALSE_PROVENANCE_COMMENTS
    }
    assert not lying, (
        f"source comment advertises measurement/validation but the registry has no "
        f"evidence: {', '.join(sorted(lying))}. Add the evidence and reclassify, or "
        "correct the comment."
    )
    # Pinned so the known-false set cannot silently grow or go stale.
    stale_pin = sorted(tr.KNOWN_FALSE_PROVENANCE_COMMENTS - claimants)
    assert not stale_pin, (
        f"pinned false-comment constants no longer claim validation: "
        f"{', '.join(stale_pin)}. Remove them from "
        "KNOWN_FALSE_PROVENANCE_COMMENTS."
    )
    for name in tr.KNOWN_FALSE_PROVENANCE_COMMENTS:
        note = tr.REGISTRY[name]["note"].lower()
        assert "false" in note, f"{name}: note must record that the comment is false"


def test_comment_attribution_detects_both_directions():
    """Discovery must attribute each comment to its own constant: FRAME_TICKS and
    TAP_FRAMES claim nothing and must not read as claimants, while _RU_PAN's block
    still carries the word."""
    for name, module in (
        ("FRAME_TICKS", "sentinel.actioncost"),
        ("TAP_FRAMES", "sentinel.playerbase"),
    ):
        assert DISCOVERED[name]["module"] == module
        assert not _claims_validation(DISCOVERED[name]["comment"])
        assert tr.REGISTRY[name]["class"] == tr.UNVALIDATED
    assert _claims_validation(DISCOVERED["_RU_PAN"]["comment"])
