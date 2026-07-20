"""Guard: the live-play path must be 100% event driven -- no host wall-clock waits.

A ``time.sleep`` there is warp-dependent by construction (the same delay spans several
times as many emulated frames with warp on), so measured frame counts silently change
with recording on/off. Waits belong on a PC or memory predicate (``driver.clock``).
"""

import ast
import io
import os
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))

LIVE_MODULES = [
    "driver/live_player.py",
    "driver/sentinel_execute.py",
    "driver/kbd_aim.py",
    "driver/core.py",
    "driver/clock.py",
    "driver/play_player.py",
    "driver/plan_audit.py",
    "driver/boot.py",
    "driver/instrument.py",
]

MARKER = "sleep-ok:"

# Every wall-clock wait allowed to remain, as (module, reason); matched EXACTLY, so a new one is a deliberate, reviewable act rather than a one-word bypass.
PINNED = {
    ("driver/boot.py", "tape-loader poll interval, no game code resident"),
    ("driver/boot.py", "docker rm teardown, outside the machine"),
    ("driver/boot.py", "docker bridge IP assignment, no PC exists"),
    ("driver/boot.py", "container relaunch backoff, no machine to poll"),
    ("driver/core.py", "docker rm teardown, outside the emulated machine"),
    ("driver/core.py", "docker bridge IP assignment, no PC exists"),
    ("driver/core.py", "VICE AVI encoder start, not the CPU"),
    ("driver/core.py", "VICE AVI muxer drain, not the CPU"),
    ("driver/core.py", "VICE AVI index flush, not the CPU"),
    ("driver/core.py", "container relaunch backoff, no machine to poll"),
}


def _root(attr):
    node = attr.value
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _sleep_names(tree):
    """(aliases of the ``time`` module, bare names bound to ``time.sleep``)."""
    mods, bare = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "time" or a.name.startswith("time."):
                    mods.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module == "time":
            for a in node.names:
                if a.name == "sleep":
                    bare.add(a.asname or a.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "sleep" and _root(node.value) in mods:
                bare.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return mods, bare


def _sleep_calls(tree):
    """Line spans of every call that resolves to ``time.sleep``."""
    mods, bare = _sleep_names(tree)
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Name) and f.id in bare) or (
            isinstance(f, ast.Attribute) and f.attr == "sleep" and _root(f) in mods
        ):
            spans.append((node.lineno, node.end_lineno or node.lineno))
    return spans


def _marked_reasons(src):
    """{lineno: reason} per ``# sleep-ok: <reason>`` COMMENT token, so a string literal
    holding the marker can never satisfy it."""
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and MARKER in tok.string:
            out[tok.start[0]] = tok.string.split(MARKER, 1)[1].strip()
    return out


def test_live_play_path_has_no_wall_clock_sleeps():
    root = os.path.dirname(HERE)
    unmarked, found = [], set()
    for rel in LIVE_MODULES:
        with open(os.path.join(root, rel)) as fh:
            src = fh.read()
        reasons = _marked_reasons(src)
        for lo, hi in _sleep_calls(ast.parse(src)):
            hits = [reasons[n] for n in range(lo, hi + 1) if reasons.get(n)]
            if hits:
                found.add((rel, hits[0]))
            else:
                unmarked.append(f"{rel}:{lo}")
    assert not unmarked, (
        "wall-clock sleep(s) in the live-play path -- drive the wait off a PC or a "
        "memory predicate (driver.clock), or, for a genuinely out-of-machine wait, "
        "mark it '# sleep-ok: <reason>' and pin it in driver/test_no_sleep.py "
        "PINNED: " + ", ".join(unmarked)
    )
    assert found == PINNED, (
        f"sleep-ok allowlist drifted; added={sorted(found - PINNED)} "
        f"removed={sorted(PINNED - found)}"
    )
