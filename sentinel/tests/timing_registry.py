"""Provenance registry for timing constants in ``sentinel/`` and ``driver/``.

DERIVED = recomputable from a ROM primitive (evidence: test_timing_derivations.py);
MEASURED = checked against a committed fixture (evidence: test_settle_accuracy.py);
UNVALIDATED = debt, pinned. Discovery parses source with ``ast``; never imports it.
"""

import ast
import pathlib

DERIVED = "DERIVED"
MEASURED = "MEASURED"
UNVALIDATED = "UNVALIDATED"
CLASSES = (DERIVED, MEASURED, UNVALIDATED)

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("sentinel", "driver")
EXCLUDE_DIRS = (ROOT / "sentinel" / "tests",)
TEST_SOURCE_GLOBS = ("sentinel/tests/test_*.py", "driver/test_*.py")

# Constant NAME substrings marking a timing/frame-cost quantity.
NAME_PATTERNS = (
    "_FRAMES",
    "_CYCLES",
    "_TICKS",
    "_SECONDS",
    "SIGMA",
    "TIMEOUT",
    "_RU_",
    "_DELAY",
    "_PERIOD",
    "_SCROLL",
    "SETTLE",
    "COOLDOWN",
    "_RATE",
    "_MS",
    "_HZ",
    "REDRAW",
    "STEPS_PER_",
    "DITHER",
    "TUNE",
    "_WAIT",
    "_STALL",
    "_SPAWN",
    "_ARM",
    "_RAMP",
    "_MASK",
)

# Keyword arguments whose numeric default is a timing/budget knob.
KWARG_NAMES = frozenset(
    {
        "timeout",
        "hold",
        "settle",
        "period",
        "delay",
        "max_steps",
        "chunk",
        "passes",
        "attempts",
    }
)

# Comment words asserting that evidence exists.
PROVENANCE_CLAIM_WORDS = ("measured", "validated")


def _plain_number(node):
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _numeric(node):
    """True if ``node`` evaluates to a number or a container of numbers."""
    if isinstance(node, ast.Constant):
        if _plain_number(node):
            return True
        if not isinstance(node.value, str):
            return False
        try:
            float(node.value)
        except ValueError:
            return False
        return True
    if isinstance(node, (ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.UnaryOp):
        return _numeric(node.operand)
    if isinstance(node, ast.BinOp):
        return _numeric(node.left) and _numeric(node.right)
    if isinstance(node, ast.Dict):
        return bool(node.values) and all(_numeric(v) for v in node.values)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return bool(node.elts) and all(_numeric(e) for e in node.elts)
    if isinstance(node, ast.Call):
        # float()/int()/os.environ.get(NAME, "<number>") wrappers.
        return any(_numeric(a) for a in node.args)
    return False


def _source_files():
    for name in SOURCE_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            if path.name.startswith("test_"):
                continue
            if any(d in path.parents for d in EXCLUDE_DIRS):
                continue
            yield path


def _dotted(path):
    return str(path.relative_to(ROOT)).replace("/", ".")[: -len(".py")]


def _comment_index(source):
    """Map line number to (comment text, is_own_line)."""
    out = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        hint = line.find("#")
        if hint < 0:
            continue
        head = line[:hint]
        if head.count('"') % 2 or head.count("'") % 2:
            continue
        out[lineno] = (line[hint + 1 :].strip(), not head.strip())
    return out


def _provenance_text(comments, lineno):
    """Trailing comment plus the contiguous own-line comment block above it."""
    block = []
    above = lineno - 1
    while above in comments and comments[above][1]:
        block.append(comments[above][0])
        above -= 1
    parts = list(reversed(block))
    if lineno in comments:
        parts.append(comments[lineno][0])
    return " ".join(parts)


def _kwarg_defaults(func):
    args = func.args
    positional = args.posonlyargs + args.args
    paired = list(
        zip(positional[len(positional) - len(args.defaults) :], args.defaults)
    )
    return paired + list(zip(args.kwonlyargs, args.kw_defaults))


def discover():
    """Scan shipped source; return name -> {"module", "lineno", "comment"}."""
    found = {}

    def record(name, module, lineno, comment):
        if name in found:
            raise AssertionError(
                f"duplicate timing constant {name!r} in {found[name]['module']} "
                f"and {module}; registry keys must be unique"
            )
        found[name] = {"module": module, "lineno": lineno, "comment": comment}

    for path in _source_files():
        source = path.read_text()
        tree = ast.parse(source)
        module = _dotted(path)
        comments = _comment_index(source)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if node.value is None or not _numeric(node.value):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and any(
                    p in target.id for p in NAME_PATTERNS
                ):
                    record(
                        target.id,
                        module,
                        node.lineno,
                        _provenance_text(comments, node.lineno),
                    )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg, default in _kwarg_defaults(node):
                if (
                    arg.arg in KWARG_NAMES
                    and default is not None
                    and _plain_number(default)
                ):
                    record(
                        f"{node.name}.{arg.arg}",
                        module,
                        node.lineno,
                        _provenance_text(comments, node.lineno),
                    )
    return found


def entry(module, provenance, note, evidence=None):
    """Build a registry value."""
    return {"module": module, "class": provenance, "evidence": evidence, "note": note}


def _u(module, note):
    return entry(module, UNVALIDATED, note)


_PRIMITIVE = "test_derived_constant_matches_primitive"
_SETTLE_FIT = "test_create_settle_prediction_is_accurate"
_PAN_FIT = "test_pan_notch_cost_matches_the_measured_plot"


def _d(module, note):
    return entry(module, DERIVED, note, _PRIMITIVE)


_AC = "sentinel.actioncost"
_PB = "sentinel.playerbase"
_PR = "sentinel.projector"
_PN = "sentinel.pancost"
_EN = "sentinel.enemies"
_MM = "sentinel.memmap"
_LOS = "sentinel.los"
_KBD = "driver.kbd_aim"
_CORE = "driver.core"
_ROW = "ray-march/sweep iteration cap; unmeasured"
_RELOAD = "ROM cooldown reload value; no derivation test"
_GUARD = "wall-clock guard; unmeasured"

REGISTRY = {
    "FRAME_TICKS": _u(_AC, "unit scalar; no test pins it to a ROM primitive"),
    "DITHER_FRAMES": _d(_AC, "977904 dither cycles / projector.FRAME_CYCLES"),
    "VIEWPOINT_REPLOT_FRAMES": _d(_AC, "TUNE_TRANSFER_FRAMES + SETTLE_FIXED_FRAMES"),
    "POST_ACTION_REPLOT_FRAMES": entry(
        _AC,
        MEASURED,
        "validated only inside the create settle sum vs frozen_ls42_audit.json (<5 f)",
        _SETTLE_FIT,
    ),
    "SETTLE": entry(
        _AC,
        MEASURED,
        "create settle within 5 f of frozen_ls42_audit.json; absorb bias is xfailed",
        _SETTLE_FIT,
    ),
    "_CLEAR_CYCLES_H": _d(_PN, "$3912 store-loop cycle count"),
    "_CLEAR_CYCLES_V": _d(_PN, "$38AD store-loop cycle count"),
    "CLEAR_FRAMES": entry(
        _PN,
        MEASURED,
        "within 1 f of the py65-measured clear subtree in golden_pan_cost.json",
        _PAN_FIT,
    ),
    "H_SCROLL": _u(_PB, "pan scroll step; no derivation test"),
    "V_SCROLL": _u(_PB, "pan scroll step; no derivation test"),
    "TOGGLE_FRAMES": entry(
        _PB,
        MEASURED,
        "inside the live_aim_subframes.json measured toggle range; envelope is loose "
        "(min <= 12 <= max), not an error bound",
        "test_charged_toggle_matches_the_measured_pair",
    ),
    "ROT_PERIOD_FRAMES": _d(_PB, "ROTATION_COOLDOWN_RELOAD x UNIT_FRAMES ($1813)"),
    "MEANIE_SPAWN_FRAMES": _d(_PB, "UPDATE_COOLDOWN_MEANIE_MADE x UNIT_FRAMES ($1869)"),
    "TAP_FRAMES": _u(_PB, "key tap hold; no derivation test"),
    "UTURN_FRAMES": entry(
        _PB,
        MEASURED,
        "live ls42 p1 u-turn, n=1 (live_ls42_hops.json); a single sample, and "
        "not yet derived from the tap_action scan/settle structure",
        "test_uturn_is_charged_as_an_action_tap_not_a_keystroke",
    ),
    "UNIT_FRAMES": _d(_PB, "3 x 256 / COOLDOWN_BRESENHAM_STEP gate+Bresenham divider"),
    "CURSOR_RAMP": _d(_PB, "popcount of the $11E0 CURSOR_REPEAT_MASK"),
    "CURSOR_REPEAT_MASK": _u(_PB, "cursor repeat mask; no derivation test"),
    "HOP_FRAMES": entry(
        _PB,
        MEASURED,
        "under both live ls42 hops (745, 879 f) and within 25%; its claimed SETTLE "
        "composition still does not add up (separately xfailed)",
        "test_hop_frames_brackets_the_measured_hops",
    ),
    "SAFE_FRAMES": _u(_PB, "post-action safety margin; unmeasured"),
    "WAIT_FRAMES": _u(_PB, "idle wait quantum; unmeasured"),
    "DRAIN_DELAY": _d(_PB, "DRAINING_COOLDOWN_RELOAD x UNIT_FRAMES ($1835)"),
    "MEANIE_ARM_FRAMES": _d(_PB, "$171B half-turn x $173A rounds x UNIT_FRAMES"),
    "FRAME_CYCLES": _d(_PR, "PAL frame cycle count 19656"),
    "BASE_CYCLES": _u(_PR, "plot_world base cycles; no fixture"),
    "SETTLE_FIXED_FRAMES": _u(_PR, "fixed settle base; no fixture"),
    "TUNE_TRANSFER_FRAMES": _u(_PR, "transfer tune wait; unmeasured"),
    "UPDATE_COOLDOWN_SCAN": _u(_EN, _RELOAD),
    "UPDATE_COOLDOWN_DRAIN": _u(_EN, _RELOAD),
    "UPDATE_COOLDOWN_MEANIE_ROTATE": _u(_EN, _RELOAD),
    "UPDATE_COOLDOWN_MEANIE_MADE": _u(_EN, _RELOAD),
    "ROTATION_COOLDOWN_RELOAD": _u(_EN, _RELOAD),
    "DRAINING_COOLDOWN_RELOAD": _u(_EN, _RELOAD),
    "COOLDOWN_STICK": _u(_EN, "cooldown stick threshold; no derivation test"),
    "ENEMIES_DRAINING_COOLDOWN": _u(_MM, "ROM address, not a duration"),
    "ENEMIES_ROTATION_COOLDOWN": _u(_MM, "ROM address, not a duration"),
    "ENEMIES_UPDATE_COOLDOWN": _u(_MM, "ROM address, not a duration"),
    "COOLDOWN_GATE": _u(_MM, "ROM address of the 1-in-3 gate"),
    "COOLDOWN_BRESENHAM": _u(_MM, "ROM address of the Bresenham accumulator"),
    "COOLDOWN_BRESENHAM_STEP": _u(_MM, "ROM Bresenham step; no derivation test"),
    "ENERGY_MASK": _u(_MM, "bit mask, not a duration"),
    "_MASK_TABLE": _u("sentinel.landscape", "bit mask table, not a duration"),
    "_STEP_SIGMA": entry(
        "sentinel.astar_player",
        MEASURED,
        "whole-step charged-vs-measured rms of the live ls42 run in "
        "live_ls42_hops.json; one run, n=11",
        "test_step_sigma_is_the_measured_whole_step_rms",
    ),
    "_march_python.max_steps": _u(_LOS, _ROW),
    "_march_jit.max_steps": _u(_LOS, _ROW),
    "check_for_line_of_sight_to_tile.max_steps": _u(_LOS, _ROW),
    "aim_target.max_steps": _u(_LOS, _ROW),
    "landable_views.max_steps": _u(_LOS, _ROW),
    "landable_sweep_with_centres.max_steps": _u(_LOS, _ROW),
    "landable_view.max_steps": _u(_LOS, _ROW),
    "landable_view_targeted.max_steps": _u(_LOS, _ROW),
    "can_see_object.max_steps": _u("sentinel.relative", _ROW),
    "_PAN_STALL_FRAMES": _d(_KBD, "playerbase.H_SCROLL + playerbase.V_SCROLL"),
    "_PAN_MAX_FRAMES": _u(
        _KBD, "below a full 464-frame pan; xfail in test_pan_max_covers_full_pan"
    ),
    "_RU_STA": _u(_KBD, "ROM read-under trap byte; no derivation test"),
    "_RU_PAN": _u(
        _KBD,
        "block asserts MEASURED, which is false for this value: it describes the "
        "monitor service rate; the 20 s is fitted hang-guard headroom",
    ),
    "_RU_COMMIT": _u(_KBD, "ROM read-under trap byte; no derivation test"),
    "_SCAN_WAIT_PASSES": _u(_KBD, "scan settle passes; unmeasured"),
    "_run_to_scan.timeout": _u(_KBD, _GUARD),
    "_one_scan_press.timeout": _u(_KBD, _GUARD),
    "tap.hold": _u(_CORE, "keypress hold frames; unmeasured"),
    "tap.settle": _u(_CORE, "post-keypress settle frames; unmeasured"),
    "_enter_play.chunk": _u(_CORE, "boot advance chunk; unmeasured"),
    "boot.attempts": _u(_CORE, "boot retry budget; unmeasured"),
    "boot_loaded.attempts": _u("driver.boot", "boot retry budget; unmeasured"),
    "save_snapshot.timeout": _u("driver.boot", _GUARD),
    "load_snapshot.timeout": _u("driver.boot", _GUARD),
    "run_frames.timeout": _u("driver.clock", _GUARD),
}

# Pinned debt; the test fails on growth and on silent validation alike.
UNVALIDATED_PIN = frozenset(
    {
        "BASE_CYCLES",
        "COOLDOWN_BRESENHAM",
        "COOLDOWN_BRESENHAM_STEP",
        "COOLDOWN_GATE",
        "COOLDOWN_STICK",
        "CURSOR_REPEAT_MASK",
        "DRAINING_COOLDOWN_RELOAD",
        "ENEMIES_DRAINING_COOLDOWN",
        "ENEMIES_ROTATION_COOLDOWN",
        "ENEMIES_UPDATE_COOLDOWN",
        "ENERGY_MASK",
        "FRAME_TICKS",
        "H_SCROLL",
        "ROTATION_COOLDOWN_RELOAD",
        "SAFE_FRAMES",
        "SETTLE_FIXED_FRAMES",
        "TAP_FRAMES",
        "TUNE_TRANSFER_FRAMES",
        "UPDATE_COOLDOWN_DRAIN",
        "UPDATE_COOLDOWN_MEANIE_MADE",
        "UPDATE_COOLDOWN_MEANIE_ROTATE",
        "UPDATE_COOLDOWN_SCAN",
        "V_SCROLL",
        "WAIT_FRAMES",
        "_MASK_TABLE",
        "_PAN_MAX_FRAMES",
        "_RU_COMMIT",
        "_RU_PAN",
        "_RU_STA",
        "_SCAN_WAIT_PASSES",
        "_enter_play.chunk",
        "_march_jit.max_steps",
        "_march_python.max_steps",
        "_one_scan_press.timeout",
        "_run_to_scan.timeout",
        "aim_target.max_steps",
        "boot.attempts",
        "boot_loaded.attempts",
        "can_see_object.max_steps",
        "check_for_line_of_sight_to_tile.max_steps",
        "landable_sweep_with_centres.max_steps",
        "landable_view.max_steps",
        "landable_view_targeted.max_steps",
        "landable_views.max_steps",
        "load_snapshot.timeout",
        "run_frames.timeout",
        "save_snapshot.timeout",
        "tap.hold",
        "tap.settle",
    }
)

# Pinned constants whose source comment advertises evidence that does not exist.
KNOWN_FALSE_PROVENANCE_COMMENTS = frozenset({"_RU_PAN"})
