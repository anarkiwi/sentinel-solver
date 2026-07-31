"""What one VIC-II badline costs the 6510, derived rather than fitted.

A badline pulls BA low three cycles before the VIC takes the bus, and the 6510 runs on
until its first READ cycle, so the steal is ``BADLINE_STEAL`` less the consecutive write
cycles the CPU happens to be performing at the window's first cycle.
"""

from sentinel import passcost

LINE_CYCLES = 63  # PAL 6569: 19656 = 312 x 63
BADLINE_FIRST_LINE = 51  # raster $33, the first $30..$F7 line with low 3 bits = YSCROLL
BADLINE_LINE_STEP = 8
BADLINE_WINDOW_CYCLE = 11  # solved live: the one cycle that derives every sampled steal
MIN_STEAL = passcost.BADLINE_STEAL - 2  # a write run is at most an RMW's or a JSR's two
RASTER_IRQ_LINE = 213  # $9589's $D5 entry: the once-a-frame $9630 body
IRQ_ENTRY = 7  # the 6510 interrupt sequence, run at an instruction boundary
IRQ_ENTRY_BRANCH = 6  # ... one less off a branch, whose IRQ poll is a cycle earlier
MARKER_OFFSET = 81  # $95E9 to $9630, the head of that body

# Write cycles by opcode, indexed from its fetch: at most an RMW or JSR/push pair.
WRITE_CYCLES = {0x08: (2,), 0x48: (2,), 0x20: (3, 4), 0x81: (5,), 0x91: (5,)}
WRITE_CYCLES.update({op: (2,) for op in (0x84, 0x85, 0x86)})
WRITE_CYCLES.update({op: (3,) for op in (0x8C, 0x8D, 0x8E, 0x94, 0x95, 0x96)})
WRITE_CYCLES.update({op: (4,) for op in (0x99, 0x9D)})
WRITE_CYCLES.update({op: (3, 4) for op in (0x06, 0x26, 0x46, 0x66, 0xC6, 0xE6)})
WRITE_CYCLES.update({op: (4, 5) for op in (0x16, 0x36, 0x56, 0x76, 0xD6, 0xF6)})
WRITE_CYCLES.update({op: (4, 5) for op in (0x0E, 0x2E, 0x4E, 0x6E, 0xCE, 0xEE)})
WRITE_CYCLES.update({op: (5, 6) for op in (0x1E, 0x3E, 0x5E, 0x7E, 0xDE, 0xFE)})


def write_run(op, index):
    """Consecutive write cycles of ``op`` starting at cycle ``index`` from its fetch."""
    cycles = WRITE_CYCLES.get(op, ())
    run = 0
    while index + run in cycles:
        run += 1
    return run


def window_positions(line_cycle=BADLINE_WINDOW_CYCLE):
    """Frame positions of the BA window on each badline, measured from raster 0."""
    return tuple(
        (BADLINE_FIRST_LINE + BADLINE_LINE_STEP * i) * LINE_CYCLES + line_cycle
        for i in range(passcost.BADLINES_PER_FRAME)
    )


def steal(op, position, windows=None):
    """The cycles a badline steals from ``op`` fetched at frame ``position``.

    ``None`` when no badline window falls at or after the fetch, i.e. the instruction
    is past the last badline of the frame and pays nothing."""
    windows = window_positions() if windows is None else windows
    window = min((w for w in windows if w >= position), default=None)
    if window is None:
        return None
    return passcost.BADLINE_STEAL - write_run(op, window - position)


def marker_position(boundary, branch=False):
    """Frame position of the $9630 marker for a raster IRQ taken at ``boundary``.

    The IRQ is taken at an instruction boundary, so the marker's own frame position is
    that boundary plus a constant -- it is aligned, not blurred.
    """
    return boundary + (IRQ_ENTRY_BRANCH if branch else IRQ_ENTRY) + MARKER_OFFSET


def frame_steal(instructions, windows=None):
    """One frame's whole badline steal, from its ``(position, opcode)`` stream.

    The stream is ascending and an instruction runs until the next one starts, so each
    BA window names an opcode and an offset into it: the law, applied 25 times.  A
    window ahead of the stream's first instruction is charged the full steal.
    """
    windows = window_positions() if windows is None else windows
    stream = list(instructions)
    total, index = 0, 0
    for window in windows:
        while index + 1 < len(stream) and stream[index + 1][0] <= window:
            index += 1
        position, op = stream[index]
        total += passcost.BADLINE_STEAL - write_run(op, window - position)
    return total
