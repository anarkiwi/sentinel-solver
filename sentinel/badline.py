"""What one VIC-II badline costs the 6510, derived rather than fitted.

A badline pulls BA low three cycles before the VIC takes the bus, and the 6510 runs on
until its first READ cycle, so the steal is ``BADLINE_STEAL`` less the consecutive write
cycles the CPU happens to be performing at the window's first cycle.
"""

import numpy as np

from sentinel import writeruns

PAL_FRAME_CYCLES = 19656  # PAL 6569: 312 raster lines x 63 cycles
BADLINE_STEAL = 43  # 40 VIC c-accesses + the 3-cycle AEC lag: the MAXIMUM, not a mode
BADLINES_PER_FRAME = 25  # raster 51..243 step 8; $D015 = 0, so there is no sprite term
SHORT_IRQ = 119  # $95E9 split chain at raster 53/93/133/173: 7 entry + 112 body
SHORT_IRQ_WRAP = 1  # the one entry a frame whose $9603 BPL wraps the split index to 4
LINE_CYCLES = 63  # PAL 6569: 19656 = 312 x 63
BADLINE_FIRST_LINE = 51  # raster $33, the first $30..$F7 line with low 3 bits = YSCROLL
BADLINE_LINE_STEP = 8
BADLINE_WINDOW_CYCLE = 11  # solved live: the one cycle that derives every sampled steal
MIN_STEAL = BADLINE_STEAL - 2  # a write run is at most an RMW's or a JSR's two
WEIGHT_SHIFT = 20  # fixed point the per-window refund of a weighted term accrues in
RASTER_IRQ_LINE = 213  # $9589's $D5 entry: the once-a-frame $9630 body
IRQ_ENTRY = 7  # the 6510 interrupt sequence, run at an instruction boundary
IRQ_POLL_CYCLE = 2  # ... polled this far into the interrupted instruction, so a TAKEN
# branch's own extra cycles (3, or 4 across a page) come off the entry: 6, or 5
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
        for i in range(BADLINES_PER_FRAME)
    )


def steal(op, position, windows=None):
    """The cycles a badline steals from ``op`` fetched at frame ``position``.

    ``None`` when no badline window falls at or after the fetch, i.e. the instruction
    is past the last badline of the frame and pays nothing."""
    windows = window_positions() if windows is None else windows
    window = min((w for w in windows if w >= position), default=None)
    if window is None:
        return None
    return BADLINE_STEAL - write_run(op, window - position)


def entry_cycles(taken_branch=0):
    """The interrupt sequences that can follow the instruction the raster caught.

    ``taken_branch`` is its own cost when it is a TAKEN branch (3, or 4 across a page),
    which polls two cycles in: a raster asserted by then starts the sequence there and
    the branch's remaining cycles come off the entry; a later one pays the whole 7.
    """
    if taken_branch <= IRQ_POLL_CYCLE:
        return (IRQ_ENTRY,)
    return (IRQ_ENTRY, IRQ_ENTRY + IRQ_POLL_CYCLE - taken_branch)


def marker_position(boundary, entry=IRQ_ENTRY):
    """Frame position of the $9630 marker for a raster IRQ taken at ``boundary``.

    The IRQ is taken at an instruction boundary, so the marker's own frame position is
    that boundary plus its own entry -- it is aligned, not blurred.
    """
    return boundary + entry + MARKER_OFFSET


FRAME_ORIGIN = RASTER_IRQ_LINE * LINE_CYCLES  # the raster the $9589 $D5 entry programs
SHORT_IRQ_LINES = (53, 93, 133, 173)  # the $9589 table's other four entries
SHORT_IRQ_WRAP_LINE = 53  # $9589 table[0] = $35: the entry whose $9603 BPL wraps to 4
NO_EVENT = 1 << 40  # past every event: a clock with no windows left to place
FRAME_STEAL_CEILING = BADLINES_PER_FRAME * BADLINE_STEAL  # every window a full steal
CLOCK_FIELDS = 7  # position, next event, event index, refund, refund residue, b (what
# the instruction the raster crossed owed), and that term's own per-window refund


def frame_events(line_cycle=BADLINE_WINDOW_CYCLE):
    """``(offset from the raster IRQ, cycles it takes)`` of everything that displaces
    the foreground, in frame order: 25 BA windows (0, priced by their own map) and the
    four split interrupts.  The raster-53 entry is the one whose $9603 BPL wraps the
    index: it falls through at 2 and pays $9605 LDX #$04, so it costs one more.
    """
    windows = [
        ((w - FRAME_ORIGIN) % PAL_FRAME_CYCLES, 0) for w in window_positions(line_cycle)
    ]
    short = [
        (
            (line * LINE_CYCLES - FRAME_ORIGIN) % PAL_FRAME_CYCLES,
            SHORT_IRQ + (SHORT_IRQ_WRAP if line == SHORT_IRQ_WRAP_LINE else 0),
        )
        for line in SHORT_IRQ_LINES
    ]
    return tuple(sorted(windows + short))


EVENT_POS = np.array([p for p, _ in frame_events()] + [NO_EVENT], dtype=np.int64)
EVENT_SHORT_IRQ = np.array([k for _, k in frame_events()] + [0], dtype=np.int64)
N_EVENTS = len(EVENT_POS) - 1


def frame_clock(armed=True):
    """A clock at the raster IRQ, its events re-armed; ``armed`` False places none."""
    clk = np.zeros(CLOCK_FIELDS, dtype=np.int64)
    clk[1] = EVENT_POS[0] if armed else NO_EVENT
    clk[2] = 0 if armed else N_EVENTS
    return clk


def owed_at(anchor, offset):
    """Cycles the instruction ``offset`` into the run at ``anchor`` has still to run.

    The 6510 finishes it before taking an interrupt, so this is ``b``: 0 where the
    offset is an instruction boundary, and 0 where no map names the run.
    """
    start = writeruns.START[anchor]
    if start < 0 or not 0 <= offset < writeruns.LENGTH_AT[anchor]:
        return 0
    return int(writeruns.OWED[start + offset])


def run_at(anchor, offset):
    """Consecutive write cycles ``offset`` cycles into the run counted from ``anchor``."""
    start = writeruns.START[anchor]
    if start < 0 or offset >= writeruns.LENGTH_AT[anchor]:
        return 0
    return int(writeruns.RUN[start + offset])


def _place(clk, anchor, cycles):
    """The slow half of :func:`charge`: the events this term's cycles reach."""
    pos, index, refund = clk[0], clk[2], 0
    start, end = pos, pos + cycles
    while EVENT_POS[index] < end:
        split = EVENT_SHORT_IRQ[index]
        if split:
            pos += split
            end += split
        else:
            spend = BADLINE_STEAL - run_at(anchor, EVENT_POS[index] - pos)
            refund += BADLINE_STEAL - spend
            pos += spend
            end += spend
        index += 1
    if start < PAL_FRAME_CYCLES <= end:  # the raster crosses this term
        clk[5] = owed_at(anchor, PAL_FRAME_CYCLES - pos)
    clk[0], clk[1], clk[2] = end, EVENT_POS[index], index
    clk[3] += refund
    return cycles - refund


def overhang(clk):
    """How far past the raster this frame's clock ran: the next frame's own carry.

    Negative where the budget ran out before the raster came round, so the frame ended
    with the foreground short of it and owes no carry at all.
    """
    return int(clk[0]) - PAL_FRAME_CYCLES


def charge(clk, anchor, cycles):
    """Spend ``cycles`` of the run at ``anchor``, less what its writes refund.

    The frame is charged ``FRAME_STEAL_CEILING`` up front, so a window landing on a
    write run gives its cycles back here; one landing on a read costs nothing more.
    """
    end = clk[0] + cycles
    if end <= clk[1]:
        if clk[0] < PAL_FRAME_CYCLES <= end:  # the raster crosses this term
            clk[5] = owed_at(anchor, PAL_FRAME_CYCLES - clk[0])
        clk[0] = end
        return cycles
    return _place(clk, anchor, cycles)


def _place_run(clk, cycles, step):
    """The slow half of :func:`_charge_step`: the events this term's cycles reach.

    The walk stops at the frame's last event: the raster IRQ interrupts a term that
    outlives the frame like any other, so what is left of it -- and ``step`` with it --
    is the next frame's carry, and the clock stays in the STARTING frame's coordinates.
    """
    index, refund = clk[2], 0
    end = clk[0] + cycles
    while EVENT_POS[index] < end:
        split = EVENT_SHORT_IRQ[index]
        if split:
            end += split
        else:
            clk[4] += step
            back = clk[4] >> WEIGHT_SHIFT
            clk[4] -= back << WEIGHT_SHIFT
            refund += back
            end += BADLINE_STEAL - back
        index += 1
    if clk[0] < PAL_FRAME_CYCLES <= end:  # no map: the raster's own b is unknown
        clk[5], clk[6] = 0, step
    clk[0], clk[1], clk[2] = end, EVENT_POS[index], index
    clk[3] += refund
    return cycles - refund


def _charge_step(clk, cycles, step):
    """Spend ``cycles`` whose every window refunds the fixed-point ``step``."""
    end = clk[0] + cycles
    if end <= clk[1]:
        if clk[0] < PAL_FRAME_CYCLES <= end:  # no map: the raster's own b is unknown
            clk[5], clk[6] = 0, step
        clk[0] = end
        return cycles
    return _place_run(clk, cycles, step)


def charge_run(clk, cycles, weight):
    """Spend ``cycles`` of a run no static map reaches, less what its writes refund.

    A term whose loop outlives its map -- the $1CDD ray-march, the $8401 chain past
    $1887's own run, the $9630 body the $95E9 walk RTIs out of -- carries its own write
    weight (:mod:`sentinel.writeweight`), so each window gets ``weight / cycles`` back.
    """
    return _charge_step(clk, cycles, (weight << WEIGHT_SHIFT) // cycles)


def carry(clk, cycles, step):
    """Advance over ``cycles`` a previous frame's budget paid; return their refund.

    The raster catches the foreground mid-term and the model charges that term whole,
    so the machine still owes its tail and runs it after the IRQ: real time on this
    frame's clock, no budget, and the crossed term's own per-window refund ``step``.
    """
    return cycles - _charge_step(clk, cycles, step)


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
        total += BADLINE_STEAL - write_run(op, window - position)
    return total
