"""The program tracer records frames and pc arrivals of an interpreted run."""

from driver import clock, prog, prog_trace
from driver.test_kbd_prog import BUDGET, _toy


def test_a_trace_records_each_frame_and_each_watched_arrival():
    bm = _toy()
    p = prog.Prog(BUDGET)
    for _ in range(3):
        p.step().run(clock.FRAME_PC, 1.0)
    outcome, rows, events = prog_trace.trace(bm, p, pcs=(0x9633,), addrs=(0xC000,))
    assert outcome.code == prog.HALT_OK and outcome.frames == 3
    assert [r[0] for r in rows] == [1, 2, 3]
    assert all(r[2] == () for r in rows) and rows[1][1] == (rows[0][1] + 1) & 0xFF
    got = [f for f, _pc in events]
    assert {pc for _f, pc in events} == {0x9633} and got == sorted(set(got))
    assert len(got) >= 2 and got[-1] <= outcome.frames
