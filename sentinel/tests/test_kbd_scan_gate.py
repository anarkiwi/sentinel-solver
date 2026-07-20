"""The gated-input-scan mechanism the sights toggle rides on (driver/kbd_aim.py).

Modelled from the ROM, no VICE: the IRQ reaches ``check_for_full_player_input`` ($9678)
once a frame while the world is not being plotted ($0CE4 bit7, $9669), and that scan
toggles the sights ($0C5F bit7) only on a SPACE edge ($11B5 skips while $1236 is set).
"""

import contextlib

import pytest

from driver import kbd_aim

_SPACE = "SPACE"
_FPS = 50.0


class _GateBm:
    """ROM input-gate model with a binmon-shaped surface (the calls KbdDriver makes)."""

    def __init__(self, sights_on=True, plot_frames=0):
        self.mem = {kbd_aim.A_SFLAG: 0x80 if sights_on else 0x00, kbd_aim.A_PLOT: 0x00}
        if plot_frames:
            self.mem[kbd_aim.A_PLOT] = 0x80
        self.plot_frames = plot_frames
        self.frames = 0
        self.scans = 0
        self.toggles = 0
        self.pressed = set()
        self.space_lock = False
        self.pc = kbd_aim.KbdDriver.PC_IRQ_SCAN_DONE

    @contextlib.contextmanager
    def halted(self):
        yield

    def exit(self):
        pass

    def mem_get(self, a, b):
        return bytes(self.mem.get(x, 0) for x in range(a, b + 1))

    def keymatrix_set(self, presses):
        self.pressed |= {p[:2] for p in presses}

    def keymatrix_release_all(self):
        self.pressed = set()

    def _frame(self):
        self.frames += 1
        if self.plot_frames:
            self.plot_frames -= 1
            if not self.plot_frames:
                self.mem[kbd_aim.A_PLOT] = 0x00

    def _scan(self):
        """check_for_full_player_input $119F: the SPACE edge test."""
        self.scans += 1
        if kbd_aim._k(_SPACE) in self.pressed:
            if not self.space_lock:
                self.mem[kbd_aim.A_SFLAG] ^= 0x80
                self.toggles += 1
            self.space_lock = True
        else:
            self.space_lock = False

    def advance_instructions(self, _n):
        if self.pc == kbd_aim.KbdDriver.PC_IRQ_SCAN:
            self.pc = "in_scan"

    def run_until_pc(self, target, timeout=5.0, condition=None):
        if target == kbd_aim.KbdDriver.PC_IRQ_SCAN_DONE:
            self._scan()  # the JSR at $9678 runs to its return address
            self.pc = target
            return
        if self.pc == target:  # binmon fast path: already halted there
            return
        if self.pc == "in_scan":
            self._scan()
        spent = 0
        while self.mem[kbd_aim.A_PLOT] & 0x80:  # gate shut: no scan can fire
            self._frame()
            spent += 1
            if spent / _FPS > timeout:
                raise TimeoutError(f"run_until_pc(${target:04X})")
        self._frame()
        self.pc = target


def _drv(bm):
    return kbd_aim.KbdDriver(bm, lambda *_a: None, quantized=True)


def test_back_to_back_toggles_each_land_on_the_first_press():
    """Sights OFF then straight back ON -- no pan runs between them, so nothing else
    clears $1236 -- still costs exactly one press each, via the idle re-arm scan."""
    bm = _GateBm(sights_on=True)
    drv = _drv(bm)
    assert drv.sights_set(False)
    assert drv.sights_set(True)
    assert bm.toggles == 2
    assert bm.scans == 4  # one idle re-arm + one press scan per toggle
    assert bm.frames <= kbd_aim._SCAN_WAIT_PASSES


def test_toggle_without_the_re_arm_scan_is_swallowed():
    """The mechanism itself: SPACE pressed through a second scan with no released-key
    scan between is skipped at $11B5 -- what ``_one_scan_press`` re-arms past."""
    bm = _GateBm(sights_on=True)
    key = kbd_aim._k(_SPACE)
    for _ in range(2):
        bm.keymatrix_set([(*key, 1)])
        bm._scan()
        bm.keymatrix_release_all()
    assert bm.toggles == 1


def test_scan_wait_outlasts_a_viewpoint_redraw():
    """A redraw holds $0CE4 for hundreds of frames, so no scan can fire; the wait
    re-arms and spends them INSIDE the primitive instead of leaking them onward."""
    bm = _GateBm(sights_on=True, plot_frames=400)
    drv = _drv(bm)
    assert drv._run_to_scan(timeout=1.0)
    assert bm.frames >= 400
    assert not bm.mem[kbd_aim.A_PLOT] & 0x80


def test_scan_wait_raises_when_the_gate_is_open():
    """A timeout with the plot flag CLEAR is a real stall (a PC that cannot recur), not
    back-pressure -- it must propagate, not spin."""

    class _Stalled(_GateBm):
        def run_until_pc(self, target, timeout=5.0, condition=None):
            raise TimeoutError("stalled")

    with pytest.raises(TimeoutError):
        _drv(_Stalled())._run_to_scan(timeout=0.1)


def test_sights_set_reaches_the_wanted_flag_across_a_redraw():
    """The whole toggle path under a redraw: it waits the plot out and lands."""
    bm = _GateBm(sights_on=True, plot_frames=120)
    drv = _drv(bm)
    assert drv.sights_set(False)
    assert not bm.mem[kbd_aim.A_SFLAG] & 0x80
    assert bm.toggles == 1
