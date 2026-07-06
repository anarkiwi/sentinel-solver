"""Unit tests for the reusable boot-snapshot helpers (driver/boot.py). No VICE or
Docker: the monitor is faked, so only the save/skip logic and the MON_CMD_DUMP body
encoding are exercised."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driver import boot  # noqa: E402


class _FakeBM:
    """Records monitor calls instead of talking to VICE."""

    def __init__(self):
        self.calls = []

    def call(self, opcode, body, timeout=None):
        self.calls.append((opcode, body, timeout))


def test_save_snapshot_encodes_mon_cmd_dump():
    bm = _FakeBM()
    boot.save_snapshot(bm, "/renders/boot.vsf")
    assert len(bm.calls) == 1
    opcode, body, _ = bm.calls[0]
    assert opcode == boot.SNAP_SAVE_OPCODE  # MON_CMD_DUMP $41
    # body = SR|SD|FL|FN with ROMs/disks off and the container path as the filename.
    fn = b"/renders/boot.vsf"
    assert body == bytes([0, 0, len(fn)]) + fn


def test_saves_when_missing(tmp_path):
    bm = _FakeBM()
    host = boot.save_boot_snapshot_if_missing(bm, str(tmp_path), log=lambda *a: None)
    assert host == os.path.join(str(tmp_path), boot.BOOT_VSF_NAME)
    assert len(bm.calls) == 1
    # it targets the /renders container path, not the host path.
    assert bm.calls[0][1].endswith(b"/renders/" + boot.BOOT_VSF_NAME.encode())


def test_skips_when_present(tmp_path):
    (tmp_path / boot.BOOT_VSF_NAME).write_bytes(b"snapshot")
    bm = _FakeBM()
    host = boot.save_boot_snapshot_if_missing(bm, str(tmp_path), log=lambda *a: None)
    assert host is None
    assert not bm.calls  # no monitor call when the snapshot already exists


def test_save_failure_is_nonfatal(tmp_path):
    class _BoomBM:
        def call(self, *a, **k):
            raise RuntimeError("socket dropped")

    host = boot.save_boot_snapshot_if_missing(
        _BoomBM(), str(tmp_path), log=lambda *a: None
    )
    assert host is None  # swallowed: a boot snapshot is an optimisation, never fatal
