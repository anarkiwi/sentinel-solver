"""The CRT recording drives asid-vice's rendered VIDEO_RECORD with the exact bodies, and
pauses while the VIC shows a viewport of one colour."""

from vice_driver.binmon import CHECK_STORE, OPCODE

from driver import crt


class _Mon:
    def __init__(self, colours=b"\x00\x00\x00"):
        self.calls, self.colours, self.reads = [], colours, []

    def call(self, opcode, body):
        self.calls.append((opcode, body))

    def mem_get(self, start, end, bank):
        self.reads.append((start, end, bank))
        return self.colours


def test_start_sends_the_rendered_recording_body():
    bm = _Mon()
    crt.start(bm, "/renders/x.mp4")
    ((op, body),) = bm.calls
    assert op == OPCODE.VIDEO_RECORD and body[0] == 2
    assert int.from_bytes(body[1:3], "little") == 720 and body[3] == len(
        "/renders/x.mp4"
    )
    assert body[4:] == b"/renders/x.mp4"


def test_a_write_to_the_background_pauses_on_one_colour_and_resumes_on_a_scheme():
    blue = _Mon(bytes([0xF6, 0x06, 0x06]))
    crt.background(blue)
    scheme = _Mon(bytes([0x00, 0x06, 0x02]))
    crt.background(scheme)
    assert blue.reads == [(0xD021, 0xD023, "io")]
    assert blue.calls == [(OPCODE.VIDEO_RECORD, b"\x03")]
    assert scheme.calls == [(OPCODE.VIDEO_RECORD, b"\x04")]
    assert crt.WATCH == {(CHECK_STORE, 0xD021): crt.background}
    assert dict(zip(crt.ARGS[::2], crt.ARGS[1::2])) == {
        "-VICIIfilter": "1",
        "-VICIIaspectmode": "2",
    }
