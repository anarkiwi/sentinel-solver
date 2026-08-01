"""The AVI-to-APNG converter: blank detection, and a pixel-exact APNG."""

import os

import numpy as np
import pytest

from driver import avi2apng

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _board(seed, height=24, width=32):
    """A frame with structure: never one flat colour."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 4, size=(height, width), dtype=np.uint8)
    frame[0, 0] = 5
    return frame


def _blank(colour, height=24, width=32, hud=avi2apng.HUD_ROWS):
    """A blanked viewport: flat under the status strip, which keeps its content."""
    frame = np.full((height, width), colour, np.uint8)
    frame[:hud, ::3] = 7
    return frame


def _decode_rgb(path):
    av = pytest.importorskip("av")
    with av.open(path) as container:
        stream = container.streams.video[0]
        return np.stack(
            [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        )


def test_blank_is_a_flat_play_area_whatever_the_hud_draws():
    frames = np.stack([_board(0), _blank(6), _board(1), _blank(0), _board(2)])
    assert avi2apng.blank_frames(frames).tolist() == [False, True, False, True, False]


def test_a_structured_frame_is_never_blank():
    frames = np.stack([_board(i) for i in range(8)])
    assert not avi2apng.blank_frames(frames).any()


def test_hud_rows_zero_keeps_a_blank_whose_strip_still_draws():
    frames = np.stack([_blank(6)])
    assert not avi2apng.blank_frames(frames, hud_rows=0)[0]
    assert avi2apng.blank_frames(frames, hud_rows=avi2apng.HUD_ROWS)[0]


def test_min_run_keeps_isolated_blanks_and_drops_sustained_ones():
    frames = np.stack(
        [_board(0), _blank(6), _board(1), _blank(6), _blank(6), _blank(6)]
    )
    kept = avi2apng.blank_frames(frames, min_run=3)
    assert kept.tolist() == [False, False, False, True, True, True]


def test_active_window_finds_the_picture_inside_a_static_border():
    frames = np.zeros((5, 20, 30), np.uint8)
    for n in range(5):
        frames[n, 4:12, 6:20] = _board(n, 8, 14)
    rows, cols = avi2apng.active_window(frames)
    assert (rows.start, rows.stop) == (4, 12)
    assert (cols.start, cols.stop) == (6, 20)


def test_active_window_on_a_still_run_keeps_everything():
    frames = np.stack([_board(0)] * 3)
    rows, cols = avi2apng.active_window(frames)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (0, 24, 0, 32)


def test_apng_round_trips_pixel_exactly(tmp_path):
    frames = np.stack([_board(i) for i in range(6)])
    frames[3] = frames[2]
    frames[3, 5, 5] ^= 1  # a one-pixel change must survive the changed-rectangle path
    palette = np.array([[i * 30, 255 - i * 30, i * 7] for i in range(8)], np.uint8)
    out = str(tmp_path / "anim.png")
    avi2apng.write_apng(out, frames, palette, [0.1] * len(frames))

    got = _decode_rgb(out)
    assert got.shape == (len(frames), 24, 32, 3)
    assert np.array_equal(got, palette[frames])


@pytest.mark.parametrize("seed", range(4))
def test_every_row_filter_round_trips(tmp_path, seed):
    """Wide noisy frames make the heuristic reach for all five filters, Paeth included."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(3, 40, 64), dtype=np.uint8)
    palette = rng.integers(0, 256, size=(256, 3), dtype=np.uint8)
    out = str(tmp_path / f"noise{seed}.png")
    avi2apng.write_apng(out, frames, palette, [0.1] * 3)
    assert np.array_equal(_decode_rgb(out), palette[frames])


def test_apng_holds_a_single_frame(tmp_path):
    frames = np.stack([_board(0)])
    palette = np.array([[i, i, i] for i in range(8)], np.uint8)
    out = str(tmp_path / "one.png")
    avi2apng.write_apng(out, frames, palette, [0.5])
    assert np.array_equal(_decode_rgb(out), palette[frames])


def test_scale_upsamples_by_whole_pixels(tmp_path):
    frames = np.stack([_board(0), _board(1)])
    palette = np.array([[i * 20, i, 0] for i in range(8)], np.uint8)
    out = str(tmp_path / "big.png")
    avi2apng.write_apng(out, frames, palette, [0.1, 0.1], scale=3)
    got = _decode_rgb(out)
    assert got.shape == (2, 72, 96, 3)
    assert np.array_equal(
        got, np.repeat(np.repeat(palette[frames], 3, axis=1), 3, axis=2)
    )


def _fake_decode(monkeypatch, frames, fps=50.0):
    """Stand in for the AVI decoder, so convert() runs without an emulator."""
    palette = np.array([[i * 30, 255 - i * 30, i * 7] for i in range(8)], np.uint8)
    monkeypatch.setattr(
        avi2apng, "decode", lambda _p, max_frames=0: (frames, palette, fps)
    )
    return palette


def test_convert_refuses_a_run_that_is_blank_throughout(tmp_path, monkeypatch):
    _fake_decode(monkeypatch, np.stack([_blank(6)] * 4))
    with pytest.raises(ValueError, match="every frame was blank"):
        avi2apng.convert("x.avi", str(tmp_path / "x.png"), log=lambda _m: None)


def test_convert_drops_the_blanks_and_keeps_the_play(tmp_path, monkeypatch):
    frames = np.stack([_board(0), _blank(6), _blank(6), _board(1), _board(2)])
    palette = _fake_decode(monkeypatch, frames)
    out = str(tmp_path / "run.png")
    info = avi2apng.convert("x.avi", out, crop=False, log=lambda _m: None)

    assert (info["src_frames"], info["blank_dropped"], info["frames"]) == (5, 2, 3)
    assert np.array_equal(_decode_rgb(out), palette[frames[[0, 3, 4]]])


def test_convert_holds_a_repeat_as_delay_instead_of_a_frame(tmp_path, monkeypatch):
    frames = np.stack([_board(0), _board(0), _board(0), _board(1)])
    _fake_decode(monkeypatch, frames, fps=10.0)
    info = avi2apng.convert(
        "x.avi", str(tmp_path / "r.png"), crop=False, log=lambda _m: None
    )

    assert info["frames"] == 2
    assert info["seconds"] == pytest.approx(0.4)


def test_convert_decimates_to_the_asked_frame_rate(tmp_path, monkeypatch):
    frames = np.stack([_board(i) for i in range(20)])
    _fake_decode(monkeypatch, frames, fps=50.0)
    info = avi2apng.convert(
        "x.avi", str(tmp_path / "d.png"), fps=10.0, crop=False, log=lambda _m: None
    )

    assert info["frames"] == 4  # stride 5 over 20 frames
    assert info["seconds"] == pytest.approx(0.4)


def test_speed_shortens_the_run_without_dropping_a_frame(tmp_path, monkeypatch):
    frames = np.stack([_board(i) for i in range(10)])
    palette = _fake_decode(monkeypatch, frames, fps=10.0)
    out = str(tmp_path / "fast.png")
    info = avi2apng.convert("x.avi", out, speed=2.0, crop=False, log=lambda _m: None)

    assert info["frames"] == 10
    assert info["seconds"] == pytest.approx(0.5)  # 1.0s of source, played twice over
    assert np.array_equal(_decode_rgb(out), palette[frames])


def test_speed_scales_a_held_repeat_too(tmp_path, monkeypatch):
    frames = np.stack([_board(0), _board(0), _board(0), _board(1)])
    _fake_decode(monkeypatch, frames, fps=10.0)
    info = avi2apng.convert(
        "x.avi", str(tmp_path / "held.png"), speed=4.0, crop=False, log=lambda _m: None
    )

    assert info["frames"] == 2
    assert info["seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_a_speed_that_is_not_positive_is_refused(tmp_path, monkeypatch, speed):
    _fake_decode(monkeypatch, np.stack([_board(0), _board(1)]))
    with pytest.raises(ValueError, match="speed must be positive"):
        avi2apng.convert(
            "x.avi", str(tmp_path / "x.png"), speed=speed, log=lambda _m: None
        )


@pytest.mark.parametrize("name", ["player_ls340_win.avi", "ls335_phase_win.avi"])
def test_converts_a_recorded_run_when_one_is_present(tmp_path, name):
    src = os.path.join(ROOT, "renders", name)
    if not os.path.exists(src):
        pytest.skip(f"{name} not recorded here")
    pytest.importorskip("av")
    out = str(tmp_path / "run.png")
    info = avi2apng.convert(src, out, fps=6.0, max_frames=1500, log=lambda _m: None)
    assert info["frames"] >= 1
    assert info["frames"] <= info["src_frames"]
    assert os.path.getsize(out) == info["bytes"]
    assert _decode_rgb(out).shape[0] == info["frames"]
