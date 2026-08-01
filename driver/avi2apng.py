#!/usr/bin/env python3
"""Convert a recorded run's ZMBV/AVI (:func:`driver.core.boot_and_play`) to an APNG.

The stream is indexed, mostly repeats, and changes locally, so the APNG stays indexed,
holds a repeat as delay, and carries only each frame's changed rectangle. Blank frames
are dropped: see :doc:`../docs/media.md` for why one flat colour identifies them.
"""

import argparse
import os
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

HUD_ROWS = 6  # status strip the game keeps drawing over a blanked viewport
DELAY_DEN = 1000  # APNG delays in milliseconds
_DISPOSE_NONE = 0
_BLEND_SOURCE = 0


def decode(path, max_frames=0):
    """Decode an indexed AVI to ``(frames, palette, fps)``.

    ``frames`` is ``(n, h, w)`` of indices into ``palette`` ``(k, 3)``. Frames carrying
    their own palette are remapped, so one table serves the whole run.
    """
    import av  # pylint: disable=import-outside-toplevel

    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        planes, table, lut_cache = [], {}, {}
        for frame in container.decode(stream):
            idx, pal = frame.to_ndarray()
            pal = np.asarray(pal)
            key = pal.tobytes()
            lut = lut_cache.get(key)
            if lut is None:
                lut = np.empty(len(pal), np.uint16)
                for i, argb in enumerate(pal):
                    lut[i] = table.setdefault(
                        tuple(int(v) for v in argb[1:4]), len(table)
                    )
                lut_cache[key] = lut
            planes.append(lut[idx])
            if max_frames and len(planes) >= max_frames:
                break
    if not planes:
        raise ValueError(f"{path}: no video frames")
    if len(table) > 256:
        raise ValueError(
            f"{path}: {len(table)} colours, more than an indexed PNG holds"
        )
    palette = np.zeros((len(table), 3), np.uint8)
    for rgb, i in table.items():
        palette[i] = rgb
    return np.stack(planes).astype(np.uint8), palette, fps


def active_window(frames):
    """Bounding box of the pixels that ever change: the picture inside the border."""
    varies = (frames != frames[0]).any(axis=0)
    if not varies.any():
        return slice(0, frames.shape[1]), slice(0, frames.shape[2])
    rows, cols = np.where(varies)
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def blank_frames(frames, hud_rows=HUD_ROWS, min_run=1):
    """Mask of frames whose play area is one flat colour, in runs of ``min_run``."""
    play = frames[:, hud_rows:, :] if hud_rows < frames.shape[1] else frames
    flat = play.reshape(len(play), -1)
    blank = flat.max(axis=1) == flat.min(axis=1)
    return _long_runs(blank, min_run) if min_run > 1 else blank


def _long_runs(mask, min_run):
    """Keep only runs of ``True`` at least ``min_run`` long."""
    out = np.zeros_like(mask)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start >= min_run:
            out[start:stop] = True
    return out


def _filtered(rows):
    """PNG-filter each row, picking the minimum sum of absolute differences."""
    prev = np.zeros(rows.shape[1], np.uint8)
    out = bytearray()
    for row in rows:
        left = np.r_[np.uint8(0), row[:-1]].astype(np.int16)
        up = prev.astype(np.int16)
        upleft = np.r_[np.uint8(0), prev[:-1]].astype(np.int16)
        cur = row.astype(np.int16)
        near_a, near_b, near_c = (
            np.abs(up - upleft),
            np.abs(left - upleft),
            np.abs(left + up - 2 * upleft),
        )
        paeth = np.where(
            (near_a <= near_b) & (near_a <= near_c),
            left,
            np.where(near_b <= near_c, up, upleft),
        )
        cands = [
            row,
            (cur - left).astype(np.uint8),
            (cur - up).astype(np.uint8),
            (cur - (left + up) // 2).astype(np.uint8),
            (cur - paeth).astype(np.uint8),
        ]
        best = min(range(5), key=lambda f: int(np.abs(cands[f].astype(np.int8)).sum()))
        out += bytes([best]) + cands[best].tobytes()
        prev = row
    return bytes(out)


def _chunk(tag, body):
    return (
        struct.pack(">I", len(body))
        + tag
        + body
        + struct.pack(">I", zlib.crc32(tag + body))
    )


def write_apng(path, frames, palette, delays, loops=0, scale=1):
    """Write ``frames`` as an APNG, each frame carrying only its changed rectangle."""
    if scale > 1:
        frames = np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
    height, width = frames.shape[1:]
    out = [b"\x89PNG\r\n\x1a\n"]
    out.append(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)))
    out.append(_chunk(b"acTL", struct.pack(">II", len(frames), loops)))
    out.append(_chunk(b"PLTE", palette.tobytes()))

    seq = 0
    for n, frame in enumerate(frames):
        if n == 0:
            box, x0, y0 = frame, 0, 0
        else:
            rows, cols = np.where(frame != frames[n - 1])
            y0, x0 = int(rows.min()), int(cols.min())
            box = frame[y0 : rows.max() + 1, x0 : cols.max() + 1]
        body = struct.pack(
            ">IIIIIHHBB",
            seq,
            box.shape[1],
            box.shape[0],
            x0,
            y0,
            min(0xFFFF, int(round(delays[n] * DELAY_DEN))),
            DELAY_DEN,
            _DISPOSE_NONE,
            _BLEND_SOURCE,
        )
        out.append(_chunk(b"fcTL", body))
        seq += 1
        data = zlib.compress(_filtered(box), 9)
        if n == 0:
            out.append(_chunk(b"IDAT", data))
        else:
            out.append(_chunk(b"fdAT", struct.pack(">I", seq) + data))
            seq += 1
    out.append(_chunk(b"IEND", b""))
    with open(path, "wb") as handle:
        handle.write(b"".join(out))
    return sum(len(part) for part in out)


def convert(
    src,
    dst,
    fps=0.0,
    hud_rows=HUD_ROWS,
    min_run=1,
    scale=1,
    crop=True,
    loops=0,
    max_frames=0,
    log=print,
):
    """Convert one recorded AVI to an APNG. Returns a summary dict."""
    frames, palette, src_fps = decode(src, max_frames=max_frames)
    total = len(frames)
    if crop:
        rows, cols = active_window(frames)
        frames = frames[:, rows, cols]
        log(
            f"picture {frames.shape[2]}x{frames.shape[1]} at ({cols.start},{rows.start})"
        )

    blank = blank_frames(frames, hud_rows=hud_rows, min_run=min_run)
    frames = frames[~blank]
    log(
        f"dropped {int(blank.sum())} blank of {total} frames ({100 * blank.mean():.1f}%)"
    )
    if frames.shape[0] == 0:
        raise ValueError(f"{src}: every frame was blank")

    stride = max(1, int(round(src_fps / fps))) if fps > 0 else 1
    frames = frames[::stride]
    step = stride / src_fps

    keep, delays = [0], [step]
    for n in range(1, len(frames)):
        if np.array_equal(frames[n], frames[keep[-1]]):
            delays[-1] += step
        else:
            keep.append(n)
            delays.append(step)
    frames = frames[keep]
    log(
        f"held {len(frames)} distinct frames at {src_fps / stride:.1f} fps, {sum(delays):.1f}s"
    )

    size = write_apng(dst, frames, palette, delays, loops=loops, scale=scale)
    return {
        "src_frames": total,
        "blank_dropped": int(blank.sum()),
        "frames": len(frames),
        "seconds": round(sum(delays), 2),
        "bytes": size,
        "path": dst,
    }


def main(argv=None):
    """Convert a recorded AVI to an APNG, dropping the hyperspace/transfer blanks."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("avi")
    parser.add_argument("-o", "--out", help="default: the AVI's name with .png")
    parser.add_argument("--fps", type=float, default=12.0, help="0 keeps every frame")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument(
        "--hud-rows", type=int, default=HUD_ROWS, help="status rows a blank keeps"
    )
    parser.add_argument(
        "--min-blank-run", type=int, default=1, help="shorter blank runs are kept"
    )
    parser.add_argument("--loops", type=int, default=0, help="0 loops forever")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-crop", action="store_true", help="keep the border")
    args = parser.parse_args(argv)

    dst = args.out or os.path.splitext(args.avi)[0] + ".png"
    info = convert(
        args.avi,
        dst,
        fps=args.fps,
        hud_rows=args.hud_rows,
        min_run=args.min_blank_run,
        scale=args.scale,
        crop=not args.no_crop,
        loops=args.loops,
        max_frames=args.max_frames,
    )
    print(
        f"{info['src_frames']} frames -> {info['frames']} ({info['seconds']}s, "
        f"{info['bytes'] / 1024:.0f} KiB) -> {info['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
