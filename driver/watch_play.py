#!/usr/bin/env python3
"""Watch a HUMAN play The Sentinel live in x64sc and log it.

Does NOT touch x64sc's source or send it any keys -- it only connects to the
stock binary monitor (a normal runtime flag, not a modification) and polls
game state + records a video while you play with your own keyboard.

Usage:
  1. Launch x64sc yourself with the binary monitor enabled, e.g.:
       x64sc -binarymonitor -binarymonitoraddress ip4://127.0.0.1:6502 \
             -autostart sentinel-gold.tap
     Play up to the point you want recorded (type the landscape code
     yourself, dismiss the preview, etc).
  2. In another terminal, run this script to start logging + video capture:
       python3 driver/watch_play.py --landscape 0
     It polls state ~4x/second (one combined memory read per poll, to avoid
     disrupting the emulator's speed pacing) into out/play_TIMESTAMP.jsonl and records
     renders/play_TIMESTAMP.avi (via asid-vice's native VIDEO_RECORD opcode,
     if your x64sc build has it -- if not, pass --no-video and just use the
     JSONL log).
  3. Play. Press Ctrl+C here when you're done (win or quit) to stop.
"""

import os, sys, time, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.dirname(HERE))

from vice_driver import BinMon
from vice_driver.binmon import BinmonError
from driver import sentinel_state as gs

# Every address sentinel_state.read_game_state() needs (tiles_table, all object
# arrays, player slot/energy/scalars) falls within [0, 0x0CDE] -- so ONE
# mem_get covering that whole span replaces the ~13 separate per-table reads
# read_game_state() would otherwise issue against a live ViceSource. Each
# mem_get briefly halts and resumes the CPU (auto_resume); at the old 10Hz
# poll rate that was ~13 halt/resume cycles x 10/sec = ~130/sec, enough
# real-time overhead to visibly throw off the emulator's speed pacing.
_SNAPSHOT_END = 0x0CDE  # inclusive; also covers our own done-flag byte


class _BatchSource:
    """A sentinel_state.MemorySource backed by ONE already-fetched buffer, so
    read_game_state()'s many per-table reads cost zero extra monitor calls."""

    def __init__(self, buf):
        self.buf = buf

    def read(self, addr, length):
        return bytes(self.buf[addr : addr + length])

    def byte(self, addr):
        return self.buf[addr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6502)
    ap.add_argument(
        "--landscape", type=int, default=None, help="for reference in the log only"
    )
    ap.add_argument(
        "--hz",
        type=float,
        default=4.0,
        help="poll rate; each poll is now a single mem_get, but keep this "
        "modest -- every poll still halts+resumes the CPU once",
    )
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--video-dir", default=os.path.join(ROOT, "renders"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"play_{stamp}.jsonl")
    video_path = os.path.join(args.video_dir, f"play_{stamp}.avi")

    print(f"connecting to binary monitor at {args.host}:{args.port} ...")
    bm = BinMon(args.host, args.port)
    bm.connect(timeout=30.0, attempts=100, retry_delay=0.3)
    bm.exit()
    print("connected.")

    if not args.no_video:
        try:
            bm.video_record(video_path)
            print(f"recording video -> {video_path}")
        except BinmonError as e:
            print(
                f"video_record not supported by this x64sc build ({e}); continuing without it"
            )
            args.no_video = True

    print(f"logging game state -> {log_path}")
    print("play now. Ctrl+C here to stop.")

    period = 1.0 / args.hz
    t_start = time.time()
    with open(log_path, "w") as logf:
        logf.write(json.dumps({"landscape": args.landscape, "t_start": t_start}) + "\n")
        try:
            while True:
                t0 = time.time()
                try:
                    buf = bm.mem_get(0, _SNAPSHOT_END)
                    st = gs.read_game_state(_BatchSource(buf))
                    p = st.player
                    rec = {
                        "t": round(t0 - t_start, 3),
                        "player": (
                            None
                            if p is None
                            else {
                                "x": p.x,
                                "y": p.y,
                                "z": p.z,
                                "zf": p.z_fraction,
                                "hang": p.h_angle,
                                "vang": p.v_angle,
                                "slot": p.slot,
                            }
                        ),
                        "energy": st.player_energy,
                        "n_objects": len(st.objects),
                        "done_flag": buf[0x0CDE],
                    }
                except (BinmonError, ConnectionError, OSError, TimeoutError) as e:
                    rec = {"t": round(t0 - t_start, 3), "error": str(e)}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            pass

    if not args.no_video:
        try:
            bm.video_stop()
            print(f"video saved -> {video_path}")
        except BinmonError as e:
            print(f"video_stop failed: {e}")
    print(f"log saved -> {log_path}")


if __name__ == "__main__":
    main()
