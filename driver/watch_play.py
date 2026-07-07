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

import os, sys, time, json, argparse, base64

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.dirname(HERE))

from vice_driver import BinMon
from vice_driver.binmon import BinmonError
from driver import sentinel_state as gs
from sentinel import memmap as mm

# Every address of mutable game state -- tiles_table ($0400), all object arrays
# (flags $0100 .. type $0A40), the sights cursor ($0CC6/$0CC7), the LOS scratch
# flag do_line_of_sight_checks ($0C6E), the PRNG LFSR ($0C7B), player energy and
# scalars, and our done-flag ($0CDE) -- falls at or below $0CDE. The py65 ROM
# oracle loads code/static data only from $0D00 up, so [0, 0x0CFF] is the WHOLE
# reconstructable game state: one mem_get captures it, and the base64 of that
# exact span can be injected verbatim into a fresh sim State OR the ROM oracle to
# replay any moment's line-of-sight bit-for-bit. Rounded up past $0CDE for margin.
_SNAPSHOT_END = 0x0CFF  # inclusive

# Byte offsets we decode into convenience fields (everything is also in "mem").
_CURSOR_X = 0x0CC6  # sights cursor cx (prepare_vector_from_player_sights $1C13)
_CURSOR_Y = 0x0CC7  # sights cursor cy ($1C2D)
_DO_LOS = 0x0C6E  # do_line_of_sight_checks; bit7 waives the looking-up rejection
_PRNG = mm.PRND_STATE  # 40-bit LFSR, 5 bytes -- makes each capture deterministic


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
        default=8.0,
        help="poll rate. Each poll is a SINGLE mem_get (one halt/resume), so even "
        "8-15Hz stays far under the ~130/sec that once disrupted pacing; denser "
        "polling better catches the exact sights aim at the instant a build fires",
    )
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out"))
    ap.add_argument("--video-dir", default=os.path.join(ROOT, "renders"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--no-full-mem",
        action="store_true",
        help="omit the base64 [0,0x0CFF] dump per record (loses exact-state replay)",
    )
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

    def snapshot(buf, t0, bracket=None):
        """Decode a captured [0,_SNAPSHOT_END] buffer into one log record. The full
        buffer is stored base64 under "mem" so the exact state can be reloaded into a
        sim State or the ROM oracle later; the decoded fields are conveniences."""
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
            # LOS/aim context, decoded from the same buffer:
            "cursor": [buf[_CURSOR_X], buf[_CURSOR_Y]],
            "do_los": buf[_DO_LOS],
            "prng": list(buf[_PRNG : _PRNG + 5]),
        }
        if bracket is not None:
            rec["bracket"] = bracket
        if not args.no_full_mem:
            rec["mem"] = base64.b64encode(bytes(buf)).decode("ascii")
        return rec

    def key_of(rec):
        """The state signature whose change marks a player action worth bracketing."""
        p = rec.get("player") or {}
        return (
            rec.get("energy"),
            rec.get("n_objects"),
            p.get("slot"),
            p.get("x"),
            p.get("y"),
            rec.get("done_flag"),
        )

    period = 1.0 / args.hz
    t_start = time.time()
    n_events = 0
    prev_key = None
    with open(log_path, "w") as logf:
        logf.write(
            json.dumps(
                {
                    "landscape": args.landscape,
                    "t_start": t_start,
                    "snapshot_span": [0, _SNAPSHOT_END],
                    "full_mem": not args.no_full_mem,
                    "schema": "watch_play/2 -- per-record base64 mem of [0,span] "
                    "for exact LOS replay; change records bracketed post-action",
                }
            )
            + "\n"
        )
        try:
            while True:
                t0 = time.time()
                try:
                    buf = bm.mem_get(0, _SNAPSHOT_END)
                    rec = snapshot(buf, t0)
                    logf.write(json.dumps(rec) + "\n")
                    # A player action (build/absorb/transfer/win) changed the state:
                    # immediately re-dump to bracket the transition tightly, so the
                    # firing aim (pre) and settled result (post) are both captured.
                    k = key_of(rec)
                    if prev_key is not None and k != prev_key and "error" not in rec:
                        try:
                            buf2 = bm.mem_get(0, _SNAPSHOT_END)
                            post = snapshot(buf2, time.time(), bracket="post")
                            logf.write(json.dumps(post) + "\n")
                        except (BinmonError, ConnectionError, OSError, TimeoutError):
                            post = rec
                        n_events += 1
                        pp = post.get("player") or {}
                        print(
                            f"  [{n_events}] t={post['t']}s change: "
                            f"e={post['energy']} nobj={post['n_objects']} "
                            f"player=({pp.get('x')},{pp.get('y')},z{pp.get('z')}) "
                            f"slot={pp.get('slot')} done={post['done_flag']}"
                            + ("  *** WIN ***" if post["done_flag"] else "")
                        )
                    prev_key = k
                except (BinmonError, ConnectionError, OSError, TimeoutError) as e:
                    rec = {"t": round(t0 - t_start, 3), "error": str(e)}
                    logf.write(json.dumps(rec) + "\n")
                logf.flush()
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            pass
    print(f"captured {n_events} state-change events.")

    if not args.no_video:
        try:
            bm.video_stop()
            print(f"video saved -> {video_path}")
        except BinmonError as e:
            print(f"video_stop failed: {e}")
    print(f"log saved -> {log_path}")


if __name__ == "__main__":
    main()
