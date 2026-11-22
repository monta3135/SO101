#!/usr/bin/env python3
"""elbow_flex を小さく動かして、手首カメラに写る青い領域がどちらへ動くかを測る。

掴む動作の前に、下ろす向きを確かめるためのもの。elbow を --step 度だけ増やし、
戻してから、その前後で青い領域の位置（画素）を比べて表示する。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from track_face import CAM_HEIGHT, CAM_WIDTH, DEFAULT_CAMERA, clamp, move_smoothly, safe_limits
from find_blue_can import find_blue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="elbow を動かして画像の動く向きを測る")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--step", type=float, default=5.0, help="elbow を動かす量（度）")
    parser.add_argument("--speed", type=float, default=10.0, help="動かす速さ（度/秒）")
    parser.add_argument("--min-area", type=float, default=1500.0)
    parser.add_argument("--max-relative-target", type=float, default=15.0)
    return parser.parse_args()


def read_blob(follower: SO101Follower, min_area: float, tries: int = 10):
    """数フレーム読んで、青い領域が見つかった最後の結果を返す。"""
    blob = None
    for _ in range(tries):
        blob = find_blue(follower.get_observation()["wrist"], min_area)
        time.sleep(0.05)
    return blob


def main() -> int:
    args = parse_args()
    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=args.camera, width=CAM_WIDTH, height=CAM_HEIGHT, fps=30
            )
        },
        max_relative_target=args.max_relative_target,
    )
    follower = SO101Follower(config)
    follower.connect(calibrate=False)
    try:
        start = follower.get_observation()
        limits = safe_limits(follower, "elbow_flex", 1.0)
        home = {f"{m}.pos": start[f"{m}.pos"] for m in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}
        elbow0 = home["elbow_flex.pos"]
        before = read_blob(follower, args.min_area)
        if before is None:
            print("青い缶が見えません。位置を合わせてから再実行してください。", flush=True)
            return 1
        print(f"開始: elbow {elbow0:+.1f}  缶 画素 ({before[0]:.0f}, {before[1]:.0f})", flush=True)

        up = dict(home)
        up["elbow_flex.pos"] = clamp(elbow0 + args.step, *limits)
        move_smoothly(follower, up, args.speed, timeout_s=30.0)
        after = read_blob(follower, args.min_area)
        if after is None:
            print(f"elbow {up['elbow_flex.pos']:+.1f}: 缶が見えなくなりました", flush=True)
        else:
            dx = after[0] - before[0]
            dy = after[1] - before[1]
            print(
                f"elbow {up['elbow_flex.pos']:+.1f}: 缶 画素 ({after[0]:.0f}, {after[1]:.0f})"
                f"  変化 dx {dx:+.0f}, dy {dy:+.0f}",
                flush=True,
            )

        move_smoothly(follower, home, args.speed, timeout_s=30.0)
        print("元の位置へ戻しました。", flush=True)
    except KeyboardInterrupt:
        print("\n停止します。", flush=True)
    finally:
        try:
            follower.disconnect()
        except Exception as error:
            print(f"切断中にエラー: {error}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
