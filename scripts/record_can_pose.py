#!/usr/bin/env python3
"""トルクを切ったアームを手で動かし、青い缶が手首カメラに映った関節角度を記録する。

モーターには指令を送らない（トルク OFF のまま読むだけ）。缶が見えた瞬間の角度を
--out の JSON に追記し、画面にも表示する。停止は Ctrl+C。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from track_face import CAM_HEIGHT, CAM_WIDTH, DEFAULT_CAMERA
from find_blue_can import find_blue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="青い缶が見えた関節角度を記録する")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--min-area", type=float, default=1500.0)
    parser.add_argument("--hz", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/can_pose.json"))
    return parser.parse_args()


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
    )
    follower = SO101Follower(config)
    follower.connect(calibrate=False)
    records: list[dict] = []
    try:
        # 手で動かせるように、モーターのトルクを切る
        follower.bus.disable_torque()
        print("トルクを切りました。手で動かしてください。Ctrl+C で終了。", flush=True)
        period = 1.0 / args.hz
        while True:
            loop_start = time.perf_counter()
            observation = follower.get_observation()
            blob = find_blue(observation["wrist"], args.min_area)
            if blob is not None:
                pose = {k.removesuffix(".pos"): round(v, 1) for k, v in observation.items() if k.endswith(".pos")}
                records.append({"pixel": [round(blob[0]), round(blob[1])], "area": round(blob[2]), "pose": pose})
                print(
                    f"缶を検出 画素 ({blob[0]:.0f}, {blob[1]:.0f}) 面積 {blob[2]:.0f}  "
                    + "  ".join(f"{k} {v:+.1f}" for k, v in pose.items()),
                    flush=True,
                )
            wait = period - (time.perf_counter() - loop_start)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\n停止します。", flush=True)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"{len(records)} 件を保存しました: {args.out}", flush=True)
        try:
            follower.disconnect()
        except Exception as error:
            print(f"切断中にエラー: {error}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
