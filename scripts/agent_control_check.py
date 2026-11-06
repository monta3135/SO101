#!/usr/bin/env python3
"""フォロワーを小さく動かし、カメラの前後画像と関節角度で制御を確認する。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

DEFAULT_CAMERA = Path(
    "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="フォロワーの制御とカメラを確認する")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--pan-delta", type=float, default=3.0)
    parser.add_argument("--gripper-delta", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/agent_check"))
    args = parser.parse_args()

    if not args.camera.exists():
        print(f"カメラが見つかりません: {args.camera}", flush=True)
        print("USB カメラを接続し、/dev/v4l/by-id の名前を確認してください。", flush=True)
        return 1

    camera_config = OpenCVCameraConfig(
        index_or_path=args.camera,
        width=640,
        height=480,
        fps=30,
    )
    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        cameras={"wrist": camera_config},
        max_relative_target={
            "shoulder_pan": args.pan_delta,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": args.gripper_delta,
        },
    )

    follower = SO101Follower(config)
    follower.connect(calibrate=False)

    try:
        print("現在の状態を読みます。", flush=True)
        before = follower.get_observation()
        before_path = args.output / "before.png"
        save_image(before_path, before["wrist"])

        print(
            f"ベースを {args.pan_delta} 度、グリッパーを {args.gripper_delta} 開きます。",
            flush=True,
        )
        time.sleep(1.0)

        action = {k: v for k, v in before.items() if k.endswith(".pos")}
        action["shoulder_pan.pos"] += args.pan_delta
        action["gripper.pos"] += args.gripper_delta
        sent = follower.send_action(action)
        time.sleep(1.5)

        after = follower.get_observation()
        after_path = args.output / "after.png"
        save_image(after_path, after["wrist"])

        # 元の姿勢に戻す
        follower.send_action({k: v for k, v in before.items() if k.endswith(".pos")})
        time.sleep(1.0)
        final = follower.get_observation()

        print("\n== 関節角度 ==", flush=True)
        print(f"{'関節':<20} {'前':>8} {'指令':>8} {'後':>8} {'戻し':>8}", flush=True)
        for key in action:
            name = key.removesuffix(".pos")
            print(
                f"{name:<20} {before[key]:>8.1f} {sent[key]:>8.1f} "
                f"{after[key]:>8.1f} {final[key]:>8.1f}",
                flush=True,
            )

        diff = np.abs(after["wrist"].astype(np.int16) - before["wrist"].astype(np.int16))
        print(f"\nカメラ画像の平均差分: {diff.mean():.2f}", flush=True)
        print(f"前画像: {before_path}", flush=True)
        print(f"後画像: {after_path}", flush=True)

        moved = (
            abs(after["shoulder_pan.pos"] - before["shoulder_pan.pos"]) > 0.5
            or abs(after["gripper.pos"] - before["gripper.pos"]) > 1.0
        )
        if not moved:
            print("関節の変化を確認できませんでした。", flush=True)
            return 2

        print("\n制御とカメラの確認が完了しました。", flush=True)
        return 0
    finally:
        follower.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
