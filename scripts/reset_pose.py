#!/usr/bin/env python3
"""フォロワーを待機姿勢へゆっくり戻す。--pan などの引数で上書きもできる。"""

from __future__ import annotations

import argparse
import time

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# docs/software.md の待機姿勢。未指定の関節（wrist_flex / wrist_roll / gripper）は現在位置のまま
REST_POSE = {
    "shoulder_pan": -5.0,
    "shoulder_lift": -111.0,
    "elbow_flex": 96.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="待機姿勢へゆっくり移動する")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--pan", type=float, help="shoulder_pan の上書き角度")
    parser.add_argument("--lift", type=float, help="shoulder_lift の上書き角度")
    parser.add_argument("--elbow", type=float, help="elbow_flex の上書き角度")
    parser.add_argument("--wrist", type=float, help="wrist_flex の上書き角度")
    parser.add_argument("--roll", type=float, help="wrist_roll の上書き角度")
    parser.add_argument("--gripper", type=float, help="gripper の上書き角度")
    parser.add_argument("--max-step", type=float, default=1.0, help="1回の送信で動く最大角度 [deg]")
    return parser.parse_args()


def safe_limits(follower: SO101Follower, motor: str, margin: float) -> tuple[float, float]:
    calibration = follower.bus.calibration[motor]
    middle = (calibration.range_min + calibration.range_max) / 2
    low = (calibration.range_min - middle) * 360 / 4095
    high = (calibration.range_max - middle) * 360 / 4095
    return low + margin, high - margin


def move_to(follower: SO101Follower, target: dict[str, float], max_step: float) -> None:
    action = dict(target)
    for _ in range(400):
        sent = follower.send_action(action)
        error = max(abs(sent[key] - target[key]) for key in target)
        if error < 0.5:
            break
        time.sleep(0.02)


def main() -> int:
    args = parse_args()
    overrides = {
        "shoulder_pan": args.pan,
        "shoulder_lift": args.lift,
        "elbow_flex": args.elbow,
        "wrist_flex": args.wrist,
        "wrist_roll": args.roll,
        "gripper": args.gripper,
    }

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        max_relative_target=args.max_step,
    )
    follower = SO101Follower(config)
    follower.connect(calibrate=False)

    try:
        start = follower.get_observation()
        limits = {m: safe_limits(follower, m, 1.0) for m in MOTORS}
        target: dict[str, float] = {}
        for motor in MOTORS:
            value = overrides[motor] if overrides[motor] is not None else REST_POSE.get(motor)
            if value is None:
                continue
            angle = float(np.clip(value, *limits[motor]))
            target[f"{motor}.pos"] = angle
            print(
                f"{motor}: {start[f'{motor}.pos']:.1f} -> {angle:.1f}",
                flush=True,
            )
        if not target:
            print("移動する関節がありません。")
            return 1

        move_to(follower, target, args.max_step)
        time.sleep(0.5)
        print("移動が完了しました。")
        return 0
    finally:
        follower.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
