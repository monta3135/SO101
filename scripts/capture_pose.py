#!/usr/bin/env python3
"""指定した絶対関節角度へ移動し、カメラで撮影して顔を検出する。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
DEFAULT_CAMERA = Path(
    "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="指定姿勢で撮影し顔を検出する")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output", type=Path, default=Path("outputs/capture"))
    parser.add_argument("--name", default="pose.png")
    parser.add_argument("--pan", type=float, help="shoulder_pan の絶対角度")
    parser.add_argument("--lift", type=float, help="shoulder_lift の絶対角度")
    parser.add_argument("--elbow", type=float, help="elbow_flex の絶対角度")
    parser.add_argument("--wrist", type=float, help="wrist_flex の絶対角度")
    parser.add_argument("--roll", type=float, help="wrist_roll の絶対角度")
    parser.add_argument("--gripper", type=float, help="gripper の絶対角度")
    parser.add_argument("--max-step", type=float, default=4.0)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/so101/face_detection_yunet_2023mar.onnx",
    )
    return parser.parse_args()


def load_detector(model: Path) -> cv2.FaceDetectorYN:
    if not model.exists():
        model.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
        print(f"顔検出モデルを取得します: {url}", flush=True)
        urllib.request.urlretrieve(url, model)
    return cv2.FaceDetectorYN.create(str(model), "", (640, 480), 0.6, 0.3, 5000)


def safe_limits(follower: SO101Follower, motor: str, margin: float) -> tuple[float, float]:
    calibration = follower.bus.calibration[motor]
    middle = (calibration.range_min + calibration.range_max) / 2
    low = (calibration.range_min - middle) * 360 / 4095
    high = (calibration.range_max - middle) * 360 / 4095
    return low + margin, high - margin


def move_to(follower: SO101Follower, target: dict[str, float], max_step: float) -> None:
    action = dict(target)
    for _ in range(200):
        sent = follower.send_action(action)
        error = max(abs(sent[key] - target[key]) for key in target)
        if error < 0.5:
            break
        time.sleep(0.02)


def main() -> int:
    args = parse_args()
    if not args.camera.exists():
        print(f"カメラが見つかりません: {args.camera}", flush=True)
        return 1

    optional = {
        "shoulder_pan": args.pan,
        "shoulder_lift": args.lift,
        "elbow_flex": args.elbow,
        "wrist_flex": args.wrist,
        "wrist_roll": args.roll,
        "gripper": args.gripper,
    }
    if all(v is None for v in optional.values()):
        print("角度の指定がありません。--pan などで1つ以上指定してください。", flush=True)
        return 1

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=args.camera, width=640, height=480, fps=30
            )
        },
        max_relative_target=args.max_step,
    )
    follower = SO101Follower(config)
    follower.connect(calibrate=False)
    detector = load_detector(args.model)
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        start = follower.get_observation()
        target = {f"{m}.pos": start[f"{m}.pos"] for m in MOTORS}
        limits = {m: safe_limits(follower, m, 1.0) for m in MOTORS}
        for motor, value in optional.items():
            if value is None:
                continue
            target[f"{motor}.pos"] = float(
                np.clip(value, *limits[motor])
            )
            print(
                f"{motor}: {start[f'{motor}.pos']:.1f} -> {target[f'{motor}.pos']:.1f}",
                flush=True,
            )

        move_to(follower, target, args.max_step)
        time.sleep(0.5)

        observation = follower.get_observation()
        image = observation["wrist"]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(bgr)
        boxes = [] if faces is None else faces[:, :4].astype(int).tolist()

        path = args.output / args.name
        Image.fromarray(image).save(path)
        print(f"画像: {path}", flush=True)
        print(f"検出した顔: {len(boxes)} 個  boxes={boxes}", flush=True)
        if boxes:
            annotated = Image.fromarray(image).copy()
            draw = ImageDraw.Draw(annotated)
            for x, y, w, h in boxes:
                draw.rectangle((x, y, x + w, y + h), outline="lime", width=4)
            annotated_path = args.output / f"annotated_{args.name}"
            annotated.save(annotated_path)
            print(f"注釈付き画像: {annotated_path}", flush=True)
        return 0
    finally:
        follower.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
