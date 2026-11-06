#!/usr/bin/env python3
"""アームを少しずつ動かしてカメラ画像を撮り、顔が写る向きを探す。"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

DEFAULT_CAMERA = Path(
    "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
STS3215_RESOLUTION = 4095


def font(size: int):
    return ImageFont.truetype(FONT_PATH, size, index=0)


def parse_offsets(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def safe_limits(follower: SO101Follower, motor: str, margin: float) -> tuple[float, float]:
    calibration = follower.bus.calibration[motor]
    middle = (calibration.range_min + calibration.range_max) / 2
    low = (calibration.range_min - middle) * 360 / STS3215_RESOLUTION
    high = (calibration.range_max - middle) * 360 / STS3215_RESOLUTION
    return low + margin, high - margin


def move_to(follower: SO101Follower, target: dict[str, float], max_step: float) -> dict[str, float]:
    action = dict(target)
    sent = dict(target)
    for _ in range(200):
        sent = follower.send_action(action)
        error = max(abs(sent[key] - target[key]) for key in target)
        if error < 0.5:
            break
        time.sleep(0.02)
    return sent


def detect_faces(
    image: np.ndarray, detector: cv2.FaceDetectorYN
) -> list[tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    boxes = []
    for face in faces:
        x, y, w, h = (int(round(v)) for v in face[:4])
        boxes.append((x, y, w, h))
    return boxes


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


def make_contact_sheet(records: list[dict], output: Path) -> None:
    cols = 4
    thumb_w, thumb_h = 320, 240
    label_h = 38
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    small = font(16)

    for index, record in enumerate(records):
        image = Image.open(record["path"]).resize((thumb_w, thumb_h))
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        label = (
            f"pan {record['pan']:+.0f} lift {record['lift']:+.0f} "
            f"elbow {record['elbow']:+.0f} wrist {record['wrist']:+.0f} "
            f"roll {record.get('roll', 0):+.0f} face {record['faces']}"
        )
        draw.text((x + 6, y + thumb_h + 8), label, font=small, fill="black")

    sheet.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="カメラで顔を探す")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output", type=Path, default=Path("outputs/face_scan"))
    parser.add_argument("--pan", default="-30,0,30", help="ベースの角度オフセット")
    parser.add_argument("--lift", default="0,15", help="肩の角度オフセット")
    parser.add_argument("--elbow", default="0,-15", help="肘の角度オフセット")
    parser.add_argument("--wrist", default="0,30", help="手首の角度オフセット")
    parser.add_argument("--roll", default="0", help="手首ロールの角度オフセット")
    parser.add_argument("--margin", type=float, default=2.0, help="可動域端からの余白")
    parser.add_argument("--max-step", type=float, default=4.0)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/so101/face_detection_yunet_2023mar.onnx",
    )
    args = parser.parse_args()

    if not args.camera.exists():
        print(f"カメラが見つかりません: {args.camera}", flush=True)
        return 1

    pan_offsets = parse_offsets(args.pan)
    lift_offsets = parse_offsets(args.lift)
    elbow_offsets = parse_offsets(args.elbow)
    wrist_offsets = parse_offsets(args.wrist)
    roll_offsets = parse_offsets(args.roll)

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=args.camera,
                width=640,
                height=480,
                fps=30,
            )
        },
        max_relative_target=args.max_step,
    )

    follower = SO101Follower(config)
    follower.connect(calibrate=False)

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    detector = load_detector(args.model)

    try:
        start = follower.get_observation()
        base = {f"{motor}.pos": start[f"{motor}.pos"] for motor in MOTORS}
        limits = {
            motor: safe_limits(follower, motor, args.margin) for motor in MOTORS
        }
        print(
            f"開始姿勢: pan {base['shoulder_pan.pos']:.1f}, "
            f"lift {base['shoulder_lift.pos']:.1f}, "
            f"elbow {base['elbow_flex.pos']:.1f}, "
            f"wrist {base['wrist_flex.pos']:.1f}",
            flush=True,
        )
        for motor, (low, high) in limits.items():
            print(
                f"  {motor}: 可動域 {low:.1f} から {high:.1f} 度",
                flush=True,
            )

        combos = itertools.product(
            pan_offsets, lift_offsets, elbow_offsets, wrist_offsets, roll_offsets
        )
        for pan, lift, elbow, wrist, roll in combos:
            target = dict(base)
            target["shoulder_pan.pos"] = float(
                np.clip(base["shoulder_pan.pos"] + pan, *limits["shoulder_pan"])
            )
            target["shoulder_lift.pos"] = float(
                np.clip(base["shoulder_lift.pos"] + lift, *limits["shoulder_lift"])
            )
            target["elbow_flex.pos"] = float(
                np.clip(base["elbow_flex.pos"] + elbow, *limits["elbow_flex"])
            )
            target["wrist_flex.pos"] = float(
                np.clip(base["wrist_flex.pos"] + wrist, *limits["wrist_flex"])
            )
            target["wrist_roll.pos"] = float(
                np.clip(base["wrist_roll.pos"] + roll, *limits["wrist_roll"])
            )

            move_to(follower, target, args.max_step)
            time.sleep(0.35)

            observation = follower.get_observation()
            image = observation["wrist"]
            faces = detect_faces(image, detector)

            name = (
                f"pan_{pan:+.0f}_lift_{lift:+.0f}_"
                f"elbow_{elbow:+.0f}_wrist_{wrist:+.0f}_roll_{roll:+.0f}.png"
            )
            path = args.output / name
            Image.fromarray(image).save(path)

            if faces:
                annotated = Image.fromarray(image).copy()
                draw = ImageDraw.Draw(annotated)
                for x, y, w, h in faces:
                    draw.rectangle((x, y, x + w, y + h), outline="lime", width=4)
                annotated.save(args.output / f"annotated_{name}")

            record = {
                "pan": float(pan),
                "lift": float(lift),
                "elbow": float(elbow),
                "wrist": float(wrist),
                "roll": float(roll),
                "faces": len(faces),
                "boxes": faces,
                "path": str(path),
            }
            records.append(record)
            print(
                f"pan {pan:+.0f} lift {lift:+.0f} "
                f"elbow {elbow:+.0f} wrist {wrist:+.0f} roll {roll:+.0f}: "
                f"顔 {len(faces)} 個",
                flush=True,
            )

        move_to(follower, base, args.max_step)
        time.sleep(0.5)

        (args.output / "results.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        )
        (args.output / "contact_sheet.png").parent.mkdir(parents=True, exist_ok=True)
        make_contact_sheet(records, args.output / "contact_sheet.png")

        found = [r for r in records if r["faces"] > 0]
        print("\n== 結果 ==", flush=True)
        if found:
            print(f"顔を検出した向き: {len(found)} 件", flush=True)
            for record in found:
                print(
                    f"  pan {record['pan']:+.0f}, lift {record['lift']:+.0f}, "
                    f"elbow {record['elbow']:+.0f}, wrist {record['wrist']:+.0f}, "
                    f"{record['faces']} 個, {record['path']}",
                    flush=True,
                )
        else:
            print("顔は検出できませんでした。", flush=True)
        print(f"コンタクトシート: {args.output / 'contact_sheet.png'}", flush=True)
        return 0
    finally:
        follower.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
