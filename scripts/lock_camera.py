#!/usr/bin/env python3
"""手首カメラの自動露出・オートホワイトバランス・オートフォーカスを切り、値を固定する。

記録とポリシー実行で画の色や明るさが変わらないようにするためのもの。
初回は自動調整が落ち着いた値でそのまま固定し、--save に保存する。
2回目以降は --load で保存した値を書き戻すと、記録時と同じ設定になる。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from track_face import CAM_HEIGHT, CAM_WIDTH, DEFAULT_CAMERA

# V4L2 の自動露出は 1 が手動、3 が自動（絞り優先）。
AUTO_EXPOSURE_MANUAL = 1
LOCKED = {
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "focus": cv2.CAP_PROP_FOCUS,
    "gain": cv2.CAP_PROP_GAIN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手首カメラの自動調整を切って値を固定する")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--settle", type=float, default=3.0, help="自動調整を待つ秒数")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", type=Path, help="固定した値を保存する JSON")
    group.add_argument("--load", type=Path, help="この JSON の値を書き戻す")
    return parser.parse_args()


def read_values(cap: cv2.VideoCapture) -> dict[str, float]:
    return {name: cap.get(prop) for name, prop in LOCKED.items()}


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.camera), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"{args.camera} を開けません。")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if args.load is None:
        # 自動調整を有効にしたまま数秒流し、今の照明に合った値へ落ち着かせる。
        end = time.monotonic() + args.settle
        while time.monotonic() < end:
            cap.read()
        values = read_values(cap)
    else:
        values = json.loads(args.load.read_text())

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    for name, prop in LOCKED.items():
        cap.set(prop, values[name])
    cap.read()

    actual = read_values(cap)
    print("固定した値:")
    for name in LOCKED:
        mark = "" if abs(actual[name] - values[name]) < 1e-6 else f"  (指定 {values[name]} が反映されていません)"
        print(f"  {name}: {actual[name]}{mark}")
    print(
        f"  auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)} "
        f"auto_wb={cap.get(cv2.CAP_PROP_AUTO_WB)} autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS)}"
    )
    cap.release()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"{args.save} に保存しました。")


if __name__ == "__main__":
    main()
