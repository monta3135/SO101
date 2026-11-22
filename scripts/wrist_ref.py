#!/usr/bin/env python3
"""手で掴ませた姿勢で、手首カメラに写るキャップの位置を測り、掴むときの目標として保存する。

トルクを切ったフォロワーで缶を「ちょうど掴みたい高さ」で挟む姿勢をとり、Enter を押す。
関節角とキャップの画素を calibration/wrist_grasp_ref.json に保存し、kinect_grasp.py は
近づきながら、キャップがこの位置に来た所で爪を閉じる。トルクは入れない。

  uv run python scripts/wrist_ref.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import so101_kin as kin
import wrist_cap
from swap_demo import check_arm
from track_face import DEFAULT_CAMERA

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "calibration/wrist_grasp_ref.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/so101-leader")
    ap.add_argument("--id", default="follower")
    args = ap.parse_args()

    check_arm(args.port, "follower")
    probe = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    probe.bus.connect()
    try:
        probe.bus.disable_torque()
        print("トルクを切りました。缶を掴みたい高さで爪に挟み、手で支えたまま Enter: ", end="", flush=True)
        input()
        pos = probe.bus.sync_read("Present_Position")
    finally:
        probe.bus.disconnect(disable_torque=True)

    subprocess.run([sys.executable, str(REPO / "scripts/lock_camera.py"), "--camera", str(DEFAULT_CAMERA),
                    "--load", str(REPO / "calibration/wrist_camera_lock.json")], check=True, capture_output=True)
    cam = cv2.VideoCapture(str(DEFAULT_CAMERA), cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Kinect と同じ USB でも入る形式
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):
        ok, img = cam.read()
    cam.release()
    if not ok:
        print("手首カメラから画像が来ません。")
        return 1
    cap = wrist_cap.find_cap(img)
    out_img = REPO / "outputs/kinect/wrist_ref.png"
    cv2.imwrite(str(out_img), wrist_cap.draw(img, cap, cap[:2] if cap else wrist_cap.GRASP_TARGET))
    if cap is None:
        print(f"キャップが見えません ({out_img})。")
        return 1
    tcp = kin.fk([pos[m] for m in kin.ARM])
    ref = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cap_px": [cap[0], cap[1]],
        "cap_area": cap[2],
        "tcp_base": tcp[:3, 3].tolist(),
        "pitch_deg": kin.pitch_of(tcp),
        "joints": {k: v for k, v in pos.items()},
    }
    OUT.write_text(json.dumps(ref, indent=2, ensure_ascii=False))
    print(f"キャップ ({cap[0]:.0f}, {cap[1]:.0f}) 面積 {cap[2]:.0f}, 爪の高さ {tcp[2, 3] * 100:.1f} cm,"
          f" 下向き {kin.pitch_of(tcp):.0f}°, gripper {pos['gripper']:.0f}")
    print(f"保存しました: {OUT}  (画像: {out_img})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
