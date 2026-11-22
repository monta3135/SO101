"""手首カメラの画像から、コーヒー缶のキャップ (オレンジと金) を見つける。

  uv run python scripts/wrist_cap.py 画像.png ...   # 見つけた位置を描いて outputs/kinect/cap_*.png に保存
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# OpenCV の色相は 0〜180。キャップのオレンジと金は 3〜28 あたり、机の茶色は彩度が低い
CAP_HSV_LOW = (3, 90, 35)
CAP_HSV_HIGH = (28, 255, 255)
# 缶の胴の中ほどを横から掴んだ状態 (爪 gripper 24 で閉じた所) での、キャップの中心の画素 (640x480)。
# 2026-11-22 に手で掴ませた姿勢 (下向き 28°) で測った。缶の軸に沿って近づく間は x がほぼ変わらない。
# 参考: ACT のデモで上から首を掴み始めた瞬間は (206, 358) (so101_lift_can_clean, 25 エピソードの中央値)。
# 左の爪が動く側なので、開くほど爪の真ん中は左に寄る
GRASP_TARGET = (258.0, 375.0)
# wrist_ref.py で手で掴ませて測った値があれば、そちらを使う
_REF = Path(__file__).resolve().parent.parent / "calibration/wrist_grasp_ref.json"
REF = json.loads(_REF.read_text()) if _REF.exists() else None
REF_CAN_HEIGHT = 0.167  # 測ったときの缶 (390 ml のボトル缶) の高さ。Kinect で 16.6〜16.9 cm
if REF is not None:
    GRASP_TARGET = tuple(REF["cap_px"])


def find_cap(image_bgr: np.ndarray, min_area: float = 800.0) -> tuple[float, float, float] | None:
    """一番大きいキャップらしい領域の中心 (画素) と面積。見つからなければ None。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CAP_HSV_LOW, CAP_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    (cx, cy), _ = cv2.minEnclosingCircle(contour)
    m = cv2.moments(contour)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    return float(cx), float(cy), float(area)


def draw(image_bgr: np.ndarray, cap, target=GRASP_TARGET) -> np.ndarray:
    img = image_bgr.copy()
    cv2.drawMarker(img, (int(target[0]), int(target[1])), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
    if cap is not None:
        cv2.circle(img, (int(cap[0]), int(cap[1])), 8, (0, 0, 255), -1)
        cv2.line(img, (int(cap[0]), int(cap[1])), (int(target[0]), int(cap[1])), (0, 0, 255), 2)
    return img


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "outputs/kinect"
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        cap = find_cap(img)
        print(path, "見つからない" if cap is None else f"中心 ({cap[0]:.0f}, {cap[1]:.0f}) 面積 {cap[2]:.0f}")
        cv2.imwrite(str(out / f"cap_{Path(path).stem}.png"), draw(img, cap))
