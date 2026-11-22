#!/usr/bin/env python3
"""Kinect の座標とアームの土台の座標を合わせる。チェッカーボードの代わりに缶を使う。

手順 (ボードと 12 V 電源をフォロワーにつなぐ):
  1. 机の上から缶をどけ、アームを待機姿勢にして Enter。今ある物を「背景」として覚える。
  2. 缶を手が届く所に置いて Enter。Kinect が缶の位置を測る。
  3. トルクの切れたアームを手で動かし、開いた爪の真ん中に缶の中心が来るように、掴む姿勢で止めて Enter。
     缶は動かさない。関節角度から爪の位置を計算する。
  4. アームを缶から離し、缶を別の場所へ動かして 2 へ。5 か所以上、手前・奥・左右に散らす。
  5. q で終わると、Kinect 座標 → 土台座標の回転と平行移動を求め、calibration/kinect_to_base.json に保存する。

掴んだ姿勢 (爪の向きと高さ) も一緒に保存し、kinect_grasp.py がその掴み方を真似する。

  uv run python scripts/kinect_calib.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot_camera_kinect import KinectCamera, KinectCameraConfig

import kinect_can as kc
import so101_kin as kin
from swap_demo import check_arm

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "calibration/kinect_to_base.json"
CAN_HEIGHT = (0.08, 0.22)  # 缶とみなす高さ (m)
CAN_WIDTH = (0.04, 0.13)


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dst ≈ R @ src + t となる回転 R と平行移動 t (Kabsch)。"""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    s = np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ s @ u.T
    return r, cd - r @ cs


def grab_frame(cam: KinectCamera, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """n 枚の深度の中央値で、ちらつきを抑える。0 (測定なし) は除いて中央値を取る。"""
    depths = []
    for _ in range(n):
        rgb = cam.async_read(1000)
        depths.append(cam.read_latest_depth()[:, :, 0].astype(np.float32))
    stack = np.stack(depths)
    stack[stack == 0] = np.nan
    depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0)
    return rgb, kc.deproject(depth)


def find_new_can(objs, background, min_shift: float = 0.05):
    """背景になかった物のうち、缶らしい大きさの一番大きいもの。"""
    cands = [
        o for o in objs
        if CAN_HEIGHT[0] <= o.height <= CAN_HEIGHT[1] and CAN_WIDTH[0] <= o.width <= CAN_WIDTH[1]
        and all(np.linalg.norm(o.base - b.base) > min_shift for b in background)
    ]
    return cands[0] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/so101-leader", help="ボードのポート (今使える 1 枚は leader の名前)")
    ap.add_argument("--id", default="follower")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--refit", action="store_true", help="アームにつながず、保存済みの点から計算し直す")
    ap.add_argument("--drop", type=int, nargs="*", default=[], help="--refit で除く点の番号 (1 始まり)")
    args = ap.parse_args()

    if args.refit:
        saved = json.loads(args.out.read_text())
        pairs = [p for i, p in enumerate(saved["pairs"], 1) if i not in args.drop]
        plane = kc.Plane(np.array(saved["plane_cam"]["normal"]), saved["plane_cam"]["d"])
        return solve(pairs, plane, args.out, saved.get("dropped", []) + [saved["pairs"][i - 1] for i in args.drop])

    check_arm(args.port, "follower")
    cam = KinectCamera(KinectCameraConfig(use_depth=True))
    cam.connect()
    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    follower.connect(calibrate=False)
    follower.bus.disable_torque()
    shots = REPO / "outputs/kinect/calib"
    shots.mkdir(parents=True, exist_ok=True)
    pairs = []
    try:
        print("トルクを切りました。アームは手で支えてください。")
        input("1. 机の上から缶をどけ、アームを待機姿勢にして Enter: ")
        _, pts = grab_frame(cam)
        plane = kc.fit_table_plane(pts)
        background = kc.find_objects(pts, plane)
        print(f"   机の平面を求めました (カメラから {plane.d * 100:.1f} cm)。背景の物 {len(background)} 個。")

        while True:
            k = len(pairs)
            ans = input(f"2. [{k + 1} 点目] 缶を置き、手を Kinect の視野から出して Enter (q で終わる): ").strip().lower()
            if ans == "q":
                break
            rgb, pts = grab_frame(cam)
            objs = kc.find_objects(pts, plane)
            can = find_new_can(objs, background)
            cv2.imwrite(str(shots / f"{k:02d}.png"), kc.annotate(rgb, [can] if can else objs))
            if can is None:
                print("   缶が見つかりません。見える所に置き直してください。見えた物:")
                for o in objs:
                    print(f"     高さ {o.height * 100:.1f} cm, 幅 {o.width * 100:.1f} cm")
                continue
            print(f"   缶: 高さ {can.height * 100:.1f} cm, 幅 {can.width * 100:.1f} cm")
            input("3. 爪の真ん中に缶の中心が来るように掴む姿勢をとって Enter: ")
            obs = follower.get_observation()
            tcp = kin.fk_from_obs(obs)
            p_base = tcp[:3, 3]
            # 爪の高さ = 土台の底 (≒ 机) からの高さ。その高さの缶の軸上の点を、カメラ側の対応点にする
            p_cam = can.base + plane.normal * p_base[2]
            pairs.append({
                "cam": p_cam.tolist(),
                "base": p_base.tolist(),
                "can_base_cam": can.base.tolist(),
                "can_height": can.height,
                "pitch_deg": kin.pitch_of(tcp),
                "joints": {m: obs[f"{m}.pos"] for m in kin.ARM + ["gripper"]},
            })
            print(f"   爪: ({p_base[0]:+.3f}, {p_base[1]:+.3f}, {p_base[2]:.3f}) m, 下向き {kin.pitch_of(tcp):.0f}°,"
                  f" gripper {obs['gripper.pos']:.0f}")
            input("4. アームを缶から離して待機姿勢に戻し、Enter: ")
    finally:
        cam.disconnect()
        follower.disconnect()
    return solve(pairs, plane, args.out, [])


def solve(pairs: list[dict], plane: kc.Plane, out: Path, dropped: list[dict]) -> int:
    if len(pairs) < 4:
        print(f"{len(pairs)} 点しかないので計算しません (4 点以上必要)。")
        return 1
    src = np.array([p["cam"] for p in pairs])
    dst = np.array([p["base"] for p in pairs])
    r, t = rigid_fit(src, dst)
    err = np.linalg.norm(src @ r.T + t - dst, axis=1)
    up = r @ plane.normal
    print("各点の残差 (cm):", " ".join(f"{i}:{e * 100:.1f}" for i, e in enumerate(err, 1)))
    print(f"残差の平均 {err.mean() * 100:.1f} cm, 最大 {err.max() * 100:.1f} cm")
    print(f"机の上向きを土台座標で見ると ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f})  (理想は 0, 0, +1)")
    if up[2] < 0.9:
        print("机の上向きが合っていません。缶の代わりに手などを拾った点がないか、outputs/kinect/calib/ の画像で確かめ、"
              "--refit --drop <番号> で除いてください。")
    out.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "R": r.tolist(),
        "t": t.tolist(),
        "residual_m": err.tolist(),
        "plane_cam": {"normal": plane.normal.tolist(), "d": plane.d},
        "pairs": pairs,
        "dropped": dropped,
    }, indent=2, ensure_ascii=False))
    print(f"保存しました: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
