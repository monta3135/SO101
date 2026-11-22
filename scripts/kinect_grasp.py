#!/usr/bin/env python3
"""Kinect で缶の位置を測り、逆運動学でそこへ手を伸ばして掴み、持ち上げて元に戻す。

kinect_calib.py で作った calibration/kinect_to_base.json を使う。掴む高さと爪の向きは、
較正のときに手で取った姿勢のうち、缶までの距離が一番近いものを真似する。

  uv run python scripts/kinect_grasp.py --dry-run   # 缶を探して計画を表示するだけ (トルクを入れない)
  uv run python scripts/kinect_grasp.py             # 計画を表示し、Enter で実行

動きは関節ごとに --speed [deg/s] 以下の直線補間。どの区間でも Ctrl-C で止まり、待機姿勢へ戻る。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot_camera_kinect import KinectCamera, KinectCameraConfig

import kinect_can as kc
import so101_kin as kin
import wrist_cap
from kinect_calib import CAN_HEIGHT, CAN_WIDTH, OUT as CALIB, grab_frame
from reset_pose import safe_limits
from swap_demo import REST_POSE, check_arm, move_slowly
from track_face import CAM_HEIGHT, CAM_WIDTH, DEFAULT_CAMERA

REPO = Path(__file__).resolve().parent.parent
WRIST_LOCK = REPO / "calibration/wrist_camera_lock.json"
REACH = (0.15, 0.45)
# Kinect で見る前の姿勢: 爪を真上に向けて土台の上に畳む (掴む点は土台から 9 cm, 高さ 30 cm)
SCAN_POSE = {"shoulder_lift.pos": -110.0, "elbow_flex.pos": 96.0, "wrist_flex.pos": -76.0}  # 土台からこの距離 (m) の範囲の物だけを缶の候補にする


def load_calib(path: Path) -> dict:
    c = json.loads(path.read_text())
    c["R"], c["t"] = np.array(c["R"]), np.array(c["t"])
    return c


def to_base(c: dict, p_cam: np.ndarray) -> np.ndarray:
    return c["R"] @ p_cam + c["t"]


def align_to_calib(pts: np.ndarray, calib_plane: kc.Plane) -> tuple[np.ndarray, float]:
    """今の机の平面が較正のときと同じに見えるよう、点を回して動かす。ずれの角度 (度) も返す。

    Kinect の首 (チルトモーター) が触られて傾くと、机の平面の向きが変わる。首振りは机の向きに
    直交しない回転なので、平面の向きと距離を合わせれば、大部分のずれが消える
    (机に沿った平行移動と、机の法線まわりの回転は平面からは分からないが、首振りではほぼ起きない)。
    """
    now = kc.fit_table_plane(pts)
    a, b = now.normal, calib_plane.normal
    v, cos = np.cross(a, b), float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx / (1 + cos)  # a を b に重ねる最小の回転
    shift = (now.d - calib_plane.d) * b
    valid = pts[..., 2] > 0
    out = pts @ rot.T + shift
    out[~valid] = 0
    return out, float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def find_can(c: dict, objs: list[kc.TableObject]):
    """手の届く範囲にある、缶らしい大きさの物のうち一番大きいもの。土台座標の底の位置も返す。"""
    for o in objs:
        base = to_base(c, o.base)
        r = float(np.hypot(base[0], base[1]))
        if (CAN_HEIGHT[0] <= o.height <= CAN_HEIGHT[1] and CAN_WIDTH[0] <= o.width <= CAN_WIDTH[1]
                and REACH[0] <= r <= REACH[1]):
            return o, base
    return None, None


def plan(c: dict, can_base: np.ndarray, can_height: float, below_top: float, backoff: float, lift: float,
         max_ik_error: float = 0.01):
    """掴む点・その手前・持ち上げた点の関節角を求める。"""
    r_can = np.hypot(can_base[0], can_base[1])
    # 缶に一番近い較正点の掴み方を真似する
    ref = min(c["pairs"], key=lambda p: np.hypot(p["base"][0] - can_base[0], p["base"][1] - can_base[1]))
    pitch = ref["pitch_deg"]
    # 高さは缶の上端から below_top だけ下 (較正で教えた高さだと缶の下の方を掴んでしまった)
    z = can_height - below_top
    if wrist_cap.REF is not None:
        # 手首カメラの目標を測ったときの掴み方 (爪の向きと缶の上端からの深さ) にそろえる
        pitch = wrist_cap.REF["pitch_deg"]
        z = can_height - (wrist_cap.REF_CAN_HEIGHT - wrist_cap.REF["tcp_base"][2])
        ref = {**ref, "pitch_deg": pitch}
    roll = ref["joints"]["wrist_roll"]
    grasp = np.array([can_base[0], can_base[1], z])
    radial = np.array([can_base[0], can_base[1], 0.0]) / r_can
    # 手本の向きで届かなければ、爪を 5° ずつ下へ傾けて届く向きを探す (土台に近い缶は水平寄りでは届かない)。
    # 高さは Kinect、左右は手首カメラの x で決めるので、向きが手本と違っても狙いはずれない
    for tilt in range(0, 30, 5):
        p_deg = pitch + tilt
        # 近づく向き: 土台から缶へ向かう水平方向を、爪の向きだけ下に傾けたもの
        approach = np.cos(np.radians(p_deg)) * radial + np.array([0, 0, -np.sin(np.radians(p_deg))])
        points = {
            "pre": grasp - approach * backoff,
            "grasp": grasp,
            "lift": grasp + np.array([0, 0, lift]),
        }
        seed = np.array([np.degrees(np.arctan2(can_base[1], can_base[0])), *[ref["joints"][m] for m in kin.ARM[1:]]])
        seed[4] = roll
        joints, report = {}, []
        for name, p in points.items():
            q, pos_err, ang_err = kin.ik(p, seed, pitch_deg=p_deg, roll_deg=roll)
            joints[name] = q
            report.append((name, p, pos_err, ang_err))
            seed = q
        if all(e <= max_ik_error and a < 5 for _, _, e, a in report):
            break
    return joints, report, {**ref, "pitch_deg": p_deg}


def read_joints(port: str, motor_id: str) -> dict[str, float]:
    """関節角 (度) を読むだけ。トルクや設定には触れない。"""
    probe = SO101Follower(SO101FollowerConfig(port=port, id=motor_id))
    probe.bus.connect()
    try:
        return probe.bus.sync_read("Present_Position")
    finally:
        probe.bus.disconnect(disable_torque=False)


def to_action(q: np.ndarray, gripper: float) -> dict[str, float]:
    a = {f"{m}.pos": float(v) for m, v in zip(kin.ARM, q)}
    a["gripper.pos"] = float(gripper)
    return a


def close_until_contact(follower: SO101Follower, start: float, squeeze: float, rate: float = 25.0) -> float:
    """爪を少しずつ閉じ、指令と実際の位置がずれた所 (缶に当たった所) から squeeze だけ強く握る。"""
    target = start
    while target > 0:
        target = max(0.0, target - rate * 0.02)
        follower.send_action({"gripper.pos": target})
        time.sleep(0.02)
        present = follower.bus.sync_read("Present_Position")["gripper"]
        if present - target > 8:
            hold = max(0.0, present - squeeze)
            follower.send_action({"gripper.pos": hold})
            print(f"   缶に当たりました (gripper {present:.0f})。{hold:.0f} で握ります。")
            time.sleep(0.5)
            return present
    print("   最後まで閉じても当たりませんでした (掴めていない)。")
    return 0.0


def wrist_frame(follower: SO101Follower) -> np.ndarray:
    for _ in range(4):  # 動いている間の古いフレームを捨てる
        img = follower.cameras["wrist"].async_read(1000)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def center_pan(follower: SO101Follower, q_pre: np.ndarray, open_g: float, args, out: Path) -> float | None:
    """缶の手前で、キャップが目標の x (爪の真ん中) に来るまで shoulder_pan だけを回す。直した量 [deg] を返す。

    pan を正に回すと腕は右 (土台座標の -y) へ動き、キャップは画面の左へ動く (実測 -14〜-18 px/deg)。
    """
    tx = wrist_cap.GRASP_TARGET[0]
    px_per_deg, dpan, prev = -15.0, 0.0, None
    for k in range(8):
        time.sleep(0.4)
        img = wrist_frame(follower)
        cap = wrist_cap.find_cap(img)
        cv2.imwrite(str(out / f"wrist_center_{k}.png"), wrist_cap.draw(img, cap))
        if cap is None:
            print("   手首カメラにキャップが見えません。止めます。")
            return None
        if prev is not None and abs(dpan - prev[1]) >= 1.5:
            measured = (cap[0] - prev[0]) / (dpan - prev[1])
            if measured < -5:
                px_per_deg = measured
        err = cap[0] - tx
        print(f"   [{k}] キャップ x={cap[0]:.0f} (目標 {tx:.0f}, ずれ {err:+.0f} px), pan の補正 {dpan:+.1f}°")
        if abs(err) <= args.wrist_tol:
            return dpan
        step = float(np.clip(-0.8 * err / px_per_deg, -4.0, 4.0))
        if abs(dpan + step) > args.max_correction:
            print(f"   補正が {args.max_correction}° を超えるので止めます。")
            return None
        prev = (cap[0], dpan)
        dpan += step
        q = q_pre.copy()
        q[0] += dpan
        move_slowly(follower, to_action(q, open_g), args.speed / 3)
    print("   8 回で収まりませんでした。止めます。")
    return None


def servo_to_cap(follower: SO101Follower, pre: np.ndarray, grasp: np.ndarray, q_pre: np.ndarray, ref: dict,
                 open_g: float, args, out: Path):
    """缶の手前から、手首カメラのキャップが「手で掴ませたときの見え方」になるまで寄せる。

    動かすのは 3 つ: shoulder_pan の補正 (deg)、手前→掴む点の直線に沿った進み (m)、上下 (m)。
    合わせるのは 3 つ: キャップの中心の x, y (px) と、大きさ (面積の比の対数 x100)。
    Kinect の位置は数 cm ずれ、見え方と位置の関係も単純ではないので、最初にそれぞれを少しずつ
    動かして効き (ヤコビアン) を実測し、そのあとはそのまま使う。
    戻り値: (関節角, 掴む点 (土台座標, pan の補正前), pan の補正量) / 失敗なら None
    """
    tx, ty = wrist_cap.GRASP_TARGET
    ref_area = wrist_cap.REF["cap_area"] if wrist_cap.REF else None
    u = (grasp - pre) / np.linalg.norm(grasp - pre)
    lo = np.array([-args.max_correction, -0.06, -0.06])
    hi = np.array([args.max_correction, float(np.linalg.norm(grasp - pre)) + 0.04, 0.06])
    max_step = np.array([3.0, 0.015, 0.015])
    q_raw = q_pre.copy()

    def pose(v):
        nonlocal q_raw
        p = pre + u * v[1] + np.array([0, 0, v[2]])
        q_raw, err, _ = kin.ik(p, q_raw, pitch_deg=ref["pitch_deg"], roll_deg=q_pre[4])
        if err > args.max_ik_error:
            return None, p
        q = q_raw.copy()
        q[0] += v[0]
        return q, p

    k = 0

    def look(v):
        nonlocal k
        time.sleep(0.4)
        img = wrist_frame(follower)
        cap = wrist_cap.find_cap(img)
        cv2.imwrite(str(out / f"wrist_servo_{k:02d}.png"), wrist_cap.draw(img, cap))
        if cap is None:
            print(f"   [{k}] キャップが見えません")
            k += 1
            return None
        size = 100 * np.log(cap[2] / ref_area) if ref_area else 0.0
        f = np.array([cap[0] - tx, cap[1] - ty, size])
        print(f"   [{k}] キャップ ({cap[0]:.0f}, {cap[1]:.0f}) 大きさ {size:+.0f}  目標とのずれ"
              f" ({f[0]:+.0f}, {f[1]:+.0f}, {f[2]:+.0f})  pan {v[0]:+.1f}° 進み {v[1] * 100:+.1f} cm 上下 {v[2] * 100:+.1f} cm")
        k += 1
        return f

    def go(v):
        q, p = pose(v)
        if q is None:
            print(f"   ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:.3f}) m に届かないので止めます。")
            return False
        move_slowly(follower, to_action(q, open_g), args.speed / 3)
        return True

    v = np.zeros(3)
    f = look(v)
    if f is None:
        return None
    # 効きを測る: 1 つずつ少し動かす
    jac = np.zeros((3, 3))
    probe = np.array([2.0, 0.015, 0.015])
    for i in range(3):
        v_i = v.copy()
        v_i[i] += probe[i]
        if not go(v_i):
            return None
        f_i = look(v_i)
        if f_i is None:
            return None
        jac[:, i] = (f_i - f) / probe[i]
        v, f = v_i, f_i
    print("   効き: " + ", ".join(f"{n} → ({c[0]:+.0f}, {c[1]:+.0f}, {c[2]:+.0f})"
                                   for n, c in zip(("pan 1°", "進み 1 cm", "上 1 cm"), (jac[:, 0], jac[:, 1] / 100, jac[:, 2] / 100))))
    tol = np.array([args.wrist_tol_near, 25.0, 30.0])
    for _ in range(15):
        if np.all(np.abs(f) <= tol):
            q, p = pose(v)
            return q, p, v[0]
        # 効きは最初に測った値のまま使い、7 割だけ動く (毎回更新すると、大きさの測りぶれで暴れた)
        dv = -0.7 * np.linalg.lstsq(jac, f, rcond=None)[0]
        dv = np.clip(dv, -max_step, max_step)
        v_new = np.clip(v + dv, lo, hi)
        dv = v_new - v
        if np.allclose(dv, 0):
            print("   動ける範囲の端に来たので止めます。")
            return None
        if not go(v_new):
            return None
        f_new = look(v_new)
        if f_new is None:
            # 動きすぎてキャップが画面から外れた。見えていた所へ戻り、歩幅を半分にしてやり直す
            max_step = max_step / 2
            if max_step[1] < 0.003 or not go(v):
                return None
            continue
        df = f_new - f
        # 当たり判定: 動かしたのに、見込みの 2 割も変わらない = 爪が缶を押して一緒に動かしている
        expected = jac @ dv
        if np.linalg.norm(expected) > 30 and np.linalg.norm(df) < 0.2 * np.linalg.norm(expected):
            print("   動かしてもキャップの見え方がほとんど変わりません。缶に当たっているので止めます。")
            return None
        v, f = v_new, f_new
    print("   15 回で収まりませんでした。止めます。")
    return None


def scan(c: dict, skeleton: np.ndarray, args):
    """Kinect で缶を探して、掴む計画を立てて表示する。見つからない・届かないときは None。"""
    cam = KinectCamera(KinectCameraConfig(use_depth=True))
    cam.connect()
    try:
        rgb, pts = grab_frame(cam)
    finally:
        cam.disconnect()
    plane = kc.Plane(np.array(c["plane_cam"]["normal"]), c["plane_cam"]["d"])
    pts, tilt = align_to_calib(pts, plane)
    print(f"Kinect の向きは較正のときから {tilt:.1f}° 変わっています" + ("。机の平面で補正しました。" if tilt > 0.5 else "。"))
    if tilt > 10:
        print("10° を超えて動いているので、較正し直してください (kinect_calib.py)。")
        return None
    # 土台は手で押さえているので、手やアームと缶が画像で重なる。真上から見た位置で分ける
    objs = kc.find_objects_topdown(pts, plane, lambda p: p @ c["R"].T + c["t"], skeleton=skeleton)
    can, can_base = find_can(c, objs)
    out = REPO / "outputs/kinect"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "grasp_target.png"), kc.annotate(rgb, [can] if can else objs))
    if can is None:
        print("手の届く範囲に缶が見つかりません。見えた物 (土台座標):")
        for o in objs:
            b = to_base(c, o.base)
            print(f"  ({b[0]:+.3f}, {b[1]:+.3f}) m, 距離 {np.hypot(b[0], b[1]):.3f} m, 高さ {o.height * 100:.1f} cm,"
                  f" 幅 {o.width * 100:.1f} cm")
        return None
    print(f"缶: 土台から ({can_base[0]:+.3f}, {can_base[1]:+.3f}) m, 距離 {np.hypot(can_base[0], can_base[1]):.3f} m,"
          f" 高さ {can.height * 100:.1f} cm")

    joints, report, ref = plan(c, can_base, can.height, args.below_top, args.backoff, args.lift, args.max_ik_error)
    print(f"爪の向き: 下向き {ref['pitch_deg']:.0f}°, 掴む高さ {report[1][1][2] * 100:.1f} cm")
    ok = True
    for (name, p, pos_err, ang_err), q in zip(report, joints.values()):
        flag = "" if pos_err <= args.max_ik_error and ang_err < 5 else "  ← 届かない"
        ok &= not flag
        print(f"  {name:5s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:.3f}) m  誤差 {pos_err * 100:.1f} cm / {ang_err:.0f}°  "
              + " ".join(f"{v:+6.1f}" for v in q) + flag)
    if not ok:
        print("届かない点があるので動かしません。缶をもう少し土台に近づける/遠ざけるなどしてください。")
        return None
    return can, can_base, joints, report, ref, out



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/so101-leader")
    ap.add_argument("--id", default="follower")
    ap.add_argument("--calib", type=Path, default=CALIB)
    ap.add_argument("--dry-run", action="store_true", help="計画を表示するだけで、アームにつながない")
    ap.add_argument("--speed", type=float, default=30.0, help="関節の最大速度 [deg/s]")
    ap.add_argument("--backoff", type=float, default=0.08, help="掴む点の手前で止まる距離 [m]")
    ap.add_argument("--lift", type=float, default=0.06, help="持ち上げる高さ [m]")
    ap.add_argument("--below-top", type=float, default=0.05, help="掴む高さ: 缶の上端からこれ [m] 下")
    ap.add_argument("--open", type=float, default=None, help="開いた爪の値 (既定は較正のときの値 +25)")
    ap.add_argument("--squeeze", type=float, default=15.0, help="当たった所からさらに閉じる量")
    ap.add_argument("--max-ik-error", type=float, default=0.01, help="逆運動学の誤差がこれ [m] を超えたら動かない")
    ap.add_argument("--snap", action="store_true", help="缶の手前まで行って手首カメラの画像を保存し、戻るだけ")
    ap.add_argument("--no-wrist", action="store_true", help="手首カメラでの補正をしない")
    ap.add_argument("--wrist-tol", type=float, default=12.0, help="缶の手前で、キャップと目標のずれがこれ [px] 以内になるまで合わせる")
    ap.add_argument("--servo3", action="store_true", help="手首カメラで左右・前後・上下の3つを合わせる (既定は左右だけ)")
    ap.add_argument("--final-dz", type=float, default=-0.03, help="--servo3 で見え方を合わせたあと、閉じる前に上下に動かす量 [m]")
    ap.add_argument("--wrist-tol-near", type=float, default=30.0, help="近づき始めてからの横のずれの許容 [px]")
    ap.add_argument("--max-correction", type=float, default=15.0, help="手首カメラで直す pan の上限 [deg]")
    args = ap.parse_args()

    c = load_calib(args.calib)
    if args.dry_run:
        # 今のアームの形を読む (読むだけ)。Kinect の点からアームを除くのに使う
        skeleton = kin.link_points([read_joints(args.port, args.id)[m] for m in kin.ARM])
        return 0 if scan(c, skeleton, args) else 1

    check_arm(args.port, "follower")
    # 手首カメラの露出などを、ACT の記録のときと同じ値に固定する
    subprocess.run([sys.executable, str(REPO / "scripts/lock_camera.py"), "--camera", str(DEFAULT_CAMERA),
                    "--load", str(WRIST_LOCK)], check=True)
    follower = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id,
        # Kinect は挿してあるだけで同じ USB コントローラの帯域を大きく予約するので、無圧縮の YUYV は
        # 入らない。圧縮した MJPG なら Kinect と同時でも 30 fps 出る (2026-11-22 に確認)
        cameras={"wrist": OpenCVCameraConfig(index_or_path=DEFAULT_CAMERA, width=CAM_WIDTH, height=CAM_HEIGHT, fps=30,
                                             fourcc="MJPG")},
    ))
    for attempt in range(3):
        try:
            follower.connect(calibrate=False)
            break
        except ConnectionError as error:
            # バスの返事が1回だけ来ないことがある。開き直してやり直す
            print(f"接続に失敗しました ({error})。やり直します。")
            for part in (follower.bus.disconnect, *(cam.disconnect for cam in follower.cameras.values())):
                try:
                    part()
                except Exception:
                    pass
            if attempt == 2:
                raise
            time.sleep(1.0)
    try:
        input("前と周りを空けて Enter: アームを畳んで爪を上に向け、Kinect で缶を探します (Ctrl-C で中止): ")
        # 土台は手で押さえるので、その手とアームが缶と重ならないよう、アームを土台の真上に畳んでから見る
        present = follower.bus.sync_read("Present_Position")
        move_slowly(follower, {**SCAN_POSE, "shoulder_pan.pos": present["shoulder_pan"],
                               "wrist_roll.pos": present["wrist_roll"], "gripper.pos": 0.0}, args.speed)
        time.sleep(0.5)
        present = follower.bus.sync_read("Present_Position")
        result = scan(c, kin.link_points([present[m] for m in kin.ARM]), args)
        if result is None:
            return 1
        can_base, joints, report, ref, out = result[1:]
        limits = {m: safe_limits(follower, m, 2.0) for m in kin.ARM}
        for name, q in joints.items():
            for m, v in zip(kin.ARM, q):
                lo, hi = limits[m]
                if not lo <= v <= hi:
                    print(f"{name} の {m} {v:+.1f}° が実機の可動域 ({lo:+.1f}〜{hi:+.1f}) の外です。動かしません。")
                    return 1
        open_g = args.open if args.open is not None else min(100.0, ref["joints"]["gripper"] + 25)
        input("Enter で掴みに行きます (Ctrl-C で中止): ")
        present = follower.bus.sync_read("Present_Position")
        start = np.array([present[m] for m in kin.ARM])

        print("1. 爪を開いて、缶の手前へ")
        move_slowly(follower, to_action(start, open_g), args.speed)
        move_slowly(follower, to_action(joints["pre"], open_g), args.speed)
        if args.snap:
            time.sleep(1.0)
            stamp = time.strftime("%H%M%S")
            for _ in range(5):  # 古いフレームを捨てる
                img = follower.cameras["wrist"].async_read(1000)
            cv2.imwrite(str(out / f"wrist_pre_{stamp}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            json.dump({"can_base": can_base.tolist(), "joints_pre": joints["pre"].tolist(),
                       "pre": report[0][1].tolist(), "grasp": report[1][1].tolist()},
                      open(out / f"wrist_pre_{stamp}.json", "w"), indent=1)
            print(f"手首カメラの画像を保存しました: {out / f'wrist_pre_{stamp}.png'}")
            return 0
        if args.no_wrist:
            print("2. 缶へ近づく")
            move_slowly(follower, to_action(joints["grasp"], open_g), args.speed / 2)
        elif not args.servo3:
            # 高さは Kinect の「机からの高さ」で決まっているので、手首カメラでは左右 (pan) だけ直す
            print("2. 手首カメラで左右を合わせてから、缶へ近づく")
            dpan = center_pan(follower, joints["pre"], open_g, args, out)
            if dpan is None:
                return 1
            for name in joints:
                joints[name] = joints[name].copy()
                joints[name][0] += dpan
            move_slowly(follower, to_action(joints["grasp"], open_g), args.speed / 2)
        else:
            print("2. 手首カメラでキャップを見ながら、掴む位置まで寄せる")
            result = servo_to_cap(follower, report[0][1], report[1][1], joints["pre"], ref, open_g, args, out)
            if result is None:
                return 1
            q_grasp, p_grasp, dpan = result
            if args.final_dz:
                # 見え方を合わせた所は高すぎた (2026-11-22: -1.5 cm でもまだ上だった)。閉じる前に上下だけ直す
                p_grasp = p_grasp + np.array([0, 0, args.final_dz])
                q_grasp, err, _ = kin.ik(p_grasp, q_grasp - [dpan, 0, 0, 0, 0],
                                         pitch_deg=ref["pitch_deg"], roll_deg=q_grasp[4])
                q_grasp[0] += dpan
                print(f"   上下に {args.final_dz * 100:+.1f} cm 動かしてから掴みます")
                move_slowly(follower, to_action(q_grasp, open_g), args.speed / 3)
            q_lift, _, _ = kin.ik(p_grasp + np.array([0, 0, args.lift]), q_grasp - [dpan, 0, 0, 0, 0],
                                  pitch_deg=ref["pitch_deg"], roll_deg=q_grasp[4])
            q_lift[0] += dpan
            q_pre = joints["pre"].copy()
            q_pre[0] += dpan
            joints = {"pre": q_pre, "grasp": q_grasp, "lift": q_lift}
        cv2.imwrite(str(out / "wrist_grasp.png"), wrist_frame(follower))
        print("3. 掴む")
        contact = close_until_contact(follower, open_g, args.squeeze)
        hold = max(0.0, contact - args.squeeze)
        print("4. 持ち上げる")
        move_slowly(follower, to_action(joints["lift"], hold), args.speed / 2)
        time.sleep(1.5)
        print("5. 元の位置に置く")
        move_slowly(follower, to_action(joints["grasp"], hold), args.speed / 2)
        move_slowly(follower, to_action(joints["grasp"], open_g), args.speed)
        move_slowly(follower, to_action(joints["pre"], open_g), args.speed / 2)
    except KeyboardInterrupt:
        print("\n中止しました。")
    finally:
        try:
            print("待機姿勢へ戻ります。")
            present = follower.bus.sync_read("Present_Position")
            move_slowly(follower, {**REST_POSE, "wrist_flex.pos": present["wrist_flex"]}, args.speed)
        finally:
            follower.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
