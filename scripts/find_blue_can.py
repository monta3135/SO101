#!/usr/bin/env python3
"""手首カメラで青いスプレー缶を探し、画面中央に来るまで追う。掴む動作はしない。

pan=0 から始め、見つからなければ pan を左右に振りながら elbow を段階的に動かして探す。
見つけたら、缶の中心が画面中央に来るように pan と上下（wrist_flex と elbow_flex の二段）で追う。
中央に入った状態を --hold 秒保ち、そのときの関節位置を表示してそこで止まる。

停止は Ctrl+C です。終了時は待機姿勢へ戻します（--no-return で戻さない）。
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from track_face import (
    CAM_HEIGHT,
    CAM_WIDTH,
    DEFAULT_CAMERA,
    MOTORS,
    approach,
    clamp,
    move_smoothly,
    pixel_to_angles,
    safe_limits,
    soft_deadband,
)

MAX_CYCLE_S = 0.1  # 処理が止まっても一度に進む時間は短く切る（秒）
BLUE_HSV_LOW = (100, 120, 70)  # OpenCV の色相は 0〜180。青は 100 台
BLUE_HSV_HIGH = (130, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="青いスプレー缶を探して画面中央に止まる")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--hz", type=float, default=30.0, help="制御の周期")
    parser.add_argument("--max-speed", type=float, default=30.0, help="関節の最大角速度（度/秒）")
    parser.add_argument("--max-accel", type=float, default=120.0, help="角速度の変化の上限（度/秒^2）")
    parser.add_argument("--search-speed", type=float, default=12.0, help="探索時の pan の速さ（度/秒）")
    parser.add_argument("--rows", type=int, default=5, help="探索で elbow を振り分ける段の数")
    parser.add_argument("--pan-gain", type=float, default=1.5, help="水平の追従の強さ（1/秒）")
    parser.add_argument("--elbow-gain", type=float, default=1.5, help="上下の追従の強さ（1/秒）")
    parser.add_argument("--deadband", type=float, default=1.0, help="この角度以内のずれは動かさない（度）")
    parser.add_argument("--tolerance", type=float, default=3.0, help="中央とみなす角度のずれ（度）")
    parser.add_argument("--hold", type=float, default=1.0, help="中央に入った状態を保つ時間（秒）")
    parser.add_argument("--fov", type=float, default=70.0, help="カメラの水平画角（度）")
    parser.add_argument("--min-area", type=float, default=1500.0, help="青い領域として扱う最小の面積（画素）")
    parser.add_argument("--wrist-recenter", type=float, default=0.15, help="手首を基準へ戻す速さ（1/秒）")
    parser.add_argument("--start-pan", type=float, default=0.0, help="開始時の shoulder_pan")
    parser.add_argument("--start-elbow", type=float, default=80.0, help="開始時の elbow_flex")
    parser.add_argument("--elbow-min", type=float, default=60.0)
    parser.add_argument("--elbow-max", type=float, default=100.0)
    parser.add_argument("--max-relative-target", type=float, default=15.0, help="1回の指令の上限（度）")
    parser.add_argument("--no-return", action="store_true", help="終了時に待機姿勢へ戻さない")
    return parser.parse_args()


def find_blue(image_rgb: np.ndarray, min_area: float) -> tuple[float, float, float] | None:
    """最大の青い領域の中心（画素）と面積を返す。見つからなければ None。"""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return x + w / 2, y + h / 2, area


def main() -> int:
    args = parse_args()
    if not args.camera.exists():
        print(f"カメラが見つかりません: {args.camera}", flush=True)
        return 1

    focal = (CAM_WIDTH / 2) / math.tan(math.radians(args.fov) / 2)
    period = 1.0 / args.hz

    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=args.camera, width=CAM_WIDTH, height=CAM_HEIGHT, fps=30
            )
        },
        max_relative_target=args.max_relative_target,
    )
    follower = SO101Follower(config)
    follower.connect(calibrate=False)

    parked: dict[str, float] | None = None
    try:
        start = follower.get_observation()
        limits = {m: safe_limits(follower, m, 1.0) for m in MOTORS}
        parked = {f"{m}.pos": start[f"{m}.pos"] for m in MOTORS}

        pan_low, pan_high = limits["shoulder_pan"]
        elbow_low = max(args.elbow_min, limits["elbow_flex"][0])
        elbow_high = min(args.elbow_max, limits["elbow_flex"][1])
        wrist_low, wrist_high = limits["wrist_flex"]
        wrist_center = clamp(parked["wrist_flex.pos"], wrist_low, wrist_high)

        goal = dict(parked)
        goal["shoulder_pan.pos"] = clamp(args.start_pan, pan_low, pan_high)
        goal["elbow_flex.pos"] = clamp(args.start_elbow, elbow_low, elbow_high)
        print(
            f"追跡姿勢へ移動します: pan {goal['shoulder_pan.pos']:.1f}, "
            f"elbow {goal['elbow_flex.pos']:.1f}",
            flush=True,
        )
        if not move_smoothly(follower, goal, args.max_speed, timeout_s=60.0):
            print("開始姿勢に届きませんでした。中断します。", flush=True)
            return 2
        print("青い缶を探します。Ctrl+C で停止。", flush=True)

        pan = goal["shoulder_pan.pos"]
        elbow = goal["elbow_flex.pos"]
        wrist = parked["wrist_flex.pos"]
        pan_speed = elbow_speed = wrist_speed = pitch_speed = 0.0
        rows = [float(v) for v in np.linspace(elbow_low, elbow_high, args.rows)]
        row_i = 0
        elbow_target = rows[row_i]
        search_dir = 1
        found = False
        found_s = 0.0  # 中央に入っている連続時間（秒）
        t_prev = time.perf_counter()
        last_print = 0.0

        while True:
            loop_start = time.perf_counter()
            dt = min(loop_start - t_prev, MAX_CYCLE_S)
            t_prev = loop_start
            max_delta = args.max_accel * dt

            observation = follower.get_observation()
            image = observation["wrist"]
            blob = find_blue(image, args.min_area)

            pan_want = elbow_want = wrist_want = 0.0
            pitch_want = 0.0
            mode = "探索中"
            if found:
                mode = "発見・保持"
            elif blob is not None:
                x, y, _ = blob
                ang_x, ang_y = pixel_to_angles((x, y), focal)
                pan_want = args.pan_gain * soft_deadband(ang_x, args.deadband)
                pitch_want = -args.elbow_gain * soft_deadband(ang_y, args.deadband)
                mode = "追跡中"
                centered = abs(ang_x) < args.tolerance and abs(ang_y) < args.tolerance
                found_s = found_s + dt if centered else 0.0
                if found_s >= args.hold:
                    found = True
                    print(
                        f"青い缶を中央に捉えました。pan {pan:+.1f}  elbow {elbow:+.1f}  "
                        f"wrist {wrist:+.1f}  ここで保持します（Ctrl+C で戻ります）",
                        flush=True,
                    )
            else:
                found_s = 0.0
                pan_want = search_dir * args.search_speed
                if (search_dir > 0 and pan >= pan_high - 0.5) or (
                    search_dir < 0 and pan <= pan_low + 0.5
                ):
                    # 端で折り返すたびに、elbow を次の段へ動かす
                    search_dir = -search_dir
                    row_i = (row_i + 1) % len(rows)
                    elbow_target = rows[row_i]

            pan_speed = approach(pan_speed, clamp(pan_want, -args.max_speed, args.max_speed), max_delta)
            if mode != "探索中":
                # 上下は手首を先に動かし、出しきれない分を肘で補う（track_face と同じ分担）
                pitch_speed = approach(pitch_speed, clamp(pitch_want, -args.max_speed, args.max_speed), max_delta)
                wrist_want = pitch_speed - args.wrist_recenter * (wrist - wrist_center)
                wrist_speed = approach(wrist_speed, clamp(wrist_want, -args.max_speed, args.max_speed), max_delta)
                elbow_want = pitch_speed - wrist_speed
            else:
                # 探索中は elbow を段の高さへ寄せ、手首は基準へ戻す
                elbow_want = args.elbow_gain * (elbow_target - elbow)
                wrist_want = -args.wrist_recenter * (wrist - wrist_center)
                wrist_speed = approach(wrist_speed, clamp(wrist_want, -args.max_speed, args.max_speed), max_delta)
            elbow_speed = approach(elbow_speed, clamp(elbow_want, -args.max_speed, args.max_speed), max_delta)

            pan += pan_speed * dt
            elbow += elbow_speed * dt
            wrist += wrist_speed * dt
            pan = clamp(pan, pan_low, pan_high)
            elbow = clamp(elbow, elbow_low, elbow_high)
            wrist = clamp(wrist, wrist_low, wrist_high)

            command = dict(parked)
            command["shoulder_pan.pos"] = pan
            command["elbow_flex.pos"] = elbow
            command["wrist_flex.pos"] = wrist
            sent = follower.send_action(command)
            pan = sent["shoulder_pan.pos"]
            elbow = sent["elbow_flex.pos"]
            wrist = sent["wrist_flex.pos"]

            now = time.perf_counter()
            if now - last_print >= 0.5:
                last_print = now
                print(
                    f"pan {pan:+6.1f}  elbow {elbow:+6.1f}  wrist {wrist:+6.1f}  {mode}",
                    flush=True,
                )

            wait = period - (time.perf_counter() - loop_start)
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        print("\n停止します。", flush=True)
    finally:
        if parked and not args.no_return:
            print("待機姿勢へ戻します。", flush=True)
            try:
                if not move_smoothly(follower, parked, args.max_speed, timeout_s=90.0):
                    print("待機姿勢に完全には戻りませんでした。手で戻してください。", flush=True)
            except Exception as error:  # 戻せなくても切断は必ず行う
                print(f"復帰中にエラー: {error}", flush=True)
        try:
            follower.disconnect()
        except Exception as error:  # 通信が切れていると切断も失敗する
            print(f"切断中にエラー: {error}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
