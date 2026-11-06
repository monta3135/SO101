#!/usr/bin/env python3
"""フォロワーに付けたカメラで人の顔をリアルタイムに追跡する。

水平は shoulder_pan、上下は wrist_flex と elbow_flex の二段で追従します。

顔のずれを角度に換算し、そのずれに比例した角速度を毎周期（既定 30Hz）で指令します。
軽い wrist_flex が細かい垂直修正を先に受け持ち、出しきれない分や手首が基準位置から
離れた分を elbow_flex が広い可動域で引き継ぐので、上下の動きに強く追従します。
角速度が変わる量には上限を付け、角度は周期ごとに積み上げて送り続けます。
目標が大きく飛ばないので、速く動いても止まったり跳ねたりせず連続して動きます。
見失っても --coast（既定 0.6 秒）の間は最後の角速度で惰性を続けるので、瞬間的な
検出のちらつきで止まりません。それでも 2 秒以上見失い続けるか、ちらつきが 3 回
続いたときは、pan をゆっくり首振りして可動域をなぞり顔を探します。土台が可動域の
端に張り付いていても、人が円を描いて反対側へ回ったときに先に向かって、画面に入った
顔を自動で再追跡します。

停止は Ctrl+C です。終了時は現在位置から待機姿勢へ滑らかに戻します。
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

DEFAULT_CAMERA = Path(
    "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
)
CAM_WIDTH = 640
CAM_HEIGHT = 480
MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
MOVE_TOLERANCE = 2.0  # 目標に届いたとみなす誤差（度）
MAX_CYCLE_S = 0.1  # 処理が一時的に止まっても一度に動く時間は短く切る（秒）
FLICKER_LIMIT = 3  # 探索を始めるまでに許す検出のちらつき回数
FLICKER_MIN_GAP = 0.4  # ちらつきと数える見失い時間の下限（秒）
FLICKER_EXIT_ERR = 20.0  # このずれより大きい画面端からの消え方のみ、ちらつきと数える


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="カメラで顔をリアルタイム追跡する")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--hz", type=float, default=30.0, help="制御と検出の周期")
    parser.add_argument("--max-speed", type=float, default=45.0, help="関節の最大角速度（度/秒）")
    parser.add_argument(
        "--max-accel",
        type=float,
        default=180.0,
        help="角速度の変化の上限（度/秒^2）。小さいほど発進と停止がなめらか",
    )
    parser.add_argument("--pan-gain", type=float, default=2.0, help="水平の追従の強さ（1/秒）")
    parser.add_argument("--elbow-gain", type=float, default=2.0, help="上下（ピッチ）の追従の強さ（1/秒）")
    parser.add_argument(
        "--deadband", type=float, default=1.0, help="この角度以内のずれは動かさない（度）"
    )
    parser.add_argument(
        "--fov", type=float, default=70.0, help="カメラの水平画角（度）。画素を角度に換算する"
    )
    parser.add_argument(
        "--smooth", type=float, default=0.05, help="検出位置のぶれを抑える時定数（秒）。0で無効"
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="顔検出スコアの下限")
    parser.add_argument("--detect-width", type=int, default=320, help="検出用に縮小する幅")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=15.0,
        help="1回の指令で現在位置から離れてよい上限（度）。最後の安全装置",
    )
    parser.add_argument("--start-pan", type=float, default=-92.5, help="開始時の shoulder_pan")
    parser.add_argument("--start-elbow", type=float, default=80.0, help="開始時の elbow_flex")
    parser.add_argument(
        "--pan-min",
        type=float,
        default=-180.0,
        help="shoulder_pan の下限。既定ではキャリブレーションの可動域いっぱいまで使う",
    )
    parser.add_argument(
        "--pan-max",
        type=float,
        default=180.0,
        help="shoulder_pan の上限。既定ではキャリブレーションの可動域いっぱいまで使う",
    )
    parser.add_argument(
        "--elbow-min",
        type=float,
        default=60.0,
        help="elbow_flex の下限。こちらを伸ばしすぎると重心が前に出て転倒しやすい",
    )
    parser.add_argument("--elbow-max", type=float, default=100.0, help="elbow_flex の上限")
    parser.add_argument(
        "--no-wrist",
        action="store_true",
        help="wrist_flex を動かさず elbow_flex だけで上下を追う",
    )
    parser.add_argument(
        "--wrist-min",
        type=float,
        default=None,
        help="wrist_flex の下限。未指定ならキャリブレーションの可動域いっぱいまで使う",
    )
    parser.add_argument(
        "--wrist-max",
        type=float,
        default=None,
        help="wrist_flex の上限。未指定ならキャリブレーションの可動域いっぱいまで使う",
    )
    parser.add_argument(
        "--wrist-recenter",
        type=float,
        default=0.15,
        help="手首を基準位置へゆっくり戻す速さ（1/秒）。0 で戻さない",
    )
    parser.add_argument(
        "--wrist-center",
        type=float,
        default=None,
        help="手首が基準とする位置。未指定なら開始時の位置",
    )
    parser.add_argument(
        "--search-speed",
        type=float,
        default=15.0,
        help="見失ったときの首振り探索の速さ（度/秒）",
    )
    parser.add_argument(
        "--search-delay",
        type=float,
        default=2.0,
        help="見失ってから首振り探索を始めるまでの時間（秒）",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="見失ったときに首振り探索をしない",
    )
    parser.add_argument(
        "--coast",
        type=float,
        default=0.6,
        help="見失った後、最後の角速度のまま惰性で動く時間（秒）。0 で無効",
    )
    parser.add_argument(
        "--acquire",
        type=float,
        default=0.25,
        help="探索から追跡に復帰するまで顔を連続検出する時間（秒）",
    )
    parser.add_argument("--display", action="store_true", help="プレビュー画面を表示する")
    parser.add_argument("--no-return", action="store_true", help="終了時に待機姿勢へ戻さない")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / ".cache/so101/face_detection_yunet_2023mar.onnx",
    )
    return parser.parse_args()


def load_detector(model: Path, width: int, height: int) -> cv2.FaceDetectorYN:
    if not model.exists():
        model.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        url = (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        )
        print(f"顔検出モデルを取得します: {url}", flush=True)
        urllib.request.urlretrieve(url, model)
    return cv2.FaceDetectorYN.create(str(model), "", (width, height), 0.6, 0.3, 5000)


def clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def approach(current: float, target: float, max_delta: float) -> float:
    """current を target へ、1周期で max_delta 以内だけ近づける。"""
    return current + clamp(target - current, -max_delta, max_delta)


def soft_deadband(value: float, band: float) -> float:
    """band 以内は 0 にし、超えた分だけを返す。境目で速度が飛ばない。"""
    return math.copysign(max(abs(value) - band, 0.0), value)


def smooth(current: float, measured: float, dt: float, time_constant: float) -> float:
    """1次遅れのフィルタで検出位置のぶれを抑える。時定数が 0 なら素通しする。"""
    if time_constant <= 0:
        return measured
    return current + (1 - math.exp(-dt / time_constant)) * (measured - current)


def safe_limits(follower: SO101Follower, motor: str, margin: float) -> tuple[float, float]:
    calibration = follower.bus.calibration[motor]
    middle = (calibration.range_min + calibration.range_max) / 2
    low = (calibration.range_min - middle) * 360 / 4095
    high = (calibration.range_max - middle) * 360 / 4095
    return low + margin, high - margin


def read_present(follower: SO101Follower) -> dict[str, float]:
    positions = follower.bus.sync_read("Present_Position", num_retry=1)
    return {f"{motor}.pos": value for motor, value in positions.items()}


def move_smoothly(
    follower: SO101Follower,
    goal: dict[str, float],
    speed: float,
    timeout_s: float,
) -> bool:
    """現在位置から goal へ、速度を 0 から立ち上げて 0 に戻すように動かす。届くまで指令を送る。"""
    start = read_present(follower)
    distance = max(abs(goal[key] - start[key]) for key in goal)
    # コサイン曲線の速度は最大で speed になる
    duration = max(1.0, math.pi / 2 * distance / speed)
    t0 = time.perf_counter()
    deadline = t0 + duration + timeout_s
    next_report = t0
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        ratio = min((now - t0) / duration, 1.0)
        eased = (1 - math.cos(math.pi * ratio)) / 2
        follower.send_action(
            {key: start[key] + (goal[key] - start[key]) * eased for key in goal}
        )
        if ratio >= 1.0:
            present = read_present(follower)
            worst = max(abs(present[key] - goal[key]) for key in goal)
            if now >= next_report:
                print(f"  目標へ収束中: 最大ずれ {worst:.1f} 度", flush=True)
                next_report = now + 1.0
            if worst < MOVE_TOLERANCE:
                return True
        elif now >= next_report:
            print(f"  移動中: {ratio:.0%}", flush=True)
            next_report = now + 1.0
        time.sleep(0.02)
    present = read_present(follower)
    for key in goal:
        if abs(present[key] - goal[key]) >= MOVE_TOLERANCE:
            print(
                f"  届かない関節: {key} 目標 {goal[key]:+.1f} 現在 {present[key]:+.1f}",
                flush=True,
            )
    return False


def pick_face(
    faces: np.ndarray | None,
    scale: float,
    threshold: float,
    previous: tuple[float, float] | None,
) -> tuple[float, float, float, float] | None:
    """検出結果から追跡する顔を1つ選ぶ。追跡中は前回の位置に一番近い顔、初回は一番大きい顔。"""
    if faces is None:
        return None
    best = None
    best_key = 0.0
    for face in faces:
        if face[14] < threshold:
            continue
        x, y, w, h = (float(v) / scale for v in face[:4])
        if previous is None:
            key = -w * h
        else:
            key = math.hypot(x + w / 2 - previous[0], y + h / 2 - previous[1])
        if best is None or key < best_key:
            best = (x, y, w, h)
            best_key = key
    return best


def pixel_to_angles(center: tuple[float, float], focal: float) -> tuple[float, float]:
    """画像中心からのずれを、カメラから見た角度（度）に換算する。右と下が正。"""
    dx = center[0] - CAM_WIDTH / 2
    dy = center[1] - CAM_HEIGHT / 2
    return math.degrees(math.atan2(dx, focal)), math.degrees(math.atan2(dy, focal))


def main() -> int:
    args = parse_args()
    if not args.camera.exists():
        print(f"カメラが見つかりません: {args.camera}", flush=True)
        return 1

    scale = args.detect_width / CAM_WIDTH
    detect_height = round(CAM_HEIGHT * scale)
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
    detector = load_detector(args.model, args.detect_width, detect_height)
    follower = SO101Follower(config)
    follower.connect(calibrate=False)

    show = args.display
    parked: dict[str, float] | None = None
    try:
        start = follower.get_observation()
        limits = {m: safe_limits(follower, m, 1.0) for m in MOTORS}
        parked = {f"{m}.pos": start[f"{m}.pos"] for m in MOTORS}

        # 可動域は、指定とキャリブレーションの狭い方を採用する
        pan_low = max(args.pan_min, limits["shoulder_pan"][0])
        pan_high = min(args.pan_max, limits["shoulder_pan"][1])
        elbow_low = max(args.elbow_min, limits["elbow_flex"][0])
        elbow_high = min(args.elbow_max, limits["elbow_flex"][1])
        wrist_hard = limits["wrist_flex"]
        wrist_low = wrist_hard[0] if args.wrist_min is None else max(args.wrist_min, wrist_hard[0])
        wrist_high = wrist_hard[1] if args.wrist_max is None else min(args.wrist_max, wrist_hard[1])
        use_wrist = not args.no_wrist and wrist_low < wrist_high
        wrist_center = parked["wrist_flex.pos"] if args.wrist_center is None else args.wrist_center
        wrist_center = clamp(wrist_center, wrist_low, wrist_high)
        print(
            f"実効可動域: pan {pan_low:+.1f}〜{pan_high:+.1f}, "
            f"elbow {elbow_low:+.1f}〜{elbow_high:+.1f}, "
            f"wrist {wrist_low:+.1f}〜{wrist_high:+.1f}",
            flush=True,
        )

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
        print("追跡を開始します。Ctrl+C で停止。", flush=True)

        pan = goal["shoulder_pan.pos"]
        elbow = goal["elbow_flex.pos"]
        wrist = parked["wrist_flex.pos"]
        pan_speed = 0.0
        elbow_speed = 0.0
        pitch_speed = 0.0  # 上下（ピッチ）の目標角速度。肘と手首が分担して出す合計
        wrist_speed = 0.0
        search_dir = 0  # 首振り探索の向き（+1 で pan を増やす方向）
        lost_s = 0.0  # 顔を見失い続けている時間（秒）
        search_active = False
        stable_s = 0.0  # 見失わずにいる連続時間（秒）
        coast_s = 0.0  # 惰性で動き続ける残り時間（秒）
        flicker_n = 0  # 連続した短い見失いの回数（ちらつき）
        intent_dir = 0  # 最後に顔と同じ方向へ動こうとした向き
        last_intent_pan = 0.0  # 惰性で使い回す直前の目標角速度
        last_intent_pitch = 0.0
        last_err_x = 0.0  # 最後に顔を検出したときの水平のずれ（度）
        angles: tuple[float, float] | None = None  # なめらか化した (水平, 垂直) のずれ（度）
        previous: tuple[float, float] | None = None  # 追跡中の顔の中心（画素）
        was_found = False
        fps = args.hz
        frame_i = 0
        t_prev = time.perf_counter()

        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - t_prev
            t_prev = loop_start
            dt = min(elapsed, MAX_CYCLE_S)
            fps = 0.9 * fps + 0.1 / max(elapsed, period)

            observation = follower.get_observation()
            image = observation["wrist"]
            small = cv2.resize(
                image, (args.detect_width, detect_height), interpolation=cv2.INTER_AREA
            )
            detector.setInputSize(small.shape[1::-1])
            _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
            box = pick_face(faces, scale, args.threshold, previous)

            found = box is not None
            if found != was_found:
                if not found and not search_active:
                    # 見失いの開始: 惰性の残り時間を設定する
                    coast_s = args.coast
                if found and lost_s >= max(args.coast, FLICKER_MIN_GAP):
                    if lost_s < args.search_delay:
                        # 探索開始より短い見失いの連続は「ちらつき」として数える。
                        # 数える条件は見失う直前のずれが画面端（±20度）だった場合のみ。
                        # 画面中央付近で消えた検出ムラまで数えると、見えている顔から
                        # 掃引して離れてしまう
                        if abs(last_err_x) >= FLICKER_EXIT_ERR:
                            flicker_n += 1
                    else:
                        flicker_n = 0
                print(
                    "顔を見つけました。追跡します。" if found else "顔を見失いました。止まります。",
                    flush=True,
                )
                was_found = found

            if box is None:
                angles = None
                if not search_active and coast_s > 0:
                    # 短い見失い: 直前の目標角速度のまま惰性で進む。瞬間的な検出抜けで
                    # 発進・停止を繰り返さず、通り抜けた先で顔が見えることもある
                    want_pan = last_intent_pan
                    want_pitch = last_intent_pitch
                else:
                    want_pan = 0.0
                    want_pitch = 0.0
            else:
                x, y, w, h = box
                center = (x + w / 2, y + h / 2)
                previous = center
                ang_x, ang_y = pixel_to_angles(center, focal)
                if angles is None:
                    angles = (ang_x, ang_y)
                else:
                    angles = (
                        smooth(angles[0], ang_x, dt, args.smooth),
                        smooth(angles[1], ang_y, dt, args.smooth),
                    )
                # 顔が右にあれば pan を増やし、下にあればカメラのピッチを減らす
                want_pan = args.pan_gain * soft_deadband(angles[0], args.deadband)
                want_pitch = -args.elbow_gain * soft_deadband(angles[1], args.deadband)
                last_err_x = angles[0]
                last_intent_pan = want_pan
                last_intent_pitch = want_pitch
                if want_pan > 0.1:
                    intent_dir = 1
                elif want_pan < -0.1:
                    intent_dir = -1

            if box is not None:
                lost_s = 0.0
                stable_s += dt
                if stable_s >= 2.0:
                    flicker_n = 0  # 安定して見えていればちらつきカウントを戻す
                if search_active and stable_s >= args.acquire:
                    search_active = False
                    print("顔が安定して見えているので追跡を再開します。", flush=True)
            else:
                stable_s = 0.0
                coast_s = max(0.0, coast_s - dt)
                lost_s += dt
                # 2 秒以上の見失いか、ちらつきの連続で探索を始める。
                # 土台は可動域の端で張り付くしかないので、人が円を描いて反対側へ
                # 回った場合に可動域をなぞって先回りし、画面に入った顔を拾える
                if not args.no_search and not search_active and (
                    lost_s >= args.search_delay or flicker_n >= FLICKER_LIMIT
                ):
                    search_active = True
                    flicker_n = 0
                    coast_s = 0.0
                    # 振り返る向き: 端に張り付いているなら可動域の反対側へ。
                    # そうでなければ、見失う直前に顔が移動していた方向へなぞる
                    if pan >= pan_high - 0.5:
                        search_dir = -1
                    elif pan <= pan_low + 0.5:
                        search_dir = 1
                    elif intent_dir != 0:
                        search_dir = intent_dir
                    else:
                        search_dir = 1 if pan <= 0 else -1
                    print("見失ったままなので首を振って顔を探します。", flush=True)
                if search_active:
                    want_pan = search_dir * args.search_speed
                    want_pitch = 0.0
                    # 端に達したら反対側へ折り返して往復する
                    if (search_dir > 0 and pan >= pan_high - 0.2) or (
                        search_dir < 0 and pan <= pan_low + 0.2
                    ):
                        search_dir = -search_dir

            # 角速度を目標へ、加速度の上限の範囲で近づける
            max_delta = args.max_accel * dt
            pan_speed = approach(
                pan_speed, clamp(want_pan, -args.max_speed, args.max_speed), max_delta
            )
            pitch_speed = approach(
                pitch_speed, clamp(want_pitch, -args.max_speed, args.max_speed), max_delta
            )

            # 上下は二段構え。軽い wrist_flex が先に動き、出しきれない分は elbow_flex が
            # 広い可動域で引き継ぐ。elbow_flex と wrist_flex は同じ向きの回転軸なので、
            # 片方の動かした分をもう片方で打ち消せば、カメラの向きを保ったまま
            # 手首を基準位置へ戻して手首の出し余力を回復できる。
            wrist_sent = 0.0  # この周期に wrist_flex が実際に出せた角速度
            if use_wrist:
                want_wrist_speed = pitch_speed - args.wrist_recenter * (wrist - wrist_center)
                wrist_speed = approach(
                    wrist_speed,
                    clamp(want_wrist_speed, -args.max_speed, args.max_speed),
                    max_delta,
                )
                wrist_before = wrist
                wrist += wrist_speed * dt
                if not wrist_low <= wrist <= wrist_high:
                    wrist = clamp(wrist, wrist_low, wrist_high)
                    wrist_speed = 0.0
                wrist_sent = (wrist - wrist_before) / max(dt, 1e-6)
            want_elbow_speed = (pitch_speed - wrist_sent) if use_wrist else pitch_speed
            elbow_speed = approach(
                elbow_speed,
                clamp(want_elbow_speed, -args.max_speed, args.max_speed),
                max_delta,
            )
            pan += pan_speed * dt
            elbow += elbow_speed * dt
            if not pan_low <= pan <= pan_high:
                pan = clamp(pan, pan_low, pan_high)
                pan_speed = 0.0
            if not elbow_low <= elbow <= elbow_high:
                elbow = clamp(elbow, elbow_low, elbow_high)
                elbow_speed = 0.0

            command = dict(parked)
            command["shoulder_pan.pos"] = pan
            command["elbow_flex.pos"] = elbow
            command["wrist_flex.pos"] = wrist
            sent = follower.send_action(command)
            # 上限で削られた分は積み上げない。実際に送れた値を次の基準にする
            pan = sent["shoulder_pan.pos"]
            elbow = sent["elbow_flex.pos"]
            wrist = sent["wrist_flex.pos"]

            if frame_i % max(1, round(args.hz / 2)) == 0:
                seen = "顔なし" if angles is None else (
                    f"ずれ ({angles[0]:+5.1f}, {angles[1]:+5.1f}) 度"
                )
                if search_active:
                    seen = "探索中"
                print(
                    f"pan {pan:+6.1f}  elbow {elbow:+6.1f}  wrist {wrist:+6.1f}  "
                    f"fps {fps:4.1f}  {seen}",
                    flush=True,
                )
            frame_i += 1

            if show:
                preview = image.copy()
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(
                        preview, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
                    )
                cv2.putText(
                    preview,
                    f"pan {pan:+.1f} elbow {elbow:+.1f} wrist {wrist:+.1f} fps {fps:.1f}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                try:
                    cv2.imshow("track", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                except cv2.error as error:  # headless 版の OpenCV には表示機能がない
                    print(f"プレビューを表示できません。表示なしで続けます: {error}", flush=True)
                    show = False

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
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        try:
            follower.disconnect()  # 表示の後始末で失敗しても必ず切断する
        except Exception as error:  # 通信が切れていると切断も失敗する
            print(f"切断中にエラー: {error}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
