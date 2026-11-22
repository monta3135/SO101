#!/usr/bin/env python3
"""使えるボードが 1 枚のまま、VLA 用のデモを 2 段階で集める。

ボードをリーダーとフォロワーで付け替えながら使う。リーダーとフォロワーは同時に動かせないので、
動きの記録と画像の記録を分ける。

  teach   ボードと 5 V 電源をリーダーにつなぐ。リーダーを手で動かし、関節角の軌道を
          outputs/teach/<セット名>/ に 1 エピソード 1 ファイルで保存する。フォロワーは動かない。
          --source follower にすると、トルクを切ったフォロワーを手で動かして記録する
          （ボードはフォロワーのまま 12 V）。
  replay  ボードと 12 V 電源をフォロワーにつなぐ。保存した軌道をフォロワーで再生し、
          カメラ画像と状態と行動を LeRobot のデータセットに記録する。エピソードの前に、
          物を teach のときと同じ位置に置き直す。
  check   つながっているアームと電圧を読んで表示するだけ。トルクは入れない。

どのモードでも、まずモーターの電圧と Homing_Offset を読む。腕の取り違えや電源の取り違えに
気づいたら、何も書き込まずに止まる。リーダーに 12 V がかかっていたら、すぐ電源を抜くよう表示する。

  uv run python scripts/swap_demo.py check
  uv run python scripts/swap_demo.py teach --set blue_can --task "pick up the blue can" --episodes 10
  uv run python scripts/swap_demo.py replay --set blue_can --dataset so101_blue_can
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

REPO = Path(__file__).resolve().parent.parent
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CALIBRATION = {
    "leader": REPO / "calibration/teleoperators/so_leader/leader.json",
    "follower": REPO / "calibration/robots/so_follower/follower.json",
}
# 実測はリーダー 4.9 V、フォロワー 11.8 V。これより外なら電源を取り違えている
VOLTAGE_RANGE = {"leader": (4.5, 6.0), "follower": (10.5, 13.0)}
LEADER_DANGER_V = 6.5
CAMERAS = {
    "wrist": "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0",
    "front": "/dev/v4l/by-id/usb-SHENZHEN_EMEET_TECHNOLOGY_CO._LTD._HD_Webcam_eMeet_C960_Ucamera002-video-index0",
}
CAMERA_LOCKS = {
    "wrist": REPO / "calibration/wrist_camera_lock.json",
    "front": REPO / "calibration/front_camera_lock.json",
}
# teach の軌道ファイル（000.json, 001.json, ...）。replayed_<データセット>.json は含めない
TRAJECTORY_GLOB = "[0-9][0-9][0-9].json"
# 待機姿勢（reset_pose.py と同じ）。replay の最後にここへ戻す
REST_POSE = {"shoulder_pan.pos": -5.0, "shoulder_lift.pos": -111.0, "elbow_flex.pos": 96.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ボード 1 枚でデモを集める（teach → replay）")
    parser.add_argument("--port", default="/dev/so101-leader", help="ボードのポート（残っている良品のボード）")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("check", help="アームと電圧を読むだけ")

    teach = sub.add_parser("teach", help="軌道を記録する")
    teach.add_argument("--set", required=True, help="軌道を保存するセット名（outputs/teach/<名前>）")
    teach.add_argument("--task", required=True, help="このエピソードの言語指示（英語推奨）")
    teach.add_argument("--episodes", type=int, default=10)
    teach.add_argument("--seconds", type=float, default=20.0, help="1 エピソードの最長時間")
    teach.add_argument("--fps", type=int, default=30)
    teach.add_argument("--source", choices=["leader", "follower"], default="leader")

    replay = sub.add_parser("replay", help="軌道をフォロワーで再生して記録する")
    replay.add_argument("--set", required=True)
    replay.add_argument("--dataset", required=True, help="データセット名（outputs/datasets/<名前>）")
    replay.add_argument("--cameras", default="wrist,front", help=f"使うカメラ（{', '.join(CAMERAS)}）")
    replay.add_argument("--speed", type=float, default=30.0, help="開始姿勢へ移る速さ [deg/s]")
    replay.add_argument("--max-step", type=float, default=10.0, help="再生中の 1 フレームの最大移動 [deg]")
    return parser.parse_args()


def read_arm(port: str) -> tuple[str, dict[str, float]]:
    """ボードの先のアームを Homing_Offset で見分け、電圧と一緒に返す。書き込みはしない。"""
    calibrations = {role: json.loads(path.read_text()) for role, path in CALIBRATION.items()}
    # 読むだけのためにバスを開く。トルクや設定には触れない
    probe = SO101Follower(SO101FollowerConfig(port=port, id="follower"))
    probe.bus.connect()
    try:
        voltage = probe.bus.sync_read("Present_Voltage", normalize=False)
        offsets = probe.bus.sync_read("Homing_Offset", normalize=False)
    finally:
        probe.bus.disconnect(disable_torque=False)

    volts = {motor: value / 10 for motor, value in voltage.items()}
    matches = {
        role: sum(offsets[m] == calibration[m]["homing_offset"] for m in MOTORS)
        for role, calibration in calibrations.items()
    }
    arm = next((role for role, count in matches.items() if count == len(MOTORS)), "不明")
    print(
        f"アーム: {arm}（較正の一致 leader {matches['leader']}/6, follower {matches['follower']}/6）  電圧: "
        + "  ".join(f"{v:.1f}" for v in volts.values())
        + " V",
        flush=True,
    )
    if arm == "leader" and max(volts.values()) > LEADER_DANGER_V:
        sys.exit(f"リーダーに {max(volts.values()):.1f} V かかっています。今すぐ DC 電源を抜いてください。")
    return arm, volts


def check_arm(port: str, role: str) -> None:
    """ボードの先が role のアームで、電圧も role のものかを確かめる。違えば終了する。"""
    arm, volts = read_arm(port)
    if arm != role:
        sys.exit(f"{role} のつもりが、つながっているのは {arm} です。付け替えを確かめてください。")
    low, high = VOLTAGE_RANGE[role]
    if not all(low <= v <= high for v in volts.values()):
        sys.exit(f"{role} の電圧は {low}〜{high} V のはずです。電源を確かめてください。")


class EnterWatcher:
    """別スレッドで Enter を待つ。pressed() で押されたかを見る。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self) -> None:
        sys.stdin.readline()
        self._event.set()

    def pressed(self) -> bool:
        return self._event.is_set()


def ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def teach(args: argparse.Namespace) -> int:
    check_arm(args.port, args.source)
    out_dir = REPO / "outputs/teach" / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "leader":
        arm = SO101Leader(SO101LeaderConfig(port=args.port, id="leader"))
        arm.connect(calibrate=False)
        read = arm.get_action
    else:
        arm = SO101Follower(SO101FollowerConfig(port=args.port, id="follower"))
        arm.connect(calibrate=False)
        arm.bus.disable_torque()  # 手で動かせるようにする
        read = arm.get_observation

    period = 1.0 / args.fps
    saved = 0
    try:
        while saved < args.episodes:
            index = len(list(out_dir.glob(TRAJECTORY_GLOB)))
            if ask(f"[{saved + 1}/{args.episodes}] 開始姿勢にして Enter（q で終了）: ") == "q":
                break
            print(f"記録中。終わったら Enter（最長 {args.seconds:.0f} 秒）", flush=True)
            watcher = EnterWatcher()
            frames = []
            start = time.perf_counter()
            while not watcher.pressed() and time.perf_counter() - start < args.seconds:
                tick = time.perf_counter()
                values = read()
                frames.append([round(float(values[f"{m}.pos"]), 2) for m in MOTORS])
                time.sleep(max(0.0, period - (time.perf_counter() - tick)))
            if not watcher.pressed():
                print("最長時間に達しました。Enter を押してください。", flush=True)
                while not watcher.pressed():
                    time.sleep(0.05)
            print(f"{len(frames)} フレーム（{len(frames) / args.fps:.1f} 秒）", flush=True)
            if len(frames) < args.fps:
                print("1 秒未満なので保存しません。", flush=True)
                continue
            if ask("Enter で保存、r でやり直し: ") == "r":
                continue
            path = out_dir / f"{index:03d}.json"
            path.write_text(
                json.dumps(
                    {"task": args.task, "fps": args.fps, "source": args.source, "joints": MOTORS, "frames": frames}
                )
            )
            saved += 1
            print(f"保存しました: {path.relative_to(REPO)}", flush=True)
    finally:
        arm.disconnect()
    print(f"{saved} エピソードを保存しました。replay の前にボードと 12 V 電源をフォロワーへ付け替えてください。")
    return 0


def lock_cameras(names: list[str]) -> None:
    """色と明るさがエピソードごとに変わらないよう、カメラの自動調整を切った値を書き戻す。"""
    for name in names:
        lock = CAMERA_LOCKS[name]
        mode = ["--load", str(lock)] if lock.exists() else ["--save", str(lock)]
        subprocess.run(
            [sys.executable, str(REPO / "scripts/lock_camera.py"), "--camera", CAMERAS[name], *mode], check=True
        )


def move_slowly(follower: SO101Follower, target: dict[str, float], speed: float) -> None:
    """現在の姿勢から target まで、最も遠い関節が speed [deg/s] になる速さで直線補間して動く。"""
    present = follower.bus.sync_read("Present_Position")
    start = np.array([present[key.removesuffix(".pos")] for key in target])
    goal = np.array(list(target.values()))
    steps = max(1, int(np.abs(goal - start).max() / speed / 0.02))
    for step in range(1, steps + 1):
        point = start + (goal - start) * step / steps
        follower.send_action(dict(zip(target, point.tolist())))
        time.sleep(0.02)
    time.sleep(0.3)


def replay(args: argparse.Namespace) -> int:
    teach_dir = REPO / "outputs/teach" / args.set
    trajectories = sorted(teach_dir.glob(TRAJECTORY_GLOB))
    if not trajectories:
        print(f"{teach_dir} に軌道がありません。", file=sys.stderr)
        return 1
    done_path = teach_dir / f"replayed_{args.dataset}.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    todo = [path for path in trajectories if path.name not in done]
    print(f"軌道 {len(trajectories)} 本のうち、未記録は {len(todo)} 本です。", flush=True)
    if not todo:
        return 0

    names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    for name in names:
        if name not in CAMERAS or not Path(CAMERAS[name]).exists():
            print(f"カメラ {name} がありません。", file=sys.stderr)
            return 1

    check_arm(args.port, "follower")
    lock_cameras(names)

    fps = json.loads(todo[0].read_text())["fps"]
    config = SO101FollowerConfig(
        port=args.port,
        id="follower",
        max_relative_target=args.max_step,
        cameras={
            name: OpenCVCameraConfig(index_or_path=CAMERAS[name], width=640, height=480, fps=30)
            for name in names
        },
    )
    follower = SO101Follower(config)
    features = {
        **hw_to_dataset_features(follower.observation_features, "observation"),
        **hw_to_dataset_features(follower.action_features, "action"),
    }
    root = REPO / "outputs/datasets" / args.dataset
    repo_id = f"local/{args.dataset}"
    if root.exists():
        dataset = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4 * len(names))
        recorded = {key for key in dataset.features if key.startswith(("observation", "action"))}
        if recorded != features.keys() or dataset.fps != fps:
            print(f"{root} はカメラか fps が違うデータセットです。", file=sys.stderr)
            return 1
    else:
        dataset = LeRobotDataset.create(
            repo_id, fps, features=features, root=root, robot_type=follower.name,
            image_writer_threads=4 * len(names),
        )

    follower.connect(calibrate=False)
    period = 1.0 / fps
    try:
        for path in todo:
            episode = json.loads(path.read_text())
            frames = episode["frames"]
            if episode["fps"] != fps:
                print(f"{path.name} は fps {episode['fps']} なので飛ばします。", flush=True)
                continue
            keys = [f"{m}.pos" for m in episode["joints"]]
            first = dict(zip(keys, frames[0]))
            while True:
                print(f"\n{path.name}: 「{episode['task']}」 {len(frames) / fps:.1f} 秒", flush=True)
                move_slowly(follower, first, args.speed)
                answer = ask("物を teach のときの位置に置いて Enter（s で飛ばす、q で終了）: ")
                if answer in ("s", "q"):
                    break
                for frame in frames:
                    tick = time.perf_counter()
                    observation = follower.get_observation()
                    action = dict(zip(keys, frame))
                    sent = follower.send_action(action)
                    dataset.add_frame(
                        {
                            **build_dataset_frame(dataset.features, observation, prefix="observation"),
                            **build_dataset_frame(dataset.features, sent, prefix="action"),
                            "task": episode["task"],
                        }
                    )
                    time.sleep(max(0.0, period - (time.perf_counter() - tick)))
                answer = ask("Enter で保存、r でやり直し、d で捨てる: ")
                if answer == "r":
                    dataset.clear_episode_buffer()
                    continue
                if answer == "d":
                    dataset.clear_episode_buffer()
                    break
                dataset.save_episode()
                done[path.name] = dataset.meta.total_episodes - 1
                done_path.write_text(json.dumps(done, indent=2))
                print(f"エピソード {done[path.name]} として保存しました。", flush=True)
                break
            if answer == "q":
                break
        move_slowly(follower, REST_POSE, args.speed)
    finally:
        dataset.finalize()
        follower.disconnect()
    print(f"記録済み {len(done)}/{len(trajectories)} 本: {root.relative_to(REPO)}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "check":
        arm, volts = read_arm(args.port)
        if arm == "不明":
            print("leader と follower のどちらの較正とも一致しません。")
            return 1
        low, high = VOLTAGE_RANGE[arm]
        ok = all(low <= v <= high for v in volts.values())
        print(f"{arm} の電圧として{'正常' if ok else 'おかしい'}です（{low}〜{high} V）。")
        return 0 if ok else 1
    return teach(args) if args.mode == "teach" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
