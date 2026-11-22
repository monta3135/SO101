#!/usr/bin/env python3
"""関節を1つずつステップとチャープで動かし、フォロワーの応答を CSV に記録する。

sim の位置サーボと実機の違い（重力による垂れ、遅れ、追従できる速さ）を測るためのもの。
同じ指令を tachyon の MuJoCo で再生して重ねるのは scripts/compare_step_response.py が行う。

まず試験姿勢（既定は前へ L 字に伸ばした姿勢）へ滑らかに移動する。--joints の関節を1つずつ、
試験姿勢から +A, 0, -A, 0, +2A, 0 度のステップでそれぞれ --dwell 秒保ち、続けて振幅
--chirp-amp 度で周波数が --chirp-f0 から --chirp-f1 Hz へ上がる正弦波（チャープ）を入れる。
動かす関節以外は試験姿勢に保つ。指令はテレオペや方策の実行と同じく、毎周期すべての関節に書く。

記録は outputs/step_response/<日時>.csv と、同名の .json（計画、サーボの設定値、キャリブレーション）。
--dry-run はロボットに接続せず、計画だけを同じ形式で書く（sim 側の動作確認用）。

開始前に計画を表示して Enter を待つ。停止は Ctrl+C。終了時は開始時の姿勢へ戻してから切断する。
既定の試験姿勢では、ハンドが土台の前方約 20cm、高さ約 23cm に来る（sim での計算）。
前と左右を空けてから動かすこと。
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import math
import statistics
import time
from pathlib import Path

import lerobot
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from track_face import MOTORS, clamp, move_smoothly, safe_limits

SHORT_NAMES = {
    "pan": "shoulder_pan",
    "lift": "shoulder_lift",
    "elbow": "elbow_flex",
    "wrist": "wrist_flex",
    "roll": "wrist_roll",
    "gripper": "gripper",
}
# --dry-run で開始姿勢の代わりに使う待機姿勢（docs/software.md の値）
DRY_RUN_START = {
    "shoulder_pan": -5.0,
    "shoulder_lift": -111.0,
    "elbow_flex": 96.0,
    "wrist_flex": -101.0,
    "wrist_roll": -179.0,
    "gripper": 1.0,
}
STEP_SCALES = (1.0, 0.0, -1.0, 0.0, 2.0, 0.0)  # ステップの段（--step に掛ける倍率）
CHIRP_RAMP_S = 0.5  # チャープの振幅を立ち上げ、立ち下げる時間。区間の境目で段差を作らない
ABORT_AFTER_S = 0.5  # 指令とのずれが大きい状態がこれだけ続いたら止める
MAX_READ_FAILURES = 25  # 位置がこの回数続けて読めなかったら止める
SLOW_READ_EVERY = 10  # 電圧と温度は 10 周期に 1 回だけ読む
VELOCITY_DEG_PER_UNIT = 360 / 4096  # Present_Velocity は 1 回転 4096 ステップの step/s
LOAD_PERCENT_PER_UNIT = 0.1  # Present_Load は最大トルクに対する 0.1% 単位
CURRENT_A_PER_UNIT = 0.0065  # teleop_power_log.py と同じ換算
SERVO_SETTINGS = (
    "P_Coefficient",
    "I_Coefficient",
    "D_Coefficient",
    "Acceleration",
    "Maximum_Acceleration",
    "Max_Torque_Limit",
    "Torque_Limit",
    "Protection_Current",
    "Return_Delay_Time",
)
FIELDNAMES = [
    "t_cmd",
    "t_obs",
    "seg",
    "segment",
    "joint",
    *[f"cmd.{m}" for m in MOTORS],
    *[f"pos.{m}" for m in MOTORS],
    *[f"vel.{m}" for m in MOTORS],
    *[f"load.{m}" for m in MOTORS],
    *[f"cur_a.{m}" for m in MOTORS],
    *[f"volt.{m}" for m in MOTORS],
    *[f"temp.{m}" for m in MOTORS],
    "loop_ms",
    "event",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="関節ごとのステップ応答とチャープ応答を記録する")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--pan", type=float, default=0.0, help="試験姿勢の shoulder_pan")
    parser.add_argument("--lift", type=float, default=-40.0, help="試験姿勢の shoulder_lift")
    parser.add_argument("--elbow", type=float, default=40.0, help="試験姿勢の elbow_flex")
    parser.add_argument("--wrist", type=float, default=0.0, help="試験姿勢の wrist_flex")
    parser.add_argument("--roll", type=float, help="試験姿勢の wrist_roll。未指定なら開始時のまま")
    parser.add_argument("--gripper", type=float, help="試験姿勢の gripper。未指定なら開始時のまま")
    parser.add_argument("--joints", default="pan,lift,elbow,wrist", help="動かす関節（カンマ区切り）")
    parser.add_argument("--step", type=float, default=10.0, help="ステップの大きさ（度）")
    parser.add_argument("--dwell", type=float, default=1.5, help="各ステップを保つ時間（秒）")
    parser.add_argument("--chirp-amp", type=float, default=5.0, help="チャープの振幅（度）")
    parser.add_argument("--chirp-f0", type=float, default=0.2, help="チャープの開始周波数（Hz）")
    parser.add_argument("--chirp-f1", type=float, default=2.5, help="チャープの終了周波数（Hz）")
    parser.add_argument("--chirp-time", type=float, default=12.0, help="チャープの長さ（秒）。0 で省く")
    parser.add_argument("--settle", type=float, default=1.0, help="関節を切り替える前に止まる時間（秒）")
    parser.add_argument("--hz", type=float, default=50.0, help="指令と記録の周期")
    parser.add_argument("--speed", type=float, default=30.0, help="試験姿勢との行き来の最大角速度（度/秒）")
    parser.add_argument("--margin", type=float, default=5.0, help="キャリブレーションの可動域から空ける角度（度）")
    parser.add_argument(
        "--abort-error", type=float, default=30.0, help="指令とのずれがこれを超え続けたら止める（度）"
    )
    parser.add_argument("--position-only", action="store_true", help="位置だけを読み、記録の周期を上げる")
    parser.add_argument("--dry-run", action="store_true", help="ロボットに接続せず計画だけを書く")
    parser.add_argument("--yes", action="store_true", help="開始前の確認を省く")
    parser.add_argument("--out", type=Path, help="CSV の保存先")
    return parser.parse_args()


def parse_joints(text: str) -> list[str]:
    joints = []
    for name in text.split(","):
        name = name.strip()
        if not name:
            continue
        motor = SHORT_NAMES.get(name, name)
        if motor not in MOTORS:
            raise SystemExit(f"知らない関節です: {name}")
        joints.append(motor)
    return joints


def test_pose(
    args: argparse.Namespace, start: dict[str, float], limits: dict[str, tuple[float, float]]
) -> tuple[dict[str, float], list[str]]:
    requested = {
        "shoulder_pan": args.pan,
        "shoulder_lift": args.lift,
        "elbow_flex": args.elbow,
        "wrist_flex": args.wrist,
        "wrist_roll": args.roll,
        "gripper": args.gripper,
    }
    pose: dict[str, float] = {}
    notes = []
    for motor in MOTORS:
        value = requested[motor]
        if value is None:
            pose[motor] = start[motor]  # 開始時のまま保つ関節は、可動域で切らずにそのまま置く
            continue
        pose[motor] = clamp(value, *limits[motor])
        if pose[motor] != value:
            notes.append(f"{motor}: 試験姿勢 {value:+.1f} を可動域内の {pose[motor]:+.1f} にしました")
    return pose, notes


def move_duration(start: dict[str, float], goal: dict[str, float], speed: float) -> float:
    """move_smoothly と同じコサイン曲線で、最大の角速度が speed になる長さ。"""
    distance = max(abs(goal[m] - start[m]) for m in MOTORS)
    return max(1.0, math.pi / 2 * distance / speed)


def build_plan(
    start: dict[str, float],
    pose: dict[str, float],
    joints: list[str],
    limits: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[list[dict], list[str]]:
    """区間の一覧を作る。各区間は時刻 t0〜t1 と、指令を決める値を持つ。"""
    plan: list[dict] = []
    notes = []
    t = 0.0

    def add(kind: str, name: str, duration: float, joint: str | None = None, **fields) -> None:
        nonlocal t
        segment = {"kind": kind, "name": name, "joint": joint, "t0": round(t, 6), "t1": round(t + duration, 6)}
        plan.append({**segment, **fields})
        t += duration

    add("move_in", "move_in", move_duration(start, pose, args.speed), start=start, goal=pose)
    add("settle", "settle", args.settle, goal=pose)
    for joint in joints:
        low, high = limits[joint]
        for scale in STEP_SCALES:
            offset = scale * args.step
            goal = dict(pose)
            goal[joint] = clamp(pose[joint] + offset, low, high)
            if abs(goal[joint] - pose[joint] - offset) > 1e-6:
                notes.append(f"{joint}: ステップ {offset:+g} 度を可動域で {goal[joint] - pose[joint]:+.1f} 度に縮めました")
            add("step", f"{joint}/step{offset:+g}", args.dwell, joint, goal=goal)
        if args.chirp_time > 0:
            amp = max(0.0, min(args.chirp_amp, pose[joint] - low, high - pose[joint]))
            if amp < args.chirp_amp:
                notes.append(f"{joint}: チャープの振幅を可動域で {amp:.1f} 度に縮めました")
            add(
                "chirp",
                f"{joint}/chirp",
                args.chirp_time,
                joint,
                goal=pose,
                amp=amp,
                f0=args.chirp_f0,
                f1=args.chirp_f1,
                ramp=CHIRP_RAMP_S,
            )
        add("hold", f"{joint}/hold", args.settle, joint, goal=pose)
    add("move_out", "move_out", move_duration(pose, start, args.speed), start=pose, goal=start)
    return plan, notes


def command_at(segment: dict, t: float) -> dict[str, float]:
    """区間の中の時刻 t での、全関節の指令。"""
    u = t - segment["t0"]
    duration = segment["t1"] - segment["t0"]
    if segment["kind"] in ("move_in", "move_out"):
        ratio = min(u / duration, 1.0)
        eased = (1 - math.cos(math.pi * ratio)) / 2
        return {m: segment["start"][m] + (segment["goal"][m] - segment["start"][m]) * eased for m in MOTORS}
    command = dict(segment["goal"])
    if segment["kind"] == "chirp":
        # 周波数が f0 から f1 へ線形に上がる正弦波。位相は周波数の積分
        phase = segment["f0"] * u + (segment["f1"] - segment["f0"]) * u * u / (2 * duration)
        envelope = max(0.0, min(1.0, u / segment["ramp"], (duration - u) / segment["ramp"]))
        command[segment["joint"]] += segment["amp"] * envelope * math.sin(2 * math.pi * phase)
    return command


def print_plan(
    start: dict[str, float],
    pose: dict[str, float],
    joints: list[str],
    plan: list[dict],
    notes: list[str],
    args: argparse.Namespace,
) -> None:
    print("開始姿勢: " + ", ".join(f"{m} {start[m]:+.1f}" for m in MOTORS), flush=True)
    print("試験姿勢: " + ", ".join(f"{m} {pose[m]:+.1f}" for m in MOTORS), flush=True)
    print(f"動かす関節: {', '.join(joints)}", flush=True)
    text = f"ステップ ±{args.step:g} 度と +{2 * args.step:g} 度を {args.dwell:g} 秒ずつ"
    if args.chirp_time > 0:
        text += f"、チャープ ±{args.chirp_amp:g} 度 {args.chirp_f0:g}→{args.chirp_f1:g}Hz を {args.chirp_time:g} 秒"
    print(f"{text}。合計 {plan[-1]['t1']:.0f} 秒、{args.hz:g}Hz で記録します。", flush=True)
    for note in notes:
        print(f"  注意: {note}", flush=True)


def write_dry_run(path: Path, plan: list[dict], hz: float) -> int:
    """ロボットなしで、計画どおりの指令だけを実機と同じ列で書く。"""
    index = 0
    ticks = int(plan[-1]["t1"] * hz)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for tick in range(ticks):
            t = tick / hz
            while plan[index]["t1"] <= t:
                index += 1
            segment = plan[index]
            command = command_at(segment, t)
            row = {
                "t_cmd": round(t, 4),
                "t_obs": round(t, 4),
                "seg": index,
                "segment": segment["name"],
                "joint": segment["joint"] or "",
            }
            row.update({f"cmd.{m}": round(command[m], 3) for m in MOTORS})
            writer.writerow(row)
    return ticks


def read_positions(follower: SO101Follower) -> dict[str, float]:
    return follower.bus.sync_read("Present_Position", num_retry=2)


def read_settings(bus) -> dict[str, dict[str, int]]:
    """比べるときの前提になるサーボの設定値を読む。"""
    settings = {}
    for name in SERVO_SETTINGS:
        try:
            settings[name] = bus.sync_read(name, normalize=False, num_retry=2)
        except Exception as error:  # 読めない項目があっても試験は続ける
            print(f"  {name} を読めませんでした: {error}", flush=True)
    return settings


def read_extra(bus, row: dict, tick: int) -> bool:
    """速度、負荷、電流と、ときどき電圧と温度を row に書く。読めなかったら False。"""
    try:
        velocity = bus.sync_read("Present_Velocity", normalize=False, num_retry=1)
        load = bus.sync_read("Present_Load", normalize=False, num_retry=1)
        current = bus.sync_read("Present_Current", normalize=False, num_retry=1)
        slow = tick % SLOW_READ_EVERY == 0
        if slow:
            voltage = bus.sync_read("Present_Voltage", normalize=False, num_retry=1)
            temperature = bus.sync_read("Present_Temperature", normalize=False, num_retry=1)
    except OSError:
        return False
    for motor in MOTORS:
        row[f"vel.{motor}"] = round(velocity[motor] * VELOCITY_DEG_PER_UNIT, 2)
        row[f"load.{motor}"] = round(load[motor] * LOAD_PERCENT_PER_UNIT, 1)
        # 符号ビットが立つことがあるので大きさだけを記録する（teleop_power_log.py と同じ）
        row[f"cur_a.{motor}"] = round((abs(current[motor]) & 0x7FFF) * CURRENT_A_PER_UNIT, 3)
        if slow:
            row[f"volt.{motor}"] = voltage[motor] / 10
            row[f"temp.{motor}"] = temperature[motor]
    return True


def check_tracking(
    present: dict[str, float], sent: dict[str, float], t: float, over_since: dict[str, float], limit: float
) -> None:
    """指令とのずれが limit を ABORT_AFTER_S 秒超え続けたら止める。どこかにぶつかったときの保護。"""
    for motor in MOTORS:
        if abs(present[motor] - sent[f"{motor}.pos"]) <= limit:
            over_since.pop(motor, None)
            continue
        since = over_since.setdefault(motor, t)
        if t - since >= ABORT_AFTER_S:
            raise RuntimeError(f"{motor} が指令から {limit:g} 度以上ずれたままです。止めます。")


def run(
    follower: SO101Follower,
    plan: list[dict],
    writer: csv.DictWriter,
    csv_file,
    args: argparse.Namespace,
    stats: dict,
) -> None:
    """計画どおりに指令を送りながら記録する。行数と周期（ms）は途中で止まっても stats に残る。"""
    period = 1.0 / args.hz
    total = plan[-1]["t1"]
    bus = follower.bus
    periods = stats["periods"]
    index = 0
    tick = 0
    failures = 0
    over_since: dict[str, float] = {}
    last_loop = None
    print(f"{0.0:6.1f}s  {plan[0]['name']}", flush=True)
    t_start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        t = loop_start - t_start
        if t >= total:
            return
        if last_loop is not None:
            periods.append((loop_start - last_loop) * 1000)
        last_loop = loop_start
        while plan[index]["t1"] <= t:
            index += 1
            print(f"{t:6.1f}s  {plan[index]['name']}", flush=True)
        segment = plan[index]

        command = command_at(segment, t)
        sent = follower.send_action({f"{m}.pos": command[m] for m in MOTORS})
        row = {
            "t_cmd": round(time.perf_counter() - t_start, 4),
            "seg": index,
            "segment": segment["name"],
            "joint": segment["joint"] or "",
            "loop_ms": round(periods[-1], 1) if periods else "",
        }
        row.update({f"cmd.{m}": round(sent[f"{m}.pos"], 3) for m in MOTORS})
        try:
            read_start = time.perf_counter()
            present = read_positions(follower)
            row["t_obs"] = round((read_start + time.perf_counter()) / 2 - t_start, 4)
        except OSError as error:
            failures += 1
            stats["read_failures"] += 1
            row["event"] = "read_failed"
            writer.writerow(row)
            stats["rows"] += 1
            if failures >= MAX_READ_FAILURES:
                raise RuntimeError(f"位置が {failures} 回続けて読めません: {error}") from error
        else:
            failures = 0
            row.update({f"pos.{m}": round(present[m], 3) for m in MOTORS})
            if not args.position_only and not read_extra(bus, row, tick):
                stats["extra_read_failures"] += 1
                row["event"] = "extra_read_failed"
            writer.writerow(row)
            stats["rows"] += 1
            check_tracking(present, sent, t, over_since, args.abort_error)

        tick += 1
        if tick % 50 == 0:
            csv_file.flush()
        time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))


def return_to_start(follower: SO101Follower, start: dict[str, float], speed: float) -> bool:
    """開始時の姿勢へ戻す。通信エラーで戻せなかったら False（トルクを切るとアームが落ちる）。"""
    goal = {f"{m}.pos": start[m] for m in MOTORS}
    for attempt in range(1, 4):
        try:
            if not move_smoothly(follower, goal, speed, timeout_s=30.0):
                print("開始時の姿勢に完全には戻りませんでした。手で戻してください。", flush=True)
            return True
        except Exception as error:  # 読み取りの失敗は一時的なことが多いので、やり直す
            print(f"復帰中にエラー（{attempt} 回目）: {error}", flush=True)
    return False


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    joints = parse_joints(args.joints)
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    path = args.out or Path("outputs/step_response") / (f"{stamp}_dry.csv" if args.dry_run else f"{stamp}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".json")

    # キャリブレーションはファイルから読まれるので、接続しなくても可動域がわかる
    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    limits = {m: safe_limits(follower, m, args.margin) for m in MOTORS}
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "lerobot_version": lerobot.__version__,
        "motors": MOTORS,
        "joints": joints,
        "limits": limits,
        "calibration": {m: dataclasses.asdict(c) for m, c in follower.bus.calibration.items()},
        "units": {
            "t_cmd": "指令を書き終えた時刻（秒）",
            "t_obs": "位置を読んだ時刻（秒、読み取りの中点）",
            "seg": "plan の何番目の区間か",
            "cmd, pos": "LeRobot の正規化値（腕の関節は度、gripper は 0〜100）",
            "vel": f"度/秒（Present_Velocity × {VELOCITY_DEG_PER_UNIT:.6f}）",
            "load": "最大トルクに対する %（Present_Load × 0.1、符号つき）",
            "cur_a": "A（Present_Current の大きさ × 0.0065）",
            "volt": "V",
            "temp": "℃",
        },
    }

    if args.dry_run:
        start = dict(DRY_RUN_START)
        pose, notes = test_pose(args, start, limits)
        plan, plan_notes = build_plan(start, pose, joints, limits, args)
        print_plan(start, pose, joints, plan, notes + plan_notes, args)
        rows = write_dry_run(path, plan, args.hz)
        meta.update(start=start, pose=pose, plan=plan, result={"stopped": "dry_run", "rows": rows})
        write_meta(meta_path, meta)
        print(f"計画を書きました: {path}", flush=True)
        return 0

    follower.connect(calibrate=False)
    started = False
    stopped = "not_started"
    stats = {"rows": 0, "read_failures": 0, "extra_read_failures": 0, "periods": []}
    start: dict[str, float] = {}
    try:
        start = read_positions(follower)
        pose, notes = test_pose(args, start, limits)
        plan, plan_notes = build_plan(start, pose, joints, limits, args)
        print_plan(start, pose, joints, plan, notes + plan_notes, args)
        meta.update(start=start, pose=pose, plan=plan, servo_settings=read_settings(follower.bus))
        write_meta(meta_path, meta)
        if not args.yes:
            input("アームの前と左右を空けて Enter で開始します（Ctrl+C で中止）: ")
        started = True
        with path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                run(follower, plan, writer, csv_file, args, stats)
                stopped = "completed"
            except KeyboardInterrupt:
                stopped = "interrupted"
                print("\n停止します。", flush=True)
            except RuntimeError as error:
                stopped = f"aborted: {error}"
                print(f"\n{error}", flush=True)
    except KeyboardInterrupt:
        print("\n開始前に中止しました。", flush=True)
    finally:
        returned = True
        if started:
            # 計画どおり終わっていればもう開始時の姿勢にいる。途中で止めた場合もここで戻す
            print("開始時の姿勢へ戻します。", flush=True)
            returned = return_to_start(follower, start, args.speed)
        try:
            if returned:
                follower.disconnect()
            else:
                # 試験姿勢のままトルクを切るとアームが落ちるので、入れたままポートだけ閉じる
                follower.bus.disconnect(disable_torque=False)
                print("戻せなかったのでトルクを入れたまま終了します。アームを支えてから電源を切るか、"
                      "reset_pose.py で戻してください。", flush=True)
        except Exception as error:  # 通信が切れていると切断も失敗する
            print(f"切断中にエラー: {error}", flush=True)

    if not started:
        return 1
    periods = stats.pop("periods")
    result = {"stopped": stopped, **stats}
    if periods:
        ordered = sorted(periods)
        result.update(
            period_ms_median=round(statistics.median(ordered), 2),
            period_ms_p90=round(ordered[int(len(ordered) * 0.9)], 2),
            period_ms_max=round(ordered[-1], 2),
        )
        print(f"周期: 中央値 {result['period_ms_median']:.1f}ms、90% {result['period_ms_p90']:.1f}ms", flush=True)
    meta["result"] = result
    write_meta(meta_path, meta)
    print(f"記録: {path}", flush=True)
    print(f"比べるとき: uv run python scripts/compare_step_response.py {path}", flush=True)
    return 0 if stopped == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
