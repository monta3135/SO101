#!/usr/bin/env python3
"""フォロワーの電圧、電流、エラー状態を CSV に記録しながらテレオペする。

切れたときは、応答が戻るまで待ってトルクの状態を読み、電源の瞬断か通信の途切れかを切り分ける。
サーボは電源が落ちるとトルク OFF で再起動するため、トルクが OFF に戻っていれば瞬断と判断できる。
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import os
import time
from pathlib import Path

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from serial import SerialException

IDS = range(1, 7)
# STS3215 の Present_Current の単位
CURRENT_A_PER_UNIT = 0.0065
FIELDNAMES = [
    "time_s",
    "event",
    "loop_ms",
    "min_voltage",
    "total_current_a",
    *[f"voltage_{i}" for i in IDS],
    *[f"current_a_{i}" for i in IDS],
    *[f"status_{i}" for i in IDS],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="電源を記録しながらテレオペする")
    parser.add_argument("--follower-port", default="/dev/so101-follower")
    parser.add_argument("--leader-port", default="/dev/so101-leader")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--log-hz", type=float, default=20.0, help="電圧と電流を読む頻度")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        help="1回の送信で動く最大角度 [deg]。未指定なら teleop.sh と同じく制限なし",
    )
    parser.add_argument("--probe-timeout", type=float, default=30.0, help="切れた後に応答を待つ秒数")
    parser.add_argument("--out", type=Path, help="CSV の保存先")
    return parser.parse_args()


def read_sample(bus) -> dict[str, float]:
    voltage = bus.sync_read("Present_Voltage", normalize=False)
    current = bus.sync_read("Present_Current", normalize=False)
    status = bus.sync_read("Status", normalize=False)

    row: dict[str, float] = {}
    for motor, motor_id in zip(bus.motors, IDS):
        row[f"voltage_{motor_id}"] = voltage[motor] / 10
        # 符号ビットが立つことがあるので大きさだけを記録する
        row[f"current_a_{motor_id}"] = round((abs(current[motor]) & 0x7FFF) * CURRENT_A_PER_UNIT, 3)
        row[f"status_{motor_id}"] = status[motor]
    row["min_voltage"] = min(voltage.values()) / 10
    row["total_current_a"] = round(sum(row[f"current_a_{i}"] for i in IDS), 3)
    return row


def probe_recovery(bus, timeout_s: float) -> tuple[float, dict[str, int]] | None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        try:
            torque = bus.sync_read("Torque_Enable", normalize=False)
        except OSError:
            time.sleep(0.05)
            continue
        return time.perf_counter() - start, torque
    return None


def verdict(recovery: tuple[float, dict[str, int]] | None, usb_present: bool) -> str:
    if not usb_present:
        return "USB ごと切れています。USB ケーブルと基板への給電を確認してください。"
    if recovery is None:
        return "応答が戻りません。電源が落ちたままか、サーボケーブルが外れています。"
    elapsed, torque = recovery
    if not any(torque.values()):
        return f"{elapsed:.2f} 秒後に全軸がトルク OFF で応答しました。電源の瞬断でサーボが再起動しています。"
    return f"{elapsed:.2f} 秒後にトルク ON のまま応答しました。電源は落ちておらず、通信だけが途切れています。"


def teleop_loop(follower, leader, writer, csv_file, args: argparse.Namespace, recent) -> tuple[str, Exception] | None:
    period = 1 / args.fps
    log_period = 1 / args.log_hz
    start = time.perf_counter()
    last_log = 0.0
    last_print = 0.0
    loop_ms = 0.0

    while True:
        loop_start = time.perf_counter()
        try:
            stage = "follower_read"
            follower.get_observation()
            stage = "leader_read"
            action = leader.get_action()
            stage = "follower_write"
            follower.send_action(action)
        except OSError as exc:
            return stage, exc

        now = time.perf_counter()
        if now - last_log >= log_period:
            last_log = now
            row = {"time_s": round(now - start, 3), "loop_ms": round(loop_ms, 1)}
            try:
                row.update(read_sample(follower.bus))
            except OSError:
                row["event"] = "sample_failed"
            writer.writerow(row)
            csv_file.flush()
            recent.append(row)

            if "min_voltage" in row and now - last_print >= 1.0:
                last_print = now
                print(f"{row['time_s']:7.1f}s  最低 {row['min_voltage']:.1f}V  合計 {row['total_current_a']:.2f}A", flush=True)

        time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
        loop_ms = (time.perf_counter() - loop_start) * 1000


def print_summary(path: Path, recent) -> None:
    rows = [row for row in recent if "min_voltage" in row]
    if rows:
        lowest = min(rows, key=lambda row: row["min_voltage"])
        highest = max(rows, key=lambda row: row["total_current_a"])
        errors = {row[f"status_{i}"] for row in rows for i in IDS} - {0}
        print(f"直前の記録 {len(rows)} 件:")
        print(f"  最低電圧 {lowest['min_voltage']:.1f}V ({lowest['time_s']:.1f}s)")
        print(f"  最大電流 {highest['total_current_a']:.2f}A ({highest['time_s']:.1f}s)")
        print(f"  エラー状態 {sorted(errors) if errors else 'なし'}")
    print(f"記録: {path}")


def main() -> int:
    args = parse_args()
    path = args.out or Path("outputs/power_log") / f"{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id="follower",
            max_relative_target=args.max_relative_target,
        )
    )
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="leader"))
    follower.connect(calibrate=False)
    leader.connect(calibrate=False)

    # 切れる直前の 10 秒分を集計に使う
    recent = collections.deque(maxlen=int(args.log_hz * 10))
    failure = None
    try:
        with path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            print("Ctrl+C で終了します。", flush=True)
            try:
                failure = teleop_loop(follower, leader, writer, csv_file, args, recent)
            except KeyboardInterrupt:
                print("\n停止しました。")

            if failure is not None:
                stage, exc = failure
                print(f"\n切れました ({stage}): {exc}", flush=True)
                writer.writerow({"event": f"failed:{stage}"})
                csv_file.flush()
                if isinstance(exc, SerialException):
                    # サーボの電源が落ちただけならこの例外にはならない
                    print("判定: ポートの読み取りエラーです。別のプロセスが同じポートを開いたか、USB が切れています。")
                elif stage == "leader_read":
                    print("リーダー側で切れました。リーダーの 5V 電源とケーブルを確認してください。")
                else:
                    usb_present = os.path.exists(args.follower_port)
                    recovery = probe_recovery(follower.bus, args.probe_timeout) if usb_present else None
                    message = verdict(recovery, usb_present)
                    print(f"判定: {message}", flush=True)
                    writer.writerow({"event": f"verdict:{message}"})
    finally:
        print_summary(path, recent)
        for name, device in (("フォロワー", follower), ("リーダー", leader)):
            try:
                device.disconnect()
            except Exception as exc:
                print(f"{name}の切断に失敗しました: {exc}")

    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
