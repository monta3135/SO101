#!/usr/bin/env python3
"""サーボの ID と実際の関節の対応を、手で動かして確認する。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_PRESENT_POSITION = 56
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
JOINTS = {
    1: "shoulder_pan（ベースの回転）",
    2: "shoulder_lift（肩の上下）",
    3: "elbow_flex（肘）",
    4: "wrist_flex（手首の上下）",
    5: "wrist_roll（手首の回転）",
    6: "gripper（グリッパー）",
}
MIN_DELTA = 20


def read_positions(packet: PacketHandler, port: PortHandler) -> dict[int, int]:
    positions: dict[int, int] = {}
    for motor_id in MOTOR_IDS:
        value, result, _ = packet.read2ByteTxRx(port, motor_id, ADDR_PRESENT_POSITION)
        if result != COMM_SUCCESS:
            raise RuntimeError(f"ID {motor_id} の位置を読めませんでした")
        positions[motor_id] = value
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description="サーボ ID と関節の対応を確認する")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mapping.json"),
        help="確認結果の保存先",
    )
    args = parser.parse_args()

    port = PortHandler(args.port)
    try:
        if not port.openPort():
            print(f"ポートを開けません: {args.port}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"ポートを開けません: {args.port} ({exc})", file=sys.stderr)
        return 1

    port.setBaudRate(args.baud)
    packet = PacketHandler(0)

    print("サーボのトルクを切った状態で確認します。")
    print("アームは手で動かせるようにしておいてください。\n")

    detected: dict[int, int] = {}

    try:
        for expected_id, joint_name in JOINTS.items():
            for attempt in range(3):
                print(f"{joint_name} だけを少し動かしたまま、Enter を押してください。")
                if attempt > 0:
                    print("動きが小さすぎました。もう一度、大きめに動かしてください。")

                before = read_positions(packet, port)
                input()
                after = read_positions(packet, port)

                deltas = {
                    motor_id: abs(after[motor_id] - before[motor_id])
                    for motor_id in MOTOR_IDS
                }
                moved_id = max(deltas, key=deltas.get)

                if deltas[moved_id] < MIN_DELTA:
                    print(f"動きを検出できませんでした（最大 {deltas[moved_id]}）。\n")
                    continue

                detected[expected_id] = moved_id
                print(
                    f"  -> 動いたのは ID {moved_id} でした"
                    f"（変化量 {deltas[moved_id]}）\n"
                )
                break
            else:
                print(f"{joint_name} の対応を確認できませんでした。\n")
                detected[expected_id] = -1
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    finally:
        port.closePort()

    print("\n== 確認結果 ==")
    print(f"{'関節':<32} {'期待ID':>6} {'実際ID':>6}  判定")
    all_ok = True
    for expected_id, joint_name in JOINTS.items():
        actual = detected.get(expected_id, -1)
        if actual == expected_id:
            verdict = "OK"
        elif actual == -1:
            verdict = "未確認"
            all_ok = False
        else:
            verdict = "NG"
            all_ok = False
        actual_text = str(actual) if actual != -1 else "-"
        print(f"{joint_name:<32} {expected_id:>6} {actual_text:>6}  {verdict}")

    reversed_pairs = [
        (expected, actual)
        for expected, actual in detected.items()
        if actual != -1 and actual != expected
    ]
    result = {
        "port": args.port,
        "detected": {str(joint_id): actual for joint_id, actual in detected.items()},
        "ok": all_ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\n結果を保存しました: {args.output}")

    if reversed_pairs:
        print("\nID と関節の対応がずれています。")
        print("ID はサーボ本体に保存されているため、ケーブルを挿し替えても ID は変わりません。")
        print("対応を直す方法は2つあります。")
        print("  1. サーボ本体を正しいジョイントに付け替える")
        print("  2. その関節のサーボだけを基板につなぎ、ID を振り直す")
        print("フォロワーは全サーボが同じギア比なので、ID の振り直しでも対応できます。")
    elif all_ok:
        print("\nすべての関節と ID が一致しています。")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
