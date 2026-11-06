#!/usr/bin/env python3
"""SO-101 のサーボバスをスキャンして、ID、位置、電圧、温度を表示する。"""

from __future__ import annotations

import argparse
import sys

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63

BAUDRATES = [1_000_000, 500_000, 115_200, 57_600, 38_400]


def parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids


def read_byte(packet: PacketHandler, port: PortHandler, motor_id: int, address: int):
    value, result, _ = packet.read1ByteTxRx(port, motor_id, address)
    return value if result == COMM_SUCCESS else None


def read_word(packet: PacketHandler, port: PortHandler, motor_id: int, address: int):
    value, result, _ = packet.read2ByteTxRx(port, motor_id, address)
    return value if result == COMM_SUCCESS else None


def scan(port: PortHandler, baudrate: int, ids: list[int]) -> list[tuple[int, int]]:
    packet = PacketHandler(0)
    found: list[tuple[int, int]] = []
    for motor_id in ids:
        model, result, _ = packet.ping(port, motor_id)
        if result == COMM_SUCCESS:
            found.append((motor_id, model))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="SO-101 のサーボバスをスキャンする")
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB シリアルのポート")
    parser.add_argument("--baud", type=int, default=1_000_000, help="ボーレート")
    parser.add_argument("--ids", default="1-20", help="スキャンする ID の範囲 (例: 1-6)")
    parser.add_argument(
        "--scan-baud",
        action="store_true",
        help="代表的なボーレートを順に試す",
    )
    args = parser.parse_args()

    ids = parse_ids(args.ids)
    baudrates = BAUDRATES if args.scan_baud else [args.baud]

    port = PortHandler(args.port)
    try:
        opened = port.openPort()
    except Exception as exc:  # シリアルポートが存在しない、権限がないなどの場合
        print(f"ポートを開けません: {args.port} ({exc})", file=sys.stderr)
        return 1

    if not opened:
        print(f"ポートを開けません: {args.port}", file=sys.stderr)
        return 1

    found_any = False
    try:
        for baudrate in baudrates:
            if not port.setBaudRate(baudrate):
                print(f"ボーレートを設定できません: {baudrate}", file=sys.stderr)
                continue

            found = scan(port, baudrate, ids)
            if not found:
                if args.scan_baud:
                    print(f"{baudrate:>9} bps: 応答なし")
                continue

            packet = PacketHandler(0)
            print(f"\n{args.port} @ {baudrate} bps")
            print(f"{'ID':>3}  {'モデル':>6}  {'位置':>6}  {'電圧':>6}  {'温度':>4}  {'トルク':>4}")
            for motor_id, model in found:
                position = read_word(packet, port, motor_id, ADDR_PRESENT_POSITION)
                voltage = read_byte(packet, port, motor_id, ADDR_PRESENT_VOLTAGE)
                temperature = read_byte(packet, port, motor_id, ADDR_PRESENT_TEMPERATURE)
                torque = read_byte(packet, port, motor_id, ADDR_TORQUE_ENABLE)

                position_text = str(position) if position is not None else "-"
                voltage_text = f"{voltage / 10:.1f}V" if voltage is not None else "-"
                temperature_text = f"{temperature}C" if temperature is not None else "-"
                torque_text = ("ON" if torque else "OFF") if torque is not None else "-"
                print(
                    f"{motor_id:>3}  {model:>6}  {position_text:>6}  "
                    f"{voltage_text:>6}  {temperature_text:>4}  {torque_text:>4}"
                )

            found_any = True
            break
    finally:
        port.closePort()

    if not found_any:
        print(
            "サーボが見つかりません。配線、電源、ボーレートを確認してください。",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
