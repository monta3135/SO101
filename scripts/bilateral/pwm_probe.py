#!/usr/bin/env python3
"""フェーズ0: STS3215 を PWM モードにして、1関節だけ PWM と応答の対応を実測する。

調べること:
  - PWM の方向ビット（bit10 か bit11）
  - Torque_Limit が PWM 出力に上限をかけるか
  - Present_Current の符号
  - PWM 値と回転速度の対応

--read-only はモード変更もトルク ON もせず、状態を読むだけ（接続の確認用）。
それ以外は Operating_Mode を 2 (PWM) にして短いパルスを打ち、終了時（例外と Ctrl+C を含む）は
PWM を 0 にしてトルクを OFF にし、Operating_Mode を 0 に戻す。戻らないと teleop.sh や方策の実行が壊れる。
EEPROM の書き込みは開始時と終了時の1回ずつ。PWM モードでは位置リミットが効かないので、
開始時の位置から --limit 度を超えたら止める。パルスは短く、電流と温度も監視する。
記録は outputs/bilateral/pwm_probe_<arm>_<joint>_<日時>.csv。

使い方: uv run scripts/bilateral/pwm_probe.py --arm follower --joint wrist_roll [--read-only]
動かす関節の周りを空けて、手で電源を切れる位置にいること。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

JOINT_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
PORTS = {"leader": "/dev/so101-leader", "follower": "/dev/so101-follower"}
STEPS_PER_DEG = 4096 / 360
MAX_TEMP_C = 55
MAX_CURRENT_UNITS = 400  # 約 2.6 A。1 単位 6.5 mA
SAFE_PWM = 150  # 0〜1000 の指令値。最初の試験はこれ以下
FIELDS = ["t", "label", "pwm_written", "pos", "vel", "current", "load", "temp"]


def encode_pwm(value: int, dir_bit: int) -> int:
    return (abs(value) | ((1 << dir_bit) if value < 0 else 0)) if value else 0


class Probe:
    def __init__(self, arm: str, joint: str, limit_deg: float):
        motor = Motor(JOINT_IDS[joint], "sts3215", MotorNormMode.DEGREES)
        self.joint = joint
        self.bus = FeetechMotorsBus(port=PORTS[arm], motors={joint: motor})
        self.limit = limit_deg * STEPS_PER_DEG
        self.rows: list[dict] = []
        self.t0 = time.perf_counter()
        self.start_pos = 0
        self.pwm_mode = False

    def read(self, name: str) -> int:
        return self.bus.read(name, self.joint, normalize=False)

    def write(self, name: str, value: int) -> None:
        self.bus.write(name, self.joint, value, normalize=False)

    def snapshot(self, label: str, pwm: int) -> dict:
        row = {
            "t": round(time.perf_counter() - self.t0, 4),
            "label": label,
            "pwm_written": pwm,
            "pos": self.read("Present_Position"),
            "vel": self.read("Present_Velocity"),
            "current": self.read("Present_Current"),
            "load": self.read("Present_Load"),
            "temp": self.read("Present_Temperature"),
        }
        self.rows.append(row)
        return row

    def pulse(self, label: str, pwm_raw: int, duration: float) -> None:
        """生の Goal_Time 値を duration 秒だけ書き続け、0 に戻す。限界を超えたら例外で止める。"""
        self.write("Goal_Time", pwm_raw)
        end = time.perf_counter() + duration
        try:
            while time.perf_counter() < end:
                row = self.snapshot(label, pwm_raw)
                self.check(row)
                self.write("Goal_Time", pwm_raw)  # 通信が切れても保持されるが、周期的に書いて確実にする
        finally:
            self.write("Goal_Time", 0)
        time.sleep(0.4)
        self.snapshot(label + "_after", 0)

    def check(self, row: dict) -> None:
        if abs(row["pos"] - self.start_pos) > self.limit:
            raise RuntimeError(f"位置が開始から {self.limit / STEPS_PER_DEG:.0f} 度を超えた")
        if row["temp"] > MAX_TEMP_C:
            raise RuntimeError(f"温度が {row['temp']} ℃")
        if abs(row["current"]) > MAX_CURRENT_UNITS:
            raise RuntimeError(f"電流が {row['current']} 単位")

    def enter_pwm(self) -> None:
        self.write("Torque_Enable", 0)
        self.write("Lock", 0)
        self.write("Operating_Mode", OperatingMode.PWM.value)
        self.write("Lock", 1)
        self.pwm_mode = True
        self.write("Goal_Time", 0)
        self.write("Torque_Enable", 1)

    def leave_pwm(self) -> None:
        errors = []
        for name, value in (("Goal_Time", 0), ("Torque_Enable", 0), ("Torque_Limit", 1000)):
            try:
                self.write(name, value)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        if self.pwm_mode:
            for _ in range(3):  # モードを戻せないと他のスクリプトが壊れるので何度か試す
                try:
                    self.write("Lock", 0)
                    self.write("Operating_Mode", OperatingMode.POSITION.value)
                    self.write("Lock", 1)
                    if self.read("Operating_Mode") == OperatingMode.POSITION.value:
                        self.pwm_mode = False
                        break
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
        if self.pwm_mode:
            print("!! Operating_Mode を 0 に戻せなかった。手動で戻すこと:", errors)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        print("記録:", path)


def summarize(rows: list[dict], label: str) -> str:
    sel = [r for r in rows if r["label"] == label]
    if not sel:
        return f"{label}: 記録なし"
    moved = (sel[-1]["pos"] - sel[0]["pos"]) / STEPS_PER_DEG
    cur = [r["current"] for r in sel]
    vel = [r["vel"] for r in sel]
    return (
        f"{label:>18}: 移動 {moved:+6.1f} 度  速度 {min(vel):+5d}..{max(vel):+5d}  "
        f"電流 {min(cur):+5d}..{max(cur):+5d}  サンプル {len(sel)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=PORTS, required=True)
    ap.add_argument("--joint", choices=JOINT_IDS, required=True)
    ap.add_argument("--limit", type=float, default=25.0, help="開始位置からの許容角度（度）")
    ap.add_argument("--pwm", type=int, default=SAFE_PWM, help="試験パルスの大きさ（0〜1000）")
    ap.add_argument("--duration", type=float, default=0.15, help="1パルスの長さ（秒）")
    ap.add_argument("--read-only", action="store_true")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="方向ビット bit10 で PWM を 50..400 まで上げ、各段を + と - の交互に打って速度と電流を表にする",
    )
    args = ap.parse_args()
    if args.pwm > 300 and not args.sweep:
        ap.error("--pwm は 300 以下にすること（方向ビットの確認が済むまで）")

    probe = Probe(args.arm, args.joint, args.limit)
    probe.bus.connect()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs/bilateral") / f"pwm_probe_{args.arm}_{args.joint}_{stamp}.csv"
    try:
        probe.start_pos = probe.read("Present_Position")
        print(
            f"{args.arm}/{args.joint}: pos={probe.start_pos} mode={probe.read('Operating_Mode')} "
            f"torque_limit={probe.read('Torque_Limit')} max_torque_limit={probe.read('Max_Torque_Limit')} "
            f"temp={probe.read('Present_Temperature')} current={probe.read('Present_Current')} "
            f"voltage={probe.read('Present_Voltage') / 10:.1f}V"
        )
        if args.read_only:
            return 0
        input("関節の周りが空いているのを確認して Enter（Ctrl+C で中止）: ")
        pos_before = probe.read("Present_Position")
        offset = probe.read("Homing_Offset")
        probe.enter_pwm()
        time.sleep(0.1)
        print(
            f"Homing_Offset={offset} 位置モード pos={pos_before} → PWM モード pos={probe.read('Present_Position')}"
        )
        probe.start_pos = probe.read("Present_Position")  # Enter を待つ間に手で動いても基準を取り直す
        if args.sweep:
            probe.write("Torque_Limit", 1000)
            print("  PWM   +側 速度(平均)  +側 電流(最大)   -側 速度(平均)  -側 電流(最大)  移動(+,-)")
            for level in (50, 100, 150, 200, 250, 300, 400):
                res = {}
                for sign, name in ((1, "+"), (-1, "-")):
                    label = f"sweep{level}{name}"
                    probe.pulse(label, encode_pwm(sign * level, 10), 0.3)
                    sel = [r for r in probe.rows if r["label"] == label][5:]  # 立ち上がりを除く
                    res[name] = (
                        sum(r["vel"] for r in sel) / max(len(sel), 1),
                        max(r["current"] for r in sel),
                        (sel[-1]["pos"] - sel[0]["pos"]) / STEPS_PER_DEG,
                    )
                print(
                    f"{level:5d}  {res['+'][0]:12.0f}  {res['+'][1]:12d}   {res['-'][0]:12.0f}  "
                    f"{res['-'][1]:12d}   {res['+'][2]:+.1f},{res['-'][2]:+.1f}"
                )
            return 0
        # 方向ビットが bit11 か bit10 かが不明なので、bit11 を先に弱く試す。
        # bit10 が方向の場合、bit11 側の値は大きな大きさとして解釈されうるので Torque_Limit を絞る。
        probe.write("Torque_Limit", 300)
        for dir_bit in (11, 10):
            for sign, name in ((1, "+"), (-1, "-")):
                label = f"bit{dir_bit}{name}"
                probe.pulse(label, encode_pwm(sign * args.pwm, dir_bit), args.duration)
                print(summarize(probe.rows, label))
                # 動かなかったり想定外に動いたら先へ進む前に確認する
            if input(f"bit{dir_bit} の ± で逆向きに同程度動いた？ [y/N]: ").lower() == "y":
                print(f"方向ビット = bit{dir_bit}")
                break
        # Torque_Limit の効果: 同じ PWM で制限を変えて速度を比べる
        for limit in (100, 300, 1000):
            probe.write("Torque_Limit", limit)
            probe.pulse(f"limit{limit}", args.pwm, args.duration)
            print(summarize(probe.rows, f"limit{limit}"))
        probe.write("Torque_Limit", 1000)
    finally:
        probe.leave_pwm()
        if probe.rows:
            probe.save(out)
        probe.bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
