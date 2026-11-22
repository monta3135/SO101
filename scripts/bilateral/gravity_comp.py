#!/usr/bin/env python3
"""フェーズ3の最初の目標: 重力と摩擦の補償だけをかけて、手で軽く動かせることを確かめる。

対象の関節だけ PWM モードにして、毎周期
    pwm = gain·g(θ) + friction_gain·Fc·tanh(ω / ω0)
を書く。g(θ) = A·cosθ + B·sinθ は identify_gravity.py の結果（calibration/bilateral/<arm>.json）。
他の関節は開始姿勢に固定（位置モードのトルク ON）。重力項は他の関節がその姿勢のときの値なので、
対象の関節の姿勢を変えても他の関節は動かさない。

g(θ) は測った角度の範囲（± --margin 度の外側）では端の値で一定にする（外挿しない）。
摩擦補償は動いている向きへ押すので、強くかけると手を離しても暴走する。--friction-gain は小さく始める（0〜0.5）。
暴走の防止: PWM は ±--pwm-max に制限し、速度が --max-vel を超えたら止める。
開始位置から --range 度を超えたら止める。温度と電流も監視する。
終了時（例外と Ctrl+C を含む）は PWM を 0、全関節のトルクを OFF、Operating_Mode を 0 に戻す。
記録は outputs/bilateral/gravity_comp_<arm>_<joint>_<日時>.csv。

使い方: uv run scripts/bilateral/gravity_comp.py --arm follower --joint wrist_flex
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path

import numpy as np
from bus import JOINTS
from identify_gravity import Holder
from lerobot.motors.feetech import OperatingMode

STEPS_PER_DEG = 4096 / 360
OMEGA0 = 60.0  # tanh の幅（step/s）。これより遅いと摩擦補償は比例的に弱まる


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("follower", "leader"), required=True)
    ap.add_argument("--joint", choices=JOINTS, required=True)
    ap.add_argument("--gravity-gain", type=float, default=1.0, help="重力項に掛ける倍率（まず 0.8 くらいから）")
    ap.add_argument("--friction-gain", type=float, default=0.0, help="摩擦補償の倍率（0〜0.5）")
    ap.add_argument("--pwm-max", type=int, default=150)
    ap.add_argument("--max-vel", type=float, default=1500.0, help="これを超えたら止める（step/s）")
    ap.add_argument("--range", type=float, default=60.0, help="開始位置からこの角度を超えたら止める（度）")
    ap.add_argument("--margin", type=float, default=10.0, help="重力を測った範囲の外へこの角度まで出てよい")
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    params = json.loads(Path(f"calibration/bilateral/{args.arm}.json").read_text())[args.joint]
    grav = params["gravity"]
    A, B = grav["A"], grav["B"]
    measured = [p["deg"] for p in grav["points"]]
    lo, hi = min(measured) - args.margin, max(measured) + args.margin
    fc = params["Fc"]
    print(f"g(θ)={A:+.1f}cosθ{B:+.1f}sinθ  測った範囲 {min(measured):.0f}〜{max(measured):.0f} 度  Fc={fc:.1f}")

    h = Holder(args.arm, args.joint, args.pwm_max, args.range)
    h.arm.connect()
    pwm_mode = False
    rows: list[list[float]] = []
    out = Path("outputs/bilateral") / (
        f"gravity_comp_{args.arm}_{args.joint}_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    try:
        s = h.arm.read_state()
        print(f"{args.arm}/{args.joint}: 位置モード 現在 {(s.pos[h.j] - h.mid) * 360 / 4095:+.1f} 度")
        print("他の関節は開始姿勢に固定（トルク ON）。対象の関節は重力・摩擦の補償だけで動かせるようになる。")
        input("手で関節を支えたまま Enter（Ctrl+C で中止）: ")
        for i, name in enumerate(JOINTS):
            if name != args.joint:
                h.arm.bus.write("Goal_Position", name, s.pos[i], normalize=False)
                h.arm.bus.write("Torque_Enable", name, 1, normalize=False)
        h.arm.set_mode(OperatingMode.PWM, [args.joint])
        pwm_mode = True
        h.arm.write_pwm_one(args.joint, 0)
        h.arm.bus.write("Torque_Enable", args.joint, 1, normalize=False)
        time.sleep(0.05)
        h.last_raw = h.arm.read_state().pos[h.j]
        h.pos = h.start = float(h.last_raw)
        print(f"開始 {h.to_deg(h.pos):+.1f} 度。{args.duration:.0f} 秒間、手で動かしてみる。")
        t0 = time.perf_counter()
        over = 0
        last_print = -1.0
        while (now := time.perf_counter() - t0) < args.duration:
            st = h.arm.read_state()
            raw = st.pos[h.j]
            h.pos += (raw - h.last_raw + 2048) % 4096 - 2048
            h.last_raw = raw
            if abs(h.pos - h.start) > h.abort:
                raise RuntimeError(f"開始から {args.range:.0f} 度を超えた")
            vel = float(st.vel[h.j])
            if abs(vel) > args.max_vel:
                raise RuntimeError(f"速度 {vel:.0f} step/s が上限を超えた")
            hot = st.temp[h.j] > 55 or st.current[h.j] > 400
            over = over + 1 if hot else 0
            if over >= 2:
                raise RuntimeError(f"温度 {st.temp[h.j]} 電流 {st.current[h.j]}")
            deg = h.to_deg(h.pos)
            th = math.radians(min(max(deg, lo), hi))
            g = A * math.cos(th) + B * math.sin(th)
            pwm = args.gravity_gain * g + args.friction_gain * fc * math.tanh(vel / OMEGA0)
            pwm = float(np.clip(pwm, -args.pwm_max, args.pwm_max))
            h.arm.write_pwm_one(args.joint, int(round(pwm)))
            rows.append([round(now, 4), round(deg, 2), vel, round(pwm, 1), st.current[h.j]])
            if now - last_print >= 0.5:  # 手を離したときに保持するか・流れるかを見られるように、半秒ごとに表示する
                last_print = now
                print(f"  t={now:5.1f}s  角度 {deg:+6.1f} 度  速度 {vel:+6.0f} step/s  PWM {pwm:+6.1f}  電流 {st.current[h.j]}")
    finally:
        try:
            h.arm.write_pwm_one(args.joint, 0)
            for name in JOINTS:
                h.arm.bus.write("Torque_Enable", name, 0, normalize=False)
        finally:
            if pwm_mode:
                for _ in range(3):
                    try:
                        h.arm.set_mode(OperatingMode.POSITION, [args.joint])
                        if h.arm.bus.read("Operating_Mode", args.joint, normalize=False) == 0:
                            break
                    except Exception as e:  # noqa: BLE001
                        print("モード復帰に失敗、再試行:", e)
                else:
                    print("!! Operating_Mode を 0 に戻せなかった。手動で戻すこと")
            h.arm.disconnect()
        if rows:
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "deg", "vel", "pwm", "current"])
                w.writerows(rows)
            arr = np.array(rows)
            print(f"記録 {out}（{len(rows)} 周期、平均 {len(rows) / arr[-1, 0]:.0f} Hz）")
            print(f"角度 {arr[:, 1].min():+.1f}〜{arr[:, 1].max():+.1f} 度、最大速度 {np.abs(arr[:, 2]).max():.0f} step/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
