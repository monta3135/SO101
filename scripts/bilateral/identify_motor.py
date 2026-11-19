#!/usr/bin/env python3
"""フェーズ2: 1関節の PWM→運動モデルを同定する。

モデル（PWM 単位で表した等価トルク）:  pwm = Fc·sign(ω) + Fv·ω + Jp·dω/dt  （重力の小さい姿勢）
  Fc: クーロン摩擦（動き出す PWM）、Fv: 粘性＋逆起電力（PWM / (step/s)）、Jp: 慣性（PWM·s² / step）
  物理トルクへの換算（Kt/R）は PWM 単位のまま扱い、必要になった段階で係数を掛ける。

手順:
  1. PWM を 50..300 で ± 交互に 0.3 秒ずつ打ち、定常速度から Fc, Fv を最小二乗で求める。
     電流も記録する（電流 vs PWM から Kt/R の比を後で見る）。
  2. PWM 200 のステップ応答の立ち上がりから時定数 T = Jp/Fv を求め、Jp = Fv·T とする。
  3. 動き出す最小の PWM（不感帯）を、ゆっくり上げて測る。

対象の関節だけ PWM モードにし、他の関節は触らない（トルクの状態もそのまま）。
重力に負ける関節（shoulder_lift, elbow_flex）は支えてから行うこと。
開始位置から --limit 度を超えたらパルスを打ち切り、--abort 度で中止する。
終了時（例外と Ctrl+C を含む）は PWM を 0、トルク OFF、Operating_Mode を 0 に戻す。
結果は calibration/bilateral/<arm>.json にマージして保存する。

使い方: uv run scripts/bilateral/identify_motor.py --arm follower --joint wrist_roll
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from bus import JOINTS, ArmBus
from lerobot.motors.feetech import OperatingMode

STEPS_PER_DEG = 4096 / 360
LEVELS = (50, 100, 150, 200, 250, 300)
PULSE_S = 0.3
MAX_TEMP_C = 55
MAX_CURRENT_UNITS = 400


class Identifier:
    def __init__(self, arm: str, joint: str, limit_deg: float, abort_deg: float):
        self.arm = ArmBus(arm)
        self.joint = joint
        self.j = JOINTS.index(joint)
        self.limit = limit_deg * STEPS_PER_DEG
        self.abort = abort_deg * STEPS_PER_DEG
        self.start = 0
        self.unwrapped = 0.0
        self.last_raw = 0
        self.over_count = 0

    def sample(self) -> tuple[float, float, int, int]:
        """(時刻, 開始からの位置 step（巻き戻し補正済み）, 電流, 温度)"""
        s = self.arm.read_state()
        raw = s.pos[self.j]
        d = (raw - self.last_raw + 2048) % 4096 - 2048
        self.unwrapped += d
        self.last_raw = raw
        if abs(self.unwrapped) > self.abort:
            raise RuntimeError(f"開始から {self.abort / STEPS_PER_DEG:.0f} 度を超えた")
        # 読み取りが化けることがあるので、2回続けて超えたときだけ止める
        over = s.temp[self.j] > MAX_TEMP_C or s.current[self.j] > MAX_CURRENT_UNITS
        self.over_count = self.over_count + 1 if over else 0
        if self.over_count >= 2:
            raise RuntimeError(f"温度 {s.temp[self.j]} 電流 {s.current[self.j]}")
        return s.t, self.unwrapped, s.current[self.j], s.temp[self.j]

    def pulse(self, pwm: int, duration: float) -> dict:
        """pwm を duration 秒だけ打つ。パルス開始位置から limit を超えたら打ち切る。"""
        t0 = time.perf_counter()
        _, p0, _, _ = self.sample()
        ts, ps, cs = [], [], []
        try:
            self.arm.write_pwm_one(self.joint, pwm)
            while time.perf_counter() - t0 < duration:
                t, p, c, _ = self.sample()
                ts.append(t - t0)
                ps.append(p - p0)
                cs.append(c)
                if abs(p - p0) > self.limit:
                    break
        finally:
            self.arm.write_pwm_one(self.joint, 0)
        time.sleep(0.4)
        self.sample()
        if abs(self.unwrapped) > 4 * STEPS_PER_DEG:
            self.recenter()
        return {"t": np.array(ts), "p": np.array(ps), "c": np.array(cs), "pwm": pwm}

    def recenter(self, pwm: int = 220, timeout: float = 3.0) -> None:
        """重力などで片側へ流れた分を、開始位置の近くまで戻す。同定のデータには含めない。"""
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < timeout:
                self.sample()
                if abs(self.unwrapped) < 3 * STEPS_PER_DEG:
                    break
                # 正の PWM は raw 位置が減る向き
                self.arm.write_pwm_one(self.joint, pwm if self.unwrapped > 0 else -pwm)
        finally:
            self.arm.write_pwm_one(self.joint, 0)
        time.sleep(0.3)
        self.sample()

    @staticmethod
    def steady_velocity(r: dict, after: float = 0.1) -> float | None:
        """after 秒以降の位置の傾き（step/s）。サンプルが足りなければ None。"""
        m = r["t"] > after
        if m.sum() < 8:
            return None
        # 正の PWM は raw 位置が減る向き。PWM と同じ向きを正にするため符号を反転する
        return -float(np.polyfit(r["t"][m], r["p"][m], 1)[0])

    def run(self) -> dict:
        res: dict = {"pulses": []}
        up, down = [], []
        for level in LEVELS:
            for sign in (1, -1):
                r = self.pulse(sign * level, PULSE_S)
                v = self.steady_velocity(r)
                if v is None:
                    print(f"  PWM {sign * level:+4d}: 打ち切りが早すぎて速度が出せない")
                    continue
                cur = float(np.mean(r["c"][r["t"] > 0.1])) if (r["t"] > 0.1).any() else 0.0
                res["pulses"].append({"pwm": sign * level, "vel": v, "current": cur})
                if abs(v) > 20:  # 動かなかった点（不感帯、重力に負けた）は当てはめから外す
                    (up if sign > 0 else down).append((v, sign * level))
                print(f"  PWM {sign * level:+4d}: 速度 {v:+7.0f} step/s  電流 {cur:5.1f}")
        # 動いた点が少ないのは、関節が押さえられている・端に当たっている・重力に負けている場合
        moving = [p for p in res["pulses"] if abs(p["vel"]) > 20]
        if len(moving) < 6:
            raise RuntimeError(
                f"動いた点が {len(moving)} 個しかない。関節が止まっている（押さえられている、端に当たっている、"
                "重力に負けている）。結果は保存しない"
            )
        # pwm = Fc·sign(ω) + Fv·ω の最小二乗。正の PWM は raw 位置が減る向きなので符号は ω と PWM で自然に合う
        rows = [(v, p) for v, p in up + down]
        # bias は向きによらない一定のトルク（重力、ケーブルやばねの力）。無負荷の関節ではほぼ 0
        A = np.array([[math.copysign(1.0, v), v, 1.0] for v, _ in rows])
        b = np.array([p for _, p in rows], dtype=float)
        (fc, fv, bias), *_ = np.linalg.lstsq(A, b, rcond=None)
        res["Fc"], res["Fv"], res["bias"] = float(fc), float(fv), float(bias)
        res["fit_max_residual"] = float(np.abs(A @ np.array([fc, fv, bias]) - b).max())
        print(
            f"Fc = {fc:.1f} PWM, Fv = {fv:.4f} PWM/(step/s)  （PWM 1 あたり {1 / fv:.1f} step/s）, "
            f"bias = {bias:+.1f} PWM, 最大残差 {res['fit_max_residual']:.1f}"
        )

        # ステップ応答から時定数 T を求め、Jp = Fv·T
        taus = []
        for level in (100, -100, 150, -150):  # 向きを交互にして位置が積み重ならないようにする
            r = self.pulse(level, 0.3)
            v_ss = self.steady_velocity(r, 0.15)
            if v_ss is None or abs(v_ss) < 1:
                continue
            # 速度の時間変化: 位置を 20 ms 窓で微分
            t, p = r["t"], r["p"]
            w = 10
            vel = -(p[w:] - p[:-w]) / (t[w:] - t[:-w])
            tm = (t[w:] + t[:-w]) / 2
            frac = vel / v_ss
            idx = np.where(frac >= 0.632)[0]
            if len(idx):
                taus.append(float(tm[idx[0]]))
        if taus:
            tau = float(np.median(taus))
            res["T"] = tau
            res["Jp"] = float(abs(fv) * tau)
            print(f"時定数 T = {tau * 1000:.0f} ms → Jp = {res['Jp']:.4f} PWM·s²/step（速度の窓 20 ms）")
        return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("leader", "follower"), required=True)
    ap.add_argument("--joint", choices=JOINTS, required=True)
    ap.add_argument("--limit", type=float, default=10.0, help="1パルスで動いてよい角度（度）")
    ap.add_argument("--abort", type=float, default=30.0, help="開始位置からこれを超えたら中止（度）")
    args = ap.parse_args()

    ident = Identifier(args.arm, args.joint, args.limit, args.abort)
    ident.arm.connect()
    pwm_mode = False
    result = None
    try:
        s = ident.arm.read_state()
        before = s.pos[ident.j]
        print(f"{args.arm}/{args.joint}: 位置モード pos={before} temp={s.temp[ident.j]}")
        input("関節の周りを空けて、重力負けする関節は支えた？ Enter で開始（Ctrl+C で中止）: ")
        ident.arm.set_mode(OperatingMode.PWM, [args.joint])
        pwm_mode = True
        ident.arm.write_pwm_one(args.joint, 0)
        ident.arm.bus.write("Torque_Enable", args.joint, 1, normalize=False)
        time.sleep(0.1)
        ident.last_raw = ident.arm.read_state().pos[ident.j]
        print(f"PWM モード pos={ident.last_raw}（位置モードとの差 {ident.last_raw - before}）")
        result = ident.run()
        result["pos_mode_offset"] = ident.last_raw - before
    finally:
        try:
            ident.arm.write_pwm_one(args.joint, 0)
            ident.arm.bus.write("Torque_Enable", args.joint, 0, normalize=False)
        finally:
            if pwm_mode:
                for _ in range(3):
                    try:
                        ident.arm.set_mode(OperatingMode.POSITION, [args.joint])
                        if ident.arm.bus.read("Operating_Mode", args.joint, normalize=False) == 0:
                            break
                    except Exception as e:  # noqa: BLE001
                        print("モード復帰に失敗、再試行:", e)
                else:
                    print("!! Operating_Mode を 0 に戻せなかった。手動で戻すこと")
            ident.arm.disconnect()
    if result:
        path = Path("calibration/bilateral") / f"{args.arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(path.read_text()) if path.exists() else {}
        data[args.joint] = {k: v for k, v in result.items() if k != "pulses"} | {"pulses": result["pulses"]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print("保存:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
