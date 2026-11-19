#!/usr/bin/env python3
"""フェーズ2: 1関節の重力項を、姿勢ごとの「保持に要る PWM」から同定する。

対象の関節だけ PWM モードにして、PWM の PID で目標角に保持し、保持中の PWM を平均する。
他の関節は位置モードのトルク ON で、開始時の姿勢に固定する。
各角度へ「下から」と「上から」の両方で近づく。摩擦は近づく向きに応じて符号が変わるので、
2回の平均が重力項 g、差の半分が摩擦 Fc になる。
重力項は g(θ) = A·cos θ + B·sin θ（θ は lerobot の度）で当てはめる。
他の関節の姿勢が変わると g も変わるので、固定した姿勢も一緒に保存する（全姿勢の重力は後で
MuJoCo のモデルと照合する）。

正の PWM は raw 位置が減る向きなので、制御は pwm = Kp·(pos − target) + Ki·∫ + Kd·vel の形になる。
PWM は ±--pwm-max に制限する。開始から --span+10 度を超えるか、温度・電流が 2 回続けて限界を超えたら止める。
終了時（例外と Ctrl+C を含む）は、開始位置へ戻してから全関節のトルクを OFF にし、Operating_Mode を 0 に戻す。
トルクが切れると、重力に負ける関節は落ちる。shoulder_lift と elbow_flex は終了時に必ず支えること。
制御中に通信が切れると PWM が残る（Goal_Time に監視タイマーはない）。重い関節は特に注意する。

使い方: uv run scripts/bilateral/identify_gravity.py --arm follower --joint wrist_flex
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
CALIBRATION = {
    "follower": "calibration/robots/so_follower/follower.json",
    "leader": "calibration/teleoperators/so_leader/leader.json",
}
KP, KI, KD = 3.0, 2.0, 0.1  # PWM/step, PWM/(step·s), PWM/(step/s)
MAX_TEMP_C = 55
MAX_CURRENT_UNITS = 400


class Holder:
    def __init__(self, arm: str, joint: str, pwm_max: int, abort_deg: float):
        self.arm = ArmBus(arm)
        self.joint = joint
        self.j = JOINTS.index(joint)
        self.pwm_max = pwm_max
        self.abort = abort_deg * STEPS_PER_DEG
        cal = json.loads(Path(CALIBRATION[arm]).read_text())[joint]
        self.home = cal["homing_offset"]
        self.mid = (cal["range_min"] + cal["range_max"]) / 2
        self.range = (cal["range_min"], cal["range_max"])
        self.last_raw = 0
        self.pos = 0.0  # 巻き戻し補正した PWM モードの raw 位置（step）
        self.start = 0.0
        self.over = 0
        self.integral = 0.0
        self.pwm = 0.0
        self.vel = 0.0
        self.t_last = 0.0

    def to_deg(self, pos_pwm: float) -> float:
        """PWM モードの raw 位置 → lerobot の度（位置モードの値 = PWM モードの値 − Homing_Offset）。"""
        raw = (pos_pwm - self.home) % 4096
        return (raw - self.mid) * 360 / 4095

    def to_pos(self, deg: float) -> float:
        """lerobot の度 → 巻き戻し補正した PWM モードの位置（開始位置の近くの枝）。"""
        raw = deg * 4095 / 360 + self.mid
        pos = raw + self.home
        k = round((self.start - pos) / 4096)
        return pos + 4096 * k

    def step(self, target: float) -> tuple[float, float, float]:
        """1周期ぶん読んで、PID の PWM を書く。(時刻, 位置, PWM) を返す。"""
        s = self.arm.read_state()
        raw = s.pos[self.j]
        self.pos += (raw - self.last_raw + 2048) % 4096 - 2048
        self.last_raw = raw
        dt = max(s.t - self.t_last, 1e-4) if self.t_last else 0.002
        self.t_last = s.t
        if abs(self.pos - self.start) > self.abort:
            raise RuntimeError(f"開始から {self.abort / STEPS_PER_DEG:.0f} 度を超えた")
        over = s.temp[self.j] > MAX_TEMP_C or s.current[self.j] > MAX_CURRENT_UNITS
        self.over = self.over + 1 if over else 0
        if self.over >= 2:
            raise RuntimeError(f"温度 {s.temp[self.j]} 電流 {s.current[self.j]}")
        self.vel += 0.3 * (s.vel[self.j] - self.vel)  # 速度を軽く平滑化
        err = self.pos - target
        self.integral = float(np.clip(self.integral + err * dt, -0.6 * self.pwm_max / KI, 0.6 * self.pwm_max / KI))
        self.pwm = float(np.clip(KP * err + KI * self.integral + KD * self.vel, -self.pwm_max, self.pwm_max))
        self.arm.write_pwm_one(self.joint, int(round(self.pwm)))
        return s.t, self.pos, self.pwm

    def move_to(self, target: float, speed_deg_s: float) -> None:
        """現在の目標から target まで、一定速度で目標を動かす。"""
        cur = self.pos
        dur = max(abs(target - cur) / (speed_deg_s * STEPS_PER_DEG), 0.01)
        t0 = time.perf_counter()
        while (frac := (time.perf_counter() - t0) / dur) < 1.0:
            self.step(cur + (target - cur) * frac)

    def hold(self, target: float, seconds: float, avg_last: float) -> dict:
        t0 = time.perf_counter()
        ts, ps, ws = [], [], []
        while (now := time.perf_counter() - t0) < seconds:
            _, p, w = self.step(target)
            if now > seconds - avg_last:
                ps.append(p)
                ws.append(w)
        return {"pwm": float(np.mean(ws)), "err_deg": float((np.mean(ps) - target) / STEPS_PER_DEG), "std": float(np.std(ws))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=CALIBRATION, required=True)
    ap.add_argument("--joint", choices=JOINTS, required=True)
    ap.add_argument("--span", type=float, default=30.0, help="開始角から ± この角度の範囲を調べる（度）")
    ap.add_argument("--points", type=int, default=5, help="角度の数")
    ap.add_argument("--pwm-max", type=int, default=250)
    ap.add_argument("--speed", type=float, default=15.0, help="目標を動かす速さ（度/秒）")
    ap.add_argument("--hold", type=float, default=2.0, help="各角度での保持時間（秒）")
    args = ap.parse_args()

    h = Holder(args.arm, args.joint, args.pwm_max, args.span + 10.0)
    h.arm.connect()
    pwm_mode = False
    result = None
    try:
        s = h.arm.read_state()
        before = s.pos[h.j]
        others = {n: s.pos[i] for i, n in enumerate(JOINTS) if n != args.joint}
        print(f"{args.arm}/{args.joint}: 位置モード pos={before} 現在 {(before - h.mid) * 360 / 4095:+.1f} 度")
        print("他の関節は開始姿勢に固定（トルク ON）。対象の関節は PWM で保持する。")
        input("重い関節は手で支えた？ Enter で開始（Ctrl+C で中止）: ")
        # 他の関節: 現在位置を目標にしてからトルク ON（元の目標へ飛ばないように）
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
        start_deg = h.to_deg(h.pos)
        lo, hi = h.range
        lo_deg, hi_deg = (lo - h.mid) * 360 / 4095 + 5, (hi - h.mid) * 360 / 4095 - 5
        degs = [d for d in np.linspace(start_deg - args.span, start_deg + args.span, args.points) if lo_deg <= d <= hi_deg]
        print(f"開始 {start_deg:+.1f} 度。調べる角度: {[round(float(d), 1) for d in degs]}")
        # 開始位置を目標に、いまの位置を保持してから始める
        h.hold(h.pos, 0.3, 0.1)
        points = []
        # 下から上へ（up）、続けて上から下へ（down）。端は両方向に出るので近づく向きが両方そろう
        order = [(d, "up") for d in degs] + [(d, "down") for d in reversed(degs)]
        for d, way in order:
            target = h.to_pos(d)
            # 近づく向きをそろえるため、up は 4 度下、down は 4 度上を経由する
            pre = target + (-4 if way == "up" else 4) * STEPS_PER_DEG
            h.move_to(pre, args.speed)
            h.move_to(target, args.speed / 2)
            r = h.hold(target, args.hold, args.hold * 0.4)
            r.update(deg=float(d), way=way)
            points.append(r)
            print(f"  {d:+7.1f} 度 {way:>4}: 保持 PWM {r['pwm']:+7.1f} (std {r['std']:.1f})  位置誤差 {r['err_deg']:+.2f} 度")
        h.move_to(h.start, args.speed)
        h.hold(h.start, 0.5, 0.1)
        # 重力と摩擦: 同じ角度の up/down の平均と差の半分
        g_pts = []
        for d in degs:
            up = [p["pwm"] for p in points if p["deg"] == float(d) and p["way"] == "up"]
            dn = [p["pwm"] for p in points if p["deg"] == float(d) and p["way"] == "down"]
            if up and dn:
                g_pts.append((float(d), (up[0] + dn[0]) / 2, (up[0] - dn[0]) / 2))
        th = np.radians([g[0] for g in g_pts])
        M = np.column_stack([np.cos(th), np.sin(th)])
        (a, b), *_ = np.linalg.lstsq(M, np.array([g[1] for g in g_pts]), rcond=None)
        resid = float(np.abs(M @ np.array([a, b]) - np.array([g[1] for g in g_pts])).max())
        print("重力項 g(θ) = A·cosθ + B·sinθ:", f"A={a:+.1f} B={b:+.1f} PWM（最大残差 {resid:.1f}）")
        print("摩擦（up と down の差の半分）:", [round(g[2], 1) for g in g_pts])
        result = {
            "A": float(a),
            "B": float(b),
            "max_residual": resid,
            "friction_half_diff": [g[2] for g in g_pts],
            "points": points,
            "other_joints_pos_raw": others,
        }
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
    if result:
        path = Path("calibration/bilateral") / f"{args.arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(path.read_text()) if path.exists() else {}
        data.setdefault(args.joint, {})["gravity"] = result
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print("保存:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
