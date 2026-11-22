#!/usr/bin/env python3
"""step_response.py の記録を、同じ指令・同じ時刻で MuJoCo に再生する（tachyon で動かす）。

このファイルの正本は 実機 PC の SO101 リポジトリ（scripts/step_replay.py）。
scripts/compare_step_response.py が実行のたびに tachyon の ~/so101/mujoco へ送ってから動かす。
record.py のモデル（build_model）とサーボの定数（KP、TIMESTEP）を使うので、~/so101/mujoco で実行する。

  cd ~/so101/mujoco && ../.venv/bin/python step_replay.py --real ../outputs/step_response/<日時>.csv

実機の値（LeRobot の度、gripper は 0〜100）を so101_new_calib の関節角に読み替える。
どちらも可動域の中央が 0 なので、腕の関節は度をそのままラジアンに直し、gripper は 0〜100 を
関節の可動域（閉じた端〜開いた端）に線形に割り当てる。ずれがわかっている関節は --offset で足す。

サーボは2通りで再生する。
  pipeline     record.py と同じ。目標角に重力補償（qfrc_bias / KP）を足して位置サーボに入れる
  no_gravcomp  同じ位置サーボで、重力補償なし
--params に fit_servo.py の <名前>_servo.json を渡すと、同定したサーボ（fitted、重力補償なし）も加える。
台やキューブとの接触は切り、自由空間での応答だけを比べる。

出力（記録と同じ場所）:
  <名前>_sim.csv    各行の t_obs での sim の角度（実機と同じ単位）
  <名前>_kstep.csv  実機の状態から始め、--kstep 行先まで予測した角度（k ステップ予測誤差用）
  <名前>_sim.json   再生の条件、sim の可動域から外れた値、ハンドの最低の高さ
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

import record as rec

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
VARIANTS = ("pipeline", "no_gravcomp", "fitted")
HAND_BODIES = ("gripper", "moving_jaw_so101_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="step_response の記録を MuJoCo で再生する")
    parser.add_argument("--real", type=Path, required=True, help="step_response.py の CSV")
    parser.add_argument("--variants", default="pipeline,no_gravcomp", help="再生するサーボ（カンマ区切り）")
    parser.add_argument("--params", type=Path, help="fit_servo.py の _servo.json。fitted を加える")
    parser.add_argument(
        "--offset",
        action="append",
        default=[],
        metavar="JOINT=VALUE",
        help="実機の値に足すずれ（腕は度、gripper は 0〜100）。何度でも指定できる",
    )
    parser.add_argument("--kstep", type=int, default=8, help="予測する行数。0 で省く")
    parser.add_argument("--kstep-stride", type=int, default=5, help="予測を始める行の間隔")
    return parser.parse_args()


def parse_offsets(items: list[str]) -> dict[str, float]:
    offsets = {}
    for item in items:
        name, _, value = item.partition("=")
        if name not in MOTORS or not value:
            raise SystemExit(f"--offset は JOINT=VALUE の形で、JOINT は {', '.join(MOTORS)} のどれか: {item}")
        offsets[name] = float(value)
    return offsets


def read_real(path: Path):
    with path.open() as csv_file:
        rows = list(csv.DictReader(csv_file))

    def column(name: str) -> np.ndarray:
        return np.array([float(row[name]) if row.get(name) else np.nan for row in rows])

    t_cmd = column("t_cmd")
    t_obs = column("t_obs")
    t_obs = np.where(np.isnan(t_obs), t_cmd, t_obs)  # 位置が読めなかった行は指令の時刻で代わりに見る
    cmd = np.stack([column(f"cmd.{m}") for m in MOTORS], axis=1)
    pos = np.stack([column(f"pos.{m}") for m in MOTORS], axis=1)
    excited = np.array([bool(row["joint"]) for row in rows])
    return t_cmd, t_obs, cmd, pos, excited


class Mapping:
    """実機の値と、sim の関節角（rad）の読み替え。"""

    def __init__(self, model: mujoco.MjModel, offsets: dict[str, float]):
        self.low = model.jnt_range[:6, 0].copy()
        self.high = model.jnt_range[:6, 1].copy()
        self.offset = np.array([offsets.get(m, 0.0) for m in MOTORS])
        self.grip_span = self.high[5] - self.low[5]

    def to_sim(self, values: np.ndarray) -> np.ndarray:
        shifted = values + self.offset
        q = np.radians(shifted)
        q[..., 5] = self.low[5] + shifted[..., 5] / 100 * self.grip_span
        return q

    def to_real(self, q: np.ndarray) -> np.ndarray:
        shifted = np.degrees(q)
        shifted[..., 5] = (q[..., 5] - self.low[5]) / self.grip_span * 100
        return shifted - self.offset


def load_model() -> mujoco.MjModel:
    model = rec.build_model()
    # 自由空間での応答だけを比べるので、台やキューブとの接触は使わない
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


def set_params(model: mujoco.MjModel, j: int, params: dict[str, float]) -> None:
    """関節 j のサーボに、fit_servo.py が同定した kp、damping、frictionloss を入れる。"""
    model.actuator_gainprm[j, 0] = params["kp"]
    model.actuator_biasprm[j, 1] = -params["kp"]
    model.dof_damping[j] = params["damping"]
    model.dof_frictionloss[j] = params["frictionloss"]


def load_fitted(path: Path) -> tuple[mujoco.MjModel, np.ndarray]:
    """同定したサーボを入れたモデルと、関節ごとのむだ時間（秒）。同定していない関節は元のまま。"""
    model = load_model()
    delays = np.zeros(6)
    for joint, params in json.loads(path.read_text())["joints"].items():
        j = MOTORS.index(joint)
        set_params(model, j, params)
        delays[j] = params["delay"]
    return model, delays


class Replayer:
    def __init__(self, model: mujoco.MjModel, gravcomp: bool, delays: np.ndarray | None = None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.gravcomp = gravcomp
        self.delays = delays
        self.hand = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in HAND_BODIES]

    def run(self, t_cmd, q_cmd, t_from, q_from, v_from, sample_times):
        """t_from の状態から始め、sample_times の各時刻の関節角（rad）とハンドの高さ（m）を返す。

        指令は t_cmd の時刻に書かれ、次の指令まで保たれる（実機と同じ零次ホールド）。
        """
        model, data = self.model, self.data
        dt = model.opt.timestep
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = q_from
        data.qvel[:6] = v_from
        mujoco.mj_forward(model, data)
        now = t_from
        active = max(int(np.searchsorted(t_cmd, now, side="right")) - 1, 0)
        q_out = np.empty((len(sample_times), 6))
        z_out = np.empty(len(sample_times))
        columns = np.arange(6)
        for j, t_sample in enumerate(sample_times):
            while now < t_sample - dt / 2:
                while active + 1 < len(t_cmd) and t_cmd[active + 1] <= now:
                    active += 1
                if self.delays is not None:
                    # 関節ごとにむだ時間だけ前の指令を使う
                    held = np.clip(np.searchsorted(t_cmd, now - self.delays, side="right") - 1, 0, None)
                    data.ctrl[:6] = q_cmd[held, columns]
                elif self.gravcomp:
                    # record.py と同じ。今の姿勢の重力とコリオリの分を目標角に足して打ち消す
                    mujoco.mj_forward(model, data)
                    data.ctrl[:6] = q_cmd[active] + data.qfrc_bias[:6] / rec.KP
                else:
                    data.ctrl[:6] = q_cmd[active]
                mujoco.mj_step(model, data)
                now += dt
            q_out[j] = data.qpos[:6]
            z_out[j] = data.xpos[self.hand, 2].min()
        return q_out, z_out


def outside(q: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, float]:
    """sim の可動域から外れている関節と、外れた量（度）。"""
    excess = np.maximum(low - q, 0) + np.maximum(q - high, 0)
    return {m: round(float(np.degrees(e)), 2) for m, e in zip(MOTORS, excess) if e > 1e-6}


def kstep(replayer, mapping, t_cmd, q_cmd, t_obs, pos, excited, k, stride):
    """関節を動かしている区間の行から、実機の状態で始めて k 行先まで予測する。

    速度は直前の行との差分で出す（先の行の値を使わない）。
    """
    q_real = mapping.to_sim(pos)
    rows = []
    for i in range(1, len(t_obs) - k, stride):
        if not excited[i] or np.isnan(q_real[i - 1 : i + k + 1]).any():
            continue
        v0 = (q_real[i] - q_real[i - 1]) / (t_obs[i] - t_obs[i - 1])
        q0 = np.clip(q_real[i], mapping.low, mapping.high)
        q, _ = replayer.run(t_cmd, q_cmd, t_obs[i], q0, v0, t_obs[i + 1 : i + k + 1])
        for h, values in enumerate(mapping.to_real(q), start=1):
            rows.append((i, h, values))
    return rows


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.params and "fitted" not in variants:
        variants.append("fitted")
    if "fitted" in variants and not args.params:
        raise SystemExit("fitted には --params で fit_servo.py の _servo.json が必要です")
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"知らないサーボです: {', '.join(unknown)}（{', '.join(VARIANTS)} から選ぶ）")
    offsets = parse_offsets(args.offset)
    t_cmd, t_obs, cmd, pos, excited = read_real(args.real)
    has_real = not np.isnan(pos).all()

    model = load_model()
    mapping = Mapping(model, offsets)
    q_cmd = mapping.to_sim(cmd)
    start = pos[0] if has_real and not np.isnan(pos[0]).any() else cmd[0]
    q_start = mapping.to_sim(start)
    command_outside = {
        m: int(n) for m, n in zip(MOTORS, ((q_cmd < mapping.low) | (q_cmd > mapping.high)).sum(axis=0)) if n
    }
    summary = {
        "real": str(args.real),
        "model": str(rec.MODEL_PATH),
        "kp": rec.KP,
        "timestep": model.opt.timestep,
        "variants": variants,
        "params": str(args.params) if args.params else None,
        "offsets": offsets,
        "start_from": "pos" if start is pos[0] else "cmd",
        "start_outside_deg": outside(q_start, mapping.low, mapping.high),
        "command_outside_rows": command_outside,
        "min_hand_z": {},
        "kstep": args.kstep if has_real else 0,
    }

    stem = args.real.with_suffix("")
    sim = {}
    kstep_rows = {}
    for variant in variants:
        if variant == "fitted":
            fitted_model, delays = load_fitted(args.params)
            replayer = Replayer(fitted_model, gravcomp=False, delays=delays)
        else:
            replayer = Replayer(model, gravcomp=variant == "pipeline")
        q0 = np.clip(q_start, mapping.low, mapping.high)
        q, z = replayer.run(t_cmd, q_cmd, t_cmd[0], q0, np.zeros(6), t_obs)
        sim[variant] = mapping.to_real(q)
        lowest = int(np.argmin(z))
        summary["min_hand_z"][variant] = {"z_m": round(float(z[lowest]), 4), "t": round(float(t_obs[lowest]), 3)}
        if summary["kstep"] > 0:
            kstep_rows[variant] = kstep(
                replayer, mapping, t_cmd, q_cmd, t_obs, pos, excited, args.kstep, args.kstep_stride
            )

    sim_path = Path(f"{stem}_sim.csv")
    with sim_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["row", "t_obs", *[f"{v}.{m}" for v in variants for m in MOTORS]])
        for i, t in enumerate(t_obs):
            writer.writerow([i, round(float(t), 4), *[round(float(x), 3) for v in variants for x in sim[v][i]]])
    if kstep_rows:
        with Path(f"{stem}_kstep.csv").open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["variant", "row", "h", *MOTORS])
            for variant, rows in kstep_rows.items():
                for i, h, values in rows:
                    writer.writerow([variant, i, h, *[round(float(x), 3) for x in values]])
    Path(f"{stem}_sim.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"sim で再生しました: {', '.join(variants)}（KP {rec.KP}、{model.opt.timestep * 1000:g}ms 刻み）")
    if summary["start_outside_deg"]:
        items = ", ".join(f"{m} {v:g} 度" for m, v in summary["start_outside_deg"].items())
        print(f"  開始姿勢が sim の可動域から外れた関節: {items}（可動域の端から始めました）")
    if command_outside:
        items = ", ".join(f"{m} {n} 行" for m, n in command_outside.items())
        print(f"  sim の可動域の外の指令: {items}（sim では端で止まります）")
    lowest = ", ".join(f"{v} {d['z_m']:.3f}m（{d['t']:.1f}s）" for v, d in summary["min_hand_z"].items())
    print(f"  ハンドの最低の高さ: {lowest}（関節の読み替えが合っている場合）")
    if kstep_rows:
        print(f"  {args.kstep} 行先の予測: {', '.join(f'{v} {len(r) // args.kstep} 回' for v, r in kstep_rows.items())}")
    print(f"  出力: {sim_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
