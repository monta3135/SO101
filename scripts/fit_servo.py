#!/usr/bin/env python3
"""step_response.py の記録に合うように、MuJoCo のサーボのパラメータを関節ごとに同定する（tachyon で動かす）。

このファイルの正本は 実機 PC の SO101 リポジトリ（scripts/fit_servo.py）。
step_replay.py と同じく ~/so101/mujoco に置き、record.py のモデルを使う。

  cd ~/so101/mujoco && ../.venv/bin/python fit_servo.py --real ../outputs/step_response/<日時>.csv

関節ごとに、その関節を動かしている区間（ステップ、チャープ、静止）だけを、記録と同じ指令・同じ時刻で
再生する。実機の角度との差の RMS が最小になるように、次の4つを scipy の Nelder-Mead（範囲つき）で探す。
  kp            位置サーボのゲイン（N·m/rad）
  damping       関節の粘性（N·m·s/rad）
  frictionloss  関節の静止摩擦（N·m）。止まる位置が近づく向きで変わる癖（不感帯）を表す
  delay         指令が効き始めるまでのむだ時間（秒）
実機には重力補償がないので、sim でも足さない。armature と forcerange は元のまま。

出力は <名前>_servo.json。step_replay.py --params に渡すと、同定したサーボ（fitted）で再生できる。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

import record as rec
from step_replay import MOTORS, Mapping, load_model, read_real, set_params

INITIAL = {"kp": rec.KP, "damping": 0.6, "frictionloss": 0.052, "delay": 0.02}
# 探す範囲。kp、damping、frictionloss は対数で探す
BOUNDS = {"kp": (2.0, 200.0), "damping": (0.01, 10.0), "frictionloss": (1e-3, 1.0), "delay": (0.0, 0.15)}
LOG_KEYS = ("kp", "damping", "frictionloss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="記録に合わせてサーボのパラメータを同定する")
    parser.add_argument("--real", type=Path, required=True, help="step_response.py の CSV")
    parser.add_argument("--iterations", type=int, default=160, help="関節ごとの Nelder-Mead の反復数")
    return parser.parse_args()


def simulate(model, data, t_cmd, q_cmd, t_from, q_from, sample_times) -> np.ndarray:
    """t_from の静止状態から、零次ホールドの指令で動かし、各 sample_times の関節角（rad）を返す。"""
    dt = model.opt.timestep
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = q_from
    mujoco.mj_forward(model, data)
    now = t_from
    active = max(int(np.searchsorted(t_cmd, now, side="right")) - 1, 0)
    out = np.empty((len(sample_times), 6))
    for k, t_sample in enumerate(sample_times):
        while now < t_sample - dt / 2:
            while active + 1 < len(t_cmd) and t_cmd[active + 1] <= now:
                active += 1
            data.ctrl[:6] = q_cmd[active]
            mujoco.mj_step(model, data)
            now += dt
        out[k] = data.qpos[:6]
    return out


def to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array([np.log(params[k]) if k in LOG_KEYS else params[k] * 10 for k in INITIAL])


def from_vector(x: np.ndarray) -> dict[str, float]:
    params = {}
    for value, key in zip(x, INITIAL):
        raw = float(np.exp(value)) if key in LOG_KEYS else float(value) / 10
        params[key] = float(np.clip(raw, *BOUNDS[key]))
    return params


def fit_joint(real_path: Path, joint: str, iterations: int) -> dict:
    t_cmd, t_obs, cmd, pos, excited = read_real(real_path)
    with real_path.open() as csv_file:
        header = csv_file.readline().strip().split(",")
        joint_col = header.index("joint")
        names = [line.rstrip("\n").split(",")[joint_col] for line in csv_file]
    rows = np.flatnonzero((np.array(names) == joint) & ~np.isnan(pos).any(axis=1))
    j = MOTORS.index(joint)
    model = load_model()
    data = mujoco.MjData(model)
    mapping = Mapping(model, {})
    q_cmd = mapping.to_sim(cmd)
    q_from = np.clip(mapping.to_sim(pos[rows[0]]), mapping.low, mapping.high)
    target = pos[rows, j]

    def rms(params: dict[str, float]) -> float:
        set_params(model, j, params)
        q = simulate(model, data, t_cmd + params["delay"], q_cmd, t_obs[rows[0]], q_from, t_obs[rows])
        return float(np.sqrt(np.mean((mapping.to_real(q)[:, j] - target) ** 2)))

    before = rms(INITIAL)
    x0 = to_vector(INITIAL)
    bounds = list(zip(to_vector({k: lo for k, (lo, _) in BOUNDS.items()}), to_vector({k: hi for k, (_, hi) in BOUNDS.items()})))
    # 最初の単体は各パラメータを少しずつずらして作る（kp などは対数で 0.7、むだ時間は 30ms）
    simplex = np.vstack([x0, x0 + np.diag([0.7, 0.7, 0.7, 0.3])])
    result = minimize(
        lambda v: rms(from_vector(v)),
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": iterations, "initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-4},
    )
    x, after = result.x, float(result.fun)
    return {"joint": joint, "params": from_vector(x), "rms_before": before, "rms_after": after, "rows": len(rows)}


def main() -> int:
    args = parse_args()
    with args.real.open() as csv_file:
        header = csv_file.readline().strip().split(",")
        joint_col = header.index("joint")
        joints = sorted({line.rstrip("\n").split(",")[joint_col] for line in csv_file} - {""}, key=MOTORS.index)
    with ProcessPoolExecutor(max_workers=len(joints)) as pool:
        results = list(pool.map(fit_joint, [args.real] * len(joints), joints, [args.iterations] * len(joints)))

    out = {
        "real": str(args.real),
        "model": str(rec.MODEL_PATH),
        "gravcomp": False,
        "initial": INITIAL,
        "joints": {r["joint"]: {**r["params"], "rms_before": r["rms_before"], "rms_after": r["rms_after"]} for r in results},
    }
    path = Path(f"{args.real.with_suffix('')}_servo.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("関節            kp      damping  friction  delay[ms]  RMS[度] 元 → 同定後")
    for r in results:
        p = r["params"]
        print(
            f"{r['joint']:14s} {p['kp']:7.2f}  {p['damping']:7.3f}  {p['frictionloss']:8.4f}  {p['delay'] * 1000:8.1f}"
            f"   {r['rms_before']:.2f} → {r['rms_after']:.2f}"
        )
    print(f"出力: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
