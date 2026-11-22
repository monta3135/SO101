"""SO-101 の順運動学と逆運動学 (numpy だけ、MuJoCo 不要)。

リンクの寸法は mjlab の so101_new_calib.xml から写した。run_tracker.py と同じく、
LeRobot の度をそのまま sim のラジアンに読み替える (どちらも可動域の中央が 0)。
座標はアームの土台の座標 (m)。z が上、土台の底が z=0。
"""

from __future__ import annotations

import numpy as np

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# (親から見た位置, 親から見た姿勢 quat (w, x, y, z)) の後に、z 軸まわりの関節が来る
_CHAIN = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.56167e-16, 1.22818e-15, -1, -4.14635e-16)),  # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (0.5, -0.5, -0.5, -0.5)),  # shoulder_lift
    ((-0.11257, -0.028, 0.0), (0.707107, 0.0, 0.0, 0.707107)),  # elbow_flex
    ((-0.1349, 0.0052, 0.0), (0.707107, 0.0, 0.0, -0.707107)),  # wrist_flex
    ((0.0, -0.0611, 0.0181), (0.0172091, -0.0172091, 0.706897, 0.706897)),  # wrist_roll
]
# gripper ボディから見た gripperframe (爪の間の、掴む点)。x 軸が指の伸びる向き
_TCP = ((-0.0079, -0.000218121, -0.0981274), (0.707107, 0.0, 0.707107, 0.0))

LIMITS_DEG = {  # so101_new_calib.xml の可動域
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.0, 162.0),
}


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _tf(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, pos
    return t


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_LINKS = [_tf(p, _quat_to_mat(q)) for p, q in _CHAIN]
_TCP_TF = _tf(_TCP[0], _quat_to_mat(_TCP[1]))


def fk(q_deg) -> np.ndarray:
    """腕の5関節 (度, ARM の順) → 掴む点の 4x4 姿勢 (土台座標)。x 列が指の向き。"""
    t = np.eye(4)
    for link, a in zip(_LINKS, np.radians(np.asarray(q_deg, float))):
        t = t @ link @ _tf(np.zeros(3), _rz(a))
    return t @ _TCP_TF


def link_points(q_deg) -> np.ndarray:
    """土台から掴む点まで、各関節の位置を並べた (7, 3)。アームの骨組みの折れ線として使う。"""
    t = np.eye(4)
    pts = [t[:3, 3].copy()]
    for link, a in zip(_LINKS, np.radians(np.asarray(q_deg, float))):
        t = t @ link @ _tf(np.zeros(3), _rz(a))
        pts.append(t[:3, 3].copy())
    pts.append((t @ _TCP_TF)[:3, 3])
    return np.array(pts)


def fk_from_obs(obs: dict) -> np.ndarray:
    return fk([obs[f"{m}.pos"] for m in ARM])


def ik(
    target: np.ndarray,
    q0_deg,
    pitch_deg: float | None = 0.0,
    roll_deg: float | None = None,
    iters: int = 200,
) -> tuple[np.ndarray, float, float]:
    """掴む点を target (m) に持っていく関節角 (度) を、減衰付き最小二乗で探す。

    pitch_deg: 指の向きの水平からの角度 (下向きが正)。None なら気にしない。
    roll_deg: wrist_roll をこの値に固定する。None なら q0 のまま。
    戻り値: (関節角, 位置の誤差 m, 向きの誤差 度)
    """
    q = np.array(q0_deg, float)
    if roll_deg is not None:
        q[4] = roll_deg
    lo = np.array([LIMITS_DEG[m][0] for m in ARM])
    hi = np.array([LIMITS_DEG[m][1] for m in ARM])
    free = [0, 1, 2, 3]  # wrist_roll は掴む点の位置にも指の向きにもほぼ効かないので動かさない

    def residual(qq):
        t = fk(qq)
        r = list(t[:3, 3] - target)
        if pitch_deg is not None:
            fwd = t[:3, 0]
            r.append(0.1 * (np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1))) - pitch_deg) / 10.0)
        return np.array(r)

    for _ in range(iters):
        r = residual(q)
        if np.linalg.norm(r[:3]) < 5e-4 and (len(r) == 3 or abs(r[3]) < 0.01 * 0.1):
            break
        jac = np.zeros((len(r), len(free)))
        for k, j in enumerate(free):
            dq = q.copy()
            dq[j] += 0.1
            jac[:, k] = (residual(dq) - r) / 0.1
        lam = 1e-4
        step = -np.linalg.solve(jac.T @ jac + lam * np.eye(len(free)), jac.T @ r)
        step = np.clip(step, -10, 10)
        q[free] = np.clip(q[free] + step, lo[free], hi[free])
    t = fk(q)
    pos_err = float(np.linalg.norm(t[:3, 3] - target))
    pitch = float(np.degrees(np.arcsin(np.clip(-t[2, 0], -1, 1))))
    return q, pos_err, (abs(pitch - pitch_deg) if pitch_deg is not None else 0.0)


def pitch_of(t: np.ndarray) -> float:
    return float(np.degrees(np.arcsin(np.clip(-t[2, 0], -1, 1))))
