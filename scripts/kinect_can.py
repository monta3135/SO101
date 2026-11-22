#!/usr/bin/env python3
"""Kinect の深度で机の平面を求め、机の上に立っている物 (缶など) の3次元位置を出す。

座標はすべて Kinect の RGB カメラ座標 (m, x 右・y 下・z 奥)。位置合わせした深度なので RGB の画素と揃う。
単体で動かすと、見つけた物を表示して、印を付けた画像を outputs/kinect/ に保存する。

  uv run python scripts/kinect_can.py            # 1回撮って検出
  uv run python scripts/kinect_can.py --watch    # 連続で検出し続ける (Ctrl-C で終了)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Kinect v1 の RGB カメラの標準的な内部パラメータ (640x480)。個体の較正はしていない。
FX = FY = 525.0
CX, CY = 319.5, 239.5


def deproject(depth_mm: np.ndarray) -> np.ndarray:
    """(H, W) の mm 深度 → (H, W, 3) の点 (m)。測れない画素は z=0。"""
    h, w = depth_mm.shape
    z = depth_mm.astype(np.float32) / 1000.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.dstack([(u - CX) * z / FX, (v - CY) * z / FY, z])


def project(p: np.ndarray) -> tuple[float, float]:
    return float(p[0] * FX / p[2] + CX), float(p[1] * FY / p[2] + CY)


@dataclass
class Plane:
    normal: np.ndarray  # 単位ベクトル。カメラの方を向く (机の上が正)
    d: float  # normal・p + d = 0

    def height(self, pts: np.ndarray) -> np.ndarray:
        """机の面からの高さ (m)。上が正。"""
        return pts @ self.normal + self.d


def fit_table_plane(pts: np.ndarray, iters: int = 300, tol: float = 0.008, seed: int = 0) -> Plane:
    """RANSAC で一番大きい平面 (= 机) を求め、最小二乗で整える。"""
    valid = pts[pts[..., 2] > 0]
    rng = np.random.default_rng(seed)
    sample = valid[rng.choice(len(valid), min(len(valid), 20000), replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ a
        count = int((np.abs(sample @ n + d) < tol).sum())
        if count > best_n:
            best, best_n = (n, d), count
    n, d = best
    inl = valid[np.abs(valid @ n + d) < tol]
    centroid = inl.mean(0)
    n = np.linalg.svd(inl - centroid, full_matrices=False)[2][2]
    d = -n @ centroid
    if d < 0:  # カメラ (原点) 側を正にする
        n, d = -n, -d
    return Plane(n, float(d))


@dataclass
class TableObject:
    top: np.ndarray  # 物の上面の中心 (カメラ座標, m)
    base: np.ndarray  # 上面の中心を机の面へ下ろした点
    height: float  # 机からの高さ (m)
    width: float  # 机に沿った方向の大きさ (m, 外接円の直径)
    pixels: int
    bbox: tuple[int, int, int, int]  # x, y, w, h (画素)


def find_objects(
    pts: np.ndarray,
    plane: Plane,
    min_height: float = 0.03,
    max_height: float = 0.30,
    min_pixels: int = 150,
) -> list[TableObject]:
    """机の面から min_height 以上浮いた画素をまとめ、塊ごとに位置と大きさを出す。"""
    hgt = plane.height(pts)
    valid = pts[..., 2] > 0
    mask = (valid & (hgt > min_height) & (hgt < max_height)).astype(np.uint8)
    # 画像の上で重なって見えても、奥行きが 3 cm 以上跳ぶ所で切り離す (アームの後ろの缶など)
    z = pts[..., 2]
    jump = np.zeros_like(mask, dtype=bool)
    jump[:, 1:] |= np.abs(np.diff(z, axis=1)) > 0.03
    jump[1:, :] |= np.abs(np.diff(z, axis=0)) > 0.03
    mask[cv2.dilate(jump.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pixels:
            continue
        sel = labels == i
        p, h = pts[sel], hgt[sel]
        # 机の面へ下ろした位置の中央値を、物の中心とする
        foot = p - np.outer(h, plane.normal)
        center = np.median(foot, axis=0)
        top_h = float(np.percentile(h, 95))
        radial = np.linalg.norm(foot - center, axis=1)
        objs.append(
            TableObject(
                top=center + plane.normal * top_h,
                base=center,
                height=top_h,
                width=float(2 * np.percentile(radial, 90)),
                pixels=int(sel.sum()),
                bbox=tuple(int(x) for x in stats[i, :4]),
            )
        )
    return sorted(objs, key=lambda o: -o.pixels)


def find_objects_topdown(
    pts: np.ndarray,
    plane: Plane,
    to_base,
    min_height: float = 0.03,
    max_height: float = 0.30,
    cell: float = 0.01,
    min_pixels: int = 150,
    skeleton: np.ndarray | None = None,
    skeleton_radius: float = 0.06,
) -> list[TableObject]:
    """机の上の点を、アームの土台座標で真上から見た 1 cm 格子に落とし、つながった塊ごとに物を出す。

    画像の上でつながっているかで分ける find_objects と違い、机の上で離れていれば別の物になる。
    土台を押さえている手やアームと、その奥の缶が画像で重なっても、缶だけを取り出せる。
    to_base: カメラ座標の点 (N, 3) を土台座標へ写す関数。
    """
    hgt = plane.height(pts)
    sel = (pts[..., 2] > 0) & (hgt > min_height) & (hgt < max_height)
    vs, us = np.nonzero(sel)
    if len(vs) == 0:
        return []
    p_cam, h = pts[sel], hgt[sel]
    p_base = to_base(p_cam)
    if skeleton is not None:
        # アームの骨組み (関節を結ぶ折れ線) から skeleton_radius 以内の点はアームとして除く
        near = np.zeros(len(p_base), bool)
        for a, b in zip(skeleton[:-1], skeleton[1:]):
            ab = b - a
            s = np.clip((p_base - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
            near |= np.linalg.norm(p_base - (a + np.outer(s, ab)), axis=1) < skeleton_radius
        keep = ~near
        vs, us, p_cam, h, p_base = vs[keep], us[keep], p_cam[keep], h[keep], p_base[keep]
        if len(vs) == 0:
            return []
    xy = p_base[:, :2]
    ij = np.floor(xy / cell).astype(int)
    lo = ij.min(0)
    ij -= lo
    grid = np.zeros(ij.max(0) + 1, np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    occ = cv2.morphologyEx((grid >= 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(occ, connectivity=8)
    lab = labels[ij[:, 0], ij[:, 1]]
    objs = []
    for i in range(1, n):
        m = lab == i
        if m.sum() < min_pixels:
            continue
        p, hh = p_cam[m], h[m]
        foot = p - np.outer(hh, plane.normal)
        center = np.median(foot, axis=0)
        top_h = float(np.percentile(hh, 95))
        radial = np.linalg.norm(foot - center, axis=1)
        u, v = us[m], vs[m]
        objs.append(TableObject(
            top=center + plane.normal * top_h,
            base=center,
            height=top_h,
            width=float(2 * np.percentile(radial, 90)),
            pixels=int(m.sum()),
            bbox=(int(u.min()), int(v.min()), int(u.max() - u.min() + 1), int(v.max() - v.min() + 1)),
        ))
    return sorted(objs, key=lambda o: -o.pixels)


def annotate(rgb: np.ndarray, objs: list[TableObject]) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for k, o in enumerate(objs):
        x, y, w, h = o.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        u, v = project(o.base)
        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 255), -1)
        cv2.putText(img, f"{k}: h{o.height * 100:.0f}cm w{o.width * 100:.0f}cm", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return img


def main() -> None:
    from lerobot_camera_kinect import KinectCamera, KinectCameraConfig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true", help="連続で検出し続ける")
    ap.add_argument("--out", default="outputs/kinect", help="印を付けた画像の保存先")
    args = ap.parse_args()

    cam = KinectCamera(KinectCameraConfig(use_depth=True))
    cam.connect()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plane = None
    try:
        while True:
            rgb = cam.async_read(1000)
            pts = deproject(cam.read_latest_depth()[:, :, 0])
            if plane is None:
                plane = fit_table_plane(pts)
                dist = plane.d
                tilt = np.degrees(np.arccos(abs(plane.normal[1])))
                print(f"机の平面: カメラから机までの垂直距離 {dist * 100:.1f} cm, 光軸の傾き 下向き {90 - tilt:.0f}°")
            objs = find_objects(pts, plane)
            print(time.strftime("%H:%M:%S"), f"{len(objs)} 個")
            for k, o in enumerate(objs):
                print(f"  {k}: 底 ({o.base[0]:+.3f}, {o.base[1]:+.3f}, {o.base[2]:.3f}) m, "
                      f"高さ {o.height * 100:.1f} cm, 幅 {o.width * 100:.1f} cm, {o.pixels} px")
            cv2.imwrite(str(out / "objects.png"), annotate(rgb, objs))
            if not args.watch:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cam.disconnect()


if __name__ == "__main__":
    main()
