#!/usr/bin/env python3
"""step_response.py の記録を tachyon の MuJoCo で再生し、実機と重ねて比べる。

  uv run python scripts/compare_step_response.py outputs/step_response/<日時>.csv

1. 記録の CSV と scripts/step_replay.py を tachyon へ送り、同じ指令を MuJoCo で再生する。
   サーボは record.py と同じ重力補償つき（pipeline）と、重力補償なし（no_gravcomp）の2通り。
2. 再生結果を記録の隣へ持ち帰り、動かした関節ごとに比べる。
   - ステップ: 落ち着いた位置と指令の差（垂れ）、変化の 50% と 90% に届くまでの時間、行き過ぎ
   - チャープ: 指令に対する振幅の比と遅れ（周波数ごと）
   - 実機の状態から 8 行（50Hz なら 160ms）先を予測したときの誤差。論文の k ステップ予測誤差と同じ考え方
3. 指令・実機・sim を重ねた図と表を <日時>_report.html に、数値を <日時>_metrics.json に書き、
   要点を画面に出す。

--sim に再生済みの <日時>_sim.csv を渡すと tachyon には接続しない。
"""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import shlex
import subprocess
import unicodedata
from pathlib import Path

import numpy as np

HOST = "tachyon"
REMOTE_CODE = "so101/mujoco"  # tachyon のホームからの相対パス
REMOTE_OUT = "so101/outputs/step_response"
REMOTE_OUT_FROM_CODE = "../outputs/step_response"
REPLAY_SCRIPT = Path(__file__).resolve().parent / "step_replay.py"
SETTLED_FRACTION = 0.3  # ステップ区間の最後のこの割合を、落ち着いた位置とみなす
MIN_CHANGE = 0.5  # 変化がこれより小さいステップは到達時間を測らない（度）
MAX_LAG_S = 0.3  # 遅れを探す範囲（秒）
GRID_S = 0.002  # チャープを解析するときの等間隔の刻み（秒）
CHIRP_WINDOWS = 5
SERIES = ("real", "pipeline", "no_gravcomp", "fitted")
BASELINES = ("ideal", "hold")
LABELS = {
    "cmd": "指令",
    "real": "実機",
    "pipeline": "sim 重力補償あり",
    "no_gravcomp": "sim 重力補償なし",
    "fitted": "sim 同定後",
    "ideal": "指令どおりと仮定",
    "hold": "止まると仮定",
}
COLORS = {"cmd": "#8c959f", "real": "#0969da", "pipeline": "#cf222e", "no_gravcomp": "#1a7f37", "fitted": "#8250df"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="step_response の記録を sim と比べる")
    parser.add_argument("csv", type=Path, help="step_response.py の CSV")
    parser.add_argument("--sim", type=Path, help="再生済みの _sim.csv。指定すると tachyon に接続しない")
    parser.add_argument("--host", default=HOST, help="sim を動かすマシン")
    parser.add_argument(
        "--offset",
        action="append",
        default=[],
        metavar="JOINT=VALUE",
        help="step_replay.py に渡す、実機の値に足すずれ。何度でも指定できる",
    )
    parser.add_argument("--kstep", type=int, default=8, help="予測する行数")
    parser.add_argument(
        "--params",
        type=Path,
        help="fit_servo.py の _servo.json。同定したサーボも並べる。既定では記録の隣にあれば使う",
    )
    return parser.parse_args()


def replay_on_host(csv_path: Path, host: str, offsets: list[str], kstep: int, params: Path | None) -> Path:
    """記録と再生スクリプトを送って sim で再生し、結果を記録の隣へ持ち帰る。"""
    name, stem = csv_path.name, csv_path.stem
    print(f"{host} の MuJoCo で再生します。", flush=True)
    subprocess.run(["ssh", host, f"mkdir -p {REMOTE_OUT}"], check=True)
    subprocess.run(["scp", "-q", str(REPLAY_SCRIPT), f"{host}:{REMOTE_CODE}/step_replay.py"], check=True)
    subprocess.run(["scp", "-q", str(csv_path), f"{host}:{REMOTE_OUT}/{name}"], check=True)
    options = "".join(f" --offset {shlex.quote(item)}" for item in offsets)
    if params:
        subprocess.run(["scp", "-q", str(params), f"{host}:{REMOTE_OUT}/{params.name}"], check=True)
        options += f" --params {REMOTE_OUT_FROM_CODE}/{shlex.quote(params.name)}"
    remote = (
        f"cd {REMOTE_CODE} && ../.venv/bin/python step_replay.py"
        f" --real {REMOTE_OUT_FROM_CODE}/{shlex.quote(name)} --kstep {kstep}{options}"
    )
    subprocess.run(["ssh", host, remote], check=True)
    for suffix in ("_sim.csv", "_sim.json"):
        subprocess.run(["scp", "-q", f"{host}:{REMOTE_OUT}/{stem}{suffix}", str(csv_path.parent)], check=True)
    summary = json.loads(csv_path.with_name(f"{stem}_sim.json").read_text())
    if summary.get("kstep"):
        subprocess.run(["scp", "-q", f"{host}:{REMOTE_OUT}/{stem}_kstep.csv", str(csv_path.parent)], check=True)
    return csv_path.with_name(f"{stem}_sim.csv")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as csv_file:
        return list(csv.DictReader(csv_file))


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.array([float(row[name]) if row.get(name) else np.nan for row in rows])


class Record:
    """実機の記録と、その sim での再生結果。"""

    def __init__(self, csv_path: Path, sim_path: Path):
        self.meta = json.loads(csv_path.with_suffix(".json").read_text())
        self.motors = self.meta["motors"]
        rows = read_rows(csv_path)
        self.t_cmd = column(rows, "t_cmd")
        t_obs = column(rows, "t_obs")
        self.t_obs = np.where(np.isnan(t_obs), self.t_cmd, t_obs)
        self.seg = column(rows, "seg").astype(int)
        self.joint = np.array([row["joint"] for row in rows])
        self.cmd = np.stack([column(rows, f"cmd.{m}") for m in self.motors], axis=1)
        self.series: dict[str, np.ndarray] = {}
        real = np.stack([column(rows, f"pos.{m}") for m in self.motors], axis=1)
        if not np.isnan(real).all():
            self.series["real"] = real

        sim_rows = read_rows(sim_path)
        if len(sim_rows) != len(rows):
            raise SystemExit(f"sim の行数（{len(sim_rows)}）が記録（{len(rows)}）と合いません。再生し直してください。")
        for variant in SERIES[1:]:
            if f"{variant}.{self.motors[0]}" in sim_rows[0]:
                self.series[variant] = np.stack([column(sim_rows, f"{variant}.{m}") for m in self.motors], axis=1)
        stem = str(sim_path)[: -len("_sim.csv")]
        summary_path = Path(f"{stem}_sim.json")
        self.sim_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        kstep_path = Path(f"{stem}_kstep.csv")
        self.kstep_rows = read_rows(kstep_path) if kstep_path.exists() else None


def mean(values) -> float:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def first_crossing(t: np.ndarray, fraction: np.ndarray, level: float) -> float:
    """fraction が初めて level に届く時刻（行の間は直線でつなぐ）。届かなければ nan。"""
    above = np.flatnonzero(fraction >= level)
    if len(above) == 0:
        return math.nan
    i = above[0]
    if i == 0:
        return float(t[0])
    f0, f1 = fraction[i - 1], fraction[i]
    return float(t[i - 1] + (level - f0) / (f1 - f0) * (t[i] - t[i - 1]))


def step_response(t: np.ndarray, y: np.ndarray, target: float, t_switch: float) -> dict | None:
    """1つのステップ区間の、垂れ（落ち着いた位置 − 指令）、到達時間、行き過ぎ。"""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(y) < 5:
        return None
    tail = t >= t[0] + (1 - SETTLED_FRACTION) * (t[-1] - t[0])
    settled = float(y[tail].mean())
    result = {"offset": settled - target, "t50_ms": math.nan, "t90_ms": math.nan, "overshoot_pct": math.nan}
    change = settled - y[0]
    if abs(change) >= MIN_CHANGE:
        # 変化の割合は、指令ではなく実際に落ち着いた位置を 1 とする。垂れがあっても到達時間を測れる
        fraction = (y - y[0]) / change
        result["t50_ms"] = (first_crossing(t, fraction, 0.5) - t_switch) * 1000
        result["t90_ms"] = (first_crossing(t, fraction, 0.9) - t_switch) * 1000
        result["overshoot_pct"] = max(0.0, float(fraction.max() - 1) * 100)
    return result


def lag_between(x: np.ndarray, y: np.ndarray, dt: float, max_lag: float) -> float:
    """y が x よりどれだけ遅れているか（秒）。相互相関が最大になるずれ。"""
    x = x - x.mean()
    y = y - y.mean()
    best_k, best = 0, -math.inf
    for k in range(int(max_lag / dt) + 1):
        if len(x) - k < 20:
            break
        score = float(np.dot(x[: len(x) - k], y[k:])) / (len(x) - k)
        if score > best:
            best_k, best = k, score
    return best_k * dt


def sine_fit(phase: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """values ≈ a·sin(phase − θ) + c に当てはめ、振幅 a と位相の遅れ θ（rad）を返す。"""
    basis = np.stack([np.sin(phase), np.cos(phase), np.ones_like(phase)], axis=1)
    (p, q, _), *_ = np.linalg.lstsq(basis, values, rcond=None)
    return math.hypot(p, q), math.atan2(-q, p)


def chirp_response(record: Record, j: int, segment: dict, y: np.ndarray) -> dict:
    """チャープ区間の、全体の遅れと、周波数の窓ごとの振幅比と遅れ。"""
    ramp = segment["ramp"]
    duration = segment["t1"] - segment["t0"]
    grid = np.arange(segment["t0"] + ramp, segment["t1"] - ramp, GRID_S)
    held = np.searchsorted(record.t_cmd, grid, side="right") - 1  # 指令は次を書くまで保たれる
    x = record.cmd[np.clip(held, 0, None), j]
    ok = np.isfinite(y)
    response = np.interp(grid, record.t_obs[ok], y[ok])
    # 窓が 1 周期より短くても測れるように、計画の位相に正弦波を当てはめて指令と応答の位相を比べる
    u = grid - segment["t0"]
    phase = 2 * math.pi * (segment["f0"] * u + (segment["f1"] - segment["f0"]) * u * u / (2 * duration))
    windows = []
    for idx in np.array_split(np.arange(len(grid)), CHIRP_WINDOWS):
        freq = segment["f0"] + (segment["f1"] - segment["f0"]) * u[idx].mean() / duration
        amp_x, lag_x = sine_fit(phase[idx], x[idx])
        amp_y, lag_y = sine_fit(phase[idx], response[idx])
        delay = (lag_y - lag_x + math.pi) % (2 * math.pi) - math.pi
        windows.append({
            "f_hz": freq,
            "gain": amp_y / amp_x if amp_x > 1e-9 else math.nan,
            "lag_ms": delay / (2 * math.pi * freq) * 1000,
            "phase_deg": math.degrees(delay),
        })
    return {
        "lag_ms": lag_between(x, response, GRID_S, MAX_LAG_S) * 1000,
        "rms_err": float(np.sqrt(np.mean((response - x) ** 2))),
        "windows": windows,
    }


def analyze_joint(record: Record, joint: str) -> dict:
    j = record.motors.index(joint)
    plan = record.meta["plan"]
    excited = record.joint == joint
    result = {}
    for name, values in record.series.items():
        y = values[:, j]
        steps, chirp = [], None
        for index, segment in enumerate(plan):
            if segment["joint"] != joint:
                continue
            rows = np.flatnonzero(record.seg == index)
            if segment["kind"] == "step" and len(rows):
                target = float(record.cmd[rows[-1], j])
                metrics = step_response(record.t_obs[rows], y[rows], target, float(record.t_cmd[rows[0]]))
                if metrics:
                    steps.append({"name": segment["name"], "target": target, **metrics})
            elif segment["kind"] == "chirp" and len(rows):
                chirp = chirp_response(record, j, segment, y)
        rms_vs_real = None
        if name != "real" and "real" in record.series:
            diff = values[excited, j] - record.series["real"][excited, j]
            diff = diff[np.isfinite(diff)]
            rms_vs_real = float(np.sqrt(np.mean(diff**2))) if len(diff) else math.nan
        summary = {
            "offset": mean(s["offset"] for s in steps),
            "t50_ms": mean(s["t50_ms"] for s in steps),
            "t90_ms": mean(s["t90_ms"] for s in steps),
            "overshoot_pct": max((s["overshoot_pct"] for s in steps if math.isfinite(s["overshoot_pct"])), default=math.nan),
            "chirp_lag_ms": chirp["lag_ms"] if chirp else math.nan,
            "chirp_last_gain": chirp["windows"][-1]["gain"] if chirp else math.nan,
            "chirp_last_f_hz": chirp["windows"][-1]["f_hz"] if chirp else math.nan,
            "rms_vs_real": rms_vs_real,
        }
        result[name] = {"steps": steps, "chirp": chirp, "summary": summary}
    return result


def kstep_errors(record: Record, k: int) -> dict | None:
    """h 行先の予測誤差（度、RMS）。sim のほかに、指令どおりに動くと仮定した予測と、止まると仮定した予測も並べる。"""
    if record.kstep_rows is None or "real" not in record.series:
        return None
    real = record.series["real"]
    variants = sorted({row["variant"] for row in record.kstep_rows}, key=SERIES.index)
    joints = record.meta["joints"]
    out = {"k": k, "by_h": {}, "by_joint": {}}
    for variant in variants:
        rows = [row for row in record.kstep_rows if row["variant"] == variant]
        start = np.array([int(row["row"]) for row in rows])
        h = np.array([int(row["h"]) for row in rows])
        predictions = {variant: np.array([[float(row[m]) for m in record.motors] for row in rows])}
        if variant == variants[0]:
            predictions["ideal"] = record.cmd[start + h]  # 指令を書いた瞬間にその角度になるとした予測
            predictions["hold"] = real[start]  # 今の位置から動かないとした予測
        actual = real[start + h]
        # 予測を始めた行で動かしていた関節の誤差だけを見る。止めている関節は開始姿勢の読み替えのずれが目立つ
        moving = np.array([record.motors.index(name) for name in record.joint[start]])
        for name, predicted in predictions.items():
            error = predicted - actual
            ok = np.isfinite(error).all(axis=1)
            out["by_h"][name] = []
            for step in range(1, k + 1):
                mask = ok & (h == step)
                out["by_h"][name].append(float(np.sqrt(np.mean(error[mask, moving[mask]] ** 2))))
            out["by_joint"][name] = {}
            for joint in joints:
                j = record.motors.index(joint)
                mask = ok & (h == k) & (record.joint[start] == joint)
                out["by_joint"][name][joint] = float(np.sqrt(np.mean(error[mask, j] ** 2))) if mask.any() else math.nan
    return out


def fmt(value, digits: int = 1, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def text_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(text_width(r[i]) for r in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        cells = [
            cell + " " * (width - text_width(cell)) if i == 0 else " " * (width - text_width(cell)) + cell
            for i, (cell, width) in enumerate(zip(row, widths))
        ]
        print("  " + "  ".join(cells), flush=True)


def summary_rows(record: Record, analysis: dict, kstep: dict | None) -> tuple[list[str], dict[str, list[list[str]]]]:
    """画面と HTML で共通の、関節ごとの要点の表。"""
    k = kstep["k"] if kstep else None
    headers = ["", "垂れ[度]", "50%[ms]", "90%[ms]", "行き過ぎ[%]", "チャープ遅れ[ms]", "高い周波数の振幅比", "sim−実機[度]"]
    if k:
        headers.append(f"{k}行先の予測[度]")
    tables = {}
    for joint, result in analysis.items():
        rows = []
        for name in SERIES:
            if name not in result:
                continue
            s = result[name]["summary"]
            row = [
                LABELS[name],
                fmt(s["offset"], 2, signed=True),
                fmt(s["t50_ms"], 0),
                fmt(s["t90_ms"], 0),
                fmt(s["overshoot_pct"], 0),
                fmt(s["chirp_lag_ms"], 0),
                f"{fmt(s['chirp_last_gain'], 2)}（{fmt(s['chirp_last_f_hz'], 1)}Hz）",
                fmt(s["rms_vs_real"], 2),
            ]
            if k:
                row.append(fmt(kstep["by_joint"].get(name, {}).get(joint), 2) if name != "real" else "—")
            rows.append(row)
        if k:
            for name in BASELINES:
                rows.append([LABELS[name], *["—"] * 7, fmt(kstep["by_joint"][name][joint], 2)])
        tables[joint] = rows
    return headers, tables


_CLIP_IDS = itertools.count()


def nice_ticks(low: float, high: float, count: int) -> tuple[list[float], int]:
    raw = (high - low) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)
    first = math.ceil(low / step)
    ticks = [i * step for i in range(first, int(math.floor(high / step)) + 1)]
    return ticks, max(0, -math.floor(math.log10(step)))


def svg_chart(
    lines: list[tuple], t0: float, t1: float, ylabel: str, width: int = 1080, height: int = 220, robust: bool = False
) -> str:
    """lines は (系列の名前, 時刻, 値, 破線か) の並び。名前で色が決まる。

    robust では縦軸を値の 2〜98% に合わせ、はみ出した線は枠で切る（ステップ直後の大きなずれで潰れないように）。
    """
    left, right, top, bottom = 60, 12, 10, 28
    shown = [y[(t >= t0) & (t <= t1) & np.isfinite(y)] for _, t, y, _ in lines]
    values = np.concatenate([v for v in shown if len(v)]) if any(len(v) for v in shown) else np.array([0.0, 1.0])
    if robust:
        values = np.percentile(values, [2, 98])
    ymin, ymax = float(values.min()), float(values.max())
    if ymax - ymin < 1e-3:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    pad = (ymax - ymin) * 0.06
    ymin, ymax = ymin - pad, ymax + pad

    def sx(t: float) -> float:
        return left + (t - t0) / (t1 - t0) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    clip = f"clip{next(_CLIP_IDS)}"
    out = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(ylabel)}">',
        f'<clipPath id="{clip}"><rect x="{left}" y="{top}" width="{width - left - right}"'
        f' height="{height - top - bottom}"/></clipPath>',
    ]
    ticks, digits = nice_ticks(ymin, ymax, 5)
    for tick in ticks:
        y = sy(tick)
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:.{digits}f}</text>')
    ticks, digits = nice_ticks(t0, t1, 10)
    for tick in ticks:
        x = sx(tick)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - bottom}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{height - bottom + 16}" text-anchor="middle">{tick:.{digits}f}</text>')
    middle = top + (height - top - bottom) / 2
    out.append(f'<text x="14" y="{middle:.1f}" transform="rotate(-90 14 {middle:.1f})" text-anchor="middle">'
               f"{html.escape(ylabel)}</text>")
    for name, t, y, dashed in lines:
        mask = (t >= t0) & (t <= t1) & np.isfinite(y)
        points = " ".join(f"{sx(a):.1f},{sy(b):.2f}" for a, b in zip(t[mask], y[mask]))
        dash = ' stroke-dasharray="5 3"' if dashed else ""
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[name]}" stroke-width="1.3"{dash}'
            f' clip-path="url(#{clip})"/>'
        )
    out.append("</svg>")
    return "\n".join(out)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def condition_items(record: Record) -> list[str]:
    meta, sim = record.meta, record.sim_summary
    result = meta.get("result", {})
    items = [f"記録: {meta['created']}（{'計画のみ' if meta.get('dry_run') else result.get('stopped', '?')}、"
             f"{result.get('rows', '?')} 行、周期の中央値 {result.get('period_ms_median', '—')}ms）"]
    items.append("試験姿勢: " + ", ".join(f"{m} {meta['pose'][m]:+.1f}" for m in record.motors))
    settings = meta.get("servo_settings", {})
    if settings:
        parts = []
        for name, values in settings.items():
            distinct = sorted(set(values.values()))
            parts.append(f"{name} {distinct[0]}" if len(distinct) == 1 else f"{name} {values}")
        items.append("サーボの設定: " + ", ".join(parts))
    if sim:
        items.append(f"sim: KP {sim['kp']}、{sim['timestep'] * 1000:g}ms 刻み、接触なし、読み替えのずれ {sim['offsets'] or 'なし'}")
        if sim.get("start_outside_deg"):
            items.append(f"開始姿勢が sim の可動域から外れた関節（端から始めた）: {sim['start_outside_deg']}")
        if sim.get("command_outside_rows"):
            items.append(f"sim の可動域の外の指令（行数、sim では端で止まる）: {sim['command_outside_rows']}")
        lowest = ", ".join(f"{LABELS[v]} {d['z_m']:.3f}m（{d['t']:.1f}s）" for v, d in sim["min_hand_z"].items())
        items.append(f"sim でのハンドの最低の高さ: {lowest}")
    return items


REPORT_STYLE = """
body { font-family: system-ui, sans-serif; margin: 24px auto; max-width: 1120px; padding: 0 16px;
       color: #1f2328; background: #ffffff; line-height: 1.5; }
h1 { font-size: 20px; } h2 { font-size: 17px; margin-top: 32px; } h3 { font-size: 14px; margin: 16px 0 4px; }
table { border-collapse: collapse; font-size: 13px; margin: 6px 0 12px; }
th, td { border: 1px solid #d0d7de; padding: 3px 8px; text-align: right; }
td:first-child, th:first-child { text-align: left; }
th { background: #f6f8fa; font-weight: 600; }
.chart { width: 100%; height: auto; display: block; }
.chart text { font-size: 11px; fill: #57606a; }
.chart .grid { stroke: #eaeef2; stroke-width: 1; }
.legend span { display: inline-block; margin-right: 16px; font-size: 13px; }
.legend i { display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }
ul { font-size: 13px; } .note { font-size: 13px; color: #57606a; }
"""


def write_report(path: Path, record: Record, analysis: dict, kstep: dict | None, title: str) -> None:
    parts = [
        f"<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title>"
        f"<style>{REPORT_STYLE}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in condition_items(record)) + "</ul>",
    ]
    headers, tables = summary_rows(record, analysis, kstep)
    parts.append("<h2>要点</h2>")
    for joint, rows in tables.items():
        parts.append(f"<h3>{html.escape(joint)}</h3>" + html_table(headers, rows))
    if kstep:
        names = [n for n in (*SERIES[1:], *BASELINES) if n in kstep["by_h"]]
        steps = [h for h in (1, 2, 4, kstep["k"]) if h <= kstep["k"]]
        rows = [[LABELS[n], *[fmt(kstep["by_h"][n][h - 1], 2) for h in sorted(set(steps))]] for n in names]
        parts.append("<h2>何行先まで当たるか（動かしている関節の RMS、度）</h2>")
        parts.append(html_table(["", *[f"{h}行先" for h in sorted(set(steps))]], rows))

    legend = "".join(
        f"<span><i style='background:{COLORS[n]}'></i>{html.escape(LABELS[n])}</span>"
        for n in ("cmd", *SERIES)
        if n == "cmd" or n in record.series
    )
    plan = record.meta["plan"]
    for joint, result in analysis.items():
        j = record.motors.index(joint)
        segments = [s for s in plan if s["joint"] == joint]
        t0, t1 = segments[0]["t0"] - 0.3, segments[-1]["t1"]
        position = [("cmd", record.t_cmd, record.cmd[:, j], True)]
        error = []
        for name in SERIES:
            if name in record.series:
                position.append((name, record.t_obs, record.series[name][:, j], False))
                error.append((name, record.t_obs, record.series[name][:, j] - record.cmd[:, j], False))
        parts.append(f"<h2>{html.escape(joint)}</h2><div class='legend'>{legend}</div>")
        parts.append(svg_chart(position, t0, t1, "角度［度］"))
        parts.append(svg_chart(error, t0, t1, "指令とのずれ［度］", height=170, robust=True))
        step_rows = []
        for name in SERIES:
            for step in result.get(name, {}).get("steps", []):
                step_rows.append([
                    f"{LABELS[name]} {step['name'].split('/')[-1]}",
                    fmt(step["target"], 1),
                    fmt(step["offset"], 2, signed=True),
                    fmt(step["t50_ms"], 0),
                    fmt(step["t90_ms"], 0),
                    fmt(step["overshoot_pct"], 0),
                ])
        parts.append("<h3>ステップ</h3>")
        parts.append(html_table(["", "指令[度]", "垂れ[度]", "50%[ms]", "90%[ms]", "行き過ぎ[%]"], step_rows))
        chirp_rows = []
        for name in SERIES:
            chirp = result.get(name, {}).get("chirp")
            if chirp:
                chirp_rows.append(
                    [LABELS[name]]
                    + [f"{fmt(w['gain'], 2)} / {fmt(w['lag_ms'], 0)}ms" for w in chirp["windows"]]
                )
        if chirp_rows:
            freqs = next(result[n]["chirp"]["windows"] for n in SERIES if n in result and result[n]["chirp"])
            parts.append("<h3>チャープ（振幅比 / 遅れ）</h3>")
            parts.append(html_table(["", *[f"{w['f_hz']:.1f}Hz" for w in freqs]], chirp_rows))

    parts.append(
        "<h2>読み方</h2><ul class='note'>"
        "<li>垂れ: ステップの最後 30% の平均位置 − 指令。負なら指令より下（手前）で止まっている</li>"
        "<li>50% / 90%: 指令が変わってから、実際に落ち着いた位置までの変化の 50% / 90% に届くまでの時間</li>"
        "<li>チャープ遅れ: 指令の正弦波と応答の相互相関が最大になるずれ。振幅比が 1 を下回る周波数から先は追従できていない</li>"
        "<li>sim−実機: 関節を動かしている間の、sim と実機の角度の差の RMS</li>"
        "<li>k 行先の予測: 実機の位置と速度から sim を始め、同じ指令で k 行先の角度を当てたときの誤差。"
        "「指令どおりと仮定」より小さくなければ、その sim のサーボは予測の役に立っていない</li>"
        "<li>関節の読み替え（実機の度 = sim のラジアン）が合っている前提。開始姿勢の外れは可動域の端に寄せている</li>"
        "</ul></body></html>"
    )
    path.write_text("\n".join(parts))


def to_json(value):
    """nan を null にして JSON に書けるようにする。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    return value


def main() -> int:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists() or not csv_path.with_suffix(".json").exists():
        print(f"記録が見つかりません: {csv_path}（同名の .json も必要です）", flush=True)
        return 1
    sim_path = args.sim
    if sim_path is None:
        try:
            params = args.params
            if params is None and csv_path.with_name(f"{csv_path.stem}_servo.json").exists():
                params = csv_path.with_name(f"{csv_path.stem}_servo.json")
            sim_path = replay_on_host(csv_path, args.host, args.offset, args.kstep, params)
        except subprocess.CalledProcessError as error:
            print(f"{args.host} での再生に失敗しました: {error}", flush=True)
            print("ssh で入れるか、~/so101/mujoco に record.py があるかを確かめてください。", flush=True)
            return 1

    record = Record(csv_path, sim_path)
    analysis = {joint: analyze_joint(record, joint) for joint in record.meta["joints"]}
    kstep = kstep_errors(record, args.kstep)
    headers, tables = summary_rows(record, analysis, kstep)
    for joint, rows in tables.items():
        print(f"\n{joint}", flush=True)
        print_table(headers, rows)
    if "real" not in record.series:
        print("\n実機の位置がない記録（--dry-run）なので、sim どうしだけを並べました。", flush=True)

    stem = csv_path.with_suffix("")
    report = Path(f"{stem}_report.html")
    write_report(report, record, analysis, kstep, f"ステップ応答の比較 {csv_path.stem}")
    metrics = Path(f"{stem}_metrics.json")
    metrics.write_text(json.dumps(to_json({"joints": analysis, "kstep": kstep}), ensure_ascii=False, indent=2) + "\n")
    print(f"\n図と表: {report}", flush=True)
    print(f"数値: {metrics}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
