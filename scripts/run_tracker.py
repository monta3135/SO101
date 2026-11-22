#!/usr/bin/env python3
"""SuperMimic で学習した追従方策で、記録したモーションをフォロワーに追わせる。

方策の計算は、SuperMimic の venv（~/supermimic_ws）で動く so101_policy_server.py を子プロセスとして
起動して任せ、標準入出力で関節の状態を渡して目標角を受け取る。LeRobot の venv には学習コードがなく、
SuperMimic の venv には LeRobot がないため、プロセスを分けている。

クリップごとに、次のモードで走らせて比べる。
  policy  方策が出した目標角をサーボに送る
  replay  参照の関節角を --replay-lead 周期だけ先読みし、そのまま目標角として送る（比べるための基準）

--backend sim は実機の代わりに、学習と同じ MuJoCo モデル（サーバーの中）を同じ周期で動かす。
単位の換算や安全のための制限が方策の動きを邪魔しないかを、実機の前に確かめるためのもの。

方策は指令がすぐ効く環境で学習しているが、実機のサーボは 1〜2 周期遅れて動く。--predict N を
付けると、送ったがまだ効いていない直近 N 周期ぶんの指令を学習と同じモデルで先に効かせ、
N 周期後の状態と参照の時刻を方策に渡す（遅れを補う）。

方策は物を見ないので、缶を掴むクリップでは録画のときと同じ場所に缶がないと掴めない。--place-can T を
付けると、各クリップの前に最初の T 秒（缶を掴み終える時刻）をグリッパを開いたまま再生して止まり、
グリッパの間に缶を置いてもらってから、同じ道を逆にたどって最初の姿勢へ戻る。

各クリップの前に、参照の最初の姿勢へ滑らかに移動して 1 秒止まってから始める。
方策の目標角は、参照から --max-deviation 度、前の周期の目標角から --max-slew 度までに切る。
関節が参照から --abort-error 度以上ずれた状態が 0.3 秒続いたら、そこで止めて残りは走らせない。
終了時（Ctrl+C を含む）は開始時の姿勢へ戻してから切断する。

角度の換算は so101_motions.py（参照を作ったスクリプト）の逆。腕の関節は度とラジアンをそのまま
読み替え、gripper の 0〜100 は関節の可動域（閉じた端〜開いた端）に線形に割り当てる。

記録は outputs/tracker/<日時>_<backend>/ に、走行ごとの CSV と summary.json。

  uv run python scripts/run_tracker.py --backend sim
  uv run python scripts/run_tracker.py --backend real --clips 0 --seconds 5
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import math
import os
import statistics
import subprocess
import termios
import time
from collections import deque
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from track_face import MOTORS, move_smoothly, read_present, safe_limits

WS = Path.home() / "supermimic_ws"
SERVER_PYTHON = WS / "supermimic/.venv/bin/python"
SERVER_SCRIPT = WS / "supermimic/src/supermimic/scripts/so101_policy_server.py"
GRIPPER_RANGE = (-0.17453, 1.74533)  # so101_new_calib.xml の gripper の可動域 [rad]
ABORT_AFTER_S = 0.3  # 参照とのずれが大きい状態がこれだけ続いたら止める
SETTLE_S = 1.0  # 最初の姿勢に着いてから始めるまで待つ時間
START_TOLERANCE = 5.0  # 最初の姿勢にこれより近ければ、届ききらなくても始める（度）
MAX_READ_FAILURES = 5  # 位置がこの回数続けて読めなかったら止める


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SuperMimic の追従方策でモーションを再生する")
    parser.add_argument("--backend", choices=("sim", "real"), default="sim")
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--run-dir", type=Path, default=WS / "runs/sm_so101_1120")
    parser.add_argument("--checkpoint", default="actor_0001200.pt")
    parser.add_argument(
        "--motion-file", type=Path, default=WS / "motions/so101_lift_can_trim/holdout_packed"
    )
    parser.add_argument("--clips", default="all", help="走らせるクリップの番号（カンマ区切り）か all")
    parser.add_argument("--modes", default="policy,replay", help="走らせるモード（カンマ区切り）")
    parser.add_argument("--seconds", type=float, help="各クリップの先頭からこの秒数だけ走らせる")
    parser.add_argument("--replay-lead", type=int, default=1, help="replay で先読みする周期数")
    parser.add_argument("--predict", type=int, default=0, help="方策に渡す状態を、この周期数だけ先へ予測する")
    parser.add_argument("--max-deviation", type=float, default=20.0, help="目標角と参照の差の上限（度）")
    parser.add_argument("--max-slew", type=float, default=10.0, help="1 周期で目標角が動いてよい量（度）")
    parser.add_argument("--abort-error", type=float, default=25.0, help="これ以上ずれ続けたら止める（度）")
    parser.add_argument("--speed", type=float, default=30.0, help="最初の姿勢との行き来の最大角速度（度/秒）")
    parser.add_argument("--margin", type=float, default=5.0, help="キャリブレーションの可動域から空ける角度（度）")
    parser.add_argument("--sim-delay", type=int, default=0, help="sim で指令が効くまでの遅れ（周期数）")
    parser.add_argument("--yes", action="store_true", help="開始前の確認を省く")
    parser.add_argument(
        "--place-can", type=float, help="缶を置く場所を見せる: 各クリップの前に、最初のこの秒数をグリッパを開いたまま再生して止まる"
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="途中で止まった走行があっても、残りを走らせる（通信が切れたときは止める）"
    )
    parser.add_argument("--out", type=Path, help="記録の保存先ディレクトリ")
    return parser.parse_args()


def to_sim(values: dict[str, float]) -> np.ndarray:
    """LeRobot の値（腕は度、gripper は 0〜100）を sim の関節角 [rad] にする。"""
    q = np.radians([values[m] for m in MOTORS])
    low, high = GRIPPER_RANGE
    q[-1] = low + values["gripper"] / 100 * (high - low)
    return q


def to_robot(q: np.ndarray) -> dict[str, float]:
    """to_sim の逆。"""
    values = {m: math.degrees(float(x)) for m, x in zip(MOTORS, q)}
    low, high = GRIPPER_RANGE
    values["gripper"] = (float(q[-1]) - low) / (high - low) * 100
    return values


def robot_limits(follower: SO101Follower, margin: float) -> dict[str, tuple[float, float]]:
    """送ってよい範囲（LeRobot の値）。gripper は 0〜100 で正規化されているので度の余白は使わない。"""
    limits = {m: safe_limits(follower, m, margin) for m in MOTORS if m != "gripper"}
    limits["gripper"] = (0.0, 100.0)
    return limits


class PolicyServer:
    """so101_policy_server.py との 1 行 1 JSON のやりとり。"""

    def __init__(self, args: argparse.Namespace, log_path: Path):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", WANDB_MODE="offline", TORCHDYNAMO_DISABLE="1")
        self.log_path = log_path
        self.log = log_path.open("w")
        command = [
            str(SERVER_PYTHON),
            str(SERVER_SCRIPT),
            "--run-dir",
            str(args.run_dir),
            "--checkpoint",
            args.checkpoint,
            "--motion-file",
            str(args.motion_file),
        ]
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=env
        )
        self._read()  # {"ready": true}

    def request(self, **message) -> dict:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        return self._read()

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"方策サーバーが止まりました。{self.log_path} を見てください。")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"方策サーバー: {reply['error']}")
        return reply

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:  # 止まっていても、止まらなくても、最後は確実に終わらせる
            self.proc.kill()
        self.log.close()


class SimArm:
    """サーバーの中の MuJoCo を、実機と同じ口（LeRobot の値の読み書き）で動かす。
    時間は待たずに、1 回の書き込みで 1 周期ぶん進める。読む角度はエンコーダの 1 目盛りに丸め、
    指令は delay 周期遅れて効かせる（実機のサーボの遅れの見積もり用）。"""

    def __init__(self, server: PolicyServer, dt: float, follower: SO101Follower, delay: int):
        self.server = server
        self.dt = dt
        self.delay = delay
        self.q = np.zeros(len(MOTORS))
        self.t = 0.0
        self.pending: list[dict[str, float]] = []
        gripper = follower.bus.calibration["gripper"]
        # 腕は 1 回転 4096 目盛り（LeRobot の度は 360/4095 刻み）、gripper は可動域を 0〜100 に割った刻み
        self.ticks = {m: 360 / 4095 for m in MOTORS}
        self.ticks["gripper"] = 100 / (gripper.range_max - gripper.range_min)

    def read(self) -> tuple[np.ndarray, float]:
        values = to_robot(self.q)
        return to_sim({m: round(values[m] / self.ticks[m]) * self.ticks[m] for m in MOTORS}), self.t

    def write(self, goal: dict[str, float]) -> None:
        self.pending.append(goal)
        goal = self.pending.pop(0) if len(self.pending) > self.delay else self.pending[0]
        reply = self.server.request(cmd="sim_step", target=to_sim(goal).tolist())
        self.q = np.array(reply["q"])
        self.t += self.dt

    def move_to(self, q: np.ndarray, speed: float) -> None:
        self.q = np.array(self.server.request(cmd="sim_reset", q=q.tolist())["q"])
        self.pending = [to_robot(q)] * self.delay
        self.t += SETTLE_S

    def wait_until(self, deadline: float) -> None:
        pass

    def clock(self) -> float:
        return self.t


class RealArm:
    def __init__(self, follower: SO101Follower):
        self.follower = follower

    def read(self) -> tuple[np.ndarray, float]:
        start = time.perf_counter()
        try:
            positions = self.follower.bus.sync_read("Present_Position", num_retry=1)
        except OSError:
            release_port(self.follower)
            raise
        return to_sim(positions), (start + time.perf_counter()) / 2

    def write(self, goal: dict[str, float]) -> None:
        self.follower.send_action({f"{m}.pos": goal[m] for m in MOTORS})

    def move_to(self, q: np.ndarray, speed: float) -> None:
        goal = {f"{m}.pos": v for m, v in to_robot(q).items()}
        if not move_smoothly(self.follower, goal, speed, timeout_s=3.0):
            # たたんだ姿勢から重力に逆らって動くと、数度届かずに止まることがある
            present = read_present(self.follower)
            worst = max(abs(present[key] - goal[key]) for key in goal)
            if worst > START_TOLERANCE:
                raise RuntimeError("最初の姿勢に届きませんでした。")
            print(f"  最大 {worst:.1f} 度のずれを残して始めます。", flush=True)
        time.sleep(SETTLE_S)

    def wait_until(self, deadline: float) -> None:
        time.sleep(max(0.0, deadline - time.perf_counter()))

    def clock(self) -> float:
        return time.perf_counter()


def connect_follower(follower: SO101Follower, tries: int = 3) -> None:
    """接続する。LeRobot は接続時の設定の書き込みをやり直さず、返事が 1 回欠けただけで失敗するので、
    何回かやり直す（やり直しのたびにトルクを切ってから設定し直す）。"""
    for attempt in range(1, tries + 1):
        try:
            follower.connect(calibrate=False)
            # ほかのプログラムが同じポートを開くと返事を取り合って走行が止まるので、閉じるまで開けなくする
            fcntl.ioctl(follower.bus.port_handler.ser.fileno(), termios.TIOCEXCL)
            return
        except ConnectionError as error:
            print(f"接続に失敗しました（{attempt}/{tries} 回目）: {error}", flush=True)
            if follower.bus.is_connected:
                follower.bus.disconnect(disable_torque=False)
    raise RuntimeError("フォロワーに接続できませんでした。")


def release_port(follower: SO101Follower) -> None:
    """読み書きの途中で例外が出ると、SDK が使用中の印を残したままにし、それ以降の通信を
    「Port is in use!」ですべて断る。印を外して、溜まった受信も捨てる。"""
    follower.bus.port_handler.is_using = False
    follower.bus.port_handler.ser.reset_input_buffer()


def clamp_goal(goal: dict[str, float], limits: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {m: min(max(goal[m], limits[m][0]), limits[m][1]) for m in MOTORS}


def show_can_place(arm, server, clip: int, ref: np.ndarray, args, limits, dt: float, out: Path, name: str) -> None:
    """缶を置く場所を見せる。最初の args.place_can 秒をグリッパを開いたまま再生して止まり、置いてもらったら
    同じ道を逆にたどって最初の姿勢へ戻る（録画で缶へ近づいた道なので、缶に当たらない）。"""
    seg = ref[: max(2, round(args.place_can / dt))].copy()
    seg[:, -1] = np.maximum.accumulate(seg[:, -1])  # gripper は一度開いたら閉じない
    for part, path_ref in (("show", seg), ("back", seg[::-1].copy())):
        print(f"[{name}] 缶の場所{'へ' if part == 'show' else 'から戻る'}: {len(path_ref) * dt:.1f} 秒", flush=True)
        result = run_clip(arm, server, clip, path_ref, "replay", args, limits, dt, out / f"clip{clip}_{part}.csv")
        if result["stopped"] != "completed":
            raise RuntimeError(result["stopped"])
        if part == "show" and not args.yes:
            input("グリッパの間に缶を置き、手を離してから Enter（Ctrl+C で中止）: ")


def run_clip(arm, server, clip: int, ref: np.ndarray, mode: str, args, limits, dt: float, path: Path) -> dict:
    """1 クリップを 1 モードで走らせ、1 周期 1 行の CSV を書く。結果の要約を返す。"""
    max_dev = math.radians(args.max_deviation)
    max_slew = math.radians(args.max_slew)
    abort = math.radians(args.abort_error)
    arm.move_to(ref[0], args.speed)
    q_prev, t_prev = arm.read()
    last_target = ref[0].copy()
    # 送ったがまだ効いていない指令（--predict 用）。移動の最後の指令は最初の姿勢
    sent = deque([ref[0]] * args.predict, maxlen=args.predict)
    over_since: float | None = None
    failures = 0
    stats = {
        "ticks": 0,
        "deviation_clamped": 0,
        "slew_clamped": 0,
        "read_failures": 0,
        "connection_lost": False,
        "act_ms": [],
        "loop_ms": [],
    }
    # pred は方策に渡した関節角（--predict なしなら q と同じ）
    names = [f"{kind}.{m}" for kind in ("ref", "q", "v", "raw", "target", "pred") for m in MOTORS]
    errors = []
    qs = []
    stopped = "completed"
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["k", "t", "loop_ms", "act_ms", *names])
        t0 = arm.clock()
        last_loop = None
        for k in range(len(ref)):
            arm.wait_until(t0 + k * dt)
            loop_start = arm.clock()
            try:
                q, t = arm.read()
            except OSError as error:  # 読めなかった周期は指令を変えずに飛ばす
                failures += 1
                stats["read_failures"] += 1
                if failures >= MAX_READ_FAILURES:
                    stopped = f"aborted: 位置が {failures} 回続けて読めません: {error}"
                    stats["connection_lost"] = True
                    break
                continue
            failures = 0
            # 最初の周期は止まっている。直前の読みとの間隔が短すぎて、差分が目盛りの揺れを拡大するので使わない
            v = (q - q_prev) / (t - t_prev) if k else np.zeros_like(q)
            act_start = time.perf_counter()
            state = {"q": q.tolist(), "v": v.tolist()}
            if mode == "policy":
                if args.predict:
                    state = server.request(cmd="predict", targets=[s.tolist() for s in sent], **state)
                reply = server.request(cmd="act", clip=clip, frame=k + args.predict, **state)
                raw = np.array(reply["target"])
            else:
                raw = ref[min(k + args.replay_lead, len(ref) - 1)].copy()
            act_ms = (time.perf_counter() - act_start) * 1000
            target = np.clip(raw, ref[k] - max_dev, ref[k] + max_dev)
            stats["deviation_clamped"] += int(np.any(target != raw))
            slewed = np.clip(target, last_target - max_slew, last_target + max_slew)
            stats["slew_clamped"] += int(np.any(slewed != target))
            goal = clamp_goal(to_robot(slewed), limits)
            arm.write(goal)
            last_target = to_sim(goal)
            sent.append(last_target)
            if last_loop is not None:
                stats["loop_ms"].append((loop_start - last_loop) * 1000)
            last_loop = loop_start
            stats["act_ms"].append(act_ms)
            stats["ticks"] += 1
            writer.writerow([
                k,
                round(t - t0, 4),
                round(stats["loop_ms"][-1], 2) if stats["loop_ms"] else "",
                round(act_ms, 2),
                *np.round(np.concatenate([ref[k], q, v, raw, last_target, state["q"]]), 5),
            ])
            errors.append(np.abs(q - ref[k]))
            qs.append(q)
            if np.any(np.abs(q - ref[k]) > abort):
                over_since = t if over_since is None else over_since
                if t - over_since >= ABORT_AFTER_S:
                    worst = MOTORS[int(np.argmax(np.abs(q - ref[k])))]
                    stopped = f"aborted: {worst} が参照から {args.abort_error:g} 度以上ずれたままです"
                    break
            else:
                over_since = None
            q_prev, t_prev = q, t
    return summarize(server, ref, np.array(qs), np.array(errors), stats, stopped)


def summarize(server, ref: np.ndarray, qs: np.ndarray, errors: np.ndarray, stats: dict, stopped: str) -> dict:
    result = {"stopped": stopped, **{k: v for k, v in stats.items() if not isinstance(v, list)}}
    if len(qs):
        body = server.request(cmd="body_error", q=qs.tolist(), ref=ref[: len(qs)].tolist())["per_step"]
        result.update(
            joint_err_rad=round(float(np.linalg.norm(errors, axis=1).mean()), 4),
            body_err_cm=round(float(np.mean(body)) * 100, 3),
            per_joint_err_deg={m: round(float(np.degrees(errors[:, i]).mean()), 2) for i, m in enumerate(MOTORS)},
            max_err_deg=round(float(np.degrees(errors).max()), 1),
        )
    for name in ("act_ms", "loop_ms"):
        values = sorted(stats[name])
        if values:
            result[f"{name}_median"] = round(statistics.median(values), 2)
            result[f"{name}_p99"] = round(values[int(len(values) * 0.99)], 2)
    return result


def main() -> int:
    args = parse_args()
    stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out = args.out or Path("outputs/tracker") / f"{stamp}_{args.backend}"
    out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in ("policy", "replay"):
            raise SystemExit(f"知らないモードです: {mode}")

    # キャリブレーションはファイルから読まれるので、接続しなくても可動域がわかる
    follower = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    limits = robot_limits(follower, args.margin)
    print("方策サーバーを起動します（20 秒ほどかかります）。", flush=True)
    server = PolicyServer(args, out / "server.log")
    info = server.request(cmd="info")
    if info["joint_names"] != MOTORS:
        raise SystemExit(f"関節の順番が違います: {info['joint_names']}")
    dt = info["dt"]
    clips = range(len(info["clips"])) if args.clips == "all" else [int(c) for c in args.clips.split(",")]
    plan = []
    for clip in clips:
        ref = np.array(server.request(cmd="reference", clip=clip)["joint_pos"])
        if args.seconds:
            ref = ref[: int(round(args.seconds / dt))]
        plan.append((clip, ref))
    total = sum(len(ref) for _, ref in plan) * dt * len(modes)
    print(f"クリップ {', '.join(str(c) for c, _ in plan)} を {', '.join(modes)} で走らせます（計 {total:.0f} 秒）。", flush=True)

    summary = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "dt": dt,
        "runs": [],
    }
    arm = None
    start_pose: np.ndarray | None = None
    try:
        if args.backend == "real":
            # 接続するとトルクが入るので、確認を取ってからつなぐ
            if not args.yes:
                input("アームの周りを空けて Enter で開始します（Ctrl+C で中止）: ")
            connect_follower(follower)
            arm = RealArm(follower)
            start_pose, _ = arm.read()
            print("開始姿勢: " + ", ".join(f"{m} {v:+.1f}" for m, v in to_robot(start_pose).items()), flush=True)
        else:
            arm = SimArm(server, dt, follower, args.sim_delay)
        for clip, ref in plan:
            if args.place_can:
                show_can_place(arm, server, clip, ref, args, limits, dt, out, info["clips"][clip]["name"])
            for mode in modes:
                name = info["clips"][clip]["name"]
                print(f"[{name}] {mode}: {len(ref) * dt:.1f} 秒", flush=True)
                result = run_clip(arm, server, clip, ref, mode, args, limits, dt, out / f"clip{clip}_{mode}.csv")
                result.update(clip=clip, name=name, mode=mode)
                summary["runs"].append(result)
                print(
                    f"  関節 {result.get('joint_err_rad', float('nan')):.4f} rad、リンク "
                    f"{result.get('body_err_cm', float('nan')):.2f} cm、最大 {result.get('max_err_deg', float('nan'))} 度、"
                    f"制限が効いた周期 {result['deviation_clamped']}/{result['slew_clamped']}（{result['stopped']}）",
                    flush=True,
                )
                # 通信が切れたら、--keep-going でも残りは走らせない（次の移動も読めずに失敗する）
                if result["stopped"] != "completed" and (result["connection_lost"] or not args.keep_going):
                    raise RuntimeError(result["stopped"])
    except KeyboardInterrupt:
        print("\n停止します。", flush=True)
        summary["interrupted"] = True
    except (RuntimeError, OSError) as error:  # OSError は移動中などに通信が切れたとき
        print(f"\n{error}", flush=True)
        summary["error"] = str(error)
    finally:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if args.backend == "real" and follower.is_connected:
            try:
                release_port(follower)
                if start_pose is not None:
                    print("開始時の姿勢へ戻します。", flush=True)
                    goal = {f"{m}.pos": v for m, v in to_robot(start_pose).items()}
                    move_smoothly(follower, goal, args.speed, timeout_s=10.0)
                follower.disconnect()
            except Exception as error:  # 通信が切れていると戻すのも切断も失敗する
                if start_pose is None:
                    # 動かす前に止まった。サーボが 1 つも返事をしないのは、たいてい電源かケーブル
                    follower.bus.port_handler.closePort()
                    print(f"サーボと通信できません: {error}。フォロワーの電源（12 V）とケーブルを確かめてください。", flush=True)
                else:
                    print(f"戻す途中でエラー: {error}。アームを支えてから電源を切ってください。", flush=True)
        server.close()
    print(f"記録: {out}", flush=True)
    return 0 if not summary.get("error") and not summary.get("interrupted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
