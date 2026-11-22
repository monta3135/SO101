#!/usr/bin/env python3
"""学習済み ACT を固定し、その出力に小さな補正（残差）を足す方策を実機の強化学習で学ぶ。

ResFiT（Residual Off-Policy RL for Finetuning BC Policies）の考え方に沿った最小構成:
  実際に送る関節目標 = ACT の出力 + 残差方策の出力 × --residual-scale
残差方策の最後の層を 0 で初期化するので、学習前は ACT とまったく同じに動く。
報酬は人がキーで付ける。掴めた瞬間に g で 0.5、置けたら s で最後に 1、失敗・時間切れは追加なし。
学習は TD3（n-step リターン、Q 関数 2 本、LayerNorm）で、エピソードの合間に Ram の CPU で回す。
成功が --min-successes 本たまるまでは Q 関数だけを学習し、残差は 0 のまま（ACT そのまま）にする。

1エピソードの流れ:
  1. 缶を置いてアームを開始姿勢にしたら n（または Enter）で開始。q で終了。
  2. 動いている間、掴めた瞬間に g を押す（エピソードは続く）。置けたら s、失敗したら f で終える。
     --episode-s 秒で失敗扱い。
  3. アームは開始時の姿勢へゆっくり戻り、その間にデータの保存と学習を行う。

データと重みは --out に保存する。同じ --out で起動し直すと続きから学習する。
--selftest は実機をつながずに、記録済みデータセットのフレームで ACT と学習の流れだけ確かめる。
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERA_KEY = "observation.images.wrist"
TASK = "Grasp the coffee can and lift it"
DEFAULT_CAMERA = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
# 残差方策に見せる画像の大きさ。ACT は元の 640x480 のまま見る。
SMALL_H, SMALL_W = 72, 96
# 関節角 [度] をネットワークに入れる前に割る値。
JOINT_SCALE = 100.0
# 探索ノイズの1ステップごとの持ち越し率（0 で毎ステップ独立）。持ち越すと震えは減るが、エピソードごとに
# 偏りが残って Q がそれを成功の原因と取り違える。本物の画像の偽データで、罰 1.0 / lr 1e-4 のとき
# 偶然の残差は 0.8 で 0.57 まで育ち、0 では 0.09 にとどまったので、独立なノイズにする。
NOISE_ALPHA = 0.0
# g（掴めた）を押したステップに付ける途中の報酬。s（置けた）は最後のステップに 1。
# 掴みは成功・置きで失敗、を掴み損ねと区別するため。
GRASP_REWARD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT に残差方策を足して実機で強化学習する")
    parser.add_argument("--base", type=Path, required=True, help="ACT のチェックポイント")
    parser.add_argument("--out", type=Path, default=Path("outputs/residual/run"))
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--id", default="follower")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-s", type=float, default=30.0)
    parser.add_argument("--max-step-deg", type=float, default=15.0, help="1ステップで動いてよい最大角度")
    # デモの開始姿勢（47本の中央値付近）。起動時の姿勢ではなく、ここへ戻してから毎回始める。
    parser.add_argument(
        "--home",
        type=float,
        nargs=6,
        default=[-17.7, -99.7, 95.7, 60.2, -73.7, 1.6],
        help="エピソードを始める姿勢 [度]（pan lift elbow wflex roll grip の順）",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        nargs=6,
        default=[3.0, 3.0, 3.0, 3.0, 3.0, 5.0],
        help="各関節の残差の最大値 [度]（pan lift elbow wflex roll grip の順）",
    )
    parser.add_argument("--noise", type=float, default=0.2, help="探索ノイズの標準偏差（残差の最大値に対する比）")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=10)
    # Ram の CPU で 1 回の更新は batch 128 で約 94 ms。400 回で約 40 秒、缶を置き直す間に終わる。
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--updates-per-step", type=float, default=0.5, help="集めた1ステップあたりの更新回数")
    parser.add_argument("--max-updates", type=int, default=400, help="1エピソード後の更新回数の上限")
    parser.add_argument("--warmup-episodes", type=int, default=2, help="この数だけ集めるまでは学習しない")
    # 報酬がすべて 0 のうちに残差方策を更新すると、当てにならない Q の傾きに引っぱられて
    # 全関節が上限まで振り切れる（最初の試行で 0.92 に張り付いた）。成功がたまるまでは Q だけ学習する。
    parser.add_argument(
        "--min-successes", type=int, default=2, help="成功エピソードがこの数たまるまで残差方策を更新しない"
    )
    # 本物の画像を使った偽データ（8 エピソード、報酬あり 3 本、残差は結果と無関係）で比べた値。
    # 画像でエピソードを見分けられるので Q が偶然の相関を学び、0.5 / lr 3e-4 では残差が 2 エピソード分の
    # 更新で 0.75 まで育った。1.0 / lr 1e-4 では 0.19 にとどまった。本物の効果を拾うのは遅くなるが安全側に倒す。
    parser.add_argument(
        "--residual-penalty", type=float, default=1.0, help="残差の大きさ（-1〜1 の2乗平均）にかける罰の重み"
    )
    parser.add_argument("--actor-lr", type=float, default=1e-4, help="残差方策の学習率")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="学習済みの残差を足して動かすだけ。探索ノイズなし、データの保存も学習もしない",
    )
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-dataset", type=Path, default=Path("outputs/datasets/so101_lift_can_trim"))
    return parser.parse_args()


# ---------------------------------------------------------------- ACT（固定）


class BasePolicy:
    """学習済み ACT を CPU で動かし、1ステップぶんの関節目標 [度] を返す。"""

    def __init__(self, path: Path):
        self.policy = ACTPolicy.from_pretrained(path)
        self.policy.config.device = "cpu"
        self.policy.to("cpu").eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    @torch.inference_mode()
    def act(self, state: np.ndarray, image: np.ndarray) -> np.ndarray:
        obs = {"observation.state": state.astype(np.float32), CAMERA_KEY: image}
        obs = prepare_observation_for_inference(obs, torch.device("cpu"), TASK, "so_follower")
        action = self.post(self.policy.select_action(self.pre(obs)))
        return action.squeeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------- 残差方策と Q 関数


class Encoder(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, SMALL_H, SMALL_W)).numel()
        self.head = nn.Sequential(nn.Linear(n, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(image).flatten(1))


def mlp(inp: int, out: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, out),
    )


class Actor(nn.Module):
    """画像特徴・関節角・ACT の出力から、[-1, 1] の残差を出す。"""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.net = mlp(feature_dim + 12, 6)
        # 学習前は残差 0、つまり ACT そのものとして動かす。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, state, base):
        return torch.tanh(self.net(torch.cat([feat, state, base], dim=1)))


class Critic(nn.Module):
    """画像エンコーダを持つ Q 関数2本。エンコーダは Q の学習でだけ更新する。"""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(feature_dim)
        self.q1 = mlp(feature_dim + 18, 1)
        self.q2 = mlp(feature_dim + 18, 1)

    def forward(self, feat, state, base, residual):
        x = torch.cat([feat, state, base, residual], dim=1)
        return self.q1(x), self.q2(x)


def to_image_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0


def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ と同じ、上下左右に数画素ずらすデータ拡張。"""
    n, _, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    out = torch.empty_like(images)
    xs = torch.randint(0, 2 * pad + 1, (n,))
    ys = torch.randint(0, 2 * pad + 1, (n,))
    for i in range(n):
        out[i] = padded[i, :, ys[i] : ys[i] + h, xs[i] : xs[i] + w]
    return out


class Agent:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.critic = Critic()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor = Actor()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.updates = 0
        self.noise = np.zeros(6)

    def reset_noise(self) -> None:
        self.noise = np.zeros(6)

    @torch.no_grad()
    def act(self, small_image: np.ndarray, state: np.ndarray, base: np.ndarray, explore: bool) -> np.ndarray:
        feat = self.critic.encoder(to_image_tensor(small_image[None]))
        s = torch.from_numpy(state[None] / JOINT_SCALE).float()
        b = torch.from_numpy(base[None] / JOINT_SCALE).float()
        r = self.actor(feat, s, b)[0].numpy()
        if explore:
            step = np.random.normal(0.0, self.args.noise, size=6) * np.sqrt(1.0 - NOISE_ALPHA**2)
            self.noise = NOISE_ALPHA * self.noise + step
            r = r + self.noise
        return np.clip(r, -1.0, 1.0).astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor], update_actor: bool) -> dict[str, float]:
        args = self.args
        img = random_shift(batch["image"])
        next_img = random_shift(batch["next_image"])
        with torch.no_grad():
            next_feat = self.critic_target.encoder(next_img)
            noise = (torch.randn_like(batch["residual"]) * 0.2).clamp(-0.5, 0.5)
            next_r = (self.actor_target(next_feat, batch["next_state"], batch["next_base"]) + noise).clamp(-1, 1)
            tq1, tq2 = self.critic_target(next_feat, batch["next_state"], batch["next_base"], next_r)
            target = batch["reward"] + batch["discount"] * torch.min(tq1, tq2)

        feat = self.critic.encoder(img)
        q1, q2 = self.critic(feat, batch["state"], batch["base"], batch["residual"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": critic_loss.item(), "q": q1.mean().item()}
        self.updates += 1
        if self.updates % 2 == 0:
            if update_actor:
                feat = feat.detach()
                r = self.actor(feat, batch["state"], batch["base"])
                q1, q2 = self.critic(feat, batch["state"], batch["base"], r)
                # 2本の Q の小さいほうを使い、Q がはっきり良くなるときだけ残差を大きくするよう大きさに罰をかける。
                actor_loss = -torch.min(q1, q2).mean() + args.residual_penalty * (r**2).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                info["actor_loss"] = actor_loss.item()
                info["residual"] = r.abs().mean().item()
            with torch.no_grad():
                for net, target_net in ((self.critic, self.critic_target), (self.actor, self.actor_target)):
                    for p, tp in zip(net.parameters(), target_net.parameters()):
                        tp.mul_(0.995).add_(p, alpha=0.005)
        return info

    def save(self, path: Path) -> None:
        torch.save(
            {
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "updates": self.updates,
            },
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu")
        for key in ("critic", "critic_target", "actor", "actor_target", "critic_opt", "actor_opt"):
            getattr(self, key).load_state_dict(state[key])
        self.updates = state["updates"]


# ---------------------------------------------------------------- リプレイバッファ


class ReplayBuffer:
    """エピソードごとの配列を持ち、n-step リターンでサンプルする。"""

    KEYS = ("image", "state", "base", "residual")

    def __init__(self, gamma: float, n_step: int):
        self.gamma = gamma
        self.n_step = n_step
        self.episodes: list[dict[str, np.ndarray]] = []

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        self.episodes.append(episode)

    @property
    def num_steps(self) -> int:
        return sum(len(ep["state"]) for ep in self.episodes)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        lengths = np.array([len(ep["state"]) for ep in self.episodes])
        ep_ids = np.random.choice(len(self.episodes), size=batch_size, p=lengths / lengths.sum())
        out = {k: [] for k in ("image", "state", "base", "residual", "next_image", "next_state", "next_base")}
        rewards, discounts = [], []
        for e in ep_ids:
            ep = self.episodes[e]
            length = len(ep["state"])
            t = np.random.randint(length)
            end = min(t + self.n_step, length)
            # 報酬は最後のステップにだけ付く（success なら 1）。
            ret = sum(self.gamma ** (k - t) * ep["reward"][k] for k in range(t, end))
            terminal = end == length
            nxt = min(end, length - 1)
            for k in ("image", "state", "base", "residual"):
                out[k].append(ep[k][t])
            out["next_image"].append(ep["image"][nxt])
            out["next_state"].append(ep["state"][nxt])
            out["next_base"].append(ep["base"][nxt])
            rewards.append(ret)
            discounts.append(0.0 if terminal else self.gamma ** (end - t))
        batch = {
            "image": to_image_tensor(np.stack(out["image"])),
            "next_image": to_image_tensor(np.stack(out["next_image"])),
            "residual": torch.from_numpy(np.stack(out["residual"])).float(),
            "reward": torch.tensor(rewards, dtype=torch.float32)[:, None],
            "discount": torch.tensor(discounts, dtype=torch.float32)[:, None],
        }
        for k in ("state", "base", "next_state", "next_base"):
            batch[k] = torch.from_numpy(np.stack(out[k]) / JOINT_SCALE).float()
        return batch


def save_episode(out: Path, index: int, episode: dict[str, np.ndarray]) -> None:
    np.savez_compressed(out / "episodes" / f"episode_{index:04d}.npz", **episode)


def load_episodes(out: Path) -> list[dict[str, np.ndarray]]:
    episodes = []
    for f in sorted(glob.glob(str(out / "episodes" / "episode_*.npz"))):
        with np.load(f) as data:
            episodes.append({k: data[k] for k in data.files})
    return episodes


def train(agent: Agent, buffer: ReplayBuffer, new_steps: int, args: argparse.Namespace) -> None:
    if len(buffer.episodes) < args.warmup_episodes:
        print(f"学習はエピソードが {args.warmup_episodes} 本たまってから始めます。")
        return
    # 掴めた・置けたのどちらかで報酬が付いたエピソードを数える。報酬が全部 0 だと比べる相手がない。
    successes = sum(int(ep["reward"].sum() > 0) for ep in buffer.episodes)
    update_actor = successes >= args.min_successes
    if not update_actor:
        print(
            f"報酬の付いたエピソードが {args.min_successes} 本たまるまでは Q 関数だけ学習し、"
            f"残差は 0 のまま（ACT そのまま）にします（今 {successes} 本）。"
        )
    n = min(args.max_updates, max(1, int(new_steps * args.updates_per_step)))
    start = time.monotonic()
    info = {}
    for _ in range(n):
        info |= agent.update(buffer.sample(args.batch), update_actor)
    print(
        f"{n} 回更新しました（{time.monotonic() - start:.0f} 秒、累計 {agent.updates} 回）: "
        + " ".join(f"{k}={v:.3f}" for k, v in info.items())
    )


def downscale(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 実機


class Keys:
    """n / Enter: 開始、g: 掴めた、s: 置けた（成功）、f: 失敗、q: 終了。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.last: str | None = None

    def dispatch(self, name: str) -> None:
        key = {"enter": "n"}.get(name, name.lower())
        if key in ("n", "g", "s", "f", "q"):
            with self.lock:
                self.last = key

    def pop(self) -> str | None:
        with self.lock:
            key, self.last = self.last, None
            return key


def read(robot) -> tuple[np.ndarray, np.ndarray]:
    obs = robot.get_observation()
    state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
    return state, obs["wrist"]


def send(robot, target: np.ndarray) -> None:
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)})


def move_to(robot, target: np.ndarray, seconds: float, fps: int) -> None:
    start, _ = read(robot)
    steps = max(1, int(seconds * fps))
    for i in range(1, steps + 1):
        send(robot, start + (target - start) * i / steps)
        time.sleep(1.0 / fps)


def run_episode(robot, base: BasePolicy, agent: Agent, keys: Keys, args) -> tuple[dict, bool, bool, bool]:
    scale = np.array(args.residual_scale, dtype=np.float32)
    frames = {k: [] for k in ("image", "state", "base", "residual")}
    base.reset()
    agent.reset_noise()
    keys.pop()
    print("開始しました。掴めたら g、置けたら s、失敗したら f を押してください。")
    success = quit_after = False
    grasp_step = None
    start = time.monotonic()
    period = 1.0 / args.fps
    while True:
        tick = time.monotonic()
        state, image = read(robot)
        base_action = base.act(state, image)
        small = downscale(image)
        residual = agent.act(small, state, base_action, explore=not args.eval)
        send(robot, base_action + residual * scale)
        for k, v in zip(frames, (small, state, base_action, residual)):
            frames[k].append(v)

        key = keys.pop()
        if key == "g" and grasp_step is None:
            grasp_step = len(frames["state"]) - 1
            print(f"掴めた（+{GRASP_REWARD}）。そのまま続けます。")
        if key in ("s", "f"):
            success = key == "s"
            break
        if key == "q":
            quit_after = True
            break
        if time.monotonic() - start > args.episode_s:
            print("時間切れなので失敗として扱います。")
            break
        time.sleep(max(0.0, period - (time.monotonic() - tick)))

    episode = {k: np.stack(v) for k, v in frames.items()}
    reward = np.zeros(len(episode["state"]), dtype=np.float32)
    # 置けたのに g を押し忘れたときは、掴めていたはずなので最後のステップに掴みの点も付ける。
    if grasp_step is None and success:
        grasp_step = len(reward) - 1
    if grasp_step is not None:
        reward[grasp_step] += GRASP_REWARD
    if success:
        reward[-1] += 1.0
    episode["reward"] = reward
    return episode, success, grasp_step is not None, quit_after


def run_robot(args: argparse.Namespace, base: BasePolicy, agent: Agent, buffer: ReplayBuffer, log: list) -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.utils.keyboard_input import create_key_listener

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.id,
            cameras={"wrist": OpenCVCameraConfig(index_or_path=args.camera, width=640, height=480, fps=30)},
            max_relative_target=args.max_step_deg,
        )
    )
    keys = Keys()
    eval_log: list[dict] = []
    if args.eval:
        if not (args.out / "agent.pt").exists():
            raise SystemExit(f"{args.out}/agent.pt がありません。先に学習してください。")
        print("動作確認モード: 学習済みの残差を足して動かすだけです（ノイズなし、学習なし）。")
    listener = create_key_listener(keys.dispatch, controls_help="n=開始 g=掴めた s=置けた f=失敗 q=終了")
    robot.connect()
    try:
        home = np.array(args.home, dtype=np.float32)
        print(f"開始姿勢 {np.round(home, 1).tolist()} へ移動します。")
        move_to(robot, home, seconds=3.0, fps=args.fps)
        while True:
            print("\n缶を置いてアームが開始姿勢に戻ったら n で開始します（q で終了）。")
            key = None
            while key not in ("n", "q"):
                key = keys.pop()
                time.sleep(0.05)
            if key == "q":
                break

            episode, success, grasped, quit_after = run_episode(robot, base, agent, keys, args)
            index = len(buffer.episodes)
            # ノイズを含めて、実際に足した残差の平均の大きさ [度]。振り切れていないかをここで見る。
            residual_deg = np.abs(episode["residual"]).mean(axis=0) * np.array(args.residual_scale)
            outcome = "置けた" if success else ("掴めたが置けなかった" if grasped else "掴めなかった")
            print(
                f"エピソード {index}: {outcome}（{len(episode['state'])} ステップ、報酬 {episode['reward'].sum():.1f}）、"
                f"残差の平均 {np.round(residual_deg, 1).tolist()} 度"
            )
            move_to(robot, home, seconds=3.0, fps=args.fps)

            if args.eval:
                eval_log.append({"success": success, "grasped": grasped, "steps": len(episode["state"])})
                (args.out / "eval_log.json").write_text(json.dumps(eval_log, indent=1))
                n = len(eval_log)
                grasps = sum(e["grasped"] for e in eval_log)
                places = sum(e["success"] for e in eval_log)
                print(f"動作確認 {n} 回: 掴めた {grasps} 回、置けた {places} 回（学習はしていません）")
                if quit_after:
                    break
                continue

            buffer.add_episode(episode)
            save_episode(args.out, index, episode)
            log.append(
                {
                    "episode": index,
                    "success": success,
                    "grasped": grasped,
                    "steps": len(episode["state"]),
                    "residual_deg": np.round(residual_deg, 2).tolist(),
                }
            )
            (args.out / "log.json").write_text(json.dumps(log, indent=1))
            recent = log[-10:]
            grasp_rate = sum(e.get("grasped", e["success"]) for e in recent) / len(recent)
            place_rate = sum(e["success"] for e in recent) / len(recent)
            print(f"直近 {len(recent)} 回: 掴めた {grasp_rate:.0%}、置けた {place_rate:.0%}")

            train(agent, buffer, len(episode["state"]), args)
            agent.save(args.out / "agent.pt")
            if quit_after:
                break
    finally:
        if listener is not None:
            listener.stop()
        robot.disconnect()


# ---------------------------------------------------------------- 実機なしの確認


def run_selftest(args: argparse.Namespace, base: BasePolicy, agent: Agent, buffer: ReplayBuffer) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/selftest", root=args.selftest_dataset, video_backend="pyav")
    for e in range(2):
        a = ds.meta.episodes[e]["dataset_from_index"]
        frames = {k: [] for k in ("image", "state", "base", "residual")}
        base.reset()
        times = []
        for i in range(a, a + 120):
            item = ds[i]
            state = item["observation.state"].numpy()
            image = (item[CAMERA_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            t = time.perf_counter()
            base_action = base.act(state, image)
            small = downscale(image)
            residual = agent.act(small, state, base_action, explore=True)
            times.append(time.perf_counter() - t)
            for k, v in zip(frames, (small, state, base_action, residual)):
                frames[k].append(v)
        episode = {k: np.stack(v) for k, v in frames.items()}
        episode["reward"] = np.zeros(len(episode["state"]), dtype=np.float32)
        episode["reward"][-1] = float(e == 0)
        diff = np.abs(episode["base"] - np.stack([ds[i]["action"].numpy() for i in range(a, a + 120)])).mean()
        print(
            f"エピソード {e}: 1ステップ平均 {np.mean(times) * 1000:.1f} ms、最大 {np.max(times) * 1000:.0f} ms、"
            f"ACT の出力とデモの行動の差 {diff:.2f} 度"
        )
        buffer.add_episode(episode)
        save_episode(args.out, e, episode)
    args.max_updates = 50
    train(agent, buffer, 100, args)
    last = agent.actor.net[-1]
    untouched = bool((last.weight == 0).all() and (last.bias == 0).all())
    print(f"成功が足りない間は残差方策を更新しない: {'OK' if untouched else 'NG'}")
    args.min_successes = 1
    args.max_updates = 200
    train(agent, buffer, 400, args)
    agent.save(args.out / "agent.pt")
    reloaded = Agent(args)
    reloaded.load(args.out / "agent.pt")
    print(f"保存と読み込み: OK（更新回数 {reloaded.updates}）")


def main() -> None:
    args = parse_args()
    (args.out / "episodes").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    base = BasePolicy(args.base)
    agent = Agent(args)
    buffer = ReplayBuffer(args.gamma, args.n_step)
    log_path = args.out / "log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    if (args.out / "agent.pt").exists():
        agent.load(args.out / "agent.pt")
        for ep in load_episodes(args.out):
            buffer.add_episode(ep)
        print(f"{args.out} から続きを読み込みました（{len(buffer.episodes)} エピソード、更新 {agent.updates} 回）。")

    if args.selftest:
        run_selftest(args, base, agent, buffer)
    else:
        run_robot(args, base, agent, buffer, log)


if __name__ == "__main__":
    main()
