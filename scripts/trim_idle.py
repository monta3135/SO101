#!/usr/bin/env python3
"""記録したデータセットの各エピソードから、冒頭の静止部分を切り詰めた新しいデータセットを作る。

開始姿勢で止まっている間は同じ画像に「止まる」動作が対応するので、ACT が動き出さなくなる。
各エピソードで、どれかの関節が開始時から --threshold 度以上動いた最初のフレームを探し、
その --keep-before 秒前より前を捨てる。--exclude のエピソードは丸ごと除く。元のデータセットは変更しない。

使い方:
  python scripts/trim_idle.py --src outputs/datasets/so101_lift_can \
      --dst outputs/datasets/so101_lift_can_trim --exclude 16 29 31 32 51
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMERA_KEY = "observation.images.wrist"
DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="エピソード冒頭の静止部分を切り詰める")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    parser.add_argument("--threshold", type=float, default=2.0, help="動き出しとみなす関節角の変化 [度]")
    parser.add_argument("--keep-before", type=float, default=0.5, help="動き出しの何秒前から残すか")
    return parser.parse_args()


def first_move(state: np.ndarray, threshold: float) -> int:
    moved = np.abs(state - state[0]).max(axis=1) > threshold
    return int(np.argmax(moved)) if moved.any() else len(state)


def main() -> None:
    args = parse_args()
    if args.dst.exists():
        raise SystemExit(f"{args.dst} はすでにあります。")

    info = json.loads((args.src / "meta/info.json").read_text())
    fps = info["fps"]
    data = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(args.src / "data/*/*.parquet"))))
    episodes = pd.concat(
        pd.read_parquet(f) for f in sorted(glob.glob(str(args.src / "meta/episodes/*/*.parquet")))
    )
    tasks = pd.read_parquet(args.src / "meta/tasks.parquet")
    task_by_index = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}

    features = {k: v for k, v in info["features"].items() if k not in DEFAULT_FEATURES}
    dst = LeRobotDataset.create(
        repo_id=f"local/{args.dst.name}",
        fps=fps,
        features=features,
        root=args.dst,
        robot_type=info["robot_type"],
        use_videos=True,
        image_writer_threads=4,
    )

    # 動画はエピソードを連結したファイルなので、ファイルごとに先頭から順に読み、
    # 各フレームを時刻からエピソードとフレーム番号に割り当てる。
    prefix = f"videos/{CAMERA_KEY}"
    kept = episodes[~episodes["episode_index"].isin(args.exclude)]
    total_before = total_after = 0
    for (chunk, file), eps in kept.groupby([f"{prefix}/chunk_index", f"{prefix}/file_index"]):
        eps = eps.sort_values(f"{prefix}/from_timestamp")
        path = args.src / info["video_path"].format(video_key=CAMERA_KEY, chunk_index=chunk, file_index=file)
        plan = []
        for _, ep in eps.iterrows():
            frames = data[data["episode_index"] == ep["episode_index"]].sort_values("frame_index")
            state = np.stack(frames["observation.state"].values)
            start = max(0, first_move(state, args.threshold) - round(args.keep_before * fps))
            plan.append((ep, frames, start))

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            decoded = container.decode(stream)
            for ep, frames, start in plan:
                from_ts = ep[f"{prefix}/from_timestamp"]
                length = len(frames)
                actions = np.stack(frames["action"].values)
                states = np.stack(frames["observation.state"].values)
                task = task_by_index[int(frames["task_index"].iloc[0])]
                n = 0
                for frame in decoded:
                    i = round((float(frame.pts * stream.time_base) - from_ts) * fps)
                    if i < 0:
                        continue
                    if i >= start:
                        dst.add_frame(
                            {
                                "action": actions[i],
                                "observation.state": states[i],
                                CAMERA_KEY: frame.to_ndarray(format="rgb24"),
                                "task": task,
                            }
                        )
                    n += 1
                    if i == length - 1:
                        break
                if n != length:
                    raise SystemExit(f"エピソード {ep['episode_index']}: 動画 {n} フレーム、データ {length} フレームで合いません。")
                dst.save_episode()
                total_before += length
                total_after += length - start
                print(f"エピソード {ep['episode_index']:2d}: {length} → {length - start} フレーム（冒頭 {start / fps:.1f} 秒を削除）")

    dst.finalize()
    print(f"{len(kept)} エピソード、{total_before} → {total_after} フレーム。{args.dst} に保存しました。")


if __name__ == "__main__":
    main()
