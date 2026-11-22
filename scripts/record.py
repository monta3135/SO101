#!/usr/bin/env python3
"""lerobot-record を、lerobot 0.6.1 の2つの問題を避けて起動する。引数は lerobot-record と同じ。

1. → (n) が続けて2回届くと、1回目で片付け時間が終わり、2回目で次のエピソードが
   0フレームのまま終わる。→ 直前の → から 1 秒以内の → は無視する。
2. 0フレームのエピソードを save_episode すると ValueError で落ち、記録が止まる。
   → 0フレームなら保存せずに捨てる。
"""

from __future__ import annotations

import time

from lerobot.datasets import lerobot_dataset
from lerobot.scripts import lerobot_record
from lerobot.utils import keyboard_input

DEBOUNCE_S = 1.0

_apply_recording_control = keyboard_input.apply_recording_control
_last_right = 0.0


def apply_recording_control(control: str, events: dict) -> None:
    global _last_right
    if control == "right":
        now = time.monotonic()
        if now - _last_right < DEBOUNCE_S:
            print("→ が続けて押されたので無視しました。")
            return
        _last_right = now
    _apply_recording_control(control, events)


_save_episode = lerobot_dataset.LeRobotDataset.save_episode


def save_episode(self, episode_data=None, parallel_encoding=True):
    if episode_data is None and not self.has_pending_frames():
        print("0フレームのエピソードなので保存しません。")
        return
    _save_episode(self, episode_data, parallel_encoding)


keyboard_input.apply_recording_control = apply_recording_control
lerobot_dataset.LeRobotDataset.save_episode = save_episode

if __name__ == "__main__":
    lerobot_record.main()
