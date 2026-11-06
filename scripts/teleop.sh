#!/usr/bin/env bash
# 固定ポート名を使ってテレオペレーションを起動する。
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv が見つかりません。uv をインストールしてください。" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo ".venv が見つかりません。docs/software.md の手順を確認してください。" >&2
  exit 1
fi

for device in /dev/so101-leader /dev/so101-follower; do
  if [[ ! -e "${device}" ]]; then
    echo "${device} がありません。" >&2
    echo "先に sudo ./scripts/pin_ports.sh を実行してください。" >&2
    exit 1
  fi
done

exec uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/so101-follower \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101-leader \
  --teleop.id=leader
