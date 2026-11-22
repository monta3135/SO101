#!/usr/bin/env bash
# tachyon で学習した ACT の最新チェックポイントを Ram の outputs/policies/ に取ってくる。
# 使い方: ./scripts/fetch_policy.sh <RUN 名> [チェックポイント名 (既定 last)]
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "使い方: $0 <RUN 名> [チェックポイント名]" >&2
  exit 1
fi
RUN=$1
CHECKPOINT="${2:-last}"

mkdir -p "outputs/policies/${RUN}"
# last はシンボリックリンクなので、-L で中身を取ってくる。
rsync -aL --partial --info=progress2 \
  "tachyon:so101/real/outputs/${RUN}/checkpoints/${CHECKPOINT}/pretrained_model/" \
  "outputs/policies/${RUN}/${CHECKPOINT}/"
echo "outputs/policies/${RUN}/${CHECKPOINT} に保存しました。"
