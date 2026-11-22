#!/usr/bin/env bash
# 学習した ACT のチェックポイントで、フォロワーを手首カメラだけで動かす。リーダーは使わない。
# 使い方: ./scripts/run_policy.sh <チェックポイントのディレクトリ> [秒数 (既定 45)] [lerobot-rollout の追加引数...]
#   例: ./scripts/run_policy.sh outputs/policies/act_lift_can_1119_1007/005000
#   Diffusion Policy はノイズ除去の回数を減らすと CPU でも速い:
#       ./scripts/run_policy.sh outputs/policies/diffusion_lift_can_1119_1203/last 45 --policy.num_inference_steps=10
# 止めるときは Ctrl+C。終了時は起動時の姿勢へゆっくり戻る。
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "使い方: $0 <チェックポイントのディレクトリ> [秒数]" >&2
  exit 1
fi
POLICY=$1
DURATION="${2:-45}"
CAMERA=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0
# 1ステップ (1/30 秒) で各関節が動いてよい最大角度 [度]。暴れたときの保険。
MAX_STEP_DEG="${MAX_STEP_DEG:-15}"
# フォロワーをつないでいる基板のポート。フォロワーの基板が壊れている間は PORT=/dev/so101-leader で起動する。
PORT="${PORT:-/dev/so101-follower}"

for path in "${PORT}" "${CAMERA}" "${POLICY}/model.safetensors"; do
  if [[ ! -e "${path}" ]]; then
    echo "${path} がありません。" >&2
    exit 1
  fi
done

# 記録したときと同じ露出・ホワイトバランスにそろえる。
uv run python scripts/lock_camera.py --camera "${CAMERA}" --load calibration/wrist_camera_lock.json

exec uv run lerobot-rollout \
  --robot.type=so101_follower \
  --robot.port="${PORT}" \
  --robot.id=follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: ${CAMERA}, width: 640, height: 480, fps: 30}}" \
  --robot.max_relative_target="${MAX_STEP_DEG}" \
  --policy.path="${POLICY}" \
  --policy.device=cpu \
  --policy.use_amp=false \
  --strategy.type=base \
  --inference.type=sync \
  --task="Grasp the coffee can and lift it" \
  --duration="${DURATION}" \
  --display_data=true \
  --play_sounds=false \
  "${@:3}"
