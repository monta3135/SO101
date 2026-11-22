#!/usr/bin/env bash
# リーダーでテレオペしながら、コーヒー缶を持ち上げるデモを ACT 用に記録する。
# 使い方: ./scripts/record_lift.sh [エピソード数]   続きから足すときは RESUME=1 を付ける。
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_EPISODES="${1:-50}"
DATASET="${DATASET:-so101_lift_can}"
CAMERA=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0

for device in /dev/so101-leader /dev/so101-follower "${CAMERA}"; do
  if [[ ! -e "${device}" ]]; then
    echo "${device} がありません。" >&2
    exit 1
  fi
done

# 色や明るさがエピソードごとに変わらないよう、カメラの自動調整を切る。
# 初回に固定した値を保存し、以降の記録とポリシー実行では同じ値を書き戻す。
CAMERA_LOCK=calibration/wrist_camera_lock.json
if [[ -e "${CAMERA_LOCK}" ]]; then
  uv run python scripts/lock_camera.py --camera "${CAMERA}" --load "${CAMERA_LOCK}"
else
  uv run python scripts/lock_camera.py --camera "${CAMERA}" --save "${CAMERA_LOCK}"
fi

exec uv run python scripts/record.py \
  --robot.type=so101_follower \
  --robot.port=/dev/so101-follower \
  --robot.id=follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: ${CAMERA}, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101-leader \
  --teleop.id=leader \
  --dataset.repo_id="local/${DATASET}" \
  --dataset.root="outputs/datasets/${DATASET}" \
  --dataset.single_task="Grasp the coffee can and lift it" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false \
  --resume="${RESUME:-false}"
