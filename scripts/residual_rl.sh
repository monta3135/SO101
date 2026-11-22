#!/usr/bin/env bash
# ACT に残差方策を足して、実機で強化学習する（scripts/residual_rl.py の起動用）。
# 使い方: ./scripts/residual_rl.sh <ACT のチェックポイント> [出力先] [residual_rl.py の追加引数...]
#   例: ./scripts/residual_rl.sh outputs/policies/act_lift_can_1119_1944/020000
#   出力先の既定は outputs/residual/<RUN 名>_<チェックポイント名>。ACT を変えるとデータも別になる。
# 操作: n で開始、g で掴めた、s で置けた、f で失敗、q で終了。同じ出力先で起動し直すと続きから学習する。
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "使い方: $0 <ACT のチェックポイント> [出力先] [追加引数...]" >&2
  exit 1
fi
BASE=$1
# 残差は ACT の出力を前提に学ぶので、ACT ごとに別のディレクトリへ保存する。
OUT="${2:-outputs/residual/$(basename "$(dirname "${BASE}")")_$(basename "${BASE}")}"
CAMERA=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0
# フォロワーをつないでいる基板のポート。フォロワーの基板が壊れて、フォロワーをリーダーの基板
# （12V 電源に差し替え）につないでいる間は PORT=/dev/so101-leader で起動する。
PORT="${PORT:-/dev/so101-follower}"

for path in "${PORT}" "${CAMERA}" "${BASE}/model.safetensors"; do
  if [[ ! -e "${path}" ]]; then
    echo "${path} がありません。" >&2
    exit 1
  fi
done

# 記録したときと同じ露出・ホワイトバランスにそろえる。
uv run python scripts/lock_camera.py --camera "${CAMERA}" --load calibration/wrist_camera_lock.json

exec uv run python scripts/residual_rl.py --base "${BASE}" --out "${OUT}" --camera "${CAMERA}" --port "${PORT}" "${@:3}"
