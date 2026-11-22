#!/usr/bin/env bash
# 記録したデータセットを tachyon に送り、模倣学習の学習を tachyon の tmux の中で始める。
# 使い方: ./scripts/train_lift.sh [ステップ数]
#   POLICY=diffusion で Diffusion Policy、POLICY=smolvla で SmolVLA、既定は ACT。
# 学習は Ram と同じ lerobot 0.6.1 の環境（tachyon の ~/so101/real/.venv）で行い、
# できたチェックポイントを Ram でそのまま読めるようにする。
set -euo pipefail

cd "$(dirname "$0")/.."

STEPS="${1:-30000}"
POLICY="${POLICY:-act}"
DATASET="${DATASET:-so101_lift_can}"
RUN="${RUN:-${POLICY}_lift_can_$(date +%m%d_%H%M)}"
REMOTE=tachyon
REMOTE_DIR=so101/real

case "${POLICY}" in
  act)
    POLICY_ARGS="--policy.type=act --policy.use_amp=true"
    SAVE_FREQ=5000
    NUM_WORKERS=4
    ;;
  diffusion)
    # DDIM で学習しておくと、Ram の CPU での推論時にノイズ除去の回数を減らせる。
    # 1サンプルで2フレームを読むのでデータ読み込みが重く、ワーカーを増やすと 2.8 → 3.1 step/s。
    POLICY_ARGS="--policy.type=diffusion --policy.noise_scheduler_type=DDIM --policy.use_amp=true"
    SAVE_FREQ=10000
    NUM_WORKERS=12
    ;;
  smolvla)
    # 事前学習済みの smolvla_base から微調整する。手首カメラは camera1 として渡す（下の rename_map）。
    POLICY_ARGS="--policy.path=lerobot/smolvla_base"
    SAVE_FREQ=5000
    NUM_WORKERS=8
    ;;
  *)
    echo "POLICY は act / diffusion / smolvla のどれかにしてください。" >&2
    exit 1
    ;;
esac

if [[ ! -e "outputs/datasets/${DATASET}/meta/info.json" ]]; then
  echo "outputs/datasets/${DATASET} がありません。先に ./scripts/record_lift.sh で記録してください。" >&2
  exit 1
fi

echo "データセットを ${REMOTE}:${REMOTE_DIR}/datasets/${DATASET} へ送ります。"
rsync -a --partial --info=progress2 \
  "outputs/datasets/${DATASET}/" "${REMOTE}:${REMOTE_DIR}/datasets/${DATASET}/"

# 照明の変化に強くするため、明るさ・コントラスト・彩度・色相をランダムに変える。
# 学習は GPU 律速なので fp16 の混合精度 (AMP) にする。3090 で約 1.4 倍速く、loss は変わらなかった。
# ssh は引数を空白でつなぎ直すので、空白を含む POLICY_ARGS は最後に置いて残り全部として受け取る。
ssh "${REMOTE}" bash -s -- "${REMOTE_DIR}" "${DATASET}" "${RUN}" "${STEPS}" \
  "${SAVE_FREQ}" "${NUM_WORKERS}" ${POLICY_ARGS} <<'EOF'
set -euo pipefail
remote_dir=$1 dataset=$2 run=$3 steps=$4 save_freq=$5 num_workers=$6
shift 6
policy_args="$*"
# smolvla_base は camera1〜3 という名前の画像を受け取るので、手首カメラを camera1 として渡す。
rename=""
case "${policy_args}" in
  *smolvla*) rename="--rename_map='{\"observation.images.wrist\":\"observation.images.camera1\"}'" ;;
esac
cd ~/"${remote_dir}"
if [[ -e "outputs/${run}" ]]; then
  echo "outputs/${run} はすでにあります。RUN=別の名前 を付けてください。" >&2
  exit 1
fi
mkdir -p logs
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
tmux new-session -d -s "${run}" \
  ".venv/bin/lerobot-train \
    ${policy_args} ${rename} \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/${dataset} \
    --dataset.root=datasets/${dataset} \
    --dataset.image_transforms.enable=true \
    --batch_size=32 \
    --num_workers=${num_workers} \
    --steps=${steps} \
    --save_freq=${save_freq} \
    --output_dir=outputs/${run} \
    --job_name=${run} \
    --wandb.enable=false \
    2>&1 | tee logs/${run}.log"
echo "tachyon の tmux セッション ${run} で学習を始めました。"
EOF

cat <<EOF
進み具合を見る:  ssh ${REMOTE} tail -f ${REMOTE_DIR}/logs/${RUN}.log
終わったら取得:  ./scripts/fetch_policy.sh ${RUN}
EOF
