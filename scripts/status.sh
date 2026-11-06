#!/usr/bin/env bash
# SO-101 の接続状態をまとめて表示する。
set -u

cd "$(dirname "$0")/.."

echo "== LeRobot =="
if [[ -x .venv/bin/python ]] && command -v uv >/dev/null 2>&1; then
  uv run python -c "import lerobot; print('lerobot', lerobot.__version__)" 2>&1 || true
else
  echo ".venv または uv がありません"
fi

echo
echo "== USB シリアルのポート =="
shopt -s nullglob
ports=(/dev/ttyACM* /dev/ttyUSB* /dev/ttyCH343USB*)
if (( ${#ports[@]} == 0 )); then
  echo "なし"
else
  for port in "${ports[@]}"; do
    echo "${port}"
  done
fi

echo
echo "== /dev/serial/by-id =="
if [[ -d /dev/serial/by-id ]]; then
  ls -l /dev/serial/by-id
else
  echo "なし"
fi

echo
echo "== 固定ポート名 =="
links=(/dev/so101-*)
if (( ${#links[@]} == 0 )); then
  echo "なし"
else
  for link in "${links[@]}"; do
    if [[ -L "${link}" ]]; then
      echo "${link} -> $(readlink -f "${link}")"
    fi
  done
fi

echo
echo "== udev ルール =="
rules=(/etc/udev/rules.d/99-so101*.rules)
if (( ${#rules[@]} == 0 )); then
  echo "なし"
else
  ls -l "${rules[@]}"
fi

echo
echo "== 所属グループ =="
id -nG
