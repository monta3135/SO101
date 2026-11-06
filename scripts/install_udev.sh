#!/usr/bin/env bash
# USB シリアルデバイスへの読み書き権限を設定する。
set -euo pipefail

RULE_FILE=/etc/udev/rules.d/99-so101.rules

if [[ ${EUID} -ne 0 ]]; then
  echo "sudo で実行してください: sudo $0" >&2
  exit 1
fi

cat > "${RULE_FILE}" <<'EOF'
# SO-101: Waveshare Bus Servo Adapter などの USB シリアル
SUBSYSTEM=="tty", SUBSYSTEMS=="usb", MODE="0666"
EOF

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty

echo "作成しました: ${RULE_FILE}"
echo "USB ケーブルを挿し直すと権限が反映されます。"
