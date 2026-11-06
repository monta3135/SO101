#!/usr/bin/env bash
# アームごとに固定のポート名 (/dev/so101-leader, /dev/so101-follower) を割り当てる。
set -euo pipefail

usage() {
  cat <<'EOF'
使い方:
  sudo ./scripts/pin_ports.sh leader /dev/ttyACM1 follower /dev/ttyACM0

引数は「役割 ポート」の順に2組渡します。
役割は leader または follower です。
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 1
fi

ROLE_A=$1
PORT_A=$2
ROLE_B=$3
PORT_B=$4

if [[ ${EUID} -ne 0 ]]; then
  echo "sudo で実行してください: sudo $0 ..." >&2
  exit 1
fi

for role in "${ROLE_A}" "${ROLE_B}"; do
  if [[ "${role}" != "leader" && "${role}" != "follower" ]]; then
    echo "役割は leader か follower を指定してください: ${role}" >&2
    exit 1
  fi
done

if [[ "${ROLE_A}" == "${ROLE_B}" ]]; then
  echo "同じ役割が2回指定されています。" >&2
  exit 1
fi

for port in "${PORT_A}" "${PORT_B}"; do
  if [[ ! -e "${port}" ]]; then
    echo "ポートが見つかりません: ${port}" >&2
    exit 1
  fi
done

if [[ "${PORT_A}" == "${PORT_B}" ]]; then
  echo "同じポートが2回指定されています。" >&2
  exit 1
fi

get_property() {
  local port=$1
  local key=$2
  udevadm info -q property -n "${port}" | sed -n "s/^${key}=//p"
}

get_usb_interface() {
  local port=$1
  udevadm info -q path -n "${port}" \
    | tr '/' '\n' \
    | grep -E '^[0-9]+-[0-9.]+:1\.[0-9]+$' \
    | tail -n 1
}

SERIAL_A=$(get_property "${PORT_A}" ID_SERIAL_SHORT)
SERIAL_B=$(get_property "${PORT_B}" ID_SERIAL_SHORT)

if [[ -n "${SERIAL_A}" && -n "${SERIAL_B}" && "${SERIAL_A}" != "${SERIAL_B}" ]]; then
  KEY_A="ATTRS{serial}==\"${SERIAL_A}\""
  KEY_B="ATTRS{serial}==\"${SERIAL_B}\""
  echo "シリアル番号で固定します。"
else
  IFACE_A=$(get_usb_interface "${PORT_A}")
  IFACE_B=$(get_usb_interface "${PORT_B}")

  if [[ -z "${IFACE_A}" || -z "${IFACE_B}" || "${IFACE_A}" == "${IFACE_B}" ]]; then
    echo "USB の接続位置を特定できませんでした。" >&2
    echo "別の USB ポートに挿し直してから再実行してください。" >&2
    exit 1
  fi

  KEY_A="KERNELS==\"${IFACE_A}\""
  KEY_B="KERNELS==\"${IFACE_B}\""
  echo "USB の接続位置で固定します。挿すポートを変えると無効になります。"
fi

RULE_FILE=/etc/udev/rules.d/99-so101-ports.rules

cat > "${RULE_FILE}" <<EOF
# SO-101: アームごとの固定ポート名
SUBSYSTEM=="tty", ${KEY_A}, SYMLINK+="so101-${ROLE_A}"
SUBSYSTEM=="tty", ${KEY_B}, SYMLINK+="so101-${ROLE_B}"
EOF

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty
sleep 1

echo "作成しました: ${RULE_FILE}"
echo "${PORT_A} -> /dev/so101-${ROLE_A}"
echo "${PORT_B} -> /dev/so101-${ROLE_B}"

for link in /dev/so101-*; do
  if [[ -L "${link}" ]]; then
    echo "  ${link} -> $(readlink -f "${link}")"
  fi
done
