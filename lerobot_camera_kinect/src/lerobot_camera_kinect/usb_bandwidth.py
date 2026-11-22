"""Kinect が予約している USB の帯域を返す・取り戻す。

初代 Kinect のカメラとマイクは、帯域を使う転送口を alt 0 に置いているので、Linux は
挿した時点でその帯域を予約し、撮影していなくても手放さない。同じ USB コントローラに
つないだ手首カメラが "Not enough bandwidth" で動かなくなる。
release() で Kinect を「未設定」(configuration 0) にすると予約が消え、抜いたのと同じになる。
acquire() で configuration 1 に戻すと、また Kinect として使える。
libfreenect の udev ルール (MODE 0666) があれば sudo は要らない。
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

VENDOR = 0x045E
PRODUCTS = {0x02AE: "camera", 0x02AD: "audio"}  # 帯域を予約する2つ (モーターは予約しない)

_lib = ctypes.CDLL("libusb-1.0.so.0")
_lib.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.libusb_exit.argtypes = [ctypes.c_void_p]
_lib.libusb_get_device_list.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))]
_lib.libusb_get_device_list.restype = ctypes.c_ssize_t
_lib.libusb_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
_lib.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.libusb_close.argtypes = [ctypes.c_void_p]
_lib.libusb_get_configuration.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.libusb_set_configuration.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.libusb_kernel_driver_active.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.libusb_detach_kernel_driver.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.libusb_reset_device.argtypes = [ctypes.c_void_p]
_lib.libusb_error_name.argtypes = [ctypes.c_int]
_lib.libusb_error_name.restype = ctypes.c_char_p


class _DeviceDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8), ("bcdUSB", ctypes.c_uint16),
        ("bDeviceClass", ctypes.c_uint8), ("bDeviceSubClass", ctypes.c_uint8),
        ("bDeviceProtocol", ctypes.c_uint8), ("bMaxPacketSize0", ctypes.c_uint8),
        ("idVendor", ctypes.c_uint16), ("idProduct", ctypes.c_uint16), ("bcdDevice", ctypes.c_uint16),
        ("iManufacturer", ctypes.c_uint8), ("iProduct", ctypes.c_uint8), ("iSerialNumber", ctypes.c_uint8),
        ("bNumConfigurations", ctypes.c_uint8),
    ]


_lib.libusb_get_device_descriptor.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DeviceDescriptor)]


def _check(rc: int, what: str) -> None:
    if rc < 0:
        raise OSError(f"{what}: {_lib.libusb_error_name(rc).decode()}")


def _set_config(value: int) -> dict[str, tuple[int, int]]:
    """Kinect のカメラとマイクの configuration を value にする。{名前: (前, 後)} を返す。"""
    ctx = ctypes.c_void_p()
    _check(_lib.libusb_init(ctypes.byref(ctx)), "libusb_init")
    result = {}
    devs = ctypes.POINTER(ctypes.c_void_p)()
    try:
        n = _lib.libusb_get_device_list(ctx, ctypes.byref(devs))
        for i in range(n):
            desc = _DeviceDescriptor()
            _lib.libusb_get_device_descriptor(devs[i], ctypes.byref(desc))
            if desc.idVendor != VENDOR or desc.idProduct not in PRODUCTS:
                continue
            name = PRODUCTS[desc.idProduct]
            handle = ctypes.c_void_p()
            _check(_lib.libusb_open(devs[i], ctypes.byref(handle)), f"open {name}")
            try:
                cur = ctypes.c_int()
                _check(_lib.libusb_get_configuration(handle, ctypes.byref(cur)), f"get_configuration {name}")
                before = cur.value
                if before != value:
                    if before != 0 and _lib.libusb_kernel_driver_active(handle, 0) == 1:
                        _check(_lib.libusb_detach_kernel_driver(handle, 0), f"detach {name}")
                    # 0 (未設定) は libusb では -1 で指定する
                    _check(_lib.libusb_set_configuration(handle, value if value else -1), f"set_configuration {name}")
                    if value:
                        # 設定し直しただけだとカメラの中身が初期化されず、最初の映像が来ないので、つなぎ直させる
                        _lib.libusb_reset_device(handle)
                    time.sleep(0.1)
                result[name] = (before, value)
            finally:
                _lib.libusb_close(handle)
        _lib.libusb_free_device_list(devs, 1)
    finally:
        _lib.libusb_exit(ctx)
    return result


def _retry(value: int, tries: int = 10) -> dict[str, tuple[int, int]]:
    # libfreenect を止めた直後は、まだ転送の後始末中で LIBUSB_ERROR_OTHER になることがある
    for k in range(tries):
        try:
            return _set_config(value)
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(0.3)
    return {}


def release() -> dict[str, tuple[int, int]]:
    """Kinect の帯域の予約を消す。Kinect が挿さっていなければ何もしない。"""
    return _retry(0)


def _configured() -> list[str]:
    """設定済み (帯域を予約している) の Kinect のカメラ・マイクを sysfs から探す。"""
    found = []
    for dev in Path("/sys/bus/usb/devices").iterdir():
        try:
            vid = int((dev / "idVendor").read_text(), 16)
            pid = int((dev / "idProduct").read_text(), 16)
        except (OSError, ValueError):
            continue
        if vid == VENDOR and pid in PRODUCTS:
            config = (dev / "bConfigurationValue").read_text().strip()
            if config not in ("", "0"):
                found.append(PRODUCTS[pid])
    return found


def ensure_released(settle: float = 2.0, timeout: float = 20.0) -> None:
    """Kinect が未設定のまま settle 秒続くまで待つ。リセットのあと Kinect が遅れて USB に
    つなぎ直されると、設定が 1 に戻って帯域がまた予約されるので、そのたびに返し直す。"""
    deadline = time.monotonic() + timeout
    quiet_since = None
    while time.monotonic() < deadline:
        if _configured():
            release()
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= settle:
            return
        time.sleep(0.2)
    raise TimeoutError(f"Kinect の帯域を返せませんでした (設定済みのまま: {_configured()})")


def acquire() -> dict[str, tuple[int, int]]:
    """Kinect を使える状態 (configuration 1) に戻す。"""
    return _retry(1)


if __name__ == "__main__":
    import sys

    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "release":
        print(release())
    elif action == "acquire":
        print(acquire())
    else:
        print("使い方: python -m lerobot_camera_kinect.usb_bandwidth release|acquire")
