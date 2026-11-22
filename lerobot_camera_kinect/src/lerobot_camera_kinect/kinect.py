"""Xbox 360 Kinect (v1) を LeRobot のカメラとして使う。

libfreenect の sync API (libfreenect_sync.so, apt の libfreenect0.5) を ctypes で呼ぶ。
裏のスレッドで RGB と深度を読み続け、最新フレームを保持する (RealSenseCamera と同じ形)。
深度は RGB に位置合わせした mm 値で、LeRobot には (H, W, 1) の uint16 として渡すので、
lerobot-record がそのまま深度動画として記録する。
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode

from .config_kinect import KinectCameraConfig

RESOLUTION_MEDIUM = 1
VIDEO_RGB = 0
DEPTH_REGISTERED = 4
WIDTH, HEIGHT = 640, 480

_sync = None
_owner: KinectCamera | None = None  # sync API はプロセスで1つの状態を持つので、同時に開けるのは1台だけ


def _lib() -> ctypes.CDLL:
    global _sync
    if _sync is None:
        try:
            _sync = ctypes.CDLL("libfreenect_sync.so.0.5")
        except OSError as e:
            raise OSError("libfreenect_sync が見つかりません。sudo apt install libfreenect-dev で入れてください。") from e
        args = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        for name in ("freenect_sync_get_video_with_res", "freenect_sync_get_depth_with_res"):
            fn = getattr(_sync, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        _sync.freenect_sync_stop.argtypes, _sync.freenect_sync_stop.restype = [], None
    return _sync


class KinectCamera(Camera):
    def __init__(self, config: KinectCameraConfig):
        super().__init__(config)
        self.config = config
        self.index = config.index
        self.color_mode = config.color_mode
        self.use_rgb = config.use_rgb
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.frame_lock = threading.Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_color_event = threading.Event()
        self.new_depth_event = threading.Event()
        self.error: str | None = None

    def __str__(self) -> str:
        return f"KinectCamera({self.index})"

    @property
    def is_connected(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        lib = ctypes.CDLL("libfreenect.so.0.5")
        ctx = ctypes.c_void_p()
        if lib.freenect_init(ctypes.byref(ctx), None) < 0:
            return []
        n = lib.freenect_num_devices(ctx)
        lib.freenect_shutdown(ctx)
        return [{"type": "Kinect", "id": i, "name": f"Kinect v1 #{i}"} for i in range(n)]

    def connect(self, warmup: bool = True) -> None:
        global _owner
        if self.is_connected:
            raise RuntimeError(f"{self} はすでに接続されています。")
        if _owner is not None and _owner is not self and _owner.is_connected:
            raise RuntimeError(f"{_owner} がすでに Kinect を使っています (sync API では1台のみ)。")
        _lib()
        _owner = self
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

        # 最初のフレームが届くまで待つ (USB の初期化で数秒かかる)
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            with self.frame_lock:
                ready = (not self.use_rgb or self.latest_color_frame is not None) and (
                    not self.use_depth or self.latest_depth_frame is not None
                )
            if ready or self.error:
                break
            time.sleep(0.05)
        if self.error or not ready:
            self.disconnect()
            raise ConnectionError(f"{self} からフレームが届きません: {self.error or 'timeout'}")
        if warmup:
            time.sleep(self.warmup_s)

    def _read_loop(self) -> None:
        lib = _lib()
        ptr, ts = ctypes.c_void_p(), ctypes.c_uint32()
        while not self.stop_event.is_set():
            color = depth = None
            # sync API は新しいフレームが来るまで待ってから返す
            if self.use_depth:
                if lib.freenect_sync_get_depth_with_res(
                    ctypes.byref(ptr), ctypes.byref(ts), self.index, RESOLUTION_MEDIUM, DEPTH_REGISTERED
                ):
                    self.error = "深度フレームを取得できませんでした (Kinect が繋がっていない?)"
                    break
                buf = (ctypes.c_uint16 * (WIDTH * HEIGHT)).from_address(ptr.value)
                depth = np.frombuffer(buf, np.uint16).reshape(HEIGHT, WIDTH, 1).copy()
            if self.use_rgb:
                if lib.freenect_sync_get_video_with_res(
                    ctypes.byref(ptr), ctypes.byref(ts), self.index, RESOLUTION_MEDIUM, VIDEO_RGB
                ):
                    self.error = "RGB フレームを取得できませんでした (Kinect が繋がっていない?)"
                    break
                buf = (ctypes.c_uint8 * (WIDTH * HEIGHT * 3)).from_address(ptr.value)
                color = np.frombuffer(buf, np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
                if self.color_mode == ColorMode.BGR:
                    color = color[:, :, ::-1].copy()
            with self.frame_lock:
                if color is not None:
                    self.latest_color_frame = color
                    self.new_color_event.set()
                if depth is not None:
                    self.latest_depth_frame = depth
                    self.new_depth_event.set()
                self.latest_timestamp = time.perf_counter()

    def _async_read(self, timeout_ms: float, read_depth: bool) -> NDArray[Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} は接続されていません。")
        event = self.new_depth_event if read_depth else self.new_color_event
        if not event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} から {timeout_ms} ms 以内に新しいフレームが来ませんでした。")
        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            event.clear()
        return frame

    def _read_latest(self, max_age_ms: int, read_depth: bool) -> NDArray[Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} は接続されていません。")
        with self.frame_lock:
            frame = self.latest_depth_frame if read_depth else self.latest_color_frame
            timestamp = self.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} はまだフレームを取得していません。")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} の最新フレームが {age_ms:.0f} ms 前のもので古すぎます。")
        return frame

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if not self.use_rgb:
            raise RuntimeError(f"{self} は use_rgb=false です。")
        frame = self._async_read(10000, read_depth=False)
        if color_mode is not None and ColorMode(color_mode) != self.color_mode:
            frame = frame[:, :, ::-1].copy()
        return frame

    def read_depth(self, timeout_ms: int = 10000) -> NDArray[Any]:
        if not self.use_depth:
            raise RuntimeError(f"{self} は use_depth=false です。")
        return self._async_read(timeout_ms, read_depth=True)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        return self._async_read(timeout_ms, read_depth=False)

    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[Any]:
        return self._async_read(timeout_ms, read_depth=True)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self._read_latest(max_age_ms, read_depth=False)

    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[Any]:
        return self._read_latest(max_age_ms, read_depth=True)

    def disconnect(self) -> None:
        global _owner
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.thread = self.stop_event = None
        if _owner is self:
            _lib().freenect_sync_stop()
            _owner = None
        with self.frame_lock:
            self.latest_color_frame = self.latest_depth_frame = self.latest_timestamp = None
            self.new_color_event.clear()
            self.new_depth_event.clear()
