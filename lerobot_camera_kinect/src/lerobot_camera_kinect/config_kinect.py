from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("kinect")
@dataclass
class KinectCameraConfig(CameraConfig):
    """Xbox 360 Kinect (v1) の設定。

    RGB と深度はどちらも 640x480 で、深度は RGB に位置合わせした mm 値 (FREENECT_DEPTH_REGISTERED)。
    測れない画素は 0。Kinect は約 0.5 m より近くを測れない。

    例: --robot.cameras="{ top: {type: kinect, use_depth: true} }"
    """

    index: int = 0
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    warmup_s: float = 2.0  # 自動露出が落ち着くまで待つ時間

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        if not self.use_rgb and not self.use_depth:
            raise ValueError("use_rgb と use_depth の少なくとも一方を true にしてください。")
        # 位置合わせした深度は 640x480 でしか出ないので、解像度は固定する
        if self.width not in (None, 640) or self.height not in (None, 480):
            raise ValueError(f"Kinect は 640x480 のみ対応です (指定: {self.width}x{self.height})。")
        if self.fps not in (None, 30):
            raise ValueError(f"Kinect は 30 fps のみ対応です (指定: {self.fps})。")
        self.width, self.height, self.fps = 640, 480, 30
