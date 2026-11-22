# lerobot_camera_kinect

Xbox 360 Kinect (v1) を LeRobot のカメラ (`type: kinect`) として使うプラグイン。
`use_depth: true` にすると、RGB に位置合わせした深度 (mm) が `<名前>_depth` として観測に加わり、
`lerobot-record` が RealSense と同じく 12 ビットの深度動画 (HEVC ロスレス, gray12le) で記録する。

## 入れ方

```bash
sudo apt install libfreenect-dev   # udev ルールも入り、sudo なしで USB に触れる
uv pip install --python .venv/bin/python -e lerobot_camera_kinect
```

## 使い方

```bash
.venv/bin/python scripts/record.py ... \
  --robot.cameras="{ top: {type: kinect, use_depth: true} }"
```

- 解像度は 640x480, 30 fps 固定 (位置合わせした深度がこの解像度でしか出ないため)。
- 深度は約 0.5 m より近い所と、光る面・黒い面が 0 (測定なし) になる。
- libfreenect の sync API を使うので、1つのプロセスで開ける Kinect は1台だけ。
- 起動時に `freenect_fetch_reg_info: send_cmd read 4 bytes (expected 118)` が出たら、
  位置合わせ用データを読めていない。開き直すと直る。
- カーネルの `gspca_kinect` ドライバが読み込まれていても動く。

## 確認したこと (2026-11-22, Ram)

- RGB・深度とも 30 fps で取れる。
- 90 フレームをデータセットに保存して読み戻し、0〜4 m の画素で元の mm 値との差は中央値 0 mm, 95% 点 1 mm。
- 3 秒で RGB 動画 2.2 MB, 深度動画 8.0 MB。
