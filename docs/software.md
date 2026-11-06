# ソフトウェアのセットアップ

このディレクトリには uv で作成した Python 3.12 の仮想環境と LeRobot 0.6.1 が入っています。
以下のコマンドは、すべてこのディレクトリで実行します。

```bash
cd /home/monta/workspace/SO101
```

`uv run` はカレントディレクトリの `.venv` を自動で使うため、仮想環境の有効化は不要です。
インストール済みのバージョンは次のコマンドで確認できます。

```bash
uv run python -c "import lerobot; print(lerobot.__version__)"
```

## udev ルールを入れる

Linux では USB シリアルのデバイスに `/dev/ttyACM0` としてアクセスします。
既定では root と `dialout` グループだけが読み書きできるため、現在のユーザーには権限がありません。

次のスクリプトを sudo で実行し、権限を設定します。

```bash
sudo ./scripts/install_udev.sh
```

スクリプトは `/etc/udev/rules.d/99-so101.rules` を作成し、`ttyACM` と `ttyUSB` に読み書き権限を与えます。
実行後は USB を挿し直してください。

## ポートを特定する

各アームの制御基板を USB で PC に接続し、AC アダプタで給電します。
基板のジャンパが B 側になっていることを確認してから、次のコマンドを実行します。

```bash
uv run lerobot-find-port
```

このコマンドは、現在見えているポートを一覧表示したあと、USB ケーブルを抜くよう促します。
フォロワーの USB を抜いて Enter を押すと、消えたポートがフォロワーのポートとして表示されます。
同じ手順をもう一度実行し、リーダーのポートも特定します。

Linux では `/dev/ttyS0` から `/dev/ttyS63` まで的大量に表示されます。
これはシリアルポートの一覧であり、USB のアームとは関係ありません。
`ttyACM`、`ttyUSB`、`ttyCH343USB` など、USB シリアルの名前の増減だけに注目します。

例:

```
Ports before disconnecting: ['/dev/ttyACM0', '/dev/ttyACM1', ...]
Remove the USB cable from your MotorsBus and press Enter when done.
The port of this MotorsBus is /dev/ttyACM0
```

## ポート名を固定する

`/dev/ttyACM0` と `/dev/ttyACM1` は接続順で変わるため、再起動のたびに入れ替わることがあります。
アームごとに固定の名前を割り当てておくと、以降のコマンドを安定して使えます。

ポートを特定したら、次の形式でスクリプトを実行します。

```bash
sudo ./scripts/pin_ports.sh leader /dev/ttyACM1 follower /dev/ttyACM0
```

引数は「役割」「ポート」の順に2組渡します。
スクリプトは USB の接続位置またはシリアル番号を記録し、次のシンボリックリンクを作成します。

- `/dev/so101-leader`
- `/dev/so101-follower`

USB を挿す物理ポートを変えると、リンクが張られなくなります。
その場合はスクリプトを実行し直してください。

## サーボの状態を確認する

ID を設定する前に、バス上にサーボが何個見えているかを確認できます。

```bash
uv run python scripts/check_servos.py --port /dev/ttyACM0
```

ID、電圧、温度が表示されます。
新品のサーボはすべて ID 1 で出荷されるため、数珠つなぎのままでは1個しか表示されません。
これは異常ではありません。

6個すべてが ID 1〜6 で表示される場合は、ID と関節の対応を確認できます。

```bash
uv run python scripts/check_mapping.py --port /dev/ttyACM0
```

画面の指示に従い、表示された関節だけを手で動かします。
すべて `OK` なら、ID 設定をやり直す必要はありません。

## モーターの ID とボーレートを設定する

サーボは1個ずつ基板に直結して設定します。
最初はグリッパーのサーボだけを基板につなぎ、他のサーボは外します。
対象のサーボを他のサーボと数珠つなぎにしたままにしないでください。

### フォロワー

```bash
uv run lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower
```

### リーダー

```bash
uv run lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader
```

コマンドを実行すると、接続するサーボの順番が表示されます。
表示されたジョイントのサーボだけを基板につなぎ、Enter を押します。

フォロワーの順番は次のとおりです。

1. gripper
2. wrist_roll
3. wrist_flex
4. elbow_flex
5. shoulder_lift
6. shoulder_pan

各サーボの設定が終わると、`'gripper' motor id set to 6` のように表示されます。
すべて終わったら、サーボバスを「基板 → ID 1 → … → ID 6」の一列につなぎ直します。

設定はサーボの不揮発メモリに書き込まれるため、次回以降は実行不要です。

## キャリブレーション

キャリブレーションは、リーダーとフォロワーが同じ物理姿勢のときに同じ位置値を返すようにする作業です。
フォロワーとリーダーの両方で実行します。

### フォロワー

```bash
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower
```

実行すると、アームを可動域の中央に動かすよう求められます。
すべての関節を中央付近に合わせて Enter を押します。

次に、`wrist_roll` 以外の関節を、それぞれ可動域の端から端までゆっくり動かします。
すべて動かしたら Enter を押します。
`wrist_roll` は全周回転するジョイントのため、可動域の記録対象から外れています。

### リーダー

```bash
uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader
```

リーダーも同じ手順です。
リーダーはトルクが切れているため、手で軽く動かせます。

キャリブレーション結果は次の場所に保存されます。

```
~/.cache/huggingface/lerobot/calibration/
```

同じ `id` で再実行すると、保存済みの設定を使うか、新しくやり直すかを聞かれます。
関節の範囲を変えた場合は `c` を入力してやり直します。

## テレオペレーション

リーダーを動かすと、フォロワーが同じ動きをします。

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader
```

ポート名を固定した場合は、次のスクリプトでも起動できます。

```bash
./scripts/teleop.sh
```

終了するときは Ctrl+C を押します。

## 動作を記録する

テレオペレーションで操作しながら、カメラ映像と関節角度をデータセットとして記録できます。

```bash
uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/so101-follower \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101-leader \
  --teleop.id=leader \
  --dataset.repo_id=monta/so101_test \
  --dataset.num_episodes=5 \
  --dataset.single_task="Pick up the cube"
```

カメラを使う場合は `--robot.cameras` に設定を追加します。
カメラのデバイス名は `lerobot-find-cameras` で確認できます。

```bash
uv run lerobot-find-cameras opencv
```

USB カメラは接続するたびに番号が変わるため、`/dev/v4l/by-id` の固定名を使うと安定します。

```bash
ls -l /dev/v4l/by-id
```

この環境のカメラは次の名前で見えます。

```
/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0
```

カメラ付きで記録する場合の例です。

```bash
uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/so101-follower \
  --robot.id=follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101-leader \
  --teleop.id=leader \
  --dataset.repo_id=monta/so101_test \
  --dataset.root=outputs/so101_test \
  --dataset.num_episodes=5 \
  --dataset.single_task="Pick up the cube" \
  --dataset.push_to_hub=false
```

カメラと2台のアームを同時に接続するには USB ポートが3つ必要です。
給電付きの USB ハブを使うと、カメラの映像が途切れにくくなります。

## トラブルシューティング

### `/dev/ttyACM0` が表示されない

USB ケーブルがデータ通信に対応しているか確認します。
充電専用ケーブルではデバイスが認識されません。
基板のジャンパが B 側か、AC アダプタが接続されているかも確認します。

```bash
lsusb
dmesg | tail -20
```

`lsusb` に新しいシリアルデバイスが現れない場合は、ケーブルか基板の問題です。

### `Permission denied` が出る

udev ルールが反映されていません。

```bash
sudo ./scripts/install_udev.sh
```

実行後に USB を挿し直します。

### サーボが見つからない

次の順番で確認します。

1. サーボバスが基板から ID 1、2、3、4、5、6 の順につながっているか
2. 各コネクタが奥まで差さっているか
3. AC アダプタの電源が入っているか
4. 同じ ID のサーボが2個ないか

ID 設定の途中で失敗した場合は、サーボを1個だけ基板につないで `lerobot-setup-motors` を再実行します。

### リーダーが重くて動かせない

リーダーとフォロワーのサーボを入れ替えていないか確認します。
リーダーはジョイントごとにギア比が異なります。
フォロワーの 1/345 サーボをリーダーに付けると、手で動かせないほど重くなります。

### フォロワーが震える、脱力する

電源の電圧が不足している可能性があります。
Standard キットは 5V、Pro キットのフォロワーは 12V です。
AC アダプタの容量が足りない場合も、動作中に電圧が落ちて同じ症状が出ます。

### キャリブレーションが途中で失敗する

可動域の記録中にアームが止まると、範囲が正しく記録されません。
各関節を端までゆっくり動かし、ケーブルが突っ張らないことを確認してから再実行します。

## カメラで人を見る

ハンドに付けたカメラで顔を捉えたり、姿勢を指定して写真を撮る方法です。

### 姿勢を指定して撮影する

```bash
uv run python scripts/capture_pose.py \
  --pan -92.5 --lift -108.9 --elbow 65.5 --wrist -101 --roll -179 \
  --output outputs/face_final --name capture.png
```

角度はキャリブレーション後の絶対値です。
撮影すると、YuNet というモデルで顔を検出して枠を描いた画像も出力します。
GPU や外部サービスは不要で、モデルは初回だけ自動ダウンロードされます。

### 顔が写る向きを探す

```bash
uv run python scripts/scan_for_face.py \
  --pan=-105,-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90,105 \
  --lift 0 --elbow=-30 --wrist 0 --roll 0 \
  --output outputs/pan_sweep
```

オフセットは実行開始時の姿勢からの相対値です。
結果は `contact_sheet.png` に一覧化され、`results.json` に検出座標が残ります。

### 動作確認済みの結果

待機姿勢（pan 約 -5、lift 約 -111、elbow 約 +96）から次の値で座っている人の顔が画角の中央に写ります。

| 関節 | 待機姿勢 | 顔を捉える姿勢 |
|---|---|---|
| shoulder_pan | 約 -5 | **-92.5** |
| shoulder_lift | 約 -111 | **-108.9** |
| elbow_flex | 約 +96 | **65.5** |
| wrist_flex | 約 -101 | **-101** |
| wrist_roll | 約 -179 | **-179** |

ポイントは shoulder_pan を約 -90 度回すことと、elbow_flex を +96 から 65.5 に伸ばすことです。
wrist_flex は可動域の下限近くで下向きにできず、下を向かせるには肘で調整します。
結果の画像は `outputs/face_final/annotated_centered.png` にあります。

ハーカーカスケード系の Haar 検出器はこの構図では逆光の影響で失敗します。
YuNet（OpenCV Zoo）は暗い顔でも安定して検出できます。

### 顔をリアルタイムに追跡する

```bash
uv run python scripts/track_face.py
```

- 水平は shoulder_pan、上下は wrist_flex と elbow_flex の二段構えで追従します
- wrist_flex が細かい上下修正を先に受け持ち、出しきれない分や基準位置から離れた分を elbow_flex が引き継ぎます
- 検出は YuNet で、CPU で約 12 fps
- 検出ずれに比例した角速度を既定 30Hz で積み上げて動くので、急に動きません。最大角速度の既定は 45 度/秒（--max-speed）
- pan と wrist_flex はキャリブレーションの可動域いっぱいまで、elbow_flex は既定で 60〜100 度まで動きます。起動時に実効可動域を表示します
- elbow_flex を 60 度より下げる（前へ伸ばす）と重心が前に出て転倒しやすくなります。状況に応じて --elbow-min で調整してください
- 見失っても --coast（既定 0.6 秒）だけは最後の角速度のまま惰性を続けるので、瞬間的な検出のちらつきでは止まりません
- それでも 2 秒以上見失うか、画面端（±20 度）付近でのちらつきが 3 回続くと、pan を既定 15 度/秒で首振りして可動域を探します（--search-speed、--search-delay、--coast で調整、--no-search で無効）
- 探索中に顔を 0.25 秒連続で検出できたら追跡へ復帰します（--acquire で調整）
- 土台の回転角は SO-101 の機械的な可動域（キャリブレーションで ±118 度前後）までで、真後ろは物理的に見えません。首振り探索によって、円を描いて戻ってきた人を自動で再追跡します
- Ctrl+C で停止すると待機姿勢へ自動で戻ります
- wrist_flex を動かしたくない（肘だけにしたい）ときは --no-wrist を付けます

プレビュー画面を出す場合は `--display` を付けます。

#### サーボ制御の注意（プロファイル再起動問題)

毎フレーム「現在位置 + 微小」を書き換えると、STS3215 は目標更新のたびに
速度プロファイルが再起動して、実質的にほとんど動きません（0.1 度/フレーム程度）。
同じ最終目標を 1 回書けば、荷重のかかる肘でも 0.6 秒で約 9 度動きます。
目標は少しずつ更新して同じ値を書き続ける方式にしてください。
`track_face.py` はこの方式（コミット方式）で実装しています。
