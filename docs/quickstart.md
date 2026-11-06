# 実機セットアップ手順書

Pro キット（リーダー 5V、フォロワー 12V）用の作業チェックリストです。
上から順に進めます。

## 0. 電源とケーブルを分ける

- [ ] 5V の AC アダプタに「5V」、12V の AC アダプタに「12V」と書いたラベルを貼る
- [ ] 12V はフォロワー専用、5V はリーダー用と決める
- [ ] USB ケーブル2本がデータ通信対応か確認する
- [ ] サーボバスの3ピンケーブルをアームごとに分ける
- [ ] 各サーボに `F1` から `F6`、`L1` から `L6` のラベルを貼る
- [ ] ケーブルの両端にも同じ番号を貼る
- [ ] 両方の基板のジャンパ2個が B 側になっていることを確認する

## 1. フォロワーのポートを確認する

- [ ] フォロワーの基板に 12V アダプタを接続する
- [ ] フォロワーの基板と PC を USB で接続する
- [ ] サーボバスはまだ基板に接続しない
- [ ] 次のコマンドを実行する

```bash
cd /home/monta/workspace/SO101
./scripts/status.sh
```

`USB シリアルのポート` に `/dev/ttyACM0` などが表示されることを確認します。
表示されない場合は、USB ケーブル、基板のジャンパ、12V アダプタを確認します。

- [ ] udev ルールを入れる

```bash
sudo ./scripts/install_udev.sh
```

- [ ] USB ケーブルを挿し直す
- [ ] ポート名を確認する

```bash
uv run lerobot-find-port
```

表示されたフォロワーのポートをメモします。
以下の例では `/dev/ttyACM0` とします。

## 2. フォロワーのモーター ID を設定する

まず、サーボの ID がすでに 1〜6 で揃っているか確認します。

```bash
uv run python scripts/check_servos.py --port /dev/ttyACM0
```

6個すべてが ID 1〜6 で表示された場合は、ID 設定をやり直す必要はありません。
次のコマンドで、ID と実際の関節の対応を確認します。

```bash
uv run python scripts/check_mapping.py --port /dev/ttyACM0
```

画面の指示に従い、表示された関節だけを手で動かします。
すべて `OK` ならキャリブレーションに進みます。

ID が揃っていない場合、または対応がずれている場合だけ、以下の ID 設定を行います。
サーボバスをすべて基板から外し、1個ずつ接続します。

```bash
uv run lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower
```

- [ ] `gripper` のサーボだけを基板につなぎ、Enter を押す
- [ ] `wrist_roll` に交換して Enter を押す
- [ ] `wrist_flex` に交換して Enter を押す
- [ ] `elbow_flex` に交換して Enter を押す
- [ ] `shoulder_lift` に交換して Enter を押す
- [ ] `shoulder_pan` に交換して Enter を押す

各サーボを接続するときは、他のサーボと数珠つなぎにしません。

すべて終わったら、サーボバスを次の順で接続します。

```
基板 → F1 → F2 → F3 → F4 → F5 → F6
```

## 3. フォロワーをキャリブレーションする

- [ ] アームを可動域の中央に置く
- [ ] 次のコマンドを実行する

```bash
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower
```

- [ ] Enter を押す
- [ ] `wrist_roll` 以外の関節を、それぞれ端から端までゆっくり動かす
- [ ] Enter を押す
- [ ] `Calibration saved to ...` が表示されることを確認する

## 4. リーダーのポートを確認する

- [ ] フォロワーの USB と電源を外す
- [ ] リーダーの基板に 5V アダプタを接続する
- [ ] リーダーの基板と PC を USB で接続する
- [ ] `./scripts/status.sh` でポートを確認する
- [ ] 必要なら `uv run lerobot-find-port` でポートを特定する

以下の例では `/dev/ttyACM0` とします。

## 5. リーダーのモーター ID を設定する

サーボバスをすべて基板から外し、1個ずつ接続します。

```bash
uv run lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader
```

画面の指示に従い、`gripper` から `shoulder_pan` まで順に設定します。

すべて終わったら、サーボバスを次の順で接続します。

```
基板 → L1 → L2 → L3 → L4 → L5 → L6
```

## 6. リーダーをキャリブレーションする

- [ ] アームを可動域の中央に置く
- [ ] 次のコマンドを実行する

```bash
uv run lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader
```

- [ ] Enter を押す
- [ ] `wrist_roll` 以外の関節を、それぞれ端から端までゆっくり動かす
- [ ] Enter を押す
- [ ] `Calibration saved to ...` が表示されることを確認する

## 7. 2台を接続してポート名を固定する

- [ ] リーダーとフォロワーの両方に電源と USB を接続する
- [ ] 次のコマンドを実行し、リーダーの USB を抜いてポートを特定する

```bash
uv run lerobot-find-port
```

- [ ] もう一度実行し、フォロワーの USB を抜いてポートを特定する

ポートが分かったら、役割とポートを指定して固定します。

```bash
sudo ./scripts/pin_ports.sh leader /dev/ttyACM1 follower /dev/ttyACM0
```

- [ ] `/dev/so101-leader` と `/dev/so101-follower` ができたことを確認する

```bash
./scripts/status.sh
```

## 8. テレオペレーションする

```bash
./scripts/teleop.sh
```

- [ ] リーダーを動かすとフォロワーが追従することを確認する
- [ ] 可動域の端でケーブルが突っ張らないことを確認する
- [ ] Ctrl+C で終了する

## 完了後の確認

- [ ] 電源を切って入れ直しても、同じポート名で接続できる
- [ ] リーダーを動かしたときにフォロワーが震えない
- [ ] キャリブレーション結果が `~/.cache/huggingface/lerobot/calibration/` に保存されている
