"""バイラテラル制御用の高速 I/O 層。

lerobot の Robot クラスを通さず、FeetechMotorsBus のポートとパケットハンドラを直接使う。
1周期で Present_Position から Present_Current までを1回の sync_read（アドレス 56〜70）でまとめて読み、
Goal_Time（PWM 指令）を1回の sync_write で書く。2本のバス（リーダーとフォロワー）は
BilateralIO がスレッドで並列に動かす。

値は生の整数のまま返す（角度の換算やキャリブレーションは上位の層の仕事）。
PWM モードの Present_Position は Homing_Offset が効かない生の値になるので、
位置モードの値との対応は docs/bilateral_plan.md のフェーズ1のメモを参照。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_IDS = {name: i + 1 for i, name in enumerate(JOINTS)}
PORTS = {"leader": "/dev/so101-leader", "follower": "/dev/so101-follower"}

ADDR_GOAL_TIME = 44
ADDR_STATE = 56  # Present_Position
STATE_LEN = 15  # 56..70: position, velocity, load, voltage, temperature, ..., current(69-70)
OFF_VEL, OFF_LOAD, OFF_VOLT, OFF_TEMP, OFF_CUR = 2, 4, 6, 7, 13
PWM_DIR_BIT = 10  # フェーズ0の実測: 方向ビットは bit10。正の PWM は raw 位置が減る向き
PWM_MAX = 1000
VEL_SIGN_BIT = 15


@dataclass
class State:
    """1周期ぶんの読み値（生の整数）。リストは JOINTS の順。"""

    t: float  # 読み終わった時刻（perf_counter）
    pos: list[int]
    vel: list[int]  # 符号つき（step/s）
    load: list[int]
    temp: list[int]
    current: list[int]  # 符号なし。向きは PWM 指令から決める


def encode_pwm(value: int) -> int:
    """符号つきの PWM（-1000〜1000）を Goal_Time の値に変換する。"""
    value = max(-PWM_MAX, min(PWM_MAX, int(value)))
    return abs(value) | ((1 << PWM_DIR_BIT) if value < 0 else 0)


def _sign_magnitude(raw: int, sign_bit: int) -> int:
    return -(raw & ~(1 << sign_bit)) if raw & (1 << sign_bit) else raw


class ArmBus:
    """1本のバス（6関節）の読み書き。"""

    def __init__(self, arm: str):
        self.arm = arm
        motors = {name: Motor(JOINT_IDS[name], "sts3215", MotorNormMode.DEGREES) for name in JOINTS}
        self.bus = FeetechMotorsBus(port=PORTS[arm], motors=motors)
        self.ids = [JOINT_IDS[name] for name in JOINTS]

    def connect(self) -> None:
        self.bus.connect()
        reader = self.bus.sync_reader
        reader.clearParam()
        reader.start_address = ADDR_STATE
        reader.data_length = STATE_LEN
        for motor_id in self.ids:
            reader.addParam(motor_id)

    def read_state(self) -> State:
        reader = self.bus.sync_reader
        comm = reader.txRxPacket()
        if comm != 0:
            raise ConnectionError(f"{self.arm}: sync_read 失敗 ({self.bus.packet_handler.getTxRxResult(comm)})")
        t = time.perf_counter()
        pos, vel, load, temp, cur = [], [], [], [], []
        for motor_id in self.ids:
            get = lambda off, n: reader.getData(motor_id, ADDR_STATE + off, n)  # noqa: E731
            pos.append(get(0, 2))
            vel.append(_sign_magnitude(get(OFF_VEL, 2), VEL_SIGN_BIT))
            load.append(get(OFF_LOAD, 2))
            temp.append(get(OFF_TEMP, 1))
            cur.append(get(OFF_CUR, 2))
        return State(t, pos, vel, load, temp, cur)

    def write_pwm(self, pwm: list[int]) -> None:
        """符号つきの PWM を6関節ぶん、1回の sync_write で書く。"""
        writer = self.bus.sync_writer
        writer.clearParam()
        writer.start_address = ADDR_GOAL_TIME
        writer.data_length = 2
        for motor_id, value in zip(self.ids, pwm):
            raw = encode_pwm(value)
            writer.addParam(motor_id, [raw & 0xFF, raw >> 8])
        comm = writer.txPacket()
        if comm != 0:
            raise ConnectionError(f"{self.arm}: sync_write 失敗 ({self.bus.packet_handler.getTxRxResult(comm)})")

    def write_pwm_one(self, joint: str, value: int) -> None:
        """1関節だけに PWM を書く。他の関節は触らない（同定で他を位置モードのままにするため）。"""
        raw = encode_pwm(value)
        self.bus.write("Goal_Time", joint, raw, normalize=False)

    def set_mode(self, mode: OperatingMode, joints: list[str] | None = None) -> None:
        """Operating_Mode を切り替える（EEPROM。起動時と終了時の1回ずつだけ呼ぶ）。joints を省くと全関節。"""
        for name in joints or JOINTS:
            self.bus.write("Torque_Enable", name, 0, normalize=False)
            self.bus.write("Lock", name, 0, normalize=False)
            self.bus.write("Operating_Mode", name, mode.value, normalize=False)
            self.bus.write("Lock", name, 1, normalize=False)

    def disconnect(self) -> None:
        self.bus.disconnect(disable_torque=False)


class BilateralIO:
    """リーダーとフォロワーを、1周期ごとにスレッドで並列に読み書きする。"""

    def __init__(self):
        self.arms = {arm: ArmBus(arm) for arm in PORTS}
        self.pool = ThreadPoolExecutor(max_workers=len(self.arms))

    def connect(self) -> None:
        for arm in self.arms.values():
            arm.connect()

    def cycle(self, pwm: dict[str, list[int]]) -> dict[str, State]:
        """両腕を並列に「読む → 書く」。返すのは書く前に読んだ状態。"""

        def run(name: str) -> State:
            state = self.arms[name].read_state()
            self.arms[name].write_pwm(pwm[name])
            return state

        futures = {name: self.pool.submit(run, name) for name in self.arms}
        return {name: f.result() for name, f in futures.items()}

    def disconnect(self) -> None:
        self.pool.shutdown(wait=True)
        for arm in self.arms.values():
            arm.disconnect()
