#!/usr/bin/env python3
"""フェーズ1: 両腕6関節の「読み → 書き」1周期の時間を測る。

モードは変えず（位置モードのまま）、トルクは入れない。書くのは Goal_Time=0 だけで、
位置モードでは移動時間の指定なので無害（トルク OFF のため腕は動かない）。
--mode single は1本ずつ順に、--mode parallel は2本を並列に回す。
"""

from __future__ import annotations

import argparse
import statistics
import time

from bus import PORTS, ArmBus, BilateralIO


def report(name: str, times: list[float]) -> None:
    ms = sorted(t * 1000 for t in times)
    p = lambda q: ms[min(len(ms) - 1, int(len(ms) * q))]  # noqa: E731
    print(
        f"{name:>10}: 平均 {statistics.mean(ms):.2f} ms ({1000 / statistics.mean(ms):.0f} Hz)  "
        f"中央 {p(0.5):.2f}  95% {p(0.95):.2f}  99% {p(0.99):.2f}  最悪 {ms[-1]:.2f}  n={len(ms)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1000)
    args = ap.parse_args()
    zero = [0] * 6

    for arm_name in PORTS:
        arm = ArmBus(arm_name)
        arm.connect()
        try:
            times = []
            for _ in range(args.cycles):
                t0 = time.perf_counter()
                arm.read_state()
                arm.write_pwm(zero)
                times.append(time.perf_counter() - t0)
            report(arm_name, times)
            s = arm.read_state()
            print("   pos", s.pos, "vel", s.vel, "temp", s.temp, "cur", s.current)
        finally:
            arm.disconnect()

    io = BilateralIO()
    io.connect()
    try:
        times = []
        for _ in range(args.cycles):
            t0 = time.perf_counter()
            io.cycle({"leader": zero, "follower": zero})
            times.append(time.perf_counter() - t0)
        report("parallel", times)
    finally:
        io.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
