"""
학습 전에 Avoid Blurp 환경의 observation, action, info 구조를 확인하는 스크립트입니다.

프로젝트 루트에서 실행:
    python scripts/check_env.py
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import gymnasium as gym
import kymnasium  # noqa: F401 - kymnasium 환경 등록을 위해 import합니다.
import numpy as np


ENV_ID = "kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1"


def describe_observation(observation: Dict[str, Any]) -> None:
    mario = np.asarray(observation.get("mario", []), dtype=np.float32)
    blurps = np.asarray(observation.get("blurps", []), dtype=np.float32)
    visible_blurps = 0

    if blurps.ndim == 2:
        visible_blurps = int(np.sum(~np.all(np.isclose(blurps, 0.0), axis=1)))

    print("observation keys:", sorted(observation.keys()))
    print("mario shape:", mario.shape, "values:", mario.tolist())
    print("blurps shape:", blurps.shape, "visible rows:", visible_blurps)

    if blurps.ndim == 2 and blurps.shape[0] > 0:
        print("first blurp row:", blurps[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--bgm", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        id=ENV_ID,
        render_mode=args.render_mode,
        bgm=args.bgm,
    )

    try:
        observation, info = env.reset()
        print("env id:", ENV_ID)
        print("action space:", env.action_space)
        print("observation space:", env.observation_space)
        print("reset info:", info)
        describe_observation(observation)

        previous_time = float(info.get("time_elapsed", 0.0) or 0.0)
        for step in range(1, args.steps + 1):
            action = int(env.action_space.sample())
            observation, reward, terminated, truncated, info = env.step(action)
            current_time = float(info.get("time_elapsed", previous_time) or previous_time)
            print(
                f"step={step:03d} "
                f"action={action} "
                f"reward={reward} "
                f"time={current_time:.4f} "
                f"delta={current_time - previous_time:.4f} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
            previous_time = current_time

            if terminated or truncated:
                print("episode ended during inspection")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
