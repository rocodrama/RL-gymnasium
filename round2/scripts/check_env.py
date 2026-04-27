import gymnasium as gym
import kymnasium
import numpy as np
import pygame
import time

env = gym.make(
    id="kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1",
    render_mode="human",
    bgm=False,
)

obs, info = env.reset()

prev_blurps = None
step = 0

while True:
    action = 0

    # pygame 이벤트 처리 필수
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        action = 1
    elif keys[pygame.K_RIGHT]:
        action = 2

    obs, reward, terminated, truncated, info = env.step(action)

    blurps = np.asarray(obs["blurps"], dtype=np.float32)
    mario = np.asarray(obs["mario"], dtype=np.float32)

    active = np.any(blurps != 0, axis=1)
    active_blurps = blurps[active]

    if step % 10 == 0:
        print("\n" + "=" * 60)
        print("step:", step, "action:", action, "time:", info.get("time_elapsed"))
        print("mario:", mario)

        print("\nactive blurps:", len(active_blurps))
        for i, b in enumerate(active_blurps[:5]):
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            print(f"{i}: raw={b}, center=({cx:.1f}, {cy:.1f})")

        print("=" * 60)

    if terminated or truncated:
        print("episode end:", "terminated=", terminated, "truncated=", truncated)
        obs, info = env.reset()

    step += 1
    time.sleep(1 / 30)