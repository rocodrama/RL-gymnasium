import gymnasium as gym
import kymnasium
import numpy as np

OBJ_CLOUD = 5


def extract_cloud_rows(tiles: np.ndarray):
    clouds = []
    for i, row in enumerate(tiles):
        x, y, obj, attr = map(int, row)
        if obj == OBJ_CLOUD:
            clouds.append((i, x, y, attr))
    return clouds


def run_check(action, max_check_steps=20):
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )

    obs, _ = env.reset()
    tiles0 = np.asarray(obs["tiles"])
    clouds0 = extract_cloud_rows(tiles0)

    cloud_indices = np.array([idx for idx, _, _, _ in clouds0], dtype=np.int32)
    cloud_positions = [(x, y) for _, x, y, _ in clouds0]

    initial_attrs = tuple(int(tiles0[idx][3]) for idx in cloud_indices)

    print(f"\n=== ACTION {action} CHECK ===")
    print("start link:", tuple(map(int, obs["link"])))
    print("initial_attrs =", initial_attrs)

    for step in range(1, max_check_steps + 1):
        obs, _, terminated, truncated, _ = env.step(action)
        tiles = np.asarray(obs["tiles"])
        link = tuple(map(int, obs["link"]))

        mismatch = False
        for idx, (x0, y0) in zip(cloud_indices, cloud_positions):
            x = int(tiles[idx][0])
            y = int(tiles[idx][1])
            if (x, y) != (x0, y0):
                mismatch = True
                print(
                    f"[WARN] row mismatch at step={step}: "
                    f"expected=({x0},{y0}), got=({x},{y})"
                )
                break

        if mismatch:
            break

        attrs = tuple(int(tiles[idx][3]) for idx in cloud_indices)

        print(f"step={step:2d} link={link} attrs={attrs}")

        if terminated or truncated:
            print(f"[INFO] episode ended at step={step}")
            break

    env.close()


if __name__ == "__main__":
    run_check(action=1)  # 회전