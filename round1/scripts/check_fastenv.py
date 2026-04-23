import gymnasium as gym
import kymnasium as kym
import numpy as np

def build_grid(tiles):
    grid = np.zeros((36, 36), dtype=np.int8)
    goal = None
    for x, y, obj, _ in tiles:
        x, y, obj = int(x), int(y), int(obj)
        if obj in [0, 1, 2, 5, 7, 11, 12, 13]:
            grid[x, y] = 1
        elif obj == 4:
            grid[x, y] = 2
            goal = (x, y)
    return grid, goal

def run_compare(action_seq):
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )
    obs, info = env.reset()

    link = tuple(map(int, obs["link"]))
    grid, goal = build_grid(obs["tiles"])

    print("start real:", link)
    print("goal:", goal)
    print("actions:", action_seq)

    for i, action in enumerate(action_seq, 1):
        obs, _, terminated, truncated, _ = env.step(action)
        link = tuple(map(int, obs["link"]))
        print(f"step={i:02d}, action={action}, real_link={link}, done={terminated or truncated}")
        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    run_compare([1, 1, 3, 3, 2, 3, 3])