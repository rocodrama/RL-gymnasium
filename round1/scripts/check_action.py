import gymnasium as gym
import kymnasium as kym

def test_actions():
    for action in [1, 2, 3]:
        env = gym.make(
            id="kymnasium/ZeldaAdventure-Stage-3",
            render_mode=None,
            bgm=False,
        )
        obs, info = env.reset()
        before = tuple(map(int, obs["link"]))

        next_obs, _, terminated, truncated, _ = env.step(action)
        after = tuple(map(int, next_obs["link"]))

        print(f"action={action} | before={before} -> after={after}")
        env.close()

if __name__ == "__main__":
    test_actions()