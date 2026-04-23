import gymnasium as gym
import kymnasium
import numpy as np

env = gym.make(
    id="kymnasium/ZeldaAdventure-Stage-3",
    render_mode=None,
    bgm=False,
)

obs, info = env.reset()

print("=== link ===")
print(obs["link"])
print(type(obs["link"]), obs["link"].shape)

print("\n=== tiles ===")
print(type(obs["tiles"]))
print(obs["tiles"].shape)
print(obs["tiles"][:50])   # 앞부분만

tiles = np.asarray(obs["tiles"])

print("\n=== x range / y range ===")
print("x:", tiles[:, 0].min(), "~", tiles[:, 0].max())
print("y:", tiles[:, 1].min(), "~", tiles[:, 1].max())

print("\n=== unique object ids ===")
print(np.unique(tiles[:, 2]))

print("\n=== counts by object id ===")
obj_ids, counts = np.unique(tiles[:, 2], return_counts=True)
for obj_id, cnt in zip(obj_ids, counts):
    print(f"object_id={int(obj_id):2d}, count={int(cnt)}")

obs, _ = env.reset()
tiles0 = np.asarray(obs["tiles"])
# (x,y) -> index 매핑
index_map = {(int(x), int(y)): i for i, (x, y, _, _) in enumerate(tiles0)}

print("=== CHECK START ===")

for step in range(10):
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    tiles = np.asarray(obs["tiles"])

    mismatch = 0

    for (x, y), i in index_map.items():
        tx, ty = int(tiles[i][0]), int(tiles[i][1])
        if (tx, ty) != (x, y):
            mismatch += 1
            break

    print(f"step {step}: mismatch={mismatch}")
env.close()