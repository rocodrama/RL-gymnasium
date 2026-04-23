import gymnasium as gym
import kymnasium
import numpy as np

OBJ_SWORD = 6
MONSTERS = {7: "turtlenack", 11: "keese", 12: "moblin", 13: "armos"}

def find_tiles(obs):
    tiles = np.asarray(obs["tiles"])
    monsters = []
    swords = []

    for row in tiles:
        x, y, obj, attr = map(int, row)
        if obj in MONSTERS:
            monsters.append((x, y, obj, attr))
        elif obj == OBJ_SWORD:
            swords.append((x, y, attr))

    return monsters, swords

def main():
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )

    obs, _ = env.reset()
    monsters, swords = find_tiles(obs)

    print("=== MONSTERS ===")
    for m in monsters:
        print(m, MONSTERS[m[2]])

    print("\n=== SWORDS ===")
    for s in swords:
        print(s)

    env.close()

if __name__ == "__main__":
    main()