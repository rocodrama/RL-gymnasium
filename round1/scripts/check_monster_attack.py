import gymnasium as gym
import kymnasium
import numpy as np
from collections import deque

# ===== 실제 env action mapping =====
ACTION_STOP = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT = 2
ACTION_FORWARD = 3
ACTION_PICKUP = 4
ACTION_ATTACK = 6

# ===== object ids =====
OBJ_WALL = 0
OBJ_POND = 1
OBJ_FIRE = 2
OBJ_GOAL = 4
OBJ_CLOUD = 5
OBJ_SWORD = 6
OBJ_TURTLENACK = 7
OBJ_KEESE = 11
OBJ_MOBLIN = 12
OBJ_ARMOS = 13

MONSTER_NAMES = {
    OBJ_TURTLENACK: "turtlenack",
    OBJ_KEESE: "keese",
    OBJ_MOBLIN: "moblin",
    OBJ_ARMOS: "armos",
}

GRID_W = 36
GRID_H = 36


def forward_delta(direction: int):
    # direction: 0 left, 1 up, 2 right, 3 down
    if direction == 0:
        return -1, 0
    if direction == 1:
        return 0, -1
    if direction == 2:
        return 1, 0
    if direction == 3:
        return 0, 1
    raise ValueError(direction)


def direction_from_delta(dx, dy):
    if (dx, dy) == (-1, 0):
        return 0
    if (dx, dy) == (0, -1):
        return 1
    if (dx, dy) == (1, 0):
        return 2
    if (dx, dy) == (0, 1):
        return 3
    raise ValueError((dx, dy))


def parse_obs(obs):
    link = tuple(map(int, obs["link"]))
    tiles = np.asarray(obs["tiles"])

    monsters = []
    swords = []
    tile_map = {}

    for row in tiles:
        x, y, obj, attr = map(int, row)
        tile_map[(x, y)] = (obj, attr)

        if obj in MONSTER_NAMES:
            monsters.append((x, y, obj, attr))
        elif obj == OBJ_SWORD:
            swords.append((x, y, attr))

    return link, tiles, monsters, swords, tile_map


def is_cloud_open(attr: int) -> bool:
    return attr in (0, 6)


def build_walkable_grid(obs, block_monsters=True, allow_target_adjacent_only=None):
    """
    allow_target_adjacent_only: target monster position (tx, ty)
    - target monster 칸은 어차피 못 들어감
    - 나머지 몬스터는 벽 처리
    """
    _, _, monsters, _, tile_map = parse_obs(obs)
    grid = np.zeros((GRID_W, GRID_H), dtype=np.int8)

    for x in range(GRID_W):
        for y in range(GRID_H):
            obj, attr = tile_map.get((x, y), (-1, 0))

            blocked = False

            if obj in (OBJ_WALL, OBJ_POND, OBJ_FIRE):
                blocked = True
            elif obj == OBJ_CLOUD and not is_cloud_open(attr):
                blocked = True
            elif block_monsters and obj in MONSTER_NAMES:
                blocked = True

            grid[x, y] = 1 if blocked else 0

    return grid


def bfs_path(grid, start, goals):
    q = deque([start])
    parent = {start: None}
    goals = set(goals)

    found = None
    while q:
        x, y = q.popleft()

        if (x, y) in goals:
            found = (x, y)
            break

        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            if grid[nx, ny] == 1:
                continue
            if (nx, ny) in parent:
                continue

            parent[(nx, ny)] = (x, y)
            q.append((nx, ny))

    if found is None:
        return None

    path = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def step_env(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated, truncated, info


def turn_to_direction(env, obs, target_dir, max_turns=10):
    for _ in range(max_turns):
        x, y, sword, direction = map(int, obs["link"])
        if direction == target_dir:
            return obs, True

        right_steps = (target_dir - direction) % 4
        left_steps = (direction - target_dir) % 4

        if right_steps <= left_steps:
            obs, _, terminated, truncated, _ = step_env(env, ACTION_TURN_RIGHT)
        else:
            obs, _, terminated, truncated, _ = step_env(env, ACTION_TURN_LEFT)

        if terminated or truncated:
            return obs, False

    return obs, False


def move_one_cell(env, obs, next_cell):
    x, y, sword, direction = map(int, obs["link"])
    nx, ny = next_cell
    dx, dy = nx - x, ny - y
    target_dir = direction_from_delta(dx, dy)

    obs, ok = turn_to_direction(env, obs, target_dir)
    if not ok:
        return obs, False

    obs, _, terminated, truncated, _ = step_env(env, ACTION_FORWARD)
    if terminated or truncated:
        return obs, False

    x2, y2, _, _ = map(int, obs["link"])
    return obs, (x2, y2) == (nx, ny)


def move_to_cell(env, obs, target, max_steps=500):
    """
    구름 때문에 막히면 stop 하면서 재탐색
    """
    for _ in range(max_steps):
        x, y, _, _ = map(int, obs["link"])
        if (x, y) == target:
            return obs, True

        grid = build_walkable_grid(obs, block_monsters=True)
        path = bfs_path(grid, (x, y), {target})

        if path is None or len(path) < 2:
            # 길이 없으면 기다렸다가 다시
            obs, _, terminated, truncated, _ = step_env(env, ACTION_STOP)
            if terminated or truncated:
                return obs, False
            continue

        next_cell = path[1]
        obs, moved = move_one_cell(env, obs, next_cell)

        if not moved:
            # cloud timing 등으로 실패 가능 -> 잠깐 기다리고 다시
            obs, _, terminated, truncated, _ = step_env(env, ACTION_STOP)
            if terminated or truncated:
                return obs, False

    return obs, False


def move_to_monster_front(env, obs, monster_pos, max_steps=800):
    tx, ty = monster_pos

    for _ in range(max_steps):
        x, y, _, _ = map(int, obs["link"])

        # 이미 인접하면 방향만 맞춘다
        if abs(x - tx) + abs(y - ty) == 1:
            dx = tx - x
            dy = ty - y
            target_dir = direction_from_delta(dx, dy)
            obs, ok = turn_to_direction(env, obs, target_dir)
            return obs, ok

        grid = build_walkable_grid(obs, block_monsters=True)

        candidate_fronts = []
        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            fx, fy = tx + dx, ty + dy
            if not (0 <= fx < GRID_W and 0 <= fy < GRID_H):
                continue
            if grid[fx, fy] == 0:
                candidate_fronts.append((fx, fy))

        if not candidate_fronts:
            obs, _, terminated, truncated, _ = step_env(env, ACTION_STOP)
            if terminated or truncated:
                return obs, False
            continue

        path = bfs_path(grid, (x, y), set(candidate_fronts))
        if path is None:
            obs, _, terminated, truncated, _ = step_env(env, ACTION_STOP)
            if terminated or truncated:
                return obs, False
            continue

        if len(path) == 1:
            dx = tx - x
            dy = ty - y
            target_dir = direction_from_delta(dx, dy)
            obs, ok = turn_to_direction(env, obs, target_dir)
            return obs, ok

        next_cell = path[1]
        obs, moved = move_one_cell(env, obs, next_cell)
        if not moved:
            obs, _, terminated, truncated, _ = step_env(env, ACTION_STOP)
            if terminated or truncated:
                return obs, False

    return obs, False


def pickup_current_sword(env, obs):
    x, y, sword, direction = map(int, obs["link"])
    _, _, _, _, tile_map = parse_obs(obs)
    obj, attr = tile_map.get((x, y), (-1, 0))

    if obj != OBJ_SWORD:
        return obs, False, None

    obs, _, terminated, truncated, _ = step_env(env, ACTION_PICKUP)
    if terminated or truncated:
        return obs, False, None

    _, _, new_sword, _ = map(int, obs["link"])
    return obs, new_sword != 0, new_sword


def monster_alive(obs, monster_pos, monster_obj):
    _, _, _, _, tile_map = parse_obs(obs)
    obj, attr = tile_map.get(monster_pos, (-1, 0))
    return obj == monster_obj


def attack_until_dead(env, obs, monster_pos, monster_obj, max_attacks=20):
    count = 0
    for _ in range(max_attacks):
        if not monster_alive(obs, monster_pos, monster_obj):
            return obs, count, True

        obs, _, terminated, truncated, _ = step_env(env, ACTION_ATTACK)
        count += 1

        if terminated or truncated:
            return obs, count, False

        if not monster_alive(obs, monster_pos, monster_obj):
            return obs, count, True

    return obs, count, False


def choose_sword(swords, target_color, mode):
    """
    mode:
    - "any": 같은 색이 아닌 검 우선, 없으면 아무 검
    - "same": 같은 색 검
    """
    if mode == "same":
        for x, y, color in swords:
            if color == target_color:
                return (x, y, color)
        return None

    if mode == "any":
        # 같은 색이 아닌 검 우선
        for x, y, color in swords:
            if color != target_color:
                return (x, y, color)
        if swords:
            return swords[0]
        return None

    return None


def run_one_case(monster, mode):
    """
    mode:
    - bare
    - any
    - same
    """
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )

    obs, _ = env.reset()
    link, tiles, monsters, swords, tile_map = parse_obs(obs)

    tx, ty, monster_obj, monster_color = monster
    monster_name = MONSTER_NAMES[monster_obj]

    # 1) 검 필요시 검 먼저 줍기
    if mode in ("any", "same"):
        sword = choose_sword(swords, monster_color, mode)
        if sword is None:
            env.close()
            return {
                "ok": False,
                "reason": f"no sword available for mode={mode}",
            }

        sx, sy, sword_color = sword

        obs, ok = move_to_cell(env, obs, (sx, sy))
        if not ok:
            env.close()
            return {
                "ok": False,
                "reason": f"failed to reach sword {sword}",
            }

        obs, ok, picked_color = pickup_current_sword(env, obs)
        if not ok:
            env.close()
            return {
                "ok": False,
                "reason": f"failed to pickup sword at {(sx, sy)}",
            }

    # 2) 몬스터 앞까지 이동
    obs, ok = move_to_monster_front(env, obs, (tx, ty))
    if not ok:
        env.close()
        return {
            "ok": False,
            "reason": f"failed to move in front of monster {(tx, ty)}",
        }

    # 3) 공격 반복
    obs, attack_count, killed = attack_until_dead(env, obs, (tx, ty), monster_obj)

    env.close()

    return {
        "ok": True,
        "monster_name": monster_name,
        "monster_color": monster_color,
        "mode": mode,
        "attacks": attack_count,
        "killed": killed,
    }


def main():
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )
    obs, _ = env.reset()
    monsters, swords = parse_obs(obs)[2], parse_obs(obs)[3]
    env.close()

    print("=== MONSTERS ===")
    for m in monsters:
        print(m, MONSTER_NAMES[m[2]])

    print("\n=== SWORDS ===")
    for s in swords:
        print(s)

    # 고유 (monster_type, color) 조합만 1개씩
    unique_monsters = {}
    for m in monsters:
        _, _, obj, color = m
        unique_monsters.setdefault((obj, color), m)

    print("\n=== TEST START ===")
    results = []

    for key, monster in unique_monsters.items():
        tx, ty, monster_obj, monster_color = monster
        print(f"\n[TEST] {MONSTER_NAMES[monster_obj]} color={monster_color} at {(tx, ty)}")

        for mode in ["bare", "any", "same"]:
            result = run_one_case(monster, mode)
            results.append((monster, mode, result))
            print(f"  mode={mode} -> {result}")

    print("\n=== SUMMARY ===")
    for monster, mode, result in results:
        tx, ty, monster_obj, monster_color = monster
        name = MONSTER_NAMES[monster_obj]

        if result["ok"]:
            print(
                f"{name:10s} color={monster_color} mode={mode:4s} "
                f"attacks={result['attacks']} killed={result['killed']}"
            )
        else:
            print(
                f"{name:10s} color={monster_color} mode={mode:4s} "
                f"FAILED: {result['reason']}"
            )


if __name__ == "__main__":
    main()