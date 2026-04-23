import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import kymnasium as kym
import numpy as np


GRID_W = 36
GRID_H = 36

FLOOR = 0
WALL = 1
GOAL = 2
INF_HITS = 10**9

DIR_LEFT = 0
DIR_UP = 1
DIR_RIGHT = 2
DIR_DOWN = 3

# 실제 env 기준
ACTION_STOP = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT = 2
ACTION_FORWARD = 3
ACTION_PICKUP = 4
ACTION_DROP = 5
ACTION_ATTACK = 6

# 내부 action index == 실제 action 값과 동일하게 사용
ACTIONS = [
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_FORWARD,
    ACTION_PICKUP,
    ACTION_DROP,
    ACTION_ATTACK,
]

OBJ_WALL = 0
OBJ_POND = 1
OBJ_FIRE = 2
OBJ_LINK = 3
OBJ_GOAL = 4
OBJ_CLOUD = 5
OBJ_SWORD = 6
OBJ_TURTLENACK = 7
OBJ_KEESE = 11
OBJ_MOBLIN = 12
OBJ_ARMOS = 13

STATIC_BLOCKING_OBJECT_IDS = {
    OBJ_WALL,
    OBJ_POND,
    OBJ_FIRE,
}

MONSTER_OBJECT_IDS = {
    OBJ_TURTLENACK,
    OBJ_KEESE,
    OBJ_MOBLIN,
    OBJ_ARMOS,
}

CLOUD_PATTERN = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
CLOUD_OPEN_ATTRS = {0, 6}

FRONT_OPEN = 0
FRONT_WALL = 1
FRONT_CLOUD_BLOCKED = 2
FRONT_SWORD = 3
FRONT_TURTLENACK = 4
FRONT_KEESE = 5
FRONT_MOBLIN = 6
FRONT_ARMOS = 7


@dataclass
class TrainConfig:
    episodes: int = 100000
    max_steps: int = 1000
    alpha: float = 0.10
    gamma: float = 0.99
    random_episodes: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    model_path: str = "round5_monster_hits.pkl"


def turn_left(direction: int) -> int:
    return [DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT][direction]


def turn_right(direction: int) -> int:
    return [DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT][direction]


def forward_delta(direction: int) -> Tuple[int, int]:
    return [(-1, 0), (0, -1), (1, 0), (0, 1)][direction]


def extract_cell(grid: np.ndarray, x: int, y: int) -> int:
    if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
        return WALL
    return int(grid[x, y])


def is_cloud_blocked(attr: int) -> bool:
    return attr not in CLOUD_OPEN_ATTRS


def monster_obj_to_front_type(obj: int) -> int:
    if obj == OBJ_TURTLENACK:
        return FRONT_TURTLENACK
    if obj == OBJ_KEESE:
        return FRONT_KEESE
    if obj == OBJ_MOBLIN:
        return FRONT_MOBLIN
    if obj == OBJ_ARMOS:
        return FRONT_ARMOS
    return FRONT_OPEN


def front_type_to_monster_obj(front_type: int) -> int:
    if front_type == FRONT_TURTLENACK:
        return OBJ_TURTLENACK
    if front_type == FRONT_KEESE:
        return OBJ_KEESE
    if front_type == FRONT_MOBLIN:
        return OBJ_MOBLIN
    if front_type == FRONT_ARMOS:
        return OBJ_ARMOS
    raise ValueError(front_type)


def build_static_layout_from_tiles(tiles: np.ndarray):
    grid = np.zeros((GRID_W, GRID_H), dtype=np.int8)
    goal_pos = None
    cloud_positions: set[Tuple[int, int]] = set()
    sword_map: Dict[Tuple[int, int], int] = {}
    monster_map: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for x, y, obj, attr in tiles:
        x = int(x)
        y = int(y)
        obj = int(obj)
        attr = int(attr)

        if obj in STATIC_BLOCKING_OBJECT_IDS:
            grid[x, y] = WALL
        elif obj == OBJ_GOAL:
            grid[x, y] = GOAL
            goal_pos = (x, y)
        else:
            grid[x, y] = FLOOR

        if obj == OBJ_CLOUD:
            cloud_positions.add((x, y))
        elif obj == OBJ_SWORD:
            sword_map[(x, y)] = attr
        elif obj in MONSTER_OBJECT_IDS:
            monster_map[(x, y)] = (obj, attr)

    if goal_pos is None:
        raise RuntimeError("goal position not found from initial tiles")

    return grid, goal_pos, cloud_positions, sword_map, monster_map


def compute_goal_features(
    x: int,
    y: int,
    direction: int,
    goal_pos: Tuple[int, int],
) -> Tuple[int, int]:
    fdx, fdy = forward_delta(direction)
    gx = goal_pos[0] - x
    gy = goal_pos[1] - y

    dot = fdx * gx + fdy * gy
    if dot > 0:
        goal_fb = 1
    elif dot < 0:
        goal_fb = -1
    else:
        goal_fb = 0

    cross = fdx * gy - fdy * gx
    if cross > 0:
        goal_lr = 1
    elif cross < 0:
        goal_lr = -1
    else:
        goal_lr = 0

    return goal_fb, goal_lr


def build_state(
    x: int,
    y: int,
    direction: int,
    sword: int,
    goal_pos: Tuple[int, int],
    front_type: int,
    front_attr: int,
    current_has_sword: int,
):
    goal_fb, goal_lr = compute_goal_features(x, y, direction, goal_pos)
    return (
        x,
        y,
        direction,
        sword,
        goal_fb,
        goal_lr,
        front_type,
        front_attr,
        current_has_sword,
    )


def build_distance_map(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    dist_map = np.full((GRID_W, GRID_H), -1, dtype=np.int32)
    q = deque([goal])
    dist_map[goal[0], goal[1]] = 0

    while q:
        x, y = q.popleft()

        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            nx, ny = x + dx, y + dy

            if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H:
                continue
            if grid[nx, ny] == WALL:
                continue
            if dist_map[nx, ny] != -1:
                continue

            dist_map[nx, ny] = dist_map[x, y] + 1
            q.append((nx, ny))

    return dist_map


def extract_static_layout():
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )
    obs, _ = env.reset()
    env.close()

    tiles = np.asarray(obs["tiles"])
    static_grid, goal_pos, cloud_positions, sword_map, monster_map = build_static_layout_from_tiles(tiles)
    link_x, link_y, sword, direction = map(int, obs["link"])

    return (
        static_grid,
        (link_x, link_y),
        direction,
        goal_pos,
        cloud_positions,
        sword_map,
        monster_map,
    )


def get_attack_reward(monster_obj: int, monster_color: int, sword_color: int) -> float:
    if sword_color == 0:
        return 0.1
    if sword_color == monster_color:
        return 6.0
    return 3.0


def required_hits(monster_obj: int, monster_color: int, sword_color: int) -> int:
    # keese: 10 / 4 / 1
    if monster_obj == OBJ_KEESE:
        if sword_color == 0:
            return 10
        if sword_color == monster_color:
            return 1
        return 4

    # moblin: 20 / 12 / 3
    if monster_obj == OBJ_MOBLIN:
        if sword_color == 0:
            return 20
        if sword_color == monster_color:
            return 3
        return 12

    # turtlenack: 불가 / 20 / 5
    if monster_obj == OBJ_TURTLENACK:
        if sword_color == 0:
            return INF_HITS
        if sword_color == monster_color:
            return 5
        return 20

    # armos: 불가 / 불가 / 5
    if monster_obj == OBJ_ARMOS:
        if sword_color == 0:
            return INF_HITS
        if sword_color != monster_color:
            return INF_HITS
        return 5

    return INF_HITS


def get_candidate_indices(state):
    front_type = state[6]
    sword = state[3]
    current_has_sword = state[8]

    candidates = [0, 1, 2]  # stop, left, right

    if front_type in (FRONT_OPEN, FRONT_SWORD):
        candidates.append(3)  # forward

    if current_has_sword == 1:
        candidates.append(4)  # pickup

    if sword != 0:
        candidates.append(5)  # drop

    if front_type in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
        candidates.append(6)  # attack

    return candidates


class FastZeldaMonsterEnv:
    def __init__(
        self,
        static_grid: np.ndarray,
        start_pos: Tuple[int, int],
        start_dir: int,
        goal_pos: Tuple[int, int],
        cloud_positions: set[Tuple[int, int]],
        sword_map: Dict[Tuple[int, int], int],
        monster_map: Dict[Tuple[int, int], Tuple[int, int]],
        max_steps: int,
    ):
        self.static_grid = static_grid.astype(np.int8, copy=True)
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.goal_pos = goal_pos
        self.cloud_positions = set(cloud_positions)
        self.initial_sword_map = dict(sword_map)
        self.initial_monster_map = dict(monster_map)
        self.max_steps = max_steps

        self.x = start_pos[0]
        self.y = start_pos[1]
        self.direction = start_dir
        self.sword = 0
        self.steps = 0

        self.sword_map: Dict[Tuple[int, int], int] = {}
        self.monster_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.monster_progress: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # value: (last_sword_color, hit_count)

        self.current_dist_map: np.ndarray | None = None

    def build_current_block_grid(self) -> np.ndarray:
        grid = self.static_grid.copy()
        for (mx, my) in self.monster_map.keys():
            grid[mx, my] = WALL
        return grid

    def rebuild_distance_map(self):
        self.current_dist_map = build_distance_map(self.build_current_block_grid(), self.goal_pos)

    def reset(self):
        self.x = self.start_pos[0]
        self.y = self.start_pos[1]
        self.direction = self.start_dir
        self.sword = 0
        self.steps = 0
        self.sword_map = dict(self.initial_sword_map)
        self.monster_map = dict(self.initial_monster_map)
        self.monster_progress = {}
        self.rebuild_distance_map()
        return self.get_state(), {}

    def current_pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def current_tile_sword_color(self) -> int:
        return self.sword_map.get((self.x, self.y), 0)

    def current_cloud_attr(self) -> int:
        return CLOUD_PATTERN[self.steps % len(CLOUD_PATTERN)]

    def is_cloud_at(self, x: int, y: int) -> bool:
        return (x, y) in self.cloud_positions

    def front_info(self) -> Tuple[int, int]:
        dx, dy = forward_delta(self.direction)
        nx, ny = self.x + dx, self.y + dy

        if extract_cell(self.static_grid, nx, ny) == WALL:
            return FRONT_WALL, 0

        if self.is_cloud_at(nx, ny):
            attr = self.current_cloud_attr()
            if is_cloud_blocked(attr):
                return FRONT_CLOUD_BLOCKED, attr

        if (nx, ny) in self.monster_map:
            monster_obj, monster_color = self.monster_map[(nx, ny)]
            return monster_obj_to_front_type(monster_obj), monster_color

        if (nx, ny) in self.sword_map:
            sword_color = self.sword_map[(nx, ny)]
            return FRONT_SWORD, sword_color

        return FRONT_OPEN, 0

    def get_state(self):
        front_type, front_attr = self.front_info()
        current_has_sword = 1 if self.current_tile_sword_color() != 0 else 0

        return build_state(
            x=self.x,
            y=self.y,
            direction=self.direction,
            sword=self.sword,
            goal_pos=self.goal_pos,
            front_type=front_type,
            front_attr=front_attr,
            current_has_sword=current_has_sword,
        )

    def step(self, action: int):
        self.steps += 1
        reward_bonus = 0.0
        monster_killed = False

        dx, dy = forward_delta(self.direction)
        nx, ny = self.x + dx, self.y + dy
        front_type, front_attr = self.front_info()

        if action == ACTION_STOP:
            pass

        elif action == ACTION_TURN_LEFT:
            self.direction = turn_left(self.direction)

        elif action == ACTION_TURN_RIGHT:
            self.direction = turn_right(self.direction)

        elif action == ACTION_FORWARD:
            blocked = front_type in (
                FRONT_WALL,
                FRONT_CLOUD_BLOCKED,
                FRONT_TURTLENACK,
                FRONT_KEESE,
                FRONT_MOBLIN,
                FRONT_ARMOS,
            )
            if not blocked:
                self.x, self.y = nx, ny

        elif action == ACTION_PICKUP:
            current_sword = self.sword_map.get((self.x, self.y))
            if current_sword is not None:
                self.sword = current_sword
                self.sword_map.pop((self.x, self.y), None)
                reward_bonus += 5.0

        elif action == ACTION_DROP:
            if self.sword != 0:
                self.sword_map[(self.x, self.y)] = self.sword
                self.sword = 0
                reward_bonus -= 5.0

        elif action == ACTION_ATTACK:
            if front_type in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
                monster_obj, monster_color = self.monster_map[(nx, ny)]
                req = required_hits(monster_obj, monster_color, self.sword)

                if req < INF_HITS:
                    last_sword, hit_count = self.monster_progress.get((nx, ny), (-1, 0))

                    # 무기가 바뀌면 카운터 리셋
                    if last_sword != self.sword:
                        hit_count = 0

                    hit_count += 1
                    self.monster_progress[(nx, ny)] = (self.sword, hit_count)

                    if hit_count >= req:
                        self.monster_map.pop((nx, ny), None)
                        self.monster_progress.pop((nx, ny), None)
                        reward_bonus += get_attack_reward(monster_obj, monster_color, self.sword)
                        monster_killed = True

        if monster_killed:
            self.rebuild_distance_map()

        terminated = (self.x, self.y) == self.goal_pos
        truncated = self.steps >= self.max_steps

        return self.get_state(), reward_bonus, terminated, truncated, {
            "monster_killed": monster_killed
        }

    def is_useful_attack(self, front_type: int, front_attr: int) -> bool:
        if front_type not in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
            return False
        monster_obj = front_type_to_monster_obj(front_type)
        return required_hits(monster_obj, front_attr, self.sword) < INF_HITS


class Round5EvalAgent(kym.Agent):
    def __init__(self, q_table=None):
        self.q_table = q_table if q_table is not None else {}

    def extract_state(self, observation):
        x, y, sword, direction = map(int, observation["link"])
        tiles = np.asarray(observation["tiles"])
        static_grid, goal_pos, _, _, _ = build_static_layout_from_tiles(tiles)

        dx, dy = forward_delta(direction)
        nx, ny = x + dx, y + dy

        front_type = FRONT_OPEN
        front_attr = 0
        current_has_sword = 0

        for r in tiles:
            tx, ty, obj, attr = map(int, r)

            if tx == x and ty == y and obj == OBJ_SWORD:
                current_has_sword = 1

            if tx == nx and ty == ny:
                if obj == OBJ_CLOUD and is_cloud_blocked(attr):
                    front_type = FRONT_CLOUD_BLOCKED
                    front_attr = attr
                elif obj == OBJ_SWORD:
                    front_type = FRONT_SWORD
                    front_attr = attr
                elif obj in MONSTER_OBJECT_IDS:
                    front_type = monster_obj_to_front_type(obj)
                    front_attr = attr

        if extract_cell(static_grid, nx, ny) == WALL:
            front_type = FRONT_WALL
            front_attr = 0

        return build_state(
            x=x,
            y=y,
            direction=direction,
            sword=sword,
            goal_pos=goal_pos,
            front_type=front_type,
            front_attr=front_attr,
            current_has_sword=current_has_sword,
        )

    def act(self, observation, info):
        state = self.extract_state(observation)
        candidate_indices = get_candidate_indices(state)

        if state not in self.q_table:
            return ACTIONS[random.choice(candidate_indices)]

        q_values = self.q_table[state]
        max_q = max(q_values[i] for i in candidate_indices)
        best_indices = [i for i in candidate_indices if q_values[i] == max_q]
        action_idx = random.choice(best_indices)
        return ACTIONS[action_idx]

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(q_table=data.get("q_table", {}))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table}, f)


class QLearningAgent:
    def __init__(self, q_table=None, epsilon: float = 0.1):
        self.q_table = q_table if q_table is not None else {}
        self.epsilon = epsilon

    def ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTIONS)

    def select_action_index(self, state):
        self.ensure_state(state)
        candidate_indices = get_candidate_indices(state)

        if random.random() < self.epsilon:
            return random.choice(candidate_indices)

        q_values = self.q_table[state]
        max_q = max(q_values[i] for i in candidate_indices)
        best_indices = [i for i in candidate_indices if q_values[i] == max_q]
        return random.choice(best_indices)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table}, f)


def train(config: TrainConfig = TrainConfig()):
    (
        static_grid,
        start_pos,
        start_dir,
        goal_pos,
        cloud_positions,
        sword_map,
        monster_map,
    ) = extract_static_layout()

    env = FastZeldaMonsterEnv(
        static_grid=static_grid,
        start_pos=start_pos,
        start_dir=start_dir,
        goal_pos=goal_pos,
        cloud_positions=cloud_positions,
        sword_map=sword_map,
        monster_map=monster_map,
        max_steps=config.max_steps,
    )

    env.reset()
    start_dist = int(env.current_dist_map[start_pos[0], start_pos[1]])

    print("========== BFS CHECK ==========")
    print("start:", start_pos)
    print("goal :", goal_pos)
    print("distance with alive monsters blocked:", start_dist)
    print("cloud count:", len(cloud_positions))
    print("sword count:", len(sword_map))
    print("monster count:", len(monster_map))
    print("================================")

    agent = QLearningAgent(epsilon=config.epsilon_start)

    success_count = 0
    recent_rewards = []
    recent_success_flags = []
    best_dist_ever = 10**9

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset()
        agent.ensure_state(state)

        visited = {env.current_pos()}
        state_visit_count = {state: 1}

        done = False
        steps = 0
        episode_reward = 0.0
        episode_success = False

        while steps < config.max_steps and not done:
            old_pos = env.current_pos()
            old_dist = int(env.current_dist_map[old_pos[0], old_pos[1]])

            if episode <= config.random_episodes:
                candidate_indices = get_candidate_indices(state)
                action_idx = random.choice(candidate_indices)
            else:
                action_idx = agent.select_action_index(state)

            action = ACTIONS[action_idx]
            front_type = state[6]
            front_attr = state[7]
            current_has_sword = state[8]

            next_state, env_bonus, terminated, truncated, info = env.step(action)
            agent.ensure_state(next_state)

            new_pos = env.current_pos()
            new_dist = int(env.current_dist_map[new_pos[0], new_pos[1]])

            reward = -0.05

            if action_idx == 0:
                reward -= 0.02

            if action_idx in [1, 2]:
                reward -= 0.01

            if action_idx == 3 and new_pos == old_pos:
                reward -= 0.20

            if new_pos in visited:
                reward -= 0.03
            else:
                visited.add(new_pos)
                reward += 0.03

            if old_dist >= 0 and new_dist >= 0:
                reward += 1.0 * (old_dist - new_dist)

            if old_dist >= 0 and new_dist > old_dist:
                reward -= 0.05

            if front_type == FRONT_CLOUD_BLOCKED and action_idx == 0:
                reward += 0.03

            # 앞에 검이 있으면 전진 쪽 유도
            if front_type == FRONT_SWORD and action_idx != 3:
                reward -= 0.10

            # 현재 칸에 검이 있으면 pickup 유도
            if current_has_sword == 1 and action_idx != 4:
                reward -= 0.20

            if action_idx == 4 and current_has_sword == 0:
                reward -= 0.20

            if action_idx == 5 and state[3] == 0:
                reward -= 0.20

            if action_idx == 6:
                if front_type not in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
                    reward -= 0.30
                elif not env.is_useful_attack(front_type, front_attr):
                    reward -= 0.30

            # 검이 없을 때 앞 몬스터별 차등 패널티
            if state[3] == 0:
                if front_type == FRONT_KEESE:
                    reward -= 0.01
                elif front_type == FRONT_MOBLIN:
                    reward -= 0.05
                elif front_type == FRONT_TURTLENACK:
                    reward -= 0.10
                elif front_type == FRONT_ARMOS:
                    reward -= 0.20

            reward += env_bonus

            state_visit_count[next_state] = state_visit_count.get(next_state, 0) + 1
            if state_visit_count[next_state] >= 20:
                truncated = True

            done = terminated or truncated

            if terminated:
                reward += 500.0
                success_count += 1
                episode_success = True

            if truncated and not terminated:
                reward -= 20.0
                if new_dist >= 0:
                    reward += max(0.0, 20.0 - 0.05 * new_dist)

            if new_dist >= 0:
                best_dist_ever = min(best_dist_ever, new_dist)

            old_q = agent.q_table[state][action_idx]
            best_next_q = max(agent.q_table[next_state])
            td_target = reward + (0.0 if done else config.gamma * best_next_q)
            agent.q_table[state][action_idx] = old_q + config.alpha * (td_target - old_q)

            state = next_state
            episode_reward += reward
            steps += 1

        if episode > config.random_episodes:
            agent.epsilon = max(
                config.epsilon_end,
                config.epsilon_start * (config.epsilon_decay ** (episode - config.random_episodes - 1))
            )

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        recent_success_flags.append(1 if episode_success else 0)
        if len(recent_success_flags) > 100:
            recent_success_flags.pop(0)

        if episode % 500 == 0 or episode <= 3:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            recent_success_rate = sum(recent_success_flags) / len(recent_success_flags)
            phase = "random" if episode <= config.random_episodes else "learn"

            print(
                f"[Episode {episode:6d}] "
                f"phase={phase}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"avg_reward={avg_reward:8.2f}, "
                f"success_count={success_count}, "
                f"best_dist_ever={best_dist_ever if best_dist_ever < 10**9 else 'inf'}, "
                f"recent_success_rate={recent_success_rate:.2f}"
            )

    agent.epsilon = 0.0
    agent.save(config.model_path)
    print(f"학습 완료: {config.model_path} 저장")


def run(model_path: str = "round5_monster_hits.pkl"):
    agent = Round5EvalAgent.load(model_path)
    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )


if __name__ == "__main__":
    # train()
    run()