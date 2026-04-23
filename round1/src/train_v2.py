import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import kymnasium as kym
import numpy as np


GRID_W = 36
GRID_H = 36

FLOOR = 0
WALL = 1
GOAL = 2

DIR_LEFT = 0
DIR_UP = 1
DIR_RIGHT = 2
DIR_DOWN = 3

# 실제 env 기준
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT = 2
ACTION_FORWARD = 3

# 내부 action index
# 0 -> left
# 1 -> right
# 2 -> forward
ACTIONS = [ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD]

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
    OBJ_TURTLENACK,
    OBJ_KEESE,
    OBJ_MOBLIN,
    OBJ_ARMOS,
}

CLOUD_PATTERN = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
CLOUD_OPEN_ATTRS = {0, 6}


@dataclass
class TrainConfig:
    episodes: int = 10000
    max_steps: int = 1000
    alpha: float = 0.10
    gamma: float = 0.99
    random_episodes: int = 2000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    model_path: str = "round2_cloud_fast.pkl"


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


def build_static_grid_goal_and_cloud_positions(
    tiles: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int], set[Tuple[int, int]]]:
    grid = np.zeros((GRID_W, GRID_H), dtype=np.int8)
    goal_pos = None
    cloud_positions: set[Tuple[int, int]] = set()

    for x, y, obj, _ in tiles:
        x = int(x)
        y = int(y)
        obj = int(obj)

        if obj in STATIC_BLOCKING_OBJECT_IDS:
            grid[x, y] = WALL
        elif obj == OBJ_GOAL:
            grid[x, y] = GOAL
            goal_pos = (x, y)
        else:
            grid[x, y] = FLOOR

        if obj == OBJ_CLOUD:
            cloud_positions.add((x, y))

    if goal_pos is None:
        raise RuntimeError("goal position not found from initial tiles")

    return grid, goal_pos, cloud_positions


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
    goal_pos: Tuple[int, int],
    front_blocked: int,
) -> Tuple[int, int, int, int, int, int]:
    goal_fb, goal_lr = compute_goal_features(x, y, direction, goal_pos)
    return (x, y, direction, goal_fb, goal_lr, front_blocked)


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
    static_grid, goal_pos, cloud_positions = build_static_grid_goal_and_cloud_positions(tiles)
    link_x, link_y, _, direction = map(int, obs["link"])

    return static_grid, (link_x, link_y), direction, goal_pos, cloud_positions


class FastZeldaCloudEnv:
    def __init__(
        self,
        static_grid: np.ndarray,
        start_pos: Tuple[int, int],
        start_dir: int,
        goal_pos: Tuple[int, int],
        cloud_positions: set[Tuple[int, int]],
        max_steps: int,
    ):
        self.static_grid = static_grid.astype(np.int8, copy=True)
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.goal_pos = goal_pos
        self.cloud_positions = set(cloud_positions)
        self.max_steps = max_steps

        self.x = start_pos[0]
        self.y = start_pos[1]
        self.direction = start_dir
        self.steps = 0

    def reset(self):
        self.x = self.start_pos[0]
        self.y = self.start_pos[1]
        self.direction = self.start_dir
        self.steps = 0
        return self.get_state(), {}

    def current_pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def current_cloud_attr(self) -> int:
        return CLOUD_PATTERN[self.steps % len(CLOUD_PATTERN)]

    def is_cloud_at(self, x: int, y: int) -> bool:
        return (x, y) in self.cloud_positions

    def compute_front_blocked(self) -> int:
        dx, dy = forward_delta(self.direction)
        nx, ny = self.x + dx, self.y + dy

        if extract_cell(self.static_grid, nx, ny) == WALL:
            return 1

        if self.is_cloud_at(nx, ny):
            attr = self.current_cloud_attr()
            if is_cloud_blocked(attr):
                return 1

        return 0

    def get_state(self):
        front_blocked = self.compute_front_blocked()
        return build_state(
            x=self.x,
            y=self.y,
            direction=self.direction,
            goal_pos=self.goal_pos,
            front_blocked=front_blocked,
        )

    def step(self, action: int):
        self.steps += 1

        if action == ACTION_TURN_LEFT:
            self.direction = turn_left(self.direction)

        elif action == ACTION_TURN_RIGHT:
            self.direction = turn_right(self.direction)

        elif action == ACTION_FORWARD:
            dx, dy = forward_delta(self.direction)
            nx, ny = self.x + dx, self.y + dy

            blocked = False

            if extract_cell(self.static_grid, nx, ny) == WALL:
                blocked = True
            elif self.is_cloud_at(nx, ny):
                attr = self.current_cloud_attr()
                if is_cloud_blocked(attr):
                    blocked = True

            if not blocked:
                self.x, self.y = nx, ny

        terminated = (self.x, self.y) == self.goal_pos
        truncated = self.steps >= self.max_steps

        return self.get_state(), 0.0, terminated, truncated, {}


class Round2EvalAgent(kym.Agent):
    def __init__(self, q_table=None):
        self.q_table = q_table if q_table is not None else {}

    def extract_state(self, observation):
        x, y, _, direction = map(int, observation["link"])
        tiles = np.asarray(observation["tiles"])

        static_grid, goal_pos, _ = build_static_grid_goal_and_cloud_positions(tiles)

        dx, dy = forward_delta(direction)
        nx, ny = x + dx, y + dy

        front_blocked = 0

        if extract_cell(static_grid, nx, ny) == WALL:
            front_blocked = 1
        else:
            # 실제 eval에서는 현재 앞칸 row만 직접 확인
            # 전체 dict/cloud map 생성 안 함
            row = None
            # tiles row order는 고정이지만 eval 쪽은 단순하게 전체 스캔 대신
            # 바로 좌표 한 칸만 찾는다.
            # 규모가 작아서 평가에는 충분히 안전함.
            for r in tiles:
                if int(r[0]) == nx and int(r[1]) == ny:
                    row = r
                    break

            if row is not None:
                obj = int(row[2])
                attr = int(row[3])
                if obj == OBJ_CLOUD and is_cloud_blocked(attr):
                    front_blocked = 1

        return build_state(
            x=x,
            y=y,
            direction=direction,
            goal_pos=goal_pos,
            front_blocked=front_blocked,
        )

    def act(self, observation, info):
        state = self.extract_state(observation)
        front_blocked = state[5]

        candidate_indices = [0, 1] if front_blocked == 1 else [0, 1, 2]

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
            self.q_table[state] = [0.0, 0.0, 0.0]

    def select_action_index(self, state):
        self.ensure_state(state)

        front_blocked = state[5]
        candidate_indices = [0, 1] if front_blocked == 1 else [0, 1, 2]

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
    static_grid, start_pos, start_dir, goal_pos, cloud_positions = extract_static_layout()
    dist_map = build_distance_map(static_grid, goal_pos)

    start_dist = int(dist_map[start_pos[0], start_pos[1]])
    print("========== BFS CHECK ==========")
    print("start:", start_pos)
    print("goal :", goal_pos)
    print("static distance:", start_dist)
    print("cloud count:", len(cloud_positions))
    print("================================")

    env = FastZeldaCloudEnv(
        static_grid=static_grid,
        start_pos=start_pos,
        start_dir=start_dir,
        goal_pos=goal_pos,
        cloud_positions=cloud_positions,
        max_steps=config.max_steps,
    )

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
            old_dist = int(dist_map[old_pos[0], old_pos[1]])

            if episode <= config.random_episodes:
                front_blocked = state[5]
                candidate_indices = [0, 1] if front_blocked == 1 else [0, 1, 2]
                action_idx = random.choice(candidate_indices)
            else:
                action_idx = agent.select_action_index(state)

            action = ACTIONS[action_idx]
            next_state, _, terminated, truncated, _ = env.step(action)
            agent.ensure_state(next_state)

            new_pos = env.current_pos()
            new_dist = int(dist_map[new_pos[0], new_pos[1]])

            reward = -0.05

            if action_idx in [0, 1]:
                reward -= 0.01

            if action_idx == 2 and new_pos == old_pos:
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


def run(model_path: str = "round2_cloud_fast.pkl"):
    agent = Round2EvalAgent.load(model_path)
    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )


if __name__ == "__main__":
    # train()
    run()