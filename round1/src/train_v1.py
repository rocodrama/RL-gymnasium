import pickle
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

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

# 내부 action index:
# 0 -> left
# 1 -> right
# 2 -> forward
ACTIONS = [ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD]

BLOCKING_OBJECT_IDS = {0, 1, 2, 5, 7, 11, 12, 13}
GOAL_OBJECT_ID = 4


@dataclass
class TrainConfig:
    episodes: int = 100000
    max_steps: int = 600
    alpha: float = 0.10
    gamma: float = 0.99
    random_episodes: int = 2000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    model_path: str = "round1_fastenv.pkl"


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


def build_grid_from_tiles(tiles: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    grid = np.zeros((GRID_W, GRID_H), dtype=np.int8)
    goal_pos = None

    for x, y, obj, _ in tiles:
        x = int(x)
        y = int(y)
        obj = int(obj)

        if obj in BLOCKING_OBJECT_IDS:
            grid[x, y] = WALL
        elif obj == GOAL_OBJECT_ID:
            grid[x, y] = GOAL
            goal_pos = (x, y)
        else:
            grid[x, y] = FLOOR

    return grid, goal_pos


def compute_goal_features(x: int, y: int, direction: int, goal_pos: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if goal_pos is None:
        return 0, 0

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


def build_state(x: int, y: int, direction: int, goal_pos: Optional[Tuple[int, int]]) -> Tuple[int, int, int, int, int]:
    goal_fb, goal_lr = compute_goal_features(x, y, direction, goal_pos)
    return (x, y, direction, goal_fb, goal_lr)


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


def extract_static_layout() -> Tuple[np.ndarray, Tuple[int, int], int, Tuple[int, int]]:
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )
    obs, _ = env.reset()
    env.close()

    link_x, link_y, _, direction = map(int, obs["link"])
    grid, goal_pos = build_grid_from_tiles(obs["tiles"])

    if goal_pos is None:
        raise RuntimeError("goal position not found from initial tiles")

    return grid, (link_x, link_y), direction, goal_pos


class FastZeldaEnv:
    def __init__(
        self,
        base_grid: np.ndarray,
        start_pos: Tuple[int, int],
        start_dir: int,
        goal_pos: Tuple[int, int],
        max_steps: int,
    ):
        self.base_grid = base_grid.astype(np.int8, copy=True)
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.goal_pos = goal_pos
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

    def get_front_cell(self) -> int:
        dx, dy = forward_delta(self.direction)
        return extract_cell(self.base_grid, self.x + dx, self.y + dy)

    def get_state(self):
        return build_state(self.x, self.y, self.direction, self.goal_pos)

    def current_pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def step(self, action: int):
        self.steps += 1

        if action == ACTION_TURN_LEFT:
            self.direction = turn_left(self.direction)
        elif action == ACTION_TURN_RIGHT:
            self.direction = turn_right(self.direction)
        elif action == ACTION_FORWARD:
            dx, dy = forward_delta(self.direction)
            nx, ny = self.x + dx, self.y + dy
            if extract_cell(self.base_grid, nx, ny) != WALL:
                self.x, self.y = nx, ny

        terminated = (self.x, self.y) == self.goal_pos
        truncated = self.steps >= self.max_steps

        return self.get_state(), 0.0, terminated, truncated, {}


class Round1EvalAgent(kym.Agent):
    def __init__(self, q_table=None):
        self.q_table = q_table if q_table is not None else {}

    def extract_state_and_front(self, observation):
        x, y, _, direction = map(int, observation["link"])
        grid, goal_pos = build_grid_from_tiles(observation["tiles"])

        dx, dy = forward_delta(direction)
        front_cell = extract_cell(grid, x + dx, y + dy)
        state = build_state(x, y, direction, goal_pos)

        return state, front_cell

    def act(self, observation, info):
        state, front_cell = self.extract_state_and_front(observation)
        candidate_indices = [0, 1] if front_cell == WALL else [0, 1, 2]

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

    def select_action_index(self, state, front_cell):
        self.ensure_state(state)
        candidate_indices = [0, 1] if front_cell == WALL else [0, 1, 2]

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
    base_grid, start_pos, start_dir, goal_pos = extract_static_layout()
    dist_map = build_distance_map(base_grid, goal_pos)

    start_dist = int(dist_map[start_pos[0], start_pos[1]])
    print("========== BFS CHECK ==========")
    print("start:", start_pos)
    print("goal :", goal_pos)
    print("distance:", start_dist)
    print("================================")

    env = FastZeldaEnv(
        base_grid=base_grid,
        start_pos=start_pos,
        start_dir=start_dir,
        goal_pos=goal_pos,
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
            front_cell = env.get_front_cell()

            if episode <= config.random_episodes:
                candidate_indices = [0, 1] if front_cell == WALL else [0, 1, 2]
                action_idx = random.choice(candidate_indices)
            else:
                action_idx = agent.select_action_index(state, front_cell)

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


def run(model_path: str = "round1_fastenv.pkl"):
    agent = Round1EvalAgent.load(model_path)
    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )


if __name__ == "__main__":
    # train()
    run()