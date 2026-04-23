import pickle
from collections import deque
from typing import Any, Dict, Tuple, Optional

import gymnasium as gym
import kymnasium as kym
import numpy as np


class Round1EvalAgent(kym.Agent):
    FLOOR = 0
    WALL = 1
    GOAL = 2

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

    DIR_LEFT = 0
    DIR_UP = 1
    DIR_RIGHT = 2
    DIR_DOWN = 3

    ACTION_TURN_LEFT = 2
    ACTION_TURN_RIGHT = 1
    ACTION_FORWARD = 3

    ACTIONS = [ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_FORWARD]

    GRID_W = 36
    GRID_H = 36

    def __init__(self, q_table=None, epsilon: float = 0.0):
        self.q_table = q_table if q_table is not None else {}
        self.epsilon = epsilon

    def turn_left(self, direction: int) -> int:
        return [self.DIR_DOWN, self.DIR_LEFT, self.DIR_UP, self.DIR_RIGHT][direction]

    def turn_right(self, direction: int) -> int:
        return [self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT][direction]

    def forward_delta(self, direction: int) -> Tuple[int, int]:
        return [(-1, 0), (0, -1), (1, 0), (0, 1)][direction]

    def build_grid_from_tiles(self, tiles: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        grid = np.zeros((self.GRID_W, self.GRID_H), dtype=np.int8)
        goal_pos = None

        for x, y, obj, attr in tiles:
            x = int(x)
            y = int(y)
            obj = int(obj)

            if obj in [
                self.OBJ_WALL,
                self.OBJ_POND,
                self.OBJ_FIRE,
                self.OBJ_CLOUD,
                self.OBJ_TURTLENACK,
                self.OBJ_KEESE,
                self.OBJ_MOBLIN,
                self.OBJ_ARMOS,
            ]:
                grid[x, y] = self.WALL
            elif obj == self.OBJ_GOAL:
                grid[x, y] = self.GOAL
                goal_pos = (x, y)
            else:
                grid[x, y] = self.FLOOR

        return grid, goal_pos

    def extract_cell(self, grid: np.ndarray, x: int, y: int) -> int:
        if x < 0 or x >= self.GRID_W or y < 0 or y >= self.GRID_H:
            return self.WALL
        return int(grid[x, y])

    def extract_state(self, observation: Any):
        x, y, sword, direction = map(int, observation["link"])
        grid, goal_pos = self.build_grid_from_tiles(observation["tiles"])

        fdx, fdy = self.forward_delta(direction)
        ldx, ldy = self.forward_delta(self.turn_left(direction))
        rdx, rdy = self.forward_delta(self.turn_right(direction))

        front_cell = self.extract_cell(grid, x + fdx, y + fdy)
        left_cell = self.extract_cell(grid, x + ldx, y + ldy)
        right_cell = self.extract_cell(grid, x + rdx, y + rdy)

        if goal_pos is None:
            goal_fb = 0
            goal_lr = 0
        else:
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

        return (
            x,
            y,
            direction,
            front_cell,
            left_cell,
            right_cell,
            goal_fb,
            goal_lr,
        )

    def act(self, observation: Any, info: Dict):
        state = self.extract_state(observation)

        front_cell = state[3]
        candidate_indices = [0, 1] if front_cell == self.WALL else [0, 1, 2]

        if state not in self.q_table:
            if front_cell == self.WALL:
                return self.ACTION_TURN_LEFT
            return self.ACTION_FORWARD

        q_values = self.q_table[state]
        max_q = max(q_values[i] for i in candidate_indices)
        best_indices = [i for i in candidate_indices if q_values[i] == max_q]
        action_idx = best_indices[0]  # 동점 랜덤 금지

        return self.ACTIONS[action_idx]

    @classmethod
    def load(cls, path: str) -> "kym.Agent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(q_table=data.get("q_table", {}), epsilon=0.0)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table}, f)


def extract_static_layout():
    env = gym.make(
        id="kymnasium/ZeldaAdventure-Stage-3",
        render_mode=None,
        bgm=False,
    )

    obs, info = env.reset()
    env.close()

    link_x, link_y, sword, direction = map(int, obs["link"])

    grid = np.zeros((36, 36), dtype=np.int8)
    goal_pos = None

    for x, y, obj, attr in obs["tiles"]:
        x = int(x)
        y = int(y)
        obj = int(obj)

        if obj in [0, 1, 2, 5, 7, 11, 12, 13]:
            grid[x, y] = 1
        elif obj == 4:
            grid[x, y] = 2
            goal_pos = (x, y)
        else:
            grid[x, y] = 0

    if goal_pos is None:
        raise RuntimeError("goal position not found from initial tiles")

    return grid, (link_x, link_y), direction, goal_pos


def turn_left(direction: int) -> int:
    return [3, 0, 1, 2][direction]


def turn_right(direction: int) -> int:
    return [1, 2, 3, 0][direction]


def forward_delta(direction: int) -> Tuple[int, int]:
    return [(-1, 0), (0, -1), (1, 0), (0, 1)][direction]


def direction_from_delta(dx: int, dy: int) -> int:
    if (dx, dy) == (-1, 0):
        return 0
    if (dx, dy) == (0, -1):
        return 1
    if (dx, dy) == (1, 0):
        return 2
    if (dx, dy) == (0, 1):
        return 3
    raise ValueError(f"invalid delta: {(dx, dy)}")


def extract_cell(grid: np.ndarray, x: int, y: int) -> int:
    if x < 0 or x >= 36 or y < 0 or y >= 36:
        return 1
    return int(grid[x, y])


def build_state(
    grid: np.ndarray,
    goal_pos: Tuple[int, int],
    x: int,
    y: int,
    direction: int,
):
    fdx, fdy = forward_delta(direction)
    ldx, ldy = forward_delta(turn_left(direction))
    rdx, rdy = forward_delta(turn_right(direction))

    front_cell = extract_cell(grid, x + fdx, y + fdy)
    left_cell = extract_cell(grid, x + ldx, y + ldy)
    right_cell = extract_cell(grid, x + rdx, y + rdy)

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

    return (
        x,
        y,
        direction,
        front_cell,
        left_cell,
        right_cell,
        goal_fb,
        goal_lr,
    )


def bfs_distance_map(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    w, h = grid.shape
    dist = np.full((w, h), -1, dtype=np.int32)

    q = deque([goal])
    dist[goal[0], goal[1]] = 0

    while q:
        x, y = q.popleft()

        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            nx, ny = x + dx, y + dy

            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if dist[nx, ny] != -1:
                continue
            if grid[nx, ny] == 1:
                continue

            dist[nx, ny] = dist[x, y] + 1
            q.append((nx, ny))

    return dist


def best_move_direction(dist_map: np.ndarray, x: int, y: int) -> Optional[int]:
    cur = dist_map[x, y]
    if cur <= 0:
        return None

    best_dir = None
    best_dist = cur

    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        nx, ny = x + dx, y + dy

        if nx < 0 or nx >= dist_map.shape[0] or ny < 0 or ny >= dist_map.shape[1]:
            continue

        nd = dist_map[nx, ny]
        if nd >= 0 and nd < best_dist:
            best_dist = nd
            best_dir = direction_from_delta(dx, dy)

    return best_dir


def best_turn_action(cur_dir: int, target_dir: int) -> int:
    left_steps = (cur_dir - target_dir) % 4
    right_steps = (target_dir - cur_dir) % 4

    if left_steps < right_steps:
        return 0   # left
    elif right_steps < left_steps:
        return 1   # right
    else:
        return 0   # 정확히 반대면 항상 left 고정


def build_q_table_from_distance_map(
    grid: np.ndarray,
    goal_pos: Tuple[int, int],
):
    dist_map = bfs_distance_map(grid, goal_pos)
    q_table = {}

    reachable = int(np.sum(dist_map >= 0))
    print("reachable cells:", reachable)

    for x in range(36):
        for y in range(36):
            if grid[x, y] == 1:
                continue
            if dist_map[x, y] < 0:
                continue

            target_dir = best_move_direction(dist_map, x, y)

            for direction in [0, 1, 2, 3]:
                state = build_state(grid, goal_pos, x, y, direction)
                front_cell = state[3]

                q = [-10.0, -10.0, -10.0]

                if (x, y) == goal_pos:
                    q = [0.0, 0.0, 0.0]

                elif target_dir is None:
                    # 예외 상태: 일단 좌회전 고정
                    q[0] = 1.0
                    q[1] = 0.0
                    q[2] = -10.0 if front_cell == 1 else -1.0

                else:
                    if direction == target_dir:
                        # 방향이 맞으면 전진 최우선
                        q[0] = 0.0
                        q[1] = 0.0
                        q[2] = 10.0 if front_cell != 1 else -10.0
                    else:
                        turn_action = best_turn_action(direction, target_dir)

                        if turn_action == 0:  # left
                            q[0] = 10.0
                            q[1] = 0.0
                            q[2] = -10.0 if front_cell == 1 else -1.0
                        else:                 # right
                            q[0] = 0.0
                            q[1] = 10.0
                            q[2] = -10.0 if front_cell == 1 else -1.0

                q_table[state] = q

    return q_table, dist_map


def train():
    grid, start_pos, start_dir, goal_pos = extract_static_layout()

    dist_map = bfs_distance_map(grid, goal_pos)
    start_dist = dist_map[start_pos[0], start_pos[1]]
    if start_dist < 0:
        raise RuntimeError("start is not reachable to goal in simplified grid")

    print("start:", start_pos)
    print("goal :", goal_pos)
    print("BFS shortest distance:", int(start_dist))

    q_table, dist_map = build_q_table_from_distance_map(
        grid=grid,
        goal_pos=goal_pos,
    )

    with open("round1_fastenv.pkl", "wb") as f:
        pickle.dump({"q_table": q_table}, f)

    print("Q-table states :", len(q_table))
    print("저장 완료: round1_fastenv_bfs.pkl")


def run():
    agent = Round1EvalAgent.load("round1_fastenv_bfs.pkl")

    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )


if __name__ == "__main__":
    # train()
    run()