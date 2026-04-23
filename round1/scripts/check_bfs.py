import gymnasium as gym
import kymnasium as kym
import numpy as np
from collections import deque


def build_grid(tiles):
    grid = np.zeros((36, 36), dtype=np.int8)
    goal = None

    for x, y, obj, _ in tiles:
        x, y, obj = int(x), int(y), int(obj)

        if obj in [0, 1, 2, 5, 7, 11, 12, 13]:
            grid[x, y] = 1
        elif obj == 4:
            grid[x, y] = 0
            goal = (x, y)

    return grid, goal


def bfs_next_step(grid, start, goal):
    q = deque([start])
    parent = {start: None}

    while q:
        x, y = q.popleft()

        if (x, y) == goal:
            break

        for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < 36 and 0 <= ny < 36):
                continue
            if grid[nx, ny] == 1:
                continue
            if (nx, ny) in parent:
                continue

            parent[(nx, ny)] = (x, y)
            q.append((nx, ny))

    if goal not in parent:
        return None

    cur = goal
    while parent[cur] != start:
        cur = parent[cur]

    return cur


def direction_from_delta(dx, dy):
    if (dx, dy) == (-1, 0):
        return 0  # left
    if (dx, dy) == (0, -1):
        return 1  # up
    if (dx, dy) == (1, 0):
        return 2  # right
    if (dx, dy) == (0, 1):
        return 3  # down
    raise ValueError((dx, dy))


class BFSAgent(kym.Agent):
    ACTION_TURN_RIGHT = 1   # 실제 env
    ACTION_TURN_LEFT = 2    # 실제 env
    ACTION_FORWARD = 3

    def __init__(self):
        self.debug_count = 0

    def act(self, observation, info):
        x, y, _, direction = map(int, observation["link"])
        grid, goal = build_grid(observation["tiles"])

        nxt = bfs_next_step(grid, (x, y), goal)
        if nxt is None:
            return self.ACTION_TURN_LEFT

        dx = nxt[0] - x
        dy = nxt[1] - y
        target_dir = direction_from_delta(dx, dy)

        if direction == target_dir:
            action = self.ACTION_FORWARD
        else:
            left_steps = (direction - target_dir) % 4
            right_steps = (target_dir - direction) % 4

            if left_steps <= right_steps:
                action = self.ACTION_TURN_LEFT
            else:
                action = self.ACTION_TURN_RIGHT

        if self.debug_count < 20:
            print(
                f"x={x}, y={y}, dir={direction}, goal={goal}, "
                f"nxt={nxt}, target_dir={target_dir}, action={action}"
            )
            self.debug_count += 1

        return action

    @classmethod
    def load(cls, path: str):
        return cls()

    def save(self, path: str):
        pass


def run():
    agent = BFSAgent()

    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )


if __name__ == "__main__":
    run()