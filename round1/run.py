import pickle
from typing import Dict, Tuple

import kymnasium as kym
import numpy as np


GRID_W = 36
GRID_H = 36
WALL = 1
INF_HITS = 10**9

DIR_LEFT = 0
DIR_UP = 1
DIR_RIGHT = 2
DIR_DOWN = 3

ACTION_STOP = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT = 2
ACTION_FORWARD = 3
ACTION_PICKUP = 4
ACTION_DROP = 5
ACTION_ATTACK = 6

IDX_STOP = 0
IDX_LEFT = 1
IDX_RIGHT = 2
IDX_FORWARD = 3
IDX_PICKUP = 4
IDX_DROP = 5
IDX_ATTACK = 6

ACTIONS = [
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_FORWARD,
    ACTION_PICKUP,
    ACTION_DROP,
    ACTION_ATTACK,
]

ACTION_PRIORITY = [
    IDX_PICKUP,
    IDX_DROP,
    IDX_ATTACK,
    IDX_FORWARD,
    IDX_LEFT,
    IDX_RIGHT,
    IDX_STOP,
]

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

CLOUD_OPEN_ATTRS = {0, 6}

FRONT_OPEN = 0
FRONT_WALL = 1
FRONT_CLOUD_BLOCKED = 2
FRONT_SWORD = 3
FRONT_TURTLENACK = 4
FRONT_KEESE = 5
FRONT_MOBLIN = 6
FRONT_ARMOS = 7


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

    for x, y, obj, _ in tiles:
        x = int(x)
        y = int(y)
        obj = int(obj)

        if obj in STATIC_BLOCKING_OBJECT_IDS:
            grid[x, y] = WALL
        else:
            grid[x, y] = 0

        if obj == OBJ_GOAL:
            goal_pos = (x, y)

    if goal_pos is None:
        raise RuntimeError("goal position not found from tiles")

    return grid, goal_pos


def compute_relative_features(
    x: int,
    y: int,
    direction: int,
    target_pos: Tuple[int, int] | None,
) -> Tuple[int, int]:
    if target_pos is None:
        return 0, 0

    fdx, fdy = forward_delta(direction)
    tx = target_pos[0] - x
    ty = target_pos[1] - y

    dot = fdx * tx + fdy * ty
    fb = 1 if dot > 0 else -1 if dot < 0 else 0

    cross = fdx * ty - fdy * tx
    lr = 1 if cross > 0 else -1 if cross < 0 else 0

    return fb, lr


def build_state(
    x: int,
    y: int,
    direction: int,
    sword: int,
    goal_pos: Tuple[int, int],
    front_type: int,
    front_attr: int,
    current_has_sword: int,
    current_tile_sword: int,
    sword_color_num: int,
    current_sword_used: int,
):
    goal_fb, goal_lr = compute_relative_features(x, y, direction, goal_pos)
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
        current_tile_sword,
        sword_color_num,
        current_sword_used,
    )


def required_hits(monster_obj: int, monster_color: int, sword_color: int) -> int:
    if monster_obj == OBJ_KEESE:
        if sword_color == 0:
            return 10
        if sword_color == monster_color:
            return 1
        return 4

    if monster_obj == OBJ_MOBLIN:
        if sword_color == 0:
            return 20
        if sword_color == monster_color:
            return 3
        return 12

    if monster_obj == OBJ_TURTLENACK:
        if sword_color == 0:
            return INF_HITS
        if sword_color == monster_color:
            return 5
        return 20

    if monster_obj == OBJ_ARMOS:
        if sword_color == 0:
            return INF_HITS
        if sword_color != monster_color:
            return INF_HITS
        return 5

    return INF_HITS


def get_candidate_indices(state):
    front_type = state[6]
    front_attr = state[7]
    sword = state[3]
    current_tile_sword = state[9]
    sword_color_num = state[10]
    current_sword_used = state[11]

    if current_tile_sword != 0 and sword != current_tile_sword and (
        sword == 0 or current_sword_used == 1 or sword_color_num < 5
    ):
        return [IDX_PICKUP]

    if (
        sword != 0
        and current_sword_used == 1
        and current_tile_sword == 0
        and front_type == FRONT_SWORD
        and front_attr != sword
    ):
        return [IDX_DROP]

    if sword == 0 and current_tile_sword != 0:
        return [IDX_PICKUP]

    if sword == 0 and current_tile_sword == 0 and front_type == FRONT_SWORD:
        return [IDX_FORWARD]

    candidates = [IDX_LEFT, IDX_RIGHT]

    if front_type == FRONT_CLOUD_BLOCKED:
        candidates.append(IDX_STOP)

    if front_type in (FRONT_OPEN, FRONT_SWORD):
        candidates.append(IDX_FORWARD)

    if current_tile_sword != 0:
        candidates.append(IDX_PICKUP)

    if (
        sword != 0
        and current_sword_used == 1
        and current_tile_sword == 0
        and front_type == FRONT_SWORD
        and front_attr != sword
    ):
        candidates.append(IDX_DROP)

    if sword != 0 and front_type in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
        monster_obj = front_type_to_monster_obj(front_type)
        if required_hits(monster_obj, front_attr, sword) < INF_HITS:
            candidates.append(IDX_ATTACK)

    return candidates


def pick_by_priority(indices):
    for idx in ACTION_PRIORITY:
        if idx in indices:
            return idx
    return indices[0]


class FinalEvalAgent(kym.Agent):
    def __init__(self, q_table=None):
        self.q_table = q_table if q_table is not None else {}
        self.collected_sword_colors: set[int] = set()
        self.initial_sword_positions: set[Tuple[int, int]] = set()
        self.dropped_sword_positions: set[Tuple[int, int]] = set()
        self.current_sword_used = True
        self.start_pos: Tuple[int, int] | None = None
        self.prev_link: Tuple[int, int, int, int] | None = None
        self.last_pos: Tuple[int, int] | None = None
        self.spin_streak = 0

    def sync_episode_memory(self, observation, info):
        x, y, sword, direction = map(int, observation["link"])
        tiles = np.asarray(observation["tiles"])

        reset_by_info = False
        if isinstance(info, dict):
            for key in ("episode_step", "elapsed_steps", "step", "steps"):
                value = info.get(key)
                if isinstance(value, (int, np.integer)) and int(value) == 0:
                    reset_by_info = True
                    break

        if self.start_pos is None:
            self.start_pos = (x, y)
        elif (
            self.prev_link is not None
            and (x, y) == self.start_pos
            and sword == 0
            and (self.prev_link[0], self.prev_link[1]) != self.start_pos
        ):
            reset_by_info = True

        if reset_by_info:
            self.collected_sword_colors.clear()
            self.initial_sword_positions = set()
            self.dropped_sword_positions = set()
            self.current_sword_used = True
            self.last_pos = None
            self.spin_streak = 0

        if self.prev_link is not None:
            prev_sword = self.prev_link[2]
            if prev_sword == 0 and sword != 0:
                self.current_sword_used = False
            elif prev_sword != 0 and sword == 0:
                self.current_sword_used = True
            elif prev_sword != 0 and sword != 0 and prev_sword != sword:
                self.current_sword_used = False
        elif sword != 0:
            self.current_sword_used = False

        if sword == 0:
            self.current_sword_used = True

        if sword != 0:
            self.collected_sword_colors.add(sword)

        sword_positions = set()
        for r in tiles:
            tx, ty, obj, _ = map(int, r)
            if obj == OBJ_SWORD:
                sword_positions.add((tx, ty))

        if not self.initial_sword_positions:
            self.initial_sword_positions = set(sword_positions)
        else:
            for pos in sword_positions:
                if pos not in self.initial_sword_positions:
                    self.dropped_sword_positions.add(pos)

        self.prev_link = (x, y, sword, direction)

    def extract_state(self, observation):
        x, y, sword, direction = map(int, observation["link"])
        tiles = np.asarray(observation["tiles"])
        static_grid, goal_pos = build_static_layout_from_tiles(tiles)

        current_has_sword = 0
        current_tile_sword = 0
        for r in tiles:
            tx, ty, obj, attr = map(int, r)
            if tx == x and ty == y and obj == OBJ_SWORD and (tx, ty) not in self.dropped_sword_positions:
                current_has_sword = 1
                current_tile_sword = attr

        def scan_front_info(test_dir: int) -> Tuple[int, int]:
            dx, dy = forward_delta(test_dir)
            nx, ny = x + dx, y + dy
            if extract_cell(static_grid, nx, ny) == WALL:
                return FRONT_WALL, 0

            front_type = FRONT_OPEN
            front_attr = 0
            for r in tiles:
                tx, ty, obj, attr = map(int, r)
                if tx != nx or ty != ny:
                    continue
                if obj == OBJ_CLOUD and is_cloud_blocked(attr):
                    return FRONT_CLOUD_BLOCKED, attr
                if obj in MONSTER_OBJECT_IDS:
                    return monster_obj_to_front_type(obj), attr
                if obj == OBJ_SWORD and (tx, ty) not in self.dropped_sword_positions:
                    front_type = FRONT_SWORD
                    front_attr = attr
            return front_type, front_attr

        front_type, front_attr = scan_front_info(direction)
        sword_color_num = len(self.collected_sword_colors)
        current_sword_used = 1 if self.current_sword_used else 0

        return build_state(
            x=x,
            y=y,
            direction=direction,
            sword=sword,
            goal_pos=goal_pos,
            front_type=front_type,
            front_attr=front_attr,
            current_has_sword=current_has_sword,
            current_tile_sword=current_tile_sword,
            sword_color_num=sword_color_num,
            current_sword_used=current_sword_used,
        )

    def act(self, observation, info):
        self.sync_episode_memory(observation, info)
        state = self.extract_state(observation)
        candidate_indices = get_candidate_indices(state)
        current_pos = (state[0], state[1])

        if state not in self.q_table:
            action_idx = pick_by_priority(candidate_indices)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values[i] for i in candidate_indices)
            best_indices = [i for i in candidate_indices if q_values[i] == max_q]
            action_idx = pick_by_priority(best_indices)

        if current_pos == self.last_pos and action_idx in (IDX_LEFT, IDX_RIGHT):
            self.spin_streak += 1
        else:
            self.spin_streak = 0

        if self.spin_streak >= 3 and IDX_FORWARD in candidate_indices:
            action_idx = IDX_FORWARD
            self.spin_streak = 0

        if (
            action_idx == IDX_ATTACK
            and state[3] != 0
            and state[6] in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS)
        ):
            monster_obj = front_type_to_monster_obj(state[6])
            if required_hits(monster_obj, state[7], state[3]) < INF_HITS:
                self.current_sword_used = True

        self.last_pos = current_pos
        return ACTIONS[action_idx]
    
    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(q_table=data.get("q_table", {}))


def run(model_path: str = "round1_agent.pkl"):
    agent = FinalEvalAgent.load(model_path)
    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )

if __name__ == "__main__":
    run()

