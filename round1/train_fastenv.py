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

# ===== 실제 환경 액션 ID =====
ACTION_STOP = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT = 2
ACTION_FORWARD = 3
ACTION_PICKUP = 4
ACTION_DROP = 5
ACTION_ATTACK = 6

# ===== Q-테이블 액션 인덱스 =====
IDX_STOP = 0
IDX_LEFT = 1
IDX_RIGHT = 2
IDX_FORWARD = 3
IDX_PICKUP = 4
IDX_DROP = 5
IDX_ATTACK = 6

# Q-테이블 인덱스를 실제 환경 액션으로 매핑
ACTIONS = [
    ACTION_STOP,        # IDX_STOP
    ACTION_TURN_LEFT,   # IDX_LEFT
    ACTION_TURN_RIGHT,  # IDX_RIGHT
    ACTION_FORWARD,     # IDX_FORWARD
    ACTION_PICKUP,      # IDX_PICKUP
    ACTION_DROP,        # IDX_DROP
    ACTION_ATTACK,      # IDX_ATTACK
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

# ===== 핵심 보상 계수 =====
FORWARD_PROGRESS_GAIN = 1.2
ATTACK_MATCH_STEP_BONUS = 0.45
ATTACK_MISMATCH_STEP_BONUS = -0.10
ATTACK_ACTION_COST = 0.05
BARE_HAND_ATTACK_PENALTY = 0.9
SWORD_COLOR_DISCOVERY_BONUS = 60.0
SWORD_COLOR_TIER_BONUS = 14.0
NEW_COLOR_PICKUP_BONUS = 12.0
UNSEEN_SWORD_PROGRESS_GAIN = 1.10
UNSEEN_SWORD_RETREAT_PENALTY = 0.90
FIRST_USE_HIT_REWARD = 2.5
SWAP_BEFORE_USE_PENALTY = 1.5


@dataclass
class TrainConfig:
    """학습 하이퍼파라미터와 보상 계수를 모아둔 설정 클래스."""
    # 학습 기본 파라미터
    episodes: int = 100000
    max_steps: int = 1000
    alpha: float = 0.10
    gamma: float = 0.99
    random_episodes: int = 10000

    # epsilon 스케줄
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9996

    # 탐색 재가열(정체 구간 탈출)
    explore_boost_epsilon: float = 0.30
    explore_boost_after_steps: int = 45
    reheat_period: int = 6000
    reheat_length: int = 1600
    reheat_epsilon: float = 0.22
    stagnation_window: int = 100
    stagnation_color_threshold: float = 3.20
    stagnation_epsilon: float = 0.30
    bootstrap_focus_colors: int = 3
    focus_ramp_success_interval: int = 4000
    max_state_repeat: int = 40

    # 목표 집중/종료 보상
    min_colors_for_goal_focus: int = 5
    early_goal_progress_gain: float = 0.80
    full_focus_goal_progress_penalty: float = 1.05
    goal_reward_base: float = 520.0
    goal_reward_per_color: float = 55.0
    goal_missing_color_penalty: float = 110.0
    non_full_goal_extra_penalty: float = 140.0
    hard_focus_episode: int = 35000
    hard_focus_missing_penalty_scale: float = 1.10
    full_color_terminal_bonus: float = 240.0
    model_path: str = "round1_final.pkl"


def turn_left(direction: int) -> int:
    """현재 방향 기준으로 좌회전한 방향 인덱스를 반환한다."""
    return [DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT][direction]


def turn_right(direction: int) -> int:
    """현재 방향 기준으로 우회전한 방향 인덱스를 반환한다."""
    return [DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT][direction]


def forward_delta(direction: int) -> Tuple[int, int]:
    """방향 인덱스를 전진 벡터 (dx, dy)로 변환한다."""
    return [(-1, 0), (0, -1), (1, 0), (0, 1)][direction]


def extract_cell(grid: np.ndarray, x: int, y: int) -> int:
    """좌표가 맵 밖이면 WALL을, 맵 안이면 해당 셀 값을 반환한다."""
    if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
        return WALL
    return int(grid[x, y])


def is_cloud_blocked(attr: int) -> bool:
    """구름 attribute가 현재 통과 불가 상태인지 판별한다."""
    return attr not in CLOUD_OPEN_ATTRS


def monster_obj_to_front_type(obj: int) -> int:
    """타일 object_id를 전방 타입(FRONT_*)으로 변환한다."""
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
    """전방 타입(FRONT_*)을 몬스터 object_id로 역변환한다."""
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
    """초기 타일에서 정적 지형/목표/검/몬스터 위치를 추출한다."""
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


def compute_relative_features(
    x: int,
    y: int,
    direction: int,
    target_pos: Tuple[int, int] | None,
) -> Tuple[int, int]:
    """타깃의 상대 방향을 전후(goal_fb), 좌우(goal_lr)로 압축한다."""
    if target_pos is None:
        return 0, 0

    fdx, fdy = forward_delta(direction)
    tx = target_pos[0] - x
    ty = target_pos[1] - y

    dot = fdx * tx + fdy * ty
    if dot > 0:
        fb = 1
    elif dot < 0:
        fb = -1
    else:
        fb = 0

    cross = fdx * ty - fdy * tx
    if cross > 0:
        lr = 1
    elif cross < 0:
        lr = -1
    else:
        lr = 0

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
    """Q-테이블 키로 사용하는 상태 튜플을 생성한다."""
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


def build_distance_map(grid: np.ndarray, goal: Tuple[int, int]) -> np.ndarray:
    """목표 지점에서 역방향 BFS로 전체 거리 맵을 계산한다."""
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
    """환경을 1회 리셋해 고정 레이아웃 정보를 추출한다."""
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
    """몬스터 처치 시 무기/속성 일치 여부에 따른 보상을 반환한다."""
    if sword_color == 0:
        return -BARE_HAND_ATTACK_PENALTY
    return 0.6 if sword_color == monster_color else 0.0


def required_hits(monster_obj: int, monster_color: int, sword_color: int) -> int:
    # 필요 타격 수 (맨손 / 색 불일치 / 색 일치)
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
    """현재 상태에서 허용 가능한 행동 후보(마스킹)를 생성한다."""
    front_type = state[6]
    front_attr = state[7]
    sword = state[3]
    current_tile_sword = state[9]
    sword_color_num = state[10]
    current_sword_used = state[11]

    # 현재 칸에 다른 검이 있으면 즉시 줍기 우선.
    # 5색을 모두 모으기 전까지는 첫 사용 전 교체도 허용한다.
    if current_tile_sword != 0 and sword != current_tile_sword and (
        sword == 0 or current_sword_used == 1 or sword_color_num < 5
    ):
        return [IDX_PICKUP]

    # 검 교체 우선 규칙:
    # 1) 손에 검이 있고 앞 칸에 다른 검이 있으면 drop 우선
    if (
        sword != 0
        and current_sword_used == 1
        and current_tile_sword == 0
        and front_type == FRONT_SWORD
        and front_attr != sword
    ):
        return [IDX_DROP]

    # 2) 맨손 + 현재 칸 검 존재 -> pickup 우선
    if sword == 0 and current_tile_sword != 0:
        return [IDX_PICKUP]

    # 3) 맨손 + 앞 칸 검 존재 -> 전진 후 pickup
    if sword == 0 and current_tile_sword == 0 and front_type == FRONT_SWORD:
        return [IDX_FORWARD]

    candidates = [IDX_LEFT, IDX_RIGHT]

    # STOP은 구름 타이밍 대기 상황에서만 허용한다.
    if front_type == FRONT_CLOUD_BLOCKED:
        candidates.append(IDX_STOP)

    if front_type in (FRONT_OPEN, FRONT_SWORD):
        candidates.append(IDX_FORWARD)

    if current_tile_sword != 0:
        candidates.append(IDX_PICKUP)

    # Drop은 교체 셋업일 때만 허용:
    # 손에 검 보유 + 현재 칸 빈 칸 + 앞 칸에 다른 검
    if (
        sword != 0
        and current_sword_used == 1
        and current_tile_sword == 0
        and front_type == FRONT_SWORD
        and front_attr != sword
    ):
        candidates.append(IDX_DROP)

    # 맨손 공격 금지 + 규칙상 불가능한 매치업 공격 차단.
    if sword != 0 and front_type in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
        monster_obj = front_type_to_monster_obj(front_type)
        if required_hits(monster_obj, front_attr, sword) < INF_HITS:
            candidates.append(IDX_ATTACK)

    return candidates


def pick_by_priority(indices):
    """행동 후보 중 고정 우선순위에 따라 최종 행동을 선택한다."""
    for idx in ACTION_PRIORITY:
        if idx in indices:
            return idx
    return indices[0]


class FastZeldaMonsterEnv:
    """학습 가속용 내부 시뮬레이터(FastEnv)."""
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
        self.all_sword_colors = set(self.initial_sword_map.values())
        self.target_sword_color_num = len(self.all_sword_colors)
        self.initial_monster_map = dict(monster_map)
        self.max_steps = max_steps

        self.x = start_pos[0]
        self.y = start_pos[1]
        self.direction = start_dir
        self.sword = 0
        self.current_sword_used = True
        self.steps = 0

        self.sword_map: Dict[Tuple[int, int], int] = {}
        self.monster_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.monster_progress: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.collected_sword_colors: set[int] = set()
        self.dropped_sword_positions: set[Tuple[int, int]] = set()
        self.current_dist_map: np.ndarray | None = None

    def build_current_block_grid(self) -> np.ndarray:
        grid = self.static_grid.copy()
        for (mx, my), (_, monster_color) in self.monster_map.items():
            # 거리 shaping 계산에서는
            # 현재 검과 같은 색 몬스터를 통과 가능 타일로 본다.
            if self.sword != 0 and monster_color == self.sword:
                continue
            grid[mx, my] = WALL
        return grid

    def rebuild_distance_map(self):
        self.current_dist_map = build_distance_map(self.build_current_block_grid(), self.goal_pos)

    def reset(self):
        self.x = self.start_pos[0]
        self.y = self.start_pos[1]
        self.direction = self.start_dir
        self.sword = 0
        self.current_sword_used = True
        self.steps = 0
        self.sword_map = dict(self.initial_sword_map)
        self.monster_map = dict(self.initial_monster_map)
        self.monster_progress = {}
        self.collected_sword_colors = set()
        self.dropped_sword_positions = set()
        self.rebuild_distance_map()
        return self.get_state(), {}

    def current_pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def current_tile_sword_color(self) -> int:
        if (self.x, self.y) in self.dropped_sword_positions:
            return 0
        return self.sword_map.get((self.x, self.y), 0)

    def current_cloud_attr(self) -> int:
        return CLOUD_PATTERN[self.steps % len(CLOUD_PATTERN)]

    def is_cloud_at(self, x: int, y: int) -> bool:
        return (x, y) in self.cloud_positions

    def nearest_unseen_sword_distance(self) -> int:
        best = 10**9
        for (sx, sy), sword_color in self.sword_map.items():
            if (sx, sy) in self.dropped_sword_positions:
                continue
            if sword_color in self.collected_sword_colors:
                continue
            d = abs(sx - self.x) + abs(sy - self.y)
            if d < best:
                best = d
        return -1 if best == 10**9 else best

    def front_info_for_direction(self, direction: int) -> Tuple[int, int]:
        dx, dy = forward_delta(direction)
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

        if (nx, ny) in self.sword_map and (nx, ny) not in self.dropped_sword_positions:
            sword_color = self.sword_map[(nx, ny)]
            return FRONT_SWORD, sword_color

        return FRONT_OPEN, 0

    def front_info(self) -> Tuple[int, int]:
        return self.front_info_for_direction(self.direction)

    def get_state(self):
        front_type, front_attr = self.front_info()
        current_tile_sword = self.current_tile_sword_color()
        current_has_sword = 1 if current_tile_sword != 0 else 0
        sword_color_num = len(self.collected_sword_colors)
        current_sword_used = 1 if self.current_sword_used else 0

        return build_state(
            x=self.x,
            y=self.y,
            direction=self.direction,
            sword=self.sword,
            goal_pos=self.goal_pos,
            front_type=front_type,
            front_attr=front_attr,
            current_has_sword=current_has_sword,
            current_tile_sword=current_tile_sword,
            sword_color_num=sword_color_num,
            current_sword_used=current_sword_used,
        )

    def step(self, action: int):
        self.steps += 1
        reward_bonus = 0.0
        monster_killed = False
        sword_state_changed = False

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
            current_pos = (self.x, self.y)
            current_sword = self.sword_map.get(current_pos)
            if current_pos in self.dropped_sword_positions:
                current_sword = None
            if current_sword is not None:
                is_new_color = current_sword not in self.collected_sword_colors
                if self.sword == 0:
                    self.sword = current_sword
                    self.current_sword_used = False
                    sword_state_changed = True
                    self.sword_map.pop(current_pos, None)
                    reward_bonus += 5.0
                elif self.sword != current_sword:
                    can_force_swap_new_color = (not self.current_sword_used) and is_new_color
                    if not self.current_sword_used and not can_force_swap_new_color:
                        reward_bonus -= SWAP_BEFORE_USE_PENALTY
                    else:
                        old_sword = self.sword
                        self.sword = current_sword
                        self.current_sword_used = False
                        sword_state_changed = True
                        self.sword_map[current_pos] = old_sword
                        reward_bonus += 2.0
                        if can_force_swap_new_color:
                            # 기본은 "사용 후 교체"를 유지하되,
                            # 3색 고착을 막기 위해 새 색 검 교체를 예외 허용한다.
                            reward_bonus -= SWAP_BEFORE_USE_PENALTY * 0.5
                self.collected_sword_colors.add(self.sword)
                if is_new_color:
                    reward_bonus += NEW_COLOR_PICKUP_BONUS

        elif action == ACTION_DROP:
            # Drop은 "앞 칸 다른 검과 교체" 준비 상황에서만 허용한다.
            can_drop_for_swap = (
                self.sword != 0
                and self.current_sword_used
                and (self.x, self.y) not in self.sword_map
                and front_type == FRONT_SWORD
                and front_attr != self.sword
            )
            if can_drop_for_swap:
                self.sword_map[(self.x, self.y)] = self.sword
                self.dropped_sword_positions.add((self.x, self.y))
                self.sword = 0
                self.current_sword_used = True
                sword_state_changed = True
                reward_bonus += 0.5

        elif action == ACTION_ATTACK:
            if self.sword == 0:
                reward_bonus -= BARE_HAND_ATTACK_PENALTY
            elif front_type in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
                monster_obj, monster_color = self.monster_map[(nx, ny)]
                req = required_hits(monster_obj, monster_color, self.sword)

                if req < INF_HITS:
                    last_sword, hit_count = self.monster_progress.get((nx, ny), (-1, 0))

                    if last_sword != self.sword:
                        hit_count = 0

                    hit_count += 1
                    self.monster_progress[(nx, ny)] = (self.sword, hit_count)
                    if not self.current_sword_used:
                        self.current_sword_used = True
                        reward_bonus += FIRST_USE_HIT_REWARD

                    if hit_count >= req:
                        self.monster_map.pop((nx, ny), None)
                        self.monster_progress.pop((nx, ny), None)
                        reward_bonus += get_attack_reward(monster_obj, monster_color, self.sword)
                        monster_killed = True
                else:
                    reward_bonus -= 0.35
            else:
                reward_bonus -= 0.35

        if monster_killed or sword_state_changed:
            self.rebuild_distance_map()

        terminated = (self.x, self.y) == self.goal_pos
        truncated = self.steps >= self.max_steps

        return self.get_state(), reward_bonus, terminated, truncated, {
            "monster_killed": monster_killed
        }

    def is_useful_attack(self, front_type: int, front_attr: int) -> bool:
        if self.sword == 0:
            return False
        if front_type not in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
            return False
        monster_obj = front_type_to_monster_obj(front_type)
        return required_hits(monster_obj, front_attr, self.sword) < INF_HITS


class FinalEvalAgent(kym.Agent):
    """평가/실환경 실행용 에이전트."""
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
        static_grid, goal_pos, _, _, _ = build_static_layout_from_tiles(tiles)

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

        # 제자리 회전이 길어지면 강제로 전진을 선택해 루프를 끊는다.
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

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(q_table=data.get("q_table", {}))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table}, f)


class QLearningAgent:
    """탭형 Q-learning 정책/테이블 관리 클래스."""
    def __init__(self, q_table=None, epsilon: float = 0.1):
        self.q_table = q_table if q_table is not None else {}
        self.epsilon = epsilon

    def ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTIONS)

    def select_action_index(self, state, epsilon: float | None = None):
        self.ensure_state(state)
        candidate_indices = get_candidate_indices(state)
        current_epsilon = self.epsilon if epsilon is None else epsilon

        if random.random() < current_epsilon:
            return random.choice(candidate_indices)

        q_values = self.q_table[state]
        max_q = max(q_values[i] for i in candidate_indices)
        best_indices = [i for i in candidate_indices if q_values[i] == max_q]
        return pick_by_priority(best_indices)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table}, f)


def train(config: TrainConfig = TrainConfig()):
    """FastEnv 기반 학습 루프."""
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
    full_color_success_count = 0
    recent_rewards = []
    recent_success_flags = []
    recent_full_color_flags = []
    recent_max_colors = []
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
        episode_full_color = False
        max_color_num_episode = state[10]
        steps_without_new_color = 0
        active_focus_colors = min(
            config.min_colors_for_goal_focus,
            config.bootstrap_focus_colors + (success_count // config.focus_ramp_success_interval),
        )

        while steps < config.max_steps and not done:
            old_pos = env.current_pos()
            old_dist = int(env.current_dist_map[old_pos[0], old_pos[1]])
            old_unseen_dist = env.nearest_unseen_sword_distance()

            # 1) 행동 선택: 초반 랜덤 탐색 + 이후 epsilon-greedy(정체 시 재가열)
            if episode <= config.random_episodes:
                candidate_indices = get_candidate_indices(state)
                action_idx = random.choice(candidate_indices)
            else:
                missing_colors = env.target_sword_color_num - state[10]
                adaptive_epsilon = agent.epsilon
                if missing_colors > 0 and steps_without_new_color >= config.explore_boost_after_steps:
                    adaptive_epsilon = max(adaptive_epsilon, config.explore_boost_epsilon)
                since_learn = episode - config.random_episodes
                in_reheat = (since_learn % config.reheat_period) < config.reheat_length
                if missing_colors > 0 and in_reheat:
                    adaptive_epsilon = max(adaptive_epsilon, config.reheat_epsilon)
                if len(recent_max_colors) >= config.stagnation_window:
                    window = recent_max_colors[-config.stagnation_window :]
                    avg_recent_colors = sum(window) / len(window)
                    if missing_colors > 0 and avg_recent_colors <= config.stagnation_color_threshold:
                        adaptive_epsilon = max(adaptive_epsilon, config.stagnation_epsilon)
                action_idx = agent.select_action_index(state, epsilon=adaptive_epsilon)

            action = ACTIONS[action_idx]
            front_type = state[6]
            front_attr = state[7]
            current_has_sword = state[8]
            current_tile_sword = state[9]
            sword_color_num = state[10]
            current_sword_used = state[11]

            next_state, env_bonus, terminated, truncated, info = env.step(action)
            agent.ensure_state(next_state)
            next_sword_color_num = next_state[10]

            new_pos = env.current_pos()
            moved = new_pos != old_pos
            new_dist = int(env.current_dist_map[new_pos[0], new_pos[1]])
            new_unseen_dist = env.nearest_unseen_sword_distance()

            reward = -0.08

            # 2) 기본 이동 shaping: 정지/회전/막힌 전진/방문 페널티
            if action_idx == IDX_STOP:
                if front_type == FRONT_CLOUD_BLOCKED:
                    reward += 0.03
                else:
                    reward -= 0.35

            if action_idx in [IDX_LEFT, IDX_RIGHT]:
                reward -= 0.02

            if action_idx == IDX_FORWARD and new_pos == old_pos:
                reward -= 0.25

            if new_pos in visited:
                reward -= 0.05
            else:
                visited.add(new_pos)
                reward += 0.03

            # 3) 목표/검 탐색 shaping: goal 거리 + 미수집 검 거리
            if moved and old_dist >= 0 and new_dist >= 0:
                goal_progress_gain = FORWARD_PROGRESS_GAIN
                if sword_color_num < active_focus_colors:
                    goal_progress_gain = config.early_goal_progress_gain
                reward += goal_progress_gain * (old_dist - new_dist)

                # 5색 집중 단계에서는 모든 색을 모으기 전 goal 직진을 억제한다.
                if (
                    active_focus_colors >= env.target_sword_color_num
                    and sword_color_num < env.target_sword_color_num
                    and old_dist > new_dist
                ):
                    reward -= config.full_focus_goal_progress_penalty * (old_dist - new_dist)

            if moved and old_dist >= 0 and new_dist > old_dist:
                reward -= 0.10

            if (
                moved
                and sword_color_num < env.target_sword_color_num
                and old_unseen_dist >= 0
                and new_unseen_dist >= 0
            ):
                reward += UNSEEN_SWORD_PROGRESS_GAIN * (old_unseen_dist - new_unseen_dist)
                if old_unseen_dist <= 3 and new_unseen_dist > old_unseen_dist:
                    reward -= UNSEEN_SWORD_RETREAT_PENALTY

            # 4) 상호작용 shaping: 검 교체/공격 규칙 페널티-보너스
            drop_swap_ready = (
                state[3] != 0
                and current_sword_used == 1
                and current_tile_sword == 0
                and front_type == FRONT_SWORD
                and front_attr != state[3]
            )

            if front_type == FRONT_SWORD and action_idx not in (IDX_FORWARD, IDX_DROP):
                reward -= 0.25

            if current_has_sword == 1 and action_idx != IDX_PICKUP:
                reward -= 0.60

            if action_idx == IDX_PICKUP and current_has_sword == 0:
                reward -= 0.20

            if action_idx == IDX_PICKUP and current_tile_sword != 0:
                reward += 0.70

            if action_idx == IDX_DROP:
                if drop_swap_ready:
                    reward += 0.25
                else:
                    reward -= 0.30

            if action_idx == IDX_ATTACK:
                reward -= ATTACK_ACTION_COST
                if state[3] == 0:
                    reward -= BARE_HAND_ATTACK_PENALTY
                elif front_type not in (FRONT_TURTLENACK, FRONT_KEESE, FRONT_MOBLIN, FRONT_ARMOS):
                    reward -= 0.30
                elif not env.is_useful_attack(front_type, front_attr):
                    reward -= 0.30
                elif state[3] == front_attr:
                    reward += ATTACK_MATCH_STEP_BONUS
                else:
                    reward += ATTACK_MISMATCH_STEP_BONUS

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
            if next_sword_color_num > sword_color_num:
                discovery_reward = 0.0
                for color_rank in range(sword_color_num + 1, next_sword_color_num + 1):
                    discovery_reward += SWORD_COLOR_DISCOVERY_BONUS + SWORD_COLOR_TIER_BONUS * color_rank
                reward += discovery_reward
                steps_without_new_color = 0
            else:
                steps_without_new_color += 1

            if next_sword_color_num > max_color_num_episode:
                max_color_num_episode = next_sword_color_num

            state_visit_count[next_state] = state_visit_count.get(next_state, 0) + 1
            if state_visit_count[next_state] >= config.max_state_repeat:
                truncated = True

            done = terminated or truncated

            if terminated:
                # 5) 종료 보상: 성공 + 색 수집 반영
                missing_for_focus = max(0, active_focus_colors - next_sword_color_num)
                missing_penalty_scale = 1.0
                if episode >= config.hard_focus_episode:
                    missing_penalty_scale = config.hard_focus_missing_penalty_scale
                reward += (
                    config.goal_reward_base
                    + config.goal_reward_per_color * next_sword_color_num
                    - config.goal_missing_color_penalty * missing_penalty_scale * missing_for_focus
                )
                missing_to_target = max(0, env.target_sword_color_num - next_sword_color_num)
                if missing_to_target > 0 and active_focus_colors >= env.target_sword_color_num:
                    reward -= config.non_full_goal_extra_penalty * missing_to_target
                if next_sword_color_num >= env.target_sword_color_num:
                    reward += config.full_color_terminal_bonus
                    full_color_success_count += 1
                    episode_full_color = True
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

        recent_full_color_flags.append(1 if episode_full_color else 0)
        if len(recent_full_color_flags) > 100:
            recent_full_color_flags.pop(0)

        recent_max_colors.append(max_color_num_episode)
        if len(recent_max_colors) > 100:
            recent_max_colors.pop(0)

        if episode % 500 == 0 or episode <= 3:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            recent_success_rate = sum(recent_success_flags) / len(recent_success_flags)
            recent_full_color_rate = sum(recent_full_color_flags) / len(recent_full_color_flags)
            avg_max_colors = sum(recent_max_colors) / len(recent_max_colors)
            phase = "random" if episode <= config.random_episodes else "learn"

            print(
                f"[Episode {episode:6d}] "
                f"phase={phase}, "
                f"epsilon={agent.epsilon:.4f}, "
                f"avg_reward={avg_reward:8.2f}, "
                f"success_count={success_count}, "
                f"full_color_success_count={full_color_success_count}, "
                f"best_dist_ever={best_dist_ever if best_dist_ever < 10**9 else 'inf'}, "
                f"recent_success_rate={recent_success_rate:.2f}, "
                f"recent_full_color_rate={recent_full_color_rate:.2f}, "
                f"avg_max_colors={avg_max_colors:.2f}, "
                f"active_focus_colors={active_focus_colors}"
            )

    agent.epsilon = 0.0
    agent.save(config.model_path)
    print(f"training done: {config.model_path}")


def run(model_path: str = "round1_final.pkl"):
    """학습된 모델을 로드해 공식 평가 환경으로 실행한다."""
    agent = FinalEvalAgent.load(model_path)
    kym.evaluate(
        env_id="kymnasium/ZeldaAdventure-Stage-3",
        agent=agent,
        bgm=True,
    )


if __name__ == "__main__":
    # train()
    run()  


