"""
2026-01 KNU RL Competition Round 2 Avoid Blurp용 병렬 DQN 학습 코드 v3 입니다.

v3 변경점(핵심):
- 생존이 최우선이므로 마리오와 Blurp의 bounding box가 닿지 않는 방향으로 보상을 재설계했습니다.
- 관측값은 절대 좌표보다 마리오 기준 상대 위치, 박스 간격, 충돌 위험도를 더 잘 보도록 변환합니다.
- Blurp는 "마리오와 곧 겹칠 가능성"이 큰 순서로 정렬합니다.
- Dueling Double DQN 구조는 유지합니다.

프로젝트 루트에서 실행:
    python src/train_v3.py

학습 완료 후 사용:
    agent = YourAgent.load("avoid_blurp_dqn_v3.pt")
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Deque, Dict, Tuple

import gymnasium as gym
import kymnasium as kym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========
# 환경 설정
# =========

ENV_ID = "kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1"
MODEL_PATH = Path("avoid_blurp_dqn_v3.pt")

OBS_DIM = 5 + 30 * 6
ACTION_DIM = 3
TRAIN_RENDER_MODE = None
TRAIN_BGM = False
# 512개 이상은 OS의 open files 제한(ulimit -n)에 걸릴 수 있습니다.
# 256은 현재 서버에서 속도와 안정성 균형이 좋은 기본값입니다.
NUM_ENVS = 256
USE_ASYNC_VECTOR_ENV = True
REQUIRE_CUDA_FOR_TRAINING = True
USE_DUMMY_SDL_FOR_TRAINING = True


# ===============
# 하이퍼파라미터
# ===============

SEED = 42
TOTAL_ENV_STEPS = 3_000_000

LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 2048
REPLAY_BUFFER_SIZE = 500_000
TRAIN_START_SIZE = 20_000
MIN_REPLAY_SIZE = 100_000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_HOLD_AFTER_WARMUP_STEPS = 100_000
EPSILON_DECAY_STEPS = 1_500_000

HIDDEN_DIM = 256
UPDATE_STEPS_PER_VECTOR_STEP = 2
TARGET_UPDATE_EVERY_ENV_STEPS = 20_000
GRAD_CLIP_NORM = 10.0

LOG_EVERY_ENV_STEPS = 5_000
SAVE_EVERY_ENV_STEPS = 250_000
RECENT_EPISODE_WINDOW = 100


# ========================
# 관측값 정규화 기준값
# ========================

X_SCALE = 640.0
Y_SCALE = 480.0
VELOCITY_SCALE = 20.0
ACCELERATION_SCALE = 10.0
FEATURE_CLIP_VALUE = 5.0


# ======================
# 보상 설계 설정값
# ======================

# 생존이 최우선 목표이므로 reward의 중심은 "오래 살아남기"와 "박스가 절대 겹치지 않기"입니다.
# 마리오 box와 Blurp box의 수평 간격, 수직 간격, 예상 충돌 시간을 직접 계산해 보상에 반영합니다.
SURVIVAL_REWARD = 0.2
TIME_DELTA_REWARD_SCALE = 8.0
NEAR_BOX_PENALTY_WEIGHT = 0.25
COLLISION_COURSE_PENALTY_WEIGHT = 1.25
DANGER_RADIUS = 170.0
HORIZONTAL_DANGER_MARGIN = 55.0
VERTICAL_DANGER_MARGIN = 120.0
MAX_DANGER_PENALTY = 2.5
COLLISION_PENALTY = -60.0
SUCCESS_REWARD = 500.0


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def log(message: str) -> None:
    print(message, flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")

    log(f"Python executable: {os.sys.executable}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"PyTorch CUDA build: {torch.version.cuda}")
    log(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
    log(f"CUDA available: {cuda_available}")
    log(f"CUDA device count: {torch.cuda.device_count()}")
    log(f"PyTorch device: {device} ({device_name})")

    if cuda_available:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        log(
            "CUDA가 보이지 않습니다. CUDA_VISIBLE_DEVICES 값이 실제 GPU 번호와 맞는지, "
            "`nvidia-smi -L`과 `python scripts/check_cuda.py`로 확인하세요."
        )


def require_cuda_if_needed(device: torch.device) -> None:
    if REQUIRE_CUDA_FOR_TRAINING and device.type != "cuda":
        raise RuntimeError(
            "학습은 GPU 사용을 전제로 설정되어 있지만 PyTorch가 CUDA를 감지하지 못했습니다. "
            "`CUDA_VISIBLE_DEVICES=2`가 올바른 물리 GPU 번호인지 확인하거나, "
            "GPU 없이 테스트만 할 경우 REQUIRE_CUDA_FOR_TRAINING=False로 바꾸세요."
        )


def configure_headless_sdl() -> None:
    if not USE_DUMMY_SDL_FOR_TRAINING:
        return

    # CUDA_VISIBLE_DEVICES는 CUDA 연산에만 영향을 줍니다.
    # pygame/SDL이 Xorg 또는 OpenGL 컨텍스트를 만들면 디스플레이가 붙은 GPU 0번에
    # 작은 Type G 프로세스가 생길 수 있으므로, 학습용 env worker에서는 dummy driver를 사용합니다.
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def make_env(rank: int = 0) -> gym.Env:
    configure_headless_sdl()
    env = gym.make(
        id=ENV_ID,
        render_mode=TRAIN_RENDER_MODE,
        bgm=TRAIN_BGM,
    )
    env.reset(seed=SEED + rank)
    return env


def make_vector_env(num_envs: int = NUM_ENVS) -> gym.vector.VectorEnv:
    env_fns = [partial(make_env, rank) for rank in range(num_envs)]
    vector_cls = gym.vector.AsyncVectorEnv if USE_ASYNC_VECTOR_ENV else gym.vector.SyncVectorEnv
    return vector_cls(env_fns)


def _extract_mario_blurps(observations: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    mario_raw = observations.get("mario", np.zeros(5, dtype=np.float32))
    blurps_raw = observations.get("blurps", np.zeros((30, 6), dtype=np.float32))

    mario = np.asarray(mario_raw, dtype=np.float32)
    blurps = np.asarray(blurps_raw, dtype=np.float32)

    if mario.ndim <= 1:
        batch_size = 1
        mario_flat = mario.reshape(1, -1)
    else:
        batch_size = mario.shape[0]
        mario_flat = mario.reshape(batch_size, -1)

    fixed_mario = np.zeros((batch_size, 5), dtype=np.float32)
    mario_copy_size = min(mario_flat.shape[1], fixed_mario.shape[1])
    fixed_mario[:, :mario_copy_size] = mario_flat[:, :mario_copy_size]

    if blurps.ndim == 2 and batch_size > 1 and blurps.shape[0] == batch_size:
        blurps_flat = blurps.reshape(batch_size, -1)
    elif blurps.ndim <= 2:
        blurps_flat = blurps.reshape(1, -1)
        if batch_size > 1:
            repeated = np.zeros((batch_size, blurps_flat.shape[1]), dtype=np.float32)
            repeated[0] = blurps_flat[0]
            blurps_flat = repeated
    else:
        blurps_flat = blurps.reshape(batch_size, -1)

    fixed_blurps_flat = np.zeros((batch_size, 30 * 6), dtype=np.float32)
    blurps_copy_size = min(blurps_flat.shape[1], fixed_blurps_flat.shape[1])
    fixed_blurps_flat[:, :blurps_copy_size] = blurps_flat[:, :blurps_copy_size]
    fixed_blurps = fixed_blurps_flat.reshape(batch_size, 30, 6)
    fixed_blurps = sort_blurps_by_threat(fixed_mario, fixed_blurps)

    return fixed_mario, fixed_blurps


def box_geometry(mario: np.ndarray, blurps: np.ndarray) -> Dict[str, np.ndarray]:
    # 마리오와 Blurp의 AABB(axis-aligned bounding box) 관계를 한 번에 계산합니다.
    # 생존은 결국 "두 박스가 겹치지 않는 것"이므로, x/y 중심보다 박스 간격이 더 직접적인 정보입니다.
    visible = ~np.all(np.isclose(blurps, 0.0), axis=2)

    mario_left = mario[:, 0:1]
    mario_top = mario[:, 1:2]
    mario_right = mario[:, 2:3]
    mario_bottom = mario[:, 3:4]
    mario_x = (mario_left + mario_right) * 0.5
    mario_y = (mario_top + mario_bottom) * 0.5
    mario_width = np.maximum(mario_right - mario_left, 1.0)
    mario_height = np.maximum(mario_bottom - mario_top, 1.0)

    blurp_left = blurps[:, :, 0]
    blurp_top = blurps[:, :, 1]
    blurp_right = blurps[:, :, 2]
    blurp_bottom = blurps[:, :, 3]
    blurp_velocity = blurps[:, :, 4]
    blurp_speed = np.abs(blurp_velocity)

    blurp_x = (blurp_left + blurp_right) * 0.5
    blurp_y = (blurp_top + blurp_bottom) * 0.5
    blurp_width = np.maximum(blurp_right - blurp_left, 1.0)
    blurp_height = np.maximum(blurp_bottom - blurp_top, 1.0)

    signed_dx = blurp_x - mario_x
    signed_dy = blurp_y - mario_y

    overlap_width = np.minimum(mario_right, blurp_right) - np.maximum(mario_left, blurp_left)
    overlap_height = np.minimum(mario_bottom, blurp_bottom) - np.maximum(mario_top, blurp_top)

    horizontal_gap = np.maximum(-overlap_width, 0.0)
    vertical_gap = np.maximum(-overlap_height, 0.0)
    box_distance = np.hypot(horizontal_gap, vertical_gap)

    x_overlap_ratio = np.clip(overlap_width / np.minimum(mario_width, blurp_width), 0.0, 1.0)
    y_overlap_ratio = np.clip(overlap_height / np.minimum(mario_height, blurp_height), 0.0, 1.0)
    current_overlap = visible & (overlap_width > 0.0) & (overlap_height > 0.0)

    # signed_dy = blurp_y - mario_y 입니다. y축은 아래가 + 방향이라고 보고,
    # signed_dy < 0인 Blurp가 velocity > 0이면 위에서 아래로 접근 중입니다.
    # signed_dy > 0인 Blurp가 velocity < 0이면 아래에서 위로 접근 중입니다.
    # 반대로 마리오 아래에 있는데 더 아래로 내려가는 Blurp는 멀어지는 중이라 위험도를 낮춥니다.
    approaching_vertical = (signed_dy * blurp_velocity) < 0.0
    vertical_approach_speed = np.maximum(-signed_dy * blurp_velocity / (np.abs(signed_dy) + 1.0), 0.0)
    directional_gap = np.where(
        approaching_vertical,
        np.maximum(np.abs(signed_dy) - (mario_height + blurp_height) * 0.5, 0.0),
        VERTICAL_DANGER_MARGIN,
    )
    lane_score = np.where(
        overlap_width > 0.0,
        1.0,
        1.0 - np.clip(horizontal_gap / HORIZONTAL_DANGER_MARGIN, 0.0, 1.0),
    )
    approach_score = np.where(
        approaching_vertical,
        1.0 / (1.0 + directional_gap / (vertical_approach_speed + 1.0)),
        0.0,
    )
    near_box_score = 1.0 / (1.0 + box_distance / DANGER_RADIUS)
    same_lane = (overlap_width > 0.0) | (horizontal_gap < HORIZONTAL_DANGER_MARGIN)
    collision_course = visible.astype(np.float32) * same_lane.astype(np.float32) * lane_score * approach_score
    touch_risk = visible.astype(np.float32) * (
        3.0 * current_overlap.astype(np.float32)
        + 2.0 * collision_course
        + near_box_score
    )

    return {
        "visible": visible,
        "mario_x": mario_x,
        "mario_y": mario_y,
        "mario_width": mario_width,
        "mario_height": mario_height,
        "blurp_x": blurp_x,
        "blurp_y": blurp_y,
        "blurp_width": blurp_width,
        "blurp_height": blurp_height,
        "signed_dx": signed_dx,
        "signed_dy": signed_dy,
        "horizontal_gap": horizontal_gap,
        "vertical_gap": vertical_gap,
        "box_distance": box_distance,
        "x_overlap_ratio": x_overlap_ratio,
        "y_overlap_ratio": y_overlap_ratio,
        "near_box_score": near_box_score,
        "collision_course": collision_course,
        "approach_score": approach_score,
        "approaching_vertical": approaching_vertical.astype(np.float32),
        "touch_risk": touch_risk,
    }


def sort_blurps_by_threat(mario: np.ndarray, blurps: np.ndarray) -> np.ndarray:
    # observation 차원은 그대로 유지하되, 30개 Blurp 행의 순서를 위험도 기준으로 정렬합니다.
    # 가장 위험한 Blurp가 항상 앞쪽 feature에 오면 MLP가 "무엇을 먼저 피해야 하는지" 배우기 쉬워집니다.
    geometry = box_geometry(mario, blurps)
    threat = geometry["touch_risk"]
    order = np.argsort(-threat, axis=1)
    batch_indices = np.arange(blurps.shape[0])[:, None]
    return blurps[batch_indices, order]


def preprocess_observations(observations: Dict[str, Any]) -> np.ndarray:
    """벡터 환경의 dict 관측값을 마리오 기준 geometry feature로 변환합니다."""
    mario, blurps = _extract_mario_blurps(observations)
    geometry = box_geometry(mario, blurps)

    mario_features = np.concatenate(
        [
            geometry["mario_x"] / X_SCALE,
            geometry["mario_y"] / Y_SCALE,
            geometry["mario_width"] / X_SCALE,
            geometry["mario_height"] / Y_SCALE,
            mario[:, 4:5] / VELOCITY_SCALE,
        ],
        axis=1,
    ).astype(np.float32)

    # 각 Blurp는 원래 좌표 대신 "마리오 기준 상대 위치와 박스 간격"으로 표현합니다.
    # 마지막 feature는 방향성까지 반영한 touch 위험도입니다.
    # 아래에 있지만 아래로 내려가는 Blurp처럼 멀어지는 물체는 위험도가 낮게 들어갑니다.
    # [상대 x, 상대 y, x축 박스 간격, y축 박스 간격, y속도, 방향성 touch 위험도]
    blurps_features = np.stack(
        [
            geometry["signed_dx"] / X_SCALE,
            geometry["signed_dy"] / Y_SCALE,
            geometry["horizontal_gap"] / X_SCALE,
            geometry["vertical_gap"] / Y_SCALE,
            blurps[:, :, 4] / VELOCITY_SCALE,
            np.clip(geometry["touch_risk"], 0.0, 5.0) / 5.0,
        ],
        axis=2,
    ).astype(np.float32)

    features = np.concatenate([mario_features, blurps_features.reshape(len(mario), -1)], axis=1)
    return np.clip(features, -FEATURE_CLIP_VALUE, FEATURE_CLIP_VALUE).astype(np.float32)


def preprocess_observation(observation: Dict[str, Any]) -> np.ndarray:
    """단일 dict 관측값을 정규화된 고정 길이 벡터 하나로 변환합니다."""
    return preprocess_observations(observation)[0]


def _vector_info_array(
    infos: Dict[str, Any],
    key: str,
    num_envs: int,
    default: np.ndarray | float,
) -> np.ndarray:
    if isinstance(default, np.ndarray):
        result = default.astype(np.float32, copy=True)
    else:
        result = np.full(num_envs, float(default), dtype=np.float32)

    if key not in infos:
        return result

    value = infos[key]
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return result

    if array.shape == ():
        result.fill(float(array.item()))
    elif array.shape[0] == num_envs:
        result[:] = array[:num_envs]

    return result


def _final_info_time(infos: Dict[str, Any], env_index: int, fallback: float) -> float:
    for key in ("final_info", "terminal_info"):
        if key not in infos:
            continue

        final_infos = infos[key]
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask)[env_index]):
            continue

        try:
            candidate = final_infos[env_index]
        except (TypeError, IndexError, KeyError):
            continue

        if isinstance(candidate, dict) and "time_elapsed" in candidate:
            return float(candidate.get("time_elapsed", fallback) or fallback)

    return float(fallback)


def close_blurp_penalties(observations: Dict[str, Any]) -> np.ndarray:
    mario, blurps = _extract_mario_blurps(observations)
    geometry = box_geometry(mario, blurps)
    visible = geometry["visible"]

    # 1. 박스 간 거리가 가까울수록 약한 패널티를 줍니다.
    #    이 값은 생존 보상을 압도하지 않고, "여유 공간이 있는 쪽으로 움직이라"는 힌트 역할만 합니다.
    near_box_penalties = NEAR_BOX_PENALTY_WEIGHT * geometry["near_box_score"]

    # 2. Blurp가 마리오 위쪽/근처에서 x축 box 라인에 들어오면 더 강한 패널티를 줍니다.
    #    즉, 단순히 가까운 물체가 아니라 실제로 닿을 가능성이 높은 물체를 피하도록 유도합니다.
    course_penalties = COLLISION_COURSE_PENALTY_WEIGHT * geometry["collision_course"]

    # 3. 이미 박스가 겹친 상황은 보통 truncated=True로 끝나지만, 비정상/경계 상황에서도 강하게 밀어냅니다.
    overlap_penalties = 2.0 * geometry["x_overlap_ratio"] * geometry["y_overlap_ratio"]

    penalties = np.where(visible, near_box_penalties + course_penalties + overlap_penalties, 0.0).sum(axis=1)

    return np.minimum(penalties, MAX_DANGER_PENALTY).astype(np.float32)


def shape_rewards(
    next_observations: Dict[str, Any],
    previous_times: np.ndarray,
    next_times: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
) -> np.ndarray:
    time_delta = np.maximum(0.0, next_times - previous_times)
    rewards = np.full(len(next_times), SURVIVAL_REWARD, dtype=np.float32)
    rewards += TIME_DELTA_REWARD_SCALE * time_delta
    rewards -= close_blurp_penalties(next_observations)
    rewards += np.where(truncations & ~terminations, COLLISION_PENALTY, 0.0).astype(np.float32)
    rewards += np.where(terminations, SUCCESS_REWARD, 0.0).astype(np.float32)
    return rewards.astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        state_batch = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        action_batch = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        next_state_batch = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        done_batch = torch.as_tensor(dones, dtype=torch.float32, device=device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_head(features)
        advantages = self.advantage_head(features)
        return value + advantages - advantages.mean(dim=1, keepdim=True)


class YourAgent(kym.Agent):
    def __init__(self, q_network: QNetwork | None = None, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else get_device()
        self.q_network = q_network if q_network is not None else QNetwork()
        self.q_network.to(self.device)
        self.q_network.eval()

    def act(self, observation: Dict[str, Any], info: Dict[str, Any]) -> int:
        del info
        state = preprocess_observation(observation)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {key: value.detach().cpu() for key, value in self.q_network.state_dict().items()}
        torch.save(
            {
                "model_state_dict": state_dict,
                "obs_dim": OBS_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dim": HIDDEN_DIM,
                "env_id": ENV_ID,
                "version": "v3",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "YourAgent":
        device = get_device()
        checkpoint = torch.load(Path(path), map_location=device)
        q_network = QNetwork(
            input_dim=int(checkpoint.get("obs_dim", OBS_DIM)),
            action_dim=int(checkpoint.get("action_dim", ACTION_DIM)),
            hidden_dim=int(checkpoint.get("hidden_dim", HIDDEN_DIM)),
        )
        q_network.load_state_dict(checkpoint["model_state_dict"])
        return cls(q_network=q_network, device=device)


Agent = YourAgent


@dataclass
class TrainingStats:
    env_steps: int
    completed_episodes: int
    epsilon: float
    exploration_phase: str
    avg_survival_time: float
    best_survival_time: float
    loss: float | None
    replay_size: int
    elapsed_seconds: float
    env_steps_per_second: float
    eta_seconds: float
    can_train: bool
    exploration_ready: bool


def epsilon_by_step(epsilon_step: int) -> float:
    """epsilon_step 기준으로 epsilon을 계산합니다.

    v3에서는 epsilon_step이 '학습(warmup 이후)부터의 누적 env_step'이 되도록 구성합니다.
    warmup 이후에도 EPSILON_HOLD_AFTER_WARMUP_STEPS 동안은 epsilon=1.0을 유지합니다.
    """
    if epsilon_step <= EPSILON_HOLD_AFTER_WARMUP_STEPS:
        return EPSILON_START

    decay_step = epsilon_step - EPSILON_HOLD_AFTER_WARMUP_STEPS
    decay_ratio = min(1.0, decay_step / EPSILON_DECAY_STEPS)
    return EPSILON_START + decay_ratio * (EPSILON_END - EPSILON_START)


def exploration_phase(learning_started: bool, epsilon_step: int) -> str:
    if not learning_started:
        return "warmup"
    if epsilon_step <= EPSILON_HOLD_AFTER_WARMUP_STEPS:
        return "hold"
    if epsilon_step < EPSILON_HOLD_AFTER_WARMUP_STEPS + EPSILON_DECAY_STEPS:
        return "decay"
    return "final"


def select_actions(
    env: gym.vector.VectorEnv,
    policy_net: QNetwork,
    states: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> np.ndarray:
    num_envs = states.shape[0]
    random_mask = np.random.random(num_envs) < epsilon
    actions = np.empty(num_envs, dtype=np.int64)

    if np.any(random_mask):
        actions[random_mask] = [int(env.single_action_space.sample()) for _ in range(int(random_mask.sum()))]

    greedy_mask = ~random_mask
    if np.any(greedy_mask):
        state_tensor = torch.as_tensor(states[greedy_mask], dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        actions[greedy_mask] = torch.argmax(q_values, dim=1).detach().cpu().numpy()

    return actions


def optimize_model(
    replay_buffer: ReplayBuffer,
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float | None:
    if len(replay_buffer) < max(TRAIN_START_SIZE, BATCH_SIZE):
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE, device)
    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        # Double DQN: 다음 행동 선택은 policy_net으로, 그 행동의 평가는 target_net으로 합니다.
        # 이렇게 하면 일반 DQN의 Q-value 과대추정을 줄여 긴 생존 정책이 더 안정적으로 학습됩니다.
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + GAMMA * (1.0 - dones) * next_q_values

    loss = F.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return float(loss.item())


def save_policy_network(policy_net: QNetwork, path: str | Path) -> None:
    agent = YourAgent(q_network=QNetwork(), device="cpu")
    agent.q_network.load_state_dict({key: value.detach().cpu() for key, value in policy_net.state_dict().items()})
    agent.save(path)


def log_training(stats: TrainingStats) -> None:
    loss_text = "n/a" if stats.loss is None else f"{stats.loss:.5f}"
    progress = 100.0 * min(1.0, stats.env_steps / TOTAL_ENV_STEPS)
    if stats.can_train:
        learning_state = "training"
    else:
        learning_state = f"collect {stats.replay_size}/{TRAIN_START_SIZE}"

    if stats.exploration_ready:
        exploration_state = stats.exploration_phase
    else:
        exploration_state = f"warmup {stats.replay_size}/{MIN_REPLAY_SIZE}"

    log(
        f"[env_step {stats.env_steps:08d}] "
        f"progress={progress:5.1f}% "
        f"state={learning_state} "
        f"explore={exploration_state} "
        f"episodes={stats.completed_episodes} "
        f"epsilon={stats.epsilon:.3f} "
        f"avg_survival={stats.avg_survival_time:.2f}s "
        f"best={stats.best_survival_time:.2f}s "
        f"loss={loss_text} "
        f"buffer={stats.replay_size} "
        f"speed={stats.env_steps_per_second:.1f} env_steps/s "
        f"elapsed={format_duration(stats.elapsed_seconds)} "
        f"eta={format_duration(stats.eta_seconds)}"
    )


def train() -> YourAgent:
    set_seed(SEED)
    device = get_device()
    print_device_info(device)
    require_cuda_if_needed(device)

    log(
        f"Vectorized env: num_envs={NUM_ENVS}, "
        f"type={'AsyncVectorEnv' if USE_ASYNC_VECTOR_ENV else 'SyncVectorEnv'}, "
        f"render_mode={TRAIN_RENDER_MODE}, bgm={TRAIN_BGM}, "
        f"dummy_sdl={USE_DUMMY_SDL_FOR_TRAINING}"
    )
    log(
        f"학습 설정: total_env_steps={TOTAL_ENV_STEPS}, "
        f"batch_size={BATCH_SIZE}, train_start_size={TRAIN_START_SIZE}, "
        f"exploration_warmup={MIN_REPLAY_SIZE}, "
        f"log_every={LOG_EVERY_ENV_STEPS}"
    )
    log(
        f"탐험 설정: 학습은 {TRAIN_START_SIZE} env_steps부터 시작, "
        f"epsilon warmup={MIN_REPLAY_SIZE} env_steps, "
        f"epsilon_hold_after_warmup={EPSILON_HOLD_AFTER_WARMUP_STEPS}, "
        f"epsilon_decay_steps={EPSILON_DECAY_STEPS}, "
        f"epsilon={EPSILON_START}->{EPSILON_END}"
    )

    log("Vectorized env 생성 중...")
    env = make_vector_env(NUM_ENVS)
    log("Vectorized env 생성 완료. 네트워크와 리플레이 버퍼를 준비합니다.")

    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    env_steps = 0
    completed_episodes = 0
    best_survival_time = 0.0
    recent_survival_times: Deque[float] = deque(maxlen=RECENT_EPISODE_WINDOW)
    last_loss: float | None = None
    next_log_step = LOG_EVERY_ENV_STEPS
    next_save_step = SAVE_EVERY_ENV_STEPS
    next_target_update_step = TARGET_UPDATE_EVERY_ENV_STEPS
    train_started_at = time.perf_counter()

    # v3: warmup 이후부터 decay를 시작하기 위한 카운터
    epsilon_step = 0

    try:
        log("환경 reset 및 초기 observation 전처리 중...")
        observations, infos = env.reset(seed=SEED)
        states = preprocess_observations(observations)
        previous_times = _vector_info_array(infos, "time_elapsed", NUM_ENVS, 0.0)
        log(f"초기화 완료. state shape={states.shape}. 학습 루프를 시작합니다.")

        while env_steps < TOTAL_ENV_STEPS:
            can_train = len(replay_buffer) >= max(TRAIN_START_SIZE, BATCH_SIZE)
            exploration_ready = len(replay_buffer) >= MIN_REPLAY_SIZE

            # v3: epsilon warmup 동안에는 epsilon_step을 증가시키지 않아,
            # decay가 탐험 데이터 수집 구간에서 소모되지 않도록 함
            epsilon = epsilon_by_step(epsilon_step if exploration_ready else 0)

            actions = select_actions(env, policy_net, states, epsilon, device)

            next_observations, _, terminations, truncations, infos = env.step(actions)
            terminations = np.asarray(terminations, dtype=bool)
            truncations = np.asarray(truncations, dtype=bool)
            dones = terminations | truncations

            next_times = _vector_info_array(infos, "time_elapsed", NUM_ENVS, previous_times)
            shaped_rewards = shape_rewards(
                next_observations=next_observations,
                previous_times=previous_times,
                next_times=next_times,
                terminations=terminations,
                truncations=truncations,
            )
            next_states = preprocess_observations(next_observations)

            for env_index in range(NUM_ENVS):
                replay_buffer.push(
                    states[env_index],
                    int(actions[env_index]),
                    float(shaped_rewards[env_index]),
                    next_states[env_index],
                    bool(dones[env_index]),
                )

                if dones[env_index]:
                    survival_time = _final_info_time(
                        infos=infos,
                        env_index=env_index,
                        fallback=float(next_times[env_index]),
                    )
                    completed_episodes += 1
                    recent_survival_times.append(survival_time)
                    best_survival_time = max(best_survival_time, survival_time)

            states = next_states
            previous_times = np.where(dones, 0.0, next_times).astype(np.float32)
            env_steps += NUM_ENVS

            if exploration_ready:
                epsilon_step += NUM_ENVS

            for _ in range(UPDATE_STEPS_PER_VECTOR_STEP):
                loss = optimize_model(
                    replay_buffer=replay_buffer,
                    policy_net=policy_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    device=device,
                )
                if loss is not None:
                    last_loss = loss

            if env_steps >= next_target_update_step:
                target_net.load_state_dict(policy_net.state_dict())
                next_target_update_step += TARGET_UPDATE_EVERY_ENV_STEPS

            if env_steps >= next_log_step:
                log_can_train = len(replay_buffer) >= max(TRAIN_START_SIZE, BATCH_SIZE)
                log_exploration_ready = len(replay_buffer) >= MIN_REPLAY_SIZE
                elapsed_seconds = time.perf_counter() - train_started_at
                env_steps_per_second = env_steps / max(elapsed_seconds, 1e-6)
                remaining_steps = max(0, TOTAL_ENV_STEPS - env_steps)
                eta_seconds = remaining_steps / max(env_steps_per_second, 1e-6)
                avg_survival_time = float(np.mean(recent_survival_times)) if recent_survival_times else 0.0

                log_training(
                    TrainingStats(
                        env_steps=env_steps,
                        completed_episodes=completed_episodes,
                        epsilon=epsilon_by_step(epsilon_step if log_exploration_ready else 0),
                        exploration_phase=exploration_phase(log_exploration_ready, epsilon_step),
                        avg_survival_time=avg_survival_time,
                        best_survival_time=best_survival_time,
                        loss=last_loss,
                        replay_size=len(replay_buffer),
                        elapsed_seconds=elapsed_seconds,
                        env_steps_per_second=env_steps_per_second,
                        eta_seconds=eta_seconds,
                        can_train=log_can_train,
                        exploration_ready=log_exploration_ready,
                    )
                )
                next_log_step += LOG_EVERY_ENV_STEPS

            if env_steps >= next_save_step:
                save_policy_network(policy_net, MODEL_PATH)
                log(f"중간 모델 저장 완료: {MODEL_PATH} (env_step={env_steps})")
                next_save_step += SAVE_EVERY_ENV_STEPS

        save_policy_network(policy_net, MODEL_PATH)
        log(f"Training complete. Saved model to: {MODEL_PATH}")
        return YourAgent.load(MODEL_PATH)
    finally:
        env.close()


if __name__ == "__main__":
    train()
