"""
2026-01 KNU RL Competition Round 2 Avoid Blurp용 병렬 DQN 학습 코드 v4 입니다.

v4 변경점(핵심):
- Atari DQN 스타일에 맞춰 observation과 reward shaping을 단순화했습니다.
- observation은 mario 5개 + blurps 30x6 flatten normalize만 사용합니다.
- reward는 생존 시간, 충돌 벌점, 120초 성공 보상 중심으로 둡니다.
- 주기적으로 모델을 저장하고, 120초 생존 후보가 나오면 greedy 검증을 통과했을 때만 성공 모델로 확정합니다.

프로젝트 루트에서 실행:
    python src/train_v4.py

학습 완료 후 사용:
    agent = YourAgent.load("avoid_blurp_dqn_v4.pt")
"""

from __future__ import annotations

import os
import random
import sys
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
MODEL_PATH = Path("avoid_blurp_dqn_v4.pt")
SUCCESS_MODEL_PATH = Path("avoid_blurp_dqn_v4_success.pt")
SUCCESS_CANDIDATE_MODEL_PATH = Path("avoid_blurp_dqn_v4_candidate.pt")

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
TOTAL_ENV_STEPS = 100_000_000

LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 2048
REPLAY_BUFFER_SIZE = 500_000
TRAIN_START_SIZE = 20_000
MIN_REPLAY_SIZE = 100_000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_HOLD_AFTER_WARMUP_STEPS = 1_000_000
EPSILON_DECAY_STEPS = 20_000_000

HIDDEN_DIM = 256
UPDATE_STEPS_PER_VECTOR_STEP = 2
TARGET_UPDATE_EVERY_ENV_STEPS = 20_000
GRAD_CLIP_NORM = 10.0

LOG_EVERY_ENV_STEPS = 100_000
SAVE_EVERY_ENV_STEPS = 1_000_000
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

# Atari식 단순 보상입니다.
# 핵심 목표가 생존이므로 "살아있으면 보상, 죽으면 큰 벌점, 120초 성공이면 큰 보상"만 강하게 둡니다.
SURVIVAL_REWARD = 1.0
TIME_DELTA_REWARD_SCALE = 5.0
COLLISION_PENALTY = -100.0
SUCCESS_REWARD = 1000.0
SUCCESS_TIME_SECONDS = 120.0

# 120초를 한 번 찍은 episode는 epsilon 탐험 덕분에 나온 우연일 수 있습니다.
# 그래서 바로 종료하지 않고, greedy 정책 평가에서 10번 연속 성공해야 성공 모델로 확정합니다.
SUCCESS_CONFIRM_EPISODES = 10
SUCCESS_CONFIRM_REQUIRED = 10
SUCCESS_CONFIRM_MAX_STEPS = 20_000


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

    return fixed_mario, fixed_blurps


def preprocess_observations(observations: Dict[str, Any]) -> np.ndarray:
    """Atari DQN 스타일로 dict 관측값을 단순 flatten + normalize합니다."""
    mario, blurps = _extract_mario_blurps(observations)

    mario_features = mario.copy()
    mario_features[:, [0, 2]] /= X_SCALE
    mario_features[:, [1, 3]] /= Y_SCALE
    mario_features[:, 4] /= VELOCITY_SCALE

    blurps_features = blurps.copy()
    blurps_features[:, :, [0, 2]] /= X_SCALE
    blurps_features[:, :, [1, 3]] /= Y_SCALE
    blurps_features[:, :, 4] /= VELOCITY_SCALE
    blurps_features[:, :, 5] /= ACCELERATION_SCALE

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


def shape_rewards(
    next_observations: Dict[str, Any],
    previous_times: np.ndarray,
    next_times: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
) -> np.ndarray:
    # 보상 설계:
    # 1. 매 step 살아 있으면 SURVIVAL_REWARD를 줘서 "오래 버티기" 자체를 가장 강하게 학습합니다.
    # 2. info["time_elapsed"]가 증가한 만큼 추가 보상을 줘서 실제 생존 시간 증가를 Q-value에 반영합니다.
    # 3. 충돌로 truncated=True가 되면 큰 벌점을 줍니다.
    # 4. 120초 생존으로 terminated=True가 되면 큰 성공 보상을 줍니다.
    time_delta = np.maximum(0.0, next_times - previous_times)
    rewards = np.full(len(next_times), SURVIVAL_REWARD, dtype=np.float32)
    rewards += TIME_DELTA_REWARD_SCALE * time_delta
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
                "version": "v4",
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

    v4에서는 epsilon_step이 '탐험 warmup 이후부터의 누적 env_step'이 되도록 구성합니다.
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


def evaluate_greedy_once(policy_net: QNetwork, device: torch.device, seed: int) -> float:
    """epsilon 없이 현재 Q-network만으로 한 episode를 평가하고 생존 시간을 반환합니다."""
    configure_headless_sdl()
    env = gym.make(id=ENV_ID, render_mode=None, bgm=False)
    was_training = policy_net.training
    policy_net.eval()

    try:
        observation, info = env.reset(seed=seed)
        survival_time = float(info.get("time_elapsed", 0.0) or 0.0)

        for _ in range(SUCCESS_CONFIRM_MAX_STEPS):
            state = preprocess_observation(observation)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = int(torch.argmax(policy_net(state_tensor), dim=1).item())

            observation, _, terminated, truncated, info = env.step(action)
            survival_time = float(info.get("time_elapsed", survival_time) or survival_time)

            if terminated or truncated:
                break

        return survival_time
    finally:
        env.close()
        if was_training:
            policy_net.train()


def confirm_success_candidate(
    policy_net: QNetwork,
    device: torch.device,
    env_steps: int,
) -> Tuple[bool, list[float]]:
    """120초 후보 모델이 우연인지 확인하기 위해 greedy 평가를 10번 연속 실행합니다."""
    survival_times: list[float] = []
    success_count = 0

    log(
        f"성공 후보 검증 시작: greedy {SUCCESS_CONFIRM_EPISODES}회 연속 "
        f"{SUCCESS_TIME_SECONDS:.0f}초 생존 필요"
    )

    for episode_index in range(SUCCESS_CONFIRM_EPISODES):
        eval_seed = SEED + 1_000_000 + env_steps + episode_index
        survival_time = evaluate_greedy_once(policy_net, device, seed=eval_seed)
        survival_times.append(survival_time)

        if survival_time >= SUCCESS_TIME_SECONDS:
            success_count += 1

        log(
            f"성공 후보 검증 {episode_index + 1}/{SUCCESS_CONFIRM_EPISODES}: "
            f"survival={survival_time:.2f}s, success={success_count}/{SUCCESS_CONFIRM_REQUIRED}"
        )

        if survival_time < SUCCESS_TIME_SECONDS:
            return False, survival_times

    return True, survival_times


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
        f"저장 설정: {SAVE_EVERY_ENV_STEPS} env_steps마다 {MODEL_PATH} 저장, "
        f"{SUCCESS_TIME_SECONDS:.0f}초 생존 시 {SUCCESS_CANDIDATE_MODEL_PATH} 저장 후 "
        f"greedy {SUCCESS_CONFIRM_EPISODES}회 연속 검증"
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

    # v4: warmup 이후부터 decay를 시작하기 위한 카운터입니다.
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

            # v4: epsilon warmup 동안에는 epsilon_step을 증가시키지 않아,
            # decay가 탐험 데이터 수집 구간에서 소모되지 않도록 합니다.
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

                    if survival_time >= SUCCESS_TIME_SECONDS:
                        candidate_env_step = env_steps + NUM_ENVS
                        save_policy_network(policy_net, SUCCESS_CANDIDATE_MODEL_PATH)
                        log(
                            f"120초 생존 후보 발견: survival_time={survival_time:.2f}s, "
                            f"env_step={candidate_env_step}. "
                            f"후보 모델 저장 완료: {SUCCESS_CANDIDATE_MODEL_PATH}"
                        )
                        confirmed, confirmation_times = confirm_success_candidate(
                            policy_net=policy_net,
                            device=device,
                            env_steps=candidate_env_step,
                        )

                        if confirmed:
                            save_policy_network(policy_net, MODEL_PATH)
                            save_policy_network(policy_net, SUCCESS_MODEL_PATH)
                            log(
                                f"성공 후보 검증 통과: times={confirmation_times}. "
                                f"모델 저장 완료: {MODEL_PATH}, {SUCCESS_MODEL_PATH}"
                            )
                            return YourAgent.load(SUCCESS_MODEL_PATH)

                        log(
                            f"성공 후보 검증 실패: times={confirmation_times}. "
                            "우연 가능성이 있어 학습을 계속합니다."
                        )

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


def run(model_path: str | Path = MODEL_PATH) -> None:
    """저장된 v4 모델을 불러와 대회 평가 방식으로 실행합니다."""
    agent = YourAgent.load(model_path)
    log(f"Loaded agent from: {model_path}")
    kym.evaluate(
        env_id=ENV_ID,
        agent=agent,
        render_mode="human",
        bgm=True,
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "train"

    if command == "train":
        train()
    elif command == "run":
        selected_model_path = Path(sys.argv[2]) if len(sys.argv) > 2 else MODEL_PATH
        run(selected_model_path)
    else:
        raise SystemExit(
            "사용법: python src/train_v4.py [train|run] [model_path]\n"
            "예시: python src/train_v4.py train\n"
            "예시: python src/train_v4.py run avoid_blurp_dqn_v4.pt"
        )
