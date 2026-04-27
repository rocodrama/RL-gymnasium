"""
2026-01 KNU RL Competition Round 2 Avoid Blurp용 병렬 DQN 학습 코드 v2 입니다.

v2 변경점(핵심):
- 탐험(ε) decay를 env_step=0부터 소모하지 않고, warmup(리플레이 버퍼가 MIN_REPLAY_SIZE에 도달) 이후부터
  본격적으로 decay가 시작되도록 조정했습니다.
  즉 warmup 구간에서는 ε=EPSILON_START를 유지해 "초반 탐험"을 확실히 보장합니다.
- warmup과 전체 학습 길이를 장기 학습 기준으로 확대했습니다.
  100k env step 동안은 replay buffer를 충분히 채우고, 이후에도 100k env step 동안 ε=1.0을 유지한 뒤
  천천히 decay합니다.

프로젝트 루트에서 실행:
    python src/train_v2.py

학습 완료 후 사용:
    agent = YourAgent.load("avoid_blurp_dqn_v2.pt")
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
MODEL_PATH = Path("avoid_blurp_dqn_v2.pt")

OBS_DIM = 5 + 30 * 6
ACTION_DIM = 3
TRAIN_RENDER_MODE = None
TRAIN_BGM = False
NUM_ENVS = 8
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
BATCH_SIZE = 1024
REPLAY_BUFFER_SIZE = 500_000
MIN_REPLAY_SIZE = 100_000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_HOLD_AFTER_WARMUP_STEPS = 100_000
EPSILON_DECAY_STEPS = 1_500_000

HIDDEN_DIM = 256
UPDATE_STEPS_PER_VECTOR_STEP = 1
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

SURVIVAL_REWARD = 0.1
TIME_DELTA_REWARD_SCALE = 1.0
CLOSE_BLURP_PENALTY_WEIGHT = 0.5
DANGER_RADIUS = 90.0
MAX_DANGER_PENALTY = 3.0
COLLISION_PENALTY = -10.0
SUCCESS_REWARD = 100.0


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
    """벡터 환경의 dict 관측값을 정규화된 고정 길이 벡터 묶음으로 변환합니다."""
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


def close_blurp_penalties(observations: Dict[str, Any]) -> np.ndarray:
    mario, blurps = _extract_mario_blurps(observations)

    mario_x = (mario[:, 0] + mario[:, 2]) * 0.5
    mario_y = (mario[:, 1] + mario[:, 3]) * 0.5
    blurp_x = (blurps[:, :, 0] + blurps[:, :, 2]) * 0.5
    blurp_y = (blurps[:, :, 1] + blurps[:, :, 3]) * 0.5

    distances = np.hypot(mario_x[:, None] - blurp_x, mario_y[:, None] - blurp_y)
    visible = ~np.all(np.isclose(blurps, 0.0), axis=2)
    danger = visible & (distances < DANGER_RADIUS)
    raw_penalties = CLOSE_BLURP_PENALTY_WEIGHT * (1.0 - distances / DANGER_RADIUS)
    penalties = np.where(danger, raw_penalties, 0.0).sum(axis=1)

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
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
                "version": "v2",
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
    learning_started: bool


def epsilon_by_step(epsilon_step: int) -> float:
    """epsilon_step 기준으로 epsilon을 계산합니다.

    v2에서는 epsilon_step이 '학습(warmup 이후)부터의 누적 env_step'이 되도록 구성합니다.
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
    if len(replay_buffer) < max(MIN_REPLAY_SIZE, BATCH_SIZE):
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE, device)
    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
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
    learning_state = "training" if stats.learning_started else f"warmup {stats.replay_size}/{MIN_REPLAY_SIZE}"

    log(
        f"[env_step {stats.env_steps:08d}] "
        f"progress={progress:5.1f}% "
        f"state={learning_state} "
        f"explore={stats.exploration_phase} "
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
        f"batch_size={BATCH_SIZE}, min_replay_size={MIN_REPLAY_SIZE}, "
        f"log_every={LOG_EVERY_ENV_STEPS}"
    )
    log(
        f"탐험 설정: warmup={MIN_REPLAY_SIZE} env_steps, "
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

    # v2: warmup 이후부터 decay를 시작하기 위한 카운터
    epsilon_step = 0

    try:
        log("환경 reset 및 초기 observation 전처리 중...")
        observations, infos = env.reset(seed=SEED)
        states = preprocess_observations(observations)
        previous_times = _vector_info_array(infos, "time_elapsed", NUM_ENVS, 0.0)
        log(f"초기화 완료. state shape={states.shape}. 학습 루프를 시작합니다.")

        while env_steps < TOTAL_ENV_STEPS:
            learning_started = len(replay_buffer) >= max(MIN_REPLAY_SIZE, BATCH_SIZE)

            # v2: warmup 동안에는 epsilon_step을 증가시키지 않아, decay가 warmup에서 소모되지 않도록 함
            epsilon = epsilon_by_step(epsilon_step if learning_started else 0)

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

            if learning_started:
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
                elapsed_seconds = time.perf_counter() - train_started_at
                env_steps_per_second = env_steps / max(elapsed_seconds, 1e-6)
                remaining_steps = max(0, TOTAL_ENV_STEPS - env_steps)
                eta_seconds = remaining_steps / max(env_steps_per_second, 1e-6)
                avg_survival_time = float(np.mean(recent_survival_times)) if recent_survival_times else 0.0

                log_training(
                    TrainingStats(
                        env_steps=env_steps,
                        completed_episodes=completed_episodes,
                        epsilon=epsilon_by_step(epsilon_step if learning_started else 0),
                        exploration_phase=exploration_phase(learning_started, epsilon_step),
                        avg_survival_time=avg_survival_time,
                        best_survival_time=best_survival_time,
                        loss=last_loss,
                        replay_size=len(replay_buffer),
                        elapsed_seconds=elapsed_seconds,
                        env_steps_per_second=env_steps_per_second,
                        eta_seconds=eta_seconds,
                        learning_started=last_loss is not None,
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
