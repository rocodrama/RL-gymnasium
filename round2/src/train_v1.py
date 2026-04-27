"""
2026-01 KNU RL Competition Round 2 Avoid Blurp용 병렬 DQN 학습 코드입니다.

프로젝트 루트에서 실행:
    python src/train_v1.py

학습 완료 후 사용:
    agent = YourAgent.load("avoid_blurp_dqn.pt")
"""

from __future__ import annotations

import random
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
MODEL_PATH = Path("avoid_blurp_dqn.pt")

OBS_DIM = 5 + 30 * 6
ACTION_DIM = 3
TRAIN_RENDER_MODE = None
TRAIN_BGM = False
NUM_ENVS = 8
USE_ASYNC_VECTOR_ENV = True


# ===============
# 하이퍼파라미터
# ===============

SEED = 42
TOTAL_ENV_STEPS = 1_000_000

LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 512
REPLAY_BUFFER_SIZE = 250_000
MIN_REPLAY_SIZE = 10_000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 250_000

HIDDEN_DIM = 256
UPDATE_STEPS_PER_VECTOR_STEP = 1
TARGET_UPDATE_EVERY_ENV_STEPS = 8_000
GRAD_CLIP_NORM = 10.0

LOG_EVERY_ENV_STEPS = 10_000
SAVE_EVERY_ENV_STEPS = 100_000
RECENT_EPISODE_WINDOW = 50


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
#
# 이 환경은 기본 보상이 항상 0에 가깝기 때문에, DQN이 "무엇이 좋은 행동인지"
# 학습할 수 있도록 직접 보상을 설계합니다.
#
# 보상 설계 요약:
# 1. 살아있는 매 스텝마다 SURVIVAL_REWARD를 줘서 오래 버티는 행동을 장려합니다.
# 2. info["time_elapsed"]가 증가한 만큼 추가 보상을 줘서 실제 생존 시간 증가를 반영합니다.
# 3. 마리오 근처에 Blurp가 있으면 거리 기반 패널티를 줘서 위험 지역을 피하게 합니다.
# 4. truncated=True로 끝나는 충돌 상황에는 큰 음수 보상을 줘서 충돌을 강하게 억제합니다.
# 5. terminated=True인 2분 생존 성공 상황에는 큰 양수 보상을 줘서 최종 목표를 명확히 합니다.

SURVIVAL_REWARD = 0.1
TIME_DELTA_REWARD_SCALE = 1.0
CLOSE_BLURP_PENALTY_WEIGHT = 0.5
DANGER_RADIUS = 90.0
MAX_DANGER_PENALTY = 3.0
COLLISION_PENALTY = -10.0
SUCCESS_REWARD = 100.0


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


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
    print(f"CUDA available: {cuda_available}")
    print(f"PyTorch device: {device} ({device_name})")

    if cuda_available:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def make_env(rank: int = 0) -> gym.Env:
    env = gym.make(
        id=ENV_ID,
        render_mode=TRAIN_RENDER_MODE,
        bgm=TRAIN_BGM,
    )
    env.reset(seed=SEED + rank)
    return env


def make_vector_env(num_envs: int = NUM_ENVS) -> gym.vector.VectorEnv:
    # AsyncVectorEnv를 사용하면 여러 게임 환경이 동시에 전이 데이터를 생성합니다.
    # CPU에서는 환경 스텝을 병렬로 수집하고, GPU에서는 리플레이 버퍼 배치를 학습합니다.
    env_fns = [partial(make_env, rank) for rank in range(num_envs)]
    vector_cls = gym.vector.AsyncVectorEnv if USE_ASYNC_VECTOR_ENV else gym.vector.SyncVectorEnv
    return vector_cls(env_fns)


def _as_fixed_array(value: Any, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    fixed = np.zeros(shape, dtype=np.float32)

    if array.size == 0:
        return fixed

    array = array.reshape(-1)
    fixed_flat = fixed.reshape(-1)
    copy_size = min(array.size, fixed_flat.size)
    fixed_flat[:copy_size] = array[:copy_size]
    return fixed


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


def slice_vector_observation(observations: Dict[str, Any], index: int) -> Dict[str, np.ndarray]:
    return {
        key: np.asarray(value)[index]
        for key, value in observations.items()
    }


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
    # VectorEnv는 에피소드가 끝난 환경을 자동으로 초기화할 수 있으므로,
    # 종료 직전 정보가 final_info/terminal_info에 따로 들어오는 경우를 우선 확인합니다.
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
    # 마리오 중심점과 각 Blurp 중심점 사이의 거리를 계산합니다.
    # DANGER_RADIUS 안에 들어온 Blurp가 많거나 가까울수록 패널티가 커집니다.
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


def close_blurp_penalty(observation: Dict[str, Any]) -> float:
    return float(close_blurp_penalties(observation)[0])


def shape_rewards(
    next_observations: Dict[str, Any],
    previous_times: np.ndarray,
    next_times: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
) -> np.ndarray:
    # 1. 기본 생존 보상: 매 스텝 살아남으면 작은 양수 보상을 줍니다.
    time_delta = np.maximum(0.0, next_times - previous_times)
    rewards = np.full(len(next_times), SURVIVAL_REWARD, dtype=np.float32)

    # 2. 생존 시간 증가량 보상: 실제 time_elapsed가 증가한 만큼 보상을 더합니다.
    rewards += TIME_DELTA_REWARD_SCALE * time_delta

    # 3. 위험 거리 패널티: 가까운 Blurp를 피하도록 거리 기반 패널티를 뺍니다.
    rewards -= close_blurp_penalties(next_observations)

    # 4. 충돌 패널티: 충돌로 truncated=True가 되면 큰 음수 보상을 줍니다.
    rewards += np.where(truncations & ~terminations, COLLISION_PENALTY, 0.0).astype(np.float32)

    # 5. 성공 보상: 2분 생존으로 terminated=True가 되면 큰 양수 보상을 줍니다.
    rewards += np.where(terminations, SUCCESS_REWARD, 0.0).astype(np.float32)
    return rewards.astype(np.float32)


def shape_reward(
    next_observation: Dict[str, Any],
    previous_info: Dict[str, Any],
    next_info: Dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> float:
    previous_time = float(previous_info.get("time_elapsed", 0.0) or 0.0)
    next_time = float(next_info.get("time_elapsed", previous_time) or previous_time)
    reward = shape_rewards(
        next_observations=next_observation,
        previous_times=np.asarray([previous_time], dtype=np.float32),
        next_times=np.asarray([next_time], dtype=np.float32),
        terminations=np.asarray([terminated], dtype=bool),
        truncations=np.asarray([truncated], dtype=bool),
    )[0]
    return float(reward)


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
        # 평가 시에는 탐색 없이 Q-value가 가장 큰 행동을 선택합니다.
        del info
        state = preprocess_observation(observation)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.q_network.state_dict().items()
        }
        torch.save(
            {
                "model_state_dict": state_dict,
                "obs_dim": OBS_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dim": HIDDEN_DIM,
                "env_id": ENV_ID,
                "version": "v1",
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
    avg_survival_time: float
    best_survival_time: float
    loss: float | None
    replay_size: int


def epsilon_by_step(env_step: int) -> float:
    decay_ratio = min(1.0, env_step / EPSILON_DECAY_STEPS)
    return EPSILON_START + decay_ratio * (EPSILON_END - EPSILON_START)


def select_actions(
    env: gym.vector.VectorEnv,
    policy_net: QNetwork,
    states: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> np.ndarray:
    # epsilon 확률로 무작위 행동을 선택하고, 나머지는 Q-network의 탐욕 행동을 선택합니다.
    # 이 함수는 num_envs개의 환경 행동을 한 번에 반환합니다.
    num_envs = states.shape[0]
    random_mask = np.random.random(num_envs) < epsilon
    actions = np.empty(num_envs, dtype=np.int64)

    if np.any(random_mask):
        actions[random_mask] = [
            int(env.single_action_space.sample())
            for _ in range(int(random_mask.sum()))
        ]

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
    # 리플레이 버퍼가 충분히 쌓이기 전에는 학습하지 않습니다.
    # 충분히 모인 뒤에는 GPU로 배치를 올려 DQN 업데이트를 수행합니다.
    if len(replay_buffer) < max(MIN_REPLAY_SIZE, BATCH_SIZE):
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE, device)

    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        # 타깃 네트워크로 다음 상태의 최대 Q-value를 계산해 목표 Q값을 만듭니다.
        next_q_values = target_net(next_states).max(dim=1).values
        target_q_values = rewards + GAMMA * (1.0 - dones) * next_q_values

    # DQN 안정화를 위해 MSE 대신 Huber 손실을 사용합니다.
    loss = F.smooth_l1_loss(current_q_values, target_q_values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # 큰 그래디언트로 인한 불안정한 업데이트를 막기 위해 그래디언트 클리핑을 적용합니다.
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()

    return float(loss.item())


def save_policy_network(policy_net: QNetwork, path: str | Path) -> None:
    agent = YourAgent(q_network=QNetwork(), device="cpu")
    agent.q_network.load_state_dict(
        {
            key: value.detach().cpu()
            for key, value in policy_net.state_dict().items()
        }
    )
    agent.save(path)


def log_training(stats: TrainingStats) -> None:
    loss_text = "n/a" if stats.loss is None else f"{stats.loss:.5f}"
    print(
        f"[env_step {stats.env_steps:08d}] "
        f"episodes={stats.completed_episodes} "
        f"epsilon={stats.epsilon:.3f} "
        f"avg_survival={stats.avg_survival_time:.2f}s "
        f"best={stats.best_survival_time:.2f}s "
        f"loss={loss_text} "
        f"buffer={stats.replay_size}"
    )


def train() -> YourAgent:
    set_seed(SEED)
    device = get_device()
    print_device_info(device)
    print(
        f"Vectorized env: num_envs={NUM_ENVS}, "
        f"type={'AsyncVectorEnv' if USE_ASYNC_VECTOR_ENV else 'SyncVectorEnv'}, "
        f"render_mode={TRAIN_RENDER_MODE}, bgm={TRAIN_BGM}"
    )

    env = make_vector_env(NUM_ENVS)

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

    try:
        observations, infos = env.reset(seed=SEED)
        states = preprocess_observations(observations)
        previous_times = _vector_info_array(infos, "time_elapsed", NUM_ENVS, 0.0)

        while env_steps < TOTAL_ENV_STEPS:
            epsilon = epsilon_by_step(env_steps)
            actions = select_actions(env, policy_net, states, epsilon, device)

            # 여러 환경을 동시에 한 스텝 진행하고 전이 데이터를 한꺼번에 수집합니다.
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
                # 각 환경에서 나온 전이 데이터를 리플레이 버퍼에 개별 저장합니다.
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

            for _ in range(UPDATE_STEPS_PER_VECTOR_STEP):
                # 벡터 스텝마다 리플레이 버퍼에서 큰 배치를 샘플링해 GPU에서 학습합니다.
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
                # 일정 스텝마다 정책 네트워크의 가중치를 타깃 네트워크에 복사합니다.
                target_net.load_state_dict(policy_net.state_dict())
                next_target_update_step += TARGET_UPDATE_EVERY_ENV_STEPS

            if env_steps >= next_log_step:
                avg_survival_time = (
                    float(np.mean(recent_survival_times))
                    if recent_survival_times
                    else 0.0
                )
                log_training(
                    TrainingStats(
                        env_steps=env_steps,
                        completed_episodes=completed_episodes,
                        epsilon=epsilon_by_step(env_steps),
                        avg_survival_time=avg_survival_time,
                        best_survival_time=best_survival_time,
                        loss=last_loss,
                        replay_size=len(replay_buffer),
                    )
                )
                next_log_step += LOG_EVERY_ENV_STEPS

            if env_steps >= next_save_step:
                save_policy_network(policy_net, MODEL_PATH)
                next_save_step += SAVE_EVERY_ENV_STEPS

        save_policy_network(policy_net, MODEL_PATH)
        print(f"Training complete. Saved model to: {MODEL_PATH}")
        return YourAgent.load(MODEL_PATH)
    finally:
        env.close()


if __name__ == "__main__":
    train()


# requirements.txt 예시:
# gymnasium
# kymnasium
# numpy
# torch
