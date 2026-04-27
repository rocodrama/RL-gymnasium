from __future__ import annotations

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

# Windows local environments can load Intel OpenMP twice through PyTorch/MKL.
# This workaround must be applied before importing numpy or torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - lets static checks run without deps.
    gym = None

try:
    import kymnasium

    AgentBase = kymnasium.Agent
except Exception:  # pragma: no cover - lets static checks run without deps.
    kymnasium = None

    class AgentBase:  # type: ignore[no-redef]
        pass


ENV_ID = "kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1"
ACTION_DIM = 3
DEFAULT_TOP_K = 10

WORLD_WIDTH = 256.0
WORLD_HEIGHT = 240.0
MAX_MARIO_SPEED = 12.0
MAX_BLURP_VX = 12.0
MAX_BLURP_VY = 18.0
MAX_BLURP_AY = 2.0
TIME_HORIZON = 120.0
COLLISION_MARGIN_X = 8.0
RISK_THRESHOLD = 0.15

MARIO_FEATURE_DIM = 4
BLURP_FEATURE_DIM = 11
RAW_STATE_DIM = 5 + 30 * 7


def state_dim_for(top_k: int = DEFAULT_TOP_K) -> int:
    del top_k
    return RAW_STATE_DIM


def _safe_float_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == shape:
        return arr
    out = np.zeros(shape, dtype=np.float32)
    src = arr.reshape(-1)
    dst = out.reshape(-1)
    n = min(src.size, dst.size)
    if n:
        dst[:n] = src[:n]
    return out


def solve_time_to_reach_y(y0: float, vy: float, ay: float, target_y: float) -> float:
    """Return the smallest positive t where y0 + vy*t + 0.5*ay*t^2 == target_y."""
    c = y0 - target_y
    a = 0.5 * ay
    b = vy

    candidates: list[float] = []
    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            candidates.append(-c / b)
    else:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = math.sqrt(disc)
            denom = 2.0 * a
            candidates.append((-b - sqrt_disc) / denom)
            candidates.append((-b + sqrt_disc) / denom)

    positive = [t for t in candidates if t > 1e-4 and math.isfinite(t)]
    return min(positive) if positive else float("inf")


@dataclass(frozen=True)
class RiskInfo:
    risk: float
    avoid_action: int
    avoid_vx_sign: float
    future_dx: float


def _extract_features_and_risk(
    observation: dict[str, Any],
    top_k: int = DEFAULT_TOP_K,
) -> tuple[np.ndarray, RiskInfo]:
    mario = _safe_float_array(observation.get("mario", np.zeros(5, dtype=np.float32)), (5,))
    blurps = _safe_float_array(
        observation.get("blurps", np.zeros((30, 7), dtype=np.float32)), (30, 7)
    )

    m_left, m_top, m_right, _m_bottom, m_vx = [float(x) for x in mario]
    mario_w = max(m_right - m_left, 1.0)
    mario_x = 0.5 * (m_left + m_right)

    mario_features = np.array(
        [
            mario_x / WORLD_WIDTH,
            np.clip(m_vx / MAX_MARIO_SPEED, -2.0, 2.0),
            np.clip(m_left / WORLD_WIDTH, -2.0, 2.0),
            np.clip((WORLD_WIDTH - m_right) / WORLD_WIDTH, -2.0, 2.0),
        ],
        dtype=np.float32,
    )

    rows: list[tuple[float, np.ndarray, float]] = []
    for blurp in blurps:
        if not np.any(np.abs(blurp) > 1e-6):
            continue

        b_left, b_top, b_right, b_bottom, b_vx, b_vy, b_ay = [float(x) for x in blurp]
        b_w = max(b_right - b_left, 1.0)
        b_cx = 0.5 * (b_left + b_right)
        b_cy = 0.5 * (b_top + b_bottom)

        # The lower edge is the first point that can reach Mario's top side.
        t_hit = solve_time_to_reach_y(b_bottom, b_vy, b_ay, m_top)
        if math.isfinite(t_hit):
            t_for_prediction = min(t_hit, TIME_HORIZON)
            predicted_blurp_x = b_cx + b_vx * t_for_prediction
            predicted_mario_x = mario_x + m_vx * t_for_prediction
            future_dx = predicted_blurp_x - predicted_mario_x
            time_score = math.exp(-t_for_prediction / 45.0)
            hit_width = 0.5 * (mario_w + b_w) + COLLISION_MARGIN_X
            x_score = math.exp(-((abs(future_dx) / max(hit_width, 1.0)) ** 2))
            risk = float(time_score * x_score)
        else:
            t_for_prediction = TIME_HORIZON
            predicted_blurp_x = b_cx + b_vx * TIME_HORIZON
            predicted_mario_x = mario_x + m_vx * TIME_HORIZON
            future_dx = predicted_blurp_x - predicted_mario_x
            risk = 0.0

        row = np.array(
            [
                1.0,
                np.clip((b_cx - mario_x) / WORLD_WIDTH, -4.0, 4.0),
                np.clip((b_cy - m_top) / WORLD_HEIGHT, -4.0, 4.0),
                np.clip(b_vx / MAX_BLURP_VX, -4.0, 4.0),
                np.clip(b_vy / MAX_BLURP_VY, -4.0, 4.0),
                np.clip(b_ay / MAX_BLURP_AY, -4.0, 4.0),
                np.clip(t_for_prediction / TIME_HORIZON, 0.0, 1.0),
                np.clip(predicted_blurp_x / WORLD_WIDTH, -4.0, 4.0),
                np.clip(predicted_mario_x / WORLD_WIDTH, -4.0, 4.0),
                np.clip(future_dx / WORLD_WIDTH, -4.0, 4.0),
                np.clip(risk, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        rows.append((risk, row, future_dx))

    rows.sort(key=lambda item: item[0], reverse=True)

    blurp_features = np.zeros((top_k, BLURP_FEATURE_DIM), dtype=np.float32)
    for idx, (_risk, row, _future_dx) in enumerate(rows[:top_k]):
        blurp_features[idx] = row

    if rows:
        best_risk, _best_row, best_future_dx = rows[0]
        if abs(best_future_dx) < 1e-3:
            left_space = m_left
            right_space = WORLD_WIDTH - m_right
            avoid_action = 1 if left_space > right_space else 2
        else:
            avoid_action = 1 if best_future_dx > 0.0 else 2
        avoid_vx_sign = -1.0 if avoid_action == 1 else 1.0
        risk_info = RiskInfo(
            risk=float(np.clip(best_risk, 0.0, 1.0)),
            avoid_action=avoid_action,
            avoid_vx_sign=avoid_vx_sign,
            future_dx=float(best_future_dx),
        )
    else:
        risk_info = RiskInfo(risk=0.0, avoid_action=0, avoid_vx_sign=0.0, future_dx=0.0)

    state = np.concatenate([mario_features, blurp_features.reshape(-1)]).astype(np.float32)
    return np.nan_to_num(state, nan=0.0, posinf=4.0, neginf=-4.0), risk_info


def preprocess_observation(observation: dict[str, Any], top_k: int = DEFAULT_TOP_K) -> np.ndarray:
    del top_k
    mario = _safe_float_array(observation.get("mario", np.zeros(5, dtype=np.float32)), (5,))
    blurps = _safe_float_array(
        observation.get("blurps", np.zeros((30, 7), dtype=np.float32)), (30, 7)
    )

    mario = mario.copy()
    mario[[0, 2]] /= WORLD_WIDTH
    mario[[1, 3]] /= WORLD_HEIGHT
    mario[4] /= MAX_MARIO_SPEED

    blurps = blurps.copy()
    blurps[:, [0, 2]] /= WORLD_WIDTH
    blurps[:, [1, 3]] /= WORLD_HEIGHT
    blurps[:, 4] /= MAX_BLURP_VX
    blurps[:, 5] /= MAX_BLURP_VY
    blurps[:, 6] /= MAX_BLURP_AY

    state = np.concatenate([mario.reshape(-1), blurps.reshape(-1)]).astype(np.float32)
    return np.nan_to_num(state, nan=0.0, posinf=4.0, neginf=-4.0)


def estimate_risk(observation: dict[str, Any], top_k: int = DEFAULT_TOP_K) -> RiskInfo:
    _state, risk_info = _extract_features_and_risk(observation, top_k)
    return risk_info


def split_vector_observation(observation: dict[str, Any], index: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(value[index]).copy() for key, value in observation.items()}


def preprocess_batch_observations(
    observations: dict[str, Any], top_k: int = DEFAULT_TOP_K
) -> np.ndarray:
    num_envs = int(np.asarray(observations["mario"]).shape[0])
    states = np.zeros((num_envs, state_dim_for(top_k)), dtype=np.float32)
    for env_idx in range(num_envs):
        states[env_idx] = preprocess_observation(split_vector_observation(observations, env_idx), top_k)
    return states


def _index_info_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        if "mario" in value and "blurps" in value:
            mario = np.asarray(value["mario"])
            blurps = np.asarray(value["blurps"])
            if mario.ndim >= 2 and blurps.ndim >= 3:
                return {"mario": mario[index].copy(), "blurps": blurps[index].copy()}
            return value
        return {k: _index_info_value(v, index) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        return arr.item()
    return arr[index]


def get_final_observation(infos: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(infos, dict):
        return None
    for key in ("final_obs", "final_observation", "terminal_observation"):
        if key not in infos:
            continue
        mask = infos.get(f"_{key}")
        if mask is not None:
            try:
                if not bool(np.asarray(mask)[index]):
                    continue
            except Exception:
                pass
        item = _index_info_value(infos[key], index)
        if isinstance(item, dict) and "mario" in item and "blurps" in item:
            return item
    return None


def get_final_info(infos: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(infos, dict) or "final_info" not in infos:
        return None
    mask = infos.get("_final_info")
    if mask is not None:
        try:
            if not bool(np.asarray(mask)[index]):
                return None
        except Exception:
            pass
    item = _index_info_value(infos["final_info"], index)
    return item if isinstance(item, dict) else None


def get_info_scalar(infos: Any, key: str, index: int) -> float | None:
    final_info = get_final_info(infos, index)
    if final_info is not None and key in final_info:
        try:
            return float(final_info[key])
        except (TypeError, ValueError):
            return None

    if not isinstance(infos, dict) or key not in infos:
        return None
    mask = infos.get(f"_{key}")
    if mask is not None:
        try:
            if not bool(np.asarray(mask)[index]):
                return None
        except Exception:
            pass
    try:
        return float(_index_info_value(infos[key], index))
    except (TypeError, ValueError):
        return None


def shaped_reward(
    reward: float,
    observation: dict[str, Any],
    action: int,
    next_observation: dict[str, Any],
    terminated: bool,
    truncated: bool,
    top_k: int = DEFAULT_TOP_K,
) -> float:
    del observation, action, next_observation, top_k

    shaped = float(reward) + 0.02
    if truncated and not terminated:
        shaped -= 1000.0
    if terminated:
        shaped += 1000.0

    return shaped


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        value = self.value_head(z)
        advantage = self.advantage_head(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class YourAgent(AgentBase):
    def __init__(
        self,
        state_dim: int | None = None,
        action_dim: int = ACTION_DIM,
        top_k: int = DEFAULT_TOP_K,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            super().__init__()
        except Exception:
            pass
        self.top_k = int(top_k)
        self.state_dim = int(state_dim or state_dim_for(self.top_k))
        self.action_dim = int(action_dim)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_net = DuelingDQN(self.state_dim, self.action_dim).to(self.device)
        self.q_net.eval()

    def act(self, observation: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        del info
        state = preprocess_observation(observation, self.top_k)
        return self.act_state(state, epsilon=0.0)

    def act_state(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def save(self, path: str) -> None:
        checkpoint = {
            "model_state_dict": {k: v.detach().cpu() for k, v in self.q_net.state_dict().items()},
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "top_k": self.top_k,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str) -> "YourAgent":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        agent = cls(
            state_dim=int(checkpoint["state_dim"]),
            action_dim=int(checkpoint.get("action_dim", ACTION_DIM)),
            top_k=int(checkpoint.get("top_k", DEFAULT_TOP_K)),
            device=device,
        )
        agent.q_net.load_state_dict(checkpoint["model_state_dict"])
        agent.q_net.eval()
        return agent


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        batch_size = int(states.shape[0])
        for i in range(batch_size):
            self.states[self.pos] = states[i]
            self.next_states[self.pos] = next_states[i]
            self.actions[self.pos] = int(actions[i])
            self.rewards[self.pos] = float(rewards[i])
            self.dones[self.pos] = float(dones[i])
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


@dataclass
class TrainConfig:
    total_env_steps: int = 10_000_000
    num_envs: int = 256
    replay_size: int = 300_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 1e-4
    target_update_interval: int = 2_000
    train_freq: int = 4
    warmup_steps: int = 10_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 300_000
    grad_clip_norm: float = 10.0
    top_k: int = DEFAULT_TOP_K
    seed: int = 42
    save_path: str = "avoid_blurp_dqn_v3.pt"
    log_interval: int = 10_000
    use_async_vector_env: bool = True


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    frac = min(max(step, 0) / cfg.epsilon_decay_steps, 1.0)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def make_env_fn(render_mode: str, seed: int, rank: int) -> Callable[[], Any]:
    def thunk() -> Any:
        if gym is None:
            raise RuntimeError("gymnasium is not installed.")
        import kymnasium as _kymnasium  # noqa: F401 - registers the environment.

        env = gym.make(
            id=ENV_ID,
            render_mode=render_mode,
            bgm=False,
        )
        env.reset(seed=seed + rank)
        try:
            env.action_space.seed(seed + rank)
        except Exception:
            pass
        return env

    return thunk


def make_vector_env(
    num_envs: int,
    render_mode: str,
    seed: int,
    use_async: bool = True,
) -> Any:
    if gym is None:
        raise RuntimeError("gymnasium is not installed.")

    env_fns = [make_env_fn(render_mode, seed, rank) for rank in range(num_envs)]
    kwargs: dict[str, Any] = {}
    try:
        from gymnasium.vector import AutoresetMode

        if hasattr(AutoresetMode, "SAME_STEP"):
            kwargs["autoreset_mode"] = AutoresetMode.SAME_STEP
    except Exception:
        pass

    vector_cls = gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
    try:
        return vector_cls(env_fns, **kwargs)
    except TypeError:
        return vector_cls(env_fns)
    except Exception:
        if use_async:
            sync_cls = gym.vector.SyncVectorEnv
            try:
                return sync_cls(env_fns, **kwargs)
            except TypeError:
                return sync_cls(env_fns)
        raise


def optimize_model(
    online_net: DuelingDQN,
    target_net: DuelingDQN,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    cfg: TrainConfig,
) -> float:
    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size)

    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = torch.argmax(online_net(next_states), dim=1, keepdim=True)
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + cfg.gamma * (1.0 - dones) * next_q_values

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online_net.parameters(), cfg.grad_clip_norm)
    optimizer.step()
    return float(loss.item())


def train(total_env_steps: int | None = None) -> YourAgent:
    cfg = TrainConfig()
    if total_env_steps is not None:
        cfg.total_env_steps = int(total_env_steps)
    env_steps_override = os.getenv("TOTAL_ENV_STEPS")
    if env_steps_override:
        cfg.total_env_steps = int(env_steps_override)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = state_dim_for(cfg.top_k)
    agent = YourAgent(state_dim=state_dim, action_dim=ACTION_DIM, top_k=cfg.top_k, device=device)
    target_net = DuelingDQN(state_dim, ACTION_DIM).to(device)
    target_net.load_state_dict(agent.q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(agent.q_net.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size, state_dim, device)

    envs = make_vector_env(
        cfg.num_envs,
        render_mode="none",
        seed=cfg.seed,
        use_async=cfg.use_async_vector_env,
    )

    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
    episode_seconds = np.zeros(cfg.num_envs, dtype=np.float32)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_seconds: deque[float] = deque(maxlen=100)
    last_loss = float("nan")
    learner_steps = 0
    train_credit = 0

    try:
        observations, _infos = envs.reset(seed=cfg.seed)
        states = preprocess_batch_observations(observations, cfg.top_k)

        global_step = 0
        agent.q_net.train()
        while global_step < cfg.total_env_steps:
            epsilon = epsilon_by_step(global_step, cfg)
            actions = np.array(
                [agent.act_state(states[i], epsilon=epsilon) for i in range(cfg.num_envs)],
                dtype=np.int64,
            )

            next_observations, raw_rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)
            next_live_states = preprocess_batch_observations(next_observations, cfg.top_k)
            buffer_next_states = np.zeros_like(next_live_states)
            shaped_rewards = np.zeros(cfg.num_envs, dtype=np.float32)

            for env_idx in range(cfg.num_envs):
                obs_i = split_vector_observation(observations, env_idx)
                final_obs = get_final_observation(infos, env_idx) if bool(dones[env_idx]) else None
                next_obs_i = (
                    final_obs
                    if final_obs is not None
                    else split_vector_observation(next_observations, env_idx)
                )
                survival_seconds = get_info_scalar(infos, "time_elapsed", env_idx)
                if survival_seconds is not None:
                    episode_seconds[env_idx] = float(survival_seconds)
                buffer_next_states[env_idx] = preprocess_observation(next_obs_i, cfg.top_k)
                shaped_rewards[env_idx] = shaped_reward(
                    float(raw_rewards[env_idx]),
                    obs_i,
                    int(actions[env_idx]),
                    next_obs_i,
                    bool(terminations[env_idx]),
                    bool(truncations[env_idx]),
                    cfg.top_k,
                )

            replay.add_batch(states, actions, shaped_rewards, buffer_next_states, dones)

            episode_returns += shaped_rewards
            episode_lengths += 1
            for env_idx, done in enumerate(dones):
                if done:
                    survival_sec = float(episode_seconds[env_idx])
                    success = bool(terminations[env_idx])
                    recent_returns.append(float(episode_returns[env_idx]))
                    recent_seconds.append(survival_sec)
                    outcome = "success" if success else "collision"
                    print(
                        f"episode env={env_idx} outcome={outcome} "
                        f"return={episode_returns[env_idx]:.2f} "
                        f"survival_sec={survival_sec:.2f} "
                        f"env_steps={episode_lengths[env_idx]} "
                        f"epsilon={epsilon:.3f} step={global_step}"
                    )
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
                    episode_seconds[env_idx] = 0.0

            states = next_live_states
            observations = next_observations
            global_step += cfg.num_envs

            if global_step >= cfg.warmup_steps and len(replay) >= cfg.batch_size:
                train_credit += cfg.num_envs
                while train_credit >= cfg.train_freq:
                    last_loss = optimize_model(agent.q_net, target_net, optimizer, replay, cfg)
                    learner_steps += 1
                    train_credit -= cfg.train_freq
                    if learner_steps % cfg.target_update_interval == 0:
                        target_net.load_state_dict(agent.q_net.state_dict())

            if global_step % cfg.log_interval < cfg.num_envs:
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                avg_seconds = float(np.mean(recent_seconds)) if recent_seconds else 0.0
                print(
                    f"step={global_step} learner_steps={learner_steps} "
                    f"epsilon={epsilon:.3f} replay={len(replay)} "
                    f"loss={last_loss:.5f} avg_return_100={avg_return:.2f} "
                    f"avg_survival_sec_100={avg_seconds:.2f}"
                )

        agent.q_net.eval()
        agent.save(cfg.save_path)
        print(f"saved agent to {cfg.save_path}")
        return agent
    finally:
        envs.close()


def evaluate_once(path: str = "avoid_blurp_dqn_v3.pt") -> float:
    if gym is None:
        raise RuntimeError("gymnasium is not installed.")
    import kymnasium as _kymnasium  # noqa: F401 - registers the environment.

    agent = YourAgent.load(path)
    env = gym.make(
        id=ENV_ID,
        render_mode="human",
        bgm=False,
    )
    total_reward = 0.0
    steps = 0
    survival_sec = 0.0
    try:
        observation, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            try:
                import pygame

                pygame.event.pump()
            except Exception:
                pass
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            try:
                survival_sec = float(info.get("time_elapsed", survival_sec))
            except (TypeError, ValueError, AttributeError):
                pass
            total_reward += float(reward)
            steps += 1
            time.sleep(1.0 / 60.0)
    finally:
        env.close()

    print(f"evaluation reward={total_reward:.2f} survival_sec={survival_sec:.2f} steps={steps}")
    return total_reward


if __name__ == "__main__":
    train()
