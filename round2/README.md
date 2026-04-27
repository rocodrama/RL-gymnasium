# 2026-01 KNU RL Competition Round 2

## Avoid Blurp!

## 1. 개요

- 하늘에서 떨어지는 Blurp를 피하며 오래 생존하는 강화학습 문제입니다.
- 핵심 목표는 **2분 생존에 가까운 안정적인 회피 정책 학습**입니다.
- 기본 reward가 항상 `0`이므로, `info["time_elapsed"]`와 Blurp 거리 정보를 활용해 reward shaping을 적용합니다.

## 2. 목표

- Blurp와 충돌하지 않고 최대한 오래 생존
- DQN으로 Q-network를 직접 학습
- 학습된 모델을 `avoid_blurp_dqn.pt`로 저장
- `YourAgent.load("avoid_blurp_dqn.pt")`로 평가 가능하게 구성

## 3. 요구사항

- Python `3.12.x`
- PyTorch 기반 DQN
- `kymnasium.Agent`를 상속한 `YourAgent` 구현
- CUDA 자동 감지. RTX 4090 환경에서는 PyTorch가 `cuda`를 사용합니다.

## 4. 설치

```bash
pip install -U kymnasium gymnasium numpy torch
```

## 5. 환경 생성

```python
import gymnasium as gym
import kymnasium

env = gym.make(
    id="kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1",
    render_mode="human",
    bgm=True,
)
```

## 6. Observation

observation은 다음 딕셔너리 형태입니다.

```python
{
    "mario": [left, top, right, bottom, velocity],
    "blurps": [
        [left, top, right, bottom, velocity, acceleration],
        ...
    ],
}
```

### 6.1 `mario`

- 길이 5 벡터
- `[left, top, right, bottom, velocity]`

### 6.2 `blurps`

- shape: `(30, 6)`
- 각 행은 `[left, top, right, bottom, velocity, acceleration]`
- 보이지 않는 Blurp는 0으로 채워집니다.

## 7. Action Space

- `0`: 정지
- `1`: 왼쪽 이동
- `2`: 오른쪽 이동

## 8. 학습 방식

- 알고리즘: DQN
- 네트워크: MLP Q-network
- 입력 state: `mario` 5개 값 + `blurps` 30x6 값을 flatten한 185차원 벡터
- 출력: action 3개에 대한 Q-value
- Replay Buffer 사용
- Target Network 사용
- epsilon-greedy exploration 사용
- Huber loss 사용
- gradient clipping 적용
- `gym.vector.AsyncVectorEnv` 기반 병렬 환경 수집
- 기본 병렬 환경 수: `NUM_ENVS = 8`
- v1 기본 batch size: `BATCH_SIZE = 512`
- v2 기본 batch size: `BATCH_SIZE = 2048`
- 학습 중 `CUDA available`, 평균 생존 시간, epsilon, loss 출력

## 9. Reward Shaping

- 매 step 생존 보상: `+0.1`
- `info["time_elapsed"]` 증가량 보상
- Mario와 가까운 Blurp에 대한 거리 기반 패널티
- 충돌 종료: `-10`
- 2분 생존 성공: `+100`

## 10. 실행 파일 안내

- v1 학습: `python src/train_v1.py`
- v2 장기 학습: `python src/train_v2.py`
- v3 geometry 생존 학습: `python src/train_v3.py`
- 환경 점검: `python scripts/check_env.py`
- CUDA 점검: `python scripts/check_cuda.py`
- 환경 점검 체크리스트: `scripts/checklist.md`
- v1 학습 결과 모델: `avoid_blurp_dqn.pt`
- v2 학습 결과 모델: `avoid_blurp_dqn_v2.pt`
- v3 학습 결과 모델: `avoid_blurp_dqn_v3.pt`

v2 학습 중에는 5,000 env step마다 진행 상황이 출력됩니다.

```text
[env_step 00005000] progress=  0.2% state=collect 5000/20000 explore=warmup 5000/100000 episodes=... epsilon=1.000 loss=n/a ...
[env_step 00020000] progress=  0.7% state=training explore=warmup 20000/100000 episodes=... epsilon=1.000 loss=...
[env_step 000100000] progress=  3.3% state=training explore=hold episodes=... epsilon=1.000 loss=...
[env_step 000250000] progress=  8.3% state=training explore=decay episodes=... epsilon=0.968 loss=... speed=... env_steps/s elapsed=... eta=...
```

`state=collect`는 DQN 업데이트 전 최소 buffer를 채우는 단계이고, `state=training`부터 실제 DQN 업데이트가 진행됩니다.
`explore=warmup` 동안에는 학습이 시작되어도 행동은 계속 epsilon 1.0 랜덤 탐험입니다.
`explore=hold`는 warmup 이후에도 epsilon 1.0을 유지하는 초반 탐험 구간입니다.
`explore=decay`부터 epsilon이 천천히 감소합니다.
`eta`는 현재 처리 속도 기준으로 남은 학습 예상 시간입니다.

v2 기본 설정은 `TOTAL_ENV_STEPS = 3_000_000`입니다. 현재 속도가 약 120 env_steps/s라면 전체 학습은 대략 7시간 전후로 예상됩니다.

CUDA가 보이지 않을 때는 먼저 아래를 확인합니다.

```bash
nvidia-smi -L
python scripts/check_cuda.py
CUDA_VISIBLE_DEVICES=2 python scripts/check_cuda.py
```

`CUDA_VISIBLE_DEVICES=2`를 사용하면 PyTorch 안에서는 해당 물리 GPU가 보통 `cuda:0`으로 다시 매핑됩니다. 이 상태에서 `torch.cuda.device_count()`가 `0`이면, 물리 GPU 번호가 다르거나 현재 쉘/conda 환경에서 NVIDIA 드라이버를 접근하지 못하는 상태입니다.

## 11. 평가 실행 예시

```python
import kymnasium as kym
from src.train_v3 import ENV_ID, YourAgent

agent = YourAgent.load("avoid_blurp_dqn_v3.pt")

kym.evaluate(
    env_id=ENV_ID,
    agent=agent,
    render_mode="human",
    bgm=True,
)
```

## 12. 버전 히스토리

### v1

- DQN 기반 첫 제출용 버전
- `src/train_v1.py` 단일 파일 안에 에이전트, 네트워크, 버퍼, 학습 루프 포함
- observation을 185차원 고정 길이 벡터로 전처리
- 좌표, 속도, 가속도 값 normalization 적용
- 생존 시간과 위험 거리 기반 reward shaping 적용
- Target Network, Replay Buffer, epsilon decay, Huber loss, gradient clipping 적용
- CUDA 자동 감지 및 GPU 학습 지원
- CUDA가 보이지 않으면 CPU로 학습하지 않고 즉시 오류 출력
- `AsyncVectorEnv`로 여러 환경 transition을 동시에 수집
- replay buffer가 충분히 쌓인 뒤 GPU에서 큰 batch로 DQN 업데이트
- 학습은 `render_mode=None`, `bgm=False`로 수행
- 모델 저장 파일명은 `avoid_blurp_dqn.pt`

### v2

- v1 DQN 구조를 유지하되 장기 학습용 하이퍼파라미터로 조정
- `TOTAL_ENV_STEPS = 3_000_000`
- `TRAIN_START_SIZE = 20_000`부터 GPU DQN 업데이트 시작
- `MIN_REPLAY_SIZE = 100_000`까지는 epsilon 1.0 랜덤 탐험 유지
- warmup 이후 `EPSILON_HOLD_AFTER_WARMUP_STEPS = 100_000` 동안 epsilon 1.0 유지
- 이후 `EPSILON_DECAY_STEPS = 1_500_000` 동안 epsilon을 `1.0 -> 0.05`로 천천히 감소
- `BATCH_SIZE = 2048`, `REPLAY_BUFFER_SIZE = 500_000`
- `LOG_EVERY_ENV_STEPS = 5_000`, `SAVE_EVERY_ENV_STEPS = 250_000`
- pygame/SDL 그래픽 컨텍스트가 GPU 0번을 잡지 않도록 dummy SDL 설정 적용
- Blurp 30개를 위험도 기준으로 정렬해 가장 위험한 물체가 앞쪽 feature에 오도록 전처리
- 생존 시간 보상을 크게 올리고 위험 패널티는 약하게 낮춘 생존 중심 reward shaping 적용
- 충돌 벌점과 2분 생존 성공 보상을 크게 올려 목표를 명확히 설정
- 단순 거리 패널티에 더해 마리오 위쪽 같은 x축 라인으로 떨어지는 Blurp에 약한 충돌 궤적 패널티 적용
- Q-network를 Dueling DQN 구조로 변경
- target 계산을 Double DQN 방식으로 변경해 Q-value 과대추정 완화
- `EPSILON_END = 0.01`로 낮춰 최종 탐험 노이즈 감소
- 모델 저장 파일명은 `avoid_blurp_dqn_v2.pt`

### v3

- v2의 병렬 학습, Dueling DQN, Double DQN 구조 유지
- 모델 저장 파일명은 `avoid_blurp_dqn_v3.pt`
- 마리오와 Blurp의 AABB bounding box 관계를 직접 계산
- observation을 절대 좌표 중심에서 마리오 기준 geometry feature 중심으로 변경
- Blurp feature는 `[상대 x, 상대 y, x축 박스 간격, y축 박스 간격, 낙하 속도, touch 위험도]`로 구성
- Blurp 30개는 마리오와 곧 겹칠 가능성이 큰 순서로 정렬
- Blurp의 y속도 방향을 반영해 마리오에게 다가오는 물체만 강하게 위험 처리
- 예를 들어 마리오 아래에 있으면서 더 아래로 내려가는 Blurp는 멀어지는 중이므로 위험도를 낮게 계산
- reward는 생존 보상과 시간 증가 보상을 유지하되, 박스 간격/충돌 궤적/실제 겹침 위험을 기준으로 패널티 계산
- `SURVIVAL_REWARD = 0.2`, `TIME_DELTA_REWARD_SCALE = 8.0`
- `COLLISION_PENALTY = -60.0`, `SUCCESS_REWARD = 500.0`
- 목적: 단순히 가까운 Blurp를 피하는 것이 아니라 마리오 box와 Blurp box가 닿지 않는 행동을 학습
