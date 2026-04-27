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
- 기본 batch size: `BATCH_SIZE = 512`
- 학습 중 `CUDA available`, 평균 생존 시간, epsilon, loss 출력

## 9. Reward Shaping

- 매 step 생존 보상: `+0.1`
- `info["time_elapsed"]` 증가량 보상
- Mario와 가까운 Blurp에 대한 거리 기반 패널티
- 충돌 종료: `-10`
- 2분 생존 성공: `+100`

## 10. 실행 파일 안내

- 학습: `python src/train_v1.py`
- 환경 점검: `python scripts/check_env.py`
- 환경 점검 체크리스트: `scripts/checklist.md`
- 학습 결과 모델: `avoid_blurp_dqn.pt`

## 11. 평가 실행 예시

```python
import kymnasium as kym
from src.train_v1 import ENV_ID, YourAgent

agent = YourAgent.load("avoid_blurp_dqn.pt")

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
- `AsyncVectorEnv`로 여러 환경 transition을 동시에 수집
- replay buffer가 충분히 쌓인 뒤 GPU에서 큰 batch로 DQN 업데이트
- 학습은 `render_mode=None`, `bgm=False`로 수행
- 모델 저장 파일명은 `avoid_blurp_dqn.pt`
