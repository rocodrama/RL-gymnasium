# AvoidBlurp DQN

`kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1` 환경을 학습하는
MLP Double-Dueling DQN 실험 코드입니다.

## 게임 정보

AvoidBlurp는 Mario가 위에서 떨어지는 Blurp를 피하면서 오래 생존하는 게임입니다.

- 목표: 120초 동안 충돌하지 않고 생존
- 성공 종료: 120초 생존 시 `terminated=True`
- 실패 종료: Blurp와 충돌 시 `truncated=True`
- 학습 렌더링: `render_mode="none"`
- 평가/시연 렌더링: `render_mode="human"`

### Action Space

```text
0 = 정지
1 = 왼쪽 이동
2 = 오른쪽 이동
```

### Observation

observation은 dict 형태입니다.

```text
obs["mario"]  shape=(5,)
[left, top, right, bottom, vx]

obs["blurps"] shape=(30, 7)
[left, top, right, bottom, vx, vy, ay]
```

Blurp는 ballistic motion을 따릅니다.

```text
y(t) = y0 + vy * t + 0.5 * ay * t^2
x(t) = x0 + vx * t
```

전처리에서는 Mario 위치/속도와 각 Blurp의 예상 충돌 시간, 미래 x 위치,
미래 거리, risk score를 계산하고, active Blurp 중 risk가 높은 상위 10개만
state vector에 사용합니다.

## 실행 방법

의존성 설치 후 학습:

```powershell
python src/train_v2.py
```

학습 step 수를 줄이거나 늘리고 싶으면:

```powershell
$env:TOTAL_ENV_STEPS=500000
python src/train_v2.py
```

CUDA 확인:

```powershell
python scripts/check_cuda.py
```

평가 예시:

```powershell
python -c "from src.train_v2 import evaluate_once; evaluate_once()"
```

## 로그 의미

학습 중 에피소드 종료 로그는 다음처럼 출력됩니다.

```text
episode env=3 outcome=collision return=-11.64 survival_sec=3.65 env_steps=55 epsilon=0.914 step=27504
```

- `env`: 8개 병렬 환경 중 번호
- `outcome`: `success` 또는 `collision`
- `return`: shaped reward 누적합
- `survival_sec`: 실제 생존 시간, `info["time_elapsed"]` 기준
- `env_steps`: 해당 에피소드의 step 수
- `epsilon`: 현재 epsilon-greedy 랜덤 행동 확률
- `step`: 전체 환경 transition 수

## Versions

### v1 - Risk Shaping DQN

파일: `src/train_v1.py`

- MLP Double-Dueling DQN
- Double DQN target 계산
- 8개 병렬 vector env 사용
- ReplayBuffer size 300000
- `batch_size=256`, `gamma=0.99`, `lr=1e-4`
- target network hard update every 2000 learner steps
- epsilon `1.0 -> 0.05`, 300000 env steps decay
- reward shaping 포함:
  - 매 step 생존 보상
  - 충돌 penalty
  - 성공 reward
  - risk_next penalty
  - risk 감소 보상
  - 위험 시 회피 방향 action/vx 보상
  - 벽 근접 penalty
  - 과속 penalty

저장 파일:

```text
avoid_blurp_dqn.pt
```

### v2 - Simple Survival Reward DQN

파일: `src/train_v2.py`

v1 구조는 유지하고 reward를 단순화한 버전입니다.

- 매 step 생존 보상: `+0.02`
- 충돌 실패: `-1000`
- 120초 생존 성공: `+1000`
- risk shaping 제거
- 벽/속도 penalty 제거
- reward clip 제거
- 에피소드 로그에 `survival_sec`, `env_steps`, `epsilon` 표시

저장 파일:

```text
avoid_blurp_dqn_v2.pt
```

v2는 "살아 있으면 조금 좋고, 충돌하면 매우 나쁘고, 120초 생존하면 매우 좋다"는
목표를 더 직접적으로 전달하기 위한 실험 버전입니다.
