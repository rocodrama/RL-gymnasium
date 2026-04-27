# 🕹️ 2026-01 KNU RL Competition Round 2

## Avoid Blurp!

> 하늘에서 떨어지는 **Blurp(보글보글)** 을 피하며 최대한 오래 생존하는 강화학습 문제입니다.  
> 시간이 지날수록 Blurp의 개수, 위치, 속도가 랜덤하게 증가하며, 목표는 **최대 2분 생존**입니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
| --- | --- |
| 문제 유형 | 강화학습 생존 문제 |
| 목표 | Blurp를 피하며 최대 2분 동안 생존 |
| 환경 ID | `kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1` |
| Python 버전 | `3.12.x` |
| 평가 기준 | 총 3회 실행 중 가장 오래 생존한 기록 |
| 마감일 | **2026년 5월 21일 08:59** |

---

## ⚙️ 실행 환경

### 필수 설치

```bash
pip install -U kymnasium
```

### Environment 생성

```python
import gymnasium as gym

env = gym.make(
    id='kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
    render_mode='human',   # "human" | "rgb_array" | "none"
    bgm=True               # render_mode="human"일 때만 동작
)
```

---

## 👀 Observation Space

```python
{
    "mario": [left, top, right, bottom, velocity],
    "blurps": [
        [left, top, right, bottom, velocity, acceleration],
        ...
    ]  # shape: [30, 6]
}
```

| 관측값 | 설명 |
| --- | --- |
| `mario` | 길이 5 벡터 |
| `blurps` | 최대 30개 객체. 보이지 않는 경우 0으로 채워짐 |
| 좌표 기준 | 좌측 상단이 `(0, 0)` |

---

## 🎮 Action Space

| Action | 설명 |
| ---: | --- |
| `0` | 정지 |
| `1` | 왼쪽 이동 |
| `2` | 오른쪽 이동 |

---

## 🎯 Reward

- 기본 reward는 항상 `0`입니다.
- 따라서 **직접 reward shaping 설계가 반드시 필요**합니다.

---

## 📊 Info

```python
{
    "time_elapsed": float
}
```

---

## 🏁 Episode 종료 조건

| 상황 | `terminated` | `truncated` |
| --- | --- | --- |
| Blurp와 충돌 | `False` | `True` |
| 2분 생존 성공 | `True` | `False` |

---

## 🎮 수동 플레이

```python
kymnasium.avoid_blurp.ManualPlayWrapper(env_id, debug=True).play()
```

| 키 | 동작 |
| --- | --- |
| `←` | 왼쪽 이동 |
| `→` | 오른쪽 이동 |

---

## 🧠 에이전트 구현 요구사항

```python
import kymnasium as kym


class YourAgent(kym.Agent):
    def act(self, observation, info):
        pass

    @classmethod
    def load(cls, path: str):
        pass

    def save(self, path: str):
        pass
```

---

## 🏋️ 학습 코드 구조

```python
def train():
    env = gym.make(
        id='kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
        render_mode='human',
        bgm=True,
    )

    # 여기서 학습 코드 작성
```

---

## 📦 제출물

### 1. 학습 코드

- `.py` 파일
- 에이전트 정의 포함
- 학습 코드 포함

### 2. 의존성 파일

#### venv 사용 시

```bash
pip list --format=freeze > requirements.txt
```

#### conda 사용 시

```bash
conda env export > environment.yml
```

---

## 🧪 평가 방식

```python
import kymnasium as kym

agent = YourAgent.load('path')

kym.evaluate(
    env_id='kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1',
    agent=agent,
    render_mode='human',
    bgm=True,
)
```

- 총 **3번 실행**합니다.
- 가장 오래 생존한 기록을 기준으로 평가합니다.

---

## 🏆 채점 기준

| 순위 | 점수 |
| ---: | ---: |
| 1위 | 30% |
| 2위 | 28.5% |
| 3위 | 27% |
| ... | ... |
| 15위 | 9% |

---

## ❌ 0점 처리 조건

- 제출 기한 내 미제출
- `requirements` 설치 후에도 실행 불가
- 학습 결과와 시연 결과 불일치
- 시연 불참 또는 시연 불가
- 학습 없이 정책 하드코딩
- 환경 변경

---

## ⏰ 마감일

> **2026년 5월 21일 08:59**
