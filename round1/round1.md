# 2026-01 KNU RL Competition Round 1

## Zelda’s Adventure

---

## 🧭 Overview

Zelda’s Adventure는 링크를 조작하여 **던전의 목표 지점까지 최단 경로로 도달하는 RL 문제**입니다.

---

## 🎯 Goal

* 링크를 움직여 **최대한 적은 행동으로 목표 지점 도달**
* 실패 시: 목표와의 거리 최소화

---

## ⚙️ Requirements

* Python 3.12.x

---

## 📦 Installation

```bash
pip install -U kymnasium
```

---

## 🕹️ Environment 생성

```python
import gymnasium as gym
import kymnasium

env = gym.make(
    id='kymnasium/ZeldaAdventure-Stage-3',
    render_mode='human',   # or "rgb_array"
    bgm=True
)
```

---

# 📥 Observation (핵심)

에이전트는 다음 딕셔너리 형태로 상태를 받습니다:

```python
{
    "link": np.array([...]),
    "tiles": np.array([...])
}
```

---

## 🧍 link (플레이어 상태)

```python
(link_x, link_y, sword, direction)
```

### 🔹 sword (검 상태)

| 값 | 의미  |
| - | --- |
| 0 | 없음  |
| 1 | 청색  |
| 2 | 적색  |
| 3 | 녹색  |
| 4 | 비취색 |
| 5 | 보라색 |
| 6 | 황색  |

### 🔹 direction (방향)

| 값 | 의미 |
| - | -- |
| 0 | 좌  |
| 1 | 상  |
| 2 | 우  |
| 3 | 하  |

---

## 🧱 tiles (환경 객체)

각 행:

```python
(x, y, object_id, attribute)
```

---

## 🧩 Object ID

| 객체   | ID |
| ---- | -- |
| 바닥   | -1 |
| 벽    | 0  |
| 연못   | 1  |
| 불    | 2  |
| 링크   | 3  |
| 목표   | 4  |
| 구름   | 5  |
| 검    | 6  |
| 터틀너크 | 7  |
| 키이스  | 11 |
| 모리블린 | 12 |
| 아모스  | 13 |

---

## 🎨 Attribute (속성)

### 링크

* 검 색상 (1~6), 없으면 0

### 구름

| 값   | 상태      |
| --- | ------- |
| 0   | 완전히 사라짐 |
| 1~3 | 등장 중    |
| 4~6 | 사라지는 중  |

➡️ **0, 6 상태에서만 통과 가능**

### 검 / 몬스터

* 1~6: 색상

### 기타

* 0

---

# 🎮 Action Space

| 값 | 행동    |
| - | ----- |
| 0 | 정지    |
| 1 | 좌회전   | -> 실제 우회전
| 2 | 우회전   | -> 실제 좌회전
| 3 | 전진    |
| 4 | 검 줍기  |
| 5 | 검 버리기 |
| 6 | 공격    |

---

## ⚔️ 공격 규칙

* 맨손 공격 가능
* 검 사용 시 더 빠름
* 같은 색 검 → 더 빠름
* 터틀너크: 검 필요
* 아모스: 같은 색 검 필요

---

## 🏁 Episode 종료 조건

* 목표 도달
* 연못 또는 불 사망
* 1000 step 초과

---

## 🎁 Reward

* 기본 보상 없음 → 직접 설계

---

# 🧠 Agent 구현

```python
import gymnasium as gym
import kymnasium as kym
from typing import Any, Dict

class YourAgent(kym.Agent):
    def act(self, observation: Any, info: Dict):
        pass

    @classmethod
    def load(cls, path: str):
        pass

    def save(self, path: str):
        pass


def train():
    env = gym.make(
        id='kymnasium/ZeldaAdventure-Stage-3',
        render_mode='human',
        bgm=True
    )

    # 학습 코드 작성
```

---

# 🏆 Evaluation

```python
import kymnasium as kym

agent = ...
kym.evaluate(
    env_id='kymnasium/ZeldaAdventure-Stage-3',
    agent=agent,
    bgm=True
)
```

---

# 🔑 핵심 요약

* Observation = `link + tiles`
* `link`: 플레이어 상태
* `tiles`: 전체 환경 객체
* reward 없음 → 설계 중요
* 목표: **최소 step으로 탈출**

---

# version bfs

# version 1

- obs를 매번 불러와 dict 생성하지 않는다.
    - tiles 한 번만 dict 생성
        - dict에는 링크, 목표, 바닥, 벽만 남긴다
            - 연못, 불, 구름, 터틀너크, 키이스, 모리블린, 아모스 -> 모두 벽으로 취급
            - 검 -> 바닥으로 취급
    - link 상태만 업데이트 (상대적으로)

- action은 7가지 중 전진, 왼쪽 회전, 오른쪽 회전만 사용
    - 앞이 벽이면 전진 금지

- 방문했던 곳 다시 방문하면 패널티

- 학습 모델 q-leaning

# version 2

- obs를 매번 전부 새로 dict 생성하지 않는다.
    - 정적 타일은 한 번만 dict / grid 생성
    - dict에는 링크, 목표, 바닥, 벽만 남긴다
        - 연못, 불, 터틀너크, 키이스, 모리블린, 아모스 -> 모두 벽으로 취급
        - 검 -> 바닥으로 취급
    - 구름은 동적 객체로 따로 처리
        - 매 step마다 tiles에서 구름의 (x, y, attribute)만 읽어서 업데이트
        - 구름 attribute
            - 0, 6 -> 통과 가능
            - 1~5 -> 막힘
    - link 상태만 업데이트 (상대적으로)

- state는 다음으로 사용
    - (x, y, direction, goal_fb, goal_lr, front_blocked)
    - front_blocked
        - 앞칸이 정적 벽이면 1
        - 앞칸이 구름인데 현재 막혀 있으면 1
        - 그 외는 0
    - action은 7가지 중 전진, 왼쪽 회전, 오른쪽 회전만 사용
        - 앞이 막혀 있으면 action 후보에서 전진 제거
        - 앞이 열린 구름이면 전진 가능
        - 앞이 닫힌 구름이면 전진 금지
- 방문했던 곳 다시 방문하면 패널티
    - 필요하면 (x, y) 기준
- reward는 version 1과 비슷하게 설계
    - 기본 step penalty
    - 회전 penalty
    - 재방문 penalty
    - goal 방향 / distance 감소 보상
    - 닫힌 구름 앞에서 기다리도록 유도할 필요가 있으면
        - 무조건 회전만 유리하지 않게 보상 조정

- 학습 모델 q-learning

# version 3

- version 2 + action:0 (정지)

# version 4

- action: 검 줍기, 검 버리기, 공격 추가
    - 앞에 검이 있을 때 줍기 시도
    - 검을 소지하고 있을 때만 검 버리기 시도
    - 앞에 몬스터가 있을 때만 공격
        - 몬스터: 맨손: 검: 색깔 검
        - Keese: 10 / 4 / 1
        - Moblin: 20 / 12 / 3
        - Turtlenack: 불가 / 20 / 5
        - Armos: 불가 / 불가 / 5
    - FRONT_SWORD는 forward 허용
    - pickup은 현재 칸에서만
        - state: (x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword)

- 공격 규칙
    * 맨손 공격 가능
    * 검 사용 시 더 빠름
    * 같은 색 검 → 더 빠름
    * 터틀너크: 검 필요
    * 아모스: 같은 색 검 필요

- 공격에 대한 보상
    - 맨손 공격 < 검 공격 < 같은 색 공격

# version 5

    - state: (x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, sword_fb, sword_lr)

# version 6
    - state: (x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, sword_color_num)