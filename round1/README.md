# 2026-01 KNU RL Competition Round 1

## Zelda's Adventure

## 1. 개요
- 링크를 조작해 목표 지점까지 도달하는 강화학습 문제입니다.
- 핵심 목표는 **성공률 유지 + step 최소화**입니다.

## 2. 목표
- 가능한 적은 행동 수로 Goal 도달
- 실패 시 Goal까지의 거리 최소화

## 3. 요구사항
- Python 3.12.x

## 4. 설치
```bash
pip install -U kymnasium
```

## 5. 환경 생성
```python
import gymnasium as gym
import kymnasium

env = gym.make(
    id="kymnasium/ZeldaAdventure-Stage-3",
    render_mode="human",  # 또는 "rgb_array"
    bgm=True,
)
```

## 6. Observation
- observation은 다음 딕셔너리 형태입니다.
```python
{
    "link": np.array([...]),
    "tiles": np.array([...]),
}
```

### 6.1 `link`
```python
(link_x, link_y, sword, direction)
```
- `sword`: 0(없음), 1~6(검 색)
- `direction`: 0(좌), 1(상), 2(우), 3(하)

### 6.2 `tiles`
```python
(x, y, object_id, attribute)
```

### 6.3 Object ID
- 바닥: -1
- 벽: 0
- 연못: 1
- 불: 2
- 링크: 3
- 목표: 4
- 구름: 5
- 검: 6
- Turtlenack: 7
- Keese: 11
- Moblin: 12
- Armos: 13

### 6.4 Attribute
- 링크: 검 색(없으면 0)
- 구름: 0, 6일 때 통과 가능 / 1~5는 막힘
- 검/몬스터: 색(1~6)

## 7. Action Space
- 0: STOP
- 1: TURN_RIGHT
- 2: TURN_LEFT
- 3: FORWARD
- 4: PICKUP
- 5: DROP
- 6: ATTACK

## 8. 중요 주의사항 (회전 반전)
- 환경 액션 기준으로 보면,
  - `action=1 (TURN_RIGHT)`가 실제로는 **왼쪽 회전처럼** 동작하고,
  - `action=2 (TURN_LEFT)`가 실제로는 **오른쪽 회전처럼** 동작합니다.
- 코드에서는 이를 보정하기 위해 Q-인덱스와 실제 액션을 분리 매핑해서 사용합니다.

## 9. 전투 규칙 요약
- 맨손 공격 가능(비효율)
- 같은 색 검일수록 타격 효율 높음
- Turtlenack: 검 필요
- Armos: 같은 색 검 필요

## 10. 에피소드 종료 조건
- Goal 도달
- 연못/불 충돌
- 최대 step 초과(기본 1000)

## 11. 버전 히스토리 요약
- `v9`는 실험 후 폐기되어 현재 기준선에서 제외했습니다.

### 공통 안내
- 아래 state는 모두 **Q-table key**로 사용한 튜플 기준입니다.

### v1
- state: `(x, y, direction, goal_fb, goal_lr)`
- 정적 맵 기반 BFS 거리 shaping 시작
- 이동/회전 중심의 기본 Q-learning
- 검/몬스터를 거의 벽처럼 취급하는 단순 접근

### v2
- state: `(x, y, direction, goal_fb, goal_lr, front_blocked)`
- 구름을 동적 객체로 분리 처리
- `front_blocked` 상태 반영
- 구름 타이밍 대기 고려

### v3
- state: `(x, y, direction, goal_fb, goal_lr, front_type)`
- `STOP` 액션 추가
- 구름 타이밍 대기 행동을 정책에 포함

### v4
- state: `(x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword)`
- `PICKUP / DROP / ATTACK` 도입
- 몬스터별 필요 타격 수 규칙 반영
- 전투 가능/불가능 매치업 반영

### v5
- state: `(x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, sword_fb, sword_lr, current_tile_sword)`
- 검 관련 상태 확장(`sword_fb`, `sword_lr`)
- 드롭/교체 로직 강화
- 맨손 공격 및 무의미 공격 제약 강화

### v6
- state: `(x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, current_tile_sword, sword_color_num)`
- `sword_fb`, `sword_lr` 제거
- `sword_color_num`(중복 없는 검 색 수집 수) 도입
- 교체 시퀀스(`drop -> forward -> pickup`) 정리
- 검 탐색/색 수집 보상 강화

### v7
- state: `(x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, current_tile_sword, sword_color_num, current_sword_used)`
- 과한 보상(특정 몬스터 가산, 특정 색 검 가산) 롤백
- BFS 갱신 시점을 몬스터 처치 + 검 상태 변경으로 확대
- 현재 검과 같은 색 몬스터를 BFS에서 통과 가능 취급

### v8
- state: `(x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, current_tile_sword, sword_color_num, current_sword_used)`
- 3검 local optimum 탈출(핵심 안정화 버전)
- 탐색 재가열(reheat), 정체 감지 epsilon boost
- 5색 집중 단계 + goal rush 억제
- 새 색 검 교체 예외(soft)로 교체 deadlock 완화

### final
- state: `(x, y, direction, sword, goal_fb, goal_lr, front_type, front_attr, current_has_sword, current_tile_sword, sword_color_num, current_sword_used)`
- 최종 실험 기준은 **v8 계열 리팩토링**
- `train_fastenv.py`: FastEnv 기반 학습용
- `train.py`: 제출용(실환경 step) 학습용
- `run.py`: 학습된 에이전트 평가 실행용

## 12. 실행 파일 안내
- 빠른 학습: `python train_fastenv.py`
- 제출용 학습: `python train.py`
- 평가 실행: `python run.py`
