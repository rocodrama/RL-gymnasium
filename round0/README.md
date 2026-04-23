# 2026-01 KNU RL Competition Round 0

## GridWorld-Crossing-26x26

## 1. 개요
- `kymnasium/GridWorld-Crossing-26x26` 환경에서 Q-learning으로 에이전트를 학습합니다.
- 에이전트는 회전/전진만으로 용암을 피하고 Goal에 도달해야 합니다.

## 2. 환경 정보
- 환경 ID: `kymnasium/GridWorld-Crossing-26x26`
- observation: `26x26` 격자 (`numpy.ndarray`)
- 격자 값(코드 기준):
  - `FLOOR = 100`
  - `WALL = 250`
  - `LAVA = 900`
  - `GOAL = 810`
  - 에이전트 방향 타일: `RIGHT=1000`, `DOWN=1001`, `LEFT=1002`, `UP=1003`

## 3. Action Space
- `0`: 왼쪽 회전 (`ACTION_LEFT`)
- `1`: 오른쪽 회전 (`ACTION_RIGHT`)
- `2`: 전진 (`ACTION_FORWARD`)

## 4. 상태(State) 정의
- Q-table key:
```python
(row, col, direction, front_cell, left_cell, right_cell)
```
- 의미:
  - `row, col`: 에이전트 좌표
  - `direction`: 현재 바라보는 방향
  - `front_cell`: 정면 한 칸의 셀 값
  - `left_cell`: 왼쪽 한 칸의 셀 값
  - `right_cell`: 오른쪽 한 칸의 셀 값

## 5. 학습 설정 (`train.py` 기준)
- `episodes = 100000`
- `max_steps = 500`
- `alpha = 0.1`
- `gamma = 0.99`
- `epsilon_start = 1.0`
- `epsilon_end = 0.05`
- `epsilon_decay = 0.9995`

## 6. 보상 설계 요약
- 기본 step 페널티: `-0.1`
- 회전 페널티: `-0.02`
- 전진 시:
  - 벽 정면: `-2.0`
  - 용암 정면: `-3.0`
  - 목표 정면: `+10.0`
  - 바닥 정면: `+0.05`
- 같은 칸 재방문: `-0.2`
- 종료 보정:
  - Goal 도달: `+50.0`
  - Lava 종료: `-50.0`
  - max_steps 초과 종료: `-10.0`

## 7. 실행 방법
```bash
cd round0
python train.py
```

## 8. 평가(run)
`train.py`의 `run()`은 `round0_agent.pkl`을 로드합니다.

```bash
cd round0
python -c "import train; train.run()"
```

## 9. 파일명 주의사항
- 현재 코드상 저장은 `round_0.pkl`, 평가는 `round0_agent.pkl`을 로드합니다.
- 학습 후 바로 `run()`을 사용할 때 파일명이 다르면 로드 오류가 날 수 있습니다.

## 10. 의존성
- `round0/requirements.txt` 참고
