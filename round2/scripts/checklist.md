# Avoid Blurp 환경 확인 Checklist

## 실행 전 확인

- [ ] Python 버전이 `3.12.x`인지 확인
- [ ] `kymnasium`, `gymnasium`, `numpy`, `torch` 설치 확인
- [ ] `python scripts/check_env.py` 실행 가능 여부 확인
- [ ] `render_mode=None`, `bgm=False`로 환경 생성 가능 여부 확인

## Observation 확인

- [ ] observation이 `dict` 형태인지 확인
- [ ] `mario` key가 존재하는지 확인
- [ ] `mario` shape가 길이 5인지 확인
- [ ] `blurps` key가 존재하는지 확인
- [ ] `blurps` shape가 `(30, 6)`인지 확인
- [ ] 보이지 않는 Blurp 행이 0으로 채워지는지 확인

## Step / 종료 조건 확인

- [ ] action space가 `Discrete(3)`인지 확인
- [ ] action `0`, `1`, `2`가 모두 실행 가능한지 확인
- [ ] 기본 reward가 대부분 `0`으로 나오는지 확인
- [ ] `info["time_elapsed"]`가 step마다 증가하는지 확인
- [ ] 충돌 종료 시 `truncated=True`로 나오는지 확인
- [ ] 2분 생존 성공 시 `terminated=True`로 나오는지 확인

## 제출 전 확인

- [ ] `python src/train_v1.py` 실행 후 `avoid_blurp_dqn.pt` 생성 확인
- [ ] `YourAgent.load("avoid_blurp_dqn.pt")`로 모델 로드 확인
- [ ] `act(observation, info)`가 항상 `int` action을 반환하는지 확인
- [ ] 평가 코드에서 환경 수정 없이 실행되는지 확인
- [ ] 학습 시작 시 `CUDA available: True`가 출력되는지 확인
- [ ] vectorized env가 `num_envs=8`로 생성되는지 확인
