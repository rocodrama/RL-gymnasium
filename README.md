# KNU RL Competition Workspace

이 저장소는 라운드별 에이전트 학습 코드를 관리하는 작업 공간입니다.

## 라운드 구성 (총 4개 기준)
1. Round 0: GridWorld-Crossing
2. Round 1: Zelda Adventure Stage 3
3. Round 2: (추가 예정)
4. Round 3: (추가 예정)

## 현재 폴더 상태
- `round0`: 구현 완료 (`train.py`, `round0_agent.pkl`, `requirements.txt`)
- `round1`: 구현 진행/정리 완료 (`train.py`, `train_fastenv.py`, `run.py`, `README.md`)
- `round2`: 폴더 미생성
- `round3`: 폴더 미생성

## 라운드별 문서
- [round0 README](round0/README.md)
- [round1 README](round1/README.md)

## 빠른 시작
### Round 0
```bash
cd round0
python train.py
```

### Round 1
```bash
cd round1
python train_fastenv.py   # 빠른 학습
# 또는
python train.py           # 제출용 학습
# 평가
python run.py
```

## 참고
- Round 2/3이 추가되면 동일한 형식으로 각 폴더에 `README.md`를 두고,
  루트 문서에 링크를 확장하면 됩니다.
