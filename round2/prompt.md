Python 3.12, gymnasium, kymnasium, numpy, torch 기반으로 
kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1 환경을 학습하는 
MLP Double-Dueling DQN 코드를 작성해줘.

환경 특징:
- observation은 dict
- obs["mario"] shape=(5,)
  [left, top, right, bottom, vx]
- obs["blurps"] shape=(30, 7)
  [left, top, right, bottom, vx, vy, ay]
- action space:
  0 = 정지
  1 = 왼쪽 이동
  2 = 오른쪽 이동
- 기본 reward는 0이므로 reward shaping 필요
- 충돌 시 truncated=True
- 2분 생존 성공 시 terminated=True
- 학습 시 render_mode="none"
- 평가/시연 시 render_mode="human"

구현 요구사항:
1. kymnasium.Agent를 상속한 YourAgent 클래스 구현
   - act(self, observation, info)
   - save(self, path)
   - load(cls, path)

2. state preprocessing 구현
   - mario feature:
     - mario center x normalized
     - mario vx normalized
     - left wall distance
     - right wall distance
   - blurp feature:
     - active 여부
     - dx, dy
     - vx, vy, ay
     - time_to_hit_mario_top
     - predicted_blurp_x
     - predicted_mario_x = mario_x + mario_vx * t_hit
     - future_dx = predicted_blurp_x - predicted_mario_x
     - risk score
   - active blurp 중 risk 높은 top_k=10개만 사용
   - 부족하면 zero padding
   - 최종 state는 float32 numpy vector

3. 물리 계산:
   - blurp y position:
     y(t) = y0 + vy*t + 0.5*ay*t^2
   - mario top에 도달하는 t_hit 계산
   - t_hit > 0 중 가장 작은 값 사용
   - x position:
     x(t) = x0 + vx*t

4. reward shaping:
   - 매 step 생존 보상 +0.02
   - truncated=True and not terminated이면 -10
   - terminated=True이면 +100
   - risk_next penalty: -0.30 * risk_next
   - risk 감소 보상: +0.15 * (risk_now - risk_next)
   - 위험할 때 올바른 회피 방향 action이면 +0.05
   - 위험할 때 현재 mario vx가 회피 방향이면 +0.04
   - 벽 가까우면 -0.03
   - 속도 너무 크면 작은 penalty
   - reward는 [-10, 100] 범위로 clip

5. DQN 구조:
   - MLP encoder: state_dim -> 256 -> 256
   - Dueling head:
     - value head: 256 -> 128 -> 1
     - advantage head: 256 -> 128 -> 3
   - Q = V + A - mean(A)

6. 학습 알고리즘:
   - Double DQN
   - ReplayBuffer size 300000 이상
   - batch_size=256
   - gamma=0.99
   - lr=1e-4
   - target network hard update every 2000 learner steps
   - train_freq=4
   - warmup_steps=10000
   - epsilon greedy:
     - start 1.0
     - end 0.05
     - decay over 300000 environment steps
   - gradient clipping 10.0

7. 8개 병렬 env 사용:
   - gym.vector.AsyncVectorEnv 또는 SyncVectorEnv 사용
   - num_envs=8
   - 각 env는 render_mode="none"
   - vector env에서 obs dict batch를 받아 각 env별 state로 변환
   - transition 저장:
     state[i], action[i], reward[i], next_state[i], done[i]
   - done = terminated or truncated
   - done된 env는 자동 reset 또는 final observation 처리 고려

8. 저장:
   - 학습 완료 후 agent.save("avoid_blurp_dqn.pt")
   - 모델 state_dict, state_dim, action_dim, top_k 저장

9. 실행 모드:
   - train() 함수
   - evaluate_once(path="avoid_blurp_dqn.pt") 함수
   - if __name__ == "__main__": train()

10. 제출 가능하게 하나의 .py 파일로 작성