import pickle  
import gymnasium as gym  
import kymnasium as kym  
import numpy as np 
from typing import Any, Dict  
import random  
from collections import defaultdict


# 환경 정의
FLOOR = 100   
WALL = 250    
LAVA = 900    
GOAL = 810    

class Round0Agent(kym.Agent):
    # 방향
    RIGHT = 1000
    DOWN = 1001
    LEFT = 1002
    UP = 1003
    
    Directions = {RIGHT, DOWN, LEFT, UP}

    # 행동
    ACTION_LEFT = 0     # 왼쪽으로 90도 회전
    ACTION_RIGHT = 1    # 오른쪽으로 90도 회전
    ACTION_FORWARD = 2  # 현재 방향으로 한 칸 이동

    Actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD]

    def __init__(self, q_table=None, epsilon=0.1):
        # (row, col, direction,front_cell,left_cell,right_cell):[Q_left, Q_right, Q_forward]
        self.q_table = q_table if q_table is not None else {}
        self.epsilon = epsilon

    # 왼쪽으로 90도 회전
    def turn_left(self, direction):
        if direction == self.RIGHT:
            return self.UP
        elif direction == self.UP:
            return self.LEFT
        elif direction == self.LEFT:
            return self.DOWN
        else:
            return self.RIGHT

    # 오른쪽으로 90도 회전
    def turn_right(self, direction):
        if direction == self.RIGHT:
            return self.DOWN
        elif direction == self.DOWN:
            return self.LEFT
        elif direction == self.LEFT:
            return self.UP
        else:
            return self.RIGHT

    # 현재 방향으로 한 칸 이동
    def forward_direction(self, direction):
        if direction == self.RIGHT:
            return 0, 1
        elif direction == self.DOWN:
            return 1, 0
        elif direction == self.LEFT:
            return 0, -1
        else:
            return -1, 0

    # 주변 칸 정보 추출
    def extract_nearby_cells(self, obs, row, col, direction):
        dr, dc = self.forward_direction(direction)
        nr, nc = row + dr, col + dc

        rows, cols = obs.shape

        # 벽
        if not (0 <= nr < rows and 0 <= nc < cols):
            return WALL

        value = int(obs[nr, nc])

        # 에이전트 방향 값이면 실제 바닥으로 처리
        if value in self.Directions:
            return FLOOR

        return value

    # 상태 추출
    def extract_state(self, observation):
        obs = np.asarray(observation)

        # 에이전트 위치 
        agent_cells = np.argwhere(np.isin(obs, list(self.Directions)))
        if len(agent_cells) == 0:
            return (0, 0, self.RIGHT, FLOOR, FLOOR, FLOOR)

        row, col = agent_cells[0]
        row, col = int(row), int(col)

        # 현재 방향
        direction = int(obs[row, col])

        # 현재 방향에서 좌,우 방향 계산
        left_dir = self.turn_left(direction)
        right_dir = self.turn_right(direction)

        # 주변 칸
        front_cell = self.extract_nearby_cells(obs, row, col, direction)
        left_cell = self.extract_nearby_cells(obs, row, col, left_dir)
        right_cell = self.extract_nearby_cells(obs, row, col, right_dir)

        # 최종 상태
        state = (
            row,
            col,
            direction,
            int(front_cell),
            int(left_cell),
            int(right_cell),
        )

        return state

    # 행동 선택
    def act(self, observation: Any, info: Dict):
        # 현재 상태 추출
        state = self.extract_state(observation)
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        # 탐험
        if random.random() < self.epsilon:
            return random.choice(self.Actions)

        # 최종 행동 선택
        q_values = self.q_table[state]
        max_q = max(q_values)

        # 동점 일 때
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]

        return random.choice(best_actions)

    @classmethod
    def load(cls, path: str) -> 'kym.Agent':
        with open(path, "rb") as f:
            data = pickle.load(f)

        return cls(q_table=data.get("q_table", {}), epsilon=0.0)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table}, f)
            
def train():
    episodes = 100000          # 전체 학습 에피소드 수
    max_steps = 500            # 한 에피소드 최대 행동 수
    alpha = 0.1                # 학습률
    gamma = 0.99               # 할인율
    epsilon_start = 1.0        # 초기 탐험 확률
    epsilon_end = 0.05         # 최소 탐험 확률
    epsilon_decay = 0.9995     # 에피소드마다 epsilon 감소율

    env = gym.make(
        id="kymnasium/GridWorld-Crossing-26x26",
        render_mode=None,
        bgm=False,
    )

    agent = Round0Agent(epsilon=epsilon_start)

    # 같은 칸 방문 횟수
    visit_counts = defaultdict(int)

    # ===== Q-table에 state 없으면 초기화 =====
    def ensure_state(state):
        if state not in agent.q_table:
            agent.q_table[state] = [0.0, 0.0, 0.0]

    success_count = 0
    recent_rewards = []

    # ===== 전체 학습 루프 =====
    for episode in range(1, episodes + 1):
        obs, info = env.reset()

        # observation을 agent가 정의한 방식으로 state로 변환
        state = agent.extract_state(obs)
        ensure_state(state)

        done = False
        steps = 0
        episode_reward = 0.0

        # 이번 에피소드에서 방문한 위치 기록
        episode_visited = set()

        while steps < max_steps and not done:
            row, col, direction, front_cell, left_cell, right_cell = state

            # 현재 위치 방문 기록
            visit_counts[(row, col)] += 1
            episode_visited.add((row, col))

            # 행동 선택
            action = agent.act(obs, info)

            next_obs, _, terminated, truncated, info = env.step(action)
            next_state = agent.extract_state(next_obs)
            ensure_state(next_state)

            next_row, next_col, next_direction, next_front, next_left, next_right = next_state

            # step 한 번
            reward = -0.1   

            # 회전 할 때
            if action in [agent.ACTION_LEFT, agent.ACTION_RIGHT]:
                reward -= 0.02

            # 전진
            if action == agent.ACTION_FORWARD:
                if front_cell == WALL:
                    reward -= 2.0
                elif front_cell == LAVA:
                    reward -= 3.0
                elif front_cell == GOAL:
                    reward += 10.0
                elif front_cell == FLOOR:
                    reward += 0.05

            # 같은 칸 방문
            if (next_row, next_col) in episode_visited:
                reward -= 0.2

            # 종료
            if terminated:
                if action == agent.ACTION_FORWARD and front_cell == GOAL:
                    reward += 50.0
                    success_count += 1
                elif action == agent.ACTION_FORWARD and front_cell == LAVA:
                    reward -= 50.0

            done = terminated or truncated

            # Q-learning 업데이트
            old_q = agent.q_table[state][action]
            best_next_q = max(agent.q_table[next_state])

            td_target = reward + (0.0 if done else gamma * best_next_q)
            agent.q_table[state][action] = old_q + alpha * (td_target - old_q)

            # 다음 상태
            obs = next_obs
            state = next_state
            episode_reward += reward
            steps += 1

        # 최대 스텝으로 게임 종료
        if steps >= max_steps and not done:
            episode_reward -= 10.0

        agent.epsilon = max(
            epsilon_end,
            epsilon_start * (epsilon_decay ** (episode - 1))
        )

        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        if episode % 500 == 0:
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            success_rate = sum(1 for r in recent_rewards if r > 20) / len(recent_rewards)

            print(
                f"[Episode {episode:5d}] "
                f"epsilon={agent.epsilon:.4f}, "
                f"avg_reward={avg_reward:8.2f}, "
                f"success_count={success_count}, "
                f"recent_success_rate={success_rate:.2f}"
            )

    agent.epsilon = 0.0
    agent.save("round_0.pkl")

    env.close()
    print("학습 완료: round_0.pkl 저장")
    
def run():
    agent = Round0Agent.load("round_0.pkl")
    
    kym.evaluate(
        env_id='kymnasium/GridWorld-Crossing-26x26',    
        agent=agent,
        bgm=True
    )
    
    
if __name__ == "__main__":
    train()