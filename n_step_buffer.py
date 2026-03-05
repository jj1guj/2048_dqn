from collections import deque
from replay_buffer import Experience

class NStepBuffer:
    """n-step遷移を計算するための一時バッファ"""
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque(maxlen=n)

    def add(self, experience):
        self.buffer.append(experience)

    def is_ready(self):
        return len(self.buffer) == self.n

    def get(self):
        """n-step遷移を計算して返す"""
        # n-step累積報酬: r_0 + γr_1 + γ²r_2 + ... + γ^(n-1)r_{n-1}
        reward = 0
        for i, exp in enumerate(self.buffer):
            reward += (self.gamma ** i) * exp.reward
            if exp.terminated:
                # 途中で終了した場合はそこまでの累積報酬を使う
                return Experience(
                    state=self.buffer[0].state,
                    action=self.buffer[0].action,
                    reward=reward,
                    next_state=exp.next_state,
                    terminated=True
                )
        # n-step先の状態
        return Experience(
            state=self.buffer[0].state,
            action=self.buffer[0].action,
            reward=reward,
            next_state=self.buffer[-1].next_state,
            terminated=False
        )

    def flush(self):
        """エピソード終了時に残りのバッファを全て処理"""
        results = []
        while len(self.buffer) > 0:
            reward = 0
            for i, exp in enumerate(self.buffer):
                reward += (self.gamma ** i) * exp.reward
                if exp.terminated:
                    results.append(Experience(
                        state=self.buffer[0].state,
                        action=self.buffer[0].action,
                        reward=reward,
                        next_state=exp.next_state,
                        terminated=True
                    ))
                    break
            else:
                results.append(Experience(
                    state=self.buffer[0].state,
                    action=self.buffer[0].action,
                    reward=reward,
                    next_state=self.buffer[-1].next_state,
                    terminated=False
                ))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()
