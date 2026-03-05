import numpy as np
import torch
from replay_buffer import Experience


class SumTree:
    """優先度付きサンプリング用のセグメントツリー"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 内部ノード + 葉
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self._update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def update(self, tree_idx, priority):
        self._update(tree_idx, priority)

    def get(self, s):
        """累積和sに対応する葉ノードを返す"""
        idx = 0
        while idx < self.capacity - 1:  # 内部ノードの間
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=50000):
        self.batch_size = batch_size
        self.alpha = alpha          # 優先度の指数 (0=均一, 1=完全優先)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 0
        self.epsilon = 1e-6         # 優先度が0にならないように
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0     # 新規データのデフォルト優先度

    def __len__(self):
        return self.tree.size

    def beta(self):
        """betaを線形にアニーリング"""
        return min(self.beta_end,
                   self.beta_start + (self.beta_end - self.beta_start) * self.frame / self.beta_frames)

    def add(self, experience):
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sampling(self):
        self.frame += 1
        batch_size = self.batch_size
        indices = []
        experiences = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                # まれにNoneが返る場合のフォールバック
                s = np.random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            experiences.append(data)
            priorities.append(priority)

        # Importance Sampling 重み
        beta = self.beta()
        total = self.tree.total()
        n = len(self)
        priorities = np.array(priorities, dtype=np.float64)
        probs = priorities / total
        weights = (n * probs) ** (-beta)
        weights = weights / weights.max()  # 正規化

        batch = Experience(*zip(*experiences))
        states = torch.tensor(np.stack(batch.state))
        actions = torch.tensor(np.array(batch.action, dtype=np.int64))
        rewards = torch.tensor(np.array(batch.reward, dtype=np.float32))
        next_states = torch.tensor(np.stack(batch.next_state))
        terminateds = torch.tensor(np.array(batch.terminated, dtype=np.int32))
        weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, terminateds, indices, weights

    def update_priorities(self, indices, td_errors):
        """TD誤差に基づいて優先度を更新"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
