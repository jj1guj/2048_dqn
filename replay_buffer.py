from collections import deque, namedtuple
import numpy as np
import random
import torch

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'terminated'])

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sampling(self):
        batch_data = random.sample(self.buffer, self.batch_size)
        batch = Experience(*zip(*batch_data))

        state = torch.tensor(np.stack(batch.state))
        action = torch.tensor(np.array(batch.action, dtype=np.int64))
        reward = torch.tensor(np.array(batch.reward, dtype=np.float32))
        next_state = torch.tensor(np.stack(batch.next_state))
        terminated = torch.tensor(np.array(batch.terminated, dtype=np.int32))
        return state, action, reward, next_state, terminated
