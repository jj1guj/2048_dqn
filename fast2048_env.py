from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fast2048 import apply_action, empty_positions, exponents_to_onehot, legal_actions, spawn_tile


class Fast2048Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size: int = 4, max_pow: int = 16):
        if size != 4:
            raise ValueError("Fast2048Env supports only size=4")
        self.size = size
        self.max_pow = max_pow
        self.board = [0] * 16

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, 4, max_pow), dtype=np.uint8
        )

    def _spawn_random_tile(self) -> None:
        empties = empty_positions(self.board)
        if not empties:
            return
        index = int(self.np_random.integers(0, len(empties)))
        pos = empties[index]
        value_exp = 1 if float(self.np_random.random()) < 0.9 else 2
        self.board = spawn_tile(self.board, pos, value_exp)

    def _max_exp(self) -> int:
        return max(self.board)

    def _is_terminal(self) -> bool:
        legal = legal_actions(self.board)
        if legal == [0]:
            _, _, moved = apply_action(self.board, 0)
            return not moved
        return False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = [0] * 16
        self._spawn_random_tile()
        self._spawn_random_tile()

        obs = exponents_to_onehot(self.board)
        info = {
            "is_legal": True,
            "max": self._max_exp(),
        }
        return obs, info

    def step(self, action: int):
        action = int(action)
        next_board, reward, moved = apply_action(self.board, action)

        if moved:
            self.board = next_board
            self._spawn_random_tile()

        terminated = self._is_terminal()
        truncated = False

        obs = exponents_to_onehot(self.board)
        info = {
            "is_legal": moved,
            "max": self._max_exp(),
        }
        return obs, float(reward), terminated, truncated, info
