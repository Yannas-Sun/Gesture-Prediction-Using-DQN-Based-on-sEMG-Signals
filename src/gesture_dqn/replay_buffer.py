"""Experience replay buffer used by DQN training."""

from __future__ import annotations

import collections
import random
from typing import Deque

import numpy as np


Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, tuple[int, ...], tuple[float, ...], np.ndarray, tuple[bool, ...]]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self) -> int:
        return len(self.buffer)
