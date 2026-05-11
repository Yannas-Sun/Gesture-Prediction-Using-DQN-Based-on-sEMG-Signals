"""Dataset-backed environment for sEMG gesture prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


class EMGEnvironment:
    """Wrap a recorded sEMG sequence as a simple classification environment."""

    def __init__(
        self,
        file_path: str | Path,
        window_size: int = 50,
        max_samples: int | None = None,
        channels: list[int] | None = None,
    ) -> None:
        data = scipy.io.loadmat(file_path, variable_names=["emg", "restimulus"])
        self.emg = np.asarray(data["emg"], dtype=np.float32)
        self.labels = np.asarray(data["restimulus"], dtype=np.int64).reshape(-1)

        if max_samples is not None:
            self.emg = self.emg[:max_samples]
            self.labels = self.labels[:max_samples]

        if channels is not None:
            self.emg = self.emg[:, channels]

        mean = np.mean(self.emg, axis=0)
        std = np.std(self.emg, axis=0) + 1e-8
        self.emg = (self.emg - mean) / std

        self.window_size = window_size
        self.idx = 0
        self.n_samples = self.emg.shape[0]
        self.num_channels = self.emg.shape[1]
        self.actions = np.unique(self.labels)
        self.num_actions = int(self.actions.max()) + 1

    def reset(self) -> np.ndarray:
        self.idx = 0
        return self.emg[self.idx : self.idx + self.window_size]

    def step(self, action: int, step_size: int = 1) -> tuple[np.ndarray, float, bool]:
        current_label_idx = self.idx + self.window_size - 1
        if current_label_idx >= self.n_samples - 1:
            return np.zeros_like(self.emg[0 : self.window_size]), 0.0, True

        true_label = int(self.labels[current_label_idx])
        reward = 1.0 if action == true_label else -1.0

        self.idx += step_size
        next_state = self.emg[self.idx : self.idx + self.window_size]
        done = self.idx + self.window_size >= self.n_samples

        return next_state, reward, done
