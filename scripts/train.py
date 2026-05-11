"""Train the DQN gesture predictor from a YAML config."""

from __future__ import annotations

import argparse

from gesture_dqn.config import load_config
from gesture_dqn.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN model for sEMG gesture prediction.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
