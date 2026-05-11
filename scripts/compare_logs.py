"""Plot comparison charts from training CSV logs."""

from __future__ import annotations

import argparse

from gesture_dqn.comparison import plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQN training logs.")
    parser.add_argument("--folder", default="outputs", help="Folder containing training CSV files.")
    parser.add_argument("--smoothing-window", type=int, default=10, help="Rolling average window.")
    parser.add_argument("--accuracy-only", action="store_true", help="Only plot accuracy.")
    args = parser.parse_args()

    output = plot_comparison(args.folder, args.smoothing_window, args.accuracy_only)
    print(f"Comparison plot saved to: {output}")


if __name__ == "__main__":
    main()
