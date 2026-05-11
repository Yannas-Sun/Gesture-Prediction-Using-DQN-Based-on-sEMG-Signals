"""Utilities for comparing training logs."""

from __future__ import annotations

import ast
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_log_file(filepath: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Read one training CSV with a configuration block and training data."""
    path = Path(filepath)
    params: dict[str, str] = {}
    data_start_line = 0

    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if "--- Training Data ---" in stripped:
            data_start_line = index + 1
            break
        if "," in stripped and "---" not in stripped:
            key, value = stripped.split(",", 1)
            params[key.strip()] = value.strip().strip('"')

    df = pd.read_csv(path, skiprows=data_start_line)
    return params, df


def plot_comparison(folder_path: str | Path = ".", smoothing_window: int = 10, accuracy_only: bool = False) -> Path:
    """Compare all training CSV files in a folder and save a PDF chart."""
    folder = Path(folder_path)
    csv_files = glob.glob(str(folder / "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    experiments = []
    for csv_file in csv_files:
        params, df = parse_log_file(csv_file)
        if not df.empty:
            experiments.append({"filename": Path(csv_file).name, "params": params, "df": df})

    diff_keys = _find_varying_keys([experiment["params"] for experiment in experiments])

    if accuracy_only:
        output = folder / "accuracy_comparison_only.pdf"
        plt.figure(figsize=(10, 6))
        axes = [plt.gca()]
    else:
        output = folder / "hyperparameters_comparison.pdf"
        _, axes_tuple = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        axes = list(axes_tuple)

    colors = plt.cm.tab10.colors
    for index, experiment in enumerate(experiments):
        df = experiment["df"]
        label = _make_label(experiment["filename"], experiment["params"], diff_keys)
        color = colors[index % len(colors)]
        _plot_metric(axes[-1], df, "Accuracy (%)", label, color, smoothing_window)
        if not accuracy_only:
            _plot_metric(axes[0], df, "Total Reward", label, color, smoothing_window)

    if accuracy_only:
        axes[0].set_title(f"Comparison: Accuracy (Smoothed window={smoothing_window})")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_xlabel("Episode")
    else:
        axes[0].set_title(f"Comparison: Total Reward (Smoothed window={smoothing_window})")
        axes[0].set_ylabel("Total Reward")
        axes[1].set_title(f"Comparison: Accuracy (Smoothed window={smoothing_window})")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_xlabel("Episode")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.6)
        axis.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    return output


def _find_varying_keys(param_sets: list[dict[str, str]]) -> list[str]:
    all_keys = set().union(*(params.keys() for params in param_sets))
    varying = []
    for key in sorted(all_keys):
        values = {params.get(key, "N/A") for params in param_sets}
        if len(values) > 1:
            varying.append(key)
    return varying


def _make_label(filename: str, params: dict[str, str], diff_keys: list[str]) -> str:
    if not diff_keys:
        return filename
    parts = []
    for key in diff_keys:
        value = params.get(key, "N/A")
        try:
            value = str(ast.literal_eval(value))
        except (SyntaxError, ValueError):
            pass
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _plot_metric(axis, df: pd.DataFrame, metric: str, label: str, color, smoothing_window: int) -> None:
    axis.plot(df["Episode"], df[metric], color=color, alpha=0.18)
    if len(df) > smoothing_window:
        smoothed = df[metric].rolling(window=smoothing_window).mean()
        axis.plot(df["Episode"], smoothed, color=color, label=label, linewidth=2)
    else:
        axis.plot(df["Episode"], df[metric], color=color, label=label, linewidth=2)
