# Gesture Prediction Using DQN Based on sEMG Signals

This repository implements a Deep Q-Network (DQN) workflow for gesture
prediction using surface electromyography (sEMG) signals. Gesture recognition is
framed as a sequential decision-making problem: an agent observes a sliding
window of multi-channel sEMG data and selects a gesture class as its action.

The original exploratory notebooks and experiment outputs are kept under
`Main/`. The reusable training code is organized as a Python package under
`src/gesture_dqn/`.

## Repository Layout

```text
config/
  default.yaml          Training, data, model, and output configuration
scripts/
  train.py              Train the DQN model from a YAML config
  compare_logs.py       Generate comparison plots from CSV training logs
src/gesture_dqn/
  comparison.py         Experiment-log comparison utilities
  config.py             YAML config loader
  environment.py        sEMG dataset environment wrapper
  models.py             1D CNN Q-network
  replay_buffer.py      DQN replay buffer
  training.py           Main DQN training loop
Main/
  main.ipynb            Original notebook implementation
  structure.ipynb       Data and sliding-window visualizations
  comparison.ipynb      Original experiment comparison notebook
  s1/                   Example NinaPro-style .mat data files
  data/, graph/         Existing experiment logs and plots
```

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Train

Run training with the default configuration:

```powershell
python scripts/train.py --config config/default.yaml
```

Training outputs are written to `outputs/` by default:

- `training_log_*.csv`
- `training_plot_*.pdf`
- `q_network_*.pt`

## Configuration

Edit `config/default.yaml` to change the dataset, input window, selected
channels, DQN hyperparameters, and output directory.

Important fields:

```yaml
data:
  file_path: Main/s1/S1_E1_A1.mat
  window_size: 50
  max_samples: 5000
  channels: [0, 1]
  step_size: 50

training:
  episodes: 500
  batch_size: 64
  learning_rate: 0.001
  gamma: 0.995
  epsilon_start: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01
```

Set `max_samples: null` to train on the full recording. Set `channels: null` to
use all available sEMG channels.

## Compare Logs

After running multiple experiments, compare all CSV logs in a folder:

```powershell
python scripts/compare_logs.py --folder outputs
```

Only plot accuracy:

```powershell
python scripts/compare_logs.py --folder outputs --accuracy-only
```

## Method Summary

- **State**: sliding window of recent sEMG samples
- **Action**: predicted gesture class
- **Reward**: `+1` for correct prediction, `-1` for incorrect prediction
- **Agent**: DQN with a 1D CNN Q-network
- **Stabilization**: replay buffer and target network

## Tutorial

A detailed Medium tutorial explaining DQN theory for bioengineers is available
here:

[Human-Level Control with Deep Q-Networks (DQN): A Guide for Bioengineers](https://medium.com/@1933476828/human-level-control-with-deep-q-networks-dqn-b9461a143ebb?postPublishedType=repub)
