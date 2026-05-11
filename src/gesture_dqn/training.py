"""Training loop for the DQN gesture predictor."""

from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

import torch
from torch import nn, optim

from gesture_dqn.environment import EMGEnvironment
from gesture_dqn.models import QNetwork
from gesture_dqn.replay_buffer import ReplayBuffer


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict[str, Any]) -> dict[str, list[float]]:
    data_config = config["data"]
    train_config = config["training"]
    model_config = config.get("model", {})
    output_config = config.get("output", {})

    set_seed(int(train_config.get("seed", 42)))
    device = resolve_device(str(train_config.get("device", "auto")))

    env = EMGEnvironment(
        file_path=data_config["file_path"],
        window_size=int(data_config["window_size"]),
        max_samples=data_config.get("max_samples"),
        channels=data_config.get("channels"),
    )

    q_net = QNetwork(
        num_channels=env.num_channels,
        num_actions=env.num_actions,
        conv1_channels=int(model_config.get("conv1_channels", 32)),
        conv2_channels=int(model_config.get("conv2_channels", 64)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
    ).to(device)
    target_net = QNetwork(
        num_channels=env.num_channels,
        num_actions=env.num_actions,
        conv1_channels=int(model_config.get("conv1_channels", 32)),
        conv2_channels=int(model_config.get("conv2_channels", 64)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=float(train_config["learning_rate"]))
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(int(train_config["buffer_capacity"]))

    episodes = int(train_config["episodes"])
    batch_size = int(train_config["batch_size"])
    gamma = float(train_config["gamma"])
    epsilon = float(train_config["epsilon_start"])
    epsilon_min = float(train_config["epsilon_min"])
    epsilon_decay = float(train_config["epsilon_decay"])
    step_size = int(data_config["step_size"])
    target_update_interval = int(train_config.get("target_update_interval", 10))

    history: dict[str, list[float]] = {
        "episode": [],
        "total_reward": [],
        "accuracy": [],
        "epsilon": [],
    }

    print(f"Training on {device}; actions={env.num_actions}, channels={env.num_channels}")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, env.num_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(torch.argmax(q_net(state_tensor)).item())

            next_state, reward, done = env.step(action, step_size)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device)

                q_current = q_net(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(next_states_t).max(1)[0]
                    q_target = rewards_t + gamma * q_next * (1 - dones_t)

                loss = loss_fn(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        accuracy = (total_reward + step_count) / (2 * step_count) * 100 if step_count else 0.0
        history["episode"].append(float(episode + 1))
        history["total_reward"].append(total_reward)
        history["accuracy"].append(accuracy)
        history["epsilon"].append(epsilon)

        print(
            f"Episode {episode + 1}/{episodes}, "
            f"Reward: {total_reward:.1f}, Acc: {accuracy:.2f}%, Eps: {epsilon:.3f}"
        )

    output_dir = Path(output_config.get("directory", "outputs"))
    save_training_outputs(output_dir, config, history, q_net, bool(output_config.get("save_checkpoint", True)))
    return history


def save_training_outputs(
    output_dir: Path,
    config: dict[str, Any],
    history: dict[str, list[float]],
    model: nn.Module,
    save_checkpoint: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    csv_path = output_dir / f"training_log_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["--- Configuration ---"])
        for section, values in config.items():
            writer.writerow([section, values])
        writer.writerow([])
        writer.writerow(["--- Training Data ---"])
        writer.writerow(["Episode", "Total Reward", "Accuracy (%)", "Epsilon"])
        writer.writerows(
            zip(
                history["episode"],
                history["total_reward"],
                history["accuracy"],
                history["epsilon"],
            )
        )

    pdf_path = output_dir / f"training_plot_{timestamp}.pdf"
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history["episode"], history["accuracy"], label="Accuracy", color="blue", linewidth=2)
    plt.title("Training Accuracy per Episode")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history["episode"], history["total_reward"], label="Total Reward", color="green", linewidth=2)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()

    if save_checkpoint:
        checkpoint_path = output_dir / f"q_network_{timestamp}.pt"
        torch.save(model.state_dict(), checkpoint_path)

    print(f"CSV log saved to: {csv_path}")
    print(f"PDF plot saved to: {pdf_path}")
