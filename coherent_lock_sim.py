#!/usr/bin/env python3
"""Simulate two-channel coherent beam combining with a learned controller.

The script provides a small end-to-end loop that mimics a single-detector,
single-actuator coherent combining setup.  A PyTorch policy is trained with a
reward equal to the measured interferometric intensity, so the controller learns
how to maximize fringe contrast without ever observing the true optical phase.

Key hardware parameters from the question are reflected in the simulation:
- One detector that measures the combined intensity with optional shot-like noise.
- One PZT actuator with 20 µm travel (40 µm optical path), 3 kHz resonance and a
  ±1 V Red Pitaya output that is amplified 15× by the PZT driver.

Run a quick training demo (CPU only) with::

    python coherent_lock_sim.py --episodes 3 --episode-length 800

Use ``--plot`` to visualize the last episode when experimenting locally.  The
code is self contained; install dependencies with ``pip install -r
requirements.txt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclasses.dataclass
class PZTSpec:
    """Mechanical and electrical limits for the actuator."""

    travel_um: float = 20.0
    optical_path_um: float = 40.0
    resonance_hz: float = 3000.0
    max_voltage: float = 150.0
    drive_gain: float = 15.0
    redpitaya_limit_v: float = 1.0
    wavelength_um: float = 1.064
    sample_rate_hz: float = 10000.0

    @property
    def volts_per_um(self) -> float:
        return self.max_voltage / self.travel_um

    @property
    def commanded_range_v(self) -> float:
        return self.redpitaya_limit_v * self.drive_gain

    @property
    def um_per_input_voltage(self) -> float:
        return self.travel_um / self.max_voltage * self.drive_gain

    @property
    def optical_phase_per_volt(self) -> float:
        opd_per_v = self.um_per_input_voltage * 2  # round trip path
        return 2 * math.pi * opd_per_v / self.wavelength_um


class PZTPlant:
    """Simplified first-order model of the PZT stack and driver."""

    def __init__(self, spec: PZTSpec) -> None:
        self.spec = spec
        self.position_um = 0.0
        self.command_history: List[float] = []
        # Convert resonance to an approximate discrete-time pole.
        tau = 1 / (2 * math.pi * spec.resonance_hz)
        self.alpha = math.exp(-1 / (spec.sample_rate_hz * tau))

    def reset(self) -> None:
        self.position_um = 0.0
        self.command_history.clear()

    def step(self, command_v: float) -> Tuple[float, float]:
        command_v = float(np.clip(command_v, -self.spec.redpitaya_limit_v, self.spec.redpitaya_limit_v))
        target_position = command_v * self.spec.um_per_input_voltage
        self.position_um = self.alpha * self.position_um + (1 - self.alpha) * target_position
        self.command_history.append(command_v)
        optical_phase = 2 * math.pi * (self.position_um * 2) / self.spec.wavelength_um
        return self.position_um, optical_phase


@dataclasses.dataclass
class BeamCombinerConfig:
    visibility: float = 0.95
    noise_std: float = 0.02
    phase_drift_std: float = 0.06
    sample_rate_hz: float = 10000.0


class CoherentCombinerEnv:
    """Environment producing noisy intensity measurements from two beams."""

    def __init__(self, pzt: PZTPlant, config: BeamCombinerConfig) -> None:
        self.pzt = pzt
        self.config = config
        self.time = 0
        self.free_phase = 0.0

    def reset(self) -> np.ndarray:
        self.time = 0
        self.free_phase = float(np.random.uniform(-math.pi, math.pi))
        self.pzt.reset()
        intensity = self._measure_intensity(0.0)
        return np.array([intensity, 0.0], dtype=np.float32)

    def _measure_intensity(self, actuator_phase: float) -> float:
        total_phase = self.free_phase + actuator_phase
        shot_like_noise = np.random.normal(scale=self.config.noise_std)
        intensity = 0.5 * (1 + self.config.visibility * math.cos(total_phase))
        return float(np.clip(intensity + shot_like_noise, 0.0, 1.1))

    def step(self, command_v: float) -> Tuple[np.ndarray, float, float]:
        _, actuator_phase = self.pzt.step(command_v)
        self.free_phase += float(np.random.normal(scale=self.config.phase_drift_std))
        intensity = self._measure_intensity(actuator_phase)
        observation = np.array([intensity, command_v], dtype=np.float32)
        self.time += 1
        return observation, intensity, (self.free_phase + actuator_phase) % (2 * math.pi)


class PolicyNet(nn.Module):
    """Tiny controller producing an analog output in the ±1 V range."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def run_episode(
    env: CoherentCombinerEnv,
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    episode_length: int,
    gamma: float,
    action_std: float,
    device: torch.device,
) -> Tuple[float, float, List[float]]:
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    log_probs: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    phases: List[float] = []

    for _ in range(episode_length):
        mean_action = policy(obs)
        dist = Normal(mean_action, action_std)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        log_probs.append(dist.log_prob(action))

        next_obs_np, intensity, phase = env.step(float(action.cpu()))
        obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        rewards.append(torch.tensor(intensity, device=device))
        phases.append(phase)

    returns: List[torch.Tensor] = []
    running_return = torch.tensor(0.0, device=device)
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    returns = torch.stack(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    policy_loss = -(torch.stack(log_probs) * returns).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return float(torch.stack(rewards).mean().cpu()), float(policy_loss.detach().cpu()), phases


def train(
    episodes: int,
    episode_length: int,
    gamma: float,
    action_std: float,
    lr: float,
    plot: bool,
    seed: int,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    spec = PZTSpec()
    env = CoherentCombinerEnv(PZTPlant(spec), BeamCombinerConfig(sample_rate_hz=spec.sample_rate_hz))
    policy = PolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    device = torch.device("cpu")

    history = []
    for episode in range(episodes):
        mean_reward, loss, phases = run_episode(
            env, policy, optimizer, episode_length=episode_length, gamma=gamma, action_std=action_std, device=device
        )
        history.append(mean_reward)
        print(f"Episode {episode + 1:3d}/{episodes} | mean intensity {mean_reward:.3f} | loss {loss:.3f}")

    if plot:
        plot_training(history, phases, spec)


def plot_training(history: Iterable[float], phases: List[float], spec: PZTSpec) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)

    axes[0].plot(history, marker="o")
    axes[0].set_title("Mean intensity per episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Normalized intensity")
    axes[0].set_ylim(0, 1.1)

    if phases:
        axes[1].plot(np.unwrap(phases))
        axes[1].set_title("Residual optical phase during last episode")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Phase [rad]")
        axes[1].axhline(0, color="k", lw=0.5)
    else:
        axes[1].text(0.5, 0.5, "No phase history recorded", ha="center")

    axes[1].text(0.02, 0.9, f"Stroke: ±{spec.commanded_range_v:.1f} V @ Red Pitaya", transform=axes[1].transAxes)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--episode-length", type=int, default=800, help="Steps per episode")
    parser.add_argument("--gamma", type=float, default=0.98, help="Return discount factor")
    parser.add_argument("--action-std", type=float, default=0.12, help="Exploration stddev for the policy")
    parser.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate")
    parser.add_argument("--plot", action="store_true", help="Display plots for the last episode")
    parser.add_argument("--seed", type=int, default=3, help="Reproducibility seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        episodes=args.episodes,
        episode_length=args.episode_length,
        gamma=args.gamma,
        action_std=args.action_std,
        lr=args.lr,
        plot=args.plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
