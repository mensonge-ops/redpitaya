#!/usr/bin/env python3
"""Self-supervised deep-learning simulator for two-beam coherent locking.

This script implements a lightweight physics model for a two-path coherent
combining experiment with a single detector and a single actuator (a PZT
stack).  It also ships with a self-supervised reinforcement-learning controller
that can be trained entirely in simulation without access to the hardware.  The
model captures the following aspects of the lab setup described by the user:

* A Red Pitaya STEMlab 125-14 generates a ±1 V control signal.
* A 15× high-voltage driver maps ±1 V to ±15 V at the PZT, corresponding to a
  maximum stroke of about 20 µm (≈ 40 µm optical path change).
* The PZT exhibits a dominant mechanical resonance around 3 kHz, modeled as a
  lightly damped second-order system.
* Two coherent beams interfere on a single detector; only their summed
  intensity is observed.

The controller is trained with a REINFORCE policy-gradient loop that maximizes
mean detector intensity.  Because the reward is derived directly from the
single detector readout, no labelled data or explicit reference phase is
required.  The trained policy can later be exported and deployed on hardware or
used as an initialization for classic control algorithms.

Typical usage inside a local IDE such as PyCharm::

    # Run a fast smoke test
    python red --simulate --episodes 2 --steps 200

    # Train longer and save a plot of the learning curve
    python red --simulate --episodes 50 --steps 800 --plot progress.png

The code is intentionally dependency-light (only PyTorch and Matplotlib for
optional plotting).  When executed without ``--simulate`` the script simply
exits; hardware bindings would be added in place of the simulated plant.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.distributions import Normal

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


@dataclasses.dataclass
class PlantConfig:
    """Physical parameters of the simulated interferometer and PZT."""

    wavelength_m: float = 1.064e-6  # 1064 nm typical for Yb lasers
    visibility: float = 0.9
    detector_noise: float = 0.02  # relative noise level
    drift_per_step: float = 0.01  # phase diffusion strength (rad)
    resonance_hz: float = 3_000.0
    damping: float = 0.05
    max_displacement_m: float = 20e-6
    hv_gain: float = 15.0  # high-voltage amplifier gain (V/V)
    max_dac_v: float = 1.0  # Red Pitaya output range
    step_rate_hz: float = 10_000.0  # simulated control loop rate


class PZTPlant:
    """Second-order PZT actuator with interferometric measurement."""

    def __init__(self, cfg: PlantConfig) -> None:
        self.cfg = cfg
        self._phase_drift = 0.0
        self._disp = 0.0
        self._vel = 0.0
        self._dt = 1.0 / cfg.step_rate_hz
        self._wn = 2 * math.pi * cfg.resonance_hz
        self._mass_scale = cfg.max_displacement_m / (cfg.hv_gain * cfg.max_dac_v)

    def reset(self) -> None:
        self._phase_drift = random.uniform(-math.pi, math.pi)
        self._disp = 0.0
        self._vel = 0.0

    def step(self, dac_v: float) -> Tuple[float, float]:
        """Advance the plant using a clamped DAC drive voltage.

        Args:
            dac_v: Commanded DAC voltage in the ±1 V range.
        Returns:
            A tuple ``(intensity, phase)`` where ``intensity`` is normalized to
            1.0 for the constructive interference case.
        """
        drive_v = max(-self.cfg.max_dac_v, min(self.cfg.max_dac_v, dac_v))
        drive_disp = drive_v * self.cfg.hv_gain * self._mass_scale

        # Discrete-time update of a damped second-order oscillator.
        # x¨ + 2ζω_n x˙ + ω_n² x = ω_n² u
        accel = self._wn ** 2 * (drive_disp - self._disp) - 2 * self.cfg.damping * self._wn * self._vel
        self._vel += accel * self._dt
        self._disp += self._vel * self._dt

        # Phase is twice displacement because of the round trip optical path.
        wavelength = self.cfg.wavelength_m
        phase = 4 * math.pi * self._disp / wavelength

        # Inject slow environmental drift.
        self._phase_drift += random.gauss(0.0, self.cfg.drift_per_step)
        total_phase = phase + self._phase_drift

        signal = 1 + self.cfg.visibility * math.cos(total_phase)
        noise = random.gauss(0.0, self.cfg.detector_noise)
        intensity = max(0.0, signal + noise)
        return intensity, total_phase


class Policy(nn.Module):
    """Small MLP policy that outputs a DAC command in volts."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(-0.7))

    def forward(self, obs: torch.Tensor) -> Normal:
        mu = self.net(obs).squeeze(-1)
        std = torch.exp(self.log_std).clamp(0.05, 0.5)
        return Normal(mu, std)


def rollout(env: PZTPlant, policy: Policy, steps: int) -> Tuple[List[torch.Tensor], List[float]]:
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    env.reset()
    obs = torch.tensor([0.0, 0.0], dtype=torch.float32)

    for _ in range(steps):
        dist = policy(obs)
        action = dist.sample()
        dac_v = float(torch.tanh(action))  # enforce ±1 V bounds
        log_probs.append(dist.log_prob(action))

        intensity, phase = env.step(dac_v)
        rewards.append(intensity)
        obs = torch.tensor([intensity, math.sin(phase)], dtype=torch.float32)

    return log_probs, rewards


def train(env: PZTPlant, policy: Policy, episodes: int, steps: int, lr: float) -> List[float]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0
    curve: List[float] = []

    for ep in range(episodes):
        log_probs, rewards = rollout(env, policy, steps)
        ep_return = sum(rewards) / len(rewards)
        baseline = 0.9 * baseline + 0.1 * ep_return if ep > 0 else ep_return
        adv = torch.tensor(rewards, dtype=torch.float32) - baseline

        loss = -(torch.stack(log_probs) * adv).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curve.append(ep_return)
        if ep % max(1, episodes // 10) == 0:
            print(f"Episode {ep:4d}: mean intensity={ep_return:.3f} baseline={baseline:.3f}")
    return curve


def plot_curve(curve: Iterable[float], path: Path) -> None:
    if plt is None:
        print("Matplotlib is unavailable; skipping plot.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.plot(list(curve))
    plt.xlabel("Episode")
    plt.ylabel("Mean normalized intensity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved plot to {path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true", help="Run the self-supervised simulation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=400, help="Steps per episode (control loop iterations)")
    parser.add_argument("--lr", type=float, default=3e-3, help="Policy learning rate")
    parser.add_argument("--plot", type=str, default="", help="Optional path to save learning-curve plot")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.simulate:
        print("Simulation flag not set; nothing to do yet. Use --simulate to run the trainer.")
        return 0

    cfg = PlantConfig()
    env = PZTPlant(cfg)
    policy = Policy()

    curve = train(env, policy, episodes=args.episodes, steps=args.steps, lr=args.lr)

    if args.plot:
        plot_curve(curve, Path(args.plot))

    print("Training finished. Example inference step:")
    env.reset()
    obs = torch.tensor([0.0, 0.0], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(5):
            dac_v = float(torch.tanh(policy(obs).mean))
            intensity, phase = env.step(dac_v)
            obs = torch.tensor([intensity, math.sin(phase)], dtype=torch.float32)
            print(f"DAC={dac_v:+.3f} V, intensity={intensity:.3f}, phase={phase:.2f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
