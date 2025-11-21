#!/usr/bin/env python3
"""Simulation-driven unsupervised locking for two-beam coherent combination.

This script builds a light-weight plant model for a single-PZT coherent
combining setup and trains a small policy network with a REINFORCE-style
update to keep the interference maximum.  It is designed for local
experimentation (for example inside PyCharm) without any Red Pitaya
hardware attached.

Hardware model
--------------
* Red Pitaya STEMlab 125-14 analog output limited to ``\u00b11 V``.
* PZT controller with ``15x`` voltage gain and ``150 V`` maximum drive.
* PZT mechanical stroke of ``20 \u03bcm`` (equivalent optical path change
  of ``40 \u03bcm``) and a ``3 kHz`` mechanical resonance.
* A single photodetector returns the combined intensity.  The simulated
  detector includes shot noise and drifting relative phase.

Learning approach
-----------------
The controller is a small multilayer perceptron implemented with only the
standard library (no third-party dependencies).  It outputs the mean of a
Gaussian policy for the commanded Red Pitaya voltage.  The policy is
updated with one-step REINFORCE using the measured intensity as the
reward, so no phase labels are required.  Exploration noise keeps the
optimizer away from flat regions and allows it to climb the interference
fringe.

Quick start
-----------
Run a short simulation and training loop (defaults are fast enough for a
laptop) and print a textual summary::

    python coherent_combiner_sim.py --episodes 5 --steps 400

To observe the lock quality over time and save a plot instead of trying
live display (useful in headless environments)::

    python coherent_combiner_sim.py --episodes 10 --steps 600 --save-plot locking.png

Configuration is exposed through command-line flags so you can tune the
PZT model, photonic parameters and learning hyper-parameters without
editing the code.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
import random
import sys
from typing import Dict, Iterable, List, Sequence, Tuple


# ---------------------- Math helpers (no NumPy needed) ---------------------
def clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def matmul_rowvec(vec: Sequence[float], mat: Sequence[Sequence[float]]) -> List[float]:
    return [sum(vec[i] * mat[i][j] for i in range(len(vec))) for j in range(len(mat[0]))]


def vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[i] + b[i] for i in range(len(a))]


def vec_tanh(vec: Sequence[float]) -> List[float]:
    return [math.tanh(x) for x in vec]


def vec_mul(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [a[i] * b[i] for i in range(len(a))]


def outer(vec_a: Sequence[float], vec_b: Sequence[float]) -> List[List[float]]:
    return [[vec_a[i] * vec_b[j] for j in range(len(vec_b))] for i in range(len(vec_a))]


def zeros_like(mat: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[0.0 for _ in row] for row in mat]


def zeros_vec(size: int) -> List[float]:
    return [0.0 for _ in range(size)]


def add_inplace(mat: List[List[float]], grad: Sequence[Sequence[float]]) -> None:
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] += grad[i][j]


# ----------------------------- Plant models -------------------------------
@dataclasses.dataclass
class PZTConfig:
    """Mechanical and electrical parameters for the simulated PZT."""

    resonance_hz: float = 3000.0
    damping: float = 0.22
    max_displacement_um: float = 20.0  # mechanical stroke
    max_voltage: float = 150.0  # controller output capability
    amplifier_gain: float = 15.0  # voltage gain from Red Pitaya to driver
    input_range_v: float = 1.0  # Red Pitaya output swing
    sample_rate_hz: float = 20_000.0


class PZTPlant:
    """Simple second-order PZT model with saturation and resonance."""

    def __init__(self, cfg: PZTConfig):
        self.cfg = cfg
        self.omega = 2 * math.pi * cfg.resonance_hz
        self.dt = 1.0 / cfg.sample_rate_hz
        self.reset()

    def reset(self) -> None:
        self.displacement_um = 0.0
        self.velocity = 0.0

    def step(self, control_v: float) -> float:
        drive_v = clip(control_v * self.cfg.amplifier_gain, -self.cfg.max_voltage, self.cfg.max_voltage)
        target_disp_um = (drive_v / self.cfg.max_voltage) * self.cfg.max_displacement_um

        acc = (self.omega**2) * (target_disp_um - self.displacement_um) - 2 * self.cfg.damping * self.omega * self.velocity
        self.velocity += acc * self.dt
        self.displacement_um += self.velocity * self.dt
        return self.displacement_um


@dataclasses.dataclass
class CombinerConfig:
    wavelength_m: float = 1064e-9
    visibility: float = 0.95
    intensity_noise: float = 0.02
    drift_std: float = 0.03  # rad per step


class CoherentCombinerEnv:
    """Environment returning a noisy intensity measurement."""

    def __init__(self, pzt_cfg: PZTConfig, comb_cfg: CombinerConfig):
        self.pzt = PZTPlant(pzt_cfg)
        self.cfg = comb_cfg
        self.reset()

    def reset(self) -> List[float]:
        self.pzt.reset()
        self.phase_bias = random.uniform(-math.pi, math.pi)
        self.drift = 0.0
        self.last_command = 0.0
        return self._observation(0.5)

    def _observation(self, measurement: float) -> List[float]:
        return [measurement, math.sin(self.phase_bias + self.drift), math.cos(self.phase_bias + self.drift), self.last_command]

    def step(self, command_v: float) -> Tuple[List[float], float]:
        self.last_command = clip(command_v, -1.0, 1.0)
        displacement = self.pzt.step(self.last_command)

        phase_shift = 4 * math.pi * (displacement * 1e-6) / self.cfg.wavelength_m
        self.drift += random.gauss(0.0, self.cfg.drift_std)
        total_phase = self.phase_bias + self.drift + phase_shift

        ideal = 0.5 * (1 + self.cfg.visibility * math.cos(total_phase))
        noisy_measurement = clip(random.gauss(ideal, self.cfg.intensity_noise), 0.0, 1.2)
        return self._observation(noisy_measurement), noisy_measurement


# ------------------------- Manual neural policy ---------------------------
@dataclasses.dataclass
class PolicyConfig:
    obs_size: int = 4
    hidden: int = 32
    action_std: float = 0.2
    learning_rate: float = 0.02
    decay: float = 0.98


class GaussianPolicy:
    """Small MLP with manual backprop for a 1D Gaussian policy."""

    def __init__(self, cfg: PolicyConfig, seed: int = 42):
        random.seed(seed)
        self.cfg = cfg
        self.params: Dict[str, List[List[float]] | List[float]] = {
            "w1": [[random.gauss(0.0, 0.3) for _ in range(cfg.hidden)] for _ in range(cfg.obs_size)],
            "b1": zeros_vec(cfg.hidden),
            "w2": [[random.gauss(0.0, 0.3) for _ in range(cfg.hidden)] for _ in range(cfg.hidden)],
            "b2": zeros_vec(cfg.hidden),
            "w3": [[random.gauss(0.0, 0.3)] for _ in range(cfg.hidden)],
            "b3": [0.0],
        }

    def forward(self, obs: List[float]) -> Tuple[float, Dict[str, List[float] | List[List[float]]]]:
        z1 = vec_add(matmul_rowvec(obs, self.params["w1"]), self.params["b1"])  # type: ignore[arg-type]
        a1 = vec_tanh(z1)
        z2 = vec_add(matmul_rowvec(a1, self.params["w2"]), self.params["b2"])  # type: ignore[arg-type]
        a2 = vec_tanh(z2)
        z3 = vec_add(matmul_rowvec(a2, self.params["w3"]), self.params["b3"])  # type: ignore[arg-type]
        mu = math.tanh(z3[0])
        cache = {"obs": obs, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
        return mu, cache

    def backward(self, cache: Dict[str, List[float]], dmu: float) -> Dict[str, List[List[float]] | List[float]]:
        dz3 = [dmu * (1 - math.tanh(cache["z3"][0]) ** 2)]
        da2 = [dz3[0] * self.params["w3"][i][0] for i in range(len(self.params["w3"]))]
        dw3 = [[cache["a2"][i] * dz3[0]] for i in range(len(cache["a2"]))]
        db3 = dz3

        dz2 = [da2[i] * (1 - math.tanh(cache["z2"][i]) ** 2) for i in range(len(cache["z2"]))]
        da1 = [sum(dz2[j] * self.params["w2"][i][j] for j in range(len(dz2))) for i in range(len(self.params["w2"]))]
        dw2 = [[cache["a1"][i] * dz2[j] for j in range(len(dz2))] for i in range(len(cache["a1"]))]
        db2 = dz2

        dz1 = [da1[i] * (1 - math.tanh(cache["z1"][i]) ** 2) for i in range(len(cache["z1"]))]
        dw1 = [[cache["obs"][i] * dz1[j] for j in range(len(dz1))] for i in range(len(cache["obs"]))]
        db1 = dz1

        return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}

    def apply_gradients(self, grads: Dict[str, List[List[float]] | List[float]], lr: float) -> None:
        for key, grad in grads.items():
            if isinstance(self.params[key][0], list):  # type: ignore[index]
                for i in range(len(self.params[key])):  # type: ignore[arg-type]
                    for j in range(len(self.params[key][0])):  # type: ignore[arg-type]
                        self.params[key][i][j] += lr * grad[i][j]  # type: ignore[index]
            else:
                for i in range(len(self.params[key])):  # type: ignore[arg-type]
                    self.params[key][i] += lr * grad[i]  # type: ignore[index]


def reinforce_update(policy: GaussianPolicy, trajectory: List[Tuple[Dict[str, List[float]], float, float]], cfg: PolicyConfig) -> None:
    decay = cfg.decay
    baseline = 0.0
    grads: Dict[str, List[List[float]] | List[float]] = {
        "w1": zeros_like(policy.params["w1"]),
        "b1": zeros_vec(len(policy.params["b1"])),
        "w2": zeros_like(policy.params["w2"]),
        "b2": zeros_vec(len(policy.params["b2"])),
        "w3": zeros_like(policy.params["w3"]),
        "b3": zeros_vec(len(policy.params["b3"])),
    }

    for cache, action, reward in trajectory:
        baseline = decay * baseline + (1 - decay) * reward
        advantage = reward - baseline
        mu = math.tanh(cache["z3"][0])
        dlogp_dmu = (action - mu) / (cfg.action_std**2)
        step_grads = policy.backward(cache, dmu=advantage * dlogp_dmu)
        for key in grads:
            if isinstance(grads[key], list) and grads[key] and isinstance(grads[key][0], list):  # type: ignore[index]
                add_inplace(grads[key], step_grads[key])  # type: ignore[arg-type]
            else:
                for i in range(len(grads[key])):  # type: ignore[arg-type]
                    grads[key][i] += step_grads[key][i]  # type: ignore[index]

    scale = cfg.learning_rate / max(1, len(trajectory))
    policy.apply_gradients(grads, lr=scale)


def run_episode(env: CoherentCombinerEnv, policy: GaussianPolicy, steps: int, cfg: PolicyConfig) -> Tuple[List[float], List[float]]:
    obs = env.reset()
    rewards: List[float] = []
    actions: List[float] = []
    trajectory: List[Tuple[Dict[str, List[float]], float, float]] = []

    for _ in range(steps):
        mu, cache = policy.forward(obs)
        action = random.gauss(mu, cfg.action_std)
        action = clip(action, -1.0, 1.0)
        obs, reward = env.step(action)
        trajectory.append((cache, action, reward))
        rewards.append(reward)
        actions.append(action)

    reinforce_update(policy, trajectory, cfg)
    return rewards, actions


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate coherent combining with unsupervised learning")
    parser.add_argument("--episodes", type=int, default=8, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps per episode")
    parser.add_argument("--save-plot", type=pathlib.Path, default=None, help="Optional path to save a PNG plot instead of displaying")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting entirely")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    return parser.parse_args(list(argv))


def maybe_plot(intensities: List[List[float]], actions: List[List[float]], save_path: pathlib.Path | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Skipping plot because matplotlib is unavailable: {exc}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for idx, (ints, acts) in enumerate(zip(intensities, actions)):
        ax1.plot(ints, label=f"episode {idx}")
        ax2.plot(acts, label=f"episode {idx}")

    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(loc="lower right")
    ax2.set_ylabel("Command (\u00b11 V)")
    ax2.set_xlabel("Step")
    ax2.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    env = CoherentCombinerEnv(PZTConfig(), CombinerConfig())
    policy = GaussianPolicy(PolicyConfig(), seed=args.seed)

    episode_rewards: List[List[float]] = []
    episode_actions: List[List[float]] = []
    avg_rewards: List[float] = []

    cfg = PolicyConfig()
    for ep in range(args.episodes):
        rewards, actions = run_episode(env, policy, steps=args.steps, cfg=cfg)
        episode_rewards.append(rewards)
        episode_actions.append(actions)
        avg_rewards.append(sum(rewards) / len(rewards))
        print(f"Episode {ep:02d}: mean intensity={avg_rewards[-1]:.3f} | recent command={actions[-1]:+.3f} V")

    print(f"Final average reward over {args.episodes} episodes: {sum(avg_rewards) / len(avg_rewards):.3f}")

    if not args.no_plot:
        maybe_plot(episode_rewards, episode_actions, save_path=args.save_plot)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
