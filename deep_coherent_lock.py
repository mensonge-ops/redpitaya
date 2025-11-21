r"""Local simulation of two-beam coherent combining with an unsupervised controller.

The script models a single PZT actuator driven from a Red Pitaya STEMlab 125-14
through a 15x high-voltage amplifier.  The actuator length is 20 µm (40 µm
optical path) with a 3 kHz mechanical resonance.  The controller uses a small
recurrent neural network trained end-to-end to maximize the detected combined
intensity without any labelled phase targets.

The default settings allow quick experimentation in PyCharm on purely simulated
measurements.  You can change noise levels, plant parameters, and training
hyperparameters via command-line options.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn, optim


@dataclass
class PlantConfig:
    """Physical parameters of the PZT and optical interferometer."""

    wavelength_nm: float = 1550.0  # Operating wavelength in nanometres
    actuator_range_um: float = 20.0  # Total stroke of the PZT
    optical_path_um: float = 40.0  # Corresponding optical path change
    controller_gain: float = 15.0  # High-voltage amplifier gain (±1 V -> ±15 V)
    controller_limit_v: float = 1.0  # Red Pitaya output limit
    resonance_hz: float = 3000.0  # Dominant mechanical resonance
    sample_rate_hz: float = 20000.0  # Simulation update rate
    visibility: float = 0.9  # Interference fringe visibility
    measurement_noise: float = 0.01  # Additive measurement noise (fractional)
    drift_per_step_rad: float = 0.01  # Random walk on the open-loop phase


class PZTSimulator:
    """Minimal PZT + interferometer model.

    The actuator is approximated by a first-order low-pass filter with its
    cutoff derived from the specified resonance.  The interferometer output is
    the normalised intensity :math:`I = 0.5(1 + V\\cos\\phi)`.
    """

    def __init__(self, config: PlantConfig) -> None:
        self.config = config
        self._phase = torch.tensor(0.0)
        self._open_loop_phase = torch.tensor(0.0)
        self._alpha = self._lowpass_alpha()

    def _lowpass_alpha(self) -> float:
        rc = 1.0 / (2 * math.pi * self.config.resonance_hz)
        dt = 1.0 / self.config.sample_rate_hz
        return float(dt / (rc + dt))

    def reset(self) -> None:
        self._phase = torch.tensor(random.uniform(-math.pi, math.pi))
        self._open_loop_phase = torch.tensor(random.uniform(-math.pi, math.pi))

    @property
    def drive_limit(self) -> float:
        return self.config.controller_limit_v

    def _drive_to_phase(self, drive_v: torch.Tensor) -> torch.Tensor:
        # Saturate at the Red Pitaya output
        drive_v = drive_v.clamp(-self.drive_limit, self.drive_limit)
        effective_v = drive_v * self.config.controller_gain
        displacement_um = (
            effective_v / (self.config.controller_gain * self.drive_limit)
        ) * (self.config.actuator_range_um / 2.0)
        optical_path_um = displacement_um * 2.0
        wavelength_um = self.config.wavelength_nm * 1e-3
        return 2 * math.pi * optical_path_um / wavelength_um

    def step(self, drive_v: torch.Tensor) -> torch.Tensor:
        commanded_phase = self._drive_to_phase(drive_v)
        self._phase = (1 - self._alpha) * self._phase + self._alpha * commanded_phase
        drift = torch.randn_like(self._open_loop_phase) * self.config.drift_per_step_rad
        self._open_loop_phase = (self._open_loop_phase + drift) % (2 * math.pi)
        total_phase = self._phase + self._open_loop_phase
        noise = torch.randn(()) * self.config.measurement_noise
        intensity = 0.5 * (1 + self.config.visibility * torch.cos(total_phase)) + noise
        return intensity.clamp(0.0, 1.0)


class RecurrentController(nn.Module):
    """GRU-based controller that outputs incremental PZT commands."""

    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh()
        )

    def forward(
        self, measurements: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # measurements: [T, B, 1]
        features, hidden_out = self.gru(measurements, hidden)
        drive = self.head(features)
        return drive.squeeze(-1), hidden_out


def run_epoch(
    plant: PZTSimulator,
    controller: RecurrentController,
    optimizer: optim.Optimizer,
    seq_len: int,
    device: torch.device,
    penalty: float,
) -> Tuple[float, float]:
    plant.reset()
    controller.train()
    optimizer.zero_grad()

    measurements: List[torch.Tensor] = []
    drives: List[torch.Tensor] = []
    hidden: torch.Tensor | None = None

    last_drive = torch.zeros(1, 1, device=device)
    for _ in range(seq_len):
        meas = plant.step(last_drive.detach().cpu()).to(device)
        meas = meas.view(1, 1, 1)
        drive, hidden = controller(meas, hidden)
        drive = drive.clamp(-1.0, 1.0)
        measurements.append(meas)
        drives.append(drive)
        last_drive = drive

    meas_tensor = torch.cat(measurements, dim=0)
    drives_tensor = torch.cat(drives, dim=0)

    mean_intensity = meas_tensor.mean()
    smooth_penalty = torch.diff(drives_tensor, dim=0).pow(2).mean()
    loss = -mean_intensity + penalty * smooth_penalty
    loss.backward()
    optimizer.step()

    return float(mean_intensity.item()), float(loss.item())


@torch.no_grad()
def simulate_closed_loop(
    plant: PZTSimulator,
    controller: RecurrentController,
    seq_len: int,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    plant.reset()
    controller.eval()
    hidden: torch.Tensor | None = None
    drive = torch.zeros(1, 1, device=device)
    intensities: List[float] = []
    drives: List[float] = []

    for _ in range(seq_len):
        meas = plant.step(drive.cpu()).to(device).view(1, 1, 1)
        drive, hidden = controller(meas, hidden)
        drive = drive.clamp(-1.0, 1.0)
        intensities.append(float(meas.item()))
        drives.append(float(drive.item()))
    return intensities, drives


def plot_results(intensities: List[float], drives: List[float]) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(intensities, label="Intensity")
    ax1.set_ylabel("Normalised intensity")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(drives, label="PZT drive (±1 V equiv.)")
    ax2.set_ylabel("Drive [V]")
    ax2.set_xlabel("Time step")
    ax2.grid(True)
    ax2.legend()
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate two-beam coherent combining with an unsupervised "
            "deep-learning controller."
        )
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--seq-len", type=int, default=400, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=3e-3, help="Optimizer learning rate")
    parser.add_argument(
        "--smooth-penalty", type=float, default=1e-2, help="Weight on drive slew penalty"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device identifier"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Plot intensity and drive after training"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    random.seed(42)

    config = PlantConfig()
    plant = PZTSimulator(config)
    controller = RecurrentController().to(device)
    optimizer = optim.Adam(controller.parameters(), lr=args.lr)

    history: List[Tuple[float, float]] = []
    for epoch in range(args.epochs):
        mean_intensity, loss = run_epoch(
            plant,
            controller,
            optimizer,
            seq_len=args.seq_len,
            device=device,
            penalty=args.smooth_penalty,
        )
        history.append((mean_intensity, loss))
        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:04d} | mean intensity={mean_intensity:.4f} "
                f"| loss={loss:.4f}"
            )

    intensities, drives = simulate_closed_loop(plant, controller, args.seq_len, device)
    print(
        f"Post-training average intensity: {sum(intensities) / len(intensities):.4f}"
    )

    if args.plot:
        plot_results(intensities, drives)

        import matplotlib.pyplot as plt

        epochs, losses = zip(*history)
        plt.figure(figsize=(6, 3))
        plt.plot([i for i, _ in enumerate(epochs, 1)], [l for _, l in history])
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
