from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from version1 import (
    EstimatorConfig,
    LockingRunner,
    MPCConfig,
    MPCController,
    PhaseEstimator,
    RunnerConfig,
    SimulationBackend,
)


def test_simulation_reaches_target_efficiency():
    n_channels = 4
    rate = 5_000.0
    amplitudes = [1.0 for _ in range(n_channels)]
    phase_per_volt = [3.2 for _ in range(n_channels)]
    backend = SimulationBackend(
        n_channels=n_channels,
        rate_hz=rate,
        amplitudes=amplitudes,
        phase_per_volt=phase_per_volt,
        noise_level=0.0005,
        drift_rate=0.1,
        seed=42,
    )
    controller = MPCController(
        MPCConfig(
            n_channels=n_channels,
            horizon=12,
            dt=1.0 / rate,
            q_weight=2.0,
            r_weight=0.05,
            voltage_limits=(-1.2, 1.2),
            phase_per_volt=phase_per_volt,
            integral_gain=0.05,
            integral_limit=0.4,
        )
    )
    estimator = PhaseEstimator(
        EstimatorConfig(
            n_channels=n_channels,
            dt=1.0 / rate,
            phase_per_volt=phase_per_volt,
            measurement_noise=1e-3,
            process_noise=5e-4,
            innovation_clip_sigma=0.0,
        ),
        amplitudes,
    )
    runner = LockingRunner(
        backend=backend,
        controller=controller,
        estimator=estimator,
        config=RunnerConfig(
            duration=0.3,
            log_interval=10_000,
            efficiency_target=0.95,
            efficiency_std_target=0.015,
            rate_hz=rate,
            enable_console_log=False,
            measurement_smoothing=0.0,
        ),
        plotter=None,
    )
    runner.run()
    summary = runner.summary()
    assert summary["efficiency_mean"] >= 0.95
    assert summary["intensity_noise"] <= 0.015
    assert "rin_peak_db" in summary
    assert "rin_peak_freq_hz" in summary
    assert summary["rin_peak_freq_hz"] >= 0


class SpikySimulationBackend(SimulationBackend):
    def __init__(self, *args, spike_interval: int = 50, spike_height: float = 100.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._counter = 0
        self._spike_interval = spike_interval
        self._spike_height = spike_height

    def read_int1(self) -> float:
        base = super().read_int1()
        self._counter += 1
        if self._spike_interval > 0 and self._counter % self._spike_interval == 0:
            return base + self._spike_height
        return base


def test_smoothing_and_outlier_rejection_limits_spikes():
    n_channels = 3
    rate = 5_000.0
    amplitudes = [1.0 for _ in range(n_channels)]
    phase_per_volt = [3.0 for _ in range(n_channels)]
    backend = SpikySimulationBackend(
        n_channels=n_channels,
        rate_hz=rate,
        amplitudes=amplitudes,
        phase_per_volt=phase_per_volt,
        noise_level=0.001,
        drift_rate=0.05,
        seed=123,
        spike_interval=40,
        spike_height=80.0,
    )
    controller = MPCController(
        MPCConfig(
            n_channels=n_channels,
            horizon=10,
            dt=1.0 / rate,
            q_weight=1.5,
            r_weight=0.05,
            voltage_limits=(-1.0, 1.0),
            phase_per_volt=phase_per_volt,
            integral_gain=0.08,
            integral_limit=0.3,
        )
    )
    estimator = PhaseEstimator(
        EstimatorConfig(
            n_channels=n_channels,
            dt=1.0 / rate,
            phase_per_volt=phase_per_volt,
            measurement_noise=2e-3,
            process_noise=5e-4,
            innovation_clip_sigma=2.5,
        ),
        amplitudes,
    )
    runner = LockingRunner(
        backend=backend,
        controller=controller,
        estimator=estimator,
        config=RunnerConfig(
            duration=0.25,
            log_interval=10_000,
            efficiency_target=0.93,
            efficiency_std_target=0.02,
            rate_hz=rate,
            enable_console_log=False,
            measurement_smoothing=0.25,
        ),
        plotter=None,
    )
    runner.run()
    summary = runner.summary()
    assert summary["efficiency_mean"] >= 0.93
    assert summary["intensity_noise"] <= 0.02
    assert "rin_peak_db" in summary
    assert "rin_peak_freq_hz" in summary
    assert summary["rin_peak_freq_hz"] >= 0
