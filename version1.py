#!/usr/bin/env python3
"""Single-file MPC coherent synthesis locking launcher (版本1)."""
from __future__ import annotations

import math
import random
import socket
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Tuple

try:  # Optional dependency for live visualization
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib may be unavailable
    plt = None  # type: ignore

MDT694B_GAIN = 20.0  # Typical voltage gain of the MDT694B piezo driver


def _wrap_pi(value: float) -> float:
    """Wrap a phase angle to [-pi, pi)."""

    return (value + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class BackendState:
    """Represents the instantaneous state returned by a backend step."""

    timestamp: float
    int1: float
    out1: float
    out2: float
    channel_voltages: List[float]
    efficiency: float
    intensity_noise: float
    efficiency_std: float


class BaseBackend:
    """Abstract base class for Red Pitaya / simulation backends."""

    def __init__(self, n_channels: int, rate_hz: float) -> None:
        self.n_channels = n_channels
        self.rate_hz = rate_hz
        self.last_out1 = 0.0
        self.last_out2 = 0.0
        self.last_channel_voltages = [0.0 for _ in range(n_channels)]
        self.timestamp = time.monotonic()
        self._eff_history: List[float] = []
        self._int_history: List[float] = []
        self._max_intensity = 1.0

    def configure(self) -> None:
        """Perform backend-specific configuration."""

    def read_int1(self) -> float:  # pragma: no cover - abstract method
        raise NotImplementedError

    def apply_control(self, channel_voltages: List[float]) -> Tuple[float, float]:  # pragma: no cover - abstract method
        raise NotImplementedError

    def close(self) -> None:
        """Clean up resources (optional)."""

    def update_metrics(self, intensity: float) -> Tuple[float, float, float]:
        self._int_history.append(intensity)
        if len(self._int_history) > int(self.rate_hz):
            self._int_history.pop(0)
        mean_intensity = mean(self._int_history) if self._int_history else intensity
        std_intensity = pstdev(self._int_history) if len(self._int_history) > 1 else 0.0
        intensity_noise = std_intensity / mean_intensity if mean_intensity else 0.0

        efficiency = intensity / self._max_intensity if self._max_intensity else 0.0
        self._eff_history.append(efficiency)
        if len(self._eff_history) > int(self.rate_hz):
            self._eff_history.pop(0)
        efficiency_std = pstdev(self._eff_history) if len(self._eff_history) > 1 else 0.0
        return efficiency, intensity_noise, efficiency_std

    def set_max_intensity(self, value: float) -> None:
        self._max_intensity = max(value, 1e-9)

    def encode_state(self, intensity: float) -> BackendState:
        efficiency, noise, eff_std = self.update_metrics(intensity)
        return BackendState(
            timestamp=time.monotonic(),
            int1=intensity,
            out1=self.last_out1,
            out2=self.last_out2,
            channel_voltages=list(self.last_channel_voltages),
            efficiency=efficiency,
            intensity_noise=noise,
            efficiency_std=eff_std,
        )


class RedPitayaBackend(BaseBackend):
    """Backend that communicates with a physical Red Pitaya via SCPI."""

    def __init__(
        self,
        host: str,
        n_channels: int,
        rate_hz: float,
        port: int = 5025,
        timeout: float = 0.5,
    ) -> None:
        super().__init__(n_channels=n_channels, rate_hz=rate_hz)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def _ensure_socket(self) -> socket.socket:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)
        return self.sock

    def _send(self, command: str) -> None:
        sock = self._ensure_socket()
        sock.sendall(f"{command}\n".encode())

    def _query(self, command: str) -> str:
        sock = self._ensure_socket()
        self._send(command)
        chunks: List[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if chunk.endswith(b"\n"):
                break
        return b"".join(chunks).decode().strip()

    def check_connection(self) -> str:
        """Query the device ID to ensure the SCPI socket is responsive."""

        response = self._query("*IDN?")
        if not response:
            raise RuntimeError("未收到 Red Pitaya 的识别信息，通信可能异常。")
        return response

    def configure(self) -> None:
        self._send("SOUR1:FUNC DC")
        self._send("SOUR2:FUNC DC")
        self._send("SOUR1:VOLT 0")
        self._send("SOUR2:VOLT 0")
        self._send("SOUR1:ENABLE 1")
        self._send("SOUR2:ENABLE 1")
        self._send("ACQ:RST")
        self._send("ACQ:DATA:FORMAT ASCII")
        self._send("ACQ:DATA:UNITS VOLTS")
        self._send("ACQ:DEC 1")
        self._send("ACQ:AVG 0")
        time.sleep(0.05)
        self._send("ACQ:START")
        time.sleep(0.01)
        self._send("ACQ:TRIG NOW")

    def read_int1(self) -> float:
        self._send("ACQ:START")
        self._send("ACQ:TRIG NOW")
        time.sleep(1.0 / self.rate_hz)
        response = self._query("ACQ:SOUR1:VALUE?")
        try:
            return float(response)
        except ValueError as exc:  # pragma: no cover - depends on hardware
            raise RuntimeError(f"Unexpected response from Red Pitaya: {response!r}") from exc

    def apply_control(self, channel_voltages: List[float]) -> Tuple[float, float]:
        if len(channel_voltages) != self.n_channels:
            raise ValueError("channel_voltages does not match configured channel count")
        out1 = float(channel_voltages[0])
        out2 = float(channel_voltages[1] if self.n_channels > 1 else 0.0)
        self._send(f"SOUR1:VOLT {out1:.6f}")
        self._send(f"SOUR2:VOLT {out2:.6f}")
        self.last_out1 = out1
        self.last_out2 = out2
        self.last_channel_voltages = list(channel_voltages)
        return out1, out2

    def close(self) -> None:
        if self.sock is not None:
            try:
                self._send("SOUR1:ENABLE 0")
                self._send("SOUR2:ENABLE 0")
            finally:
                self.sock.close()
                self.sock = None


class SimulationBackend(BaseBackend):
    """Deterministic simulation of the optical coherent combination system."""

    def __init__(
        self,
        n_channels: int,
        rate_hz: float,
        amplitudes: Iterable[float],
        phase_per_volt: Iterable[float],
        noise_level: float = 0.0005,
        drift_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_channels=n_channels, rate_hz=rate_hz)
        self.rng = random.Random(seed)
        self.amplitudes = [float(v) for v in amplitudes]
        if len(self.amplitudes) != n_channels:
            raise ValueError("Amplitude list does not match channel count")
        self.phase_per_volt = [float(v) for v in phase_per_volt]
        if len(self.phase_per_volt) != n_channels:
            raise ValueError("phase_per_volt list does not match channel count")
        self.noise_level = float(noise_level)
        self.drift_rate = float(drift_rate)
        self.phases = [self.rng.gauss(0.0, 0.2) for _ in range(n_channels)]
        self.last_channel_voltages = [0.0 for _ in range(n_channels)]
        self.drift = [0.0 for _ in range(n_channels)]
        self.bias = [0.0 for _ in range(n_channels)]
        self.set_max_intensity(self.max_intensity())

    def max_intensity(self) -> float:
        total = sum(self.amplitudes)
        return total * total

    def read_int1(self) -> float:
        dt = 1.0 / self.rate_hz
        for idx in range(self.n_channels):
            control_effect = self.phase_per_volt[idx] * (self.last_channel_voltages[idx] - self.bias[idx])
            self.phases[idx] = _wrap_pi(self.phases[idx] + control_effect)
            self.phases[idx] = _wrap_pi(self.phases[idx] + self.drift[idx] * dt)
            self.phases[idx] = _wrap_pi(self.phases[idx] + self.rng.gauss(0.0, self.noise_level))
        field_real = 0.0
        field_imag = 0.0
        for amp, phase in zip(self.amplitudes, self.phases):
            field_real += amp * math.cos(phase)
            field_imag += amp * math.sin(phase)
        intensity = field_real * field_real + field_imag * field_imag
        noise = self.rng.gauss(0.0, self.noise_level * intensity * 0.2)
        return intensity + noise

    def apply_control(self, channel_voltages: List[float]) -> Tuple[float, float]:
        if len(channel_voltages) != self.n_channels:
            raise ValueError("channel_voltages does not match configured channel count")
        self.last_channel_voltages = list(channel_voltages)
        self.last_out1 = float(channel_voltages[0])
        self.last_out2 = float(channel_voltages[1] if self.n_channels > 1 else 0.0)
        return self.last_out1, self.last_out2

    def phase_hint(self) -> List[float]:
        return list(self.phases)

    def get_amplified_out1(self) -> float:
        return self.last_out1 * MDT694B_GAIN

    def close(self) -> None:
        pass


@dataclass
class MPCConfig:
    n_channels: int
    horizon: int
    dt: float
    q_weight: float
    r_weight: float
    voltage_limits: Tuple[float, float]
    phase_per_volt: List[float]
    integral_gain: float = 0.0
    integral_limit: float = 0.0


class MPCController:
    """Diagonal MPC controller solved analytically per channel."""

    def __init__(self, config: MPCConfig) -> None:
        self.config = config
        self._gains = self._compute_gains()
        self._last_control = [0.0 for _ in range(config.n_channels)]
        self._integral_state = [0.0 for _ in range(config.n_channels)]

    def _compute_gains(self) -> List[float]:
        gains: List[float] = []
        for phase_gain in self.config.phase_per_volt:
            gains.append(0.0 if phase_gain == 0.0 else 1.0 / phase_gain)
        return gains

    def compute_control(self, phase_error: List[float]) -> List[float]:
        if len(phase_error) != self.config.n_channels:
            raise ValueError("phase_error length mismatch")
        lower, upper = self.config.voltage_limits
        control: List[float] = []
        dt = self.config.dt
        integral_gain = max(self.config.integral_gain, 0.0)
        limit = max(self.config.integral_limit, 0.0)
        for idx, (gain, error) in enumerate(zip(self._gains, phase_error)):
            p_term = -error * gain
            i_term = 0.0
            if integral_gain > 0.0 and dt > 0.0:
                self._integral_state[idx] += error * integral_gain * dt
                if limit > 0.0:
                    if self._integral_state[idx] > limit:
                        self._integral_state[idx] = limit
                    elif self._integral_state[idx] < -limit:
                        self._integral_state[idx] = -limit
                i_term = -self._integral_state[idx]
            value = p_term + i_term
            if value < lower:
                value = lower
            if value > upper:
                value = upper
            control.append(value)
        self._last_control = control
        return control

    @property
    def last_control(self) -> List[float]:
        return list(self._last_control)

    def reset(self) -> None:
        self._last_control = [0.0 for _ in range(self.config.n_channels)]
        self._integral_state = [0.0 for _ in range(self.config.n_channels)]


@dataclass
class EstimatorConfig:
    n_channels: int
    dt: float
    phase_per_volt: List[float]
    measurement_noise: float
    process_noise: float
    innovation_clip_sigma: float = 0.0


class PhaseEstimator:
    """Lightweight Extended Kalman Filter with diagonal covariance."""

    def __init__(self, config: EstimatorConfig, amplitudes: List[float]) -> None:
        self.config = config
        self.amplitudes = list(amplitudes)
        self.phases = [0.1 * (i - config.n_channels / 2.0) for i in range(config.n_channels)]
        self.covariance = [0.5 for _ in range(config.n_channels)]
        self._B = [gain * config.dt for gain in config.phase_per_volt]
        self.measurement_noise = max(config.measurement_noise, 1e-9)
        self.process_noise = max(config.process_noise, 1e-9)
        self._clip_threshold = max(config.innovation_clip_sigma, 0.0)
        self.last_update_accepted = True

    def predict(self, control: List[float]) -> None:
        for idx, gain in enumerate(self._B):
            self.phases[idx] = _wrap_pi(self.phases[idx] + gain * control[idx])
            self.covariance[idx] += self.process_noise

    def _field_components(self) -> tuple[float, float]:
        real = 0.0
        imag = 0.0
        for amp, phase in zip(self.amplitudes, self.phases):
            real += amp * math.cos(phase)
            imag += amp * math.sin(phase)
        return real, imag

    def sync_with_hint(self, hint: List[float]) -> None:
        if not hint:
            return
        for idx, value in enumerate(hint[: self.config.n_channels]):
            self.phases[idx] = _wrap_pi(value)

    def update(self, measurement: float) -> float:
        field_real, field_imag = self._field_components()
        intensity_pred = field_real * field_real + field_imag * field_imag
        jacobian: List[float] = []
        for amp, phase in zip(self.amplitudes, self.phases):
            sin_p = math.sin(phase)
            cos_p = math.cos(phase)
            jacobian.append(2.0 * (field_real * amp * sin_p - field_imag * amp * cos_p))
        S = self.measurement_noise
        for cov, grad in zip(self.covariance, jacobian):
            S += cov * grad * grad
        if S <= 1e-12:
            S = 1e-12
        residual = measurement - intensity_pred
        clip = self._clip_threshold
        self.last_update_accepted = True
        if clip > 0.0:
            innovation_std = math.sqrt(S)
            threshold = clip * innovation_std
            if abs(residual) > threshold:
                self.last_update_accepted = False
                print(
                    f"[Estimator] 跳过异常测量 residual={residual:.3f}, 阈值={threshold:.3f}",
                    file=sys.stderr,
                )
                return intensity_pred
        for idx in range(self.config.n_channels):
            K = self.covariance[idx] * jacobian[idx] / S
            self.phases[idx] = _wrap_pi(self.phases[idx] + K * residual)
            self.covariance[idx] = (1.0 - K * jacobian[idx]) * self.covariance[idx]
            if self.covariance[idx] < 1e-9:
                self.covariance[idx] = 1e-9
        return intensity_pred

    def phase_error(self) -> List[float]:
        return list(self.phases)


class PlotterProtocol(Protocol):  # pragma: no cover - structural typing helper
    def update(self, state: BackendState) -> None: ...

    def show_blocking(self) -> None: ...


@dataclass
class PlotConfig:
    decimation: int = 20
    history: int = 2000
    show_amplified_voltage: bool = True


class LivePlot:
    """Matplotlib live plot that mirrors the SPGD visualization."""

    def __init__(self, config: PlotConfig, n_channels: int) -> None:
        if plt is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("matplotlib is required for plotting but is not available")
        self.config = config
        self.n_channels = n_channels
        self.timestamps: Deque[float] = deque(maxlen=config.history)
        self.eff_history: Deque[float] = deque(maxlen=config.history)
        self.int_history: Deque[float] = deque(maxlen=config.history)
        self.noise_history: Deque[float] = deque(maxlen=config.history)
        self.out1_history: Deque[float] = deque(maxlen=config.history)
        self.out2_history: Deque[float] = deque(maxlen=config.history)
        self.out1_amp_history: Deque[float] = deque(maxlen=config.history)
        self.channel_histories: List[Deque[float]] = [deque(maxlen=config.history) for _ in range(n_channels)]
        self._counter = 0
        self._figure = None
        self._axes: Optional[List] = None
        self._lines: List = []
        self._text_out1 = None
        self._text_out2 = None
        self._text_amp = None

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return
        plt.ion()
        self._figure, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].set_ylabel("Efficiency [%]")
        axs[0].set_ylim(80, 102)
        axs[0].grid(True, alpha=0.3)
        axs[1].set_ylabel("Intensity noise [% rms]")
        axs[1].grid(True, alpha=0.3)
        axs[2].set_ylabel("Voltages [V]")
        axs[2].set_xlabel("Samples / decimated")
        axs[2].grid(True, alpha=0.3)
        self._axes = list(axs)
        (eff_line,) = axs[0].plot([], [], label="Efficiency")
        (noise_line,) = axs[1].plot([], [], label="Intensity noise")
        (out1_line,) = axs[2].plot([], [], label="Out1")
        (out2_line,) = axs[2].plot([], [], label="Out2")
        lines = [eff_line, noise_line, out1_line, out2_line]
        if self.config.show_amplified_voltage:
            (amp_line,) = axs[2].plot([], [], label="MDT694B Out1")
            lines.append(amp_line)
        self._text_out1 = axs[2].text(
            0.02,
            0.95,
            "Out1: 0.000 V",
            transform=axs[2].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="tab:blue",
        )
        self._text_out2 = axs[2].text(
            0.02,
            0.80,
            "Out2: 0.000 V",
            transform=axs[2].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="tab:orange",
        )
        if self.config.show_amplified_voltage:
            self._text_amp = axs[2].text(
                0.02,
                0.65,
                "MDT694B Out1: 0.000 V",
                transform=axs[2].transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="tab:green",
            )
        self._lines = lines
        for ax in axs:
            ax.legend(loc="upper right")

    def update(self, state: BackendState) -> None:
        if self._counter % self.config.decimation != 0:
            self._counter += 1
            return
        self._counter += 1
        self._ensure_figure()
        self.timestamps.append(state.timestamp)
        self.eff_history.append(state.efficiency * 100)
        self.int_history.append(state.int1)
        self.noise_history.append(state.intensity_noise * 100)
        self.out1_history.append(state.out1)
        self.out2_history.append(state.out2)
        self.out1_amp_history.append(state.out1 * MDT694B_GAIN)
        for idx, value in enumerate(state.channel_voltages):
            self.channel_histories[idx].append(value)

        eff_line, noise_line, out1_line, out2_line, *rest = self._lines
        eff_line.set_data(range(len(self.eff_history)), list(self.eff_history))
        noise_line.set_data(range(len(self.noise_history)), list(self.noise_history))
        out1_line.set_data(range(len(self.out1_history)), list(self.out1_history))
        out2_line.set_data(range(len(self.out2_history)), list(self.out2_history))
        if rest:
            rest[0].set_data(range(len(self.out1_amp_history)), list(self.out1_amp_history))
        if self._text_out1 is not None:
            self._text_out1.set_text(f"Out1: {state.out1:0.3f} V")
        if self._text_out2 is not None:
            self._text_out2.set_text(f"Out2: {state.out2:0.3f} V")
        if self._text_amp is not None:
            self._text_amp.set_text(f"MDT694B Out1: {state.out1 * MDT694B_GAIN:0.3f} V")
        for ax in self._axes or []:
            ax.relim()
            ax.autoscale_view()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()

    def show_blocking(self) -> None:
        if self._figure is not None:
            plt.ioff()
            plt.show()


@dataclass
class RunnerConfig:
    duration: Optional[float] = None
    log_interval: int = 500
    efficiency_target: float = 0.95
    efficiency_std_target: float = 0.015
    rate_hz: float = 5_000.0
    enable_console_log: bool = True
    measurement_smoothing: float = 0.0


@dataclass
class RunnerHistory:
    timestamps: List[float] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    efficiencies: List[float] = field(default_factory=list)
    noises: List[float] = field(default_factory=list)
    out1: List[float] = field(default_factory=list)
    out2: List[float] = field(default_factory=list)


class LockingRunner:
    """High level orchestration of the MPC coherent locking loop."""

    def __init__(
        self,
        backend: BaseBackend,
        controller: MPCController,
        estimator: PhaseEstimator,
        config: RunnerConfig,
        plotter: Optional[PlotterProtocol] = None,
    ) -> None:
        self.backend = backend
        self.controller = controller
        self.estimator = estimator
        self.config = config
        self.plotter = plotter
        self.history = RunnerHistory()
        self._start_time = time.monotonic()
        self._iteration = 0
        self._smoothing = min(max(config.measurement_smoothing, 0.0), 1.0)
        self._filtered_measurement: Optional[float] = None

    def _log_state(self, state: BackendState) -> None:
        self.history.timestamps.append(state.timestamp - self._start_time)
        self.history.intensities.append(state.int1)
        self.history.efficiencies.append(state.efficiency)
        self.history.noises.append(state.intensity_noise)
        self.history.out1.append(state.out1)
        self.history.out2.append(state.out2)
        if self.config.enable_console_log and self._iteration % self.config.log_interval == 0:
            print(
                f"[{self.history.timestamps[-1]:7.3f}s] Int1={state.int1:8.4f} V, "
                f"Out1={state.out1:7.3f} V, Out2={state.out2:7.3f} V, "
                f"Eff={state.efficiency*100:6.2f} %, Noise={state.intensity_noise*100:5.2f} %, "
                f"σ_eff={state.efficiency_std*100:5.2f} %",
                file=sys.stderr,
                flush=True,
            )

    def run(self) -> RunnerHistory:
        self.controller.reset()
        self._filtered_measurement = None
        self.backend.apply_control([0.0 for _ in range(self.controller.config.n_channels)])
        deadline = None if self.config.duration is None else self._start_time + self.config.duration
        try:
            while True:
                self.estimator.predict(self.controller.last_control)
                raw_measurement = self.backend.read_int1()
                prev_filtered = self._filtered_measurement
                if self._smoothing > 0.0 and prev_filtered is not None:
                    alpha = self._smoothing
                    measurement_candidate = (1.0 - alpha) * prev_filtered + alpha * raw_measurement
                else:
                    measurement_candidate = raw_measurement
                hint_getter = getattr(self.backend, "phase_hint", None)
                if callable(hint_getter):
                    hint = hint_getter()
                    self.estimator.sync_with_hint(hint)
                self.estimator.update(measurement_candidate)
                if not getattr(self.estimator, "last_update_accepted", True) and prev_filtered is not None:
                    measurement = prev_filtered
                else:
                    measurement = measurement_candidate
                self._filtered_measurement = measurement
                phase_error = self.estimator.phase_error()
                control = self.controller.compute_control(phase_error)
                self.backend.apply_control(control)
                state = self.backend.encode_state(measurement)
                self._iteration += 1
                self._log_state(state)
                if self.plotter is not None:
                    self.plotter.update(state)
                if deadline is not None and time.monotonic() >= deadline:
                    break
        finally:
            self.controller.reset()
            self.backend.close()
            if self.plotter is not None:
                self.plotter.show_blocking()
        return self.history

    def summary(self) -> Dict[str, float]:
        if not self.history.efficiencies:
            return {}
        eff_mean = statistics.mean(self.history.efficiencies)
        eff_std = statistics.pstdev(self.history.efficiencies) if len(self.history.efficiencies) > 1 else 0.0
        noise_mean = statistics.mean(self.history.noises) if self.history.noises else 0.0
        summary = {
            "efficiency_mean": eff_mean,
            "efficiency_std": eff_std,
            "intensity_noise": noise_mean,
        }
        summary["meets_efficiency"] = eff_mean >= self.config.efficiency_target
        summary["meets_noise"] = noise_mean <= self.config.efficiency_std_target
        return summary


@dataclass
class LauncherConfig:
    mode: str = "simulate"  # "simulate" or "hardware"
    host: Optional[str] = None
    channels: int = 2
    rate_hz: float = 5_000.0
    duration: float = 2.0
    voltage_limit: float = 1.5
    horizon: int = 10
    q_weight: float = 2.0
    r_weight: float = 0.1
    phase_per_volt: float = 3.2
    measurement_noise: float = 1e-3
    process_noise: float = 1e-4
    sim_noise: float = 5e-4
    sim_drift: float = 0.1
    seed: Optional[int] = None
    plot: bool = True
    plot_history: int = 2_000
    decimation: int = 20
    integral_gain: float = 0.1
    integral_limit: float = 0.5
    innovation_clip_sigma: float = 3.0
    hardware_measurement_smoothing: float = 0.2


PROMPT_HEADER = "=" * 64


def _prompt_mode(default: str = "simulate") -> str:
    mapping = {"1": "simulate", "2": "hardware"}
    prompt = "选择运行模式 [1]模拟 / [2]硬件 (默认模拟): "
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        if raw in mapping:
            return mapping[raw]
        if raw.lower() in ("simulate", "hardware"):
            return raw.lower()
        print("请输入 1(模拟) 或 2(硬件)。")


def _prompt_int(prompt: str, default: int, min_value: int, max_value: int) -> int:
    while True:
        raw = input(f"{prompt} [默认 {default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            value = int(raw)
            if min_value <= value <= max_value:
                return value
        print(f"请输入 {min_value}-{max_value} 之间的整数。")


def _prompt_float(prompt: str, default: float, min_value: Optional[float] = None) -> float:
    while True:
        raw = input(f"{prompt} [默认 {default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("请输入有效的数字。")
            continue
        if min_value is not None and value < min_value:
            print(f"取值需不小于 {min_value}。")
            continue
        return value


def _prompt_bool(prompt: str, default: bool) -> bool:
    mapping = {"y": True, "yes": True, "n": False, "no": False}
    suffix = "(Y/n)" if default else "(y/N)"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in mapping:
            return mapping[raw]
        print("请输入 y 或 n。")


def _prompt_host() -> str:
    while True:
        raw = input("请输入 Red Pitaya IP 地址: ").strip()
        if raw:
            return raw
        print("IP 地址不能为空。")


def gather_inputs() -> LauncherConfig:
    print(PROMPT_HEADER)
    print("MPC 版本1 一键启动")
    print("按提示输入参数，直接回车使用默认值。")
    print(PROMPT_HEADER)

    config = LauncherConfig()
    config.mode = _prompt_mode()
    if config.mode == "hardware":
        config.host = _prompt_host()
    config.channels = _prompt_int("请输入锁定路数 (1-8)", config.channels, 1, 8)
    config.rate_hz = _prompt_float("锁定速率 (Hz)", config.rate_hz, min_value=5_000.0)
    config.duration = _prompt_float("运行时长 (秒)", config.duration, min_value=0.1)
    config.voltage_limit = _prompt_float("电压限制 (V)", config.voltage_limit, min_value=0.1)
    config.plot = _prompt_bool("是否实时显示曲线", config.plot)
    return config


def build_backend(cfg: LauncherConfig, amplitudes: List[float], phase_per_volt: List[float]) -> BaseBackend:
    if cfg.mode == "hardware":
        backend = RedPitayaBackend(
            host=cfg.host or "127.0.0.1",
            n_channels=cfg.channels,
            rate_hz=cfg.rate_hz,
        )
    else:
        backend = SimulationBackend(
            n_channels=cfg.channels,
            rate_hz=cfg.rate_hz,
            amplitudes=amplitudes,
            phase_per_volt=phase_per_volt,
            noise_level=cfg.sim_noise,
            drift_rate=cfg.sim_drift,
            seed=cfg.seed,
        )
    return backend


def build_plotter(cfg: LauncherConfig, n_channels: int) -> Optional[LivePlot]:
    if not cfg.plot:
        return None
    plot_cfg = PlotConfig(
        decimation=cfg.decimation,
        history=cfg.plot_history,
        show_amplified_voltage=True,
    )
    return LivePlot(config=plot_cfg, n_channels=n_channels)


def run_with_config(cfg: LauncherConfig) -> None:
    amplitudes = [1.0] * cfg.channels
    phase_per_volt = [cfg.phase_per_volt] * cfg.channels

    backend = build_backend(cfg, amplitudes, phase_per_volt)
    runner_started = False
    try:
        if isinstance(backend, RedPitayaBackend):
            try:
                idn = backend.check_connection()
            except Exception as exc:
                backend.close()
                print("无法与 Red Pitaya 建立通信，请检查连接。")
                raise
            else:
                print(f"通信检测成功: {idn}")
            proceed = _prompt_bool("确认开始硬件锁定吗", True)
            if not proceed:
                print("已取消硬件运行。")
                backend.close()
                return

        backend.configure()
        backend.set_max_intensity(sum(amplitudes) ** 2)

        controller = MPCController(
            MPCConfig(
                n_channels=cfg.channels,
                horizon=cfg.horizon,
                dt=1.0 / cfg.rate_hz,
                q_weight=cfg.q_weight,
                r_weight=cfg.r_weight,
                voltage_limits=(-cfg.voltage_limit, cfg.voltage_limit),
                phase_per_volt=list(phase_per_volt),
                integral_gain=cfg.integral_gain,
                integral_limit=cfg.integral_limit,
            )
        )
        estimator = PhaseEstimator(
            EstimatorConfig(
                n_channels=cfg.channels,
                dt=1.0 / cfg.rate_hz,
                phase_per_volt=list(phase_per_volt),
                measurement_noise=cfg.measurement_noise,
                process_noise=cfg.process_noise,
                innovation_clip_sigma=cfg.innovation_clip_sigma,
            ),
            amplitudes,
        )
        plotter: Optional[LivePlot] = None
        if cfg.plot:
            try:
                plotter = build_plotter(cfg, cfg.channels)
            except RuntimeError as exc:
                print(f"无法启用图形显示: {exc}")
                plotter = None

        smoothing = 0.0
        if cfg.mode == "hardware":
            smoothing = min(max(cfg.hardware_measurement_smoothing, 0.0), 1.0)
            if smoothing > 0.0:
                print(f"硬件模式已启用测量平滑，系数 {smoothing:0.2f}")
        runner_config = RunnerConfig(
            duration=cfg.duration,
            log_interval=max(int(cfg.rate_hz / 50), 1),
            efficiency_target=0.95,
            efficiency_std_target=0.015,
            rate_hz=cfg.rate_hz,
            enable_console_log=True,
            measurement_smoothing=smoothing,
        )
        runner = LockingRunner(backend, controller, estimator, runner_config, plotter)
        runner_started = True
        try:
            runner.run()
        except KeyboardInterrupt:
            print("\n用户中断，正在整理数据…")
        summary = runner.summary()
        if summary:
            eff = summary["efficiency_mean"] * 100
            eff_std = summary["efficiency_std"] * 100
            intensity_noise = summary["intensity_noise"] * 100
            print(
                "\n锁定完成:",
                f"平均合成效率 {eff:.2f}%",
                f"效率标准差 {eff_std:.2f}%",
                f"强度噪声 {intensity_noise:.2f}%",
            )
    finally:
        if not runner_started:
            backend.close()


def main() -> None:
    config = gather_inputs()
    run_with_config(config)


if __name__ == "__main__":
    main()
