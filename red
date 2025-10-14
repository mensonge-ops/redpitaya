#!/usr/bin/env python3
"""MPC coherent synthesis locking launcher (版本1-fixed).

This script runs a model-predictive-control (MPC) coherent combination
lock either against a real Red Pitaya device via SCPI or against a
self-contained simulation backend.  It implements the fixes listed in
版本1-fixed:

* Use ``draw_idle()``/``plt.pause`` to avoid UI freezes and increase the
  default plotting decimation rate so Matplotlib does not run at the
  full control loop speed.
* Pace the main control loop with a metronome based on ``rate_hz`` so
  that the GUI is not starved by a 100% CPU busy loop.
* Optimise the ``RedPitayaBackend`` fast-path to read values via the
  ``VALUE?`` SCPI command to avoid unnecessary ``STOP``/``RST`` cycles.
* Simplify ``SimulationBackend`` to generate measurements directly
  instead of emulating SCPI strings.
* Deduplicate branches inside ``PhaseEstimator.update``.

NumPy/Matplotlib are optional.  When they are missing the script will
still run in headless mode and simply skip live plots or RIN FFTs.
"""
from __future__ import annotations

import argparse
import math
import random
import socket
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Deque, Dict, Iterable, List, Optional, Protocol, Tuple

try:  # Optional dependency for live visualization
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib may be unavailable
    plt = None  # type: ignore

# ---- Optional dependency for fast RIN FFT ----
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None  # type: ignore

MDT694B_GAIN = 15.0  # MDT694B: 0-10 V in -> 0-150 V out（示意）


# ----------------------------- Utils -----------------------------
def _wrap_pi(value: float) -> float:
    """Wrap a phase angle to [-pi, pi)."""
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def compute_rin_spectrum(samples: Iterable[float], sample_rate: float) -> Tuple[List[float], List[float]]:
    """Compute single-sideband RIN (dBc/Hz) using NumPy FFT.

    如果当前环境未安装 NumPy，则返回空谱。
    """
    if np is None:
        # NumPy 不可用则跳过 RIN 绘制
        return [], []
    x = np.asarray([float(v) for v in samples], dtype=float)
    n = int(x.size)
    if n < 2 or sample_rate <= 0.0:
        return [], []
    m = float(x.mean())
    if not np.isfinite(m) or m <= 0.0:
        return [], []
    # 分数强度波动
    frac = (x - m) / (m + 1e-30)
    # 汉宁窗 + 能量归一化
    window = np.hanning(n)
    U = np.mean(window * window)  # 窗功率归一化
    xw = frac * window
    # 单边功率谱密度 (per Hz)
    X = np.fft.rfft(xw)
    psd = (np.abs(X) ** 2) / (sample_rate * (n ** 2) * U)
    if psd.size > 2:
        psd[1:-1] *= 2.0  # 单边谱
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    rin_db = 10.0 * np.log10(psd + 1e-30)
    return freqs.tolist(), rin_db.tolist()


# ----------------------------- Data types -----------------------------
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


# ----------------------------- Base Backend -----------------------------
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
        efficiency = max(0.0, min(1.0, efficiency))
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

# ----------------------------- Red Pitaya Backend -----------------------------
class RedPitayaBackend(BaseBackend):
    """Backend that communicates with a physical Red Pitaya via SCPI (raw-socket)."""

    def __init__(
        self,
        host: str,
        n_channels: int,
        rate_hz: float,
        port: int = 5_000,
        timeout: float = 5.0,
    ) -> None:
        super().__init__(n_channels=n_channels, rate_hz=rate_hz)
        self.host = host
        self.port = port
        self.timeout = float(timeout)
        self.sock: Optional[socket.socket] = None
        self._lk = threading.Lock()  # 串行化所有 query，防止串包

    # ---------- 连接 ----------
    def _ensure_socket(self) -> None:
        if self.sock is not None:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise ConnectionError(f"无法连接到 Red Pitaya ({self.host}:{self.port})：{exc}") from exc
        s.settimeout(self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s

    def _reconnect(self) -> None:
        self.close()
        time.sleep(0.03)
        self._ensure_socket()

    def close(self) -> None:
        if self.sock:
            try:
                try:
                    self._write_line("SOUR1:ENABLE 0")
                except Exception:
                    pass
                try:
                    self._write_line("SOUR2:ENABLE 0")
                except Exception:
                    pass
            finally:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

    # ---------- 行 I/O ----------
    def _write_line(self, command: str) -> None:
        self._ensure_socket()
        assert self.sock is not None
        payload = (command.rstrip("\r\n") + "\n").encode("ascii", "ignore")
        try:
            self.sock.sendall(payload)
        except OSError as exc:
            self.close()
            raise ConnectionError(f"向 Red Pitaya 发送指令失败 ({self.host}:{self.port})：{exc}") from exc

    def _read_line_raw(self, *, read_timeout: Optional[float] = None) -> str:
        self._ensure_socket()
        assert self.sock is not None
        old_to = self.sock.gettimeout()
        if read_timeout is not None:
            self.sock.settimeout(read_timeout)
        buf = bytearray()
        try:
            while True:
                try:
                    chunk = self.sock.recv(4096)
                except (socket.timeout, OSError) as exc:
                    raise ConnectionError(f"接收 Red Pitaya 返回数据失败 ({self.host}:{self.port})：{exc}") from exc
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in chunk or b"\r" in chunk:
                    break
        finally:
            if read_timeout is not None:
                try:
                    self.sock.settimeout(old_to)
                except Exception:
                    pass
        if not buf:
            raise ConnectionError(f"未能从 Red Pitaya ({self.host}:{self.port}) 收到有效响应。")
        return bytes(buf).decode("ascii", "ignore").rstrip("\r\n")

    def _query(self, command: str, *, retries: int = 2, read_timeout: Optional[float] = None) -> str:
        with self._lk:
            for k in range(retries + 1):
                try:
                    self._write_line(command)
                    return self._read_line_raw(read_timeout=read_timeout)
                except ConnectionError:
                    if k < retries:
                        self._reconnect()
                        time.sleep(0.02)
                        continue
                    raise

    def _drain_input(self, budget: int = 65_536) -> None:
        if not self.sock:
            return
        old_to = self.sock.gettimeout()
        total = 0
        try:
            self.sock.settimeout(0.0)  # non-blocking
            while total < budget:
                try:
                    chunk = self.sock.recv(min(4096, budget - total))
                except (BlockingIOError, InterruptedError, socket.timeout):
                    break
                except OSError:
                    break
                if not chunk:
                    break
                total += len(chunk)
        finally:
            self.sock.settimeout(old_to)

    # ---------- 设备配置/触发 ----------
    def check_connection(self) -> str:
        resp = self._query("*IDN?")
        if not resp:
            raise RuntimeError("未收到 Red Pitaya 的识别信息，通信可能异常。")
        return resp

    def _arm_immediate_trigger(self) -> None:
        self._query("ACQ:TRIG:LEV 0;*OPC?")
        try:
            self._query("ACQ:TRIG:DLY 0;*OPC?")
        except Exception:
            pass
        try:
            self._query("ACQ:TRIG:SOURCE NOW;*OPC?")
            self._query("ACQ:START;*OPC?")
            return
        except Exception:
            pass
        try:
            self._query("ACQ:TRIG NOW;*OPC?")
            self._query("ACQ:START;*OPC?")
            return
        except Exception:
            pass
        self._query("ACQ:START;*OPC?")
        self._query("ACQ:TRIG NOW;*OPC?")

    def configure(self) -> None:
        self._query("SOUR1:FUNC DC;*OPC?")
        self._query("SOUR2:FUNC DC;*OPC?")
        self._query("SOUR1:VOLT 0;*OPC?")
        self._query("SOUR2:VOLT 0;*OPC?")
        self._query("SOUR1:ENABLE 1;*OPC?")
        self._query("SOUR2:ENABLE 1;*OPC?")
        self._query("ACQ:RST;*OPC?")
        self._query("ACQ:DATA:FORMAT ASCII;*OPC?")
        self._query("ACQ:DATA:UNITS VOLTS;*OPC?")
        self._query("ACQ:DEC 1;*OPC?")
        self._query("ACQ:AVG 0;*OPC?")
        self._query("ACQ:TRIG:LEV 0;*OPC?")
        self._arm_immediate_trigger()

    # ---------- 读取与控制 ----------
    def read_int1(self) -> float:
        # 快速路径：配置已由 configure() 完成，这里只读当前值；避免每次 STOP/RST 造成阻塞
        poll_delay = min(0.001, max(1.0 / (10.0 * max(self.rate_hz, 1.0)), 0.0001))
        ready_states = {"TD", "TS", "STOP", "DONE"}

        # 尝试短回复
        for _ in range(2):
            try:
                s = self._query("ACQ:SOUR1:VALUE?", read_timeout=self.timeout)
                v = self._extract_first_float(s)
                if v is not None:
                    return v
            except ConnectionError:
                pass
            time.sleep(poll_delay)

        # 回退到 DATA:LAST?/CH1?/DATA?
        self._drain_input()
        st = ""
        for cmd in ("ACQ:DATA:LAST?", "ACQ:DATA:CH1?", "ACQ:SOUR1:DATA?"):
            try:
                s = self._query(cmd, read_timeout=self.timeout * 2.0)
                arr = self._parse_floats(s)
                if arr:
                    return arr[-1]
                if s.strip().upper() in ready_states:
                    time.sleep(poll_delay)
                    continue
            except ConnectionError:
                continue

        raise ConnectionError(f"读取数据超时/无应答（最后状态={st!r}）")

    @staticmethod
    def _extract_first_float(response: str) -> Optional[float]:
        for tok in response.replace(",", " ").replace(";", " ").split():
            try:
                return float(tok)
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_floats(csv_line: str) -> List[float]:
        out: List[float] = []
        for tok in csv_line.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except ValueError:
                pass
        return out

    def apply_control(self, channel_voltages: List[float]) -> Tuple[float, float]:
        if len(channel_voltages) != self.n_channels:
            raise ValueError("channel_voltages does not match configured channel count")
        out1 = float(channel_voltages[0])
        out2 = float(channel_voltages[1] if self.n_channels > 1 else 0.0)
        self._write_line(f"SOUR1:VOLT {out1:.6f}")
        self._write_line(f"SOUR2:VOLT {out2:.6f}")
        self.last_out1 = out1
        self.last_out2 = out2
        self.last_channel_voltages = list(channel_voltages)
        return out1, out2


# ----------------------------- Simulation Backend -----------------------------
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
        self.set_max_intensity(self.max_intensity())

    def max_intensity(self) -> float:
        total = sum(self.amplitudes)
        return total * total

    def read_int1(self) -> float:
        """根据内部物理量生成强度测量（快速、无 SCPI）。"""
        # 相位随漂移与控制量更新
        for i in range(self.n_channels):
            self.drift[i] += self.rng.gauss(0.0, self.drift_rate / max(self.rate_hz, 1.0))
            self.phases[i] = _wrap_pi(
                self.phases[i]
                + self.phase_per_volt[i] * self.last_channel_voltages[i]
                + self.drift[i]
            )

        # 合成场 → 强度
        real = 0.0
        imag = 0.0
        for amp, ph in zip(self.amplitudes, self.phases):
            real += amp * math.cos(ph)
            imag += amp * math.sin(ph)
        intensity = real * real + imag * imag

        # 加测量噪声（RIN 近似）
        noise = self.rng.gauss(0.0, self.noise_level * max(intensity, 1e-6))
        y = max(intensity + noise, 1e-9)
        return y

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


# ----------------------------- MPC Controller -----------------------------
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
    """Diagonal MPC controller solved analytically per channel (简化版)。"""

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
                    self._integral_state[idx] = max(-limit, min(limit, self._integral_state[idx]))
                i_term = -self._integral_state[idx]
            value = p_term + i_term
            value = max(lower, min(upper, value))
            control.append(value)
        self._last_control = control
        return control

    @property
    def last_control(self) -> List[float]:
        return list(self._last_control)

    def reset(self) -> None:
        self._last_control = [0.0 for _ in range(self.config.n_channels)]
        self._integral_state = [0.0 for _ in range(self.config.n_channels)]


# ----------------------------- Estimator (EKF-lite) -----------------------------
@dataclass
class EstimatorConfig:
    n_channels: int
    dt: float
    phase_per_volt: List[float]
    measurement_noise: float
    process_noise: float
    innovation_clip_sigma: float = 0.0
    meas_gain_init: float = 1.0
    adapt_gain: float = 0.02
    gain_limits: Tuple[float, float] = (0.05, 20.0)


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
        self.meas_gain = max(config.meas_gain_init, 1e-9)
        self.adapt_gain = max(config.adapt_gain, 0.0)
        self._g_lo, self._g_hi = config.gain_limits

    def predict(self, control: List[float]) -> None:
        for idx, gain in enumerate(self._B):
            self.phases[idx] = _wrap_pi(self.phases[idx] + gain * control[idx])
            self.covariance[idx] += self.process_noise

    def _field_components(self) -> Tuple[float, float]:
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
        # 1) 计算场与模型强度
        field_real, field_imag = self._field_components()
        intensity_model = field_real * field_real + field_imag * field_imag

        # 2) 应用测量增益 g
        g = self.meas_gain
        intensity_pred = g * intensity_model

        # 3) 雅可比（同样乘 g）
        jacobian: List[float] = []
        for amp, phase in zip(self.amplitudes, self.phases):
            sin_p = math.sin(phase)
            cos_p = math.cos(phase)
            dI = 2.0 * (field_real * amp * sin_p - field_imag * amp * cos_p)
            jacobian.append(g * dI)

        # 4) 创新协方差 S
        S = self.measurement_noise
        for cov, grad in zip(self.covariance, jacobian):
            S += cov * grad * grad
        S = max(S, 1e-12)

        # 5) 残差
        residual = measurement - intensity_pred

        # 6) 在线自适应测量增益 g
        if self.adapt_gain > 0.0:
            denom = intensity_model * intensity_model + 1e-9
            self.meas_gain += self.adapt_gain * residual * intensity_model / denom
            self.meas_gain = max(self._g_lo, min(self._g_hi, self.meas_gain))

        # 7) 离群裁剪（可选）
        clip = self._clip_threshold
        self.last_update_accepted = True
        if clip > 0.0:
            innovation_std = math.sqrt(S)
            if abs(residual) > clip * innovation_std:
                self.last_update_accepted = False
                print(
                    f"[Estimator] 跳过异常测量 residual={residual:.3f}, 阈值={clip * innovation_std:.3f}, g={self.meas_gain:.3f}",
                    file=sys.stderr,
                )
                return intensity_pred

        # 8) 卡尔曼更新
        for idx in range(self.config.n_channels):
            K = self.covariance[idx] * jacobian[idx] / S
            self.phases[idx] = _wrap_pi(self.phases[idx] + K * residual)
            self.covariance[idx] = (1.0 - K * jacobian[idx]) * self.covariance[idx]
            if self.covariance[idx] < 1e-9:
                self.covariance[idx] = 1e-9
        return intensity_pred

    def phase_error(self) -> List[float]:
        return list(self.phases)


# ----------------------------- Plotting -----------------------------
class PlotterProtocol(Protocol):  # pragma: no cover - structural typing helper
    def update(self, state: BackendState) -> None:
        ...

    def show_blocking(self) -> None:
        ...


@dataclass
class PlotConfig:
    decimation: int = 200  # 降帧，避免卡 UI（rate_hz/decimation ≈ 刷新 FPS）
    history: int = 2_000
    show_amplified_voltage: bool = True
    sample_rate: float = 5_000.0
    rin_window: float = 0.1


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
        self.out1_history: Deque[float] = deque(maxlen=config.history)
        self.out2_history: Deque[float] = deque(maxlen=config.history)
        self.out1_amp_history: Deque[float] = deque(maxlen=config.history)
        self.channel_histories: List[Deque[float]] = [deque(maxlen=config.history) for _ in range(n_channels)]
        self._counter = 0
        self._figure = None
        self._axes: Optional[List] = None
        self._line_eff = None
        self._line_rin = None
        self._line_out1 = None
        self._line_out2 = None
        self._line_amp = None
        self._text_out1 = None
        self._text_out2 = None
        self._text_amp = None

    def _ensure_figure(self) -> None:
        if self._figure is not None:
            return
        plt.ion()
        self._figure, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs[0].set_ylabel("Efficiency [%]")
        axs[0].set_ylim(80, 102)
        axs[0].grid(True, alpha=0.3)
        axs[1].set_ylabel("RIN [dBc/Hz]")
        axs[1].set_xlabel("Frequency [Hz]")
        axs[1].grid(True, alpha=0.3)
        axs[1].set_xlim(0, self.config.sample_rate / 2.0)
        axs[1].set_ylim(-140, -60)
        axs[2].set_ylabel("Voltages [V]")
        axs[2].set_xlabel("Samples / decimated")
        axs[2].grid(True, alpha=0.3)
        self._axes = list(axs)
        (eff_line,) = axs[0].plot([], [], label="Efficiency")
        (rin_line,) = axs[1].plot([], [], label="RIN")
        (out1_line,) = axs[2].plot([], [], label="Out1")
        (out2_line,) = axs[2].plot([], [], label="Out2")
        if self.config.show_amplified_voltage:
            (amp_line,) = axs[2].plot([], [], label="MDT694B Out1")
            self._line_amp = amp_line
        self._line_eff = eff_line
        self._line_rin = rin_line
        self._line_out1 = out1_line
        self._line_out2 = out2_line
        self._text_out1 = axs[2].text(
            0.02,
            0.95,
            "Out1: 0.000 V",
            transform=axs[2].transAxes,
            va="top",
            ha="left",
            fontsize=10,
        )
        self._text_out2 = axs[2].text(
            0.02,
            0.80,
            "Out2: 0.000 V",
            transform=axs[2].transAxes,
            va="top",
            ha="left",
            fontsize=10,
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
            )
        axs[0].legend(loc="upper right")
        axs[1].legend(loc="upper right")
        axs[2].legend(loc="upper right")

    def update(self, state: BackendState) -> None:
        if self._counter % max(1, self.config.decimation) != 0:
            self._counter += 1
            return
        self._counter += 1
        self._ensure_figure()
        self.timestamps.append(state.timestamp)
        self.eff_history.append(state.efficiency * 100)
        self.int_history.append(state.int1)
        self.out1_history.append(state.out1)
        self.out2_history.append(state.out2)
        self.out1_amp_history.append(state.out1 * MDT694B_GAIN)
        for idx, value in enumerate(state.channel_voltages):
            self.channel_histories[idx].append(value)

        if self._line_eff is not None:
            self._line_eff.set_data(range(len(self.eff_history)), list(self.eff_history))
        if self._line_rin is not None:
            window = max(int(self.config.rin_window * self.config.sample_rate), 2)
            if len(self.int_history) >= window:
                samples = list(self.int_history)[-window:]
                freqs, rin = compute_rin_spectrum(samples, self.config.sample_rate)
                self._line_rin.set_data(freqs, rin)
                if rin and self._axes is not None:
                    ymin = min(rin)
                    ymax = max(rin)
                    if math.isfinite(ymin) and math.isfinite(ymax):
                        if ymin == ymax:
                            ymin -= 1.0
                            ymax += 1.0
                        margin = max((ymax - ymin) * 0.1, 1.0)
                        self._axes[1].set_ylim(ymin - margin, ymax + margin)
                if self._axes is not None:
                    self._axes[1].set_xlim(0, self.config.sample_rate / 2.0)
            else:
                self._line_rin.set_data([], [])
        if self._line_out1 is not None:
            self._line_out1.set_data(range(len(self.out1_history)), list(self.out1_history))
        if self._line_out2 is not None:
            self._line_out2.set_data(range(len(self.out2_history)), list(self.out2_history))
        if self._line_amp is not None:
            self._line_amp.set_data(range(len(self.out1_amp_history)), list(self.out1_amp_history))
        if self._text_out1 is not None:
            self._text_out1.set_text(f"Out1: {state.out1:0.3f} V")
        if self._text_out2 is not None:
            self._text_out2.set_text(f"Out2: {state.out2:0.3f} V")
        if self._text_amp is not None:
            self._text_amp.set_text(f"MDT694B Out1: {state.out1 * MDT694B_GAIN:0.3f} V")
        if self._axes is not None:
            for idx, ax in enumerate(self._axes):
                if idx == 1:
                    continue
                ax.relim()
                ax.autoscale_view()
        if self._figure is not None:
            self._figure.canvas.draw_idle()
            try:
                plt.pause(0.001)
            except Exception:
                pass

    def show_blocking(self) -> None:
        if self._figure is not None:
            plt.ioff()
            plt.show()


# ----------------------------- Runner -----------------------------
@dataclass
class RunnerConfig:
    duration: Optional[float] = None
    log_interval: int = 500
    efficiency_target: float = 0.95
    efficiency_std_target: float = 0.015
    rate_hz: float = 5_000.0
    enable_console_log: bool = True
    measurement_smoothing: float = 0.0
    rin_window: float = 0.1


@dataclass
class RunnerHistory:
    timestamps: List[float] = field(default_factory=list)
    intensities: List[float] = field(default_factory=list)
    efficiencies: List[float] = field(default_factory=list)
    noises: List[float] = field(default_factory=list)
    out1: List[float] = field(default_factory=list)
    out2: List[float] = field(default_factory=list)

    def rin_spectrum(self, rate_hz: float, window: float) -> Tuple[List[float], List[float]]:
        window_samples = max(int(window * rate_hz), 2)
        if len(self.intensities) < 2:
            return [], []
        if len(self.intensities) < window_samples:
            samples = self.intensities
        else:
            samples = self.intensities[-window_samples:]
        return compute_rin_spectrum(samples, rate_hz)


class LockingRunner:
    """High level orchestration of the MPC coherent locking loop."""

    def __init__(
        self,
        backend: BaseBackend,
        controller: MPCController,
        estimator: PhaseEstimator,
        config: RunnerConfig,
        plotter: Optional[LivePlot] = None,
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
            rin_info = ""
            freqs, rin = self.history.rin_spectrum(self.config.rate_hz, self.config.rin_window)
            if freqs and rin:
                peak_idx = max(range(len(rin)), key=lambda idx: rin[idx])
                peak_freq = freqs[peak_idx]
                peak_rin = rin[peak_idx]
                rin_info = f", RIN_peak={peak_rin:6.1f} dBc/Hz@{peak_freq:6.0f} Hz"
            print(
                f"[{self.history.timestamps[-1]:7.3f}s] Int1={state.int1:8.4f} V, "
                f"Out1={state.out1:7.3f} V, Out2={state.out2:7.3f} V, "
                f"Eff={state.efficiency*100:6.2f} %, σ_eff={state.efficiency_std*100:5.2f} %"
                f"{rin_info}",
                file=sys.stderr,
                flush=True,
            )

    def run(self) -> RunnerHistory:
        self.controller.reset()
        self._filtered_measurement = None
        self.backend.apply_control([0.0 for _ in range(self.controller.config.n_channels)])
        deadline = None if self.config.duration is None else self._start_time + self.config.duration

        # 控制循环按 dt 节拍运行，防止 100% CPU 占用导致 GUI 卡死
        dt = 1.0 / max(self.config.rate_hz, 1.0)
        next_tick = time.perf_counter()

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

                # 节拍：尽量对齐 rate_hz，如果超时则立即下一轮并重置基准
                next_tick += dt
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()

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
        freqs, rin = self.history.rin_spectrum(self.config.rate_hz, self.config.rin_window)
        if freqs and rin:
            peak_idx = max(range(len(rin)), key=lambda idx: rin[idx])
            summary["rin_peak_db"] = float(rin[peak_idx])
            summary["rin_peak_freq_hz"] = float(freqs[peak_idx])
        return summary


# ----------------------------- Launcher & CLI -----------------------------
@dataclass
class LauncherConfig:
    mode: str = "simulate"  # "simulate" or "hardware"
    host: Optional[str] = None
    port: int = 5_000
    channels: int = 2
    rate_hz: float = 5_000.0
    duration: Optional[float] = 2.0
    voltage_limit: float = 1.0
    horizon: int = 10
    q_weight: float = 2.0
    r_weight: float = 0.1
    phase_per_volt: float = 3.2
    measurement_noise: float = 2.0
    process_noise: float = 1e-3
    sim_noise: float = 5e-4
    sim_drift: float = 0.1
    seed: Optional[int] = None
    plot: bool = True
    plot_history: int = 2_000
    decimation: int = 200
    integral_gain: float = 0.1
    integral_limit: float = 0.5
    innovation_clip_sigma: float = 0.0
    hardware_measurement_smoothing: float = 0.3
    rin_window: float = 0.1
    log_interval: int = 500
    efficiency_target: float = 0.95
    efficiency_std_target: float = 0.015


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
    print("MPC 版本1 一键启动 (fixed)")
    print("按提示输入参数，直接回车使用默认值。")
    print(PROMPT_HEADER)

    config = LauncherConfig()
    config.mode = _prompt_mode()
    if config.mode == "hardware":
        config.host = _prompt_host()
        config.port = _prompt_int("请输入端口", config.port, 1, 65_535)
        config.duration = None
    config.channels = _prompt_int("请输入锁定路数 (1-8)", config.channels, 1, 8)
    config.rate_hz = _prompt_float("锁定速率 (Hz)", config.rate_hz, min_value=500.0)  # 建议从 500 Hz 起步
    if config.mode != "hardware":
        config.duration = _prompt_float("运行时长 (秒)", config.duration or 2.0, min_value=0.1)
    config.voltage_limit = _prompt_float("电压限制 (V)", config.voltage_limit, min_value=0.1)
    config.plot = _prompt_bool("是否实时显示曲线", config.plot)
    return config


def build_backend(cfg: LauncherConfig, amplitudes: List[float], phase_per_volt: List[float]) -> BaseBackend:
    if cfg.mode == "hardware":
        if not cfg.host:
            raise ValueError("Hardware mode requires a --host or interactive host input")
        backend: BaseBackend = RedPitayaBackend(
            host=cfg.host,
            n_channels=cfg.channels,
            rate_hz=cfg.rate_hz,
            port=cfg.port,
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

def build_controller(cfg: LauncherConfig, phase_per_volt: List[float]) -> MPCController:
    dt = 1.0 / max(cfg.rate_hz, 1.0)
    controller_cfg = MPCConfig(
        n_channels=cfg.channels,
        horizon=cfg.horizon,
        dt=dt,
        q_weight=cfg.q_weight,
        r_weight=cfg.r_weight,
        voltage_limits=(-cfg.voltage_limit, cfg.voltage_limit),
        phase_per_volt=phase_per_volt,
        integral_gain=cfg.integral_gain,
        integral_limit=cfg.integral_limit,
    )
    return MPCController(controller_cfg)


def build_estimator(cfg: LauncherConfig, amplitudes: List[float], phase_per_volt: List[float]) -> PhaseEstimator:
    dt = 1.0 / max(cfg.rate_hz, 1.0)
    est_cfg = EstimatorConfig(
        n_channels=cfg.channels,
        dt=dt,
        phase_per_volt=phase_per_volt,
        measurement_noise=cfg.measurement_noise,
        process_noise=cfg.process_noise,
        innovation_clip_sigma=cfg.innovation_clip_sigma,
    )
    return PhaseEstimator(est_cfg, amplitudes)


def build_plotter(cfg: LauncherConfig) -> Optional[LivePlot]:
    if not cfg.plot:
        return None
    try:
        plot_cfg = PlotConfig(
            decimation=max(1, cfg.decimation),
            history=max(10, cfg.plot_history),
            sample_rate=cfg.rate_hz,
            rin_window=cfg.rin_window,
        )
        return LivePlot(plot_cfg, cfg.channels)
    except RuntimeError as exc:
        print(f"[Plot] 无法启用实时绘图：{exc}", file=sys.stderr)
        return None


def run_with_config(cfg: LauncherConfig) -> Tuple[LockingRunner, RunnerHistory]:
    amplitudes = [1.0 for _ in range(cfg.channels)]
    phase_per_volt = [cfg.phase_per_volt for _ in range(cfg.channels)]
    backend = build_backend(cfg, amplitudes, phase_per_volt)
    backend.configure()
    backend.set_max_intensity(sum(amplitudes) ** 2)
    controller = build_controller(cfg, phase_per_volt)
    estimator = build_estimator(cfg, amplitudes, phase_per_volt)
    runner_cfg = RunnerConfig(
        duration=cfg.duration if cfg.mode != "hardware" else None,
        rate_hz=cfg.rate_hz,
        measurement_smoothing=cfg.hardware_measurement_smoothing if cfg.mode == "hardware" else 0.0,
        rin_window=cfg.rin_window,
        log_interval=cfg.log_interval,
        efficiency_target=cfg.efficiency_target,
        efficiency_std_target=cfg.efficiency_std_target,
    )
    plotter = build_plotter(cfg)
    runner = LockingRunner(
        backend=backend,
        controller=controller,
        estimator=estimator,
        config=runner_cfg,
        plotter=plotter,
    )
    history = runner.run()
    return runner, history


def _args_to_config(args: argparse.Namespace) -> LauncherConfig:
    cfg = LauncherConfig()
    cfg.mode = args.mode
    cfg.host = args.host
    cfg.port = args.port
    cfg.channels = args.channels
    cfg.rate_hz = args.rate_hz
    cfg.duration = args.duration
    if cfg.mode == "hardware" and args.duration is None:
        cfg.duration = None
    cfg.voltage_limit = args.voltage_limit
    cfg.horizon = args.horizon
    cfg.q_weight = args.q_weight
    cfg.r_weight = args.r_weight
    cfg.phase_per_volt = args.phase_per_volt
    cfg.measurement_noise = args.measurement_noise
    cfg.process_noise = args.process_noise
    cfg.sim_noise = args.sim_noise
    cfg.sim_drift = args.sim_drift
    cfg.seed = args.seed
    cfg.plot = args.plot
    cfg.plot_history = args.plot_history
    cfg.decimation = args.decimation
    cfg.integral_gain = args.integral_gain
    cfg.integral_limit = args.integral_limit
    cfg.innovation_clip_sigma = args.innovation_clip_sigma
    cfg.hardware_measurement_smoothing = args.hardware_smoothing
    cfg.rin_window = args.rin_window
    cfg.log_interval = args.log_interval
    cfg.efficiency_target = args.efficiency_target
    cfg.efficiency_std_target = args.efficiency_std_target
    return cfg


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPC coherent synthesis locking launcher")
    parser.add_argument("--mode", choices=["simulate", "hardware"], default="simulate")
    parser.add_argument("--host", help="Red Pitaya hostname/IP (hardware mode)")
    parser.add_argument("--port", type=int, default=5_000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--rate-hz", type=float, default=5_000.0)
    parser.add_argument("--duration", type=float, help="Run duration in seconds (simulation mode)")
    parser.add_argument("--voltage-limit", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--q-weight", type=float, default=2.0)
    parser.add_argument("--r-weight", type=float, default=0.1)
    parser.add_argument("--phase-per-volt", type=float, default=3.2)
    parser.add_argument("--measurement-noise", type=float, default=2.0)
    parser.add_argument("--process-noise", type=float, default=1e-3)
    parser.add_argument("--sim-noise", type=float, default=5e-4)
    parser.add_argument("--sim-drift", type=float, default=0.1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plot", dest="plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument("--plot-history", type=int, default=2_000)
    parser.add_argument("--decimation", type=int, default=200)
    parser.add_argument("--integral-gain", type=float, default=0.1)
    parser.add_argument("--integral-limit", type=float, default=0.5)
    parser.add_argument("--innovation-clip-sigma", type=float, default=0.0)
    parser.add_argument("--hardware-smoothing", type=float, default=0.3)
    parser.add_argument("--rin-window", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--efficiency-target", type=float, default=0.95)
    parser.add_argument("--efficiency-std-target", type=float, default=0.015)
    parser.add_argument("--interactive", action="store_true", help="Use interactive prompt instead of CLI args")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.interactive:
        cfg = gather_inputs()
    else:
        cfg = _args_to_config(args)
        if cfg.mode == "hardware" and not cfg.host:
            print("[Error] Hardware mode requires --host", file=sys.stderr)
            return 2
        if cfg.mode == "simulate" and args.duration is None and cfg.duration is None:
            cfg.duration = 2.0
    try:
        runner, history = run_with_config(cfg)
        summary = runner.summary()
    except KeyboardInterrupt:
        print("\n[Runner] Interrupted by user.", file=sys.stderr)
        return 1

    if summary:
        print("\n===== Summary =====")
        for key, value in summary.items():
            print(f"{key}: {value}")
    else:
        print("No data collected.")

    if history.intensities:
        print(f"Samples collected: {len(history.intensities)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
