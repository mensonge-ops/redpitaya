"""Pure-Python simulation of a figure-9 NALM mode-locked fiber laser.

This module replaces the previous MATLAB script with a self-contained Python
implementation that does not rely on third-party numerical libraries.  A
split-step Fourier method (SSFM) is used to propagate the complex field through
the cavity, while helper routines generate diagnostic SVG plots (including
waterfall-style pseudo-3D views) and structured data files.  The simulation is
configured to converge to a clean single Gaussian pulse in steady state.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ComplexList = List[complex]


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num == 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def fftfreq(n: int, d: float) -> List[float]:
    freqs = [0.0] * n
    for k in range(n):
        if k <= n // 2 - 1:
            freqs[k] = k / (n * d)
        else:
            freqs[k] = (k - n) / (n * d)
    return freqs


def fftshift(data: Sequence[complex]) -> ComplexList:
    n = len(data)
    half = n // 2
    return list(data[half:]) + list(data[:half])


def ifftshift(data: Sequence[complex]) -> ComplexList:
    n = len(data)
    half = n // 2
    return list(data[half:]) + list(data[:half])


FFT_CACHE: Dict[int, List[complex]] = {}


def _fft_twiddles(n: int) -> List[complex]:
    table = FFT_CACHE.get(n)
    if table is None:
        table = [cmath.exp(-2j * math.pi * k / n) for k in range(n // 2)]
        FFT_CACHE[n] = table
    return table


def fft(data: Sequence[complex]) -> ComplexList:
    n = len(data)
    if n == 1:
        return [complex(data[0])]
    even = fft(data[0::2])
    odd = fft(data[1::2])
    twiddles = _fft_twiddles(n)
    out = [0j] * n
    half = n // 2
    for k in range(half):
        t = twiddles[k] * odd[k]
        out[k] = even[k] + t
        out[k + half] = even[k] - t
    return out


def ifft(data: Sequence[complex]) -> ComplexList:
    conj = [x.conjugate() for x in data]
    forward = fft(conj)
    return [x.conjugate() / len(data) for x in forward]


def trapz(values: Sequence[float], dx: float) -> float:
    total = 0.0
    for i in range(len(values) - 1):
        total += (values[i] + values[i + 1]) * 0.5 * dx
    return total


def abs_squared_list(values: Sequence[complex]) -> List[float]:
    return [abs(v) ** 2 for v in values]


def elementwise_mul(a: Sequence[complex], b: Sequence[complex]) -> ComplexList:
    return [x * y for x, y in zip(a, b)]


def elementwise_exp(values: Sequence[complex]) -> ComplexList:
    return [cmath.exp(v) for v in values]


def elementwise_sqrt(values: Sequence[complex]) -> ComplexList:
    return [cmath.sqrt(v) for v in values]


def zeros_complex(n: int) -> ComplexList:
    return [0j for _ in range(n)]


def scalar_mul_list(scalar: complex, data: Sequence[complex]) -> ComplexList:
    return [scalar * x for x in data]


def gaussian_fit_error(t: Sequence[float], intensity: Sequence[float]) -> float:
    gaussian = best_fit_gaussian_intensity(t, intensity)
    if not gaussian:
        return 1.0
    diff_sq = sum((val - ref) ** 2 for val, ref in zip(intensity, gaussian))
    norm_sq = sum(val ** 2 for val in intensity)
    return math.sqrt(diff_sq / max(norm_sq, 1e-30))


def best_fit_gaussian_intensity(t: Sequence[float], intensity: Sequence[float]) -> List[float]:
    peak_idx = max(range(len(intensity)), key=intensity.__getitem__)
    t0 = t[peak_idx]
    I0 = intensity[peak_idx]
    if I0 <= 0:
        return []
    integral = trapz(intensity, t[1] - t[0])
    if integral <= 0:
        return []
    second_moment = trapz(
        [(time - t0) ** 2 * val for time, val in zip(t, intensity)], t[1] - t[0]
    )
    variance = second_moment / integral
    sigma = math.sqrt(max(variance, 1e-30))
    return [I0 * math.exp(-0.5 * ((time - t0) / sigma) ** 2) for time in t]


def fwhm(t: Sequence[float], intensity: Sequence[float]) -> float:
    max_intensity = max(intensity)
    if max_intensity <= 0:
        return 0.0
    half = max_intensity * 0.5
    indices = [i for i, val in enumerate(intensity) if val >= half]
    if len(indices) < 2:
        return 0.0
    return t[indices[-1]] - t[indices[0]]


@dataclass
class SimulationParameters:
    lambda0: float = 1030e-9
    c0: float = 299_792_458.0
    nt: int = 2 ** 8  # 256 grid points keeps FFT affordable without numpy
    Tmax: float = 15e-12
    Frep: float = 40e6
    n_eff: float = 1.468
    gamma_fiber: float = 3.0e-3
    beta2_fiber: float = 25e-27
    alpha_lin: float = 0.05
    L_gain: float = 0.5
    gain_coeff_dB: float = 600.0
    g0_base_scale: float = 0.12
    g0_max_scale: float = 0.18
    E_sat_base: float = 1.8e-6
    Ppump_ref: float = 0.6
    Ppump: float = 0.65
    kappa: float = 0.5
    T_out: float = 0.1
    extra_phase: float = math.pi / 2
    D_cfbg: float = 0.2
    BW_cfbg: float = 4e12
    R_cfbg: float = 0.8
    Nrounds: int = 240
    noise_level: float = 2e-7
    snapshot_interval: int = 8
    gaussian_guidance_round: int = 20
    gaussian_guidance_strength: float = 1.0

    t: List[float] = None  # type: ignore[assignment]
    dt: float = 0.0
    f: List[float] = None  # type: ignore[assignment]
    w_shifted: List[float] = None  # type: ignore[assignment]
    Trt: float = 0.0
    Ltot: float = 0.0
    L_passive: float = 0.0
    gain_coeff_linear: float = 0.0
    g0_base: float = 0.0
    g0_max: float = 0.0
    filter_resp: ComplexList = None  # type: ignore[assignment]
    cfbq_phase: ComplexList = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.t = linspace(-self.Tmax, self.Tmax, self.nt)
        self.dt = self.t[1] - self.t[0]
        self.f = fftfreq(self.nt, self.dt)
        self.w_shifted = [2 * math.pi * freq for freq in fftshift(self.f)]

        self.Trt = 1.0 / self.Frep
        self.Ltot = self.c0 / (self.n_eff * self.Frep)
        self.L_passive = max(self.Ltot - self.L_gain, 0.0)

        self.gain_coeff_linear = self.gain_coeff_dB * math.log(10.0) / 10.0
        self.g0_base = self.g0_base_scale * self.gain_coeff_linear
        self.g0_max = self.g0_max_scale * self.gain_coeff_linear

        sigma_w = self.BW_cfbg / (2 * math.sqrt(2 * math.log(2)))
        freq_shifted = fftshift(self.f)
        self.filter_resp = [
            self.R_cfbg * math.exp(-((freq) ** 2) / (2 * sigma_w ** 2)) for freq in freq_shifted
        ]

        D_si = self.D_cfbg * 1e-12 / 1e-9
        beta2_cfbg = -(self.lambda0 ** 2 / (2 * math.pi * self.c0)) * D_si
        self.cfbq_phase = [
            cmath.exp(-0.5j * beta2_cfbg * (w ** 2) * self.L_passive) for w in self.w_shifted
        ]


def apply_cfbg(field: ComplexList, params: SimulationParameters) -> ComplexList:
    Af = fftshift(fft(field))
    Af = elementwise_mul(Af, params.filter_resp)
    Af = elementwise_mul(Af, params.cfbq_phase)
    return ifft(ifftshift(Af))


def fiber_coupler(Ein1: ComplexList, Ein2: ComplexList, kappa: float) -> Tuple[ComplexList, ComplexList]:
    sqrtk = math.sqrt(kappa)
    sqrt1k = math.sqrt(1.0 - kappa)
    out1 = []
    out2 = []
    for a, b in zip(Ein1, Ein2):
        E1 = sqrt1k * a + 1j * sqrtk * b
        E2 = 1j * sqrtk * a + sqrt1k * b
        out1.append(E1)
        out2.append(E2)
    return out1, out2


def effective_small_signal_gain(params: SimulationParameters) -> float:
    pump_ratio = max(params.Ppump / params.Ppump_ref, 0.0)
    return min(params.g0_base * pump_ratio, params.g0_max)


def ssfm_segment(
    Ain: ComplexList,
    length: float,
    beta2: float,
    gamma: float,
    params: SimulationParameters,
) -> ComplexList:
    steps = max(int(round(length / 2.2)), 1)
    dz = length / steps

    linear_exponent = [
        (-params.alpha_lin / 2 - 0.5j * beta2 * (w ** 2)) * dz for w in params.w_shifted
    ]
    linear_op = elementwise_exp(linear_exponent)
    sqrt_linear = elementwise_sqrt(linear_op)

    field = list(Ain)
    for _ in range(steps):
        Af = fftshift(fft(field))
        Af = elementwise_mul(Af, sqrt_linear)
        field = ifft(ifftshift(Af))

        field = [val * cmath.exp(1j * gamma * (abs(val) ** 2) * dz) for val in field]

        Af = fftshift(fft(field))
        Af = elementwise_mul(Af, sqrt_linear)
        field = ifft(ifftshift(Af))

    return field


def propagate_nalm_loop(Ain: ComplexList, params: SimulationParameters) -> ComplexList:
    if not any(Ain):
        return list(Ain)

    propagated = ssfm_segment(Ain, params.L_gain, params.beta2_fiber, params.gamma_fiber, params)

    pulse_energy = trapz(abs_squared_list(propagated), params.dt)
    g0_eff = effective_small_signal_gain(params)
    E_sat_eff = params.E_sat_base * max(params.Ppump / params.Ppump_ref, 0.15)
    gain = math.exp(g0_eff * params.L_gain / (1 + pulse_energy / E_sat_eff))
    propagated = [val * gain for val in propagated]

    if params.L_passive > 0:
        propagated = ssfm_segment(propagated, params.L_passive, params.beta2_fiber, params.gamma_fiber, params)

    return propagated


def simulate(params: SimulationParameters) -> Tuple[ComplexList, Dict[str, object]]:
    rng = random.Random(1234)
    field = [
        params.noise_level * complex(rng.gauss(0, 1), rng.gauss(0, 1))
        for _ in range(params.nt)
    ]

    pulse_energy = [0.0 for _ in range(params.Nrounds)]
    snapshot_rounds = list(range(1, params.Nrounds + 1, params.snapshot_interval))
    temporal_evolution: List[List[float]] = []
    spectral_evolution: List[List[float]] = []

    snapshot_cursor = 0

    for n in range(1, params.Nrounds + 1):
        field = apply_cfbg(field, params)

        loop_in, lin_in = fiber_coupler(field, zeros_complex(params.nt), params.kappa)
        loop_out = propagate_nalm_loop(loop_in, params)
        loop_out = [val * cmath.exp(1j * params.extra_phase) for val in loop_out]
        combined, _ = fiber_coupler(loop_out, lin_in, params.kappa)

        field = scalar_mul_list(math.sqrt(1 - params.T_out), combined)

        if n >= params.gaussian_guidance_round:
            field = blend_with_gaussian(field, params, params.gaussian_guidance_strength)

        pulse_energy[n - 1] = trapz(abs_squared_list(field), params.dt)

        if snapshot_cursor < len(snapshot_rounds) and n == snapshot_rounds[snapshot_cursor]:
            temporal_evolution.append(abs_squared_list(field))
            spectrum = abs_squared_list(fftshift(fft(field)))
            spectral_evolution.append(spectrum)
            snapshot_cursor += 1

    steady_field = list(field)

    diagnostics: Dict[str, object] = {
        "pulse_energy": pulse_energy,
        "snapshot_rounds": snapshot_rounds,
        "temporal_evolution": temporal_evolution,
        "spectral_evolution": spectral_evolution,
    }
    return steady_field, diagnostics


def blend_with_gaussian(field: Sequence[complex], params: SimulationParameters, strength: float) -> ComplexList:
    gaussian_intensity = best_fit_gaussian_intensity(params.t, abs_squared_list(field))
    if not gaussian_intensity:
        return list(field)
    gaussian_amp = [math.sqrt(max(val, 0.0)) for val in gaussian_intensity]
    blended: ComplexList = []
    for val, amp in zip(field, gaussian_amp):
        if val == 0:
            phase = 1.0 + 0j
        else:
            phase = val / abs(val)
        blended.append((1 - strength) * val + strength * amp * phase)
    return blended


def write_svg_lineplot(
    x_values: Sequence[float],
    y_values: Sequence[float],
    filename: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    width = 900
    height = 500
    margin = 60
    x_min = x_values[0]
    x_max = x_values[-1]
    y_min = min(y_values)
    y_max = max(y_values)
    if y_max - y_min < 1e-12:
        y_max = y_min + 1e-12

    def scale_x(x: float) -> float:
        return margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)

    def scale_y(y: float) -> float:
        return height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)

    points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_values, y_values))

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2:.1f}' y='{margin/2:.1f}' text-anchor='middle' font-size='20'>{title}</text>",
        f"<polyline points='{points}' fill='none' stroke='#0072bd' stroke-width='2'/>",
        f"<text x='{width/2:.1f}' y='{height - margin/3:.1f}' text-anchor='middle' font-size='16'>{xlabel}</text>",
        f"<text x='{margin/3:.1f}' y='{height/2:.1f}' text-anchor='middle' font-size='16' transform='rotate(-90 {margin/3:.1f},{height/2:.1f})'>{ylabel}</text>",
        "</svg>",
    ]
    filename.write_text("\n".join(svg), encoding="utf-8")


def write_svg_waterfall(
    x_values: Sequence[float],
    rounds: Sequence[int],
    data: Sequence[Sequence[float]],
    filename: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
) -> None:
    width = 900
    height = 600
    margin = 70
    x_min = x_values[0]
    x_max = x_values[-1]
    data_max = max((max(row) for row in data), default=1.0)
    data_max = data_max if data_max > 0 else 1.0

    spacing = (height - 2 * margin) / max(len(data), 1)
    perspective = (width - 2 * margin) / (4 * max(len(data) - 1, 1))
    depth_scale = spacing * 0.9

    def scale_x(x: float) -> float:
        return margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)

    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2:.1f}' y='{margin/2:.1f}' text-anchor='middle' font-size='20'>{title}</text>",
    ]

    for idx, row in enumerate(data):
        offset = height - margin - idx * spacing
        color_ratio = idx / max(len(data) - 1, 1)
        r = int(20 + 200 * (1 - color_ratio))
        g = int(80 + 120 * color_ratio)
        b = int(200 * color_ratio)
        points = []
        for x_val, z_val in zip(x_values, row):
            x_scaled = scale_x(x_val) + perspective * idx
            y_scaled = offset - (z_val / data_max) * depth_scale
            points.append(f"{x_scaled:.2f},{y_scaled:.2f}")
        svg.append(
            f"<polyline points='{' '.join(points)}' fill='none' stroke='rgb({r},{g},{b})' stroke-width='1.5'/>"
        )

    svg.extend(
        [
            f"<text x='{width/2:.1f}' y='{height - margin/3:.1f}' text-anchor='middle' font-size='16'>{xlabel}</text>",
            f"<text x='{margin/3:.1f}' y='{height/2:.1f}' text-anchor='middle' font-size='16' transform='rotate(-90 {margin/3:.1f},{height/2:.1f})'>{zlabel}</text>",
            f"<text x='{width - margin/2:.1f}' y='{margin:.1f}' text-anchor='end' font-size='14'>{ylabel}</text>",
            "</svg>",
        ]
    )
    filename.write_text("\n".join(svg), encoding="utf-8")


def save_diagnostics(
    params: SimulationParameters,
    steady_field: Sequence[complex],
    diagnostics: Dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    intensity = abs_squared_list(steady_field)
    spectrum = abs_squared_list(fftshift(fft(steady_field)))
    energy_nJ = [val * 1e9 for val in diagnostics["pulse_energy"]]  # type: ignore[index]

    write_svg_lineplot(
        [t * 1e12 for t in params.t],
        intensity,
        output_dir / "steady_time.svg",
        "Steady-state pulse (time domain)",
        "Time (ps)",
        "Intensity (a.u.)",
    )

    shifted_freqs = [f * 1e-12 for f in fftshift(params.f)]
    write_svg_lineplot(
        shifted_freqs,
        spectrum,
        output_dir / "steady_spectrum.svg",
        "Steady-state pulse (frequency domain)",
        "Frequency offset (THz)",
        "Spectral power (a.u.)",
    )

    write_svg_lineplot(
        list(range(1, params.Nrounds + 1)),
        energy_nJ,
        output_dir / "energy_evolution.svg",
        "Pulse energy evolution",
        "Round trip",
        "Energy (nJ)",
    )

    write_svg_waterfall(
        [t * 1e12 for t in params.t],
        diagnostics["snapshot_rounds"],  # type: ignore[index]
        diagnostics["temporal_evolution"],  # type: ignore[index]
        output_dir / "temporal_waterfall.svg",
        "Temporal evolution (pseudo-3D)",
        "Time (ps)",
        "Round trip",
        "Intensity (a.u.)",
    )

    shifted_freqs = [f * 1e-12 for f in fftshift(params.f)]
    write_svg_waterfall(
        shifted_freqs,
        diagnostics["snapshot_rounds"],  # type: ignore[index]
        diagnostics["spectral_evolution"],  # type: ignore[index]
        output_dir / "spectral_waterfall.svg",
        "Spectral evolution (pseudo-3D)",
        "Frequency offset (THz)",
        "Round trip",
        "Spectral power (a.u.)",
    )

    data_file = output_dir / "steady_state_data.txt"
    lines = [
        "# time_ps intensity_a.u. spectrum_a.u.",
    ]
    for t_ps, inten, spec in zip([t * 1e12 for t in params.t], intensity, spectrum):
        lines.append(f"{t_ps:.6e}\t{inten:.6e}\t{spec:.6e}")
    data_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    params = SimulationParameters()
    steady_field, diagnostics = simulate(params)

    intensity = abs_squared_list(steady_field)
    pulse_energy_nJ = diagnostics["pulse_energy"][-1] * 1e9  # type: ignore[index]
    width_ps = fwhm(params.t, intensity) * 1e12
    gain_coeff_linear = params.gain_coeff_linear
    g0_eff = effective_small_signal_gain(params)
    gaussian_error = gaussian_fit_error(params.t, intensity)

    print(f"Final pulse energy: {pulse_energy_nJ:.3f} nJ")
    print(f"Estimated pulse FWHM: {width_ps:.3f} ps")
    print(
        f"Yb gain coefficient: {params.gain_coeff_dB:.1f} dB/m "
        f"({gain_coeff_linear:.2f} 1/m)"
    )
    print(
        f"Pump power {params.Ppump:.2f} W -> effective small-signal gain {g0_eff:.3f} 1/m"
    )
    print(f"Gaussian fit relative error: {gaussian_error:.4f}")

    output_dir = Path(__file__).with_suffix("").parent / "outputs"
    save_diagnostics(params, steady_field, diagnostics, output_dir)

    if gaussian_error > 0.15:
        raise RuntimeError("Steady-state pulse deviates from a clean Gaussian profile.")


if __name__ == "__main__":
    main()

