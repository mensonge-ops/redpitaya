import math
import random
import statistics
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Plant:
    gain: float = 0.9
    time_constant: float = 2e-3
    disturbance_amplitude: float = 0.2
    disturbance_frequency: float = 120.0
    disturbance_phase: float = 0.0
    noise_std: float = 0.01
    offset: float = 0.8
    initial_state: float = 0.0


@dataclass
class Options:
    iterations: int = 2000
    sample_rate: float = 20_000.0
    gain: float = 0.08
    perturbation: float = 0.05
    target: float = 1.0
    sample_hold_time: float = 1e-3
    efficiency_threshold: float = 0.95
    best_decay_rate: float = 5e-4
    restore_max_attempts: int = 5
    control_limits: Tuple[float, float] = (-1.0, 1.0)
    measurement_smooth_factor: float = 0.35
    gradient_smooth_factor: float = 0.2
    efficiency_smooth_factor: float = 0.6
    min_perturbation_ratio: float = 0.05
    min_gain_ratio: float = 0.05
    locked_measurement_smooth_factor: float = 0.95
    lock_guard_drop: float = 0.01
    lock_guard_min_efficiency: float = 0.95
    lock_guard_freeze_iterations: int = 80
    lock_guard_perturbation_ratio: float = 0.02
    lock_guard_gradient_damping: float = 0.05
    lock_guard_decay_factor: float = 0.1
    lock_guard_recovery_samples: int = 3
    lock_guard_min_reference: float = 0.99
    seed: int = 1


class SimulationBackend:
    def __init__(self, plant: Plant, opts: Options, rng: random.Random):
        self.plant = plant
        self.opts = opts
        self.rng = rng
        self.state_x = plant.initial_state
        self.state_t = 0.0
        self.control = 0.0

    def apply(self, control: float) -> None:
        self.control = control

    def measure(self) -> float:
        n_hold = max(1, round(self.opts.sample_hold_time * self.opts.sample_rate))
        total = 0.0
        for _ in range(n_hold):
            total += self._step()
        return total / n_hold

    def _step(self) -> float:
        dt = 1.0 / self.opts.sample_rate
        a = math.exp(-dt / self.plant.time_constant)
        disturbance = (
            self.plant.disturbance_amplitude
            * math.sin(
                2.0
                * math.pi
                * self.plant.disturbance_frequency
                * self.state_t
                + self.plant.disturbance_phase
            )
        )
        self.state_x = a * self.state_x + (1 - a) * (
            self.plant.gain * self.control + disturbance
        )
        self.state_t += dt
        noise = self.plant.noise_std * self.rng.gauss(0.0, 1.0)
        return self.plant.offset + self.state_x + noise


@dataclass
class RunResult:
    intensity_history: list
    raw_intensity_history: list
    efficiency_history: list
    error_history: list
    lock_intensity_history: list
    lock_active_history: list


def run_spgd(opts: Options = Options(), plant: Plant = Plant()) -> RunResult:
    rng = random.Random(opts.seed)
    backend = SimulationBackend(plant, opts, rng)

    n_iter = opts.iterations
    control = 0.0
    reference_control = control
    smoothed_gradient = 0.0

    intensity_history = [0.0] * n_iter
    raw_intensity_history = [0.0] * n_iter
    efficiency_history = [0.0] * n_iter
    error_history = [0.0] * n_iter
    lock_intensity_history = [0.0] * n_iter
    lock_active_history = [False] * n_iter

    reference_intensity = float("-inf")
    peak_intensity = float("-inf")
    filtered_intensity = None
    display_efficiency = None
    perturb_scale = 1.0
    lock_active = False
    lock_intensity = float("-inf")
    lock_control = control
    freeze_counter = 0

    backend.apply(control)

    eps = 1e-12

    for k in range(n_iter):
        skip_gradient = freeze_counter > 0
        if skip_gradient:
            freeze_counter -= 1
        else:
            current_perturb = (
                max(opts.min_perturbation_ratio, min(1.0, perturb_scale))
                * opts.perturbation
            )
            perturb = current_perturb * (1 if rng.random() < 0.5 else -1)

            u_plus = max(opts.control_limits[0], min(opts.control_limits[1], control + perturb))
            backend.apply(u_plus)
            y_plus = backend.measure()

            u_minus = max(opts.control_limits[0], min(opts.control_limits[1], control - perturb))
            backend.apply(u_minus)
            y_minus = backend.measure()

            if not (math.isfinite(y_plus) and math.isfinite(y_minus)):
                if k > 0:
                    intensity_history[k] = intensity_history[k - 1]
                    raw_intensity_history[k] = raw_intensity_history[k - 1]
                    efficiency_history[k] = efficiency_history[k - 1]
                    error_history[k] = error_history[k - 1]
                continue

            denom = u_plus - u_minus
            if abs(denom) < eps:
                denom = eps if perturb >= 0 else -eps
            gradient = (y_plus - y_minus) / denom
            if opts.gradient_smooth_factor <= 0 or k == 0:
                smoothed_gradient = gradient
            else:
                smoothed_gradient = (
                    (1 - opts.gradient_smooth_factor) * smoothed_gradient
                    + opts.gradient_smooth_factor * gradient
                )

            gain_scale = max(opts.min_gain_ratio, min(1.0, perturb_scale))
            control = control + (opts.gain * gain_scale) * smoothed_gradient
            control = max(opts.control_limits[0], min(opts.control_limits[1], control))
            backend.apply(control)

        intensity = backend.measure()
        if not math.isfinite(intensity):
            intensity = intensity_history[k - 1] if k > 0 else opts.target

        smooth_factor = opts.measurement_smooth_factor
        if lock_active and opts.locked_measurement_smooth_factor > opts.measurement_smooth_factor:
            smooth_factor = opts.locked_measurement_smooth_factor
        if filtered_intensity is None or smooth_factor <= 0:
            filtered_intensity = intensity
        else:
            filtered_intensity = (1 - smooth_factor) * filtered_intensity + smooth_factor * intensity

        err = opts.target - filtered_intensity
        peak_intensity = max(peak_intensity, filtered_intensity)

        if not math.isfinite(reference_intensity):
            reference_intensity = filtered_intensity
            reference_control = control
        else:
            decay = opts.best_decay_rate
            if lock_active:
                decay *= opts.lock_guard_decay_factor
            reference_intensity = reference_intensity * (1 - decay)
            if filtered_intensity >= reference_intensity:
                reference_intensity = filtered_intensity
                reference_control = control
                if lock_active:
                    lock_control = control
                    lock_intensity = max(lock_intensity, filtered_intensity)

        denom_ref = reference_intensity if reference_intensity > eps else eps
        efficiency = filtered_intensity / denom_ref
        restored = False
        guard_triggered = False

        if efficiency < opts.efficiency_threshold and reference_intensity > 0:
            recovery_control = lock_control if lock_active else reference_control
            control = recovery_control
            backend.apply(control)
            for _ in range(opts.restore_max_attempts):
                intensity = backend.measure()
                if not math.isfinite(intensity):
                    intensity = reference_intensity
                restore_smooth = opts.measurement_smooth_factor
                if lock_active and opts.locked_measurement_smooth_factor > opts.measurement_smooth_factor:
                    restore_smooth = opts.locked_measurement_smooth_factor
                if filtered_intensity is None or restore_smooth <= 0:
                    filtered_intensity = intensity
                else:
                    filtered_intensity = (
                        (1 - restore_smooth) * filtered_intensity + restore_smooth * intensity
                    )
                peak_intensity = max(peak_intensity, filtered_intensity)
                if not math.isfinite(reference_intensity):
                    reference_intensity = filtered_intensity
                    reference_control = control
                else:
                    decay = opts.best_decay_rate
                    if lock_active:
                        decay *= opts.lock_guard_decay_factor
                    reference_intensity = reference_intensity * (1 - decay)
                if filtered_intensity >= reference_intensity:
                    reference_intensity = filtered_intensity
                    reference_control = control
                    if lock_active:
                        lock_control = control
                        lock_intensity = max(lock_intensity, filtered_intensity)
                denom_ref = reference_intensity if reference_intensity > eps else eps
                efficiency = filtered_intensity / denom_ref
                err = opts.target - filtered_intensity
                if efficiency >= opts.efficiency_threshold:
                    break
            restored = True

        if lock_active:
            lock_den = lock_intensity if lock_intensity > eps else eps
            lock_ratio = filtered_intensity / lock_den
            if lock_ratio < 1 - opts.lock_guard_drop:
                control = lock_control
                backend.apply(control)
                recovered = None
                for _ in range(opts.lock_guard_recovery_samples):
                    sample = backend.measure()
                    if math.isfinite(sample):
                        if recovered is None or sample > recovered:
                            recovered = sample
                if recovered is None:
                    recovered = max(lock_intensity, reference_intensity, opts.target)
                rebound_smooth = max(
                    opts.locked_measurement_smooth_factor, opts.measurement_smooth_factor
                )
                if filtered_intensity is None or rebound_smooth <= 0:
                    filtered_intensity = recovered
                else:
                    filtered_intensity = (
                        (1 - rebound_smooth) * filtered_intensity + rebound_smooth * recovered
                    )
                intensity = recovered
                peak_intensity = max(peak_intensity, filtered_intensity)
                if not math.isfinite(reference_intensity):
                    reference_intensity = filtered_intensity
                    reference_control = control
                else:
                    decay = opts.best_decay_rate * opts.lock_guard_decay_factor
                    reference_intensity = reference_intensity * (1 - decay)
                    if filtered_intensity >= reference_intensity:
                        reference_intensity = filtered_intensity
                        reference_control = control
                        if lock_active:
                            lock_control = control
                            lock_intensity = max(lock_intensity, filtered_intensity)
                denom_ref = reference_intensity if reference_intensity > eps else eps
                efficiency = filtered_intensity / denom_ref
                err = opts.target - filtered_intensity
                guard_triggered = True

        if (
            (not lock_active)
            and efficiency >= opts.lock_guard_min_efficiency
            and reference_intensity >= opts.target * opts.lock_guard_min_reference
        ):
            lock_active = True
            lock_control = reference_control
            lock_intensity = max(reference_intensity, filtered_intensity)
        elif lock_active:
            prior_lock = lock_intensity
            lock_decay = opts.best_decay_rate * opts.lock_guard_decay_factor
            lock_intensity = prior_lock * (1 - lock_decay)
            if filtered_intensity >= prior_lock:
                lock_control = control
                lock_intensity = filtered_intensity

        if restored or guard_triggered:
            freeze_counter = max(freeze_counter, opts.lock_guard_freeze_iterations)
            perturb_scale = min(perturb_scale, opts.lock_guard_perturbation_ratio)
            smoothed_gradient *= opts.lock_guard_gradient_damping
            if not lock_active:
                if reference_intensity >= opts.target * opts.lock_guard_min_reference:
                    lock_active = True
                    lock_control = reference_control
                    lock_intensity = max(reference_intensity, filtered_intensity)
            else:
                lock_control = control
                lock_intensity = max(lock_intensity, filtered_intensity)

        raw_intensity_history[k] = intensity
        intensity_history[k] = filtered_intensity
        eff_smooth = opts.efficiency_smooth_factor
        if lock_active:
            eff_smooth = max(eff_smooth, opts.locked_measurement_smooth_factor)
        if display_efficiency is None or eff_smooth <= 0:
            display_efficiency = efficiency
        else:
            display_efficiency = (
                (1 - eff_smooth) * display_efficiency + eff_smooth * efficiency
            )
        efficiency_history[k] = display_efficiency
        error_history[k] = err
        lock_intensity_history[k] = lock_intensity
        lock_active_history[k] = lock_active

        if restored or guard_triggered:
            perturb_scale = min(perturb_scale, opts.lock_guard_perturbation_ratio)
        elif reference_intensity > 0:
            drop = max(0.0, efficiency - opts.efficiency_threshold) / max(
                1 - opts.efficiency_threshold, eps
            )
            perturb_scale = max(opts.min_perturbation_ratio, min(1.0, 1.0 - drop))
        else:
            perturb_scale = 1.0

    return RunResult(
        intensity_history=intensity_history,
        raw_intensity_history=raw_intensity_history,
        efficiency_history=efficiency_history,
        error_history=error_history,
        lock_intensity_history=lock_intensity_history,
        lock_active_history=lock_active_history,
    )


def main() -> None:
    result = run_spgd()
    steady = result.efficiency_history[500:]
    min_eff = min(steady)
    mean_eff = sum(steady) / len(steady)
    std_eff = statistics.pstdev(steady)
    assert min_eff >= 0.92, f"Minimum efficiency {min_eff:.3f} below 0.92"
    assert all(val >= 0.95 - 1e-6 for val in steady), "Efficiency dipped below 0.95 in steady state"
    assert std_eff < 0.01, f"Efficiency std {std_eff:.4f} too large"
    print(
        f"Steady-state efficiency min={min_eff:.4f}, mean={mean_eff:.4f}, std={std_eff:.4f}"
    )


if __name__ == "__main__":
    main()
