"""Minimal numerical helpers implemented with the Python standard library."""
from __future__ import annotations

"""Minimal numerical helpers implemented with the Python standard library."""

import cmath
import math
from typing import Iterable, List, Sequence


def linspace(start: float, stop: float, num: int) -> List[float]:
    """Return evenly spaced numbers over a specified interval."""
    if num <= 0:
        return []
    if num == 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [float(start + step * i) for i in range(num)]


def arange(stop: int) -> List[int]:
    """Return evenly spaced values in the half-open interval [0, stop)."""
    return list(range(stop))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def trapz(y: Sequence[float], dx: float = 1.0) -> float:
    if len(y) < 2:
        return 0.0
    return dx * (0.5 * (y[0] + y[-1]) + sum(y[1:-1]))


def clip(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def abs_list(values: Sequence[complex]) -> List[float]:
    return [abs(v) for v in values]


def abs_squared(values: Sequence[complex]) -> List[float]:
    return [abs(v) ** 2 for v in values]


def add(a: Sequence[complex], b: Sequence[complex]) -> List[complex]:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[complex], b: Sequence[complex]) -> List[complex]:
    return [x - y for x, y in zip(a, b)]


def scale(a: Sequence[complex], scalar: complex) -> List[complex]:
    return [scalar * x for x in a]


def mul(a: Sequence[complex], b: Sequence[complex]) -> List[complex]:
    return [x * y for x, y in zip(a, b)]


def pow_scalar(a: Sequence[complex], exponent: int) -> List[complex]:
    return [x ** exponent for x in a]


def conj(a: Sequence[complex]) -> List[complex]:
    return [x.conjugate() for x in a]


def complex_exp(values: Sequence[complex]) -> List[complex]:
    return [cmath.exp(v) for v in values]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _bit_reversed_indices(n: int) -> List[int]:
    bits = n.bit_length() - 1
    indices = [0] * n
    for i in range(n):
        rev = 0
        value = i
        for _ in range(bits):
            rev = (rev << 1) | (value & 1)
            value >>= 1
        indices[i] = rev
    return indices


def fft(seq: Sequence[complex]) -> List[complex]:
    seq = list(seq)
    n = len(seq)
    if n == 0:
        return []
    if n == 1:
        return [seq[0]]
    if not _is_power_of_two(n):
        return [sum(seq[j] * cmath.exp(-2j * math.pi * k * j / n) for j in range(n)) for k in range(n)]

    indices = _bit_reversed_indices(n)
    data = [seq[idx] for idx in indices]

    length = 2
    while length <= n:
        half = length // 2
        angle = -2j * math.pi / length
        w_len = cmath.exp(angle)
        for start in range(0, n, length):
            w = 1 + 0j
            for pos in range(start, start + half):
                u = data[pos]
                v = w * data[pos + half]
                data[pos] = u + v
                data[pos + half] = u - v
                w *= w_len
        length *= 2
    return data


def ifft(seq: Sequence[complex]) -> List[complex]:
    seq = list(seq)
    n = len(seq)
    if n == 0:
        return []
    conj_seq = [x.conjugate() for x in seq]
    forward = fft(conj_seq)
    return [x.conjugate() / n for x in forward]


def fftshift(seq: Sequence[complex]) -> List[complex]:
    seq = list(seq)
    n = len(seq)
    half = n // 2
    if n % 2 == 0:
        return seq[half:] + seq[:half]
    return seq[half + 1 :] + [seq[half]] + seq[: half + 1]


def fftfreq(n: int, d: float) -> List[float]:
    if n <= 0:
        return []
    val = 1.0 / (n * d)
    half = n // 2
    freqs = [val * i for i in range(half)]
    if n % 2 == 0:
        freqs.append(-val * half)
        freqs.extend([-val * i for i in range(half - 1, 0, -1)])
    else:
        freqs.append(val * half)
        freqs.extend([-val * i for i in range(half, 0, -1)])
    return freqs


def rfft(seq: Sequence[complex]) -> List[complex]:
    full = fft(seq)
    n = len(seq)
    return full[: n // 2 + 1]


def rfftfreq(n: int, d: float) -> List[float]:
    if n <= 0:
        return []
    val = 1.0 / (n * d)
    return [val * i for i in range(n // 2 + 1)]


def unwrap(phases: Sequence[float]) -> List[float]:
    phases = list(phases)
    if not phases:
        return []
    unwrapped = [phases[0]]
    offset = 0.0
    for prev, current in zip(phases, phases[1:]):
        delta = current - prev
        while delta > math.pi:
            delta -= 2 * math.pi
            offset -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
            offset += 2 * math.pi
        unwrapped.append(current + offset)
    return unwrapped


def angle(values: Sequence[complex]) -> List[float]:
    return [math.atan2(v.imag, v.real) for v in values]


def array_like(data: Iterable[complex]) -> List[complex]:
    return [complex(x) for x in data]


def real_array(data: Iterable[float]) -> List[float]:
    return [float(x) for x in data]


def zeros(n: int, complex_: bool = False) -> List[complex]:
    if complex_:
        return [0j for _ in range(n)]
    return [0.0 for _ in range(n)]

