"""Minimal NumPy fallback used for environments without the real package.

This module only implements the tiny subset of the :mod:`numpy` API that is
required by the Red Pitaya locking script.  It is intentionally lightweight
and not a drop-in replacement for full NumPy.  The functionality provided
here is sufficient for small vector operations, basic random number
generation and a couple of convenience routines.
"""
from __future__ import annotations

import cmath
import math
import random as _random
from typing import Iterable, Iterator, List, Sequence


class ndarray(List[float]):
    """Very small 1-D array type supporting basic arithmetic."""

    def __init__(
        self,
        iterable: Iterable[complex] = (),
        *,
        dtype=float,
        shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__(dtype(value) for value in iterable)
        if shape is None:
            shape = (len(self),)
        self._shape = tuple(int(dim) for dim in shape)

    # NumPy compatibility -------------------------------------------------
    @property
    def size(self) -> int:
        result = 1
        for dim in self._shape:
            result *= dim
        return result

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    def copy(self) -> "ndarray":
        return ndarray(self, dtype=lambda x: x, shape=self._shape)

    def ravel(self) -> "ndarray":
        return ndarray(self, dtype=lambda x: x, shape=(len(self),))

    # Arithmetic operations ----------------------------------------------
    def _binary_op(self, other, op) -> "ndarray":
        if isinstance(other, ndarray):
            if len(self) != len(other):
                raise ValueError("operands could not be broadcast together")
            values = [op(a, b) for a, b in zip(self, other)]
        else:
            values = [op(a, other) for a in self]
        dtype = complex if any(isinstance(v, complex) for v in values) else float
        shape = self._shape if isinstance(other, ndarray) and self._shape == other._shape else (len(values),)
        return ndarray(values, dtype=dtype, shape=shape)

    def __add__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other) -> "ndarray":
        return self.__add__(other)

    def __sub__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "ndarray":
        if isinstance(other, ndarray):
            return other.__sub__(self)
        values = [other - a for a in self]
        dtype = complex if any(isinstance(v, complex) for v in values) else float
        return ndarray(values, dtype=dtype, shape=(len(values),))

    def __mul__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other) -> "ndarray":
        return self.__mul__(other)

    def __truediv__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other) -> "ndarray":
        if isinstance(other, ndarray):
            return other.__truediv__(self)
        values = [other / a for a in self]
        dtype = complex if any(isinstance(v, complex) for v in values) else float
        return ndarray(values, dtype=dtype, shape=(len(values),))

    def __neg__(self) -> "ndarray":
        return ndarray(-a for a in self)

    def __pow__(self, exponent) -> "ndarray":
        if isinstance(exponent, ndarray):
            if len(self) != len(exponent):
                raise ValueError("operands could not be broadcast together")
            values = [a ** b for a, b in zip(self, exponent)]
        else:
            values = [a ** exponent for a in self]
        dtype = complex if any(isinstance(v, complex) for v in values) else float
        shape = self._shape if isinstance(exponent, ndarray) and self._shape == exponent._shape else (len(values),)
        return ndarray(values, dtype=dtype, shape=shape)

    def __rpow__(self, base) -> "ndarray":
        if isinstance(base, ndarray):
            return base.__pow__(self)
        values = [base ** a for a in self]
        dtype = complex if any(isinstance(v, complex) for v in values) else float
        return ndarray(values, dtype=dtype, shape=(len(values),))

    # In-place operators -------------------------------------------------
    def _assign_from(self, result: "ndarray") -> "ndarray":
        self[:] = result
        self._shape = result._shape
        return self

    def __iadd__(self, other) -> "ndarray":
        return self._assign_from(self.__add__(other))

    def __isub__(self, other) -> "ndarray":
        return self._assign_from(self.__sub__(other))

    def __imul__(self, other) -> "ndarray":
        return self._assign_from(self.__mul__(other))

    def __itruediv__(self, other) -> "ndarray":
        return self._assign_from(self.__truediv__(other))

    # Allow ``list(ndarray)`` to work as expected.
    def __iter__(self) -> Iterator[float]:  # type: ignore[override]
        return super().__iter__()

    # Comparison operators returning boolean arrays ----------------------
    def _compare(self, other, op) -> "ndarray":
        if isinstance(other, ndarray):
            if len(self) != len(other):
                raise ValueError("operands could not be broadcast together")
            return ndarray((op(a, b) for a, b in zip(self, other)), dtype=bool)
        return ndarray((op(a, other) for a in self), dtype=bool)

    def __lt__(self, other) -> "ndarray":
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other) -> "ndarray":
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other) -> "ndarray":
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other) -> "ndarray":
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other) -> "ndarray":  # type: ignore[override]
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other) -> "ndarray":  # type: ignore[override]
        return self._compare(other, lambda a, b: a != b)

    # Boolean indexing ---------------------------------------------------
    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, ndarray):
            if len(key) != len(self):
                raise IndexError("boolean index did not match array length")
            if not key:
                return ndarray()
            if all(isinstance(flag, bool) for flag in key):
                return ndarray(value for value, flag in zip(self, key) if flag)
        return super().__getitem__(key)


def array(data: Iterable[complex], dtype=float, *, shape: Sequence[int] | None = None) -> ndarray:
    return ndarray(data, dtype=dtype, shape=shape)


def asarray(data: Iterable[complex], dtype=float) -> ndarray:
    if isinstance(data, ndarray):
        return ndarray((dtype(x) for x in data), dtype=dtype, shape=data.shape)
    if isinstance(data, Sequence) and data and isinstance(data[0], (list, tuple, ndarray)):
        rows = [asarray(row, dtype=dtype) for row in data]
        if not rows:
            return ndarray((), dtype=dtype, shape=(0,))
        row_len = rows[0].size
        if any(row.size != row_len for row in rows):
            raise ValueError("cannot create ndarray from ragged nested sequences")
        flat: List[complex] = []
        for row in rows:
            flat.extend(row)
        return ndarray(flat, dtype=dtype, shape=(len(rows), row_len))
    return ndarray(data, dtype=dtype)


def _resolve_shape(length) -> Sequence[int]:
    if isinstance(length, (tuple, list)):
        return tuple(int(dim) for dim in length)
    return (int(length),)


def zeros(length: int | Sequence[int], dtype=float) -> ndarray:
    shape = _resolve_shape(length)
    total = 1
    for dim in shape:
        total *= dim
    return ndarray((dtype(0) for _ in range(total)), dtype=dtype, shape=shape)


def empty(length: int | Sequence[int], dtype=float) -> ndarray:
    # ``empty`` in NumPy returns arbitrary memory; using zeros keeps things simple.
    return zeros(length, dtype=dtype)


def concatenate(values: Sequence[Iterable[complex]]) -> ndarray:
    result: List[complex] = []
    for value in values:
        result.extend(value)
    return ndarray(result, shape=(len(result),))


def arange(stop: int, dtype=float) -> ndarray:
    values = [dtype(i) for i in range(int(stop))]
    return ndarray(values, dtype=dtype, shape=(len(values),))


def mean(data: Iterable[float]) -> float:
    data_list = list(data)
    if not data_list:
        return 0.0
    return sum(data_list) / len(data_list)


def min(data: Iterable[float]) -> float:
    return builtins.min(data)  # type: ignore[name-defined]


def max(data: Iterable[float]) -> float:
    return builtins.max(data)  # type: ignore[name-defined]


def abs(value):
    if isinstance(value, ndarray):
        return ndarray((abs(x) for x in value))
    return builtins.abs(value)  # type: ignore[name-defined]


def _exp_scalar(x):
    if isinstance(x, complex):
        return cmath.exp(x)
    return math.exp(x)


def exp(value):
    if isinstance(value, ndarray):
        values = [_exp_scalar(x) for x in value]
        dtype = complex if any(isinstance(v, complex) for v in values) else float
        return ndarray(values, dtype=dtype)
    return _exp_scalar(value)


def clip(data: Iterable[float], low: float, high: float) -> ndarray:
    return ndarray((builtins.min(builtins.max(x, low), high) for x in data))


def geomspace(start: float, stop: float, num: int) -> ndarray:
    if num <= 1:
        return ndarray([float(start)])
    log_start = math.log(start)
    log_stop = math.log(stop)
    step = (log_stop - log_start) / (num - 1)
    return ndarray(math.exp(log_start + step * k) for k in range(num))


def array2string(data: Iterable[float], precision: int = 4) -> str:
    formatted = []
    fmt = f"{{:.{precision}f}}"
    for value in data:
        if isinstance(value, bool):
            formatted.append("True" if value else "False")
        elif isinstance(value, (int, float)):
            formatted.append(fmt.format(value))
        else:
            formatted.append(str(value))
    return "[" + ", ".join(formatted) + "]"


def unwrap(phases: Iterable[float]) -> ndarray:
    phases_list = list(phases)
    if not phases_list:
        return ndarray()
    unwrapped = [phases_list[0]]
    two_pi = 2.0 * math.pi
    for current in phases_list[1:]:
        delta = current - unwrapped[-1]
        while delta > math.pi:
            current -= two_pi
            delta = current - unwrapped[-1]
        while delta < -math.pi:
            current += two_pi
            delta = current - unwrapped[-1]
        unwrapped.append(current)
    return ndarray(unwrapped)


def log10(values: Iterable[float]) -> ndarray:
    return ndarray(math.log10(v) for v in values)


def all(values: Iterable[bool]) -> bool:
    return builtins.all(values)  # type: ignore[name-defined]


def empty_like(other: Iterable[complex]) -> ndarray:
    return zeros(len(list(other)))


inf = float("inf")


class _RandomGenerator:
    def __init__(self, seed=None) -> None:
        self._rng = _random.Random(seed)

    def choice(self, options: Sequence[float], size: int) -> ndarray:
        return ndarray(self._rng.choice(options) for _ in range(size))

    def normal(self, *, scale: float, size: int) -> ndarray:
        return ndarray(self._rng.gauss(0.0, scale) for _ in range(size))

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)


class _RandomModule:
    def default_rng(self, seed=None) -> _RandomGenerator:
        return _RandomGenerator(seed)


random = _RandomModule()


# ``np.ndarray`` is frequently used for annotations in the main script.
__all__ = [
    "ndarray",
    "array",
    "asarray",
    "zeros",
    "empty",
    "concatenate",
    "arange",
    "mean",
    "min",
    "max",
    "abs",
    "exp",
    "clip",
    "geomspace",
    "array2string",
    "unwrap",
    "log10",
    "all",
    "inf",
    "random",
]


# ``builtins`` is only required to keep ``min``/``max`` from shadowing.
import builtins  # noqa: E402  (import after __all__ for clarity)
