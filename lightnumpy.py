"""Minimal drop-in replacement for a handful of NumPy features."""
from __future__ import annotations

import builtins
import cmath
import math
import random as _stdlib_random
from collections.abc import Sequence
from typing import Iterator, List, Tuple, Union

Number = Union[int, float, complex]


def _ensure_sequence(value: Union["ndarray", Sequence, Number]) -> Union["ndarray", Sequence, Number]:
    if isinstance(value, ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return ndarray(value)
    return value


class ndarray:
    """Very small list-backed array type supporting basic arithmetic."""

    def __init__(self, data: Union["ndarray", Sequence, Number] = ()):  # type: ignore[assignment]
        if isinstance(data, ndarray):
            self._data = data.tolist()
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            self._data = [self._coerce_element(elem) for elem in data]
        elif data == ():
            self._data = []
        else:
            self._data = [self._coerce_scalar(data)]

    @staticmethod
    def _coerce_element(value: Union["ndarray", Sequence, Number]) -> Union[Number, "ndarray"]:
        if isinstance(value, ndarray):
            return ndarray(value.tolist())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return ndarray(value)
        return ndarray._coerce_scalar(value)

    @staticmethod
    def _coerce_scalar(value: Number) -> Number:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, complex):
            return value
        return float(value)

    # Basic container protocol -------------------------------------------------
    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __getitem__(self, item):
        result = self._data[item]
        if isinstance(item, slice):
            return ndarray(result)
        if isinstance(result, list):
            return ndarray(result)
        return result

    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            if isinstance(value, ndarray):
                self._data[key] = value.tolist()
            elif isinstance(value, Sequence):
                self._data[key] = list(value)
            else:
                raise TypeError("Slice assignment requires a sequence")
        else:
            if isinstance(value, ndarray):
                self._data[key] = value.tolist()
            else:
                self._data[key] = value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ndarray({self.tolist()!r})"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return "[" + ", ".join(str(item) for item in self) + "]"

    # Numeric helpers ----------------------------------------------------------
    def _binary_op(self, other, op) -> "ndarray":
        other = _ensure_sequence(other)
        if isinstance(other, ndarray):
            if len(self) != len(other):
                raise ValueError("Arrays must have the same length")
            return ndarray([op(a, b) for a, b in zip(self, other)])
        return ndarray([op(a, other) for a in self])

    def __add__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other) -> "ndarray":
        return self.__add__(other)

    def __sub__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other) -> "ndarray":
        other = _ensure_sequence(other)
        if isinstance(other, ndarray):
            return other.__sub__(self)
        return ndarray([other - a for a in self])

    def __mul__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other) -> "ndarray":
        return self.__mul__(other)

    def __truediv__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other) -> "ndarray":
        other = _ensure_sequence(other)
        if isinstance(other, ndarray):
            return other.__truediv__(self)
        return ndarray([other / a for a in self])

    def __neg__(self) -> "ndarray":
        return ndarray([-a for a in self])

    def __abs__(self) -> "ndarray":
        return ndarray([abs(a) for a in self])

    def __mod__(self, other) -> "ndarray":
        return self._binary_op(other, lambda a, b: a % b)

    def __rmod__(self, other) -> "ndarray":
        other = _ensure_sequence(other)
        if isinstance(other, ndarray):
            return other.__mod__(self)
        return ndarray([other % a for a in self])

    def __pow__(self, exponent) -> "ndarray":
        exponent = _ensure_sequence(exponent)
        if isinstance(exponent, ndarray):
            return ndarray([a ** b for a, b in zip(self, exponent)])
        return ndarray([a ** exponent for a in self])

    def __rpow__(self, base) -> "ndarray":
        base = _ensure_sequence(base)
        if isinstance(base, ndarray):
            return base.__pow__(self)
        return ndarray([base ** a for a in self])

    def copy(self) -> "ndarray":
        return ndarray(self.tolist())

    def tolist(self):  # pragma: no cover - simple conversion
        result: List = []
        for item in self._data:
            if isinstance(item, ndarray):
                result.append(item.tolist())
            elif isinstance(item, list):
                result.append(ndarray(item).tolist())
            else:
                result.append(item)
        return result

    # Properties ---------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        def _shape(data) -> Tuple[int, ...]:
            if isinstance(data, list) and data:
                first = data[0]
                return (len(data),) + _shape(first)
            if isinstance(data, list):
                return (len(data),)
            return ()

        return _shape(self.tolist())

    @property
    def size(self) -> int:
        def _size(data) -> int:
            if isinstance(data, list):
                return sum(_size(elem) for elem in data)
            return 1

        return _size(self.tolist())


# Constructors ----------------------------------------------------------------
def asarray(obj, dtype=None):
    if isinstance(obj, ndarray):
        data = obj.tolist()
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        converted = []
        for item in obj:
            if isinstance(item, (Sequence, ndarray)) and not isinstance(item, (str, bytes, bytearray)):
                converted.append(asarray(item, dtype).tolist())
            else:
                converted.append(_convert_scalar(item, dtype))
        return ndarray(converted)
    else:
        return ndarray([_convert_scalar(obj, dtype)])
    return ndarray(_apply_dtype(data, dtype))


def _convert_scalar(value, dtype):
    if dtype is None:
        if isinstance(value, complex):
            return complex(value)
        return float(value)
    if dtype in (float, int):
        return dtype(value)
    return value


def _apply_dtype(data, dtype):
    if isinstance(data, list):
        return [_apply_dtype(elem, dtype) for elem in data]
    return _convert_scalar(data, dtype)


def array(obj, dtype=None):  # pragma: no cover - convenience wrapper
    return asarray(obj, dtype=dtype)


def zeros(shape, dtype=float):
    if isinstance(shape, int):
        return ndarray([_convert_scalar(0, dtype) for _ in range(shape)])
    if isinstance(shape, tuple) and len(shape) == 1:
        return zeros(shape[0], dtype=dtype)
    if isinstance(shape, tuple) and len(shape) == 0:
        return ndarray([])
    if isinstance(shape, tuple) and len(shape) == 2:
        rows, cols = shape
        return ndarray([[ _convert_scalar(0, dtype) for _ in range(cols)] for _ in range(rows)])
    raise NotImplementedError("Only 1D or 2D zeros are supported")


def empty(shape, dtype=float):
    if isinstance(shape, int):
        return ndarray([] if shape == 0 else [None for _ in range(shape)])
    if isinstance(shape, tuple) and len(shape) == 1:
        return empty(shape[0], dtype=dtype)
    if isinstance(shape, tuple) and len(shape) == 0:
        return ndarray([])
    return zeros(shape, dtype=dtype)


def arange(start, stop=None, step=1.0, dtype=float):
    if stop is None:
        stop = start
        start = 0.0
    values = []
    current = float(start)
    step = float(step)
    if step == 0:
        raise ValueError("step must be non-zero")
    if step > 0:
        while current < float(stop):
            values.append(_convert_scalar(current, dtype))
            current += step
    else:
        while current > float(stop):
            values.append(_convert_scalar(current, dtype))
            current += step
    return ndarray(values)


def geomspace(start, stop, num):
    if num <= 0:
        return ndarray([])
    if start <= 0 or stop <= 0:
        raise ValueError("geomspace requires positive start/stop")
    step = (math.log(stop) - math.log(start)) / (num - 1) if num > 1 else 0.0
    values = [start * math.exp(step * i) for i in range(num)]
    return ndarray(values)


def concatenate(seq: Sequence[Union[ndarray, Sequence, Number]]):
    result: List = []
    for item in seq:
        if isinstance(item, ndarray):
            result.extend(item.tolist())
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            result.extend(list(item))
        else:
            result.append(item)
    return ndarray(result)


# Reduction helpers -----------------------------------------------------------
def mean(values: Union[ndarray, Sequence, Number]) -> float:
    arr = asarray(values)
    if arr.size == 0:
        return 0.0
    data = arr.tolist()
    flat: List[float] = []
    _flatten_numeric(data, flat)
    return sum(flat) / len(flat)


def _flatten_numeric(data, output: List[float]) -> None:
    if isinstance(data, list):
        for item in data:
            _flatten_numeric(item, output)
    else:
        output.append(float(data))


def min(values: Union[ndarray, Sequence, Number]):  # type: ignore[override]
    arr = asarray(values)
    flat: List[float] = []
    _flatten_numeric(arr.tolist(), flat)
    return builtins.min(flat) if flat else float("inf")


def max(values: Union[ndarray, Sequence, Number]):  # type: ignore[override]
    arr = asarray(values)
    flat: List[float] = []
    _flatten_numeric(arr.tolist(), flat)
    return builtins.max(flat) if flat else float("-inf")


def clip(values, lower, upper):
    arr = asarray(values)
    return ndarray([builtins.min(builtins.max(x, lower), upper) for x in arr])


def argmin(values):
    arr = asarray(values)
    best_idx = 0
    best_val = None
    for idx, val in enumerate(arr):
        scalar = val
        if isinstance(scalar, ndarray):
            scalar = scalar.tolist()
        if best_val is None or scalar < best_val:
            best_val = scalar
            best_idx = idx
    return best_idx


def array2string(arr, precision=8):
    array_obj = asarray(arr)
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(float(x)) for x in array_obj) + "]"


def abs(values):  # type: ignore[override]
    if isinstance(values, ndarray):
        return ndarray([abs(x) for x in values])
    return builtins.abs(values)


def angle(values):
    arr = asarray(values)
    return ndarray([math.atan2(val.imag if isinstance(val, complex) else 0.0, val.real if isinstance(val, complex) else float(val)) for val in arr])


def unwrap(values):
    arr = asarray(values)
    if arr.size == 0:
        return arr.copy()
    result: List[float] = []
    data = [float(v) for v in arr]
    prev = data[0]
    offset = 0.0
    result.append(prev)
    for angle_val in data[1:]:
        delta = angle_val - prev
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        result.append(angle_val + offset)
        prev = angle_val
    return ndarray(result)


def log10(values):
    arr = asarray(values)
    return ndarray([math.log10(float(v)) if float(v) > 0 else float("-inf") for v in arr])


def exp(values):
    if isinstance(values, ndarray):
        return ndarray([cmath.exp(v) for v in values])
    return cmath.exp(values) if isinstance(values, complex) else math.exp(values)


# Random number support -------------------------------------------------------
class _Generator:
    def __init__(self, seed: Union[int, None] = None) -> None:
        self._random = _stdlib_random.Random(seed)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Union[int, None] = None):
        if size is None:
            return loc + scale * self._random.gauss(0.0, 1.0)
        return ndarray([loc + scale * self._random.gauss(0.0, 1.0) for _ in range(int(size))])

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Union[int, None] = None):
        if size is None:
            return self._random.uniform(low, high)
        return ndarray([self._random.uniform(low, high) for _ in range(int(size))])

    def choice(self, values: Sequence, size: Union[int, None] = None):
        if size is None:
            return self._random.choice(list(values))
        population = list(values)
        return ndarray([self._random.choice(population) for _ in range(int(size))])


class _RandomModule:
    def default_rng(self, seed: Union[int, None] = None) -> _Generator:
        return _Generator(seed)


random = _RandomModule()


# Constants -------------------------------------------------------------------
pi = math.pi
inf = float("inf")


__all__ = [
    "ndarray",
    "array",
    "asarray",
    "zeros",
    "empty",
    "arange",
    "geomspace",
    "concatenate",
    "mean",
    "min",
    "max",
    "clip",
    "argmin",
    "array2string",
    "abs",
    "angle",
    "unwrap",
    "log10",
    "exp",
    "random",
    "pi",
    "inf",
    "IS_LIGHTWEIGHT",
]


IS_LIGHTWEIGHT = True
