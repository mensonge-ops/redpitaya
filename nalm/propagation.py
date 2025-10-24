"""Propagation routines for the generalized nonlinear Schrödinger equation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

from .physics import (
    Linearoperator_w,
    NonLinearoperator_w,
    Raman_response_w,
    filter_lorentz_tf,
    gain_saturated2,
)

__all__ = ["PropagationRecord", "IP_CQEM_FD"]


@dataclass
class PropagationRecord:
    """Store intermediate data along the propagation coordinate."""

    z: np.ndarray
    spectra: np.ndarray
    fields: np.ndarray


def _as_namespace(segment) -> SimpleNamespace:
    if isinstance(segment, SimpleNamespace):
        return SimpleNamespace(**segment.__dict__)
    if hasattr(segment, "to_namespace"):
        return segment.to_namespace()
    if isinstance(segment, dict):
        return SimpleNamespace(**segment)
    raise TypeError("Unsupported segment description")


def IP_CQEM_FD(
    u0: np.ndarray,
    dt: float,
    dz: float,
    segment,
    fo: float,
    tol: float,
    record: bool = False,
    quiet: bool = True,
) -> Tuple[np.ndarray, int, Optional[PropagationRecord]]:
    """Solve the GNLSE using the interaction picture and CQE method."""

    u0 = np.asarray(u0, dtype=np.complex128)
    nt = u0.size
    w = np.fft.fftshift(2.0 * np.pi * np.arange(-nt // 2, nt // 2) / (dt * nt))
    t = np.arange(-nt // 2, nt // 2) * dt

    mod = _as_namespace(segment)
    hrw, fr = Raman_response_w(t, mod)

    ufft = np.fft.fft(u0)
    propagated = 0.0
    u1 = u0.copy()
    nf = 1

    gain_tf = None
    alpha_base = mod.alpha
    if getattr(mod, "gssdB", None) is not None:
        gain_tf = filter_lorentz_tf(u1, mod.fbw, mod.fc, fo, 1.0 / (dt * nt))

    z_samples = []
    spectra = []
    fields = []

    while propagated < mod.L:
        step = dz
        if propagated + step > mod.L:
            step = mod.L - propagated

        if gain_tf is not None:
            Pin0 = np.sum(np.abs(u1) ** 2) / nt
            gain = gain_saturated2(Pin0, mod.gssdB, mod.PsatdBm) * gain_tf
            alpha_effective = alpha_base - gain
        else:
            alpha_effective = mod.alpha

        LOP = Linearoperator_w(alpha_effective, mod.betaw, w)

        denom = w + 2.0 * np.pi * fo
        PhotonN = np.sum(np.abs(ufft) ** 2 / denom)
        PhotonN_z = np.sum(
            np.exp(-step * np.fft.fftshift(alpha_effective)) * np.abs(ufft) ** 2 / denom
        )

        halfstep = np.exp(LOP * step / 2.0)
        uip = halfstep * ufft
        k1 = halfstep * step * NonLinearoperator_w(u1, mod.gamma, w, fo, fr, hrw, dt, mod)
        uhalf2 = np.fft.ifft(uip + k1 / 2.0)
        k2 = step * NonLinearoperator_w(uhalf2, mod.gamma, w, fo, fr, hrw, dt, mod)
        uhalf3 = np.fft.ifft(uip + k2 / 2.0)
        k3 = step * NonLinearoperator_w(uhalf3, mod.gamma, w, fo, fr, hrw, dt, mod)
        uhalf4 = np.fft.ifft(halfstep * (uip + k3))
        k4 = step * NonLinearoperator_w(uhalf4, mod.gamma, w, fo, fr, hrw, dt, mod)

        uaux = halfstep * (uip + k1 / 6.0 + k2 / 3.0 + k3 / 3.0) + k4 / 6.0
        propagated += step

        error = abs(np.sum(np.abs(uaux) ** 2 / denom) - PhotonN_z) / PhotonN_z
        if error > 2.0 * tol:
            propagated -= step
            dz = step / 2.0
            continue

        ufft = uaux
        u1 = np.fft.ifft(ufft)

        if error > tol:
            dz = step / (2.0 ** 0.2)
        elif error < 0.5 * tol:
            dz = step * (2.0 ** 0.2)

        if record:
            z_samples.append(propagated)
            spectra.append(np.fft.fftshift(np.abs(ufft)))
            fields.append(u1.copy())

        nf += 16

        if not quiet:
            print(f"\rPropagation {propagated / mod.L * 100:.2f}%", end="")

    if record:
        plotdata = PropagationRecord(
            z=np.asarray(z_samples),
            spectra=np.asarray(spectra),
            fields=np.asarray(fields),
        )
    else:
        plotdata = None

    return u1, nf, plotdata
