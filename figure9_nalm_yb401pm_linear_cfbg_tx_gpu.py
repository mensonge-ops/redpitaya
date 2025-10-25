# figure9_nalm_yb401pm_linear_cfbg_tx_gpu.py
# -----------------------------------------------------------------------------
# Figure‑9 NALM fiber laser (explicit linear arm) with CFBG at the arm end.
# * Linear arm is propagated out-and-back explicitly.
# * CFBG applies spectral amplitude + quadratic phase at the arm endpoint.
# * The **laser output** is taken from the **CFBG transmission** port.
# * Yb401‑PM gain fiber (0.6 m) with small‑signal gain (gssdB), energy saturation.
# * Auto GPU (CuPy) fallback to CPU (NumPy) without code changes.
# * Robust but lightweight diagnostics (spectral FWHM, evolution plots).
# -----------------------------------------------------------------------------
from __future__ import annotations
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib
for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(backend)
        break
    except Exception:
        pass
# ============================== Backend (GPU/CPU) ==============================
USE_GPU = True  # try GPU; falls back to CPU automatically
try:
    import cupy as cp
    xp = cp
    xpf = cp.fft
    BACKEND = "GPU (CuPy)"
except Exception:
    import numpy as np
    xp = np
    xpf = np.fft
    BACKEND = "CPU (NumPy)"

C_MPS   = 299_792_458.0
C_NM_PS = 299_792.458

# ------------------------------- Helpers -------------------------------------
def to_cpu(a):
    try:
        import cupy as _cp
        if isinstance(a, _cp.ndarray):
            return _cp.asnumpy(a)
    except Exception:
        pass
    return a

def fftshift_fft(u):
    return xpf.fftshift(xpf.fft(u))

def ifft_ifftshift(U):
    return xpf.ifft(xpf.ifftshift(U))

def rand_sech(nt, Tps, noise_amp=0.25, seed=42):
    import numpy as _np
    rng = _np.random.default_rng(seed)
    t = _np.linspace(-Tps/2, -Tps/2 + Tps*(nt-1)/nt, nt)
    tau = Tps/10
    u = 1/_np.cosh(t/tau)
    if noise_amp > 0:
        amp = 1 + noise_amp*(rng.random(nt) - 0.5)
        phs = _np.exp(1j*2*_np.pi*rng.random(nt)*noise_amp)
        u = u * amp * phs
    u = u / _np.max(_np.abs(u))
    return xp.asarray(u, dtype=xp.complex128), xp.asarray(t, dtype=float)

def coupler(u1, u2, rho):
    k = math.sqrt(max(rho, 0.0))
    t = math.sqrt(max(1.0 - rho, 0.0))
    return t*u1 + 1j*k*u2, 1j*k*u1 + t*u2

# ------------------------------- Metrics -------------------------------------
def fwhm_nm_on_frequency_axis(y, nu_THz, lambda0_nm, c_nm_ps=C_NM_PS):
    y = y / (float(y.max()) + 1e-300)
    th = 0.5
    idx = xp.where(y >= th)[0]
    if idx.size == 0:
        return 0.0
    iL2 = int(idx[0]); iR1 = int(idx[-1])
    # left
    if iL2 > 0:
        y1, y2 = float(y[iL2-1]), float(y[iL2]); den = y2 - y1
        wL = 0.0 if abs(den) < 1e-12 else (th - y1)/den
        wL = min(1.0, max(0.0, wL))
        nuL = float(nu_THz[iL2-1]) + wL*(float(nu_THz[iL2]) - float(nu_THz[iL2-1]))
    else:
        nuL = float(nu_THz[iL2])
    # right
    if iR1 < y.size-1:
        y1, y2 = float(y[iR1]), float(y[iR1+1]); den = y2 - y1
        wR = 0.0 if abs(den) < 1e-12 else (th - y1)/den
        wR = min(1.0, max(0.0, wR))
        nuR = float(nu_THz[iR1]) + wR*(float(nu_THz[iR1+1]) - float(nu_THz[iR1]))
    else:
        nuR = float(nu_THz[iR1])
    dnu = max(0.0, nuR - nuL)
    return (lambda0_nm**2 / c_nm_ps) * dnu

# ============================ Fiber model (RK4IP) =============================
@dataclass
class Fiber:
    L: float                 # km
    gamma: float             # W^-1 km^-1
    alpha: float             # km^-1 (power loss)
    betaw: object            # xp.array([0,0,beta2, beta3]) in ps^n/km
    gssdB: float | None = None
    Esat_pJ: float | None = None
    fbw_THz: float | None = None
    fc_THz: float | None = None


def IP_CQEM_FD(u_in, dt_ps, dz_km, fiber: Fiber, f_THz, fo_THz, is_gain: bool):
    """ RK4IP: dispersion (β2/β3), Kerr, loss, distributed small-signal gain + energy saturation. """
    u = u_in.astype(xp.complex128).copy()
    nt = u.shape[0]
    Nz = max(1, int(math.ceil(fiber.L / dz_km)))
    dz_eff = fiber.L / Nz

    Omega = 2*math.pi*f_THz
    betaw = fiber.betaw
    if betaw.shape[0] < 3:
        pad = xp.zeros(3-betaw.shape[0]); betaw = xp.concatenate([betaw, pad])

    Dlin = -0.5*fiber.alpha + 0j
    Dlin = Dlin * xp.ones(nt, dtype=xp.complex128)
    for n in range(2, betaw.shape[0]):
        bn = float(betaw[n])
        if bn != 0.0:
            Dlin += 1j * (bn / math.factorial(n)) * (Omega**n)

    # gain bandwidth (small-signal spectral shape)
    Gf_half = 1.0
    if is_gain and (fiber.fbw_THz is not None) and (fiber.fc_THz is not None) and fiber.L > 0:
        sigma = fiber.fbw_THz / (2*math.sqrt(2*math.log(2)))
        GainFilter = xp.exp(-0.5 * ((f_THz + fo_THz - fiber.fc_THz)/max(sigma,1e-30))**2)
        Gf_half = GainFilter**(dz_eff/(2*fiber.L))

    # distributed small-signal power gain exponent
    g_per_km = 0.0
    if is_gain and (fiber.gssdB is not None) and (fiber.L > 0):
        G_lin = 10**(fiber.gssdB/10.0)
        g_per_km = math.log(G_lin) / fiber.L

    ExpD_half = xp.exp(Dlin * dz_eff/2.0)
    gamma = fiber.gamma
    def NL(x):
        return 1j * gamma * (xp.abs(x)**2) * x

    Esat_J = fiber.Esat_pJ*1e-12 if (is_gain and fiber.Esat_pJ is not None) else None

    for _ in range(Nz):
        # energy saturation (per step)
        sat = 1.0
        if Esat_J is not None:
            Epulse = float(xp.sum(xp.abs(u)**2)) * dt_ps * 1e-12
            sat = 1.0 / (1.0 + Epulse/max(Esat_J, 1e-30))

        A_gain = 1.0
        if is_gain and (g_per_km != 0.0):
            A_gain = math.exp(0.5 * g_per_km * sat * dz_eff)

        U = fftshift_fft(u) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k1 = dz_eff * NL(u_lin)

        U = fftshift_fft(u + 0.5*k1) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k2 = dz_eff * NL(u_lin)

        U = fftshift_fft(u + 0.5*k2) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k3 = dz_eff * NL(u_lin)

        U = fftshift_fft(u + k3) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u_lin = ifft_ifftshift(U)
        k4 = dz_eff * NL(u_lin)

        u = u + (k1 + 2*k2 + 2*k3 + k4)/6.0
        U = fftshift_fft(u) * ExpD_half
        if isinstance(Gf_half, xp.ndarray):
            U *= Gf_half
        u = ifft_ifftshift(U)

        u *= A_gain

    return u, {}

# ========================== CFBG (endpoint node) ==============================
def cfbg_node(E_in, f_THz, fo_THz, Rmax=0.20, FWHM_nm=16.0, lam0_nm=1030.0,
              phi2_reflect_ps2=0.0, phi2_trans_ps2=0.0,
              IL_tx_dB=0.0, IL_rx_dB=0.0):
    """
    Endpoint CFBG: returns (E_reflected_back, E_transmitted_output)
    R(λ) = Rmax * exp(-0.5*((λ-λ0)/σ)^2);  A_r = sqrt(R), A_t = sqrt(1-R)
    Reflect arm applies exp(+j*0.5*φ2_reflect*Ω^2); transmit arm can optionally have φ2.
    """
    # wavelength grid per bin
    nu_abs_THz = f_THz + fo_THz
    lam_nm = C_NM_PS / xp.maximum(nu_abs_THz, 1e-30)
    sigma_nm = FWHM_nm/(2*math.sqrt(2*math.log(2)))
    R_lambda = Rmax * xp.exp(-0.5 * ((lam_nm - lam0_nm)/max(sigma_nm,1e-30))**2)
    R_lambda = xp.clip(R_lambda, 0.0, 1.0)
    A_r = xp.sqrt(R_lambda) * 10**(-IL_rx_dB/20.0)
    A_t = xp.sqrt(1.0 - R_lambda) * 10**(-IL_tx_dB/20.0)

    Omega = 2*math.pi*f_THz
    Hphi_r = xp.exp(1j * 0.5 * phi2_reflect_ps2 * (Omega**2)) if phi2_reflect_ps2 != 0.0 else 1.0
    Hphi_t = xp.exp(1j * 0.5 * phi2_trans_ps2   * (Omega**2)) if phi2_trans_ps2   != 0.0 else 1.0

    U = fftshift_fft(E_in)
    U_ref = U * (A_r * Hphi_r)
    U_tx  = U * (A_t * Hphi_t)

    E_reflected = ifft_ifftshift(U_ref)
    E_transmit  = ifft_ifftshift(U_tx)

    # π phase upon reflection from grating (optional, set True if needed)
    E_reflected = -E_reflected
    return E_reflected, E_transmit

# =============================== Main sim ====================================
def main():
    print(f"Backend: {BACKEND}")

    # ---- design goals ----
    TARGET_FWHM_NM = 16.0
    TARGET_FREP_MHZ = 40.0
    lambda0_nm = 1030.0
    fo_THz = C_NM_PS / lambda0_nm

    nt   = 2**12
    Tps  = 120.0
    dt   = Tps/nt
    f_THz = xp.arange(-nt/2, nt/2, dtype=float) * (1.0/Tps)
    nu_THz = f_THz + fo_THz
    lamb_nm = C_NM_PS / nu_THz

    # ---- seed ----
    u, t_ps = rand_sech(nt, Tps, noise_amp=0.2, seed=42)
    P0_init = 55.0
    u *= math.sqrt(P0_init)

    # ---- Yb401‑PM / passive fiber params ----
    MFD_um = 6.0
    Aeff_um2 = math.pi*(MFD_um/2.0)**2
    n2 = 2.6e-20
    gamma = 2*math.pi*(n2)/(lambda0_nm*1e-9)/(Aeff_um2*1e-12) * 1e3   # W^-1 km^-1
    beta2_ps2 = 20.2
    beta3_ps3 = 36.8e-3
    betaw = xp.asarray([0.0, 0.0, beta2_ps2, beta3_ps3])

    # ---- geometry scaling (km) ----
    n_g = 1.45
    L_target_m = C_MPS / (n_g * (TARGET_FREP_MHZ*1e6))
    L_target_km = L_target_m / 1e3
    base_total_m = 9.0
    scale = L_target_m / base_total_m

    base_lengths = {
        "smf1": 0.00029,
        "amf":  0.00060,
        "smf2": 0.00120,
        "smf3": 0.00500 - 0.00029 - 0.00060 - 0.00120,
        "L_lin1": 0.00100,
        "L_lin2": 0.00100,
    }

    # apply scaling so that the geometry meets the 40 MHz repetition rate
    smf1 = Fiber(L=base_lengths["smf1"]*scale, gamma=gamma, alpha=0.0, betaw=betaw)
    amf  = Fiber(L=base_lengths["amf"]*scale,  gamma=gamma, alpha=0.0, betaw=betaw,
                 gssdB=30.0, Esat_pJ=11.0, fbw_THz=C_NM_PS/(lambda0_nm**2)*40.0, fc_THz=C_NM_PS/lambda0_nm)
    smf2 = Fiber(L=base_lengths["smf2"]*scale, gamma=gamma, alpha=0.0, betaw=betaw)
    smf3 = Fiber(L=base_lengths["smf3"]*scale, gamma=gamma, alpha=0.0, betaw=betaw)

    L_lin1 = base_lengths["L_lin1"]*scale
    L_lin2 = base_lengths["L_lin2"]*scale
    smf_lin1 = Fiber(L=L_lin1, gamma=gamma, alpha=0.0, betaw=betaw)
    smf_lin2 = Fiber(L=L_lin2, gamma=gamma, alpha=0.0, betaw=betaw)

    # Couplers & NRPS
    rho_nalm = 0.50
    nrps_phi = math.pi/2
    rho_out  = 0.12

    # Repetition estimate (linear-cavity formula)
    L_cavity_km = (smf1.L + amf.L + smf2.L + smf3.L) + 2*(L_lin1 + L_lin2)
    f_rep = C_MPS / (n_g*(L_cavity_km*1e3))
    print(f"Estimated f_rep ≈ {f_rep/1e6:.2f} MHz (cavity length ~ {L_cavity_km*1e3:.3f} m)")

    # ---- Set CFBG dispersion to cancel net fiber D ----
    def D_total_fiber_ps_per_nm(fibers, lam_nm):
        factor = -(2*math.pi*C_NM_PS)/(lam_nm**2)
        Dsum = 0.0
        for fb in fibers:
            b2 = float(fb.betaw[2]) if fb.betaw.shape[0] >= 3 else 0.0
            Dsum += (factor * b2) * fb.L
        return Dsum
    def phi2_from_Dtotal_ps2(D_ps_per_nm, lam_nm):
        return float(- (lam_nm**2 / (2*math.pi*C_NM_PS)) * D_ps_per_nm)

    fibers_for_D = [smf1, amf, smf2, smf3, smf_lin1, smf_lin2]
    D_fiber = D_total_fiber_ps_per_nm(fibers_for_D, lambda0_nm)
    D_cfbg  = -D_fiber
    phi2_ref = phi2_from_Dtotal_ps2(D_cfbg, lambda0_nm)
    phi2_tx  = 0.0
    print(f"Fiber D = {D_fiber:+.3f} ps/nm, set CFBG φ2_ref = {phi2_ref:+.4f} ps^2 (net≈0)")

    # ---- loop settings ----
    dz = 0.5e-6
    N_round = 260

    spec_out_evo = xp.zeros((N_round, nt))
    time_out_evo = xp.zeros((N_round, nt))
    fwhm_evo_nm  = xp.zeros(N_round)

    E = u.copy()

    for rr in range(N_round):
        Ein2 = xp.zeros_like(E)
        cw, ccw = coupler(E, Ein2, rho_nalm)

        cw, _ = IP_CQEM_FD(cw, dt, dz, smf1, f_THz, fo_THz, is_gain=False)
        cw, _ = IP_CQEM_FD(cw, dt, dz, amf,  f_THz, fo_THz, is_gain=True)
        cw, _ = IP_CQEM_FD(cw, dt, dz, smf2, f_THz, fo_THz, is_gain=False)
        cw = cw * xp.exp(0.5j*nrps_phi)
        cw, _ = IP_CQEM_FD(cw, dt, dz, smf3, f_THz, fo_THz, is_gain=False)

        ccw, _ = IP_CQEM_FD(ccw, dt, dz, smf3, f_THz, fo_THz, is_gain=False)
        ccw = ccw * xp.exp(-0.5j*nrps_phi)
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, smf2, f_THz, fo_THz, is_gain=False)
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, amf,  f_THz, fo_THz, is_gain=True)
        ccw, _ = IP_CQEM_FD(ccw, dt, dz, smf1, f_THz, fo_THz, is_gain=False)

        E_to_linear, _dump = coupler(cw, ccw, rho_nalm)

        E_lin, _ = IP_CQEM_FD(E_to_linear, dt, dz, smf_lin1, f_THz, fo_THz, is_gain=False)
        E_lin, _ = IP_CQEM_FD(E_lin,       dt, dz, smf_lin2, f_THz, fo_THz, is_gain=False)

        E_reflect, E_tx = cfbg_node(E_lin, f_THz, fo_THz,
                                    Rmax=0.18, FWHM_nm=TARGET_FWHM_NM, lam0_nm=lambda0_nm,
                                    phi2_reflect_ps2=phi2_ref, phi2_trans_ps2=phi2_tx,
                                    IL_tx_dB=0.1, IL_rx_dB=0.0)

        time_out_evo[rr, :] = xp.abs(E_tx)**2
        SPEC = fftshift_fft(E_tx)
        SPEC = (xp.abs(SPEC)**2)
        if float(SPEC.max()) > 0:
            SPEC = SPEC / float(SPEC.max())
        spec_out_evo[rr, :] = SPEC
        fwhm_evo_nm[rr] = fwhm_nm_on_frequency_axis(SPEC, nu_THz, lambda0_nm, C_NM_PS)

        E_back, _ = IP_CQEM_FD(E_reflect, dt, dz, smf_lin2, f_THz, fo_THz, is_gain=False)
        E, _      = IP_CQEM_FD(E_back,    dt, dz, smf_lin1, f_THz, fo_THz, is_gain=False)

        if (rr % 10) == 0:
            peak_power = float(xp.max(time_out_evo[rr, :]))
            print(f"Round {rr+1}: FWHM ≈ {float(fwhm_evo_nm[rr]):.2f} nm, Peak ≈ {peak_power:.1f} W")

    rr = N_round - 1
    t_np = to_cpu(t_ps); lamb_np = to_cpu(lamb_nm)
    It_np = to_cpu(time_out_evo[rr, :])
    Sp_np = to_cpu(spec_out_evo[rr, :])

    plt.figure(1); plt.clf()
    plt.plot(t_np, It_np); plt.grid(True)
    plt.xlabel('Time (ps)'); plt.ylabel('Power (W)')
    plt.title('Output (CFBG transmission) – time domain')

    plt.figure(2); plt.clf()
    plt.plot(lamb_np, Sp_np); plt.grid(True)
    plt.xlabel('Wavelength (nm)'); plt.ylabel('Normalized spectral power')
    plt.title('Output (CFBG transmission) – spectrum')

    plt.figure(3); plt.clf()
    import numpy as _np
    plt.subplot(2,1,1)
    plt.imshow(to_cpu(time_out_evo), aspect='auto', origin='lower',
               extent=[t_np[0], t_np[-1], 1, N_round])
    plt.colorbar(); plt.xlabel('Time (ps)'); plt.ylabel('Round trip')
    plt.title('Temporal evolution (output)')
    plt.subplot(2,1,2)
    plt.imshow(to_cpu(spec_out_evo), aspect='auto', origin='lower',
               extent=[lamb_np[0], lamb_np[-1], 1, N_round])
    plt.colorbar(); plt.xlabel('Wavelength (nm)'); plt.ylabel('Round trip')
    plt.title('Spectral evolution (output)')
    plt.tight_layout()

    print(f"\nFinal output spectral FWHM (nm) ≈ {float(fwhm_evo_nm[-1]):.2f}")
    if abs(float(fwhm_evo_nm[-1]) - TARGET_FWHM_NM) <= 0.8:
        print("Target 16 nm mode-locking bandwidth achieved.")
    else:
        print("Warning: final FWHM deviates from 16 nm target – consider tuning gain or filter.")
    plt.show()

if __name__ == "__main__":
    main()
