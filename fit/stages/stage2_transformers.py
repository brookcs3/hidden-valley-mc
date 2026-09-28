# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Stage 2: the three transformer paths (both compressor stages out). Per core, bounded least squares on:
  - the small-signal response (log sweep at -30 dBFS): gain, low-frequency corner, high shelf, low pass;
  - the harmonic grid (10 frequencies x 13 levels, -12 .. +24 dBFS): driver-stage a2 and a3, the core's knee flux, knee hardness and
    knee asymmetry at the onset, its decay with the held peak flux, and the saturation pulse. Fundamental gain is weighted 5x, H2/H3 1x, higher harmonics less.
Linear and nonlinear parameters are fitted in two passes, then jointly."""
import os, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import F, ITEMS, MODEL, load_cal, save_cal, render_item, feat_residual, protocol  # noqa: E402

CORES = ["Nickel", "Iron", "Steel"]
LIN = ["x_gain_db", "x_fl_hz", "x_hs_hz", "x_hs_db", "x_lp_hz"]
NL = ["x_a2", "x_a3", "x_sat_db", "x_q", "x_asym", "x_asym_p", "x_pulse"]

def put(cal, k, names, vec):
    for n, v in zip(names, vec):
        cal[MODEL.fields[n][0] + k] = v
    return cal

def get(cal, k, names):
    return [cal[MODEL.fields[n][0] + k] for n in names]

def digital_response(p, freqs, fs=48000.0):
    """closed-form response of the linear path: gain, forward-Euler leaky-integrator high pass (pole 1 - 2 pi fl / fs), TPT high shelf,
    TPT low pass (the same forms the C++ runs)"""
    gain_db, fl, hs_hz, hs_db, lp_hz = p
    w = 2 * np.pi * np.asarray(freqs, dtype=float) / fs
    z1 = np.exp(-1j * w)
    hp = (1 - z1) / (1 - (1 - 2 * np.pi * fl / fs) * z1)
    def tpt_lp(fc):
        g = np.tan(np.pi * fc / fs)
        return g * (1 + z1) / ((1 + g) - (1 - g) * z1)
    sh = 1.0
    if hs_hz > 0:
        lp_ = tpt_lp(hs_hz); sh = lp_ + 10 ** (hs_db / 20) * (1 - lp_)
    lp = tpt_lp(lp_hz) if lp_hz > 0 else 1.0
    return 20 * np.log10(np.abs(10 ** (gain_db / 20) * hp * sh * lp))

def fit_core(cal, k, core):
    resp_id = f"xf_resp_{core}"
    grid = [i for i in ITEMS if i.startswith(f"xf_{core}_f") and int(i.rsplit("_", 1)[1]) < 24]   # +24 dBFS is the reference's output ceiling, not modelled
    # linear pass: closed-form response against the measured sweep response (nonlinearity is negligible at -30 dBFS)
    ref = np.array([v for v in F[resp_id] if v is not None]); freqs = [f for f, v in zip(protocol.RESP_FREQS, F[resp_id]) if v is not None]
    best = None
    for hs0 in (3000.0, 6000.0, 12000.0):
        for lp0 in (30000.0, 60000.0, 120000.0):
            p0 = [ref[10], 1.7, hs0, -0.1, lp0]
            lo = [-1.0, 0.3, 1000.0, -3.0, 15000.0]; hi = [1.0, 12.0, 22000.0, 1.0, 600000.0]
            r = least_squares(lambda p: digital_response(p, freqs) - ref, p0, bounds=(lo, hi), x_scale=[0.1, 0.5, 1000.0, 0.1, 20000.0])
            if best is None or r.cost < best.cost: best = r
    r = best
    cal = put(cal, k, LIN, r.x)
    print(f"  {core} linear: {dict(zip(LIN, np.round(r.x, 4)))}  rms {np.sqrt(np.mean(r.fun**2)):.4f} dB, max {np.max(np.abs(r.fun)):.4f} dB")
    # nonlinear pass: the harmonic grid
    def rnl(p):
        c = put(cal.copy(), k, NL, p)
        return np.concatenate([feat_residual(i, render_item(ITEMS[i], c)) for i in grid])
    p0 = get(cal, k, NL)
    lo = [1e-8, -5e-2, -6.0, 1.5, -0.5, 0.0, -0.02]; hi = [5e-3, 5e-2, 12.0, 30.0, 0.5, 6.0, 0.02]
    p0 = np.clip(p0, lo, hi)
    best = None
    for start in (p0, np.array(list(p0[:6]) + [-p0[6] if abs(p0[6]) > 1e-6 else 1e-3])):   # the pulse's basin is not found from every start
        r_ = least_squares(rnl, np.clip(start, lo, hi), bounds=(lo, hi), x_scale=[1e-5, 1e-4, 0.5, 1.0, 0.01, 0.2, 0.01], diff_step=1e-3, max_nfev=60, loss="soft_l1", f_scale=2.0)
        if best is None or r_.cost < best.cost: best = r_
    r = best
    cal = put(cal, k, NL, r.x)
    res = r.fun
    print(f"  {core} nonlinear: {dict(zip(NL, [float(f'{v:.6g}') for v in r.x]))}  rms {np.sqrt(np.mean(res**2)):.3f} dB (weighted)")
    return cal

def run():
    cal = load_cal()
    for k, core in enumerate(CORES):
        cal = fit_core(cal, k, core)
    save_cal(cal, {"stage2": "transformer paths: response (xf_resp_*) then harmonic grid (xf_*_f*)"}, fields=LIN + NL)

if __name__ == "__main__":
    run()
