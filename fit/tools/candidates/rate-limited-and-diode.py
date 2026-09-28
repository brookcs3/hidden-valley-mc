# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Detector candidates, family "rate-limited-and-diode": nonlinear charge/discharge paths (diode + resistor detectors) on top of the
log rectifier -> storage node -> gain computer topology of the original harness (fit/tools/detector_candidates.py, modes 0-4 kept).
New modes (all: e = 20 log10 |x| clamped at FLOOR, v = node in dB, gr = C_r(v - T_k); release reference R = T_k - depth):
  10  saturating attack: dv = min(e - v, K) * kA when e > v; release one-pole toward R                               extras: depth, K
  11  linear release: dv = -(30 dB / tr) / fs per sample, stops at R; attack one-pole toward e                      extras: depth
  12  two release paths: toward R with kR and toward the instantaneous e with a fixed constant tau2                  extras: depth, tau2
  13  attack toward e + k_off (offset), release toward R                                                            extras: depth, k_off
  14  diode drop: attack (toward e) only when e > v + delta; release toward R                                        extras: depth, delta
  15  hybrid release: one-pole toward R plus a constant current sink (slope s dB/s), stops at R                       extras: depth, s
  16  soft saturating attack: dv = K tanh((e - v)/K) * kA; release toward R                                          extras: depth, K
  21  parallel RC (diode + always-on bleed): dv = kA (e - v)+ - kR (v - R); the discharge does not switch off while charging  extras: depth
  22  21 with a saturating attack drive (clamp K)                                                                     extras: depth, K
  23  21 with a diode knee: drive = s softplus((e - v - Vd)/s), i.e. an exponential diode with series resistor         extras: depth, s, Vd
  24  21 with attack toward e + k_off                                                                                 extras: depth, k_off
  25  21 with a second release path toward the instantaneous e (tau2), always on when e < v                           extras: depth, tau2
  26  21 with a constant current sink added to the bleed (slope s dB/s), stops at R                                   extras: depth, s
  27  21 with a saturating bleed: pull-down = kR D0 (1 - exp(-(v - R)/D0)) (diode/transistor-limited discharge)      extras: depth, D0
  28  21 with a power-law bleed: pull-down = kR D0 ((v - R)/D0)^alpha                                                extras: depth, alpha (D0 = 20)
  29  27 with attack toward e + k_off                                                                                 extras: depth, D0, k_off
  30  parallel RC with the attack in the LINEAR domain (V += kA (|x| - V)+), bleed in dB toward R                    extras: depth
  32  parallel RC with an exponential (diode) attack drive: dv = kA s (10^((e - v)/s) - 1)+, s in dB (s=8.686: linear domain; s->inf: dB) extras: depth, s
  33  parallel RC on e = 20 log10(|x| - g X_T), X_T = 10^(T_k/20): threshold as a bias current subtracted from the rectified signal
      before the log converter; the static curve is remapped so the 1 ms calibration is unchanged                     extras: depth, g
  34  33 with attack toward e + k_off                                                                                 extras: depth, g, k_off
  35  parallel RC whose bleed returns to Rf = (1 - b) R + b max(e, R): the release resistor hangs across the attack diode
      (returns to the thresholded log level) with a second path to the reference; below threshold identical to 21     extras: depth, b
  36  35 with attack toward e + k_off                                                                                 extras: depth, b, k_off
Findings (see the structured report of the run that produced this file):
  * every mode that keeps E's branching (release only while e <= v) collapses back onto E (0.29 dB steady rms): the fitter switches
    the added mechanism off (K -> bound, k_off -> 0, tau2 -> bound, sink -> 0); the linear release (11) and the hard diode gate (14)
    are worse.
  * mode 21, the parallel RC (the bleed resistor is always connected; in E it was switched off while the diode conducts, which
    under-counts the discharge by 1/(1 - duty) ~ 4x at 30 ms attack) is the winner: 0.179 dB steady rms (max 0.71), bursts 0.0126
    weighted (max 0.77) on the stage-3 curves, 0.141 / 0.0103 after the curves are re-derived for it (--recal). It also reproduces
    the reference's attack-edge time constant shortening with faster recover (1/(kA + kR)) and the odd-harmonic levels (fact 6).
  * the remaining structure is a level tilt at slow attack (-0.6 dB at -25 dBFS, +0.5 dB at +2 dBFS, 30 ms): the steady state alone
    wants the bleed to grow with level only half as fast (equivalent depth ~ 21 dB), the burst tails and the bdepth series pin a
    one-pole toward T_k - 0.5 dB. Sub-linear bleeds (27, 28, 26), a thresholded return (35), a linear-domain threshold bias (33),
    exponential / linear-domain attack drives (32, 30) and the diode knee (23) are all rejected by the data.
usage: python3 fit/tools/candidates/rate-limited-and-diode.py [--modes 10,11,...] [--verbose] [--nfev N] [--recal] [--profile]
       [--ex0 depth,extra,...] [--fixdepth D] [--steady-only]   (default: every mode of the family, ~40 min)"""
import os, sys, time, numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, protocol, FS  # noqa: E402

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
XG = -20.0 + np.arange(96)
cal = load_cal()
GAIN = cal[MODEL.field("d_gain_db")]; XGN = float(cal[MODEL.fields["x_gain_db"][0]]); GAIN12 = float(GAIN[11] + XGN)
CURVES = cal[MODEL.field("d_curve")].reshape(6, 96).copy(); T = cal[MODEL.field("d_thr_db")].copy()
FLOOR = -100.0
LIN_DB = 30.0   # mode 11/15: the recover constant is the time to fall LIN_DB dB


@njit(cache=True)
def curve_lin(c, x):
    if x <= XG[0]: return c[0]
    if x >= XG[-1]: return c[-1] + (c[-1] - c[-2]) * (x - XG[-1])
    i = int(x - XG[0]); t = x - XG[0] - i
    return c[i] * (1 - t) + c[i + 1] * t


@njit(cache=True)
def run_detector(a, fs, ta, tr, mode, c, Tk, ex):
    """a: |sidechain|. ex: extras vector (ex[0] = depth for every mode >= 4). Returns the gain reduction (dB) per sample."""
    n = a.shape[0]
    gr = np.empty(n)
    kA = 1.0 - np.exp(-1.0 / (ta * fs)); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    fl = 10.0 ** (FLOOR / 20.0)
    depth = ex[0]
    R = Tk - depth
    v = FLOOR; g = 1.0; grs = 0.0; env = 0.0
    K = ex[1] if ex.shape[0] > 1 else 1e9
    step = LIN_DB / tr / fs                       # mode 11: dB per sample
    k2 = 1.0 - np.exp(-1.0 / (ex[1] * fs)) if (mode == 12 and ex.shape[0] > 1) else 0.0
    if mode == 25: k2 = 1.0 - np.exp(-1.0 / (ex[1] * fs))
    slope = ex[1] / fs if (mode == 15 or mode == 26) else 0.0
    Vd = ex[2] if ex.shape[0] > 2 else 0.0
    XT = K * 10.0 ** (Tk / 20.0) if (mode == 33 or mode == 34) else 0.0
    for i in range(n):
        if mode == 0:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += (e - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 1:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            gt = 10.0 ** (-max(curve_lin(c, e - Tk), 0.0) / 20.0)
            if gt < g: g += (gt - g) * kA
            else: g += (gt - g) * kR
            gr[i] = -20.0 * np.log10(g)
        elif mode == 2:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            gt = max(curve_lin(c, e - Tk), 0.0)
            if gt > grs: grs += (gt - grs) * kA
            else: grs += (gt - grs) * kR
            gr[i] = grs
        elif mode == 4:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 10:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += min(e - v, K) * kA
            else: v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 11:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            elif v > R: v = max(v - step, R)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 12:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += (R - v) * kR + (e - v) * k2
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 13:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            et = e + K
            if et > v: v += (et - v) * kA
            else: v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 14:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v + K: v += (e - v) * kA
            elif e <= v: v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 15:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            elif v > R: v = max(v + (R - v) * kR - slope, R)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 16:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += K * np.tanh((e - v) / K) * kA
            else: v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 21:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e - v
            v += (d * kA if d > 0.0 else 0.0) + (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 22:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e - v
            v += (min(d, K) * kA if d > 0.0 else 0.0) + (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 23:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            z = (e - v - Vd) / K
            drive = K * (z if z > 30.0 else np.log1p(np.exp(z)))
            v += drive * kA + (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 24:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e + K - v
            v += (d * kA if d > 0.0 else 0.0) + (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 25:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e - v
            v += (d * kA if d > 0.0 else d * k2) + (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 26:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e - v
            v += (d * kA if d > 0.0 else 0.0) + (R - v) * kR
            if v > R: v = max(v - slope, R)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 27 or mode == 29:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e + Vd - v
            u = v - R
            pull = K * (1.0 - np.exp(-u / K)) if u > 0.0 else u
            v += (d * kA if d > 0.0 else 0.0) - pull * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 30:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v:
                V = 10.0 ** (v / 20.0); V += (a[i] - V) * kA; v = 20.0 * np.log10(V)
            v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 32:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e - v
            if d > 0.0: v += K * (10.0 ** (d / K) - 1.0) * kA
            v += (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 33 or mode == 34:
            y = a[i] - XT
            e = 20.0 * np.log10(y) if y > fl else FLOOR
            d = e + Vd - v
            v += (d * kA if d > 0.0 else 0.0) + (R - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 35 or mode == 36:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e + Vd - v
            Rf = R + K * (e - R) if e > R else R
            v += (d * kA if d > 0.0 else 0.0) + (Rf - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 28:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            d = e - v
            u = v - R
            pull = 20.0 * (u / 20.0) ** K if u > 0.0 else u
            v += (d * kA if d > 0.0 else 0.0) - pull * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        else:
            if a[i] > env: env += (a[i] - env) * kA
            else: env += (a[i] - env) * kR
            e = 20.0 * np.log10(env) if env > fl else FLOOR
            gr[i] = max(curve_lin(c, e - Tk), 0.0)
    return gr


CUR = [CURVES]


def remap_curves(g):
    """mode 33/34: the detector sees u = x + 20 log10(1 - g 10^(-x/20)) at the peak of a sine x dB above threshold (1 ms attack);
    the static family GR = C(x) must be unchanged, so the gain computer uses C_g(u) = C(x(u))."""
    if g <= 0.0: return CURVES
    xs = np.linspace(20.0 * np.log10(g) + 1e-3, 90.0, 40001)
    us = xs + 20.0 * np.log10(1.0 - g * 10.0 ** (-xs / 20.0))
    out = np.empty_like(CURVES)
    for r in range(6):
        cx = np.interp(xs, XG, CURVES[r]); cx = np.where(xs > XG[-1], CURVES[r][-1] + (CURVES[r][-1] - CURVES[r][-2]) * (xs - XG[-1]), cx)
        out[r] = np.interp(XG, us, cx, left=0.0)
    return np.ascontiguousarray(out)


def env_dB(item_id, ta, tr, mode, ex):
    it = ITEMS[item_id]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    x = protocol.stimulus(st, fs)[0]
    r = RATIOS.index(s.get("discrete_ratio", "4:1")); Tk = T[int(s["discrete_threshold"]) - 1]
    gr = run_detector(np.abs(x), float(fs), ta, tr, mode, CUR[0][r], Tk, ex)
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def steady(ta, tr, mode, ex, level=-10.0, f=1000.0, thr=16, ratio="4:1", secs=4.0):
    t = np.arange(int(secs * FS)) / FS
    x = np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))
    gr = run_detector(x, float(FS), ta, tr, mode, CUR[0][RATIOS.index(ratio)], T[thr - 1], ex)
    per = int(round(FS / f)); seg = gr[-per * 50:]
    return 20 * np.log10(np.mean(10 ** (-seg / 20.0))) + GAIN12


BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS[:5]] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
N_SS = 36 + 12 + 6


# extras layout per mode: list of (name, init, lo, hi, scale). ex[0] is always depth (dB).
DEPTH = ("depth_dB", 0.0, -40.0, 80.0, 2.0)
EXTRAS = {
    4: [DEPTH],
    10: [DEPTH, ("K_dB", 12.0, 0.3, 80.0, 2.0)],
    11: [DEPTH],
    12: [DEPTH, ("tau2_s", 0.5, 0.005, 20.0, 0.1)],
    13: [DEPTH, ("k_off_dB", 0.0, -20.0, 20.0, 1.0)],
    14: [DEPTH, ("delta_dB", 1.0, 0.0, 20.0, 1.0)],
    15: [DEPTH, ("slope_dB_s", 50.0, 0.0, 2000.0, 20.0)],
    16: [DEPTH, ("K_dB", 12.0, 0.3, 80.0, 2.0)],
    21: [DEPTH],
    22: [DEPTH, ("K_dB", 12.0, 0.3, 80.0, 2.0)],
    23: [DEPTH, ("knee_s_dB", 3.0, 0.05, 30.0, 0.5), ("Vd_dB", 0.0, -20.0, 20.0, 1.0)],
    24: [DEPTH, ("k_off_dB", 0.0, -20.0, 20.0, 1.0)],
    25: [DEPTH, ("tau2_s", 0.5, 0.005, 20.0, 0.1)],
    26: [DEPTH, ("slope_dB_s", 50.0, 0.0, 2000.0, 20.0)],
    27: [DEPTH, ("D0_dB", 30.0, 1.0, 400.0, 5.0)],
    28: [DEPTH, ("alpha", 0.8, 0.2, 1.5, 0.05)],
    29: [DEPTH, ("D0_dB", 30.0, 1.0, 400.0, 5.0), ("k_off_dB", 0.0, -20.0, 20.0, 1.0)],
    30: [DEPTH],
    32: [DEPTH, ("s_dB", 20.0, 2.0, 300.0, 2.0)],
    33: [DEPTH, ("g_bias", 0.5, 0.0, 3.0, 0.05)],
    34: [DEPTH, ("g_bias", 0.5, 0.0, 3.0, 0.05), ("k_off_dB", 0.0, -20.0, 20.0, 1.0)],
    35: [DEPTH, ("b_frac", 0.4, 0.0, 1.0, 0.05)],
    36: [DEPTH, ("b_frac", 0.4, 0.0, 1.0, 0.05), ("k_off_dB", 0.0, -20.0, 20.0, 1.0)],
}
NAMES = {
    4: "E   release toward T-depth (baseline)",
    10: "a   saturating attack, clamp K",
    11: "b   linear release (30 dB / tr) to T-depth",
    12: "c   two release paths: T-depth (tr) + instantaneous e (tau2)",
    13: "d   attack toward e + k_off",
    14: "e   diode drop: attack only when e > v + delta",
    15: "b'  hybrid release: one-pole to T-depth + constant sink",
    16: "a'  soft saturating attack, K tanh",
    21: "P   parallel RC: diode attack + always-on bleed to T-depth",
    22: "P+a parallel RC + saturating attack (clamp K)",
    23: "P+knee parallel RC + softplus diode knee (s) and drop (Vd)",
    24: "P+d parallel RC + attack toward e + k_off",
    25: "P+c parallel RC + second release path toward e (tau2)",
    26: "P+b' parallel RC + constant current sink",
    27: "P+sat parallel RC + saturating bleed (D0)",
    28: "P+pow parallel RC + power-law bleed (alpha)",
    29: "P+sat+d parallel RC + saturating bleed + k_off",
    30: "P+lin parallel RC, attack in the linear domain",
    32: "P+exp parallel RC, exponential (diode) attack drive, scale s",
    33: "P+bias parallel RC, threshold as a bias subtracted before the log (g)",
    34: "P+bias+d 33 + attack toward e + k_off",
    35: "P+thr parallel RC, bleed returns to (1-b) R + b max(e, R)",
    36: "P+thr+d 35 + attack toward e + k_off",
}


def split(p, mode):
    ta = p[:6]; tr = p[6:11]; ex = np.ascontiguousarray(p[11:], dtype=np.float64)
    return ta, tr, ex


def per_attack_ex(ex, mode, ai):
    return ex


def residuals(p, mode):
    ta, tr, ex = split(p, mode)
    CUR[0] = remap_curves(float(ex[1])) if mode in (33, 34) else CURVES
    out = []
    for ai, a in enumerate(ATTACKS):
        exa = per_attack_ex(ex, mode, ai)
        for rci, rc in enumerate(RECOVERS[:5]):
            out.append(3.0 * (steady(ta[ai], tr[rci], mode, exa) - F[f"disc_ar_{a}_{rc}"]))
        for l in (-25, 2):
            out.append(3.0 * (steady(ta[ai], tr[2], mode, exa, level=float(l)) - F[f"disc_al_{a}_{l}"]))
    for a in (0.1, 1.0, 30.0):
        ai = ATTACKS.index(a); exa = per_attack_ex(ex, mode, ai)
        for f in (100.0, 5000.0):
            out.append(3.0 * (steady(ta[ai], tr[2], mode, exa, f=f) - F[f"disc_af_{a}_f{int(f)}"]))
    for iid in BURSTS:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], mode, per_attack_ex(ex, mode, ai)); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append((0.0 if STEADY_ONLY[0] else 1.0) * w * (e[:n] - ref[:n]))
    return np.concatenate([np.atleast_1d(o) for o in out])


STEADY_ONLY = [False]


def recalibrate(mode, ex, ta=0.54e-3, tr=0.336, iters=3, verbose=True):
    """Re-derive the static curves for this detector at the calibration setting (1 ms / 0.5 s, the model's defaults, where the
    stage-3 static family was fitted through the old E detector): C_r(x) <- C_r(x) + (GR_ref(x) - GR_model(x)), with GR_ref(x)
    the pooled static cloud over all thresholds in x = L - T_k (fact 1) and GR_model from this mode at threshold 16."""
    global CURVES
    ref = {}
    for r_, r in enumerate(RATIOS):
        pts = []
        for k in range(1, 25):
            for L in range(-60, 13, 3):
                key = f"disc_static_{r}_t{k}_{L}"
                if key in F: pts.append((L - T[k - 1], GAIN12 - F[key]))
        pts = np.array(sorted(pts)); ref[r_] = pts
    new = CURVES.copy()
    for it in range(iters):
        CUR[0] = new
        worst = 0.0
        for r_ in range(6):
            pts = ref[r_]
            xs = XG.copy(); Tk = T[15]
            model = np.array([GAIN12 - steady(ta, tr, mode, ex, level=float(Tk + x), ratio=RATIOS[r_]) for x in xs])
            # reference on the grid: local linear regression of the cloud within +-2 dB
            refg = np.empty_like(xs)
            for j, x in enumerate(xs):
                m = np.abs(pts[:, 0] - x) < 2.0
                if m.sum() >= 3:
                    A = np.vstack([np.ones(m.sum()), pts[m, 0] - x]).T; coef = np.linalg.lstsq(A, pts[m, 1], rcond=None)[0]; refg[j] = coef[0]
                else: refg[j] = np.nan
            ok = ~np.isnan(refg) & (refg > 0.05)
            corr = np.where(ok, refg - model, 0.0)
            worst = max(worst, float(np.max(np.abs(corr[ok]))) if ok.any() else 0.0)
            new[r_] = np.maximum(new[r_] + corr, 0.0)
        if verbose: print(f"    recal iter {it + 1}: max |correction| {worst:.3f} dB")
    CURVES = np.ascontiguousarray(new); CUR[0] = CURVES
    if verbose:
        for thr in (8, 16, 22):
            row = []
            for L in (-39, -24, -9, 3):
                key = f"disc_static_4:1_t{thr}_{L}"
                if key in F: row.append(f"L{L}:{steady(ta, tr, mode, ex, level=float(L), thr=thr) - F[key]:+.3f}")
            print(f"    recal check 4:1 thr {thr}: " + " ".join(row))
    return CURVES


def report(mode, r, verbose):
    res = r.fun; ta, tr, ex = split(r.x, mode)
    ss = res[:N_SS] / 3.0; bu = res[N_SS:]
    names = [n for n, *_ in EXTRAS[mode]]
    print(f"[{mode}] {NAMES[mode]}")
    print(f"    attack ms {np.round(ta * 1e3, 3).tolist()}")
    print(f"    recover s {np.round(tr, 4).tolist()}")
    print("    extras    " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, ex)))
    print(f"    steady-state rms {np.sqrt(np.mean(ss ** 2)):.3f} dB  max {np.max(np.abs(ss)):.3f} dB | bursts rms {np.sqrt(np.mean(bu ** 2)):.4f} (weighted)  max {np.max(np.abs(bu)):.3f}"
          f" | nfev {r.nfev} cost {r.cost:.2f}")
    if verbose:
        print("    steady residuals (attack rows; cols recover 0.1/0.25/0.5/0.8/1.2 then level -25/+2):")
        k = 0
        for ai, a in enumerate(ATTACKS):
            row = ss[k:k + 7]; k += 7
            print(f"      atk {a:5.1f}: {np.round(row, 2).tolist()}")
        print("    frequency residuals (0.1/1/30 ms x 100 Hz/5 kHz):", np.round(ss[42:48], 2).tolist())
    sys.stdout.flush()
    return dict(mode=mode, ta=ta, tr=tr, ex=ex, ss_rms=float(np.sqrt(np.mean(ss ** 2))), ss_max=float(np.max(np.abs(ss))),
                b_rms=float(np.sqrt(np.mean(bu ** 2))), b_max=float(np.max(np.abs(bu))))


def profile(mode, x):
    """per-item burst residual summary for a parameter vector"""
    ta, tr, ex = split(x, mode)
    print("    burst profile (item: rms / max |res| dB unweighted, at ms):")
    for iid in BURSTS:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], mode, per_attack_ex(ex, mode, ai)); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        d = e[:n] - ref[:n]; j = int(np.argmax(np.abs(d)))
        print(f"      {iid:24s} rms {np.sqrt(np.mean(d ** 2)):.3f}  max {d[j]:+.3f} @ {j} ms   attack-phase(500-700) max {np.max(np.abs(d[500:700])):.3f}  release(2500-3500) max {np.max(np.abs(d[2500:3500])):.3f}")


def traces(mode, x):
    ta, tr, ex = split(x, mode)
    print("    release traces (model - ref, dB) at ms after burst end: 0 5 10 20 40 80 160 320 640")
    for iid in ["disc_bdepth_-30", "disc_bdepth_-20", "disc_burst_5.0_0.25 s", "disc_bdepth_0", "disc_burst_30.0_0.1 s", "disc_burst_30.0_1.2 s", "disc_burst_0.1_0.1 s"]:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], mode, per_attack_ex(ex, mode, ai)); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        d = e[:n] - ref[:n]; t0 = 2500
        print(f"      {iid:24s} " + " ".join(f"{d[t0 + k]:+.2f}" for k in (0, 5, 10, 20, 40, 80, 160, 320, 640)) + f"   ref GR at +0/+80/+320: {GAIN12 - ref[t0]:.1f} {GAIN12 - ref[t0 + 80]:.1f} {GAIN12 - ref[t0 + 320]:.1f}")


def fit(mode, nfev=80, verbose=False, p0=None, ex0=None, fixdepth=None, prof=False):
    base = [a * 1e-3 for a in ATTACKS] + [0.08, 0.13, 0.32, 0.32, 0.46]
    lo = [1e-5] * 6 + [0.01] * 5; hi = [1.0] * 6 + [5.0] * 5
    xs = [1e-3] * 6 + [0.05] * 5
    for n, init, l, h, sc in EXTRAS[mode]:
        base.append(init); lo.append(l); hi.append(h); xs.append(sc)
    if ex0 is not None:
        for k, v in enumerate(ex0): base[11 + k] = v
    if fixdepth is not None:
        base[11] = fixdepth; lo[11] = fixdepth - 1e-6; hi[11] = fixdepth + 1e-6
    if p0 is not None: base = list(p0)
    t0 = time.time()
    r = least_squares(residuals, np.array(base), bounds=(lo, hi), args=(mode,), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
    out = report(mode, r, verbose)
    if prof: profile(mode, r.x); traces(mode, r.x)
    print(f"    ({time.time() - t0:.0f} s)")
    return out, r


def main():
    modes = [4, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36]
    if "--modes" in sys.argv: modes = [int(m) for m in sys.argv[sys.argv.index("--modes") + 1].split(",")]
    nfev = int(sys.argv[sys.argv.index("--nfev") + 1]) if "--nfev" in sys.argv else 80
    verbose = "--verbose" in sys.argv
    STEADY_ONLY[0] = "--steady-only" in sys.argv
    ex0 = [float(v) for v in sys.argv[sys.argv.index("--ex0") + 1].split(",")] if "--ex0" in sys.argv else None
    fixdepth = float(sys.argv[sys.argv.index("--fixdepth") + 1]) if "--fixdepth" in sys.argv else None
    for m in modes:
        if "--recal" in sys.argv:
            ex_r = np.array([init for _, init, *_ in EXTRAS[m]]);
            if ex0 is not None:
                for k, v in enumerate(ex0): ex_r[k] = v
            recalibrate(m, ex_r)
        fit(m, nfev=nfev, verbose=verbose, ex0=ex0, fixdepth=fixdepth, prof="--profile" in sys.argv)


if __name__ == "__main__":
    main()
