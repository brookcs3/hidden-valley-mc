# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Detector candidates, family "RMS / power domain". Copy of fit/tools/detector_candidates.py with new modes (the old ones are kept):
  4   E   log rectifier -> dB one-pole; attack toward e, release toward (Tk - depth)                                  [baseline]
  10  P   square -> one-pole on the power with attack/release branching (x^2 > p charges) -> 10 log10 -> curve         (a)
  11  PE  square -> power one-pole; attack toward x^2, release toward the threshold-referred power 10^((Tk-depth)/10)   (a, E-style release)
  12  PL  attack in the linear POWER domain toward x^2, release in dB toward (Tk - depth)                              ("x^2 in mode E")
  13  AL  attack in the linear AMPLITUDE domain toward |x|, release in dB toward (Tk - depth)                          (amplitude analogue of 12)
  14  R   dbx-2252-style log: e = 10 log10(x^2 + eps), eps = 10^((Tk - eps_db)/10); dB one-pole with attack/release    (c)
  15  RP  same as 14 with a fixed fast pre-averaging pole on x^2 before the log (tau_pre)                               (c, pre-smoothed)
  16  H   hybrid: E detector mixed in dB with a true-RMS averager (fixed tau_rms): e = w*vE + (1-w)*10log10(p_rms)      (d)
  17  RE  like 14 (log with threshold-referred floor) but the release goes toward (Tk - depth) as in E                  (c + E)
  20  GM  attack one-pole on |x|^m with m fitted (m -> 0 is E, 1 amplitude, 2 power), dB release toward (Tk - depth)
  21  GM2 attack and release one-pole on |x|^m, release toward the threshold-referred floor
  22/23   charge target k dB above the level (no-ops: a pure shift of the table; kept for the record)
  24  EL  E with the release leak always on (charge and discharge act simultaneously)
  25  DL  softplus diode charge (knee w dB) + always-on leak
  26  ALL amplitude-domain attack + always-on dB leak
  27  EL2 E + switched release + always-on leak with its own conductance toward a deeper reference
  28  ELC E + switched release + always-on constant-current discharge (Dc*kR dB per sample)
  29  ELC' the winner in clean form: attack v += (e - v) kA - L kR ; release v += ((Tk - depth) - v) kR ; node starts at rest
  30  E0  E with the node started at rest (initial-condition control for 4)
usage: cd <repo> && python3 fit/tools/candidates/rms-and-power-domain.py [--modes 4,10,11] [--nfev 80] [--verbose] [--p0 <13 comma-separated values>]
results of the run of 2026-09-27 are summarised in the report; the winner is
  python3 fit/tools/candidates/rms-and-power-domain.py --modes 29 --nfev 200 --verbose --p0 1e-5,1.09e-4,5.87e-4,2.59e-3,8.19e-3,24.736e-3,0.0941,0.1341,0.3328,0.3334,0.4883,0.517,17.209"""
import os, sys, numpy as np
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


@njit(cache=True)
def curve_lin(c, x):
    if x <= XG[0]: return c[0]
    if x >= XG[-1]: return c[-1] + (c[-1] - c[-2]) * (x - XG[-1])
    i = int(x - XG[0]); t = x - XG[0] - i
    return c[i] * (1 - t) + c[i + 1] * t


@njit(cache=True)
def run_detector(a, fs, ta, tr, mode, c, Tk, ex):
    """a: |sidechain|. ex: extra parameters (mode specific). Returns the gain reduction (dB) per sample."""
    n = a.shape[0]
    gr = np.empty(n)
    kA = 1.0 - np.exp(-1.0 / (ta * fs)); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    fl = 10.0 ** (FLOOR / 20.0); fl2 = fl * fl
    v = FLOOR; g = 1.0; grs = 0.0; env = 0.0; p = fl2; ppre = fl2; prms = fl2
    depth = ex[0]
    P0 = 10.0 ** ((Tk - depth) / 10.0)
    eps = 10.0 ** ((Tk - ex[0]) / 10.0)   # modes 14/15/17: ex[0] is the floor below threshold (dB)
    kPre = 1.0
    if mode == 15: kPre = 1.0 - np.exp(-1.0 / (ex[1] * 1e-3 * fs))
    kRms = 1.0
    if mode == 16: kRms = 1.0 - np.exp(-1.0 / (ex[2] * 1e-3 * fs))
    ym = 0.0; Y0 = 0.0; kup = 1.0
    if mode == 20 or mode == 21 or mode == 22:
        ym = 10.0 ** (ex[1] * FLOOR / 20.0); Y0 = 10.0 ** (ex[1] * (Tk - depth) / 20.0)
    if mode == 22: kup = 10.0 ** (ex[2] / 20.0)
    wk = 1.0
    if mode == 25: wk = ex[1]
    if mode == 24 or mode == 29 or mode == 30: v = Tk - depth   # start at rest (the node sits at the release reference, as the C++ does)
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
        elif mode == 3:
            if a[i] > env: env += (a[i] - env) * kA
            else: env += (a[i] - env) * kR
            e = 20.0 * np.log10(env) if env > fl else FLOOR
            gr[i] = max(curve_lin(c, e - Tk), 0.0)
        elif mode == 4:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 10:
            x2 = a[i] * a[i]
            if x2 > p: p += (x2 - p) * kA
            else: p += (x2 - p) * kR
            e = 10.0 * np.log10(p) if p > fl2 else FLOOR
            gr[i] = max(curve_lin(c, e - Tk), 0.0)
        elif mode == 11:
            x2 = a[i] * a[i]
            if x2 > p: p += (x2 - p) * kA
            else: p += (P0 - p) * kR
            e = 10.0 * np.log10(p) if p > fl2 else FLOOR
            gr[i] = max(curve_lin(c, e - Tk), 0.0)
        elif mode == 12:
            x2 = a[i] * a[i]
            if x2 > p:
                p += (x2 - p) * kA
                v = 10.0 * np.log10(p) if p > fl2 else FLOOR
            else:
                v += ((Tk - depth) - v) * kR
                p = 10.0 ** (v / 10.0)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 13:
            if a[i] > env:
                env += (a[i] - env) * kA
                v = 20.0 * np.log10(env) if env > fl else FLOOR
            else:
                v += ((Tk - depth) - v) * kR
                env = 10.0 ** (v / 20.0)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 14:
            e = 10.0 * np.log10(a[i] * a[i] + eps)
            if e > v: v += (e - v) * kA
            else: v += (e - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 15:
            ppre += (a[i] * a[i] - ppre) * kPre
            e = 10.0 * np.log10(ppre + eps)
            if e > v: v += (e - v) * kA
            else: v += (e - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 16:
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            prms += (a[i] * a[i] - prms) * kRms
            er = 10.0 * np.log10(prms) if prms > fl2 else FLOOR
            w = ex[1]
            gr[i] = max(curve_lin(c, w * v + (1.0 - w) * er - Tk), 0.0)
        elif mode == 17:
            e = 10.0 * np.log10(a[i] * a[i] + eps)
            if e > v: v += (e - v) * kA
            else: v += ((Tk - ex[1]) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 20:
            # generalized-domain attack: one-pole on y = |x|^m (m = ex[1]; m -> 0 is E's dB one-pole, 1 amplitude, 2 power),
            # dB release toward (Tk - depth). y and v are kept consistent across the branch switch.
            mm = ex[1]
            xm = a[i] ** mm
            if xm > ym:
                ym += (xm - ym) * kA
                v = (20.0 / mm) * np.log10(ym) if ym > 0.0 else FLOOR
            else:
                v += ((Tk - depth) - v) * kR
                ym = 10.0 ** (mm * v / 20.0)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 22:
            # 20 with the attack charging toward a target k dB ABOVE the rectified level (log-domain diode offset): y -> (|x| 10^(k/20))^m
            mm = ex[1]
            xm = (a[i] * kup) ** mm
            if xm > ym:
                ym += (xm - ym) * kA
                v = (20.0 / mm) * np.log10(ym) if ym > 0.0 else FLOOR
            else:
                v += ((Tk - depth) - v) * kR
                ym = 10.0 ** (mm * v / 20.0)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 23:
            # E with the attack charging toward e + k (control for 22 in the pure log domain)
            e = 20.0 * np.log10(a[i]) + ex[1] if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 24:
            # EL: E with the release leak ALWAYS on (the release resistor is never switched out of the RC): attack charge when e > v,
            # plus the discharge toward (Tk - depth) every sample. DC input settles at (v - V0) tr/(ta + tr) below the peak.
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 25:
            # DL: diode charge (softplus with knee width w dB, exponential below, linear above) + always-on leak toward (Tk - depth)
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            z = (e - v) / wk
            sp = (z + np.log1p(np.exp(-z))) if z > 0.0 else np.log1p(np.exp(z))
            v += wk * sp * kA
            v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 26:
            # ALL: amplitude-domain attack (AL) + always-on dB leak toward (Tk - depth)
            if a[i] > env:
                env += (a[i] - env) * kA
                v = 20.0 * np.log10(env) if env > fl else FLOOR
            v += ((Tk - depth) - v) * kR
            env = 10.0 ** (v / 20.0)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 27:
            # EL2: switched release (branch, toward Tk - depth, kR) + a separate always-on leak with conductance g*kR toward a deeper
            # reference (Tk - depth - dd): a second resistor from the node to a lower rail. Extras: depth, g, dd.
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            v += ((Tk - depth - ex[2]) - v) * kR * ex[1]
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 28:
            # ELC: switched release (branch, toward Tk - depth) + an always-on CONSTANT-CURRENT discharge of Dc*kR dB per sample
            # (level-independent pull; a bias current source on the node). Extras: depth, Dc.
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            v -= ex[1] * kR
            if v < Tk - depth - ex[1]: v = Tk - depth - ex[1]
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 29:
            # ELC clean form (the winner): while charging, the node also loses a fixed L dB x kR per sample (a constant-current leak,
            # level-independent); while releasing it is E (one-pole toward Tk - depth). Extras: depth, L.
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v:
                v += (e - v) * kA - ex[1] * kR
            else:
                v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 30:
            # E with the node started at rest (control: same as 4 but with the C++ initial condition)
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 21:
            # 20 with the release also in the |x|^m domain toward the threshold-referred floor (single domain, dbx-2252 shape when m=2)
            mm = ex[1]
            xm = a[i] ** mm
            if xm > ym: ym += (xm - ym) * kA
            else: ym += (Y0 - ym) * kR
            v = (20.0 / mm) * np.log10(ym) if ym > 0.0 else FLOOR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        else:
            gr[i] = 0.0
    return gr


# extra parameters per mode: (name, init, lo, hi, x_scale)
EXTRAS = {
    0: [], 1: [], 2: [], 3: [],
    4: [("depth", 10.0, -40.0, 80.0, 2.0)],
    10: [],
    11: [("depth", 10.0, -40.0, 80.0, 2.0)],
    12: [("depth", 10.0, -40.0, 80.0, 2.0)],
    13: [("depth", 10.0, -40.0, 80.0, 2.0)],
    14: [("eps_db", 10.0, -20.0, 80.0, 2.0)],
    15: [("eps_db", 10.0, -20.0, 80.0, 2.0), ("tau_pre_ms", 0.3, 0.005, 20.0, 0.1)],
    16: [("depth", 10.0, -40.0, 80.0, 2.0), ("w", 0.7, 0.0, 1.0, 0.1), ("tau_rms_ms", 30.0, 0.5, 2000.0, 10.0)],
    17: [("eps_db", 10.0, -20.0, 80.0, 2.0), ("depth", 10.0, -40.0, 80.0, 2.0)],
    20: [("depth", 1.0, -40.0, 80.0, 2.0), ("m", 1.0, 0.05, 3.0, 0.1)],
    21: [("depth", 1.0, -40.0, 80.0, 2.0), ("m", 1.0, 0.05, 3.0, 0.1)],
    22: [("depth", 1.0, -40.0, 80.0, 2.0), ("m", 1.0, 0.05, 3.0, 0.1), ("k_db", 1.0, -6.0, 30.0, 0.5)],
    23: [("depth", 1.0, -40.0, 80.0, 2.0), ("k_db", 1.0, -6.0, 30.0, 0.5)],
    24: [("depth", 1.0, -40.0, 80.0, 2.0)],
    25: [("depth", 1.0, -40.0, 80.0, 2.0), ("knee_db", 2.0, 0.05, 30.0, 0.5)],
    26: [("depth", 1.0, -40.0, 80.0, 2.0)],
    27: [("depth", 1.0, -40.0, 80.0, 2.0), ("g", 0.6, 0.0, 3.0, 0.1), ("dd", 20.0, 0.0, 120.0, 5.0)],
    28: [("depth", 1.0, -40.0, 80.0, 2.0), ("Dc", 25.0, 0.0, 120.0, 5.0)],
    29: [("depth", 0.5, -40.0, 80.0, 2.0), ("L", 17.0, 0.0, 120.0, 5.0)],
    30: [("depth", 10.0, -40.0, 80.0, 2.0)],
}
NAMES = {0: "A log->dB one-pole->curve", 1: "B curve->gain smoothing (linear)", 2: "C curve->GR smoothing (dB)", 3: "D linear envelope->log->curve",
         4: "E log->one-pole, release toward threshold - depth",
         10: "P  power one-pole, branching on x^2 (a)", 11: "PE power one-pole, release toward threshold-referred power (a+E)",
         12: "PL linear-power attack, dB release toward threshold - depth (x^2 in E)", 13: "AL linear-amplitude attack, dB release toward threshold - depth",
         14: "R  log(x^2 + eps) with threshold-referred eps, dB one-pole (2252-style) (c)", 15: "RP same as R with fixed pre-averaging of x^2",
         16: "H  hybrid E-peak / true-RMS mix in dB (d)", 17: "RE log(x^2 + eps) floor + release toward threshold - depth",
         20: "GM attack one-pole on |x|^m (m fitted), dB release toward threshold - depth", 21: "GM2 attack and release one-pole on |x|^m (m fitted), release toward threshold-referred floor",
         22: "GMK |x|^m attack toward level + k dB, dB release toward threshold - depth", 23: "EK E with attack toward e + k dB (log-domain control)",
         24: "EL E with the release leak always on (simultaneous charge and discharge)", 25: "DL softplus diode charge (knee w dB) + always-on leak",
         26: "ALL amplitude-domain attack + always-on dB leak toward threshold - depth",
         27: "EL2 E + switched release + always-on leak (conductance g*kR) toward a deeper reference (depth + dd)",
         28: "ELC E + switched release + always-on constant-current discharge (Dc*kR dB per sample)",
         29: "ELC' clean form: attack v += (e-v) kA - L kR; release E toward threshold - depth (node starts at rest)",
         30: "E0 E with the node started at rest (initial-condition control)"}
BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS[:5]] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]


def _ex(mode, p):
    ex = np.asarray(p[11:], dtype=float)
    return ex if len(ex) else np.zeros(1)


def env_dB(item_id, ta, tr, mode, ex):
    it = ITEMS[item_id]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    x = protocol.stimulus(st, fs)[0]
    r = RATIOS.index(s.get("discrete_ratio", "4:1")); Tk = T[int(s["discrete_threshold"]) - 1]
    gr = run_detector(np.abs(x), float(fs), ta, tr, mode, CURVES[r], Tk, ex)
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def steady(ta, tr, mode, ex, level=-10.0, f=1000.0, thr=16, ratio="4:1", secs=3.0):
    t = np.arange(int(secs * FS)) / FS
    x = np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))
    gr = run_detector(x, float(FS), ta, tr, mode, CURVES[RATIOS.index(ratio)], T[thr - 1], ex)
    per = int(round(FS / f)); seg = gr[-per * 50:]
    return 20 * np.log10(np.mean(10 ** (-seg / 20.0))) + GAIN12


def residuals(p, mode):
    ta = p[:6]; tr = p[6:11]; ex = _ex(mode, p)
    out = []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            out.append(3.0 * (steady(ta[ai], tr[rci], mode, ex) - F[f"disc_ar_{a}_{rc}"]))
        for l in (-25, 2):
            out.append(3.0 * (steady(ta[ai], tr[2], mode, ex, level=float(l)) - F[f"disc_al_{a}_{l}"]))
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            out.append(3.0 * (steady(ta[ATTACKS.index(a)], tr[2], mode, ex, f=f) - F[f"disc_af_{a}_f{int(f)}"]))
    for iid in BURSTS:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], mode, ex); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate([np.atleast_1d(o) for o in out])


def report(mode, r, verbose):
    res = r.fun; n_ss = 36 + 12 + 6
    names = [e[0] for e in EXTRAS[mode]]
    extra = " ".join(f"{nm}={val:.3f}" for nm, val in zip(names, r.x[11:]))
    print(f"[{mode}] {NAMES[mode]}")
    print(f"    attack ms {np.round(r.x[:6] * 1e3, 3)} recover s {np.round(r.x[6:11], 4)} {extra}")
    ss = res[:n_ss] / 3.0
    print(f"    steady-state rms {np.sqrt(np.mean(ss ** 2)):.3f} dB max {np.max(np.abs(ss)):.2f} | bursts rms {np.sqrt(np.mean(res[n_ss:] ** 2)):.3f} (weighted) max |burst| {np.max(np.abs(res[n_ss:])):.2f} | nfev {r.nfev} cost {r.cost:.2f}")
    if verbose:
        print("    steady residuals (attack rows; cols recover 0.1/0.25/0.5/0.8/1.2 then level -25/+2):")
        k = 0
        for ai, a in enumerate(ATTACKS):
            row = res[k:k + 7] / 3.0; k += 7
            print(f"      atk {a:5.1f}: {np.round(row, 2)}")
        print("    frequency residuals (0.1/1/30 ms x 100 Hz/5 kHz):", np.round(res[42:48] / 3.0, 2))
    sys.stdout.flush()


def main():
    modes = [4, 10, 11, 12, 13, 14, 15, 16, 17]
    if "--modes" in sys.argv: modes = [int(m) for m in sys.argv[sys.argv.index("--modes") + 1].split(",")]
    nfev = int(sys.argv[sys.argv.index("--nfev") + 1]) if "--nfev" in sys.argv else 80
    verbose = "--verbose" in sys.argv
    p0 = [a * 1e-3 for a in ATTACKS] + [0.08, 0.13, 0.32, 0.32, 0.46]
    lo = [1e-5] * 6 + [0.002] * 5; hi = [1.0] * 6 + [5.0] * 5
    for mode in modes:
        ex = EXTRAS[mode]
        pp = p0 + [e[1] for e in ex]; l_ = lo + [e[2] for e in ex]; h_ = hi + [e[3] for e in ex]
        if "--p0" in sys.argv:   # full initial vector override (attack s x6, recover s x5, extras), e.g. to restart from another mode's solution
            pp = [float(s) for s in sys.argv[sys.argv.index("--p0") + 1].split(",")]
        xs = [1e-3] * 6 + [0.05] * 5 + [e[4] for e in ex]
        r = least_squares(residuals, pp, bounds=(l_, h_), args=(mode,), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
        report(mode, r, verbose)


if __name__ == "__main__":
    main()
