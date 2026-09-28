# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Detector candidates, family "cascaded and decoupled" (Giannoulis, Massberg & Reiss 2012, JAES 60(6), "Digital dynamic range
compressor design, a tutorial and analysis", https://www.eecs.qmul.ac.uk/~josh/documents/2012/GiannoulisMassbergReiss-dynamicrangecompression-JAES2012.pdf).
Copy of fit/tools/detector_candidates.py with extra modes; the original modes 0-4 are kept. Every new mode also fits a global curve
shift `xoff` (added to v - Tk: a constant shift of all thresholds, so the static shift invariance is untouched) and the release depth.
   4  E        log -> branching one-pole (attack toward e, release toward ref = Tk - depth)                       [control, + xoff]
  20  LEAK     decoupled in the analog sense: v += kA (e - v)+ - kR (v - ref); the release conductance is always on
  21  SD_REF   Giannoulis smooth decoupled: y1 = max(e, y1 + (ref - y1) kR); y += (y1 - y) kA
  22  SD_LEV   smooth decoupled, release toward the level: y1 = max(e, y1 + (e - y1) kR); y += (y1 - y) kA
  23  PRE_LOG  fixed pre-smoothing pole tp on the log level, then E
  24  PRE_LIN  fixed pre-smoothing pole tp on |x| (rectifier RC), then log, then E
  25  POST     E, then a fixed post-smoothing pole tp
  26  CASC2_BR two cascaded branching nodes with the same ta / tr (second-order attack)
  27  CASC2_LIN E node, then a linear pole with the same ta (second-order attack)
  28  LEAK_POST LEAK, then a fixed post-smoothing pole tp
  29  LEAK_CASC2 two cascaded LEAK nodes with the same ta / tr
  30  LEAK_D   LEAK with the attack diode offset: v += kA (e - v - dd)+ - kR (v - ref)
  31  PARTIAL  LEAK with the discharge scaled by lam while the attack path conducts: v += kA (e - v)+ - kR (v - ref) (lam if e > v else 1)
  32  PARTIAL_POST PARTIAL, then a fixed post-smoothing pole (tq fixed at 0.5 ms; lam fitted)
  33  CONST    constant discharge while the attack path conducts: e > v: v += kA (e - v) - kR D0 ; else v += kR (ref - v)
  34  REFMAX   release reference is the greater of the threshold reference and the log level minus D0 (diode-OR of two references):
               X = max(ref, e - D0); v += kA (e - v)+ - kR (v - X)
  35  PARALLEL two release resistors: v += kA (e - v)+ - kR (v - ref) - kR2 (v - (e - D0))+, tr2 = c * tr (D0, c fitted)
  36  REFMAX_LEAK REFMAX plus a slow leak toward a floor tied to the threshold: v += ... - kG (v - (Tk - 60)), tg fitted
  37  GATEPEAK peak-hold node p (instant attack, release toward ref) supplies the charge, gated by the instantaneous level:
               p = max(e, p + (ref - p) kR); if e > v: v += (p - v) kA; v -= (v - max(ref, e - D0)) kR
usage: add --phys to start the fit from the transient attack constants (0.1/0.5/1.7/4.7/11/36 ms) and D0 = 14.6
usage: python3 fit/tools/candidates/cascaded-and-decoupled.py --modes 20,21 [--nfev 80] [--verbose]"""
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
EXTRA = {"depth": 0.25, "xoff": 0.0, "tp": 1e-3, "tq": 1.0}
NAMES = {4: "E (control, + xoff)", 20: "LEAK decoupled leak", 21: "SD_REF smooth decoupled -> ref", 22: "SD_LEV smooth decoupled -> level",
         23: "PRE_LOG pre-pole (log) -> E", 24: "PRE_LIN pre-pole (lin) -> log -> E", 25: "POST E -> post-pole", 26: "CASC2_BR two branching nodes",
         27: "CASC2_LIN E -> linear pole(ta)", 28: "LEAK_POST leak -> post-pole", 29: "LEAK_CASC2 two leak nodes", 30: "LEAK_D leak + diode offset",
         31: "PARTIAL leak scaled by lam during conduction", 32: "PARTIAL_POST partial leak -> 0.5 ms post-pole",
         33: "CONST constant discharge D0*kR during conduction", 34: "REFMAX release toward max(ref, e - D0)",
         35: "PARALLEL two release paths (ref, e - D0)", 36: "REFMAX_LEAK REFMAX + slow leak to Tk - 60",
         37: "GATEPEAK peak-held charge source, gated by e > v, REFMAX release"}
HAS_TP = {23, 24, 25, 28, 30, 31, 32, 33, 34, 35, 36, 37}
HAS_TQ = {35, 36}


@njit(cache=False)
def curve_lin(c, x):
    if x <= XG[0]: return c[0]
    if x >= XG[-1]: return c[-1] + (c[-1] - c[-2]) * (x - XG[-1])
    i = int(x - XG[0]); t = x - XG[0] - i
    return c[i] * (1 - t) + c[i + 1] * t


@njit(cache=False)
def run_detector(a, fs, ta, tr, mode, c, Tk, depth, xoff, tp, tq=1.0):
    n = a.shape[0]
    gr = np.empty(n)
    kA = 1.0 - np.exp(-1.0 / (ta * fs)); kR = 1.0 - np.exp(-1.0 / (tr * fs)); kP = 1.0 - np.exp(-1.0 / (max(tp, 1e-9) * fs))
    kR2 = 1.0 - np.exp(-1.0 / (max(tq, 1e-9) * tr * fs)); kG = 1.0 - np.exp(-1.0 / (max(tq, 1e-9) * fs))
    fl = 10.0 ** (FLOOR / 20.0)
    ref = Tk - depth
    v = ref; y = ref; y1 = ref; es = FLOOR; env = 0.0; pk = ref
    for i in range(n):
        e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
        if mode == 4:
            if e > v: v += (e - v) * kA
            else: v += (ref - v) * kR
            out = v
        elif mode == 20:
            v += (max(e - v, 0.0)) * kA - (v - ref) * kR
            out = v
        elif mode == 21:
            y1 = y1 + (ref - y1) * kR
            if e > y1: y1 = e
            y += (y1 - y) * kA
            out = y
        elif mode == 22:
            y1 = y1 + (e - y1) * kR
            if e > y1: y1 = e
            y += (y1 - y) * kA
            out = y
        elif mode == 23:
            es += (e - es) * kP
            if es > v: v += (es - v) * kA
            else: v += (ref - v) * kR
            out = v
        elif mode == 24:
            env += (a[i] - env) * kP
            e2 = 20.0 * np.log10(env) if env > fl else FLOOR
            if e2 > v: v += (e2 - v) * kA
            else: v += (ref - v) * kR
            out = v
        elif mode == 25:
            if e > v: v += (e - v) * kA
            else: v += (ref - v) * kR
            y += (v - y) * kP
            out = y
        elif mode == 26:
            if e > v: v += (e - v) * kA
            else: v += (ref - v) * kR
            if v > y: y += (v - y) * kA
            else: y += (ref - y) * kR
            out = y
        elif mode == 27:
            if e > v: v += (e - v) * kA
            else: v += (ref - v) * kR
            y += (v - y) * kA
            out = y
        elif mode == 28:
            v += (max(e - v, 0.0)) * kA - (v - ref) * kR
            y += (v - y) * kP
            out = y
        elif mode == 29:
            v += (max(e - v, 0.0)) * kA - (v - ref) * kR
            y += (max(v - y, 0.0)) * kA - (y - ref) * kR
            out = y
        elif mode == 30:
            v += (max(e - v - tp, 0.0)) * kA - (v - ref) * kR   # tp reused as the diode offset (dB)
            out = v
        elif mode == 31:
            if e > v: v += (e - v) * kA - (v - ref) * kR * tp   # tp reused as lam
            else: v += (ref - v) * kR
            out = v
        elif mode == 33:
            if e > v: v += (e - v) * kA - tp * kR   # tp reused as D0 (dB)
            else: v += (ref - v) * kR
            out = v
        elif mode == 34:
            X = e - tp
            if X < ref: X = ref
            v += max(e - v, 0.0) * kA - (v - X) * kR
            out = v
        elif mode == 35:
            v += max(e - v, 0.0) * kA - (v - ref) * kR - max(v - (e - tp), 0.0) * kR2
            out = v
        elif mode == 36:
            X = e - tp
            if X < ref: X = ref
            v += max(e - v, 0.0) * kA - (v - X) * kR - (v - (Tk - 60.0)) * kG
            out = v
        elif mode == 37:
            pk = pk + (ref - pk) * kR
            if e > pk: pk = e
            X = e - tp
            if X < ref: X = ref
            if e > v: v += (pk - v) * kA
            v -= (v - X) * kR
            out = v
        elif mode == 32:
            if e > v: v += (e - v) * kA - (v - ref) * kR * tp
            else: v += (ref - v) * kR
            kQ = 1.0 - np.exp(-1.0 / (0.5e-3 * fs))
            y += (v - y) * kQ
            out = y
        else:
            out = e
        gr[i] = max(curve_lin(c, out - Tk + xoff), 0.0)
    return gr


def env_dB(item_id, ta, tr, mode):
    it = ITEMS[item_id]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    x = protocol.stimulus(st, fs)[0]
    r = RATIOS.index(s.get("discrete_ratio", "4:1")); Tk = T[int(s["discrete_threshold"]) - 1]
    gr = run_detector(np.abs(x), float(fs), ta, tr, mode, CURVES[r], Tk, EXTRA["depth"], EXTRA["xoff"], EXTRA["tp"], EXTRA["tq"])
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def steady(ta, tr, mode, level=-10.0, f=1000.0, thr=16, ratio="4:1", secs=3.0):
    t = np.arange(int(secs * FS)) / FS
    x = np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))
    gr = run_detector(x, float(FS), ta, tr, mode, CURVES[RATIOS.index(ratio)], T[thr - 1], EXTRA["depth"], EXTRA["xoff"], EXTRA["tp"], EXTRA["tq"])
    per = int(round(FS / f)); seg = gr[-per * 50:]
    return 20 * np.log10(np.mean(10 ** (-seg / 20.0))) + GAIN12


BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS[:5]] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
N_SS = 36 + 12 + 6


def unpack(p, mode):
    EXTRA["depth"] = p[11]; EXTRA["xoff"] = p[12]
    EXTRA["tp"] = p[13] if mode in HAS_TP else 1e-3
    EXTRA["tq"] = p[14] if mode in HAS_TQ else 1.0
    return p[:6], p[6:11]


def residuals(p, mode):
    ta, tr = unpack(p, mode)
    out = []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            out.append(3.0 * (steady(ta[ai], tr[rci], mode) - F[f"disc_ar_{a}_{rc}"]))
        for l in (-25, 2):
            out.append(3.0 * (steady(ta[ai], tr[2], mode, level=float(l)) - F[f"disc_al_{a}_{l}"]))
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            out.append(3.0 * (steady(ta[ATTACKS.index(a)], tr[2], mode, f=f) - F[f"disc_af_{a}_f{int(f)}"]))
    for iid in BURSTS:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], mode); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate([np.atleast_1d(o) for o in out])


def report(name, mode, r, verbose):
    res = r.fun; ss = res[:N_SS] / 3.0; bu = res[N_SS:]
    extra = f" depth {r.x[11]:.2f} dB xoff {r.x[12]:.2f} dB"
    if mode in (31, 32): extra += f" lam {r.x[13]:.3f}"
    elif mode in (33, 34, 37): extra += f" D0 {r.x[13]:.2f} dB"
    elif mode == 35: extra += f" D0 {r.x[13]:.2f} dB tr2/tr {r.x[14]:.3f}"
    elif mode == 36: extra += f" D0 {r.x[13]:.2f} dB tg {r.x[14]:.2f} s"
    elif mode == 30: extra += f" dd {r.x[13]:.2f} dB"
    elif mode in HAS_TP: extra += f" tp {r.x[13] * 1e3:.3f} ms"
    print(f"[{mode}] {name}: attack ms {np.round(r.x[:6] * 1e3, 3)} recover s {np.round(r.x[6:11], 4)}{extra}")
    print(f"    steady-state rms {np.sqrt(np.mean(ss ** 2)):.3f} dB max {np.max(np.abs(ss)):.2f} | bursts rms {np.sqrt(np.mean(bu ** 2)):.3f} (weighted) | max |burst| {np.max(np.abs(bu)):.2f} | nfev {r.nfev} cost {r.cost:.2f}")
    if verbose:
        print("    steady residuals (attack rows; cols recover 0.1/0.25/0.5/0.8/1.2 then level -25/+2):")
        k = 0
        for a in ATTACKS:
            print(f"      atk {a:5.1f}: {np.round(ss[k:k + 7], 2)}"); k += 7
        print("    frequency residuals (0.1/1/30 ms x 100 Hz/5 kHz):", np.round(ss[42:48], 2))
        # per-burst rms (unweighted, whole envelope)
        worst = []
        for iid in BURSTS:
            s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
            ta, tr = unpack(r.x, mode)
            e = env_dB(iid, ta[ai], tr[rci], mode); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
            d = e[:n] - ref[:n]; worst.append((float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d))), iid))
        worst.sort(reverse=True)
        print("    worst bursts (rms, max, item):", [(round(a, 3), round(b, 2), c) for a, b, c in worst[:4]])
    sys.stdout.flush()


def main():
    modes = [int(m) for m in next((a.split("=")[1] for a in sys.argv if a.startswith("--modes=")), "20").split(",")]
    nfev = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--nfev=")), "80"))
    verbose = "--verbose" in sys.argv
    p0 = [a * 1e-3 for a in ATTACKS] + [0.09, 0.17, 0.33, 0.33, 0.478] + [0.25, 0.0]
    phys = "--phys" in sys.argv
    if phys: p0 = [0.1e-3, 0.5e-3, 1.7e-3, 4.7e-3, 11e-3, 36e-3] + [0.09, 0.131, 0.33, 0.33, 0.478] + [0.25, 0.1]
    lo = [1e-5] * 6 + [0.01] * 5 + [-40.0, -6.0]; hi = [1.0] * 6 + [5.0] * 5 + [80.0, 6.0]
    xs = [1e-3] * 6 + [0.05] * 5 + [2.0, 0.3]
    for mode in modes:
        pp, l_, h_, x_ = list(p0), list(lo), list(hi), list(xs)
        if mode in HAS_TP:
            if mode == 30: pp += [1.0]; l_ += [0.0]; h_ += [20.0]; x_ += [0.5]
            elif mode in (31, 32): pp += [0.7]; l_ += [0.0]; h_ += [1.0]; x_ += [0.1]
            elif mode in (33, 34, 37): pp += [14.6 if phys else 8.0]; l_ += [0.0]; h_ += [60.0]; x_ += [1.0]
            elif mode == 35: pp += [14.6, 1.0]; l_ += [0.0, 0.05]; h_ += [60.0, 50.0]; x_ += [1.0, 0.3]
            elif mode == 36: pp += [14.6 if phys else 20.0, 5.0]; l_ += [0.0, 0.3]; h_ += [60.0, 100.0]; x_ += [1.0, 1.0]
            elif mode in (25, 28): pp += [1e-3]; l_ += [2e-5]; h_ += [0.05]; x_ += [5e-4]
            else: pp += [3e-4]; l_ += [2e-5]; h_ += [5e-3]; x_ += [1e-4]
        r = least_squares(residuals, pp, bounds=(l_, h_), args=(mode,), x_scale=x_, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
        report(NAMES.get(mode, str(mode)), mode, r, verbose)


if __name__ == "__main__":
    main()
