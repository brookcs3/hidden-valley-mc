# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Which detector topology does the reference's discrete stage have? Four candidates, each fitted (attack and recover time constants)
to the same data: the steady-state table against attack and recover (disc_ar_*), the level and frequency checks (disc_al_*, disc_af_*)
and every burst envelope (disc_burst_*, disc_bdepth_*). The static curves and thresholds from stage 3a are used as they are.
  A  log rectifier -> one-pole in dB (attack when rising, recover when falling) -> gain computer        (the first C++ implementation)
  B  log rectifier -> gain computer -> smooth the GAIN, linear amplitude domain (attack when falling, recover when rising)
  C  log rectifier -> gain computer -> smooth the gain reduction in dB
  D  linear envelope of |x| (attack/recover one-poles) -> log -> gain computer
usage: python3 fit/tools/detector_candidates.py"""
import os, sys, numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, protocol, FS  # noqa: E402

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
XG = -20.0 + np.arange(96)
cal = load_cal()
GAIN = cal[MODEL.field("d_gain_db")]; XGN = float(cal[MODEL.fields["x_gain_db"][0]]); GAIN12 = float(GAIN[11] + XGN)
CURVES = cal[MODEL.field("d_curve")].reshape(6, 96).copy(); T = cal[MODEL.field("d_thr_db")].copy()
FLOOR = -100.0
DEPTH = [20.0]
DEPTHS = np.zeros(6)


@njit(cache=True)
def curve_lin(c, x):
    if x <= XG[0]: return c[0]
    if x >= XG[-1]: return c[-1] + (c[-1] - c[-2]) * (x - XG[-1])
    i = int(x - XG[0]); t = x - XG[0] - i
    return c[i] * (1 - t) + c[i + 1] * t


@njit(cache=True)
def run_detector(a, fs, ta, tr, mode, c, Tk, depth=20.0):
    """a: |sidechain| (or signed for D uses |a| too). Returns the gain reduction (dB) per sample."""
    n = a.shape[0]
    gr = np.empty(n)
    kA = 1.0 - np.exp(-1.0 / (ta * fs)); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    fl = 10.0 ** (FLOOR / 20.0)
    v = FLOOR; g = 1.0; grs = 0.0; env = 0.0
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
            # release toward the threshold minus a fixed depth (a dbx-style discharge to the threshold reference)
            e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += ((Tk - depth) - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        else:
            if a[i] > env: env += (a[i] - env) * kA
            else: env += (a[i] - env) * kR
            e = 20.0 * np.log10(env) if env > fl else FLOOR
            gr[i] = max(curve_lin(c, e - Tk), 0.0)
    return gr


def env_dB(item_id, ta, tr, mode, depth=None):
    it = ITEMS[item_id]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    if depth is not None: DEPTH[0] = depth
    x = protocol.stimulus(st, fs)[0]
    r = RATIOS.index(s.get("discrete_ratio", "4:1")); Tk = T[int(s["discrete_threshold"]) - 1]
    gr = run_detector(np.abs(x), float(fs), ta, tr, mode, CURVES[r], Tk, DEPTH[0])
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def steady(ta, tr, mode, level=-10.0, f=1000.0, thr=16, ratio="4:1", secs=3.0, depth=None):
    if depth is not None: DEPTH[0] = depth
    t = np.arange(int(secs * FS)) / FS
    x = np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))
    gr = run_detector(x, float(FS), ta, tr, mode, CURVES[RATIOS.index(ratio)], T[thr - 1], DEPTH[0])
    per = int(round(FS / f)); seg = gr[-per * 50:]
    return 20 * np.log10(np.mean(10 ** (-seg / 20.0))) + GAIN12


BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS[:5]] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]


def residuals(p, mode):
    ta = p[:6]; tr = p[6:11]
    if len(p) == 12: DEPTH[0] = p[11]
    dep = (lambda rci: p[11 + rci]) if len(p) > 12 else (lambda rci: DEPTH[0])
    m = 4 if mode == 5 else mode
    out = []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            out.append(3.0 * (steady(ta[ai], tr[rci], m, depth=dep(rci)) - F[f"disc_ar_{a}_{rc}"]))
        for l in (-25, 2):
            out.append(3.0 * (steady(ta[ai], tr[2], m, level=float(l), depth=dep(2)) - F[f"disc_al_{a}_{l}"]))
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            out.append(3.0 * (steady(ta[ATTACKS.index(a)], tr[2], m, f=f, depth=dep(2)) - F[f"disc_af_{a}_f{int(f)}"]))
    for iid in BURSTS:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], m, depth=dep(rci)); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate([np.atleast_1d(o) for o in out])


def main():
    p0 = np.array([a * 1e-3 for a in ATTACKS] + [0.08, 0.13, 0.32, 0.32, 0.46])
    lo = [1e-5] * 6 + [0.01] * 5; hi = [1.0] * 6 + [5.0] * 5
    cands = ((5, "F release toward threshold - depth, depth per recover position"),) if "--f-only" in sys.argv else \
            ((4, "E log->one-pole, release toward threshold - depth"),) if "--e-only" in sys.argv else \
            ((0, "A log->dB one-pole->curve"), (1, "B curve->gain smoothing (linear)"), (2, "C curve->GR smoothing (dB)"), (3, "D linear envelope->log->curve"), (4, "E release toward threshold - depth"))
    for mode, name in cands:
        if mode == 5: pp, l_, h_ = (list(p0) + [0.0] * 5, lo + [-40.0] * 5, hi + [80.0] * 5)
        elif mode == 4: pp, l_, h_ = (list(p0) + [10.0], lo + [-40.0], hi + [80.0])
        else: pp, l_, h_ = (p0, lo, hi)
        r = least_squares(residuals, pp, bounds=(l_, h_), args=(mode,), x_scale=[1e-3] * 6 + [0.05] * 5 + [2.0] * (len(pp) - 11), diff_step=1e-3, max_nfev=80, loss="soft_l1", f_scale=1.0)
        res = r.fun
        n_ss = 36 + 12 + 6
        print(f"{name}: attack ms {np.round(r.x[:6] * 1e3, 3)} recover s {np.round(r.x[6:11], 4)}" + (f" depth {np.round(r.x[11:], 2)} dB" if mode >= 4 else ""))
        print(f"    steady-state rms {np.sqrt(np.mean((res[:n_ss] / 3.0) ** 2)):.3f} dB | bursts rms {np.sqrt(np.mean(res[n_ss:] ** 2)):.3f} (weighted) | max |burst| {np.max(np.abs(res[n_ss:])):.2f}")
        if "--verbose" in sys.argv:
            tab = (res[:36] / 3.0).reshape(6, 6)   # attack x (5 recovers + 2 levels interleaved)... print the raw first 36 in order
            print("    steady residuals (attack rows; cols recover 0.1/0.25/0.5/0.8/1.2 then level -25/+2):")
            k = 0
            for ai, a in enumerate(ATTACKS):
                row = res[k:k + 7] / 3.0; k += 7
                print(f"      atk {a:5.1f}: {np.round(row, 2)}")
            print("    frequency residuals (0.1/1/30 ms x 100 Hz/5 kHz):", np.round(res[42:48] / 3.0, 2))

if __name__ == "__main__":
    main()
