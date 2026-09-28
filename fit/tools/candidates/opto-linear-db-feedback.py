# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical stage candidate "linear-db-feedback" (hypothesis A).

Hypothesis: the reference's optical stage is a FEEDBACK loop whose light/cell law is linear in dB. With the sidechain tapped from the
divider output v = x / (1 + cond) (before the make-up), the loop computes

    S_dB   = 20 log10(detector(|v|))                       detector: (i) peak-ish one-pole on |v| (attack ta, release tr)
                                                                     (ii) RMS-like averager: one-pole on v^2 (tau), square root
    GR*    = k * softplus_w(S_dB + D_thr)                  linear-in-dB law; softplus of width w dB (w -> 0: hard knee at -D_thr dBFS)
    c*     = 10^(GR*/20) - 1                               target conductance ("light")
    cond   = sum_i w_i s_i,  s_i chasing c* with attack/release one-poles (the three states of src/dsp/Opto.hpp)

so that above the knee the closed loop gives a fixed ratio 1 + k (static slope k / (1 + k) dB per dB) and a knee as hard as the law's.
The make-up and the no-compression trim of each threshold position are taken from the reference's own -50 dBFS gain (see the note on
the trim law in the output). Stage amplifier terms b2, b3 come from fit/data/constants.json.

Stages: A  static family (thresholds 6, 10, 14, 20, 24 x levels -50..14 dBFS, 1 kHz) with k, w, the five drives and the detector
           time constants free, the three conductance states fixed at the stage-4 values;
        B  joint fit of the same statics and the burst envelopes (opto_burst_*, opto_blen_*, opto_pulses), states free as well;
        C  joint fit of the statics of 6, 10, 14, 18, 20, 22, 24, the bursts AND the harmonics under gain reduction (H3, H5 of the
           1 kHz grid t14/t18/t22 x -20/-10/0 dBFS plus t18 -10 dBFS at 100 Hz and 4 kHz): can one parameter set carry all three?
        D  as C plus one parameter alpha: the cell's time constants shrink as (1 + light)^-alpha (light = the target conductance c*,
           or the conductance itself with --proxy cond), the CdS "faster when lit harder" behaviour. Motivation: the reference's H3 at
           1 kHz rises about 0.85 dB per dB of 20 log10(cond) (GR 3 -> 18 dB: H3 -62 -> -41 dBc) while any log-domain law with fixed
           linear ripple filtering gives a GR-independent modulation depth.
        After each stage: the drive of position 12 fitted 1-D on its static curve (held out; after A/B also 18 and 22), the
        100 Hz / 3 kHz / 8 kHz static curves at position 20, the burst envelopes and the harmonics as checks, and the
        fit/data/discriminate_opto.json checks rendered with the loop: (A) the 0.5 dB knee sweeps at positions 20 and 10 (knee level,
        width, slope above), (D) the no-GR gain against position at -70 and -50 dBFS (the law gives zero GR there; the bias is taken
        from the reference as a per-position trim and is reported as such, not reproduced), (E) the above-knee steps at position 20
        (-20 -> -10 -> -20, -10 -> 0 -> -10, -20 -> +5 -> -20 dBFS: 50 / 90 % attack and release times, the tail 0.1..2 s).

usage: cd <repo> && python3 -u fit/tools/candidates/opto-linear-db-feedback.py [--det peak|rms|both] [--stages ABCD] [--nfev N]
                                                                                 [--proxy target|cond] [--quick]
"""
import os, sys, time, json, argparse, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares, minimize_scalar
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, protocol, FS  # noqa: E402

cal = load_cal()
B2 = float(cal[MODEL.fields["o_b2"][0]]); B3 = float(cal[MODEL.fields["o_b3"][0]])
W0 = cal[MODEL.field("o_w")].copy(); TATT0 = cal[MODEL.field("o_tatt")].copy(); TREL0 = cal[MODEL.field("o_trel")].copy()
THRS_AB = [6, 10, 14, 20, 24]
THRS_C = [6, 10, 14, 18, 20, 22, 24]
LEVELS = list(range(-50, 15, 2))
BASE = {k: float(F[f"opto_static_t{k}_-50"]) for k in range(1, 25)}   # make-up (position 12) + the position's no-compression trim, dB
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
HARMS = ([f"opto_harm_t{t}_{l}_f1000" for t in (14, 18, 22) for l in (-20, -10, 0) if not (t == 14 and l == -20)]
         + ["opto_harm_t18_-10_f100", "opto_harm_t18_-10_f4000"])
HW3, HW5 = 0.25, 0.10          # stages C/D: weight of an H3 / H5 dB error against a static dB error
STATIC_FIT_SECS = 1.0          # the attacks are all under 10 ms: 1 s of sine is settled; the report re-renders the protocol's 2.5 s
PROXY = [0]                    # 0: the time-constant scaling follows the light (target), 1: the conductance


# ------------------------------------------------------------------------------------------------ the loop
@njit(cache=True)
def run_loop(x, n, fs, det, dta, dtr, k, wk, D, w, tatt, trel, alpha, proxy, b2, b3, y):
    """one channel of the feedback loop; y[:n] = divider output (before make-up)"""
    kA = 1.0 - np.exp(-1.0 / (dta * fs)); kR = 1.0 - np.exp(-1.0 / (dtr * fs))
    ra = np.empty(3); rr = np.empty(3); katt = np.empty(3); krel = np.empty(3)
    for j in range(3):
        ra[j] = 1.0 / (tatt[j] * fs); rr[j] = 1.0 / (trel[j] * fs)
        katt[j] = 1.0 - np.exp(-ra[j]); krel[j] = 1.0 - np.exp(-rr[j])
    st = np.zeros(3)
    env = 0.0; p = 0.0; cond = 0.0
    for i in range(n):
        xi = x[i]
        xa = xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        y[i] = v
        a = abs(v)
        if det == 0:
            if a > env: env += (a - env) * kA
            else: env += (a - env) * kR
        else:
            p += (a * a - p) * kA
            env = np.sqrt(p)
        S = 20.0 * np.log10(env + 1e-12)
        u = S + D
        if wk < 1e-6:
            sp = u if u > 0.0 else 0.0
        else:
            z = u / wk
            sp = u if z > 30.0 else (wk * np.log1p(np.exp(z)) if z > -30.0 else 0.0)
        target = 10.0 ** (k * sp / 20.0) - 1.0
        if alpha > 0.0:
            sc = (1.0 + (target if proxy == 0 else cond)) ** alpha
            for j in range(3):
                katt[j] = 1.0 - np.exp(-ra[j] * sc); krel[j] = 1.0 - np.exp(-rr[j] * sc)
        cond = 0.0
        for j in range(3):
            if target > st[j]: st[j] += (target - st[j]) * katt[j]
            else: st[j] += (target - st[j]) * krel[j]
            cond += w[j] * st[j]
    return y


@njit(cache=True, parallel=True)
def run_many(X, lens, fs, det, dta, dtr, k, wk, Ds, w, tatt, trel, alpha, proxy, b2, b3):
    m = X.shape[0]
    Y = np.zeros_like(X)
    for r in prange(m):
        run_loop(X[r], lens[r], fs, det, dta, dtr, k, wk, Ds[r], w, tatt, trel, alpha, proxy, b2, b3, Y[r])
    return Y


class P:
    """the loop's shape parameters (everything but the per-position drives)"""
    def __init__(self, det, k, wk, dta, dtr, w, tatt, trel, alpha=0.0):
        self.det, self.k, self.wk, self.dta, self.dtr, self.w, self.tatt, self.trel, self.alpha = det, k, wk, dta, dtr, np.asarray(w), np.asarray(tatt), np.asarray(trel), alpha

    def render(self, X, lens, Ds):
        return run_many(X, lens, float(FS), self.det, self.dta, self.dtr, self.k, self.wk, np.asarray(Ds, dtype=float), self.w, self.tatt, self.trel,
                        self.alpha, PROXY[0], B2, B3)

    def render1(self, x, D):
        y = np.zeros(len(x))
        run_loop(x, len(x), float(FS), self.det, self.dta, self.dtr, self.k, self.wk, D, self.w, self.tatt, self.trel, self.alpha, PROXY[0], B2, B3, y)
        return y

    def __str__(self):
        s = (f"k {self.k:.3f} (ratio {1 + self.k:.2f}:1, slope {self.k / (1 + self.k):.3f}) knee width {self.wk:.2f} dB | detector ta {self.dta * 1e3:.3f} ms "
             f"tr {self.dtr * 1e3:.2f} ms | states w {np.round(self.w, 3).tolist()} attack ms {np.round(self.tatt * 1e3, 2).tolist()} "
             f"release ms {np.round(self.trel * 1e3, 1).tolist()}")
        if self.alpha > 0: s += f" | alpha {self.alpha:.3f} ({'light' if PROXY[0] == 0 else 'cond'}-driven)"
        return s


# ------------------------------------------------------------------------------------------------ features (vectorised mirrors of protocol.feature)
def lockin(y, f, fs, n0, n1):
    t = np.arange(n0, n1) / fs
    return 2.0 * np.mean(y[n0:n1] * np.exp(-2j * np.pi * f * t))


def last_window(n, f, fs, last_s):
    per = fs / f; nper = int(round(last_s * f))
    n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
    return n0, n1


def gain_db_lockin(y, f, fs, level, last_s=0.5):
    n0, n1 = last_window(len(y), f, fs, last_s)
    return float(20 * np.log10(abs(lockin(y, f, fs, n0, n1)) + 1e-30) - level)


def harm_lockin(y, f, fs, level, last_s=1.0):
    """(gain_db, H3 dBc, H5 dBc) over the last window, as protocol.feature type harm"""
    n0, n1 = last_window(len(y), f, fs, last_s)
    c1 = abs(lockin(y, f, fs, n0, n1)); c3 = abs(lockin(y, 3 * f, fs, n0, n1)); c5 = abs(lockin(y, 5 * f, fs, n0, n1))
    return float(20 * np.log10(c1 + 1e-30) - level), float(20 * np.log10(c3 / c1 + 1e-30)), float(20 * np.log10(c5 / c1 + 1e-30))


def env_db(y, x, f, fs):
    """per-period gain in dB (lock-in over one period, hop one period), as protocol.feature type env"""
    per = int(round(fs / f)); m = len(y) // per
    t = np.arange(per) / fs; e = np.exp(-2j * np.pi * f * t)
    cy = np.abs((y[:m * per].reshape(m, per) * e).mean(axis=1)); cx = np.abs((x[:m * per].reshape(m, per) * e).mean(axis=1))
    return 20 * np.log10(cy / (cx + 1e-30) + 1e-30)


# ------------------------------------------------------------------------------------------------ the data sets
class StaticSet:
    def __init__(self, thrs, levels, secs):
        n = int(round(secs * FS)); t = np.arange(n) / FS
        self.thrs = list(thrs); rows = []
        self.X = np.empty((len(thrs) * len(levels), n)); r = 0
        for k in thrs:
            for l in levels:
                self.X[r] = 10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t); rows.append((k, l)); r += 1
        self.rows = rows
        self.lens = np.full(len(rows), n, dtype=np.int64)
        self.ref = np.array([F[f"opto_static_t{k}_{l}"] for k, l in rows])
        self.base = np.array([BASE[k] for k, l in rows]); self.level = np.array([float(l) for k, l in rows])
        self.idx = np.array([self.thrs.index(k) for k, l in rows])

    def gains(self, P_, Ds):
        Y = P_.render(self.X, self.lens, np.asarray(Ds)[self.idx])
        return np.array([gain_db_lockin(Y[r, :self.lens[r]], 1000.0, FS, self.level[r]) + self.base[r] for r in range(self.X.shape[0])])


BX_list = [protocol.stimulus(ITEMS[i]["stim"], FS)[0] for i in BURSTS]
BLEN = np.array([len(x) for x in BX_list], dtype=np.int64)
BX = np.zeros((len(BURSTS), int(BLEN.max())))
for r, x in enumerate(BX_list):
    BX[r, :len(x)] = x
BREF = [np.asarray(F[i]) for i in BURSTS]
BW = []
for ref in BREF:
    n = len(ref); BW.append(np.where(ref < BASE[20] - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0))

HX = np.array([protocol.stimulus(ITEMS[i]["stim"], FS)[0] for i in HARMS])
HLEN = np.full(len(HARMS), HX.shape[1], dtype=np.int64)
HTHR = [int(ITEMS[i]["set"]["optical_threshold"]) for i in HARMS]
HF = [float(ITEMS[i]["stim"]["f"]) for i in HARMS]; HLVL = [float(ITEMS[i]["stim"]["level"]) for i in HARMS]
HREF = [(F[i]["gain_db"], F[i]["h"][1], F[i]["h"][3]) for i in HARMS]


def burst_envs(P_, D20):
    Y = P_.render(BX, BLEN, np.full(len(BURSTS), D20))
    return [env_db(Y[r, :BLEN[r]], BX[r, :BLEN[r]], 1000.0, FS) + BASE[20] for r in range(len(BURSTS))]


def harms(P_, Dmap):
    """[(gain_db, H3, H5)] for HARMS; Dmap: threshold position -> drive"""
    Y = P_.render(HX, HLEN, [Dmap[t] for t in HTHR])
    out = []
    for r in range(len(HARMS)):
        g, h3, h5 = harm_lockin(Y[r], HF[r], FS, HLVL[r]); out.append((g + BASE[HTHR[r]], h3, h5))
    return out


# ------------------------------------------------------------------------------------------------ parameter packing
# p = [k, wk, D_thr..., dta, dtr, (w1 w2 w3, tatt1..3, trel1..3, (alpha))]
def unpack(p, det, nthr, states_fixed):
    k, wk = p[0], p[1]; Ds = np.array(p[2:2 + nthr]); dta, dtr = p[2 + nthr], p[3 + nthr]
    alpha = 0.0
    if states_fixed is not None:
        w, tatt, trel = states_fixed
    else:
        o = 4 + nthr
        w = np.abs(np.array(p[o:o + 3])); w = w / w.sum(); tatt = np.array(p[o + 3:o + 6]); trel = np.array(p[o + 6:o + 9])
        if len(p) > o + 9: alpha = float(p[o + 9])
    return P(det, k, wk, dta, dtr, w, tatt, trel, alpha), Ds


def resid_static(p, det, sset, states_fixed):
    P_, Ds = unpack(p, det, len(sset.thrs), states_fixed)
    return sset.gains(P_, Ds) - sset.ref


def resid_joint(p, det, sset, with_harm):
    P_, Ds = unpack(p, det, len(sset.thrs), None)
    out = [sset.gains(P_, Ds) - sset.ref]
    for e, ref, wt in zip(burst_envs(P_, Ds[sset.thrs.index(20)]), BREF, BW):
        n = min(len(e), len(ref)); out.append(wt[:n] * (e[:n] - ref[:n]))
    if with_harm:
        Dmap = {t: Ds[sset.thrs.index(t)] for t in sset.thrs}
        hr = []
        for (g, h3, h5), (rg, rh3, rh5) in zip(harms(P_, Dmap), HREF):
            hr += [g - rg, HW3 * (h3 - rh3), HW5 * (h5 - rh5)]
        out.append(np.array(hr))
    return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ reports
def report_static(P_, Ds, thrs, label):
    """the protocol's 2.5 s renders: rms / max per threshold"""
    s = StaticSet(thrs, LEVELS, 2.5)
    e = s.gains(P_, Ds) - s.ref
    print(f"  [{label}] static (2.5 s renders) all: rms {np.sqrt(np.mean(e ** 2)):.3f} dB max {np.max(np.abs(e)):.2f} dB")
    per = {}
    for j, kk in enumerate(thrs):
        sel = s.idx == j; per[kk] = (float(np.sqrt(np.mean(e[sel] ** 2))), float(np.max(np.abs(e[sel]))))
        print(f"     t{kk:2d}: rms {per[kk][0]:.3f} max {per[kk][1]:.2f} | resid by level: {np.round(e[sel], 2).tolist()}")
    print(f"  [{label}] per-threshold rms/max: " + ", ".join(f"t{kk} {per[kk][0]:.3f}/{per[kk][1]:.2f}" for kk in thrs))
    return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))), per


def fit_drive(P_, thr):
    """1-D fit of one position's drive on its static curve (shape fixed): held-out check"""
    s = StaticSet([thr], LEVELS, STATIC_FIT_SECS)
    cost = lambda D: float(np.sum((s.gains(P_, [D]) - s.ref) ** 2))
    r = minimize_scalar(cost, bounds=(-60.0, 80.0), method="bounded", options={"xatol": 1e-3})
    s2 = StaticSet([thr], LEVELS, 2.5); e = s2.gains(P_, [r.x]) - s2.ref
    return float(r.x), float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def report_bursts(P_, D20, D12):
    envs = burst_envs(P_, D20)
    allw = []; allp = []
    for iid, e, ref, wt in zip(BURSTS, envs, BREF, BW):
        n = min(len(e), len(ref)); d = e[:n] - ref[:n]; allw.append(wt[:n] * d); allp.append(d)
        it = ITEMS[iid]["stim"]; n0 = int(round(it["pre_s"] * 1000))
        if iid == "opto_pulses":
            print(f"  {iid}: plain rms {np.sqrt(np.mean(d ** 2)):.3f} max {np.max(np.abs(d)):.2f}")
        else:
            n1 = n0 + int(round(it["burst_s"] * 1000))
            pts = [n0, n0 + 1, n0 + 3, n0 + 10, n0 + 100, n1 - 1, n1 + 1, n1 + 10, n1 + 100, n1 + 500, n1 + 1000, min(n1 + 2500, n - 1)]
            print(f"  {iid}: plain rms {np.sqrt(np.mean(d ** 2)):.3f} max {np.max(np.abs(d)):.2f} | model/ref at burst+0,1,3,10,100 ms, end-1, end+1,10,100,500,1000,2500 ms:")
            print("      model", np.round(e[pts], 2).tolist()); print("      ref  ", np.round(ref[pts], 2).tolist())
    aw = np.concatenate(allw); ap = np.concatenate(allp)
    burst_rms = float(np.sqrt(np.mean(aw ** 2)))
    print(f"  bursts: weighted rms {burst_rms:.3f} (stage-4 weighting) | plain rms {np.sqrt(np.mean(ap ** 2)):.3f} dB max {np.max(np.abs(ap)):.2f}")
    it = ITEMS["opto_burst_thr12"]; x = protocol.stimulus(it["stim"], FS)[0]
    e = env_db(P_.render1(x, D12), x, 1000.0, FS) + BASE[12]; ref = np.asarray(F["opto_burst_thr12"]); n = min(len(e), len(ref)); d = e[:n] - ref[:n]
    print(f"  opto_burst_thr12 (drive of position 12 from its static curve): plain rms {np.sqrt(np.mean(d ** 2)):.3f} max {np.max(np.abs(d)):.2f}")
    return burst_rms


def report_freq(P_, D20):
    for f in (100, 3000, 8000):
        Lf = list(range(-40, 11, 4)); n = int(2.5 * FS); t = np.arange(n) / FS
        X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * f * t) for l in Lf]); lens = np.full(len(Lf), n, dtype=np.int64)
        Y = P_.render(X, lens, np.full(len(Lf), D20))
        g = np.array([gain_db_lockin(Y[r], float(f), FS, Lf[r]) + BASE[20] for r in range(len(Lf))])
        ref = np.array([F[f"opto_static_f{f}_{l}"] for l in Lf]); d = g - ref
        print(f"  f{f} static at t20 (not fitted): rms {np.sqrt(np.mean(d ** 2)):.3f} max {np.max(np.abs(d)):.2f} | resid {np.round(d, 2).tolist()}")


def report_harm(P_, Dmap):
    notes = []; e3 = []; e5 = []
    for iid, (g, h3, h5), (rg, rh3, rh5) in zip(HARMS, harms(P_, Dmap), HREF):
        s = f"{iid}: H3 model {h3:.1f} ref {rh3:.1f} | H5 model {h5:.1f} ref {rh5:.1f} | gain {g:.2f}/{rg:.2f}"
        print("  " + s); notes.append(s); e3.append(h3 - rh3); e5.append(h5 - rh5)
    print(f"  harmonics: H3 error rms {np.sqrt(np.mean(np.square(e3))):.1f} dB (max {np.max(np.abs(e3)):.1f}), H5 error rms {np.sqrt(np.mean(np.square(e5))):.1f} dB (max {np.max(np.abs(e5)):.1f})")
    return notes


DISC = json.load(open(os.path.join(HERE, "..", "..", "data", "discriminate_opto.json")))


def rms_db(a):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(a))) + 1e-20))


def step_signal(a, b, secs=(2.0, 2.0, 2.0), f=1000.0):
    """fit/measure/discriminate_opto.py's step stimulus"""
    t = np.arange(int(round(sum(secs) * FS))) / FS
    env = np.full_like(t, 10 ** (a / 20)); n1 = int(secs[0] * FS); n2 = n1 + int(secs[1] * FS)
    env[n1:n2] = 10 ** (b / 20)
    return env * np.sin(2 * np.pi * f * t)


def per_cycle_gain(x, y, f=1000.0):
    """discriminate_opto.py's per-cycle RMS gain, dB"""
    per = int(round(FS / f)); m = len(x) // per
    xr = np.sqrt(np.mean(x[:m * per].reshape(m, per) ** 2, axis=1)); yr = np.sqrt(np.mean(y[:m * per].reshape(m, per) ** 2, axis=1))
    return 20 * np.log10(yr / xr)


def t_to(frac, seg, g_start, g_end):
    target = g_start + frac * (g_end - g_start)
    for i, v in enumerate(seg):
        if (g_end < g_start and v <= target) or (g_end > g_start and v >= target):
            return i
    return None


def report_discriminate(P_, Dmap, label):
    """the discriminate_opto.py measurements (A), (D), (E) rendered with the loop and compared with the reference's numbers"""
    out = {}
    # (A) fine knee sweeps: 3 s sines, RMS gain over the last second, as the reference script measured them
    for thr in (20, 10):
        kd = DISC["knee"][str(thr)]; lv = np.array(kd["levels"]); gref = np.array(kd["gain_db"])
        n = int(3.0 * FS); t = np.arange(n) / FS
        X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
        Y = P_.render(X, lens, np.full(len(lv), Dmap[thr]))
        g = np.array([rms_db(Y[r, -FS:]) - rms_db(X[r, -FS:]) for r in range(len(lv))]) + BASE[thr]
        grm = g[0] - g; grr = gref[0] - gref; e = g - gref
        cm = [float(np.interp(v, grm, lv)) for v in (1.0, 3.0, 6.0)]; cr = [float(np.interp(v, grr, lv)) for v in (1.0, 3.0, 6.0)]
        sm = (g[-1] - g[-5]) / (lv[-1] - lv[-5]); sr = (gref[-1] - gref[-5]) / (lv[-1] - lv[-5])
        # onset: first level with GR > 0.1 dB
        om = float(lv[np.argmax(grm > 0.1)]); orf = float(lv[np.argmax(grr > 0.1)])
        print(f"  [{label}] (A) knee sweep t{thr}: GR > 0.1 dB from {om:.1f} dBFS (ref {orf:.1f}); GR crosses 1/3/6 dB at "
              f"{cm[0]:.2f}/{cm[1]:.2f}/{cm[2]:.2f} dBFS (ref {cr[0]:.2f}/{cr[1]:.2f}/{cr[2]:.2f}); slope over the top 2 dB {sm:.3f} dB/dB (ref {sr:.3f}); "
              f"gain error rms {np.sqrt(np.mean(e ** 2)):.3f} max {np.max(np.abs(e)):.2f} dB")
        print("      GR per 0.5 dB model:", np.round(grm, 2).tolist()); print("      GR per 0.5 dB ref  :", np.round(grr, 2).tolist())
        out[f"knee{thr}"] = (cm, cr, float(sm), float(sr), float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))))
    # (D) the no-GR gain against position: the law's own GR at -70 / -50 dBFS, and the trim the harness imports from the reference
    n = int(2.0 * FS); t = np.arange(n) / FS
    ks = [k for k in (12, 20, 22, 24) if k in Dmap]
    for lvl in (-70.0, -50.0):
        X = np.array([10 ** (lvl / 20) * np.sin(2 * np.pi * 1000.0 * t) for k in ks]); lens = np.full(len(ks), n, dtype=np.int64)
        Y = P_.render(X, lens, np.array([Dmap[k] for k in ks]))
        gr = {k: round(-(rms_db(Y[r, -FS:]) - rms_db(X[r, -FS:])), 4) for r, k in enumerate(ks)}
        print(f"  [{label}] (D) {lvl:.0f} dBFS, the loop's own GR by position (dB, no trim): {gr}")
    trims = {k: round(BASE[k] - BASE[1], 3) for k in ks}
    print(f"  [{label}] (D) the reference's no-GR loss against position 1 (dB): {trims} is NOT produced by the loop (the hard/soft-knee "
          f"law gives zero light below the knee at every level); the harness applies it as a per-position trim read from the reference")
    print("      trim / A^2 against the fitted drives (A = 10^(D/20)): " + ", ".join(f"t{k} {trims[k] / 10 ** (Dmap[k] / 10):+.2e}" for k in ks))
    # (E) above-knee steps at position 20
    out["steps"] = {}
    for key, sd in DISC["steps"].items():
        a, b = sd["from"], sd["to"]; ref = np.array(sd["per_cycle_gain_db"])
        x = step_signal(a, b); gc = per_cycle_gain(x, P_.render1(x, Dmap[20])) + BASE[20]
        n1, n2 = 2000, 4000
        rows = []
        for nm, g_ in (("model", gc), ("ref", ref)):
            up = g_[n1:n2]; down = g_[n2:]
            ga = float(np.mean(g_[n1 - 200:n1 - 1])); gb = float(np.mean(g_[n2 - 200:n2 - 1])); ga2 = float(np.mean(g_[-200:]))
            rows.append((nm, ga, gb, ga2, t_to(0.5, up, ga, gb), t_to(0.9, up, ga, gb), t_to(0.5, down, gb, ga2), t_to(0.9, down, gb, ga2), down))
        m, r = rows
        tail = m[8][100:2000] - r[8][100:2000]; d = gc[:len(ref)] - ref[:len(gc)]
        print(f"  [{label}] (E) step {a:+.0f} -> {b:+.0f} -> {a:+.0f}: gain model {m[1]:+.2f} -> {m[2]:+.2f} -> {m[3]:+.2f} (ref {r[1]:+.2f} -> {r[2]:+.2f} -> {r[3]:+.2f}); "
              f"attack 50/90 % model {m[4]}/{m[5]} ms (ref {r[4]}/{r[5]}); release 50/90 % model {m[6]}/{m[7]} ms (ref {r[6]}/{r[7]}); "
              f"tail 0.1..2 s |model - ref| max {np.max(np.abs(tail)):.3f} dB; whole trajectory rms {np.sqrt(np.mean(d ** 2)):.3f} dB")
        print("      attack first 40 ms (per 2 ms) model:", np.round(gc[n1:n1 + 40:2], 2).tolist())
        print("      attack first 40 ms (per 2 ms) ref  :", np.round(ref[n1:n1 + 40:2], 2).tolist())
        print("      release first 100 ms (per 5 ms) model:", np.round(m[8][:100:5], 2).tolist())
        print("      release first 100 ms (per 5 ms) ref  :", np.round(r[8][:100:5], 2).tolist())
        print("      release 0.1..2 s (per 100 ms) model:", np.round(m[8][100:2000:100], 2).tolist())
        print("      release 0.1..2 s (per 100 ms) ref  :", np.round(r[8][100:2000:100], 2).tolist())
        out["steps"][key] = dict(model=(m[4], m[5], m[6], m[7]), ref=(r[4], r[5], r[6], r[7]), tail_max=float(np.max(np.abs(tail))), rms=float(np.sqrt(np.mean(d ** 2))))
    return out


def closed_form_floor():
    """what a pure linear-in-dB closed loop (no dynamics, no ripple) can do: per threshold, fit slope a and knee T to the points
    below base - 0.3 dB; report the rms. Slope a = k / (1 + k)."""
    print("closed-form floor (gain = base - a * max(0, L - T), hard knee), per threshold:")
    for kk in THRS_C + [12]:
        ref = np.array([F[f"opto_static_t{kk}_{l}"] for l in LEVELS]); L = np.array(LEVELS, dtype=float)
        sel = ref < BASE[kk] - 0.3
        A = np.vstack([L[sel], np.ones(sel.sum())]).T; a, b = np.linalg.lstsq(A, ref[sel], rcond=None)[0]
        T = (BASE[kk] - b) / a
        pred = BASE[kk] + np.minimum(0.0, a * (L - T)); e = pred - ref
        print(f"   t{kk:2d}: slope {-a:.3f} dB/dB (k = {-a / (1 + a):.2f}, ratio {1 / (1 + a):.2f}:1) knee {T:.1f} dBFS | rms {np.sqrt(np.mean(e ** 2)):.3f} max {np.max(np.abs(e)):.2f}")


def trim_note():
    print("no-compression trim (gain at -50 dBFS minus position 1's) against the stage-4 drive A (linear): trim_dB / A^2")
    for kk in (10, 14, 18, 20, 22, 24):
        A = 10 ** (cal[MODEL.fields["o_thr_db"][0] + kk - 1] / 20); tr = BASE[kk] - BASE[1]
        print(f"   t{kk:2d}: trim {tr:+.2f} dB, A {A:.1f}, trim/A^2 {tr / A / A:+.2e}")
    print("reference H3 at 1 kHz against gain reduction (dBc; GR = -(gain - no-GR gain)):")
    rows = sorted((-(F[i]["gain_db"] - BASE[int(ITEMS[i]["set"]["optical_threshold"])]), F[i]["h"][1], i) for i in HARMS if i.endswith("f1000"))
    print("   " + ", ".join(f"GR {g:.1f}: {h:.1f}" for g, h, _ in rows))
    g = np.array([r[0] for r in rows]); h = np.array([r[1] for r in rows])
    print(f"   H3 against 20 log10(cond): slope {np.polyfit(20 * np.log10(10 ** (g / 20) - 1), h, 1)[0]:.2f} dB/dB")


def dstr(Dmap):
    return {t: round(float(Dmap[t]), 2) for t in sorted(Dmap)}


def run(det, nfev, stages):
    name = "peak one-pole on |v| (attack/release)" if det == 0 else "RMS averager (one-pole on v^2; tr unused)"
    print(f"\n=== detector ({'i' if det == 0 else 'ii'}): {name} ===")
    t0 = time.time(); res = dict(det=det)
    sAB = StaticSet(THRS_AB, LEVELS, STATIC_FIT_SECS)
    knee0 = {6: 7.3, 10: 15.9, 12: 19.0, 14: 20.9, 18: 24.8, 20: 26.8, 22: 28.7, 24: 30.7}   # closed-form knees, -dBFS
    p0 = [1.65, 1.0] + [knee0[t] for t in THRS_AB] + [0.5e-3 if det == 0 else 3e-3, 5e-3]
    nt = len(THRS_AB)
    lo = [0.5, 0.01] + [-60] * nt + [2e-5, 1e-4]; hi = [5.0, 15.0] + [80] * nt + [0.05, 1.0]; xs = [0.05, 0.2] + [1] * nt + [1e-4, 1e-3]
    fixed = (W0, TATT0, TREL0)
    # ---- stage A: statics, states fixed
    r = least_squares(resid_static, p0, bounds=(lo, hi), args=(det, sAB, fixed), x_scale=xs, diff_step=2e-3, max_nfev=nfev)
    PA, Ds = unpack(r.x, det, nt, fixed)
    print(f"  stage A ({time.time() - t0:.0f} s, nfev {r.nfev}): {PA} | drives {dstr(dict(zip(THRS_AB, Ds)))}")
    print(f"  stage A static (1 s fit renders): rms {np.sqrt(np.mean(r.fun ** 2)):.3f} dB max {np.max(np.abs(r.fun)):.2f}")
    res["rmsA"], res["maxA"], _ = report_static(PA, Ds, THRS_AB, "A")
    DA = dict(zip(THRS_AB, Ds))
    for thr in (12, 18, 22):
        D, rms_, max_ = fit_drive(PA, thr); DA[thr] = D
        print(f"  [A] held-out t{thr}: drive {D:.2f} dB | static rms {rms_:.3f} max {max_:.2f}")
    print("  [A] harmonics (states = stage 4, not fitted):"); report_harm(PA, DA)
    res["discA"] = report_discriminate(PA, DA, "A")
    if "B" not in stages:
        return res
    # ---- stage B: statics + bursts, states free
    t1 = time.time()
    loB = lo + [0.01] * 3 + [5e-5] * 3 + [1e-3] * 3; hiB = hi + [1.0] * 3 + [0.2] * 3 + [5.0] * 3; xsB = xs + [0.05] * 3 + [1e-3] * 3 + [0.01, 0.02, 0.1]
    pB0 = np.clip(list(r.x) + list(W0) + list(TATT0) + list(TREL0), loB, hiB)
    rB = least_squares(resid_joint, pB0, bounds=(loB, hiB), args=(det, sAB, False), x_scale=xsB, diff_step=2e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
    PB, Ds = unpack(rB.x, det, nt, None); ns = len(sAB.ref)
    print(f"  stage B ({time.time() - t1:.0f} s, nfev {rB.nfev}): {PB} | drives {dstr(dict(zip(THRS_AB, Ds)))}")
    print(f"  stage B static (1 s fit renders): rms {np.sqrt(np.mean(rB.fun[:ns] ** 2)):.3f} dB max {np.max(np.abs(rB.fun[:ns])):.2f} | bursts weighted rms {np.sqrt(np.mean(rB.fun[ns:] ** 2)):.3f}")
    res["rmsB"], res["maxB"], res["perB"] = report_static(PB, Ds, THRS_AB, "B")
    DB = dict(zip(THRS_AB, Ds))
    for thr in (12, 18, 22):
        D, rms_, max_ = fit_drive(PB, thr); DB[thr] = D
        print(f"  [B] held-out t{thr}: drive {D:.2f} dB | static rms {rms_:.3f} max {max_:.2f}")
    print("  [B] drive table (dB):", dstr(DB), "| steps 20->22->24:", round(DB[22] - DB[20], 2), round(DB[24] - DB[22], 2))
    res["burstB"] = report_bursts(PB, DB[20], DB[12])
    report_freq(PB, DB[20])
    print("  [B] harmonics (not fitted):"); res["harmB"] = report_harm(PB, DB)
    res["discB"] = report_discriminate(PB, DB, "B")
    res.update(PB=PB, DB=DB)
    if "C" not in stages:
        print(f"  total {time.time() - t0:.0f} s"); return res
    # ---- stage C: statics (7 positions) + bursts + harmonics
    sC = StaticSet(THRS_C, LEVELS, STATIC_FIT_SECS); ntc = len(THRS_C)
    loC = [0.5, 0.01] + [-60] * ntc + [2e-5, 1e-4] + [0.01] * 3 + [5e-5] * 3 + [1e-3] * 3
    hiC = [5.0, 15.0] + [80] * ntc + [0.05, 1.0] + [1.0] * 3 + [0.2] * 3 + [5.0] * 3
    xsC = [0.05, 0.2] + [1] * ntc + [1e-4, 1e-3] + [0.05] * 3 + [1e-3] * 3 + [0.01, 0.02, 0.1]
    pC0 = np.clip([PB.k, PB.wk] + [DB[t] for t in THRS_C] + [PB.dta, PB.dtr] + list(PB.w) + list(PB.tatt) + list(PB.trel), loC, hiC)
    prev = pC0
    for stage in [s for s in "CD" if s in stages]:
        t2 = time.time()
        if stage == "D":
            p_0 = list(prev) + [0.5]; lo_ = loC + [0.0]; hi_ = hiC + [3.0]; xs_ = xsC + [0.05]
        else:
            p_0, lo_, hi_, xs_ = prev, loC, hiC, xsC
        rS = least_squares(resid_joint, p_0, bounds=(lo_, hi_), args=(det, sC, True), x_scale=xs_, diff_step=2e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
        PS, Ds = unpack(rS.x, det, ntc, None); ns = len(sC.ref); nh = 3 * len(HARMS)
        print(f"  stage {stage} ({time.time() - t2:.0f} s, nfev {rS.nfev}): {PS} | drives {dstr(dict(zip(THRS_C, Ds)))}")
        print(f"  stage {stage} static (1 s fit renders): rms {np.sqrt(np.mean(rS.fun[:ns] ** 2)):.3f} dB max {np.max(np.abs(rS.fun[:ns])):.2f} | "
              f"bursts weighted rms {np.sqrt(np.mean(rS.fun[ns:-nh] ** 2)):.3f}")
        res[f"rms{stage}"], res[f"max{stage}"], res[f"per{stage}"] = report_static(PS, Ds, THRS_C, stage)
        DS = dict(zip(THRS_C, Ds))
        D12, rms12, max12 = fit_drive(PS, 12); DS[12] = D12
        print(f"  [{stage}] held-out t12: drive {D12:.2f} dB | static rms {rms12:.3f} max {max12:.2f}")
        print(f"  [{stage}] drive table (dB):", dstr(DS))
        res[f"burst{stage}"] = report_bursts(PS, DS[20], DS[12])
        report_freq(PS, DS[20])
        print(f"  [{stage}] harmonics (fitted, H3 weight {HW3} / H5 weight {HW5} per dB):"); res[f"harm{stage}"] = report_harm(PS, DS)
        res[f"disc{stage}"] = report_discriminate(PS, DS, stage)
        res.update({f"P{stage}": PS, f"D{stage}": DS})
        prev = list(rS.x) if stage == "C" else prev
    print(f"  total {time.time() - t0:.0f} s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="both", choices=["peak", "rms", "both"])
    ap.add_argument("--stages", default="ABCD")
    ap.add_argument("--nfev", type=int, default=60)
    ap.add_argument("--proxy", default="target", choices=["target", "cond"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    nfev = 3 if a.quick else a.nfev
    PROXY[0] = 0 if a.proxy == "target" else 1
    # sanity: the vectorised features mirror protocol.feature
    it = ITEMS["opto_blen_0.2"]; x = protocol.stimulus(it["stim"], FS)[0]
    e1 = env_db(x * 0.5, x, 1000.0, FS); e2 = np.asarray(protocol.feature(it, np.stack([x * 0.5, x * 0.5]), FS))
    assert np.max(np.abs(e1[:len(e2)] - e2)) < 1e-3, "env mirror mismatch"
    it = ITEMS[HARMS[0]]; x = protocol.stimulus(it["stim"], FS)[0]; yy = x * (1 + 0.01 * np.cos(4 * np.pi * 1000.0 * np.arange(len(x)) / FS))
    fh = protocol.feature(it, np.stack([yy, yy]), FS); g, h3, h5 = harm_lockin(yy, 1000.0, FS, float(it["stim"]["level"]))
    assert abs(fh["h"][1] - h3) < 0.05 and abs(fh["gain_db"] - g) < 1e-3, "harm mirror mismatch"
    print(f"stage amplifier b2 {B2:.3e} b3 {B3:.3e} | stage-4 states w {np.round(W0, 3).tolist()} attack ms {np.round(TATT0 * 1e3, 2).tolist()} release ms {np.round(TREL0 * 1e3, 1).tolist()}")
    print(f"reference no-GR gain by position (dB at -50 dBFS): { {t: round(BASE[t], 2) for t in (1, 6, 10, 12, 14, 18, 20, 22, 24)} }")
    print(f"8 kHz against 1 kHz at -40 dBFS, position 20 (audio-path HF loss with no GR): {F['opto_static_f8000_-40'] - F['opto_static_t20_-40']:+.3f} dB")
    closed_form_floor()
    trim_note()
    out = []
    for det in ([0] if a.det == "peak" else [1] if a.det == "rms" else [0, 1]):
        out.append(run(det, nfev, a.stages))
    print("\n=== summary ===")
    for o in out:
        s = f"  det {'peak' if o['det'] == 0 else 'rms '}: static A rms {o['rmsA']:.3f}/max {o['maxA']:.2f}"
        for st in "BCD":
            if f"P{st}" in o:
                s += (f" | {st}: k {o[f'P{st}'].k:.3f} knee w {o[f'P{st}'].wk:.2f} dB, static rms {o[f'rms{st}']:.3f}/max {o[f'max{st}']:.2f}, "
                      f"bursts weighted rms {o[f'burst{st}']:.3f}")
        print(s)
        for st in "ABCD":
            if f"disc{st}" in o:
                dd = o[f"disc{st}"]; k20 = dd["knee20"]
                print(f"      {st}: knee t20 1/3/6 dB at {k20[0][0]:.2f}/{k20[0][1]:.2f}/{k20[0][2]:.2f} (ref {k20[1][0]:.2f}/{k20[1][1]:.2f}/{k20[1][2]:.2f}), slope {k20[2]:.3f} (ref {k20[3]:.3f}); "
                      + "steps 50/90 att, 50/90 rel ms: " + "; ".join(f"{kk} model {v['model']} ref {v['ref']} tail max {v['tail_max']:.2f}" for kk, v in dd["steps"].items()))


if __name__ == "__main__":
    main()
