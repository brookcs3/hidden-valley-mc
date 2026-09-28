# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical ripple, direction (a), key "opto-ripple-a-pre-smooth": does smoothing the sidechain DRIVE before the panel's hard turn-on
(rather than the light after it) give the reference's ripple harmonics without softening the static knee?

Hypothesis. The light of the fitted stage (src/dsp/Opto.hpp, fit/stages/stage4_opto.py) is a pulse train: the rectified sine minus the
turn-on o_vth, clipped at zero, so its 4f component passes through zero as the conduction duty cycle changes with level, and the
model's H5 has nulls the reference does not have. If the EL panel is driven through a source resistance into its own capacitance, or
through a detector, the turn-on acts on a smoothed drive: the pulse has no sharp edges, the null fills in, and the static mean light
(hence the knee and the 0.62 dB/dB slope) is unchanged as long as the smoothing does not reach the knee. Variants tested, each placed
between the rectifier and the turn-on unless stated:
  rc        one-pole on |s| after the rectifier (the panel as an RC), time constant tau; output normalised by pi/2 so a sine of
            amplitude d still reads d in the mean
  peak      peak detector: rise time constant tau_r (down to instant), exponential fall tau_f
  rms       one-pole on s^2 then square root (an RMS-like averager), normalised by sqrt 2
  pre       one-pole on s BEFORE the rectifier (a linear pole in the sidechain, a control: it should fail the HF rows)
  slew      peak detector whose fall is a constant slew S (drive units per second): the relative ripple falls with level
  rclevel   the rc variant with a level-dependent time constant tau = tau0 (drive / o_vth)^(-kappa)
For each variant only its own parameters are fitted, plus a global drive offset dA (dB, added to every o_thr_db entry: the knee must
not move at 1 kHz), the persistence o_tau_el (whose job overlaps) and the panel exponent o_n: the calibration's n was fitted for the
pulse-train light of the instantaneous turn-on (whose mean grows as (d - vth)^(n + 1/2) near the knee), and a turn-on acting on a
smoothed drive has a different mean-light law, so freezing n would refute the smoothing on the statics alone. A control ("none")
refits dA, tau_el and n with no smoother, so each variant's gain is read against it. Everything else is the calibration in
fit/data/constants.json at the start of the run. The fit is on the static series of positions 14/18/20/22 (weight 2 per dB) and the 27 opto_harm_t* items
(gain weight 2, H3 1, H5 0.5, H7 0.25; the feat_residual floor rule). The report for every variant: H3/H5/H7 error rms by
frequency, the gain error, statics at t20 and the 100 Hz / 3 kHz / 8 kHz series, the fine knee (A) at t20 and t10, the nine burst
envelopes (stage-4 weights), the HF rows and the no-GR rows. The mirror is stage 4's run_one with the detector options added, and is
checked against the C++ engine first.

usage: cd <repo> && python3 -u fit/tools/candidates/opto-ripple-pre-smooth.py [--quick] [--only rc,peak,...]
"""
import json, os, sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS, DATA  # noqa: E402

np.set_printoptions(linewidth=200, suppress=True)
FSF = float(FS)
LEVELS = list(range(-50, 15, 2))
FIT_POS = (14, 18, 20, 22)
HARM_IDS = [f"opto_harm_t{t}_{l}_f{f}" for t in (14, 18, 22) for l in (-20, -10, 0) for f in (100, 1000, 4000)]
BURSTS = [f"opto_burst_{l}" for l in (-26, -18, -10, -2)] + [f"opto_blen_{b}" for b in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
DISC = json.load(open(os.path.join(DATA, "discriminate_opto.json")))
DETS = {"none": 0, "rc": 1, "peak": 2, "rms": 3, "pre": 4, "slew": 5, "rclevel": 6}
QUICK = "--quick" in sys.argv
ONLY = None
for a in sys.argv:
    if a.startswith("--only"): ONLY = sys.argv[sys.argv.index(a) + 1].split(",")

cal = load_cal()
def cidx(name, i=0): return MODEL.fields[name][0] + i
def cget(name, i=0): return float(cal[cidx(name, i)])
G0 = cget("o_gain_db", 11) + cget("x_gain_db", 0); G0_LIN = 10.0 ** (G0 / 20.0)


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def lp_matched(fc, Q, fs):
    w0 = 2.0 * np.pi * fc / fs; q = 1.0 / (2.0 * Q)
    if q <= 1.0: a1 = -2.0 * np.exp(-q * w0) * np.cos(np.sqrt(1.0 - q * q) * w0)
    else: a1 = -2.0 * np.exp(-q * w0) * np.cosh(np.sqrt(q * q - 1.0) * w0)
    a2 = np.exp(-2.0 * q * w0)
    A0 = (1.0 + a1 + a2) ** 2; A1 = (1.0 - a1 + a2) ** 2; A2 = -4.0 * a2
    p1 = np.sin(w0 / 2.0) ** 2; p0 = 1.0 - p1; p2 = 4.0 * p0 * p1
    R1 = (A0 * p0 + A1 * p1 + A2 * p2) * Q * Q
    B0 = A0; B1 = (R1 - B0 * p0) / p1
    b0 = 0.5 * (np.sqrt(B0) + np.sqrt(B1)); b1 = np.sqrt(B0) - b0
    return b0, b1, 0.0, a1, a2


@njit(cache=True)
def run_one(x, n, fs, A, vth, nexp, gam, tau_el, w, tatt, trel, L0, mu, b2, b3, fc, Q, det, p1, p2, out):
    """stage 4's mirror of OptoStage::process plus a drive smoother between the rectifier and the turn-on (det, p1, p2)"""
    kEl = 1.0 - np.exp(-1.0 / (tau_el * fs))
    kA0 = 1.0 - np.exp(-1.0 / (tatt[0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (tatt[1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (tatt[2] * fs))
    r0 = 1.0 / (trel[0] * fs); r1 = 1.0 / (trel[1] * fs); r2 = 1.0 / (trel[2] * fs)
    lb0, lb1, lb2, la1, la2 = lp_matched(fc, Q, fs)
    c0 = L0 ** gam
    k1 = 1.0 - np.exp(-1.0 / (p1 * fs))          # rc / rms / pre / peak rise
    k2 = 1.0 - np.exp(-1.0 / (p2 * fs)) if p2 > 0.0 else 1.0   # peak fall
    slew = p2 / fs                                # slew fall, drive units per sample
    normRc = np.pi / 2.0; normRms = np.sqrt(2.0)
    z1 = 0.0; z2 = 0.0; L = 0.0; s0 = c0; s1 = c0; s2 = c0; cond = c0
    ds = 0.0; ms = 0.0; yl = 0.0
    for i in range(n):
        xi = x[i]
        xa = xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        out[i] = v
        y = lb0 * v + z1
        z1 = lb1 * v - la1 * y + z2
        z2 = lb2 * v - la2 * y
        if det == 4:
            yl += (y - yl) * k1
            d = A * abs(yl); dd = d
        else:
            d = A * abs(y); dd = d
        if det == 1:
            ds += (d - ds) * k1; dd = ds * normRc
        elif det == 2:
            if d > ds: ds += (d - ds) * k1
            else: ds += (d - ds) * k2
            dd = ds
        elif det == 3:
            ms += (d * d - ms) * k1; dd = np.sqrt(ms) * normRms
        elif det == 5:
            if d > ds: ds += (d - ds) * k1
            else:
                ds -= slew
                if ds < d: ds = d
            dd = ds
        elif det == 6:
            lvl = ds * normRc / vth
            if lvl < 0.1: lvl = 0.1
            tau = p1 * lvl ** (-p2)
            kk = 1.0 - np.exp(-1.0 / (tau * fs))
            ds += (d - ds) * kk; dd = ds * normRc
        e = dd - vth
        Linst = e ** nexp if e > 0.0 else 0.0
        L += (Linst - L) * kEl
        target = (L + L0) ** gam
        q0 = 1.0 - np.exp(-(1.0 + mu * s0) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2) * r2)
        s0 += (target - s0) * (kA0 if target > s0 else q0)
        s1 += (target - s1) * (kA1 if target > s1 else q1)
        s2 += (target - s2) * (kA2 if target > s2 else q2)
        cond = w[0] * s0 + w[1] * s1 + w[2] * s2


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, vth, nexp, gam, tau_el, w, tatt, trel, L0, mu, b2, b3, fc, Q, det, p1, p2, out):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], vth, nexp, gam, tau_el, w, tatt, trel, L0[j], mu, b2, b3, fc, Q, det, p1, p2, out[j])


@njit(cache=True, parallel=True)
def gain_batch(Y, lens, fs, f, last_s, out):
    """gain of the fundamental (dB re unit amplitude) over the last last_s seconds, whole periods, as protocol 'gain_db'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; per = fs / f
        nper = max(1, int(round(last_s * f)))
        n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
        re = 0.0; im = 0.0
        for i in range(n0, n1):
            t = i / fs
            re += Y[j, i] * np.cos(2.0 * np.pi * f * t); im -= Y[j, i] * np.sin(2.0 * np.pi * f * t)
        out[j] = 20.0 * np.log10(2.0 * np.sqrt(re * re + im * im) / (n1 - n0) + 1e-30)


class Shape:
    """the calibration's optical fields plus the candidate's detector and its offsets"""
    def __init__(self):
        self.b2, self.b3 = cget("o_b2"), cget("o_b3")
        self.thr = np.array(cal[MODEL.field("o_thr_db")], dtype=float).copy()
        self.n, self.gam, self.vth, self.tau_el = cget("o_n"), cget("o_gamma"), cget("o_vth"), cget("o_tau_el")
        self.w = np.array([cget("o_w", i) for i in range(3)]); self.tatt = np.array([cget("o_tatt", i) for i in range(3)]); self.trel = np.array([cget("o_trel", i) for i in range(3)])
        self.mu, self.leak, self.leak_q = cget("o_rel_mu"), cget("o_leak"), cget("o_leak_q")
        self.fc, self.Q = cget("o_sc_lp_hz"), cget("o_sc_lp_q")
        self.det, self.p1, self.p2, self.dA = 0, 1e-4, 0.0, 0.0

    def copy(self):
        s = Shape.__new__(Shape); s.__dict__.update(self.__dict__); return s
    def A(self, k): return 10.0 ** ((self.thr[k - 1] + self.dA) / 20.0)
    def cond0(self, k): return self.leak * 10.0 ** (self.leak_q * (self.thr[k - 1] - self.thr[19]) / 20.0)
    def leak_light(self, k):
        c0 = self.cond0(k); return c0 ** (1.0 / self.gam) if c0 > 0.0 else 0.0

    def render(self, X, lens, thrs, fs=FSF):
        A = np.array([self.A(k) for k in thrs]); L0 = np.array([self.leak_light(k) for k in thrs]); Y = np.zeros_like(X)
        run_batch(X, lens, fs, A, self.vth, self.n, self.gam, self.tau_el, self.w, self.tatt, self.trel, L0, self.mu, self.b2, self.b3,
                  self.fc, self.Q, self.det, self.p1, self.p2, Y)
        return Y


class Batch:
    def __init__(self, ids):
        self.ids = ids
        xs = [protocol.stimulus(ITEMS[i]["stim"], ITEMS[i]["fs"])[0] for i in ids]
        self.lens = np.array([len(x) for x in xs]); self.X = np.zeros((len(ids), self.lens.max()))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.thr = [int(ITEMS[i]["set"]["optical_threshold"]) for i in ids]
        self.fs = ITEMS[ids[0]]["fs"]

    def feats(self, sh):
        Y = sh.render(self.X, self.lens, self.thr, float(self.fs))
        return [protocol.feature(ITEMS[i], np.stack([Y[j, :self.lens[j]] * G0_LIN] * 2), self.fs) for j, i in enumerate(self.ids)]


def sines(levels, f=1000.0, secs=2.5, fs=FS):
    n = int(round(secs * fs)); t = np.arange(n) / fs
    return np.array([10 ** (l / 20) * np.sin(2 * np.pi * f * t) for l in levels]), np.full(len(levels), n, dtype=np.int64)


def static_gains(sh, thrs, levels, f=1000.0, secs=2.5):
    X, lens = sines(levels, f, secs); Y = sh.render(X, lens, thrs)
    g = np.zeros(len(levels)); gain_batch(Y, lens, FSF, f, 0.5, g)
    return g - np.array(levels, dtype=float) + G0


def rms(e): return float(np.sqrt(np.mean(np.square(np.asarray(e, dtype=float))))) if len(e) else float("nan")


# ------------------------------------------------------------------------------------------------ residuals and report
def harm_resid(iid, m, wg=2.0, wts=(1.0, 0.5, 0.25)):
    """gain and H3/H5/H7 residuals with feat_residual's floor rule (-120 floor, x0.3 where the reference is below -110 dBc)"""
    ref = F[iid]; out = [wg * (m["gain_db"] - ref["gain_db"])]
    for idx, wt in zip((1, 3, 5), wts):
        a, b = m["h"][idx], ref["h"][idx]
        if a is None or b is None: continue
        a, b = max(a, -120.0), max(b, -120.0)
        out.append(wt * (a - b) * (0.3 if b < -110 else 1.0))
    return out


STAT = Batch([f"opto_static_t{k}_{l}" for k in FIT_POS for l in (LEVELS[::2] if QUICK else LEVELS)])
STAT_REF = np.array([F[i] for i in STAT.ids])
HARM = Batch(HARM_IDS)


def resid(sh):
    out = [2.0 * (np.array(STAT.feats(sh)) - STAT_REF)]
    for i, m in zip(HARM_IDS, HARM.feats(sh)): out.append(np.array(harm_resid(i, m)))
    return np.concatenate(out)


def lp2q(f, fc, Q):
    """analog second-order low pass magnitude, dB"""
    w = f / fc
    return -10.0 * np.log10((1 - w * w) ** 2 + (w / Q) ** 2)


def harm_table(sh, label, verbose=True):
    """H3/H5/H7 model - reference over the 27 items, by frequency; items with a reference below -100 dBc are excluded from the rms.
    r = o_vth / (A |LP(f)| v^) is the turn-on relative to the divider output amplitude the model settles at: the light of the plain
    hard turn-on is the pulse train max(0, sin - r), whose 4f / 2f ratio nulls at r = 0.42 and whose 6f / 2f nulls at 0.28 and 0.75"""
    err = {f: {3: [], 5: [], 7: []} for f in (100, 1000, 4000)}; gerr = []; rows = []
    for iid, m in zip(HARM_IDS, HARM.feats(sh)):
        ref = F[iid]; f = int(ITEMS[iid]["stim"]["f"]); gerr.append(m["gain_db"] - ref["gain_db"])
        cells = []
        for hk, idx in ((3, 1), (5, 3), (7, 5)):
            a, b = m["h"][idx], ref["h"][idx]
            if a is None or b is None: cells.append("   n/a      "); continue
            cells.append(f"{a:6.1f}/{b:6.1f}")
            if b > -100.0: err[f][hk].append(a - b)
        k = int(ITEMS[iid]["set"]["optical_threshold"]); lvl = ITEMS[iid]["stim"]["level"]
        vhat = 10.0 ** ((lvl + m["gain_db"] - G0) / 20.0); lpm = 10.0 ** (lp2q(float(f), sh.fc, sh.Q) / 20.0)
        rr = sh.vth / (sh.A(k) * lpm * vhat)
        rows.append(f"   {iid:24s} gain {m['gain_db']:+6.2f}/{ref['gain_db']:+6.2f} | H3 {cells[0]} | H5 {cells[1]} | H7 {cells[2]} | r {rr:.2f}")
    if verbose:
        print(f"   harmonics under GR [{label}], model / ref dBc:"); print("\n".join(rows))
    summ = {}
    for f in (100, 1000, 4000):
        summ[f] = tuple(rms(err[f][k]) for k in (3, 5, 7))
        print(f"   {f:5d} Hz: H3 err rms {summ[f][0]:.1f} max {np.max(np.abs(err[f][3])):.1f} | H5 rms {summ[f][1]:.1f} max {np.max(np.abs(err[f][5])):.1f} | "
              + (f"H7 rms {summ[f][2]:.1f} max {np.max(np.abs(err[f][7])):.1f}" if err[f][7] else "H7 n/a") + f"   (n = {len(err[f][3])}/{len(err[f][5])}/{len(err[f][7])})")
    allh = {k: rms(err[100][k] + err[1000][k] + err[4000][k]) for k in (3, 5, 7)}
    print(f"   all 27: H3 rms {allh[3]:.1f} | H5 rms {allh[5]:.1f} | H7 rms {allh[7]:.1f} | gain err rms {rms(gerr):.3f} max {np.max(np.abs(gerr)):.2f} dB")
    return summ, allh, rms(gerr)


def envelope(sh, iid):
    it = ITEMS[iid]; b = Batch([iid]); return np.asarray(b.feats(sh)[0])


def bursts(sh):
    allw = []; allp = []; lines = []
    for iid in BURSTS + ["opto_burst_thr12"]:
        e = envelope(sh, iid); ref = np.asarray(F[iid]); n = min(len(e), len(ref)); d = e[:n] - ref[:n]
        wt = np.where(ref[:n] < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)
        if iid in BURSTS: allw.append(wt * d); allp.append(d)
        lines.append(f"{iid} {rms(d):.3f}/{np.max(np.abs(d)):.2f}")
    aw = np.concatenate(allw); ap = np.concatenate(allp)
    print(f"   bursts (nine items): weighted rms {rms(aw):.3f} max {np.max(np.abs(aw)):.2f} | plain rms {rms(ap):.3f} max {np.max(np.abs(ap)):.2f} | per item rms/max: " + ", ".join(lines))
    return rms(aw), rms(ap)


def fine_knee(sh):
    out = {}
    n = int(3.0 * FS); t = np.arange(n) / FS
    for thr in (20, 10):
        kd = DISC["knee"][str(thr)]; lv = np.array(kd["levels"]); gref = np.array(kd["gain_db"])
        X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
        Y = sh.render(X, lens, [thr] * len(lv))
        g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(lv))]) + G0
        grm = g[0] - g; grr = gref[0] - gref; e = g - gref
        cm = [float(np.interp(v, grm, lv)) for v in (0.5, 1.0, 3.0)]; cr = [float(np.interp(v, grr, lv)) for v in (0.5, 1.0, 3.0)]
        sm = (g[-1] - g[-5]) / (lv[-1] - lv[-5]); sr = (gref[-1] - gref[-5]) / (lv[-1] - lv[-5])
        i0 = int(np.argmax(grr > 0.005)) - 1
        print(f"   (A) t{thr}: rms {rms(e):.3f} max {np.max(np.abs(e)):.2f} | GR 0.5/1/3 dB at {cm[0]:.2f}/{cm[1]:.2f}/{cm[2]:.2f} (ref {cr[0]:.2f}/{cr[1]:.2f}/{cr[2]:.2f}), "
              f"width 0.5->3 {cm[2] - cm[0]:.2f} (ref {cr[2] - cr[0]:.2f}), top slope {sm:.3f} (ref {sr:.3f}) | GR per 0.5 dB from {lv[i0]:.1f}: model {np.round(grm[i0:i0 + 7], 2).tolist()} ref {np.round(grr[i0:i0 + 7], 2).tolist()}")
        out[thr] = (rms(e), float(np.max(np.abs(e))), cm[2] - cm[0])
    return out


def statics_report(sh):
    out = {}
    for k in (20, 12):
        ref = np.array([F[f"opto_static_t{k}_{l}"] for l in LEVELS]); e = static_gains(sh, [k] * len(LEVELS), LEVELS) - ref
        out[k] = (rms(e), float(np.max(np.abs(e))))
    for f in (100, 3000, 8000):
        lv = list(range(-40, 11, 2)); ref = np.array([F[f"opto_static_f{f}_{l}"] for l in lv])
        e = static_gains(sh, [20] * len(lv), lv, f=float(f)) - ref; out[f] = (rms(e), float(np.max(np.abs(e))))
    print(f"   statics t20 rms {out[20][0]:.3f} max {out[20][1]:.2f} | t12 {out[12][0]:.3f}/{out[12][1]:.2f} | at t20: 100 Hz {out[100][0]:.3f}/{out[100][1]:.2f}, "
          f"3 kHz {out[3000][0]:.3f}/{out[3000][1]:.2f}, 8 kHz {out[8000][0]:.3f}/{out[8000][1]:.2f} (mirror: no Nickel HF shelf)")
    # HF rows and the no-GR rows
    fr = np.array(DISC["hf"]["freqs"]); n = int(3.0 * FS); t = np.arange(n) / FS; hf = {}
    for lvl in (-40.0, -10.0, 0.0):
        ref = np.array(DISC["hf"]["rows"][str(lvl)])
        X = np.array([10 ** (lvl / 20) * np.sin(2 * np.pi * f * t) for f in fr]); lens = np.full(len(fr), n, dtype=np.int64)
        Y = sh.render(X, lens, [20] * len(fr))
        g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(fr))]) + G0
        hf[lvl] = g - ref
    print("   HF rows model-ref (1/2/4/6/8/10/12/16 kHz): " + " | ".join(f"{l:+.0f} dBFS {np.round(hf[l], 2).tolist()} rms {rms(hf[l]):.3f}" for l in hf))
    n2 = int(2.0 * FS); t2 = np.arange(n2) / FS
    ks = [int(k) for k in DISC["nogr"]["-50.0"]]; ref = np.array([DISC["nogr"]["-50.0"][str(k)] for k in ks])
    X = np.array([10 ** (-50.0 / 20) * np.sin(2 * np.pi * 1000.0 * t2) for k in ks]); lens = np.full(len(ks), n2, dtype=np.int64)
    Y = sh.render(X, lens, ks)
    g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(ks))]) + G0
    print(f"   (D) no-GR at -50 dBFS, positions {ks}: model-ref {np.round(g - ref, 3).tolist()} max {np.max(np.abs(g - ref)):.3f}")
    out["hf"] = {l: rms(hf[l]) for l in hf}; out["nogr"] = float(np.max(np.abs(g - ref)))
    return out


def full_report(sh, label):
    print(f"\n== {label}: det {sh.det} p1 {sh.p1:.3e} p2 {sh.p2:.3e} dA {sh.dA:+.3f} dB tau_el {sh.tau_el * 1e3:.4f} ms n {sh.n:.4f}")
    summ, allh, gerr = harm_table(sh, label)
    st = statics_report(sh); kn = fine_knee(sh); bw, bp = bursts(sh)
    print(f"   SUMMARY {label:10s}| H3 rms 100/1k/4k {summ[100][0]:.1f}/{summ[1000][0]:.1f}/{summ[4000][0]:.1f} | H5 {summ[100][1]:.1f}/{summ[1000][1]:.1f}/{summ[4000][1]:.1f} | "
          f"H7 {summ[100][2]:.1f}/{summ[1000][2]:.1f} | all27 H3 {allh[3]:.2f} H5 {allh[5]:.2f} H7 {allh[7]:.2f} gain {gerr:.3f} | t20 {st[20][0]:.3f} | 100 Hz {st[100][0]:.3f} "
          f"3k {st[3000][0]:.3f} 8k {st[8000][0]:.3f} | knee t20 {kn[20][0]:.3f} (w {kn[20][2]:.2f}) t10 {kn[10][0]:.3f} | bursts {bw:.3f} ({bp:.3f}) | HF -10 {st['hf'][-10.0]:.3f} 0 {st['hf'][0.0]:.3f}")
    return dict(h=summ, allh=allh, gain=gerr, st=st, knee=kn, bursts=(bw, bp))


# ------------------------------------------------------------------------------------------------ the fits
# per variant: (parameters after dA and log10 tau_el) name, log10 lower, log10 upper, grid starts (log10)
CANDS = {
    "none":    [],
    "rc":      [("log10 tau", -5.0, -1.5, [-4.3, -4.0, -3.7, -3.4, -3.1, -2.5, -2.0])],
    "peak":    [("log10 tau_rise", -6.3, -3.0, [-6.0, -5.0, -4.3]), ("log10 tau_fall", -4.7, -1.5, [-4.0, -3.5, -3.0, -2.3])],
    "rms":     [("log10 tau", -5.0, -1.5, [-4.3, -4.0, -3.7, -3.4, -3.1, -2.5, -2.0])],
    "pre":     [("log10 tau", -5.5, -3.0, [-4.6, -4.0, -3.5])],
    "slew":    [("log10 tau_rise", -6.3, -3.0, [-6.0, -4.5]), ("log10 slew", 2.5, 5.5, [3.3, 3.8, 4.3, 4.8])],
    "rclevel": [("log10 tau0", -5.0, -1.5, [-4.3, -3.7, -3.1, -2.5]), ("kappa", -1.5, 1.5, [-1.0, 0.0, 1.0])],
}


# the smoothing forced to a value that engages at 1 kHz (about a quarter cycle), with only dA and tau_el fitted: the residuals
# of the hypothesis as stated, whatever the free fit prefers
FORCED = {"rc": (-3.6,), "peak": (-6.0, -3.3), "rms": (-3.6,), "pre": (-4.0,), "slew": (-6.0, 4.0), "rclevel": (-3.6, 1.0)}


def fit_forced(sh0, name, t0):
    spec = CANDS[name]; fixed = FORCED[name]
    f = lambda q: resid(apply(sh0, name, np.array([q[0], q[1], q[2]] + list(fixed))))
    r = least_squares(f, [0.0, np.log10(sh0.tau_el), np.log10(sh0.n)], bounds=([-6.0, -6.0, -0.3], [6.0, -3.0, 0.5]), x_scale=[0.5, 0.2, 0.05], diff_step=1e-2, max_nfev=4 if QUICK else 12)
    print(f"   {name} forced at " + ", ".join(f"{b[0]} {v}" for b, v in zip(spec, fixed)) + f": cost {r.cost:.2f}, dA {r.x[0]:+.3f}, log10 tau_el {r.x[1]:.3f}, n {10 ** r.x[2]:.3f}  [{time.time() - t0:.0f} s]")
    return apply(sh0, name, np.array(list(r.x) + list(fixed))), r.cost


NFIX = 3   # dA, log10 tau_el, log10 n precede the variant's own parameters


def apply(sh0, name, p):
    s = sh0.copy(); s.det = DETS[name]; s.dA = float(p[0]); s.tau_el = 10.0 ** p[1]; s.n = 10.0 ** p[2]
    spec = CANDS[name]
    if len(spec) > 0: s.p1 = 10.0 ** p[NFIX]
    if len(spec) > 1: s.p2 = p[NFIX + 1] if spec[1][0] == "kappa" else 10.0 ** p[NFIX + 1]
    return s


def fit_candidate(sh0, name, t0):
    spec = CANDS[name]
    lo = [-6.0, -6.0, -0.3] + [b[1] for b in spec]; hi = [6.0, -3.0, 0.5] + [b[2] for b in spec]
    xs = [0.5, 0.2, 0.05] + [0.2] * len(spec)
    import itertools
    grid = list(itertools.product(*[b[3] for b in spec])) if spec else [()]
    if QUICK: grid = grid[::max(1, len(grid) // 3)]
    print(f"\n-- fitting {name}: parameters dA, log10 tau_el, log10 n, " + ", ".join(b[0] for b in spec) + f"; {len(grid)} grid starts")
    best = None
    for g in grid:
        p0 = np.array([0.0, np.log10(sh0.tau_el), np.log10(sh0.n)] + list(g))
        c0 = 0.5 * np.sum(resid(apply(sh0, name, p0)) ** 2)
        r = least_squares(lambda p: resid(apply(sh0, name, p)), p0, bounds=(lo, hi), x_scale=xs, diff_step=2e-2, max_nfev=6 if QUICK else 10)
        print(f"   start {np.round(g, 2).tolist()}: cost {c0:.2f} (dA 0) -> {r.cost:.2f} at x {np.round(r.x, 3).tolist()}  [{time.time() - t0:.0f} s]")
        if best is None or r.cost < best.cost: best = r
    r = least_squares(lambda p: resid(apply(sh0, name, p)), best.x, bounds=(lo, hi), x_scale=xs, diff_step=5e-3, max_nfev=8 if QUICK else 30, xtol=1e-8, ftol=1e-6)
    if r.cost < best.cost: best = r
    print(f"   {name} best: cost {best.cost:.2f}, " + ", ".join(f"{nm} {v:.3f}" for nm, v in zip(["dA", "log10 tau_el", "log10 n"] + [b[0] for b in spec], best.x)) + f" (n {10 ** best.x[2]:.3f})  [{time.time() - t0:.0f} s]")
    return apply(sh0, name, best.x), best.cost


# ------------------------------------------------------------------------------------------------ main
def validate(sh):
    chk = ["opto_static_t20_-10", "opto_static_t10_-20", "opto_burst_-10", "opto_harm_t18_-10_f1000", "opto_harm_t22_0_f100"]
    worst = 0.0
    for i in chk:
        m = Batch([i]).feats(sh)[0]; e = render_item(ITEMS[i], cal)
        if isinstance(m, dict): d = max(abs(m["gain_db"] - e["gain_db"]), abs(m["h"][1] - e["h"][1]), abs(m["h"][3] - e["h"][3]))
        elif isinstance(m, list): d = float(np.max(np.abs(np.array(m) - np.array(e)[:len(m)])))
        else: d = abs(m - e)
        worst = max(worst, d); print(f"   mirror vs engine {i}: max |diff| {d:.4f} dB")
    return worst


def main():
    t0 = time.time()
    sh = Shape()
    print(f"calibration snapshot: vth {sh.vth:.3f} n {sh.n:.3f} gamma {sh.gam:.3f} tau_el {sh.tau_el * 1e3:.4f} ms w {np.round(sh.w, 3).tolist()} tatt ms {np.round(sh.tatt * 1e3, 1).tolist()} "
          f"trel ms {np.round(sh.trel * 1e3, 1).tolist()} mu {sh.mu:.2f} leak {sh.leak:.4f} q {sh.leak_q:.3f} LP {sh.fc:.0f} Hz Q {sh.Q:.3f} b2 {sh.b2:.2e} b3 {sh.b3:.2e} G0 {G0:.3f}")
    print("1. mirror against the C++ engine (current constants.json)")
    w = validate(sh)
    if w > 0.05: print(f"!! mirror differs from the engine by {w:.3f} dB; the mirror is what is reported below")
    tt = time.time(); r0 = resid(sh); print(f"   one objective evaluation: {len(r0)} residuals, {time.time() - tt:.2f} s, cost {0.5 * np.sum(r0 ** 2):.2f}  [{time.time() - t0:.0f} s]")
    results = {}
    results["baseline"] = full_report(sh, "baseline")
    names = ONLY or ["none", "rc", "peak", "rms", "slew", "rclevel", "pre"]
    for name in names:
        if name in FORCED:
            s, cost = fit_forced(sh, name, t0)
            results[name + "*"] = full_report(s, name + "* (forced)"); results[name + "*"]["cost"] = cost; results[name + "*"]["shape"] = s
        s, cost = fit_candidate(sh, name, t0)
        results[name] = full_report(s, name); results[name]["cost"] = cost; results[name]["shape"] = s
    print("\n== summary (H rms in dB by frequency; statics/knee/bursts in dB)")
    print(f"   {'variant':10s} {'cost':>7s} | {'H3 100/1k/4k':>16s} | {'H5 100/1k/4k':>16s} | {'H7 100/1k':>10s} | {'gain':>5s} | {'t20':>5s} {'100Hz':>5s} {'3k':>5s} | {'kneeT20':>7s} {'kneeT10':>7s} | {'bursts':>6s} | {'HF-10':>5s} {'HF0':>5s} | parameters")
    for name, r in results.items():
        s = r.get("shape"); h = r["h"]
        par = "" if s is None else f"dA {s.dA:+.2f} tau_el {s.tau_el * 1e3:.3f} ms n {s.n:.3f} p1 {s.p1:.3e} p2 {s.p2:.3e}"
        print(f"   {name:10s} {r.get('cost', 0.5 * np.sum(r0 ** 2)):7.2f} | {h[100][0]:4.1f}/{h[1000][0]:4.1f}/{h[4000][0]:4.1f}     | {h[100][1]:4.1f}/{h[1000][1]:4.1f}/{h[4000][1]:4.1f}     | {h[100][2]:4.1f}/{h[1000][2]:4.1f}  | {r['gain']:5.3f} | "
              f"{r['st'][20][0]:5.3f} {r['st'][100][0]:5.3f} {r['st'][3000][0]:5.3f} | {r['knee'][20][0]:7.3f} {r['knee'][10][0]:7.3f} | {r['bursts'][0]:6.3f} | {r['st']['hf'][-10.0]:5.3f} {r['st']['hf'][0.0]:5.3f} | {par}")
    print(f"   total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
