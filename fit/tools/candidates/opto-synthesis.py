# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical stage synthesis check, key "synthesis": the model chosen in docs/opto-fix.md, evaluated on everything the document states.

The model is the feedback loop of src/dsp/Opto.hpp with the "physical-law-hard-turnon" V6 shape (hard turn-on with the exponent on
the cell, a light leak proportional to a power of the sidechain drive, a self-quenched (bimolecular) release) plus the second-order
sidechain low pass found by "sidechain-shape-and-taper". Nothing about the shape is fitted here: the V6 raw vector is taken as
reported (and reproduced by its verifier), and only (a) one knee level per threshold position (the shift family, the allowed form)
and (b) the low pass corner and Q are fitted, because no harness had produced those with this shape.

  1. the mirror is validated against the C++ engine with the stage-4 calibration (statics, an envelope, a harmonic item);
  2. the V6 shape with one free knee per position on opto_static_t{1..24}_*: the knee table, its C++ o_thr_db equivalents, the
     per-position residuals, and the no-GR gain law (D) under the V6 leak (b 0.0202, q 2.154) and under the direct fit of the
     no-GR rows (b 0.0209, q 2.08);
  3. the sidechain low pass: opto_static_f{100,3000,8000}_* and the HF rows of discriminate_opto.json without a low pass, with the
     reported (5147 Hz, Q 0.718), and with fc, Q refitted on the 3 kHz + 8 kHz static series;
  4. with the low pass in place: the burst family, opto_burst_thr12, sr_opto_{44100,96000}, the 27 under-GR harmonic items
     (H3/H5/H7), the fine knee sweeps (A), the no-GR rows (D) at -70 and -50 dBFS, the above-knee steps (E), the extended rows (B).

usage: cd <repo> && python3 -u fit/tools/candidates/opto-synthesis.py
"""
import json, os, sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares, minimize_scalar
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402

np.set_printoptions(linewidth=200, suppress=True)
DISC = json.load(open(os.path.join(HERE, "..", "..", "data", "discriminate_opto.json")))
cal = load_cal()
def cget(name, i=0): return float(cal[MODEL.fields[name][0] + i])
B2, B3 = cget("o_b2"), cget("o_b3")
G0 = cget("o_gain_db", 11) + cget("x_gain_db", 0)     # make-up at position 12 plus the Nickel path's midband gain, dB
LEVELS = list(range(-50, 15, 2))
FSF = float(FS)

# the V6 raw vector as reported by opto-physical-law-hard-turnon.py and reproduced by its verifier (/tmp/opto_hard_turnon_verify.log)
V6 = {"knee_6": -8.2877, "knee_10": -16.909123, "knee_14": -21.769433, "knee_18": -25.552051, "knee_20": -27.498282, "knee_22": -29.6237,
      "knee_24": -31.649124, "logC": 0.804515, "p": 1.657263, "log_tau_el": -4.177286, "w0": 0.29804, "w1": 0.787815, "w2": 0.249402,
      "la0": -2.073795, "la1": -1.803165, "la2": -2.068737, "lr0": -0.756595, "lr1": -0.879338, "lr2": -1.027291, "logb": -1.694579,
      "q": 2.154234, "logmuc": 0.885802}
LP_REPORTED = (5147.0, 0.718)      # sidechain-shape-and-taper, from the HF rows


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def lp_coeffs(fc, Q, fs, kind):
    """second-order low pass (b0, b1, b2, a1, a2), direct form II transposed. kind 0: RBJ bilinear (matched at fc only; the bilinear
    warp adds attenuation toward Nyquist); kind 1: Vicanek's magnitude-matched design (poles by impulse invariance, zeros matched
    at DC, fc and Nyquist), which follows the analog curve up to Nyquist"""
    w0 = 2.0 * np.pi * fc / fs
    if kind == 0:
        cw = np.cos(w0); al = np.sin(w0) / (2.0 * Q); a0 = 1.0 + al
        return (1.0 - cw) / 2.0 / a0, (1.0 - cw) / a0, (1.0 - cw) / 2.0 / a0, -2.0 * cw / a0, (1.0 - al) / a0
    q = 1.0 / (2.0 * Q)
    if q <= 1.0:
        a1 = -2.0 * np.exp(-q * w0) * np.cos(np.sqrt(1.0 - q * q) * w0)
    else:
        a1 = -2.0 * np.exp(-q * w0) * np.cosh(np.sqrt(q * q - 1.0) * w0)
    a2 = np.exp(-2.0 * q * w0)
    A0 = (1.0 + a1 + a2) ** 2; A1 = (1.0 - a1 + a2) ** 2; A2 = -4.0 * a2
    p1 = np.sin(w0 / 2.0) ** 2; p0 = 1.0 - p1; p2 = 4.0 * p0 * p1
    R1 = (A0 * p0 + A1 * p1 + A2 * p2) * Q * Q
    B0 = A0; B1 = (R1 - B0 * p0) / p1
    b0 = 0.5 * (np.sqrt(B0) + np.sqrt(B1)); b1 = np.sqrt(B0) - b0
    return b0, b1, 0.0, a1, a2


@njit(cache=True)
def run_one(x, n, fs, A, C, nexp, gam, tau_el, w, tatt, trel, cond0, muc, b2, b3, lp, fc, Q, lpk, out):
    """one channel of the loop: divider v = xa / (1 + cond), sidechain [low pass] -> d = A |v| -> e = d - 1 -> light e^nexp ->
    persistence tau_el -> target = C (L + l0)^gam with l0 the leak -> three states, release rate x (1 + muc s_i); out = v"""
    kEl = 1.0 - np.exp(-1.0 / (tau_el * fs)) if tau_el > 0.0 else 1.0
    kA0 = 1.0 - np.exp(-1.0 / (tatt[0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (tatt[1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (tatt[2] * fs))
    kR0 = 1.0 - np.exp(-1.0 / (trel[0] * fs)); kR1 = 1.0 - np.exp(-1.0 / (trel[1] * fs)); kR2 = 1.0 - np.exp(-1.0 / (trel[2] * fs))
    l0 = (cond0 / C) ** (1.0 / gam) if cond0 > 0.0 else 0.0
    lb0, lb1, lb2, la1, la2 = lp_coeffs(fc, Q, fs, lpk)
    z1 = 0.0; z2 = 0.0
    L = 0.0; s0 = 0.0; s1 = 0.0; s2 = 0.0; cond = 0.0
    for i in range(n):
        xi = x[i]
        xa = xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        out[i] = v
        u = v
        if lp == 1:
            y = lb0 * u + z1
            z1 = lb1 * u - la1 * y + z2
            z2 = lb2 * u - la2 * y
            u = y
        d = A * abs(u)
        e = d - 1.0
        Linst = e ** nexp if e > 0.0 else 0.0
        L += (Linst - L) * kEl
        target = C * (L + l0) ** gam
        if muc > 0.0:
            q0 = 1.0 - np.exp(-(1.0 + muc * s0) / (trel[0] * fs)); q1 = 1.0 - np.exp(-(1.0 + muc * s1) / (trel[1] * fs)); q2 = 1.0 - np.exp(-(1.0 + muc * s2) / (trel[2] * fs))
        else:
            q0 = kR0; q1 = kR1; q2 = kR2
        s0 += (target - s0) * (kA0 if target > s0 else q0)
        s1 += (target - s1) * (kA1 if target > s1 else q1)
        s2 += (target - s2) * (kA2 if target > s2 else q2)
        cond = w[0] * s0 + w[1] * s1 + w[2] * s2


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, C, nexp, gam, tau_el, w, tatt, trel, cond0, muc, b2, b3, lp, fc, Q, lpk, out):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], C, nexp, gam, tau_el, w, tatt, trel, cond0[j], muc, b2, b3, lp, fc, Q, lpk, out[j])


@njit(cache=True, parallel=True)
def lockin_batch(Y, lens, fs, f, last_s, out):
    """gain of the fundamental (dB re unit amplitude) over the last `last_s` seconds, whole periods, as protocol.feature 'gain_db'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; per = fs / f
        nper = max(1, int(round(last_s * f)))
        n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
        re = 0.0; im = 0.0
        for i in range(n0, n1):
            t = i / fs
            re += Y[j, i] * np.cos(2.0 * np.pi * f * t); im -= Y[j, i] * np.sin(2.0 * np.pi * f * t)
        m = n1 - n0
        out[j] = 20.0 * np.log10(2.0 * np.sqrt(re * re + im * im) / m + 1e-30)


@njit(cache=True, parallel=True)
def env_batch(Y, X, lens, fs, f, out, counts):
    """gain of the fundamental per period (lock-in of output over input, hop one period), dB, as protocol.feature 'env'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; per = fs / f; m = int(n / per); counts[j] = m
        for k in range(m):
            n0 = int(round(k * per)); n1 = int(round((k + 1) * per))
            yr = 0.0; yi = 0.0; xr = 0.0; xi = 0.0
            for i in range(n0, n1):
                cs = np.cos(2.0 * np.pi * f * i / fs); sn = np.sin(2.0 * np.pi * f * i / fs)
                yr += Y[j, i] * cs; yi -= Y[j, i] * sn; xr += X[j, i] * cs; xi -= X[j, i] * sn
            out[j, k] = 20.0 * np.log10(np.sqrt(yr * yr + yi * yi) / (np.sqrt(xr * xr + xi * xi) + 1e-30) + 1e-30)


class Model:
    """the shape (everything but the per-position knee) plus the knee table; renders batches"""
    def __init__(self, d, lp=False, fc=LP_REPORTED[0], Q=LP_REPORTED[1], b=None, q=None):
        self.C = 10.0 ** d["logC"]; self.p = d["p"]; self.tau_el = 10.0 ** d["log_tau_el"]
        w = np.abs(np.array([d["w0"], d["w1"], d["w2"]])); self.w = w / w.sum()
        self.tatt = 10.0 ** np.array([d["la0"], d["la1"], d["la2"]]); self.trel = 10.0 ** np.array([d["lr0"], d["lr1"], d["lr2"]])
        self.b = 10.0 ** d["logb"] if b is None else b; self.q = d["q"] if q is None else q
        self.muc = 10.0 ** d["logmuc"]
        self.knee = {k: d[f"knee_{k}"] for k in (6, 10, 14, 18, 20, 22, 24)}
        self.lp, self.fc, self.Q = lp, fc, Q; self.lpk = 1
        self.nexp, self.gam = 1.0, self.p

    def A(self, k): return 10.0 ** (-self.knee[k] / 20.0)
    def cond0(self, k): return self.b * (self.A(k) / self.A(20)) ** self.q
    def vth(self): return self.C ** (1.0 / self.p)
    def thr_db(self, k): return 20.0 * np.log10(self.vth()) - self.knee[k]

    def render(self, X, lens, thrs, fs=FSF, Y=None):
        A = np.array([self.A(k) for k in thrs]); c0 = np.array([self.cond0(k) for k in thrs])
        if Y is None: Y = np.zeros_like(X)
        run_batch(X, lens, fs, A, self.C, self.nexp, self.gam, self.tau_el, self.w, self.tatt, self.trel, c0, self.muc, B2, B3,
                  1 if self.lp else 0, self.fc, self.Q, self.lpk, Y)
        return Y


def sines(levels, f=1000.0, secs=2.5, fs=FS):
    n = int(round(secs * fs)); t = np.arange(n) / fs
    X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * f * t) for l in levels])
    return X, np.full(len(levels), n, dtype=np.int64)


def static_gains(m, thrs, levels, f=1000.0, secs=2.5):
    X, lens = sines(levels, f, secs); Y = m.render(X, lens, thrs)
    g = np.zeros(len(levels)); lockin_batch(Y, lens, FSF, f, 0.5, g)
    return g - np.array(levels, dtype=float) + G0


def rms(e): return float(np.sqrt(np.mean(np.square(e))))


# ------------------------------------------------------------------------------------------------ 1. validation against the C++ engine
def validate():
    n, gam, vth, tau_el = cget("o_n"), cget("o_gamma"), cget("o_vth"), cget("o_tau_el")
    thr_db = cal[MODEL.field("o_thr_db")]
    m = Model(V6); m.nexp, m.gam = n, gam; m.C = vth ** (n * gam); m.tau_el = tau_el; m.muc = 0.0; m.b = 0.0
    m.w = np.array([cget("o_w", i) for i in range(3)]); m.tatt = np.array([cget("o_tatt", i) for i in range(3)]); m.trel = np.array([cget("o_trel", i) for i in range(3)])
    m.knee = {k: -(thr_db[k - 1] - 20 * np.log10(vth)) for k in range(1, 25)}
    ids = [f"opto_static_t20_{l}" for l in LEVELS]
    cpp = np.array([float(render_item(ITEMS[i], cal)) for i in ids]); mine = static_gains(m, [20] * len(LEVELS), LEVELS)
    print(f"1. mirror vs C++ engine (stage-4 calibration): statics t20 max |diff| {np.max(np.abs(mine - cpp)):.4f} dB", end="")
    it = ITEMS["opto_burst_-10"]; x = protocol.stimulus(it["stim"], FS)[0]
    X = x[None, :].copy(); lens = np.array([len(x)]); Y = m.render(X, lens, [20]); E = np.zeros((1, len(x) // 48 + 2)); cnt = np.zeros(1, dtype=np.int64)
    env_batch(Y, X, lens, FSF, 1000.0, E, cnt); e_cpp = np.asarray(render_item(it, cal)); e_m = E[0, :cnt[0]] + G0
    print(f" | opto_burst_-10 envelope max |diff| {np.max(np.abs(e_m - e_cpp[:len(e_m)])):.4f} dB", end="")
    it = ITEMS["opto_harm_t18_-10_f1000"]; x = protocol.stimulus(it["stim"], FS)[0]
    Y = m.render(x[None, :].copy(), np.array([len(x)]), [18]); y = Y[0] * 10 ** (G0 / 20)
    hm = protocol.feature(it, np.stack([y, y]), FS); hc = render_item(it, cal)
    print(f" | opto_harm_t18_-10_f1000 H3 {hm['h'][1]:.1f}/{hc['h'][1]:.1f} H5 {hm['h'][3]:.1f}/{hc['h'][3]:.1f} (mirror/C++)")


# ------------------------------------------------------------------------------------------------ 2. the knee table
def fit_knees(m, label):
    print(f"\n2. V6 shape, one free knee per position ({label}); fit on 1.5 s renders, report on the protocol's 2.5 s")
    print("   pos  knee dBFS  o_thr_db | all 33 levels rms / max | compressing (GR>0.3) n, rms | no-GR gain model / ref (dB) | GR at +14 model / ref")
    rows = {}
    for k in range(1, 25):
        ref = np.array([F[f"opto_static_t{k}_{l}"] for l in LEVELS])
        def resid(knee, secs):
            m.knee[k] = knee
            return static_gains(m, [k] * len(LEVELS), LEVELS, secs=secs) - ref
        r = minimize_scalar(lambda kn: float(np.sum(resid(kn, 1.5) ** 2)), bounds=(-45.0, 40.0), method="bounded", options={"xatol": 1e-3})
        m.knee[k] = r.x; e = resid(r.x, 2.5)
        gr = ref[0] - ref; comp = gr > 0.3
        g0m = static_gains(m, [k], [-50.0])[0]
        rows[k] = dict(knee=r.x, thr=m.thr_db(k), rms=rms(e), mx=float(np.max(np.abs(e))), nc=int(comp.sum()), crms=rms(e[comp]) if comp.any() else float("nan"),
                       g0m=g0m, g0r=ref[0], gr14m=g0m - (e[-1] + ref[-1]), gr14r=ref[0] - ref[-1], e=e)
        rw = rows[k]
        print(f"   {k:3d}  {rw['knee']:8.2f}  {rw['thr']:8.2f} | {rw['rms']:.3f} / {rw['mx']:.2f} | {rw['nc']:2d}, {rw['crms']:.3f} | {rw['g0m']:+.3f} / {rw['g0r']:+.3f} | {rw['gr14m']:.2f} / {rw['gr14r']:.2f}")
    ks = np.arange(1, 25); knees = np.array([rows[k]["knee"] for k in ks])
    print("   knee dBFS 1..24:", np.round(knees, 2).tolist())
    print("   o_thr_db  1..24:", np.round([rows[k]["thr"] for k in ks], 2).tolist())
    print("   steps of o_thr_db (dB):", np.round(np.diff([rows[k]["thr"] for k in ks]), 2).tolist())
    allE = np.concatenate([rows[k]["e"] for k in ks]); compE = np.concatenate([rows[k]["e"][(F_gr(k) > 0.3)] for k in ks])
    fitted = [6, 10, 14, 18, 20, 22, 24]; unf = [k for k in range(3, 25) if k not in fitted]
    print(f"   family 24 x 33: rms {rms(allE):.3f} max {np.max(np.abs(allE)):.2f} | compressing points only ({len(compE)}): rms {rms(compE):.3f} | "
          f"7 fitted positions rms {rms(np.concatenate([rows[k]['e'] for k in fitted])):.3f} | 15 unfitted positions 3..23 rms {rms(np.concatenate([rows[k]['e'] for k in unf])):.3f} max {max(rows[k]['mx'] for k in unf):.2f}")
    g0e = np.array([rows[k]["g0m"] - rows[k]["g0r"] for k in ks])
    print(f"   no-GR gain (flat region) model - ref by position: {np.round(g0e, 3).tolist()} | rms {rms(g0e):.4f} max {np.max(np.abs(g0e)):.3f}")
    # two-segment step law on the o_thr_db table, positions 5..24 (the sidechain agent's form)
    thr = np.array([rows[k]["thr"] for k in ks]); best = None
    for kb in range(6, 22):
        def seg(p): return np.where(ks < kb, p[0] + p[1] * (ks - kb), p[0] + p[2] * (ks - kb))
        r = least_squares(lambda p: (seg(p) - thr)[4:], [thr[19] - (20 - kb), 2.0, 1.0])
        if best is None or rms(r.fun) < best[1]: best = (kb, rms(r.fun), r.x, float(np.max(np.abs(r.fun))))
    print(f"   two-segment law on positions 5..24: break at {best[0]}, {best[2][1]:.2f} dB/step below, {best[2][2]:.3f} above, rms {best[1]:.3f} max {best[3]:.2f}")
    return rows


def F_gr(k):
    ref = np.array([F[f"opto_static_t{k}_{l}"] for l in LEVELS]); return ref[0] - ref


# ------------------------------------------------------------------------------------------------ 3. the sidechain low pass
def freq_statics(m, label):
    out = {}
    for f in (100, 3000, 8000):
        lv = list(range(-40, 11, 2)); ref = np.array([F[f"opto_static_f{f}_{l}"] for l in lv])
        e = static_gains(m, [20] * len(lv), lv, f=float(f)) - ref; out[f] = e
        print(f"   [{label}] f{f:5d} at t20: rms {rms(e):.3f} max {np.max(np.abs(e)):.2f} | model-ref by level -40..10: {np.round(e, 2).tolist()}")
    return out


def hf_rows(m, label):
    fr = np.array(DISC["hf"]["freqs"]); n = int(3.0 * FS); t = np.arange(n) / FS
    for lvl in (-40.0, -10.0, 0.0):
        ref = np.array(DISC["hf"]["rows"][str(lvl)])
        X = np.array([10 ** (lvl / 20) * np.sin(2 * np.pi * f * t) for f in fr]); lens = np.full(len(fr), n, dtype=np.int64)
        Y = m.render(X, lens, [20] * len(fr))
        g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(fr))]) + G0
        e = g - ref
        print(f"   [{label}] HF row {lvl:+.0f} dBFS, model-ref by 1/2/4/6/8/10/12/16 kHz: {np.round(e, 2).tolist()} | rms {rms(e):.3f} max {np.max(np.abs(e)):.2f}")


def fit_lp(m):
    lv = list(range(-40, 11, 2))
    refs = {f: np.array([F[f"opto_static_f{f}_{l}"] for l in lv]) for f in (3000, 8000)}
    def res(p):
        m.fc, m.Q = p
        return np.concatenate([static_gains(m, [20] * len(lv), lv, f=float(f)) - refs[f] for f in (3000, 8000)])
    r = least_squares(res, list(LP_REPORTED), bounds=([1000.0, 0.3], [20000.0, 3.0]), x_scale=[500.0, 0.1], diff_step=1e-3)
    m.fc, m.Q = r.x
    print(f"   refit of the low pass on opto_static_f3000_* + f8000_* (26 points): fc {r.x[0]:.0f} Hz, Q {r.x[1]:.3f} | rms {rms(r.fun):.3f} max {np.max(np.abs(r.fun)):.2f}")
    rb = least_squares(lambda p: res([p[0], 0.7071]), [LP_REPORTED[0]], bounds=([1000.0], [20000.0]), x_scale=[500.0], diff_step=1e-3)
    print(f"   Butterworth (Q 0.7071) refit: fc {rb.x[0]:.0f} Hz | rms {rms(rb.fun):.3f} max {np.max(np.abs(rb.fun)):.2f}")
    m.fc, m.Q = r.x
    return r.x


# ------------------------------------------------------------------------------------------------ 4. dynamics, harmonics, discriminating measurements
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]


def envelope(m, iid):
    it = ITEMS[iid]; fs = it["fs"]; x = protocol.stimulus(it["stim"], fs)[0]
    X = x[None, :].copy(); lens = np.array([len(x)]); Y = m.render(X, lens, [int(it["set"]["optical_threshold"])], fs=float(fs))
    E = np.zeros((1, int(len(x) / (fs / it["stim"]["f"])) + 2)); cnt = np.zeros(1, dtype=np.int64)
    env_batch(Y, X, lens, float(fs), float(it["stim"]["f"]), E, cnt)
    return E[0, :cnt[0]] + G0


def bursts(m):
    print("\n4a. burst family (weights as stage 4), model-ref:")
    allw = []; allp = []
    for iid in BURSTS + ["opto_burst_thr12", "sr_opto_44100", "sr_opto_96000"]:
        e = envelope(m, iid); ref = np.asarray(F[iid]); n = min(len(e), len(ref)); d = e[:n] - ref[:n]
        wt = np.where(ref[:n] < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)
        if iid in BURSTS: allw.append(wt * d); allp.append(d)
        st = ITEMS[iid]["stim"]; f = st["f"]; n0 = int(round(st["pre_s"] * f))
        if "burst_s" in st:
            n1 = n0 + int(round(st["burst_s"] * f))
            pts = [n1 + 5, n1 + 20, n1 + 50, n1 + 100, n1 + 300, n1 + 1000]; pts = [p for p in pts if p < n]
            tail = " ".join(f"{d[p]:+.2f}" for p in pts)
            print(f"   {iid:16s} rms {rms(d):.3f} max {np.max(np.abs(d)):.2f} | onset periods 1-4 model {np.round(e[n0:n0 + 4], 2).tolist()} ref {np.round(ref[n0:n0 + 4], 2).tolist()} | after the end +5/20/50/100/300/1000 ms: {tail}")
        else:
            print(f"   {iid:16s} rms {rms(d):.3f} max {np.max(np.abs(d)):.2f}")
    aw = np.concatenate(allw); ap = np.concatenate(allp)
    print(f"   nine items: weighted rms {rms(aw):.3f} max {np.max(np.abs(aw)):.2f} | plain rms {rms(ap):.3f} max {np.max(np.abs(ap)):.2f}")


def harmonics(m, label):
    ids = [f"opto_harm_t{t}_{l}_f{f}" for t in (14, 18, 22) for l in (-20, -10, 0) for f in (100, 1000, 4000)]
    print(f"\n4b. harmonics under GR ({label}), model / ref dBc:")
    err = {100: ([], []), 1000: ([], []), 4000: ([], [])}; g_err = []
    for iid in ids:
        it = ITEMS[iid]; x = protocol.stimulus(it["stim"], FS)[0]
        Y = m.render(x[None, :].copy(), np.array([len(x)]), [int(it["set"]["optical_threshold"])]); y = Y[0] * 10 ** (G0 / 20)
        h = protocol.feature(it, np.stack([y, y]), FS); ref = F[iid]; f = int(it["stim"]["f"])
        h7 = f"{h['h'][5]:.1f}/{ref['h'][5]:.1f}" if h["h"][5] is not None and ref["h"][5] is not None else "n/a"
        print(f"   {iid:24s} gain {h['gain_db']:+.2f}/{ref['gain_db']:+.2f} | H3 {h['h'][1]:.1f}/{ref['h'][1]:.1f} | H5 {h['h'][3]:.1f}/{ref['h'][3]:.1f} | H7 {h7}")
        err[f][0].append(h["h"][1] - ref["h"][1]); err[f][1].append(h["h"][3] - ref["h"][3]); g_err.append(h["gain_db"] - ref["gain_db"])
    for f in (100, 1000, 4000):
        print(f"   {f:5d} Hz (9 items): H3 error rms {rms(err[f][0]):.1f} max {np.max(np.abs(err[f][0])):.1f} | H5 error rms {rms(err[f][1]):.1f} max {np.max(np.abs(err[f][1])):.1f}")
    print(f"   gain error over the 27 items: rms {rms(g_err):.3f} max {np.max(np.abs(g_err)):.2f} dB")


def step_signal(a, b, secs=(2.0, 2.0, 2.0), f=1000.0):
    t = np.arange(int(round(sum(secs) * FS))) / FS
    env = np.full_like(t, 10 ** (a / 20)); n1 = int(secs[0] * FS); n2 = n1 + int(secs[1] * FS)
    env[n1:n2] = 10 ** (b / 20)
    return env * np.sin(2 * np.pi * f * t)


def per_cycle_gain(x, y, f=1000.0):
    per = int(round(FS / f)); n = len(x) // per
    xr = np.sqrt(np.mean(x[:n * per].reshape(n, per) ** 2, axis=1)); yr = np.sqrt(np.mean(y[:n * per].reshape(n, per) ** 2, axis=1))
    return 20 * np.log10(yr / xr)


def t_to(frac, seg, g_start, g_end):
    target = g_start + frac * (g_end - g_start)
    for i, v in enumerate(seg):
        if (g_end < g_start and v <= target) or (g_end > g_start and v >= target):
            return i
    return None


def discriminate(m):
    print("\n4c. discriminate_opto.json:")
    n = int(3.0 * FS); t = np.arange(n) / FS
    for thr in (20, 10):
        kd = DISC["knee"][str(thr)]; lv = np.array(kd["levels"]); gref = np.array(kd["gain_db"])
        X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
        Y = m.render(X, lens, [thr] * len(lv))
        g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(lv))]) + G0
        grm = g[0] - g; grr = gref[0] - gref; e = g - gref
        cm = [float(np.interp(v, grm, lv)) for v in (0.5, 1.0, 3.0, 6.0)]; cr = [float(np.interp(v, grr, lv)) for v in (0.5, 1.0, 3.0, 6.0)]
        sm = (g[-1] - g[-5]) / (lv[-1] - lv[-5]); sr = (gref[-1] - gref[-5]) / (lv[-1] - lv[-5])
        print(f"   (A) t{thr}: GR crosses 0.5/1/3/6 dB at {cm[0]:.2f}/{cm[1]:.2f}/{cm[2]:.2f}/{cm[3]:.2f} dBFS (ref {cr[0]:.2f}/{cr[1]:.2f}/{cr[2]:.2f}/{cr[3]:.2f}); "
              f"width 0.5->3 dB {cm[2] - cm[0]:.2f} (ref {cr[2] - cr[0]:.2f}); top slope {sm:.3f} (ref {sr:.3f}); rms {rms(e):.3f} max {np.max(np.abs(e)):.2f} dB")
        i0 = int(np.argmax(grr > 0.005)) - 1
        print(f"        GR per 0.5 dB from {lv[i0]:.1f} dBFS, model {np.round(grm[i0:i0 + 8], 2).tolist()} ref {np.round(grr[i0:i0 + 8], 2).tolist()}")
    # (B) extended rows
    lv = np.array(DISC["extended"]["levels"]); gref = np.array(DISC["extended"]["gain_db"])
    X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
    Y = m.render(X, lens, [20] * len(lv))
    g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(lv))]) + G0
    print(f"   (B) extended rows at t20, levels {lv.tolist()}: model {np.round(g, 2).tolist()} ref {np.round(gref, 2).tolist()} | model-ref {np.round(g - gref, 2).tolist()}")
    # (D) no-GR rows at -70 and -50
    n2 = int(2.0 * FS); t2 = np.arange(n2) / FS
    for lvl in ("-70.0", "-50.0"):
        ks = [int(k) for k in DISC["nogr"][lvl]]; ref = np.array([DISC["nogr"][lvl][str(k)] for k in ks])
        X = np.array([10 ** (float(lvl) / 20) * np.sin(2 * np.pi * 1000.0 * t2) for k in ks]); lens = np.full(len(ks), n2, dtype=np.int64)
        Y = m.render(X, lens, ks)
        g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(ks))]) + G0
        print(f"   (D) {lvl} dBFS, positions {ks}: model {np.round(g, 3).tolist()} ref {np.round(ref, 3).tolist()} | model-ref max {np.max(np.abs(g - ref)):.3f} dB")
    # (E) steps
    for key, v in DISC["steps"].items():
        a, b = v["from"], v["to"]; ref = np.array(v["per_cycle_gain_db"])
        x = step_signal(a, b); Y = m.render(x[None, :].copy(), np.array([len(x)]), [20]); gc = per_cycle_gain(x, Y[0] * 10 ** (G0 / 20))
        n1, n2_ = 2000, 4000; rows = []
        for g_ in (gc, ref):
            up = g_[n1:n2_]; down = g_[n2_:]
            ga = float(np.mean(g_[n1 - 200:n1 - 1])); gb = float(np.mean(g_[n2_ - 200:n2_ - 1])); ga2 = float(np.mean(g_[-200:]))
            rows.append((ga, gb, ga2, t_to(0.5, up, ga, gb), t_to(0.9, up, ga, gb), t_to(0.5, down, gb, ga2), t_to(0.9, down, gb, ga2), down[100] - ga2, down[500] - ga2, down[1000] - ga2, down))
        mm, rr = rows; d = gc[:len(ref)] - ref[:len(gc)]
        print(f"   (E) {a:+.0f}->{b:+.0f}->{a:+.0f}: gains model {mm[0]:+.2f}/{mm[1]:+.2f}/{mm[2]:+.2f} ref {rr[0]:+.2f}/{rr[1]:+.2f}/{rr[2]:+.2f} | attack 50/90 % {mm[3]}/{mm[4]} ms (ref {rr[3]}/{rr[4]}) | "
              f"release 50/90 % {mm[5]}/{mm[6]} ms (ref {rr[5]}/{rr[6]}) | tail at 100/500/1000 ms model {mm[7]:+.3f}/{mm[8]:+.3f}/{mm[9]:+.3f} ref {rr[7]:+.3f}/{rr[8]:+.3f}/{rr[9]:+.3f} | trajectory rms {rms(d):.3f} max {np.max(np.abs(d)):.2f}")
        print(f"        release 0..30 ms per 3 ms: model {np.round(mm[10][:30:3], 2).tolist()} ref {np.round(rr[10][:30:3], 2).tolist()}")


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    print(f"G0 {G0:.3f} dB (make-up 12 + Nickel), b2 {B2:.2e} b3 {B3:.2e}")
    validate()
    m = Model(V6)
    print(f"\nV6 shape: C {m.C:.3f} p {m.p:.3f} -> vth {m.vth():.3f}, slope p/(1+p) {m.p / (1 + m.p):.3f}; tau_el {m.tau_el * 1e3:.3f} ms; w {np.round(m.w, 3).tolist()} "
          f"attack ms {np.round(m.tatt * 1e3, 2).tolist()} release ms {np.round(m.trel * 1e3, 1).tolist()}; mu_c {m.muc:.2f}; leak b {m.b:.4f} q {m.q:.3f}")
    rows = fit_knees(m, "leak b 0.0202 q 2.154 as fitted in V6")
    m2 = Model(V6, b=0.0209, q=2.08); fit_knees(m2, "leak b 0.0209 q 2.08, the direct fit of the no-GR rows")
    print(f"\n   time {time.time() - t0:.0f} s")
    # 3. the sidechain low pass, with the V6 leak and the knees just fitted
    print("\n3. sidechain low pass at threshold 20")
    m.lp = False; freq_statics(m, "no low pass"); hf_rows(m, "no low pass")
    print("   digital magnitude of the two designs at 48 kHz against the analog second order (dB relative to 1 kHz), fc 5147 Q 0.718:")
    for kind, name in ((0, "RBJ bilinear"), (1, "Vicanek matched")):
        cf = lp_coeffs(LP_REPORTED[0], LP_REPORTED[1], FSF, kind)
        def H(f):
            z = np.exp(-2j * np.pi * f / FSF)
            return 20 * np.log10(abs((cf[0] + cf[1] * z + cf[2] * z * z) / (1 + cf[3] * z + cf[4] * z * z)))
        print(f"     {name:16s}: " + ", ".join(f"{f / 1000:g}k {H(f) - H(1000.0):+.2f}" for f in (2000, 4000, 6000, 8000, 10000, 12000, 16000, 20000)))
    print("     analog          : " + ", ".join(f"{f / 1000:g}k {lp2q(f, *LP_REPORTED) - lp2q(1000.0, *LP_REPORTED):+.2f}" for f in (2000, 4000, 6000, 8000, 10000, 12000, 16000, 20000)))
    m.lp = True; m.fc, m.Q = LP_REPORTED
    m.lpk = 0; freq_statics(m, f"RBJ LP {m.fc:.0f} Hz Q {m.Q:.3f}"); hf_rows(m, f"RBJ LP {m.fc:.0f} Hz Q {m.Q:.3f}")
    m.lpk = 1; freq_statics(m, f"matched LP {m.fc:.0f} Hz Q {m.Q:.3f}"); hf_rows(m, f"matched LP {m.fc:.0f} Hz Q {m.Q:.3f}")
    fc, Q = fit_lp(m); freq_statics(m, f"matched LP refit {fc:.0f} Hz Q {Q:.3f}"); hf_rows(m, f"matched LP refit {fc:.0f} Hz Q {Q:.3f}")
    print(f"   attenuation of the refit low pass relative to 1 kHz: " + ", ".join(f"{f} Hz {lp2q(f, fc, Q) - lp2q(1000.0, fc, Q):+.2f}" for f in (100, 1000, 3000, 4000, 6000, 8000, 12000, 16000)) + " dB")
    # 4. with the reported low pass in place
    m.fc, m.Q = LP_REPORTED
    print(f"\n4. dynamics, harmonics and the discriminating measurements, low pass {m.fc:.0f} Hz Q {m.Q:.3f} in the sidechain")
    bursts(m)
    harmonics(m, "with the low pass")
    m.lp = False; harmonics(m, "without the low pass (for the 4 kHz items)"); m.lp = True
    tau0 = m.tau_el
    for te in (1e-5, 3e-5):
        m.tau_el = te; harmonics(m, f"persistence tau_el {te * 1e3:.2f} ms instead of {tau0 * 1e3:.3f} (no refit)")
    m.tau_el = tau0
    discriminate(m)
    # the calibration values the C++ would carry
    print("\n5. C++ calibration equivalents of the chosen model:")
    print(f"   o_n 1.0, o_gamma {m.p:.4f}, o_vth {m.vth():.4f}, o_tau_el {m.tau_el:.3e}, o_w {np.round(m.w, 4).tolist()}, o_tatt {np.round(m.tatt, 5).tolist()}, o_trel {np.round(m.trel, 4).tolist()}, "
          f"o_rel_mu {m.muc:.3f}, o_leak {m.b:.4f}, o_leak_q {m.q:.3f}, o_sc_lp_hz {LP_REPORTED[0]:.0f}, o_sc_lp_q {LP_REPORTED[1]:.3f}")
    print(f"   o_thr_db 1..24: {np.round([rows[k]['thr'] for k in range(1, 25)], 2).tolist()}")
    print(f"   total {time.time() - t0:.0f} s")


def lp2q(f, fc, Q):
    w = f / fc
    return -10.0 * np.log10((1 - w * w) ** 2 + (w / Q) ** 2)


if __name__ == "__main__":
    main()
