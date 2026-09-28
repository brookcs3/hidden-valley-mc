# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical stage candidate, key "physical-law-hard-turnon" (hypothesis C).

Keep the feedback structure of src/dsp/Opto.hpp (divider v = x / (1 + cond), sidechain from v, EL light law, CdS cell law, three
conductance states with attack/release) but fix its degrees of freedom:
  (1) hard turn-on: the panel emits nothing below its threshold voltage. In the C++ the knee hardness is set by o_vth (the fit pinned it
      at its 0.5 bound). Here the law is re-parametrised as cond* = C * (A |v| / vth - 1)^p with vth = 1, so the free quantities are the
      KNEE LEVEL per threshold position (A = 10^(-knee_dBFS / 20)), the knee hardness C and the law exponent p. Back in C++ terms:
      vth = C^(1/p), o_thr_db = 20 log10(vth) - knee_dBFS.
  (2) the light exponent n and the cell exponent gamma only matter through p = n gamma above the knee (static slope p / (1 + p)), so p is
      fitted once and placed either on the cell (n = 1, gamma = p, "cell" placement) or on the light (n = p, gamma = 1, "light").
  (3) a bias that costs static gain with sidechain drive: cond0 = b (A / A_20)^q in parallel with the cell (variant "leak": the same
      amount added as light before the cell law).
  (4) an RMS detector (square, one-pole tau_rms, root) before the light law instead of the instantaneous |d| with persistence tau_el.
  (5) (an addition, only to test (E) below) light-assisted release: each state's release rate is multiplied by (1 + mu * target), so
      the cell recovers fast while it is still lit (a step between two above-knee levels) and slowly in the dark (a burst that ends
      below the knee). The linear three-state cell cannot do both with one set of time constants.
  (6) (the same purpose) self-quenched release: each state's release rate is multiplied by (1 + mu_c * s_i), i.e. dc/dt ~ -(c + mu_c c^2)
      in release, bimolecular recombination; the reference's release trajectories read as d(1/c)/dt ~ 85 /s from 20 dB down to 1 dB of
      gain reduction. (7) both multipliers together.
Fitted jointly against the reference features: the static family (thresholds 6, 10, 14, 18, 20, 22, 24 x levels -50..14 dBFS, 1 kHz;
18 and 22 are there because the harmonic items use them), the nine burst/pulse envelopes and H3/H5 of the three under-GR harmonic items.
The numba mirror is validated against the C++ engine (render_item) with the stage-4 calibration mapped onto it before any fit.
After each fit the discriminating measurements of fit/data/discriminate_opto.json are reproduced and compared: (A) the 0.5 dB knee sweeps
at thresholds 20 and 10, (D) the no-GR gain against threshold, (E) the steps between two above-knee levels.

usage: cd <repo> && python3 -u fit/tools/candidates/opto-physical-law-hard-turnon.py [--variants V1,V2,V2L,V3,V3L,V4,V5,V6,V7] [--nfev 60] [--quick]
  --quick   every other level (17 per threshold) and no second start, for a fast look
"""
import json
import os, sys, time, numpy as np  # noqa: E402
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402

ARGS = sys.argv[1:]
def opt(name, default):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default
QUICK = "--quick" in ARGS
NFEV = int(opt("--nfev", "60"))
VARIANTS = opt("--variants", "V1,V2,V2L,V3,V3L,V4,V5,V6,V7").split(",")
HARM_W = float(opt("--harm-weight", "1.0"))      # multiplies the H3/H5/gain residuals of the three under-GR harmonic items
REFINE = "--refine" in ARGS                      # a second least_squares pass from the solution, finer difference step, linear loss
DISC = json.load(open(os.path.join(HERE, "..", "..", "data", "discriminate_opto.json")))

cal = load_cal()
def cget(name, i=0): return float(cal[MODEL.fields[name][0] + i])
B2, B3 = cget("o_b2"), cget("o_b3")
G0 = cget("o_gain_db", 11) + cget("x_gain_db", 0)     # make-up at position 12 plus the Nickel path's midband gain, dB
THRS = [6, 10, 14, 18, 20, 22, 24]; REPORT_THRS = [6, 10, 14, 20, 24]
LEVELS = list(range(-50, 15, 4 if QUICK else 2))
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
HARMS = ["opto_harm_t18_-10_f1000", "opto_harm_t18_-10_f100", "opto_harm_t22_0_f1000"]
FSF = float(FS)


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def run_one(x, n, fs, A, C, nexp, gam, tau_el, det, tau_rms, w, tatt, trel, cond0, leak, mu, muc, b2, b3, out):
    kEl = 1.0 - np.exp(-1.0 / (tau_el * fs)) if tau_el > 0.0 else 1.0
    kR = 1.0 - np.exp(-1.0 / (tau_rms * fs)) if tau_rms > 0.0 else 1.0
    kA0 = 1.0 - np.exp(-1.0 / (tatt[0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (tatt[1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (tatt[2] * fs))
    kR0 = 1.0 - np.exp(-1.0 / (trel[0] * fs)); kR1 = 1.0 - np.exp(-1.0 / (trel[1] * fs)); kR2 = 1.0 - np.exp(-1.0 / (trel[2] * fs))
    l0 = (cond0 / C) ** (1.0 / gam) if (leak == 1 and cond0 > 0.0) else 0.0
    L = 0.0; P = 0.0; s0 = 0.0; s1 = 0.0; s2 = 0.0; cond = 0.0
    for i in range(n):
        xi = x[i]
        xa = xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        out[i] = v
        d = A * abs(v)
        if det == 1:
            P += (d * d - P) * kR
            d = np.sqrt(P)
        e = d - 1.0
        Linst = e ** nexp if e > 0.0 else 0.0
        L += (Linst - L) * kEl
        if leak == 1:
            target = C * (L + l0) ** gam
        else:
            target = (C * L ** gam if L > 0.0 else 0.0) + cond0
        if mu > 0.0 or muc > 0.0:
            r = 1.0 + mu * target
            q0 = 1.0 - np.exp(-(r + muc * s0) / (trel[0] * fs)); q1 = 1.0 - np.exp(-(r + muc * s1) / (trel[1] * fs)); q2 = 1.0 - np.exp(-(r + muc * s2) / (trel[2] * fs))
        else:
            q0 = kR0; q1 = kR1; q2 = kR2
        s0 += (target - s0) * (kA0 if target > s0 else q0)
        s1 += (target - s1) * (kA1 if target > s1 else q1)
        s2 += (target - s2) * (kA2 if target > s2 else q2)
        cond = w[0] * s0 + w[1] * s1 + w[2] * s2


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, C, nexp, gam, tau_el, det, tau_rms, w, tatt, trel, cond0, leak, mu, muc, b2, b3, out):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], C, nexp, gam, tau_el, det, tau_rms, w, tatt, trel, cond0[j], leak, mu, muc, b2, b3, out[j])


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


# ------------------------------------------------------------------------------------------------ the data sets
class Batch:
    def __init__(self, ids):
        self.ids = ids
        xs = [protocol.stimulus(ITEMS[i]["stim"], FS)[0] for i in ids]
        self.lens = np.array([len(x) for x in xs], dtype=np.int64)
        self.X = np.zeros((len(xs), int(self.lens.max())))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.Y = np.zeros_like(self.X)
        self.thr = np.array([int(ITEMS[i]["set"]["optical_threshold"]) for i in ids])
        self.level = np.array([float(ITEMS[i]["stim"]["level"]) for i in ids])
        self.gain_out = np.zeros(len(ids))

STATIC_IDS = [f"opto_static_t{k}_{l}" for k in THRS for l in LEVELS]
S = Batch(STATIC_IDS); B = Batch(BURSTS); H = Batch(HARMS)
REF_S = np.array([F[i] for i in STATIC_IDS])


def burst_weights(iid, n):
    ref = np.asarray(F[iid]); return np.where(ref[:n] < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)


# ------------------------------------------------------------------------------------------------ parameters
class Spec:
    """a named parameter vector with bounds; log-domain entries are stored as log10"""
    def __init__(self):
        self.names, self.x0, self.lo, self.hi, self.scale = [], [], [], [], []
    def add(self, name, x0, lo, hi, scale):
        self.names.append(name); self.x0.append(x0); self.lo.append(lo); self.hi.append(hi); self.scale.append(scale)
    def unpack(self, p):
        return {n: p[i] for i, n in enumerate(self.names)}


def model_params(d, variant):
    """Spec values -> the mirror's arguments (per-threshold drive A and bias cond0 for each threshold in THRS)"""
    knee = {k: d[f"knee_{k}"] for k in THRS}
    C = 10.0 ** d["logC"]; p = d["p"]
    nexp, gam = (p, 1.0) if variant["place"] == "light" else (1.0, p)
    tau_el = 10.0 ** d["log_tau_el"] if "log_tau_el" in d else 0.0
    tau_rms = 10.0 ** d["log_tau_rms"] if "log_tau_rms" in d else 0.0
    w = np.abs(np.array([d["w0"], d["w1"], d["w2"]])); w = w / w.sum()
    tatt = 10.0 ** np.array([d["la0"], d["la1"], d["la2"]]); trel = 10.0 ** np.array([d["lr0"], d["lr1"], d["lr2"]])
    A = {k: 10.0 ** (-knee[k] / 20.0) for k in THRS}
    if "logb" in d:
        b = 10.0 ** d["logb"]; q = d["q"]
        cond0 = {k: b * (A[k] / A[20]) ** q for k in THRS}
    else:
        cond0 = {k: 0.0 for k in THRS}
    mu = 10.0 ** d["logmu"] if "logmu" in d else 0.0
    muc = 10.0 ** d["logmuc"] if "logmuc" in d else 0.0
    return dict(A=A, C=C, nexp=nexp, gam=gam, tau_el=tau_el, det=variant["det"], tau_rms=tau_rms, w=w, tatt=tatt, trel=trel,
                cond0=cond0, leak=variant["leak"], mu=mu, muc=muc, knee=knee, p=p)


def render_batch(bt, m):
    A = np.array([m["A"][k] for k in bt.thr]); c0 = np.array([m["cond0"][k] for k in bt.thr])
    run_batch(bt.X, bt.lens, FSF, A, m["C"], m["nexp"], m["gam"], m["tau_el"], m["det"], m["tau_rms"], m["w"], m["tatt"], m["trel"],
              c0, m["leak"], m["mu"], m["muc"], B2, B3, bt.Y)


def statics(m):
    render_batch(S, m)
    lockin_batch(S.Y, S.lens, FSF, 1000.0, 0.5, S.gain_out)
    return S.gain_out - S.level + G0


ENV_OUT = np.zeros((len(BURSTS), int(B.lens.max() // 48) + 2)); ENV_N = np.zeros(len(BURSTS), dtype=np.int64)


def envs(m):
    render_batch(B, m)
    env_batch(B.Y, B.X, B.lens, FSF, 1000.0, ENV_OUT, ENV_N)
    return [ENV_OUT[j, :ENV_N[j]] + G0 for j in range(len(BURSTS))]


def envs_protocol(m):
    """the same through protocol.feature (slow; used once to check env_batch)"""
    render_batch(B, m); out = []
    for j, iid in enumerate(B.ids):
        y = B.Y[j, :B.lens[j]] * 10.0 ** (G0 / 20.0)
        out.append(np.asarray(protocol.feature(ITEMS[iid], np.stack([y, y]), FS)))
    return out


def harms(m):
    render_batch(H, m); out = []
    for j, iid in enumerate(H.ids):
        y = H.Y[j, :H.lens[j]] * 10.0 ** (G0 / 20.0)
        out.append(protocol.feature(ITEMS[iid], np.stack([y, y]), FS))
    return out


def residuals(p, spec, variant, parts=None):
    m = model_params(spec.unpack(p), variant)
    r_s = 2.0 * (statics(m) - REF_S)
    r_b = []
    for e, iid in zip(envs(m), BURSTS):
        ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        r_b.append(burst_weights(iid, n) * (e[:n] - ref[:n]))
    r_b = np.concatenate(r_b)
    r_h = []
    for h, iid in zip(harms(m), HARMS):
        ref = F[iid]
        r_h += [2.0 * (h["gain_db"] - ref["gain_db"]), 1.0 * (h["h"][1] - ref["h"][1]), 0.7 * (h["h"][3] - ref["h"][3])]
    r_h = HARM_W * np.array(r_h)
    if parts is not None: parts.update(s=r_s / 2.0, b=r_b, h=r_h, m=m)
    return np.concatenate([r_s, r_b, r_h])


def report(spec, p, variant, label):
    parts = {}; residuals(p, spec, variant, parts)
    m = parts["m"]; rs = parts["s"].reshape(len(THRS), len(LEVELS))
    sel = [THRS.index(k) for k in REPORT_THRS]
    rms5 = float(np.sqrt(np.mean(rs[sel] ** 2))); max5 = float(np.max(np.abs(rs[sel])))
    rms20 = float(np.sqrt(np.mean(rs[THRS.index(20)] ** 2))); max20 = float(np.max(np.abs(rs[THRS.index(20)])))
    rmsb = float(np.sqrt(np.mean(parts["b"] ** 2))); maxb = float(np.max(np.abs(parts["b"])))
    d = spec.unpack(p)
    vth = m["C"] ** (1.0 / m["p"])
    print(f"\n=== {label} ===")
    print(f"  p (n*gamma) {m['p']:.3f} -> slope {m['p'] / (1 + m['p']):.3f} dB/dB | C {m['C']:.3f} (C++ vth = C^(1/p) = {vth:.3f}) | placement {variant['place']} | "
          f"detector {'RMS tau_rms %.3f ms' % (m['tau_rms'] * 1e3) if variant['det'] == 1 else 'inst, tau_el %.3f ms' % (m['tau_el'] * 1e3)}")
    print(f"  states: w {np.round(m['w'], 3)} attack ms {np.round(m['tatt'] * 1e3, 2)} release ms {np.round(m['trel'] * 1e3, 1)}")
    if m["mu"] > 0.0 or m["muc"] > 0.0: print(f"  release multiplier: 1 + {m['mu']:.2f} * target + {m['muc']:.2f} * state")
    print("  raw parameter vector:", json.dumps({k: round(float(v), 6) for k, v in d.items()}))
    if "logb" in d: print(f"  bias: cond0 at thr 20 = {10 ** d['logb']:.4f} ({20 * np.log10(1 + 10 ** d['logb']):.3f} dB), exponent q {d['q']:.2f} in A/A20 ({'light leak' if variant['leak'] else 'parallel conductance'})")
    print("  knee dBFS by threshold:", {k: round(m['knee'][k], 2) for k in THRS})
    print("  C++ o_thr_db equivalents:", {k: round(20 * np.log10(vth) - m['knee'][k], 2) for k in THRS})
    print(f"  STATIC (thr 6/10/14/20/24 x {len(LEVELS)} levels): rms {rms5:.3f} dB max {max5:.2f} | thr 20 only: rms {rms20:.3f} max {max20:.2f} | all 7 thr: rms {np.sqrt(np.mean(rs ** 2)):.3f}")
    for k in THRS:
        print(f"    t{k:2d} model-ref (dB) by level {LEVELS[0]}..{LEVELS[-1]}: {np.round(rs[THRS.index(k)], 2)}")
    print(f"  BURSTS weighted rms {rmsb:.3f} max {maxb:.2f}")
    for e, iid in zip(envs(m), BURSTS):
        ref = np.asarray(F[iid]); n = min(len(e), len(ref)); err = e[:n] - ref[:n]
        end = 500 + int(round(ITEMS[iid]["stim"].get("burst_s", 0.0) * 1000)) if "burst_s" in ITEMS[iid]["stim"] else 500 + 8 * 250
        tail = [f"{e[end + t] - ref[end + t]:+.2f}" for t in (5, 20, 50, 100, 500, 1000, 2000) if end + t < n]
        print(f"    {iid:16s} rms {np.sqrt(np.mean(err ** 2)):.3f} max {np.max(np.abs(err)):.2f} | first 6 periods after onset: model {np.round(e[500:506], 2)} ref {np.round(ref[500:506], 2)} | model-ref at +5/20/50/100/500/1000/2000 ms after the end: {' '.join(tail)}")
    print("  HARMONICS under GR (model / ref):")
    for h, iid in zip(harms(m), HARMS):
        ref = F[iid]
        print(f"    {iid}: gain {h['gain_db']:.2f}/{ref['gain_db']:.2f} | H3 {h['h'][1]:.1f}/{ref['h'][1]:.1f} | H5 {h['h'][3]:.1f}/{ref['h'][3]:.1f} | H7 {h['h'][5]:.1f}/{ref['h'][5]:.1f}")
    disc = discriminate(disc_render_mirror(m), label)
    return dict(rms5=rms5, max5=max5, rms20=rms20, rmsb=rmsb, maxb=maxb, m=m, d=d, harms=harms(m), disc=disc)


# ------------------------------------------------------------------------------------------------ the discriminating measurements
def rms_gain_last_s(x, y, secs=1.0):
    n = int(FS * secs)
    return float(20 * np.log10(np.sqrt(np.mean(y[-n:] ** 2)) / np.sqrt(np.mean(x[-n:] ** 2))))


def step_signal(a, b, secs=(2.0, 2.0, 2.0), f=1000.0):
    t = np.arange(int(round(sum(secs) * FS))) / FS
    env = np.full_like(t, 10 ** (a / 20)); n1 = int(secs[0] * FS); n2 = n1 + int(secs[1] * FS)
    env[n1:n2] = 10 ** (b / 20)
    return env * np.sin(2 * np.pi * f * t)


def per_cycle_gain(x, y, f=1000.0):
    per = int(round(FS / f)); m = len(x) // per
    xr = np.sqrt(np.mean(x[:m * per].reshape(m, per) ** 2, axis=1)); yr = np.sqrt(np.mean(y[:m * per].reshape(m, per) ** 2, axis=1))
    return 20 * np.log10(yr / xr)


def t_to(frac, seg, g_start, g_end):
    target = g_start + frac * (g_end - g_start)
    for i, v in enumerate(seg):
        if (g_end < g_start and v <= target) or (g_end > g_start and v >= target):
            return i
    return None


class DiscBatch:
    """(A) 0.5 dB knee sweeps at thresholds 20 and 10 (3 s sines), (E) three above-knee steps at threshold 20, in one render batch"""
    def __init__(self):
        xs, self.thr, self.tag = [], [], []
        for thr in (20, 10):
            for l in DISC["knee"][str(thr)]["levels"]:
                xs.append(protocol.stimulus({"kind": "sine", "level": float(l), "f": 1000.0, "secs": 3.0}, FS)[0]); self.thr.append(thr); self.tag.append(("A", thr, l))
        for key, v in DISC["steps"].items():
            xs.append(step_signal(v["from"], v["to"])); self.thr.append(20); self.tag.append(("E", key, 0))
        self.lens = np.array([len(x) for x in xs], dtype=np.int64)
        self.X = np.zeros((len(xs), int(self.lens.max())))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.Y = np.zeros_like(self.X); self.thr = np.array(self.thr)


DB = None


def discriminate(render, label, knees=None):
    """render(DiscBatch) fills DB.Y (make-up NOT applied). Prints (A), (D) and (E) against fit/data/discriminate_opto.json.
    (D) uses the static family: the model's -50 dBFS gain per threshold against the reference's, as a loss relative to threshold 1."""
    global DB
    if DB is None: DB = DiscBatch()
    render(DB)
    g = 10.0 ** (G0 / 20.0)
    print(f"  --- discriminating checks, {label}")
    out = {}
    for thr in (20, 10):
        ref = np.array(DISC["knee"][str(thr)]["gain_db"]); lv = np.array(DISC["knee"][str(thr)]["levels"])
        idx = [j for j, t in enumerate(DB.tag) if t[0] == "A" and t[1] == thr]
        mod = np.array([rms_gain_last_s(DB.X[j, :DB.lens[j]], DB.Y[j, :DB.lens[j]] * g) for j in idx])
        grr = ref[0] - ref; grm = mod[0] - mod
        kr = float(np.interp(0.5, grr, lv)); km = float(np.interp(0.5, grm, lv))
        k3r = float(np.interp(3.0, grr, lv)); k3m = float(np.interp(3.0, grm, lv))
        sr = (ref[-1] - ref[-5]) / (lv[-1] - lv[-5]); sm = (mod[-1] - mod[-5]) / (lv[-1] - lv[-5])
        e = mod - ref
        print(f"  (A) thr {thr}: GR crosses 0.5/3 dB at {km:.2f}/{k3m:.2f} dBFS (ref {kr:.2f}/{k3r:.2f}); width 0.5->3 dB {k3m - km:.2f} (ref {k3r - kr:.2f}); "
              f"top slope {sm:+.3f} (ref {sr:+.3f}) dB/dB; model-ref rms {np.sqrt(np.mean(e ** 2)):.3f} max {np.max(np.abs(e)):.2f} dB")
        print(f"      GR per 0.5 dB, model: {np.round(grm[12:24], 2).tolist()}")
        print(f"                     ref: {np.round(grr[12:24], 2).tolist()}")
        out[f"A{thr}"] = (float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))), k3m - km, k3r - kr)
    # (D)
    r1 = float(F["opto_static_t1_-50"])
    print("  (D) no-GR gain at -50 dBFS, loss relative to threshold 1 (model / ref), dB:", end="")
    for k in THRS:
        j = STATIC_IDS.index(f"opto_static_t{k}_-50")
        print(f"  t{k} {r1 - S.gain_out[j] + S.level[j] - G0:.3f}/{r1 - F[STATIC_IDS[j]]:.3f}", end="")
    print()
    # (E)
    for key, v in DISC["steps"].items():
        j = [jj for jj, t in enumerate(DB.tag) if t[0] == "E" and t[1] == key][0]
        gr = np.array(v["per_cycle_gain_db"]); gm = per_cycle_gain(DB.X[j, :DB.lens[j]], DB.Y[j, :DB.lens[j]] * g)
        n1, n2 = 2000, 4000
        def stats(gc):
            up = gc[n1:n2]; down = gc[n2:]
            ga = float(np.mean(gc[n1 - 200:n1 - 1])); gb = float(np.mean(gc[n2 - 200:n2 - 1])); ga2 = float(np.mean(gc[-200:]))
            return ga, gb, ga2, t_to(0.5, up, ga, gb), t_to(0.9, up, ga, gb), t_to(0.5, down, gb, ga2), t_to(0.9, down, gb, ga2), down[100] - ga2, down[500] - ga2, down[1000] - ga2
        sm_, sr_ = stats(gm), stats(gr)
        print(f"  (E) step {v['from']:+.0f}->{v['to']:+.0f}->{v['from']:+.0f}: gains model {sm_[0]:+.2f}/{sm_[1]:+.2f}/{sm_[2]:+.2f} ref {sr_[0]:+.2f}/{sr_[1]:+.2f}/{sr_[2]:+.2f} | "
              f"attack 50/90 % at {sm_[3]}/{sm_[4]} ms (ref {sr_[3]}/{sr_[4]}) | release 50/90 % at {sm_[5]}/{sm_[6]} ms (ref {sr_[5]}/{sr_[6]}) | "
              f"tail at 100/500/1000 ms: model {sm_[7]:+.3f}/{sm_[8]:+.3f}/{sm_[9]:+.3f} ref {sr_[7]:+.3f}/{sr_[8]:+.3f}/{sr_[9]:+.3f} dB")
        e = gm[:len(gr)] - gr[:len(gm)]
        print(f"      per-cycle model-ref rms {np.sqrt(np.mean(e ** 2)):.3f} dB, max {np.max(np.abs(e)):.2f} dB; release, first 100 ms per 5 ms model {np.round(gm[n2:n2 + 100:5], 2).tolist()}")
        out[f"E{key}"] = (float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))), sm_[9], sr_[9], sm_[7], sr_[7], sm_[6], sr_[6])
    return out


def disc_render_mirror(m):
    def r(bt):
        A = np.array([m["A"][k] for k in bt.thr]); c0 = np.array([m["cond0"][k] for k in bt.thr])
        run_batch(bt.X, bt.lens, FSF, A, m["C"], m["nexp"], m["gam"], m["tau_el"], m["det"], m["tau_rms"], m["w"], m["tatt"], m["trel"],
                  c0, m["leak"], m["mu"], m["muc"], B2, B3, bt.Y)
    return r


def disc_render_cpp(bt):
    """the C++ engine (stage-4 calibration) on the same signals; its make-up is divided out so the caller's G0 applies as for the mirror"""
    g = 10.0 ** (G0 / 20.0)
    for j in range(bt.X.shape[0]):
        y = MODEL.render(bt.X[j, :bt.lens[j]], FS, {"optical_bypass": "In", "optical_threshold": int(bt.thr[j])}, cal=cal)[0]
        bt.Y[j, :bt.lens[j]] = y / g


# ------------------------------------------------------------------------------------------------ variants
def make_spec(variant, start):
    s = Spec()
    # knee levels from the stage-4 threshold table laid onto a -27 dBFS knee at threshold 20
    thr_db = cal[MODEL.field("o_thr_db")]
    for k in THRS:
        s.add(f"knee_{k}", -27.0 - (thr_db[k - 1] - thr_db[19]), -45.0, 5.0, 0.5)
    if start == "stage4":
        s.add("logC", np.log10(cget("o_vth") ** (cget("o_n") * cget("o_gamma"))), -2.0, 3.0, 0.1)
        s.add("p", cget("o_n") * cget("o_gamma"), 0.8, 3.0, 0.05)
        w = [cget("o_w", i) for i in range(3)]; ta = [cget("o_tatt", i) for i in range(3)]; tr = [cget("o_trel", i) for i in range(3)]
        tau_el = cget("o_tau_el")
    else:   # "physical": a harder knee, the slope the reference shows, a symmetric fast state and slower trapping states
        s.add("logC", np.log10(3.0), -2.0, 3.0, 0.1)
        s.add("p", 1.63, 0.8, 3.0, 0.05)
        w = [0.05, 0.70, 0.25]; ta = [1.0, 0.007, 0.05]; tr = [1.0, 0.007, 0.05]
        tau_el = 1e-3
    if variant["det"] == 1:
        s.add("log_tau_rms", np.log10(max(tau_el, 1e-3)), -4.5, -1.5, 0.1)
    else:
        s.add("log_tau_el", np.log10(max(tau_el, 5e-5)), -4.5, -1.5, 0.1)
    for i in range(3): s.add(f"w{i}", w[i], 0.005, 1.0, 0.05)
    for i in range(3): s.add(f"la{i}", np.log10(ta[i]), -4.0, 0.0, 0.1)
    for i in range(3): s.add(f"lr{i}", np.log10(tr[i]), -3.0, 1.0, 0.1)
    if variant["bias"]:
        s.add("logb", np.log10(0.02), -4.0, -0.5, 0.1)
        s.add("q", 2.2, 0.5, 4.0, 0.1)
    if variant.get("lar"):
        s.add("logmu", np.log10(10.0), -1.0, 3.0, 0.1)
    if variant.get("sq"):
        s.add("logmuc", np.log10(30.0), -1.0, 3.0, 0.1)
    return s


VAR = {
    "V1": dict(place="cell", det=0, leak=0, bias=False, title="(1)+(2): hard turn-on by knee level + hardness C, product exponent on the cell; no bias"),
    "V2": dict(place="cell", det=0, leak=0, bias=True, title="V1 + (3) bias cond0 = b (A/A20)^q in parallel"),
    "V2L": dict(place="cell", det=0, leak=1, bias=True, title="V1 + (3) bias as a light leak before the cell law"),
    "V3": dict(place="cell", det=1, leak=0, bias=True, title="V2 with (4) an RMS detector (tau_rms) before the light law, no persistence pole"),
    "V3L": dict(place="cell", det=1, leak=1, bias=True, title="V3 (RMS detector) with the bias as a light leak"),
    "V4": dict(place="light", det=0, leak=0, bias=True, title="V2 with the exponent on the light (n = p, gamma = 1)"),
    "V5": dict(place="cell", det=0, leak=1, bias=True, lar=True, title="V2L + (5) light-assisted release, rate x (1 + mu * target)"),
    "V6": dict(place="cell", det=0, leak=1, bias=True, sq=True, title="V2L + (6) self-quenched release, rate x (1 + mu_c * state)"),
    "V7": dict(place="cell", det=0, leak=1, bias=True, lar=True, sq=True, title="V2L + (5) + (6): rate x (1 + mu * target + mu_c * state)"),
}


def fit(variant, start, nfev):
    spec = make_spec(variant, start)
    x0 = np.clip(np.array(spec.x0), spec.lo, spec.hi)
    t0 = time.time()
    r = least_squares(residuals, x0, bounds=(spec.lo, spec.hi), args=(spec, variant), x_scale=np.array(spec.scale), diff_step=2e-3,
                      max_nfev=nfev, loss="soft_l1", f_scale=1.0)
    print(f"  [{start} start] cost {r.cost:.2f} after {r.nfev} evals, {time.time() - t0:.0f} s, status {r.status}")
    if REFINE:
        t0 = time.time()
        r = least_squares(residuals, r.x, bounds=(spec.lo, spec.hi), args=(spec, variant), x_scale=np.array(spec.scale), diff_step=5e-4,
                          max_nfev=2 * nfev, xtol=1e-10, ftol=1e-6)
        print(f"  [{start} start, refined] cost {r.cost:.2f} after {r.nfev} evals, {time.time() - t0:.0f} s, status {r.status}")
    return spec, r


# ------------------------------------------------------------------------------------------------ main
def validate_mirror():
    """the stage-4 calibration mapped onto the mirror against the C++ engine on the same items"""
    n, gam, vth, tau_el = cget("o_n"), cget("o_gamma"), cget("o_vth"), cget("o_tau_el")
    thr_db = cal[MODEL.field("o_thr_db")]
    p = n * gam
    m = dict(A={k: 10.0 ** (thr_db[k - 1] / 20.0) / vth for k in THRS}, C=vth ** p, nexp=n, gam=gam, tau_el=tau_el, det=0, tau_rms=0.0,
             w=np.array([cget("o_w", i) for i in range(3)]), tatt=np.array([cget("o_tatt", i) for i in range(3)]),
             trel=np.array([cget("o_trel", i) for i in range(3)]), cond0={k: 0.0 for k in THRS}, leak=0, mu=0.0, muc=0.0, knee={}, p=p)
    t0 = time.time(); ms = statics(m); t1 = time.time() - t0
    print(f"mirror: {len(STATIC_IDS)} statics in {t1:.2f} s")
    ids = [f"opto_static_t20_{l}" for l in LEVELS]
    cpp = np.array([float(render_item(ITEMS[i], cal)) for i in ids])
    mine = ms.reshape(len(THRS), len(LEVELS))[THRS.index(20)]
    print(f"  mirror vs C++ (stage-4 cal), static t20: max |diff| {np.max(np.abs(mine - cpp)):.4f} dB")
    e_cpp = np.asarray(render_item(ITEMS["opto_burst_-10"], cal)); e_m = envs(m)[2]
    print(f"  mirror vs C++, opto_burst_-10 envelope: max |diff| {np.max(np.abs(e_m - e_cpp[:len(e_m)])):.4f} dB")
    dmax = max(float(np.max(np.abs(a[:min(len(a), len(b))] - b[:min(len(a), len(b))]))) for a, b in zip(envs(m), envs_protocol(m)))
    print(f"  env_batch vs protocol.feature over the nine envelopes: max |diff| {dmax:.5f} dB")
    hm = harms(m)[0]; hc = render_item(ITEMS[HARMS[0]], cal)
    print(f"  mirror vs C++, {HARMS[0]}: H3 {hm['h'][1]:.1f}/{hc['h'][1]:.1f}  H5 {hm['h'][3]:.1f}/{hc['h'][3]:.1f}")
    # the stage-4 point's residuals on this harness's data, for a like-for-like baseline
    all_cpp = np.array([float(render_item(ITEMS[i], cal)) for i in STATIC_IDS]).reshape(len(THRS), len(LEVELS))
    rs = all_cpp - REF_S.reshape(len(THRS), len(LEVELS)); sel = [THRS.index(k) for k in REPORT_THRS]
    print(f"  BASELINE (C++ engine, stage-4 cal) static thr 6/10/14/20/24: rms {np.sqrt(np.mean(rs[sel] ** 2)):.3f} dB max {np.max(np.abs(rs[sel])):.2f} | thr 20: rms {np.sqrt(np.mean(rs[THRS.index(20)] ** 2)):.3f} max {np.max(np.abs(rs[THRS.index(20)])):.2f}")
    print(f"    t20 model-ref by level: {np.round(rs[THRS.index(20)], 2)}")
    rb = []
    for iid in BURSTS:
        e = np.asarray(render_item(ITEMS[iid], cal)); ref = np.asarray(F[iid]); n_ = min(len(e), len(ref)); rb.append(burst_weights(iid, n_) * (e[:n_] - ref[:n_]))
    rb = np.concatenate(rb)
    print(f"  BASELINE bursts weighted rms {np.sqrt(np.mean(rb ** 2)):.3f} max {np.max(np.abs(rb)):.2f}")
    statics(m)   # S.gain_out for (D)
    discriminate(disc_render_cpp, "BASELINE (C++ engine, stage-4 cal)")


def profile():
    v = VAR["V2"]; s = make_spec(v, "physical"); m = model_params(s.unpack(np.array(s.x0)), v)
    for fn in (statics, envs, harms):
        fn(m); t0 = time.time(); fn(m); print(f"  {fn.__name__}: {time.time() - t0:.3f} s")
    for bt, nm in ((S, "statics"), (B, "bursts")):
        render_batch(bt, m); t0 = time.time(); render_batch(bt, m); print(f"  render {nm} only: {time.time() - t0:.3f} s")


def main():
    print(f"levels per threshold {len(LEVELS)}, thresholds {THRS}, G0 {G0:.3f} dB, b2 {B2:.2e} b3 {B3:.2e}, nfev {NFEV}, quick {QUICK}")
    if "--profile" in ARGS:
        profile(); return
    validate_mirror()
    if "--validate-only" in ARGS: return
    results = {}
    for name in VARIANTS:
        v = VAR[name]
        print(f"\n##### {name}: {v['title']}")
        best = None
        for start in (("stage4", "physical") if not QUICK else ("physical",)):
            spec, r = fit(v, start, NFEV)
            if best is None or r.cost < best[1].cost: best = (spec, r, start)
        spec, r, start = best
        results[name] = report(spec, r.x, v, f"{name} (best start: {start})")
    print("\n##### SUMMARY")
    print(f"  {'variant':6s} {'static rms5':>11s} {'max5':>6s} {'t20 rms':>8s} {'burst rms':>9s} {'H3 err (3 items)':>22s} {'H5 err (3 items)':>22s} {'(A) t20 rms/width':>18s} {'(E) tail@1s -10->-20':>20s} {'(E) +5->-20 tail@100ms, t90':>26s}")
    for name, res in results.items():
        h3 = [res["harms"][j]["h"][1] - F[i]["h"][1] for j, i in enumerate(HARMS)]; h5 = [res["harms"][j]["h"][3] - F[i]["h"][3] for j, i in enumerate(HARMS)]
        a20 = res["disc"]["A20"]; e1 = res["disc"]["E-20_-10"]; e3 = res["disc"]["E-20_5"]
        print(f"  {name:6s} {res['rms5']:11.3f} {res['max5']:6.2f} {res['rms20']:8.3f} {res['rmsb']:9.3f} {str(np.round(h3, 1)):>22s} {str(np.round(h5, 1)):>22s} {a20[0]:8.3f}/{a20[2]:5.2f}({a20[3]:.2f}) {e1[2]:+8.3f} (ref {e1[3]:+.3f}) {e3[4]:+8.3f} (ref {e3[5]:+.3f}), {e3[6]} ms (ref {e3[7]})")


if __name__ == "__main__":
    main()
