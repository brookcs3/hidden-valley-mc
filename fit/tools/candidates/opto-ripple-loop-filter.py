# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical ripple harmonics, key "opto-ripple-c-loop-filter": directions (d) and (e) of the ripple hole in docs/opto-fix.md section 6,
plus a direct characterisation of the reference's ripple law.

Hypotheses tested (each fits only the parameters it adds; the V6 shape, the knee table and the matched sidechain low pass of
docs/opto-fix.md are held):
  (d1) the sidechain low pass Q (and corner) shape the ripple that is fed back;
  (d2) an added real pole in the sidechain before the rectifier;
  (d3) an added lead/lag (zero + pole) section in the sidechain before the rectifier;
  (d4) the persistence pole placed AFTER the cell law (on the target the states chase) instead of on the light;
  (d5) a pole on the conductance itself (after the states, before the divider);
  (e)  the divider's even-order behaviour: whether the amplifier terms b2, b3 act on the stage input (engine) or on the divider
       output (what the H2 rows under gain reduction say), and whether that placement moves H3/H5/H7;
  control: the persistence pole alone, refitted on the same objective, so every candidate is scored against the same-objective
       control and not against the unrefitted synthesis numbers.
Each candidate is scored on: H3/H5/H7 error rms over the 27 opto_harm_t* items by frequency, the static rms at threshold 20
(33 levels), the fine-knee rms (discriminate_opto.json (A), thresholds 20 and 10), the burst weighted rms (stage-4 weights over
the nine burst items) and the 3 kHz / 8 kHz static series.

The law section reads the 27 reference items directly: per position and frequency, the gain reduction, the mean conductance,
H2/H3/H5/H7, the relative conductance ripple implied by H3, the H5-H3 and H7-H3 ratios, the frequency slopes, and the pulse-train
Fourier table of the model's own light waveform, and states what any mechanism must produce.

usage: cd <repo> && python3 -u fit/tools/candidates/opto-ripple-loop-filter.py [--quick] [--v6]
The base model is the current fit in fit/data/constants.json (the stage-4 refit of docs/opto-fix.md section 3.3, which freed o_n and
the persistence and is what src/dsp/Opto.hpp runs); --v6 uses instead the V6 shape / knee table / low pass of the document, the
base of the first run of this harness (/tmp/opto_ripple_loop_final.log). Runs on the numba mirror of OptoStage::process (validated
against the C++ engine at the start of the run); nothing under src/, fit/stages/ or fit/data/ is touched.
"""
import json, os, sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS, DATA  # noqa: E402

np.set_printoptions(linewidth=220, suppress=True)
QUICK = "--quick" in sys.argv
FSF = float(FS)
DISC = json.load(open(os.path.join(DATA, "discriminate_opto.json")))
cal0 = load_cal()
def cget(name, i=0): return float(cal0[MODEL.fields[name][0] + i])
def cidx(name, i=0): return MODEL.fields[name][0] + i
G0 = cget("o_gain_db", 11) + cget("x_gain_db", 0)     # make-up at position 12 plus the Nickel path's midband gain, dB
G0_LIN = 10.0 ** (G0 / 20.0)
B2, B3 = cget("o_b2"), cget("o_b3")
LEVELS = list(range(-50, 15, 2))
POS = (14, 18, 22); LVLS = (-20, -10, 0); FREQS = (100, 1000, 4000)
HARM_IDS = [f"opto_harm_t{t}_{l}_f{f}" for t in POS for l in LVLS for f in FREQS]
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
HF_LEVELS = list(range(-40, 11, 4))

# the chosen model of docs/opto-fix.md: the V6 shape (section 2.3 raw vector), the 24-entry knee table (section 4.1), the matched
# low pass (section 5.2)
V6 = {"logC": 0.804515, "p": 1.657263, "log_tau_el": -4.177286, "w": (0.29804, 0.787815, 0.249402), "la": (-2.073795, -1.803165, -2.068737),
      "lr": (-0.756595, -0.879338, -1.027291), "logb": -1.694579, "q": 2.154234, "logmuc": 0.885802}
THR_DB_V6 = np.array([-9.4, -1.43, 6.27, 11.37, 15.54, 17.99, 20.26, 22.41, 24.53, 26.62, 28.63, 29.57,
                      30.56, 31.48, 32.46, 33.39, 34.37, 35.32, 36.30, 37.28, 38.29, 39.32, 40.40, 41.36])
LP_V6 = (5147.0, 0.718)
USE_V6 = "--v6" in sys.argv
THR_DB = THR_DB_V6 if USE_V6 else np.array(cal0[MODEL.field("o_thr_db")], dtype=float).copy()
LP = LP_V6 if USE_V6 else (cget("o_sc_lp_hz"), cget("o_sc_lp_q"))

# parameter vector layout for the kernel
(iVTH, iN, iGAM, iTAUEL, iW0, iW1, iW2, iTA0, iTA1, iTA2, iTR0, iTR1, iTR2, iMU, iB2, iB3, iFC, iQ, iAMPOUT, iEXTRA, iFZ, iFP, iTPOST, iTCOND, iMU2) = range(25)


class Shape:
    """the optical stage's parameters in the calibration's terms plus this harness's options"""
    def __init__(self, v6=USE_V6):
        if v6:
            C = 10.0 ** V6["logC"]; self.gam = V6["p"]; self.n = 1.0; self.vth = C ** (1.0 / self.gam)
            self.tau_el = 10.0 ** V6["log_tau_el"]
            w = np.abs(np.array(V6["w"])); self.w = w / w.sum()
            self.tatt = 10.0 ** np.array(V6["la"]); self.trel = 10.0 ** np.array(V6["lr"])
            self.leak = 10.0 ** V6["logb"]; self.leak_q = V6["q"]; self.mu = 10.0 ** V6["logmuc"]
        else:   # the current fit (fit/data/constants.json), field by field
            self.n = cget("o_n"); self.gam = cget("o_gamma"); self.vth = cget("o_vth"); self.tau_el = cget("o_tau_el")
            self.w = np.array([cget("o_w", i) for i in range(3)]); self.tatt = np.array([cget("o_tatt", i) for i in range(3)]); self.trel = np.array([cget("o_trel", i) for i in range(3)])
            self.leak = cget("o_leak"); self.leak_q = cget("o_leak_q"); self.mu = cget("o_rel_mu")
        self.thr = THR_DB.copy(); self.fc, self.Q = LP; self.b2, self.b3 = B2, B3
        self.amp_out = 0; self.extra = 0; self.fz = 1e6; self.fp = 1e6; self.tau_post = 0.0; self.tau_cond = 0.0; self.mu2 = 0.0
    @property
    def pexp(self): return self.n * self.gam     # the static law's exponent: cond ~ (d - vth)^(n gamma)

    def copy(self):
        s = Shape.__new__(Shape); s.__dict__.update(self.__dict__); s.thr = self.thr.copy(); s.w = self.w.copy(); return s
    def A(self, k): return 10.0 ** (self.thr[k - 1] / 20.0)
    def cond0(self, k): return self.leak * 10.0 ** (self.leak_q * (self.thr[k - 1] - self.thr[19]) / 20.0)
    def L0(self, k):
        c0 = self.cond0(k); return c0 ** (1.0 / self.gam) if c0 > 0.0 else 0.0
    def P(self):
        return np.array([self.vth, self.n, self.gam, self.tau_el, *self.w, *self.tatt, *self.trel, self.mu, self.b2, self.b3, self.fc, self.Q,
                         float(self.amp_out), float(self.extra), self.fz, self.fp, self.tau_post, self.tau_cond, self.mu2])

    def to_cal(self):
        """a calibration vector for the C++ engine (only the engine's own fields: the placement/extra options have no engine form)"""
        c = cal0.copy()
        c[MODEL.field("o_thr_db")] = self.thr
        c[cidx("o_n")] = self.n; c[cidx("o_gamma")] = self.gam; c[cidx("o_vth")] = self.vth; c[cidx("o_tau_el")] = self.tau_el
        for i in range(3): c[cidx("o_w", i)] = self.w[i]; c[cidx("o_tatt", i)] = self.tatt[i]; c[cidx("o_trel", i)] = self.trel[i]
        c[cidx("o_rel_mu")] = self.mu; c[cidx("o_leak")] = self.leak; c[cidx("o_leak_q")] = self.leak_q
        c[cidx("o_sc_lp_hz")] = self.fc; c[cidx("o_sc_lp_q")] = self.Q; c[cidx("o_b2")] = self.b2; c[cidx("o_b3")] = self.b3
        return c


# ------------------------------------------------------------------------------------------------ the mirror, with the options
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
def run_one(x, n, fs, A, L0, P, out, cond_out):
    """OptoStage::process (sidechain filter out, no hwUnit, no material options) with the options of this harness:
    P[iAMPOUT] 1: the amplifier terms act on the divider output instead of the stage input;
    P[iEXTRA] 1: a first-order lead/lag (1 + s/wz)/(1 + s/wp) in the sidechain after the low pass (bilinear, prewarped at fp);
    P[iTPOST] > 0: one-pole on the target after the cell law; P[iTCOND] > 0: one-pole on the conductance after the states;
    P[iMU2]: a quadratic term in the release quench, rate = (1 + mu s + mu2 s^2) / trel"""
    vth = P[iVTH]; nexp = P[iN]; gam = P[iGAM]; b2 = P[iB2]; b3 = P[iB3]; mu = P[iMU]; mu2 = P[iMU2]
    kEl = 1.0 - np.exp(-1.0 / (P[iTAUEL] * fs)) if P[iTAUEL] > 0.0 else 1.0
    kA0 = 1.0 - np.exp(-1.0 / (P[iTA0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (P[iTA1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (P[iTA2] * fs))
    r0 = 1.0 / (P[iTR0] * fs); r1 = 1.0 / (P[iTR1] * fs); r2 = 1.0 / (P[iTR2] * fs)
    w0 = P[iW0]; w1 = P[iW1]; w2 = P[iW2]
    lb0, lb1, lb2, la1, la2 = lp_matched(P[iFC], P[iQ], fs)
    amp_out = P[iAMPOUT] > 0.5; extra = P[iEXTRA] > 0.5
    eb0 = 1.0; eb1 = 0.0; ea1 = 0.0
    if extra:
        fp = min(P[iFP], 0.45 * fs); fz = P[iFZ]
        wp = 2.0 * np.pi * fp; wz = 2.0 * np.pi * fz; cc = wp / np.tan(np.pi * fp / fs)
        eb0 = (wp / wz) * (wz + cc) / (wp + cc); eb1 = (wp / wz) * (wz - cc) / (wp + cc); ea1 = (wp - cc) / (wp + cc)
    kPost = 1.0 - np.exp(-1.0 / (P[iTPOST] * fs)) if P[iTPOST] > 0.0 else 1.0
    kCond = 1.0 - np.exp(-1.0 / (P[iTCOND] * fs)) if P[iTCOND] > 0.0 else 1.0
    c0 = L0 ** gam
    z1 = 0.0; z2 = 0.0; L = 0.0; s0 = c0; s1 = c0; s2 = c0; cond = c0; Tf = c0; cf = c0; yprev = 0.0; y2prev = 0.0
    for i in range(n):
        xi = x[i]
        xa = xi if amp_out else xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        out[i] = v + b2 * v * v + b3 * v * v * v if amp_out else v
        y = lb0 * v + z1
        z1 = lb1 * v - la1 * y + z2
        z2 = lb2 * v - la2 * y
        if extra:
            y2 = eb0 * y + eb1 * yprev - ea1 * y2prev
            yprev = y; y2prev = y2; y = y2
        e = A * abs(y) - vth
        Linst = e ** nexp if e > 0.0 else 0.0
        L += (Linst - L) * kEl
        target = (L + L0) ** gam
        Tf += (target - Tf) * kPost
        target = Tf
        q0 = 1.0 - np.exp(-(1.0 + mu * s0 + mu2 * s0 * s0) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1 + mu2 * s1 * s1) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2 + mu2 * s2 * s2) * r2)
        s0 += (target - s0) * (kA0 if target > s0 else q0)
        s1 += (target - s1) * (kA1 if target > s1 else q1)
        s2 += (target - s2) * (kA2 if target > s2 else q2)
        craw = w0 * s0 + w1 * s1 + w2 * s2
        cf += (craw - cf) * kCond
        cond = cf
        cond_out[i] = cond


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, L0, P, out, cond_out):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], L0[j], P, out[j], cond_out[j])


@njit(cache=True, parallel=True)
def lockin_batch(Y, lens, fs, f, last_s, out):
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
    for j in prange(Y.shape[0]):
        n = lens[j]; per = fs / f; m = int(n / per); counts[j] = m
        for k in range(m):
            n0 = int(round(k * per)); n1 = int(round((k + 1) * per))
            yr = 0.0; yi = 0.0; xr = 0.0; xi = 0.0
            for i in range(n0, n1):
                cs = np.cos(2.0 * np.pi * f * i / fs); sn = np.sin(2.0 * np.pi * f * i / fs)
                yr += Y[j, i] * cs; yi -= Y[j, i] * sn; xr += X[j, i] * cs; xi -= X[j, i] * sn
            out[j, k] = 20.0 * np.log10(np.sqrt(yr * yr + yi * yi) / (np.sqrt(xr * xr + xi * xi) + 1e-30) + 1e-30)


class Batch:
    """a set of protocol items rendered together on the mirror"""
    def __init__(self, ids, secs=None):
        self.ids = ids
        xs = []
        for i in ids:
            st = dict(ITEMS[i]["stim"])
            if secs is not None and st["kind"] == "sine": st["secs"] = secs
            xs.append(protocol.stimulus(st, ITEMS[i]["fs"])[0])
        self.lens = np.array([len(x) for x in xs]); self.X = np.zeros((len(ids), self.lens.max()))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.Y = np.zeros_like(self.X); self.C = np.zeros_like(self.X)
        self.thr = np.array([int(ITEMS[i]["set"]["optical_threshold"]) for i in ids])
        self.f = float(ITEMS[ids[0]]["stim"]["f"]); self.kind = ITEMS[ids[0]]["feat"]["type"]

    def render(self, sh):
        A = np.array([sh.A(k) for k in self.thr]); L0 = np.array([sh.L0(k) for k in self.thr])
        run_batch(self.X, self.lens, FSF, A, L0, sh.P(), self.Y, self.C)

    def gains(self, sh):
        """steady gain of the fundamental (dB) as protocol 'gain_db', one frequency per batch"""
        self.render(sh); g = np.zeros(len(self.ids)); lockin_batch(self.Y, self.lens, FSF, self.f, 0.5, g)
        return g - np.array([ITEMS[i]["stim"]["level"] for i in self.ids], dtype=float) + G0

    def harms(self, sh):
        self.render(sh); out = []
        for j, i in enumerate(self.ids):
            y = self.Y[j, :self.lens[j]] * G0_LIN
            out.append(protocol.feature(ITEMS[i], np.stack([y, y]), ITEMS[i]["fs"]))
        return out

    def envs(self, sh):
        """per-period gain (dB) as protocol 'env'; the burst items share f = 1 kHz"""
        self.render(sh); E = np.zeros((len(self.ids), int(self.lens.max() / (FSF / self.f)) + 2)); cnt = np.zeros(len(self.ids), dtype=np.int64)
        env_batch(self.Y, self.X, self.lens, FSF, self.f, E, cnt)
        return [E[j, :cnt[j]] + G0 for j in range(len(self.ids))]


def rms(e): e = np.asarray(e, dtype=float); return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")
def amax(e): e = np.asarray(e, dtype=float); return float(np.max(np.abs(e))) if e.size else float("nan")


# ------------------------------------------------------------------------------------------------ the scoring set
class Score:
    def __init__(self):
        self.harm = Batch(HARM_IDS)
        self.stat20 = Batch([f"opto_static_t20_{l}" for l in LEVELS])
        self.stat3k = Batch([f"opto_static_f3000_{l}" for l in HF_LEVELS]); self.stat8k = Batch([f"opto_static_f8000_{l}" for l in HF_LEVELS])
        self.bursts = [Batch([i]) for i in BURSTS]
        self.knee = {}
        for thr in ("20", "10"):
            kd = DISC["knee"][thr]; n = int(3.0 * FS); t = np.arange(n) / FS
            X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in kd["levels"]])
            self.knee[thr] = (X, np.full(len(X), n, dtype=np.int64), np.zeros_like(X), np.zeros_like(X), np.array(kd["gain_db"]), np.array(kd["levels"]))
        self.ref_stat20 = np.array([F[i] for i in self.stat20.ids]); self.ref_3k = np.array([F[i] for i in self.stat3k.ids]); self.ref_8k = np.array([F[i] for i in self.stat8k.ids])

    def harm_table(self, sh):
        """model/ref harmonics for the 27 items: dict id -> (model feat, ref feat)"""
        return {i: (m, F[i]) for i, m in zip(HARM_IDS, self.harm.harms(sh))}

    def knee_rms(self, sh, thr):
        X, lens, Y, C, ref, lv = self.knee[thr]; k = int(thr)
        run_batch(X, lens, FSF, np.full(len(X), sh.A(k)), np.full(len(X), sh.L0(k)), sh.P(), Y, C)
        g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(X))]) + G0
        return rms(g - ref), float(np.max(np.abs(g - ref)))

    def burst_resid(self, sh, weighted=True):
        out = []
        for b in self.bursts:
            e = b.envs(sh)[0]; ref = np.asarray(F[b.ids[0]]); n = min(len(e), len(ref)); d = e[:n] - ref[:n]
            wt = np.where(ref[:n] < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)
            out.append(wt * d if weighted else d)
        return np.concatenate(out)

    def harm_resid(self, table, wg=2.0, w3=1.0, w5=0.5, w7=0.25, w5_4k=0.25):
        out = []
        for i, (m, r) in table.items():
            f = int(ITEMS[i]["stim"]["f"])
            out.append(wg * (m["gain_db"] - r["gain_db"]))
            for idx, wt in ((1, w3), (3, w5_4k if f == 4000 else w5), (5, w7)):
                if m["h"][idx] is None or r["h"][idx] is None or r["h"][idx] < -100: continue
                out.append(wt * (max(m["h"][idx], -110.0) - max(r["h"][idx], -110.0)))
        return np.array(out)

    def resid(self, sh):
        """the fit objective: harmonics (27 items), statics at t20 (x10), bursts (stage-4 weights x5), 3 kHz + 8 kHz statics (x1): the statics
        and bursts are weighted as constraints (0.1 dB of static error costs as much as 1 dB of H3), since the task holds them fixed"""
        return np.concatenate([self.harm_resid(self.harm_table(sh)), 10.0 * (self.stat20.gains(sh) - self.ref_stat20), 5.0 * self.burst_resid(sh),
                               self.stat3k.gains(sh) - self.ref_3k, self.stat8k.gains(sh) - self.ref_8k])

    def report(self, sh, label, full=True):
        table = self.harm_table(sh)
        err = {f: {2: [], 3: [], 5: [], 7: []} for f in FREQS}; g = []
        for i, (m, r) in table.items():
            f = int(ITEMS[i]["stim"]["f"]); g.append(m["gain_db"] - r["gain_db"])
            for h, idx in ((2, 0), (3, 1), (5, 3), (7, 5)):
                if m["h"][idx] is None or r["h"][idx] is None or r["h"][idx] < (-125 if h == 2 else -100): continue
                err[f][h].append(m["h"][idx] - r["h"][idx])
        s20 = self.stat20.gains(sh) - self.ref_stat20; e3 = self.stat3k.gains(sh) - self.ref_3k; e8 = self.stat8k.gains(sh) - self.ref_8k
        bw = self.burst_resid(sh); bp = self.burst_resid(sh, weighted=False)
        k20 = self.knee_rms(sh, "20"); k10 = self.knee_rms(sh, "10")
        allh = {h: np.concatenate([err[f][h] for f in FREQS]) for h in (3, 5, 7)}
        print(f"  [{label}]")
        print("     H3 err rms/max by 100/1k/4k: " + " | ".join(f"{rms(err[f][3]):.1f}/{amax(err[f][3]):.1f}" for f in FREQS) + f"  all {rms(allh[3]):.1f}")
        print("     H5 err rms/max by 100/1k/4k: " + " | ".join(f"{rms(err[f][5]):.1f}/{amax(err[f][5]):.1f}" for f in FREQS) + f"  all {rms(allh[5]):.1f}")
        print("     H7 err rms/max by 100/1k   : " + " | ".join(f"{rms(err[f][7]):.1f}/{amax(err[f][7]):.1f}" for f in FREQS[:2]) + f"  all {rms(allh[7]):.1f}")
        print("     H2 err rms/max by 100/1k/4k: " + " | ".join(f"{rms(err[f][2]):.1f}/{amax(err[f][2]):.1f}" for f in FREQS))
        print(f"     gain over 27 items rms {rms(g):.3f} max {np.max(np.abs(g)):.2f} | static t20 rms {rms(s20):.3f} max {np.max(np.abs(s20)):.2f} | "
              f"fine knee (A) t20 {k20[0]:.3f}/{k20[1]:.2f} t10 {k10[0]:.3f}/{k10[1]:.2f} | bursts weighted {rms(bw):.3f} plain {rms(bp):.3f}/{np.max(np.abs(bp)):.2f} | "
              f"3 kHz {rms(e3):.3f} 8 kHz {rms(e8):.3f}")
        if full:
            print("     item                    gain m/r     H3 m/r        H5 m/r        H7 m/r        H2 m/r")
            for i, (m, r) in table.items():
                def s(idx):
                    return "   n/a     " if m["h"][idx] is None or r["h"][idx] is None else f"{m['h'][idx]:6.1f}/{r['h'][idx]:6.1f}"
                print(f"     {i:24s} {m['gain_db']:+6.2f}/{r['gain_db']:+6.2f} {s(1)} {s(3)} {s(5)} {s(0)}")
        return dict(h3=rms(allh[3]), h5=rms(allh[5]), h7=rms(allh[7]), h3f={f: rms(err[f][3]) for f in FREQS}, h5f={f: rms(err[f][5]) for f in FREQS},
                    h7f={f: rms(err[f][7]) for f in FREQS[:2]}, gain=rms(g), stat20=rms(s20), knee20=k20[0], knee10=k10[0], burst=rms(bw), s3k=rms(e3), s8k=rms(e8))


# ------------------------------------------------------------------------------------------------ 0. mirror against the engine
def validate(sh):
    c = sh.to_cal()
    ids = ["opto_static_t20_-10", "opto_static_t20_2", "opto_static_t10_-20", "opto_harm_t18_-10_f1000", "opto_harm_t22_0_f100"]
    worst = 0.0
    for i in ids:
        b = Batch([i]); e = render_item(ITEMS[i], c)
        if b.kind == "harm":
            m = b.harms(sh)[0]; d = max(abs(m["gain_db"] - e["gain_db"]), abs(m["h"][1] - e["h"][1]), abs(m["h"][3] - e["h"][3]))
            print(f"   mirror vs engine {i}: gain {m['gain_db']:.3f}/{e['gain_db']:.3f} H3 {m['h'][1]:.2f}/{e['h'][1]:.2f} H5 {m['h'][3]:.2f}/{e['h'][3]:.2f} (max |diff| {d:.4f})")
        else:
            m = b.gains(sh)[0]; d = abs(m - e); print(f"   mirror vs engine {i}: {m:.4f}/{e:.4f} (|diff| {d:.4f} dB)")
        worst = max(worst, d)
    b = Batch(["opto_burst_-10"]); m = b.envs(sh)[0]; e = np.asarray(render_item(ITEMS["opto_burst_-10"], c)); n = min(len(m), len(e))
    d = float(np.max(np.abs(m[:n] - e[:n]))); worst = max(worst, d)
    print(f"   mirror vs engine opto_burst_-10 envelope: max |diff| {d:.4f} dB")
    return worst


# ------------------------------------------------------------------------------------------------ 1. the reference's ripple law
def flat_gain(k): return float(F[f"opto_static_t{k}_-50"])


def reference_law():
    print("\n1. THE REFERENCE RIPPLE LAW (27 opto_harm_t* items; GR = flat gain at -50 dBFS minus the item's gain; cbar = 10^(GR/20) - 1;")
    print("   a2 = 2 10^(H3/20) is the 2f ripple of the conductance relative to (1 + cbar), since v = x / (1 + c) and x sin(wt) a cos(2wt) -> a/2 at 3f)")
    print("   pos  lvl   f    gain    GR    20log c   H2     H3     H5     H7  | H2-H2nogr  a2=2e^H3  a2/(1+cbar)*... rel ripple dc/cbar | H5-H3  H7-H3")
    nogr = {l: F[f"opto_harm_nogr_{l}"]["h"][0] for l in (-20, -10, 0)}
    rows = []
    for k in POS:
        for l in LVLS:
            for f in FREQS:
                r = F[f"opto_harm_t{k}_{l}_f{f}"]; gr = flat_gain(k) - r["gain_db"]; cb = 10 ** (gr / 20) - 1
                h2, h3, h5, h7 = r["h"][0], r["h"][1], r["h"][3], r["h"][5]
                a2 = 2 * 10 ** (h3 / 20); rel = a2 * (1 + cb) / max(cb, 1e-9)
                rows.append(dict(k=k, l=l, f=f, gr=gr, cb=cb, h2=h2, h3=h3, h5=h5, h7=h7, a2=a2, rel=rel))
                h7s = f"{h7:6.1f}" if h7 is not None else "   n/a"
                d53 = f"{h5 - h3:6.1f}"; d73 = f"{h7 - h3:6.1f}" if h7 is not None else "   n/a"
                print(f"   {k:3d} {l:4d} {f:5d} {r['gain_db']:+6.2f} {gr:5.2f} {20 * np.log10(max(cb, 1e-9)):7.1f} {h2:6.1f} {h3:6.1f} {h5:6.1f} {h7s} | {h2 - nogr[l]:+6.1f}      {a2:.2e}   {rel:.4f} | {d53} {d73}")
    # (i) H2 under GR against the no-GR H2 at the same input level: even term on the output or on the input?
    print("\n   (i) even harmonic: H2 under GR minus the no-GR H2 at the same input level, against -GR (output placement predicts H2 = H2nogr - GR, input placement 0):")
    h2rms = {}
    for f in FREQS:
        d = [(r["h2"] - nogr[r["l"]], -r["gr"]) for r in rows if r["f"] == f and r["gr"] > 1.5]
        h2rms[f] = rms([a - b for a, b in d])
        print(f"       {f:5d} Hz: mean(H2 - H2nogr + GR) {np.mean([a - b for a, b in d]):+.2f} dB, rms about -GR {rms([a - b for a, b in d]):.2f}, rms about 0 {rms([a for a, b in d]):.2f}  (items with GR > 1.5 dB: {len(d)})")
    # (ii) frequency law per harmonic
    print("\n   (ii) frequency dependence (dB per octave) of H3, H5, H7 per position and level: 100 -> 1000 Hz (3.32 oct) and 1000 -> 4000 Hz (2 oct)")
    s3a, s3b, s5a, s5b, s7a = [], [], [], [], []
    for k in POS:
        for l in LVLS:
            R = {f: next(r for r in rows if r["k"] == k and r["l"] == l and r["f"] == f) for f in FREQS}
            if R[100]["gr"] < 1.5: continue
            a3 = (R[1000]["h3"] - R[100]["h3"]) / 3.32; b3 = (R[4000]["h3"] - R[1000]["h3"]) / 2.0
            a5 = (R[1000]["h5"] - R[100]["h5"]) / 3.32; b5 = (R[4000]["h5"] - R[1000]["h5"]) / 2.0
            a7 = (R[1000]["h7"] - R[100]["h7"]) / 3.32
            s3a.append(a3); s3b.append(b3); s5a.append(a5); s5b.append(b5); s7a.append(a7)
            print(f"       t{k} {l:+3d} dBFS (GR {R[1000]['gr']:5.2f}): H3 {a3:+.2f} / {b3:+.2f}   H5 {a5:+.2f} / {b5:+.2f}   H7 {a7:+.2f} / n/a   (GR change 1k->4k {R[4000]['gr'] - R[1000]['gr']:+.2f} dB)")
    print(f"       mean: H3 {np.mean(s3a):+.2f} / {np.mean(s3b):+.2f}, H5 {np.mean(s5a):+.2f} / {np.mean(s5b):+.2f}, H7 {np.mean(s7a):+.2f} dB/oct")
    law = dict(slope3a=np.mean(s3a), slope3b=np.mean(s3b), slope5a=np.mean(s5a), slope5b=np.mean(s5b), slope7a=np.mean(s7a), slope5b_min=min(s5b), slope5b_max=max(s5b), slope5a_sd=np.std(s5a))
    print("       note: at 4 kHz H5 sits at 20 kHz and H7 (28 kHz) would alias onto it at 48 kHz if the reference runs the stage without oversampling;")
    print("       the 1k -> 4k H5 slopes are therefore not trusted as a ripple law; the 100 -> 1000 Hz slopes are.")
    # (iii) level law per position and frequency
    print("\n   (iii) level dependence: slope of H3 / H5 / H7 (dBc) against 20 log10 cbar, per position and frequency (between the -20/-10 and -10/0 dBFS items)")
    for f in FREQS[:2]:
        for k in POS:
            R = [next(r for r in rows if r["k"] == k and r["l"] == l and r["f"] == f) for l in LVLS]
            x = [20 * np.log10(max(r["cb"], 1e-9)) for r in R]
            def sl(key, a, b): return (R[b][key] - R[a][key]) / (x[b] - x[a]) if R[a][key] is not None and R[b][key] is not None else float("nan")
            print(f"       {f:5d} Hz t{k}: 20log c = {x[0]:6.1f} {x[1]:6.1f} {x[2]:6.1f} | H3 slope {sl('h3', 0, 1):+.2f} {sl('h3', 1, 2):+.2f} | H5 {sl('h5', 0, 1):+.2f} {sl('h5', 1, 2):+.2f} | H7 {sl('h7', 0, 1):+.2f} {sl('h7', 1, 2):+.2f} dB/dB")
    # (iv) relative ripple against cbar: a power law fit over the compressing items
    print("\n   (iv) relative conductance ripple dc2f/cbar = 2 10^(H3/20) (1 + cbar) / cbar against cbar, items with GR > 1.5 dB:")
    for f in FREQS[:2]:
        pts = [(r["cb"], r["rel"], r["gr"]) for r in rows if r["f"] == f and r["gr"] > 1.5]
        X = np.log10([p[0] for p in pts]); Y = np.log10([p[1] for p in pts]); A = np.vstack([X, np.ones_like(X)]).T
        beta, c = np.linalg.lstsq(A, Y, rcond=None)[0]; res = Y - A @ [beta, c]
        print(f"       {f:5d} Hz: dc/cbar = {10 ** c:.4f} cbar^{beta:.2f}  (rms of log10 residual {rms(res):.3f} = {rms(res) * 20:.1f} dB); points (GR, dc/cbar): "
              + ", ".join(f"({p[2]:.1f}, {p[1]:.4f})" for p in sorted(pts, key=lambda p: p[2])))
        law[f"pl_{f}"] = (10 ** c, beta, rms(res) * 20)
    print("       a linear averager with a fixed time constant would give dc/cbar falling with GR (the pulse's 2f share of its mean falls from 2 at the knee")
    print("       to 0.67 for a full rectified sine); the measured dc/cbar RISES with cbar, so the cell's effective averaging time falls with conductance.")
    # (iv b) the effective integrator time constant the reference implies: dc2f/cbar = (T1/mean) / (2 pi 2f tau_eff), with T1/mean the 2f share
    # of the model's own pulse at the r the static law puts at that GR, against the model's quenched release constant trel / (1 + mu cbar)
    sh = Shape(); th = np.linspace(0, np.pi, 4001); pe = sh.pexp
    def gr_of_r(r):
        g = np.maximum(0.0, np.sin(th) - r) ** pe; return 20 * np.log10(1 + (1.0 / r) ** pe * trapz(g, th) / np.pi)
    def t1_share(r):
        g = np.maximum(0.0, np.sin(th) - r) ** pe; return abs(2 * trapz(g * np.cos(2 * th), th) / np.pi) / (trapz(g, th) / np.pi)
    def r_of_gr(gr):
        lo, hi = 0.001, 0.999
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if gr_of_r(mid) > gr: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)
    print("\n   (iv b) effective averaging time of the cell implied by the reference's 2f ripple, tau_eff = (T1/mean) / (4 pi f dc/cbar) with T1/mean the 2f share of")
    print("       the model's own pulse at the r the static law puts at that GR, against the model's quenched release constant trel_mean / (1 + mu cbar):")
    trel_mean = float(np.sum(sh.w * sh.trel)); taus = {}
    for f in FREQS[:2]:
        pts = sorted([r for r in rows if r["f"] == f], key=lambda r: r["gr"])
        line = []; taus[f] = []
        for r in pts:
            rr = r_of_gr(r["gr"]); tau = t1_share(rr) / (4 * np.pi * f * r["rel"]); tq = trel_mean / (1 + sh.mu * r["cb"])
            taus[f].append((r["gr"], tau, tq))
            line.append(f"GR {r['gr']:4.1f}: r {rr:.2f} T1/mean {t1_share(rr):.2f} tau_eff {tau * 1e3:5.1f} ms (model quench {tq * 1e3:5.1f} ms)")
        print(f"       {f:5d} Hz:\n         " + "\n         ".join(line))
    law["taus"] = taus
    law["tau_ratio_100_1k"] = float(np.max([abs(a[1] / b[1] - 1.0) for a, b in zip(taus[100], taus[1000])]))
    law["quench_ratio"] = (float(np.min([t / q for _, t, q in taus[100]])), float(np.max([t / q for _, t, q in taus[100]])))
    # (v) H5-H3 and H7-H3 against GR: the shape of the ripple
    print("\n   (v) ripple shape: H5-H3 and H7-H3 (dB) against GR at 100 Hz and 1 kHz, sorted by GR:")
    shape = {}
    for f in FREQS[:2]:
        pts = sorted([(r["gr"], r["h5"] - r["h3"], (r["h7"] - r["h3"]) if r["h7"] is not None else float("nan"), r["k"], r["l"]) for r in rows if r["f"] == f])
        print(f"       {f:5d} Hz: " + ", ".join(f"t{p[3]}/{p[4]:+d} GR {p[0]:.1f}: {p[1]:+.1f}/{p[2]:+.1f}" for p in pts))
        d5 = min(pts, key=lambda p: p[1]); d7 = min((p for p in pts if not np.isnan(p[2])), key=lambda p: p[2])
        shape[f] = dict(knee53=pts[0][1], knee73=pts[0][2], dip5=(d5[0], d5[1]), dip7=(d7[0], d7[2]), top53=pts[-1][1], top73=pts[-1][2])
    law["shape"] = shape; law["h2rms"] = h2rms
    a = [p[1] for p in sorted([(r["gr"], r["h5"] - r["h3"]) for r in rows if r["f"] == 100])]; b = [p[1] for p in sorted([(r["gr"], r["h5"] - r["h3"]) for r in rows if r["f"] == 1000])]
    law["shape_100_vs_1k"] = float(np.max(np.abs(np.array(a) - np.array(b))))
    return rows, law


trapz = getattr(np, "trapezoid", None) or np.trapz


def pulse_table(gam):
    """Fourier coefficients of the model's own target waveform g(theta) = max(0, sin theta - r)^gam over a half cycle: T_k at 2kf relative to T_1,
    with the 1/f integration of a linear averager (-6 dB at 4f, -9.5 dB at 6f), and the GR the static law puts at each r (vth = 1, leak ignored)"""
    print(f"\n   (vi) the model's light pulse train, target = max(0, d sin(theta) - vth)^{gam:.3f} (n gamma), r = vth / d: harmonic shares and the GR of the static law at that r")
    print("       r      duty   GR(dB)  T1/mean  T2/T1 dB  T3/T1 dB | predicted H5-H3  H7-H3 (linear averager, 1/f)")
    th = np.linspace(0, np.pi, 20001)
    null = None; prev = None
    for r in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        g = np.maximum(0.0, np.sin(th) - r) ** gam
        mean = trapz(g, th) / np.pi
        T = [2 * trapz(g * np.cos(2 * k * th), th) / np.pi for k in (1, 2, 3)]
        d = 1.0 / r; cb = d ** gam * mean; gr = 20 * np.log10(1 + cb)
        duty = 1 - 2 * np.arcsin(r) / np.pi
        t2 = 20 * np.log10(abs(T[1]) / abs(T[0])); t3 = 20 * np.log10(abs(T[2]) / abs(T[0]))
        if prev is not None and np.sign(T[1]) != np.sign(prev[1]): null = (prev[0], r, prev[2], gr)
        prev = (r, T[1], gr)
        print(f"       {r:.2f}  {duty:.3f}  {gr:6.2f}  {abs(T[0]) / mean:6.3f}  {t2:+7.1f}  {t3:+7.1f} | {t2 - 6.0:+7.1f} {t3 - 9.5:+7.1f}   (sign T2 {'+' if T[1] > 0 else '-'})")
    if null: print(f"       the 4f (T2) null lies between r {null[0]:.2f} and {null[1]:.2f}, GR {null[2]:.1f} to {null[3]:.1f} dB with vth = 1 drive units (GR is independent of vth's value)")
    return null


def state_law(law, null, model_null):
    """the law any mechanism must produce, stated from the numbers computed above (law: reference_law's dict; null: pulse_table's 4f null
    of the pulse train; model_null: (GR, 4f/2f dB) of the rendered base model's deepest conductance 4f null over the nine 1 kHz items)"""
    t = {gr: tau for gr, tau, _ in law["taus"][100]}; grs = sorted(t); pick = [grs[0], grs[1], grs[3], grs[-1]]
    s1 = law["shape"][1000]; s0 = law["shape"][100]; pl = law["pl_1000"]
    print("\n   THE LAW ANY MECHANISM MUST PRODUCE (from (i)-(vi), all numbers computed above):")
    print(f"     1. one 1/f integrator and nothing else between the light and the divider from 100 Hz to 4 kHz: H3 / H5 / H7 fall {-law['slope3a']:.1f} / {-law['slope5a']:.1f} / {-law['slope7a']:.1f} dB per octave")
    print(f"        from 100 Hz to 1 kHz and H3 keeps {-law['slope3b']:.1f} dB/oct to 4 kHz, so no pole below about 10 kHz acts on the light or the conductance;")
    print(f"     2. the integrator's time constant falls with conductance: tau_eff " + " / ".join(f"{t[g] * 1e3:.1f}" for g in pick) + " ms at " + " / ".join(f"{g:.1f}" for g in pick) + " dB of GR, the same at 100 Hz")
    print(f"        and 1 kHz to {law['tau_ratio_100_1k'] * 100:.0f} %, i.e. dc/cbar = {pl[0]:.4f} cbar^{pl[1]:.2f} (1 kHz / f) to {pl[2]:.1f} dB; the base model's quenched release (1 + mu c) / trel gives tau_eff / quench between {law['quench_ratio'][0]:.2f} and {law['quench_ratio'][1]:.2f};")
    print(f"     3. the ripple's harmonic shape is a property of the waveform, not of a filter: H5-H3 is the same at 100 Hz and 1 kHz within {law['shape_100_vs_1k']:.1f} dB at every GR;")
    print(f"        at 1 kHz H5-H3 is {s1['knee53']:+.0f} dB at the knee (a narrow pulse), dips to {s1['dip5'][1]:+.0f} dB at {s1['dip5'][0]:.1f} dB of GR and recovers to {s1['top53']:+.0f} dB at 18 dB ({s0['dip5'][1]:+.0f} at {s0['dip5'][0]:.1f} dB, {s0['top53']:+.0f} at 100 Hz);")
    print(f"        H7-H3 is {s1['knee73']:+.0f} at the knee, dips to {s1['dip7'][1]:+.0f} at {s1['dip7'][0]:.1f} dB of GR and recovers to {s1['top73']:+.0f}; a fixed-time smoother (a quarter cycle at 1 kHz")
    print("        is 0.25 ms, nothing at 100 Hz) cannot produce a cycle-locked shape, so whatever moves the dip must scale with the period;")
    if null:
        print(f"     4. the base model's own pulse train with a linear averager puts the 4f null at {null[3]:.0f}-{null[2]:.0f} dB of GR and its rendered loop (asymmetric states) has its deepest 4f/2f")
        print(f"        ({model_null[1]:+.0f} dB) at {model_null[0]:.1f} dB of GR; the reference's dip is at {s1['dip5'][0]:.1f} dB (1 kHz) / {s0['dip5'][0]:.1f} dB (100 Hz) and only {s1['dip5'][1]:+.0f} dB deep: the mechanism must move the null and fill it, not remove it;")
    print(f"     5. the even harmonics are the downstream amplifier's: H2 = H2(no GR, same input) - GR to {law['h2rms'][100]:.1f} / {law['h2rms'][1000]:.1f} / {law['h2rms'][4000]:.1f} dB rms at 100 / 1k / 4k,")
    print("        so the b2 term acts on the divider output and takes no part in the ripple;")
    print(f"     6. the 4 kHz H5 rows (20 kHz) do not follow any law the other rows follow (1k -> 4k slopes {-law['slope5b_max']:.1f} to {-law['slope5b_min']:.1f} dB/oct against {-law['slope5a']:.1f} +- {law['slope5a_sd']:.1f} for")
    print("        100 -> 1k) and should not be fitted as ripple.")


# ------------------------------------------------------------------------------------------------ 2. candidates
def fit_candidate(sc, base, label, names, x0, lo, hi, xs, apply, nfev=40, alt_starts=None):
    """fits the named parameters plus a global shift of the drive table (dthr, dB: the whole knee table moves together, the allowed
    shift family), since removing the persistence pole moves the static knee (the pole smooths the light before the convex cell law)"""
    t0 = time.time(); print(f"\n   {label}:")
    names = list(names) + ["dthr"]; x0 = list(x0) + [base.thr[19] - THR_DB[19]]; lo = list(lo) + [-4.0]; hi = list(hi) + [4.0]; xs = list(xs) + [0.2]
    def resid(p):
        s = base.copy(); apply(s, p[:-1]); s.thr = THR_DB + p[-1]; return sc.resid(s)
    starts = [x0] + [list(a) + [x0[-1]] for a in (alt_starts or [])]
    r = None
    for st in starts:
        rr = least_squares(resid, st, bounds=(lo, hi), x_scale=xs, diff_step=2e-3, max_nfev=nfev)
        print(f"     start {np.round(st, 3).tolist()} -> cost {rr.cost:.2f} at {np.round(rr.x, 4).tolist()} (nfev {rr.nfev})")
        if r is None or rr.cost < r.cost: r = rr
    s = base.copy(); apply(s, r.x[:-1]); s.thr = THR_DB + r.x[-1]
    print(f"   {label}: " + ", ".join(f"{n} {v:.5g}" for n, v in zip(names, r.x)) + f" | cost {r.cost:.3f} (start {0.5 * np.sum(resid(np.array(x0)) ** 2):.1f}), nfev {r.nfev}, {time.time() - t0:.0f} s")
    return s, r


def main():
    T0 = time.time()
    print(f"G0 {G0:.3f} dB (make-up 12 + Nickel), b2 {B2:.2e} b3 {B3:.2e}; engine cal layout {MODEL.layout_hash}; base = {'V6 shape of docs/opto-fix.md' if USE_V6 else 'fit/data/constants.json (stage-4 refit)'}")
    base = Shape()
    print(f"base shape: n {base.n:.3f} vth {base.vth:.3f} gamma {base.gam:.3f} tau_el {base.tau_el * 1e3:.4f} ms w {np.round(base.w, 3).tolist()} tatt ms {np.round(base.tatt * 1e3, 2).tolist()} "
          f"trel ms {np.round(base.trel * 1e3, 1).tolist()} mu {base.mu:.2f} leak {base.leak:.4f} q {base.leak_q:.3f} LP {base.fc:.0f} Hz Q {base.Q:.3f}")
    print("\n0. mirror against the C++ engine (calibration vector built from the base shape):")
    worst = validate(base)
    print(f"   worst |diff| {worst:.4f} dB" + ("" if worst < 0.05 else "  !! the mirror and the engine disagree (the engine may be mid-edit); the mirror is used as the reference model here"))
    rows, law = reference_law()
    null = pulse_table(base.pexp)
    sc = Score()
    print("\n2. BASELINE: the base model on the mirror (nothing refitted)")
    R = {}
    R["baseline"] = sc.report(base, f"baseline: {'V6 + knee table + LP 5147/0.718' if USE_V6 else 'constants.json'}", full=True)
    # the model's own conductance ripple on three 1 kHz items: is the H5 null a null of the conductance's 4f component?
    print("\n   model conductance ripple (last 1 s, lock-in of cond at 2f/4f/6f relative to its mean) on the nine 1 kHz items, and the same from the reference's H3/H5/H7:")
    b = Batch([f"opto_harm_t{k}_{l}_f1000" for k in POS for l in LVLS]); b.render(base); model_null = (0.0, 0.0)
    for j, i in enumerate(b.ids):
        c = b.C[j, -FS:]; t = np.arange(FS) / FSF; cb = c.mean()
        comps = [abs(2 * np.mean((c - cb) * np.exp(-2j * np.pi * 1000.0 * m * t))) for m in (2, 4, 6)]
        r = F[i]; grr = flat_gain(int(ITEMS[i]["set"]["optical_threshold"])) - r["gain_db"]; cbr = 10 ** (grr / 20) - 1
        ref2 = 2 * 10 ** (r["h"][1] / 20) * (1 + cbr)
        if 20 * np.log10(comps[1] / comps[0]) < model_null[1]: model_null = (grr, 20 * np.log10(comps[1] / comps[0]))
        print(f"     {i:24s} model cbar {cb:.3f} dc 2f/4f/6f {comps[0]:.2e} {comps[1]:.2e} {comps[2]:.2e} (4f/2f {20 * np.log10(comps[1] / comps[0]):+.1f} dB) | "
              f"ref cbar {cbr:.3f} dc2f {ref2:.2e} (4f/2f from H5-H3 {r['h'][3] - r['h'][1]:+.1f} dB)")
    state_law(law, null, model_null)
    # which knob of the loop moves the 4f null? single-knob perturbations, H5-H3 at the nine 1 kHz items (model) against the reference
    print("\n   SWEEP: which knob of the loop moves the 4f null? H5-H3 (dB) at the nine 1 kHz items (GR-sorted: t14/-20, t18/-20, t22/-20, t14/-10, t18/-10, t22/-10, t14/0, t18/0, t22/0),")
    print("   single-knob perturbations of the baseline, nothing refitted; the last columns are the 1 kHz H3 / H5 error rms and the static t20 rms")
    ids9 = sorted([f"opto_harm_t{k}_{l}_f1000" for k in POS for l in LVLS], key=lambda i: flat_gain(int(ITEMS[i]["set"]["optical_threshold"])) - F[i]["gain_db"])
    b9 = Batch(ids9)
    def row9(sh, label):
        hs = b9.harms(sh); d = [m["h"][3] - m["h"][1] for m in hs]
        e3 = rms([m["h"][1] - F[i]["h"][1] for i, m in zip(ids9, hs)]); e5 = rms([m["h"][3] - F[i]["h"][3] for i, m in zip(ids9, hs)])
        s20 = rms(sc.stat20.gains(sh) - sc.ref_stat20)
        print(f"     {label:44s} " + " ".join(f"{v:6.1f}" for v in d) + f" | {e3:4.1f} {e5:4.1f} | {s20:.3f}")
    print(f"     {'reference':44s} " + " ".join(f"{F[i]['h'][3] - F[i]['h'][1]:6.1f}" for i in ids9))
    row9(base, "baseline")
    for lab, fn in (("tau_el 0.01 ms", lambda s: setattr(s, "tau_el", 1e-5)),
                    ("mu x 0.3", lambda s: setattr(s, "mu", base.mu * 0.3)), ("mu x 3", lambda s: setattr(s, "mu", base.mu * 3.0)),
                    ("mu2 = 1 (quadratic quench)", lambda s: setattr(s, "mu2", 1.0)), ("mu2 = 5", lambda s: setattr(s, "mu2", 5.0)),
                    ("attack x 0.3 (all states)", lambda s: setattr(s, "tatt", base.tatt * 0.3)), ("attack x 3", lambda s: setattr(s, "tatt", base.tatt * 3.0)),
                    ("release x 0.3 (all states)", lambda s: setattr(s, "trel", base.trel * 0.3)), ("release x 3", lambda s: setattr(s, "trel", base.trel * 3.0)),
                    ("gamma x 0.85", lambda s: setattr(s, "gam", base.gam * 0.85)), ("gamma x 1.15", lambda s: setattr(s, "gam", base.gam * 1.15)),
                    ("n x 1.3, gamma / 1.3 (same static law)", lambda s: (setattr(s, "n", base.n * 1.3), setattr(s, "gam", base.gam / 1.3))),
                    ("n x 0.7, gamma / 0.7 (same static law)", lambda s: (setattr(s, "n", base.n * 0.7), setattr(s, "gam", base.gam / 0.7))),
                    ("n 1, gamma = n gamma (same static law)", lambda s: (setattr(s, "n", 1.0), setattr(s, "gam", base.pexp))),
                    ("vth x 0.5, table -6.02 dB (same knee level)", lambda s: (setattr(s, "vth", base.vth * 0.5), setattr(s, "thr", THR_DB - 20 * np.log10(2.0)))),
                    ("vth x 2, table +6.02 dB (same knee level)", lambda s: (setattr(s, "vth", base.vth * 2.0), setattr(s, "thr", THR_DB + 20 * np.log10(2.0)))),
                    ("amplifier on the output", lambda s: setattr(s, "amp_out", 1))):
        s = base.copy(); fn(s); row9(s, lab)
    # (e) the amplifier placement
    print("\n3. (e) THE AMPLIFIER TERMS ON THE DIVIDER OUTPUT (the H2 rows), no other change:")
    e_out = base.copy(); e_out.amp_out = 1
    R["amp_out"] = sc.report(e_out, "b2, b3 on the divider output", full=False)
    # controls and candidates
    nfev = 12 if QUICK else 40
    print("\n4. CANDIDATES, each fitted on the same objective (27 harmonic items H3 w1 / H5 w0.5 (4 kHz 0.25) / H7 w0.25 / gain w2, statics t20 x2, bursts stage-4 weights, 3 kHz + 8 kHz statics)")
    def ap_tau(s, p): s.tau_el = 10.0 ** p[0]
    ctrl, _ = fit_candidate(sc, base, "control: tau_el alone", ["log_tau_el"], [np.log10(base.tau_el)], [-6.0], [-3.0], [0.1], ap_tau, nfev)
    R["control"] = sc.report(ctrl, f"control tau_el {ctrl.tau_el * 1e3:.4f} ms", full=False)
    ctrl_out = ctrl.copy(); ctrl_out.amp_out = 1
    R["control_out"] = sc.report(ctrl_out, "control + amplifier on the output", full=False)
    # d1: Q, then fc + Q
    def ap_q(s, p): s.tau_el = 10.0 ** p[0]; s.Q = p[1]
    c1, _ = fit_candidate(sc, ctrl, "(d1) low pass Q + tau_el", ["log_tau_el", "Q"], [np.log10(ctrl.tau_el), ctrl.Q], [-6.0, 0.3], [-3.0, 3.0], [0.1, 0.1], ap_q, nfev)
    R["d1_Q"] = sc.report(c1, f"(d1) Q {c1.Q:.3f}, tau_el {c1.tau_el * 1e3:.4f} ms", full=False)
    def ap_fq(s, p): s.tau_el = 10.0 ** p[0]; s.fc = p[1]; s.Q = p[2]
    c1b, _ = fit_candidate(sc, ctrl, "(d1) low pass fc + Q + tau_el", ["log_tau_el", "fc", "Q"], [np.log10(ctrl.tau_el), ctrl.fc, ctrl.Q], [-6.0, 1500.0, 0.3], [-3.0, 20000.0, 3.0], [0.1, 500.0, 0.1], ap_fq, nfev)
    R["d1_fcQ"] = sc.report(c1b, f"(d1) fc {c1b.fc:.0f} Hz Q {c1b.Q:.3f}, tau_el {c1b.tau_el * 1e3:.4f} ms", full=False)
    # d2: an added real pole (zero at infinity)
    def ap_pole(s, p): s.tau_el = 10.0 ** p[0]; s.extra = 1; s.fz = 1e7; s.fp = 10.0 ** p[1]
    c2, _ = fit_candidate(sc, ctrl, "(d2) added sidechain pole + tau_el", ["log_tau_el", "log_fp"], [np.log10(ctrl.tau_el), np.log10(8000.0)], [-6.0, 2.5], [-3.0, 4.3], [0.1, 0.1], ap_pole, nfev)
    R["d2_pole"] = sc.report(c2, f"(d2) pole {c2.fp:.0f} Hz, tau_el {c2.tau_el * 1e3:.4f} ms", full=False)
    # d3: lead/lag: zero and pole
    def ap_ll(s, p): s.tau_el = 10.0 ** p[0]; s.extra = 1; s.fz = 10.0 ** p[1]; s.fp = 10.0 ** p[2]
    c3, _ = fit_candidate(sc, ctrl, "(d3) sidechain lead/lag + tau_el", ["log_tau_el", "log_fz", "log_fp"], [np.log10(ctrl.tau_el), np.log10(3000.0), np.log10(9000.0)],
                          [-6.0, 2.0, 2.0], [-3.0, 4.3, 4.3], [0.1, 0.1, 0.1], ap_ll, nfev)
    R["d3_leadlag"] = sc.report(c3, f"(d3) zero {c3.fz:.0f} Hz pole {c3.fp:.0f} Hz, tau_el {c3.tau_el * 1e3:.4f} ms", full=False)
    # d4: persistence after the cell law (pre-cell pole removed)
    def ap_post(s, p): s.tau_el = 1e-6; s.tau_post = 10.0 ** p[0]
    c4, _ = fit_candidate(sc, ctrl, "(d4) pole on the target after the cell law (no pre-cell pole)", ["log_tau_post"], [np.log10(6.6e-5)], [-6.0], [-2.5], [0.1], ap_post, nfev)
    R["d4_post"] = sc.report(c4, f"(d4) tau_post {c4.tau_post * 1e3:.4f} ms", full=False)
    def ap_both(s, p): s.tau_el = 10.0 ** p[0]; s.tau_post = 10.0 ** p[1]
    c4b, _ = fit_candidate(sc, ctrl, "(d4b) pre-cell + post-cell poles", ["log_tau_el", "log_tau_post"], [np.log10(ctrl.tau_el), np.log10(6.6e-5)], [-6.0, -6.0], [-3.0, -2.5], [0.1, 0.1], ap_both, nfev)
    R["d4b_both"] = sc.report(c4b, f"(d4b) tau_el {c4b.tau_el * 1e3:.4f} ms + tau_post {c4b.tau_post * 1e3:.4f} ms", full=False)
    # d5: pole on the conductance after the states
    def ap_cond(s, p): s.tau_el = 1e-6; s.tau_cond = 10.0 ** p[0]
    c5, _ = fit_candidate(sc, ctrl, "(d5) pole on the conductance after the states (no pre-cell pole)", ["log_tau_cond"], [np.log10(6.6e-5)], [-6.0], [-2.5], [0.1], ap_cond, nfev)
    R["d5_cond"] = sc.report(c5, f"(d5) tau_cond {c5.tau_cond * 1e3:.4f} ms", full=False)
    # d6, d7: the loop's own dynamics (the states are the ripple integrator): quench, attack scale, quadratic quench, no pre-cell pole
    def ap_dyn(s, p): s.tau_el = 1e-5; s.mu = 10.0 ** p[0]; s.tatt = base.tatt * 10.0 ** p[1]
    c6, _ = fit_candidate(sc, ctrl, "(d6) loop dynamics: mu + attack scale (tau_el 0.01 ms)", ["log_mu", "log_att_scale"], [np.log10(base.mu), 0.0], [-1.0, -1.5], [3.0, 1.5], [0.1, 0.1], ap_dyn, nfev,
                          alt_starts=[[np.log10(base.mu * 3), 0.0], [np.log10(base.mu), np.log10(3.0)], [np.log10(base.mu * 3), np.log10(3.0)]])
    R["d6_dyn"] = sc.report(c6, f"(d6) mu {c6.mu:.2f}, attack x {c6.tatt[0] / base.tatt[0]:.3f}", full=False)
    def ap_dyn2(s, p): s.tau_el = 1e-5; s.mu = 10.0 ** p[0]; s.tatt = base.tatt * 10.0 ** p[1]; s.mu2 = 10.0 ** p[2]
    c7, _ = fit_candidate(sc, ctrl, "(d7) loop dynamics: mu + attack scale + quadratic quench mu2", ["log_mu", "log_att_scale", "log_mu2"], [np.log10(base.mu), 0.0, -1.0], [-1.0, -1.5, -3.0], [3.0, 1.5, 2.0], [0.1, 0.1, 0.2], ap_dyn2, nfev,
                          alt_starts=[[np.log10(base.mu), 0.0, 0.0], [np.log10(base.mu), np.log10(3.0), 0.0], [np.log10(base.mu * 3), np.log10(3.0), -1.0]])
    R["d7_dyn2"] = sc.report(c7, f"(d7) mu {c7.mu:.2f}, attack x {c7.tatt[0] / base.tatt[0]:.3f}, mu2 {c7.mu2:.3f}", full=True)
    # d8, d9: the persistence pole at its floor with the shape refitted (gamma, vth, dthr), then with the loop dynamics as well: the fair
    # "same objective" references for a model without the 2.4 kHz pole, since that pole also shaped the above-knee static law
    def ap_shape(s, p): s.tau_el = 1e-5; s.gam = p[0]; s.vth = 10.0 ** p[1]
    c8, _ = fit_candidate(sc, base, "(d8) no pre-cell pole, shape refit: gamma + vth", ["gamma", "log_vth"], [base.gam, np.log10(base.vth)], [0.7 * base.gam, np.log10(base.vth) - 0.3], [1.6 * base.gam, np.log10(base.vth) + 0.3], [0.05, 0.05], ap_shape, nfev)
    R["d8_shape"] = sc.report(c8, f"(d8) gamma {c8.gam:.3f} vth {c8.vth:.3f}, tau_el 0.01 ms", full=False)
    def ap_shape_dyn(s, p): s.tau_el = 1e-5; s.gam = p[0]; s.vth = 10.0 ** p[1]; s.mu = 10.0 ** p[2]; s.tatt = base.tatt * 10.0 ** p[3]
    c9, _ = fit_candidate(sc, base, "(d9) no pre-cell pole, shape + loop dynamics: gamma + vth + mu + attack scale", ["gamma", "log_vth", "log_mu", "log_att_scale"],
                          [c8.gam, np.log10(c8.vth), np.log10(base.mu), 0.0], [0.7 * base.gam, np.log10(base.vth) - 0.3, -1.0, -1.5], [1.6 * base.gam, np.log10(base.vth) + 0.3, 3.0, 1.5], [0.05, 0.05, 0.1, 0.1], ap_shape_dyn, nfev,
                          alt_starts=[[c8.gam, np.log10(c8.vth), np.log10(base.mu * 3), np.log10(2.0)]])
    R["d9_shape_dyn"] = sc.report(c9, f"(d9) gamma {c9.gam:.3f} vth {c9.vth:.3f} mu {c9.mu:.2f} attack x {c9.tatt[0] / base.tatt[0]:.3f}, tau_el 0.01 ms", full=True)
    print("     static t20 model-ref by level -50..14:", np.round(sc.stat20.gains(c9) - sc.ref_stat20, 2).tolist())
    print(f"     static slope from the law alone p/(1+p), p = n gamma = {c9.pexp:.3f}: {c9.pexp / (1 + c9.pexp):.3f}; the states' attack/release asymmetry changes the effective slope, which is why gamma can move")
    # the best of the loop-filter family with the amplifier on the output, full table
    best_key = min((k for k in R if k not in ("baseline", "amp_out")), key=lambda k: R[k]["h3"] ** 2 + 0.25 * R[k]["h5"] ** 2 + 0.0625 * R[k]["h7"] ** 2)
    print(f"\n5. best of the family by the harmonic objective: {best_key}")
    print("   summary (rms dB): key | H3 all (100/1k/4k) | H5 all (100/1k/4k) | H7 all (100/1k) | gain | static t20 | knee t20/t10 | bursts w | 3k / 8k")
    for k, v in R.items():
        print(f"   {k:12s} | {v['h3']:4.1f} ({v['h3f'][100]:.1f}/{v['h3f'][1000]:.1f}/{v['h3f'][4000]:.1f}) | {v['h5']:4.1f} ({v['h5f'][100]:.1f}/{v['h5f'][1000]:.1f}/{v['h5f'][4000]:.1f}) | "
              f"{v['h7']:4.1f} ({v['h7f'][100]:.1f}/{v['h7f'][1000]:.1f}) | {v['gain']:.3f} | {v['stat20']:.3f} | {v['knee20']:.3f}/{v['knee10']:.3f} | {v['burst']:.3f} | {v['s3k']:.3f}/{v['s8k']:.3f}")
    print(f"\n   total {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
