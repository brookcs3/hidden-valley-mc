# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical ripple synthesis, key "opto-ripple-synthesis": the check behind docs/opto-ripple-fix.md.

The three ripple harnesses (opto-ripple-pre-smooth.py, opto-ripple-asym-phosphor.py, opto-ripple-loop-filter.py) each scored their
candidates against the current fit/data/constants.json on the same numba mirror of OptoStage::process. This harness puts the pieces
the synthesis adopts together and scores them on everything the task lists, from runs of its own:
  1. the mirror against the C++ engine (current constants);
  2. the baseline on the mirror at 48 kHz, on the mirror with the loop at 96 kHz, and through the engine in HQ 2X (quality 1): the 27
     opto_harm_t* items, so the 4 kHz rows can be read with and without the 48 kHz loop's aliasing of the light pulse;
  3. (e) the amplifier terms b2, b3 on the divider output instead of the stage input (the H2 rows under gain reduction);
  4. the loop's own state law, refitted under a knee-constrained objective (statics t20 x10, fine knee sweeps (A) x10, bursts x5,
     3 kHz + 8 kHz statics x1, harmonics: gain x2, H3 x1, H5 x0.5 except the 4 kHz H5 rows at 0, H7 x0.25), with (e) in place:
       F1  gamma, vth, mu, an attack scale and the knee shift (the loop-filter harness's d9k refit, redone here with (e));
       F2  a free exponent on the release quench, rate = (1 + mu s^rho) / trel, with mu and the knee shift (rho = 1 is the model);
       F3  F1 + rho.
  5. for every variant: H3/H5/H7/H2 error rms over the 27 items by frequency, the static rms at t20 (33 levels) and t12, the 100 Hz
     series at t20, the fine knees (A) at 20 and 10, the nine burst envelopes (stage-4 weights), the 3 kHz / 8 kHz series, the no-GR
     rows (D) and the above-knee steps (E) (release 50/90 %, tail at 0.5 s).
Nothing under src/, fit/stages/ or fit/data/ is written.
usage: cd <repo> && python3 -u fit/tools/candidates/opto-ripple-synthesis.py [--quick] [--nofit]
"""
import json, os, sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS, DATA  # noqa: E402

np.set_printoptions(linewidth=220, suppress=True)
QUICK = "--quick" in sys.argv; NOFIT = "--nofit" in sys.argv
FSF = float(FS)
DISC = json.load(open(os.path.join(DATA, "discriminate_opto.json")))
cal0 = load_cal()
def cget(name, i=0): return float(cal0[MODEL.fields[name][0] + i])
def cidx(name, i=0): return MODEL.fields[name][0] + i
G0 = cget("o_gain_db", 11) + cget("x_gain_db", 0)     # make-up at position 12 plus the Nickel path's midband gain, dB
G0_LIN = 10.0 ** (G0 / 20.0)
LEVELS = list(range(-50, 15, 2))
POS = (14, 18, 22); LVLS = (-20, -10, 0); FREQS = (100, 1000, 4000)
HARM_IDS = [f"opto_harm_t{t}_{l}_f{f}" for t in POS for l in LVLS for f in FREQS]
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
HF_LEVELS = list(range(-40, 11, 4))
THR_DB = np.array(cal0[MODEL.field("o_thr_db")], dtype=float).copy()

(iVTH, iN, iGAM, iTAUEL, iW0, iW1, iW2, iTA0, iTA1, iTA2, iTR0, iTR1, iTR2, iMU, iB2, iB3, iFC, iQ, iAMPOUT, iRHO) = range(20)


class Shape:
    """the optical stage's parameters in the calibration's terms plus this harness's two options: the amplifier placement and the
    quench exponent rho"""
    def __init__(self):
        self.n = cget("o_n"); self.gam = cget("o_gamma"); self.vth = cget("o_vth"); self.tau_el = cget("o_tau_el")
        self.w = np.array([cget("o_w", i) for i in range(3)]); self.tatt = np.array([cget("o_tatt", i) for i in range(3)]); self.trel = np.array([cget("o_trel", i) for i in range(3)])
        self.leak = cget("o_leak"); self.leak_q = cget("o_leak_q"); self.mu = cget("o_rel_mu")
        self.thr = THR_DB.copy(); self.fc, self.Q = cget("o_sc_lp_hz"), cget("o_sc_lp_q"); self.b2, self.b3 = cget("o_b2"), cget("o_b3")
        self.amp_out = 0; self.rho = 1.0

    def copy(self):
        s = Shape.__new__(Shape); s.__dict__.update(self.__dict__); s.thr = self.thr.copy(); s.w = self.w.copy(); s.tatt = self.tatt.copy(); s.trel = self.trel.copy(); return s
    def A(self, k): return 10.0 ** (self.thr[k - 1] / 20.0)
    def cond0(self, k): return self.leak * 10.0 ** (self.leak_q * (self.thr[k - 1] - self.thr[19]) / 20.0)
    def L0(self, k):
        c0 = self.cond0(k); return c0 ** (1.0 / self.gam) if c0 > 0.0 else 0.0
    def P(self):
        return np.array([self.vth, self.n, self.gam, self.tau_el, *self.w, *self.tatt, *self.trel, self.mu, self.b2, self.b3, self.fc, self.Q, float(self.amp_out), self.rho])
    def describe(self):
        return (f"n {self.n:.3f} gamma {self.gam:.4f} vth {self.vth:.3f} tau_el {self.tau_el * 1e6:.1f} us | w {np.round(self.w, 3).tolist()} attack ms {np.round(self.tatt * 1e3, 1).tolist()} "
                f"release ms {np.round(self.trel * 1e3, 1).tolist()} mu {self.mu:.3f} rho {self.rho:.3f} | thr20 {self.thr[19]:+.3f} dB (shift {self.thr[19] - THR_DB[19]:+.3f}) | amp {'output' if self.amp_out else 'input'}")


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
def run_one(x, n, fs, A, L0, P, out, cond_out):
    """OptoStage::process (sidechain filter out, no hwUnit, no material options): divider v = xa / (1 + cond); sidechain low pass;
    d = A |s|; light (d - vth)^n above the turn-on; persistence; target = (L + L0)^gam; three states, linear attack, release rate
    (1 + mu s^rho) / trel. P[iAMPOUT] 1: the amplifier terms act on the divider output (the sidechain sees the clean divider output)"""
    vth = P[iVTH]; nexp = P[iN]; gam = P[iGAM]; b2 = P[iB2]; b3 = P[iB3]; mu = P[iMU]; rho = P[iRHO]
    kEl = 1.0 - np.exp(-1.0 / (P[iTAUEL] * fs))
    kA0 = 1.0 - np.exp(-1.0 / (P[iTA0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (P[iTA1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (P[iTA2] * fs))
    r0 = 1.0 / (P[iTR0] * fs); r1 = 1.0 / (P[iTR1] * fs); r2 = 1.0 / (P[iTR2] * fs)
    w0 = P[iW0]; w1 = P[iW1]; w2 = P[iW2]
    lb0, lb1, lb2, la1, la2 = lp_matched(P[iFC], P[iQ], fs)
    amp_out = P[iAMPOUT] > 0.5
    uf = 1.0 / np.sqrt(-3.0 * b3) if b3 < 0.0 else 1e30
    c0 = L0 ** gam
    z1 = 0.0; z2 = 0.0; L = 0.0; s0 = c0; s1 = c0; s2 = c0; cond = c0
    for i in range(n):
        xi = x[i]
        if amp_out:
            v = xi / (1.0 + cond)
            vc = v if v < uf else uf
            vc = vc if vc > -uf else -uf
            out[i] = vc + b2 * vc * vc + b3 * vc * vc * vc
        else:
            xc = xi if xi < uf else uf
            xc = xc if xc > -uf else -uf
            v = (xc + b2 * xc * xc + b3 * xc * xc * xc) / (1.0 + cond)
            out[i] = v
        y = lb0 * v + z1
        z1 = lb1 * v - la1 * y + z2
        z2 = lb2 * v - la2 * y
        e = A * abs(y) - vth
        Linst = e ** nexp if e > 0.0 else 0.0
        L += (Linst - L) * kEl
        target = (L + L0) ** gam
        if rho == 1.0:
            q0 = 1.0 - np.exp(-(1.0 + mu * s0) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2) * r2)
        else:
            q0 = 1.0 - np.exp(-(1.0 + mu * s0 ** rho) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1 ** rho) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2 ** rho) * r2)
        s0 += (target - s0) * (kA0 if target > s0 else q0)
        s1 += (target - s1) * (kA1 if target > s1 else q1)
        s2 += (target - s2) * (kA2 if target > s2 else q2)
        cond = w0 * s0 + w1 * s1 + w2 * s2
        cond_out[i] = cond


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, L0, P, out, cond_out):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], L0[j], P, out[j], cond_out[j])


@njit(cache=True, parallel=True)
def lockin_batch(Y, lens, fs, freqs, last_s, out):
    """gain of the fundamental (dB re unit amplitude) over the last last_s seconds, whole periods, as protocol 'gain_db'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; f = freqs[j]; per = fs / f
        nper = max(1, int(round(last_s * f)))
        n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
        re = 0.0; im = 0.0
        for i in range(n0, n1):
            t = i / fs
            re += Y[j, i] * np.cos(2.0 * np.pi * f * t); im -= Y[j, i] * np.sin(2.0 * np.pi * f * t)
        out[j] = 20.0 * np.log10(2.0 * np.sqrt(re * re + im * im) / (n1 - n0) + 1e-30)


@njit(cache=True, parallel=True)
def harm_batch(Y, lens, fs, freqs, last_s, out):
    """H1..H8 complex amplitudes over the last last_s seconds (whole periods), as protocol 'harm'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; f = freqs[j]; per = fs / f
        nper = max(4, int(round(last_s * f)))
        n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
        m = n1 - n0
        for k in range(1, 9):
            re = 0.0; im = 0.0
            if k * f < fs / 2 * 0.95:
                for i in range(n0, n1):
                    t = i / fs
                    re += Y[j, i] * np.cos(2.0 * np.pi * k * f * t); im -= Y[j, i] * np.sin(2.0 * np.pi * k * f * t)
            out[j, k - 1] = 2.0 * (re + 1j * im) / m


@njit(cache=True, parallel=True)
def env_batch(Y, X, lens, fs, f, out, counts):
    """gain of the fundamental per period (lock-in of output over input, hop one period), dB, as protocol 'env'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; per = fs / f; m = int(n / per); counts[j] = m
        for k in range(m):
            n0 = int(round(k * per)); n1 = int(round((k + 1) * per))
            yr = 0.0; yi = 0.0; xr = 0.0; xi = 0.0
            for i in range(n0, n1):
                cs = np.cos(2.0 * np.pi * f * i / fs); sn = np.sin(2.0 * np.pi * f * i / fs)
                yr += Y[j, i] * cs; yi -= Y[j, i] * sn; xr += X[j, i] * cs; xi -= X[j, i] * sn
            out[j, k] = 20.0 * np.log10(np.sqrt(yr * yr + yi * yi) / (np.sqrt(xr * xr + xi * xi) + 1e-30) + 1e-30)


def rms(e): e = np.asarray(e, dtype=float); e = e[~np.isnan(e)]; return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")
def amax(e): e = np.asarray(e, dtype=float); e = e[~np.isnan(e)]; return float(np.max(np.abs(e))) if e.size else float("nan")


class Batch:
    """protocol items rendered together on the mirror; fs overrides the loop's sample rate (stimulus and features at that rate)"""
    def __init__(self, ids, fs=None):
        self.ids = ids; self.fs = float(fs if fs is not None else ITEMS[ids[0]]["fs"])
        xs = [protocol.stimulus(ITEMS[i]["stim"], int(self.fs))[0] for i in ids]
        self.lens = np.array([len(x) for x in xs]); self.X = np.zeros((len(ids), self.lens.max()))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.Y = np.zeros_like(self.X); self.C = np.zeros_like(self.X)
        self.thr = [int(ITEMS[i]["set"]["optical_threshold"]) for i in ids]
        self.freqs = np.array([float(ITEMS[i]["stim"]["f"]) for i in ids])
        self.levels = np.array([float(ITEMS[i]["stim"]["level"]) for i in ids])

    def render(self, sh):
        A = np.array([sh.A(k) for k in self.thr]); L0 = np.array([sh.L0(k) for k in self.thr])
        run_batch(self.X, self.lens, self.fs, A, L0, sh.P(), self.Y, self.C)

    def gains(self, sh, last_s=0.5):
        self.render(sh); g = np.zeros(len(self.ids)); lockin_batch(self.Y, self.lens, self.fs, self.freqs, last_s, g)
        return g - self.levels + G0

    def harms(self, sh, last_s=1.0):
        """gain_db and H2..H8 in dBc per item (None above 0.95 Nyquist), as protocol.feature 'harm'"""
        self.render(sh)
        H = np.zeros((len(self.ids), 8), dtype=np.complex128); harm_batch(self.Y, self.lens, self.fs, self.freqs, last_s, H)
        out = []
        for j, i in enumerate(self.ids):
            a = np.abs(H[j]); hs = [None if k * self.freqs[j] >= self.fs / 2 * 0.95 else float(20 * np.log10(a[k - 1] / (a[0] + 1e-30) + 1e-30)) for k in range(2, 9)]
            out.append({"gain_db": float(20 * np.log10(a[0] * G0_LIN + 1e-30) - self.levels[j]), "h": hs})
        return out

    def envs(self, sh):
        self.render(sh); E = np.zeros((len(self.ids), int(self.lens.max() / (self.fs / 1000.0)) + 2)); cnt = np.zeros(len(self.ids), dtype=np.int64)
        env_batch(self.Y, self.X, self.lens, self.fs, 1000.0, E, cnt)
        return [E[j, :cnt[j]] + G0 for j in range(len(self.ids))]


def burst_weight(iid, n):
    ref = np.asarray(F[iid])[:n]
    return np.where(ref < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)


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
        if (g_end < g_start and v <= target) or (g_end > g_start and v >= target): return i
    return None


# ------------------------------------------------------------------------------------------------ the scoring set
class Score:
    def __init__(self):
        self.harm = Batch(HARM_IDS)
        self.stat20 = Batch([f"opto_static_t20_{l}" for l in LEVELS]); self.stat12 = Batch([f"opto_static_t12_{l}" for l in LEVELS])
        self.stat3k = Batch([f"opto_static_f3000_{l}" for l in HF_LEVELS]); self.stat8k = Batch([f"opto_static_f8000_{l}" for l in HF_LEVELS])
        self.s100 = Batch([f"opto_static_f100_{l}" for l in range(-40, 11, 2)])
        self.bursts = Batch(BURSTS + ["opto_burst_thr12"])
        self.knee = {}
        for thr in ("20", "10"):
            kd = DISC["knee"][thr]; n = int(3.0 * FS); t = np.arange(n) / FS
            X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in kd["levels"]])
            self.knee[thr] = (X, np.full(len(X), n, dtype=np.int64), np.zeros_like(X), np.zeros_like(X), np.array(kd["gain_db"]), np.array(kd["levels"]))
        keys = list(DISC["steps"].keys())
        xs = [step_signal(DISC["steps"][k]["from"], DISC["steps"][k]["to"]) for k in keys]
        self.step_keys = keys; self.step_X = np.array(xs); self.step_lens = np.full(len(xs), len(xs[0]), dtype=np.int64)
        self.step_Y = np.zeros_like(self.step_X); self.step_C = np.zeros_like(self.step_X)
        n2 = int(2.0 * FS); t2 = np.arange(n2) / FS
        self.nogr_ks = [int(k) for k in DISC["nogr"]["-50.0"]]; self.nogr_ref = np.array([DISC["nogr"]["-50.0"][str(k)] for k in self.nogr_ks])
        self.nogr_X = np.array([10 ** (-50.0 / 20) * np.sin(2 * np.pi * 1000.0 * t2) for _ in self.nogr_ks]); self.nogr_lens = np.full(len(self.nogr_ks), n2, dtype=np.int64)
        self.nogr_Y = np.zeros_like(self.nogr_X); self.nogr_C = np.zeros_like(self.nogr_X)
        self.ref = {b: np.array([F[i] for i in b.ids]) for b in (self.stat20, self.stat12, self.stat3k, self.stat8k, self.s100)}

    def knee_resid(self, sh):
        out = []
        for thr, (X, lens, Y, C, ref, lv) in self.knee.items():
            k = int(thr); run_batch(X, lens, FSF, np.full(len(X), sh.A(k)), np.full(len(X), sh.L0(k)), sh.P(), Y, C)
            g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -FS:] ** 2)) / np.sqrt(np.mean(X[j, -FS:] ** 2))) for j in range(len(X))]) + G0
            out.append(g - ref)
        return out

    def burst_resid(self, sh):
        envs = self.bursts.envs(sh); w = []; p = []; per = []
        for i, m in zip(self.bursts.ids, envs):
            ref = np.asarray(F[i]); n = min(len(m), len(ref)); d = m[:n] - ref[:n]
            if i in BURSTS: w.append(burst_weight(i, n) * d); p.append(d)
            per.append((i, rms(d), amax(d)))
        return np.concatenate(w), np.concatenate(p), per

    def harm_err(self, sh):
        """per item: gain err, H3 err, H5 err, H7 err, H2 err (nan where the reference is absent or below -100 dBc; H2 below -125);
        the odd harmonics of the model are floored at -110 dBc (a null is a null), H2 is not (the reference's H2 sits at -109 to -122)"""
        ms = self.harm.harms(sh); rows = []
        for i, m in zip(HARM_IDS, ms):
            r = F[i]; e = [m["gain_db"] - r["gain_db"]]
            for idx, floor, mfloor in ((1, -100.0, -110.0), (3, -100.0, -110.0), (5, -100.0, -110.0), (0, -125.0, -200.0)):
                a, b = m["h"][idx], r["h"][idx]
                e.append(np.nan if (a is None or b is None or b < floor) else max(a, mfloor) - b)
            rows.append(e)
        return np.array(rows), ms

    def resid(self, sh, knee_w=10.0):
        E, _ = self.harm_err(sh)
        f4k = np.array([i.endswith("_f4000") for i in HARM_IDS])
        w5 = np.where(f4k, 0.0, 0.5)
        r = [2.0 * E[:, 0], np.nan_to_num(E[:, 1]), w5 * np.nan_to_num(E[:, 2]), 0.25 * np.nan_to_num(E[:, 3]),
             10.0 * (self.stat20.gains(sh) - self.ref[self.stat20]), 5.0 * self.burst_resid(sh)[0],
             self.stat3k.gains(sh) - self.ref[self.stat3k], self.stat8k.gains(sh) - self.ref[self.stat8k]]
        if knee_w > 0.0: r += [knee_w * k for k in self.knee_resid(sh)]
        return np.concatenate(r)

    def report(self, sh, label, items=True):
        E, ms = self.harm_err(sh)
        print(f"\n== {label}\n   {sh.describe()}")
        byf = {}
        for f in FREQS:
            sel = np.array([i.endswith(f"_f{f}") for i in HARM_IDS])
            byf[f] = [rms(E[sel, c]) for c in (1, 2, 3, 4)] + [amax(E[sel, 1]), amax(E[sel, 2])]
        allh = [rms(E[:, c]) for c in (1, 2, 3, 4)]
        print("   H3 err rms/max by 100/1k/4k: " + " | ".join(f"{byf[f][0]:.2f}/{byf[f][4]:.1f}" for f in FREQS) + f"  all {allh[0]:.2f}")
        print("   H5 err rms/max by 100/1k/4k: " + " | ".join(f"{byf[f][1]:.2f}/{byf[f][5]:.1f}" for f in FREQS) + f"  all {allh[1]:.2f}")
        print("   H7 err rms by 100/1k       : " + " | ".join(f"{byf[f][2]:.2f}" for f in FREQS[:2]) + f"  all {allh[2]:.2f}")
        print("   H2 err rms by 100/1k/4k    : " + " | ".join(f"{byf[f][3]:.2f}" for f in FREQS) + f"  all {allh[3]:.2f}")
        e20 = self.stat20.gains(sh) - self.ref[self.stat20]; e12 = self.stat12.gains(sh) - self.ref[self.stat12]
        e3 = self.stat3k.gains(sh) - self.ref[self.stat3k]; e8 = self.stat8k.gains(sh) - self.ref[self.stat8k]; e100 = self.s100.gains(sh) - self.ref[self.s100]
        bw, bp, per = self.burst_resid(sh)
        kn = self.knee_resid(sh)
        # knee width and top slope
        kdesc = []
        for (thr, (X, lens, Y, C, ref, lv)), e in zip(self.knee.items(), kn):
            g = ref + e; grm = g[0] - g; grr = ref[0] - ref
            cm = [float(np.interp(v, grm, lv)) for v in (0.5, 3.0)]; cr = [float(np.interp(v, grr, lv)) for v in (0.5, 3.0)]
            sm = (g[-1] - g[-5]) / (lv[-1] - lv[-5]); sr = (ref[-1] - ref[-5]) / (lv[-1] - lv[-5])
            kdesc.append(f"t{thr} rms {rms(e):.3f} max {amax(e):.2f} (0.5 dB at {cm[0]:.2f}, ref {cr[0]:.2f}; width 0.5->3 {cm[1] - cm[0]:.2f}, ref {cr[1] - cr[0]:.2f}; top slope {sm:.3f}, ref {sr:.3f})")
        print(f"   gain over 27 items rms {rms(E[:, 0]):.3f} max {amax(E[:, 0]):.2f} | static t20 rms {rms(e20):.3f} max {amax(e20):.2f} | t12 {rms(e12):.3f}/{amax(e12):.2f} | 100 Hz at t20 {rms(e100):.3f}/{amax(e100):.2f} | 3 kHz {rms(e3):.3f} 8 kHz {rms(e8):.3f} (mirror: no Nickel shelf)")
        print("   fine knee (A): " + " | ".join(kdesc))
        print(f"   bursts (nine items): weighted rms {rms(bw):.4f} max {amax(bw):.2f} | plain rms {rms(bp):.3f} max {amax(bp):.2f} | per item rms/max: " + ", ".join(f"{i} {r:.3f}/{m:.2f}" for i, r, m in per))
        # (D) no-GR rows at -50 dBFS
        run_batch(self.nogr_X, self.nogr_lens, FSF, np.array([sh.A(k) for k in self.nogr_ks]), np.array([sh.L0(k) for k in self.nogr_ks]), sh.P(), self.nogr_Y, self.nogr_C)
        g = np.array([20 * np.log10(np.sqrt(np.mean(self.nogr_Y[j, -FS:] ** 2)) / np.sqrt(np.mean(self.nogr_X[j, -FS:] ** 2))) for j in range(len(self.nogr_ks))]) + G0
        print(f"   (D) no-GR at -50 dBFS, positions {self.nogr_ks}: model-ref {np.round(g - self.nogr_ref, 3).tolist()} max {amax(g - self.nogr_ref):.3f}")
        # (E) steps at t20
        run_batch(self.step_X, self.step_lens, FSF, np.full(len(self.step_keys), sh.A(20)), np.full(len(self.step_keys), sh.L0(20)), sh.P(), self.step_Y, self.step_C)
        steps = {}
        for j, key in enumerate(self.step_keys):
            v = DISC["steps"][key]; ref = np.array(v["per_cycle_gain_db"]); gc = per_cycle_gain(self.step_X[j], self.step_Y[j] * G0_LIN)
            n1, n2 = 2000, 4000; rows = []
            for g_ in (gc, ref):
                up = g_[n1:n2]; down = g_[n2:]
                ga = float(np.mean(g_[n1 - 200:n1 - 1])); gb = float(np.mean(g_[n2 - 200:n2 - 1])); ga2 = float(np.mean(g_[-200:]))
                rows.append((ga, gb, ga2, t_to(0.5, up, ga, gb), t_to(0.9, up, ga, gb), t_to(0.5, down, gb, ga2), t_to(0.9, down, gb, ga2), down[100] - ga2, down[500] - ga2))
            mm, rr = rows; d = gc[:len(ref)] - ref[:len(gc)]; rel = d[n2:]
            steps[key] = (rms(rel), amax(rel))
            print(f"   (E) {v['from']:+.0f}->{v['to']:+.0f}->{v['from']:+.0f}: GR {mm[0] - mm[1]:.2f} (ref {rr[0] - rr[1]:.2f}) | attack 50/90 % {mm[3]}/{mm[4]} ms (ref {rr[3]}/{rr[4]}) | release 50/90 % {mm[5]}/{mm[6]} ms (ref {rr[5]}/{rr[6]}) | "
                  f"tail at 100/500 ms {mm[7]:+.3f}/{mm[8]:+.3f} (ref {rr[7]:+.3f}/{rr[8]:+.3f}) | release trajectory rms {rms(rel):.3f} max {amax(rel):.2f}")
        if items:
            print("   item                    gain m/r     H3 m/r        H5 m/r        H7 m/r        H2 m/r")
            for i, m in zip(HARM_IDS, ms):
                r = F[i]
                def s(idx): return "   n/a     " if m["h"][idx] is None or r["h"][idx] is None else f"{m['h'][idx]:6.1f}/{r['h'][idx]:6.1f}"
                print(f"   {i:24s} {m['gain_db']:+6.2f}/{r['gain_db']:+6.2f} {s(1)} {s(3)} {s(5)} {s(0)}")
        return dict(h3=allh[0], h5=allh[1], h7=allh[2], h2=allh[3], h3f={f: byf[f][0] for f in FREQS}, h5f={f: byf[f][1] for f in FREQS}, h7f={f: byf[f][2] for f in FREQS[:2]},
                    gain=rms(E[:, 0]), stat20=rms(e20), stat12=rms(e12), s100=rms(e100), knee20=rms(kn[0]), knee10=rms(kn[1]), burst=rms(bw), burstp=rms(bp), s3k=rms(e3), s8k=rms(e8),
                    steps=steps)


def harm_summary(ms, label):
    """H3 / H5 / H7 error rms by frequency from a list of 27 harm features (model), and the 4 kHz H5 rows by item"""
    E = {f: [[], [], []] for f in FREQS}; line = []
    for i, m in zip(HARM_IDS, ms):
        r = F[i]; f = int(ITEMS[i]["stim"]["f"])
        for k, idx in enumerate((1, 3, 5)):
            if m["h"][idx] is not None and r["h"][idx] is not None and r["h"][idx] > -100: E[f][k].append(max(m["h"][idx], -110.0) - r["h"][idx])
        if f == 4000 and r["h"][3] > -100: line.append(f"{i[10:-6]} {m['h'][3]:.1f}/{r['h'][3]:.1f}")
    print(f"   {label}: " + " | ".join(f"{f} Hz H3 {rms(E[f][0]):.2f} H5 {rms(E[f][1]):.2f} H7 {rms(E[f][2]):.2f}" for f in FREQS))
    print("      4 kHz H5 model/ref by item: " + ", ".join(line))
    return E


# ------------------------------------------------------------------------------------------------ fits
def fit(sc, base, label, names, x0, lo, hi, xs, apply, nfev, starts=None):
    t0 = time.time(); print(f"\n-- fit {label}: parameters {names} + dthr (the whole drive table shifts together)")
    names = list(names) + ["dthr"]; lo = list(lo) + [-4.0]; hi = list(hi) + [4.0]; xs = list(xs) + [0.2]
    def unpack(p):
        s = base.copy(); apply(s, p[:-1]); s.thr = THR_DB + p[-1]; return s
    best = None
    for st in ([list(x0) + [0.0]] + [list(a) for a in (starts or [])]):
        c0 = 0.5 * float(np.sum(sc.resid(unpack(np.array(st))) ** 2))
        r = least_squares(lambda p: sc.resid(unpack(p)), np.array(st), bounds=(lo, hi), x_scale=xs, diff_step=2e-3, max_nfev=nfev)
        print(f"   start {np.round(st, 4).tolist()}: cost {c0:.1f} -> {r.cost:.2f} at {np.round(r.x, 4).tolist()} (nfev {r.nfev}) [{time.time() - t0:.0f} s]")
        if best is None or r.cost < best.cost: best = r
    s = unpack(best.x)
    print(f"   {label}: " + ", ".join(f"{n} {v:.5g}" for n, v in zip(names, best.x)) + f" | cost {best.cost:.2f}")
    return s, best


def validate(sh):
    c = cal0.copy()
    ids = ["opto_static_t20_-10", "opto_static_t20_2", "opto_static_t10_-20", "opto_harm_t18_-10_f1000", "opto_harm_t22_0_f100", "opto_harm_t14_0_f4000"]
    worst = 0.0
    for i in ids:
        b = Batch([i]); e = render_item(ITEMS[i], c)
        if ITEMS[i]["feat"]["type"] == "harm":
            m = b.harms(sh)[0]; d = max(abs(m["gain_db"] - e["gain_db"]), abs(m["h"][1] - e["h"][1]), abs(m["h"][3] - e["h"][3]), abs(m["h"][0] - e["h"][0]))
            print(f"   mirror vs engine {i}: gain {m['gain_db']:.3f}/{e['gain_db']:.3f} H2 {m['h'][0]:.2f}/{e['h'][0]:.2f} H3 {m['h'][1]:.2f}/{e['h'][1]:.2f} H5 {m['h'][3]:.2f}/{e['h'][3]:.2f} (max |diff| {d:.4f})")
        else:
            m = b.gains(sh)[0]; d = abs(m - e); print(f"   mirror vs engine {i}: {m:.4f}/{e:.4f} (|diff| {d:.4f} dB)")
        worst = max(worst, d)
    b = Batch(["opto_burst_-10"]); m = b.envs(sh)[0]; e = np.asarray(render_item(ITEMS["opto_burst_-10"], c)); n = min(len(m), len(e))
    d = float(np.max(np.abs(m[:n] - e[:n]))); worst = max(worst, d)
    print(f"   mirror vs engine opto_burst_-10 envelope: max |diff| {d:.4f} dB")
    return worst


def main():
    T0 = time.time()
    base = Shape()
    print(f"G0 {G0:.3f} dB (make-up 12 + Nickel); engine cal layout {MODEL.layout_hash}; {os.cpu_count()} cpus\nbase (fit/data/constants.json): {base.describe()}")
    print("\n0. mirror against the C++ engine (current constants, 48 kHz, quality 0):")
    worst = validate(base)
    print(f"   worst |diff| {worst:.4f} dB" + ("" if worst < 0.05 else "  !! the mirror and the engine disagree (the engine may be mid-edit); the mirror is what is scored below"))
    sc = Score()
    R = {}
    print("\n1. BASELINE on the mirror at 48 kHz")
    R["baseline"] = sc.report(base, "baseline: constants.json, amplifier on the stage input, loop at 48 kHz", items=True)
    print("\n2. THE 4 kHz ROWS AND THE LOOP'S RATE (27 items; the reference's 4 kHz H5 sits at 20 kHz)")
    harm_summary(sc.harm.harms(base), "mirror, loop at 48 kHz")
    b96 = Batch(HARM_IDS, fs=96000); harm_summary(b96.harms(base), "mirror, loop at 96 kHz (stimulus and features at 96 kHz)")
    ms = [render_item(ITEMS[i], cal0, quality=1) for i in HARM_IDS]; harm_summary(ms, "C++ engine, HQ 2X (quality 1: the channel at 96 kHz between the half-band pair)")
    ms0 = [render_item(ITEMS[i], cal0, quality=0) for i in HARM_IDS]; harm_summary(ms0, "C++ engine, STANDARD (quality 0)")
    print(f"   [{time.time() - T0:.0f} s]")
    print("\n3. (e) THE AMPLIFIER TERMS ON THE DIVIDER OUTPUT (no other change)")
    eo = base.copy(); eo.amp_out = 1
    R["amp_out"] = sc.report(eo, "(e) b2, b3 on the divider output, before the make-up", items=True)
    print("   the same at 96 kHz:"); harm_summary(b96.harms(eo), "mirror, loop at 96 kHz, amplifier on the output")
    if not NOFIT:
        nfev = 8 if QUICK else 30
        print("\n4. THE STATE LAW UNDER THE KNEE-CONSTRAINED OBJECTIVE, with (e) in place (statics t20 x10, knee (A) x10, bursts x5, 3k/8k x1; gain x2, H3 x1, H5 x0.5 (4 kHz H5 x0), H7 x0.25)")
        r0 = sc.resid(eo); print(f"   cost of (e) as is: {0.5 * float(np.sum(r0 ** 2)):.2f} ({len(r0)} residuals)")
        def ap1(s, p): s.gam = p[0]; s.vth = 10.0 ** p[1]; s.mu = 10.0 ** p[2]; s.tatt = base.tatt * 10.0 ** p[3]
        f1, _ = fit(sc, eo, "F1 gamma + vth + mu + attack scale", ["gamma", "log_vth", "log_mu", "log_att"], [base.gam, np.log10(base.vth), np.log10(base.mu), 0.0],
                    [0.9, np.log10(base.vth) - 0.3, -1.0, -1.5], [2.2, np.log10(base.vth) + 0.3, 3.0, 1.5], [0.05, 0.05, 0.1, 0.1], ap1, nfev,
                    starts=[[1.3918, 0.81318, 0.91019, 0.091131, 0.19701]])
        R["F1"] = sc.report(f1, "F1: gamma, vth, mu, attack scale refit (+ knee shift), amplifier on the output", items=False)
        def ap2(s, p): s.rho = p[0]; s.mu = 10.0 ** p[1]
        f2, _ = fit(sc, eo, "F2 quench exponent rho + mu", ["rho", "log_mu"], [1.0, np.log10(base.mu)], [0.2, -1.0], [1.6, 3.0], [0.05, 0.1], ap2, nfev,
                    starts=[[0.6, np.log10(base.mu) + 0.2, 0.0]])
        R["F2"] = sc.report(f2, "F2: release rate (1 + mu s^rho) / trel, rho and mu fitted (+ knee shift), amplifier on the output", items=False)
        def ap3(s, p): s.rho = p[0]; s.gam = p[1]; s.vth = 10.0 ** p[2]; s.mu = 10.0 ** p[3]; s.tatt = base.tatt * 10.0 ** p[4]
        f3, _ = fit(sc, eo, "F3 rho + gamma + vth + mu + attack scale", ["rho", "gamma", "log_vth", "log_mu", "log_att"],
                    [0.8, f1.gam, np.log10(f1.vth), np.log10(f1.mu), np.log10(f1.tatt[0] / base.tatt[0])],
                    [0.2, 0.9, np.log10(base.vth) - 0.3, -1.0, -1.5], [1.6, 2.2, np.log10(base.vth) + 0.3, 3.0, 1.5], [0.05, 0.05, 0.05, 0.1, 0.1], ap3, nfev,
                    starts=[[f2.rho, base.gam, np.log10(base.vth), np.log10(f2.mu), 0.0, f2.thr[19] - THR_DB[19]]])
        R["F3"] = sc.report(f3, "F3: rho + gamma, vth, mu, attack scale (+ knee shift), amplifier on the output", items=True)
        for k, s in (("F1", f1), ("F2", f2), ("F3", f3)): print(f"   {k} at 96 kHz:"); harm_summary(b96.harms(s), f"mirror, loop at 96 kHz, {k}")
    print("\n5. SUMMARY (rms dB): key | H3 all (100/1k/4k) | H5 all (100/1k/4k) | H7 all (100/1k) | H2 all | gain | static t20 / t12 / 100 Hz | knee t20/t10 | bursts w/plain | 3k/8k | steps release rms (three steps)")
    for k, v in R.items():
        st = "/".join(f"{v['steps'][key][0]:.3f}" for key in v["steps"])
        print(f"   {k:9s} | {v['h3']:4.2f} ({v['h3f'][100]:.1f}/{v['h3f'][1000]:.1f}/{v['h3f'][4000]:.1f}) | {v['h5']:4.2f} ({v['h5f'][100]:.1f}/{v['h5f'][1000]:.1f}/{v['h5f'][4000]:.1f}) | "
              f"{v['h7']:4.2f} ({v['h7f'][100]:.1f}/{v['h7f'][1000]:.1f}) | {v['h2']:4.2f} | {v['gain']:.3f} | {v['stat20']:.3f} / {v['stat12']:.3f} / {v['s100']:.3f} | {v['knee20']:.3f}/{v['knee10']:.3f} | "
              f"{v['burst']:.4f}/{v['burstp']:.3f} | {v['s3k']:.3f}/{v['s8k']:.3f} | {st}")
    print(f"\n   total {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
