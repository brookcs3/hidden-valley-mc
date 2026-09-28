# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical ripple harmonics, key "opto-ripple-b-asym-phosphor": directions (b) and (c) of docs/opto-fix.md section 6.1.

Hypothesis under test: what sits between the panel's hard turn-on and the cell smooths the light over about a quarter cycle, so that
the reference's ripple has a clean 6 dB/octave law (H3: -27.9 / -48.0 / -61.0 dBc at 100 Hz / 1 kHz / 4 kHz, t18 / -10 dBFS) and an H5
whose null sits where the reference's does. The candidates are placed AFTER the turn-on, so the static knee (the fine sweeps A) is
untouched in principle:
  (b)  asymmetric phosphor persistence: the light follows the panel drive with one time constant when rising and another, slower, when
       falling (an exponential tail on every light pulse); and a variant whose fall rate speeds up with the light itself;
  (c1) a light-dependent cell response: a one-pole on the light after the turn-on whose rate rises with the illumination (the CdS cell
       speeds up when lit), and separately the same illumination dependence applied to the states' attack (the release is already
       quenched by the conductance);
  (c2) two cells under one panel: a second conductance in parallel with the three-state cell that sees the light through its own,
       slower, persistence pole and carries a share of the conductance;
  (c3) the cell law after the integration: the states chase the light (n free) and the conductance is (sum w s)^gamma.
Each candidate adds two or three parameters. They are fitted together with a global knee shift, a light-gain trim and the rising
persistence, everything else held at the current fit/data/constants.json values (stage 4 as refitted: n 1.404, gamma 1.342, vth 6.48,
tau_el 10 us, the three states with 100-255 ms attacks), on the 27 opto_harm_t* items (gain, H3, H5, H7), the static level series at
positions 14, 18, 20, 22, the 100 Hz series at 20, the fine knees (A) and the nine burst envelopes with the stage-4 weights. The
control is the current model with the same free trims and nothing added; a second control refits the state constants as well.
Reported for each: H3 / H5 / H7 error rms over the 27 items by frequency, the static rms at position 20 (33 levels), the fine-knee
rms (A) at positions 20 and 10, the burst rms (weighted and plain), the 100 Hz static series.

Why the reference's null cannot be a time constant: its H5 minimum is at 6-7 dB of GR at 100 Hz, 1 kHz and 4 kHz alike (items
t14/-10 and t22/-20 at every frequency), so whatever places it is a function of the conduction angle, not of time. A one-pole on the
light or an asymmetric persistence is frequency dependent by construction; a level-dependent attack/release rate ratio of states that
are integrators at every audio frequency (the current constants) is not. The diagnostics print the model's null positions by frequency.

Result (logs /tmp/opto_ripple_asym_diag2.log, _srate.log, _fit5.log at 48 kHz, _fs96.log at 96 kHz): the current constants already put the
null at 6.7 / 7.9 dB of GR at 100 Hz / 1 kHz (reference 6.5-6.9) through the attack/release rate asymmetry of the states (with linear
symmetric states the model has no null at all). The 4 kHz H5 pattern (7.2 dB rms, a null at t18/-10 instead of t14/-10) is aliasing
of the light pulse's harmonics (8, 16, 24, 32 kHz...) in the 48 kHz loop: with the same constants and the loop at 96 / 192 kHz the
4 kHz H5 error is 3.9 / 4.6 dB rms and the null lands on t14/-10 (-89.0 / -90.8 against -91.4 dBc); the engine's quality 1 (2x) mode
gives the same (7.21 -> 4.03). With the aliasing removed, (b) fits tau_fall = tau_rise (nothing added), (c1) and (c1') go to their
no-effect bounds, (c2) buys 0.002 dB of burst rms as a fourth state while worsening H3 at 100 Hz / 1 kHz, and every refit of the state
constants trades H5 at 1 kHz (3.3 -> 4.7-5.7 dB rms) for the bursts (0.0075 -> 0.0035). None of (b), (c) is the ripple mechanism.

The loop is rendered by a numba mirror of OptoStage::process (the stage-4 mirror with the candidates added; validated against the C++
engine at the start of the run on the current calibration). Nothing under src/ or fit/stages/ or fit/data/ is written.
usage: cd <repo> && python3 -u fit/tools/candidates/opto-ripple-asym-phosphor.py [--diag] [--nullscan] [--srate] [--fit] [--fs96] [--quick]
  --diag      diagnostics (base model per-item table, level scans, ripple components)
  --nullscan  the anatomy of the model's H5 null (level scans with the state law simplified)
  --srate     the 27 items with the loop at 48 / 96 / 192 kHz on the mirror and through the engine at quality 0 / 1
  --fit       the candidate fits at 48 kHz
  --fs96      the reduced candidate list with the loop at 96 kHz
  with no flag: diag, nullscan, srate and fit (about 12 minutes on 16 cores); --fs96 must be asked for
"""
import json, os, sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402

np.set_printoptions(linewidth=220, suppress=True)
DISC = json.load(open(os.path.join(HERE, "..", "..", "data", "discriminate_opto.json")))
FSF = float(FS)
LEVELS = list(range(-50, 15, 2))
HARM_IDS = [f"opto_harm_t{t}_{l}_f{f}" for f in (100, 1000, 4000) for t in (14, 18, 22) for l in (-20, -10, 0)]
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
cal_now = load_cal()
def cget(name, i=0): return float(cal_now[MODEL.fields[name][0] + i])
G0 = cget("o_gain_db", 11) + cget("x_gain_db", 0)     # make-up at position 12 plus the Nickel path's midband gain, dB
G0_LIN = 10.0 ** (G0 / 20.0)

KINDS = {0: "control (symmetric persistence)", 1: "(b) asymmetric persistence rise/fall", 2: "(b') asymmetric, fall rate x (1 + kappa L)",
         3: "(c1) light-dependent cell low pass after the turn-on", 4: "(c1') illumination-dependent state attack",
         5: "(c2) second cell in parallel with its own persistence", 6: "(c3) cell law after the integration: states chase the light (n, gamma free), cond = (sum w s)^gamma",
         7: "(c3+b) cell law after the integration with an asymmetric persistence",
         8: "(c4) release quench with a free exponent: rate (1 + mu s^rho) / trel", 9: "(c5) state attack slowing with conductance: rate 1 / (tatt (1 + kappa s))",
         10: "(c6) excursion-dependent attack: rate (1 + beta (target - s)) / (tatt scale): fast on a step, slow on the ripple"}


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
def run_one(x, n, fs, A, vth, nexp, gam, tau_r, w, tatt, trel, L0, mu, b2, b3, fc, Q, kind, p1, p2, p3, out, cout):
    """OptoStage::process (sidechain filter out, no hwUnit, no material options) plus the candidate `kind`:
    divider v = xa / (1 + cond); sidechain low pass; d = A |s|; light (d - vth)^n above the turn-on; persistence; target = (L + L0)^gam;
    three states, linear attack, quenched release; out = v, cout = cond.
      kind 0: L one-pole tau_r both ways (the chosen model, tau_r = o_tau_el)
      kind 1: L rises with tau_r, falls with tau_f = p1
      kind 2: L rises with tau_r, falls with rate (1 + p2 L) / p1
      kind 3: after the persistence a second one-pole Lc with rate (1 + p2 Lc) / p1 both ways; the cell sees Lc
      kind 4: the states' attack rate is (1 + p1 s_i) / (p2 tatt_i)
      kind 5: cond = (1 - p1) sum_i w_i s_i + p1 s_b, s_b chasing (Lb + L0)^gam with Lb = L through a pole p2, attack/release of state 1 x p3
      kind 6: the states chase the light L + L0 itself (linear attack, quenched release in light units) and cond = (sum_i w_i s_i)^gam
      kind 8: the release rate is (1 + mu s_i^p1) / trel_i (p1 = 1 is the chosen model)
      kind 9: the attack rate is 1 / (p2 tatt_i (1 + p1 s_i)): the attack slows as the cell conducts (an asymmetry present at low GR)
      kind 10: the attack rate is (1 + p1 (target - s_i)) / (p2 tatt_i): a burst's overdrive is attacked fast, the ripple slowly"""
    kr = 1.0 - np.exp(-1.0 / (tau_r * fs))
    kf = 1.0 - np.exp(-1.0 / (p1 * fs)) if ((kind == 1 or kind == 7) and p1 > 0.0) else kr
    tsc = p2 if (kind == 4 or kind == 9 or kind == 10) else 1.0
    rho = p1 if kind == 8 else 1.0
    kA0 = 1.0 - np.exp(-1.0 / (tsc * tatt[0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (tsc * tatt[1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (tsc * tatt[2] * fs))
    rA0 = 1.0 / (tsc * tatt[0] * fs); rA1 = 1.0 / (tsc * tatt[1] * fs); rA2 = 1.0 / (tsc * tatt[2] * fs)
    r0 = 1.0 / (trel[0] * fs); r1 = 1.0 / (trel[1] * fs); r2 = 1.0 / (trel[2] * fs)
    kb = 1.0 - np.exp(-1.0 / (p2 * fs)) if (kind == 5 and p2 > 0.0) else 1.0
    kAb = 1.0 - np.exp(-1.0 / (p3 * tatt[1] * fs)) if kind == 5 else 0.0
    rb = 1.0 / (p3 * trel[1] * fs) if kind == 5 else 0.0
    lb0, lb1, lb2, la1, la2 = lp_matched(fc, Q, fs)
    c0 = L0 ** gam
    sinit = L0 if (kind == 6 or kind == 7) else c0
    z1 = 0.0; z2 = 0.0; L = 0.0; Lc = 0.0; Lb = 0.0; s0 = sinit; s1 = sinit; s2 = sinit; sb = c0; cond = c0
    for i in range(n):
        xi = x[i]
        xa = xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        out[i] = v; cout[i] = cond
        y = lb0 * v + z1
        z1 = lb1 * v - la1 * y + z2
        z2 = lb2 * v - la2 * y
        e = A * abs(y) - vth
        Linst = e ** nexp if e > 0.0 else 0.0
        if kind == 1 or kind == 7:
            L += (Linst - L) * (kr if Linst > L else kf)
        elif kind == 2:
            if Linst > L: L += (Linst - L) * kr
            else: L += (Linst - L) * (1.0 - np.exp(-(1.0 + p2 * L) / (p1 * fs)))
        else:
            L += (Linst - L) * kr
        if kind == 3:
            Lc += (L - Lc) * (1.0 - np.exp(-(1.0 + p2 * Lc) / (p1 * fs)))
            Lu = Lc
        else:
            Lu = L
        target = (Lu + L0) if (kind == 6 or kind == 7) else (Lu + L0) ** gam
        if kind == 8:
            q0 = 1.0 - np.exp(-(1.0 + mu * s0 ** rho) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1 ** rho) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2 ** rho) * r2)
        else:
            q0 = 1.0 - np.exp(-(1.0 + mu * s0) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2) * r2)
        if kind == 4:
            a0 = 1.0 - np.exp(-(1.0 + p1 * s0) * rA0); a1 = 1.0 - np.exp(-(1.0 + p1 * s1) * rA1); a2 = 1.0 - np.exp(-(1.0 + p1 * s2) * rA2)
        elif kind == 9:
            a0 = 1.0 - np.exp(-rA0 / (1.0 + p1 * s0)); a1 = 1.0 - np.exp(-rA1 / (1.0 + p1 * s1)); a2 = 1.0 - np.exp(-rA2 / (1.0 + p1 * s2))
        elif kind == 10:
            a0 = 1.0 - np.exp(-rA0 * (1.0 + p1 * (target - s0))); a1 = 1.0 - np.exp(-rA1 * (1.0 + p1 * (target - s1))); a2 = 1.0 - np.exp(-rA2 * (1.0 + p1 * (target - s2)))
        else:
            a0 = kA0; a1 = kA1; a2 = kA2
        s0 += (target - s0) * (a0 if target > s0 else q0)
        s1 += (target - s1) * (a1 if target > s1 else q1)
        s2 += (target - s2) * (a2 if target > s2 else q2)
        fast = w[0] * s0 + w[1] * s1 + w[2] * s2
        if kind == 5:
            Lb += (L - Lb) * kb
            tb = (Lb + L0) ** gam
            qb = 1.0 - np.exp(-(1.0 + mu * sb) * rb)
            sb += (tb - sb) * (kAb if tb > sb else qb)
            cond = (1.0 - p1) * fast + p1 * sb
        elif kind == 6 or kind == 7:
            cond = fast ** gam
        else:
            cond = fast


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, vth, nexp, gam, tau_r, w, tatt, trel, L0, mu, b2, b3, fc, Q, kind, p1, p2, p3, out, cout):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], vth, nexp, gam, tau_r, w, tatt, trel, L0[j], mu, b2, b3, fc, Q, kind, p1, p2, p3, out[j], cout[j])


@njit(cache=True, parallel=True)
def lockin_batch(Y, lens, fs, freqs, last_s, out):
    """gain of the fundamental (dB re unit amplitude) over the last `last_s` seconds, whole periods, as protocol.feature 'gain_db'"""
    for j in prange(Y.shape[0]):
        n = lens[j]; f = freqs[j]; per = fs / f
        nper = max(1, int(round(last_s * f)))
        n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
        re = 0.0; im = 0.0
        for i in range(n0, n1):
            t = i / fs
            re += Y[j, i] * np.cos(2.0 * np.pi * f * t); im -= Y[j, i] * np.sin(2.0 * np.pi * f * t)
        m = n1 - n0
        out[j] = 20.0 * np.log10(2.0 * np.sqrt(re * re + im * im) / m + 1e-30)


@njit(cache=True, parallel=True)
def harm_batch(Y, lens, fs, freqs, last_s, out):
    """H1..H8 complex amplitudes over the last `last_s` seconds (whole periods), as protocol.feature 'harm'"""
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


class Shape:
    """the optical stage's parameters in the C++ calibration's terms plus the candidate (kind, p1, p2, p3)"""
    def __init__(self, cal=None):
        cal = cal_now if cal is None else cal        # the current fit/data/constants.json is the baseline
        g = lambda name, i=0: float(cal[MODEL.fields[name][0] + i])
        self.b2, self.b3 = g("o_b2"), g("o_b3")
        self.thr = np.array(cal[MODEL.field("o_thr_db")], dtype=float).copy()
        self.n, self.gam, self.vth, self.tau_r = g("o_n"), g("o_gamma"), g("o_vth"), g("o_tau_el")
        self.w = np.array([g("o_w", i) for i in range(3)]); self.tatt = np.array([g("o_tatt", i) for i in range(3)]); self.trel = np.array([g("o_trel", i) for i in range(3)])
        self.mu, self.leak, self.leak_q = g("o_rel_mu"), g("o_leak"), g("o_leak_q")
        self.fc, self.Q = g("o_sc_lp_hz"), g("o_sc_lp_q")
        self.kind, self.p1, self.p2, self.p3 = 0, 0.0, 0.0, 1.0
        self.shift = 0.0        # global knee shift, dB (added to every o_thr_db)
        self.dlogC = 0.0        # light gain trim: multiplies the light by 10^(dlogC / gam) (i.e. C by 10^dlogC)

    def copy(self):
        s = Shape.__new__(Shape); s.__dict__.update(self.__dict__); s.thr = self.thr.copy(); s.w = self.w.copy(); return s

    def thr_db(self, k): return float(self.thr[k - 1]) + self.shift
    def knee(self, k): return 20.0 * np.log10(self.vth) - self.thr_db(k)
    def cond0(self, k): return self.leak * 10.0 ** (self.leak_q * (self.thr_db(k) - self.thr_db(20)) / 20.0)
    def law(self):
        """(n, gamma) in force: kinds 6/7 carry their own pair in p1/p2 when set"""
        if self.kind == 6 and self.p1 > 0.0: return float(self.p1), float(self.p2)
        return float(self.n), float(self.gam)

    def leak_light(self, k):
        c0 = self.cond0(k); gam = self.law()[1]; return c0 ** (1.0 / gam) if c0 > 0.0 else 0.0

    def render(self, X, lens, thrs, fs=FSF, Y=None, Cd=None):
        # the light gain trim is applied as a drive scale on (d - vth) and on L0: light' = light * 10^(dlogC/gam) == C' = C 10^dlogC
        n, gam = self.law()
        gl = 10.0 ** (self.dlogC / gam)
        A = np.array([gl * 10.0 ** (self.thr_db(k) / 20.0) for k in thrs]); L0 = np.array([gl * self.leak_light(k) for k in thrs])
        if Y is None: Y = np.zeros_like(X)
        if Cd is None: Cd = np.zeros_like(X)
        run_batch(X, lens, fs, A, gl * self.vth, n, gam, self.tau_r, self.w, self.tatt, self.trel, L0, self.mu, self.b2, self.b3, self.fc, self.Q,
                  int(self.kind), float(self.p1), float(self.p2), float(self.p3), Y, Cd)
        return Y


def rms(e):
    e = np.asarray(e, dtype=float); e = e[~np.isnan(e)]
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def amax(e):
    e = np.asarray(e, dtype=float); e = e[~np.isnan(e)]
    return float(np.max(np.abs(e))) if len(e) else float("nan")


# ------------------------------------------------------------------------------------------------ batches of protocol items
class Batch:
    """protocol items rendered together on the mirror; `fs` overrides the loop's sample rate (the stimulus is generated and the
    features are taken at that rate, so the reference features compare directly)"""
    def __init__(self, ids, secs=None, fs=None):
        self.ids = ids
        self.fs = float(fs if fs is not None else ITEMS[ids[0]]["fs"])
        xs = []
        for i in ids:
            st = dict(ITEMS[i]["stim"])
            if secs is not None and st["kind"] == "sine": st["secs"] = secs
            xs.append(protocol.stimulus(st, int(self.fs))[0])
        self.lens = np.array([len(x) for x in xs]); self.X = np.zeros((len(ids), self.lens.max()))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.Y = np.zeros_like(self.X); self.C = np.zeros_like(self.X)
        self.thr = [int(ITEMS[i]["set"]["optical_threshold"]) for i in ids]
        self.freqs = np.array([float(ITEMS[i]["stim"]["f"]) for i in ids])

    def gains(self, sh, last_s=0.5):
        sh.render(self.X, self.lens, self.thr, self.fs, self.Y, self.C)
        g = np.zeros(len(self.ids)); lockin_batch(self.Y, self.lens, self.fs, self.freqs, last_s, g)
        return g - np.array([ITEMS[i]["stim"]["level"] for i in self.ids], dtype=float) + G0

    def harms(self, sh, last_s=1.0):
        """gain_db and H2..H8 in dBc per item (None where above 0.95 Nyquist), like protocol.feature 'harm'"""
        sh.render(self.X, self.lens, self.thr, self.fs, self.Y, self.C)
        H = np.zeros((len(self.ids), 8), dtype=np.complex128); harm_batch(self.Y, self.lens, self.fs, self.freqs, last_s, H)
        out = []
        for j, i in enumerate(self.ids):
            a = np.abs(H[j]); hs = []
            for k in range(2, 9):
                hs.append(None if k * self.freqs[j] >= self.fs / 2 * 0.95 else float(20 * np.log10(a[k - 1] / (a[0] + 1e-30) + 1e-30)))
            out.append({"gain_db": float(20 * np.log10(a[0] * G0_LIN + 1e-30) - ITEMS[i]["stim"]["level"]), "h": hs})
        return out

    def ripple(self, sh, last_s=1.0):
        """the conductance's mean and its 2f/4f/6f components (relative, dB) over the last second, per item"""
        sh.render(self.X, self.lens, self.thr, self.fs, self.Y, self.C)
        rows = []
        for j in range(len(self.ids)):
            n = self.lens[j]; f = self.freqs[j]; per = self.fs / f; nper = int(round(last_s * f))
            n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per)); seg = self.C[j, n0:n1]; t = np.arange(n0, n1) / self.fs
            c0 = float(np.mean(seg)); comps = [abs(2.0 * np.mean(seg * np.exp(-2j * np.pi * k * f * t))) for k in (2, 4, 6)]
            rows.append((c0, [20 * np.log10(c / c0 + 1e-30) for c in comps]))
        return rows

    def envs(self, sh):
        sh.render(self.X, self.lens, self.thr, self.fs, self.Y, self.C)
        E = np.zeros((len(self.ids), int(self.lens.max() / (self.fs / 1000.0)) + 2)); cnt = np.zeros(len(self.ids), dtype=np.int64)
        env_batch(self.Y, self.X, self.lens, self.fs, 1000.0, E, cnt)
        return [E[j, :cnt[j]] + G0 for j in range(len(self.ids))]


def burst_weight(iid, n):
    ref = np.asarray(F[iid])[:n]
    return np.where(ref < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)


# ------------------------------------------------------------------------------------------------ validation against the C++ engine
def validate():
    sh = Shape(cal_now)
    chk = ["opto_static_t20_-10", "opto_static_t20_2", "opto_static_t10_-20", "opto_burst_-10", "opto_harm_t18_-10_f1000", "opto_harm_t18_-10_f100"]
    worst = 0.0
    for i in chk:
        b = Batch([i]); e = render_item(ITEMS[i], cal_now)
        t = ITEMS[i]["feat"]["type"]
        if t == "gain_db": m = b.gains(sh)[0]; d = abs(m - e)
        elif t == "env": m = b.envs(sh)[0]; n = min(len(m), len(e)); d = float(np.max(np.abs(m[:n] - np.asarray(e)[:n])))
        else:
            m = b.harms(sh)[0]; d = max(abs(m["gain_db"] - e["gain_db"]), abs(m["h"][1] - e["h"][1]), abs(m["h"][3] - e["h"][3]))
        worst = max(worst, d)
        print(f"  mirror vs engine {i}: max |diff| {d:.4f} dB")
    print(f"  baseline = the current constants.json: n {sh.n:.3f} gamma {sh.gam:.3f} vth {sh.vth:.3f} tau_el {sh.tau_r * 1e6:.1f} us | w {np.round(sh.w, 3).tolist()} attack ms {np.round(sh.tatt * 1e3, 1).tolist()} release ms {np.round(sh.trel * 1e3, 1).tolist()} mu {sh.mu:.2f} | leak {sh.leak:.4f} q {sh.leak_q:.3f} | LP {sh.fc:.0f} Hz Q {sh.Q:.3f}")
    if worst > 0.05: print("!! the mirror does not reproduce the engine; stopping"); sys.exit(1)


# ------------------------------------------------------------------------------------------------ scoring
class Scorer:
    def __init__(self, quick=False, fs=None):
        self.fsx = float(fs if fs is not None else FS); fsx = self.fsx        # the loop's sample rate for every batch (48 kHz = the protocol's)
        self.bh = Batch(HARM_IDS, fs=fsx)                                           # protocol lengths (3 s, last 1 s)
        self.bh_fit = Batch(HARM_IDS, secs=1.5, fs=fsx)                             # shorter renders for the fit (last 0.5 s)
        self.bs20 = Batch([f"opto_static_t20_{l}" for l in LEVELS], fs=fsx)
        fit_lv = LEVELS[::2] if quick else LEVELS
        self.stat_fit_ids = [f"opto_static_t{k}_{l}" for k in (14, 18, 20, 22) for l in fit_lv]
        self.bs_fit = Batch(self.stat_fit_ids, secs=1.5, fs=fsx)
        self.bb = Batch(BURSTS, fs=fsx)
        self.b100 = Batch([f"opto_static_f100_{l}" for l in range(-40, 11, 2)], fs=fsx)
        self.b100_fit = Batch([f"opto_static_f100_{l}" for l in range(-40, 11, 4)], secs=1.5, fs=fsx)   # guard: the 100 Hz statics must stay with the 1 kHz ones
        # fine knee sweeps (A)
        self.knee = {}
        for thr in ("20", "10"):
            kd = DISC["knee"][thr]; lv = np.array(kd["levels"]); n = int(3.0 * fsx); t = np.arange(n) / fsx
            X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in lv])
            self.knee[thr] = (lv, np.array(kd["gain_db"]), X, np.full(len(lv), n, dtype=np.int64), np.zeros_like(X), np.zeros_like(X))
        self.ref_h = [F[i] for i in HARM_IDS]
        # the fine sweeps again at 1.5 s for the fit residual
        self.knee_fit = {}
        for thr in ("20", "10"):
            kd = DISC["knee"][thr]; lv = np.array(kd["levels"]); n = int(1.5 * fsx); t = np.arange(n) / fsx
            X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * 1000.0 * t) for l in lv])
            self.knee_fit[thr] = (lv, np.array(kd["gain_db"]), X, np.full(len(lv), n, dtype=np.int64), np.zeros_like(X), np.zeros_like(X))

    def knee_resid(self, sh, fit=True):
        out = []; h = int(self.fsx) // 2
        for thr, (lv, gref, X, lens, Y, Cd) in (self.knee_fit if fit else self.knee).items():
            sh.render(X, lens, [int(thr)] * len(lv), self.fsx, Y, Cd)
            g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -h:] ** 2)) / np.sqrt(np.mean(X[j, -h:] ** 2))) for j in range(len(lv))]) + G0
            out.append(g - gref)
        return np.concatenate(out)

    def harm_errors(self, sh, fit=False):
        """per item: (gain err, H3 err, H5 err, H7 err or nan); references below -100 dBc are treated as absent"""
        b = self.bh_fit if fit else self.bh
        ms = b.harms(sh, 0.5 if fit else 1.0); rows = []
        for m, r in zip(ms, self.ref_h):
            e = [m["gain_db"] - r["gain_db"]]
            for idx in (1, 3, 5):
                a, bb = m["h"][idx], r["h"][idx]
                e.append(np.nan if (a is None or bb is None or bb < -100.0) else max(a, -110.0) - bb)
            rows.append(e)
        return np.array(rows), ms

    def resid(self, sh, wg=0.5, w3=1.0, w5=0.5, w7=0.3, ws=40.0, wk=40.0, wb=20.0):
        """harmonics (gain 0.5, H3 1, H5 0.5, H7 0.3 per dB), statics x 40 per dB (1 kHz at four positions and the 100 Hz series at
        position 20, so that a fixed-time smoothing cannot buy harmonics with a frequency-dependent static), fine knees x 40, bursts x 20:
        the chosen model's cost is then about 860 harmonics / 530 statics / 40 knees / 1030 bursts (sum of squares / 2), so doubling
        the static or burst error costs more than any harmonic gain on offer, while a candidate is free to improve them"""
        E, _ = self.harm_errors(sh, fit=True)
        out = [wg * E[:, 0], w3 * np.nan_to_num(E[:, 1]), w5 * np.nan_to_num(E[:, 2]), w7 * np.nan_to_num(E[:, 3])]
        out.append(ws * (self.bs_fit.gains(sh) - np.array([F[i] for i in self.stat_fit_ids])))
        out.append(ws * (self.b100_fit.gains(sh) - np.array([F[i] for i in self.b100_fit.ids])))
        out.append(wk * self.knee_resid(sh))
        for i, m in zip(BURSTS, self.bb.envs(sh)):
            n = min(len(m), len(F[i])); out.append(wb * burst_weight(i, n) * (np.array(m[:n]) - np.array(F[i][:n])))
        return np.concatenate(out)

    def report(self, sh, label, items=False):
        E, ms = self.harm_errors(sh)
        print(f"\n== {label}  [loop at {self.fsx / 1e3:g} kHz]")
        print(f"   kind {sh.kind} [{KINDS[sh.kind]}] p1 {sh.p1:.4g} p2 {sh.p2:.4g} p3 {sh.p3:.4g} | tau_rise {sh.tau_r * 1e6:.1f} us | knee shift {sh.shift:+.3f} dB | dlogC {sh.dlogC:+.4f}")
        by_f = {}
        for f in (100, 1000, 4000):
            sel = [j for j, i in enumerate(HARM_IDS) if i.endswith(f"_f{f}")]
            by_f[f] = [rms(E[sel, 1]), rms(E[sel, 2]), rms(E[sel, 3]), amax(E[sel, 1]), amax(E[sel, 2])]
            print(f"   {f:5d} Hz: H3 err rms {by_f[f][0]:.2f} (max {by_f[f][3]:.1f}) | H5 rms {by_f[f][1]:.2f} (max {by_f[f][4]:.1f}) | H7 rms {by_f[f][2]:.2f} | gain err rms {rms(E[sel, 0]):.3f}")
        h3all = rms(E[:, 1]); h5all = rms(E[:, 2]); h7all = rms(E[:, 3])
        print(f"   all 27: H3 {h3all:.2f} | H5 {h5all:.2f} | H7 {h7all:.2f} | gain {rms(E[:, 0]):.3f} dB")
        if items:
            print("   item                        GR    | H3 model/ref  err | H5 model/ref  err | H7 model/ref  err")
            for j, i in enumerate(HARM_IDS):
                m, r = ms[j], self.ref_h[j]
                h7 = f"{m['h'][5]:6.1f}/{r['h'][5]:6.1f} {E[j, 3]:+5.1f}" if not np.isnan(E[j, 3]) else "  n/a"
                print(f"   {i:26s} {-r['gain_db'] + F[f'opto_static_t{ITEMS[i]['set']['optical_threshold']}_-50']:5.1f} | {m['h'][1]:6.1f}/{r['h'][1]:6.1f} {E[j, 1]:+5.1f} | {m['h'][3]:6.1f}/{r['h'][3]:6.1f} {E[j, 2]:+5.1f} | {h7}")
        # statics at t20, the 100 Hz series
        e20 = self.bs20.gains(sh) - np.array([F[i] for i in self.bs20.ids]); e100 = self.b100.gains(sh) - np.array([F[i] for i in self.b100.ids])
        # bursts
        allw = []; allp = []; per = []
        for i, m in zip(BURSTS, self.bb.envs(sh)):
            n = min(len(m), len(F[i])); d = np.array(m[:n]) - np.array(F[i][:n]); allw.append(burst_weight(i, n) * d); allp.append(d); per.append(rms(d))
        bw = rms(np.concatenate(allw)); bp = rms(np.concatenate(allp))
        # fine knees
        kn = {}; h = int(self.fsx)
        for thr, (lv, gref, X, lens, Y, Cd) in self.knee.items():
            sh.render(X, lens, [int(thr)] * len(lv), self.fsx, Y, Cd)
            g = np.array([20 * np.log10(np.sqrt(np.mean(Y[j, -h:] ** 2)) / np.sqrt(np.mean(X[j, -h:] ** 2))) for j in range(len(lv))]) + G0
            grm = g[0] - g; grr = gref[0] - gref
            cm = [float(np.interp(v, grm, lv)) for v in (0.5, 3.0)]; cr = [float(np.interp(v, grr, lv)) for v in (0.5, 3.0)]
            kn[thr] = (rms(g - gref), float(np.max(np.abs(g - gref))), cm[1] - cm[0], cr[1] - cr[0], cm[0], cr[0])
        print(f"   statics t20 (33 levels): rms {rms(e20):.3f} max {np.max(np.abs(e20)):.2f} | 100 Hz series at t20: rms {rms(e100):.3f} max {np.max(np.abs(e100)):.2f}")
        print(f"   bursts (9 items): weighted rms {bw:.4f} | plain rms {bp:.3f} | per item {np.round(per, 3).tolist()}")
        print(f"   (A) fine knee t20: rms {kn['20'][0]:.3f} max {kn['20'][1]:.2f}, width 0.5->3 dB {kn['20'][2]:.2f} (ref {kn['20'][3]:.2f}), 0.5 dB at {kn['20'][4]:.2f} (ref {kn['20'][5]:.2f}) | "
              f"t10: rms {kn['10'][0]:.3f} max {kn['10'][1]:.2f}, width {kn['10'][2]:.2f} (ref {kn['10'][3]:.2f})")
        return dict(h3=by_f, h3all=h3all, h5all=h5all, h7all=h7all, stat20=rms(e20), s100=rms(e100), bw=bw, bp=bp, kn20=kn["20"][0], kn10=kn["10"][0])


# ------------------------------------------------------------------------------------------------ diagnostics
def diagnostics(S):
    base = Shape()
    print("\n1. the current model, per item (fit/data/constants.json)")
    S.report(base, "baseline, constants.json as is", items=True)
    b2 = base.copy(); b2.tau_r = 2e-6
    S.report(b2, "baseline with the persistence pole removed (tau_el 2 us, no refit)", items=True)
    # the ripple components of the conductance, model, and what the reference's harmonics imply
    print("\n2. conductance ripple of the chosen model (2f / 4f / 6f components relative to the mean conductance, dB) against the reference's H3/H5/H7")
    print("   for a slow gain ripple the sideband law is H(2k+1) ~ c_2k / (2 (1 + c0)) to first order; the last column is that first-order H3 from c2")
    rows = S.bh.ripple(base)
    for j, i in enumerate(HARM_IDS):
        c0, comps = rows[j]; r = S.ref_h[j]
        h3_lin = 20 * np.log10(10 ** (comps[0] / 20) * c0 / (2 * (1 + c0)) + 1e-30)
        print(f"   {i:26s} cond {c0:6.3f} | c2 {comps[0]:6.1f} c4 {comps[1]:6.1f} c6 {comps[2]:6.1f} dB re c0 | ref H3 {r['h'][1]:6.1f} H5 {r['h'][3]:6.1f} | c2->H3 {h3_lin:6.1f}")
    # level scans at position 20: where are the model's H5 / H7 nulls, per frequency
    print("\n3. level scans at position 20 (model): H3 / H5 / H7 against GR, per frequency; the reference's 27 items are on the same GR axis in section 1")
    for f in (100.0, 1000.0, 4000.0):
        lv = np.arange(-30.0, 8.1, 2.0); n = int(1.5 * FS); t = np.arange(n) / FS
        X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * f * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
        Y = np.zeros_like(X); Cd = np.zeros_like(X)
        for sh, lab in ((base, "as is"), (b2, "no pole")):
            sh.render(X, lens, [20] * len(lv), FSF, Y, Cd)
            H = np.zeros((len(lv), 8), dtype=np.complex128); harm_batch(Y, lens, FSF, np.full(len(lv), f), 0.5, H)
            a = np.abs(H); g = 20 * np.log10(a[:, 0] + 1e-30) - lv + G0; gr = g[0] - g
            h = lambda k: 20 * np.log10(a[:, k - 1] / (a[:, 0] + 1e-30) + 1e-30)
            print(f"   f {f:6.0f} tau {lab:7s}  GR: " + " ".join(f"{v:6.1f}" for v in gr))
            print(f"                       H3: " + " ".join(f"{v:6.1f}" for v in h(3)))
            print(f"                       H5: " + " ".join(f"{v:6.1f}" for v in h(5)))
            if 7 * f < FS / 2 * 0.95: print(f"                       H7: " + " ".join(f"{v:6.1f}" for v in h(7)))
    # the reference on a GR axis
    print("\n4. the reference's harmonics sorted by GR (from the item gains), per frequency")
    for f in (100, 1000, 4000):
        sel = [(-(F[i]["gain_db"]) + F[f"opto_static_t{ITEMS[i]['set']['optical_threshold']}_-50"], i) for i in HARM_IDS if i.endswith(f"_f{f}")]
        sel.sort()
        print(f"   f {f:5d}  GR: " + " ".join(f"{g:6.1f}" for g, _ in sel))
        print(f"            H3: " + " ".join(f"{F[i]['h'][1]:6.1f}" for _, i in sel))
        print(f"            H5: " + " ".join(f"{F[i]['h'][3]:6.1f}" for _, i in sel))
        print(f"            H7: " + " ".join(f"{(F[i]['h'][5] if F[i]['h'][5] is not None else float('nan')):6.1f}" for _, i in sel))


def null_scan(S):
    """where does the model's H5 null come from: level scans at position 20, 1 kHz, with the state law simplified step by step"""
    print("\n5. H5 / c4 null anatomy at position 20, 1 kHz: GR, H5 and the conductance's 4f/2f ratio (dB) against level, per variant")
    base = Shape()
    variants = [("current model", base)]
    v = base.copy(); v.mu = 0.0; v.trel = v.tatt.copy(); variants.append(("linear symmetric states (mu 0, trel = tatt)", v))
    v = base.copy(); v.mu = 0.0; variants.append(("mu 0 (attack faster than release everywhere)", v))
    v = base.copy(); v.tau_r = 1e-6; variants.append(("no persistence pole", v))
    v = base.copy(); v.mu = 0.0; v.trel = v.tatt.copy(); v.tau_r = 1e-6; variants.append(("linear symmetric states, no pole", v))
    v = base.copy(); v.trel = v.tatt / 3.0; v.mu = 0.0; variants.append(("mu 0, release 3x faster than attack", v))
    v = base.copy(); v.kind = 6; variants.append(("(c3) cell law after the integration, constants as is", v))
    for k in (0.3, 0.5, 2.0):
        v = base.copy(); v.tatt = v.tatt * k; variants.append((f"attack constants x {k:g} (release-dominated asymmetry at lower GR)", v))
    v = base.copy(); v.tatt = v.tatt * 0.5; v.tau_r = 2e-6; variants.append(("attack constants x 0.5, no persistence pole", v))
    lv = np.arange(-30.0, 8.1, 2.0); n = int(1.5 * FS); t = np.arange(n) / FS; f = 1000.0
    X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * f * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
    Y = np.zeros_like(X); Cd = np.zeros_like(X)
    for lab, sh in variants:
        sh.render(X, lens, [20] * len(lv), FSF, Y, Cd)
        H = np.zeros((len(lv), 8), dtype=np.complex128); harm_batch(Y, lens, FSF, np.full(len(lv), f), 0.5, H)
        a = np.abs(H); g = 20 * np.log10(a[:, 0] + 1e-30) - lv + G0; gr = g[0] - g
        h5 = 20 * np.log10(a[:, 4] / (a[:, 0] + 1e-30) + 1e-30); h3 = 20 * np.log10(a[:, 2] / (a[:, 0] + 1e-30) + 1e-30)
        r = []
        for j in range(len(lv)):
            seg = Cd[j, -FS // 2:]; tt = np.arange(n - FS // 2, n) / FSF
            c2 = abs(2 * np.mean(seg * np.exp(-2j * np.pi * 2 * f * tt))); c4 = abs(2 * np.mean(seg * np.exp(-2j * np.pi * 4 * f * tt)))
            r.append(20 * np.log10(c4 / (c2 + 1e-30) + 1e-30))
        print(f"   [{lab}]")
        print("      GR:    " + " ".join(f"{x:6.1f}" for x in gr))
        print("      H3:    " + " ".join(f"{x:6.1f}" for x in h3))
        print("      H5:    " + " ".join(f"{x:6.1f}" for x in h5))
        print("      c4/c2: " + " ".join(f"{x:6.1f}" for x in r))
    print("\n   and the full report for the attack-scaled variants (no other change):")
    for k in (0.5, 2.0):
        v = base.copy(); v.tatt = v.tatt * k; S.report(v, f"attack constants x {k:g}", items=False)
    v = base.copy(); v.mu = 0.0; S.report(v, "release quench removed (mu 0), no other change", items=False)


# ------------------------------------------------------------------------------------------------ the loop's sample rate
def srate_scan():
    """the 27 items on the mirror with the loop at 48, 96 and 192 kHz (same constants), and through the engine at quality 0 and 1
    (2x oversampling wraps the optical stage): a hard turn-on on a rectified 4 kHz sine has light harmonics at 8k, 16k, 24k, 32k...
    which alias in a 48 kHz loop; the 4f component (16 kHz) is what H5 carries"""
    print("\n6. the loop's sample rate: the current constants, 27 items, mirror at 48 / 96 / 192 kHz (model / ref dBc; error rms by frequency)")
    sh = Shape()
    for fs in (48000, 96000, 192000):
        b = Batch(HARM_IDS, fs=fs); ms = b.harms(sh)
        E = {100: [[], [], [], []], 1000: [[], [], [], []], 4000: [[], [], [], []]}
        line = []
        for iid, m in zip(HARM_IDS, ms):
            r = F[iid]; f = int(ITEMS[iid]["stim"]["f"])
            E[f][0].append(m["gain_db"] - r["gain_db"])
            for k, idx in enumerate((1, 3, 5)):
                if m["h"][idx] is not None and r["h"][idx] is not None and r["h"][idx] > -100: E[f][k + 1].append(m["h"][idx] - r["h"][idx])
            if f == 4000 and r["h"][3] > -100: line.append(f"{iid[10:-6]} {m['h'][3]:.1f}/{r['h'][3]:.1f}")
        print(f"   loop at {fs / 1e3:g} kHz: " + " | ".join(f"{f} Hz H3 {rms(E[f][1]):.2f} H5 {rms(E[f][2]):.2f} H7 {rms(E[f][3]):.2f} gain {rms(E[f][0]):.3f}" for f in (100, 1000, 4000)))
        print("      4 kHz H5 by item: " + ", ".join(line))
    print("   through the C++ engine at 48 kHz, quality 0 (no oversampling) and 1 (2x, the optical stage inside): 1 kHz and 4 kHz items")
    ids = [i for i in HARM_IDS if i.endswith("_f1000") or i.endswith("_f4000")]
    for q in (0, 1):
        E = {1000: [[], []], 4000: [[], []]}; line = []
        for iid in ids:
            m = render_item(ITEMS[iid], cal_now, quality=q); r = F[iid]; f = int(ITEMS[iid]["stim"]["f"])
            if r["h"][3] > -100: E[f][0].append(m["h"][1] - r["h"][1]); E[f][1].append(m["h"][3] - r["h"][3])
            if f == 4000 and r["h"][3] > -100: line.append(f"{iid[10:-6]} {m['h'][3]:.1f}/{r['h'][3]:.1f}")
        print(f"   engine quality {q}: " + " | ".join(f"{f} Hz H3 {rms(E[f][0]):.2f} H5 {rms(E[f][1]):.2f}" for f in (1000, 4000)) + "\n      4 kHz H5 by item: " + ", ".join(line))
    print("   4 kHz level scan at position 20, H5 against GR, loop at 48 kHz and at 192 kHz (the reference's minimum is at 6-7 dB of GR at every frequency):")
    for fs in (48000, 192000):
        lv = np.arange(-30.0, 8.1, 2.0); n = int(1.5 * fs); t = np.arange(n) / fs; f = 4000.0
        X = np.array([10 ** (l / 20) * np.sin(2 * np.pi * f * t) for l in lv]); lens = np.full(len(lv), n, dtype=np.int64)
        Y = np.zeros_like(X); Cd = np.zeros_like(X)
        sh.render(X, lens, [20] * len(lv), float(fs), Y, Cd)
        H = np.zeros((len(lv), 8), dtype=np.complex128); harm_batch(Y, lens, float(fs), np.full(len(lv), f), 0.5, H)
        a = np.abs(H); g = 20 * np.log10(a[:, 0] + 1e-30) - lv + G0; gr = g[0] - g
        print(f"      {fs / 1e3:3g} kHz  GR: " + " ".join(f"{v:6.1f}" for v in gr))
        print(f"               H5: " + " ".join(f"{v:6.1f}" for v in 20 * np.log10(a[:, 4] / (a[:, 0] + 1e-30) + 1e-30)))


# ------------------------------------------------------------------------------------------------ candidate fits
def fit_candidate(S, kind, p0, lo, hi, xs, label, dyn=False, max_nfev=40):
    """fit the candidate's parameters (log10 for time constants and rates, see unpack) + log10 tau_rise + knee shift + dlogC; with
    dyn=True the state constants w, tatt, trel and mu are refitted as well (needed where the candidate changes what a state means)"""
    base = Shape(); base.kind = kind
    npar = len(p0)
    def unpack(v):
        s = base.copy()
        ps = list(v[:npar]) + [0.0, 0.0, 1.0][npar:]
        if kind == 1: s.p1 = 10.0 ** ps[0]
        if kind == 2: s.p1 = 10.0 ** ps[0]; s.p2 = 10.0 ** ps[1]
        if kind == 3: s.p1 = 10.0 ** ps[0]; s.p2 = 10.0 ** ps[1]
        if kind == 4: s.p1 = 10.0 ** ps[0]; s.p2 = ps[1]
        if kind == 5: s.p1 = ps[0]; s.p2 = 10.0 ** ps[1]; s.p3 = 10.0 ** ps[2]
        if kind == 6: s.p1 = ps[0]; s.p2 = ps[1]
        if kind == 7: s.p1 = 10.0 ** ps[0]
        if kind == 8: s.p1 = ps[0]
        if kind == 9: s.p1 = 10.0 ** ps[0]; s.p2 = ps[1]
        if kind == 10: s.p1 = 10.0 ** ps[0]; s.p2 = 10.0 ** ps[1]
        s.tau_r = 10.0 ** v[npar]; s.shift = v[npar + 1]; s.dlogC = v[npar + 2]
        if dyn:
            q = v[npar + 3:]
            w = np.abs(np.array(q[0:3])); s.w = w / w.sum(); s.tatt = 10.0 ** np.array(q[3:6]); s.trel = 10.0 ** np.array(q[6:9]); s.mu = 10.0 ** q[9]
        return s
    x0 = list(p0) + [np.log10(base.tau_r), 0.0, 0.0]; LO = list(lo) + [-5.5, -3.0, -0.3]; HI = list(hi) + [-3.5, 3.0, 0.3]; XS = list(xs) + [0.1, 0.2, 0.02]
    if dyn:
        mu0 = np.log10(base.mu) + (0.3 if kind in (6, 7) else 0.0)     # in light units the quench per unit state is larger (s = cond^(1/gamma))
        x0 += list(base.w) + list(np.log10(base.tatt)) + list(np.log10(base.trel)) + [mu0]
        LO += [0.005] * 3 + [-4.0] * 3 + [-3.0] * 3 + [-1.0]; HI += [1.0] * 3 + [0.0] * 3 + [1.0] * 3 + [3.0]; XS += [0.05] * 3 + [0.1] * 3 + [0.1] * 3 + [0.1]
    t0 = time.time()
    r = least_squares(lambda v: S.resid(unpack(v)), np.array(x0), bounds=(LO, HI), x_scale=XS, diff_step=2e-3, max_nfev=max_nfev)
    s = unpack(r.x)
    print(f"\n   fit [{label}]: cost {r.cost:.1f} nfev {r.nfev} ({time.time() - t0:.0f} s) x {np.round(r.x, 4).tolist()}")
    if dyn: print(f"   w {np.round(s.w, 3).tolist()} attack ms {np.round(s.tatt * 1e3, 2).tolist()} release ms {np.round(s.trel * 1e3, 1).tolist()} mu {s.mu:.2f}")
    return s, r


def main():
    quick = "--quick" in sys.argv; only = [a for a in sys.argv[1:] if a.startswith("--") and a != "--quick"]
    do_diag = "--diag" in only or not only; do_fit = "--fit" in only or not only; do_null = "--nullscan" in only or not only
    do_srate = "--srate" in only or not only; do_fs96 = "--fs96" in only
    t0 = time.time()
    print(f"G0 {G0:.3f} dB, b2 {cget('o_b2'):.2e} b3 {cget('o_b3'):.2e}; {os.cpu_count()} cpus")
    validate()
    S = Scorer(quick=quick)
    print(f"  batches ready [{time.time() - t0:.0f} s]")
    if do_diag:
        diagnostics(S)
        print(f"\n  diagnostics done [{time.time() - t0:.0f} s]")
    if do_null:
        null_scan(S)
    if do_srate:
        srate_scan()
        print(f"\n  sample-rate scan done [{time.time() - t0:.0f} s]")
    if do_fs96:
        # the candidates again with the loop at 96 kHz (the aliasing of the 4 kHz items removed): the reduced list
        S = Scorer(quick=quick, fs=96000); results = {}
        base = Shape(); r0 = S.resid(base)
        print(f"\n7. the loop at 96 kHz: cost of the current model under the fit weights: {0.5 * float(np.sum(r0 ** 2)):.1f}")
        results["baseline"] = S.report(base, "the current model as is (no fit)", items=True)
        s, _ = fit_candidate(S, 0, [], [], [], [], "control: tau_el, knee shift, light trim only")
        results["control"] = S.report(s, "control refit (tau_el, knee shift, light trim)", items=False)
        s, _ = fit_candidate(S, 1, [np.log10(1e-4)], [-5.5], [-2.5], [0.1], "(b) asym persistence: tau_fall")
        results["b"] = S.report(s, "(b) asymmetric persistence", items=False)
        s, _ = fit_candidate(S, 4, [np.log10(1.0), 1.0], [-2.0, 0.2], [2.0, 5.0], [0.2, 0.1], "(c1') state attack rate x (1 + mu_a s), tatt scale")
        results["c1b"] = S.report(s, "(c1') illumination-dependent state attack", items=False)
        s, _ = fit_candidate(S, 5, [0.3, np.log10(3e-4), 0.0], [0.02, -5.5, -1.5], [0.9, -2.0, 1.5], [0.05, 0.1, 0.1], "(c2) second cell, share / persistence / speed")
        results["c2"] = S.report(s, "(c2) second cell in parallel with its own persistence", items=False)
        s, _ = fit_candidate(S, 0, [], [], [], [], "control + dynamics refit (w, tatt, trel, mu)", dyn=True, max_nfev=60)
        results["ctrl-dyn"] = S.report(s, "control with the state constants refitted", items=True)
        s, _ = fit_candidate(S, 4, [np.log10(1.0), 1.0], [-2.0, 0.2], [2.0, 5.0], [0.2, 0.1], "(c1') illumination-dependent attack, dynamics refitted", dyn=True, max_nfev=60)
        results["c1b-dyn"] = S.report(s, "(c1') illumination-dependent state attack with the state constants refitted", items=True)
        print("\n== summary at 96 kHz (H3 / H5 / H7 error rms over the 27 items; statics t20; fine knee t20/t10; bursts weighted/plain)")
        for k, v in results.items():
            print(f"   {k:8s} H3 {v['h3all']:.2f} (100 Hz {v['h3'][100][0]:.2f}, 1 kHz {v['h3'][1000][0]:.2f}, 4 kHz {v['h3'][4000][0]:.2f}) | H5 {v['h5all']:.2f} ({v['h3'][100][1]:.1f}, {v['h3'][1000][1]:.1f}, {v['h3'][4000][1]:.1f}) | "
                  f"H7 {v['h7all']:.2f} | stat t20 {v['stat20']:.3f} | 100 Hz {v['s100']:.3f} | knee {v['kn20']:.3f}/{v['kn10']:.3f} | bursts {v['bw']:.4f}/{v['bp']:.3f}")
    if do_fit:
        results = {}
        base = Shape(); r0 = S.resid(base)
        print(f"   cost of the current model under the fit weights: {0.5 * float(np.sum(r0 ** 2)):.1f}")
        results["baseline"] = S.report(base, "the current model as is (no fit)", items=False)
        s, _ = fit_candidate(S, 0, [], [], [], [], "control: tau_el, knee shift, light trim only")
        results["control"] = S.report(s, "control refit (tau_el, knee shift, light trim)", items=False)
        s, _ = fit_candidate(S, 1, [np.log10(1e-4)], [-5.5], [-2.5], [0.1], "(b) asym persistence: tau_fall")
        results["b"] = S.report(s, "(b) asymmetric persistence", items=True)
        s, _ = fit_candidate(S, 2, [np.log10(1e-4), np.log10(1.0)], [-5.5, -3.0], [-2.5, 3.0], [0.1, 0.2], "(b') asym persistence, fall rate x (1 + kappa L)")
        results["b2"] = S.report(s, "(b') asymmetric persistence with light-dependent fall", items=False)
        s, _ = fit_candidate(S, 3, [np.log10(1e-4), np.log10(1.0)], [-5.5, -3.0], [-2.0, 3.0], [0.1, 0.2], "(c1) cell low pass, rate x (1 + kappa Lc)")
        results["c1"] = S.report(s, "(c1) light-dependent cell low pass after the turn-on", items=False)
        s, _ = fit_candidate(S, 4, [np.log10(1.0), 1.0], [-2.0, 0.2], [2.0, 5.0], [0.2, 0.1], "(c1') state attack rate x (1 + mu_a s), tatt scale")
        results["c1b"] = S.report(s, "(c1') illumination-dependent state attack", items=True)
        s, _ = fit_candidate(S, 9, [np.log10(0.3), 1.0], [-2.0, 0.2], [1.5, 5.0], [0.2, 0.1], "(c5) state attack slowing with conductance, tatt scale")
        results["c5"] = S.report(s, "(c5) state attack slowing with conductance", items=False)
        s, _ = fit_candidate(S, 5, [0.3, np.log10(3e-4), 0.0], [0.02, -5.5, -1.5], [0.9, -2.0, 1.5], [0.05, 0.1, 0.1], "(c2) second cell, share / persistence / speed")
        results["c2"] = S.report(s, "(c2) second cell in parallel with its own persistence", items=True)
        s, _ = fit_candidate(S, 0, [], [], [], [], "control + dynamics refit (w, tatt, trel, mu)", dyn=True, max_nfev=80)
        results["ctrl-dyn"] = S.report(s, "control with the state constants refitted", items=True)
        s, _ = fit_candidate(S, 4, [np.log10(1.0), 1.0], [-2.0, 0.2], [2.0, 5.0], [0.2, 0.1], "(c1') illumination-dependent attack, dynamics refitted", dyn=True, max_nfev=80)
        results["c1b-dyn"] = S.report(s, "(c1') illumination-dependent state attack with the state constants refitted", items=True)
        s, _ = fit_candidate(S, 5, [0.3, np.log10(3e-4), 0.0], [0.02, -5.5, -1.5], [0.9, -2.0, 1.5], [0.05, 0.1, 0.1], "(c2) second cell, dynamics refitted", dyn=True, max_nfev=80)
        results["c2-dyn"] = S.report(s, "(c2) second cell with the state constants refitted", items=True)
        s, _ = fit_candidate(S, 6, [base.n, base.gam], [0.3, 0.5], [3.0, 4.0], [0.05, 0.05], "(c3) cell law after the integration: n, gamma, dynamics refitted", dyn=True, max_nfev=80)
        results["c3"] = S.report(s, "(c3) cell law after the integration: states chase the light, cond = (sum w s)^gamma", items=True)
        print("\n== summary (H3 / H5 / H7 error rms over the 27 items; statics t20; fine knee t20/t10; bursts weighted/plain)")
        for k, v in results.items():
            print(f"   {k:8s} H3 {v['h3all']:.2f} (100 Hz {v['h3'][100][0]:.2f}, 1 kHz {v['h3'][1000][0]:.2f}, 4 kHz {v['h3'][4000][0]:.2f}) | H5 {v['h5all']:.2f} ({v['h3'][100][1]:.1f}, {v['h3'][1000][1]:.1f}, {v['h3'][4000][1]:.1f}) | "
                  f"H7 {v['h7all']:.2f} | stat t20 {v['stat20']:.3f} | 100 Hz {v['s100']:.3f} | knee {v['kn20']:.3f}/{v['kn10']:.3f} | bursts {v['bw']:.4f}/{v['bp']:.3f}")
    print(f"\n  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
