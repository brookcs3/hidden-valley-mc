# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical stage, hypothesis B ("feedforward-fixed-ratio"): the reference implements the optical stage as a FEED-FORWARD gain computer
(log level of the INPUT above a threshold, fixed ratio, hard or short soft knee) followed by opto-like smoothing of the gain-reduction
target (the three-state attack/release of src/dsp/Opto.hpp applied to the target, in the conductance or in the dB domain), i.e. a
digital "opto-style" compressor rather than a physical feedback loop. The same computer and smoothers are also run inside a feedback
wrapper (sidechain = the divider output) so that feed-forward and feedback are compared at their respective best fits.

The mirror (numba, one sample at a time, the protocol's stimuli and features):
    sidechain  u = x (feed-forward) or u = x * g[n-1] (feedback); then the sidechain low pass implied by discriminate_opto.json (C):
               second-order Butterworth at 5.2 kHz (fixed, switchable with --no-sclp)
    detector   det 0: |u| instantaneous   det 1: RMS, one-pole on u^2 (tau_d)   det 2: peak, instant attack / one-pole release (tau_d)
    computer   L = 20 log10(env); GR_t = s_eff * knee(L - T, W)   (quadratic knee of width W dB; s = 1 - 1/ratio;
               s_eff = s feed-forward, s / (1 - s) feedback, so both wrappers have the same closed-loop ratio)
    target     conductance c* = 10^(GR_t/20) - 1 (or the dB value for the dB-domain smoother); light lag: one-pole tau_L
    smoother   sm 0: three states, one-poles in c (attack/release constant per state, weights)   sm 1: the same in dB
               sm 2: photocell ODE, bimolecular recombination: dc/dt = b_a (c*^2 - c^2) rising, b_r (c*^2 - c^2) - k_r (c - c*)
                     falling, plus a slow linear state (weight w_s, tau_s)
               sm 3: the same rise, but the fall is TARGET-INDEPENDENT: dc/dt = -(b_r c^2 + k_r c), clamped at c*  (the only
                     feed-forward smoother that can copy the reference's release, see (E) below; kept as the devil's advocate)
               sm 4: hybrid: linear one-pole rise in c (tau_a), the bimolecular fall of sm 2
    gain       g = 1 / (1 + c); out = x * g * make-up  (make-up = the position's measured no-GR gain, so the bias (D) is absorbed)
Per configuration: (1) static fit at threshold 20 (T, s, W [, tau_d]); (2) cell dynamics on the burst family (opto_burst_*, opto_blen_*,
opto_pulses, stage-4 weights); (3) static refit at 20, T_k for thresholds 6..24 with s and W shared (and a per-threshold slope check);
(4) metrics: static rms/max over thresholds 6/10/14/20/24 x -50..14, the fine knee sweep (A), the bias law (D), the burst residuals and
release law, the above-knee steps (E) as a HOLD-OUT (attack/release windows, the release-identity test: the step -10 -> -20 release
against the burst -10 -> -50 release, identical in the reference for the first 5 ms), harmonics under GR; (5) a joint refit of the
dynamics on bursts + steps and the same metrics again. The C++ feedback model (fit/data/constants.json) gets the same metrics.
usage: cd <repo> && python3 -u fit/tools/candidates/opto-feedforward-fixed-ratio.py [--configs A,B,...] [--nfev 50] [--no-cpp] [--no-ref] [--no-sclp] [--warm]
       (2-3.5 min per configuration; run --warm once, then one process per configuration in parallel: --configs X --no-cpp --no-ref)
       A  det0 FF sm0 (the hypothesis as stated: Opto.hpp's states on a feed-forward target, conductance domain)
       B  det0 FF sm1 (dB domain)      C  det1(RMS) FF sm0      H  det2(peak) FF sm0
       D  det0 FF sm2 (photocell ODE)  K  det0 FF sm3 (target-independent release)
       E  det0 FB sm2 (the feedback alternative, same computer)   J  det0 FB sm0 (feedback with the linear states)
       F  det1 FB sm2   I  det2 FB sm2   G  det1 FF sm2   M  det0 FB sm4   N  det1 FB sm4   P  det1 FF sm4"""
import os, sys, time, json, numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.signal import butter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402

LEVELS = list(range(-50, 15, 2))
THRS = [6, 10, 14, 20, 24]
THRS_ALL = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
BURSTS = [f"opto_burst_{lb}" for lb in (-26, -18, -10, -2)] + [f"opto_blen_{bl}" for bl in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
HARMS = ["opto_harm_t18_-10_f1000", "opto_harm_t18_-10_f100", "opto_harm_t18_-10_f4000", "opto_harm_t22_0_f1000", "opto_harm_t14_-10_f1000",
         "opto_harm_t18_-20_f1000", "opto_harm_t22_-10_f1000"]
DISC = json.load(open(os.path.join(HERE, "..", "..", "data", "discriminate_opto.json")))
STEPS = {k: (v["from"], v["to"], np.asarray(v["per_cycle_gain_db"])) for k, v in DISC["steps"].items()}
STEP_WIN = [(1990, 2200), (3990, 4300)]   # cycles used for the joint fit and the step metrics
NFEV = 50
SCLP = "--no-sclp" not in sys.argv
LPC = np.zeros(5)
if SCLP:
    sos = butter(2, 5200.0, fs=FS, output="sos")[0]
    LPC[:] = [sos[0], sos[1], sos[2], sos[4], sos[5]]


def g0_of(thr):
    """the position's no-GR gain (make-up included), dB: the flat part of its measured static curve"""
    return float(np.mean([F[f"opto_static_t{thr}_{l}"] for l in range(-50, -39, 2)]))


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def run_loop(x, fs, T, s, W, tau_d, det, sm, fb, tau_L, dyn, lp, lpc):
    """linear gain per sample (without make-up). dyn: sm 0/1 -> [w0 w1 w2 ta0 ta1 ta2 tr0 tr1 tr2]; sm 2/3 -> [b_a b_r k_r w_s tau_s]
    (b in 1/(c s), k_r in 1/s); sm 4 -> [tau_a b_r k_r w_s tau_s]"""
    n = x.shape[0]; dt = 1.0 / fs
    g = np.empty(n)
    kd = 1.0 - np.exp(-dt / tau_d) if tau_d > 0.0 else 1.0
    kL = 1.0 - np.exp(-dt / tau_L) if tau_L > 0.0 else 1.0
    seff = s / (1.0 - s) if fb == 1 else s
    hw = 0.5 * W
    b0 = lpc[0]; b1 = lpc[1]; b2 = lpc[2]; a1 = lpc[3]; a2 = lpc[4]
    z1 = 0.0; z2 = 0.0
    ms = 0.0; env = 0.0; Lt = 0.0; c = 0.0; ss = 0.0; gprev = 1.0
    st0 = 0.0; st1 = 0.0; st2 = 0.0
    kA0 = kA1 = kA2 = kR0 = kR1 = kR2 = 0.0; w0 = w1 = w2 = 0.0
    ba = br = kr = ws = 0.0; kS = 0.0; kAa = 0.0
    if sm <= 1:
        w0 = dyn[0]; w1 = dyn[1]; w2 = dyn[2]
        kA0 = 1.0 - np.exp(-dt / dyn[3]); kA1 = 1.0 - np.exp(-dt / dyn[4]); kA2 = 1.0 - np.exp(-dt / dyn[5])
        kR0 = 1.0 - np.exp(-dt / dyn[6]); kR1 = 1.0 - np.exp(-dt / dyn[7]); kR2 = 1.0 - np.exp(-dt / dyn[8])
    else:
        ba = dyn[0]; br = dyn[1]; kr = dyn[2]; ws = dyn[3]
        kS = 1.0 - np.exp(-dt / dyn[4])
        if sm == 4: kAa = 1.0 - np.exp(-dt / dyn[0])
    for i in range(n):
        u = x[i] * gprev if fb == 1 else x[i]
        if lp == 1:
            yv = b0 * u + z1
            z1 = b1 * u - a1 * yv + z2
            z2 = b2 * u - a2 * yv
            u = yv
        a = abs(u)
        if det == 0:
            env = a
        elif det == 1:
            ms += (u * u - ms) * kd
            env = np.sqrt(ms)
        else:
            if a > env: env = a
            else: env += (a - env) * kd
        L = 20.0 * np.log10(env + 1e-12)
        over = L - T
        if over <= -hw: gr = 0.0
        elif over >= hw: gr = seff * over
        else: gr = seff * (over + hw) * (over + hw) / (2.0 * W)
        if sm == 1:
            Lt += (gr - Lt) * kL
            st0 += (Lt - st0) * (kA0 if Lt > st0 else kR0)
            st1 += (Lt - st1) * (kA1 if Lt > st1 else kR1)
            st2 += (Lt - st2) * (kA2 if Lt > st2 else kR2)
            GR = w0 * st0 + w1 * st1 + w2 * st2
            gi = 10.0 ** (-GR / 20.0)
        else:
            cs = 10.0 ** (gr / 20.0) - 1.0
            if sm == 0:
                Lt += (cs - Lt) * kL
                st0 += (Lt - st0) * (kA0 if Lt > st0 else kR0)
                st1 += (Lt - st1) * (kA1 if Lt > st1 else kR1)
                st2 += (Lt - st2) * (kA2 if Lt > st2 else kR2)
                ct = w0 * st0 + w1 * st1 + w2 * st2
            else:
                Lt += (cs * cs - Lt) * kL          # the light, proportional to the equilibrium c^2
                tgt = np.sqrt(Lt)
                if Lt > c * c:
                    if sm == 4: c += (tgt - c) * kAa
                    else:
                        c += ba * (Lt - c * c) * dt
                        if c > tgt: c = tgt
                else:
                    if sm == 2: c += (br * (Lt - c * c) - kr * (c - tgt)) * dt
                    else: c -= (br * c * c + kr * c) * dt
                    if c < tgt: c = tgt
                ss += (c - ss) * kS
                ct = (1.0 - ws) * c + ws * ss
            gi = 1.0 / (1.0 + ct)
        g[i] = gi; gprev = gi
    return g


# ------------------------------------------------------------------------------------------------ stimuli and features (protocol / discriminate equivalents)
def lockin_gain_db(y, x, f, fs, last_s):
    per = fs / f; nper = max(1, int(round(last_s * f)))
    n1 = int(round(int(len(y) / per) * per)); n0 = int(round(n1 - nper * per))
    t = np.arange(n0, n1) / fs
    cy = 2.0 * np.mean(y[n0:n1] * np.exp(-2j * np.pi * f * t)); cx = 2.0 * np.mean(x[n0:n1] * np.exp(-2j * np.pi * f * t))
    return float(20 * np.log10(abs(cy) / abs(cx)))


def env_db(y, x, f, fs):
    per = int(round(fs / f)); m = len(y) // per
    t = np.arange(per) / fs; w = np.exp(-2j * np.pi * f * t)
    cy = np.abs((y[:m * per].reshape(m, per) * w).mean(axis=1)); cx = np.abs((x[:m * per].reshape(m, per) * w).mean(axis=1))
    return 20 * np.log10(cy / (cx + 1e-30) + 1e-30)


def step_signal(a, b, secs=(2.0, 2.0, 2.0), f=1000.0, fs=FS):
    t = np.arange(int(round(sum(secs) * fs))) / fs
    env = np.full_like(t, 10 ** (a / 20)); n1 = int(secs[0] * fs); n2 = n1 + int(secs[1] * fs)
    env[n1:n2] = 10 ** (b / 20)
    return env * np.sin(2 * np.pi * f * t)


def per_cycle_gain(x, y, f=1000.0, fs=FS):
    per = int(round(fs / f)); m = len(x) // per
    xr = np.sqrt(np.mean(x[:m * per].reshape(m, per) ** 2, axis=1)); yr = np.sqrt(np.mean(y[:m * per].reshape(m, per) ** 2, axis=1))
    return 20 * np.log10(yr / xr)


def rms_gain_db(x, y, fs=FS):
    n = int(fs); r = lambda a: np.sqrt(np.mean(a[-n:] ** 2))
    return float(20 * np.log10(r(y) / r(x)))


class Cfg:
    def __init__(self, name, det, sm, fb):
        self.name, self.det, self.sm, self.fb = name, det, sm, fb
        self.T = {20: -26.8}; self.s = 0.62; self.W = 1.0; self.tau_d = 0.5e-3 if det else 0.0; self.tau_L = 0.3e-3
        self.dyn = np.array([0.6, 0.3, 0.1, 1e-3, 3e-3, 20e-3, 5e-3, 30e-3, 300e-3]) if sm <= 1 else \
            np.array([2e-3 if sm == 4 else 140.0, 80.0, 6.0, 0.02, 0.3])

    def gain(self, x, fs, thr):
        return run_loop(x, float(fs), self.T[thr], self.s, self.W, self.tau_d, self.det, self.sm, self.fb, self.tau_L, self.dyn, 1 if SCLP else 0, LPC)

    def out(self, x, fs, thr):
        return x * self.gain(x, fs, thr) * 10 ** (g0_of(thr) / 20.0)

    def render(self, item):
        st = item["stim"]; fs = item["fs"]; thr = int(item["set"]["optical_threshold"])
        x = protocol.stimulus(st, fs)[0]
        return x, self.out(x, fs, thr)

    def static_gain(self, thr, level, f=1000.0, secs=2.5):
        it = {"stim": {"kind": "sine", "level": float(level), "f": f, "secs": secs}, "fs": FS, "set": {"optical_threshold": thr}}
        x, y = self.render(it)
        return lockin_gain_db(y, x, f, FS, 0.5)

    def knee_gain(self, thr, level):
        """discriminate_opto's measure: 3 s sine, RMS gain over the last second"""
        x = protocol.stimulus({"kind": "sine", "level": float(level), "f": 1000.0, "secs": 3.0}, FS)[0]
        return rms_gain_db(x, self.out(x, FS, thr))

    def env(self, item_id):
        it = ITEMS[item_id]; x, y = self.render(it)
        return env_db(y, x, it["stim"]["f"], it["fs"])

    def step(self, key):
        a, b, _ = STEPS[key]; x = step_signal(a, b)
        return per_cycle_gain(x, self.out(x, FS, 20))

    def harm(self, item_id):
        it = ITEMS[item_id]; x, y = self.render(it)
        return protocol.feature(it, np.stack([y, y]), it["fs"])

    def state(self):
        return (dict(self.T), self.s, self.W, self.tau_d, self.tau_L, self.dyn.copy())

    def restore(self, st):
        self.T, self.s, self.W, self.tau_d, self.tau_L, self.dyn = dict(st[0]), st[1], st[2], st[3], st[4], st[5].copy()


# ------------------------------------------------------------------------------------------------ residuals and metrics
def static_resid(cfg, thr, levels=LEVELS):
    return np.array([cfg.static_gain(thr, l) - F[f"opto_static_t{thr}_{l}"] for l in levels])


def burst_resid(envs, weighted=True):
    out = []
    for iid, e in envs.items():
        ref = np.asarray(F[iid]); n = min(len(e), len(ref)); d = np.asarray(e[:n]) - ref[:n]
        if weighted:
            w = np.where(ref[:n] < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)
            out.append(w * d)
        else:
            out.append(d)
    return np.concatenate(out)


def step_resid(steps):
    out = []
    for k, gc in steps.items():
        ref = STEPS[k][2]
        for a, b in STEP_WIN:
            out.append((gc[a:b] - ref[a:b]) / np.sqrt((b - a) / 100.0))
    return np.concatenate(out)


def burst_metrics(envs):
    """plain rms/max, stage-4 weighted rms, and the attack (first 20 periods of the burst), release (first 100 periods after it) and
    tail (the rest) rms, over the burst family"""
    att, rel, tail = [], [], []
    for iid, e in envs.items():
        ref = np.asarray(F[iid]); n = min(len(e), len(ref)); d = np.asarray(e[:n]) - ref[:n]
        st = ITEMS[iid]["stim"]; f = st["f"]; n0 = int(round(st["pre_s"] * f))
        if st["kind"] == "burst":
            n1 = n0 + int(round(st["burst_s"] * f))
            att.append(d[n0:n0 + 20]); rel.append(d[n1:n1 + 100]); tail.append(d[n1 + 100:])
        else:
            att.append(d[n0:n0 + 20])
            on = int(round(st["on_s"] * f)); off = int(round(st["off_s"] * f))
            for k in range(st["count"]):
                a = n0 + k * (on + off) + on
                rel.append(d[a:a + min(100, off)])
            tail.append(d[n0 + st["count"] * (on + off):])
    plain = burst_resid(envs, False); wr = burst_resid(envs, True)
    r = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
    return dict(rms=float(np.sqrt(np.mean(plain ** 2))), max=float(np.max(np.abs(plain))), wrms=float(np.sqrt(np.mean(wr ** 2))),
                attack=r(att), release=r(rel), tail=r(tail))


def step_metrics(steps, envs):
    """(E): rms/max over the attack window (40 cycles) and the release window (100 cycles) of the three steps, and the release-identity
    test: the step -10 -> -20 release minus the burst -10 -> -50 release, cycles 0-4 and 5-9 after the step"""
    att, rel = [], []
    for k, gc in steps.items():
        ref = STEPS[k][2]; d = gc - ref
        att.append(d[2000:2040]); rel.append(d[4000:4100])
    att = np.concatenate(att); rel = np.concatenate(rel)
    ident = steps["-20_-10"][4000:4010] - np.asarray(envs["opto_burst_-10"])[2500:2510]
    return dict(s_attack=float(np.sqrt(np.mean(att ** 2))), s_release=float(np.sqrt(np.mean(rel ** 2))),
                s_max=float(max(np.abs(att).max(), np.abs(rel).max())), ident=ident)


REF_IDENT = STEPS["-20_-10"][2][4000:4010] - np.asarray(F["opto_burst_-10"])[2500:2510]


def release_law(env, g0, n1, npts=30):
    """dc/dt = -k c^m read off an envelope after the burst end (period n1): m from log(-dc/dt) against log(c), central differences"""
    e = np.asarray(env[n1 + 1:n1 + npts + 1]); c = 10 ** ((g0 - e) / 20.0) - 1.0
    c = np.maximum(c, 1e-6); dc = -(c[2:] - c[:-2]) / 2.0   # per period (1 ms at 1 kHz)
    sel = (dc > 0) & (c[1:-1] > 0.02)
    if sel.sum() < 4:
        return float("nan"), float("nan")
    A = np.vstack([np.log(c[1:-1][sel]), np.ones(sel.sum())]).T
    m, lk = np.linalg.lstsq(A, np.log(dc[sel]), rcond=None)[0]
    return float(m), float(np.exp(lk))


def attack_fractions(steps, envs):
    """first-cycle fraction of the gain change (dB) for the burst -50 -> -10 and the three steps"""
    e = np.asarray(envs["opto_burst_-10"]); pre = e[:20].mean(); fin = e[1500:2400].mean()
    out = {"burst -50>-10": (e[500] - pre) / (fin - pre)}
    for k, gc in steps.items():
        p1 = gc[1990:1999].mean(); p2 = gc[3990:3999].mean()
        out[f"step {k}"] = (gc[2000] - p1) / (p2 - p1)
    return out


def knee_check(cfg, thr):
    """(A): the model on discriminate_opto's fine sweep (0.5 dB steps) against the reference: gain residual rms/max and the GR per step
    around the knee"""
    lv = np.asarray(DISC["knee"][str(thr)]["levels"]); ref = np.asarray(DISC["knee"][str(thr)]["gain_db"])
    mod = np.array([cfg.knee_gain(thr, l) for l in lv])
    d = mod - ref; gr_m = mod[0] - mod; gr_r = ref[0] - ref
    i0 = int(np.argmax(gr_r > 0.1)) - 2
    return dict(rms=float(np.sqrt(np.mean(d ** 2))), max=float(np.abs(d).max()), levels=lv[i0:i0 + 8], gr_model=gr_m[i0:i0 + 8], gr_ref=gr_r[i0:i0 + 8])


def bias_law(T):
    """(D): loss of no-GR gain relative to threshold 1 against the sidechain drive (T_20 - T_k, dB): exponent q of loss ~ drive^q"""
    ks = [k for k in (8, 12, 16, 20, 22, 24) if k in T]
    loss = np.array([g0_of(1) - g0_of(k) for k in ks]); drive = np.array([T[20] - T[k] for k in ks])
    sel = loss > 2e-3
    q = np.polyfit(drive[sel], np.log(loss[sel]), 1)[0] * 20.0 / np.log(10.0)
    return float(q), ks, loss, drive


def fmt(v, nd=3):
    return np.array2string(np.asarray(v, dtype=float), precision=nd, suppress_small=True, max_line_width=250)


# ------------------------------------------------------------------------------------------------ fits
def fit_static20(cfg, nfev, with_tau_d=True):
    """T, s, W (and the detector time constant in the pre-fit only: after the dynamics fit it belongs to the bursts)"""
    td = bool(cfg.det) and with_tau_d
    p0 = [cfg.T[20], cfg.s, np.log10(cfg.W)] + ([np.log10(cfg.tau_d)] if td else [])
    lo = [-40.0, 0.3, -2.0] + ([-4.5] if td else []); hi = [-10.0, 0.9, 1.3] + ([-1.0] if td else [])
    def res(p):
        cfg.T[20] = p[0]; cfg.s = p[1]; cfg.W = 10 ** p[2]
        if td: cfg.tau_d = 10 ** p[3]
        return static_resid(cfg, 20)
    r = least_squares(res, p0, bounds=(lo, hi), x_scale=[0.5, 0.02, 0.1] + ([0.1] if td else []), diff_step=1e-3, max_nfev=nfev)
    res(r.x)
    return r


def fit_thr(cfg, thr, nfev=20, with_slope=False):
    """T_k (and, for the slope check, s_k) of another threshold position; with_slope restores the shared s afterwards"""
    cfg.T[thr] = cfg.T[20] + 1.0 * (thr - 20); s_shared = cfg.s
    def res(p):
        cfg.T[thr] = p[0]
        if with_slope: cfg.s = p[1]
        return static_resid(cfg, thr)
    p0 = [cfg.T[thr]] + ([cfg.s] if with_slope else [])
    r = least_squares(res, p0, bounds=([-60.0] + ([0.3] if with_slope else []), [10.0] + ([0.9] if with_slope else [])),
                      x_scale=[0.5] + ([0.02] if with_slope else []), diff_step=1e-3, max_nfev=nfev)
    res(r.x); s_k = cfg.s
    if with_slope: cfg.s = s_shared
    return r, s_k


def fit_dyn(cfg, nfev, with_steps):
    if cfg.sm <= 1:
        p0 = list(cfg.dyn[:3]) + list(np.log10(cfg.dyn[3:])) + [np.log10(cfg.tau_L)]
        lo = [0.0] * 3 + [-4.5] * 3 + [-3.5] * 3 + [-5.0]; hi = [1.0] * 3 + [-0.5] * 3 + [1.0] * 3 + [-2.0]
        xs = [0.05] * 3 + [0.1] * 6 + [0.1]
        def unpack(p):
            w = np.abs(np.array(p[:3])); w = w / max(w.sum(), 1e-9)
            cfg.dyn = np.concatenate([w, 10 ** np.array(p[3:9])]); cfg.tau_L = 10 ** p[9]
    else:
        p0 = list(np.log10(cfg.dyn[:3])) + [cfg.dyn[3], np.log10(cfg.dyn[4]), np.log10(cfg.tau_L)]
        lo = [-4.5 if cfg.sm == 4 else 0.0, 0.0, -2.0, 0.0, -2.0, -5.0]; hi = [-1.0 if cfg.sm == 4 else 3.5, 3.5, 3.0, 0.3, 1.0, -2.0]
        xs = [0.1, 0.1, 0.1, 0.01, 0.1, 0.1]
        def unpack(p):
            cfg.dyn = np.array([10 ** p[0], 10 ** p[1], 10 ** p[2], p[3], 10 ** p[4]]); cfg.tau_L = 10 ** p[5]
    if cfg.det:   # the detector time constant is a dynamic parameter too
        p0 = list(p0) + [np.log10(cfg.tau_d)]; lo = lo + [-4.5]; hi = hi + [-1.0]; xs = xs + [0.1]
    def res(p):
        unpack(p)
        if cfg.det: cfg.tau_d = 10 ** p[-1]
        out = [burst_resid({i: cfg.env(i) for i in BURSTS}, True)]
        if with_steps: out.append(step_resid({k: cfg.step(k) for k in STEPS}))
        return np.concatenate(out)
    r = least_squares(res, np.clip(p0, lo, hi), bounds=(lo, hi), x_scale=xs, diff_step=2e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
    res(r.x)
    return r


def describe(cfg):
    d = f"T20 {cfg.T[20]:.2f} dBFS, s {cfg.s:.4f} (ratio {1 / (1 - cfg.s):.2f}:1), knee W {cfg.W:.2f} dB, tau_L {cfg.tau_L * 1e3:.3f} ms"
    if cfg.det: d += f", tau_d {cfg.tau_d * 1e3:.3f} ms"
    if cfg.sm <= 1:
        d += f" | w {fmt(cfg.dyn[:3])} attack ms {fmt(cfg.dyn[3:6] * 1e3, 2)} release ms {fmt(cfg.dyn[6:9] * 1e3, 1)}"
    else:
        rise = f"tau_a {cfg.dyn[0] * 1e3:.2f} ms" if cfg.sm == 4 else f"b_a {cfg.dyn[0]:.1f}"
        d += f" | {rise} b_r {cfg.dyn[1]:.1f} /(c s), k_r {cfg.dyn[2]:.2f} /s, slow w {cfg.dyn[3]:.3f} tau {cfg.dyn[4] * 1e3:.0f} ms"
    return d


# ------------------------------------------------------------------------------------------------ the reference's own numbers, and the C++ model
def reference_section():
    print("== reference (fit/data/reference_features.json, discriminate_opto.json)")
    print("  release law dc/dt = -k c^m (c = 10^(GR/20) - 1) from the envelopes, first 30 ms after the burst:")
    for iid in ("opto_burst_-18", "opto_burst_-10", "opto_burst_-2", "opto_burst_thr12"):
        it = ITEMS[iid]; st = it["stim"]; n1 = int(round((st["pre_s"] + st["burst_s"]) * st["f"]))
        thr = int(it["set"]["optical_threshold"]); m, k = release_law(F[iid], g0_of(thr), n1)
        print(f"    {iid}: m {m:.2f}, k {k:.4f} /(c^m ms)")
    envs = {i: np.asarray(F[i]) for i in BURSTS}; steps = {k: v[2] for k, v in STEPS.items()}
    print(f"  (E) release identity, step -10>-20 minus burst -10>-50, cycles 0-9 after the step (dB): {fmt(REF_IDENT, 3)}")
    print("      the two releases coincide until the divider output at -20 dBFS reaches the knee (gain about -6.7 dB), then part")
    print(f"  (E) first-cycle attack fractions (dB): " + ", ".join(f"{k} {v:.3f}" for k, v in attack_fractions(steps, envs).items()))
    d = DISC["nogr"]["-50.0"]
    print(f"  (D) no-GR gain by threshold at -50 dBFS: " + " ".join(f"{k}:{v:+.3f}" for k, v in d.items()))
    drives = {8: 12.41, 12: 19.46, 16: 23.22, 20: 27.05, 22: 29.08, 24: 31.13}   # stage 4 shift table (build_stage4.log)
    q, ks, loss, drv = bias_law({k: -v for k, v in drives.items()})
    print(f"  (D) with the stage-4 shift table as drive: loss {fmt(loss, 3)} dB at drive {fmt(drv, 1)} dB -> loss ~ drive^{q:.2f}")


def cpp_section():
    print("== C++ feedback model (stage 4 calibration) on the same metrics")
    cal = load_cal(); t0 = time.time()
    envs = {i: np.asarray(render_item(ITEMS[i], cal)) for i in BURSTS}
    bm = burst_metrics(envs)
    print(f"  bursts: plain rms {bm['rms']:.3f} max {bm['max']:.2f} | weighted rms {bm['wrms']:.3f} | attack {bm['attack']:.3f} release {bm['release']:.3f} tail {bm['tail']:.3f} dB")
    m, k = release_law(envs["opto_burst_-10"], g0_of(20), 2500)
    print(f"  release law on opto_burst_-10: m {m:.2f} k {k:.4f}")
    steps = {}
    for key, (a, b, _) in STEPS.items():
        x = step_signal(a, b); y = MODEL.render(x, FS, {"optical_bypass": "In", "optical_threshold": 20}, cal=cal)[0]
        steps[key] = per_cycle_gain(x, y)
    sm = step_metrics(steps, envs)
    print(f"  (E) steps hold-out: attack rms {sm['s_attack']:.3f} release rms {sm['s_release']:.3f} max {sm['s_max']:.2f} dB | identity cycles 0-4 {fmt(sm['ident'][:5], 3)} (ref {fmt(REF_IDENT[:5], 3)})")
    print(f"  (E) first-cycle attack fractions: " + ", ".join(f"{k} {v:.3f}" for k, v in attack_fractions(steps, envs).items()))
    res = {thr: np.array([render_item(ITEMS[f"opto_static_t{thr}_{l}"], cal) - F[f"opto_static_t{thr}_{l}"] for l in LEVELS]) for thr in THRS}
    allr = np.concatenate(list(res.values()))
    print(f"  static (5 thresholds x 33 levels): rms {np.sqrt(np.mean(allr ** 2)):.3f} max {np.abs(allr).max():.2f} | t20 rms {np.sqrt(np.mean(res[20] ** 2)):.3f} max {np.abs(res[20]).max():.2f}")
    print(f"  t20 residual by level: {fmt(res[20], 2)}")
    for iid in HARMS[:4]:
        h = render_item(ITEMS[iid], cal); ref = F[iid]
        print(f"  {iid}: H3 {h['h'][1]:.1f} (ref {ref['h'][1]:.1f}) H5 {h['h'][3]:.1f} (ref {ref['h'][3]:.1f}) gain {h['gain_db']:.2f} (ref {ref['gain_db']:.2f})")
    print(f"  ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------------------------------------ one configuration
def metrics(cfg, tag, out):
    envs = {i: cfg.env(i) for i in BURSTS}
    bm = burst_metrics(envs)
    steps = {k: cfg.step(k) for k in STEPS}
    sm = step_metrics(steps, envs)
    out.update({f"{tag}_{k}": v for k, v in bm.items()}); out.update({f"{tag}_{k}": v for k, v in sm.items() if k != "ident"})
    out[f"{tag}_ident_err"] = float(np.sqrt(np.mean((sm["ident"][:5] - REF_IDENT[:5]) ** 2)))
    print(f"  [{tag}] BURSTS: plain rms {bm['rms']:.3f} max {bm['max']:.2f} | weighted rms {bm['wrms']:.3f} | attack {bm['attack']:.3f} release {bm['release']:.3f} tail {bm['tail']:.3f} dB")
    for iid in ("opto_burst_-18", "opto_burst_-10", "opto_burst_-2"):
        e = envs[iid]; ref = np.asarray(F[iid])
        print(f"    {iid} attack periods 1-6: model {fmt(e[500:506], 2)} ref {fmt(ref[500:506], 2)}")
        print(f"    {'':>{len(iid)}} release 0/3/6/9/15/30/60/100 ms: model {fmt(e[[2500, 2503, 2506, 2509, 2515, 2530, 2560, 2600]], 2)} ref {fmt(ref[[2500, 2503, 2506, 2509, 2515, 2530, 2560, 2600]], 2)}")
        m, k = release_law(e, g0_of(20), 2500); print(f"    {'':>{len(iid)}} release law: m {m:.2f} k {k:.4f} (ref m {release_law(ref, g0_of(20), 2500)[0]:.2f})")
    e = envs["opto_pulses"]; ref = np.asarray(F["opto_pulses"])
    print(f"    pulses, 2nd/3rd pulse attack periods: model {fmt(e[[750, 751, 752, 1000, 1001, 1002]], 2)} ref {fmt(ref[[750, 751, 752, 1000, 1001, 1002]], 2)}")
    print(f"  [{tag}] (E) STEPS: attack rms {sm['s_attack']:.3f} release rms {sm['s_release']:.3f} max {sm['s_max']:.2f} dB")
    for k, gc in steps.items():
        ref = STEPS[k][2]
        print(f"    step {k}: attack cycles 0-5: model {fmt(gc[2000:2006], 2)} ref {fmt(ref[2000:2006], 2)}")
        print(f"    {'':>{len(k) + 5}} release 0/2/4/6/10/15/30/60 ms: model {fmt(gc[[4000, 4002, 4004, 4006, 4010, 4015, 4030, 4060]], 2)} ref {fmt(ref[[4000, 4002, 4004, 4006, 4010, 4015, 4030, 4060]], 2)}")
    print(f"    release identity (step -10>-20 minus burst -10>-50), cycles 0-9: model {fmt(sm['ident'], 3)}")
    print(f"    {'':>16} reference {fmt(REF_IDENT, 3)}  -> model error over cycles 0-4: {out[f'{tag}_ident_err']:.3f} dB")
    print(f"    first-cycle attack fractions: " + ", ".join(f"{k} {v:.3f}" for k, v in attack_fractions(steps, envs).items()))
    return envs


def run_cfg(cfg, nfev):
    t0 = time.time()
    print(f"\n== {cfg.name}: det {cfg.det} ({['instantaneous |x|', 'RMS one-pole', 'peak'][cfg.det]}), smoother {cfg.sm} "
          f"({['3-state linear in c', '3-state linear in dB', 'bimolecular photocell ODE', 'bimolecular rise, target-independent decay', 'linear rise in c, bimolecular fall'][cfg.sm]}), "
          f"{'FEEDBACK' if cfg.fb else 'FEED-FORWARD'}")
    r = fit_static20(cfg, nfev)
    print(f"  static20 pre-fit: rms {np.sqrt(np.mean(r.fun ** 2)):.3f} max {np.abs(r.fun).max():.2f} | {describe(cfg)}")
    r = fit_dyn(cfg, nfev, with_steps=False)
    print(f"  dynamics fit on the bursts: nfev {r.nfev}, weighted rms {np.sqrt(np.mean(r.fun ** 2)):.4f}")
    r = fit_static20(cfg, nfev, with_tau_d=False)
    print(f"  static20 refit: rms {np.sqrt(np.mean(r.fun ** 2)):.3f} max {np.abs(r.fun).max():.2f}")
    print(f"  {describe(cfg)}")
    out = {}
    # other thresholds, shared s and W; and the per-threshold slope check
    res = {20: r.fun}; sk = {20: cfg.s}
    for thr in THRS_ALL:
        if thr == 20: continue
        rr, _ = fit_thr(cfg, thr); res[thr] = rr.fun
    for thr in THRS:
        if thr != 20: _, sk[thr] = fit_thr(cfg, thr, with_slope=True)
    allr = np.concatenate([res[t] for t in THRS])
    comp = np.concatenate([res[t][np.array([F[f"opto_static_t{t}_{l}"] for l in LEVELS]) < g0_of(t) - 0.3] for t in THRS])
    out.update(static_rms=float(np.sqrt(np.mean(allr ** 2))), static_max=float(np.abs(allr).max()), t20_rms=float(np.sqrt(np.mean(res[20] ** 2))),
               t20_max=float(np.abs(res[20]).max()), comp_rms=float(np.sqrt(np.mean(comp ** 2))))
    print(f"  T by position: {' '.join(f'{t}:{cfg.T[t]:.2f}' for t in sorted(cfg.T))} | steps/position 6-10 {(cfg.T[10] - cfg.T[6]) / 4:.2f}, 10-14 {(cfg.T[14] - cfg.T[10]) / 4:.2f}, 14-20 {(cfg.T[20] - cfg.T[14]) / 6:.2f}, 20-24 {(cfg.T[24] - cfg.T[20]) / 4:.2f} dB")
    print(f"  STATIC 5 thresholds x 33 levels: rms {out['static_rms']:.3f} max {out['static_max']:.2f} dB (compressing region rms {out['comp_rms']:.3f}) | per threshold rms "
          + " ".join(f"t{t}:{np.sqrt(np.mean(res[t] ** 2)):.3f}" for t in THRS))
    print(f"  per-threshold slope if s were free: " + " ".join(f"t{t}:{sk[t]:.3f}" for t in THRS) + f" (shared {cfg.s:.3f})")
    print(f"  t20 residual by level (-50..14): {fmt(res[20], 2)}")
    for t in (6, 24):
        print(f"  t{t} residual by level (-50..14): {fmt(res[t], 2)}")
    for thr in (20, 10):
        kc = knee_check(cfg, thr); out[f"knee{thr}_rms"] = kc["rms"]; out[f"knee{thr}_max"] = kc["max"]
        print(f"  (A) fine knee, threshold {thr}: gain residual rms {kc['rms']:.3f} max {kc['max']:.2f} dB | levels {fmt(kc['levels'], 1)}")
        print(f"      GR model {fmt(kc['gr_model'], 2)}\n      GR ref   {fmt(kc['gr_ref'], 2)}")
    q, ks, loss, drv = bias_law(cfg.T); out["bias_q"] = q
    print(f"  (D) no-GR loss vs this fit's drive (T20 - Tk): {fmt(loss, 3)} dB at {fmt(drv, 1)} dB -> loss ~ drive^{q:.2f}; the feed-forward computer gives 0 dB below the knee (absorbed in the make-up here)")
    metrics(cfg, "bursts-fit", out)
    for f in (100, 3000, 8000):
        d = np.array([cfg.static_gain(20, l, f=float(f)) - F[f"opto_static_f{f}_{l}"] for l in range(-40, 11, 4)])
        print(f"  hold-out static {f} Hz (-40..8 step 4): {fmt(d, 2)}")
    for iid in HARMS:
        h = cfg.harm(iid); ref = F[iid]
        hf = lambda v: "n/a" if v is None else f"{v:.1f}"
        print(f"  {iid}: H3 {hf(h['h'][1])} (ref {hf(ref['h'][1])}) H5 {hf(h['h'][3])} (ref {hf(ref['h'][3])}) H7 {hf(h['h'][5])} (ref {hf(ref['h'][5])}) gain {h['gain_db']:.2f} (ref {ref['gain_db']:.2f})")
        if iid == HARMS[0]: out["h3"], out["h5"] = h["h"][1], h["h"][3]
    # joint refit: bursts + steps
    r = fit_dyn(cfg, nfev, with_steps=True)
    rs = fit_static20(cfg, nfev, with_tau_d=False)
    print(f"  JOINT refit (bursts + steps): nfev {r.nfev}; static20 rms {np.sqrt(np.mean(rs.fun ** 2)):.3f} max {np.abs(rs.fun).max():.2f}")
    print(f"  {describe(cfg)}")
    out["joint_t20_rms"] = float(np.sqrt(np.mean(rs.fun ** 2)))
    metrics(cfg, "joint", out)
    print(f"  ({time.time() - t0:.0f} s)")
    return out


def main():
    global NFEV
    if "--nfev" in sys.argv: NFEV = int(sys.argv[sys.argv.index("--nfev") + 1])
    cfgs = {"A": Cfg("A", 0, 0, 0), "B": Cfg("B", 0, 1, 0), "C": Cfg("C", 1, 0, 0), "H": Cfg("H", 2, 0, 0), "D": Cfg("D", 0, 2, 0),
            "K": Cfg("K", 0, 3, 0), "E": Cfg("E", 0, 2, 1), "J": Cfg("J", 0, 0, 1), "F": Cfg("F", 1, 2, 1), "I": Cfg("I", 2, 2, 1),
            "G": Cfg("G", 1, 2, 0), "M": Cfg("M", 0, 4, 1), "N": Cfg("N", 1, 4, 1), "P": Cfg("P", 1, 4, 0)}
    sel = sys.argv[sys.argv.index("--configs") + 1].split(",") if "--configs" in sys.argv else list(cfgs)
    if "--warm" in sys.argv:   # compile the kernel and write numba's cache from this file itself (parallel --configs runs then share it)
        cfgs["A"].gain(np.zeros(4800), FS, 20); print("kernel compiled"); return
    if "--no-ref" not in sys.argv: reference_section()
    if "--no-cpp" not in sys.argv: cpp_section()
    results = {}
    for k in sel:
        if k in cfgs: results[k] = run_cfg(cfgs[k], NFEV)
    if results:
        print("\n== summary (static over thresholds 6/10/14/20/24; steps (E) as a hold-out after the burst fit, then after the joint fit)")
        for k, o in results.items():
            print(f"  {k}: static rms {o['static_rms']:.3f} max {o['static_max']:.2f} (t20 {o['t20_rms']:.3f}/{o['t20_max']:.2f}) knee20 {o['knee20_rms']:.3f} | "
                  f"bursts rms {o['bursts-fit_rms']:.3f} max {o['bursts-fit_max']:.2f} w {o['bursts-fit_wrms']:.3f} | steps hold-out attack {o['bursts-fit_s_attack']:.3f} "
                  f"release {o['bursts-fit_s_release']:.3f} max {o['bursts-fit_s_max']:.2f} ident-err {o['bursts-fit_ident_err']:.3f} | joint: bursts rms {o['joint_rms']:.3f} "
                  f"steps attack {o['joint_s_attack']:.3f} release {o['joint_s_release']:.3f} max {o['joint_s_max']:.2f} ident-err {o['joint_ident_err']:.3f} | H3 {o['h3']:.1f} H5 {o['h5']:.1f}")


if __name__ == "__main__":
    main()
