# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Detector candidates, mixed linear/log family (copy of fit/tools/detector_candidates.py with extra modes; the original is untouched).
Modes 0-4 are the original ones. New modes (all take the release-target offset `depth` and an attack-target offset `aoff`, in dB):
  40  E control: log rectifier -> dB one-pole, attack toward e, release toward Tk - depth (mode 4 with toff and the rest start)
  10  (a) attack in the LINEAR amplitude domain toward |x|, release in dB toward Tk - depth
  11  (b) attack in dB toward e, release as an exponential decay in AMPLITUDE toward 10^((Tk - depth)/20)
  12  (c) half-wave rectifier (positive half cycles only feed the attack), mode-E release
  13  (d) release on the detector (instant attack, release toward Tk - depth), attack one-pole on the GAIN REDUCTION after the curve
  14  (d2) as 13 but the GR smoother also falls with the recover constant
  15  E with the release path always active (physical RC: charge and discharge currents add during attack)
  16  E plus a fixed leak toward the instantaneous log level (two-path release), leak constant p2 = log10(tau_leak s)
  17  mode 15 plus the fixed leak of 16 (always-on release to the threshold reference + leak toward the log level)
  18  mode 15 with a saturating release conductance: rate = kR (v - rest) / (1 + (v - rest) / S), S = 10^p2 dB (current-source limit)
  19  mode 15 with a soft-diode attack: rate = kA V0 ln(1 + exp((e - v) / V0)), V0 = 10^p2 dB (Shockley tail below the node)
  20  mode 15 with a saturating attack (diode + series R): rate = kA u / (1 + u / Sa), u = e - v > 0, Sa = 10^p2 dB
  21  19 and 20 together (V0 = 10^p2, Sa = 10^p3)
  22  mode 15 with a soft-knee attack (diode below its knee): rate = kA (sqrt(u^2 + V0^2) - V0), V0 = 10^p2 dB
  23  mode 15 with a power-law attack: rate = kA u^g (u in dB), g = p2
  24  mode 15 with an attack conductance that grows with the level above the reference: rate = kA u (1 + (v - rest) / Sv), Sv = 10^p2 dB
  25  (b2) mode 11 with the amplitude-domain release always active (dB attack adds on top of the exponential amplitude decay)
  26  (b3) mode 25 with a half-wave rectifier
  27  (b4) generalised domain release, always active: u = 10^(q v / 20) decays toward 10^(q rest / 20) with kR, q = p2 (1 amplitude, 2 power, ->0 log)
  28  (b5) mode 27 with a half-wave rectifier
  30  mode 15 with the attack conductance scaled by the INPUT level above the reference: rate = kA u (1 + max(e - rest, 0) / Sv), Sv = 10^p2 dB
  31  transdiode log-amp pre-stage before mode 15: the log output el slews at (8.69/tau0) (A^g - A^(g-1) B) dB/s, A = a / a0, B = 10^((el - e0)/20),
      a0 = -10 dBFS, tau0 = 10^p2 s (log-amp time constant at -10 dBFS), g = p3 (1 = real transdiode: bandwidth proportional to the current)
  32  mode 31 with a fixed slowdown exponent g = 1 and a soft-knee attack diode V0 = 10^p3 dB (mode 22 law)
  34  mode 30 with a saturating attack current (diode + series R): rate = kA (1 + Xe / Sv) u / (1 + u / Sa), Xe = max(e - rest, 0), Sa = 10^p3 dB
  35  mode 30 with a soft-knee attack diode: rate = kA (1 + Xe / Sv) (sqrt(u^2 + V0^2) - V0), V0 = 10^p3 dB
  36  mode 30 with a free exponent on the level law: rate = kA u ((Xe + Sv) / Sv)^g, g = p3 (1 = mode 30)
  37  mode 30 with a hybrid level reference: rate = kA u (1 + (al Xe + (1 - al) Xv) / Sv), Xv = max(v - rest, 0), al = p3 (1 = mode 30, 0 = mode 24)
  39  mode 30 with a soft-clipped sidechain: a' = Lc tanh(a / Lc), Lc = 10^(p3 / 20) (sidechain headroom in dBFS)
  33  mode 15 with the release conductance scaled by beta = p2 while the attack diode conducts (probe, not physical)
  29  (b6) mode 27 with the release target BELOW the floor by a second offset: decay toward 10^(q (rest - p3) / 20) but the node never falls below rest
Extra parameters: depth (release target = threshold - depth), toff (global threshold shift: the curve is read at v - Tk - toff and the
release target is Tk + toff - depth; stage 3 fitted T with mode E's own detector offset, another topology needs its own), p2, p3.
The node starts at the release target (the reference's burst onsets are immediate at every recover position, so its node sits at the
knee after the 0.5 s pre-roll; starting at -100 dB leaves the model 4-20 dB short at the slow recover positions). Mode 4 keeps the
original -100 dB start so that it reproduces the original harness numbers.
Result (see the candidate log in the task report): mode 30 is the winner (steady rms 0.079 dB / max 0.27, bursts 0.010 weighted / max 0.91)
against mode E's 0.289 / 1.30 and 0.046 / 1.05. Mode 15 (always-on release) is the first step (0.175 / 0.011), the input-level-scaled
attack conductance (30) the second. The release law was checked directly on the reference burst envelopes (slope / (v - Tk) constant
per recover position from 35 dB down to 1 dB, identical for the -30 / -20 / 0 dBFS bursts): a proportional log-domain discharge.
usage: python3 fit/tools/candidates/mixed-domain.py [--modes 40,15,30] [--verbose] [--nfev 80] [--tight] [--eval p1,p2,...] [--trace item,...]"""
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
LN10_20 = np.log(10.0) / 20.0


@njit(cache=True)
def curve_lin(c, x):
    if x <= XG[0]: return c[0]
    if x >= XG[-1]: return c[-1] + (c[-1] - c[-2]) * (x - XG[-1])
    i = int(x - XG[0]); t = x - XG[0] - i
    return c[i] * (1 - t) + c[i + 1] * t


@njit(cache=True)
def run_detector(xs, fs, ta, tr, mode, c, Tk0, depth, toff, p2, p3):
    """xs: SIGNED sidechain samples. Returns the gain reduction (dB) per sample."""
    n = xs.shape[0]
    gr = np.empty(n)
    kA = 1.0 - np.exp(-1.0 / (ta * fs)); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    kL = 1.0 - np.exp(-1.0 / ((10.0 ** p2) * fs)); sat = 10.0 ** p2; v0 = 10.0 ** p2; sa = 10.0 ** p3
    if mode == 21: v0 = 10.0 ** p2; sa = 10.0 ** p3
    fl = 10.0 ** (FLOOR / 20.0)
    Tk = Tk0 + toff
    rest = Tk - depth; rest_lin = 10.0 ** (rest / 20.0); aoff = 0.0; ga = 1.0
    e0 = -10.0; tau0 = 10.0 ** p2; el = FLOOR
    if mode == 32: v0 = 10.0 ** p3
    if mode == 39: sa = 10.0 ** (p3 / 20.0)
    v = FLOOR if mode == 4 else rest; g = 1.0; grs = 0.0; env = 0.0 if mode == 4 else rest_lin
    for i in range(n):
        a = abs(xs[i])
        if mode == 0:
            e = 20.0 * np.log10(a) if a > fl else FLOOR
            if e > v: v += (e - v) * kA
            else: v += (e - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 1:
            e = 20.0 * np.log10(a) if a > fl else FLOOR
            gt = 10.0 ** (-max(curve_lin(c, e - Tk), 0.0) / 20.0)
            if gt < g: g += (gt - g) * kA
            else: g += (gt - g) * kR
            gr[i] = -20.0 * np.log10(g)
        elif mode == 2:
            e = 20.0 * np.log10(a) if a > fl else FLOOR
            gt = max(curve_lin(c, e - Tk), 0.0)
            if gt > grs: grs += (gt - grs) * kA
            else: grs += (gt - grs) * kR
            gr[i] = grs
        elif mode == 3:
            if a > env: env += (a - env) * kA
            else: env += (a - env) * kR
            e = 20.0 * np.log10(env) if env > fl else FLOOR
            gr[i] = max(curve_lin(c, e - Tk), 0.0)
        elif mode == 4 or mode == 40:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            if e > v: v += (e - v) * kA
            else: v += (rest - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 10:
            # (a) linear attack toward |x|, log release toward the threshold reference
            al = a * ga
            if al > env: env += (al - env) * kA
            else:
                vv = 20.0 * np.log10(env) if env > fl else FLOOR
                vv += (rest - vv) * kR
                env = 10.0 ** (vv / 20.0)
            v = 20.0 * np.log10(env) if env > fl else FLOOR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 11:
            # (b) dB attack, amplitude-domain exponential release toward the threshold-referenced level
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            if e > v: v += (e - v) * kA
            else:
                env = 10.0 ** (v / 20.0)
                env += (rest_lin - env) * kR
                v = 20.0 * np.log10(env) if env > fl else FLOOR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 12:
            # (c) half-wave: only positive half cycles feed the attack
            ah = xs[i] if xs[i] > 0.0 else 0.0
            e = (20.0 * np.log10(ah) if ah > fl else FLOOR) + aoff
            if e > v: v += (e - v) * kA
            else: v += (rest - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 13 or mode == 14:
            # (d) detector: instant attack, recover toward the threshold reference; attack one-pole on the GR after the curve
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            if e > v: v = e
            else: v += (rest - v) * kR
            gt = max(curve_lin(c, v - Tk), 0.0)
            if gt > grs: grs += (gt - grs) * kA
            elif mode == 14: grs += (gt - grs) * kR
            else: grs = gt
            gr[i] = grs
        elif mode == 15:
            # E with the release conductance always active
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v: dv += (e - v) * kA
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 16:
            # E plus a fixed leak toward the instantaneous log level while below the node
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            if e > v: v += (e - v) * kA
            else: v += (rest - v) * kR + (e - v) * kL
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 17:
            # always-on release to the threshold reference plus a fixed leak toward the log level while below the node
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v: dv += (e - v) * kA
            else: dv += (e - v) * kL
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 18:
            # always-on release with a saturating conductance (constant-current limit S dB): rate = kR x / (1 + x / S)
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            xr = v - rest
            dv = -kR * xr / (1.0 + abs(xr) / sat)
            if e > v: dv += (e - v) * kA
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 19:
            # always-on release, soft-diode attack (softplus of e - v with softness V0 dB): conducts a little below the node too
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            u = (e - v) / v0
            phi = v0 * (u + np.log1p(np.exp(-u))) if u > 0.0 else v0 * np.log1p(np.exp(u))
            v += (rest - v) * kR + phi * kA
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 20:
            # always-on release, saturating attack (diode + series R): rate = kA u / (1 + u / Sa)
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v:
                u = e - v
                dv += kA * u / (1.0 + u / sa)
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 22:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v:
                u = e - v
                dv += kA * (np.sqrt(u * u + v0 * v0) - v0)
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 23:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v: dv += kA * (e - v) ** p2
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 24:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v: dv += kA * (e - v) * (1.0 + max(v - rest, 0.0) / sat)
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 25 or mode == 26:
            # (b2/b3) amplitude-domain release always active, dB one-pole attack
            ah = a if mode == 25 else (xs[i] if xs[i] > 0.0 else 0.0)
            e = (20.0 * np.log10(ah) if ah > fl else FLOOR) + aoff
            env = 10.0 ** (v / 20.0)
            env += (rest_lin - env) * kR
            v = 20.0 * np.log10(env) if env > fl else FLOOR
            if e > v: v += (e - v) * kA
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 27 or mode == 28 or mode == 29:
            # (b4/b5/b6) release in the domain u = 10^(q v / 20), always active; attack dB one-pole
            ah = a if mode != 28 else (xs[i] if xs[i] > 0.0 else 0.0)
            e = (20.0 * np.log10(ah) if ah > fl else FLOOR) + aoff
            q = p2
            tgt = rest - p3 if mode == 29 else rest
            u = 10.0 ** (q * v / 20.0); ut = 10.0 ** (q * tgt / 20.0)
            u += (ut - u) * kR
            if u > 1e-300: v = (20.0 / q) * np.log10(u)
            else: v = FLOOR
            if mode == 29 and v < rest: v = rest
            if e > v: v += (e - v) * kA
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 30:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v: dv += kA * (e - v) * (1.0 + max(e - rest, 0.0) / sat)
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 31 or mode == 32:
            # transdiode log amp: bandwidth grows with the rectified current; el is its output (dB)
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            A = 10.0 ** ((e - e0) / 20.0); B = 10.0 ** ((el - e0) / 20.0)
            gg = p3 if mode == 31 else 1.0
            # exact logistic step for dB/dt = B (A - B) A^(g-1) / tau0 with A held over the sample (g = 1: a transdiode log amp)
            if A < 1e-6: A = 1e-6
            if B < 1e-12: B = 1e-12
            B = A / (1.0 + (A / B - 1.0) * np.exp(-(A ** gg) / (tau0 * fs)))
            el = e0 + 20.0 * np.log10(B) if B > 1e-12 else FLOOR
            if el < FLOOR: el = FLOOR
            dv = (rest - v) * kR
            if el > v:
                u = el - v
                if mode == 32: dv += kA * (np.sqrt(u * u + v0 * v0) - v0)
                else: dv += kA * u
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 37:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v: dv += kA * (e - v) * (1.0 + (p3 * max(e - rest, 0.0) + (1.0 - p3) * max(v - rest, 0.0)) / sat)
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 34 or mode == 35 or mode == 36 or mode == 39:
            if mode == 39: a = sa * np.tanh(a / sa)
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            dv = (rest - v) * kR
            if e > v:
                u = e - v; xe = max(e - rest, 0.0)
                if mode == 34: dv += kA * (1.0 + xe / sat) * u / (1.0 + u / sa)
                elif mode == 35: dv += kA * (1.0 + xe / sat) * (np.sqrt(u * u + sa * sa) - sa)
                elif mode == 36: dv += kA * u * ((xe + sat) / sat) ** p3
                else: dv += kA * (1.0 + xe / sat) * u
            v += dv
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 33:
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            if e > v: v += (e - v) * kA + (rest - v) * kR * p2
            else: v += (rest - v) * kR
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
        elif mode == 21:
            # soft-diode attack with a series-R saturation: phi = softplus(u; V0), rate = kA phi / (1 + phi / Sa)
            e = (20.0 * np.log10(a) if a > fl else FLOOR) + aoff
            u = (e - v) / v0
            phi = v0 * (u + np.log1p(np.exp(-u))) if u > 0.0 else v0 * np.log1p(np.exp(u))
            v += (rest - v) * kR + kA * phi / (1.0 + phi / sa)
            gr[i] = max(curve_lin(c, v - Tk), 0.0)
    return gr


def env_dB(item_id, ta, tr, mode, depth, toff, p2, p3):
    it = ITEMS[item_id]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    x = protocol.stimulus(st, fs)[0]
    r = RATIOS.index(s.get("discrete_ratio", "4:1")); Tk = T[int(s["discrete_threshold"]) - 1]
    gr = run_detector(x, float(fs), ta, tr, mode, CURVES[r], Tk, depth, toff, p2, p3)
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def steady(ta, tr, mode, depth, toff, p2, p3, level=-10.0, f=1000.0, thr=16, ratio="4:1", secs=3.0):
    t = np.arange(int(secs * FS)) / FS
    x = 10 ** (level / 20) * np.sin(2 * np.pi * f * t)
    gr = run_detector(x, float(FS), ta, tr, mode, CURVES[RATIOS.index(ratio)], T[thr - 1], depth, toff, p2, p3)
    per = int(round(FS / f)); seg = gr[-per * 50:]
    return 20 * np.log10(np.mean(10 ** (-seg / 20.0))) + GAIN12


def h3_dbc(ta, tr, mode, depth, toff, p2, p3, f, level=-10.0, thr=16, secs=2.0):
    """third harmonic of the detector-only chain output (x * gain) on a steady sine, dBc, over the last 0.5 s"""
    t = np.arange(int(secs * FS)) / FS
    x = 10 ** (level / 20) * np.sin(2 * np.pi * f * t)
    gr = run_detector(x, float(FS), ta, tr, mode, CURVES[RATIOS.index("4:1")], T[thr - 1], depth, toff, p2, p3)
    y = x * 10 ** (-gr / 20.0); n = int(0.5 * FS); per = int(round(FS / f)); n = (n // per) * per
    ys = y[-n:]; ts = t[-n:]
    c1 = np.abs(np.mean(ys * np.exp(-2j * np.pi * f * ts))); c3 = np.abs(np.mean(ys * np.exp(-2j * np.pi * 3 * f * ts)))
    c2 = np.abs(np.mean(ys * np.exp(-2j * np.pi * 2 * f * ts)))
    if "--h2" in sys.argv: return 20 * np.log10(c2 / c1 + 1e-30)
    return 20 * np.log10(c3 / c1 + 1e-30)


BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS[:5]] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
N_SS = 36 + 12 + 6


def unpack(p, mode):
    ta = p[:6]; tr = p[6:11]
    depth = p[11] if len(p) > 11 else 0.0
    toff = p[12] if len(p) > 12 else 0.0
    p2 = p[13] if len(p) > 13 else 0.0
    p3 = p[14] if len(p) > 14 else 0.0
    return ta, tr, depth, toff, p2, p3


def residuals(p, mode):
    ta, tr, depth, toff, p2, p3 = unpack(p, mode)
    out = []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            out.append(3.0 * (steady(ta[ai], tr[rci], mode, depth, toff, p2, p3) - F[f"disc_ar_{a}_{rc}"]))
        for l in (-25, 2):
            out.append(3.0 * (steady(ta[ai], tr[2], mode, depth, toff, p2, p3, level=float(l)) - F[f"disc_al_{a}_{l}"]))
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            out.append(3.0 * (steady(ta[ATTACKS.index(a)], tr[2], mode, depth, toff, p2, p3, f=f) - F[f"disc_af_{a}_f{int(f)}"]))
    for iid in BURSTS:
        s = ITEMS[iid]["set"]; ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
        e = env_dB(iid, ta[ai], tr[rci], mode, depth, toff, p2, p3); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate([np.atleast_1d(o) for o in out])


NAMES = {4: "E (original: depth only)", 40: "E+aoff control", 10: "(a) linear attack / log release to threshold", 11: "(b) dB attack / amplitude release to threshold level",
         12: "(c) half-wave rectifier + E release", 13: "(d) detector release + GR attack smoother (instant GR fall)", 14: "(d2) detector release + GR attack smoother (GR falls with kR)",
         15: "E with release always active (simultaneous RC)", 16: "E + fixed leak toward the log level (two-path release)",
         17: "mode 15 + fixed leak toward the log level", 18: "mode 15 with saturating release conductance (S dB)",
         19: "mode 15 with soft-diode (softplus) attack, V0 dB", 20: "mode 15 with saturating attack (diode + series R), Sa dB", 21: "mode 15 with softplus attack and series-R saturation",
         30: "mode 15, attack conductance scaled by input level above rest (Sv)", 31: "transdiode log-amp pre-stage (tau0 at -10 dBFS, exponent g) + mode 15",
         32: "transdiode log-amp (g=1) + soft-knee diode V0 + mode 15", 33: "mode 15 with release scaled by beta while conducting (probe)",
         39: "mode 30 + soft-clipped sidechain (headroom Lc dBFS)", 37: "mode 30 with hybrid level reference (alpha input, 1-alpha node)", 34: "mode 30 + saturating attack current (Sa)", 35: "mode 30 + soft-knee attack diode (V0)", 36: "mode 30 with exponent g on the level law",
         25: "(b2) dB attack / amplitude release always active", 26: "(b3) mode 25 + half-wave rectifier",
         27: "(b4) dB attack / power-q domain release always active, q free", 28: "(b5) mode 27 + half-wave", 29: "(b6) mode 27, target below the floor by p3, clamped at rest",
         22: "mode 15 with soft-knee attack (sqrt(u^2+V0^2)-V0)", 23: "mode 15 with power-law attack u^g", 24: "mode 15 with level-scaled attack conductance (1+(v-rest)/Sv)"}
# extra parameter layout per mode: (initial, lo, hi, x_scale) for depth, toff, p2, p3
D_ = (10.0, -40.0, 80.0, 2.0); TO_ = (0.0, -6.0, 6.0, 0.5)
EXTRAS = {4: [D_],
          40: [D_, TO_], 10: [D_, TO_], 11: [(10.0, -20.0, 80.0, 2.0), TO_], 12: [D_, TO_], 13: [D_, TO_], 14: [D_, TO_], 15: [D_, TO_],
          16: [D_, TO_, (0.0, -2.0, 1.5, 0.3)],
          17: [D_, TO_, (0.3, -2.0, 1.5, 0.3)],
          18: [D_, TO_, (1.5, 0.0, 2.5, 0.3)],
          19: [D_, TO_, (0.3, -1.0, 1.3, 0.3)],
          20: [D_, TO_, (2.2, 0.0, 2.5, 0.3)],
          21: [D_, TO_, (0.3, -1.0, 1.3, 0.3), (1.3, 0.0, 2.5, 0.3)],
          22: [D_, TO_, (0.0, -1.0, 1.3, 0.3)],
          23: [D_, TO_, (1.0, 0.4, 2.5, 0.1)],
          24: [D_, TO_, (1.5, 0.3, 3.0, 0.3)],
          30: [D_, TO_, (1.1, 0.3, 3.0, 0.3)],
          31: [D_, TO_, (-4.3, -6.0, -2.5, 0.3), (1.0, 0.2, 1.5, 0.1)],
          32: [D_, TO_, (-4.3, -6.0, -2.5, 0.3), (0.0, -1.0, 1.0, 0.3)],
          34: [D_, TO_, (1.24, 0.3, 3.0, 0.3), (1.6, 0.8, 2.5, 0.2)],
          39: [D_, TO_, (1.24, 0.3, 3.0, 0.3), (0.0, -12.0, 20.0, 2.0)],
          37: [D_, TO_, (1.24, 0.3, 3.0, 0.3), (0.8, 0.0, 1.0, 0.1)],
          35: [D_, TO_, (1.24, 0.3, 3.0, 0.3), (0.0, -1.0, 1.0, 0.3)],
          36: [D_, TO_, (1.24, 0.3, 3.0, 0.3), (1.0, 0.3, 2.5, 0.1)],
          33: [D_, TO_, (0.8, 0.0, 1.5, 0.1)],
          25: [(10.0, -20.0, 80.0, 2.0), TO_], 26: [(10.0, -20.0, 80.0, 2.0), TO_],
          27: [(10.0, -20.0, 80.0, 2.0), TO_, (1.0, 0.05, 3.0, 0.2)], 28: [(10.0, -20.0, 80.0, 2.0), TO_, (1.0, 0.05, 3.0, 0.2)],
          29: [(5.0, -20.0, 80.0, 2.0), TO_, (1.0, 0.05, 3.0, 0.2), (10.0, 0.0, 60.0, 2.0)]}


def main():
    args = sys.argv[1:]
    modes = [int(m) for m in args[args.index("--modes") + 1].split(",")] if "--modes" in args else [40, 15, 30]
    nfev = int(args[args.index("--nfev") + 1]) if "--nfev" in args else 80
    p0 = [a * 1e-3 for a in ATTACKS] + [0.08, 0.13, 0.32, 0.32, 0.46]
    lo = [1e-5] * 6 + [0.01] * 5; hi = [1.0] * 6 + [5.0] * 5
    for mode in modes:
        ex = EXTRAS[mode]
        pp = p0 + [e[0] for e in ex]; l_ = lo + [e[1] for e in ex]; h_ = hi + [e[2] for e in ex]
        xs = [1e-3] * 6 + [0.05] * 5 + [e[3] for e in ex]
        if "--eval" in args:   # evaluate a given parameter vector (comma separated) instead of fitting
            class R: pass
            r = R(); r.x = np.array([float(t) for t in args[args.index("--eval") + 1].split(",")]); r.fun = residuals(r.x, mode); r.nfev = 0; r.cost = float(np.sum(r.fun ** 2) / 2)
        else:
            tol = 1e-10 if "--tight" in args else 1e-8
            r = least_squares(residuals, pp, bounds=(l_, h_), args=(mode,), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0, ftol=tol, xtol=tol, gtol=tol)
        res = r.fun; ta, tr, depth, toff, p2, p3 = unpack(r.x, mode)
        ss = res[:N_SS] / 3.0; bb = res[N_SS:]
        extra = (f" tau_leak {10 ** p2:.3f} s" if mode in (16, 17) else f" S {10 ** p2:.2f} dB" if mode == 18 else f" V0 {10 ** p2:.3f} dB" if mode == 19
                 else f" Sa {10 ** p2:.2f} dB" if mode == 20 else f" V0 {10 ** p2:.3f} dB Sa {10 ** p3:.2f} dB" if mode == 21
                 else f" V0 {10 ** p2:.3f} dB" if mode == 22 else f" g {p2:.3f}" if mode == 23 else f" Sv {10 ** p2:.2f} dB" if mode == 24
                 else f" Sv {10 ** p2:.2f} dB" if mode == 30 else f" Sv {10 ** p2:.2f} dB Sa {10 ** p3:.2f} dB" if mode == 34
                 else f" Sv {10 ** p2:.2f} dB alpha {p3:.3f}" if mode == 37 else f" Sv {10 ** p2:.2f} dB Lc {p3:.2f} dBFS" if mode == 39 else f" Sv {10 ** p2:.2f} dB V0 {10 ** p3:.3f} dB" if mode == 35 else f" Sv {10 ** p2:.2f} dB g {p3:.3f}" if mode == 36 else f" tau0 {10 ** p2 * 1e3:.3f} ms g {p3:.3f}" if mode == 31
                 else f" tau0 {10 ** p2 * 1e3:.3f} ms V0 {10 ** p3:.3f} dB" if mode == 32 else f" beta {p2:.3f}" if mode == 33
                 else f" q {p2:.3f}" if mode in (27, 28) else f" q {p2:.3f} sub-floor {p3:.2f} dB" if mode == 29 else "")
        print(f"MODE {mode} {NAMES[mode]}: attack ms {np.round(ta * 1e3, 3).tolist()} recover s {np.round(tr, 4).tolist()} depth {depth:.2f} dB toff {toff:.2f} dB"
              + extra + f" | p {np.round(r.x, 5).tolist()} | nfev {r.nfev} cost {r.cost:.2f}")
        print(f"    steady-state rms {np.sqrt(np.mean(ss ** 2)):.3f} dB max {np.max(np.abs(ss)):.2f} | bursts rms {np.sqrt(np.mean(bb ** 2)):.3f} (weighted) max |burst| {np.max(np.abs(bb)):.2f}")
        if "--verbose" in args:
            print("    steady residuals model-ref gain dB (attack rows; cols recover 0.1/0.25/0.5/0.8/1.2 then level -25/+2):")
            k = 0
            for ai, a in enumerate(ATTACKS):
                row = ss[k:k + 7]; k += 7
                print(f"      atk {a:5.1f}: {np.round(row, 2).tolist()}")
            print("    frequency residuals (0.1 ms: 100 Hz, 5 kHz | 1 ms: 100, 5k | 30 ms: 100, 5k):", np.round(ss[42:48], 2).tolist())
            # per-burst weighted rms
            k = 0; per = []
            for iid in BURSTS:
                s = ITEMS[iid]["set"]; ref = np.asarray(F[iid]); ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
                e = env_dB(iid, ta[ai], tr[rci], mode, depth, toff, p2, p3); n = min(len(e), len(ref)); seg = bb[k:k + n]; k += n
                per.append((iid, float(np.sqrt(np.mean(seg ** 2))), float(np.max(np.abs(seg)))))
            per.sort(key=lambda t: -t[1])
            print("    worst bursts (weighted rms, max):", [(i, round(a, 3), round(b, 2)) for i, a, b in per[:6]])
            print("    bdepth bursts:", [(i, round(a, 3), round(b, 2)) for i, a, b in per if "bdepth" in i])
            if "--trace" in args:
                for iid in args[args.index("--trace") + 1].split(","):
                    s_ = ITEMS[iid]["set"]; ref = np.asarray(F[iid]); ai = ATTACKS.index(float(s_["discrete_attack"])); rci = RECOVERS.index(s_["discrete_recover"])
                    e = env_dB(iid, ta[ai], tr[rci], mode, depth, toff, p2, p3); n = min(len(e), len(ref)); d = e[:n] - ref[:n]
                    j = int(np.argmax(np.abs(d)))
                    print(f"    trace {iid}: worst at period {j}: model {e[j]:.2f} ref {ref[j]:.2f}; step at 500 -> model {np.round(e[498:512], 2).tolist()} ref {np.round(ref[498:512], 2).tolist()}")
                    print(f"      release at 2500 -> model {np.round(e[2498:2540:3], 2).tolist()} ref {np.round(ref[2498:2540:3], 2).tolist()}")
            # H3 diagnostic (reference: 0.1 ms: -38.65 / -58.46 / -70.68 dBc at 100 / 1k / 5k; 10 ms: -42.02 / -61.9 / -74.0)
            h = {a: [round(float(h3_dbc(ta[ATTACKS.index(a)], tr[2], mode, depth, toff, p2, p3, f)), 1) for f in (100.0, 1000.0, 5000.0)] for a in (0.1, 10.0)}
            print(f"    H3 dBc at 100 Hz/1 kHz/5 kHz (recover 0.5 s): 0.1 ms {h[0.1]} (ref -38.6/-58.5/-70.7) | 10 ms {h[10.0]} (ref -42.0/-61.9/-74.0)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
