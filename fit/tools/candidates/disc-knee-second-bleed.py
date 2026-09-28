# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage, soft-ratio knee: can the node fall below the release reference? Directions (c) and (d) of the knee hole.

THE PROBLEM. The detector node rests at rest = T - depth (depth 0.40 dB) and cannot follow a level below it (only the bleed acts), so
every sine at or below rest leaves the node at rest and the gain computer reads one value, C_r(-depth). The reference shows zero
gain reduction at low levels, yet its 1.2:1, 2:1 and 3:1 static curves still compress at the rest level (0.49 / 1.08 / 0.58 dB) and
fade to zero over the 2 dB below it; 4:1, 6:1 and FLOOD start above rest. Pinning the node-domain curves to zero at rest makes
silence exact and costs up to 1.1 dB at the soft-ratio knees (stage-3 rendered statics 0.136 -> 0.216 dB rms, max 0.73).

HYPOTHESES, all fitted on one residual (the stage-3b dynamic residual: the 54 steady cells x 3, every burst envelope per period,
the DUAL items; plus the whole 6 x 24 x 25 static grid with the curves refitted in the TRUE node domain at every evaluation, so
the static family is always the best the mechanism can do):
  (d) mode 0  the current node with the static family refitted in the node domain: the node is simulated on every static point
              (mean over the last 0.5 s at the capture setting, 1 ms / 0.5 s) instead of assumed at L + d0, the curve is the
              isotonic fit in that domain pinned to zero at the silence node, and the rendered check carries the within-cycle
              ripple through the C++ PCHIP. If the node's own onset off rest with a hard knee could make the soft-ratio statics,
              this fit would show it; the band at and below rest is its irreducible residual (the node has no ripple there).
  (c) mode 1  two-way bleed to rest (as now) plus a slow leak toward rest - D with conductance kR / lam. Silence equilibrium
              rest - D kL / (kR + kL). Algebraically this is the current node with a deeper rest and a faster bleed.
      mode 2  as mode 1 with the leak's time constant fixed in seconds (tg), so the equivalent depth depends on the recover position.
      mode 3  two-way bleed plus a leak toward the log floor (-100 dB), conductance kR / lam: the GERMANIUM hook already in
              DiscreteStage (leakRatio). Its silence equilibrium depends on the threshold's absolute level, so the node map is
              simulated per threshold position and the 24-threshold grid judges it.
      mode 4  the bleed to rest is one-way (a diode + resistor: conducts only while the node is above rest) and the slow leak toward
              rest - D is what takes the node below rest. Above rest this is the current node up to the leak's small addition;
              below rest a quiet signal is tracked down to rest - D (the diode charges, only the leak discharges) and silence
              rests at rest - D. The statics follow for any leak speed (they start from silence and charge UP); only the burst
              onsets (the node starts D dB lower) and the reference's post-burst tail (how fast the node sinks below rest) can
              set the leak's constant, which is why --reference measures that tail on the plug-in.
      mode 5  mode 4 with the leak's time constant fixed in seconds.
      mode 7  one-way bleed to rest plus a leak toward the log floor: silence goes to the floor, a quiet level is tracked all the
              way down, so a burst after a -50 dBFS pre-roll starts 12 dB below rest at threshold 16.
  (b') mode 6 the node stays at rest, but the release reference and the curve's origin move together with the RATIO switch,
              T_r = T + off_r (0 for 4:1, 6:1, FLOOD). Kept from the first run of this harness as the cross-check of direction (b),
              which is another harness's subject (disc-knee-per-ratio.py).
The curves are never taken from constants.json: every mode refits them in its own node domain (pooled isotonic regression, pinned
to zero at the silence node, the threshold convention shifted so that the silence node lands on a knot of the 1 dB grid, so that
the C++ PCHIP is exactly zero there). Reported per mode: rendered statics through the mirror (exact ripple, lock-in gain, PCHIP)
on the stage-3 report set (thresholds 4 / 12 / 20 x levels -30 / -15 / 0), the 198-item set (levels -30..0 step 3), the whole grid
and the knee band; the standing gain reduction at silence per ratio; the steady-state table; the bursts; the DUAL items; against
the current constants evaluated the same way (the mirror is checked against the C++ engine on them first).

RESULTS (2026-09-27, appended after the runs; the numbers are in the log build/disc-knee-second-bleed*.log):
  see the end of this docstring, filled in by hand from the runs.

usage: cd <repo> && python3 -u fit/tools/candidates/disc-knee-second-bleed.py [--modes 0,4,5,7,1,2,3] [--nfev 40] [--no-cpp] [--cpp]
           [--fix D,lam] [--scan] [--wstatic W] [--reference] [--reference-only]
  --fix D,lam   hold the leak's D (dB) and lam (or tg in seconds for modes 2 and 5) and fit only the sixteen stage-3 constants
  --scan        modes 4 / 5: evaluate a grid of (D, lam or tg) with the detector at the current constants before the joint fit
  --wstatic W   weight of the static grid in the residual (1.0)
  --cpp         also render the mode-0 / mode-6 result through the C++ engine (those need no code change)
  --r3x         reference and current model side by side: the 30 ms onset after pre-roll tones from 3 dB below to 4 dB above rest
  --reference   black-box renders of the reference plug-in (needs the licensed plug-in and Pedalboard, fit/measure/pa.py): the fine
                static knee at threshold 16, the post-burst tail at three recover positions, the burst onset after pre-rolls at
                several quiet levels, and a below-rest tone from silence and after a burst. The numbers are written to
                build/disc-knee-second-bleed-reference.json; when that file exists every report also simulates the same stimuli
                through the candidate (its fitted node and node-domain curves at 1.2:1, 2:1 and 4:1) and prints them side by side.
  (numba caches need a real file: never pipe this through stdin)"""
import os, sys, time, json, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402
import stage3_discrete as S3  # noqa: E402  (fit_static, pava, curve_at, the engine's curve grid; nothing here writes constants)

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
N = S3.N; X0 = S3.X0; DX = S3.DX; XG = S3.XG
FLOOR = -100.0
cal = load_cal()
GAIN = cal[MODEL.field("d_gain_db")]; XGN = float(cal[MODEL.fields["x_gain_db"][0]]); GAIN12 = float(GAIN[11] + XGN)
S3.GAIN12 = GAIN12
LEVELS = list(range(-60, 13, 3))
NAMES = {0: "(d) current node, curves refitted in the true node domain",
         1: "(c) two-way bleed to rest + slow leak to rest - D, kL = kR / lam",
         2: "(c) two-way bleed to rest + slow leak to rest - D, fixed tg",
         3: "(c) two-way bleed to rest + leak to the log floor (GERMANIUM hook), kL = kR / lam",
         4: "(c) one-way bleed to rest (diode) + slow leak to rest - D, kL = kR / lam",
         5: "(c) one-way bleed to rest (diode) + slow leak to rest - D, fixed tg",
         6: "(b') node at rest; the release reference and the curve origin move with the ratio switch, T_r = T + off_r",
         7: "(c) one-way bleed to rest (diode) + leak to the log floor, kL = kR / lam"}
OFF0 = np.zeros(6)
ABS_MODES = (3,)          # the node map depends on the threshold's absolute level: simulate it per threshold
ONEWAY_MODES = (4, 5, 7)
OUT = os.path.join(HERE, "..", "..", "..", "build")


# ------------------------------------------------------------------------------------------------ the curve, as the C++ evaluates it
@njit(cache=True)
def pchip_build(y):
    """Fritsch-Carlson slopes, as Curve::build (DX = 1)"""
    m = np.empty(N); d = np.empty(N - 1)
    for i in range(N - 1): d[i] = (y[i + 1] - y[i]) / DX
    m[0] = d[0]; m[N - 1] = d[N - 2]
    for i in range(1, N - 1): m[i] = 0.0 if d[i - 1] * d[i] <= 0.0 else 0.5 * (d[i - 1] + d[i])
    for i in range(N - 1):
        if d[i] == 0.0:
            m[i] = 0.0; m[i + 1] = 0.0; continue
        a = m[i] / d[i]; b = m[i + 1] / d[i]; s = a * a + b * b
        if s > 9.0:
            t = 3.0 / np.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]
    return m


@njit(cache=True)
def pchip_at(y, m, x):
    u = (x - X0) / DX
    if u <= 0.0: return y[0]
    if u >= N - 1.0: return y[N - 1] + m[N - 1] * (x - (X0 + DX * (N - 1.0)))
    i = int(u); t = u - i; t2 = t * t; t3 = t2 * t
    return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * DX * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * DX * m[i + 1]


# ------------------------------------------------------------------------------------------------ the node
@njit(cache=True)
def leak_coefs(mode, fs, tr, lam, kR, rest, D):
    """returns (kL, target, one-way bleed, silence node); rest and the target in the same (absolute) frame"""
    if mode == 0 or mode == 6:
        return 0.0, rest, False, rest
    if mode == 1 or mode == 4:
        kL = 1.0 - np.exp(-1.0 / (lam * tr * fs)); tgt = rest - D
    elif mode == 2 or mode == 5:
        kL = 1.0 - np.exp(-1.0 / (lam * fs)); tgt = rest - D
    else:   # 3, 7
        kL = 1.0 - np.exp(-1.0 / (lam * tr * fs)); tgt = FLOOR
    oneway = mode == 4 or mode == 5 or mode == 7
    vs = tgt if oneway else (kR * rest + kL * tgt) / (kR + kL)
    return kL, tgt, oneway, vs


@njit(cache=True)
def node(a, fs, ta, tr, dual, t2, c2, Tk, depth, sv, mode, D, lam):
    """a: |sidechain|. The current detector (always-on bleed toward rest, level-scaled attack diode, DUAL second node) plus the
    candidate second bleed. Starts at the silence equilibrium, as DiscreteStage::reset would."""
    n = a.shape[0]; v = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    k2 = (1.0 - np.exp(-1.0 / (t2 * fs))) if dual else 0.0
    fl = 10.0 ** (FLOOR / 20.0)
    rest = Tk - depth
    kL, tgt, oneway, vs = leak_coefs(mode, fs, tr, lam, kR, rest, D)
    x = vs; w = vs
    for i in range(n):
        e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
        if oneway:
            dv = (rest - x) * kR if x > rest else 0.0
        else:
            dv = (rest - x) * kR
        dv += (tgt - x) * kL
        if e > x:
            xe = e - rest
            f = 1.0 + xe / sv if xe > 0.0 else 1.0
            dv += (e - x) * (1.0 - np.exp(-rA * f))
        x += dv
        if dual:
            flow = (x - w) * k2
            x -= flow
            w += flow / c2
        v[i] = x
    return v


@njit(parallel=True, cache=True)
def sine_grid(levels, f0s, tas, trs, Tks, ridx, curves, slopes, fs, depth, sv, mode, D, lam, secs, last):
    """steady sines, one per entry: returns (lock-in gain in dB through the PCHIP curve of ridx, mean node over the last `last` s).
    The gain is the fundamental's complex amplitude over the last `last` seconds relative to the input, as the protocol's gain_db."""
    m = levels.shape[0]; gain = np.empty(m); vmean = np.empty(m)
    for j in prange(m):
        n = int(secs * fs); n0 = n - int(last * fs)
        amp = 10.0 ** (levels[j] / 20.0); wv = 2.0 * np.pi * f0s[j] / fs
        a = np.empty(n)
        for i in range(n): a[i] = abs(amp * np.sin(wv * i))
        v = node(a, fs, tas[j], trs[j], False, 1.0, 1.0, Tks[j], depth, sv, mode, D, lam)
        r = ridx[j]; sr = 0.0; si = 0.0; sm = 0.0
        for i in range(n0, n):
            g = pchip_at(curves[r], slopes[r], v[i] - Tks[j])
            if g < 0.0: g = 0.0
            s = np.sin(wv * i); cs = np.cos(wv * i)
            y = amp * s * (10.0 ** (-g / 20.0))
            sr += y * s; si += y * cs
            sm += v[i]
        gain[j] = 20.0 * np.log10(2.0 * np.sqrt(sr * sr + si * si) / (n - n0) / amp)
        vmean[j] = sm / (n - n0)
    return gain, vmean


@njit(cache=True)
def gr_of_node(v, Tk, y, m):
    """gain reduction per sample from the PCHIP curve (y, m) read at v - Tk, never below zero"""
    n = v.shape[0]; g = np.empty(n)
    for i in range(n):
        x = pchip_at(y, m, v[i] - Tk)
        g[i] = x if x > 0.0 else 0.0
    return g


@njit(cache=True)
def env_lockin(x, gr, f, fs):
    """the protocol's 'env' feature on y = x 10^(-gr/20): gain of the fundamental per period by lock-in, dB"""
    per = fs / f; m = int(x.shape[0] / per); out = np.empty(m); w = 2.0 * np.pi * f / fs
    for k in range(m):
        n0 = int(np.floor(k * per + 0.5)); n1 = int(np.floor((k + 1) * per + 0.5))
        cyr = 0.0; cyi = 0.0; cxr = 0.0; cxi = 0.0
        for i in range(n0, n1):
            c = np.cos(w * i); s = np.sin(w * i)
            y = x[i] * 10.0 ** (-gr[i] / 20.0)
            cyr += y * c; cyi -= y * s; cxr += x[i] * c; cxi -= x[i] * s
        out[k] = 20.0 * np.log10(np.sqrt(cyr * cyr + cyi * cyi) / (np.sqrt(cxr * cxr + cxi * cxi) + 1e-30) + 1e-30)
    return out


# ------------------------------------------------------------------------------------------------ the static family in the node domain
def pooled_pava(x, y):
    """isotonic regression after pooling identical x (the pinned node maps many levels onto one node value)"""
    xr = np.round(x, 6)
    ux, inv = np.unique(xr, return_inverse=True)
    cnt = np.bincount(inv).astype(float); sy = np.bincount(inv, weights=y)
    fit_u = S3.pava(ux, sy / cnt, cnt)
    return ux, fit_u, fit_u[inv]


def static_setup():
    data = []
    for ri, r in enumerate(RATIOS):
        for k in range(1, 25):
            for L in LEVELS:
                data.append((ri, k - 1, float(L), GAIN12 - F[f"disc_static_{r}_t{k}_{L}"]))
    d = np.array(data)
    return d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2], d[:, 3]


YS = np.concatenate([np.arange(-70.0, -8.0, 1.0), np.arange(-8.0, 8.0, 0.05), np.arange(8.0, 80.0, 1.0)])
YS_COARSE = np.concatenate([np.arange(-70.0, -6.0, 2.0), np.arange(-6.0, 6.0, 0.25), np.arange(6.0, 80.0, 2.0)])
ZERO_CURVES = np.zeros((6, N)); ZERO_SLOPES = np.zeros((6, N))


def node_map(pd, mode, Tabs):
    """g(y): mean node relative to T on a 2.5 s sine y dB above T at the capture setting (1 ms, 0.5 s), from the silence start.
    Tabs: the absolute threshold(s) at which to simulate: one value for the shift-invariant modes, all 24 for the absolute ones."""
    ta, tr, t2, c2, depth, sv, D, lam = pd
    ys = YS_COARSE if len(Tabs) > 1 else YS
    m = len(ys); out = []
    for T in Tabs:
        _, vm = sine_grid(ys + T, np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), np.full(m, T), np.zeros(m, dtype=np.int64),
                          ZERO_CURVES, ZERO_SLOPES, float(FS), depth, sv, mode, D, lam, 2.5, 0.5)
        out.append(vm - T)
    return ys, out


def silence_node(pd, mode, Tabs):
    """the silence equilibrium relative to T, at the absolute threshold Tabs"""
    ta, tr, t2, c2, depth, sv, D, lam = pd
    kR = 1.0 - np.exp(-1.0 / (tr[2] * FS))
    return float(leak_coefs(mode, float(FS), tr[2], lam, kR, Tabs - depth, D)[3] - Tabs)


class Static:
    def __init__(self, Tp):
        self.Tp = Tp
        self.ri, self.ki, self.L, self.G = static_setup()
        self.y0 = -20.0 - Tp[15]   # the stage-3 convention for d0: level -20 dBFS at threshold 16

    def fit(self, pd, mode, off=OFF0):
        """curves refitted in the node domain: returns residual (3600), node-domain x per point, fitted values, d0, x_sil, curve points.
        off: per-ratio shift of the reference (mode 6); each ratio's node domain is relative to its own T + off_r"""
        T16 = self.Tp[15]   # d0 is found below; the map at T' is close enough for the absolute modes
        ys, g = node_map(pd, mode, [T16] if mode not in ABS_MODES else list(self.Tp))
        d0 = float(np.interp(self.y0, ys, g[0] if mode not in ABS_MODES else g[15]) - self.y0)
        y = self.L - self.Tp[self.ki] - d0 - off[self.ri]
        if mode in ABS_MODES:
            xn = np.empty_like(y)
            for k in range(24):
                m = self.ki == k
                xn[m] = np.interp(y[m], ys, g[k])
        else:
            xn = np.interp(y, ys, g[0])
        xs = silence_node(pd, mode, T16 + d0)
        fit = np.empty_like(self.G); curves_pts = []
        for r in range(6):
            m = self.ri == r
            ux, fu, fpts = pooled_pava(xn[m], self.G[m])
            fpts = np.where(xn[m] <= xs + 1e-6, 0.0, fpts)
            fu = np.where(ux <= xs + 1e-6, 0.0, fu)
            fit[m] = fpts; curves_pts.append((ux, fu))
        return fit - self.G, xn, fit, d0, xs, curves_pts


def curves_on_grid(curves_pts, xs, delta):
    """node-domain curves on the 1 dB grid after the convention shift delta (x' = x - delta), pinned at and below the silence knot"""
    C = np.zeros((6, N)); ks = int(round(xs - delta))
    for r in range(6):
        ux, fu = curves_pts[r]
        ux = ux - delta
        c = np.interp(XG, ux, fu, left=0.0)
        beyond = XG > ux.max()
        if np.any(beyond):
            tail = max(0, len(fu) - 40)
            slope = (fu[-1] - fu[tail]) / max(1e-6, ux[-1] - ux[tail])
            c[beyond] = fu[-1] + slope * (XG[beyond] - ux.max())
        c[XG <= ks + 1e-9] = 0.0
        C[r] = np.maximum.accumulate(np.maximum(c, 0.0))
    S = np.stack([pchip_build(C[r]) for r in range(6)])
    return C, S


# ------------------------------------------------------------------------------------------------ the dynamic residual (stage 3b)
BURST_IDS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS] + [f"disc_dual_blen_{b}" for b in (0.1, 0.5, 2.0, 8.0)] + ["disc_dual_pulses"] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
XSIG = {iid: protocol.stimulus(ITEMS[iid]["stim"], ITEMS[iid]["fs"])[0] for iid in BURST_IDS}
STIM = {iid: np.abs(XSIG[iid]) for iid in BURST_IDS}
ENV_N = {iid: min(len(F[iid]), len(STIM[iid]) // int(round(ITEMS[iid]["fs"] / ITEMS[iid]["stim"]["f"]))) for iid in BURST_IDS}


def env_of_item(iid, curves_L, Tp, pd, mode, d0):
    """gain per period (dB, lock-in as the protocol's env feature) of the mirror on a burst item, through the L-domain 4:1 curve"""
    ta, tr, t2, c2, depth, sv, D, lam = pd
    it = ITEMS[iid]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    T = Tp[int(s["discrete_threshold"]) - 1] + d0
    ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
    v = node(STIM[iid], float(fs), ta[ai], tr[rci], dual, t2, c2, T, depth, sv, mode, D, lam)
    r = RATIOS.index(s.get("discrete_ratio", "4:1"))
    gr = np.maximum(S3.curve_at(curves_L[r], v - T), 0.0)
    gr[v <= T - depth] = 0.0
    return env_lockin(XSIG[iid], gr, float(st["f"]), float(fs)) + GAIN12


def steady_cells(curves_L, Tp, pd, mode, d0):
    """the 54 steady cells the stage-3 way: L-domain 4:1 curve at the mean node; returns model - reference gain (dB)"""
    ta, tr, t2, c2, depth, sv, D, lam = pd
    T = Tp[15] + d0
    lv, f0, tas, trs, refs = [], [], [], [], []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            lv.append(-10.0); f0.append(1000.0); tas.append(ta[ai]); trs.append(tr[rci]); refs.append(F[f"disc_ar_{a}_{rc}"])
        for l in (-25, 2):
            lv.append(float(l)); f0.append(1000.0); tas.append(ta[ai]); trs.append(tr[2]); refs.append(F[f"disc_al_{a}_{l}"])
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            lv.append(-10.0); f0.append(f); tas.append(ta[ATTACKS.index(a)]); trs.append(tr[2]); refs.append(F[f"disc_af_{a}_f{int(f)}"])
    m = len(lv)
    _, vm = sine_grid(np.array(lv), np.array(f0), np.array(tas), np.array(trs), np.full(m, T), np.zeros(m, dtype=np.int64),
                      ZERO_CURVES, ZERO_SLOPES, float(FS), depth, sv, mode, D, lam, 3.0, 0.5)
    pred = GAIN12 - np.maximum(S3.curve_at(curves_L[3], vm - T), 0.0)
    return pred - np.array(refs)


def dyn_resid(curves_L, Tp, pd, mode, d0):
    out = [3.0 * steady_cells(curves_L, Tp, pd, mode, d0)]
    for iid in BURST_IDS:
        ref = np.asarray(F[iid]); e = env_of_item(iid, curves_L, Tp, pd, mode, d0); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate(out)


def burst_split(res):
    """(weighted rms of the non-DUAL bursts, max), (DUAL blen + pulses rms, max) from the dynamic residual after the 54 cells; per item"""
    k = 54; nd, du = [], []; per = {}
    for iid in BURST_IDS:
        n = ENV_N[iid]; seg = res[k:k + n]; k += n
        (du if "dual" in iid else nd).append(seg); per[iid] = (float(np.sqrt(np.mean(seg ** 2))), float(np.max(np.abs(seg))))
    nd = np.concatenate(nd); du = np.concatenate(du)
    return (float(np.sqrt(np.mean(nd ** 2))), float(np.max(np.abs(nd)))), (float(np.sqrt(np.mean(du ** 2))), float(np.max(np.abs(du)))), per


# ------------------------------------------------------------------------------------------------ parameters
def unpack(p, mode):
    ta = list(p[0:6]); tr = list(p[6:12]); t2 = float(p[12]); c2 = float(p[13]); depth = float(p[14]); sv = float(10.0 ** p[15])
    if mode == 0 or mode == 6:
        D, lam = 0.0, 1.0
    else:
        D = float(p[16]); lam = float(10.0 ** p[17])
    return ta, tr, t2, c2, depth, sv, D, lam


def off_of(p, mode):
    off = np.zeros(6)
    if mode == 6: off[:3] = p[16:19]
    return off


def start_and_bounds(mode):
    p0 = list(cal[MODEL.field("d_tatt")]) + list(cal[MODEL.field("d_trel")]) + [float(cal[MODEL.field("d_dual_t2")][0]), float(cal[MODEL.field("d_dual_c2")][0]),
                                                                                   float(cal[MODEL.field("d_rel_depth_db")][0]), float(np.log10(cal[MODEL.field("d_att_sv_db")][0]))]
    lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, -20.0, 0.3]; hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 40.0, 3.0]
    xs = [1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 1.0, 0.3]
    if mode in (1, 4):
        p0 += [2.5, np.log10(3.0)]; lo += [0.3, -1.0]; hi += [40.0, 3.5]; xs += [1.0, 0.3]
    elif mode in (3, 7):
        p0 += [0.0, np.log10(30.0)]; lo += [0.0, 0.0]; hi += [0.0 + 1e-9, 4.0]; xs += [1.0, 0.3]
    elif mode in (2, 5):
        p0 += [2.5, np.log10(1.0)]; lo += [0.3, -1.5]; hi += [40.0, 2.0]; xs += [1.0, 0.3]
    elif mode == 6:
        p0 += [-2.2, -2.2, -1.0]; lo += [-6.0] * 3; hi += [0.0] * 3; xs += [0.5] * 3
    return np.clip(p0, lo, hi), lo, hi, xs


# ------------------------------------------------------------------------------------------------ reports
REPORT54 = [(r, k, L) for r in range(6) for k in (4, 12, 20) for L in (-30, -15, 0)]
REPORT198 = [(r, k, L) for r in range(6) for k in (4, 12, 20) for L in range(-30, 1, 3)]
FULL = [(r, k, L) for r in range(6) for k in range(1, 25) for L in LEVELS]


def rendered_static(items, C, S, Tfin, depth_fin, pd, mode, off=OFF0):
    """mirror render of static items: gain through the node + PCHIP curve with the exact ripple; returns model - reference (dB)"""
    ta, tr, t2, c2, _, sv, D, lam = pd
    lv = np.array([float(L) for _, _, L in items]); Tk = np.array([Tfin[k - 1] + off[r] for r, k, _ in items]); ri = np.array([r for r, _, _ in items], dtype=np.int64)
    m = len(items)
    g, _ = sine_grid(lv, np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), Tk, ri, C, S, float(FS), depth_fin, sv, mode, D, lam, 2.5, 0.5)
    ref = np.array([F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in items])
    return g + GAIN12 - ref


def rms_max(e):
    e = np.asarray(e); return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def report(mode, p, Tp, curves_L, st, label, cpp=False):
    pd = unpack(p, mode); ta, tr, t2, c2, depth, sv, D, lam = pd; off = off_of(p, mode)
    sres, xn, fit, d0, xs, cpts = st.fit(pd, mode, off)
    # convention shift: put the silence node on a knot (for the absolute modes: the silence node at threshold 16)
    ks = int(round(xs)); delta = xs - ks
    if ks < X0: ks = int(X0); delta = 0.0   # silence far below the grid (floor leaks): nothing to pin, no shift
    Tfin = Tp + d0 + delta; depth_fin = depth + delta
    C, S = curves_on_grid(cpts, xs, delta)
    print(f"  {label}: attack ms {np.round(np.array(ta) * 1e3, 3).tolist()} recover s {np.round(tr, 4).tolist()} t2 {t2:.4f} c2 {c2:.2f} depth {depth:.3f} Sv {sv:.2f}"
          + (f" off {np.round(off[:3], 3).tolist()}" if mode == 6 else (f" D {D:.3f} dB {'tg' if mode in (2, 5) else 'lam'} {lam:.3f}" + (f" (tg = lam tr: {lam * tr[2]:.2f} s at 0.5 s)" if mode in (1, 3, 4, 7) else "")) if mode else ""))
    print(f"    d0 {d0:+.3f} dB; silence node {xs:+.3f} dB below T (stage-3 convention); convention shift {delta:+.3f} -> rest at {-depth_fin:+.3f}, silence knot {ks:+d}")
    rest_rel = d0 - depth   # rest relative to Tp, in the level domain (T = Tp + d0, rest = T - depth)
    band = st.L - st.Tp[st.ki] - rest_rel   # relative to the 4:1 rest, so the bands are comparable across modes
    sel = st.G > 0.3
    print(f"    node-domain static family: rms {rms_max(sres)[0]:.3f} max {rms_max(sres)[1]:.2f} over all 3600 | compressing (>0.3 dB) rms {rms_max(sres[sel])[0]:.3f} max {rms_max(sres[sel])[1]:.2f}")
    for lo_, hi_, nm in ((-99, -2.5, "L < rest - 2.5"), (-2.5, 0.0, "rest - 2.5 <= L <= rest"), (0.0, 3.0, "rest < L <= rest + 3"), (3.0, 99, "L > rest + 3")):
        m = (band > lo_) & (band <= hi_)
        per = " ".join(f"{RATIOS[r]} {rms_max(sres[m & (st.ri == r)])[1]:.2f}" for r in range(6))
        print(f"      band {nm:24s}: rms {rms_max(sres[m])[0]:.3f} max {rms_max(sres[m])[1]:.2f} | max per ratio: {per}")
    e54 = rendered_static(REPORT54, C, S, Tfin, depth_fin, pd, mode, off); e198 = rendered_static(REPORT198, C, S, Tfin, depth_fin, pd, mode, off)
    efull = rendered_static(FULL, C, S, Tfin, depth_fin, pd, mode, off)
    fr = np.array([r for r, _, _ in FULL]); fb = np.array([L - Tp[k - 1] - rest_rel for _, k, L in FULL]); knee = (fb > -4.0) & (fb <= 2.0)
    print(f"    RENDERED statics (mirror, exact ripple, PCHIP): 54-item set rms {rms_max(e54)[0]:.3f} max {rms_max(e54)[1]:.2f} | 198-item set rms {rms_max(e198)[0]:.3f} max {rms_max(e198)[1]:.2f}"
          f" | full grid rms {rms_max(efull)[0]:.3f} max {rms_max(efull)[1]:.2f} | knee band (rest - 4 < L <= rest + 2) rms {rms_max(efull[knee])[0]:.3f} max {rms_max(efull[knee])[1]:.2f}")
    print("      full-grid rms/max per ratio:", " ".join(f"{RATIOS[r]} {rms_max(efull[fr == r])[0]:.3f}/{rms_max(efull[fr == r])[1]:.2f}" for r in range(6)))
    sil = [float(pchip_at(C[r], S[r], float(ks))) for r in range(6)]
    # the true silence check: a -50 dBFS sine at threshold 1 (as law_disc_gain_*) and a -80 dBFS sine at threshold 24, every ratio
    m = 6
    gs1, _ = sine_grid(np.full(m, -50.0), np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), Tfin[0] + off, np.arange(6, dtype=np.int64), C, S, float(FS), depth_fin, sv, mode, D, lam, 2.0, 0.5)
    gs24, _ = sine_grid(np.full(m, -80.0), np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), Tfin[23] + off, np.arange(6, dtype=np.int64), C, S, float(FS), depth_fin, sv, mode, D, lam, 2.0, 0.5)
    print(f"    standing GR at silence per ratio: curve at the silence knot {np.round(sil, 4).tolist()} | rendered -50 dBFS at threshold 1: {np.round(-gs1, 4).tolist()} | -80 dBFS at threshold 24: {np.round(-gs24, 4).tolist()}")
    dres = dyn_resid(curves_L, Tp, pd, mode, d0)
    ss = dres[:54] / 3.0; (brms, bmax), (drms, dmax), per_item = burst_split(dres)
    print(f"    steady table rms {rms_max(ss)[0]:.3f} max {rms_max(ss)[1]:.2f} | bursts weighted rms {brms:.3f} max {bmax:.2f} | DUAL (blen, pulses) rms {drms:.3f} max {dmax:.2f}")
    k = 0
    rows = []
    for ai, a in enumerate(ATTACKS):
        rows.append(f"{a:5.1f} ms: " + " ".join(f"{v:+.2f}" for v in ss[k:k + 7])); k += 7
    print("      steady residual rows (recover 0.1/0.25/0.5/0.8/1.2 | level -25/+2):", " || ".join(rows))
    worst = sorted(per_item.items(), key=lambda t: -t[1][0])[:5]
    print("      worst bursts (weighted rms, max):", [(i, round(a, 3), round(b, 2)) for i, (a, b) in worst])
    # onset and release of one slow-attack burst and the 1 ms / 0.5 s tail
    for iid, idx in (("disc_burst_30.0_0.5 s", (500, 502, 505, 510, 520, 540)), ("disc_burst_1.0_0.5 s", (2600, 2800, 3000, 3300, 3600, 4000))):
        e = env_of_item(iid, curves_L, Tp, pd, mode, d0); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        print(f"      {iid} (period, model, ref):", [(j, round(float(e[j]), 2), round(float(ref[j]), 2)) for j in idx if j < n])
    xc = reference_check(mode, pd, C, S, Tfin, depth_fin, off)
    if cpp and mode in (0, 6):
        # the C++ engine on these constants. Mode 6's per-ratio reference is emulated without a code change: every item runs one
        # ratio, so the whole threshold table is shifted by that ratio's offset for its render (exactly what a d_ratio_off field would do)
        def cal_for(r):
            cal2 = cal.copy()
            cal2[MODEL.field("d_thr_db")] = Tfin + off[r]; cal2[MODEL.field("d_curve")] = C.reshape(-1); cal2[MODEL.field("d_tatt")] = ta; cal2[MODEL.field("d_trel")] = tr
            cal2[MODEL.field("d_dual_t2")] = t2; cal2[MODEL.field("d_dual_c2")] = c2; cal2[MODEL.field("d_rel_depth_db")] = depth_fin; cal2[MODEL.field("d_att_sv_db")] = sv
            return cal2
        cals = [cal_for(r) for r in range(6)]
        for nm, items, em in (("54-item", REPORT54, e54), ("198-item", REPORT198, e198)):
            ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t{k}_{L}"], cals[r]) - F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in items])
            print(f"    C++ ENGINE on these constants, {nm} set: rms {rms_max(ec)[0]:.3f} max {rms_max(ec)[1]:.2f}; mirror - C++ max |diff| {np.max(np.abs(em - ec)):.4f} dB")
        ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t12_-60"], cals[r]) - F[f"disc_static_{RATIOS[r]}_t12_-60"] for r in range(6)])
        print(f"    C++ ENGINE standing GR at silence (disc_static_*_t12_-60, model - reference gain, dB): {np.round(-ec, 4).tolist()}")
        for iid in ("disc_burst_1.0_0.5 s", "disc_burst_30.0_0.1 s", "disc_burst_1.0_Dual", "disc_dual_blen_0.1", "law_disc_gain_12"):
            e = render_item(ITEMS[iid], cals[3]); ref = np.asarray(F[iid]); d = np.atleast_1d(np.asarray(e) - ref)
            print(f"      C++ {iid}: rms {rms_max(d)[0]:.3f} max {rms_max(d)[1]:.2f}")
        if os.path.exists(REF_JSON):
            # the soft-ratio tails through the engine (validates the mirror's mode-0 / mode-6 dynamics at 2:1 and 1.2:1)
            R = json.load(open(REF_JSON))
            def cenv(st, ri, **kw):
                it = {"id": "x", "fs": FS, "stim": st, "set": {"discrete_bypass": "In", "discrete_threshold": 16, "discrete_attack": 1.0, "discrete_recover": "0.5 s", "discrete_ratio": RATIOS[ri], **kw}, "feat": {"type": "env"}}
                return GAIN12 - np.asarray(render_item(it, cals[ri]))
            for r, ref in R["R2"]["0.5 s"].items():
                ri = RATIOS.index(r); print(f"      C++ R2 tail {r} (recover 0.5 s): {at(cenv(r2_stim(), ri), 2.5)} | reference {ref}")
            lvl = R["rest"] - 1.0; st, st2 = r4_stims(lvl); ri = 1
            print(f"      C++ R4 2:1 tone at rest - 1: from silence {at(cenv(st, ri), 0.5)} | {R['R4']['2:1']['-1.0']['silence']}; after the burst {at(cenv(st2, ri), 2.5)} | {R['R4']['2:1']['-1.0']['after']}")
            st = r3_stim(R["rest"] - 1.0); e = cenv(st, ri, discrete_attack=30.0); ref = R["R3"]["2:1"][f"{R['rest'] - 1.0:.4f}"]
            print(f"      C++ R3 2:1 onset after a pre-roll at rest - 1 (30 ms): before {float(np.mean(e[1950:2000])):.3f} periods {[round(float(e[2000 + j]), 2) for j in R3_IDX]} | reference {ref['before']} {[round(v, 2) for v in ref['periods']]}")
    sys.stdout.flush()
    return dict(mode=mode, label=label, p=[float(v) for v in p], Tfin=Tfin.tolist(), depth_fin=depth_fin, e54=rms_max(e54), e198=rms_max(e198), efull=rms_max(efull),
                knee=rms_max(efull[knee]), sil=sil, sil_render=[float(v) for v in -gs1], steady=rms_max(ss), bursts=(brms, bmax), dual=(drms, dmax), xcheck=xc)


def baseline_check(Tp, curves_L, cpp=True):
    """the current constants through the mirror and (optionally) through the C++: validates the mirror (curves from constants.json, not refitted)"""
    C = cal[MODEL.field("d_curve")].reshape(6, N).copy(); S = np.stack([pchip_build(C[r]) for r in range(6)])
    T = cal[MODEL.field("d_thr_db")].copy(); depth = float(cal[MODEL.field("d_rel_depth_db")][0]); sv = float(cal[MODEL.field("d_att_sv_db")][0])
    pd = (list(cal[MODEL.field("d_tatt")]), list(cal[MODEL.field("d_trel")]), float(cal[MODEL.field("d_dual_t2")][0]), float(cal[MODEL.field("d_dual_c2")][0]), depth, sv, 0.0, 1.0)
    e54 = rendered_static(REPORT54, C, S, T, depth, pd, 0); e198 = rendered_static(REPORT198, C, S, T, depth, pd, 0); efull = rendered_static(FULL, C, S, T, depth, pd, 0)
    print(f"BASELINE (constants.json as they are), mirror: 54-item statics rms {rms_max(e54)[0]:.3f} max {rms_max(e54)[1]:.2f} | 198-item {rms_max(e198)[0]:.3f} max {rms_max(e198)[1]:.2f} | full grid {rms_max(efull)[0]:.3f} max {rms_max(efull)[1]:.2f}")
    if cpp:
        ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t{k}_{L}"], cal) - F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in REPORT54])
        print(f"  C++ engine on the same constants, 54-item set: rms {rms_max(ec)[0]:.3f} max {rms_max(ec)[1]:.2f}; mirror - C++ max |diff| {np.max(np.abs(e54 - ec)):.4f} dB")
        ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t12_-60"], cal) - F[f"disc_static_{RATIOS[r]}_t12_-60"] for r in range(6)])
        print(f"  C++ engine standing GR at silence (disc_static_*_t12_-60), per ratio: {np.round(-ec, 4).tolist()}")
        for iid in ("disc_burst_1.0_0.5 s", "disc_burst_30.0_0.1 s", "disc_burst_1.0_Dual"):
            e = render_item(ITEMS[iid], cal); ref = np.asarray(F[iid]); d = np.atleast_1d(np.asarray(e) - ref)
            print(f"    C++ {iid}: rms {rms_max(d)[0]:.3f} max {rms_max(d)[1]:.2f}")
    sil = [float(pchip_at(C[r], S[r], -depth)) for r in range(6)]
    print(f"  mirror standing GR at silence per ratio, PCHIP at rest ({-depth:+.3f} dB): {np.round(sil, 3).tolist()}")
    d0 = float(T[15] - Tp[15])
    dres = dyn_resid(curves_L, Tp, pd, 0, d0); ss = dres[:54] / 3.0; (brms, bmax), (drms, dmax), _ = burst_split(dres)
    print(f"  dynamics on the current constants (stage-3 method, lock-in envelopes): steady rms {rms_max(ss)[0]:.3f} max {rms_max(ss)[1]:.2f} | bursts rms {brms:.3f} max {bmax:.2f} | DUAL rms {drms:.3f} max {dmax:.2f}")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ the reference plug-in
REF_JSON = os.path.join(OUT, "disc-knee-second-bleed-reference.json")
TS = (0.25, 0.5, 1.0, 2.0, 4.0)
R3_IDX = (1, 2, 3, 5, 8, 12, 20, 40)


def r2_stim():
    return {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 4.5, "f": 1000.0}


def r3_stim(pre):
    return {"kind": "burst", "pre": float(pre), "level": -10, "pre_s": 2.0, "burst_s": 0.5, "post_s": 0.1, "f": 1000.0}


def r4_stims(lvl):
    return ({"kind": "burst", "pre": -120, "level": lvl, "pre_s": 0.5, "burst_s": 6.0, "post_s": 0.1, "f": 1000.0},
            {"kind": "burst", "pre": lvl, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 6.0, "f": 1000.0})


def at(gr, t_end, ts=TS, f=1000.0):
    return [round(float(np.mean(gr[int((t_end + t) * f) - 50:int((t_end + t) * f)])), 3) for t in ts]


def reference_probe():
    """black-box renders of the reference: what the fade below the soft-ratio knee does in time. Writes REF_JSON."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
    import pa
    p = pa.Ref(); print(f"reference {p.version} {p.sha256[:16]}")
    T16 = float(cal[MODEL.field("d_thr_db")][15]); depth = float(cal[MODEL.field("d_rel_depth_db")][0]); rest = T16 - depth
    print(f"model's threshold 16: T {T16:.2f} dBFS, rest {rest:.2f} dBFS (levels below are dBFS peak of the sine)")
    S = dict(discrete_bypass="In", discrete_threshold=16, discrete_attack=1.0, discrete_recover="0.5 s")
    R = {"rest": rest, "T16": T16, "R1": {}, "R2": {}, "R3": {}, "R4": {}}

    def env(st, **kw):
        y = p.run(protocol.stimulus(st, pa.FS), **pa.both(**{**S, **kw}))
        return GAIN12 - np.asarray(protocol.feature({"feat": {"type": "env"}, "stim": st}, y, pa.FS))

    def gain(st, **kw):
        y = p.run(protocol.stimulus(st, pa.FS), **pa.both(**{**S, **kw}))
        return GAIN12 - float(protocol.feature({"feat": {"type": "gain_db", "last_s": 0.5}, "stim": st}, y, pa.FS))

    print("\n(R1) fine static knee at threshold 16 (2.5 s sines from silence, GR over the last 0.5 s, dB), level relative to the model's rest:")
    lv = np.round(np.arange(-4.0, 2.01, 0.25), 2)
    for r in ("1.2:1", "2:1", "3:1", "4:1"):
        g = [gain({"kind": "sine", "level": float(rest + d), "f": 1000.0, "secs": 2.5}, discrete_ratio=r) for d in lv]
        R["R1"][r] = [[float(d), float(v)] for d, v in zip(lv, g)]
        print(f"   {r:>5}: " + " ".join(f"{d:+.2f}:{v:.3f}" for d, v in zip(lv, g)))

    print("\n(R2) post-burst tail: -50 -> -10 dBFS (2 s) -> -50, threshold 16, attack 1 ms; GR (50-period mean) at 0.25/0.5/1/2/4 s after the burst end, per recover position")
    for rc in ("0.1 s", "0.5 s", "1.2 s"):
        R["R2"][rc] = {}
        for r in ("1.2:1", "2:1", "4:1"):
            gr = env(r2_stim(), discrete_ratio=r, discrete_recover=rc)
            R["R2"][rc][r] = at(gr, 2.5)
            print(f"   recover {rc:>5} {r:>5}: {at(gr, 2.5)}   (in the burst, last 50 periods: {round(float(np.mean(gr[2450:2500])), 3)})")

    print("\n(R3) burst onset after a 2 s pre-roll at several quiet levels: -10 dBFS burst, attack 30 ms, recover 0.5 s; GR per period after the onset")
    for r in ("4:1", "2:1"):
        R["R3"][r] = {}
        for pre in (-120.0, -80.0, -50.0, rest - 3.0, rest - 2.0, rest - 1.0, rest - 0.25):
            gr = env(r3_stim(pre), discrete_ratio=r, discrete_attack=30.0)
            before = round(float(np.mean(gr[1950:2000])), 3); pers = [round(float(gr[2000 + j]), 3) for j in R3_IDX]
            R["R3"][r][f"{pre:.4f}"] = {"before": before, "periods": pers}
            print(f"   {r:>5} pre-roll {pre:8.2f} dBFS (rest {pre - rest:+.2f}): before the onset {before}; periods {R3_IDX}: {pers}")

    print("\n(R4) tone 1 / 2 dB below the model's rest at 2:1 and 1.2:1: from silence (6 s), and after a 2 s burst at -10 dBFS (GR at 0.25/0.5/1/2/4 s after)")
    for r in ("2:1", "1.2:1"):
        R["R4"][r] = {}
        for d in (-1.0, -2.0):
            lvl = float(rest + d); st, st2 = r4_stims(lvl)
            a = at(env(st, discrete_ratio=r), 0.5); b = at(env(st2, discrete_ratio=r), 2.5)
            R["R4"][r][f"{d:.1f}"] = {"silence": a, "after": b}
            print(f"   {r:>5} tone {lvl:.2f} dBFS (rest {d:+.1f}): from silence {a} | after the burst {b}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(REF_JSON, "w"), indent=1)
    print(f"written {REF_JSON}")
    sys.stdout.flush()


def reference_check(mode, pd, C, S, Tfin, depth_fin, off):
    """the R2 / R3 / R4 stimuli through the candidate's node and node-domain curves, against the reference's numbers"""
    if not os.path.exists(REF_JSON):
        print("    (no reference JSON: run --reference once to enable the dynamic cross-check at the soft ratios)"); return None
    R = json.load(open(REF_JSON))
    ta, tr, t2, c2, depth, sv, D, lam = pd
    T16 = Tfin[15]

    def sim(st, r, ai, rci):
        x = protocol.stimulus(st, FS)[0]; ri = RATIOS.index(r)
        v = node(np.abs(x), float(FS), ta[ai], tr[rci], False, t2, c2, T16 + off[ri], depth_fin, sv, mode, D, lam)
        return -env_lockin(x, gr_of_node(v, T16 + off[ri], C[ri], S[ri]), 1000.0, float(FS))

    errs = []
    print("    REFERENCE CROSS-CHECK at the soft ratios (model | reference), GR in dB:")
    for rc, d in R["R2"].items():
        rci = RECOVERS.index(rc); row = []
        for r, ref in d.items():
            m = at(sim(r2_stim(), r, 2, rci), 2.5); errs += [a - b for a, b in zip(m, ref)]
            row.append(f"{r} {m} | {ref}")
        print(f"      R2 tail, recover {rc}: " + "  ;  ".join(row))
    for r, d in R["R3"].items():
        for pre, ref in d.items():
            gr = sim(r3_stim(float(pre)), r, 5, 2)
            before = round(float(np.mean(gr[1950:2000])), 3); pers = [round(float(gr[2000 + j]), 2) for j in R3_IDX]
            errs += [before - ref["before"]] + [a - b for a, b in zip(pers, ref["periods"])]
            print(f"      R3 onset {r} after a pre-roll at {float(pre):8.2f}: before {before} | {ref['before']}; periods {pers} | {[round(v, 2) for v in ref['periods']]}")
    for r, d in R["R4"].items():
        for dd, ref in d.items():
            lvl = R["rest"] + float(dd); st, st2 = r4_stims(lvl)
            a = at(sim(st, r, 2, 2), 0.5); b = at(sim(st2, r, 2, 2), 2.5)
            errs += [x - y for x, y in zip(a, ref["silence"])] + [x - y for x, y in zip(b, ref["after"])]
            print(f"      R4 tone {r} at rest {dd}: from silence {a} | {ref['silence']}; after the burst {b} | {ref['after']}")
    e = np.array(errs)
    print(f"      cross-check rms {rms_max(e)[0]:.3f} max {rms_max(e)[1]:.2f} dB over {len(e)} numbers")
    return rms_max(e)


def r3_extra():
    """the burst onset (30 ms attack) after 2 s pre-roll tones from 3 dB below to 2 dB above the model's 4:1 rest, at 4:1 and 2:1,
    on the reference and on the current model (constants.json, mode 0): where is each one's node after a quiet tone? Nothing written."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
    import pa
    p = pa.Ref(); print(f"reference {p.version} {p.sha256[:16]}")
    T16 = float(cal[MODEL.field("d_thr_db")][15]); depth = float(cal[MODEL.field("d_rel_depth_db")][0]); rest = T16 - depth
    S = dict(discrete_bypass="In", discrete_threshold=16, discrete_attack=30.0, discrete_recover="0.5 s")
    C = cal[MODEL.field("d_curve")].reshape(6, N).copy(); Sl = np.stack([pchip_build(C[r]) for r in range(6)])
    ta = list(cal[MODEL.field("d_tatt")]); tr = list(cal[MODEL.field("d_trel")]); sv = float(cal[MODEL.field("d_att_sv_db")][0])
    print(f"(R3x) -10 dBFS burst after a 2 s pre-roll tone, attack 30 ms, recover 0.5 s, threshold 16 (rest {rest:.2f} dBFS): GR before the onset and at periods {R3_IDX}")
    for r in ("4:1", "2:1"):
        ri = RATIOS.index(r)
        for d in (-3.0, -1.0, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            st = r3_stim(rest + d); x = protocol.stimulus(st, pa.FS)
            y = p.run(x, **pa.both(**S, discrete_ratio=r))
            ref = GAIN12 - np.asarray(protocol.feature({"feat": {"type": "env"}, "stim": st}, y, pa.FS))
            v = node(np.abs(x[0]), float(FS), ta[5], tr[2], False, 1.0, 1.0, T16, depth, sv, 0, 0.0, 1.0)
            gr = gr_of_node(v, T16, C[ri], Sl[ri]); gr[v <= T16 - depth] = 0.0
            mod = -env_lockin(x[0], gr, 1000.0, float(FS))
            vb = float(np.mean(v[int(1.95 * FS):int(2.0 * FS)])) - rest
            print(f"   {r:>4} pre-roll rest {d:+.2f}: REF before {float(np.mean(ref[1950:2000])):.3f} periods {[round(float(ref[2000 + j]), 2) for j in R3_IDX]}"
                  f" | MODEL before {float(np.mean(mod[1950:2000])):.3f} periods {[round(float(mod[2000 + j]), 2) for j in R3_IDX]} (model node before the onset: rest {vb:+.2f})")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    if "--r3x" in args:
        r3_extra(); return
    if "--reference" in args or "--reference-only" in args:
        reference_probe()
        if "--reference-only" in args: return
    modes = [int(m) for m in args[args.index("--modes") + 1].split(",")] if "--modes" in args else [0, 4, 5, 7, 1, 2, 3]
    nfev = int(args[args.index("--nfev") + 1]) if "--nfev" in args else 40
    w_static = float(args[args.index("--wstatic") + 1]) if "--wstatic" in args else 1.0
    fix = [float(t) for t in args[args.index("--fix") + 1].split(",")] if "--fix" in args else None
    off_given = [float(t) for t in args[args.index("--off") + 1].split(",")] if "--off" in args else None
    print(f"GAIN12 {GAIN12:.4f}; reference GR at -50 dBFS, threshold 1, gain 12: {GAIN12 - F['law_disc_gain_12']:+.4f} dB; curve grid N {N} X0 {X0} DX {DX}")
    Tp, curves_L = S3.fit_static()
    st = Static(Tp)
    baseline_check(Tp, curves_L, cpp="--no-cpp" not in args)
    results = {}
    tag = "-".join(str(m) for m in modes) + (f"-fix{fix[0]:g}-{fix[1]:g}" if fix else "") + ("-scan" if "--scan" in args else "")
    for mode in modes:
        print(f"\nMODE {mode}: {NAMES[mode]}")
        p0, lo, hi, xs = start_and_bounds(mode)
        tail = np.array([])
        if fix is not None and mode in (1, 2, 3, 4, 5, 7):
            p0[16] = fix[0]; p0[17] = np.log10(fix[1]); tail = p0[16:18].copy()
            p0, lo, hi, xs = p0[:16], lo[:16], hi[:16], xs[:16]
            print(f"  leak held at D {fix[0]} {'tg' if mode in (2, 5) else 'lam'} {fix[1]}")
        elif mode in (3, 7):
            tail = p0[16:17].copy(); p0, lo, hi, xs = np.concatenate([p0[:16], p0[17:18]]), lo[:16] + lo[17:18], hi[:16] + hi[17:18], xs[:16] + xs[17:18]
        if off_given is not None and mode == 6:
            p0[16:19] = off_given; tail = p0[16:19].copy()
            p0, lo, hi, xs = p0[:16], lo[:16], hi[:16], xs[:16]
            print(f"  per-ratio reference offsets held at {off_given} (1.2:1, 2:1, 3:1)")
        if mode in (3, 7) and fix is None:
            full = lambda p: np.concatenate([p[:16], tail, p[16:17]])
        else:
            full = lambda p: np.concatenate([p, tail])
        rep = {}
        t0 = time.time()
        rep["start"] = report(mode, full(p0), Tp, curves_L, st, "start (current constants, curves refitted in the node domain)")
        print(f"    (one evaluation + report: {time.time() - t0:.1f} s)")
        if "--scan" in args and mode in (1, 2, 4, 5):
            print("  scan of the leak (detector at the current constants): D x " + ("tg (s)" if mode in (2, 5) else "lam (tg = lam x 0.33 s)"))
            grid = [(D, g) for D in (1.5, 2.0, 2.5, 3.0, 4.0) for g in ((0.1, 0.3, 1.0, 3.0, 10.0) if mode in (2, 5) else (0.3, 1.0, 3.0, 10.0, 30.0))]
            best = None
            for D, g in grid:
                pp = p0.copy(); pp[16] = D; pp[17] = np.log10(g)
                pd = unpack(full(pp), mode); sres, _, _, d0, _, _ = st.fit(pd, mode)
                dres = dyn_resid(curves_L, Tp, pd, mode, d0); ss = dres[:54] / 3.0; (b, bm), (du, dm), _ = burst_split(dres)
                cost = float(np.sum(np.concatenate([dres, w_static * sres]) ** 2) / 2)
                print(f"    D {D:4.1f} {'tg' if mode in (2, 5) else 'lam'} {g:5.1f}: static node-domain rms {rms_max(sres)[0]:.4f} | steady {rms_max(ss)[0]:.3f}/{rms_max(ss)[1]:.2f} bursts {b:.3f}/{bm:.2f} DUAL {du:.3f}/{dm:.2f} | cost {cost:.2f}")
                if best is None or cost < best[0]: best = (cost, D, g)
            print(f"    scan best: D {best[1]} {'tg' if mode in (2, 5) else 'lam'} {best[2]} (cost {best[0]:.2f}); the joint fit starts there")
            p0[16] = best[1]; p0[17] = np.log10(best[2])
            rep["scan"] = report(mode, full(p0), Tp, curves_L, st, "scan best (detector at the current constants)")
        def resid(p):
            pd = unpack(full(p), mode)
            sres, _, _, d0, _, _ = st.fit(pd, mode, off_of(full(p), mode))
            return np.concatenate([dyn_resid(curves_L, Tp, pd, mode, d0), w_static * sres])
        t0 = time.time()
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
        print(f"  fit: nfev {r.nfev} cost {r.cost:.3f} ({time.time() - t0:.0f} s)")
        rep["fitted"] = report(mode, full(r.x), Tp, curves_L, st, "fitted", cpp=("--cpp" in args))
        print("  p:", np.round(full(r.x), 6).tolist())
        results[mode] = rep
        os.makedirs(OUT, exist_ok=True)
        json.dump(results, open(os.path.join(OUT, f"disc-knee-second-bleed-{tag}.json"), "w"), indent=1)
        sys.stdout.flush()
    print("\nSUMMARY (statics: rendered rms/max on the 54-item report set, the 198-item set, the full grid, the knee band; silence: curve at the silence knot per ratio)")
    for mode, rep in results.items():
        for key, R in rep.items():
            print(f"  mode {mode} {key:7s}: statics54 {R['e54'][0]:.3f}/{R['e54'][1]:.2f} 198 {R['e198'][0]:.3f}/{R['e198'][1]:.2f} full {R['efull'][0]:.3f}/{R['efull'][1]:.2f} knee {R['knee'][0]:.3f}/{R['knee'][1]:.2f}"
                  f" | silence {np.round(R['sil'], 3).tolist()} | steady {R['steady'][0]:.3f}/{R['steady'][1]:.2f} bursts {R['bursts'][0]:.3f}/{R['bursts'][1]:.2f} DUAL {R['dual'][0]:.3f}/{R['dual'][1]:.2f}"
                  + (f" | soft-ratio cross-check {R['xcheck'][0]:.3f}/{R['xcheck'][1]:.2f}" if R.get('xcheck') else ""))


if __name__ == "__main__":
    main()
