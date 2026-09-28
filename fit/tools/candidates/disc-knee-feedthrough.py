# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage, soft-ratio knee: direction (a), a feed-through around the storage node. RESULT: REFUTED (see the end).

HYPOTHESIS. The storage node v cannot fall below its rest point (T - depth): on a steady sine below rest only the bleed acts, so
every level below rest maps onto one node value and the gain computer reads one gain reduction, C_r(-depth). The reference's
1.2:1, 2:1 and 3:1 curves keep falling smoothly to zero over the last 1-2 dB below that level (0.48 / 1.08 / 0.58 dB at the rest
level; 4:1 and steeper are already at zero there). The candidate: the gain computer does not see the node alone but the node plus
a small term from a second, UNCLAMPED envelope w of the rectified level (a branching one-pole in dB, attack taw, release trw toward
the log level, no bleed to the reference), which does follow the level down. The node keeps its always-on bleed to the reference,
so the release still targets the threshold reference and the steady table, bursts and DUAL items are produced by the node as now.

VARIANTS (the gain computer's argument z, GR = C_r(z - T) with C_r resampled onto the z domain so the static family stays exact):
  V0  z = v                                                   the current model and constants (the control; reproduces stage 3)
  V1  z = (1 - alpha) v + alpha w                             a plain blend: the feed-through acts at every level
  V2  z = v + alpha * min(w - (rest - woff), 0)               the feed-through acts only below the reference (a diode from w
                                                              into the gain computer's node that conducts once w is below rest)
  V4  GR = C_r(v - T) * S(w), S = clip((w - (rest - woff - wd)) / wd, 0, 1)   a multiplicative gate on the level: one common
                                                              fade shape scaled by each ratio's value at rest (unpinned curves)
  V5  V2 with a fast follower (trw fixed 1 ms): the direct feed-through of the rectified level, ripple and all
  V6  z = v with a hard zero at rest (GR = 0 whenever v <= rest): the control the task describes, silence exact by construction.
  (max(v, w): analytically a no-op for the fade, since max(v, w) >= v = rest below rest, so GR >= C(-depth) there; not run.)

FIT. Each variant is fitted in two passes on the full residual: (A) the mechanism constants alone with the detector at the current
constants.json values; (B) jointly with the sixteen detector constants of stage 3b (six attacks, six recovers, DUAL t2 / c2, depth,
log10 Sv). Residual = 3 x steady table (disc_ar_*, disc_al_*, disc_af_*: 54 cells, full-chain simulation) + burst envelopes
(disc_burst_*, disc_bdepth_*, disc_dual_blen_*, disc_dual_pulses, weighted as stage 3) + the whole static grid (six ratios x 24
thresholds x 25 levels = 3600 items, weight 1, evaluated through shift invariance from one fine x = L - T' grid per evaluation).
The curves are the stage-3a shift family (fit_static, imported from fit/stages/stage3_discrete.py), resampled onto the z domain
inside every residual evaluation, read with the C++ Fritsch-Carlson monotone cubic on the 1 dB grid. The mirror was checked
against the C++ engine on the current constants: stage-3 report set 0.217 / 0.73 dB (C++ 0.216 / 0.73), standing gain reduction in
silence per ratio 0.326 / 0.732 / 0.451 / 0.044 / 0.001 / 0.002 dB (C++ 0.326 / 0.731 / 0.451 / 0.044 / 0.002 / 0.002).

RESULTS (2026-09-27; statics = full-chain rms / max in dB; bursts and DUAL weighted as stage 3):
  - The current constants do NOT give exact silence at the soft ratios: the C++ reads the PCHIP at v - T = -depth = -0.40, between
    the pinned grid point at -1 and the unpinned one at 0, and stands at +0.33 / +0.73 / +0.45 dB at 1.2:1 / 2:1 / 3:1 (4:1
    +0.04). The whole 0.216 / 0.73 static figure is that standing value (the stage-3 report set, thresholds 4 / 12 / 20 at levels
    -30 / -15 / 0, contains no point in the fade band at all: with a hard zero at rest it reads 0.018 / 0.04 while the knee band
    x in [-4, 2] reads 0.226 / 1.21). Measure the knee on the knee band or the whole grid, not on that set.
  - V2 (feed-through below rest) reproduces the statics to the shift family's floor with the dynamics untouched: pass A alpha
    0.63, woff 0.24, taw -> 0.03 ms, trw -> 9.6 s (its 10 s bound): whole grid 0.024 / 0.18, knee band 0.035 / 0.16, report set
    0.019 / 0.04, silence 0.000 at every ratio, steady 0.084 / 0.23, bursts 0.011 / 0.98, DUAL 0.018 / 0.09 (all three identical
    to the current constants). Pass B (joint) keeps alpha 0.61 and moves the detector to depth 0.08, Sv 24, attack 1.39 ms:
    steady 0.076 / 0.26, bursts 0.010 / 0.92, DUAL 0.010 / 0.07; the refit control (V6 with the static grid weighted 0, the
    stage-3b residual alone) lands on the same detector (depth 0.077, Sv 24.5, steady 0.076, bursts 0.010, DUAL 0.010), so the
    dynamic gains are the refit's, not the mechanism's.
  - The follower must be slower than every release in the protocol: with trw held at 1 s / 3 s the DUAL items go to 0.168 /
    0.117 rms at the scan optimum and the fit then removes the mechanism (alpha 0.012 / 0.016, statics back to 0.12 / 0.10).
    V1 (plain blend) is removed by the fit the same way (alpha 0.019, statics 0.133 / 0.65; joint 0.056 / 0.50 at the price of
    depth 0.63 / Sv 25), V5 (1 ms follower) too (alpha 0.006, statics 0.116 / 0.53). V4 (gate) works nearly as well as V2 with
    the same slow follower (grid 0.028 / 0.30, knee 0.057 / 0.30: one common fade shape cannot do 3:1 and 1.2:1 at once).
  - THE REFERENCE REFUTES THE SLOW FOLLOWER (python3 ... --reference-tail, black-box renders of the reference plug-in): after a
    2 s burst at -10 dBFS (threshold 16, 1 ms, 0.5 s) into -50 dBFS its gain reduction at 2:1 is 5.27 / 1.00 / 0.082 / 0.050 /
    0.001 dB at 0.5 / 1 / 2 / 4 / 8 s, on the same trajectory as 4:1 (5.67 / 1.07 / 0.084 / 0.050 / 0.001) and 1.2:1 (3.45 /
    0.67 / 0.070 / 0.050 / 0.001); V2 predicts 0.94-1.25 dB standing until the follower crosses rest at about 10 s. A tone 1.5 dB
    below the rest level (-39.5 dBFS) settles to its from-silence value within 2 s of such a burst (2:1: 0.266 dB; 1.2:1: 0.137),
    where V2 predicts 0.94 / 0.42 dB for 10 s. So the reference's fade below the knee follows the level down at the release rate
    with no memory, and every (a)-variant that follows the level that fast is removed by the 4:1 DUAL tails. Within (a) that is
    a contradiction: below rest the argument must be <= v (max(v, w) gives no fade), a w that falls faster than v cuts the 4:1
    DUAL tail, a w that falls slower than v cannot produce the measured tail. The knee needs the gain computer's argument to keep
    falling below the 4:1 rest point at the release rate (1.6 dB below it by 2 s at 2:1), which is direction (b) or (c), not (a).

usage: cd <repo> && python3 -u fit/tools/candidates/disc-knee-feedthrough.py [--variants 0,1,2,4,5,6] [--nfev-a 40] [--nfev-b 40]
       [--skip-joint] [--scan] [--trw SECONDS] [--static-w W] [--eval JSON:variant:key] [--reference-tail]
  --scan   V2 / V4: a coarse grid over (alpha, woff) / (wd, woff) with the follower at 0.1 ms / 10 s before pass A (the resampling
           onto the 1 dB grid makes the static residual jagged in these constants and least_squares alone stalls near its start)
  --trw    hold the follower's release at this value (seconds) instead of fitting it: the sensitivity of the bursts to it
  --static-w W   weight of the static grid in the residual (0 = the stage-3b detector residual alone: the refit control)
  --eval JSON:variant:key   re-report a saved parameter vector (the runs write /tmp/disc-knee-feedthrough-<tag>.json)
  --reference-tail   render the discriminating stimuli through the reference plug-in (needs the licensed plug-in and Pedalboard;
           fit/measure/pa.py) and print its post-burst tail at 1.2:1 / 2:1 / 3:1 / 4:1 and the below-rest tone from silence and
           after a burst; nothing is written
  (numba caches need a real file: never pipe this through stdin)"""
import os, sys, time, json, numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, protocol, FS  # noqa: E402
import stage3_discrete as s3  # noqa: E402  (fit_static, curve_at; nothing here writes constants)

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
N = s3.N; XG = s3.XG; DX = s3.DX; X0 = s3.X0
FLOOR = -100.0
cal = load_cal()
GAIN = cal[MODEL.field("d_gain_db")]; XGN = float(cal[MODEL.fields["x_gain_db"][0]]); GAIN12 = float(GAIN[11] + XGN)
s3.GAIN12 = GAIN12
T_CUR = cal[MODEL.field("d_thr_db")].copy()
LEVELS = list(range(-60, 13, 3))
VNAMES = {0: "V0 z = v (control, pinned curves)", 1: "V1 plain blend z = (1-a) v + a w", 2: "V2 below-rest feed-through z = v + a min(w - (rest - woff), 0)",
          4: "V4 multiplicative gate GR = C(v - T) S(w)", 5: "V5 = V2 with a 1 ms follower (direct feed-through of the level)",
          6: "V6 z = v with a hard zero at rest (properly pinned control)"}


# ------------------------------------------------------------------------------------------------ the curve (as Discrete.hpp)
@njit(cache=True)
def pchip_build(y, m):
    """Fritsch-Carlson slopes on the uniform grid, same arithmetic as Curve::build"""
    d = np.empty(N - 1)
    for i in range(N - 1): d[i] = (y[i + 1] - y[i]) / DX
    m[0] = d[0]; m[N - 1] = d[N - 2]
    for i in range(1, N - 1): m[i] = 0.0 if d[i - 1] * d[i] <= 0.0 else 0.5 * (d[i - 1] + d[i])
    for i in range(N - 1):
        if d[i] == 0.0:
            m[i] = 0.0; m[i + 1] = 0.0; continue
        a = m[i] / d[i]; b = m[i + 1] / d[i]; s = a * a + b * b
        if s > 9.0:
            t = 3.0 / np.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]


@njit(cache=True)
def pchip_at(y, m, x):
    u = (x - X0) / DX
    if u <= 0.0: return y[0]
    if u >= N - 1.0: return y[N - 1] + m[N - 1] * (x - (X0 + DX * (N - 1)))
    i = int(u); t = u - i; t2 = t * t; t3 = t2 * t
    return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * DX * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * DX * m[i + 1]


# ------------------------------------------------------------------------------------------------ the chain mirror
@njit(cache=True)
def chain(a, fs, ta, tr, dual, t2, c2, Tk, depth, sv, mode, alpha, woff, wd, taw, trw, cy, cm):
    """a: |sidechain|. Node v exactly as DiscreteStage::process (always-on bleed toward Tk - depth, level-scaled attack diode,
    DUAL second node); follower w (branching one-pole in dB, unclamped); z per variant; GR = C(z - Tk) [* S]. Returns
    (gr, z, s) per sample."""
    n = a.shape[0]
    gr = np.empty(n); zz = np.empty(n); ss = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    k2 = 1.0 - np.exp(-1.0 / (t2 * fs)) if dual else 0.0
    kAw = 1.0 - np.exp(-1.0 / (taw * fs)); kRw = 1.0 - np.exp(-1.0 / (trw * fs))
    fl = 10.0 ** (FLOOR / 20.0)
    rest = Tk - depth
    v = rest; w2 = rest; w = FLOOR
    for i in range(n):
        e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
        dv = (rest - v) * kR
        if e > v:
            xe = e - rest
            f = 1.0 + xe / sv if xe > 0.0 else 1.0
            dv += (e - v) * (1.0 - np.exp(-rA * f))
        v += dv
        if dual:
            flow = (v - w2) * k2
            v -= flow
            w2 += flow / c2
        # the unclamped follower
        if e > w: w += (e - w) * kAw
        else: w += (e - w) * kRw
        s = 1.0
        if mode == 1:
            z = (1.0 - alpha) * v + alpha * w
        elif mode == 2 or mode == 5:
            d = w - (rest - woff)
            z = v + alpha * d if d < 0.0 else v
        elif mode == 4:
            z = v
            s = (w - (rest - woff - wd)) / wd
            if s < 0.0: s = 0.0
            if s > 1.0: s = 1.0
        else:
            z = v
        g = pchip_at(cy, cm, z - Tk) * s
        if mode == 6 and v <= rest + 1e-9: g = 0.0
        if g < 0.0: g = 0.0
        gr[i] = g; zz[i] = z; ss[i] = s
    return gr, zz, ss


def sine(level, secs, f=1000.0, fs=FS):
    t = np.arange(int(secs * fs)) / fs
    return np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))


# ------------------------------------------------------------------------------------------------ parameters
class P:
    """the parameter vector: 16 detector constants (as stage 3b) + the variant's mechanism constants"""
    def __init__(self, variant, trw_fixed=None):
        self.variant = variant; self.trw_fixed = trw_fixed
        d = cal
        self.det0 = list(d[MODEL.field("d_tatt")]) + list(d[MODEL.field("d_trel")]) + [float(d[MODEL.field("d_dual_t2")][0]), float(d[MODEL.field("d_dual_c2")][0]),
                                                                                       float(d[MODEL.field("d_rel_depth_db")][0]), float(np.log10(d[MODEL.field("d_att_sv_db")][0]))]
        self.det_lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, -20.0, 0.3]; self.det_hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 40.0, 3.0]
        self.det_xs = [1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 1.0, 0.3]
        # mechanism: (name, start, lo, hi, x_scale)
        if variant in (0, 6): mech = []
        elif variant == 1: mech = [("alpha", 0.3, 0.0, 1.0, 0.1), ("log_taw", -3.0, -4.5, -1.0, 0.3), ("log_trw", -1.0, -3.0, 1.0, 0.3)]
        elif variant == 2: mech = [("alpha", 0.6, 0.0, 3.0, 0.1), ("woff", 0.0, -6.0, 6.0, 0.5), ("log_taw", -3.0, -4.5, -1.0, 0.3), ("log_trw", -1.0, -3.0, 1.0, 0.3)]
        elif variant == 4: mech = [("woff", 0.0, -6.0, 6.0, 0.5), ("wd", 2.0, 0.2, 8.0, 0.5), ("log_taw", -3.0, -4.5, -1.0, 0.3), ("log_trw", -1.0, -3.0, 1.0, 0.3)]
        elif variant == 5: mech = [("alpha", 0.6, 0.0, 3.0, 0.1), ("woff", 0.0, -6.0, 6.0, 0.5), ("log_taw", -3.5, -4.5, -2.0, 0.3)]
        if trw_fixed is not None: mech = [m for m in mech if m[0] != "log_trw"]
        self.mech = mech
        self.names = ["ta0.1", "ta0.5", "ta1", "ta5", "ta10", "ta30", "tr0.1", "tr0.25", "tr0.5", "tr0.8", "tr1.2", "trDual", "t2", "c2", "depth", "log_sv"] + [m[0] for m in mech]

    def full0(self): return np.array(self.det0 + [m[1] for m in self.mech])
    def unpack(self, p):
        p = np.asarray(p, dtype=float)
        det = dict(ta=p[0:6], tr=p[6:12], t2=float(p[12]), c2=float(p[13]), depth=float(p[14]), sv=float(10.0 ** p[15]))
        mech = dict(alpha=0.0, woff=0.0, wd=1.0, taw=1e-3, trw=0.1)
        for i, m in enumerate(self.mech):
            val = float(p[16 + i])
            if m[0].startswith("log_"): mech[m[0][4:]] = 10.0 ** val
            else: mech[m[0]] = val
        if self.variant == 5: mech["trw"] = 1e-3
        if self.trw_fixed is not None: mech["trw"] = self.trw_fixed
        return det, mech


# ------------------------------------------------------------------------------------------------ the static family on the z domain
XS = np.concatenate([np.arange(-10.0, -4.0, 0.5), np.arange(-4.0, 4.0, 0.1), np.arange(4.0, 12.0, 0.5), np.arange(12.0, 76.0, 1.0)])
ZERO_Y = np.zeros(N); ZERO_M = np.zeros(N)


def zbar_grid(det, mech, mode, xs=XS, secs=1.2, tail=0.2):
    """mean z (and mean gate S) on a steady 1 kHz sine at x = L - T (the level relative to the threshold), capture setting (1 ms, 0.5 s)"""
    Tk = T_CUR[15]; nt = int(tail * FS)
    zb = np.empty(len(xs)); sb = np.empty(len(xs))
    for i, x in enumerate(xs):
        a = sine(x + Tk, secs)
        _, z, s = chain(a, float(FS), det["ta"][2], det["tr"][2], False, det["t2"], det["c2"], Tk, det["depth"], det["sv"], mode,
                        mech["alpha"], mech["woff"], mech["wd"], mech["taw"], mech["trw"], ZERO_Y, ZERO_M)
        zb[i] = z[-nt:].mean(); sb[i] = s[-nt:].mean()
    return zb - Tk, sb


def resample_curves(cref, Tp, zb, sb, mode, depth, d_cap):
    """C_z on the 1 dB grid such that C_z(zbar(x)) [* sbar(x)] = C_ref(x), x = L - T'. zb is z - T on XS; T' = T - d_cap. The static
    family's argument is x = L - T' = (L - T) + d_cap, so the level x_s = XS + d_cap in the family's frame."""
    xf = XS + d_cap
    out = np.zeros((6, N)); ms = np.zeros((6, N))
    if mode == 4:
        keep = np.concatenate([[True], np.diff(zb) > 1e-6]) & (sb > 0.02)
    else:
        keep = np.concatenate([[True], np.diff(zb) > 1e-6])
    zk = zb[keep]; xk = xf[keep]; sk = sb[keep]
    for r in range(6):
        cr = s3.curve_at(cref[r], xk)
        if mode == 4: cr = cr / sk
        c = np.interp(XG, zk, cr, left=np.nan, right=np.nan)
        lo = XG < zk[0]; hi = XG > zk[-1]
        if mode == 4:
            c[lo] = cr[0]   # never read (v >= rest): held for continuity
        else:
            c[lo] = 0.0
        if np.any(hi):
            slope = (cr[-1] - cr[-6]) / max(1e-6, zk[-1] - zk[-6])
            c[hi] = cr[-1] + slope * (XG[hi] - zk[-1])
        c = np.maximum.accumulate(np.maximum(c, 0.0))
        out[r] = c; pchip_build(c, ms[r])
    return out, ms


# ------------------------------------------------------------------------------------------------ residual pieces
BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS[:5]] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
DUAL_B = [f"disc_burst_{a}_Dual" for a in ATTACKS]
DUAL5 = [f"disc_dual_blen_{b}" for b in (0.1, 0.5, 2.0, 8.0)] + ["disc_dual_pulses"]
ALL_BURSTS = BURSTS + DUAL_B + DUAL5
STATIC_ITEMS = [(r, k, L) for r in range(6) for k in range(1, 25) for L in LEVELS]
STATIC_REF = np.array([GAIN12 - F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in STATIC_ITEMS])
STATIC_R = np.array([r for r, k, L in STATIC_ITEMS]); STATIC_K = np.array([k - 1 for r, k, L in STATIC_ITEMS]); STATIC_L = np.array([float(L) for r, k, L in STATIC_ITEMS])
REPORT_MASK = np.array([(k in (4, 12, 20) and L in (-30, -15, 0)) for r, k, L in STATIC_ITEMS])      # the stage-3 report set (54)
BAND_MASK = np.array([(k in (4, 12, 20) and -30 <= L <= 0) for r, k, L in STATIC_ITEMS])            # every grid level in that band (198)


def steady_gr(det, mech, mode, cy, cm, ai, rci, Tk, level=-10.0, f=1000.0, secs=3.0, dual=False):
    a = sine(level, secs, f=f)
    gr, _, _ = chain(a, float(FS), det["ta"][ai], det["tr"][rci], dual, det["t2"], det["c2"], Tk, det["depth"], det["sv"], mode,
                     mech["alpha"], mech["woff"], mech["wd"], mech["taw"], mech["trw"], cy, cm)
    per = int(round(FS / f)); seg = gr[-per * 50:]
    return -20 * np.log10(np.mean(10 ** (-seg / 20.0)))


def env_item(iid, det, mech, mode, curves, ms, T):
    it = ITEMS[iid]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    x = np.abs(protocol.stimulus(st, fs)[0])
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
    r = RATIOS.index(s.get("discrete_ratio", "4:1")); Tk = T[int(s["discrete_threshold"]) - 1]
    gr, _, _ = chain(x, float(fs), det["ta"][ai], det["tr"][rci], dual, det["t2"], det["c2"], Tk, det["depth"], det["sv"], mode,
                     mech["alpha"], mech["woff"], mech["wd"], mech["taw"], mech["trw"], curves[r], ms[r])
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def burst_res(iid, e):
    ref = np.asarray(F[iid]); n = min(len(e), len(ref))
    w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
    return w * (e[:n] - ref[:n])


class Harness:
    def __init__(self, variant, cref, Tp, trw_fixed=None, static_w=1.0):
        self.variant = variant; self.mode = variant; self.cref = cref; self.Tp = Tp; self.P = P(variant, trw_fixed)
        self.nev = 0; self.static_w = static_w

    def build(self, p):
        """everything a residual needs: the z-domain curves for these constants, zbar on the x grid"""
        det, mech = self.P.unpack(p)
        zb, sb = zbar_grid(det, mech, self.mode)
        # d_cap: the node offset at the capture setting, as stage 3 defines it (level -20 dBFS at T16); T' = T - d_cap
        d_cap = float(np.interp(-20.0 - T_CUR[15], XS, zb) + T_CUR[15] - (-20.0))
        curves, ms = resample_curves(self.cref, self.Tp, zb, sb, self.mode, det["depth"], d_cap)
        self.T = self.Tp + d_cap   # the thresholds in the z frame for these constants (stage 3: T = T' + d0)
        return det, mech, zb, sb, d_cap, curves, ms

    def static_pred_cheap(self, zb, sb, d_cap, curves, ms):
        """static grid through shift invariance: x_item = L - T'_k = (L - T_k) + d_cap; z from the zbar grid, then the curve"""
        xr = STATIC_L - self.Tp[STATIC_K] - d_cap   # = L - T_k in the node frame
        z = np.interp(xr, XS, zb); s = np.interp(xr, XS, sb)
        out = np.empty(len(z))
        for r in range(6):
            m = STATIC_R == r
            out[m] = np.array([pchip_at(curves[r], ms[r], zi) for zi in z[m]]) * s[m]
        return np.maximum(out, 0.0)

    def static_pred_full(self, det, mech, d_cap, curves, ms):
        """the same grid with the full chain (ripple included): steady GR on a fine x grid per ratio, then interpolated"""
        Tk = T_CUR[15]; per = 48; xs = XS
        out = np.empty(len(STATIC_REF))
        xr = STATIC_L - self.Tp[STATIC_K] - d_cap
        for r in range(6):
            g = np.empty(len(xs))
            for i, x in enumerate(xs):
                a = sine(x + Tk, 1.5)
                gr, _, _ = chain(a, float(FS), det["ta"][2], det["tr"][2], False, det["t2"], det["c2"], Tk, det["depth"], det["sv"], self.mode,
                                 mech["alpha"], mech["woff"], mech["wd"], mech["taw"], mech["trw"], curves[r], ms[r])
                seg = gr[-per * 100:]
                g[i] = -20 * np.log10(np.mean(10 ** (-seg / 20.0)))
            m = STATIC_R == r
            out[m] = np.interp(xr[m], xs, g)
        return out

    def residual(self, p):
        static_w = self.static_w
        self.nev += 1
        det, mech, zb, sb, d_cap, curves, ms = self.build(p)
        out = []
        for ai, a in enumerate(ATTACKS):
            for rci, rc in enumerate(RECOVERS[:5]):
                out.append(3.0 * (GAIN12 - steady_gr(det, mech, self.mode, curves[3], ms[3], ai, rci, self.T[15]) - F[f"disc_ar_{a}_{rc}"]))
            for lvl in (-25, 2):
                out.append(3.0 * (GAIN12 - steady_gr(det, mech, self.mode, curves[3], ms[3], ai, 2, self.T[15], level=float(lvl)) - F[f"disc_al_{a}_{lvl}"]))
        for a in (0.1, 1.0, 30.0):
            for f in (100.0, 5000.0):
                out.append(3.0 * (GAIN12 - steady_gr(det, mech, self.mode, curves[3], ms[3], ATTACKS.index(a), 2, self.T[15], f=f) - F[f"disc_af_{a}_f{int(f)}"]))
        for iid in ALL_BURSTS:
            out.append(burst_res(iid, env_item(iid, det, mech, self.mode, curves, ms, self.T)))
        st = self.static_pred_cheap(zb, sb, d_cap, curves, ms) - STATIC_REF
        out.append(static_w * st)
        return np.concatenate([np.atleast_1d(o) for o in out])

    def report(self, p, label):
        det, mech, zb, sb, d_cap, curves, ms = self.build(p)
        print(f"  [{label}] constants: attack ms {np.round(det['ta'] * 1e3, 3).tolist()} recover s {np.round(det['tr'], 4).tolist()} t2 {det['t2']:.4f} c2 {det['c2']:.2f} "
              f"depth {det['depth']:.3f} Sv {det['sv']:.2f} | mech {({k: round(v, 4) for k, v in mech.items()})} | d_cap {d_cap:+.3f}")
        # steady
        ss = []
        for ai, a in enumerate(ATTACKS):
            for rci, rc in enumerate(RECOVERS[:5]):
                ss.append(GAIN12 - steady_gr(det, mech, self.mode, curves[3], ms[3], ai, rci, self.T[15]) - F[f"disc_ar_{a}_{rc}"])
            for lvl in (-25, 2):
                ss.append(GAIN12 - steady_gr(det, mech, self.mode, curves[3], ms[3], ai, 2, self.T[15], level=float(lvl)) - F[f"disc_al_{a}_{lvl}"])
        for a in (0.1, 1.0, 30.0):
            for f in (100.0, 5000.0):
                ss.append(GAIN12 - steady_gr(det, mech, self.mode, curves[3], ms[3], ATTACKS.index(a), 2, self.T[15], f=f) - F[f"disc_af_{a}_f{int(f)}"])
        ss = np.array(ss)
        bb = np.concatenate([burst_res(i, env_item(i, det, mech, self.mode, curves, ms, self.T)) for i in BURSTS])
        db = np.concatenate([burst_res(i, env_item(i, det, mech, self.mode, curves, ms, self.T)) for i in DUAL_B])
        d5 = np.concatenate([burst_res(i, env_item(i, det, mech, self.mode, curves, ms, self.T)) for i in DUAL5])
        stc = self.static_pred_cheap(zb, sb, d_cap, curves, ms) - STATIC_REF
        stf = self.static_pred_full(det, mech, d_cap, curves, ms) - STATIC_REF
        knee = np.abs(STATIC_L - self.Tp[STATIC_K] + 1.0) <= 3.0   # x = L - T' in [-4, 2]
        rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
        mx = lambda e: float(np.max(np.abs(e)))
        # silence
        sil = np.zeros(int(1.0 * FS))
        stand = []
        for r in range(6):
            gr, _, _ = chain(sil, float(FS), det["ta"][2], det["tr"][2], False, det["t2"], det["c2"], T_CUR[15], det["depth"], det["sv"], self.mode,
                             mech["alpha"], mech["woff"], mech["wd"], mech["taw"], mech["trw"], curves[r], ms[r])
            stand.append(float(gr[-1]))
        crest = [float(pchip_at(curves[r], ms[r], -det["depth"])) for r in range(6)]
        print(f"  [{label}] STATICS stage-3 report set (54, full chain): rms {rms(stf[REPORT_MASK]):.3f} max {mx(stf[REPORT_MASK]):.2f} | thr 4/12/20 x levels -30..0 step 3 (198): rms {rms(stf[BAND_MASK]):.3f} max {mx(stf[BAND_MASK]):.2f} | "
              f"whole grid (3600): rms {rms(stf):.3f} max {mx(stf):.2f} | knee band x in [-4,2] ({int(knee.sum())}): rms {rms(stf[knee]):.3f} max {mx(stf[knee]):.2f} | cheap (no ripple) 54: rms {rms(stc[REPORT_MASK]):.3f}")
        per_r = [f"{RATIOS[r]} {rms(stf[REPORT_MASK & (STATIC_R == r)]):.3f}/{mx(stf[REPORT_MASK & (STATIC_R == r)]):.2f}" for r in range(6)]
        print(f"  [{label}] statics report set per ratio (rms/max): {'  '.join(per_r)}")
        print(f"  [{label}] standing GR at silence per ratio: {np.round(stand, 4).tolist()} | curve value at the rest point C(-depth): {np.round(crest, 3).tolist()}")
        print(f"  [{label}] STEADY (54 cells): rms {rms(ss):.3f} max {mx(ss):.2f} | BURSTS (33, weighted): rms {rms(bb):.3f} max {mx(bb):.2f} | "
              f"DUAL (blen x4 + pulses, weighted): rms {rms(d5):.3f} max {mx(d5):.2f} | disc_burst_*_Dual (6): rms {rms(db):.3f} max {mx(db):.2f}")
        k = 0
        print(f"  [{label}] steady residual table (attack rows; recover 0.1/0.25/0.5/0.8/1.2 | level -25/+2):")
        for ai, a in enumerate(ATTACKS):
            print(f"      atk {a:5.1f}: {np.round(ss[k:k + 7], 2).tolist()}"); k += 7
        print(f"      freq (0.1 ms 100/5k, 1 ms, 30 ms): {np.round(ss[42:48], 2).tolist()}")
        # the knee trace: the model's static GR against x for the three soft ratios, around the rest point
        xs_show = np.arange(-4.0, 1.01, 0.5)
        xr_show = xs_show - d_cap
        print(f"  [{label}] knee trace, x = L - T' from -4 to +1 (rest at x = {d_cap - det['depth']:+.2f}); model (full chain) vs reference family:")
        for r in (0, 1, 2, 3):
            g = []
            for x in xr_show:
                a = sine(x + T_CUR[15], 1.5)
                gr, _, _ = chain(a, float(FS), det["ta"][2], det["tr"][2], False, det["t2"], det["c2"], T_CUR[15], det["depth"], det["sv"], self.mode,
                                 mech["alpha"], mech["woff"], mech["wd"], mech["taw"], mech["trw"], curves[r], ms[r])
                g.append(-20 * np.log10(np.mean(10 ** (-gr[-4800:] / 20.0))))
            print(f"      {RATIOS[r]:>5}: model {np.round(g, 2).tolist()}\n             ref   {np.round(s3.curve_at(self.cref[r], xs_show), 2).tolist()}")
        return dict(static_report_rms=rms(stf[REPORT_MASK]), static_report_max=mx(stf[REPORT_MASK]), static_band_rms=rms(stf[BAND_MASK]), static_band_max=mx(stf[BAND_MASK]), static_all_rms=rms(stf), static_all_max=mx(stf),
                    knee_rms=rms(stf[knee]), knee_max=mx(stf[knee]), steady_rms=rms(ss), steady_max=mx(ss), burst_rms=rms(bb), burst_max=mx(bb),
                    dual_rms=rms(d5), dual_max=mx(d5), dualb_rms=rms(db), dualb_max=mx(db), standing=stand, p=[float(v) for v in p], names=self.P.names)


def fit(h, p0, free, nfev, label):
    """least_squares over the free indices of p0 (others held), soft_l1 as stage 3"""
    p0 = np.asarray(p0, dtype=float); free = np.asarray(free)
    lo = np.array(h.P.det_lo + [m[2] for m in h.P.mech]); hi = np.array(h.P.det_hi + [m[3] for m in h.P.mech]); xs = np.array(h.P.det_xs + [m[4] for m in h.P.mech])
    base = p0.copy()
    def rf(q):
        p = base.copy(); p[free] = q
        return h.residual(p)
    t0 = time.time(); h.nev = 0
    r = least_squares(rf, np.clip(p0[free], lo[free], hi[free]), bounds=(lo[free], hi[free]), x_scale=xs[free], diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
    p = base.copy(); p[free] = r.x
    print(f"  [{label}] fit: nfev {r.nfev} ({h.nev} evaluations, {time.time() - t0:.0f} s) cost {r.cost:.3f} status {r.status}")
    return p, r


def scan(h, p0):
    """coarse grid over the mechanism's two shape constants (follower held at 0.1 ms attack, 10 s release unless --trw): returns the
    best start for pass A by the soft-L1 cost"""
    names = h.P.names
    def setp(p, name, val):
        if name in names: p[names.index(name)] = val
    grids = {2: ("alpha", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], "woff", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
             4: ("wd", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "woff", [-1.0, -0.5, 0.0, 0.5, 1.0])}
    if h.variant not in grids: return p0
    n1, g1, n2, g2 = grids[h.variant]
    best = (np.inf, None); rows = []
    for a in g1:
        for b in g2:
            p = np.array(p0, dtype=float); setp(p, n1, a); setp(p, n2, b); setp(p, "log_taw", -4.0); setp(p, "log_trw", 1.0)
            r = h.residual(p); c = float(np.sum(np.sqrt(1.0 + r * r) - 1.0))
            rows.append((a, b, c))
            if c < best[0]: best = (c, p.copy())
    print(f"  [scan] {n1} x {n2} grid (cost): " + "; ".join(f"{a:g}/{b:g}: {c:.1f}" for a, b, c in rows))
    print(f"  [scan] best cost {best[0]:.2f} at {dict(zip(names[16:], np.round(best[1][16:], 3).tolist()))}")
    return best[1]


def reference_tail():
    """the discriminating measurement on the reference plug-in: does the gain reduction below the soft ratios' knee follow the level
    down at the release rate (no memory), or does it stand for seconds after a burst as a slow follower would make it?"""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
    import pa
    p = pa.Ref(); print(f"reference {p.version} {p.sha256[:16]}")
    S = dict(discrete_bypass="In", discrete_threshold=16, discrete_attack=1.0, discrete_recover="0.5 s")
    def env(st, ratio):
        y = p.run(protocol.stimulus(st, pa.FS), **pa.both(**S, discrete_ratio=ratio))
        return GAIN12 - np.asarray(protocol.feature({"feat": {"type": "env"}, "stim": st}, y, pa.FS))
    def at(gr, t_end, ts):
        return [round(float(np.mean(gr[int((t_end + t) * 1000) - 50:int((t_end + t) * 1000)])), 3) for t in ts]
    ts = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
    print("(i) burst -50 -> -10 (2 s) -> -50, thr 16, 1 ms / 0.5 s: reference GR (dB, 50-period mean) at 0.5/1/2/4/8/12 s after the burst end")
    for r in ("1.2:1", "2:1", "3:1", "4:1"):
        st = {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 12.5, "f": 1000.0}
        gr = env(st, r); print(f"   {r:>5}: {at(gr, 2.5, ts)}   (during the burst, last 50 periods: {round(float(np.mean(gr[2450:2500])), 3)})")
    print("(ii) tone about 1.5 dB below the rest level at 2:1 and 1.2:1: from silence, and after a 2 s burst at -10")
    for r in ("2:1", "1.2:1"):
        for lvl in (-39.5, -40.5):
            st = {"kind": "burst", "pre": -120, "level": lvl, "pre_s": 0.5, "burst_s": 12.5, "post_s": 0.1, "f": 1000.0}
            a = at(env(st, r), 0.5, ts)
            st2 = {"kind": "burst", "pre": lvl, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 12.5, "f": 1000.0}
            b = at(env(st2, r), 2.5, ts)
            print(f"   {r:>5} tone {lvl} dBFS: from silence {a} | after the burst {b}")


def main():
    args = sys.argv[1:]
    if "--reference-tail" in args:
        reference_tail(); return
    trw_fixed = float(args[args.index("--trw") + 1]) if "--trw" in args else None
    static_w = float(args[args.index("--static-w") + 1]) if "--static-w" in args else 1.0
    ev = args[args.index("--eval") + 1] if "--eval" in args else None   # JSON:variant:key -> report that saved vector only
    variants = [int(v) for v in args[args.index("--variants") + 1].split(",")] if "--variants" in args else [0, 1, 2, 4, 5]
    nfev_a = int(args[args.index("--nfev-a") + 1]) if "--nfev-a" in args else 40
    nfev_b = int(args[args.index("--nfev-b") + 1]) if "--nfev-b" in args else 40
    print(f"GAIN12 {GAIN12:.4f}; current constants: depth {float(cal[MODEL.field('d_rel_depth_db')][0]):.3f} Sv {float(cal[MODEL.field('d_att_sv_db')][0]):.2f}")
    Tp, cref = s3.fit_static()
    results = {}
    for v in variants:
        print(f"\n=== VARIANT {v}: {VNAMES[v]}")
        h = Harness(v, cref, Tp, trw_fixed, static_w)
        p0 = h.P.full0()
        if trw_fixed is not None: print(f"  follower release held at {trw_fixed} s")
        if static_w != 1.0: print(f"  static grid weight {static_w} in the residual")
        if ev:
            path, vv, key = ev.split(":")
            d = json.load(open(path))[vv][key]
            print(f"  re-reporting {path} variant {vv} [{key}]: {dict(zip(d['names'], np.round(d['p'], 5).tolist()))}")
            results[v] = {key: h.report(np.array(d["p"]), f"eval {key}")}
            continue
        t0 = time.time(); r0 = h.residual(p0); print(f"  residual length {len(r0)}, one evaluation {time.time() - t0:.2f} s (first includes numba compile)")
        t0 = time.time(); h.residual(p0); print(f"  one evaluation {time.time() - t0:.2f} s")
        rep = {"start": h.report(p0, "start")}
        if "--scan" in args and h.P.mech:
            p0 = scan(h, p0)
            rep["scan"] = h.report(p0, "scan best")
        if h.P.mech:
            free = list(range(16, 16 + len(h.P.mech)))
            pA, _ = fit(h, p0, free, nfev_a, "A mech-only")
            rep["A"] = h.report(pA, "A mech-only")
        else:
            pA = p0
        if "--skip-joint" not in args:
            free = list(range(16 + len(h.P.mech)))
            pB, _ = fit(h, pA, free, nfev_b, "B joint")
            rep["B"] = h.report(pB, "B joint")
        results[v] = rep
        sys.stdout.flush()
        tag = f"v{'-'.join(map(str, variants))}" + ("-scan" if "--scan" in args else "") + (f"-trw{trw_fixed:g}" if trw_fixed is not None else "")
        json.dump(results, open(os.path.join("/tmp", f"disc-knee-feedthrough-{tag}.json"), "w"), indent=1)
    print("\n=== SUMMARY (static report-set rms/max | steady rms | bursts rms | DUAL rms | standing GR max)")
    for v, rep in results.items():
        for k, d in rep.items():
            print(f"  V{v} {k:12s}: statics 54 {d['static_report_rms']:.3f}/{d['static_report_max']:.2f} (198 {d['static_band_rms']:.3f}/{d['static_band_max']:.2f}, grid {d['static_all_rms']:.3f}/{d['static_all_max']:.2f}, knee {d['knee_rms']:.3f}/{d['knee_max']:.2f}) | "
                  f"steady {d['steady_rms']:.3f}/{d['steady_max']:.2f} | bursts {d['burst_rms']:.3f}/{d['burst_max']:.2f} | DUAL {d['dual_rms']:.3f}/{d['dual_max']:.2f} | silence {max(abs(s) for s in d['standing']):.4f}")


if __name__ == "__main__":
    main()
