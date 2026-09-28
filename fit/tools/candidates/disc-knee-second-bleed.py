# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage, soft-ratio knee: can the node fall below the release reference? Directions (c) and (d) of the knee hole.

The problem. The detector node rests at rest = T - depth and cannot follow a level below it (only the bleed acts), so every sine at or
below rest leaves the node at rest and the gain reduction at C_r(-depth). The reference shows zero gain reduction at low levels, yet
its static curves at 1.2:1, 2:1 and 3:1 still compress at the rest level (0.49 / 1.08 / 0.58 dB) and fade to zero over the 2 dB
below it; 4:1, 6:1 and FLOOD start above rest (+0.4 / +1.3 / +3.6 dB). Pinning the node-domain curves to zero at rest costs up to
1.1 dB at the soft-ratio knees.

Hypotheses tested here, all on the same residual (the stage-3 dynamic residual: steady cells x 3, burst envelopes per sample, DUAL
items; plus the full 6 x 24 x 25 static grid with the curves refitted in the TRUE node domain at every evaluation):
  (d) mode 0  the current node, with the static family refitted in the node domain by simulating the node on every static point
              (mean node over the item's last 0.5 s, and the exact lock-in gain through the per-sample curve with the within-cycle
              ripple). If the node's own onset off rest with a hard curve could make the soft-ratio statics, this fit would show it.
              It cannot reproduce any gain reduction at a level at or below rest: that band is the irreducible residual it reports.
  (c) mode 1  a second, slow bleed from the node toward rest - D, conductance kR / lam (equilibrium at silence rest - D / (1 + lam),
              the same at every recover position); the bleed to rest stays two-way (a resistor).
      mode 2  as mode 1 with the leak's time constant fixed in seconds (tg), so the silence equilibrium depends on the recover position.
      mode 3  the leak toward the log floor (-100 dB), conductance kR / lam: the GERMANIUM hook already in DiscreteStage (leakRatio);
              its silence equilibrium depends on the threshold's absolute level, which the 24-threshold grid can see.
      mode 4  the bleed to rest is one-way (conducts only while the node is above rest, a diode + resistor) and the slow leak toward
              rest - D is what discharges the node below rest. Above rest this is exactly the current node; below rest the node
              tracks a quiet signal down to rest - D, and rests at rest - D in silence.
      mode 5  mode 4 with the leak's time constant fixed in seconds.
Every mode starts from the current constants; the curves are never taken from constants.json but refitted (isotonic regression in
the node domain, pinned to zero at the silence node; the threshold convention is shifted so that the silence node sits exactly on
a knot of the 1 dB curve grid, so that the C++ PCHIP is exactly zero there).
Reported per mode: rendered statics through the mirror (exact ripple, lock-in gain, PCHIP curve) on the stage-3 report set (thresholds
4, 12, 20 x levels -30, -15, 0) and on the fuller set (levels -30..0 step 3), the standing gain reduction at silence per ratio, the
steady-state table, the bursts and the DUAL items, against the current best (statics 0.216 / 0.73, steady ~0.1, bursts ~0.01, DUAL 0.011).
The mirror is checked against the C++ engine on the baseline first.
usage: cd <repo> && python3 -u fit/tools/candidates/disc-knee-second-bleed.py [--modes 0,1,2,3,4,5] [--nfev 40] [--no-cpp]"""
import os, sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402
import stage3_discrete as S3  # noqa: E402

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
XG = -20.0 + np.arange(96); N = 96; X0 = -20.0
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
         5: "(c) one-way bleed to rest (diode) + slow leak to rest - D, fixed tg"}


# ------------------------------------------------------------------------------------------------ the curve, as the C++ evaluates it
@njit(cache=True)
def pchip_build(y):
    """Fritsch-Carlson slopes, as Curve::build"""
    m = np.empty(N); d = np.empty(N - 1)
    for i in range(N - 1): d[i] = y[i + 1] - y[i]
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
    u = x - X0
    if u <= 0.0: return y[0]
    if u >= N - 1.0: return y[N - 1] + m[N - 1] * (x - (X0 + N - 1.0))
    i = int(u); t = u - i; t2 = t * t; t3 = t2 * t
    return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * m[i + 1]


# ------------------------------------------------------------------------------------------------ the node
@njit(cache=True)
def leak_coefs(mode, fs, tr, lam, kR, rest, D):
    """returns (kL, target, oneway, silence node)"""
    if mode == 0:
        return 0.0, rest, False, rest
    if mode == 1 or mode == 4:
        kL = 1.0 - np.exp(-1.0 / (lam * tr * fs)); tgt = rest - D
    elif mode == 2 or mode == 5:
        kL = 1.0 - np.exp(-1.0 / (lam * fs)); tgt = rest - D
    else:
        kL = 1.0 - np.exp(-1.0 / (lam * tr * fs)); tgt = FLOOR
    oneway = mode == 4 or mode == 5
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
    The gain is the fundamental's amplitude over the last `last` seconds relative to the input, as the protocol's gain_db feature."""
    m = levels.shape[0]; gain = np.empty(m); vmean = np.empty(m)
    for j in prange(m):
        n = int(secs * fs); n0 = n - int(last * fs)
        amp = 10.0 ** (levels[j] / 20.0); wv = 2.0 * np.pi * f0s[j] / fs
        a = np.empty(n)
        for i in range(n): a[i] = abs(amp * np.sin(wv * i))
        v = node(a, fs, tas[j], trs[j], False, 1.0, 1.0, Tks[j], depth, sv, mode, D, lam)
        r = ridx[j]; sr = 0.0; sm = 0.0
        for i in range(n0, n):
            g = pchip_at(curves[r], slopes[r], v[i] - Tks[j])
            if g < 0.0: g = 0.0
            s = np.sin(wv * i)
            sr += a[i] * (10.0 ** (-g / 20.0)) * s * (1.0 if s >= 0.0 else -1.0)   # a is |x|: restore the sign
            sm += v[i]
        gain[j] = 20.0 * np.log10(2.0 * sr / (n - n0) / amp)
        vmean[j] = sm / (n - n0)
    return gain, vmean


# ------------------------------------------------------------------------------------------------ the static family in the node domain
def pooled_pava(x, y):
    """isotonic regression after pooling identical x (the pinned node maps many levels onto one node value)"""
    xr = np.round(x, 6)
    ux, inv = np.unique(xr, return_inverse=True)
    cnt = np.bincount(inv).astype(float); sy = np.bincount(inv, weights=y)
    fit_u = S3.pava(ux, sy / cnt, cnt)
    return ux, fit_u, fit_u[inv]


def static_setup(Tp):
    data = []
    for ri, r in enumerate(RATIOS):
        for k in range(1, 25):
            for L in LEVELS:
                data.append((ri, k - 1, float(L), GAIN12 - F[f"disc_static_{r}_t{k}_{L}"]))
    d = np.array(data)
    return d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2], d[:, 3]


YS = np.concatenate([np.arange(-70.0, -8.0, 1.0), np.arange(-8.0, 8.0, 0.05), np.arange(8.0, 80.0, 1.0)])
ZERO_CURVES = np.zeros((6, N)); ZERO_SLOPES = np.zeros((6, N))


def node_map(pd, mode):
    """g(y): mean node (relative to T) on a 2.5 s sine y dB above T at the capture setting (1 ms, 0.5 s), from the silence start"""
    ta, tr, t2, c2, depth, sv, D, lam = pd
    m = len(YS)
    _, vm = sine_grid(YS, np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), np.zeros(m), np.zeros(m, dtype=np.int64),
                      ZERO_CURVES, ZERO_SLOPES, float(FS), depth, sv, mode, D, lam, 2.5, 0.5)
    return vm


def silence_node(pd, mode):
    ta, tr, t2, c2, depth, sv, D, lam = pd
    kR = 1.0 - np.exp(-1.0 / (tr[2] * FS))
    return leak_coefs(mode, float(FS), tr[2], lam, kR, -depth, D)[3]   # relative to T


class Static:
    def __init__(self, Tp):
        self.Tp = Tp
        self.ri, self.ki, self.L, self.G = static_setup(Tp)
        self.y0 = -20.0 - Tp[15]   # the stage-3 convention for d0: level -20 dBFS at threshold 16

    def fit(self, pd, mode):
        """curves refitted in the node domain: returns residual (3600), node-domain x per point, fitted values, d0, x_sil"""
        g = node_map(pd, mode)
        d0 = float(np.interp(self.y0, YS, g) - self.y0)
        y = self.L - self.Tp[self.ki] - d0
        xn = np.interp(y, YS, g)
        xs = silence_node(pd, mode)
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
STIM = {iid: np.abs(protocol.stimulus(ITEMS[iid]["stim"], ITEMS[iid]["fs"])[0]) for iid in BURST_IDS}
ENV_N = {iid: min(len(F[iid]), len(STIM[iid]) // int(round(ITEMS[iid]["fs"] / ITEMS[iid]["stim"]["f"]))) for iid in BURST_IDS}


def env_of_item(iid, curves_L, Tp, pd, mode, d0):
    ta, tr, t2, c2, depth, sv, D, lam = pd
    it = ITEMS[iid]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    T = Tp[int(s["discrete_threshold"]) - 1] + d0
    ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
    v = node(STIM[iid], float(fs), ta[ai], tr[rci], dual, t2, c2, T, depth, sv, mode, D, lam)
    r = RATIOS.index(s.get("discrete_ratio", "4:1"))
    gr = np.maximum(S3.curve_at(curves_L[r], v - T), 0.0)
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


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
    """(weighted rms of the non-DUAL bursts, max), (DUAL blen + pulses rms, max) from the dynamic residual after the 54 cells"""
    k = 54; nd, du = [], []
    for iid in BURST_IDS:
        n = ENV_N[iid]; seg = res[k:k + n]; k += n
        (du if "dual" in iid else nd).append(seg)
    nd = np.concatenate(nd); du = np.concatenate(du)
    return (float(np.sqrt(np.mean(nd ** 2))), float(np.max(np.abs(nd)))), (float(np.sqrt(np.mean(du ** 2))), float(np.max(np.abs(du))))


# ------------------------------------------------------------------------------------------------ parameters
def unpack(p, mode):
    ta = list(p[0:6]); tr = list(p[6:12]); t2 = float(p[12]); c2 = float(p[13]); depth = float(p[14]); sv = float(10.0 ** p[15])
    if mode == 0:
        D, lam = 0.0, 1.0
    else:
        D = float(p[16]); lam = float(10.0 ** p[17])
    return ta, tr, t2, c2, depth, sv, D, lam


def start_and_bounds(mode):
    p0 = list(cal[MODEL.field("d_tatt")]) + list(cal[MODEL.field("d_trel")]) + [float(cal[MODEL.field("d_dual_t2")][0]), float(cal[MODEL.field("d_dual_c2")][0]),
                                                                                   float(cal[MODEL.field("d_rel_depth_db")][0]), float(np.log10(cal[MODEL.field("d_att_sv_db")][0]))]
    lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, -20.0, 0.3]; hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 40.0, 3.0]
    xs = [1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 1.0, 0.3]
    if mode in (1, 3, 4):
        p0 += [3.0, np.log10(30.0)]; lo += [0.3, 0.0]; hi += [40.0, 3.5]; xs += [1.0, 0.3]
    elif mode in (2, 5):
        p0 += [3.0, np.log10(3.0)]; lo += [0.3, -1.5]; hi += [40.0, 1.5]; xs += [1.0, 0.3]
    return np.clip(p0, lo, hi), lo, hi, xs


# ------------------------------------------------------------------------------------------------ reports
REPORT54 = [(r, k, L) for r in range(6) for k in (4, 12, 20) for L in (-30, -15, 0)]
REPORT198 = [(r, k, L) for r in range(6) for k in (4, 12, 20) for L in range(-30, 1, 3)]
FULL = [(r, k, L) for r in range(6) for k in range(1, 25) for L in LEVELS]


def rendered_static(items, C, S, Tfin, depth_fin, pd, mode):
    """mirror render of static items: gain through the node + PCHIP curve with the exact ripple; returns model - reference (dB)"""
    ta, tr, t2, c2, _, sv, D, lam = pd
    lv = np.array([float(L) for _, _, L in items]); Tk = np.array([Tfin[k - 1] for _, k, _ in items]); ri = np.array([r for r, _, _ in items], dtype=np.int64)
    m = len(items)
    g, _ = sine_grid(lv, np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), Tk, ri, C, S, float(FS), depth_fin, sv, mode, D, lam, 2.5, 0.5)
    ref = np.array([F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in items])
    return g + GAIN12 - ref


def rms_max(e):
    e = np.asarray(e); return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def report(mode, p, Tp, curves_L, st, label, cpp=False):
    pd = unpack(p, mode); ta, tr, t2, c2, depth, sv, D, lam = pd
    sres, xn, fit, d0, xs, cpts = st.fit(pd, mode)
    # convention shift: put the silence node on a knot
    ks = int(round(xs)); delta = xs - ks
    Tfin = Tp + d0 + delta; depth_fin = depth + delta
    C, S = curves_on_grid(cpts, xs, delta)
    print(f"  {label}: attack ms {np.round(np.array(ta) * 1e3, 3).tolist()} recover s {np.round(tr, 4).tolist()} t2 {t2:.4f} c2 {c2:.2f} depth {depth:.3f} Sv {sv:.2f}"
          + (f" D {D:.3f} dB {'tg' if mode in (2, 5) else 'lam'} {lam:.3f}" if mode else ""))
    print(f"    d0 {d0:+.3f} dB; silence node {xs:+.3f} dB below T (stage-3 convention); convention shift {delta:+.3f} -> rest at {-depth_fin:+.3f}, silence knot {ks:+d}")
    rest_rel = d0 - depth   # rest relative to Tp, in the level domain (T = Tp + d0, rest = T - depth)
    band = st.L - st.Tp[st.ki] - rest_rel
    sel = st.G > 0.3
    print(f"    node-domain static family: rms {rms_max(sres)[0]:.3f} max {rms_max(sres)[1]:.2f} over all 3600 | compressing (>0.3 dB) rms {rms_max(sres[sel])[0]:.3f} max {rms_max(sres[sel])[1]:.2f}")
    for lo_, hi_, nm in ((-99, -2.5, "L < rest - 2.5"), (-2.5, 0.0, "rest - 2.5 <= L <= rest"), (0.0, 3.0, "rest < L <= rest + 3"), (3.0, 99, "L > rest + 3")):
        m = (band > lo_) & (band <= hi_)
        per = " ".join(f"{RATIOS[r]} {rms_max(sres[m & (st.ri == r)])[1]:.2f}" for r in range(6))
        print(f"      band {nm:24s}: rms {rms_max(sres[m])[0]:.3f} max {rms_max(sres[m])[1]:.2f} | max per ratio: {per}")
    e54 = rendered_static(REPORT54, C, S, Tfin, depth_fin, pd, mode); e198 = rendered_static(REPORT198, C, S, Tfin, depth_fin, pd, mode)
    efull = rendered_static(FULL, C, S, Tfin, depth_fin, pd, mode)
    print(f"    RENDERED statics (mirror, exact ripple, PCHIP): 54-item set rms {rms_max(e54)[0]:.3f} max {rms_max(e54)[1]:.2f} | 198-item set rms {rms_max(e198)[0]:.3f} max {rms_max(e198)[1]:.2f} | full grid rms {rms_max(efull)[0]:.3f} max {rms_max(efull)[1]:.2f}")
    fr = np.array([r for r, _, _ in FULL])
    print("      full-grid rms per ratio:", " ".join(f"{RATIOS[r]} {rms_max(efull[fr == r])[0]:.3f}/{rms_max(efull[fr == r])[1]:.2f}" for r in range(6)))
    sil = [float(pchip_at(C[r], S[r], float(ks))) for r in range(6)]
    # the true silence check: a -50 dBFS sine at threshold 1 (as law_disc_gain_*), every ratio
    m = 6
    gs, _ = sine_grid(np.full(m, -50.0), np.full(m, 1000.0), np.full(m, ta[2]), np.full(m, tr[2]), np.full(m, Tfin[0]), np.arange(6, dtype=np.int64), C, S, float(FS), depth_fin, sv, mode, D, lam, 2.0, 0.5)
    print(f"    standing GR at silence per ratio (curve at the silence knot): {np.round(sil, 4).tolist()} | rendered -50 dBFS at threshold 1: {np.round(-gs, 4).tolist()}")
    dres = dyn_resid(curves_L, Tp, pd, mode, d0)
    ss = dres[:54] / 3.0; (brms, bmax), (drms, dmax) = burst_split(dres)
    print(f"    steady table rms {rms_max(ss)[0]:.3f} max {rms_max(ss)[1]:.2f} | bursts weighted rms {brms:.3f} max {bmax:.2f} | DUAL (blen, pulses) rms {drms:.3f} max {dmax:.2f}")
    k = 0
    rows = []
    for ai, a in enumerate(ATTACKS):
        rows.append(f"{a:5.1f} ms: " + " ".join(f"{v:+.2f}" for v in ss[k:k + 7])); k += 7
    print("      steady residual rows (recover 0.1/0.25/0.5/0.8/1.2 | level -25/+2):", " || ".join(rows))
    # release tail of one burst, the last dB of gain reduction
    e = env_of_item("disc_burst_1.0_0.5 s", curves_L, Tp, pd, mode, d0); ref = np.asarray(F["disc_burst_1.0_0.5 s"]); n = min(len(e), len(ref))
    tail = [(j, round(float(e[j]), 2), round(float(ref[j]), 2)) for j in (2600, 2800, 3000, 3300, 3600, 4000) if j < n]
    print("      burst 1 ms / 0.5 s release tail (period, model, ref):", tail)
    if cpp:
        cal2 = cal.copy()
        cal2[MODEL.field("d_thr_db")] = Tfin; cal2[MODEL.field("d_curve")] = C.reshape(-1); cal2[MODEL.field("d_tatt")] = ta; cal2[MODEL.field("d_trel")] = tr
        cal2[MODEL.field("d_dual_t2")] = t2; cal2[MODEL.field("d_dual_c2")] = c2; cal2[MODEL.field("d_rel_depth_db")] = depth_fin; cal2[MODEL.field("d_att_sv_db")] = sv
        cal2[MODEL.field("d_goff_db")] = 0.0
        ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t{k}_{L}"], cal2) - F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in REPORT54])
        print(f"    C++ engine on the same constants (mode 0 only is faithful): 54-item rms {rms_max(ec)[0]:.3f} max {rms_max(ec)[1]:.2f}; mirror - C++ max |diff| {np.max(np.abs(e54 - ec)):.4f} dB")
    sys.stdout.flush()
    return dict(mode=mode, p=p, C=C, Tfin=Tfin, depth_fin=depth_fin, e54=rms_max(e54), e198=rms_max(e198), efull=rms_max(efull), sil=sil, steady=rms_max(ss), bursts=(brms, bmax), dual=(drms, dmax))


def baseline_check(Tp, curves_L):
    """the current constants through the mirror and through the C++: validates the mirror (curves from constants.json, not refitted)"""
    C = cal[MODEL.field("d_curve")].reshape(6, N).copy(); S = np.stack([pchip_build(C[r]) for r in range(6)])
    T = cal[MODEL.field("d_thr_db")].copy(); depth = float(cal[MODEL.field("d_rel_depth_db")][0]); sv = float(cal[MODEL.field("d_att_sv_db")][0])
    pd = (list(cal[MODEL.field("d_tatt")]), list(cal[MODEL.field("d_trel")]), float(cal[MODEL.field("d_dual_t2")][0]), float(cal[MODEL.field("d_dual_c2")][0]), depth, sv, 0.0, 1.0)
    e54 = rendered_static(REPORT54, C, S, T, depth, pd, 0)
    ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t{k}_{L}"], cal) - F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in REPORT54])
    print(f"BASELINE (constants.json as they are): C++ engine 54-item statics rms {rms_max(ec)[0]:.3f} max {rms_max(ec)[1]:.2f}")
    print(f"  mirror on the same constants: rms {rms_max(e54)[0]:.3f} max {rms_max(e54)[1]:.2f}; mirror - C++ max |diff| {np.max(np.abs(e54 - ec)):.4f} dB")
    xs = -depth
    sil = [float(pchip_at(C[r], S[r], xs)) for r in range(6)]
    print(f"  standing GR at silence per ratio, C++ PCHIP at rest ({xs:+.3f} dB, between the knots -1 and 0): {np.round(sil, 3).tolist()}")
    d0 = float(T[15] - Tp[15])
    dres = dyn_resid(curves_L, Tp, pd, 0, d0); ss = dres[:54] / 3.0; (brms, bmax), (drms, dmax) = burst_split(dres)
    print(f"  dynamics on the current constants (stage-3 method): steady rms {rms_max(ss)[0]:.3f} max {rms_max(ss)[1]:.2f} | bursts rms {brms:.3f} max {bmax:.2f} | DUAL rms {drms:.3f} max {dmax:.2f}")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ main
def main():
    args = sys.argv[1:]
    modes = [int(m) for m in args[args.index("--modes") + 1].split(",")] if "--modes" in args else [0, 1, 2, 3, 4, 5]
    nfev = int(args[args.index("--nfev") + 1]) if "--nfev" in args else 40
    w_static = float(args[args.index("--wstatic") + 1]) if "--wstatic" in args else 1.0
    print(f"GAIN12 {GAIN12:.4f}; reference GR at -50 dBFS, threshold 1, gain 12: {GAIN12 - F['law_disc_gain_12']:+.4f} dB")
    Tp, curves_L = S3.fit_static()
    st = Static(Tp)
    if "--no-cpp" not in args:
        baseline_check(Tp, curves_L)
    results = {}
    for mode in modes:
        print(f"\nMODE {mode}: {NAMES[mode]}")
        p0, lo, hi, xs = start_and_bounds(mode)
        t0 = time.time()
        report(mode, p0, Tp, curves_L, st, "start (current constants, curves refitted in the node domain)")
        print(f"    (one evaluation + report: {time.time() - t0:.1f} s)")
        def resid(p):
            pd = unpack(p, mode)
            sres, _, _, d0, _, _ = st.fit(pd, mode)
            return np.concatenate([dyn_resid(curves_L, Tp, pd, mode, d0), w_static * sres])
        t0 = time.time()
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
        print(f"  fit: nfev {r.nfev} cost {r.cost:.3f} ({time.time() - t0:.0f} s)")
        results[mode] = report(mode, r.x, Tp, curves_L, st, "fitted", cpp=("--cpp" in args))
        print("  p:", np.round(r.x, 6).tolist())
    print("\nSUMMARY (current best: statics 0.216 / 0.73, steady ~0.1, bursts ~0.01, DUAL 0.011)")
    for mode, R in results.items():
        print(f"  mode {mode}: statics54 {R['e54'][0]:.3f}/{R['e54'][1]:.2f} statics198 {R['e198'][0]:.3f}/{R['e198'][1]:.2f} full {R['efull'][0]:.3f}/{R['efull'][1]:.2f} | silence {np.round(R['sil'], 3).tolist()} | steady {R['steady'][0]:.3f}/{R['steady'][1]:.2f} bursts {R['bursts'][0]:.3f}/{R['bursts'][1]:.2f} DUAL {R['dual'][0]:.3f}/{R['dual'][1]:.2f}")


if __name__ == "__main__":
    main()
