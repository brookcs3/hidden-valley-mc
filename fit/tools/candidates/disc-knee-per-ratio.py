# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage, soft-ratio knee: direction (b), the six ratio positions do not share one rest reference.

Hypothesis. The detector node cannot fall below its rest point rest = T - depth (depth 0.40 dB from the 4:1 dynamics), yet the
reference's 1.2:1, 2:1 and 3:1 static curves begin 1.9 / 1.8 / 0.7 dB BELOW that rest (their knees start at x = L - T = -2.3, -2.2,
-1.1 dB; 4:1, 6:1 and FLOOD start at +0.1, +1.0, +3.5 dB). A curve read off a node that sits at rest for every level below it can
only give one value there, so pinning it to zero costs the whole knee of the soft ratios. If instead each ratio position carries its
own rest reference (a per-ratio depth, or a per-ratio threshold offset applied to the node and the curve alike), the soft ratios' rest
can sit below their knee while 4:1 keeps the depth the bursts, the steady-state table and DUAL fitted, all of which are at 4:1.

What this harness does, all from the reference static grid disc_static_{ratio}_t{k}_{L} (6 x 24 x 25) and the current detector:
  1. simulates the node on every static point at the capture setting (attack 1 ms, recover 0.5 s) with the stage-3 detector mirror,
     giving the true node level (mean over the last 0.5 s, plus the last ten periods for the within-cycle ripple);
  2. fits one monotone curve per ratio in the TRUE node domain (isotonic regression), under
       F0  no node at all: a monotone function of L - T (the floor any curve-of-level model can reach, T fixed);
       F1  shared rest (depth 0.40), curve pinned to zero at rest (the current model done properly), and unpinned (the standing GR it wants);
       F2  per-ratio depth d_r: rest_r = T - d_r, the node simulated with d_r, curve pinned at rest_r, scanned over d_r;
       F3  per-ratio threshold offset o_r with the shared depth: rest = T + o_r - 0.40, curve read at v - T - o_r, scanned over o_r;
       F4  shared rest, curve shifted per ratio (a reparametrisation of F1: shown to be one);
  3. realises the winning curves on the C++ grid (PCHIP mirror of Discrete.hpp's Curve) under the current layout (v - T from -20 dB,
     1 dB steps) and under a rest-relative layout (v - rest from 0), with the grid values refined through the PCHIP itself, and
     evaluates the exact lock-in gain with the node's within-cycle ripple;
  4. renders the stage-3 report set through the C++ engine for the current constants and for the realisable variant (integer depths
     for the soft ratios, so their rest lands on a grid point), and reports the standing gain reduction at silence per ratio;
  5. re-evaluates the 4:1 dynamics (steady-state table, level and frequency checks, bursts, DUAL) through the mirror for the current
     constants and for the node-domain 4:1 curve, since nothing else in the change touches them.

usage: cd <repo> && python3 -u fit/tools/candidates/disc-knee-per-ratio.py [--no-render] [--depths 0.4,1,2,3,4] [--margin 0.5] [--ripple-fit]
"""
import os, sys, time, json, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, feat_residual, protocol, FS  # noqa: E402

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
LEVELS = list(range(-60, 13, 3))
cal0 = load_cal()
GAIN = cal0[MODEL.field("d_gain_db")]; XGN = float(cal0[MODEL.fields["x_gain_db"][0]]); GAIN12 = float(GAIN[11] + XGN)
T = cal0[MODEL.field("d_thr_db")].copy()
CURVES0 = cal0[MODEL.field("d_curve")].reshape(6, 96).copy()
DEPTH0 = float(cal0[MODEL.field("d_rel_depth_db")][0]); SV = float(cal0[MODEL.field("d_att_sv_db")][0])
TA = list(cal0[MODEL.field("d_tatt")]); TR = list(cal0[MODEL.field("d_trel")])
T2 = float(cal0[MODEL.field("d_dual_t2")][0]); C2 = float(cal0[MODEL.field("d_dual_c2")][0])
FLOOR = float(cal0[MODEL.field("d_floor_db")][0])
TAIL = 480   # ten periods of 1 kHz at 48 kHz, whole periods, phase aligned with the stimulus
ARGS = sys.argv[1:]


# ------------------------------------------------------------------------------------------------ detector mirror
@njit(cache=True)
def detector(a, fs, ta, tr, floor_db, dual, t2, c2, goff, Tk, depth, sv):
    """stage-3 mirror of DiscreteStage::process: always-on bleed toward Tk - depth, attack diode with the level law, DUAL second node"""
    n = a.shape[0]
    v = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    k2 = 1.0 - np.exp(-1.0 / (t2 * fs)) if dual else 0.0
    floor_lin = 10.0 ** (floor_db / 20.0)
    rest = Tk - depth
    x = rest; w = rest
    for i in range(n):
        e = (20.0 * np.log10(a[i]) if a[i] > floor_lin else floor_db) + goff
        dv = (rest - x) * kR
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


@njit(cache=True)
def sim_sine(level_db, f, fs, secs, Tk, depth, ta, tr, sv, floor_db, tail):
    """steady sine through the detector: mean node over the last 0.5 s and its last `tail` samples"""
    n = int(secs * fs)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    fl = 10.0 ** (floor_db / 20.0); rest = Tk - depth
    amp = 10.0 ** (level_db / 20.0)
    v = rest; s = 0.0; cnt = 0
    vt = np.empty(tail)
    n0 = n - int(0.5 * fs)
    for i in range(n):
        a = abs(amp * np.sin(2.0 * np.pi * f * i / fs))
        e = 20.0 * np.log10(a) if a > fl else floor_db
        dv = (rest - v) * kR
        if e > v:
            xe = e - rest
            fct = 1.0 + xe / sv if xe > 0.0 else 1.0
            dv += (e - v) * (1.0 - np.exp(-rA * fct))
        v += dv
        if i >= n0:
            s += v; cnt += 1
        if i >= n - tail:
            vt[i - (n - tail)] = v
    return s / cnt, vt


@njit(parallel=True, cache=True)
def sim_batch(levels, Tks, depth, ta, tr, sv, floor_db, fs, secs, tail):
    n = levels.shape[0]
    vm = np.empty(n); vt = np.empty((n, tail))
    for i in prange(n):
        m, t = sim_sine(levels[i], 1000.0, fs, secs, Tks[i], depth, ta, tr, sv, floor_db, tail)
        vm[i] = m; vt[i, :] = t
    return vm, vt


# ------------------------------------------------------------------------------------------------ PCHIP mirror of Discrete.hpp Curve
class Grid:
    def __init__(self, x0, dx, n):
        self.x0, self.dx, self.n = x0, dx, n
        self.xg = x0 + dx * np.arange(n)

    def slopes(self, y):
        d = np.diff(y) / self.dx
        m = np.empty(self.n); m[0] = d[0]; m[-1] = d[-1]
        for i in range(1, self.n - 1):
            m[i] = 0.0 if d[i - 1] * d[i] <= 0.0 else 0.5 * (d[i - 1] + d[i])
        for i in range(self.n - 1):
            if d[i] == 0.0:
                m[i] = m[i + 1] = 0.0; continue
            a, b = m[i] / d[i], m[i + 1] / d[i]; s = a * a + b * b
            if s > 9.0:
                t = 3.0 / np.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]
        return m

    def at(self, y, x):
        """vectorised Curve::at"""
        m = self.slopes(y)
        x = np.asarray(x, dtype=float); u = (x - self.x0) / self.dx
        out = np.empty_like(u)
        lo = u <= 0.0; hi = u >= self.n - 1; mid = ~(lo | hi)
        out[lo] = y[0]
        out[hi] = y[-1] + m[-1] * (x[hi] - (self.x0 + self.dx * (self.n - 1)))
        i = u[mid].astype(int); t = u[mid] - i; t2 = t * t; t3 = t2 * t
        out[mid] = ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * self.dx * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * self.dx * m[i + 1])
        return out


GRID_T = Grid(-20.0, 1.0, 96)     # the current layout: gain reduction against v - T
GRID_R = Grid(0.0, 1.0, 96)       # the proposed layout: gain reduction against v - rest (the first point IS the rest point)


# ------------------------------------------------------------------------------------------------ isotonic regression
def pava(x, y, w):
    order = np.argsort(x, kind="stable"); ys = y[order]; ws = w[order]
    vals = list(ys); wts = list(ws); idx = [[i] for i in range(len(ys))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            tw = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tw
            wts[i] = tw; idx[i] += idx[i + 1]
            del vals[i + 1]; del wts[i + 1]; del idx[i + 1]
            if i > 0: i -= 1
        else:
            i += 1
    out = np.empty(len(ys))
    for v, ids in zip(vals, idx):
        out[ids] = v
    res = np.empty(len(ys)); res[order] = out
    return res


def iso_fit(xn, G, w, pin_at=None):
    """monotone fit of G against xn; with pin_at = x0 the curve is 0 at and below x0 (points there are predicted 0 and excluded from
    the regression, an anchor of weight 1e6 at x0 keeps the rest above 0). Returns (pred, knots_x, knots_y) with knots sorted/unique."""
    if pin_at is None:
        fit = pava(xn, G, w)
        pred = fit
        xs, ys = xn, fit
    else:
        above = xn > pin_at + 1e-9
        xa = np.concatenate([[pin_at], xn[above]]); ya = np.concatenate([[0.0], G[above]]); wa = np.concatenate([[1e6], w[above]])
        fit = pava(xa, ya, wa)
        pred = np.zeros(len(G)); pred[above] = fit[1:]
        xs, ys = xa, fit
    order = np.argsort(xs, kind="stable"); xs, ys = xs[order], ys[order]
    ux = np.unique(xs)
    uy = np.array([ys[xs == u].mean() for u in ux])
    uy = np.maximum.accumulate(uy)
    return pred, ux, uy


def knots_to_grid(ux, uy, grid, origin):
    """resample a knot curve (x in v - T units) onto a grid whose x is v - T - origin; linear beyond the knots, non-decreasing"""
    xg = grid.xg + origin
    y = np.interp(xg, ux, uy, left=0.0)
    beyond = xg > ux[-1]
    if np.any(beyond):
        tail = max(0, len(uy) - 40)
        slope = (uy[-1] - uy[tail]) / max(1e-6, ux[-1] - ux[tail])
        y[beyond] = uy[-1] + slope * (xg[beyond] - ux[-1])
    return np.maximum.accumulate(np.maximum(y, 0.0))


def lockin_gr(y, grid, origin, vt):
    """exact gain reduction the 'gain_db' feature sees: lock-in of sin * 10^(-GR(t)/20) over the tail periods (vt in v - T units)"""
    n, tail = vt.shape
    gr = grid.at(y, (vt - origin).reshape(-1)).reshape(n, tail)
    g = 10.0 ** (-gr / 20.0)
    ph = 2.0 * np.pi * np.arange(tail) / 48.0
    c = 2.0 * np.mean(g * np.sin(ph)[None, :] * np.exp(-1j * ph)[None, :], axis=1)
    return -20.0 * np.log10(np.abs(c))


def refine_grid(y0, grid, origin, xn, G, w, vt=None, first_free=0, n_free=None, pin0=True):
    """least-squares refinement of the grid values through the PCHIP (the C++ evaluator), monotone by construction (increments >= 0).
    xn: node level in v - T units; the grid's x is v - T - origin. If vt (tail trajectories, v - T) is given the residual is the exact
    lock-in gain with the within-cycle ripple instead of the curve at the mean node."""
    y0 = y0.copy(); n = grid.n
    if n_free is None: n_free = n - first_free
    inc0 = np.diff(np.concatenate([[0.0], y0]))
    free = np.arange(first_free, first_free + n_free)
    def build(p):
        inc = inc0.copy(); inc[free] = p
        if pin0: inc[0] = 0.0
        return np.cumsum(inc)
    def resid(p):
        y = build(p)
        if vt is None:
            return w * (grid.at(y, xn - origin) - G)
        return w * (lockin_gr(y, grid, origin, vt) - G)
    p0 = np.maximum(inc0[free], 0.0)
    r = least_squares(resid, p0, bounds=(np.zeros(len(free)), np.full(len(free), 10.0)), x_scale=0.05, diff_step=1e-3, max_nfev=60, loss="linear")
    return build(r.x)


# ------------------------------------------------------------------------------------------------ static data
def static_table():
    ri, ki, Lv, G = [], [], [], []
    for r_, r in enumerate(RATIOS):
        for k in range(1, 25):
            for L in LEVELS:
                ri.append(r_); ki.append(k - 1); Lv.append(float(L)); G.append(GAIN12 - F[f"disc_static_{r}_t{k}_{L}"])
    ri, ki, Lv, G = map(np.array, (ri, ki, Lv, G))
    return ri, ki, Lv, G, Lv - T[ki]


def rms(a): return float(np.sqrt(np.mean(np.square(a))))
def mx(a): return float(np.max(np.abs(a)))


def report_resid(tag, res, x, G, per_ratio=None):
    knee = (x > -6.0) & (x < 4.0); comp = G > 0.3
    s = f"{tag}: all {rms(res):.4f}/{mx(res):.3f}  knee(-6<x<4) {rms(res[knee]):.4f}/{mx(res[knee]):.3f}  compressing(G>0.3) {rms(res[comp]):.4f}/{mx(res[comp]):.3f}"
    if per_ratio is not None:
        s += "  per ratio rms " + " ".join(f"{RATIOS[r]}:{rms(res[per_ratio == r]):.4f}" for r in range(6))
    print(s)


# ------------------------------------------------------------------------------------------------ dynamics (stage-3 residual definitions, 4:1 only)
BURSTS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS] + [f"disc_dual_blen_{b}" for b in (0.1, 0.5, 2.0, 8.0)] + ["disc_dual_pulses"] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]


def d0_for(ta, tr, depth, level=-20.0, secs=2.5, f=1000.0, Tk=-37.5):
    t = np.arange(int(secs * FS)) / FS
    a = np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))
    v = detector(a, float(FS), ta, tr, FLOOR, False, T2, C2, 0.0, Tk, depth, SV)
    return float(v[-int(0.5 * FS):].mean() - level)


def env_of_item(iid, curve_fn, ta, tr, depth):
    it = ITEMS[iid]; st = it["stim"]; fs = it["fs"]; s = it["set"]
    x = protocol.stimulus(st, fs)[0]
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    Tk = T[int(s["discrete_threshold"]) - 1]
    v = detector(np.abs(x), float(fs), ta, tr, FLOOR, dual, T2, C2, 0.0, Tk, depth, SV)
    gr = np.maximum(curve_fn(v - Tk), 0.0)
    per = int(round(fs / st["f"])); m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def dynamics(curve_fn, depth, tag):
    """curve_fn: x = v - T -> GR for the 4:1 curve. Prints steady / bursts / DUAL rms like stage 3."""
    T16 = T[15]; ss = []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            d = d0_for(TA[ai], TR[rci], depth, level=-10.0, secs=3.0, Tk=T16)
            ss.append(float(curve_fn(np.array([-10.0 + d - T16]))[0]) - (GAIN12 - F[f"disc_ar_{a}_{rc}"]))
        for lvl in (-25, 2):
            d = d0_for(TA[ai], TR[2], depth, level=float(lvl), secs=3.0, Tk=T16)
            ss.append(float(curve_fn(np.array([lvl + d - T16]))[0]) - (GAIN12 - F[f"disc_al_{a}_{lvl}"]))
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            d = d0_for(TA[ATTACKS.index(a)], TR[2], depth, level=-10.0, secs=3.0, f=f, Tk=T16)
            ss.append(float(curve_fn(np.array([-10.0 + d - T16]))[0]) - (GAIN12 - F[f"disc_af_{a}_f{int(f)}"]))
    ss = np.array(ss)
    bres, dres = [], []
    for iid in BURSTS:
        ref = np.asarray(F[iid]); s_ = ITEMS[iid]["set"]
        ai = ATTACKS.index(float(s_["discrete_attack"])); rci = RECOVERS.index(s_["discrete_recover"])
        e = env_of_item(iid, curve_fn, TA[ai], TR[rci], depth)
        n = min(len(e), len(ref)); w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        (dres if "dual" in iid else bres).append(w * (e[:n] - ref[:n]))
    bres = np.concatenate(bres); dres = np.concatenate(dres)
    print(f"  dynamics [{tag}]: steady rms {rms(ss):.3f} max {mx(ss):.2f} | bursts (non-DUAL) weighted rms {rms(bres):.3f} max {mx(bres):.2f} | DUAL weighted rms {rms(dres):.3f} max {mx(dres):.2f}")
    return rms(ss), rms(bres), rms(dres)


# ------------------------------------------------------------------------------------------------ renders
REPORT54 = [(r, k, L) for r in RATIOS for k in (4, 12, 20) for L in (-30, -15, 0)]
REPORT198 = [(r, k, L) for r in RATIOS for k in (4, 12, 20) for L in range(-30, 1, 3)]


def render_statics(cal_for_ratio, items, tag):
    errs = []; per = {r: [] for r in RATIOS}
    for r, k, L in items:
        iid = f"disc_static_{r}_t{k}_{L}"
        e = float(feat_residual(iid, render_item(ITEMS[iid], cal_for_ratio(r)))[0])
        errs.append(e); per[r].append(e)
    errs = np.array(errs)
    print(f"  rendered statics [{tag}] ({len(items)} items): rms {rms(errs):.3f} max {mx(errs):.2f} | per ratio rms " + " ".join(f"{r}:{rms(np.array(per[r])):.3f}" for r in RATIOS))
    worst = sorted(zip(np.abs(errs), items), reverse=True)[:4]
    print("    worst:", [(f"{r}_t{k}_{L}", round(float(a), 3)) for a, (r, k, L) in worst])
    return rms(errs), mx(errs)


def render_silence(cal_for_ratio, tag):
    out = []
    for r in RATIOS:
        iid = f"disc_static_{r}_t12_-60"
        out.append(GAIN12 - float(render_item(ITEMS[iid], cal_for_ratio(r))))
    print(f"  rendered standing GR at silence [{tag}] per ratio (dB): {np.round(out, 4).tolist()}")
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ri, ki, Lv, G, x = static_table()
    w = np.ones(len(G))
    print(f"static grid: {len(G)} points, GAIN12 {GAIN12:.4f}, depth {DEPTH0:.4f}, Sv {SV:.2f}, ta(1 ms) {TA[2] * 1e3:.3f} ms, tr(0.5 s) {TR[2]:.4f} s")
    lo = x < -12
    print(f"reference gain reduction far below the knee (x < -12, {lo.sum()} points): mean {G[lo].mean():.4f} rms {rms(G[lo]):.4f} max {G[lo].max():.4f} dB (the capture noise floor)")
    onset = {}
    for r in range(6):
        m = ri == r; xs_ = x[m]; gs_ = G[m]; o = np.argsort(xs_)
        first = next((xs_[o][j] for j in range(len(o)) if gs_[o][j] > 0.03 and xs_[o][j] > -6), 0.0); onset[r] = first
        print(f"  {RATIOS[r]:>6}: knee onset (first x with GR > 0.03 dB) at x = L - T = {first:+.2f} dB, i.e. {first + DEPTH0:+.2f} dB from the rest point")

    # ---- the node on every static point (600 distinct (L, T) pairs shared by the six ratios)
    pairs = {}
    for i in range(len(G)):
        pairs.setdefault((Lv[i], ki[i]), []).append(i)
    pk = list(pairs.keys()); pl = np.array([p[0] for p in pk]); pT = np.array([T[p[1]] for p in pk])
    def node_for_depth(depth, Tshift=0.0):
        vm, vt = sim_batch(pl, pT + Tshift, depth, TA[2], TR[2], SV, FLOOR, float(FS), 2.5, TAIL)
        xn = np.empty(len(G)); tails = np.empty((len(G), TAIL))
        for j, key in enumerate(pk):
            for i in pairs[key]:
                xn[i] = vm[j] - (T[ki[i]] + Tshift); tails[i] = vt[j] - (T[ki[i]] + Tshift)
        return xn, tails
    xn0, vt0 = node_for_depth(DEPTH0)
    print(f"node simulated on all points ({time.time() - t0:.1f} s). Node against x = L - T at depth {DEPTH0:.2f}:")
    m41 = ri == 3
    for xt in (-3, -2.5, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 40):
        j = np.argmin(np.abs(x[m41] - xt))
        print(f"    x {x[m41][j]:+6.2f}: node - T {xn0[m41][j]:+6.3f}  (node above rest {xn0[m41][j] + DEPTH0:6.3f}), ripple pk-pk {np.ptp(vt0[m41][j]):.3f} dB")

    # ---- mirror check against the engine on a few points (current curves, current layout, exact lock-in)
    if "--no-render" not in ARGS:
        chk = [(3, 12, -27), (1, 12, -27), (0, 12, -30), (3, 12, -12), (2, 4, -3), (3, 20, -30)]
        print("mirror vs engine (current constants): item, mirror lock-in GR, mirror mean-node GR, engine GR (all dB)")
        for r, k, L in chk:
            i = np.where((ri == r) & (ki == k - 1) & (Lv == L))[0][0]
            gl = lockin_gr(CURVES0[r], GRID_T, 0.0, vt0[i:i + 1])[0]; gm = GRID_T.at(CURVES0[r], np.array([xn0[i]]))[0]
            iid = f"disc_static_{RATIOS[r]}_t{k}_{L}"; ge = GAIN12 - float(render_item(ITEMS[iid], cal0))
            print(f"    {iid:>26}: {gl:7.3f} {gm:7.3f} {ge:7.3f}   ref {G[i]:.3f}")

    # ---- F0: monotone in the level domain, no node (the floor)
    print("\n=== F0 monotone curve of L - T per ratio, no node (the floor with T fixed)")
    resF0 = np.empty(len(G)); floorF0 = {}
    for r in range(6):
        m = ri == r; pred, _, _ = iso_fit(x[m], G[m], w[m]); resF0[m] = pred - G[m]; floorF0[r] = rms(resF0[m])
    report_resid("F0", resF0, x, G, ri)

    # ---- F1: shared rest, true node domain, pinned and unpinned
    print(f"\n=== F1 shared rest (depth {DEPTH0:.2f}), curves fitted in the true node domain")
    resP = np.empty(len(G)); resU = np.empty(len(G))
    for r in range(6):
        m = ri == r
        pred, ux, uy = iso_fit(xn0[m], G[m], w[m], pin_at=-DEPTH0); resP[m] = pred - G[m]
        predu, uxu, uyu = iso_fit(xn0[m], G[m], w[m]); resU[m] = predu - G[m]
        at_rest = xn0[m] <= -DEPTH0 + 1e-9
        print(f"  {RATIOS[r]:>6}: points at rest {at_rest.sum():3d}, their reference GR mean {G[m][at_rest].mean():.3f} max {G[m][at_rest].max():.3f} dB; "
              f"unpinned fit wants {predu[at_rest].mean():.3f} dB standing GR; pinned rms {rms(resP[m]):.4f} max {mx(resP[m]):.3f} | unpinned rms {rms(resU[m]):.4f}")
    report_resid("F1 pinned", resP, x, G, ri)
    report_resid("F1 unpinned", resU, x, G, ri)

    # ---- F4: shared rest, per-ratio curve shift (reparametrisation check)
    print("\n=== F4 shared rest, curve read at v - T - s_r (curve refitted): the residual cannot depend on s_r")
    for s in (-2.0, 0.0, 2.0):
        res = np.empty(len(G))
        for r in range(6):
            m = ri == r; pred, _, _ = iso_fit(xn0[m] - s, G[m], w[m], pin_at=-DEPTH0 - s); res[m] = pred - G[m]
        print(f"  s_r = {s:+.1f}: all rms {rms(res):.5f} (F1 pinned {rms(resP):.5f})")

    # ---- F2: per-ratio depth
    depths = [float(d) for d in ARGS[ARGS.index("--depths") + 1].split(",")] if "--depths" in ARGS else [DEPTH0, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0]
    print("\n=== F2 per-ratio depth d_r: node simulated with rest = T - d_r, curve pinned at rest. rms (all points) per ratio against d_r; floor F0 in the header")
    print("     d_r   " + " ".join(f"{RATIOS[r]:>8}" for r in range(6)) + "     | node above rest at x = -d+0.5 / +1 / +2 (node onset)")
    print("     F0    " + " ".join(f"{floorF0[r]:8.4f}" for r in range(6)))
    F2 = {}; nodes = {DEPTH0: (xn0, vt0)}
    for d in depths:
        if d not in nodes: nodes[d] = node_for_depth(d)
        xn, vt = nodes[d]; row = []
        for r in range(6):
            m = ri == r; pred, ux, uy = iso_fit(xn[m], G[m], w[m], pin_at=-d); F2[(d, r)] = (rms(pred - G[m]), mx(pred - G[m]), ux, uy); row.append(F2[(d, r)][0])
        o41 = np.argsort(x[m41]); ons = [float(np.interp(-d + dx_, x[m41][o41], xn[m41][o41])) + d for dx_ in (0.5, 1.0, 2.0)]
        print(f"  {d:6.2f}  " + " ".join(f"{v:8.4f}" for v in row) + "     | " + " / ".join(f"{o:.3f}" for o in ons))
    best_d = {}; near_d = {}
    for r in range(6):
        cands = sorted(depths); fl = floorF0[r]
        ok = [d for d in cands if F2[(d, r)][0] <= fl + 0.0003]; ok2 = [d for d in cands if F2[(d, r)][0] <= fl + 0.002]
        best_d[r] = min(ok) if ok else min(cands, key=lambda d: F2[(d, r)][0]); near_d[r] = min(ok2) if ok2 else best_d[r]
    print("  smallest d_r reaching the F0 floor (within 0.0003 dB rms) per ratio:", {RATIOS[r]: best_d[r] for r in range(6)},
          "| within 0.002:", {RATIOS[r]: near_d[r] for r in range(6)})
    print("  F2 max |residual| at that d_r per ratio:", {RATIOS[r]: round(F2[(best_d[r], r)][1], 3) for r in range(6)})

    # ---- F3: per-ratio threshold offset o_r with the shared depth (node and curve shifted together)
    print(f"\n=== F3 per-ratio offset o_r, shared depth {DEPTH0:.2f}: rest = T + o_r - depth, curve read at v - T - o_r")
    offs = [-3.0, -2.35, -1.85, -1.6, -1.1, -0.6, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    print("     o_r   " + " ".join(f"{RATIOS[r]:>8}" for r in range(6)))
    F3 = {}
    for o in offs:
        xn, vt = node_for_depth(DEPTH0, Tshift=o); row = []
        for r in range(6):
            m = ri == r; pred, ux, uy = iso_fit(xn[m], G[m], w[m], pin_at=-DEPTH0); F3[(o, r)] = (rms(pred - G[m]), ux, uy); row.append(F3[(o, r)][0])
        print(f"  {o:6.2f}  " + " ".join(f"{v:8.4f}" for v in row))
    print("  (o_r < 0 lowers the rest: o_r = depth - d_r should reproduce the F2 row at d_r if the node onset shape does not depend on the depth;\n"
          "   o_r > 0 raises it, and each ratio breaks where the rest crosses its knee onset: 4:1 between +0.5 and +1, 6:1 between +1.5 and +2, FLOOD above +3)")

    # ---- knee shape in the node domain: is the curve hard once the node onset is accounted for?
    print("\n=== knee shape: reference GR against level above the knee onset (L domain) vs the fitted node-domain curve above rest, per ratio, at its d_r")
    for r in range(6):
        d = best_d[r]; _, _, ux, uy = F2[(d, r)]
        m = ri == r; o = np.argsort(x[m]); xs_, gs_ = x[m][o], G[m][o]
        first = onset[r]
        lvl = [float(np.interp(first + dd, xs_, gs_)) for dd in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)]
        nod = [float(np.interp(-d + dd, ux, uy)) for dd in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)]
        slope5 = (float(np.interp(first + 8.0, xs_, gs_)) - float(np.interp(first + 5.0, xs_, gs_))) / 3.0
        print(f"  {RATIOS[r]:>6} d_r {d:.2f}: L-domain GR at onset+0.25/0.5/1/2/3/5 dB {np.round(lvl, 3).tolist()} | node-domain curve at rest+0.25/0.5/1/2/3/5 {np.round(nod, 3).tolist()} | slope 5..8 dB above onset: {slope5:.3f} dB/dB")

    # ---- grid realisation of F2 at best_d: current layout (v - T grid, integer depths for the soft ratios) and rest-relative layout
    print("\n=== grid realisation (PCHIP as in Discrete.hpp), evaluated at the mean node and with the exact lock-in (ripple)")
    soft = [0, 1, 2]
    margin = float(ARGS[ARGS.index("--margin") + 1]) if "--margin" in ARGS else 0.5   # rest this much below the floor-reaching depth, so the PCHIP has room to rise
    d_int = {r: (float(int(np.ceil(best_d[r] + margin - 1e-9))) if r in soft else DEPTH0) for r in range(6)}
    d_R = {r: (best_d[r] + margin if r in soft else DEPTH0) for r in range(6)}
    print(f"  margin below the floor-reaching depth: {margin} dB")
    print(f"  integer depths for the current layout: {[(RATIOS[r], d_int[r]) for r in range(6)]}; rest-relative layout depths: {[(RATIOS[r], d_R[r]) for r in range(6)]}")
    for d in set(list(d_int.values()) + list(d_R.values())):
        if d not in nodes: nodes[d] = node_for_depth(d)
    curves_T = np.zeros((6, 96)); curves_R = np.zeros((6, 96))
    resT = np.empty(len(G)); resTl = np.empty(len(G)); resR = np.empty(len(G)); resRl = np.empty(len(G))
    for r in range(6):
        m = ri == r
        # current layout, integer depth: knots from the F2 fit at that depth, grid in v - T, zero at and below -d (a grid point)
        d = d_int[r]; xn, vt = nodes[d]
        _, ux, uy = iso_fit(xn[m], G[m], w[m], pin_at=-d)
        y = knots_to_grid(ux, uy, GRID_T, 0.0); y[GRID_T.xg <= -d + 1e-9] = 0.0; y = np.maximum.accumulate(y)
        k0 = int(np.searchsorted(GRID_T.xg, -d + 1e-9))
        y = refine_grid(y, GRID_T, 0.0, xn[m], G[m], w[m], first_free=k0, n_free=min(20, 96 - k0), pin0=False)
        y[:k0] = 0.0
        curves_T[r] = y
        resT[m] = GRID_T.at(y, xn[m]) - G[m]; resTl[m] = lockin_gr(y, GRID_T, 0.0, vt[m]) - G[m]
        # rest-relative layout: grid x' = v - rest, first point is the rest point (exactly zero), best (non-integer) depth
        d = d_R[r]; xn, vt = nodes[d]
        _, ux, uy = iso_fit(xn[m], G[m], w[m], pin_at=-d)
        y = knots_to_grid(ux, uy, GRID_R, -d)   # grid.xg - d are the v - T positions of the rest-relative grid points
        y[0] = 0.0; y = np.maximum.accumulate(y)
        y = refine_grid(y, GRID_R, -d, xn[m], G[m], w[m], first_free=1, n_free=20, pin0=True)
        if "--ripple-fit" in ARGS:
            y = refine_grid(y, GRID_R, -d, xn[m], G[m], w[m], vt=vt[m], first_free=1, n_free=12, pin0=True)
        curves_R[r] = y
        resR[m] = GRID_R.at(y, xn[m] + d) - G[m]; resRl[m] = lockin_gr(y, GRID_R, -d, vt[m]) - G[m]
        print(f"  {RATIOS[r]:>6}: current layout d={d_int[r]:.2f}: mean-node rms {rms(resT[m]):.4f} max {mx(resT[m]):.3f}, lock-in rms {rms(resTl[m]):.4f} max {mx(resTl[m]):.3f}, at silence {GRID_T.at(curves_T[r], np.array([-d_int[r]]))[0]:.4f} | "
              f"rest-relative d={d_R[r]:.2f}: mean-node rms {rms(resR[m]):.4f} max {mx(resR[m]):.3f}, lock-in rms {rms(resRl[m]):.4f} max {mx(resRl[m]):.3f}, at silence {curves_R[r][0]:.4f}")
    for r in range(6):
        m = ri == r; idx = np.where(m)[0]; o = np.argsort(-np.abs(resR[m]))[:3]
        print(f"    rest-relative worst points {RATIOS[r]:>6}: " + ", ".join(f"t{ki[idx[j]] + 1}/{int(Lv[idx[j]])} x {x[idx[j]]:+.2f} node-rest {nodes[d_R[r]][0][idx[j]] + d_R[r]:.2f} ref {G[idx[j]]:.3f} err {resR[idx[j]]:+.3f}" for j in o))
    # what the knee resolution costs: the rest-relative layout on finer uniform grids (same PCHIP), mean node
    for step, npts in ((0.5, 192), (0.25, 384)):
        gf = Grid(0.0, step, npts); resF = np.empty(len(G))
        for r in range(6):
            m = ri == r; d = d_R[r]; xn, vt = nodes[d]
            _, ux, uy = iso_fit(xn[m], G[m], w[m], pin_at=-d)
            y = knots_to_grid(ux, uy, gf, -d); y[0] = 0.0; y = np.maximum.accumulate(y)
            y = refine_grid(y, gf, -d, xn[m], G[m], w[m], first_free=1, n_free=int(12 / step), pin0=True)
            resF[m] = gf.at(y, xn[m] + d) - G[m]
        report_resid(f"rest-relative layout, {step} dB grid ({npts} points), mean node", resF, x, G, ri)
    report_resid("current layout, integer soft depths, mean node", resT, x, G)
    report_resid("current layout, integer soft depths, lock-in", resTl, x, G)
    report_resid("rest-relative layout, mean node", resR, x, G)
    report_resid("rest-relative layout, lock-in", resRl, x, G)
    # the same evaluation for the on-disk constants (mirror), for a like-for-like number
    res0 = np.empty(len(G)); res0l = np.empty(len(G))
    for r in range(6):
        m = ri == r; res0[m] = GRID_T.at(CURVES0[r], xn0[m]) - G[m]; res0l[m] = lockin_gr(CURVES0[r], GRID_T, 0.0, vt0[m]) - G[m]
    report_resid("on-disk constants (mirror), mean node", res0, x, G, ri)
    report_resid("on-disk constants (mirror), lock-in", res0l, x, G, ri)
    print("  mirror standing GR at silence, on-disk constants (PCHIP at -depth):", np.round([GRID_T.at(CURVES0[r], np.array([-DEPTH0]))[0] for r in range(6)], 4).tolist())
    # the stage-3 report sets, from the mirror, for every variant
    def mirror_set(items, curves, grid, origin_of, nodes_of):
        e = []
        for r_, k, L in items:
            r = RATIOS.index(r_); i = np.where((ri == r) & (ki == k - 1) & (Lv == L))[0][0]
            xn, vt = nodes_of(r); e.append(lockin_gr(curves[r], grid, origin_of(r), vt[i:i + 1])[0] - G[i])
        e = np.array(e); return rms(e), mx(e)
    for tag, items in (("54", REPORT54), ("198", REPORT198)):
        a = mirror_set(items, CURVES0, GRID_T, lambda r: 0.0, lambda r: nodes[DEPTH0])
        b = mirror_set(items, curves_T, GRID_T, lambda r: 0.0, lambda r: nodes[d_int[r]])
        c = mirror_set(items, curves_R, GRID_R, lambda r: -d_R[r], lambda r: nodes[d_R[r]])
        print(f"  mirror report set {tag}: on-disk {a[0]:.3f}/{a[1]:.2f} | current layout + integer soft depths {b[0]:.3f}/{b[1]:.2f} | rest-relative layout {c[0]:.3f}/{c[1]:.2f}")

    # ---- C++ renders: on-disk, and the current-layout variant (per-ratio depth passed through the single field, one ratio per render)
    if "--no-render" not in ARGS:
        print("\n=== C++ engine renders")
        def cal_disk(r): return cal0
        def cal_new(r):
            r = RATIOS.index(r) if isinstance(r, str) else r
            c = cal0.copy(); c[MODEL.field("d_curve")] = curves_T.reshape(-1); c[MODEL.field("d_rel_depth_db")] = d_int[r]; return c
        render_silence(cal_disk, "on-disk"); render_silence(cal_new, "per-ratio depth, current layout")
        render_statics(cal_disk, REPORT54, "on-disk, 54"); render_statics(cal_new, REPORT54, "per-ratio depth, 54")
        render_statics(cal_disk, REPORT198, "on-disk, 198"); render_statics(cal_new, REPORT198, "per-ratio depth, 198")
        knee_items = [(r_, k, L) for r_ in RATIOS for k in (4, 12, 20) for L in LEVELS if -4.0 < L - T[k - 1] < 4.0]
        render_statics(cal_disk, knee_items, "on-disk, knee items"); render_statics(cal_new, knee_items, "per-ratio depth, knee items")
        for tag, cf in (("on-disk", cal_disk), ("per-ratio depth", cal_new)):
            e = [float(feat_residual(f"both_{l}", render_item(ITEMS[f"both_{l}"], cf(1)))[0]) for l in (-30, -20, -10, 0)]
            print(f"  both_* (2:1 + opto) [{tag}]: residuals {np.round(e, 3).tolist()}")
        e = [float(feat_residual(f"law_disc_gain_{g}", render_item(ITEMS[f"law_disc_gain_{g}"], cal_new(3)))[0]) for g in (1, 12, 24)]
        print(f"  law_disc_gain_1/12/24 [per-ratio depth]: {np.round(e, 4).tolist()}")

    # ---- dynamics at 4:1 (mirror): on-disk 4:1 curve; node-domain 4:1 curve on the current layout; on the rest-relative layout
    print("\n=== 4:1 dynamics (mirror, stage-3 residual definitions)")
    dynamics(lambda xx: GRID_T.at(CURVES0[3], xx), DEPTH0, "on-disk")
    dynamics(lambda xx: GRID_T.at(curves_T[3], xx), DEPTH0, "node-domain 4:1 curve, current layout")
    dynamics(lambda xx: GRID_R.at(curves_R[3], xx + d_R[3]), DEPTH0, "node-domain 4:1 curve, rest-relative layout")

    out = {"best_depth": {RATIOS[r]: best_d[r] for r in range(6)}, "integer_depth": {RATIOS[r]: d_int[r] for r in range(6)},
           "curves_current_layout": curves_T.tolist(), "curves_rest_relative": curves_R.tolist(), "depth_rest_relative": {RATIOS[r]: d_R[r] for r in range(6)}}
    p = os.path.join(HERE, "..", "..", "..", "build", "disc-knee-per-ratio.json")
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"))
    print(f"\ncurves and depths written to {os.path.abspath(p)}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
