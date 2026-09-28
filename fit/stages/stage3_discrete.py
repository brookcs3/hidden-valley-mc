# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Stage 3: the discrete stage.

3a. Static family. For every ratio position r, threshold k and level L, the measured gain reduction is GR(r, k, L). The shift family
    GR(r, k, L) = G_r(L - T'_k) (one shift per threshold position, one monotone curve per ratio) is fitted by alternating projections:
    isotonic (pool-adjacent-violators) curves on the shifted data, then the shift of each threshold position by a 1-D search. The
    residual is printed; if the shift model did not hold it would show there. T' is the threshold table directly (the node after the
    gain computer carries no detector offset).
3b. Detector and curves. A numba mirror of the C++ node (docs/disc-knee-fix.md: log rectifier -> gain computer through the monotone
    cubic curve -> one storage node in dB of gain reduction with an always-on bleed to zero and an attack diode whose conductance grows
    with the target gain reduction; DUAL second node) runs on the steady-tone matrix, every burst and tail envelope and the static
    family on a fine level grid. The six curves are refitted inside every evaluation by a fixed point (simulate the node on the fine
    grid at the capture setting, correct the table by the residual against the target family, re-monotonise), so the curve the
    engine runs is the curve the static grid measured through the node. The fifteen constants (attack, recover, the DUAL pair and the
    conductance law's scale) are fitted jointly with scipy.optimize.least_squares.
3c. Make-up interaction: the detector offset that comes with each DISCRETE GAIN position, read through the node's steady state.
3d. Gain-cell terms a2, a3 from the no-GR harmonic series, rendered through the whole model. a3 is bounded to be compressive.
The per-ratio attack tables (d_att_scale, d_att_a0, d_att_sat_db, docs/disc-ballistics.md) are carried at the identity: they were
fitted on a capture outside this protocol and under the earlier node, and are refitted only once their items are in the protocol.
"""
import os, sys, ctypes, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares, minimize_scalar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import F, ITEMS, MODEL, load_cal, save_cal, render_item, feat_residual, protocol, FS  # noqa: E402
import hvmc_core  # noqa: E402

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
_L = hvmc_core.lib()
_n = ctypes.c_int(); _x0 = ctypes.c_double(); _dx = ctypes.c_double()
_L.hvmc_curve_grid(ctypes.byref(_n), ctypes.byref(_x0), ctypes.byref(_dx))
N = int(_n.value); X0 = float(_x0.value); DX = float(_dx.value)   # the gain computer's grid, from the engine
XG = X0 + DX * np.arange(N)
FLOOR = -100.0
GAIN12 = 0.0   # make-up of position 12 plus the Nickel path gain, set in run()
FSF = float(FS)
LEVELS = list(range(-60, 13, 3))
YS = np.concatenate([np.arange(-20.0, -8.0, 1.0), np.arange(-8.0, 8.0, 0.25), np.arange(8.0, 76.0, 1.0)])   # the fine level grid


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def pchip_build(y):
    """the engine's monotone cubic (Fritsch-Carlson) tangents for one curve table"""
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


@njit(cache=True)
def node_gr(a, fs, ta, tr, dual, t2, c2, Tk, goff, sv, a0, sa, y, m):
    """DiscreteStage::process's detector: returns the gain reduction per sample. a: |sidechain|; (y, m) the ratio's curve table
    and tangents; ta already carries the ratio's attack scale."""
    n = a.shape[0]; g = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs)); k2 = (1.0 - np.exp(-1.0 / (t2 * fs))) if dual else 0.0
    fl = 10.0 ** (FLOOR / 20.0); inv_sa = 1.0 / sa if sa > 1e-3 else 0.0
    s = 0.0; w = 0.0
    for i in range(n):
        e = (20.0 * np.log10(a[i]) if a[i] > fl else FLOOR) + goff
        gt = pchip_at(y, m, e - Tk)
        if gt < 0.0: gt = 0.0
        dg = -s * kR
        if gt > s:
            f = a0 + gt / sv
            gap = gt - s
            gs = gap / (1.0 + gap * inv_sa) if inv_sa > 0.0 else gap
            dg += gs * (1.0 - np.exp(-rA * f))
        s += dg
        if dual:
            flow = (s - w) * k2; s -= flow; w += flow / c2
        if s < 1e-7: s = 0.0
        g[i] = s
    return g


@njit(parallel=True, cache=True)
def sine_grid(levels, f0s, tas, trs, Tks, ridx, C, S, fs, sv, a0s, sas, secs, last):
    """steady sines, one per entry: the lock-in gain of the fundamental in dB over the last `last` s (the protocol's gain_db)"""
    mm = levels.shape[0]; gain = np.empty(mm)
    for j in prange(mm):
        n = int(secs * fs); n0 = n - int(last * fs)
        amp = 10.0 ** (levels[j] / 20.0); wv = 2.0 * np.pi * f0s[j] / fs
        a = np.empty(n)
        for i in range(n): a[i] = abs(amp * np.sin(wv * i))
        r = ridx[j]
        g = node_gr(a, fs, tas[j], trs[j], False, 1.0, 1.0, Tks[j], 0.0, sv, a0s[j], sas[j], C[r], S[r])
        sr = 0.0; si = 0.0
        for i in range(n0, n):
            s = np.sin(wv * i); cs = np.cos(wv * i)
            yv = amp * s * (10.0 ** (-g[i] / 20.0))
            sr += yv * s; si += yv * cs
        gain[j] = 20.0 * np.log10(2.0 * np.sqrt(sr * sr + si * si) / (n - n0) / amp)
    return gain


@njit(cache=True)
def env_lockin(x, gr, f, fs):
    """the protocol's 'env' feature on y = x 10^(-gr/20): the gain of the fundamental per period by lock-in, dB"""
    per = fs / f; mm = int(x.shape[0] / per); out = np.empty(mm); w = 2.0 * np.pi * f / fs
    for k in range(mm):
        n0 = int(np.floor(k * per + 0.5)); n1 = int(np.floor((k + 1) * per + 0.5))
        cyr = 0.0; cyi = 0.0; cxr = 0.0; cxi = 0.0
        for i in range(n0, n1):
            c = np.cos(w * i); s = np.sin(w * i)
            yv = x[i] * 10.0 ** (-gr[i] / 20.0)
            cyr += yv * c; cyi -= yv * s; cxr += x[i] * c; cxi -= x[i] * s
        out[k] = 20.0 * np.log10(np.sqrt(cyr * cyr + cyi * cyi) / (np.sqrt(cxr * cxr + cxi * cxi) + 1e-30) + 1e-30)
    return out


def curve_at(c, x):
    """linear interpolation on the grid with linear extrapolation beyond it (the C++ uses a monotone cubic; close enough for fitting)"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.interp(x, XG, c)
    hi = x > XG[-1]
    if np.any(hi):
        y[hi] = c[-1] + (c[-1] - c[-2]) / DX * (x[hi] - XG[-1])
    return y

def pava(x, y, w):
    """isotonic (non-decreasing) regression of y against x with weights w; returns fitted values in the order of x"""
    order = np.argsort(x); ys = y[order]; ws = w[order]
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

def gr_static(r, k, L):
    return GAIN12 - F[f"disc_static_{r}_t{k}_{L}"]

def fit_static():
    """3a: returns T' (24) and curves (6, N)"""
    levels = list(range(-60, 13, 3))
    data = []
    for ri, r in enumerate(RATIOS):
        for k in range(1, 25):
            for L in levels:
                data.append((ri, k - 1, float(L), gr_static(r, k, L)))
    data = np.array(data)
    ri, ki, Lv, G = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2], data[:, 3]
    W = np.where(G > 0.3, 1.0, 0.3)
    Tp = 2.8 - 2.69 * np.arange(24)
    curves = np.zeros((6, N))
    for it in range(6):
        x = Lv - Tp[ki]
        for r in range(6):
            m = ri == r
            yfit = pava(x[m], G[m], W[m])
            xs = x[m]; order = np.argsort(xs)
            xs, yf = xs[order], yfit[order]
            c = np.interp(XG, xs, yf)
            c[XG < xs.min()] = yf[0]
            beyond = XG > xs.max()
            if np.any(beyond):
                tail = max(0, len(yf) - 40)
                slope = (yf[-1] - yf[tail]) / max(1e-6, xs[-1] - xs[tail])
                c[beyond] = yf[-1] + slope * (XG[beyond] - xs.max())
            curves[r] = np.maximum.accumulate(np.maximum(c, 0.0))
        for k in range(24):
            m = ki == k
            def cost(s):
                tot = 0.0
                for r in range(6):
                    mm = m & (ri == r)
                    tot += np.sum(W[mm] * (curve_at(curves[r], Lv[mm] - s) - G[mm]) ** 2)
                return tot
            res = minimize_scalar(cost, bounds=(Tp[k] - 4.0, Tp[k] + 4.0), method="bounded", options={"xatol": 1e-3})
            Tp[k] = res.x
        pred = np.array([curve_at(curves[ri[i]], Lv[i] - Tp[ki[i]])[0] for i in range(len(G))])
        resid = pred - G
        sel = G > 0.3
        print(f"  static pass {it}: rms {np.sqrt(np.mean(resid[sel] ** 2)):.3f} dB, max {np.max(np.abs(resid[sel])):.2f} dB over {int(sel.sum())} compressing points")
    print("  threshold steps (dB):", np.round(-np.diff(Tp), 2))
    return Tp, curves


# ------------------------------------------------------------------------------------------------ the target family and the curves
class Static:
    """the target family G_r(y), y = L - T'_k: the pooled isotonic regression per ratio (the floor of any curve-of-level model)"""
    def __init__(self, Tp):
        data = []
        for ri, r in enumerate(RATIOS):
            for k in range(1, 25):
                for L in LEVELS:
                    data.append((ri, k - 1, float(L), gr_static(r, k, L)))
        d = np.array(data)
        self.ri, self.ki, self.L, self.G = d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2], d[:, 3]
        y = self.L - Tp[self.ki]
        fit = np.empty_like(self.G); self.steps = []
        for r in range(6):
            m = self.ri == r
            xr = np.round(y[m], 6); ux, inv = np.unique(xr, return_inverse=True)
            cnt = np.bincount(inv).astype(float); sy = np.bincount(inv, weights=self.G[m])
            fu = np.maximum(pava(ux, sy / cnt, cnt), 0.0)
            self.steps.append((ux, fu)); fit[m] = fu[inv]
        self.floor = fit - self.G
        self.Gt = np.stack([np.interp(YS, ux, fu, left=0.0) for ux, fu in self.steps])   # (6, len(YS))


def rms_max(e):
    e = np.asarray(e); return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


class Det:
    """the detector constants and the per-ratio tables in one place"""
    def __init__(self, cal):
        self.ta = np.array(cal[MODEL.field("d_tatt")], dtype=float); self.tr = np.array(cal[MODEL.field("d_trel")], dtype=float)
        self.t2 = float(cal[MODEL.field("d_dual_t2")][0]); self.c2 = float(cal[MODEL.field("d_dual_c2")][0])
        self.sv = float(cal[MODEL.field("d_att_sv_db")][0])
        self.att = np.array(cal[MODEL.field("d_att_scale")], dtype=float); self.a0 = np.array(cal[MODEL.field("d_att_a0")], dtype=float)
        self.sa = np.array(cal[MODEL.field("d_att_sat_db")], dtype=float)

    def pack(self): return np.array(list(self.ta) + list(self.tr) + [self.t2, self.c2, np.log10(self.sv)])
    def unpack(self, p):
        d = Det.__new__(Det); d.__dict__.update(self.__dict__)
        d.ta = np.array(p[0:6]); d.tr = np.array(p[6:12]); d.t2 = float(p[12]); d.c2 = float(p[13]); d.sv = float(10.0 ** p[14])
        return d


def fit_curves_gr(st, det, T16, C0=None, iters=5, damp=0.8, verbose=False):
    """the curves on the engine's grid by fixed point: simulate the node on YS (relative to T16) at the capture setting (attack 1 ms,
    recover 0.5 s), correct the table by the residual against the target family, re-monotonise, repeat. Returns C, S, residual."""
    ny = len(YS)
    C = np.stack([np.maximum.accumulate(np.maximum(np.interp(XG, YS, st.Gt[r]), 0.0)) for r in range(6)]) if C0 is None else C0.copy()
    lv = np.tile(YS + T16, 6); ri = np.repeat(np.arange(6), ny).astype(np.int64); mm = len(lv)
    tas = np.array([det.ta[2] * det.att[r] for r in ri]); a0s = np.array([det.a0[r] for r in ri]); sas = np.array([det.sa[r] for r in ri])
    res = None
    for it in range(iters):
        S = np.stack([pchip_build(C[r]) for r in range(6)])
        gain = sine_grid(lv, np.full(mm, 1000.0), tas, np.full(mm, det.tr[2]), np.full(mm, T16), ri, C, S, FSF, det.sv, a0s, sas, 2.5, 0.5)
        res = st.Gt - (-gain).reshape(6, ny)
        if verbose: print(f"      curve fixed point {it}: residual on the fine grid rms {rms_max(res)[0]:.4f} max {rms_max(res)[1]:.3f}")
        if it == iters - 1: break
        for r in range(6):
            dC = np.interp(XG, YS, res[r], left=0.0, right=res[r][-1])
            C[r] = np.maximum.accumulate(np.maximum(C[r] + damp * dC, 0.0))
    # pin the grid to exactly zero below each ratio's onset: the node would stand on a fixed-point residue in silence otherwise
    for r in range(6):
        on = np.argmax(st.Gt[r] > 0.03) if (st.Gt[r] > 0.03).any() else len(YS) - 1
        C[r][XG < YS[on]] = 0.0
        C[r] = np.maximum.accumulate(C[r])
    S = np.stack([pchip_build(C[r]) for r in range(6)])
    return C, S, res


# ------------------------------------------------------------------------------------------------ the dynamic items
BURST_IDS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS] + [f"disc_dual_blen_{b}" for b in (0.1, 0.5, 2.0, 8.0)] + ["disc_dual_pulses"] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
TAIL_IDS = [i for i in ITEMS if i.startswith("disc_tail_")]
DYN_IDS = BURST_IDS + TAIL_IDS
XSIG = {iid: protocol.stimulus(ITEMS[iid]["stim"], ITEMS[iid]["fs"])[0] for iid in DYN_IDS}
STIM = {iid: np.abs(XSIG[iid]) for iid in DYN_IDS}


def gr_item(iid, C, S, T, det):
    it = ITEMS[iid]; s = it["set"]; fs = float(it["fs"])
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    Tk = T[int(s["discrete_threshold"]) - 1]
    ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
    r = RATIOS.index(s.get("discrete_ratio", "4:1"))
    return node_gr(STIM[iid], fs, det.ta[ai] * det.att[r], det.tr[rci], dual, det.t2, det.c2, Tk, 0.0, det.sv, det.a0[r], det.sa[r], C[r], S[r])


def env_of_item(iid, C, S, T, det):
    return env_lockin(XSIG[iid], gr_item(iid, C, S, T, det), float(ITEMS[iid]["stim"]["f"]), float(ITEMS[iid]["fs"])) + GAIN12


def steady_cells(C, S, T, det):
    """the 54 steady cells: attack x recover at -10 dBFS, the level checks, the frequency checks (all 4:1, threshold 16)"""
    lv, f0, tas, trs, refs = [], [], [], [], []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            lv.append(-10.0); f0.append(1000.0); tas.append(det.ta[ai]); trs.append(det.tr[rci]); refs.append(F[f"disc_ar_{a}_{rc}"])
        for l in (-25, 2):
            lv.append(float(l)); f0.append(1000.0); tas.append(det.ta[ai]); trs.append(det.tr[2]); refs.append(F[f"disc_al_{a}_{l}"])
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            lv.append(-10.0); f0.append(f); tas.append(det.ta[ATTACKS.index(a)]); trs.append(det.tr[2]); refs.append(F[f"disc_af_{a}_f{int(f)}"])
    mm = len(lv)
    gain = sine_grid(np.array(lv), np.array(f0), np.array(tas) * det.att[3], np.array(trs), np.full(mm, T[15]), np.full(mm, 3, dtype=np.int64), C, S, FSF, det.sv, np.full(mm, det.a0[3]), np.full(mm, det.sa[3]), 3.0, 0.5)
    return gain + GAIN12 - np.array(refs)


def dyn_resid(C, S, T, det):
    out = [3.0 * steady_cells(C, S, T, det)]
    for iid in DYN_IDS:
        ref = np.asarray(F[iid]); e = env_of_item(iid, C, S, T, det); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate(out)


def split_dyn(res):
    k = 54; groups = {"bursts": [], "DUAL": [], "tails": []}
    for iid in DYN_IDS:
        n = min(len(F[iid]), int(len(STIM[iid]) / (ITEMS[iid]["fs"] / ITEMS[iid]["stim"]["f"])))
        seg = res[k:k + n]; k += n
        groups["tails" if iid.startswith("disc_tail_") else ("DUAL" if "dual" in iid else "bursts")].append(seg)
    return {g: rms_max(np.concatenate(v)) for g, v in groups.items() if v}


def run():
    global GAIN12
    cal = load_cal()
    gain = cal[MODEL.field("d_gain_db")]
    xg = float(cal[MODEL.fields["x_gain_db"][0]])
    GAIN12 = float(gain[11] + xg)
    # ---- 3a static family
    Tp, _ = fit_static()
    st = Static(Tp)
    print(f"  target family (pooled isotonic per ratio against L - T'): floor rms {rms_max(st.floor)[0]:.4f} max {rms_max(st.floor)[1]:.3f} over 3600 points")
    T = Tp.copy()   # no detector offset: the threshold table is stage 3a's
    # ---- 3b the joint fit, curves refitted inside every evaluation
    det = Det(cal)
    C, S, sres = fit_curves_gr(st, det, T[15], iters=6, verbose=True)
    lo = [1e-6] * 6 + [0.01] * 6 + [0.005, 1.5, 0.3]; hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 3.0]
    xs = [1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 0.3]
    cache = {"C": C.copy()}
    wst = np.sqrt(3600.0 / (6 * len(YS)))
    def resid(p):
        d = det.unpack(p)
        Cc, Sc, sr = fit_curves_gr(st, d, T[15], C0=cache["C"], iters=2)
        cache["C"] = Cc
        return np.concatenate([dyn_resid(Cc, Sc, T, d), wst * sr.reshape(-1)])
    r = least_squares(resid, np.clip(det.pack(), lo, hi), bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=120, loss="soft_l1", f_scale=1.0)
    det = det.unpack(r.x)
    C, S, sres = fit_curves_gr(st, det, T[15], C0=cache["C"], iters=6, verbose=True)
    dres = dyn_resid(C, S, T, det); parts = split_dyn(dres[54:])
    print(f"  detector fit: nfev {r.nfev} cost {r.cost:.3f} | attack {np.round(det.ta * 1e3, 3)} ms, recover {np.round(det.tr, 4)} s, dual t2 {det.t2:.4f} c2 {det.c2:.2f}, Sv {det.sv:.2f} dB")
    print(f"    steady-state rms {np.sqrt(np.mean((dres[:54] / 3.0) ** 2)):.3f} dB (max {np.max(np.abs(dres[:54] / 3.0)):.2f}) | " + " | ".join(f"{k} {v[0]:.3f}/{v[1]:.2f}" for k, v in parts.items()) + f" | static fixed point rms {rms_max(sres)[0]:.4f}")
    # ---- 3c make-up interaction, through the node's steady state at 4:1
    ny = len(YS); lv = YS + T[15]
    P41 = -sine_grid(lv, np.full(ny, 1000.0), np.full(ny, det.ta[2] * det.att[3]), np.full(ny, det.tr[2]), np.full(ny, T[15]), np.full(ny, 3, dtype=np.int64), C, S, FSF, det.sv, np.full(ny, det.a0[3]), np.full(ny, det.sa[3]), 2.5, 0.5)
    goff = np.zeros(24)
    for g_ in range(1, 25):
        grm = float(gain[g_ - 1] + xg - F[f"disc_gi_{g_}"])
        x_needed = float(np.interp(grm, P41, YS))
        goff[g_ - 1] = x_needed - (-10.0 - T[15])
    goff -= goff[11]   # the static grid was captured at position 12
    print("  make-up detector offsets (dB):", np.round(goff, 2))
    # ---- write
    cal[MODEL.field("d_thr_db")] = T
    cal[MODEL.field("d_curve")] = C.reshape(-1)
    cal[MODEL.field("d_tatt")] = det.ta; cal[MODEL.field("d_trel")] = det.tr
    cal[MODEL.field("d_floor_db")] = FLOOR; cal[MODEL.field("d_dual_t2")] = det.t2; cal[MODEL.field("d_dual_c2")] = det.c2
    cal[MODEL.field("d_att_sv_db")] = det.sv
    cal[MODEL.field("d_goff_db")] = goff
    # ---- 3d gain-cell terms
    ids = [f"disc_harm_nogr_{l}" for l in (-30, -20, -10, 0, 6)]
    def rh(p):
        c = cal.copy(); c[MODEL.field("d_a2")] = p[0]; c[MODEL.field("d_a3")] = p[1]
        return np.concatenate([feat_residual(i, render_item(ITEMS[i], c))[:3] for i in ids])
    r2 = least_squares(rh, [2.9e-3, -1.4e-3], bounds=([-0.05, -0.05], [0.05, 0.0]), x_scale=[1e-4, 1e-4], diff_step=1e-3)   # a3 <= 0: a gain cell compresses
    cal[MODEL.field("d_a2")] = r2.x[0]; cal[MODEL.field("d_a3")] = r2.x[1]
    print(f"  gain cell: a2 {r2.x[0]:.4e}, a3 {r2.x[1]:.4e}, rms {np.sqrt(np.mean(r2.fun ** 2)):.2f} dB")
    if os.environ.get("HVMC_NO_SAVE"):
        print("  (HVMC_NO_SAVE set: constants not written)"); return
    save_cal(cal, {"stage3": "discrete: static family (disc_static_*), the node after the gain computer fitted with the curves through it (disc_ar_*, disc_al_*, disc_af_*, disc_burst_*, disc_dual_*, disc_bdepth_*, disc_tail_*), make-up offsets (disc_gi_*), cell terms (disc_harm_nogr_*); docs/disc-knee-fix.md"},
             fields=["d_thr_db", "d_curve", "d_tatt", "d_trel", "d_floor_db", "d_att_sv_db", "d_dual_t2", "d_dual_c2", "d_goff_db", "d_a2", "d_a3"])
    # ---- report: full renders through the engine
    errs = []
    for r_ in RATIOS:
        for k in (4, 12, 20):
            for L in (-30, -15, 0):
                iid = f"disc_static_{r_}_t{k}_{L}"
                errs.append(feat_residual(iid, render_item(ITEMS[iid], cal))[0])
    errs = np.array(errs)
    print(f"  rendered static check (54 items): rms {np.sqrt(np.mean(errs ** 2)):.3f} dB, max {np.max(np.abs(errs)):.2f} dB")
    knee = []
    for r_ in RATIOS:
        for k in (4, 12, 20):
            for L in LEVELS:
                if -4.0 < L - T[k - 1] < 4.0:
                    iid = f"disc_static_{r_}_t{k}_{L}"; knee.append(feat_residual(iid, render_item(ITEMS[iid], cal))[0])
    knee = np.array(knee)
    print(f"  rendered knee band (-4 < x < 4, {len(knee)} items): rms {np.sqrt(np.mean(knee ** 2)):.3f} dB, max {np.max(np.abs(knee)):.2f} dB")
    sil = []
    for r_ in RATIOS:
        it = dict(ITEMS[f"disc_static_{r_}_t1_-60"]); it = {**it, "stim": {**it["stim"], "level": -70.0}}
        sil.append(render_item(it, cal) - (GAIN12))
    print("  standing gain at -70 dBFS relative to the make-up, per ratio (dB):", np.round(sil, 4))
    for iid in ("disc_burst_1.0_0.5 s", "disc_burst_30.0_0.1 s", "disc_burst_1.0_Dual", "disc_tail_2:1_0.5 s"):
        if iid not in ITEMS: continue
        e = feat_residual(iid, render_item(ITEMS[iid], cal))
        print(f"  rendered {iid}: rms {np.sqrt(np.mean(e ** 2)):.3f} dB, max {np.max(np.abs(e)):.2f} dB")

if __name__ == "__main__":
    run()
