# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Stage 3: the discrete stage.

3a. Static family. For every ratio position r, threshold k and level L, the measured gain reduction is GR(r, k, L). The model says
    GR = C_r(v - T_k) with v the detector node, which on a steady sine at the capture's attack/recover setting sits at L + d0. So
    GR(r, k, L) = C_r(L - T'_k) with T'_k = T_k - d0: one shift per threshold position and one curve per ratio. Fitted by alternating
    projections: monotone (pool-adjacent-violators) curves on the shifted data, then the shift of each threshold position by a 1-D
    search. The residual is printed; if the shift model did not hold it would show there.
3b. Detector. A numba mirror of the C++ detector (log rectifier, always-on bleed toward the threshold reference, level-dependent
    attack conductance, DUAL second node; docs/detector-fix.md) runs on the
    steady-tone matrix and the burst stimuli; the gain trajectory C_r(v(t) - T) is compared with the reference, and the attack and
    recover time constants, the floor and the DUAL network are fitted. d0 is recomputed from the fitted detector, and T = T' + d0.
3c. Make-up interaction: the detector offset that comes with each DISCRETE GAIN position, from the measured GR at each position.
3d. Gain-cell terms a2, a3 from the no-GR harmonic series, rendered through the whole model (so the Nickel path is accounted for).
"""
import os, sys, numpy as np
from numba import njit
from scipy.optimize import least_squares, minimize_scalar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import F, ITEMS, MODEL, load_cal, save_cal, render_item, feat_residual, protocol, FS  # noqa: E402

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
N = 96; X0 = -20.0; DX = 1.0
XG = X0 + DX * np.arange(N)
GAIN12 = 0.0   # make-up of position 12 plus the Nickel path gain, set in run()


# ------------------------------------------------------------------------------------------------ detector mirror
@njit(cache=True)
def detector(a, fs, ta, tr, floor_db, dual, t2, c2, goff, Tk, depth, sv):
    """a: |sidechain| samples. Returns the node v (dB) per sample. Same equations as DiscreteStage::process: the bleed (release)
    always conducting toward Tk - depth, the attack diode conducting toward the rectified log level with a conductance that grows
    with the level above the reference (1 + (e - rest) / sv)."""
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


def env_of_item(item_id, Tp, curves, ta, tr, floor, t2, c2, d0, depth, sv):
    """gain trajectory of the model on a burst item, from the detector mirror (no render); dB per period, like the 'env' feature"""
    it = ITEMS[item_id]; st = it["stim"]; fs = it["fs"]
    x = protocol.stimulus(st, fs)[0]
    s = it["set"]
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    T = Tp[int(s["discrete_threshold"]) - 1] + d0
    v = detector(np.abs(x), float(fs), ta, tr, floor, dual, t2, c2, 0.0, T, depth, sv)
    r = RATIOS.index(s.get("discrete_ratio", "4:1"))
    gr = np.maximum(curve_at(curves[r], v - T), 0.0)
    per = int(round(fs / st["f"]))
    m = len(gr) // per
    lin = (10 ** (-gr[:m * per] / 20.0)).reshape(m, per).mean(axis=1)
    return 20 * np.log10(lin) + GAIN12


def d0_for(ta, tr, floor, t2, c2, fs=FS, level=-20.0, secs=2.5, f=1000.0, Tk=-37.5, depth=0.0, sv=17.4):
    """detector offset on a steady sine: v_ss - peak level (it depends weakly on the level above the threshold reference)"""
    t = np.arange(int(secs * fs)) / fs
    a = np.abs(10 ** (level / 20) * np.sin(2 * np.pi * f * t))
    v = detector(a, float(fs), ta, tr, floor, False, t2, c2, 0.0, Tk, depth, sv)
    return float(v[-int(0.5 * fs):].mean() - level)


def run():
    global GAIN12
    cal = load_cal()
    gain = cal[MODEL.field("d_gain_db")]
    xg = float(cal[MODEL.fields["x_gain_db"][0]])
    GAIN12 = float(gain[11] + xg)
    # ---- 3a static
    Tp, curves = fit_static()
    # ---- 3b detector: one joint fit of the time constants, the DUAL network and the discharge depth over the steady-state table
    # (attack x recover, plus the level and frequency checks) and every burst envelope. The threshold enters the detector (the
    # release discharges toward it), so d0 is evaluated at the threshold of each item.
    floor = -100.0
    t2 = float(cal[MODEL.field("d_dual_t2")][0]); c2 = float(cal[MODEL.field("d_dual_c2")][0])
    ta = list(cal[MODEL.field("d_tatt")]); tr = list(cal[MODEL.field("d_trel")]); depth = 0.0
    burst_ids = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS] + [f"disc_dual_blen_{b}" for b in (0.1, 0.5, 2.0, 8.0)] + ["disc_dual_pulses"] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
    def unpack(p):
        return list(p[0:6]), list(p[6:12]), float(p[12]), float(p[13]), float(p[14]), float(10.0 ** p[15])
    def resid_all(p):
        tta, ttr, tt2, tc2, dep, sv = unpack(p)
        T16 = Tp[15]
        d_cap = d0_for(tta[2], ttr[2], floor, tt2, tc2, Tk=T16, depth=dep, sv=sv)   # the capture setting (attack 1 ms, recover 0.5 s)
        T = T16 + d_cap
        out = []
        for ai, a in enumerate(ATTACKS):
            for rci, rc in enumerate(RECOVERS[:5]):
                d = d0_for(tta[ai], ttr[rci], floor, tt2, tc2, level=-10.0, secs=3.0, Tk=T, depth=dep, sv=sv)
                out.append(3.0 * (float(curve_at(curves[3], -10.0 + d - T)[0]) - (GAIN12 - F[f"disc_ar_{a}_{rc}"])))
            for lvl in (-25, 2):
                d = d0_for(tta[ai], ttr[2], floor, tt2, tc2, level=float(lvl), secs=3.0, Tk=T, depth=dep, sv=sv)
                out.append(3.0 * (float(curve_at(curves[3], lvl + d - T)[0]) - (GAIN12 - F[f"disc_al_{a}_{lvl}"])))
        for a in (0.1, 1.0, 30.0):
            for f in (100.0, 5000.0):
                d = d0_for(tta[ATTACKS.index(a)], ttr[2], floor, tt2, tc2, level=-10.0, secs=3.0, f=f, Tk=T, depth=dep, sv=sv)
                out.append(3.0 * (float(curve_at(curves[3], -10.0 + d - T)[0]) - (GAIN12 - F[f"disc_af_{a}_f{int(f)}"])))
        for iid in burst_ids:
            ref = np.asarray(F[iid]); s_ = ITEMS[iid]["set"]
            ai = ATTACKS.index(float(s_["discrete_attack"])); rci = RECOVERS.index(s_["discrete_recover"])
            e = env_of_item(iid, Tp, curves, tta[ai], ttr[rci], floor, tt2, tc2, d_cap, dep, sv)
            n = min(len(e), len(ref))
            w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
            out.append(w * (e[:n] - ref[:n]))
        return np.concatenate([np.atleast_1d(o) for o in out])
    p0 = [0.035e-3, 0.42e-3, 1.76e-3, 6.4e-3, 19.3e-3, 58.0e-3] + [0.095, 0.136, 0.334, 0.334, 0.489, tr[5]] + [t2, c2, 0.55, np.log10(17.4)]
    lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, float(os.environ.get("HVMC_DEPTH_MIN", -20.0)), 0.3]; hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 40.0, 3.0]
    r = least_squares(resid_all, np.clip(p0, lo, hi), bounds=(lo, hi), x_scale=[1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 1.0, 0.3], diff_step=1e-3, max_nfev=120, loss="soft_l1", f_scale=1.0)
    ta, tr, t2, c2, depth, sv = unpack(r.x)
    n_ss = 36 + 12 + 6
    print(f"  detector fit: attack {np.round(np.array(ta) * 1e3, 3)} ms, recover {np.round(tr, 4)} s, dual t2 {t2:.4f} c2 {c2:.2f}, depth {depth:.2f} dB, Sv {sv:.2f} dB")
    d_cap = d0_for(ta[2], tr[2], floor, t2, c2, Tk=Tp[15], depth=depth, sv=sv)
    dual_res = []
    for iid in burst_ids:
        if "dual" not in iid: continue
        ref = np.asarray(F[iid]); s_ = ITEMS[iid]["set"]
        e = env_of_item(iid, Tp, curves, ta[ATTACKS.index(float(s_["discrete_attack"]))], tr[RECOVERS.index(s_["discrete_recover"])], floor, t2, c2, d_cap, depth, sv)
        n = min(len(e), len(ref)); w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        dual_res.append(w * (e[:n] - ref[:n]))
    dual_res = np.concatenate(dual_res)
    print(f"    DUAL items (blen, pulses): weighted rms {np.sqrt(np.mean(dual_res ** 2)):.3f}, max {np.max(np.abs(dual_res)):.2f}")
    dd = d0_for(ta[2], tr[2], floor, t2, c2, Tk=Tp[15], depth=depth, sv=sv)
    print(f"  detector offset d0 at the capture setting: {dd:.3f} dB")
    T = Tp + dd
    # the offset varies slowly with the level above the release reference; resample the curves onto the node domain so that the
    # static family stays exact: C_node(x + delta(x) - d0) = C_ref(x)
    # x = L - T + dd is the static family's argument, so the sine level that produces it is L = x + T - dd. Below the release
    # reference the node cannot follow the level down (only the bleed acts), so every such level maps onto the rest point; the
    # curve is pinned to zero there, which is what the reference does at low levels (its soft onset in that last dB is the one
    # thing the pinned node cannot reproduce, and the printed figure below is that error).
    xs = np.arange(-25.0, 80.0, 0.5)
    dx = np.array([d0_for(ta[2], tr[2], floor, t2, c2, level=x + T[15] - dd, Tk=T[15], depth=depth, sv=sv) - dd for x in xs])
    xnode = xs + dx
    above = xnode > -depth + 1e-6
    print(f"  node-domain offset over the working range (x > -depth): {dx[above].min():+.2f} .. {dx[above].max():+.2f} dB")
    print("  reference gain reduction at the rest level, per ratio (dB, the knee error the pinned node accepts):",
          np.round([float(curve_at(curves[r_], -depth + dd)[0]) for r_ in range(6)], 3))
    for r_ in range(6):
        cref = curve_at(curves[r_], xs)
        cn = np.interp(XG, xnode[above], cref[above], left=0.0)
        cn[XG <= -depth] = 0.0
        curves[r_] = np.maximum.accumulate(np.maximum(cn, 0.0))
    print("  standing gain reduction at the rest point after pinning, per ratio (dB):", np.round([float(curve_at(curves[r_], -depth)[0]) for r_ in range(6)], 3))
    # ---- 3c make-up interaction
    goff = np.zeros(24)
    c4 = curves[3]
    for g in range(1, 25):
        grm = float(gain[g - 1] + xg - F[f"disc_gi_{g}"])
        x_needed = float(np.interp(grm, c4, XG))
        goff[g - 1] = x_needed - (-10.0 + dd - T[15])
    goff -= goff[11]   # the static grid was captured at position 12
    print("  make-up detector offsets (dB):", np.round(goff, 2))
    # ---- write
    cal[MODEL.field("d_thr_db")] = T
    cal[MODEL.field("d_curve")] = curves.reshape(-1)
    cal[MODEL.field("d_tatt")] = ta; cal[MODEL.field("d_trel")] = tr
    cal[MODEL.field("d_floor_db")] = floor; cal[MODEL.field("d_dual_t2")] = t2; cal[MODEL.field("d_dual_c2")] = c2
    cal[MODEL.field("d_rel_depth_db")] = depth; cal[MODEL.field("d_att_sv_db")] = sv
    cal[MODEL.field("d_goff_db")] = goff
    # ---- 3d gain-cell terms
    ids = [f"disc_harm_nogr_{l}" for l in (-30, -20, -10, 0, 6)]
    def rh(p):
        c = cal.copy(); c[MODEL.field("d_a2")] = p[0]; c[MODEL.field("d_a3")] = p[1]
        return np.concatenate([feat_residual(i, render_item(ITEMS[i], c))[:3] for i in ids])
    r = least_squares(rh, [2.9e-3, 1.4e-3], bounds=([-0.05, -0.05], [0.05, 0.05]), x_scale=[1e-4, 1e-4], diff_step=1e-3)
    cal[MODEL.field("d_a2")] = r.x[0]; cal[MODEL.field("d_a3")] = r.x[1]
    print(f"  gain cell: a2 {r.x[0]:.4e}, a3 {r.x[1]:.4e}, rms {np.sqrt(np.mean(r.fun ** 2)):.2f} dB")
    if os.environ.get("HVMC_NO_SAVE"):
        print("  (HVMC_NO_SAVE set: constants not written)"); return
    save_cal(cal, {"stage3": "discrete: static family (disc_static_*), detector (disc_ar_*, disc_al_*, disc_burst_*, disc_dual_*), make-up offsets (disc_gi_*), cell terms (disc_harm_nogr_*)"},
             fields=["d_thr_db", "d_curve", "d_tatt", "d_trel", "d_floor_db", "d_rel_depth_db", "d_att_sv_db", "d_dual_t2", "d_dual_c2", "d_goff_db", "d_a2", "d_a3"])
    # ---- report: full renders on a sample of items
    errs = []
    for r_ in RATIOS:
        for k in (4, 12, 20):
            for L in (-30, -15, 0):
                iid = f"disc_static_{r_}_t{k}_{L}"
                errs.append(feat_residual(iid, render_item(ITEMS[iid], cal))[0])
    errs = np.array(errs)
    print(f"  rendered static check (54 items): rms {np.sqrt(np.mean(errs ** 2)):.3f} dB, max {np.max(np.abs(errs)):.2f} dB")
    for iid in ("disc_burst_1.0_0.5 s", "disc_burst_30.0_0.1 s", "disc_burst_1.0_Dual"):
        e = feat_residual(iid, render_item(ITEMS[iid], cal))
        print(f"  rendered {iid}: rms {np.sqrt(np.mean(e ** 2)):.3f} dB, max {np.max(np.abs(e)):.2f} dB")

if __name__ == "__main__":
    run()
