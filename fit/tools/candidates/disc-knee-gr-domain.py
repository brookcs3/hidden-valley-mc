# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage, soft-ratio knee: the storage node AFTER the gain computer (the timing runs in the gain-reduction domain).
Synthesis check for docs/disc-knee-fix.md; the three candidate harnesses (disc-knee-feedthrough.py, disc-knee-per-ratio.py,
disc-knee-second-bleed.py) all keep the node in the level domain and read the ratio curve off it.

WHY THIS FORM. The reference's post-burst tail (disc-knee-second-bleed.py --reference, R2: -50 -> -10 dBFS for 2 s -> -50, threshold
16, attack 1 ms) has the same gain-reduction FRACTION at every ratio position: GR(t) / GR(in the burst) at 0.25 / 0.5 / 1 s is
0.471 / 0.202 / 0.039 at 1.2:1, 0.476 / 0.205 / 0.039 at 2:1 and 0.478 / 0.206 / 0.039 at 4:1 (0.5 s recover), although the three
static curves have very different shapes (1.2:1 runs from 0.27 to 0.6 dB per dB, 4:1 from 0.94 to 1.0). A node in the level domain
released through a progressive curve cannot do that: the per-ratio-rest form (mode 6 of the second-bleed harness) gives 0.366 / 0.127
/ 0.019 at 1.2:1, i.e. 1.8 / 1.3 / 0.34 dB low. A node that stores the gain reduction itself releases every ratio on one trajectory
by construction, rests at zero gain reduction (exact silence at every ratio with no pin, no snap and no depth), starts compressing
at each ratio's own static onset (the curve's own zero, 1.9 dB below the 4:1 onset at 1.2:1 and 2:1), holds a tone below the 4:1
onset at its from-silence value after a burst (R4), and leaves the 4:1 node floored below its onset (R3). At 4:1 it is the current
detector up to the curve's local slope (0.94 .. 1.0), which is why the 4:1-only fits of docs/detector-fix.md could not tell the two
orders apart.

THE MODEL (per sample; e the log-rectified sidechain level, dB):
    gt = C_r(e - T)                      instantaneous target gain reduction from the ratio curve (never below 0)
    dg = -g kR                           bleed toward zero gain reduction, always on (kR from the recover position)
    if gt > g: f = 1 + max(e - (T - dref), 0) / Sv     (law L: the diode's conductance grows with the level above a reference)
                    or 1 + gt / Sv                        (law G: it grows with the target gain reduction itself)
               dg += (gt - g) (1 - exp(-f / (ta fs)))
    g += dg;  DUAL: flow = (g - w) k2; g -= flow; w += flow / c2
    gain reduction = g
The curves are fitted in the model's own domain: the target static family (pooled isotonic regression of the 3600 disc_static
points against L - T'_k, T' from stage 3a) is matched by a fixed-point iteration that simulates the node on a fine level grid and
corrects the 1 dB PCHIP table, so the curve the report evaluates is the curve the engine would run.

REPORTED, against the current constants through the same mirror (level node, on-disk PCHIP curves, the engine's hard zero at rest):
rendered statics (54-item report set, 198-item set, full 3600 grid, knee band), standing GR at silence per ratio, the stage-3b
steady table, bursts and DUAL items (lock-in envelopes, stage-3 weights), and the reference probes R1..R4 of the second-bleed harness
(read from build/disc-knee-second-bleed-reference.json). Pass A: detector constants as on disk. Pass B (--fit): the sixteen stage-3b
constants refitted jointly (curves refitted inside every evaluation), the reference probes kept OUT of the residual as the cross-check.

usage: cd <repo> && python3 -u fit/tools/candidates/disc-knee-gr-domain.py [--law L|G|both] [--fit] [--nfev 12] [--no-cpp] [--wstatic 1.0]
  (numba caches need a real file: never pipe this through stdin)"""
import os, sys, time, json, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol, FS  # noqa: E402
import stage3_discrete as S3  # noqa: E402  (fit_static, pava, the engine's curve grid; nothing here writes constants)

RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
N = S3.N; X0 = S3.X0; DX = S3.DX; XG = S3.XG
FLOOR = -100.0
cal = load_cal()
GAIN = cal[MODEL.field("d_gain_db")]; XGN = float(cal[MODEL.fields["x_gain_db"][0]]); GAIN12 = float(GAIN[11] + XGN)
S3.GAIN12 = GAIN12
LEVELS = list(range(-60, 13, 3))
OUT = os.path.join(HERE, "..", "..", "..", "build")
REF_JSON = os.path.join(OUT, "disc-knee-second-bleed-reference.json")
ARGS = sys.argv[1:]


# ------------------------------------------------------------------------------------------------ the curve, as the C++ evaluates it
@njit(cache=True)
def pchip_build(y):
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


# ------------------------------------------------------------------------------------------------ the two nodes
@njit(cache=True)
def node_level(a, fs, ta, tr, dual, t2, c2, Tk, depth, sv, y, m):
    """the current DiscreteStage::process (bleed to T - depth always on, level-law attack diode, DUAL, the snap, the hard zero at
    rest): returns the gain reduction per sample through the PCHIP curve (y, m)"""
    n = a.shape[0]; g = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs)); k2 = (1.0 - np.exp(-1.0 / (t2 * fs))) if dual else 0.0
    fl = 10.0 ** (FLOOR / 20.0); rest = Tk - depth; v = rest; w = rest
    for i in range(n):
        e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
        dv = (rest - v) * kR
        if e > v:
            xe = e - rest; f = 1.0 + xe / sv if xe > 0.0 else 1.0
            dv += (e - v) * (1.0 - np.exp(-rA * f))
        v += dv
        if dual:
            flow = (v - w) * k2; v -= flow; w += flow / c2
        if v < rest + 1e-9: v = rest
        x = pchip_at(y, m, v - Tk)
        g[i] = 0.0 if (x < 0.0 or v <= rest) else x
    return g


@njit(cache=True)
def node_gr(a, fs, ta, tr, dual, t2, c2, Tk, dref, sv, law, y, m):
    """the storage node after the gain computer: returns the gain reduction per sample"""
    n = a.shape[0]; g = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs)); k2 = (1.0 - np.exp(-1.0 / (t2 * fs))) if dual else 0.0
    fl = 10.0 ** (FLOOR / 20.0); rest = Tk - dref; s = 0.0; w = 0.0
    for i in range(n):
        e = 20.0 * np.log10(a[i]) if a[i] > fl else FLOOR
        gt = pchip_at(y, m, e - Tk)
        if gt < 0.0: gt = 0.0
        dg = -s * kR
        if gt > s:
            xe = (e - rest) if law == 0 else gt
            f = 1.0 + xe / sv if xe > 0.0 else 1.0
            dg += (gt - s) * (1.0 - np.exp(-rA * f))
        s += dg
        if dual:
            flow = (s - w) * k2; s -= flow; w += flow / c2
        if s < 0.0: s = 0.0
        g[i] = s
    return g


@njit(parallel=True, cache=True)
def sine_grid(levels, f0s, tas, trs, Tks, ridx, C, S, fs, dpar, sv, law, secs, last, which):
    """steady sines, one per entry: lock-in gain in dB over the last `last` s (as the protocol's gain_db) and the mean gain
    reduction there. which 0: the level node (dpar = depth); which 1: the GR node (dpar = dref)."""
    mm = levels.shape[0]; gain = np.empty(mm); gmean = np.empty(mm)
    for j in prange(mm):
        n = int(secs * fs); n0 = n - int(last * fs)
        amp = 10.0 ** (levels[j] / 20.0); wv = 2.0 * np.pi * f0s[j] / fs
        a = np.empty(n)
        for i in range(n): a[i] = abs(amp * np.sin(wv * i))
        r = ridx[j]
        if which == 0:
            g = node_level(a, fs, tas[j], trs[j], False, 1.0, 1.0, Tks[j], dpar, sv, C[r], S[r])
        else:
            g = node_gr(a, fs, tas[j], trs[j], False, 1.0, 1.0, Tks[j], dpar, sv, law, C[r], S[r])
        sr = 0.0; si = 0.0; sm = 0.0
        for i in range(n0, n):
            s = np.sin(wv * i); cs = np.cos(wv * i)
            yv = amp * s * (10.0 ** (-g[i] / 20.0))
            sr += yv * s; si += yv * cs; sm += g[i]
        gain[j] = 20.0 * np.log10(2.0 * np.sqrt(sr * sr + si * si) / (n - n0) / amp)
        gmean[j] = sm / (n - n0)
    return gain, gmean


@njit(cache=True)
def env_lockin(x, gr, f, fs):
    """the protocol's 'env' feature on y = x 10^(-gr/20): gain of the fundamental per period by lock-in, dB"""
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


# ------------------------------------------------------------------------------------------------ the static family
def static_setup():
    data = []
    for ri, r in enumerate(RATIOS):
        for k in range(1, 25):
            for L in LEVELS:
                data.append((ri, k - 1, float(L), GAIN12 - F[f"disc_static_{r}_t{k}_{L}"]))
    d = np.array(data)
    return d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2], d[:, 3]


YS = np.concatenate([np.arange(-20.0, -8.0, 1.0), np.arange(-8.0, 8.0, 0.25), np.arange(8.0, 76.0, 1.0)])


class Static:
    """the target family G_r(y), y = L - T'_k: pooled isotonic regression per ratio (the floor of any curve-of-level model with T' fixed)"""
    def __init__(self, Tp):
        self.Tp = Tp
        self.ri, self.ki, self.L, self.G = static_setup()
        y = self.L - Tp[self.ki]
        self.steps = []; fit = np.empty_like(self.G)
        for r in range(6):
            m = self.ri == r
            xr = np.round(y[m], 6); ux, inv = np.unique(xr, return_inverse=True)
            cnt = np.bincount(inv).astype(float); sy = np.bincount(inv, weights=self.G[m])
            fu = S3.pava(ux, sy / cnt, cnt); fu = np.maximum(fu, 0.0)
            self.steps.append((ux, fu)); fit[m] = fu[inv]
        self.floor = fit - self.G
        self.Gt = np.stack([np.interp(YS, ux, fu, left=0.0) for ux, fu in self.steps])   # (6, len(YS))


def rms_max(e):
    e = np.asarray(e); return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def fit_curves_gr(st, pd, law, C0=None, iters=5, damp=0.8, verbose=False):
    """the GR-node-domain curves on the 1 dB grid by fixed point: simulate the node on YS (relative to T16) at the capture setting,
    correct the table by the residual against the target family, repeat. Returns C, S, the last residual on YS (6, len(YS))."""
    ta, tr, t2, c2, dref, sv = pd
    T16 = st.Tp[15]; ny = len(YS)
    C = np.stack([np.maximum.accumulate(np.maximum(np.interp(XG, YS, st.Gt[r]), 0.0)) for r in range(6)]) if C0 is None else C0.copy()
    lv = np.tile(YS + T16, 6); ri = np.repeat(np.arange(6), ny).astype(np.int64); mm = len(lv)
    res = None
    for it in range(iters):
        S = np.stack([pchip_build(C[r]) for r in range(6)])
        gain, _ = sine_grid(lv, np.full(mm, 1000.0), np.full(mm, ta[2]), np.full(mm, tr[2]), np.full(mm, T16), ri, C, S, float(FS), dref, sv, law, 2.5, 0.5, 1)
        P = (-gain).reshape(6, ny)
        res = st.Gt - P
        if verbose: print(f"      curve fixed point {it}: residual on the fine grid rms {rms_max(res)[0]:.4f} max {rms_max(res)[1]:.3f}")
        if it == iters - 1: break
        for r in range(6):
            dC = np.interp(XG, YS, res[r], left=0.0, right=res[r][-1])
            C[r] = np.maximum.accumulate(np.maximum(C[r] + damp * dC, 0.0))
    S = np.stack([pchip_build(C[r]) for r in range(6)])
    return C, S, res


# ------------------------------------------------------------------------------------------------ the dynamic residual (stage 3b)
BURST_IDS = [f"disc_burst_{a}_{rc}" for a in ATTACKS for rc in RECOVERS] + [f"disc_dual_blen_{b}" for b in (0.1, 0.5, 2.0, 8.0)] + ["disc_dual_pulses"] + [f"disc_bdepth_{l}" for l in (-30, -20, 0)]
XSIG = {iid: protocol.stimulus(ITEMS[iid]["stim"], ITEMS[iid]["fs"])[0] for iid in BURST_IDS}
STIM = {iid: np.abs(XSIG[iid]) for iid in BURST_IDS}
ENV_N = {iid: min(len(F[iid]), len(STIM[iid]) // int(round(ITEMS[iid]["fs"] / ITEMS[iid]["stim"]["f"]))) for iid in BURST_IDS}


def gr_item(iid, C, S, T, pd, law, which, ta_i=None, tr_i=None):
    ta, tr, t2, c2, dpar, sv = pd
    it = ITEMS[iid]; s = it["set"]; fs = float(it["fs"])
    dual = s.get("discrete_recover", "0.5 s") == "Dual"
    Tk = T[int(s["discrete_threshold"]) - 1]
    ai = ATTACKS.index(float(s["discrete_attack"])); rci = RECOVERS.index(s["discrete_recover"])
    r = RATIOS.index(s.get("discrete_ratio", "4:1"))
    if which == 0: return node_level(STIM[iid], fs, ta[ai], tr[rci], dual, t2, c2, Tk, dpar, sv, C[r], S[r])
    return node_gr(STIM[iid], fs, ta[ai], tr[rci], dual, t2, c2, Tk, dpar, sv, law, C[r], S[r])


def env_of_item(iid, C, S, T, pd, law, which):
    return env_lockin(XSIG[iid], gr_item(iid, C, S, T, pd, law, which), float(ITEMS[iid]["stim"]["f"]), float(ITEMS[iid]["fs"])) + GAIN12


def steady_cells(C, S, T, pd, law, which):
    ta, tr, t2, c2, dpar, sv = pd
    lv, f0, tas, trs, refs = [], [], [], [], []
    for ai, a in enumerate(ATTACKS):
        for rci, rc in enumerate(RECOVERS[:5]):
            lv.append(-10.0); f0.append(1000.0); tas.append(ta[ai]); trs.append(tr[rci]); refs.append(F[f"disc_ar_{a}_{rc}"])
        for l in (-25, 2):
            lv.append(float(l)); f0.append(1000.0); tas.append(ta[ai]); trs.append(tr[2]); refs.append(F[f"disc_al_{a}_{l}"])
    for a in (0.1, 1.0, 30.0):
        for f in (100.0, 5000.0):
            lv.append(-10.0); f0.append(f); tas.append(ta[ATTACKS.index(a)]); trs.append(tr[2]); refs.append(F[f"disc_af_{a}_f{int(f)}"])
    mm = len(lv)
    gain, _ = sine_grid(np.array(lv), np.array(f0), np.array(tas), np.array(trs), np.full(mm, T[15]), np.full(mm, 3, dtype=np.int64), C, S, float(FS), dpar, sv, law, 3.0, 0.5, which)
    return gain + GAIN12 - np.array(refs)


def dyn_resid(C, S, T, pd, law, which):
    out = [3.0 * steady_cells(C, S, T, pd, law, which)]
    for iid in BURST_IDS:
        ref = np.asarray(F[iid]); e = env_of_item(iid, C, S, T, pd, law, which); n = min(len(e), len(ref))
        w = np.where(ref[:n] < GAIN12 - 0.5, 1.0, 0.25) / np.sqrt(n / 100.0)
        out.append(w * (e[:n] - ref[:n]))
    return np.concatenate(out)


def burst_split(res):
    k = 54; nd, du = [], []; per = {}
    for iid in BURST_IDS:
        n = ENV_N[iid]; seg = res[k:k + n]; k += n
        (du if "dual" in iid else nd).append(seg); per[iid] = (float(np.sqrt(np.mean(seg ** 2))), float(np.max(np.abs(seg))))
    nd = np.concatenate(nd); du = np.concatenate(du)
    return rms_max(nd), rms_max(du), per


# ------------------------------------------------------------------------------------------------ the reference probes
TS = (0.25, 0.5, 1.0, 2.0, 4.0)
R3_IDX = (1, 2, 3, 5, 8, 12, 20, 40)


def r2_stim(): return {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 4.5, "f": 1000.0}
def r3_stim(pre): return {"kind": "burst", "pre": float(pre), "level": -10, "pre_s": 2.0, "burst_s": 0.5, "post_s": 0.1, "f": 1000.0}
def r4_stims(lvl):
    return ({"kind": "burst", "pre": -120, "level": lvl, "pre_s": 0.5, "burst_s": 6.0, "post_s": 0.1, "f": 1000.0},
            {"kind": "burst", "pre": lvl, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 6.0, "f": 1000.0})
def at(gr, t_end, ts=TS, f=1000.0):
    return [round(float(np.mean(gr[int((t_end + t) * f) - 50:int((t_end + t) * f)])), 3) for t in ts]


def reference_check(C, S, T, pd, law, which):
    if not os.path.exists(REF_JSON):
        print("    (no reference JSON: run disc-knee-second-bleed.py --reference once)"); return None
    R = json.load(open(REF_JSON)); ta, tr, t2, c2, dpar, sv = pd; T16 = T[15]
    def sim(stim, r, ai, rci):
        x = protocol.stimulus(stim, FS)[0]; ri = RATIOS.index(r)
        if which == 0: g = node_level(np.abs(x), float(FS), ta[ai], tr[rci], False, t2, c2, T16, dpar, sv, C[ri], S[ri])
        else: g = node_gr(np.abs(x), float(FS), ta[ai], tr[rci], False, t2, c2, T16, dpar, sv, law, C[ri], S[ri])
        return -env_lockin(x, g, 1000.0, float(FS))
    errs = {"R1": [], "R2": [], "R3": [], "R4": []}
    print("    REFERENCE PROBES (model | reference), GR in dB; R1 levels relative to the current model's 4:1 rest (-38.14 dBFS):")
    for r, pts in R["R1"].items():
        ri = RATIOS.index(r); d = np.array([p[0] for p in pts]); ref = np.array([p[1] for p in pts]); mm = len(d)
        gain, _ = sine_grid(R["rest"] + d, np.full(mm, 1000.0), np.full(mm, ta[2]), np.full(mm, tr[2]), np.full(mm, T16), np.full(mm, ri, dtype=np.int64), C, S, float(FS), dpar, sv, law, 2.5, 0.5, which)
        g = -gain; errs["R1"] += list(g - ref)
        sel = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24]
        print(f"      R1 knee {r:>5}: " + " ".join(f"{d[i]:+.2f}:{g[i]:.3f}|{ref[i]:.3f}" for i in sel))
    for rc, dd in R["R2"].items():
        rci = RECOVERS.index(rc); row = []
        for r, ref in dd.items():
            m = at(sim(r2_stim(), r, 2, rci), 2.5); errs["R2"] += [a - b for a, b in zip(m, ref)]
            row.append(f"{r} {m} | {ref}")
        print(f"      R2 tail, recover {rc}: " + "  ;  ".join(row))
    for r, dd in R["R3"].items():
        for pre, ref in dd.items():
            gr = sim(r3_stim(float(pre)), r, 5, 2)
            before = round(float(np.mean(gr[1950:2000])), 3); pers = [round(float(gr[2000 + j]), 2) for j in R3_IDX]
            errs["R3"] += [before - ref["before"]] + [a - b for a, b in zip(pers, ref["periods"])]
            print(f"      R3 onset {r} after a pre-roll at {float(pre):8.2f}: before {before} | {ref['before']}; periods {pers} | {[round(v, 2) for v in ref['periods']]}")
    for r, dd in R["R4"].items():
        for d, ref in dd.items():
            lvl = R["rest"] + float(d); s1, s2 = r4_stims(lvl)
            a = at(sim(s1, r, 2, 2), 0.5); b = at(sim(s2, r, 2, 2), 2.5)
            errs["R4"] += [x - y for x, y in zip(a, ref["silence"])] + [x - y for x, y in zip(b, ref["after"])]
            print(f"      R4 tone {r} at rest {d}: from silence {a} | {ref['silence']}; after the burst {b} | {ref['after']}")
    allv = np.concatenate([np.array(v) for v in errs.values()])
    summ = {k: rms_max(np.array(v)) for k, v in errs.items()}; summ["all"] = rms_max(allv)
    print("      probe residual rms/max: " + " | ".join(f"{k} {a:.3f}/{b:.2f}" for k, (a, b) in summ.items()))
    return summ


# ------------------------------------------------------------------------------------------------ reports
REPORT54 = [(r, k, L) for r in range(6) for k in (4, 12, 20) for L in (-30, -15, 0)]
REPORT198 = [(r, k, L) for r in range(6) for k in (4, 12, 20) for L in range(-30, 1, 3)]
FULL = [(r, k, L) for r in range(6) for k in range(1, 25) for L in LEVELS]
T_DISK = cal[MODEL.field("d_thr_db")].copy(); DEPTH_DISK = float(cal[MODEL.field("d_rel_depth_db")][0]); SV_DISK = float(cal[MODEL.field("d_att_sv_db")][0])
PD_DISK = (list(cal[MODEL.field("d_tatt")]), list(cal[MODEL.field("d_trel")]), float(cal[MODEL.field("d_dual_t2")][0]), float(cal[MODEL.field("d_dual_c2")][0]), DEPTH_DISK, SV_DISK)


def rendered_static(items, C, S, T, pd, law, which):
    ta, tr, t2, c2, dpar, sv = pd
    lv = np.array([float(L) for _, _, L in items]); Tk = np.array([T[k - 1] for _, k, _ in items]); ri = np.array([r for r, _, _ in items], dtype=np.int64); mm = len(items)
    g, _ = sine_grid(lv, np.full(mm, 1000.0), np.full(mm, ta[2]), np.full(mm, tr[2]), Tk, ri, C, S, float(FS), dpar, sv, law, 2.5, 0.5, which)
    ref = np.array([F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in items])
    return g + GAIN12 - ref


def report(label, C, S, T, pd, law, which, st, sres=None):
    ta, tr, t2, c2, dpar, sv = pd
    print(f"  {label}: attack ms {np.round(np.array(ta) * 1e3, 3).tolist()} recover s {np.round(tr, 4).tolist()} t2 {t2:.4f} c2 {c2:.2f} {'depth' if which == 0 else 'dref'} {dpar:.3f} Sv {sv:.2f}" + (f" law {'L' if law == 0 else 'G'}" if which else ""))
    if sres is not None:
        print(f"    curve fixed point: residual against the target family on the fine grid rms {rms_max(sres)[0]:.4f} max {rms_max(sres)[1]:.3f}"
              f" (knee -4..2: {rms_max(sres[:, (YS > -4) & (YS <= 2)])[0]:.4f}/{rms_max(sres[:, (YS > -4) & (YS <= 2)])[1]:.3f})")
    on = [float(XG[np.argmax(C[r] > 0.03)]) for r in range(6)]
    print(f"    curve onsets (first grid x with C > 0.03 dB, x = e - T): {on}")
    rest_rel = (T_DISK[15] - DEPTH_DISK) - st.Tp[15]   # the current model's 4:1 rest relative to T', so the bands match the other harnesses
    e54 = rendered_static(REPORT54, C, S, T, pd, law, which); e198 = rendered_static(REPORT198, C, S, T, pd, law, which); efull = rendered_static(FULL, C, S, T, pd, law, which)
    fr = np.array([r for r, _, _ in FULL]); fb = np.array([L - st.Tp[k - 1] - rest_rel for _, k, L in FULL]); knee = (fb > -4.0) & (fb <= 2.0)
    print(f"    RENDERED statics (mirror, exact ripple, PCHIP): 54-item set rms {rms_max(e54)[0]:.3f} max {rms_max(e54)[1]:.2f} | 198-item set rms {rms_max(e198)[0]:.3f} max {rms_max(e198)[1]:.2f}"
          f" | full grid rms {rms_max(efull)[0]:.3f} max {rms_max(efull)[1]:.2f} | knee band (rest - 4 < L <= rest + 2) rms {rms_max(efull[knee])[0]:.3f} max {rms_max(efull[knee])[1]:.2f}")
    print("      full-grid rms/max per ratio:", " ".join(f"{RATIOS[r]} {rms_max(efull[fr == r])[0]:.3f}/{rms_max(efull[fr == r])[1]:.2f}" for r in range(6)))
    print("      knee-band rms/max per ratio:", " ".join(f"{RATIOS[r]} {rms_max(efull[knee & (fr == r)])[0]:.3f}/{rms_max(efull[knee & (fr == r)])[1]:.2f}" for r in range(6)))
    worst = np.argsort(-np.abs(efull))[:4]
    print("      worst grid points:", [(f"{RATIOS[FULL[i][0]]}_t{FULL[i][1]}_{FULL[i][2]}", round(float(efull[i]), 3)) for i in worst])
    mm = 6
    gs1, _ = sine_grid(np.full(mm, -50.0), np.full(mm, 1000.0), np.full(mm, ta[2]), np.full(mm, tr[2]), np.full(mm, T[0]), np.arange(6, dtype=np.int64), C, S, float(FS), dpar, sv, law, 2.0, 0.5, which)
    gs24, _ = sine_grid(np.full(mm, -80.0), np.full(mm, 1000.0), np.full(mm, ta[2]), np.full(mm, tr[2]), np.full(mm, T[23]), np.arange(6, dtype=np.int64), C, S, float(FS), dpar, sv, law, 2.0, 0.5, which)
    print(f"    standing GR at silence per ratio: -50 dBFS at threshold 1 {np.round(-gs1, 4).tolist()} | -80 dBFS at threshold 24 {np.round(-gs24, 4).tolist()}")
    # standing GR after a signal: the R2 stimulus, 4 s after the burst, at every ratio
    x = protocol.stimulus(r2_stim(), FS)[0]; post = []
    for r in range(6):
        g = node_level(np.abs(x), float(FS), ta[2], tr[2], False, t2, c2, T[15], dpar, sv, C[r], S[r]) if which == 0 else node_gr(np.abs(x), float(FS), ta[2], tr[2], False, t2, c2, T[15], dpar, sv, law, C[r], S[r])
        post.append(round(float(np.mean(g[int(6.4 * FS):int(6.5 * FS)])), 4))
    print(f"    standing GR 4 s after a -10 dBFS burst (threshold 16, 1 ms / 0.5 s) per ratio: {post}")
    dres = dyn_resid(C, S, T, pd, law, which)
    ss = dres[:54] / 3.0; (brms, bmax), (drms, dmax), per_item = burst_split(dres)
    print(f"    steady table rms {rms_max(ss)[0]:.3f} max {rms_max(ss)[1]:.2f} | bursts weighted rms {brms:.3f} max {bmax:.2f} | DUAL (blen, pulses) rms {drms:.3f} max {dmax:.2f}")
    k = 0; rows = []
    for ai, a in enumerate(ATTACKS):
        rows.append(f"{a:5.1f} ms: " + " ".join(f"{v:+.2f}" for v in ss[k:k + 7])); k += 7
    print("      steady residual rows (recover 0.1/0.25/0.5/0.8/1.2 | level -25/+2):", " || ".join(rows))
    print("      frequency checks (100 Hz / 5 kHz at 0.1, 1, 30 ms):", " ".join(f"{v:+.2f}" for v in ss[42:48]))
    worst = sorted(per_item.items(), key=lambda t: -t[1][0])[:5]
    print("      worst bursts (weighted rms, max):", [(i, round(a, 3), round(b, 2)) for i, (a, b) in worst])
    for iid, idx in (("disc_burst_30.0_0.5 s", (500, 501, 502, 505, 510, 520, 540)), ("disc_burst_1.0_0.5 s", (2600, 2800, 3000, 3300, 3600, 4000)), ("disc_dual_blen_8.0", None)):
        e = env_of_item(iid, C, S, T, pd, law, which); ref = np.asarray(F[iid]); n = min(len(e), len(ref))
        if idx is None:
            idx = tuple(int(v) for v in np.linspace(n * 0.55, n - 1, 6))
        print(f"      {iid} (period, model, ref):", [(j, round(float(e[j]), 2), round(float(ref[j]), 2)) for j in idx if j < n])
    xc = reference_check(C, S, T, pd, law, which)
    sys.stdout.flush()
    return dict(label=label, e54=rms_max(e54), e198=rms_max(e198), efull=rms_max(efull), knee=rms_max(efull[knee]), sil=[float(v) for v in -gs1], post=post,
                steady=rms_max(ss), bursts=(brms, bmax), dual=(drms, dmax), probes=xc, onsets=on, pd=[list(map(float, ta)), list(map(float, tr)), t2, c2, dpar, sv], law=law)


def cpp_check(C, S, T):
    """the level-node mirror against the C++ engine on the current constants (validates the mirror's static and burst evaluation)"""
    e54 = rendered_static(REPORT54, C, S, T, PD_DISK, 0, 0)
    ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t{k}_{L}"], cal) - F[f"disc_static_{RATIOS[r]}_t{k}_{L}"] for r, k, L in REPORT54])
    print(f"  C++ engine, 54-item set: rms {rms_max(ec)[0]:.3f} max {rms_max(ec)[1]:.2f}; mirror - C++ max |diff| {np.max(np.abs(e54 - ec)):.4f} dB")
    ec = np.array([render_item(ITEMS[f"disc_static_{RATIOS[r]}_t12_-60"], cal) - F[f"disc_static_{RATIOS[r]}_t12_-60"] for r in range(6)])
    print(f"  C++ engine standing GR at silence (disc_static_*_t12_-60), per ratio: {np.round(-ec, 4).tolist()}")
    for iid in ("disc_burst_1.0_0.5 s", "disc_burst_30.0_0.1 s", "disc_burst_1.0_Dual"):
        e = np.asarray(render_item(ITEMS[iid], cal)); m = env_of_item(iid, C, S, T, PD_DISK, 0, 0); ref = np.asarray(F[iid]); n = min(len(e), len(m), len(ref))
        print(f"    C++ {iid}: rms vs reference {rms_max(e[:n] - ref[:n])[0]:.3f}; mirror - C++ rms {rms_max(m[:n] - e[:n])[0]:.4f} max {rms_max(m[:n] - e[:n])[1]:.3f}")
    # the post-burst standing value through the engine at every ratio (the R2 stimulus)
    stim = r2_stim(); vals = []
    for r in RATIOS:
        it = {"id": "x", "fs": FS, "stim": stim, "set": {"discrete_bypass": "In", "discrete_threshold": 16, "discrete_attack": 1.0, "discrete_recover": "0.5 s", "discrete_ratio": r}, "feat": {"type": "env"}}
        g = GAIN12 - np.asarray(render_item(it, cal)); vals.append(round(float(np.mean(g[6450:6500])), 4))
    print(f"  C++ engine standing GR 4 s after a -10 dBFS burst (threshold 16, 1 ms / 0.5 s) per ratio: {vals}")
    sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ main
def main():
    law_arg = ARGS[ARGS.index("--law") + 1] if "--law" in ARGS else "both"
    laws = [0, 1] if law_arg == "both" else [0 if law_arg == "L" else 1]
    nfev = int(ARGS[ARGS.index("--nfev") + 1]) if "--nfev" in ARGS else 12
    w_static = float(ARGS[ARGS.index("--wstatic") + 1]) if "--wstatic" in ARGS else 1.0
    print(f"GAIN12 {GAIN12:.4f}; curve grid N {N} X0 {X0} DX {DX}; reference GR at -50 dBFS, threshold 1: {GAIN12 - F['law_disc_gain_12']:+.4f} dB")
    Tp, _ = S3.fit_static()
    st = Static(Tp)
    print(f"target family (pooled isotonic per ratio against L - T'): floor rms {rms_max(st.floor)[0]:.4f} max {rms_max(st.floor)[1]:.3f} over 3600 points")
    results = {}
    # baseline: the current constants through the level node (the engine's form)
    C0 = cal[MODEL.field("d_curve")].reshape(6, N).copy(); S0 = np.stack([pchip_build(C0[r]) for r in range(6)])
    print("\nBASELINE: the current constants (level node, on-disk curves, hard zero at rest)")
    if "--no-cpp" not in ARGS: cpp_check(C0, S0, T_DISK)
    results["baseline"] = report("on-disk constants", C0, S0, T_DISK, PD_DISK, 0, 0, st)
    # pass A per law: detector constants as on disk, curves refitted in the GR-node domain
    T = Tp.copy()   # the GR node has no offset to fold into the thresholds: T = T'
    fitted = {}
    for law in laws:
        nm = "L" if law == 0 else "G"
        print(f"\nGR NODE, law {nm}: pass A (detector constants as on disk)")
        pd = tuple(PD_DISK)
        t0 = time.time(); C, S, sres = fit_curves_gr(st, pd, law, iters=6, verbose=True); print(f"    (curve fit {time.time() - t0:.1f} s)")
        results[f"A_{nm}"] = report(f"pass A law {nm}", C, S, T, pd, law, 1, st, sres)
        fitted[law] = (C, pd)
        if "--fit" in ARGS:
            print(f"\nGR NODE, law {nm}: pass B (joint refit of the sixteen stage-3b constants; curves refitted inside every evaluation)")
            p0 = list(pd[0]) + list(pd[1]) + [pd[2], pd[3], pd[4], float(np.log10(pd[5]))]
            lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, -5.0, 0.3]; hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 10.0, 3.0]
            xs = [1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 0.5, 0.3]
            cache = {"C": C.copy()}
            wst = w_static * np.sqrt(3600.0 / (6 * len(YS)))
            def resid(p):
                pdp = (list(p[0:6]), list(p[6:12]), float(p[12]), float(p[13]), float(p[14]), float(10.0 ** p[15]))
                Cc, Sc, sr = fit_curves_gr(st, pdp, law, C0=cache["C"], iters=2)
                cache["C"] = Cc
                return np.concatenate([dyn_resid(Cc, Sc, T, pdp, law, 1), wst * sr.reshape(-1)])
            t0 = time.time()
            r = least_squares(resid, np.clip(p0, lo, hi), bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
            print(f"  fit: nfev {r.nfev} cost {r.cost:.3f} ({time.time() - t0:.0f} s)")
            p = r.x; pdp = (list(p[0:6]), list(p[6:12]), float(p[12]), float(p[13]), float(p[14]), float(10.0 ** p[15]))
            C, S, sres = fit_curves_gr(st, pdp, law, C0=cache["C"], iters=6, verbose=True)
            results[f"B_{nm}"] = report(f"pass B law {nm}", C, S, T, pdp, law, 1, st, sres)
            print("  p:", np.round(p, 6).tolist())
            fitted[law] = (C, pdp)
        os.makedirs(OUT, exist_ok=True)
        json.dump({"results": results, "Tp": Tp.tolist(), "curves": {("L" if l == 0 else "G"): {"C": c.tolist(), "pd": [list(map(float, q[0])), list(map(float, q[1])), q[2], q[3], q[4], q[5]]} for l, (c, q) in fitted.items()}},
                  open(os.path.join(OUT, "disc-knee-gr-domain.json"), "w"), indent=1)
    print("\nSUMMARY (statics: rendered rms/max on the 54-item set, the 198-item set, the full grid, the knee band | silence per ratio | standing 4 s after a burst | dynamics | reference probes)")
    for key, R in results.items():
        pr = R["probes"]
        print(f"  {key:9s}: statics54 {R['e54'][0]:.3f}/{R['e54'][1]:.2f} 198 {R['e198'][0]:.3f}/{R['e198'][1]:.2f} full {R['efull'][0]:.3f}/{R['efull'][1]:.2f} knee {R['knee'][0]:.3f}/{R['knee'][1]:.2f}"
              f" | silence {np.round(R['sil'], 3).tolist()} | post-burst {R['post']} | steady {R['steady'][0]:.3f}/{R['steady'][1]:.2f} bursts {R['bursts'][0]:.3f}/{R['bursts'][1]:.2f} DUAL {R['dual'][0]:.3f}/{R['dual'][1]:.2f}"
              + (f" | probes all {pr['all'][0]:.3f}/{pr['all'][1]:.2f} R1 {pr['R1'][0]:.3f}/{pr['R1'][1]:.2f} R2 {pr['R2'][0]:.3f}/{pr['R2'][1]:.2f} R3 {pr['R3'][0]:.3f}/{pr['R3'][1]:.2f} R4 {pr['R4'][0]:.3f}/{pr['R4'][1]:.2f}" if pr else ""))


if __name__ == "__main__":
    main()
