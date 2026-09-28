# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Transformer path candidate, key "xfmr-even-c-output-stage": where do the reference's even harmonics come from?

The reference law (fit/data/reference_features.json, xf_{core}_f{f}_{level}, tabulated by this script with --law):
  - the odd order is a pure flux-law saturator: at equal flux drive (level - 20 log10(f / 20)) H3 is frequency independent to 1 dB;
  - the even order has an ONSET BURST that is also flux-law scaled and frequency independent: H2 = H4 = -78.8 dBc at the onset and a
    peak of about -59.5 dBc six dB above it, on every core and at 20 / 40 / 80 / 160 Hz alike;
  - past the peak the even order falls to a floor that DECAYS with drive at fixed frequency (20 Hz Nickel: -63, -64, -66, -67, -69 dBc
    from +9 to +21 dBFS) and, at equal flux drive, falls about 7 dB per octave with frequency (+9 dB over the onset: -63 at 20 Hz,
    -71 at 40 Hz, -80 at 80 Hz); Iron's floor sits about 3 dB above its driver's 1 kHz H2 at every frequency (the driver's even term
    passes through the saturating core), Nickel's driver is 20 dB quieter and its floor is the residual of the burst.
  - in the linear region the even order is the driver's quadratic, 1 dB per dB, at all frequencies.
The C++ model (src/dsp/Transformer.hpp, stage 2) puts the driver's a2 u^2 BEFORE the flux integrator. That term has a mean a2 A^2 / 2 and
the leaky integrator gives it a DC gain of 1 / r (r = 2 pi x_fl_hz): a flux offset that grows with the square of the level and shifts
the core's spikes, so the model's even order grows 0.5 - 1 dB per dB where the reference's decays (the C++ with x_a2 = 0 already
follows the reference's 20 Hz decay within 3 dB; its H2 phase flips to 90 degrees exactly where the growth begins).

Hypotheses tested here (direction (c): even order after the core; (d): the loss term against the asymmetry; and a hybrid core):
  B0   the C++ structure refitted (control): a2 before the core, leak r phi, knee-hardness asymmetry;
  C1   a2 moved AFTER the core, an output-stage quadratic on the differentiated flux (no DC reaches the integrator);
  C1L  C1 with the output even term level-limited, a2 y^2 / (1 + (y / L)^2), L fitted;
  C2   a2 before the core but AC-coupled (a2 (u^2 - <u^2>), the driver's coupling capacitor), the 2f term still drives the core;
  D1   C1 with the loss on the SATURATED flux, phi += T (v - r S(phi)) (source resistance against the magnetising current);
  D2   C1 with the asymmetry only above a flux level: asym_eff = asym |phi/phik|^8 / (thr^8 + |phi/phik|^8), thr fitted;
  D3   C1 with a CEILING asymmetry (phik (1 +- asym)) instead of a hardness asymmetry;
  D4   C1 with a rate loss: phi += T (v - r phi - re y), re fitted (eddy-like, widens the spikes more at higher frequency);
  D5   C1 with a FLUX OFFSET (remanence, S = sat(phi + off phik)) instead of the hardness asymmetry (the earlier rejection of an offset
       was made with the driver's DC term still reaching the integrator);
  D6   C1 with both the hardness asymmetry and the flux offset.
Each candidate is a numba mirror of the transformer path (gain, driver, core, high shelf, low pass; both compressor stages out) validated
against the C++ engine on the stage-2 calibration before any fit, then fitted per core with bounded least squares (soft L1) on the grid
-12 .. +21 dBFS x 10 frequencies (+24 dBFS is the reference's output ceiling and is left out, as tests/pb_reference.py does), on the
residual of feat_residual (gain x5, H2/H3 x1, H4/H5 x0.5, H6..H8 x0.25, harmonics the reference has below -80 dBc ignored).
Reported per core with the test suite's metric: gain rms/max, odd rms/max, even rms/max over the same grid (harmonics where the reference
sits above -80 dBc, the model floored at -120 dBc).

usage: cd <repo> && python3 -u fit/tools/candidates/xfmr-even-output.py [--law] [--variants B0,C1,...] [--cores Nickel,Iron,Steel]
                                                                         [--nfev 60] [--validate-only]
"""
import json, os, sys, time
import numpy as np
from numba import njit, prange
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, feat_residual, protocol, FS  # noqa: E402

ARGS = sys.argv[1:]
def opt(name, default):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default
VARIANTS = opt("--variants", "B0,C1,C1L,C2,D1,D2,D3,D4,D5,D6").split(",")
CORES = opt("--cores", "Nickel,Iron,Steel").split(",")
NFEV = int(opt("--nfev", "60"))
FREQS = [20, 30, 40, 60, 80, 120, 160, 320, 1000, 5000]
LEVELS = list(range(-12, 22, 3))
HW = np.array([1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25])
GAIN_W = 5.0
FLOOR = -80.0     # reference harmonics below this are not compared (tests/pb_reference.py HARM_FLOOR)
MFLOOR = -120.0
FSF = float(FS)
cal = load_cal()
def cget(name, k): return float(cal[MODEL.fields[name][0] + k])


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def sat(ph, phik_p, phik_n, qp, qn):
    if ph >= 0.0:
        az = ph / phik_p; q = qp
    else:
        az = -ph / phik_n; q = qn
    if az <= 0.02:
        return ph
    return ph / np.exp(np.log1p(np.exp(q * np.log(az))) / q)


@njit(cache=True)
def run_one(x, fs, gain_db, a2, a3, sat_db, q, asym, fl_hz, hs_hz, hs_db, lp_hz, mode_a2, mode_loss, mode_asym, L, thr, re, off):
    """one channel of the transformer path; modes: a2 0 = before the core, 1 = after (output stage), 2 = before but AC-coupled;
    loss 0 = r phi, 1 = r S(phi), 2 = r phi + re y; asym 0 = knee hardness, 1 = hardness above thr, 2 = ceiling, 3 = flux offset only
    (asym is the offset in knee-flux units), 4 = hardness plus the offset off (knee-flux units)"""
    n = x.shape[0]; y = np.empty(n)
    T = 1.0 / fs
    g = 10.0 ** (gain_db / 20.0)
    r = 2.0 * np.pi * fl_hz
    phik = 10.0 ** (sat_db / 20.0) / (2.0 * np.pi * 20.0)
    qp = q * (1.0 + asym); qn = q * (1.0 - asym)
    phik_p = phik; phik_n = phik
    if mode_asym == 2:
        qp = q; qn = q; phik_p = phik * (1.0 + asym); phik_n = phik * (1.0 - asym)
    if mode_asym == 1:
        qp = q; qn = q
    dphi = 0.0
    if mode_asym == 3:
        qp = q; qn = q; dphi = asym * phik
    if mode_asym == 4:
        dphi = off * phik
    if qp < 1.0: qp = 1.0
    if qn < 1.0: qn = 1.0
    # corners at or above 0.49 fs are clamped there, as the C++ FirstOrder::set does
    g_hs = np.tan(np.pi * min(hs_hz, 0.49 * fs) / fs) if hs_hz > 0.0 else 0.0
    gl_hs = 10.0 ** (hs_db / 20.0)
    g_lp = np.tan(np.pi * min(lp_hz, 0.49 * fs) / fs) if lp_hz > 0.0 else 0.0
    kdc = 1.0 - np.exp(-1.0 / (0.05 * fs))
    phi = 0.0; sPrev = 0.0; s_hs = 0.0; s_lp = 0.0; dc = 0.0; yPrev = 0.0
    L2 = L * L
    for i in range(n):
        u = g * x[i]
        u2 = u * u
        if mode_a2 == 0:
            v = u + a2 * u2 + a3 * u2 * u
        elif mode_a2 == 1:
            v = u + a3 * u2 * u
        else:
            dc += (u2 - dc) * kdc
            v = u + a2 * (u2 - dc) + a3 * u2 * u
        # core
        if mode_loss == 0:
            phi += T * (v - r * phi)
        elif mode_loss == 1:
            phi += T * (v - r * sat(phi, phik_p, phik_n, qp, qn))
        else:
            phi += T * (v - r * phi - re * yPrev)
        if mode_asym == 1:
            az = abs(phi) / phik
            a8 = az ** 8
            ae = asym * a8 / (thr ** 8 + a8)
            s = sat(phi, phik_p, phik_n, q * (1.0 + ae), q * (1.0 - ae))
        else:
            s = sat(phi + dphi, phik_p, phik_n, qp, qn)
        yv = (s - sPrev) * fs
        sPrev = s
        yPrev = yv
        if mode_a2 == 1:
            if L > 0.0:
                yv = yv + a2 * yv * yv / (1.0 + yv * yv / L2)
            else:
                yv = yv + a2 * yv * yv
        # high shelf (TPT), low pass (TPT)
        if hs_hz > 0.0:
            vv = (yv - s_hs) * g_hs / (1.0 + g_hs); lp = vv + s_hs; s_hs = lp + vv
            yv = lp + gl_hs * (yv - lp)
        if lp_hz > 0.0:
            vv = (yv - s_lp) * g_lp / (1.0 + g_lp); lp = vv + s_lp; s_lp = lp + vv
            yv = lp
        y[i] = yv
    return y


@njit(parallel=True, cache=True)
def run_grid(X, fs, p, modes, extra):
    m = X.shape[0]; Y = np.empty_like(X)
    for j in prange(m):
        Y[j] = run_one(X[j], fs, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], modes[0], modes[1], modes[2],
                       extra[0], extra[1], extra[2], extra[3])
    return Y


# ------------------------------------------------------------------------------------------------ features and residuals
def grid_ids(core):
    return [f"xf_{core}_f{f}_{l}" for f in FREQS for l in LEVELS]

STIM = {}
def stimuli(ids):
    X = np.stack([protocol.stimulus(ITEMS[i]["stim"], FS)[0] for i in ids])
    return X

LOCKIN = {}
def lockin_plan(i):
    """the harm feature of protocol.feature with the lock-in exponentials cached per item (same segment and harmonics as protocol.py)"""
    if i in LOCKIN: return LOCKIN[i]
    it = ITEMS[i]; ft, st = it["feat"], it["stim"]
    f = st["f"]; per = FS / f
    n = int(round(st["secs"] * FS))
    nper = max(4, int(round(ft.get("last_s", 0.5) * f)))
    n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
    t = np.arange(n0, n1) / FS
    ks = [k for k in range(1, 9) if k == 1 or k * f < FS / 2 * 0.95]
    E = np.stack([np.exp(-2j * np.pi * k * f * t) for k in ks])
    LOCKIN[i] = (n0, n1, ks, E, st["level"])
    return LOCKIN[i]

def features(ids, Y):
    out = []
    for i, y in zip(ids, Y):
        n0, n1, ks, E, level = lockin_plan(i)
        c = 2.0 * (E @ y[n0:n1]) / (n1 - n0)
        c1 = abs(c[0]); hs = [None] * 7
        for j, k in enumerate(ks[1:], start=1):
            hs[k - 2] = round(float(20 * np.log10(abs(c[j]) / (c1 + 1e-30) + 1e-30)), 2)
        out.append({"h": hs, "gain_db": round(float(20 * np.log10(c1 + 1e-30) - level), 4)})
    return out

def check_features():
    """the cached lock-in against protocol.feature on a few rendered items"""
    ids = ["xf_Nickel_f20_12", "xf_Iron_f5000_21", "xf_Steel_f1000_-12", "xf_Iron_f60_18"]
    X = stimuli(ids); Y = run_grid(X, FSF, pvec(base_params(1)), np.array([0, 0, 0]), np.array([0.0, 1.0, 0.0, 0.0]))
    fa = features(ids, Y); worst = 0.0
    for i, y, a in zip(ids, Y, fa):
        b = protocol.feature(ITEMS[i], np.stack([y, y]), FS)
        worst = max(worst, abs(a["gain_db"] - b["gain_db"]))
        for u, v in zip(a["h"], b["h"]):
            if (u is None) != (v is None): worst = np.inf
            elif u is not None and max(u, v) > -100.0: worst = max(worst, abs(u - v))
    print(f"  cached lock-in against protocol.feature: worst |difference| {worst:.4f} dB (gain or any harmonic above -100 dBc)")
    return worst

def residual_vec(ids, feats):
    """the stage-2 residual (common.feat_residual: gain x5, H2/H3 x1, H4/H5 x0.5, H6..H8 x0.25, floor -120 dBc, x0.3 where the reference
    is below -110): every harmonic counts, so the 1 kHz quadratic law pins a2 (with the test's -80 dBc floor Nickel's and Steel's
    linear-region H2, all below -80 dBc, would leave a2 free and the fit inflates it to shape the deep-saturation floor)"""
    return np.concatenate([feat_residual(i, m) for i, m in zip(ids, feats)])

def metrics(ids, feats):
    """tests/pb_reference.py pooled figures: gain, odd, even (rms, max, n) and the worst item of each"""
    g, od, ev = [], [], []; worst = {"odd": (0, ""), "even": (0, "")}
    for i, m in zip(ids, feats):
        ref = F[i]; g.append(m["gain_db"] - ref["gain_db"])
        for k, (a, b) in enumerate(zip(m["h"], ref["h"]), start=2):
            if a is None or b is None or b <= FLOOR: continue
            e = max(a, MFLOOR) - b
            kind = "odd" if k % 2 else "even"
            (od if k % 2 else ev).append(e)
            if abs(e) > worst[kind][0]: worst[kind] = (abs(e), f"{i} H{k}")
    st = lambda v: (float(np.sqrt(np.mean(np.square(v)))), float(np.max(np.abs(v))), len(v)) if len(v) else (0.0, 0.0, 0)
    per = {}
    for i, m in zip(ids, feats):
        ref = F[i]
        for k, (a, b) in enumerate(zip(m["h"], ref["h"]), start=2):
            if a is None or b is None or b <= FLOOR: continue
            per.setdefault(k, []).append(max(a, MFLOOR) - b)
    return {"gain": st(g), "odd": st(od), "even": st(ev), "worst": worst, "per": {k: st(v) for k, v in sorted(per.items())}}

def fmt(mt):
    g, o, e = mt["gain"], mt["odd"], mt["even"]
    per = " ".join(f"H{k} {v[0]:.1f}/{v[1]:.0f}({v[2]})" for k, v in mt["per"].items())
    return (f"gain rms {g[0]:.3f} max {g[1]:.2f} | odd rms {o[0]:.2f} max {o[1]:.1f} (n {o[2]}) | even rms {e[0]:.2f} max {e[1]:.1f} (n {e[2]})"
            f"  worst odd {mt['worst']['odd'][1]}, even {mt['worst']['even'][1]}\n    per order rms/max(n): {per}")


# ------------------------------------------------------------------------------------------------ the candidates
# name: (mode_a2, mode_loss, mode_asym, extra names)
CAND = {
    "B0":  (0, 0, 0, []),
    "C1":  (1, 0, 0, []),
    "C1L": (1, 0, 0, ["L"]),
    "C2":  (2, 0, 0, []),
    "D1":  (1, 1, 0, []),
    "D2":  (1, 0, 1, ["thr"]),
    "D3":  (1, 0, 2, []),
    "D4":  (1, 2, 0, ["re"]),
    "D5":  (1, 0, 3, []),
    "D6":  (1, 0, 4, ["off"]),
}
NL = ["a2", "a3", "sat_db", "q", "asym"]
BOUNDS = {"a2": (1e-8, 5e-3), "a3": (-5e-2, -1e-7), "sat_db": (-6.0, 12.0), "q": (1.5, 30.0), "asym": (-0.5, 0.5),
          "L": (0.5, 100.0), "thr": (0.3, 3.0), "re": (0.0, 2.0), "off": (-0.3, 0.3)}
SCALE = {"a2": 1e-5, "a3": 1e-4, "sat_db": 0.5, "q": 1.0, "asym": 0.01, "L": 2.0, "thr": 0.1, "re": 0.05, "off": 0.005}
X0 = {"L": 10.0, "thr": 1.0, "re": 0.02, "off": 0.01}

def base_params(k):
    return dict(gain_db=cget("x_gain_db", k), a2=cget("x_a2", k), a3=cget("x_a3", k), sat_db=cget("x_sat_db", k), q=cget("x_q", k),
                asym=cget("x_asym", k), fl_hz=cget("x_fl_hz", k), hs_hz=cget("x_hs_hz", k), hs_db=cget("x_hs_db", k), lp_hz=cget("x_lp_hz", k))

def pvec(d):
    return np.array([d["gain_db"], d["a2"], d["a3"], d["sat_db"], d["q"], d["asym"], d["fl_hz"], d["hs_hz"], d["hs_db"], d["lp_hz"]])

def evaluate(ids, X, d, cand, extra):
    ma, ml, ms, _ = CAND[cand]
    ex = np.array([extra.get("L", 0.0), extra.get("thr", 1.0), extra.get("re", 0.0), extra.get("off", 0.0)])
    Y = run_grid(X, FSF, pvec(d), np.array([ma, ml, ms]), ex)
    return features(ids, Y)

def fit_core(core, k, cand):
    ids = grid_ids(core); X = stimuli(ids)
    names = NL + CAND[cand][3]
    d0 = base_params(k)
    p0 = [d0[n] for n in NL] + [X0[n] for n in CAND[cand][3]]
    lo = [BOUNDS[n][0] for n in names]; hi = [BOUNDS[n][1] for n in names]
    p0 = np.clip(p0, lo, hi)
    def unpack(p):
        d = dict(d0); ex = {}
        for n, v in zip(names, p):
            if n in NL: d[n] = v
            else: ex[n] = v
        return d, ex
    def rfun(p):
        d, ex = unpack(p)
        return residual_vec(ids, evaluate(ids, X, d, cand, ex))
    t0 = time.time()
    r0 = rfun(p0)
    best = None
    starts = [p0]
    # a second start with a smaller asymmetry for the post-core candidates (the stage-2 values were fitted with the DC term); the flux
    # offset candidates start from both signs
    if cand != "B0":
        p1 = p0.copy(); p1[names.index("asym")] *= 0.5 if CAND[cand][2] != 3 else -1.0
        starts.append(p1)
    for s in starts:
        r = least_squares(rfun, s, bounds=(lo, hi), x_scale=[SCALE[n] for n in names], diff_step=1e-3, max_nfev=NFEV, loss="soft_l1",
                          f_scale=2.0)
        if best is None or r.cost < best.cost: best = r
    d, ex = unpack(best.x)
    feats = evaluate(ids, X, d, cand, ex)
    mt = metrics(ids, feats)
    print(f"  {core} {cand}: {dict((n, float(f'{v:.6g}')) for n, v in zip(names, best.x))}  cost {best.cost:.1f} (start {0.5 * np.sum(r0**2):.1f})"
          f"  {time.time() - t0:.0f} s")
    print(f"    {fmt(mt)}")
    return {"params": {n: float(v) for n, v in zip(names, best.x)}, "metrics": mt, "feats": {i: f for i, f in zip(ids, feats)}}


# ------------------------------------------------------------------------------------------------ the reference law
def law():
    for core in ("Nickel", "Iron", "Steel"):
        print(f"\n{core}: H2 / H3 / H4 (dBc) against level at each frequency (columns {LEVELS})")
        for f in FREQS:
            for hk, nm in ((0, "H2"), (1, "H3"), (2, "H4")):
                row = [F[f'xf_{core}_f{f}_{l}']['h'][hk] for l in LEVELS]
                print(f"  {f:5d} {nm} " + " ".join("   ---" if v is None else f"{v:6.1f}" for v in row))
        print(f"{core}: H2 at equal flux drive D = level - 20 log10(f / 20) (columns: D in dB at 20 Hz; rows: frequency)")
        Ds = [0, 3, 6, 9, 12, 15, 18, 21]
        for f in (20, 40, 80, 160):
            row = []
            for D in Ds:
                l = D + int(round(20 * np.log10(f / 20)))
                row.append(f"{F[f'xf_{core}_f{f}_{l}']['h'][0]:6.1f}" if l in LEVELS else "   ---")
            print(f"  {f:5d} " + " ".join(row))
        print(f"{core}: H3 at equal flux drive")
        for f in (20, 40, 80, 160):
            row = []
            for D in Ds:
                l = D + int(round(20 * np.log10(f / 20)))
                row.append(f"{F[f'xf_{core}_f{f}_{l}']['h'][1]:6.1f}" if l in LEVELS else "   ---")
            print(f"  {f:5d} " + " ".join(row))
        print(f"{core}: H2 minus the driver's H2 (the 1 kHz value at the same level), deep in saturation (level >= onset + 9 dB)")
        for f in (20, 30, 40, 60, 80):
            row = []
            for l in LEVELS:
                D = l - 20 * np.log10(f / 20)
                if D < 9: row.append("   ---"); continue
                row.append(f"{F[f'xf_{core}_f{f}_{l}']['h'][0] - F[f'xf_{core}_f1000_{l}']['h'][0]:6.1f}")
            print(f"  {f:5d} " + " ".join(row))


# ------------------------------------------------------------------------------------------------ main
def validate():
    """the mirror (B0 structure, stage-2 calibration) against the C++ engine on every grid point"""
    worst = 0.0; worst_id = ""
    for k, core in enumerate(("Nickel", "Iron", "Steel")):
        ids = grid_ids(core); X = stimuli(ids)
        feats = evaluate(ids, X, base_params(k), "B0", {})
        for i, m in zip(ids, feats):
            c = render_item(ITEMS[i], cal)
            e = abs(m["gain_db"] - c["gain_db"])
            if not np.isfinite(e): e = np.inf
            for a, b in zip(m["h"], c["h"]):
                if a is None or b is None: continue
                if max(a, b) > -100: e = max(e, abs(a - b))
            if e > worst: worst, worst_id = e, i
        mt = metrics(ids, feats)
        print(f"  mirror {core} (stage-2 values): {fmt(mt)}")
    print(f"  mirror against the C++ engine: worst |difference| {worst:.3f} dB (gain or any harmonic above -100 dBc) at {worst_id}")
    return worst

def main():
    if "--law" in ARGS:
        law(); return
    print("validating the mirror against the C++ engine")
    if check_features() > 0.01: print("feature mismatch: stopping"); return
    w = validate()
    if "--validate-only" in ARGS: return
    if w > 0.5:
        print("mirror mismatch above 0.5 dB: stopping"); return
    results = {}
    for cand in VARIANTS:
        print(f"\n== {cand}: a2 {'before' if CAND[cand][0] == 0 else 'after' if CAND[cand][0] == 1 else 'before, AC-coupled'} the core, "
              f"loss {['r phi', 'r S(phi)', 'r phi + re y'][CAND[cand][1]]}, asym {['hardness', 'hardness above thr', 'ceiling', 'flux offset', 'hardness + flux offset'][CAND[cand][2]]}")
        for core in CORES:
            k = ("Nickel", "Iron", "Steel").index(core)
            results[f"{cand}/{core}"] = fit_core(core, k, cand)
    print("\n== summary (grid -12..+21 dBFS x 10 frequencies; test-suite metric)")
    print(f"{'cand':5s} {'core':7s} {'gain rms':>8s} {'gain max':>8s} {'odd rms':>8s} {'odd max':>8s} {'even rms':>8s} {'even max':>8s}")
    for key, r in results.items():
        cand, core = key.split("/"); mt = r["metrics"]
        print(f"{cand:5s} {core:7s} {mt['gain'][0]:8.3f} {mt['gain'][1]:8.2f} {mt['odd'][0]:8.2f} {mt['odd'][1]:8.1f} {mt['even'][0]:8.2f} {mt['even'][1]:8.1f}")
    # the H2 tables of the best candidate per core against the reference
    for core in CORES:
        keys = [k for k in results if k.endswith("/" + core)]
        best = min(keys, key=lambda k: results[k]["metrics"]["even"][0])
        print(f"\n{core}: best even rms {best}; model H2 minus reference H2 where the reference sits above -80 dBc (columns {LEVELS})")
        for f in FREQS:
            row = []
            for l in LEVELS:
                i = f"xf_{core}_f{f}_{l}"; a = results[best]["feats"][i]["h"][0]; b = F[i]["h"][0]
                row.append("    ." if b <= FLOOR else f"{max(a, MFLOOR) - b:5.1f}")
            print(f"  {f:5d} " + " ".join(row))
    out = {k: {"params": v["params"], "metrics": {kk: vv for kk, vv in v["metrics"].items() if kk not in ("worst", "per")}} for k, v in results.items()}
    json.dump(out, open("/tmp/xfmr-even-output.json", "w"), indent=1)

if __name__ == "__main__":
    main()
