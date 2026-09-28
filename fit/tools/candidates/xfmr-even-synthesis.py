# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Transformer even harmonics, the synthesis run (docs/xfmr-even-fix.md): the surviving candidates of the three hypothesis keys on ONE
mirror and ONE metric, plus the form none of them ran.

The three keys agree on the reference law (fit/data/reference_features.json, xf_{core}_f{f}_{level}; --law of xfmr-even-output.py):
below the knee the even order is the driver's quadratic (1 dB/dB, frequency independent); at the flux-law onset (D = level - 20 log10(f/20)
= 0..6 dB) a burst with a flat even spectrum (H2 = H4 = -60 dBc at D = 3..6) that is the same on every core and at every frequency; past
it a floor that DECAYS with drive at fixed frequency (Nickel 20 Hz -63 -> -69 dBc over +9..+21) and, at EQUAL flux drive, falls 6-8 dB
per octave with frequency (Nickel D = 9: -63 / -71 / -80 dBc at 20 / 40 / 80 Hz).

Every candidate the keys fitted makes the asymmetry a function of the flux DEPTH alone (static knee gate D2r, peak-hold decay V5/V8/V9):
at equal D the flux waveform is the same in normalised time at every frequency, the even feature is a fixed FLUX shape, the
differentiator scales it with omega like the fundamental, and its dBc is the same at every frequency. That is why all of them leave
the same 8 dB rms: the reference's floor falls 6-8 dB per octave, which is the signature of an even feature of fixed VOLTAGE size
(not differentiated). The physical place for such a feature is the primary side: the driver's source impedance differs between
sourcing and sinking (a single-ended Class-A stage), so the magnetising-current pulses the saturating core draws on both polarities
leave an even-symmetric voltage drop (R+ - R-)/2 |i_m| that is the same size at every frequency, a narrow pulse per half cycle at the
onset (flat even spectrum) and a notch at the zero crossings deep in (narrowing with drive: the even order decays). An even-symmetric
function of a half-wave-symmetric waveform has only even harmonics and DC, so it cannot touch the odd-order fit.

Candidates (per core, the stage-2 grid at -12..+21 dBFS, stage-2 residual, test-suite metric; a2 AFTER the core in all of them, which the
output key showed is equivalent to AC-coupling the driver and removes the driver's rectified DC from the flux integrator):
  D2r   the output key's best: knee-hardness asymmetry gated below a flux level, asym_eff = asym thr^8 / (thr^8 + |phi/phik|^8)  [thr]
  PH    the remanence key's V8 with the two unidentified constants FIXED (coupling handled by a2-after-core, hold 300 ms): asym_eff =
        asym / max(env, 0.3)^p, env the peak of |phi|/phik held with a 300 ms release                                          [p]
  PH2   PH with p = 2 (the remanence key's V9: no new field)                                                                    []
  M1    the fixed-voltage even feature alone, symmetric knee (asym = 0): y = dS/dt + ke Vk P(phi), Vk = 10^(sat_db/20) (the 20 Hz knee
        level), P = |phi/phik|^q / (1 + |phi/phik|^q) the saturation indicator (both polarities)                              [ke]
  M1a   M1 with the static knee-hardness asymmetry free as well                                                            [ke]
  M1r   M1 with the D2r gate                                                                                          [thr, ke]
  M1p   M1 with the PH decay                                                                                            [p, ke]
  D2c   D2r with the driver's even term BEFORE the core and AC-coupled (a2 (u^2 - <u^2>), the mean a 50 ms one-pole; the output key's
        C2 coupling): the placement the C++ should use, since the after-core placement puts a2 A^2 / 2 of DC on the output    [thr]
  PHc   PH with the same AC-coupled placement                                                                                  [p]
  M1pc  M1p with the AC-coupled placement and the pulse's own mean removed the same way (a transformer secondary passes no DC): the
        C++-ready form of the best candidate                                                                                [p, ke]
Result (2026-09-28, nfev 60, mirror 0.010 dB worst against the C++ engine; even rms/max, Nickel / Iron / Steel, pooled in brackets;
logs build_xfmr_even_synthesis.log and build_xfmr_even_synthesis_cpp.log): D2r 9.12/23.3, 7.15/22.7, 8.11/24.6 [8.08] (reproduces the
output key); PH 8.61/25.2, 5.40/25.6, 8.83/33.0 [7.60], PH2 [8.00]; M1 alone 16.98, 16.91, 19.26 [17.68] (refuted: with no asymmetry
the pulse cannot make the burst); M1a [9.67] (with the static asymmetry the fit turns the pulse off); M1r 7.53, 7.16, 7.84 [7.48];
M1p 5.93/15.7, 6.40/24.6, 6.67/30.8 [6.34]; D2c [8.74] and PHc [8.61] (the before-the-core coupling costs 0.5-1 dB for the gate and the
decay alone); M1pc 5.65/15.2, 5.71/25.0, 6.61/30.2 [5.98], odd 2.05/9.3, 4.96/49.4, 1.97/6.9 [3.44], gain 0.023, 0.044, 0.034, with
a2 8.274e-6 / 8.326e-5 / 8.137e-6, a3 -1.3753e-4 / -8.4828e-4 / -1.3174e-4, sat_db 5.502 / 5.330 / 5.861, q 8.434 / 8.884 / 8.474,
asym -0.01224 / +0.01397 / -0.01240, p 1.648 / 1.990 / 1.764, ke +1.473e-3 / -1.638e-3 / +1.128e-3: the same three constants on
every core. Adopted in docs/xfmr-even-fix.md. What remains: the onset burst 4-5 dB weak at every frequency, the 120/160 Hz +21 dBFS
points 10-12 dB low (the foot of the reference's +24 dBFS ceiling), the reference's nulls 5-12 dB off.
Reported per core with tests/pb_reference.py's metric (harmonics where the reference is above -80 dBc, model floored at -120 dBc, levels
<= +21): gain rms/max, odd rms/max, even rms/max, and the H2 residual table of the best. The mirror is validated against the C++ engine
(fit/common.render_item) on the stage-2 calibration before any fit.

usage: cd <repo> && python3 -u fit/tools/candidates/xfmr-even-synthesis.py [--variants D2r,PH,...] [--cores Nickel,Iron,Steel] [--nfev 60]
                                                                            [--validate-only]
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
VARIANTS = opt("--variants", "D2r,PH,PH2,M1,M1a,M1r,M1p,D2c,PHc,M1pc").split(",")
OUT = "/tmp/xfmr-even-synthesis" + ("-" + "-".join(VARIANTS) if "--variants" in ARGS else "") + ".json"
CORES = opt("--cores", "Nickel,Iron,Steel").split(",")
NFEV = int(opt("--nfev", "60"))
FREQS = [20, 30, 40, 60, 80, 120, 160, 320, 1000, 5000]
LEVELS = list(range(-12, 22, 3))
FLOOR = -80.0
MFLOOR = -120.0
FSF = float(FS)
TAU_HOLD = 0.300      # PH: the peak hold's release, s (the remanence key's V9 value; unidentified by steady sines beyond a few cycles)
ENV_MIN = 0.3         # PH: the decay's floor in knee-flux units (V8/V9)
cal = load_cal()
def cget(name, k): return float(cal[MODEL.fields[name][0] + k])


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def sat(ph, phik, qp, qn):
    az = abs(ph) / phik
    if az <= 0.02:
        return ph
    q = qp if ph >= 0.0 else qn
    lz = q * np.log(az)
    if lz > 700.0:
        return phik if ph > 0.0 else -phik
    return ph / np.exp(np.log1p(np.exp(lz)) / q)


@njit(cache=True)
def run_one(x, fs, gain_db, a2, a3, sat_db, q, asym, fl_hz, hs_hz, hs_db, lp_hz, gate, thr, p, ke, a2_after):
    """one channel of the transformer path: gain, driver (a3 before the core; a2 before or after), flux core, high shelf, low pass.
    gate 0: static knee-hardness asymmetry; 1: gated below thr (D2r); 2: peak-hold power-law decay (PH). ke: the fixed-voltage even
    feature on the core output. a2_after 0: the even term before the core (the C++ as it is); 1: after the core; 2: before the core with
    its running mean removed (AC-coupled, 50 ms one-pole on u^2), and the pulse's mean removed the same way."""
    n = x.shape[0]; y = np.empty(n)
    T = 1.0 / fs
    g = 10.0 ** (gain_db / 20.0)
    r = 2.0 * np.pi * fl_hz
    Vk = 10.0 ** (sat_db / 20.0)
    phik = Vk / (2.0 * np.pi * 20.0)
    qp = q * (1.0 + asym); qn = q * (1.0 - asym)
    if qp < 1.0: qp = 1.0
    if qn < 1.0: qn = 1.0
    thr8 = thr ** 8
    krel = 1.0 - np.exp(-1.0 / (TAU_HOLD * fs))
    g_hs = np.tan(np.pi * min(hs_hz, 0.49 * fs) / fs) if hs_hz > 0.0 else 0.0
    gl_hs = 10.0 ** (hs_db / 20.0)
    g_lp = np.tan(np.pi * min(lp_hz, 0.49 * fs) / fs) if lp_hz > 0.0 else 0.0
    phi = 0.0; sPrev = 0.0; s_hs = 0.0; s_lp = 0.0; env = 0.0; dc = 0.0; pdc = 0.0
    kdc = 1.0 - np.exp(-1.0 / (0.05 * fs))
    for i in range(n):
        u = g * x[i]
        u2 = u * u
        if a2_after == 1:
            v = u + a3 * u2 * u
        elif a2_after == 2:
            dc += (u2 - dc) * kdc
            v = u + a2 * (u2 - dc) + a3 * u2 * u
        else:
            v = u + a2 * u2 + a3 * u2 * u
        phi += T * (v - r * phi)
        az = abs(phi) / phik
        if gate == 0:
            s = sat(phi, phik, qp, qn)
        elif gate == 1:
            az8 = ((az * az) * (az * az)) ** 2
            ae = asym * thr8 / (thr8 + az8)
            s = sat(phi, phik, max(q * (1.0 + ae), 1.0), max(q * (1.0 - ae), 1.0))
        else:
            if az > env: env = az
            else: env += (az - env) * krel
            ae = asym / max(env, ENV_MIN) ** p
            s = sat(phi, phik, max(q * (1.0 + ae), 1.0), max(q * (1.0 - ae), 1.0))
        yv = (s - sPrev) * fs
        sPrev = s
        if ke != 0.0:
            P = 0.0
            if az > 0.02:
                lz = q * np.log(az)
                if lz > 700.0:
                    P = 1.0
                else:
                    e = np.exp(lz); P = e / (1.0 + e)
            if a2_after == 2:
                pdc += (P - pdc) * kdc
                yv += ke * Vk * (P - pdc)
            else:
                yv += ke * Vk * P
        if a2_after == 1:
            yv = yv + a2 * yv * yv
        if hs_hz > 0.0:
            vv = (yv - s_hs) * g_hs / (1.0 + g_hs); lp = vv + s_hs; s_hs = lp + vv
            yv = lp + gl_hs * (yv - lp)
        if lp_hz > 0.0:
            vv = (yv - s_lp) * g_lp / (1.0 + g_lp); lp = vv + s_lp; s_lp = lp + vv
            yv = lp
        y[i] = yv
    return y


@njit(parallel=True, cache=True)
def run_grid(X, fs, pv, gate, thr, p, ke, a2_after):
    m = X.shape[0]; Y = np.empty_like(X)
    for j in prange(m):
        Y[j] = run_one(X[j], fs, pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], pv[7], pv[8], pv[9], gate, thr, p, ke, a2_after)
    return Y


# ------------------------------------------------------------------------------------------------ features, residuals, metrics
def grid_ids(core):
    return [f"xf_{core}_f{f}_{l}" for f in FREQS for l in LEVELS]

def stimuli(ids):
    return np.stack([protocol.stimulus(ITEMS[i]["stim"], FS)[0] for i in ids])

LOCKIN = {}
def lockin_plan(i):
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

def residual_vec(ids, feats):
    return np.concatenate([feat_residual(i, m) for i, m in zip(ids, feats)])

def metrics(ids, feats):
    g, od, ev = [], [], []; worst = {"odd": (0, ""), "even": (0, "")}; per = {}
    for i, m in zip(ids, feats):
        ref = F[i]; g.append(m["gain_db"] - ref["gain_db"])
        for k, (a, b) in enumerate(zip(m["h"], ref["h"]), start=2):
            if a is None or b is None or b <= FLOOR: continue
            e = max(a, MFLOOR) - b
            kind = "odd" if k % 2 else "even"
            (od if k % 2 else ev).append(e); per.setdefault(k, []).append(e)
            if abs(e) > worst[kind][0]: worst[kind] = (abs(e), f"{i} H{k}")
    st = lambda v: (float(np.sqrt(np.mean(np.square(v)))), float(np.max(np.abs(v))), len(v)) if len(v) else (0.0, 0.0, 0)
    return {"gain": st(g), "odd": st(od), "even": st(ev), "worst": worst, "per": {k: st(v) for k, v in sorted(per.items())}}

def per_freq_even(ids, feats):
    out = {}
    for f in FREQS:
        v = []
        for i, m in zip(ids, feats):
            if int(i.split("_f")[1].split("_")[0]) != f: continue
            for k, (a, b) in enumerate(zip(m["h"], F[i]["h"]), start=2):
                if a is None or b is None or b <= FLOOR or k % 2: continue
                v.append(max(a, MFLOOR) - b)
        if v: out[f] = (float(np.sqrt(np.mean(np.square(v)))), float(np.max(np.abs(v))), len(v))
    return out

def fmt(mt):
    g, o, e = mt["gain"], mt["odd"], mt["even"]
    per = " ".join(f"H{k} {v[0]:.1f}/{v[1]:.0f}({v[2]})" for k, v in mt["per"].items())
    return (f"gain rms {g[0]:.3f} max {g[1]:.2f} | odd rms {o[0]:.2f} max {o[1]:.1f} (n {o[2]}) | even rms {e[0]:.2f} max {e[1]:.1f} (n {e[2]})"
            f"  worst odd {mt['worst']['odd'][1]}, even {mt['worst']['even'][1]}\n    per order rms/max(n): {per}")


# ------------------------------------------------------------------------------------------------ candidates
# name: (gate, a2_after, extra names, asym free)
CAND = {
    "D2r": (1, 1, ["thr"], True),
    "PH":  (2, 1, ["p"], True),
    "PH2": (2, 1, [], True),
    "M1":  (0, 1, ["ke"], False),
    "M1a": (0, 1, ["ke"], True),
    "M1r": (1, 1, ["thr", "ke"], True),
    "M1p": (2, 1, ["p", "ke"], True),
    "D2c": (1, 2, ["thr"], True),
    "PHc": (2, 2, ["p"], True),
    "M1pc": (2, 2, ["p", "ke"], True),
}
NL = ["a2", "a3", "sat_db", "q", "asym"]
BOUNDS = {"a2": (1e-8, 5e-3), "a3": (-5e-2, -1e-7), "sat_db": (-6.0, 12.0), "q": (1.5, 30.0), "asym": (-0.5, 0.5),
          "thr": (0.1, 3.0), "p": (0.0, 6.0), "ke": (-0.3, 0.3)}
SCALE = {"a2": 1e-5, "a3": 1e-4, "sat_db": 0.5, "q": 1.0, "asym": 0.01, "thr": 0.1, "p": 0.2, "ke": 0.01}
X0 = {"thr": 0.85, "p": 2.0, "ke": 0.01}
FIXED = {"thr": 1.0, "p": 2.0, "ke": 0.0}

def base_params(k):
    return dict(gain_db=cget("x_gain_db", k), a2=cget("x_a2", k), a3=cget("x_a3", k), sat_db=cget("x_sat_db", k), q=cget("x_q", k),
                asym=cget("x_asym", k), fl_hz=cget("x_fl_hz", k), hs_hz=cget("x_hs_hz", k), hs_db=cget("x_hs_db", k), lp_hz=cget("x_lp_hz", k))

def pvec(d):
    return np.array([d["gain_db"], d["a2"], d["a3"], d["sat_db"], d["q"], d["asym"], d["fl_hz"], d["hs_hz"], d["hs_db"], d["lp_hz"]])

def evaluate(ids, X, d, cand, extra):
    gate, a2_after, _, _ = CAND[cand]
    ex = dict(FIXED); ex.update(extra)
    Y = run_grid(X, FSF, pvec(d), gate, ex["thr"], ex["p"], ex["ke"], a2_after)
    return features(ids, Y)

def fit_core(core, k, cand):
    ids = grid_ids(core); X = stimuli(ids)
    gate, a2_after, extras, asym_free = CAND[cand]
    names = NL + extras
    d0 = base_params(k)
    lo = [BOUNDS[n][0] for n in names]; hi = [BOUNDS[n][1] for n in names]
    if not asym_free:
        d0["asym"] = 0.0; lo[names.index("asym")] = -1e-9; hi[names.index("asym")] = 1e-9
    if gate == 2: d0["asym"] = 0.1      # the decaying forms want a larger asymmetry at the knee (the remanence key's start)
    p0 = [d0[n] for n in NL] + [X0[n] for n in extras]
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
    starts = [p0]
    if asym_free and gate != 2:
        p1 = p0.copy(); p1[names.index("asym")] *= 0.5; starts.append(p1)
    if "ke" in extras:
        p2 = p0.copy(); p2[names.index("ke")] = -X0["ke"]; starts.append(p2)
        p3 = p0.copy(); p3[names.index("ke")] = 0.1 * X0["ke"]; starts.append(p3)   # a small pulse: the fitted values sit near 0.0015
    best = None
    for s in starts:
        r = least_squares(rfun, s, bounds=(lo, hi), x_scale=[SCALE[n] for n in names], diff_step=1e-3, max_nfev=NFEV, loss="soft_l1",
                          f_scale=2.0)
        print(f"    start {dict((n, float(f'{v:.4g}')) for n, v in zip(names, s) if n not in NL or n == 'asym')}: cost {r.cost:.1f} nfev {r.nfev}")
        if best is None or r.cost < best.cost: best = r
    d, ex = unpack(best.x)
    feats = evaluate(ids, X, d, cand, ex)
    mt = metrics(ids, feats)
    print(f"  {core} {cand}: {dict((n, float(f'{v:.6g}')) for n, v in zip(names, best.x))}  cost {best.cost:.1f} (start {0.5 * np.sum(r0**2):.1f})"
          f"  {time.time() - t0:.0f} s")
    print(f"    {fmt(mt)}")
    pf = per_freq_even(ids, feats)
    print("    even per frequency rms/max(n): " + " ".join(f"{f}Hz {v[0]:.1f}/{v[1]:.0f}({v[2]})" for f, v in pf.items()))
    return {"params": {n: float(v) for n, v in zip(names, best.x)}, "metrics": mt, "per_freq_even": pf, "feats": {i: f for i, f in zip(ids, feats)}}


# ------------------------------------------------------------------------------------------------ main
def validate():
    """the mirror (static asymmetry, a2 before the core, stage-2 calibration) against the C++ engine on every grid point"""
    worst = 0.0; worst_id = ""
    for k, core in enumerate(("Nickel", "Iron", "Steel")):
        ids = grid_ids(core); X = stimuli(ids)
        Y = run_grid(X, FSF, pvec(base_params(k)), 0, 1.0, 2.0, 0.0, 0)
        feats = features(ids, Y)
        for i, m in zip(ids, feats):
            c = render_item(ITEMS[i], cal)
            e = abs(m["gain_db"] - c["gain_db"])
            if not np.isfinite(e): e = np.inf
            for a, b in zip(m["h"], c["h"]):
                if a is None or b is None: continue
                if max(a, b) > -100: e = max(e, abs(a - b))
            if e > worst: worst, worst_id = e, i
        print(f"  mirror {core} (stage-2 values): {fmt(metrics(ids, feats))}")
    print(f"  mirror against the C++ engine: worst |difference| {worst:.3f} dB (gain or any harmonic above -100 dBc) at {worst_id}")
    return worst

def pooled(results, cands):
    print("\n== pooled over the three cores (n-weighted rms), even / odd / gain")
    for cand in cands:
        keys = [k for k in results if k.startswith(cand + "/")]
        if len(keys) < 3: continue
        def pool(kind):
            n = sum(results[k]["metrics"][kind][2] for k in keys)
            return float(np.sqrt(sum(results[k]["metrics"][kind][0] ** 2 * results[k]["metrics"][kind][2] for k in keys) / n))
        gn = float(np.sqrt(np.mean([results[k]["metrics"]["gain"][0] ** 2 for k in keys])))
        print(f"  {cand:4s} even {pool('even'):.2f}  odd {pool('odd'):.2f}  gain {gn:.3f}")

def main():
    print("validating the mirror against the C++ engine")
    w = validate()
    if "--validate-only" in ARGS: return
    if w > 0.5:
        print("mirror mismatch above 0.5 dB: stopping"); return
    results = {}
    for cand in VARIANTS:
        gate, a2_after, extras, asym_free = CAND[cand]
        print(f"\n== {cand}: a2 {['before', 'after', 'before, AC-coupled'][a2_after]} the core, gate {['static', 'below thr (D2r)', 'peak-hold decay (PH)'][gate]}, "
              f"asym {'free' if asym_free else 'fixed 0'}, extras {extras}")
        for core in CORES:
            k = ("Nickel", "Iron", "Steel").index(core)
            results[f"{cand}/{core}"] = fit_core(core, k, cand)
    print("\n== summary (grid -12..+21 dBFS x 10 frequencies; test-suite metric)")
    print(f"{'cand':5s} {'core':7s} {'gain rms':>8s} {'gain max':>8s} {'odd rms':>8s} {'odd max':>8s} {'even rms':>8s} {'even max':>8s}")
    for key, r in results.items():
        cand, core = key.split("/"); mt = r["metrics"]
        print(f"{cand:5s} {core:7s} {mt['gain'][0]:8.3f} {mt['gain'][1]:8.2f} {mt['odd'][0]:8.2f} {mt['odd'][1]:8.1f} {mt['even'][0]:8.2f} {mt['even'][1]:8.1f}")
    pooled(results, VARIANTS)
    for core in CORES:
        keys = [k for k in results if k.endswith("/" + core)]
        best = min(keys, key=lambda k: results[k]["metrics"]["even"][0])
        print(f"\n{core}: best even rms {best}; model H2 minus reference H2 where the reference sits above -80 dBc (columns {LEVELS})")
        for f in FREQS[:7]:
            row = []
            for l in LEVELS:
                i = f"xf_{core}_f{f}_{l}"; a = results[best]["feats"][i]["h"][0]; b = F[i]["h"][0]
                row.append("    ." if b <= FLOOR else f"{max(a, MFLOOR) - b:5.1f}")
            print(f"  {f:5d} " + " ".join(row))
        print(f"{core}: {best} model H4 minus reference H4")
        for f in FREQS[:7]:
            row = []
            for l in LEVELS:
                i = f"xf_{core}_f{f}_{l}"; a = results[best]["feats"][i]["h"][2]; b = F[i]["h"][2]
                row.append("    ." if a is None or b is None or b <= FLOOR else f"{max(a, MFLOOR) - b:5.1f}")
            print(f"  {f:5d} " + " ".join(row))
    out = {k: {"params": v["params"], "per_freq_even": v["per_freq_even"],
               "metrics": {kk: vv for kk, vv in v["metrics"].items() if kk not in ("worst", "per")}} for k, v in results.items()}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
