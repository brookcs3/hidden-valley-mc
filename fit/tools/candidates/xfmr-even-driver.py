# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Transformer even harmonics, direction (a): a SYMMETRIC core (x_asym = 0) with the even order coming from the driver stage alone,
either the fixed polynomial a2 x^2, a level-dependent a2, a Class-A stage with its own asymmetric ceiling ahead of the core, or a
Class-A standing current (a constant, or envelope-shifted, DC term in the driver output that the leaky flux integrator turns into a
flux offset). The core, filters and integrator are a numba replica of src/dsp/Transformer.hpp at 48 kHz without oversampling
(validated against the C++ engine through fit/common.render_item, see --validate); src/ is not touched.

Hypothesis under test: the reference's even harmonics can be reproduced by a driver-side asymmetry with a symmetric core, keeping
the odd-order fit (must stay within 3.5 dB rms on the stage-2 grid).

Reference even-order law (extracted first, --extract prints the tables): below the core's onset, H2 is a 1 dB/dB, frequency-independent
line per core (a fixed a2; Iron's ten times Nickel's / Steel's); at the onset points of the 6 dB/octave flux law (20 Hz +3 dBFS,
30 Hz +6, 40 Hz +9, 60 Hz +12, 80 Hz +15, 120 Hz +18, 160 Hz +21) an even component of fixed spectral shape appears (H2 = H4 within
0.3 dB, H6 8 dB lower, H8 28 dB lower) with a magnitude of -60 +- 3 dBc that is the SAME on all three cores, then decays slowly with
drive (Nickel 20 Hz: -59 dBc at +6 to -70 at +24; Steel to -74; Iron only to -64, where its ten-times-larger driver DC term shows).

Candidates (per core, fitted on the stage-2 grid at levels <= +21 dBFS with the stage-2 residual, reported with the test-suite metric
of tests/pb_reference.py: harmonics where the reference is above -80 dBc, even = H2/H4/H6/H8, odd = H3/H5/H7, model floor -120 dBc):
  asym    the current model (knee-hardness asymmetry), the current constants and a refit: the baseline
  sym     symmetric core, driver a2 x^2 + a3 x^3 only
  a2env   symmetric core, a2 (1 + k env): the even term grows with the signal envelope (the germanium Class-A form)
  classA  symmetric core, driver polynomial followed by an asymmetric soft ceiling (cp tanh(y / cp) above, cn below), before the core
  caenv   symmetric core, the repo's own Class-A module (src/dsp/ClassA.hpp): a2 (1 + a2Env env), then the soft ceiling
          y / (1 + |y/C|^8)^(1/8) with C = ceilDb above and ceilDb - asymDb below; a2Env, ceilDb, asymDb fitted
  dc      symmetric core, constant DC term in the driver output (Class-A standing current): flux offset = bias * phi_k
  dcenv   symmetric core, DC term that shifts with the envelope (self-biasing stage): bias0 + bias1 * env
  dcdec   symmetric core, standing bias that collapses with drive (operating point runs toward cutoff): bias0 / (1 + env / E0)
  acdc    symmetric core, AC-coupled driver (the rectified DC of a2 x^2, a2 A^2 / 2, is removed from the drive using the envelope:
          a2 (u^2 - pi^2 env^2 / 8); a real coupling capacitor at 0.2 Hz does the same but its start-up transient (the state has to
          reach the quadrature steady state) is still 0.1 % of A in a 2 s render, which alone makes 2 % of phi_k of flux offset)
          plus a constant standing bias in the primary: flux offset = bias * phi_k
  acdec   symmetric core, AC-coupled driver plus a standing bias that collapses with drive: bias0 / (1 + env / E0)
  acasym  CONTROL (not direction a): the current knee asymmetry with the driver AC-coupled: how much of the deep-saturation even
          excess is the driver's rectified DC (a2 x^2 integrates to a flux offset growing with A^2) rather than the knee
Result (2026-09-27 23:00-23:50, nfev 40, the three cores in parallel, logs build_xfmr_even_driver_{extract,offsetmap,Nickel,Iron,
Steel,post}.log; pooled over the three cores with the test metric, even rms / odd rms / gain rms): the current model 13.04 / 3.54 / 0.042.
Driver-only even order with a symmetric core is REFUTED: sym 18.99 / 3.42, a2env 19.19 / 3.42, classA 18.47 / 2.11, caenv 19.12 / 2.02,
every one worse than the current model, because the onset term is missing (20 Hz +3: sym H2 -101.7 dBc Nickel, -85.2 Iron, against
-60.5 / -60.4 in the reference). A standing bias reproduces the onset spike (dc: -70 / -62 dBc) but is then 12-23 dB too loud deep in
(Nickel 20 Hz +12: -53.0 against -64.4; Iron 60 Hz +18: -48.6 against -66.5): dc 14.19, dcenv 12.74, dcdec 15.20, acdc 13.17,
acdec 12.58 (the best driver-side form, 0.5 dB better than the current model, even max 29.9). The structural reason is in --offsetmap:
the equivalent static flux offset the reference needs is 1.1-1.6 % of phi_k 3 dB below onset, 0.8-1.3 % at onset, 0.13-0.3 % at +3 dB and
0.01-0.07 % from +6 dB on, and at ONE level it spans 100x across frequency (Nickel +18 dBFS: 0.093 % at 20 Hz, 0.028 % at 40, 0.010 % at
60, 1.25 % at 120 Hz), whereas the driver sees the same signal at every frequency and so delivers one offset per level. The control
acasym (the current knee asymmetry, the driver's rectified a2 x^2 DC kept out of the flux) is the best figure here: 9.96 / 3.46 / 0.040
(Nickel 10.03, Iron 9.37, Steel 10.62), so the driver's DC integrating into the flux is a real part of the deep-saturation excess, but the
frequency-dependent collapse remains (even max 24.6-31.2, e.g. Nickel 60 Hz +18 H4 -53.5 against -94.1). Side finding: a symmetric soft
ceiling at about 36 dBFS in Iron's driver (classA, caenv) cuts Iron's odd rms 4.95 -> 2.28 and odd max 49.4 -> 18.5 (xf_Iron_f1000_21 H5).
usage: cd hidden-valley-mc && python3 -u fit/tools/candidates/xfmr-even-driver.py [--extract] [--validate] [--offsetmap] [--fit]
       [--cores=Nickel,Iron] [--cands=asym,sym,dc,...] [--nfev=60] [--refit-asym]   (no flags: everything)"""
import os, sys, time, numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item, feat_residual, protocol, FS  # noqa: E402

CORES = ["Nickel", "Iron", "Steel"]
FREQS = [20, 30, 40, 60, 80, 120, 160, 320, 1000, 5000]
LEVELS = list(range(-12, 25, 3))
FIT_MAX_LEVEL = 21          # the +24 dBFS points hit the reference's output ceiling (tests: xfmr_ceiling), not modelled
HARM_FLOOR = -80.0; MODEL_FLOOR = -120.0
M_ASYM, M_SYM, M_A2ENV, M_CLASSA, M_DC, M_DCENV, M_DCDEC, M_ACDC, M_ACDEC, M_ACASYM, M_CAENV = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
MODES = {"asym": M_ASYM, "sym": M_SYM, "a2env": M_A2ENV, "classA": M_CLASSA, "dc": M_DC, "dcenv": M_DCENV, "dcdec": M_DCDEC,
         "acdc": M_ACDC, "acdec": M_ACDEC, "acasym": M_ACASYM, "caenv": M_CAENV}
cal0 = load_cal()

def getf(cal, name, k):
    return float(cal[MODEL.fields[name][0] + k])


@njit(cache=True)
def sat(ph, phik, q):
    az = abs(ph / phik)
    if az <= 0.02:
        return ph
    return ph / np.exp(np.log1p(np.exp(q * np.log(az))) / q)


@njit(cache=True)
def run_path(x, fs, g, a2, a3, r, phik, qp, qn, hs_g, hs_gain, hs_on, lp_g, lp_on, mode, p1, p2, p3):
    """x -> gain -> driver (mode) -> flux core (symmetric unless qp != qn) -> high shelf -> low pass. Returns y (float64)."""
    n = x.shape[0]
    y = np.empty(n)
    T = 1.0 / fs
    phi = 0.0; sprev = 0.0; hs_s = 0.0; lp_s = 0.0; env = 0.0
    kEnv = 1.0 - np.exp(-1.0 / ((1.0 / (2.0 * np.pi * 5.0)) * fs))
    cp = 10.0 ** (p1 / 20.0); cn = 10.0 ** (p2 / 20.0); e0 = 10.0 ** (p2 / 20.0)
    cpa = 10.0 ** (p2 / 20.0); cna = cpa * 10.0 ** (-p3 / 20.0)   # caenv: the ClassA.hpp ceiling per polarity (ceilDb, asymDb)
    kdc = np.pi * np.pi / 8.0   # <u^2> = A^2 / 2 with A = pi env / 2 for a sine
    for i in range(n):
        u = g * x[i]
        env += (abs(u) - env) * kEnv
        a2e = a2
        if mode == 2 or mode == 10:
            a2e = a2 * (1.0 + p1 * env)
        v = u + a2e * u * u + a3 * u * u * u
        if mode == 3:
            if v > 0.0:
                v = cp * np.tanh(v / cp)
            else:
                v = -cn * np.tanh(-v / cn)
        elif mode == 10:
            # src/dsp/ClassA.hpp: y / (1 + |y / C|^8)^(1/8), C = cp above, cp 10^(-asymDb/20) below
            C = cpa if v > 0.0 else cna
            v = v / (1.0 + abs(v / C) ** 8.0) ** 0.125
        elif mode == 4:
            v += p1 * r * phik
        elif mode == 5:
            v += (p1 + p2 * env) * r * phik
        elif mode == 6:
            v += p1 / (1.0 + env / e0) * r * phik
        elif mode == 7 or mode == 8 or mode == 9:
            # AC-coupled driver: the rectified DC of the even term does not reach the primary; then the standing bias (constant,
            # or collapsing with the envelope)
            v -= a2 * kdc * env * env
            if mode == 7:
                v += p1 * r * phik
            elif mode == 8:
                v += p1 / (1.0 + env / e0) * r * phik
        phi += T * (v - r * phi)
        s = sat(phi, phik, qp if phi > 0.0 else qn)
        yc = (s - sprev) * fs
        sprev = s
        # TPT high shelf
        if hs_on:
            vv = (yc - hs_s) * hs_g / (1.0 + hs_g); lpv = vv + hs_s; hs_s = lpv + vv
            yc = lpv + hs_gain * (yc - lpv)
        if lp_on:
            vv = (yc - lp_s) * lp_g / (1.0 + lp_g); lpv = vv + lp_s; lp_s = lpv + vv
            yc = lpv
        y[i] = yc
    return y


class Path:
    """one core's path from the calibration vector, with the nonlinear parameters overridable"""
    def __init__(self, cal, k):
        self.k = k
        self.gain_db = getf(cal, "x_gain_db", k); self.fl = getf(cal, "x_fl_hz", k)
        self.hs_hz = getf(cal, "x_hs_hz", k); self.hs_db = getf(cal, "x_hs_db", k); self.lp_hz = getf(cal, "x_lp_hz", k)
        self.a2 = getf(cal, "x_a2", k); self.a3 = getf(cal, "x_a3", k); self.sat_db = getf(cal, "x_sat_db", k)
        self.q = getf(cal, "x_q", k); self.asym = getf(cal, "x_asym", k)

    def render(self, x, fs, mode, p1=0.0, p2=0.0, p3=0.0, a2=None, a3=None, sat_db=None, q=None, asym=None):
        a2 = self.a2 if a2 is None else a2; a3 = self.a3 if a3 is None else a3
        sat_db = self.sat_db if sat_db is None else sat_db; q = self.q if q is None else q
        asym = (self.asym if asym is None else asym) if mode in (M_ASYM, M_ACASYM) else 0.0
        g = 10 ** (self.gain_db / 20); r = 2 * np.pi * self.fl
        phik = 10 ** (sat_db / 20) / (2 * np.pi * 20.0)
        qp = max(q * (1 + asym), 1.0); qn = max(q * (1 - asym), 1.0)
        hs_on = self.hs_hz > 0; hs_g = np.tan(np.pi * min(self.hs_hz, 0.49 * fs) / fs) if hs_on else 0.0
        lp_on = self.lp_hz > 0; lp_g = np.tan(np.pi * min(self.lp_hz, 0.49 * fs) / fs) if lp_on else 0.0
        return run_path(np.asarray(x, dtype=np.float64), float(fs), g, a2, a3, r, phik, qp, qn, hs_g, 10 ** (self.hs_db / 20), hs_on,
                        lp_g, lp_on, mode, float(p1), float(p2), float(p3))


def grid_items(core, max_level=FIT_MAX_LEVEL):
    return [f"xf_{core}_f{f}_{l}" for f in FREQS for l in LEVELS if l <= max_level]


def feature_of(item_id, y, fs=FS):
    yy = np.stack([y, y]).astype(np.float32).astype(np.float64)   # the engine outputs float32 like the plugin
    return protocol.feature(ITEMS[item_id], yy, fs)


def render_feats(path, core, mode, p, max_level=FIT_MAX_LEVEL):
    """p: dict of overrides (a2, a3, sat_db, q, asym, p1, p2) -> {item_id: feature}"""
    out = {}
    for i in grid_items(core, max_level):
        it = ITEMS[i]
        x = protocol.stimulus(it["stim"], it["fs"])[0].astype(np.float32).astype(np.float64)
        y = path.render(x, it["fs"], mode, **p)
        out[i] = feature_of(i, y, it["fs"])
    return out


def metric(feats, core):
    """the test-suite metric on the grid at levels <= FIT_MAX_LEVEL: even/odd rms and max where the reference is above -80 dBc,
    gain rms; plus a 'quiet' figure: mean excess of the model's even harmonics where the reference sits between -100 and -80 dBc"""
    ev, od, gn, quiet = [], [], [], []
    for i, m in feats.items():
        ref = F[i]
        gn.append(m["gain_db"] - ref["gain_db"])
        for k, (a, b) in enumerate(zip(m["h"], ref["h"]), start=2):
            if a is None or b is None:
                continue
            a = max(a, MODEL_FLOOR)
            if b > HARM_FLOOR:
                (od if k % 2 else ev).append(a - b)
            elif b > -100.0 and k % 2 == 0:
                quiet.append(max(a - b, 0.0))
    ev, od, gn, quiet = map(np.asarray, (ev, od, gn, quiet))
    rms = lambda v: float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")
    mx = lambda v: float(np.max(np.abs(v))) if len(v) else float("nan")
    return {"even_rms": rms(ev), "even_max": mx(ev), "odd_rms": rms(od), "odd_max": mx(od), "gain_rms": rms(gn), "gain_max": mx(gn),
            "quiet_excess_mean": float(np.mean(quiet)) if len(quiet) else float("nan"), "n_even": len(ev), "n_odd": len(od)}


def fmt(m):
    return (f"even rms {m['even_rms']:5.2f} max {m['even_max']:5.1f} | odd rms {m['odd_rms']:5.2f} max {m['odd_max']:5.1f} | "
            f"gain rms {m['gain_rms']:5.3f} max {m['gain_max']:5.2f} | quiet excess {m['quiet_excess_mean']:5.1f} dB (n even {m['n_even']}, odd {m['n_odd']})")


# ------------------------------------------------------------------------------------------------ the reference law
def extract():
    print("=== Reference even-order law (H2 dBc; rows level dBFS, cols Hz). '*' marks the 6 dB/octave onset points.")
    onset = {20: 3, 30: 6, 40: 9, 60: 12, 80: 15, 120: 18, 160: 21}
    for core in CORES:
        for name, idx in (("H2", 0), ("H4", 2), ("H3", 1)):
            print(f"\n{core} {name}")
            print("lvl   " + " ".join(f"{f:>7d}" for f in FREQS))
            for l in LEVELS:
                row = []
                for f in FREQS:
                    h = F[f"xf_{core}_f{f}_{l}"]["h"][idx]
                    mark = "*" if onset.get(f) == l else " "
                    row.append("   None" if h is None else f"{h:6.1f}{mark}")
                print(f"{l:4d}  " + " ".join(row))
    print("\n=== Below onset: H2 slope against level (dB per dB) at 1 kHz and 320 Hz, -12..+18 dBFS, per core")
    for core in CORES:
        for f in (1000, 320):
            h = [F[f"xf_{core}_f{f}_{l}"]["h"][0] for l in range(-12, 19, 3)]
            sl = np.polyfit(range(-12, 19, 3), h, 1)[0]
            print(f"  {core:6s} {f:5d} Hz: slope {sl:5.2f} dB/dB, H2 at 0 dBFS {h[4]:7.1f} dBc -> a2 ~ {2 * 10 ** (h[4] / 20):.2e}")
    print("\n=== At the onset points: H2 H4 H6 H8 | H3 H5 H7 per core (the even component's shape and size, and its sameness across cores)")
    for f, l in onset.items():
        for core in CORES:
            h = F[f"xf_{core}_f{f}_{l}"]["h"]
            print(f"  {f:4d} Hz {l:+3d} {core:6s}  even {h[0]:6.1f} {h[2]:6.1f} {h[4]:6.1f} {h[6]:6.1f} | odd {h[1]:6.1f} {h[3]:6.1f} {h[5]:6.1f}")
    print("\n=== Deep saturation at 20 Hz: H2 against level per core (the decay law beyond the onset)")
    for core in CORES:
        print(f"  {core:6s} " + " ".join(f"{l:+3d}:{F[f'xf_{core}_f20_{l}']['h'][0]:6.1f}" for l in range(3, 25, 3)))
    print("\nWhat a mechanism must produce: (1) a fixed per-core a2 below onset; (2) at the flux-law onset an even component of about -60 dBc")
    print("with H2 = H4, independent of the core and of the driver's a2 (Iron's a2 is 10x and its onset H2 is the same); (3) a slow decay")
    print("of that component with drive (about 0.6 dB/dB for Nickel/Steel at 20 Hz), Iron's held up by its larger driver DC term.")


# ------------------------------------------------------------------------------------------------ validation against the C++ engine
def validate():
    print("=== Python path against the C++ engine (current constants, mode asym), a few grid items")
    worst = 0.0
    for core, k in zip(CORES, range(3)):
        path = Path(cal0, k)
        for i in (f"xf_{core}_f20_0", f"xf_{core}_f20_12", f"xf_{core}_f60_18", f"xf_{core}_f1000_6", f"xf_{core}_f5000_-12"):
            it = ITEMS[i]
            mc = render_item(it, cal0)
            x = protocol.stimulus(it["stim"], it["fs"])[0].astype(np.float32).astype(np.float64)
            mp = feature_of(i, path.render(x, it["fs"], M_ASYM), it["fs"])
            d = [abs(a - b) for a, b in zip(mc["h"], mp["h"]) if a is not None and b is not None and max(a, b) > -110]
            dg = abs(mc["gain_db"] - mp["gain_db"]); worst = max(worst, dg, max(d) if d else 0.0)
            hh = lambda a, b: "   None/   None" if a is None or b is None else f"{a:7.1f}/{b:7.1f}"
            print(f"  {i:22s} gain C++ {mc['gain_db']:7.3f} py {mp['gain_db']:7.3f} | H2 {hh(mc['h'][0], mp['h'][0])} "
                  f"H3 {hh(mc['h'][1], mp['h'][1])} H4 {hh(mc['h'][2], mp['h'][2])} H5 {hh(mc['h'][3], mp['h'][3])} | worst diff {max(d) if d else 0:.2f} dB")
    print(f"  worst difference (harmonics above -110 dBc and gain): {worst:.3f} dB")


# ------------------------------------------------------------------------------------------------ the law in mechanism terms
def offset_map():
    """the static flux offset (fraction of phi_k) a DC-free symmetric core with the current a2/a3/sat/q needs at each grid point to
    reproduce the reference's H2: what any asymmetry mechanism must deliver, against excess over the onset law and frequency"""
    print("=== Required static flux offset (% of phi_k) for the reference's H2, DC-free symmetric core (mode acdc, current constants).")
    print("    rows: level minus the onset law 3 + 20 log10(f / 20) dBFS; '<0' = the model is above the reference with zero offset (H2 model>ref)")
    onset = lambda f: 3 + 20 * np.log10(f / 20)
    freqs = [20, 30, 40, 60, 80, 120, 160]
    for core, k in zip(CORES, range(3)):
        path = Path(cal0, k)
        def h2(f, l, bias):
            i = f"xf_{core}_f{f}_{l}"; it = ITEMS[i]
            x = protocol.stimulus(it["stim"], it["fs"])[0]
            return feature_of(i, path.render(x, it["fs"], M_ACDC, p1=bias), it["fs"])["h"][0]
        print(f"\n{core}: excess | " + " ".join(f"{f:>15d}" for f in freqs))
        for ex in (-3, 0, 3, 6, 9, 12, 15):
            row = []
            for f in freqs:
                l = ex + onset(f); lg = min(LEVELS, key=lambda v: abs(v - l))
                if abs(lg - l) > 1.6 or lg > FIT_MAX_LEVEL:
                    row.append("."); continue
                ref = F[f"xf_{core}_f{f}_{lg}"]["h"][0]; zero = h2(f, lg, 0.0)
                if zero >= ref:
                    row.append(f"<0({zero:5.1f}>{ref:5.1f})"); continue
                lo, hi = 1e-5, 0.3
                for _ in range(16):
                    mid = np.sqrt(lo * hi)
                    if h2(f, lg, mid) < ref: lo = mid
                    else: hi = mid
                row.append(f"{100 * np.sqrt(lo * hi):6.3f}%@{lg:+3d}")
            print(f"  {ex:+3d}    | " + " ".join(f"{r:>15s}" for r in row))
        # the decisive test of any driver-side (level-only) law: at ONE level the driver sees the same signal at every frequency, so
        # a DC, envelope or ceiling mechanism delivers one flux offset per level; the reference needs a different one per frequency
        print(f"  {core}: required offset at EQUAL LEVEL across frequency (a level-only mechanism gives one value per row)")
        for lg in (9, 12, 15, 18, 21):
            row = []
            for f in freqs:
                if lg - onset(f) < -3.5:
                    row.append("."); continue
                ref = F[f"xf_{core}_f{f}_{lg}"]["h"][0]; zero = h2(f, lg, 0.0)
                if zero >= ref:
                    row.append("<0"); continue
                lo, hi = 1e-5, 0.3
                for _ in range(16):
                    mid = np.sqrt(lo * hi)
                    if h2(f, lg, mid) < ref: lo = mid
                    else: hi = mid
                row.append(f"{100 * np.sqrt(lo * hi):6.3f}%")
            print(f"  {lg:+3d} dBFS | " + " ".join(f"{r:>15s}" for r in row))


# ------------------------------------------------------------------------------------------------ fitting
PARAMS = {
    #          names                              lo                                  hi                               x_scale
    "asym":   (["a2", "a3", "sat_db", "q", "asym"], [-5e-3, -5e-2, -6.0, 1.5, -0.5], [5e-3, 5e-2, 12.0, 30.0, 0.5], [1e-5, 1e-4, 0.5, 1.0, 0.01]),
    "sym":    (["a2", "a3", "sat_db", "q"], [-5e-3, -5e-2, -6.0, 1.5], [5e-3, 5e-2, 12.0, 30.0], [1e-5, 1e-4, 0.5, 1.0]),
    "a2env":  (["a2", "a3", "sat_db", "q", "p1"], [-5e-3, -5e-2, -6.0, 1.5, -0.5], [5e-3, 5e-2, 12.0, 30.0, 20.0], [1e-5, 1e-4, 0.5, 1.0, 0.1]),
    "classA": (["a2", "a3", "sat_db", "q", "p1", "p2"], [-5e-3, -5e-2, -6.0, 1.5, 6.0, 6.0], [5e-3, 5e-2, 12.0, 30.0, 60.0, 60.0], [1e-5, 1e-4, 0.5, 1.0, 1.0, 1.0]),
    "dc":     (["a2", "a3", "sat_db", "q", "p1"], [-5e-3, -5e-2, -6.0, 1.5, -0.3], [5e-3, 5e-2, 12.0, 30.0, 0.3], [1e-5, 1e-4, 0.5, 1.0, 1e-3]),
    "dcenv":  (["a2", "a3", "sat_db", "q", "p1", "p2"], [-5e-3, -5e-2, -6.0, 1.5, -0.3, -0.3], [5e-3, 5e-2, 12.0, 30.0, 0.3, 0.3], [1e-5, 1e-4, 0.5, 1.0, 1e-3, 1e-3]),
    "dcdec":  (["a2", "a3", "sat_db", "q", "p1", "p2"], [-5e-3, -5e-2, -6.0, 1.5, -0.3, -20.0], [5e-3, 5e-2, 12.0, 30.0, 0.3, 40.0], [1e-5, 1e-4, 0.5, 1.0, 1e-3, 1.0]),
    "acdc":   (["a2", "a3", "sat_db", "q", "p1"], [-5e-3, -5e-2, -6.0, 1.5, -0.3], [5e-3, 5e-2, 12.0, 30.0, 0.3], [1e-5, 1e-4, 0.5, 1.0, 1e-3]),
    "acdec":  (["a2", "a3", "sat_db", "q", "p1", "p2"], [-5e-3, -5e-2, -6.0, 1.5, -0.3, -20.0], [5e-3, 5e-2, 12.0, 30.0, 0.3, 40.0], [1e-5, 1e-4, 0.5, 1.0, 1e-3, 1.0]),
    "acasym": (["a2", "a3", "sat_db", "q", "asym"], [-5e-3, -5e-2, -6.0, 1.5, -0.5], [5e-3, 5e-2, 12.0, 30.0, 0.5], [1e-5, 1e-4, 0.5, 1.0, 0.01]),
    "caenv":  (["a2", "a3", "sat_db", "q", "p1", "p2", "p3"], [-5e-3, -5e-2, -6.0, 1.5, -0.5, 6.0, -6.0], [5e-3, 5e-2, 12.0, 30.0, 20.0, 60.0, 6.0],
               [1e-5, 1e-4, 0.5, 1.0, 0.1, 1.0, 0.1]),
}
STARTS = {"a2env": [[0.5], [3.0]], "classA": [[30.0, 24.0], [40.0, 30.0]], "dc": [[0.01], [-0.01]], "dcenv": [[0.01, -0.001], [0.01, 0.001]],
          "dcdec": [[0.01, 6.0], [0.02, 0.0]], "acdc": [[0.01], [-0.01]], "acdec": [[0.01, 6.0], [0.02, 0.0]],
          "caenv": [[3.0, 24.0, 1.0], [0.5, 30.0, 3.0]]}


def fit_candidate(core, k, cand, nfev, verbose=False):
    path = Path(cal0, k); mode = MODES[cand]
    names, lo, hi, xs = PARAMS[cand]
    items = grid_items(core)
    base = {"a2": path.a2, "a3": path.a3, "sat_db": path.sat_db, "q": path.q, "asym": path.asym, "p1": 0.0, "p2": 0.0, "p3": 0.0}

    def resid(pv):
        p = dict(zip(names, pv))
        fe = render_feats(path, core, mode, p)
        return np.concatenate([feat_residual(i, fe[i]) for i in items])

    best = None
    for extra in STARTS.get(cand, [[]]):
        p0 = [base[n] for n in names]
        for j, v in enumerate(extra):
            p0[len(names) - len(extra) + j] = v
        p0 = np.clip(p0, lo, hi)
        t0 = time.time()
        r = least_squares(resid, p0, bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=2.0)
        if verbose:
            print(f"    start {extra}: cost {r.cost:.2f} nfev {r.nfev} ({time.time() - t0:.0f} s) -> {dict(zip(names, [float(f'{v:.5g}') for v in r.x]))}")
        if best is None or r.cost < best.cost:
            best = r
    p = dict(zip(names, best.x))
    fe = render_feats(path, core, mode, p)
    return p, metric(fe, core), fe, float(best.cost)


def show_points(fe, core, label):
    """the task's diagnostic points: even harmonics where the reference is quiet (-100 to -75 dBc) but the model is loud"""
    rows = []
    for i, m in fe.items():
        ref = F[i]
        for k in (0, 2):
            a, b = m["h"][k], ref["h"][k]
            if a is None or b is None:
                continue
            if -100.0 < b <= -75.0 and max(a, MODEL_FLOOR) - b > 10.0:
                rows.append((i, k + 2, max(a, MODEL_FLOOR), b))
    rows.sort(key=lambda t: t[3] - t[2])
    print(f"    {label}: {len(rows)} even points with the model >10 dB above a reference at or below -75 dBc; worst 6:")
    for i, h, a, b in rows[:6]:
        print(f"      {i:22s} H{h} model {a:7.1f} ref {b:7.1f}")


def main():
    args = sys.argv[1:]
    def opt(name, default):
        for a in args:
            if a.startswith(f"--{name}="):
                return a.split("=", 1)[1]
        return default
    cores = opt("cores", "Nickel,Iron,Steel").split(",")
    cands = opt("cands", "asym,sym,a2env,classA,caenv,dc,dcenv,dcdec,acdc,acdec,acasym").split(",")
    nfev = int(opt("nfev", "60"))
    if "--extract" in args or len(args) == 0:
        extract()
    if "--validate" in args or len(args) == 0:
        validate()
    if "--offsetmap" in args or len(args) == 0:
        offset_map()
    if "--fit" in args or len(args) == 0:
        results = {}
        for core in cores:
            k = CORES.index(core)
            path = Path(cal0, k)
            print(f"\n=== {core}: current constants (asym knee), Python path, levels <= +{FIT_MAX_LEVEL}")
            fe = render_feats(path, core, M_ASYM, {})
            m0 = metric(fe, core); print("    " + fmt(m0)); show_points(fe, core, "current")
            results[(core, "current")] = m0
            for cand in cands:
                if cand == "asym" and "--refit-asym" not in args:
                    continue
                t0 = time.time()
                print(f"\n--- {core} {cand}: fitting {PARAMS[cand][0]} (max_nfev {nfev})")
                p, m, fe, cost = fit_candidate(core, k, cand, nfev, verbose=True)
                print(f"    fitted {dict((n, float(f'{v:.5g}')) for n, v in p.items())}  stage-2 cost {cost:.2f}  ({time.time() - t0:.0f} s)")
                print("    " + fmt(m)); show_points(fe, core, cand)
                results[(core, cand)] = m
        print("\n=== Summary (test metric: levels <= +21, reference harmonics above -80 dBc)")
        print(f"{'core':7s} {'candidate':9s} {'even rms':>8s} {'even max':>8s} {'odd rms':>8s} {'odd max':>8s} {'gain rms':>8s} {'quiet+':>7s}")
        for (core, cand), m in results.items():
            print(f"{core:7s} {cand:9s} {m['even_rms']:8.2f} {m['even_max']:8.1f} {m['odd_rms']:8.2f} {m['odd_max']:8.1f} {m['gain_rms']:8.3f} {m['quiet_excess_mean']:7.1f}")


if __name__ == "__main__":
    main()
