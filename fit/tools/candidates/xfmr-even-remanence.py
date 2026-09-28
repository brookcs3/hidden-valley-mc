# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Transformer even-harmonic candidate, key "xfmr-even-b-remanence" (direction (b): history-dependent asymmetry).

Hypothesis. The reference's even harmonics on the transformer grid (fit/data/reference_features.json, xf_{core}_f{freq}_{level}) do not
grow with saturation depth the way the model's static knee asymmetry (x_asym, qp/qn = q (1 +- asym)) makes them grow: they are the
driver's a2 (1 dB per dB, frequency independent) below the knee, a spike with a FLAT even spectrum (H2 = H4 = -57..-60 dBc, the same on
all three cores at 20..80 Hz whatever the level the onset happens at) at the onset of saturation, and a floor of -63..-85 dBc once the
core is through the knee that falls about 9 dB per octave with frequency at constant depth (Nickel/Steel; Iron's floor is its own
driver's a2). Direction (b) says the asymmetry that makes the onset spike must depend on recent history, so that it fades once the core
is driven through the knee every half cycle: a flux offset (remanence) that builds with the signal and relaxes with a time constant,
and the variant where the asymmetry parameter itself falls with the tracked saturation depth.

Result (see the SUMMARY the run prints and the .result.<core>.json files). The remanence proper (V2, V6: a relaxing flux offset built
from the excess flux) is refuted: its net offset is a DC bias proportional to the rectified excess, so its even orders grow with depth
and the even residual gets worse than the static knee (14.7..31 dB rms against 11.8..13.7). The depth-tracking asymmetry (V5, V8: the
knee asymmetry divided by a held measure of how far past the knee the core has been driven, with the driver AC-coupled so a2 cannot
magnetise the core) halves the even residual (about 6..7 dB rms, from 11.8..13.7) at unchanged odd residual and gain; V9 is the same
form with its three shape constants fixed (p = 2, 1 Hz, 300 ms), so it adds no calibration field and only refits the stage-2 five.
What remains (max 16..27 dB) is the frequency dependence of the deep-saturation floor and the 120/160 Hz +21 spikes, which no function
of depth alone produces.

What this file does, in order:
  1. extracts the reference law: per core, H2/H4 against level at each frequency and against frequency at constant gain reduction, the
     sub-knee slope of H2 (dB per dB), the onset spike and its spectrum flatness (H2 - H4), and the deep-saturation floor;
  2. mirrors src/dsp/Transformer.hpp (gain, driver polynomial, flux core, shelf, low pass, float32 in/out) in numba and validates the
     mirror against the C++ engine (fit/common.py render_item) on the current constants;
  3. decomposes the current model's even harmonics: static asym on/off, driver DC through the leaky integrator on/off;
  4. fits candidate topologies per core on the grid (levels -12..+21 dBFS; +24 is the reference's output ceiling, pooled apart by the
     test suite) with the stage-2 parameters (a2, a3, sat_db, q, asym) plus the candidate's own, and reports the test-suite metrics
     (tests/pb_reference.py residuals: even/odd pooled where the reference is above -80 dBc, model floored at -120 dBc) and a
     "quiet" even metric over all grid points with both sides floored at -100 dBc, per core.

Candidates (variant ids):
  V0  the C++ as it is (static knee asymmetry), refitted here as the control
  V1  V0 + the driver AC-coupled (one-pole high pass at c1 Hz before the flux integrator: the a2 term's DC cannot magnetise the core)
  V2  remanence: symmetric knee; a flux offset state  off' = kb * excess * (1 + eps * sign(excess)) - off / tau, excess = phi - S(phi),
      applied as S(phi + off); c = (kb, tau_ms, eps)
  V3  depth-decaying asymmetry: qp/qn = q (1 +- asym_eff), asym_eff = asym / (1 + (env / d0)^2), env a peak tracker of |phi| / phi_k
      with release tau_env; c = (d0, tau_env_ms)
  V4  excess drain (a relaxing offset in the flux itself): phi' = v - r phi - kc * (phi - S(phi)), static asym; c = (kc)
  V5  V3 + V1 (depth-decaying asymmetry with the driver AC-coupled)
  V6  V2 + V1
  V7  memoryless knee-confined asymmetry (round 2): asym_eff(phi) = asym / (1 + (|phi| / phi_k / d0)^p) evaluated per sample, driver
      AC-coupled; c = (d0, p, fdc_hz). A static curve whose two polarities differ only on the approach to the knee.
  V8  peak-hold power-law decay (round 2): asym_eff = asym / max(env, 0.3)^p, env the peak of |phi| / phi_k held with release tau_env,
      driver AC-coupled; c = (p, fdc_hz, tau_env_ms). The asymmetry seen by the core falls with the level it is driven to.
  V9  the recommended fixed form (round 3): V8 with p = 2, fdc = 1 Hz and tau_env = 300 ms fixed, so only the stage-2 fields
      (a2, a3, sat_db, q, asym) are fitted: no new calibration field.

usage: cd <repo> && python3 -u fit/tools/candidates/xfmr-even-remanence.py [--explore] [--variants V0,V1,...] [--cores Nickel,Iron,Steel]
                                                                       [--nfev 40] [--quick]
  --explore   the law extraction, the mirror validation and the decomposition only (no fits)
  --quick     every other frequency in the fit (fast look)
  --starts N  multi-start on the asymmetry (1: as fitted; 3: x1, x0.5, x2), the lowest cost kept
The fits are slow (about 8 min per start of an 8-parameter variant); run one process per core and variant in parallel, e.g.
  for c in Nickel Iron Steel; do for v in V0 V2 V6 V3 V5 V8 V9; do python3 -u fit/tools/candidates/xfmr-even-remanence.py --cores $c \
      --variants $v --starts 3 > build_xfmr_even_${c}_$v.log 2>&1 & done; done
Each writes xfmr-even-remanence.result.<core>.<variant>.json next to this file.
"""
import json, os, sys, time
import numpy as np
from numba import njit
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "measure"))
from common import F, ITEMS, MODEL, load_cal, render_item, protocol  # noqa: E402

ARGS = sys.argv[1:]
def opt(name, default):
    for i, a in enumerate(ARGS):
        if a == name and i + 1 < len(ARGS): return ARGS[i + 1]
        if a.startswith(name + "="): return a[len(name) + 1:]
    return default
EXPLORE = "--explore" in ARGS
QUICK = "--quick" in ARGS
VARIANTS = opt("--variants", "V0,V1,V2,V3,V4,V5,V6,V7,V8,V9").split(",")
CORES = opt("--cores", "Nickel,Iron,Steel").split(",")
NFEV = int(opt("--nfev", "40"))
STARTS = int(opt("--starts", "1"))       # asymmetry starts: x1, x0.5, x2 (and their negatives are equivalent in magnitude)

FS = 48000.0
ALL_FREQS = [20, 30, 40, 60, 80, 120, 160, 320, 1000, 5000]
LEVELS = list(range(-12, 25, 3))
FIT_LEVELS = [l for l in LEVELS if l < 24]          # +24 dBFS: the reference's output ceiling (tests/pb_reference.py CEILING_DBFS)
CORE_INDEX = {"Nickel": 0, "Iron": 1, "Steel": 2}
LIN = ["x_gain_db", "x_fl_hz", "x_hs_hz", "x_hs_db", "x_lp_hz"]
NL = ["x_a2", "x_a3", "x_sat_db", "x_q", "x_asym"]
HARM_FLOOR = -80.0; MODEL_FLOOR = -120.0; QUIET_FLOOR = -100.0

def ref_h(core, f, lvl, k):
    """reference harmonic H(k) in dBc (k = 2..8), or None"""
    return F[f"xf_{core}_f{f}_{lvl}"]["h"][k - 2]

def ref_gain(core, f, lvl):
    return F[f"xf_{core}_f{f}_{lvl}"]["gain_db"]

# ================================================================================================ 1. the reference law
def extract_law():
    print("=" * 110)
    print("1. REFERENCE EVEN-ORDER LAW (H2 / H4 in dBc; gain in dB)")
    print("=" * 110)
    for core in ("Nickel", "Iron", "Steel"):
        print(f"\n--- {core}: H2 | H4 | H3 against level, rows = frequency (levels {LEVELS})")
        for f in ALL_FREQS:
            cells = []
            for l in LEVELS:
                h2, h4, h3 = ref_h(core, f, l, 2), ref_h(core, f, l, 4), ref_h(core, f, l, 3)
                cells.append(f"{h2:6.1f}/{h4:6.1f}/{h3:6.1f}")
            print(f"{f:5d} " + " ".join(cells))
        # sub-knee slope of H2 against level: points with H3 < -70 dBc (no core action) and H2 above the -110 dBc scatter
        xs, ys = [], []
        for f in ALL_FREQS:
            for l in LEVELS:
                if ref_h(core, f, l, 3) < -70 and ref_h(core, f, l, 2) > -110:
                    xs.append(l); ys.append(ref_h(core, f, l, 2))
        A = np.vstack([xs, np.ones(len(xs))]).T; sl, ic = np.linalg.lstsq(A, ys, rcond=None)[0]
        res = np.array(ys) - (sl * np.array(xs) + ic)
        print(f"  sub-knee H2 law: {sl:.3f} dB/dB, H2(0 dBFS) = {ic:.1f} dBc, scatter {np.std(res):.2f} dB over {len(xs)} points"
              f"  (a2 = {2 * 10 ** (ic / 20):.3e} from H2 = a2 A / 2)")
        # the onset spike: per frequency, the level with the largest H2 and its flatness
        print("  onset spike per frequency: level, H2, H4, H2-H4, H3, gain")
        for f in ALL_FREQS:
            cand = [(ref_h(core, f, l, 2), l) for l in FIT_LEVELS]
            h2, l = max(cand)
            print(f"    {f:5d} Hz: +{l:>3d} dBFS  H2 {h2:6.1f}  H4 {ref_h(core, f, l, 4):6.1f}  flat {h2 - ref_h(core, f, l, 4):5.1f}"
                  f"  H3 {ref_h(core, f, l, 3):6.1f}  gain {ref_gain(core, f, l):6.2f}")
        # deep saturation: H2 against frequency at (about) constant gain reduction
        print("  deep saturation: H2/H4 at the grid point nearest each gain-reduction depth (GR dB: freq -> level H2/H4)")
        for depth in (2.0, 4.5, 7.0, 10.0):
            row = []
            for f in ALL_FREQS[:7]:
                best = None
                for l in FIT_LEVELS:
                    gr = ref_gain(core, f, -12) - ref_gain(core, f, l)
                    if best is None or abs(gr - depth) < abs(best[0] - depth): best = (gr, l)
                if abs(best[0] - depth) < 1.5:
                    row.append(f"{f}Hz +{best[1]} {ref_h(core, f, best[1], 2):.0f}/{ref_h(core, f, best[1], 4):.0f}")
            print(f"    GR {depth:4.1f} dB: " + "  ".join(row))
        # the deep-saturation floor against frequency: H2/H4 at the first level where H3 is above -12 dBc (the core fully through the
        # knee), against the driver's own H2 = a2 A / 2 at that level read off the 1 kHz row (where the core does nothing)
        print("  deep-saturation floor: first level with H3 > -12 dBc: freq -> level H2/H4 (driver's H2 at 1 kHz, same level)")
        row = []
        for f in ALL_FREQS[:7]:
            deep = [l for l in LEVELS if ref_h(core, f, l, 3) > -12.0]
            if deep:
                l = deep[0]
                row.append(f"{f}Hz +{l} {ref_h(core, f, l, 2):.0f}/{ref_h(core, f, l, 4):.0f} ({ref_h(core, 1000, l, 2):.0f})")
        print("    " + "  ".join(row))
        # the +24 dBFS row (pooled apart by the test suite): which frequencies clip there
        print("  +24 dBFS row (not fitted): freq -> gain H2/H3: " + "  ".join(
            f"{f}Hz {ref_gain(core, f, 24):.1f} {ref_h(core, f, 24, 2):.0f}/{ref_h(core, f, 24, 3):.0f}" for f in (80, 120, 160, 320, 1000, 5000)))
    print("\nWhat a mechanism must produce (levels -12..+21 dBFS, the fitted range):\n"
          "  (i) below the knee H2 = a2 A / 2 only (1 dB per dB at 1 kHz, H4 at the floor), frequency independent: the driver's even term\n"
          "      must not magnetise the core;\n"
          "  (ii) at the onset of saturation (H3 between -45 and -20 dBc) a spike whose even spectrum is flat (H2 = H4 = H6 within 3 dB, an\n"
          "      impulsive once-per-cycle feature) at -57..-60 dBc on every core at 20..80 Hz, whatever the level the onset happens at\n"
          "      (+6 at 20 Hz .. +18 at 80 Hz): to first order a function of depth alone. The 120 Hz +21 point (-46.5, 12 dB above that)\n"
          "      and the 160 Hz +21 point (-54, before the onset) sit at the foot of the +24 dBFS anomaly: at +24 the reference clips at\n"
          "      160 Hz (H2 -29) and 320 Hz (H2 -12, H2 above H3, gain -2..-3 dB) but is clean at 1 kHz and 5 kHz (H2, H3 on the driver's\n"
          "      law, higher output than 320 Hz), so that ceiling is frequency selective and not an output clipper; the fitted range keeps\n"
          "      only its foot;\n"
          "  (iii) once the core is through the knee (H3 above -12 dBc) the even orders fall to a floor that is frequency dependent at equal\n"
          "      depth: Nickel/Steel about -63..-66 dBc at 20 Hz, -67 at 30, -71 at 40, -75..-80 at 60..80 Hz (about 9 dB per octave, i.e.\n"
          "      the even feature's absolute size falls with frequency faster than the fundamental's ceiling rises), then a slow -0.5 dB per\n"
          "      dB with level while H3 keeps rising to -1 dBc; Iron's floor (-63..-66 at every frequency) is its own driver's a2 (ten times\n"
          "      Nickel's), which hides the core's floor above 30 Hz. So the asymmetry must be largest at the entry into saturation and fade\n"
          "      once the core is driven through the knee every half cycle, and what is left must shrink with frequency.")

# ================================================================================================ 2. the numba mirror
V0, V1, V2, V3, V4, V5, V6, V7, V8, V9 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

@njit(cache=True)
def _sat(ph, phik, qp, qn):
    az = abs(ph / phik)
    if az <= 0.02: return ph
    q = qp if ph > 0.0 else qn
    return ph / np.exp(np.log1p(np.exp(q * np.log(az))) / q)

@njit(cache=True)
def xf_kernel(x, fs, gain_db, a2, a3, fl, sat_db, q, asym, hs_hz, hs_db, lp_hz, variant, c1, c2, c3, c4):
    n = x.size; y = np.empty(n)
    T = 1.0 / fs
    g = 10.0 ** (gain_db / 20.0)
    r = 2.0 * np.pi * fl
    phik = 10.0 ** (sat_db / 20.0) / (2.0 * np.pi * 20.0)
    qp = q * (1.0 + asym); qn = q * (1.0 - asym)
    if qp < 1.0: qp = 1.0
    if qn < 1.0: qn = 1.0
    # shelf and low pass (TPT first order, Common.hpp FirstOrder)
    hs_on = hs_hz > 0.0; ghs = np.tan(np.pi * min(hs_hz, 0.49 * fs) / fs) if hs_on else 0.0; gl = 10.0 ** (hs_db / 20.0); shs = 0.0
    lp_on = lp_hz > 0.0; glp = np.tan(np.pi * min(lp_hz, 0.49 * fs) / fs) if lp_on else 0.0; slp = 0.0
    # candidate states
    if variant == V9:   # the fixed form: V8 with p = 2, coupling 1 Hz, hold 300 ms
        variant = V8; c1 = 2.0; c2 = 1.0; c3 = 300.0
    dcblock = variant == V1 or variant == V5 or variant == V6 or variant == V7 or variant == V8
    fdc = c1 if variant == V1 else (c2 if variant == V8 else c3)   # V5/V6/V7 carry the coupling corner in c3, V8 in c2
    kdc = 1.0 - np.exp(-2.0 * np.pi * fdc / fs) if dcblock else 0.0
    dc = 0.0
    off = 0.0; env = 0.0
    kb = c1 if (variant == V2 or variant == V6) else 0.0
    tau = c2 * 1e-3 if (variant == V2 or variant == V6) else 1.0
    eps = c3 if variant == V2 else (c2 if variant == V6 else 0.0)   # V6: c = (kb, eps, fdc), tau fixed by c2 -> see below
    if variant == V6:
        tau = 0.02; eps = c2
    d0 = c1 if (variant == V3 or variant == V5) else 1.0
    krel = (1.0 - np.exp(-1.0 / (c2 * 1e-3 * fs))) if (variant == V3 or variant == V5) else 0.0
    kc = c1 if variant == V4 else 0.0
    if variant == V8:
        krel = 1.0 - np.exp(-1.0 / (c3 * 1e-3 * fs))
    phi = 0.0; sPrev = 0.0
    for i in range(n):
        u = g * x[i]
        v = u + a2 * u * u + a3 * u * u * u
        if dcblock:
            dc += (v - dc) * kdc
            v = v - dc
        if variant == V4:
            phi += T * (v - r * phi - kc * (phi - _sat(phi, phik, q, q)))
        else:
            phi += T * (v - r * phi)
        if variant == V3 or variant == V5:
            a = abs(phi) / phik
            if a > env: env = a
            else: env += (a - env) * krel
            ae = asym / (1.0 + (env / d0) * (env / d0))
            qpe = q * (1.0 + ae); qne = q * (1.0 - ae)
            if qpe < 1.0: qpe = 1.0
            if qne < 1.0: qne = 1.0
            s = _sat(phi, phik, qpe, qne)
        elif variant == V7 or variant == V8:
            a = abs(phi) / phik
            if variant == V7:
                ae = asym / (1.0 + (a / c1) ** c2)
            else:
                if a > env: env = a
                else: env += (a - env) * krel
                ae = asym / max(env, 0.3) ** c1
            qpe = q * (1.0 + ae); qne = q * (1.0 - ae)
            if qpe < 1.0: qpe = 1.0
            if qne < 1.0: qne = 1.0
            s = _sat(phi, phik, qpe, qne)
        elif variant == V2 or variant == V6:
            ex = phi - _sat(phi, phik, q, q)
            sg = 1.0 if ex > 0.0 else -1.0
            off += T * (kb * ex * (1.0 + eps * sg) - off / tau)
            s = _sat(phi + off, phik, q, q)
        else:
            s = _sat(phi, phik, qp, qn)
        yv = (s - sPrev) * fs; sPrev = s
        if hs_on:
            vv = (yv - shs) * ghs / (1.0 + ghs); lpv = vv + shs; shs = lpv + vv
            yv = lpv + gl * (yv - lpv)
        if lp_on:
            vv = (yv - slp) * glp / (1.0 + glp); lpv = vv + slp; slp = lpv + vv
            yv = lpv
        y[i] = yv
    return y

def core_params(cal, k):
    return [float(cal[MODEL.fields[n][0] + k]) for n in ("x_gain_db", "x_a2", "x_a3", "x_fl_hz", "x_sat_db", "x_q", "x_asym", "x_hs_hz", "x_hs_db", "x_lp_hz")]

_STIM = {}
def stim(f, lvl):
    key = (f, lvl)
    if key not in _STIM:
        x = protocol.stimulus({"kind": "sine", "level": lvl, "f": float(f), "secs": 2.0}, int(FS))[0]
        _STIM[key] = np.ascontiguousarray(x.astype(np.float32).astype(np.float64))   # the engine takes float32 input
    return _STIM[key]

_LOCK = {}
def lockins(f):
    """precomputed complex exponentials for the harm feature (last 0.5 s, whole periods), as protocol.feature does"""
    if f not in _LOCK:
        n = stim(f, 0).size; per = FS / f
        nper = max(4, int(round(0.5 * f)))
        n1 = int(round(int(n / per) * per)); n0 = int(round(n1 - nper * per))
        t = np.arange(n0, n1) / FS
        E = np.stack([np.exp(-2j * np.pi * k * f * t) for k in range(1, 9)])
        _LOCK[f] = (n0, n1, E)
    return _LOCK[f]

def harm_feature(y, f, lvl):
    """{'h': [H2..H8 dBc or None], 'gain_db'} from a rendered channel (float32-rounded like the engine's output)"""
    y = y.astype(np.float32).astype(np.float64)
    n0, n1, E = lockins(f)
    c = 2.0 * (E @ y[n0:n1]) / (n1 - n0)
    a1 = abs(c[0]) + 1e-30
    hs = []
    for k in range(2, 9):
        if k * f >= FS / 2 * 0.95: hs.append(None)
        else: hs.append(float(20 * np.log10(abs(c[k - 1]) / a1 + 1e-30)))
    return {"h": hs, "gain_db": float(20 * np.log10(a1) - lvl)}

def render_grid(p, variant, c, freqs, levels):
    """-> {(f, lvl): feature}"""
    out = {}
    for f in freqs:
        for lvl in levels:
            y = xf_kernel(stim(f, lvl), FS, *p, variant, *c)
            out[(f, lvl)] = harm_feature(y, f, lvl)
    return out

def validate_mirror(cal):
    print("\n" + "=" * 110); print("2. MIRROR VALIDATION against the C++ engine (current constants): |dH| over harmonics the reference has above -100 dBc")
    print("=" * 110)
    worst = 0.0
    for core in ("Nickel", "Iron", "Steel"):
        k = CORE_INDEX[core]; p = core_params(cal, k)
        for f, lvl in ((20, 3), (60, 18), (1000, 21), (320, 12), (40, 9)):
            item = ITEMS[f"xf_{core}_f{f}_{lvl}"]
            ref = render_item(item, cal)
            mine = harm_feature(xf_kernel(stim(f, lvl), FS, *p, V0, 0.0, 0.0, 0.0, 0.0), f, lvl)
            d = [abs(a - b) for a, b in zip(mine["h"], ref["h"]) if a is not None and b is not None and b > -100]
            dg = abs(mine["gain_db"] - ref["gain_db"])
            worst = max(worst, max(d) if d else 0.0, dg)
            print(f"  {core:6s} {f:5d} Hz {lvl:+3d} dBFS: gain {ref['gain_db']:7.3f} / {mine['gain_db']:7.3f}  "
                  f"H2 {ref['h'][0]:7.1f} / {mine['h'][0]:7.1f}  H3 {ref['h'][1]:7.1f} / {mine['h'][1]:7.1f}  max|dH| {max(d) if d else 0:.3f} dB")
    print(f"  worst deviation {worst:.3f} dB")
    return worst

# ================================================================================================ 3. metrics
def metrics(core, feats, levels=FIT_LEVELS, freqs=ALL_FREQS):
    """test-suite pooled residuals (even/odd where the reference is above -80 dBc, model floored at -120) and a quiet-even metric"""
    gain, odd, even, quiet, h2_all = [], [], [], [], []
    for f in freqs:
        for lvl in levels:
            m = feats[(f, lvl)]; r = F[f"xf_{core}_f{f}_{lvl}"]
            gain.append(m["gain_db"] - r["gain_db"])
            for k, (a, b) in enumerate(zip(m["h"], r["h"]), start=2):
                if a is None or b is None: continue
                if k % 2 == 0: quiet.append(max(a, QUIET_FLOOR) - max(b, QUIET_FLOOR))
                if b <= HARM_FLOOR: continue
                (odd if k % 2 else even).append(max(a, MODEL_FLOOR) - b)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else 0.0
    mx = lambda v: float(np.max(np.abs(v))) if len(v) else 0.0
    return {"gain_rms": rms(gain), "gain_max": mx(gain), "odd_rms": rms(odd), "odd_max": mx(odd), "odd_n": len(odd),
            "even_rms": rms(even), "even_max": mx(even), "even_n": len(even), "quiet_even_rms": rms(quiet), "quiet_even_max": mx(quiet)}

def fmt(m):
    return (f"gain {m['gain_rms']:.3f}/{m['gain_max']:.2f}  odd {m['odd_rms']:5.2f}/{m['odd_max']:5.1f} (n={m['odd_n']})  "
            f"even {m['even_rms']:5.2f}/{m['even_max']:5.1f} (n={m['even_n']})  quiet-even {m['quiet_even_rms']:5.2f}/{m['quiet_even_max']:5.1f}")

def h2_table(core, feats, title, freqs=ALL_FREQS, levels=LEVELS):
    print(f"  {title}: model H2 (reference H2) rows = frequency, cols = levels {levels}")
    for f in freqs:
        print(f"  {f:5d} " + " ".join(f"{feats[(f, l)]['h'][0]:6.1f}({ref_h(core, f, l, 2):4.0f})" for l in levels))

# ================================================================================================ 4. decomposition of the current model
def decompose(cal):
    print("\n" + "=" * 110); print("3. DECOMPOSITION of the current model's even harmonics (current constants)"); print("=" * 110)
    for core in ("Nickel", "Iron", "Steel"):
        k = CORE_INDEX[core]; p = core_params(cal, k)
        print(f"\n--- {core}")
        cases = [("as is (static asym)", p, V0, (0.0, 0.0, 0.0, 0.0)),
                 ("asym = 0", p[:6] + [0.0] + p[7:], V0, (0.0, 0.0, 0.0, 0.0)),
                 ("asym = 0, driver AC-coupled 1 Hz", p[:6] + [0.0] + p[7:], V1, (1.0, 0.0, 0.0, 0.0)),
                 ("static asym, driver AC-coupled 1 Hz", p, V1, (1.0, 0.0, 0.0, 0.0))]
        for title, pp, var, c in cases:
            feats = render_grid(pp, var, c, ALL_FREQS, LEVELS)
            m = metrics(core, feats)
            print(f"  {title:38s} {fmt(m)}")
            if core == "Iron" or title.startswith("as is"):
                h2_table(core, feats, title, freqs=[20, 60, 120, 320, 1000], levels=FIT_LEVELS)

# ================================================================================================ 5. candidate fits
# per variant: names of the extra parameters, starting values, bounds, x_scale
CAND = {
    "V0": ([], [], [], []),
    "V1": (["fdc_hz"], [1.0], [(0.2, 8.0)], [0.5]),
    "V2": (["kb", "tau_ms", "eps"], [200.0, 20.0, 0.1], [(0.0, 5000.0), (1.0, 500.0), (-1.0, 1.0)], [50.0, 5.0, 0.05]),
    "V3": (["d0", "tau_env_ms"], [1.2, 30.0], [(0.3, 6.0), (2.0, 300.0)], [0.1, 5.0]),
    "V4": (["kc"], [50.0], [(0.0, 5000.0)], [10.0]),
    "V5": (["d0", "tau_env_ms", "fdc_hz"], [1.2, 30.0, 1.0], [(0.3, 6.0), (2.0, 300.0), (0.2, 8.0)], [0.1, 5.0, 0.5]),
    "V6": (["kb", "eps", "fdc_hz"], [200.0, 0.1, 1.0], [(0.0, 5000.0), (-1.0, 1.0), (0.2, 8.0)], [50.0, 0.05, 0.5]),
    "V7": (["d0", "p", "fdc_hz"], [0.6, 2.0, 1.0], [(0.1, 3.0), (0.5, 8.0), (0.2, 8.0)], [0.05, 0.2, 0.5]),
    "V8": (["p", "fdc_hz", "tau_env_ms"], [2.0, 1.0, 300.0], [(0.0, 6.0), (0.2, 8.0), (1.0, 2000.0)], [0.2, 0.5, 50.0]),
    "V9": ([], [], [], []),
}
VID = {"V0": V0, "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "V6": V6, "V7": V7, "V8": V8, "V9": V9}
NL_LO = [-5e-3, -5e-2, -6.0, 1.5, -0.5]; NL_HI = [5e-3, 5e-2, 12.0, 30.0, 0.5]; NL_SCALE = [1e-5, 1e-4, 0.5, 1.0, 0.01]
HW = {2: 1.0, 3: 1.0, 4: 0.5, 5: 0.5, 6: 0.25, 7: 0.25, 8: 0.25}

def fit_residual(core, feats, freqs, levels):
    """fit objective: gain 5x; harmonics weighted by order, both sides floored at -100 dBc (quiet must stay quiet), +24 dBFS excluded"""
    out = []
    for f in freqs:
        for lvl in levels:
            m = feats[(f, lvl)]; r = F[f"xf_{core}_f{f}_{lvl}"]
            out.append(5.0 * (m["gain_db"] - r["gain_db"]))
            for k, (a, b) in enumerate(zip(m["h"], r["h"]), start=2):
                if a is None or b is None: continue
                a, b = max(a, QUIET_FLOOR), max(b, QUIET_FLOOR)
                if a <= QUIET_FLOOR and b <= QUIET_FLOOR: out.append(0.0); continue
                out.append(HW[k] * (a - b))
    return np.array(out)

def resp_check(core, p, var, c):
    """the mirror's small-signal response on the stage-2 sweep item against the reference (tests/pb_reference.py xfmr_resp)"""
    item = ITEMS[f"xf_resp_{core}"]
    x = protocol.stimulus(item["stim"], int(FS))[0].astype(np.float32).astype(np.float64)
    y = xf_kernel(np.ascontiguousarray(x), FS, *p, var, *c).astype(np.float32).astype(np.float64)
    feat = protocol.feature(item, np.stack([y, y]), int(FS))
    d = np.array([a - b for a, b in zip(feat, F[f"xf_resp_{core}"]) if a is not None and b is not None])
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))

def fit_variant(cal, core, vname, freqs):
    k = CORE_INDEX[core]; p = core_params(cal, k)
    lin = [p[0], p[3], p[7], p[8], p[9]]           # gain, fl, hs_hz, hs_db, lp_hz stay (fitted analytically from the sweep)
    names, c0, cb, cs = CAND[vname]; var = VID[vname]
    nl0 = [p[1], p[2], p[4], p[5], p[6]]
    if var in (V2, V6): nl0[4] = 0.0                # symmetric knee in the remanence variants
    if var in (V7, V8): nl0[4] = 0.1                # the decaying forms want a larger asymmetry at the knee
    if var == V9: nl0[4] = 0.012
    x0 = np.array(nl0 + c0); lo = np.array(NL_LO + [b[0] for b in cb]); hi = np.array(NL_HI + [b[1] for b in cb])
    if var in (V2, V6): lo[4] = -1e-9; hi[4] = 1e-9
    x0 = np.clip(x0, lo, hi)
    xs = np.array(NL_SCALE + cs)
    def unpack(x):
        c = list(x[5:]) + [0.0] * (4 - len(names))
        return [lin[0], x[0], x[1], lin[1], x[2], x[3], x[4], lin[2], lin[3], lin[4]], c
    def resid(x):
        pp, c = unpack(x)
        return fit_residual(core, render_grid(pp, var, c, freqs, FIT_LEVELS), freqs, FIT_LEVELS)
    t0 = time.time()
    r0 = resid(x0)
    r = None
    for mult in ([1.0], [1.0, 0.5, 2.0])[STARTS > 1]:
        xs0 = x0.copy(); xs0[4] = np.clip(x0[4] * mult, lo[4], hi[4])
        ri = least_squares(resid, xs0, bounds=(lo, hi), x_scale=xs, diff_step=1e-3, max_nfev=NFEV, loss="soft_l1", f_scale=2.0)
        if STARTS > 1: print(f"    start asym x{mult}: cost {ri.cost:.2f} (nfev {ri.nfev})")
        if r is None or ri.cost < r.cost: r = ri
    pp, c = unpack(r.x)
    feats = render_grid(pp, var, c, ALL_FREQS, LEVELS)
    m = metrics(core, feats)
    print(f"\n  {core} {vname}: {time.time() - t0:.0f} s, nfev {r.nfev}, weighted rms {np.sqrt(np.mean(r0**2)):.2f} -> {np.sqrt(np.mean(r.fun**2)):.2f} dB")
    print(f"    params: a2 {r.x[0]:.4g} a3 {r.x[1]:.4g} sat_db {r.x[2]:.3f} q {r.x[3]:.3f} asym {r.x[4]:.4f} " +
          " ".join(f"{n} {v:.4g}" for n, v in zip(names, r.x[5:])))
    print(f"    {fmt(m)}")
    h2_table(core, feats, f"{vname} fitted", freqs=[20, 60, 120, 320, 1000], levels=FIT_LEVELS)
    rr, rm = resp_check(core, pp, var, c)
    print(f"    small-signal sweep response (xf_resp_{core}, -30 dBFS, 28 frequencies): rms {rr:.4f} dB, max {rm:.4f} dB")
    print("    per-frequency even residual (ref > -80 dBc; rms/max dB, n) and the +24 dBFS ceiling points (not fitted):")
    for f in ALL_FREQS:
        mf = metrics(core, feats, freqs=[f])
        if mf["even_n"]: print(f"      {f:5d} Hz: even {mf['even_rms']:5.2f}/{mf['even_max']:5.1f} (n={mf['even_n']})  odd {mf['odd_rms']:5.2f}/{mf['odd_max']:5.1f}")
    m24 = metrics(core, feats, levels=[24])
    print(f"      +24 dBFS: {fmt(m24)}")
    return {"variant": vname, "core": core, "x": [float(v) for v in r.x], "names": NL + names, "metrics": m, "resp_rms": rr, "resp_max": rm}

def main():
    cal = load_cal()
    extract_law()
    worst = validate_mirror(cal)
    if worst > 0.3:
        print("MIRROR DOES NOT MATCH THE ENGINE: stop"); sys.exit(1)
    decompose(cal)
    if EXPLORE: return
    print("\n" + "=" * 110); print("4. CANDIDATE FITS (levels -12..+21 dBFS; metrics as tests/pb_reference.py, xfmr group)"); print("=" * 110)
    freqs = ALL_FREQS[::2] if QUICK else ALL_FREQS
    results = []
    for core in CORES:
        base = metrics(core, render_grid(core_params(cal, CORE_INDEX[core]), V0, (0.0, 0.0, 0.0, 0.0), ALL_FREQS, LEVELS))
        print(f"\n### {core}  current constants: {fmt(base)}")
        for v in VARIANTS:
            results.append(fit_variant(cal, core, v, freqs))
    print("\n" + "=" * 110); print("SUMMARY (per core, fitted): even rms/max | odd rms/max | gain rms | quiet-even rms"); print("=" * 110)
    for rr in results:
        m = rr["metrics"]
        print(f"  {rr['core']:6s} {rr['variant']}: even {m['even_rms']:5.2f}/{m['even_max']:5.1f}  odd {m['odd_rms']:5.2f}/{m['odd_max']:5.1f}  "
              f"gain {m['gain_rms']:.3f}  quiet-even {m['quiet_even_rms']:5.2f}/{m['quiet_even_max']:5.1f}   "
              + " ".join(f"{n}={v:.4g}" for n, v in zip(rr["names"], rr["x"])))
    out = os.path.join(HERE, f"xfmr-even-remanence.result.{'-'.join(CORES)}.{'-'.join(VARIANTS)}.json")   # parallel per-core, per-variant runs do not clobber each other
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
