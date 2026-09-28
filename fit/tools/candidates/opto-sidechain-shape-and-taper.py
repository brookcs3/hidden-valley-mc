# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Optical stage, hypothesis D: the sidechain path and the threshold taper. Analysis of the reference's static features
(fit/data/reference_features.json), with the engine rendered only to show what the current model does at each frequency.

  1. Frequency: opto_static_f{100,3000,8000}_{l} against opto_static_t20_{l}. Sidechain roll-off (the knee moves, the no-GR gain does
     not) or audio-path loss (the no-GR gain at -40 moves too)? The level shift of each frequency's curve is fitted against the 1 kHz
     template and a low pass of free order and corner is fitted to the shifts; the "detector averaging" alternative (a pole after the
     rectifier) is bounded analytically and with the engine.
  2. Threshold law: the knee level of every position (GR = 1 dB and 3 dB crossings, a hinge fit, and the shift onto the 1 kHz
     template built from positions 20 and 21 interleaved on a 1 dB grid); a two-segment step law on the knee levels.
  3. No-GR gain against threshold (the flat region of every position): "linear trim" against "leak proportional to a power of the
     sidechain gain", and whether the leak sits inside the feedback loop (it is partly absorbed once compressing) or after the tap.
  4. Shape against threshold: every position's curve shifted onto the template; the residual, slope multiplier and knee width per position.
  5. Static law: the template fitted with the current light law (c = K (v - v0)^p, soft onset) and with a hinge-power law
     (c = K ((v/vk)^p - 1), linear onset), both solved as the feedback divider, plus the additive leak.

usage: cd <repo> && python3 -u fit/tools/candidates/opto-sidechain-shape-and-taper.py [--no-render]"""
import os, sys, numpy as np
from scipy.optimize import least_squares, minimize_scalar, brentq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "stages"))
from common import F, ITEMS, MODEL, load_cal, render_item  # noqa: E402

np.set_printoptions(linewidth=160, suppress=True)
RENDER = "--no-render" not in sys.argv
LEVELS = np.arange(-50, 15, 2)            # opto_static_t{k}_{l}
FLEVELS = np.arange(-40, 11, 2)           # opto_static_f{f}_{l}
FLAT_MAX = -36                             # every position is flat up to here (position 24 knees at -30)
G = np.array([[F[f"opto_static_t{k}_{l}"] for l in LEVELS] for k in range(1, 25)])   # gain dB, includes make-up 12
g0 = np.array([G[k][LEVELS <= FLAT_MAX].mean() for k in range(24)])
flat_spread = max(np.ptp(G[k][LEVELS <= FLAT_MAX]) for k in range(24))


def rms(e): return float(np.sqrt(np.mean(np.square(e))))


def interp_curve(xs, ys, x):
    """linear interpolation with linear extrapolation at both ends"""
    y = np.interp(x, xs, ys)
    lo = x < xs[0]; hi = x > xs[-1]
    if lo.any(): y[lo] = ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x[lo] - xs[0])
    if hi.any(): y[hi] = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x[hi] - xs[-1])
    return y


def hinge(L, s, Lk, w):
    """soft hinge: slope s above Lk, knee width w (dB); w -> 0 is a hard knee"""
    z = (L - Lk) / max(w, 1e-3)
    return s * max(w, 1e-3) * np.logaddexp(0.0, z)


def fit_hinge(L, gr, s0=0.62, Lk0=None, w0=1.0, fix_s=None):
    if Lk0 is None: Lk0 = L[np.argmax(gr > 1.0)] - 1.0
    if fix_s is None:
        r = least_squares(lambda p: hinge(L, p[0], p[1], p[2]) - gr, [s0, Lk0, w0], bounds=([0.2, -60, 1e-3], [1.2, 40, 10]))
        return r.x, rms(r.fun), float(np.max(np.abs(r.fun)))
    r = least_squares(lambda p: hinge(L, fix_s, p[0], p[1]) - gr, [Lk0, w0], bounds=([-60, 1e-3], [40, 10]))
    return np.array([fix_s, r.x[0], r.x[1]]), rms(r.fun), float(np.max(np.abs(r.fun)))


def crossing(L, gr, level):
    """first level where gr crosses `level`, linear interpolation; None if it never does"""
    idx = np.where(gr >= level)[0]
    if len(idx) == 0 or idx[0] == 0: return None
    i = idx[0]
    return float(L[i - 1] + (level - gr[i - 1]) / (gr[i] - gr[i - 1]) * (L[i] - L[i - 1]))


def best_shift(Lm, ym, Lt, yt, bounds=(-45, 45), sel=None):
    """shift s minimising sum (interp(template, Lm + s) - ym)^2 over the selected points"""
    if sel is None: sel = np.ones(len(Lm), bool)
    cost = lambda s: float(np.sum((interp_curve(Lt, yt, Lm[sel] + s) - ym[sel]) ** 2))
    r = minimize_scalar(cost, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    e = interp_curve(Lt, yt, Lm[sel] + r.x) - ym[sel]
    return float(r.x), rms(e), float(np.max(np.abs(e)))


# ============================================================================ 0. the template: positions 20 and 21 interleaved
GR = g0[:, None] - G                       # gain reduction relative to each position's own flat gain
k20, k21 = 19, 20
# the shift of position 21 onto position 20 (2 dB grid, GR domain, compressing points only)
s21, s21_rms, s21_max = best_shift(LEVELS, GR[k21], LEVELS, GR[k20], bounds=(-6, 6), sel=GR[k21] > 0.3)
print("=== 0. template")
print(f"  flat-region spread (max over positions, levels <= {FLAT_MAX}): {flat_spread:.3f} dB")
print(f"  position 21 onto 20: shift {s21:+.3f} dB (rms {s21_rms:.3f}, max {s21_max:.3f}); interleaving with a 1.000 dB shift")
# position 21 has 1 dB more drive: its point at level L is position 20's curve at L + 1
TL = np.concatenate([LEVELS, LEVELS + 1.0]); order = np.argsort(TL); TL = TL[order]
T_GR = np.concatenate([GR[k20], GR[k21]])[order]                       # GR relative to own flat gain, 1 dB grid, at position 20's levels
T_G = np.concatenate([G[k20], G[k21]])[order]                          # absolute gain
T_SRC = np.concatenate([np.zeros(len(LEVELS), int), np.ones(len(LEVELS), int)])[order]
comp = T_GR > 0.3
p_hT, hT_rms, hT_max = fit_hinge(TL[comp], T_GR[comp])
print(f"  template hinge fit (compressing points): slope {p_hT[0]:.4f} dB/dB (ratio {1 / (1 - p_hT[0]):.2f}:1), knee {p_hT[1]:.2f} dBFS, width {p_hT[2]:.2f} dB | rms {hT_rms:.3f} max {hT_max:.3f}")
p_hT0, hT0_rms, hT0_max = fit_hinge(TL[comp], T_GR[comp], w0=0.01)
# local slopes on the 1 dB grid, to show the wobble
d1 = np.diff(T_GR[comp])
print(f"  local slope on the 1 dB grid above the knee: mean {d1.mean():.3f}, sd {d1.std():.3f}, min {d1.min():.2f}, max {d1.max():.2f}")
print("  1 dB-grid GR (dB) from the first compressing point:", np.round(T_GR[comp], 2))
# straight-line fit above 3 dB of GR: is the residual wobble structured or noise-like?
sel3 = T_GR > 3.0
A = np.vstack([TL[sel3], np.ones(sel3.sum())]).T; coef = np.linalg.lstsq(A, T_GR[sel3], rcond=None)[0]
lin_res = T_GR[sel3] - A @ coef
print(f"  straight line above 3 dB GR: slope {coef[0]:.4f}, residual rms {rms(lin_res):.3f} max {np.max(np.abs(lin_res)):.3f}; residual by source (20 even / 21 odd) rms {rms(lin_res[T_SRC[sel3] == 0]):.3f} / {rms(lin_res[T_SRC[sel3] == 1]):.3f}")
# the same for the 100 Hz curve (smooth?) and the two other frequencies
GF = {f: np.array([F[f"opto_static_f{f}_{l}"] for l in FLEVELS]) for f in (100, 3000, 8000)}
g0f = {f: GF[f][FLEVELS <= -30].mean() for f in GF}
for f in (100, 3000, 8000):
    grf = g0f[f] - GF[f]; s3 = (grf > 3.0) & (FLEVELS <= (2 if f == 8000 else 14))
    A = np.vstack([FLEVELS[s3], np.ones(s3.sum())]).T; c = np.linalg.lstsq(A, grf[s3], rcond=None)[0]
    print(f"  {f:5d} Hz straight line above 3 dB GR (L <= {2 if f == 8000 else 14}): slope {c[0]:.4f}, residual rms {rms(grf[s3] - A @ c):.3f} max {np.max(np.abs(grf[s3] - A @ c)):.3f}")

# ============================================================================ 1. frequency
print("\n=== 1. frequency behaviour at threshold 20 (opto_static_f* against opto_static_t20_*)")
gr20 = GR[k20]
print(f"  no-GR gain at -40 dBFS: 1 kHz {F['opto_static_t20_-40']:.3f} | 100 Hz {F['opto_static_f100_-40']:.3f} | 3 kHz {F['opto_static_f3000_-40']:.3f} | 8 kHz {F['opto_static_f8000_-40']:.3f} dB")
resp = F["stage_resp_opto"]; RESP_F = [10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 8000, 10000, 12000, 14000, 15000, 16000, 18000, 20000, 22000]
r1k = resp[RESP_F.index(1000)]
print(f"  stage_resp_opto (threshold 1, audio path only) relative to 1 kHz: 100 Hz {resp[RESP_F.index(100)] - r1k:+.3f}, 3 kHz {resp[RESP_F.index(3000)] - r1k:+.3f}, 8 kHz {resp[RESP_F.index(8000)] - r1k:+.3f} dB")
shifts = {}
for f in (100, 3000, 8000):
    grf = g0f[f] - GF[f]
    top = 2 if f == 8000 else 14
    sel = (grf > 0.7) & (FLEVELS <= top)
    s, e_rms, e_max = best_shift(FLEVELS, grf, TL, T_GR, bounds=(-20, 20), sel=sel)
    # also allow a slope multiplier: grf ~ m * T(L + s)
    def res2(p):
        return p[1] * interp_curve(TL, T_GR, FLEVELS[sel] + p[0]) - grf[sel]
    r2 = least_squares(res2, [s, 1.0])
    kn1 = crossing(FLEVELS, grf, 1.0); kn3 = crossing(FLEVELS, grf, 3.0)
    shifts[f] = s
    print(f"  {f:5d} Hz: no-GR gain {g0f[f]:+.3f} (1 kHz {g0[k20]:+.3f}) | knee(1 dB) {kn1:.2f} knee(3 dB) {kn3:.2f} dBFS (1 kHz: {crossing(LEVELS, gr20, 1.0):.2f} / {crossing(LEVELS, gr20, 3.0):.2f}) | "
          f"level shift onto the 1 kHz template {s:+.2f} dB (rms {e_rms:.3f}, max {e_max:.3f}, n {sel.sum()}) | with free slope: shift {r2.x[0]:+.2f}, slope x{r2.x[1]:.3f}, rms {rms(r2.fun):.3f}")
gr8 = g0f[8000] - GF[8000]; hi8 = FLEVELS >= 4
A = np.vstack([FLEVELS[hi8], np.ones(hi8.sum())]).T; c8 = np.linalg.lstsq(A, gr8[hi8], rcond=None)[0]
out8 = FLEVELS[hi8] + GF[8000][hi8]
print(f"  8 kHz above +4 dBFS: slope {c8[0]:.3f} dB/dB (output level {out8.min():.2f}..{out8.max():.2f} dBFS: the output is pinned; not a sidechain effect)")
# the low-pass fit: attenuation a(f) = 10 log10(1 + (f/fc)^(2N)); 1 kHz is the reference (a = 0 by definition, so the true a(1k) is absorbed)
fa = np.array([100.0, 3000.0, 8000.0]); aa = np.array([-shifts[100], -shifts[3000], -shifts[8000]])   # a positive shift means the knee is higher
def lp_att(f, fc, N): return 10.0 * np.log10(1.0 + (f / fc) ** (2.0 * N))
def lp_res(p, N=None):
    fc = p[0]; n = p[1] if N is None else N
    return lp_att(fa, fc, n) - lp_att(1000.0, fc, n) - aa
r_free = least_squares(lp_res, [5000.0, 2.0], bounds=([500, 0.5], [40000, 6]), x_scale=[1000, 0.5])
r_1 = least_squares(lambda p: lp_res(p, 1.0), [4000.0], bounds=([500], [40000]))
r_2 = least_squares(lambda p: lp_res(p, 2.0), [5000.0], bounds=([500], [40000]))
print(f"  sidechain attenuation relative to 1 kHz: 100 Hz {aa[0]:+.2f}, 3 kHz {aa[1]:+.2f}, 8 kHz {aa[2]:+.2f} dB")
print(f"  low pass, free order: fc {r_free.x[0]:.0f} Hz, order {r_free.x[1]:.2f} | residual dB {np.round(r_free.fun, 2)}")
print(f"  low pass, first order: fc {r_1.x[0]:.0f} Hz | residual dB {np.round(r_1.fun, 2)} (rms {rms(r_1.fun):.2f})")
print(f"  low pass, second order: fc {r_2.x[0]:.0f} Hz | residual dB {np.round(r_2.fun, 2)} (rms {rms(r_2.fun):.2f})")
fc1_from8k = 8000.0 / np.sqrt(10 ** (aa[2] / 10) - 1)
print("  sensitivity to the 3 kHz attenuation (its estimates: free-slope fit 0.27, 3 dB crossing 0.33, 1 dB crossing 0.40, template shift 0.48 dB):")
for a3 in (0.27, 0.33, 0.40, 0.48):
    twoN = np.log((10 ** (aa[2] / 10) - 1) / (10 ** (a3 / 10) - 1)) / np.log(8000.0 / 3000.0)
    fcN = 8000.0 / (10 ** (aa[2] / 10) - 1) ** (1.0 / twoN)
    print(f"    a(3 kHz) {a3:.2f} dB -> order {twoN / 2:.2f}, fc {fcN:.0f} Hz")
def lp2q_att(f, fc, Q):
    w = f / fc
    return 10.0 * np.log10((1 - w * w) ** 2 + (w / Q) ** 2)
for Q, name in ((0.7071, 'Butterworth pair (Q 0.707)'), (0.5, 'two real poles at the same corner (Q 0.5)')):
    rq = least_squares(lambda p: lp2q_att(fa, p[0], Q) - lp2q_att(1000.0, p[0], Q) - aa, [5000.0], bounds=([500], [40000]))
    print(f"  second order, {name}: fc {rq.x[0]:.0f} Hz | residual dB {np.round(rq.fun, 2)} (rms {rms(rq.fun):.2f})")
rq = least_squares(lambda p: lp2q_att(fa[1:], p[0], p[1]) - lp2q_att(1000.0, p[0], p[1]) - aa[1:], [5000.0, 0.7], bounds=([500, 0.3], [40000, 3.0]), x_scale=[1000, 0.1])
print(f"  second order, fc and Q free (3 kHz and 8 kHz points): fc {rq.x[0]:.0f} Hz, Q {rq.x[1]:.2f} | residual dB {np.round(rq.fun, 3)}")
# is the wobble of the 1 kHz curve the same pattern at 3 kHz and 8 kHz (a feature of the static law) or not?
wob_T = T_GR[sel3] - (A_T := np.vstack([TL[sel3], np.ones(sel3.sum())]).T) @ np.linalg.lstsq(A_T, T_GR[sel3], rcond=None)[0]
for f in (100, 3000, 8000):
    grf = g0f[f] - GF[f]; s3 = (grf > 3.0) & (FLEVELS <= (2 if f == 8000 else 14))
    Af = np.vstack([FLEVELS[s3], np.ones(s3.sum())]).T; wob_f = grf[s3] - Af @ np.linalg.lstsq(Af, grf[s3], rcond=None)[0]
    wob_1k = np.interp(FLEVELS[s3] + shifts[f], TL[sel3], wob_T)
    cc = np.corrcoef(wob_f, wob_1k)[0, 1]
    print(f"  wobble pattern {f:5d} Hz against 1 kHz (shifted): correlation {cc:+.2f}, rms {rms(wob_f):.3f} vs {rms(wob_1k):.3f}, rms of the difference {rms(wob_f - wob_1k):.3f}")
print(f"  a first-order pole placed from the 8 kHz point alone: fc {fc1_from8k:.0f} Hz would attenuate 3 kHz by {lp_att(3000, fc1_from8k, 1):.2f} dB (measured {aa[1]:.2f})")
# the "detector averaging" alternative: a pole after the rectifier does not change the mean of the light; it removes ripple, and with
# a cell law exponent gamma the mean conductance changes by mean(L^gamma) against (mean L)^gamma at most
cal = load_cal(); n_fit = float(cal[MODEL.fields["o_n"][0]]); gm_fit = float(cal[MODEL.fields["o_gamma"][0]]); p_fit = n_fit * gm_fit
th = np.linspace(0, np.pi, 20001); Lw = np.abs(np.sin(th)) ** n_fit
full = np.mean(Lw ** gm_fit); none = np.mean(Lw) ** gm_fit
print(f"  post-rectifier averaging bound (n {n_fit:.2f}, gamma {gm_fit:.2f}): full ripple / no ripple mean conductance {full / none:.3f} = {20 * np.log10(full / none):.2f} dB, "
      f"i.e. a level shift of at most {20 * np.log10(full / none) / p_fit:.2f} dB between frequencies: cannot give {aa[2]:.1f} dB")
if RENDER:
    print("  engine renders (current constants): gain difference (f minus 1 kHz) model against reference, compressing levels")
    m1k = np.array([render_item(ITEMS[f"opto_static_t20_{l}"], cal) for l in FLEVELS])
    G1k = np.array([F[f"opto_static_t20_{l}"] for l in FLEVELS])
    for f in (100, 3000, 8000):
        mf = np.array([render_item(ITEMS[f"opto_static_f{f}_{l}"], cal) for l in FLEVELS])
        top = 2 if f == 8000 else 14
        sel = ((g0f[f] - GF[f]) > 0.7) & (FLEVELS <= top)
        dm = (mf - m1k)[sel]; dr = (GF[f] - G1k)[sel]
        # emulate a sidechain attenuation of a(f) dB by lowering the threshold-20 drive
        c2 = cal.copy(); c2[MODEL.fields["o_thr_db"][0] + 19] -= aa[list(fa).index(float(f))]
        mf2 = np.array([render_item(ITEMS[f"opto_static_f{f}_{l}"], c2) for l in FLEVELS])
        dm2 = (mf2 - m1k)[sel]
        print(f"    {f:5d} Hz: model diff mean {dm.mean():+.2f} dB (ref {dr.mean():+.2f}); diff-vs-diff rms {rms(dm - dr):.2f} | with the drive lowered by {aa[list(fa).index(float(f))]:.2f} dB: rms {rms(dm2 - dr):.2f}, max {np.max(np.abs(dm2 - dr)):.2f}")

# ============================================================================ 2. threshold law
print("\n=== 2. knee level per position")
droop = g0[0] - G[0]                       # position 1: the stage amplifier's own droop (no compression below +12, see below)
# a cubic amplifier droops the fundamental by a term proportional to amplitude^2: fit on position 1 at 0..+10 dBFS, extrapolate to +14
selD = (LEVELS >= 0) & (LEVELS <= 10)
kd = np.sum(droop[selD] * 10 ** (LEVELS[selD] / 10)) / np.sum(10 ** (LEVELS[selD] / 5))
droop_pred14 = kd * 10 ** (14 / 10)
print(f"  position 1: gain falls {g0[0]:+.3f} -> {G[0][-1]:+.3f} dB by +14 dBFS; an amplitude^2 droop fitted on 0..+10 predicts {droop_pred14:.3f} dB at +14 (measured {droop[-1]:.3f}): excess {droop[-1] - droop_pred14:+.3f} dB")
print(f"  position 2: GR {GR[1][-2]:.2f} dB at +12, {GR[1][-1]:.2f} dB at +14 (minus the droop: {GR[1][-2] - droop[-2]:.2f}, {GR[1][-1] - droop[-1]:.2f})")
GRc = GR - droop[None, :]                  # GR corrected for the amplifier droop
rows = []
for k in range(24):
    gr = GRc[k]
    kn1 = crossing(LEVELS, gr, 1.0); kn3 = crossing(LEVELS, gr, 3.0)
    comp_k = gr > 0.3
    n_c = int(comp_k.sum())
    if 0 < n_c < 3: comp_k = gr > 0.1; n_c = int(comp_k.sum())   # positions 2 and 3: use the first partial point too
    if n_c >= 4:
        ph, h_rms, h_max = fit_hinge(LEVELS[comp_k], gr[comp_k])
        sl = np.polyfit(LEVELS[gr > 3.0], gr[gr > 3.0], 1)[0] if (gr > 3.0).sum() >= 3 else np.nan
    elif n_c >= 1:
        ph, h_rms, h_max = fit_hinge(LEVELS[comp_k], gr[comp_k], fix_s=p_hT[0]); ph[2] = np.nan; sl = np.nan
    else:
        ph, h_rms, h_max, sl = np.array([np.nan] * 3), np.nan, np.nan, np.nan
    # shift onto the template (GR domain, compressing points), and in absolute gain; points that would land above the template's
    # top level are left out (positions 22..24 reach 2..4 dB beyond it)
    if n_c >= 1:
        kn1_20 = crossing(LEVELS, GRc[k20], 1.0)
        if kn1 is not None: comp_k = comp_k & (LEVELS + (kn1_20 - kn1) <= LEVELS.max() + 0.5)
        n_c = int(comp_k.sum())
        sg, sg_rms, sg_max = best_shift(LEVELS, gr, TL, T_GR, bounds=(-45, 45), sel=comp_k)
        sa, sa_rms, sa_max = best_shift(LEVELS, G[k], TL, T_G, bounds=(-45, 45), sel=comp_k)
        # slope multiplier against the template
        r2 = least_squares(lambda p: p[1] * interp_curve(TL, T_GR, LEVELS[comp_k] + p[0]) - gr[comp_k], [sg, 1.0]) if n_c >= 4 else None
        mult = float(r2.x[1]) if r2 is not None else np.nan
    else:
        sg = sg_rms = sg_max = sa = sa_rms = sa_max = mult = np.nan
    rows.append(dict(k=k + 1, g0=g0[k], kn1=kn1, kn3=kn3, Lk=ph[1], s=ph[0], w=ph[2], h_rms=h_rms, sl=sl, sg=sg, sg_rms=sg_rms, sg_max=sg_max, sa=sa, sa_rms=sa_rms, sa_max=sa_max, mult=mult, n=n_c))
print("  pos  noGR_gain  knee1dB  knee3dB | hinge: knee  slope  width  rms | slope>3dB | shift onto template (GR): s  rms  max | (abs gain): s  rms  max | slope x")
for r in rows:
    fmt = lambda v, w=6, d=2: (f"{v:{w}.{d}f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else " " * (w - 3) + "---")
    print(f"  {r['k']:3d}  {r['g0']:+8.3f}  {fmt(r['kn1'])}  {fmt(r['kn3'])} | {fmt(r['Lk'])}  {fmt(r['s'], 6, 3)}  {fmt(r['w'])}  {fmt(r['h_rms'], 5, 3)} | {fmt(r['sl'], 6, 3)} | "
          f"{fmt(r['sg'], 7)} {fmt(r['sg_rms'], 5, 3)} {fmt(r['sg_max'], 5, 3)} | {fmt(r['sa'], 7)} {fmt(r['sa_rms'], 5, 3)} {fmt(r['sa_max'], 5, 3)} | {fmt(r['mult'], 6, 3)}")
# the knee table: use the template shift (GR domain) as the drive relative to position 20; knee level = template knee - shift
knee_T = p_hT[1]
Lk = np.array([knee_T - r["sg"] if not np.isnan(r["sg"]) else np.nan for r in rows])
drive = np.array([r["sg"] for r in rows])          # dB relative to position 20 (positive: more sidechain gain)
print("  drive relative to position 20 (dB):", np.round(drive, 2))
print("  steps (dB):", np.round(np.diff(drive), 2))
# two-segment step law on positions with a measured knee (>= 3 compressing points)
ok = np.array([r["n"] >= 3 for r in rows]); ks = np.arange(1, 25)
best = None
for kb in range(4, 22):
    def seg(p):
        a, b1, b2 = p
        return np.where(ks < kb, a + b1 * (ks - kb), a + b2 * (ks - kb))
    r = least_squares(lambda p: (seg(p) - drive)[ok], [drive[19] - (20 - kb), 2.0, 1.0])
    if best is None or rms(r.fun) < best[1]: best = (kb, rms(r.fun), r.x, seg(r.x))
kb, seg_rms, seg_p, seg_fit = best
print(f"  two-segment step law (positions with a knee: {ks[ok].min()}..{ks[ok].max()}): break at position {kb}, {seg_p[1]:.2f} dB/step below, {seg_p[2]:.3f} dB/step above, rms {seg_rms:.2f} dB")
print("  residual per position (measured - law):", np.round((drive - seg_fit)[ok], 2), "for positions", ks[ok])
ok5 = ok & (ks >= 5)
best5 = None
for kb in range(6, 22):
    def seg(p):
        a, b1, b2 = p
        return np.where(ks < kb, a + b1 * (ks - kb), a + b2 * (ks - kb))
    r = least_squares(lambda p: (seg(p) - drive)[ok5], [drive[19] - (20 - kb), 2.0, 1.0])
    if best5 is None or rms(r.fun) < best5[1]: best5 = (kb, rms(r.fun), r.x, seg(r.x))
print(f"  same on positions 5..24: break at {best5[0]}, {best5[2][1]:.3f} dB/step below, {best5[2][2]:.3f} dB/step above, rms {best5[1]:.3f} dB, max {np.max(np.abs((drive - best5[3])[ok5])):.3f}")
print("  residual per position 5..24:", np.round((drive - best5[3])[ok5], 2))

# ============================================================================ 3. the no-GR gain against threshold
print("\n=== 3. no-GR gain against threshold (flat region, levels <= -36)")
print("  g0 per position:", np.round(g0, 3))
loss = g0[0] - g0
print("  loss relative to position 1 (dB):", np.round(loss, 3))
okL = ~np.isnan(drive)
# (i) linear trim in position, (ii) linear trim in drive dB, (iii) leak c0 = c00 * 10^(p * drive / 20) as a conductance
r_i = np.polyfit(ks, g0, 1); e_i = g0 - np.polyval(r_i, ks)
r_ii = np.polyfit(drive[okL], g0[okL], 1); e_ii = g0[okL] - np.polyval(r_ii, drive[okL])
def leak_model(p, d): return p[0] - 20 * np.log10(1 + p[1] * 10 ** (p[2] * d / 20))
g_amp = float(g0[0])                        # amplifier + make-up gain: position 1's flat gain (its leak is below 0.001 dB)
r_iii = least_squares(lambda p: leak_model([g_amp, p[0], p[1]], drive[okL]) - g0[okL], [0.02, 2.0], bounds=([1e-6, 0.2], [1.0, 6.0]))
p_slope = p_hT[0] / (1 - p_hT[0])
r_iv = least_squares(lambda p: leak_model([g_amp, p[0], p_slope], drive[okL]) - g0[okL], [0.02], bounds=([1e-6], [1.0]))
print(f"  (i)   linear trim in position: {r_i[0]:+.4f} dB/position, rms {rms(e_i):.3f}, max {np.max(np.abs(e_i)):.3f}")
print(f"  (ii)  linear trim in drive:    {r_ii[0]:+.4f} dB per dB of drive, rms {rms(e_ii):.3f}, max {np.max(np.abs(e_ii)):.3f}")
print(f"  (iii) leak conductance c0 = c00 * A^p (A relative to position 20, g_amp fixed {g_amp:+.3f} dB): c00 {r_iii.x[0]:.4f}, p {r_iii.x[1]:.2f}, rms {rms(r_iii.fun):.3f}, max {np.max(np.abs(r_iii.fun)):.3f}")
print(f"  (iv)  leak with p fixed to the slope's p/(1+p) inverse ({p_slope:.2f}): c00 {r_iv.x[0]:.4f}, rms {rms(r_iv.fun):.3f}, max {np.max(np.abs(r_iv.fun)):.3f}")
r_v = least_squares(lambda p: (g_amp - p[0] * 10 ** (p[1] * drive[okL] / 20)) - g0[okL], [0.18, 2.0], bounds=([1e-6, 0.2], [5.0, 6.0]))
print(f"  (v)   loss in dB as a pure power of the drive, loss = a * A^q: a {r_v.x[0]:.4f} dB, q {r_v.x[1]:.2f}, rms {rms(r_v.fun):.3f}, max {np.max(np.abs(r_v.fun)):.3f} (the 20 log10(1 + c0) form of (iii) is preferred if it fits better)")
print("  leak residual per position:", np.round(r_iii.fun, 3))
c0 = 10 ** ((g_amp - g0) / 20) - 1.0        # measured leak conductance per position
print("  measured leak conductance c0 per position:", np.round(c0, 4))
# in the loop or after the tap? offset of each position's absolute gain against the template (shifted), in the compressing region
print("  where does the loss sit? absolute-gain offset of each position against the template shifted by its GR-domain drive, in the compressing region.")
print("  A gain factor anywhere outside the conductance sum (input side or after the tap) gives the full flat-gain difference dg0 at every level;")
print("  an additive conductance inside the loop is swamped as the signal conductance grows (numerical prediction from the template):")
print("  pos  dg0(flat)  offset mean  offset first/last 3  | gain factor: dg0 | additive in-loop conductance: mean first/last | rms against each")
s_T = p_hT[0]
# a conductance-domain template: c_sig(Lv) at position 20 = c_tot - c0, from the interleaved template (each source minus its own leak)
c0_src = np.array([c0[k20], c0[k21]])
c_tot_T = 10 ** ((g_amp - T_G) / 20) - 1.0
c_sig_T = np.maximum(c_tot_T - c0_src[T_SRC], 0.0)
Lv_T = TL - (g_amp - T_G)                  # divider output level relative to the input convention (make-up and amplifier gain removed)
selT = c_sig_T > 0
Lv_grid = Lv_T[selT]; c_grid = c_sig_T[selT]
o = np.argsort(Lv_grid); Lv_grid = Lv_grid[o]; c_grid = c_grid[o]
def c_sig(Lv, shift):
    """the signal conductance of position 20 at divider output level Lv + shift (dB); zero below the template's first point"""
    z = Lv + shift
    if z <= Lv_grid[0]: return 0.0
    if z >= Lv_grid[-1]:   # extend with the power law of the last decade
        pw = np.log(c_grid[-1] / c_grid[-6]) / (Lv_grid[-1] - Lv_grid[-6]) * 20 / np.log(10)
        return c_grid[-1] * 10 ** (pw * (z - Lv_grid[-1]) / 20)
    return float(np.interp(z, Lv_grid, c_grid))
def solve_static(Lx, shift, leak, law=None):
    """feedback divider: v (1 + leak + c(v)) = x; returns the gain in dB (without amplifier/make-up). law(Lv) -> c, default the template"""
    cfun = (lambda Lv: c_sig(Lv, shift)) if law is None else law
    out = np.empty(len(Lx))
    for i, L in enumerate(Lx):
        x = 10 ** (L / 20)
        f = lambda v: v * (1 + leak + cfun(20 * np.log10(v))) - x
        lo_, hi_ = x * 1e-6, x * 1.0000001
        if not (f(lo_) < 0 < f(hi_)):
            raise RuntimeError(f"static solve bracket failed: L {L} shift {shift} leak {leak} f(lo) {f(lo_)} f(hi) {f(hi_)}")
        v = brentq(f, lo_, hi_, xtol=1e-12, rtol=1e-12)
        out[i] = 20 * np.log10(v / x)
    return out
for k in (20, 21, 22, 23, 24, 16, 12, 8):
    kk = k - 1
    comp_k = (GRc[kk] > 0.3) & (LEVELS + drive[kk] <= LEVELS.max() + 0.5)
    if comp_k.sum() < 3: continue
    off = G[kk][comp_k] - interp_curve(TL, T_G, LEVELS[comp_k] + drive[kk])
    dg0 = g0[kk] - g0[k20]
    # the leak-in-loop prediction: template conductance shifted by the drive, plus this position's leak
    pred_leak = g_amp + solve_static(LEVELS[comp_k], drive[kk], c0[kk])
    pred_ref = g_amp + solve_static(LEVELS[comp_k], drive[kk], c0[k20])   # same shift with position 20's leak: the "no change" reference
    off_leak = pred_leak - pred_ref
    print(f"  {k:3d}  {dg0:+8.3f}  {off.mean():+10.3f}  {off[:3].mean():+6.3f}/{off[-3:].mean():+6.3f} | {dg0:+10.3f} | {off_leak.mean():+.3f} {off_leak[:3].mean():+.3f}/{off_leak[-3:].mean():+.3f} | gain factor {rms(off - dg0):.3f}, in-loop conductance {rms(off - off_leak):.3f}")

# ============================================================================ 4. shape against threshold: the family predicted from the template
print("\n=== 4. shape against threshold")
print("  template shift residual per position (GR domain) is in the table above; slope multiplier x and hinge width per position:")
mults = np.array([r["mult"] for r in rows]); widths = np.array([r["w"] for r in rows]); slopes = np.array([r["sl"] for r in rows])
okm = ~np.isnan(mults)
print(f"  slope multiplier: positions {ks[okm].min()}..{ks[okm].max()}: mean {np.nanmean(mults):.4f}, sd {np.nanstd(mults):.4f}, min {np.nanmin(mults):.3f} (pos {ks[np.nanargmin(mults)]}), max {np.nanmax(mults):.3f} (pos {ks[np.nanargmax(mults)]})")
print(f"  slope above 3 dB GR (linear fit): mean {np.nanmean(slopes):.4f}, sd {np.nanstd(slopes):.4f}; per position:", np.round(slopes, 3))
print(f"  hinge knee width: mean {np.nanmean(widths):.2f} dB, sd {np.nanstd(widths):.2f}; per position:", np.round(widths, 2))
trend = np.polyfit(ks[okm], mults[okm], 1)
print(f"  slope multiplier trend with position: {trend[0] * 10:+.4f} per 10 positions")
# the full family predicted by "template + drive shift + additive in-loop leak": residual over every position and level
pred_all = np.full_like(G, np.nan); drive_leak = np.full(24, np.nan)
for kk in range(24):
    if np.isnan(drive[kk]): continue
    inr = LEVELS + drive[kk] <= LEVELS.max() + 0.5
    def cost(d):
        return float(np.sum((g_amp - droop[inr] + solve_static(LEVELS[inr], d, c0[kk]) - G[kk][inr]) ** 2))
    r = minimize_scalar(cost, bounds=(drive[kk] - 3, drive[kk] + 3), method="bounded", options={"xatol": 1e-4})
    drive_leak[kk] = r.x
    pred_all[kk][inr] = g_amp - droop[inr] + solve_static(LEVELS[inr], r.x, c0[kk])
e_all = (pred_all - G)[~np.isnan(pred_all)]
print(f"  family from template + shift (refitted) + additive in-loop leak: rms {rms(e_all):.3f} dB, max {np.max(np.abs(e_all)):.3f} over {e_all.size} points (positions with a knee, within the template range)")
# and the same with the leak after the tap (a trim): pred = template shifted + dg0
pred_trim = np.full_like(G, np.nan)
for kk in range(24):
    if np.isnan(drive[kk]): continue
    inr = LEVELS + drive[kk] <= LEVELS.max() + 0.5
    pred_trim[kk][inr] = interp_curve(TL, T_G, LEVELS[inr] + drive[kk]) + (g0[kk] - g0[k20])
    flat = (LEVELS + drive[kk] < TL[comp][0] - 1.0) & inr
    pred_trim[kk][flat] = g0[kk] - droop[flat]
e_trim = (pred_trim - G)[~np.isnan(pred_trim)]
print(f"  family from template + shift + gain factor (trim):        rms {rms(e_trim):.3f} dB, max {np.max(np.abs(e_trim)):.3f}")
def step_regularity(d, name):
    st = np.diff(d); hi = st[10:23]; lo = st[4:10]      # steps 11->12 .. 23->24 and 5->6 .. 10->11
    print(f"  {name}: steps 11..24 mean {hi.mean():.3f} sd {hi.std():.3f} (odd/even alternation {np.abs(np.diff(hi)).mean():.3f}); steps 5..11 mean {lo.mean():.3f} sd {lo.std():.3f}")
step_regularity(drive, "drive table, GR-domain shift (gain-factor model)")
step_regularity(drive_leak, "drive table refitted under the in-loop conductance model")
print("  in-loop model drives:", np.round(drive_leak, 2))
for kk in range(24):
    if np.isnan(drive[kk]): continue
    ok_ = ~np.isnan(pred_all[kk]); e1 = (pred_all[kk] - G[kk])[ok_]; e2 = (pred_trim[kk] - G[kk])[ok_]
    print(f"    pos {kk + 1:2d}: leak-in-loop rms {rms(e1):.3f} max {np.max(np.abs(e1)):.3f} | trim rms {rms(e2):.3f} max {np.max(np.abs(e2)):.3f}")

# ============================================================================ 5. the static law: current light law against a hinge-power law
print("\n=== 5. static law fitted to the template (feedback divider solved; ripple ignored)")
Lx_T = TL[TL >= TL[comp][0] - 4]; y_T = T_G[TL >= TL[comp][0] - 4]; src_T = T_SRC[TL >= TL[comp][0] - 4]
leak_T = c0_src[src_T]
def law_A(p):   # current: c = K (v - v0)^p above v0 (soft onset, exponent p)
    K, Lv0, pw = p; v0 = 10 ** (Lv0 / 20)
    return lambda Lv: (K * max(10 ** (Lv / 20) - v0, 0.0) ** pw)
def law_B(p):   # hinge-power: c = K ((v/vk)^p - 1) above vk (linear onset with slope K p, asymptote p)
    K, Lvk, pw = p; vk = 10 ** (Lvk / 20)
    return lambda Lv: (K * max((10 ** (Lv / 20) / vk) ** pw - 1.0, 0.0))
loss_src = np.array([g_amp - g0[k20], g_amp - g0[k21]])
def resid_law(p, law, inloop=False):
    out = np.empty(len(Lx_T))
    for src in (0, 1):
        m = src_T == src
        if inloop: out[m] = g_amp + solve_static(Lx_T[m], 0.0, c0_src[src], law(p)) - y_T[m]
        else: out[m] = g_amp - loss_src[src] + solve_static(Lx_T[m], 0.0, 0.0, law(p)) - y_T[m]
    return out
Lk_out = knee_T - (g_amp - g0[k20])          # the knee in divider-output level
rA = least_squares(resid_law, [1.0, Lk_out - 6, 2.0], args=(law_A,), bounds=([1e-4, -80, 0.5], [1e4, 20, 6]), x_scale=[1, 1, 0.1])
rB = least_squares(resid_law, [1.0, Lk_out, 1.6], args=(law_B,), bounds=([1e-3, -80, 0.5], [1e3, 20, 6]), x_scale=[0.1, 1, 0.1])
rB1 = least_squares(lambda p: resid_law([1.0, p[0], p[1]], law_B), [Lk_out, 1.6], bounds=([-80, 0.5], [20, 6]), x_scale=[1, 0.1])
rA_in = least_squares(resid_law, rA.x, args=(law_A, True), bounds=([1e-4, -80, 0.5], [1e4, 20, 6]), x_scale=[1, 1, 0.1])
rB_in = least_squares(resid_law, rB.x, args=(law_B, True), bounds=([1e-3, -80, 0.5], [1e3, 20, 6]), x_scale=[0.1, 1, 0.1])
print("  (the leak enters as a gain factor; with it as an additive in-loop conductance instead: law A rms %.3f, law B rms %.3f)" % (rms(rA_in.fun), rms(rB_in.fun)))
print(f"  law A (current, c = K (v - v0)^p): K {rA.x[0]:.3f}, v0 {rA.x[1]:.2f} dBFS, p {rA.x[2]:.3f} | rms {rms(rA.fun):.3f}, max {np.max(np.abs(rA.fun)):.3f}")
print(f"  law B (hinge-power, c = K ((v/vk)^p - 1)): K {rB.x[0]:.3f}, vk {rB.x[1]:.2f} dBFS, p {rB.x[2]:.3f} (asymptotic slope {rB.x[2] / (1 + rB.x[2]):.3f}) | rms {rms(rB.fun):.3f}, max {np.max(np.abs(rB.fun)):.3f}")
print(f"  law B with K = 1 (pure hard knee, fixed ratio): vk {rB1.x[0]:.2f} dBFS, p {rB1.x[1]:.3f} | rms {rms(rB1.fun):.3f}, max {np.max(np.abs(rB1.fun)):.3f}")
print("  residual law A by relative level:", np.round(rA.fun, 2))
print("  residual law B by relative level:", np.round(rB.fun, 2))
# the whole family with law B + measured leak + per-position drive (fitted 1-D), the static number for the report
print("  law B over all 24 positions (drive per position refitted, the flat-gain loss as a gain factor, amplifier droop from position 1):")
lawB = law_B(rB.x)
tot = []; drives_B = np.full(24, np.nan); rows_B = []
for kk in range(24):
    comp_k = GRc[kk] > 0.3
    if comp_k.sum() == 0:
        # no knee: the curve is the leak and the droop only
        pred = g0[kk] - droop; e = pred - G[kk]; tot.append(e); rows_B.append((kk + 1, np.nan, rms(e), np.max(np.abs(e)))); continue
    def cost(d):
        pred = g0[kk] - droop[comp_k] + solve_static(LEVELS[comp_k], 0.0, 0.0, lambda Lv: lawB(Lv + d))
        return float(np.sum((pred - G[kk][comp_k]) ** 2))
    d0 = drive[kk] if not np.isnan(drive[kk]) else 0.0
    r = minimize_scalar(cost, bounds=(d0 - 4, d0 + 4), method="bounded", options={"xatol": 1e-4})
    drives_B[kk] = r.x
    pred = g0[kk] - droop + solve_static(LEVELS, 0.0, 0.0, lambda Lv: lawB(Lv + r.x))
    e = pred - G[kk]; tot.append(e); rows_B.append((kk + 1, r.x, rms(e), np.max(np.abs(e))))
tot = np.concatenate(tot)
static_rms = rms(tot); static_max = float(np.max(np.abs(tot)))
print(f"  all positions, all levels (24 x 33): rms {static_rms:.3f} dB, max {static_max:.3f} dB")
for k_, d_, r_, m_ in rows_B:
    print(f"    pos {k_:2d}: drive {d_:+7.2f}  rms {r_:.3f}  max {m_:.3f}")
print("  drive table relative to position 20 (law B):", np.round(drives_B, 2))
a20 = float(cal[MODEL.fields["o_thr_db"][0] + 19])
print(f"  as o_thr_db with the current position-20 drive ({a20:.2f} dB) as the anchor:", np.round(a20 + drives_B, 2))
# the frequency series with law B and the fitted second-order low pass
fc2 = float(r_2.x[0])
print(f"  frequency series with law B and a second-order sidechain low pass at {fc2:.0f} Hz (drive lowered by its attenuation):")
for f in (100, 3000, 8000):
    att = lp_att(f, fc2, 2.0) - lp_att(1000.0, fc2, 2.0)
    aud = g0f[f] - g0[k20]                    # the audio-path loss at this frequency (applied after the loop, as measured at -40)
    top = 2 if f == 8000 else 14
    sel = FLEVELS <= top
    pred = g0[k20] + aud + solve_static(FLEVELS[sel], 0.0, 0.0, lambda Lv: lawB(Lv + drives_B[k20] - att))
    e = pred - GF[f][sel]
    print(f"    {f:5d} Hz: attenuation {att:.2f} dB, rms {rms(e):.3f}, max {np.max(np.abs(e)):.3f} (levels <= {top})")

# ============================================================================ 6. the discriminating captures (fit/data/discriminate_opto.json)
import json  # noqa: E402
D = json.load(open(os.path.join(HERE, "..", "..", "data", "discriminate_opto.json")))
print("\n=== 6. discriminate_opto.json: (A) fine knee sweeps, (C) HF rows, (D) no-GR bias, (E) above-knee steps and the slow tail")
# ---- (A) fine knee sweeps at thresholds 20 and 10 against law B with the section-5 drives
fineA = {}
for thr in ("20", "10"):
    kk = int(thr) - 1
    Lf = np.array(D["knee"][thr]["levels"]); gf = np.array(D["knee"][thr]["gain_db"]); grf = gf[0] - gf
    fineA[int(thr)] = (Lf, gf, grf)
    kn01 = crossing(Lf, grf, 0.1); kn1 = crossing(Lf, grf, 1.0); kn3 = crossing(Lf, grf, 3.0)
    onset = float(Lf[np.argmax(grf > 0.05)])
    sl = np.polyfit(Lf[grf > 3], grf[grf > 3], 1)[0]
    ph, h_rms, h_max = fit_hinge(Lf[grf > 0.05], grf[grf > 0.05])
    pred = g0[kk] + solve_static(Lf, 0.0, 0.0, lambda Lv: lawB(Lv + drives_B[kk]))
    e = pred - gf
    pkn1 = crossing(Lf, g0[kk] - pred, 1.0); pkn3 = crossing(Lf, g0[kk] - pred, 3.0)
    d05 = np.diff(grf[grf > 1.0]) / 0.5
    print(f"  (A) threshold {thr}: flat gain {gf[0]:+.3f} (protocol flat {g0[kk]:+.3f}); GR first > 0.05 at {onset:.1f}, crosses 0.1/1/3 dB at {kn01:.2f}/{kn1:.2f}/{kn3:.2f} dBFS "
          f"(1 -> 3 dB in {kn3 - kn1:.2f} dB); slope above 3 dB {sl:.3f}; hinge: knee {ph[1]:.2f} slope {ph[0]:.3f} width {ph[2]:.2f} dB (rms {h_rms:.3f})")
    print(f"      law B + drive {drives_B[kk]:+.2f} (from the 2 dB grid): rms {rms(e):.3f} max {np.max(np.abs(e)):.3f} dB over {len(Lf)} points; predicted 1/3 dB crossings {pkn1:.2f}/{pkn3:.2f}; "
          f"local slope per 0.5 dB above 1 dB GR: mean {d05.mean():.3f} sd {d05.std():.3f} min {d05.min():.2f} max {d05.max():.2f}")
# is the wobble the same function of GR at both thresholds (a property of the law, e.g. a table) or independent (noise)?
wob = {}
for thr in (20, 10):
    Lf, gf, grf = fineA[thr]; s = grf > 1.0
    A_ = np.vstack([Lf[s], np.ones(s.sum())]).T; wob[thr] = (grf[s], grf[s] - A_ @ np.linalg.lstsq(A_, grf[s], rcond=None)[0])
w10 = np.interp(wob[20][0], wob[10][0], wob[10][1])
print(f"      wobble about a straight line (GR > 1 dB): rms {rms(wob[20][1]):.3f} (thr 20) / {rms(wob[10][1]):.3f} (thr 10); correlation as a function of GR {np.corrcoef(wob[20][1], w10)[0, 1]:+.2f}")
# the join between the fine sweep (rms over the last second) and the protocol grid (lock-in): same plugin state, two capture runs
Lf20, gf20, grf20 = fineA[20]
join = [(l, gf20[list(Lf20).index(l)] - F[f"opto_static_t20_{int(l)}"]) for l in (-30.0, -26.0, -22.0, -18.0, -16.0)]
print("      fine sweep minus protocol grid at the shared levels (dB):", [(int(l), round(v, 3)) for l, v in join])

# ---- the knee width that an instantaneous threshold produces: law A applied to |v sin| then cycle-averaged, against law A on the amplitude
KA, v0A_db, pA = rA.x; v0A = 10 ** (v0A_db / 20)
thg = np.linspace(0, np.pi / 2, 4001)
_trap = getattr(np, "trapezoid", None) or np.trapz
def lawA_cycle(Lv):
    a = 10 ** (Lv / 20)
    e = a * np.sin(thg) - v0A
    m = _trap(np.where(e > 0, np.abs(e) ** pA, 0.0), thg) / (np.pi / 2)
    return KA * m / _trap(np.sin(thg) ** pA, thg) * (np.pi / 2)    # normalised so the asymptote matches the amplitude law
Lq = np.arange(-34.0, -10.0, 0.25)
for name, law in (("amplitude (envelope) law A", law_A(rA.x)), ("instantaneous law A, cycle-averaged", lawA_cycle)):
    gq = g0[k20] + solve_static(Lq, 0.0, 0.0, law); grq = g0[k20] - gq
    c01, c1, c3 = crossing(Lq, grq, 0.1), crossing(Lq, grq, 1.0), crossing(Lq, grq, 3.0)
    ef = np.interp(Lf20, Lq, gq) - gf20
    print(f"      {name:38s}: GR crosses 0.1/1/3 dB at {c01:.2f}/{c1:.2f}/{c3:.2f} dBFS (0.1 -> 3 dB in {c3 - c01:.2f} dB; measured {crossing(Lf20, grf20, 3.0) - crossing(Lf20, grf20, 0.1):.2f}); "
          f"against the fine sweep rms {rms(ef):.3f} max {np.max(np.abs(ef)):.3f}")

# ---- (C) the sidechain response from the HF rows: GR at -10 and 0 dBFS by frequency -> equivalent 1 kHz input level -> attenuation
Hf = np.array(D["hf"]["freqs"]); r40 = np.array(D["hf"]["rows"]["-40.0"]); rows_hf = {-10.0: np.array(D["hf"]["rows"]["-10.0"]), 0.0: np.array(D["hf"]["rows"]["0.0"])}
print(f"  (C) audio path at -40 dBFS relative to 1 kHz (dB): {np.round(r40 - r40[0], 3)} (largest {np.max(np.abs(r40 - r40[0])):.2f} at {Hf[np.argmax(np.abs(r40 - r40[0]))]:.0f} Hz: not the mechanism)")
# the 1 kHz static curve as GR(level): fine sweep for the knee, the interleaved template above it
inv_L = np.concatenate([Lf20[grf20 > 0.005], TL[TL > Lf20[-1]]]); inv_GR = np.concatenate([grf20[grf20 > 0.005], T_GR[TL > Lf20[-1]]])
o = np.argsort(inv_GR); inv_L = inv_L[o]; inv_GR = inv_GR[o]
att_rows = {}; bound_rows = {}
for lvl, row in rows_hf.items():
    grh = r40 - row
    Leq = np.interp(grh, inv_GR, inv_L)
    att = lvl - Leq                                   # sidechain attenuation at f (dB); positive = the sidechain sees less
    bound = grh < inv_GR[0]                           # no GR at all: only a lower bound on the attenuation
    att_rows[lvl] = att - att[0]; bound_rows[lvl] = bound
    print(f"      at {lvl:+.0f} dBFS: GR by frequency {np.round(grh, 2)} -> attenuation relative to 1 kHz {np.round(att - att[0], 2)}" + (f" (bound only at {Hf[bound].astype(int)} Hz)" if bound.any() else ""))
sel_c = np.concatenate([~bound_rows[l] for l in (-10.0, 0.0)]); f_c = np.concatenate([Hf, Hf])[sel_c]; a_c = np.concatenate([att_rows[-10.0], att_rows[0.0]])[sel_c]
def fitC(model, p0, lo, hi, xs, name):
    r = least_squares(lambda p: model(f_c, p) - model(1000.0, p) - a_c, p0, bounds=(lo, hi), x_scale=xs)
    e = r.fun; print(f"      {name:52s} " + " ".join(f"{v:9.0f}" if abs(v) > 50 else f"{v:9.3f}" for v in r.x) + f" | rms {rms(e):.2f} max {np.max(np.abs(e)):.2f} dB | residual by f: {np.round(e, 2)}")
    return r
mC = {}
mC["lp1"] = fitC(lambda f, p: lp_att(f, p[0], 1.0), [4000.0], [500], [60000], [1000], "first-order low pass: fc")
mC["lp2"] = fitC(lambda f, p: lp_att(f, p[0], 2.0), [5000.0], [500], [60000], [1000], "second-order Butterworth: fc")
mC["lpN"] = fitC(lambda f, p: lp_att(f, p[0], p[1]), [5000.0, 2.0], [500, 0.5], [60000, 6], [1000, 0.5], "low pass, free order: fc, N")
mC["lp2q"] = fitC(lambda f, p: lp2q_att(f, p[0], p[1]), [5000.0, 0.7], [500, 0.3], [60000, 3.0], [1000, 0.1], "second order, fc and Q free")
mC["2real"] = fitC(lambda f, p: lp_att(f, p[0], 1.0) + lp_att(f, p[1], 1.0), [4000.0, 12000.0], [500, 500], [60000, 60000], [1000, 1000], "two real poles: fc1, fc2")
mC["lp3"] = fitC(lambda f, p: lp_att(f, p[0], 3.0), [6000.0], [500], [60000], [1000], "third-order Butterworth: fc")
# the same attenuation must also explain the 3 kHz / 8 kHz static series (section 1): a(3k), a(8k) from the recommended filter
fc_rec, q_rec = mC["lp2q"].x
print(f"      recommended second order fc {fc_rec:.0f} Hz Q {q_rec:.2f}: a(3 kHz) {lp2q_att(3000, fc_rec, q_rec) - lp2q_att(1000, fc_rec, q_rec):.2f} dB (section 1: {aa[1]:.2f}), a(8 kHz) {lp2q_att(8000, fc_rec, q_rec) - lp2q_att(1000, fc_rec, q_rec):.2f} dB (section 1: {aa[2]:.2f}); "
      f"a(1 kHz) itself {lp2q_att(1000, fc_rec, q_rec):.3f} dB and a(100 Hz) {lp2q_att(100, fc_rec, q_rec):.4f} dB (absorbed in the drive table)")

# ---- (D) the no-GR gain at -70 and -50 dBFS for 8 positions: bias or floor, and the leak law
ksD = np.array([int(k) for k in D["nogr"]["-50.0"]]); g70 = np.array([D["nogr"]["-70.0"][str(k)] for k in ksD]); g50 = np.array([D["nogr"]["-50.0"][str(k)] for k in ksD])
print(f"  (D) positions {ksD}: gain at -50 {np.round(g50, 3)}; -70 minus -50 (dB) {np.round(g70 - g50, 5)} (largest {np.max(np.abs(g70 - g50)):.5f}: a bias, not a compression floor)")
pred_D = leak_model([g_amp, r_iii.x[0], r_iii.x[1]], drive[ksD - 1])
pred_D[np.isnan(drive[ksD - 1])] = g_amp
print(f"      leak law from section 3 (c0 = {r_iii.x[0]:.4f} A^{r_iii.x[1]:.2f}) predicts {np.round(pred_D, 3)}: residual rms {rms(pred_D - g50):.4f} max {np.max(np.abs(pred_D - g50)):.4f} dB")
lossD = g50[0] - g50
print(f"      loss relative to position 1 at -50: {np.round(lossD, 3)} (the parent's numbers: 0.006 at 8, 0.03 at 12, 0.073 at 16, 0.18 at 20, 0.285 at 22, 0.445 at 24)")
okD = ~np.isnan(drive[ksD - 1]) & (ksD >= 8)
qD = np.polyfit(drive[ksD - 1][okD] / 20 * np.log(10), np.log(lossD[okD]), 1)[0]
print(f"      log(loss) against log(A) on positions 8..24: exponent {qD:.2f} (2 = the square of the sidechain drive)")

# ---- (E) above-knee steps: attack and release times, the absence of a tail, and what an additive in-loop tail would have done
print("  (E) above-knee steps at threshold 20 (per-cycle gain, 1 ms cycles; step at 2.0 s, back at 4.0 s):")
tails = {}
for key, s in D["steps"].items():
    gc = np.array(s["per_cycle_gain_db"]); a, b = s["from"], s["to"]
    pre = gc[1800:1999].mean(); mid = gc[3800:3999].mean(); end = gc[-200:].mean()
    up = gc[2000:4000]; dn = gc[4000:]
    t50a = int(np.argmax(up <= pre + 0.5 * (mid - pre))); t90a = int(np.argmax(up <= pre + 0.9 * (mid - pre)))
    t50r = int(np.argmax(dn >= mid + 0.5 * (end - mid))); t90r = int(np.argmax(dn >= mid + 0.9 * (end - mid)))
    resid = end - dn                                   # GR still to be released (dB), positive
    t_done = int(np.argmax(resid < 0.01))
    sb = float(np.interp(b, TL, T_G)) if b not in LEVELS else float(F[f"opto_static_t20_{int(b)}"])
    print(f"      {a:+.0f} -> {b:+.0f} -> {a:+.0f} dBFS: gain {pre:+.3f} -> {mid:+.3f} -> {end:+.3f} dB (static curve: {F[f'opto_static_t20_{int(a)}']:+.3f} / {sb:+.3f}); "
          f"attack 50/90 % at {t50a}/{t90a} ms; release 50/90 % at {t50r}/{t90r} ms, within 0.01 dB by {t_done} ms; residual GR at 0.1/0.5/1/2 s: {resid[100]:.4f}/{resid[500]:.4f}/{resid[1000]:.4f}/{resid[1990]:.4f} dB")
    tails[key] = (a, b, resid)
# the slow tail below the knee: from the protocol bursts (pre -50, burst 2 s, post 3 s; env = per-cycle gain), as a conductance
print("      slow tail after bursts that fall BELOW the knee (opto_burst_*, opto_blen_*): residual GR (dB) and the equivalent conductance c_tail = 10^(GR/20) - 1")
print("      item              GR in burst | residual GR at +0.1 / +0.3 / +1 / +2 / +2.9 s | c_tail at +0.3 s | c_tail / c_burst")
tail_c = {}
for iid, n_end in [(f"opto_burst_{lb}", 2500) for lb in (-26, -18, -10, -2)] + [(f"opto_blen_{bl}", 500 + int(bl * 1000)) for bl in (0.05, 0.2, 1.0, 4.0)]:
    env = np.asarray(F[iid]); base = env[50:450].mean()
    gr_b = base - env[n_end - 100:n_end - 1].mean()
    res_t = [base - env[n_end + t] for t in (100, 300, 1000, 2000, 2900)]
    c_t = 10 ** (res_t[1] / 20) - 1; c_b = 10 ** (gr_b / 20) - 1
    tail_c[iid] = (gr_b, res_t, c_t, c_b)
    print(f"      {iid:16s}  {gr_b:6.2f} dB   | {res_t[0]:.3f} / {res_t[1]:.3f} / {res_t[2]:.3f} / {res_t[3]:.3f} / {res_t[4]:.3f} | {c_t:.4f} | {c_t / max(c_b, 1e-9):.4f}")
# what would the -10 dBFS burst's tail do to the -10 -> -20 step if it were an additive in-loop conductance? closed-loop static solve
gr_b, res_t, c_t, c_b = tail_c["opto_burst_-10"]
c20 = 10 ** ((g_amp - float(F["opto_static_t20_-20"])) / 20) - 1.0 - c0[k20]
print(f"      opto_burst_-10: c_tail 0.3 s after the burst {c_t:.4f}; the signal conductance at -20 dBFS is {c20:.3f}")
for t_ms, res_v in zip((100, 300, 1000, 2000), res_t[:4]):
    c_tl = 10 ** (res_v / 20) - 1
    with_tail = g_amp + solve_static(np.array([-20.0]), 0.0, c0[k20] + c_tl)[0]; without = g_amp + solve_static(np.array([-20.0]), 0.0, c0[k20])[0]
    meas = tails["-20_-10"][2][min(t_ms, 1990)]
    print(f"        at +{t_ms:4d} ms: additive in-loop conductance {c_tl:.4f} would leave {without - with_tail:.3f} dB of extra GR at -20 dBFS; measured after the -10 -> -20 step: {meas:.4f} dB")
print("      a floor that is only seen when the fast conductance has fallen below it (c = max(fast, floor), or the cell's dark-resistance recovery) leaves 0 above the knee, as measured")

# ---- law A (amplitude law, the better fit of the knee) over all 24 positions, same procedure as law B
print("  law A (amplitude, c = K (a - v0)^p) over all 24 positions (drive per position refitted, leak as a gain factor, droop from position 1):")
lawA_ = law_A(rA.x)
totA = []; drives_A = np.full(24, np.nan); rows_A = []
for kk in range(24):
    comp_k = GRc[kk] > 0.3
    if comp_k.sum() == 0:
        pred = g0[kk] - droop; e = pred - G[kk]; totA.append(e); rows_A.append((kk + 1, np.nan, rms(e), np.max(np.abs(e)))); continue
    def costA(d):
        pred = g0[kk] - droop[comp_k] + solve_static(LEVELS[comp_k], 0.0, 0.0, lambda Lv: lawA_(Lv + d))
        return float(np.sum((pred - G[kk][comp_k]) ** 2))
    d0 = drives_B[kk] if not np.isnan(drives_B[kk]) else 0.0
    r = minimize_scalar(costA, bounds=(d0 - 4, d0 + 4), method="bounded", options={"xatol": 1e-4})
    drives_A[kk] = r.x
    pred = g0[kk] - droop + solve_static(LEVELS, 0.0, 0.0, lambda Lv: lawA_(Lv + r.x))
    e = pred - G[kk]; totA.append(e); rows_A.append((kk + 1, r.x, rms(e), np.max(np.abs(e))))
totA = np.concatenate(totA)
staticA_rms = rms(totA); staticA_max = float(np.max(np.abs(totA)))
print(f"  all positions, all levels (24 x 33): rms {staticA_rms:.3f} dB, max {staticA_max:.3f} dB (law B: {static_rms:.3f} / {static_max:.3f})")
print("  per position rms:", np.round([r_ for _, _, r_, _ in rows_A], 3))
print("  drive table relative to position 20 (law A):", np.round(drives_A, 2))
print(f"  as o_thr_db with the current position-20 drive ({a20:.2f} dB) as the anchor:", np.round(a20 + drives_A, 2))
for thr in (20, 10):
    Lf, gf, grf = fineA[thr]; kk = thr - 1
    e = g0[kk] + solve_static(Lf, 0.0, 0.0, lambda Lv: lawA_(Lv + drives_A[kk])) - gf
    print(f"  fine sweep at threshold {thr} with law A and its 2 dB-grid drive: rms {rms(e):.3f} max {np.max(np.abs(e)):.3f} dB")
# the tail time constant below the knee (from the -10 dBFS burst: residual GR at 0.1, 0.3, 1 s)
gr_b, res_t, c_t, c_b = tail_c["opto_burst_-10"]
tau1 = 0.2 / np.log(res_t[0] / res_t[1]); tau2 = 0.7 / np.log(res_t[1] / res_t[2])
print(f"  tail below the knee (opto_burst_-10): exponential time constant {tau1 * 1e3:.0f} ms (0.1 -> 0.3 s) and {tau2 * 1e3:.0f} ms (0.3 -> 1 s); amplitude extrapolated to the burst end {res_t[0] * np.exp(0.1 / tau1):.2f} dB; "
      f"independent of burst depth (-18..-2 dBFS) and length (0.05..4 s), see the table")

print("\n=== summary numbers")
print(f"  static law A (amplitude) + leak + drives: rms {staticA_rms:.3f} max {staticA_max:.3f} dB over 24 x 33 points; law A: K {rA.x[0]:.2f}, v0 {rA.x[1]:.2f} dBFS (divider output), p {rA.x[2]:.3f}")
print(f"  template: slope {p_hT[0]:.3f} dB/dB, knee width {p_hT[2]:.2f} dB, knee {knee_T:.2f} dBFS at position 20; wobble about a straight line rms {rms(lin_res):.3f}")
print(f"  sidechain low pass: order {r_free.x[1]:.2f}, fc {r_free.x[0]:.0f} Hz (second order: fc {fc2:.0f} Hz; first order refuted: residual {np.round(r_1.fun, 2)})")
print(f"  step law (positions 5..24): {best5[2][1]:.2f} dB/step below position {best5[0]}, {best5[2][2]:.2f} above; rms {best5[1]:.3f}")
print(f"  flat-gain loss: 20 log10(1 + {r_iii.x[0]:.4f} A^{r_iii.x[1]:.2f}) (A relative to position 20), rms {rms(r_iii.fun):.3f}; linear trim in position rms {rms(e_i):.3f}, in drive {rms(e_ii):.3f}")
print(f"  placement: gain factor outside the loop and additive in-loop conductance both describe the family (rms {rms(e_trim):.3f} vs {rms(e_all):.3f}); see the drive-step regularity above")
print(f"  static law B + leak + drives: rms {static_rms:.3f} max {static_max:.3f} dB over 24 x 33 points")
print(f"  (C) sidechain low pass from the 8-frequency HF rows: second order fc {mC['lp2q'].x[0]:.0f} Hz Q {mC['lp2q'].x[1]:.2f} (rms {rms(mC['lp2q'].fun):.2f}); Butterworth fc {mC['lp2'].x[0]:.0f} (rms {rms(mC['lp2'].fun):.2f}); "
      f"free order N {mC['lpN'].x[1]:.2f}; first order rms {rms(mC['lp1'].fun):.2f}; two real poles {mC['2real'].x[0]:.0f}/{mC['2real'].x[1]:.0f} Hz rms {rms(mC['2real'].fun):.2f}")
print(f"  (D) leak law predicts the -50 dBFS row within {np.max(np.abs(pred_D - g50)):.4f} dB; -70 and -50 agree within {np.max(np.abs(g70 - g50)):.5f} dB")
