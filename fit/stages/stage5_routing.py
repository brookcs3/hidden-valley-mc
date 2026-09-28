# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Stage 5: the shared routing constants and the profile constants.

5a. Sidechain high-pass corner sc_hz: the discrete stage's gain reduction against frequency with the filter in (scf_disc_*_In) at
    -10 dBFS, rendered through the engine; one scalar, minimised on the discrete items (the static family is exact) and reported on
    the optical items and on the filter-out items (which must not move).
5b. Stereo link d_link: the discrete stage's gain on each channel with the left at -10 dBFS and the right swept (link_disc_*_ch*);
    the model sums the two sidechain signals before the rectifier, d_link is the scale of that sum. Reported: the anti-phase pair
    (must not compress), the 1000/1100 Hz pair, and the optical link items.
5c. Profile constants that are not fitted to the reference but taken from the hardware evidence (research/re, plan section 6.5):
    MEASURED UNIT: o_hw_lp_hz 18 kHz (first order; -1.1 dB at 10 kHz and -3.8 dB at 20 kHz on the torn-down unit bracket 16.9 to
    18.6 kHz), the opto HF loss that grows with GR (one owner's data: -0.25 dB at 10 kHz at no GR, -1.5 dB at 3 dB GR; a one-pole
    whose corner is o_hw_grloss_hz / (1 + o_hw_grloss_k * cond), solved from those two points), and d_hwunit_a2 (H2 -50 dBc at
    about -3 dBu = -17 dBFS: a2 = 2 * 10^(-50/20) / 10^(-17/20)). HARDWARE: Iron's LF lift and noise stay at the priors in
    src/dsp/Calibration.hpp (from the Flotown renders and the teardown FFT). These are written with a note so the generated header
    records their provenance."""
import os, sys, numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import F, ITEMS, MODEL, load_cal, save_cal, render_item  # noqa: E402

SCF_F = (20, 30, 40, 60, 90, 130, 200, 400, 1000)
LINK_R = (-80, -40, -30, -20, -15, -10, -5)


def set_named(cal, name, val):
    cal[MODEL.fields[name][0]] = val


def resid(ids, cal):
    return np.array([render_item(ITEMS[i], cal) - F[i] for i in ids])


def run():
    cal = load_cal()
    # ---- 5a sidechain corner
    fit_ids = [f"scf_disc_{f}_In" for f in SCF_F]
    def cost(fc):
        c = cal.copy(); set_named(c, "sc_hz", fc)
        return float(np.sum(resid(fit_ids, c) ** 2))
    r = minimize_scalar(cost, bounds=(20.0, 400.0), method="bounded", options={"xatol": 0.5})
    set_named(cal, "sc_hz", r.x)
    e_in = resid(fit_ids, cal); e_out = resid([f"scf_disc_{f}_Out" for f in SCF_F], cal)
    e_opto = resid([f"scf_opto_{f}_In" for f in SCF_F], cal)
    print(f"  sidechain high-pass: {r.x:.1f} Hz | discrete, filter in: rms {np.sqrt(np.mean(e_in ** 2)):.3f} dB max {np.max(np.abs(e_in)):.2f}"
          f" | filter out: rms {np.sqrt(np.mean(e_out ** 2)):.3f} | optical, filter in: rms {np.sqrt(np.mean(e_opto ** 2)):.3f} max {np.max(np.abs(e_opto)):.2f}")
    print("  per frequency (Hz: dB residual, discrete in):", {f: round(float(e), 2) for f, e in zip(SCF_F, e_in)})
    # ---- 5b stereo link
    link_ids = [f"link_disc_{lr}_ch{ch}" for lr in LINK_R for ch in (0, 1)]
    def cost2(k):
        c = cal.copy(); set_named(c, "d_link", k)
        return float(np.sum(resid(link_ids, c) ** 2))
    r2 = minimize_scalar(cost2, bounds=(0.25, 1.5), method="bounded", options={"xatol": 1e-3})
    set_named(cal, "d_link", r2.x)
    e = resid(link_ids, cal)
    print(f"  stereo link scale: {r2.x:.3f} (0.5 = the sum of the two channels, halved) | discrete link items: rms {np.sqrt(np.mean(e ** 2)):.3f} dB max {np.max(np.abs(e)):.2f}")
    print("  right level -> residual ch0/ch1:", {lr: (round(float(e[2 * i]), 2), round(float(e[2 * i + 1]), 2)) for i, lr in enumerate(LINK_R)})
    for iid in ("link_disc_anti", "link_disc_f1100", "link_opto_anti", "link_opto_f1100"):
        m = render_item(ITEMS[iid], cal); print(f"  {iid}: model {m:.2f} ref {F[iid]:.2f}")
    eo = resid([f"link_opto_{lr}_ch{ch}" for lr in LINK_R for ch in (0, 1)], cal)
    print(f"  optical link items (shape from stage 4): rms {np.sqrt(np.mean(eo ** 2)):.3f} dB max {np.max(np.abs(eo)):.2f}")
    # ---- 5c profile constants from the hardware evidence
    f0, hf = 10000.0, np.sqrt(1.0 / 10 ** (-0.25 / 10) - 1.0)   # |H|^2 = 10^(-0.025): (f/fc)^2 = 1/|H|^2 - 1
    fc0 = f0 / hf                                              # corner at no gain reduction
    hg = np.sqrt(1.0 / 10 ** (-1.5 / 10) - 1.0); fc3 = f0 / hg  # corner at 3 dB of gain reduction
    cond3 = 10 ** (3.0 / 20) - 1.0
    k = (fc0 / fc3 - 1.0) / cond3
    a2 = 2.0 * 10 ** (-50.0 / 20) / 10 ** (-17.0 / 20)
    set_named(cal, "o_hw_lp_hz", 18000.0); set_named(cal, "o_hw_grloss_hz", fc0); set_named(cal, "o_hw_grloss_k", k); set_named(cal, "d_hwunit_a2", a2)
    print(f"  MEASURED UNIT: opto path pole 18000 Hz; HF loss corner {fc0:.0f} Hz at no GR falling as 1/(1 + {k:.2f} cond); cell a2 {a2:.4f} (H2 -50 dBc at -17 dBFS)")
    save_cal(cal, {"stage5": f"sidechain corner from scf_disc_*_In, link scale from link_disc_*; MEASURED UNIT constants from the hardware evidence (not fitted to the reference): 18 kHz pole, HF loss corner {fc0:.0f} Hz / k {k:.2f}, cell a2 {a2:.4f}; HARDWARE Iron lift and noise at their priors; CLASS A profile constants (ca_*) derived in docs/class-a-profile.md, at their priors"},
             fields=["sc_hz", "d_link", "o_hw_lp_hz", "o_hw_grloss_hz", "o_hw_grloss_k", "d_hwunit_a2", "hw_iron_lift_db", "hw_iron_lift_hz", "hw_iron_lift_q", "hw_iron_noise_db", "ca_in_db", "ca_out_db", "ca_in_fl_hz", "ca_in_lp_hz", "ca_a2", "ca_a3", "ca_a2_env", "ca_ceil_db", "ca_ceil_q", "ca_ceil_asym_db", "ca_d_a2_scale", "ca_noise_db"])

if __name__ == "__main__":
    run()
