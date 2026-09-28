# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discriminating tests (build plan 6.3) that fix the block diagram before any fitting. Prints a report and writes discriminate.json."""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import pa
from pa import both, sine, rms_db, harmonics, FS
P = pa.Ref(); R = {"reference": {"version": P.version, "sha256": P.sha256}}
print("reference", P.version, P.sha256[:16])
ss = lambda y, ch=0, a=1.5: rms_db(y[ch, int(a * FS):])

if os.environ.get("SKIP12"):
    pass
else:
  pass
# 1. opto sidechain tap: GR vs optical gain at fixed input and threshold
g_unity = {}
rows = []
for g in (1, 4, 7, 10, 12, 16, 20, 24):
    y0 = P.run(sine(-40, secs=2), **both(optical_bypass="In", optical_threshold=20, optical_gain=g))
    y1 = P.run(sine(-10, secs=2), **both(optical_bypass="In", optical_threshold=20, optical_gain=g))
    gain_db = ss(y0) + 40 + 3.0103
    gr = (-10 - 3.0103 + gain_db) - ss(y1)
    rows.append((g, round(gain_db, 3), round(gr, 3)))
R["opto_gain_vs_gr"] = rows; print("1) optical gain -> (makeup dB, GR dB at -10 dBFS, thr 20):", rows)

# 2. discrete: GR vs discrete gain (feed-forward check)
rows = []
for g in (1, 7, 12, 18, 24):
    y0 = P.run(sine(-50, secs=2), **both(discrete_bypass="In", discrete_threshold=16, discrete_gain=g))
    y1 = P.run(sine(-10, secs=2), **both(discrete_bypass="In", discrete_threshold=16, discrete_gain=g))
    gain_db = ss(y0) + 50 + 3.0103
    rows.append((g, round(gain_db, 3), round((-10 - 3.0103 + gain_db) - ss(y1), 3)))
R["disc_gain_vs_gr"] = rows; print("2) discrete gain -> (makeup dB, GR dB):", rows)

# 3. Stereo mode: do the right-channel controls still act?
x = sine(-10, secs=2)
yA = P.run(np.stack([x, x]), **{**both(discrete_bypass="In", discrete_threshold=16), "mode": "Stereo", "discrete_threshold_2": 4})
yB = P.run(np.stack([x, x]), **{**both(discrete_bypass="In", discrete_threshold=16), "mode": "Dual Mono", "discrete_threshold_2": 4})
R["stereo_controls"] = {"stereo_L": ss(yA, 0), "stereo_R": ss(yA, 1), "dual_L": ss(yB, 0), "dual_R": ss(yB, 1)}
print("3) R threshold 4 vs L 16:", {k: round(v, 2) for k, v in R["stereo_controls"].items()})
yA = P.run(np.stack([x, x]), **{"mode": "Stereo", "transformer_2": "Iron"})
h1, _ = harmonics(yA[1, FS:], 1000); h0, _ = harmonics(yA[0, FS:], 1000)
R["stereo_transformer_R_iron"] = {"L_h2": h0[0], "R_h2": h1[0]}; print("   stereo, R transformer Iron: H2 L/R", round(h0[0], 1), round(h1[0], 1))

# 4. Mix: where is the dry tapped?
x = sine(-10, secs=2)
st = both(discrete_bypass="In", discrete_threshold=20, transformer="Iron")
y0 = P.run(x, **{**st, "mix": 0.0}); y100 = P.run(x, **{**st, "mix": 100.0}); y50 = P.run(x, **{**st, "mix": 50.0})
ypath = P.run(x, **both(transformer="Iron"))
R["mix"] = {"mix0_minus_input_db": rms_db(y0[0] - x.astype(np.float32)), "mix0_minus_xfmr_path_db": rms_db(y0[0] - ypath[0]),
            "mix0_level": ss(y0), "mix50_level": ss(y50), "mix100_level": ss(y100)}
print("4) mix:", {k: round(v, 2) for k, v in R["mix"].items()})

# 5. sidechain filter on the opto: 50 Hz vs 1 kHz
rows = {}
for f in (50.0, 1000.0):
    for sc in ("Out", "In"):
        y = P.run(sine(-10, f=f, secs=3), **both(optical_bypass="In", optical_threshold=20, sidechain_filter=sc))
        y0 = P.run(sine(-40, f=f, secs=3), **both(optical_bypass="In", optical_threshold=20, sidechain_filter=sc))
        rows[f"{int(f)}Hz_{sc}"] = round((ss(y0, a=2) + 30) - ss(y, a=2), 2)
R["opto_scfilter_gr"] = rows; print("5) opto GR with SC filter:", rows)

# 6. transformer clip onset vs frequency (Nickel, Iron): level where THD reaches -40 dB, and THD at +6 dBFS
rows = {}
for core in ("Nickel", "Iron", "Steel"):
    for f in (20.0, 40.0, 80.0, 160.0, 320.0, 1000.0):
        thds = []
        for lvl in np.arange(-6, 19, 1.0):
            y = P.run(sine(lvl, f=f, secs=1.5), **both(transformer=core))
            _, thd = harmonics(y[0, FS // 2:], f)
            thds.append((float(lvl), round(thd, 1)))
        onset = next((l for l, t in thds if t > -40), None)
        rows[f"{core}_{int(f)}"] = {"onset_-40dB": onset, "thd_at_+6": dict(thds)[6.0], "thd_at_+12": dict(thds)[12.0], "curve": thds}
        print(f"6) {core:6s} {int(f):5d} Hz: THD>-40 dB from {onset} dBFS; THD +6: {dict(thds)[6.0]}  +12: {dict(thds)[12.0]}  +18: {dict(thds)[18.0]}")
R["xfmr_clip"] = rows

# 7. discrete stage H2 vs level at no GR, and op-amp/limiter headroom with the discrete stage in
rows = []
for lvl in (-30, -20, -10, 0, 6):
    y = P.run(sine(lvl, secs=2), **both(discrete_bypass="In", discrete_threshold=1, discrete_gain=7))
    hd, thd = harmonics(y[0, FS:], 1000)
    rows.append((lvl, round(hd[0], 1), round(hd[1], 1), round(thd, 1)))
R["disc_h2_vs_level"] = rows; print("7) discrete no-GR (level, H2, H3, THD):", rows)
rows = []
for lvl in (-30, -10, 0, 6):
    y = P.run(sine(lvl, secs=2), **both(optical_bypass="In", optical_threshold=1, optical_gain=12))
    hd, thd = harmonics(y[0, FS:], 1000)
    rows.append((lvl, round(hd[0], 1), round(hd[1], 1), round(thd, 1)))
R["opto_nogr_h_vs_level"] = rows; print("   opto no-GR (level, H2, H3, THD):", rows)

# 8. detector type: sine vs square at equal peak, and at equal RMS
t = np.arange(3 * FS) / FS
sq = np.sign(np.sin(2 * np.pi * 1000 * t)) * 10 ** (-10 / 20)
from scipy.signal import butter, sosfilt
sq = sosfilt(butter(8, 18000, fs=FS, output="sos"), sq)   # band-limited square
si = sine(-10, secs=3)
res = {}
for name, sig in (("sine_pk-10", si), ("square_pk-10", sq), ("square_rms_eq_sine", sq * 10 ** (-3.0103 / 20))):
    y = P.run(sig, **both(discrete_bypass="In", discrete_threshold=16, discrete_attack=1.0, discrete_recover="0.5 s"))
    y0 = P.run(sig * 10 ** (-40 / 20), **both(discrete_bypass="In", discrete_threshold=16))
    res[name] = round((ss(y0, a=2) + 40) - ss(y, a=2), 2)
R["crest"] = res; print("8) discrete GR by waveform:", res)
res = {}
for name, sig in (("sine_pk-10", si), ("square_pk-10", sq), ("square_rms_eq_sine", sq * 10 ** (-3.0103 / 20))):
    y = P.run(sig, **both(optical_bypass="In", optical_threshold=20))
    y0 = P.run(sig * 10 ** (-40 / 20), **both(optical_bypass="In", optical_threshold=20))
    res[name] = round((ss(y0, a=2) + 40) - ss(y, a=2), 2)
R["crest_opto"] = res; print("   opto GR by waveform:", res)
json.dump(R, open(os.path.join(os.path.dirname(__file__), "..", "data", "discriminate.json"), "w"), indent=1)
