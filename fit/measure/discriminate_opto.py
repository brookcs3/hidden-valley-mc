# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discriminating measurements of the reference's OPTICAL stage that the main protocol lacks, taken to settle its static law:
 A. a fine level sweep across the knee (0.5 dB steps) at thresholds 20 and 10, 1 kHz: knee level and width, slope above it;
 B. the static curve extended to +26 dBFS at threshold 20: does the slope stay constant;
 C. sidechain high-frequency behaviour: gain at -40 (no GR), -10 and 0 dBFS from 2 to 16 kHz at threshold 20; the -40 dBFS row is
    the audio path, the GR difference isolates the sidechain response;
 D. the no-GR gain against threshold at -70 and -50 dBFS (bias or compression floor);
 E. steps between two ABOVE-knee levels (-20 -> -10 -> -20, -10 -> 0 -> -10, -20 -> +5 -> -20) at threshold 20: per-cycle gain
    trajectories, to separate a feedback loop (rate depends on the distance from equilibrium) from feed-forward smoothing.
Writes fit/data/discriminate_opto.json and prints a report. Needs the licensed reference plug-in (fit/measure/pa.py).
usage: python3 -u fit/measure/discriminate_opto.py"""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import pa
from pa import both, sine, rms_db, FS

P = pa.Ref()
R = {"meta": {"reference": "Plugin Alliance / Brainworx 'Shadow Hills Mastering Compressor' (VST3, licensed copy)", "version": P.version, "sha256": P.sha256, "fs": FS,
              "gain_db": "steady output level minus input level over the last second, channel 0, dB; includes the optical make-up at position 12"}}
print("reference", P.version, P.sha256[:16])
OPTO = dict(optical_bypass="In")


def gain(level, f=1000.0, secs=3.0, **kw):
    x = sine(level, f=f, secs=secs)
    y = P.run(x, **both(**OPTO, **kw))
    n = int(FS)
    return rms_db(y[0, -n:]) - rms_db(x[-n:])


# A. fine knee sweep
R["knee"] = {}
for thr, lo, hi in ((20, -34.0, -16.0), (10, -24.0, -6.0)):
    levels = np.arange(lo, hi + 0.01, 0.5)
    g = [gain(float(l), optical_threshold=thr) for l in levels]
    R["knee"][str(thr)] = {"levels": levels.tolist(), "gain_db": g}
    g0 = g[0]
    gr = [g0 - v for v in g]
    k1 = float(np.interp(1.0, gr, levels)); k3 = float(np.interp(3.0, gr, levels)); k6 = float(np.interp(6.0, gr, levels))
    slope_hi = (g[-1] - g[-5]) / (levels[-1] - levels[-5])
    print(f"A) threshold {thr}: gain at {lo:.0f} dBFS {g0:+.2f}; GR crosses 1/3/6 dB at {k1:.2f}/{k3:.2f}/{k6:.2f} dBFS; slope over the top 2 dB {slope_hi:+.3f} dB/dB")
    print("   GR (dB) per 0.5 dB:", np.round(gr, 2).tolist())

# B. extended level
levels = [-10.0, -4.0, 2.0, 8.0, 14.0, 20.0, 26.0]
g = [gain(l, optical_threshold=20) for l in levels]
R["extended"] = {"levels": levels, "gain_db": g}
print("B) threshold 20, level -> gain:", [(l, round(v, 2)) for l, v in zip(levels, g)], " slopes:", np.round(np.diff(g) / np.diff(levels), 3).tolist())

# C. sidechain HF
freqs = [1000.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 16000.0]
R["hf"] = {"freqs": freqs, "rows": {}}
for lvl in (-40.0, -10.0, 0.0):
    g = [gain(lvl, f=f, optical_threshold=20) for f in freqs]
    R["hf"]["rows"][str(lvl)] = g
    print(f"C) level {lvl:+.0f} dBFS, gain by frequency:", [(int(f), round(v, 2)) for f, v in zip(freqs, g)])
gr10 = [R["hf"]["rows"]["-40.0"][i] - R["hf"]["rows"]["-10.0"][i] for i in range(len(freqs))]
gr0 = [R["hf"]["rows"]["-40.0"][i] - R["hf"]["rows"]["0.0"][i] for i in range(len(freqs))]
print("   GR at -10 dBFS by frequency:", [(int(f), round(v, 2)) for f, v in zip(freqs, gr10)])
print("   GR at   0 dBFS by frequency:", [(int(f), round(v, 2)) for f, v in zip(freqs, gr0)])

# D. no-GR gain vs threshold
R["nogr"] = {}
for lvl in (-70.0, -50.0):
    row = {str(k): gain(lvl, secs=2.0, optical_threshold=k) for k in (1, 4, 8, 12, 16, 20, 22, 24)}
    R["nogr"][str(lvl)] = row
    print(f"D) level {lvl:.0f} dBFS, gain by threshold:", {k: round(v, 3) for k, v in row.items()})

# E. above-knee steps: per-cycle gain trajectories
def step_signal(a, b, secs=(2.0, 2.0, 2.0), f=1000.0):
    t = np.arange(int(round(sum(secs) * FS))) / FS
    env = np.full_like(t, 10 ** (a / 20)); n1 = int(secs[0] * FS); n2 = n1 + int(secs[1] * FS)
    env[n1:n2] = 10 ** (b / 20)
    return env * np.sin(2 * np.pi * f * t), env


def per_cycle_gain(x, y, f=1000.0):
    per = int(round(FS / f)); m = len(x) // per
    xr = np.sqrt(np.mean(x[:m * per].reshape(m, per) ** 2, axis=1)); yr = np.sqrt(np.mean(y[:m * per].reshape(m, per) ** 2, axis=1))
    return (20 * np.log10(yr / xr)).tolist()


R["steps"] = {}
for a, b in ((-20.0, -10.0), (-10.0, 0.0), (-20.0, 5.0)):
    x, _ = step_signal(a, b)
    y = P.run(x, **both(**OPTO, optical_threshold=20))
    gc = per_cycle_gain(x, y[0].astype(np.float64))
    R["steps"][f"{a:.0f}_{b:.0f}"] = {"from": a, "to": b, "per_cycle_gain_db": gc, "cycle_s": 1.0 / 1000.0}
    n1 = 2000; n2 = 4000
    def t_to(frac, seg, g_start, g_end):
        target = g_start + frac * (g_end - g_start)
        for i, v in enumerate(seg):
            if (g_end < g_start and v <= target) or (g_end > g_start and v >= target):
                return i
        return None
    up = gc[n1:n2]; down = gc[n2:]
    g_a = float(np.mean(gc[n1 - 200:n1 - 1])); g_b = float(np.mean(gc[n2 - 200:n2 - 1])); g_a2 = float(np.mean(gc[-200:]))
    print(f"E) step {a:+.0f} -> {b:+.0f} -> {a:+.0f} dBFS: gain {g_a:+.2f} -> {g_b:+.2f} -> {g_a2:+.2f} dB;"
          f" attack reaches 50/90 % at {t_to(0.5, up, g_a, g_b)}/{t_to(0.9, up, g_a, g_b)} ms; release reaches 50/90 % at {t_to(0.5, down, g_b, g_a2)}/{t_to(0.9, down, g_b, g_a2)} ms")
    print("   attack, first 40 ms (dB per 2 ms):", np.round(up[:40:2], 2).tolist())
    print("   release, first 100 ms (dB per 5 ms):", np.round(down[:100:5], 2).tolist())
    print("   release, 0.1..2 s (dB per 100 ms):", np.round(down[100:2000:100], 2).tolist())

out = os.path.join(os.path.dirname(__file__), "..", "data", "discriminate_opto.json")
json.dump(R, open(out, "w"), indent=1)
print("wrote", os.path.relpath(out))
