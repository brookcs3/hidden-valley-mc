# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""The reference's output ceiling: both stages out, each transformer, sines from 20 Hz to 5 kHz at +12 to +30 dBFS; gain of the
fundamental and H2 / H3. Writes fit/data/discriminate_ceiling.json. usage: python3 -u fit/measure/discriminate_ceiling.py"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import pa
from pa import both, sine, FS
P = pa.Ref()
R = {"meta": {"version": P.version, "sha256": P.sha256}, "rows": {}}
LEVELS = [12, 15, 18, 20, 21, 22, 23, 24, 26, 28, 30]
FREQS = [20.0, 60.0, 160.0, 320.0, 1000.0, 5000.0]
def lock(y, f, n0):
    t = np.arange(len(y) - n0) / FS; seg = y[n0:]
    return 2 * np.mean(seg * np.exp(-2j * np.pi * f * t))
for core in ("Nickel", "Iron", "Steel"):
    for f in FREQS:
        row = []
        for L in LEVELS:
            x = sine(L, f=f, secs=1.5); y = P.run(x, **both(transformer=core))[0].astype(np.float64)
            n0 = int(0.5 * FS); c1 = lock(y, f, n0)
            g = 20 * np.log10(abs(c1)) - L
            h2 = 20 * np.log10(abs(lock(y, 2 * f, n0)) / abs(c1) + 1e-30); h3 = 20 * np.log10(abs(lock(y, 3 * f, n0)) / abs(c1) + 1e-30)
            peak = 20 * np.log10(np.max(np.abs(y[n0:])))
            row.append({"level": L, "gain_db": round(g, 3), "h2": round(h2, 1), "h3": round(h3, 1), "peak_dbfs": round(peak, 2)})
        R["rows"][f"{core}_{int(f)}"] = row
        print(core, int(f), [(r["level"], r["gain_db"], r["peak_dbfs"]) for r in row])
json.dump(R, open(os.path.join(os.path.dirname(__file__), "..", "data", "discriminate_ceiling.json"), "w"), indent=1)
