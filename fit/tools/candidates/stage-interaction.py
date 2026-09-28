# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""The interaction between the reference's optical stage (first) and its discrete stage (second): measurements and hypotheses.
Findings in docs/stage-interaction.md. Nothing under src/, fit/stages/ or fit/data/ is touched.

The probe. In STEREO the discrete detectors are linked (summed sidechain) and the optical ones are not. With the left channel loud and the
right at -40 dBFS (below every optical knee), the right channel carries the same discrete gain reduction as the left and no optical gain
reduction, so its level change reads the discrete gain reduction directly, and left minus right reads the optical stage in situ. Every
level is referred to the same settings with the left channel at -60 dBFS, so the make-up laws and the inter-stage gain cancel.

parts: measure (the reference; cached in build/stage-interaction/measure.json), model (our engine on the same grid), hyp (the hypotheses
against the measurements), all
usage: python3 fit/tools/candidates/stage-interaction.py [--parts measure,model,hyp] [--force]"""
import argparse, json, os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "fit")); sys.path.insert(0, os.path.join(ROOT, "fit", "measure"))
import hvmc_core  # noqa: E402
FS = 48000
OUT = os.path.join(ROOT, "build", "stage-interaction"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "measure.json")

def db(a): return 20.0 * np.log10(max(float(a), 1e-20))
def rms(a): return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=np.float64)))))
def tone(dbfs, secs=3.0, f=1000.0, fs=FS):
    t = np.arange(int(round(secs * fs))) / fs
    return 10 ** (dbfs / 20) * np.sin(2 * np.pi * f * t)
def stereo(l, r): return np.stack([np.asarray(l, dtype=np.float32), np.asarray(r, dtype=np.float32)])
def last_db(y, secs=1.0): n = int(secs * FS); return [db(rms(y[0, -n:])), db(rms(y[1, -n:]))]

# settings in the reference's names (pa.both expands to both channels; in STEREO the left controls govern both)
def S(**kw):
    s = dict(mode="Stereo", optical_bypass="Out", discrete_bypass="Out", optical_threshold=1, optical_gain=12, discrete_threshold=1, discrete_gain=12,
             discrete_ratio="4:1", discrete_attack=1.0, discrete_recover="0.5 s", transformer="Nickel", sidechain_filter="Out")
    s.update(kw); return s
def OPTO(thr, **kw): return dict(optical_bypass="In", optical_threshold=thr, **kw)
def DISC(thr, ratio="4:1", **kw): return dict(discrete_bypass="In", discrete_threshold=thr, discrete_ratio=ratio, **kw)

class Ref:
    def __init__(self):
        import pa; self.pa = pa; self.P = pa.Ref(); self.n = 0
    def run(self, x, s):
        self.n += 1
        return self.P.run(x, FS, **self.pa.both(**s)).astype(np.float64)

R_PROBE = -40.0; L_REF = -60.0
LEVELS = [-40, -35, -30, -25, -20, -15, -10, -5, 0, 3, 6]
OPTO_THR = [14, 18, 20, 22, 24]
DISC_SETS = [("Flood", 20), ("4:1", 16), ("2:1", 14), ("Flood", 12), ("4:1", 22)]

def measure(force=False):
    res = json.load(open(CACHE)) if (os.path.exists(CACHE) and not force) else {}
    R = Ref(); t0 = time.time()
    def key(*a): return "|".join(str(v) for v in a)
    def probe(s, lvl, r_lvl=R_PROBE):
        """[L, R] output levels in dBFS with L at lvl and R at r_lvl"""
        k = key("probe", json.dumps(s, sort_keys=True), lvl, r_lvl)
        if k not in res: res[k] = last_db(R.run(stereo(tone(lvl), tone(r_lvl)), s))
        return res[k]
    def rel(s, lvl, r_lvl=R_PROBE):
        """[dL, dR]: level change of each channel from L at L_REF to L at lvl (negative = gain reduction)"""
        a = probe(s, lvl, r_lvl); b = probe(s, L_REF, r_lvl); return [a[0] - b[0], a[1] - b[1]]

    # 0. baselines: no compression anywhere, each stage's gain and the inter-stage gain, against the optical threshold
    print("== 0. no-compression gains at L = -60 dBFS (dB, L channel): none, opto, disc, both, extra = both - opto - disc + none")
    n0 = probe(S(), L_REF)[0]; d0 = probe(S(**DISC(1)), L_REF)[0]
    for thr in [1] + OPTO_THR:
        o = probe(S(**OPTO(thr)), L_REF)[0]; b = probe(S(**OPTO(thr), **DISC(1)), L_REF)[0]
        res[key("extra", thr)] = b - o - d0 + n0
        print(f"   opto thr {thr:2d}: none {n0:+.3f} opto {o:+.3f} disc {d0:+.3f} both {b:+.3f} extra {b - o - d0 + n0:+.3f}")
    # the +2.14 against the discrete ratio and threshold (at no compression) and the filter
    for ratio, dthr in (("Flood", 1), ("2:1", 1), ("1.2:1", 1)):
        b = probe(S(**OPTO(1), **DISC(dthr, ratio)), L_REF)[0]; d = probe(S(**DISC(dthr, ratio)), L_REF)[0]; o = probe(S(**OPTO(1)), L_REF)[0]
        print(f"   {ratio} thr {dthr}: extra {b - o - d + n0:+.3f}")

    # 1. link symmetry: discrete alone in STEREO; does R (at -40) carry L's gain reduction exactly?
    print("\n== 1. discrete alone, STEREO, R at -40 dBFS: gain reduction read on L and on R (dB) against L level; and R at -60 / -20")
    for ratio, dthr in DISC_SETS[:2]:
        s = S(**DISC(dthr, ratio)); rows = []
        for lvl in LEVELS:
            dl, dr = rel(s, lvl); rows.append(f"{lvl:+d}: {-dl:5.2f}/{-dr:5.2f}")
        print(f"   {ratio} thr {dthr}: " + "  ".join(rows))
        for r_lvl in (-60.0, -20.0):
            dl, dr = rel(s, 0, r_lvl); print(f"      L 0 dBFS with R at {r_lvl:+.0f}: GR L {-dl:5.2f} R {-dr:5.2f}")

    # 2. the optical stage alone (its GR against level per threshold), and the discrete stage in situ with the optical at threshold 1
    print("\n== 2. optical alone: GR on L (dB) against level per threshold; R (at -40) must be untouched")
    for thr in OPTO_THR:
        s = S(**OPTO(thr)); rows = []
        for lvl in LEVELS:
            dl, dr = rel(s, lvl); rows.append(f"{lvl:+d}: {-dl:5.2f}({-dr:+.2f})")
        print(f"   thr {thr}: " + "  ".join(rows))
    print("\n== 2b. discrete in situ (optical IN at threshold 1, so the +2.14 is present): GR read on R against L level, and L - R")
    for ratio, dthr in DISC_SETS:
        s = S(**OPTO(1), **DISC(dthr, ratio)); rows = []
        for lvl in LEVELS:
            dl, dr = rel(s, lvl); rows.append(f"{lvl:+d}: {-dr:5.2f}({dl - dr:+.2f})")
        print(f"   {ratio} thr {dthr}: " + "  ".join(rows))

    # 3. the main grid: both stages, R reads the discrete GR, L - R reads the optical stage in situ
    print("\n== 3. both stages: per level, GR_d from R | optical in situ from L - R (dB, positive = gain reduction)")
    for thr in OPTO_THR:
        for ratio, dthr in DISC_SETS:
            s = S(**OPTO(thr), **DISC(dthr, ratio)); rows = []
            for lvl in LEVELS:
                dl, dr = rel(s, lvl); rows.append(f"{lvl:+d}: {-dr:5.2f}|{-(dl - dr):5.2f}")
            print(f"   opto {thr:2d} -> {ratio:5s} thr {dthr:2d}: " + "  ".join(rows))

    # 4. make-up axes at opto 22 -> FLOOD 20 and 4:1 16, L at 0 and -10 dBFS
    print("\n== 4. make-up: opto 22 -> FLOOD 20 / 4:1 16; GR_d from R | optical in situ from L - R, against DISCRETE GAIN then OPTICAL GAIN")
    for ratio, dthr in DISC_SETS[:2]:
        for lvl in (0, -10):
            rows = []
            for dg in (1, 4, 7, 12, 18, 24):
                dl, dr = rel(S(**OPTO(22), **DISC(dthr, ratio), discrete_gain=dg), lvl); rows.append(f"dg{dg:2d}: {-dr:5.2f}|{-(dl - dr):5.2f}")
            print(f"   {ratio} thr {dthr} L {lvl:+d}: " + "  ".join(rows))
            rows = []
            for og in (1, 6, 12, 18, 24):
                dl, dr = rel(S(**OPTO(22), **DISC(dthr, ratio), optical_gain=og), lvl); rows.append(f"og{og:2d}: {-dr:5.2f}|{-(dl - dr):5.2f}")
            print(f"   {ratio} thr {dthr} L {lvl:+d}: " + "  ".join(rows))
            rows = []
            for og in (1, 12, 24):
                dl, dr = rel(S(**OPTO(22), discrete_bypass="Out", optical_gain=og), lvl); rows.append(f"og{og:2d}: {-(dl - dr):5.2f}")
            print(f"   (optical alone, thr 22, L {lvl:+d}: " + "  ".join(rows) + ")")

    # 5. time signature: L steps -50 -> 0 dBFS at 1 s, R steady at -40; per-period gain of L and R (referred to the settled L = -60 render)
    print("\n== 5. step -50 -> 0 dBFS on L at t = 1 s, R at -40: per 1 kHz period, GR_d (from R) and optical in situ (L - R); opto 22")
    t = np.arange(int(2.0 * FS)) / FS
    xl = np.where(t < 1.0, 10 ** (-50 / 20), 1.0) * np.sin(2 * np.pi * 1000 * t); xr = 10 ** (R_PROBE / 20) * np.sin(2 * np.pi * 1000 * t)
    x = stereo(xl, xr); n = FS // 1000; i0 = FS
    for ratio, dthr in DISC_SETS[:2]:
        for att in (0.1, 1.0, 30.0):
            s = S(**OPTO(22), **DISC(dthr, ratio), discrete_attack=att); k = key("step", json.dumps(s, sort_keys=True))
            if k not in res:
                y = R.run(x, s); y0 = R.run(stereo(10 ** (-60 / 20) * np.sin(2 * np.pi * 1000 * t), xr), s)
                so = S(**OPTO(22)); yo = R.run(x, so); yo0 = R.run(stereo(10 ** (-60 / 20) * np.sin(2 * np.pi * 1000 * t), xr), so)
                per = lambda a, a0, c, ref_gain: [db(rms(a[c, i0 + j * n: i0 + (j + 1) * n])) - db(rms(a0[c, -FS:])) - ref_gain for j in range(500)]
                res[k] = {"L": per(y, y0, 0, 50.0), "R": per(y, y0, 1, 0.0), "Lo": per(yo, yo0, 0, 50.0)}
            r = res[k]; ks = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 400]
            print(f"   {ratio} thr {dthr} att {att:4.1f}: " + " ".join(f"p{j}: {-r['R'][j]:.1f}|{-(r['L'][j] - r['R'][j]):.1f}|{-r['Lo'][j]:.1f}" for j in ks))
    print("      (columns per period: GR_d | optical in situ | optical alone)")

    # 6. the one-instance L waveform against the series of parts (reference optical alone -> +2.14 dB -> reference discrete alone)
    print("\n== 6. waveform: one instance L against the series of parts with the +2.14 dB put in; gain difference, null after gain match, H2/H3")
    xs = stereo(tone(0.0), tone(R_PROBE))
    for thr in (22,):
        for ratio, dthr in DISC_SETS[:3]:
            for lvl in (0, -10):
                k = key("wave", thr, ratio, dthr, lvl)
                if k not in res:
                    xs = stereo(tone(lvl), tone(R_PROBE))
                    yb = R.run(xs, S(**OPTO(thr), **DISC(dthr, ratio)))
                    yo = R.run(xs, S(**OPTO(thr))); ex = 10 ** (res[key("extra", thr)] / 20)
                    ys = R.run((yo * ex).astype(np.float32), S(**DISC(dthr, ratio)))
                    n1 = -FS; a = yb[0, n1:]; b = ys[0, n1:]; g = db(rms(a)) - db(rms(b)); bn = b * 10 ** (g / 20)
                    import pa
                    ha, _ = pa.harmonics(a, 1000.0); hb, _ = pa.harmonics(b, 1000.0)
                    res[k] = {"gain_diff": g, "null": db(rms(a - bn)) - db(rms(a)), "h_one": ha[:3], "h_series": hb[:3], "R_diff": db(rms(yb[1, n1:])) - db(rms(ys[1, n1:]))}
                r = res[k]
                print(f"   opto {thr} -> {ratio} thr {dthr} at {lvl:+d}: L one - series {r['gain_diff']:+.2f} dB (R {r['R_diff']:+.2f}); null after gain match {r['null']:.1f} dB; H2/H3/H4 one {r['h_one'][0]:.1f}/{r['h_one'][1]:.1f}/{r['h_one'][2]:.1f} series {r['h_series'][0]:.1f}/{r['h_series'][1]:.1f}/{r['h_series'][2]:.1f}")
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s; cached in {CACHE}")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", default="measure"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if "measure" in a.parts or "all" in a.parts: measure(a.force)
