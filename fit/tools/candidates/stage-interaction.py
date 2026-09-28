# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""The interaction between the reference's optical stage (first) and its discrete stage (second): measurements and hypotheses.
Findings in docs/stage-interaction.md: the reference adds (1/200) (x - v), x the input and v the optical divider output, after the
transformer; there is no back-off of the optical loop. Nothing under src/, fit/stages/ or fit/data/ is touched.

The probe. In STEREO the discrete detectors are linked (summed sidechain) and the optical ones are not. With the left channel loud and the
right at -40 dBFS (below every optical knee), the right channel carries the same discrete gain reduction as the left and no optical gain
reduction, so its level change reads the discrete gain reduction directly, and left minus right reads the optical stage in situ. Every
level is referred to the same settings with the left channel at -60 dBFS, so the make-up laws and the inter-stage gain cancel.

parts: measure, measure2 .. measure6 (the reference, cached in build/stage-interaction/measure.json; every render is skipped if cached),
analyse (the cell table, the discrete-detector check, the bleed law fit, the step, the hypotheses on our engine's laws, our engine with the
law added; writes build/stage-interaction/tables.md and fit.json), all
usage: python3 fit/tools/candidates/stage-interaction.py [--parts measure,measure2,measure3,measure4,measure5,measure6,analyse] [--force]"""
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

def measure2(force=False):
    """second pass: baselines for the make-up rows, the cross-drive grid (R at 20 kHz drives the linked discrete detector while L's
    optical loop only sees L), a probe tone on L, and the optical stage's H3 against its gain reduction"""
    res = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    R = Ref(); t0 = time.time(); import pa
    def key(*a): return "|".join(str(v) for v in a)
    def probe(s, lvl, r_lvl=R_PROBE):
        k = key("probe", json.dumps(s, sort_keys=True), lvl, r_lvl)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(lvl), tone(r_lvl)), s))
        return res[k]
    def rel(s, lvl, r_lvl=R_PROBE):
        a = probe(s, lvl, r_lvl); b = probe(s, L_REF, r_lvl); return [a[0] - b[0], a[1] - b[1]]

    print("== 7. make-up axes with their own -60 dBFS baselines: GR_d from R | optical in situ | L output dBFS rms | excess = alone - in situ")
    for ratio, dthr, lvls in (("Flood", 20, (-10, 0, 6)), ("4:1", 22, (0, 6)), ("4:1", 16, (6,))):
        for lvl in lvls:
            alone = (lvl - L_REF) - rel(S(**OPTO(22)), lvl)[0]
            rows = []
            for dg in (1, 4, 7, 12, 18, 24):
                s = S(**OPTO(22), **DISC(dthr, ratio), discrete_gain=dg); dl, dr = rel(s, lvl); gro = (lvl - L_REF) - (dl - dr)
                rows.append(f"dg{dg:2d}: {-dr:5.2f}|{gro:5.2f}|{probe(s, lvl)[0]:6.1f}|{alone - gro:5.2f}")
            print(f"   {ratio} thr {dthr} L {lvl:+d} (alone {alone:.2f}): " + "  ".join(rows))
            rows = []
            for og in (1, 6, 12, 18, 24):
                s = S(**OPTO(22), **DISC(dthr, ratio), optical_gain=og); dl, dr = rel(s, lvl); gro = (lvl - L_REF) - (dl - dr)
                al = (lvl - L_REF) - rel(S(**OPTO(22), optical_gain=og), lvl)[0]
                rows.append(f"og{og:2d}: {-dr:5.2f}|{gro:5.2f}|{probe(s, lvl)[0]:6.1f}|{al - gro:5.2f}")
            print(f"   {ratio} thr {dthr} L {lvl:+d}: " + "  ".join(rows))

    print("\n== 8. cross-drive: calibration. discrete alone, R alone at 20 kHz against 1 kHz: GR on R; optical alone at thr 22, R at 20 kHz: GR on R")
    def probe2(s, la, fa, lb, fb):
        k = key("probe2", json.dumps(s, sort_keys=True), la, fa, lb, fb)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(la, f=fa), tone(lb, f=fb)), s))
        return res[k]
    for ratio, dthr in (("Flood", 20), ("4:1", 16)):
        s = S(**DISC(dthr, ratio)); rows = []
        for lb in (-30, -20, -10, 0, 6):
            for fb in (1000.0, 20000.0):
                g = probe2(s, -80, 1000.0, lb, fb)[1] - probe2(s, -80, 1000.0, -60, fb)[1]; rows.append(f"{lb:+d}@{fb/1000:.0f}k: {(lb + 60) - g:5.2f}")
        print(f"   disc {ratio} thr {dthr}: " + "  ".join(rows))
    s = S(**OPTO(22)); rows = []
    for lb in (-20, -10, 0, 6):
        for fb in (1000.0, 20000.0):
            g = probe2(s, -80, 1000.0, lb, fb)[1] - probe2(s, -80, 1000.0, -60, fb)[1]; rows.append(f"{lb:+d}@{fb/1000:.0f}k: {(lb + 60) - g:5.2f}")
    print(f"   opto thr 22 on R: " + "  ".join(rows))
    print("\n== 8b. cross-drive grid, opto 22: L at 1 kHz level A, R at 20 kHz level B; cells: GR_d (from R) | L total GR | excess = GR_o,alone(A) + GR_d - total")
    for ratio, dthr in (("Flood", 20), ("4:1", 16)):
        s = S(**OPTO(22), **DISC(dthr, ratio))
        for la in (-30, -20, -10, 0, 6):
            alone = (la + 60) - (probe2(S(**OPTO(22)), la, 1000.0, -80, 20000.0)[0] - probe2(S(**OPTO(22)), -60, 1000.0, -80, 20000.0)[0])
            rows = []
            for lb in (-60, -20, -10, 0, 6):
                y = probe2(s, la, 1000.0, lb, 20000.0); y0 = probe2(s, -60, 1000.0, -80, 20000.0); yb0 = probe2(s, -60, 1000.0, lb, 20000.0)
                grd = (lb + 60) - (y[1] - probe2(s, -60, 1000.0, -60, 20000.0)[1]) if lb > -60 else 0.0
                # discrete GR from R needs R's own no-GR level: R at -60 with L at -60 (no compression anywhere)
                tot = (la + 60) - (y[0] - y0[0])
                rows.append(f"B{lb:+d}: {grd:5.2f}|{tot:5.2f}|{alone + grd - tot:5.2f}")
            print(f"   {ratio} thr {dthr} A {la:+d} (alone {alone:5.2f}): " + "  ".join(rows))

    print("\n== 9. probe tone: L = 1 kHz at level A plus 7 kHz at -40 dBFS; gain of each component (lock-in) in situ against the series parts; opto 22 -> FLOOD 20")
    def lockin(y, f, n0):
        t = np.arange(len(y) - n0) / FS; seg = y[n0:]
        return 20 * np.log10(np.abs(np.mean(seg * np.exp(-2j * np.pi * f * t))) * 2 + 1e-20)
    for la in (-10, 0, 6):
        k = key("ptone", la)
        if k not in res or force:
            xl = tone(la) + tone(-40, f=7000.0); x = stereo(xl, tone(R_PROBE)); n0 = 2 * FS
            yb = R.run(x, S(**OPTO(22), **DISC(20, "Flood"))); yo = R.run(x, S(**OPTO(22)))
            ex = 10 ** (res[key("extra", 22)] / 20); ys = R.run((yo * ex).astype(np.float32), S(**DISC(20, "Flood")))
            res[k] = {f"{lab}_{f}": lockin(yy[0], f, n0) - lockin(x[0].astype(np.float64), f, n0) for lab, yy in (("one", yb), ("series", ys), ("opto", yo)) for f in (1000.0, 7000.0)}
        r = res[k]
        print(f"   A {la:+d}: one instance 1k {r['one_1000.0']:+.2f} 7k {r['one_7000.0']:+.2f} | series 1k {r['series_1000.0']:+.2f} 7k {r['series_7000.0']:+.2f} | excess 1k {r['one_1000.0'] - r['series_1000.0']:+.2f} 7k {r['one_7000.0'] - r['series_7000.0']:+.2f} | opto alone 1k {r['opto_1000.0']:+.2f} 7k {r['opto_7000.0']:+.2f}")

    print("\n== 10. the optical stage's H3 against its gain reduction (alone, thr 22 and 14, L level); and in situ H3 at the grid cells of 6")
    for thr in (22, 14):
        rows = []
        for lvl in (-20, -15, -10, -5, 0, 3, 6):
            k = key("h3alone", thr, lvl)
            if k not in res or force:
                y = R.run(stereo(tone(lvl), tone(R_PROBE)), S(**OPTO(thr))); h, _ = pa.harmonics(y[0, -FS:], 1000.0)
                gr = (lvl - L_REF) - rel(S(**OPTO(thr)), lvl)[0]; res[k] = {"gr": gr, "h2": h[0], "h3": h[1], "h5": h[3]}
            r = res[k]; rows.append(f"{lvl:+d}: GR {r['gr']:5.2f} H3 {r['h3']:6.1f}")
        print(f"   thr {thr}: " + "  ".join(rows))
    rows = []
    for ratio, dthr, lvl in (("Flood", 20, 0), ("Flood", 20, 6), ("4:1", 22, 0), ("4:1", 22, 6), ("4:1", 16, 6), ("Flood", 20, -10)):
        k = key("h3insitu", ratio, dthr, lvl)
        if k not in res or force:
            s = S(**OPTO(22), **DISC(dthr, ratio)); y = R.run(stereo(tone(lvl), tone(R_PROBE)), s); h, _ = pa.harmonics(y[0, -FS:], 1000.0)
            dl, dr = rel(s, lvl); res[k] = {"gro": (lvl - L_REF) - (dl - dr), "grd": -dr, "h3": h[1], "h2": h[0]}
            yd = R.run((R.run(stereo(tone(lvl), tone(R_PROBE)), S(**OPTO(22))) * 10 ** (res[key("extra", 22)] / 20)).astype(np.float32), S(**DISC(dthr, ratio)))
            hs, _ = pa.harmonics(yd[0, -FS:], 1000.0); res[k]["h3_series"] = hs[1]
        r = res[k]; rows.append(f"{ratio} {dthr} L{lvl:+d}: GR_o in situ {r['gro']:5.2f} H3 {r['h3']:6.1f} (series H3 {r['h3_series']:6.1f})")
    print("   in situ: " + "\n            ".join(rows))
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s")
    return res

def measure3(force=False):
    """third pass: the no-compression baselines with R at -60 dBFS for every cached setting (at discrete thresholds 20 and 22 the probe at
    -40 dBFS already puts the linked detector 2 to 3 dB into gain reduction), and the phase of the extra signal (complex lock-in of
    probe tones at 100 Hz, 3, 7 and 15 kHz riding on the 1 kHz tone, one instance against the series of parts)"""
    res = json.load(open(CACHE)); R = Ref(); t0 = time.time()
    def key(*a): return "|".join(str(v) for v in a)
    settings = set()
    for k in list(res):
        if k.startswith("probe|"):
            settings.add(k.split("|")[1])
    for sj in sorted(settings):
        k = key("probe", sj, -60.0, -60.0)
        if k not in res or force:
            s = json.loads(sj); res[k] = last_db(R.run(stereo(tone(-60.0), tone(-60.0)), s))
    print(f"baselines: {len(settings)} settings")
    print("\n== 11. phase of the extra signal: probes at -40 dBFS on L with the 1 kHz tone at 0 dBFS; complex gain one instance / series, opto 22 -> FLOOD 20 and 4:1 22")
    probes = (100.0, 3000.0, 7000.0, 15000.0)
    for ratio, dthr in (("Flood", 20), ("4:1", 22)):
        for la in (0, 6):
            k = key("phase", ratio, dthr, la)
            if k not in res or force:
                xl = tone(la) + sum(tone(-40, f=f) for f in probes); x = stereo(xl, tone(R_PROBE)); n0 = 2 * FS
                yb = R.run(x, S(**OPTO(22), **DISC(dthr, ratio))); yo = R.run(x, S(**OPTO(22)))
                ex = 10 ** (res[key("extra", 22)] / 20); ys = R.run((yo * ex).astype(np.float32), S(**DISC(dthr, ratio)))
                def li(y, f):
                    t = np.arange(len(y) - n0) / FS; return complex(np.mean(y[n0:] * np.exp(-2j * np.pi * f * t)))
                out = {}
                for f in (1000.0,) + probes:
                    a = li(yb[0], f); b = li(ys[0], f); c = li(x[0].astype(np.float64), f)
                    out[str(f)] = {"ratio_db": 20 * np.log10(abs(a / b)), "ratio_deg": float(np.degrees(np.angle(a / b))),
                                   "extra_db": 20 * np.log10(abs((a - b) / c)), "extra_deg": float(np.degrees(np.angle((a - b) / c)))}
                res[k] = out
            r = res[k]
            print(f"   {ratio} thr {dthr} L {la:+d}: " + "  ".join(f"{float(f)/1000:g}k: one/series {v['ratio_db']:+.2f} dB {v['ratio_deg']:+.1f} deg; extra/input {v['extra_db']:.1f} dB {v['extra_deg']:+.0f} deg" for f, v in r.items()))
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s")
    return res

def measure4(force=False):
    """decisive cells: does the bleed need discrete gain reduction? (discrete at threshold 1 with both make-ups at 1, so the main
    path is 24 dB down); does it exist with the discrete stage out? (optical alone at make-up 1 against 12); and the discrete GR
    dependence at fixed optical GR (thresholds 12 to 24 at 4:1 and FLOOD with L at 0 and +6)"""
    res = json.load(open(CACHE)); R = Ref(); t0 = time.time()
    def key(*a): return "|".join(str(v) for v in a)
    def probe(s, lvl, r_lvl=R_PROBE):
        k = key("probe", json.dumps(s, sort_keys=True), lvl, r_lvl)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(lvl), tone(r_lvl)), s))
        return res[k]
    def base(s):
        k = key("probe", json.dumps(s, sort_keys=True), -60.0, -60.0)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(-60.0), tone(-60.0)), s))
        return res[k]
    print("== 12. both stages in, DISCRETE at threshold 1 (no discrete GR): total GR on L against the optical alone at the same make-up; excess")
    for thr in (22, 24):
        for og, dg in ((12, 12), (1, 12), (12, 1), (1, 1)):
            rows = []
            for lvl in (-10, 0, 6):
                s = S(**OPTO(thr), **DISC(1), optical_gain=og, discrete_gain=dg); b = base(s); y = probe(s, lvl)
                grL = (lvl + 60) - (y[0] - b[0]); grd = 20.0 - (y[1] - b[1])
                sa = S(**OPTO(thr), optical_gain=og); ba = base(sa); ya = probe(sa, lvl); gra = (lvl + 60) - (ya[0] - ba[0])
                rows.append(f"L{lvl:+d}: GR_L {grL:5.2f} (GR_d {grd:4.2f}) alone {gra:5.2f} excess {gra + grd - grL:5.2f}")
            print(f"   opto {thr} og{og:2d} dg{dg:2d}: " + "  ".join(rows))
    print("\n== 12b. optical alone: GR against make-up position at thresholds 22 and 24 (a bleed that bypasses the make-up shows as less GR at position 1)")
    for thr in (22, 24):
        for lvl in (-10, 0, 6):
            rows = []
            for og in (1, 3, 6, 12, 18, 24):
                sa = S(**OPTO(thr), optical_gain=og); ba = base(sa); ya = probe(sa, lvl); rows.append(f"og{og:2d}: {(lvl + 60) - (ya[0] - ba[0]):5.2f}")
            print(f"   thr {thr} L{lvl:+d}: " + "  ".join(rows))
    print("\n== 12c. discrete threshold sweep at opto 22 (og 12, dg 12): GR_d | excess, L at 0 and +6")
    for ratio in ("4:1", "Flood", "2:1"):
        for lvl in (0, 6):
            rows = []
            for dthr in (8, 12, 14, 16, 18, 20, 22, 24):
                s = S(**OPTO(22), **DISC(dthr, ratio)); b = base(s); y = probe(s, lvl)
                grL = (lvl + 60) - (y[0] - b[0]); grd = 20.0 - (y[1] - b[1])
                sa = S(**OPTO(22)); ba = base(sa); ya = probe(sa, lvl); gra = (lvl + 60) - (ya[0] - ba[0])
                rows.append(f"t{dthr:2d}: {grd:5.2f}|{gra + grd - grL:5.2f}")
            print(f"   {ratio:5s} L{lvl:+d}: " + "  ".join(rows))
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s")

def measure5(force=False):
    """the discrete stage alone at its curve plateau: is the gain reduction the same on L (+6 dBFS) and R (-40 dBFS)? (the R probe
    assumes it); and the optical stage alone at position 24 against 12 (the loop's own make-up dependence, for the fit)"""
    res = json.load(open(CACHE)); R = Ref(); t0 = time.time()
    def key(*a): return "|".join(str(v) for v in a)
    def probe(s, lvl, r_lvl=R_PROBE):
        k = key("probe", json.dumps(s, sort_keys=True), lvl, r_lvl)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(lvl), tone(r_lvl)), s))
        return res[k]
    def base(s):
        k = key("probe", json.dumps(s, sort_keys=True), -60.0, -60.0)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(-60.0), tone(-60.0)), s))
        return res[k]
    print("== 13. discrete alone (optical out) and with the optical in at threshold 1, STEREO: GR on L against GR on R, L at 0 and +6 dBFS")
    for opt in (False, True):
        for ratio, dthr in (("4:1", 24), ("2:1", 24), ("Flood", 24), ("4:1", 22), ("Flood", 20), ("4:1", 16)):
            s = S(**DISC(dthr, ratio), **(OPTO(1) if opt else {})); b = base(s); rows = []
            for lvl in (-10, 0, 6):
                y = probe(s, lvl); rows.append(f"L{lvl:+d}: L {(lvl + 60) - (y[0] - b[0]):5.2f} R {20.0 - (y[1] - b[1]):5.2f}")
            print(f"   {'opto thr1 + ' if opt else 'alone       '}{ratio:5s} thr {dthr}: " + "  ".join(rows))
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s")

def measure6(force=False):
    """where the bleed enters: at a bleed-dominated cell (opto 22 og12 -> FLOOD 20 dg1, L +6: the output is 15 dB above the series)
    the output follows the core's gain if the bleed is added before the transformer and does not if after; the filter and the mode"""
    res = json.load(open(CACHE)); R = Ref(); t0 = time.time()
    def key(*a): return "|".join(str(v) for v in a)
    def probe(s, lvl, r_lvl=R_PROBE):
        k = key("probe", json.dumps(s, sort_keys=True), lvl, r_lvl)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(lvl), tone(r_lvl)), s))
        return res[k]
    def base(s):
        k = key("probe", json.dumps(s, sort_keys=True), -60.0, -60.0)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(-60.0), tone(-60.0)), s))
        return res[k]
    print("== 14. opto 22 -> FLOOD 20 with discrete gain 1, L +6 (bleed-dominated) and discrete gain 24 (main-path-dominated): L output level per transformer, mode, filter")
    for dg in (1, 24):
        for extra in ({}, {"transformer": "Iron"}, {"transformer": "Steel"}, {"mode": "Dual Mono"}, {"sidechain_filter": "In"}):
            s = S(**OPTO(22), **DISC(20, "Flood"), discrete_gain=dg, **extra); b = base(s); y = probe(s, 6)
            print(f"   dg{dg:2d} {str(extra) if extra else 'Nickel/Stereo/filter out':32s}: no-GR gain {b[0] + 63.0103:+.3f} dB, L out {y[0]:.3f} dBFS rms, GR_d {20 - (y[1] - b[1]):.2f}, total GR on L {66 - (y[0] - b[0]):.2f}")
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s")

def measure7(force=False):
    """the inter-stage gain against the make-up positions (no compression, L and R at -60): both minus opto minus disc plus none"""
    res = json.load(open(CACHE)); R = Ref(); t0 = time.time()
    def key(*a): return "|".join(str(v) for v in a)
    def base(s):
        k = key("probe", json.dumps(s, sort_keys=True), -60.0, -60.0)
        if k not in res or force: res[k] = last_db(R.run(stereo(tone(-60.0), tone(-60.0)), s))
        return res[k]
    n0 = base(S())[0]
    print("== 15. the inter-stage gain (dB) at no compression against OPTICAL GAIN (rows) and DISCRETE GAIN (columns), optical threshold 22, discrete threshold 1")
    for og in (1, 6, 12, 18, 24):
        o = base(S(**OPTO(22), optical_gain=og))[0]; rows = []
        for dg in (1, 12, 24):
            d = base(S(**DISC(1), discrete_gain=dg))[0]; b = base(S(**OPTO(22), **DISC(1), optical_gain=og, discrete_gain=dg))[0]; rows.append(f"dg{dg:2d}: {b - o - d + n0:+.3f}")
        print(f"   og{og:2d} (opto alone gain {o + 63.0103:+.2f}): " + "  ".join(rows))
    json.dump(res, open(CACHE, "w"), indent=0)
    print(f"\n{R.n} reference renders, {time.time() - t0:.0f} s")

# ------------------------------------------------------------------------------------------------ analysis
def cells(res):
    """every both-stages probe in the cache as a cell with corrected baselines. Levels are rms dB; gains are rms differences.
    G0: the chain's no-compression gain at the setting (L and R at -60). GR_d: from R (R at -40 against R at -60 with L at -60,
    minus the 20 dB step). GR_L: the left channel's total gain reduction in situ. GR_o_alone: the optical stage alone at the same
    threshold, make-up and level. excess = GR_o_alone + GR_d - GR_L. k_db: the extra signal as a fraction of the input,
    20 log(out_lin - main_lin), where out and main are the in-situ and series outputs relative to the input."""
    out = []
    for k, v in res.items():
        if not k.startswith("probe|"): continue
        _, sj, lvl, rl = k.split("|"); lvl = float(lvl); rl = float(rl); s = json.loads(sj)
        if s.get("optical_bypass") != "In" or s.get("discrete_bypass") != "In" or rl != R_PROBE or lvl <= L_REF or s.get("mode") != "Stereo": continue
        if int(s.get("optical_threshold", 1)) == 1: continue
        b = res.get("|".join(["probe", sj, "-60.0", "-60.0"]))
        if b is None: continue
        so = dict(s); so["discrete_bypass"] = "Out"; so.pop("discrete_threshold", None); so.pop("discrete_ratio", None)
        so = S(**{kk: vv for kk, vv in so.items() if kk in ("optical_threshold", "optical_gain")}, optical_bypass="In")
        def get(sd, l, r):
            for ls in (str(l), str(int(l)) if float(l).is_integer() else str(l)):
                for rs in (str(r), str(int(r)) if float(r).is_integer() else str(r)):
                    x = res.get("|".join(["probe", json.dumps(sd, sort_keys=True), ls, rs]))
                    if x is not None: return x
        ao = get(so, lvl, R_PROBE); a0 = get(so, L_REF, R_PROBE)
        if ao is None or a0 is None: continue
        G0 = b[0] + 60.0 + 3.0103          # the chain's no-compression gain (rms of a sine sits 3.01 dB under its peak)
        gr_d = 20.0 - (v[1] - b[1]); gr_L = (lvl + 60.0) - (v[0] - b[0]); gr_oa = (lvl + 60.0) - (ao[0] - a0[0])
        excess = gr_oa + gr_d - gr_L
        out_lin = 10 ** ((G0 - gr_L) / 20); main_lin = 10 ** ((G0 - gr_oa - gr_d) / 20); extra = out_lin - main_lin
        out.append(dict(thr=int(s["optical_threshold"]), og=int(s["optical_gain"]), dthr=int(s["discrete_threshold"]), ratio=s["discrete_ratio"], dg=int(s["discrete_gain"]),
                        lvl=lvl, G0=G0, gr_d=gr_d, gr_L=gr_L, gr_oa=gr_oa, gr_oi=gr_L - gr_d, excess=excess,
                        k_db=20 * np.log10(abs(extra) + 1e-12) * (1 if extra > 0 else np.nan), out_db=G0 - gr_L, main_db=G0 - gr_oa - gr_d))
    return out

def lin(db): return 10.0 ** (np.asarray(db, dtype=float) / 20.0)
def dbl(x): return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-20))

class Laws:
    """the reference's stage-alone laws from the cache (or our engine's, rendered): the optical stage's gain reduction against input
    level per (threshold, make-up), and the discrete stage's gain reduction against its detector level (both stages in, optical at
    threshold 1, so the inter-stage gain is inside) per (ratio, threshold, make-up)."""
    def __init__(self, res, ours=None):
        self.res = res; self.ours = ours; self.cache = {}
    def probe(self, s, lvl, r_lvl):
        if self.ours is None:
            for ls in (str(float(lvl)), str(int(lvl)) if float(lvl).is_integer() else str(lvl)):
                for rs in (str(float(r_lvl)), str(int(r_lvl)) if float(r_lvl).is_integer() else str(r_lvl)):
                    x = self.res.get("|".join(["probe", json.dumps(s, sort_keys=True), ls, rs]))
                    if x is not None: return x
            return None
        return self.ours(s, lvl, r_lvl)
    def opto(self, thr, og, lvls):
        """[GR_o alone at each level] (as measured, i.e. including any bleed)"""
        s = S(**OPTO(thr), optical_gain=og); b = self.probe(s, -60.0, -60.0) or self.probe(s, -60.0, R_PROBE)
        return np.array([(l + 60.0) - (self.probe(s, l, R_PROBE)[0] - b[0]) for l in lvls])
    def disc(self, ratio, dthr, dg=12):
        """(detector level dBFS peak, GR_d) table with the optical in at threshold 1: the level the discrete detector sees is
        L + G_om(1, og 12) + d_inter (the R probe's share added: 0.5 (L + R) in phase)"""
        k = ("disc", ratio, dthr, dg)
        if k not in self.cache:
            s = S(**OPTO(1), **DISC(dthr, ratio), discrete_gain=dg); b = self.probe(s, -60.0, -60.0)
            lv, gr = [], []
            for l in LEVELS + [-60.0, -50.0, -45.0]:
                y = self.probe(s, l, R_PROBE)
                if y is None: continue
                lv.append(dbl(0.5 * (lin(l) + lin(R_PROBE)))); gr.append(20.0 - (y[1] - b[1]))
            o = np.argsort(lv); self.cache[k] = (np.array(lv)[o], np.array(gr)[o])
        return self.cache[k]
    def gr_d(self, ratio, dthr, det_level, dg=12):
        lv, gr = self.disc(ratio, dthr, dg)
        if len(lv) < 2 or det_level < lv[0] - 0.5 or det_level > lv[-1] + 0.5: return float("nan")
        return float(np.interp(det_level, lv, gr))
    def G_om(self, thr, og):
        s = S(**OPTO(thr), optical_gain=og); b = self.probe(s, -60.0, -60.0) or self.probe(s, -60.0, R_PROBE); return b[0] + 63.0103

def bleed_k(c, kappa_db, p, hpar, hform):
    """the extra signal as a fraction of the input for a cell: kappa (1 - g_o)^p h(GR_d)"""
    go = lin(-c["gr_ot"]); h = hfun(c["gr_d"], hpar, hform)
    return lin(kappa_db) * (1.0 - go) ** p * h

def hfun(grd, hp, form):
    np.seterr(all="ignore"); grd = np.asarray(grd, dtype=float); gd = lin(-grd)
    if form == "ramp":   return lin(np.minimum(0.0, hp[0] * (grd - hp[1])))          # h_dB = min(0, a (GR_d - D))
    if form == "pow":    return hp[0] + (1.0 - hp[0]) * (1.0 - gd) ** hp[1]          # h0 + (1 - h0)(1 - g_d)^q
    if form == "hill":   return 1.0 - (1.0 - hp[0]) / (1.0 + (grd / hp[1]) ** hp[2])  # h0 -> 1 around D
    if form == "const":  return np.ones_like(grd)
    raise ValueError(form)

def true_opto(c, kappa0):
    """the optical stage's true gain reduction from its alone measurement, which itself carries the bleed at h(0): solve
    alone = -20 log(g G + kappa0 (1 - g)) + 20 log(G) for g (G the make-up and path gain)"""
    G = lin(c["G_om"]); target = lin(-c["gr_oa"]) * G
    g = lin(-c["gr_oa"])
    for _ in range(30): g = (target - kappa0 * (1.0 - g)) / G
    return -dbl(max(g, 1e-9))

def fit_bleed(C, form, fit_p=False, cut=0.3):
    from scipy.optimize import least_squares
    F = [c for c in C if c["excess"] >= cut and c["og"] >= 6]
    init = {"ramp": [0.2, 30.0], "pow": [0.5, 4.0], "hill": [0.5, 15.0, 2.0], "const": []}[form]
    x0 = [-47.0] + ([1.0] if fit_p else []) + init
    def unpack(x):
        kappa = x[0]; p = x[1] if fit_p else 1.0; hp = x[2:] if fit_p else x[1:]; return kappa, p, hp
    def resid(x):
        kappa, p, hp = unpack(x); r = []
        h0 = float(hfun(0.0, hp, form)); k0 = lin(kappa) * h0
        for c in F:
            c["gr_ot"] = true_opto(c, k0)
            out = lin(c["G0"] - c["gr_ot"] - c["gr_d"]) + bleed_k(c, kappa, p, hp, form)
            r.append(dbl(out) - c["out_db"])
        return np.array(r)
    r = least_squares(resid, x0, loss="soft_l1", f_scale=0.3)
    kappa, p, hp = unpack(r.x); h0 = float(hfun(0.0, hp, form)); k0 = lin(kappa) * h0
    for c in C:
        c["gr_ot"] = true_opto(c, k0); pred = dbl(lin(c["G0"] - c["gr_ot"] - c["gr_d"]) + bleed_k(c, kappa, p, hp, form))
        c["pred_excess"] = pred - (c["G0"] - c["gr_oa"] - c["gr_d"]); c["res"] = pred - c["out_db"]
    fitted = np.array([c["res"] for c in F]); allr = np.array([c["res"] for c in C])
    return dict(form=form, kappa_db=kappa, p=p, hp=list(hp), h0=h0, rms_fit=float(np.sqrt(np.mean(fitted ** 2))), max_fit=float(np.abs(fitted).max()),
                rms_all=float(np.sqrt(np.mean(allr ** 2))), max_all=float(np.abs(allr).max()), n=len(F))

def analyse():
    res = json.load(open(CACHE)); C = cells(res); L = Laws(res)
    for c in C: c["G_om"] = L.G_om(c["thr"], c["og"])
    md = []
    def P(*a):
        print(*a); md.append(" ".join(str(x) for x in a))
    P(f"## A. cells: {len(C)}; corrected baselines. excess = optical alone + discrete (from R) - total on L (dB); k = extra signal / input (dB)\n")
    C.sort(key=lambda c: (c["ratio"], c["dthr"], c["thr"], c["og"], c["dg"], c["lvl"]))
    P("| opto thr | og | discrete | dthr | dg | L dBFS | GR_d (R) | opto alone | opto in situ | excess | out dBFS | k dB |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in C:
        if c["excess"] > 0.15 or c["lvl"] >= 0 or c["dg"] != 12 or c["og"] != 12:
            P(f"| {c['thr']} | {c['og']} | {c['ratio']} | {c['dthr']} | {c['dg']} | {c['lvl']:+.0f} | {c['gr_d']:.2f} | {c['gr_oa']:.2f} | {c['gr_oi']:.2f} | {c['excess']:.2f} | {c['out_db'] - 3.01:.1f} | {c['k_db']:.1f} |")

    # B. is the discrete detector blind to the excess? predict GR_d from the discrete law at the series level and compare with R
    fit_bleed(C, "const", False)   # sets gr_ot, the optical stage's true gain reduction (its alone measurement carries the bleed)
    P("\n## B. the discrete detector: GR_d read on R against the discrete law evaluated at the series level (the optical divider output, i.e. the true optical GR, plus the make-up and the inter-stage gain)\n")
    P("| cell | L | GR_d measured (R) | predicted from the series level | difference | predicted if the detector saw the in-situ optical output |"); P("|---|---|---|---|---|---|")
    diffs = []
    for c in C:
        if c["dg"] != 12: continue
        det = dbl(0.5 * (lin(c["lvl"] - c["gr_ot"] + c["G_om"] - L.G_om(1, 12)) + lin(R_PROBE)))
        pred = L.gr_d(c["ratio"], c["dthr"], det); det2 = dbl(0.5 * (lin(c["lvl"] - c["gr_oi"] + c["G_om"] - L.G_om(1, 12)) + lin(R_PROBE))); pred2 = L.gr_d(c["ratio"], c["dthr"], det2)
        if np.isnan(pred) or np.isnan(pred2): continue
        c["gr_d_series"] = pred; c["gr_d_backoff"] = pred2; diffs.append(c["gr_d"] - pred)
        if c["excess"] > 1.0: P(f"| opto {c['thr']} og{c['og']} -> {c['ratio']} {c['dthr']} | {c['lvl']:+.0f} | {c['gr_d']:.2f} | {pred:.2f} | {c['gr_d'] - pred:+.2f} | {pred2:.2f} ({pred2 - c['gr_d']:+.1f}) |")
    d = np.array(diffs); P(f"\nall {len(d)} dg-12 cells: R minus series prediction mean {d.mean():+.2f}, rms {np.sqrt(np.mean(d**2)):.2f}, max |{np.abs(d).max():.2f}| dB")
    for og in (1, 6, 12, 18, 24):
        dd = np.array([c["gr_d"] - c["gr_d_series"] for c in C if c["dg"] == 12 and c["og"] == og and "gr_d_series" in c])
        if len(dd): P(f"optical gain {og:2d}: {len(dd)} cells, mean {dd.mean():+.2f}, rms {np.sqrt(np.mean(dd**2)):.2f}, max |{np.abs(dd).max():.2f}| dB")
    heavy = [c for c in C if c["dg"] == 12 and c["excess"] > 2.0 and "gr_d_series" in c]
    P(f"cells with excess over 2 dB ({len(heavy)}): R minus series prediction mean {np.mean([c['gr_d'] - c['gr_d_series'] for c in heavy]):+.2f} dB; if the detector saw the in-situ optical output it would read {np.mean([c['gr_d_backoff'] - c['gr_d'] for c in heavy]):+.2f} dB more on average")

    # C. the bleed law
    P("\n## C. the bleed law: out = G g_o g_d x + kappa (1 - g_o)^p h(GR_d) x, fitted on cells with excess >= 0.3 dB and optical make-up >= 6\n")
    P("| form of h | kappa dB | p | h parameters | h(0) | rms / max residual on fitted cells (dB) | rms / max on all cells (dB) | n |"); P("|---|---|---|---|---|---|---|---|")
    fits = {}
    for form, fp in (("const", False), ("const", True), ("ramp", False), ("pow", False), ("hill", False)):
        f = fit_bleed(C, form, fp); fits[(form, fp)] = f
        P(f"| {form} | {f['kappa_db']:.2f} | {f['p']:.2f} | {', '.join(f'{v:.3g}' for v in f['hp'])} | {f['h0']:.2f} | {f['rms_fit']:.2f} / {f['max_fit']:.2f} | {f['rms_all']:.2f} / {f['max_all']:.2f} | {f['n']} |")
    P("\nthe constant on subsets (h = 1, p = 1):\n")
    P("| subset | kappa dB | 1/kappa | rms / max residual (dB) | n |"); P("|---|---|---|---|---|")
    for name, pick in (("all", lambda c: True), ("FLOOD only", lambda c: c["ratio"] == "Flood"), ("4:1 only", lambda c: c["ratio"] == "4:1"), ("2:1 only", lambda c: c["ratio"] == "2:1"),
                       ("excess over 3 dB", lambda c: c["excess"] > 3.0), ("discrete gain 1", lambda c: c["dg"] == 1), ("discrete gain 18 or 24", lambda c: c["dg"] >= 18),
                       ("optical gain 6", lambda c: c["og"] == 6), ("optical gain 18 or 24", lambda c: c["og"] >= 18), ("optical threshold 14", lambda c: c["thr"] == 14), ("optical threshold 24", lambda c: c["thr"] == 24),
                       ("discrete at threshold 1 (no discrete GR)", lambda c: c["dthr"] == 1), ("L at -10 dBFS", lambda c: c["lvl"] == -10.0), ("L at +6 dBFS", lambda c: c["lvl"] == 6.0)):
        sub = [c for c in C if pick(c)]
        if sum(1 for c in sub if c["excess"] >= 0.3 and c["og"] >= 6) < 2:
            sub2 = [dict(c) for c in sub]; f = fit_bleed(sub2, "const", False, cut=0.15) if sub2 else None
        else: sub2 = [dict(c) for c in sub]; f = fit_bleed(sub2, "const", False)
        if f: P(f"| {name} | {f['kappa_db']:.2f} | {1.0 / lin(f['kappa_db']):.1f} | {f['rms_fit']:.3f} / {f['max_fit']:.3f} | {f['n']} |")
    best = fits[("const", False)]; fit_bleed(C, "const", False)   # leave the constant law's predictions on the cells
    P(f"\nchosen: the constant, kappa {best['kappa_db']:.2f} dB = 1/{1.0 / lin(best['kappa_db']):.1f}, no dependence on the discrete stage")
    P("\nmeasured k against the fit, per cell with excess >= 1 dB (k_fit = kappa (1 - g_o) h):\n")
    P("| cell | L | GR_o true | GR_d | excess meas | excess pred | residual |"); P("|---|---|---|---|---|---|---|")
    for c in C:
        if c["excess"] >= 1.0: P(f"| opto {c['thr']} og{c['og']} -> {c['ratio']} {c['dthr']} dg{c['dg']} | {c['lvl']:+.0f} | {c['gr_ot']:.2f} | {c['gr_d']:.2f} | {c['excess']:.2f} | {c['pred_excess']:.2f} | {c['res']:+.2f} |")
    # the k against GR_o (cross-drive) and against GR_d (threshold sweep) as the two marginals, measured against the law
    P("\nmarginals: k = extra / input in dB. Against optical GR at heavy discrete GR (cross-drive grid, R at 20 kHz -10 and -20 dBFS):\n")
    P("| L level (1 kHz) | GR_o alone | GR_d (R at -20) | k meas | k law | GR_d (R at -10) | k meas | k law |"); P("|---|---|---|---|---|---|---|---|")
    def p2(s, la, fa, lb, fb): return res.get("|".join(["probe2", json.dumps(s, sort_keys=True), str(la), str(fa), str(lb), str(fb)]))
    for ratio, dthr in (("Flood", 20), ("4:1", 16)):
        s = S(**OPTO(22), **DISC(dthr, ratio)); y0 = p2(s, -60, 1000.0, -80, 20000.0); yr0 = p2(s, -60, 1000.0, -60, 20000.0)
        so = S(**OPTO(22)); G0 = y0[0] + 63.0103
        for la in (-30, -20, -10, 0, 6):
            alone = (la + 60) - (p2(so, la, 1000.0, -80, 20000.0)[0] - p2(so, -60, 1000.0, -80, 20000.0)[0]); row = [f"| {la:+d} ({ratio} {dthr}) | {alone:.2f}"]
            for lb in (-20, -10):
                y = p2(s, la, 1000.0, lb, 20000.0); grd = (lb + 60) - (y[1] - yr0[1]); tot = (la + 60) - (y[0] - y0[0])
                c = dict(G0=G0, gr_oa=alone, gr_d=grd, G_om=L.G_om(22, 12)); c["gr_ot"] = true_opto(c, lin(best["kappa_db"]) * best["h0"])
                extra = lin(G0 - tot) - lin(G0 - c["gr_ot"] - grd); klaw = dbl(bleed_k(c, best["kappa_db"], 1.0, best["hp"], "const"))
                row.append(f"| {grd:.2f} | {dbl(extra) if extra > 0 else float('nan'):.1f} | {klaw:.1f}")
            P(" ".join(row) + " |")
    P("\nAgainst discrete GR at fixed optical GR (discrete threshold sweep, opto 22, L +6 and 0):\n")
    P("| ratio | L | dthr | GR_d | excess | k meas | k law |"); P("|---|---|---|---|---|---|---|")
    for c in C:
        if c["thr"] == 22 and c["og"] == 12 and c["dg"] == 12 and c["lvl"] in (0.0, 6.0) and c["excess"] > 0.05 and c["dthr"] in (12, 14, 16, 18, 20, 22, 24):
            klaw = dbl(bleed_k(c, best["kappa_db"], 1.0, best["hp"], "const")); P(f"| {c['ratio']} | {c['lvl']:+.0f} | {c['dthr']} | {c['gr_d']:.2f} | {c['excess']:.2f} | {c['k_db']:.1f} | {klaw:.1f} |")

    # D. the step: per period from p1, the law applied to the measured alone envelopes
    P("\n## D. the step (-50 -> 0 dBFS on L, R at -40, opto 22): the law applied per 1 kHz period to the measured optical-alone and discrete (R) envelopes; L measured minus predicted (dB)\n")
    P("| discrete | attack ms | p1 | p2 | p3 | p5 | p10 | p20 | p50 | p100 | p200 | p400 | rms p1..p400 |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ratio, dthr in (("Flood", 20), ("4:1", 16)):
        s0 = S(**OPTO(22), **DISC(dthr, ratio)); b60 = res["|".join(["probe", json.dumps(s0, sort_keys=True), "-60.0", "-60.0"])]; b40 = res["|".join(["probe", json.dumps(s0, sort_keys=True), "-60.0", "-40.0"])]
        off = (b40[1] - b60[1]) - 20.0          # the probe's own discrete GR at the -40 baseline (negative)
        G0 = b60[0] + 63.0103
        for att in (0.1, 1.0, 30.0):
            s = S(**OPTO(22), **DISC(dthr, ratio), discrete_attack=att); r = res["|".join(["step", json.dumps(s, sort_keys=True)])]
            Lm = np.array(r["L"]) + 50.0; Rm = np.array(r["R"]); Lo = np.array(r["Lo"]) + 50.0     # dB re the settled -60 renders (R at -40): per-period gains
            grd = -(Rm + off); gro = 60.0 - Lo                                                     # optical alone GR per period (the alone render's -60 baseline has no discrete stage)
            c = [dict(G0=G0, gr_oa=g, gr_d=d, G_om=L.G_om(22, 12)) for g, d in zip(gro, grd)]
            pred = []
            for cc in c:
                cc["gr_ot"] = true_opto(cc, lin(best["kappa_db"]) * best["h0"]); pred.append(dbl(lin(G0 - cc["gr_ot"] - cc["gr_d"]) + bleed_k(cc, best["kappa_db"], 1.0, best["hp"], "const")))
            Lmeas = Lm + b40[0] + 3.0103                              # per-period L level re the 0 dBFS input (Lm is re the -60 render with R at -40, whose level is b40)
            d = Lmeas - np.array(pred); ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400]
            P(f"| {ratio} {dthr} | {att} | " + " | ".join(f"{d[k]:+.2f}" for k in ks) + f" | {np.sqrt(np.mean(d[1:400]**2)):.2f} |")

    # E. the hypotheses on the engine's laws: fixed points
    P("\n## E. hypotheses: static fixed points on our engine's stage laws, against the measured L excess and the R reading\n")
    O = Ours(); LO = Laws(res, ours=O.probe)
    def phi_table(thr, og=12):
        lv = np.arange(-40.0, 12.1, 1.0); gr = LO.opto(thr, og, lv); s = lv - gr; return s, gr
    hyp_rows = []
    def fixed_point(c, hyp, par):
        """returns (GR_o, GR_d) under the hypothesis for a cell"""
        thr, og, ratio, dthr, dg, lvl = c["thr"], c["og"], c["ratio"], c["dthr"], c["dg"], c["lvl"]
        sc, grc = phi_table(thr, og); dG = LO.G_om(thr, og) - LO.G_om(1, 12); Gdm = float(hvmc_core.Model().cal[hvmc_core.Model().field("d_gain_db")][dg - 1]) if False else O.gdm(dg)
        def det(gro): return dbl(0.5 * (lin(lvl - gro + dG) + lin(R_PROBE)))
        best = None
        for gro in np.arange(0.0, 30.0, 0.05):
            grd = LO.gr_d(ratio, dthr, det(gro), dg)
            if hyp == "series": sc_lvl = lvl - gro
            elif hyp == "H1": sc_lvl = lvl - gro - grd
            elif hyp == "H2": sc_lvl = dbl(lin(lvl - gro) * (1.0 + par * lin(dG + 2.14 + Gdm - grd)))
            elif hyp == "H3": sc_lvl = lvl - gro
            target = float(np.interp(sc_lvl, sc, grc))
            if hyp == "H3": target = dbl(1.0 + (lin(target) - 1.0) * np.exp(-par * grd))
            err = abs(gro - target)
            if best is None or err < best[0]: best = (err, gro, grd)
        return best[1], best[2]
    sel = [c for c in C if c["og"] in (1, 12) and c["dg"] in (1, 12) and c["lvl"] in (-10.0, 0.0, 6.0) and c["thr"] in (14, 22) and (c["ratio"], c["dthr"]) in (("Flood", 20), ("4:1", 22), ("4:1", 16), ("2:1", 14))]
    P(f"{len(sel)} cells (opto 14 / 22, FLOOD 20, 4:1 22, 4:1 16, 2:1 14, make-ups 1 and 12, L -10 / 0 / +6). Residuals: L excess (measured minus predicted) and R reading (measured GR_d minus predicted), dB rms / max\n")
    P("| hypothesis | parameter | L excess rms / max | R reading rms / max | note |"); P("|---|---|---|---|---|")
    from scipy.optimize import minimize_scalar
    def run_hyp(hyp, par):
        eL, eR = [], []
        for c in sel:
            gro, grd = fixed_point(c, hyp, par)
            ours_series = fixed_point(c, "series", 0.0)
            pred_excess = (ours_series[0] - gro)  # the optical back-off under the hypothesis, in dB of L level, plus the discrete change
            pred_excess += (ours_series[1] - grd)
            eL.append(c["excess"] - pred_excess); eR.append(c["gr_d"] - grd)
        return np.array(eL), np.array(eR)
    eL, eR = run_hyp("series", 0.0); P(f"| series of parts (our engine, +2.14 in) | - | {np.sqrt(np.mean(eL**2)):.2f} / {np.abs(eL).max():.2f} | {np.sqrt(np.mean(eR**2)):.2f} / {np.abs(eR).max():.2f} | the R reading matches: the discrete detector sees the series signal |")
    eL, eR = run_hyp("H1", 0.0); P(f"| H1 optical sidechain = discrete cell output (pre make-up) | - | {np.sqrt(np.mean(eL**2)):.2f} / {np.abs(eL).max():.2f} | {np.sqrt(np.mean(eR**2)):.2f} / {np.abs(eR).max():.2f} | the loop opens fully: the discrete detector would see it |")
    for f in (0.1, 0.3, 1.0):
        eL, eR = run_hyp("H2", f); P(f"| H2 optical sidechain = own output + f x discrete output | f = {f} | {np.sqrt(np.mean(eL**2)):.2f} / {np.abs(eL).max():.2f} | {np.sqrt(np.mean(eR**2)):.2f} / {np.abs(eR).max():.2f} | |")
    for b in (0.02, 0.05, 0.1):
        eL, eR = run_hyp("H3", b); P(f"| H3 optical conductance x exp(-beta GR_d) | beta = {b} | {np.sqrt(np.mean(eL**2)):.2f} / {np.abs(eL).max():.2f} | {np.sqrt(np.mean(eR**2)):.2f} / {np.abs(eR).max():.2f} | |")
    # H4: the +2.14 placement. audio-only after the detector: R would read GR_d at the level without the 2.14
    eR4 = []
    for c in sel:
        dG = LO.G_om(c["thr"], c["og"]) - LO.G_om(1, 12); grd_a = LO.gr_d(c["ratio"], c["dthr"], dbl(0.5 * (lin(c["lvl"] - c["gr_oa"] + dG - 2.14) + lin(R_PROBE))), c["dg"]); eR4.append(c["gr_d"] - grd_a)
    eR4 = np.array(eR4); P(f"| H4 the +2.14 on the audio only, after the discrete detector | - | (excess unchanged) | {np.sqrt(np.mean(eR4**2)):.2f} / {np.abs(eR4).max():.2f} | the R reading says the detector has the +2.14 (row 1) |")
    # H5: d_goff. the R reading already carries it; the excess against dg is what remains
    dgc = [c for c in C if c["thr"] == 22 and c["og"] == 12 and c["ratio"] == "Flood" and c["dthr"] == 20 and c["lvl"] == 0.0]
    P(f"| H5 the make-up dependence is d_goff on an over-compressing curve | - | excess at FLOOD 20 / L 0 by discrete gain 1..24: " + ", ".join(f"{c['excess']:.1f}" for c in sorted(dgc, key=lambda c: c['dg'])) + " | R reads GR_d " + ", ".join(f"{c['gr_d']:.1f}" for c in sorted(dgc, key=lambda c: c['dg'])) + " | the detector offset is in the R reading and in the series alike; it leaves the excess untouched |")

    # F. our engine on the grid, with and without the law applied after it
    P("\n## F. our engine on the grid: its own excess, and the law applied after it (L level, ours minus reference, dB)\n")
    P("| cell | L | ref excess | ours excess | ours L minus ref L | with the bleed added after our engine |"); P("|---|---|---|---|---|---|")
    rows = []
    for c in sel:
        s = S(**OPTO(c["thr"]), **DISC(c["dthr"], c["ratio"]), optical_gain=c["og"], discrete_gain=c["dg"])
        y = O.probe(s, c["lvl"], R_PROBE); b = O.probe(s, -60.0, -60.0); so = S(**OPTO(c["thr"]), optical_gain=c["og"]); ya = O.probe(so, c["lvl"], R_PROBE); ba = O.probe(so, -60.0, R_PROBE)
        G0 = b[0] + 63.0103; grd = 20.0 - (y[1] - b[1]); grL = (c["lvl"] + 60) - (y[0] - b[0]); gra = (c["lvl"] + 60) - (ya[0] - ba[0])
        exc = gra + grd - grL; cc = dict(G0=G0, gr_oa=gra, gr_ot=gra, gr_d=grd, G_om=LO.G_om(c["thr"], c["og"]))
        with_bleed = dbl(lin(G0 - gra - grd) + bleed_k(cc, best["kappa_db"], 1.0, best["hp"], "const"))
        ref_out = c["out_db"]; d0 = (G0 - grL) - ref_out; d1 = with_bleed - ref_out; rows.append((d0, d1))
        P(f"| opto {c['thr']} og{c['og']} -> {c['ratio']} {c['dthr']} dg{c['dg']} | {c['lvl']:+.0f} | {c['excess']:.2f} | {exc:.2f} | {d0:+.2f} | {d1:+.2f} |")
    r = np.array(rows); P(f"\nrms over these cells: ours minus reference {np.sqrt(np.mean(r[:,0]**2)):.2f} dB (max {np.abs(r[:,0]).max():.2f}); with the bleed {np.sqrt(np.mean(r[:,1]**2)):.2f} dB (max {np.abs(r[:,1]).max():.2f})")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md))
    json.dump({"best": best, "fits": {f"{k[0]}{'_p' if k[1] else ''}": v for k, v in fits.items()}}, open(os.path.join(OUT, "fit.json"), "w"), indent=1)
    return C

class Ours:
    """our engine (fit/hvmc_core.py) on the same probes, cached"""
    def __init__(self):
        self.M = hvmc_core.Model(); self.path = os.path.join(OUT, "ours.json")
        self.res = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self.gain = self.M.cal[self.M.field("d_gain_db")]
    def gdm(self, dg): return float(self.gain[dg - 1])
    def probe(self, s, lvl, r_lvl):
        k = "|".join(["probe", json.dumps(s, sort_keys=True), str(float(lvl)), str(float(r_lvl))])
        if k not in self.res:
            y = self.M.render(stereo(tone(lvl), tone(r_lvl)), FS, s); self.res[k] = last_db(y); json.dump(self.res, open(self.path, "w"))
        return self.res[k]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", default="measure"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    parts = a.parts.split(",")
    if "measure" in parts or "all" in parts: measure(a.force)
    if "measure2" in parts or "all" in parts: measure2(a.force)
    if "measure3" in parts or "all" in parts: measure3(a.force)
    if "measure4" in parts or "all" in parts: measure4(a.force)
    if "measure5" in parts or "all" in parts: measure5(a.force)
    if "measure6" in parts or "all" in parts: measure6(a.force)
    if "measure7" in parts or "all" in parts: measure7(a.force)
    if "analyse" in parts or "all" in parts: analyse()
