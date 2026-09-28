#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Programme-material audit: where the model departs from the licensed reference plug-in on real and synthetic programme, which
block each departure lives in, and what would remove it. Generalises fit/tools/real_audio_compare.py.

Four parts, each a table on stdout (and, with --out, a markdown file plus a JSON of every number):
  grid       every signal x every setting: rms level difference, the gain-trajectory difference (ours minus reference, per 10 ms
             and per 100 ms window: rms, max, where in time the max falls and whether that window is an onset, a release or
             steady), the null depth, band-wise spectrum differences and the crest factor of input and both outputs;
  localise   for every grid case whose level or 100 ms trajectory difference exceeds 0.3 dB: the same case with the other stage
             bypassed and with the transformer at Nickel (attribution), the residual after one fitted gain offset and after one
             fitted threshold offset (the calibration's threshold table shifted by a scalar through the C interface, nothing in
             src/ changed), and the mean difference over onset, release and steady windows (attack-, release- or static-shaped);
  charge     the discrete detector's charge law beyond a sine: steady gain reduction on a sine, two-tone, multi-tone, square and
             pink-noise signals at equal peak and at equal rms, ours against the reference, per attack position; the same on the
             optical stage;
  routing    the sidechain filter's insertion gain (gain reduction with the filter in minus out, on tones above its corner and on
             programme) and the stereo link scale (STEREO minus DUAL MONO on material with different left and right content),
             ours against the reference;
  steps      the pink-noise level steps as a static curve on noise, per ratio and ballistic setting;
  series     the two-stage settings decomposed: each plugin's discrete stage fed with the reference's optical output;
  ballistics whether the reference's detector changes with the ratio switch (step response at 4:1, FLOOD and 2:1).
Results of every part merge into <out>/results.json, so parts can be rerun one at a time.

The reference is driven through fit/measure/pa.py (its own parameter names); the model through fit/hvmc_core.py, the ctypes bridge
to the engine the plugin runs (bit-identical to the VST3 through Pedalboard: --check-vst3 prints the null). Signals: the real mix
(default ~/shadow/listen_test/01_dry_house_mix.wav) at 0, -6 and -12 dB, and synthetic programme built here with numpy.
usage: python3 fit/tools/program_audit.py [--mix path] [--parts grid,localise,steps,charge,routing,series,ballistics] [--quick] [--out build/program_audit]
       [--plots] [--check-vst3]
Needs the licensed reference plug-in; not part of the test suite."""
import argparse, json, os, sys, time
import numpy as np
from scipy import signal as sps
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "fit", "measure")); sys.path.insert(0, os.path.join(ROOT, "fit")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import pa, hvmc_core  # noqa: E402

FS = 48000
BANDS = [20, 60, 150, 400, 1000, 2500, 6000, 12000, 20000]
BAND_NAMES = ["20-60", "60-150", "150-400", "400-1k", "1-2.5k", "2.5-6k", "6-12k", "12-20k"]
DEPART = 0.3          # dB: a case beyond this in level or 100 ms trajectory rms is localised
SLOPE_DB = 3.0        # dB per window: input envelope rise above this is an onset, fall below minus this a release
MASK_DBFS = -70.0     # windows with the input below this are not compared

# ------------------------------------------------------------------------------------------------------------ helpers
def db(x): return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))
def rms(x): return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))
def crest_db(y): return float(db(np.abs(y).max()) - db(rms(y)))

def env_db(y, fs, ms):
    """per-window rms in dB of a (2, n) or (n,) signal, both channels' power pooled"""
    y = np.atleast_2d(np.asarray(y, dtype=np.float64)); n = int(fs * ms / 1000); m = y.shape[1] // n
    p = np.mean(y[:, :m * n].reshape(y.shape[0], m, n) ** 2, axis=(0, 2))
    return db(np.sqrt(p))

def classify(ein):
    """per window: 'onset' if the input rose more than SLOPE_DB since the previous window, 'release' if it fell, else 'steady'"""
    d = np.diff(ein, prepend=ein[0])
    cls = np.full(len(ein), "steady", dtype=object)
    cls[d > SLOPE_DB] = "onset"; cls[d < -SLOPE_DB] = "release"
    return cls

def traj(x, yo, yr, fs, ms):
    """gain-trajectory difference ours minus reference per window, the mask, the input envelope"""
    ein = env_db(x, fs, ms); go = env_db(yo, fs, ms) - ein; gr = env_db(yr, fs, ms) - ein
    mask = ein > MASK_DBFS
    return go - gr, mask, ein

def bands_db(y, fs):
    y = np.atleast_2d(np.asarray(y, dtype=np.float64)); w = np.hanning(y.shape[1])
    P = np.mean([np.abs(np.fft.rfft(c * w)) ** 2 for c in y], axis=0); f = np.fft.rfftfreq(y.shape[1], 1 / fs)
    return np.array([10 * np.log10(P[(f >= a) & (f < b)].sum() + 1e-30) for a, b in zip(BANDS[:-1], BANDS[1:])])

def null_db(yo, yr): return float(db(rms(yo - yr)) - db(rms(yr)))

def metrics(x, yo, yr, fs):
    n = min(x.shape[1], yo.shape[1], yr.shape[1]); x, yo, yr = x[:, :n], yo[:, :n], yr[:, :n]
    out = {"level_db": float(db(rms(yo)) - db(rms(yr))), "gr_ref_db": float(db(rms(x)) - db(rms(yr))), "gr_ours_db": float(db(rms(x)) - db(rms(yo))),
           "null_db": null_db(yo, yr), "crest_in": crest_db(x), "crest_ref": crest_db(yr), "crest_ours": crest_db(yo),
           "bands_db": (bands_db(yo, fs) - bands_db(yr, fs)).round(2).tolist()}
    for ms in (10, 100):
        d, mask, ein = traj(x, yo, yr, fs, ms); cls = classify(ein)
        dm = d[mask]
        i = int(np.argmax(np.abs(np.where(mask, d, 0.0))))
        out[f"d{ms}_rms"] = float(rms(dm)) if len(dm) else 0.0
        out[f"d{ms}_max"] = float(d[i]); out[f"d{ms}_max_t"] = float(i * ms / 1000); out[f"d{ms}_max_class"] = str(cls[i])
        out[f"d{ms}_mean"] = float(np.mean(dm)) if len(dm) else 0.0
        for c in ("onset", "release", "steady"):
            sel = mask & (cls == c)
            out[f"d{ms}_{c}"] = float(np.mean(d[sel])) if sel.any() else float("nan")
            out[f"d{ms}_{c}_n"] = int(sel.sum())
    return out

# ------------------------------------------------------------------------------------------------------------ signals
def pink(n, rng, fs=FS, hp_hz=20.0):
    """pink noise (1/f power) band-limited to hp_hz .. fs/2, unit rms. Without the high pass an FFT 1/f spectrum carries almost half
    its power below 20 Hz, and that subsonic drive saturates both transformer cores (a flux ceiling that falls 6 dB per octave)."""
    X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1.0 / fs); f[0] = f[1]
    X = X / np.sqrt(f); X[f < hp_hz] = 0.0
    y = np.fft.irfft(X, n); return y / rms(y)

def to_peak(y, dbfs): return y / np.abs(y).max() * 10 ** (dbfs / 20)
def to_rms(y, dbfs): return y / rms(y) * 10 ** (dbfs / 20)
def stereo(l, r=None): return np.stack([l, l if r is None else r]).astype(np.float32)

def synth_signals(fs=FS):
    rng = np.random.default_rng(7); S = {}
    # kick and bass loop, 120 bpm: a pitched kick on every beat, an 8th-note bass line alternating 65 and 80 Hz, 8 s
    T = 8.0; n = int(T * fs); t = np.arange(n) / fs; y = np.zeros(n)
    for k in range(int(T * 2)):
        i0 = int(k * 0.5 * fs); tt = t[: n - i0]
        f = 55 + 95 * np.exp(-tt / 0.04); ph = 2 * np.pi * np.cumsum(f) / fs
        y[i0:] += np.sin(ph) * np.exp(-tt / 0.12) * (1 - np.exp(-tt / 0.001))
    for k in range(int(T * 4)):
        i0 = int(k * 0.25 * fs); i1 = min(i0 + int(0.22 * fs), n); tt = t[: i1 - i0]; f0 = 65.0 if k % 2 == 0 else 80.0
        e = (1 - np.exp(-tt / 0.002)) * (0.35 + 0.65 * np.exp(-tt / 0.15)); e[-int(0.01 * fs):] *= np.linspace(1, 0, int(0.01 * fs))
        y[i0:i1] += 0.6 * e * (np.sin(2 * np.pi * f0 * tt) + 0.25 * np.sin(2 * np.pi * 2 * f0 * tt))
    S["kickbass"] = (stereo(to_peak(y, -3)), fs, "kick + bass loop 120 bpm, 60-80 Hz, peak -3 dBFS")
    # snare-like noise burst train: band-passed noise bursts with a 200 Hz body on the backbeat, ghost notes between, 8 s
    y = np.zeros(n); bp = sps.butter(2, [150, 6000], btype="band", fs=fs, output="sos")
    for k in range(int(T * 2)):
        i0 = int((k * 0.5 + 0.25) * fs); i1 = min(i0 + int(0.25 * fs), n); tt = t[: i1 - i0]
        burst = sps.sosfilt(bp, rng.standard_normal(i1 - i0)) * np.exp(-tt / 0.08) * (1 - np.exp(-tt / 0.001))
        body = np.sin(2 * np.pi * 200 * tt) * np.exp(-tt / 0.06)
        y[i0:i1] += burst / 3 + 0.5 * body
        ig = int((k * 0.5 + 0.125) * fs); ig1 = min(ig + int(0.1 * fs), n); tg = t[: ig1 - ig]
        y[ig:ig1] += 0.25 * sps.sosfilt(bp, rng.standard_normal(ig1 - ig)) / 3 * np.exp(-tg / 0.04)
    y = to_peak(y, -6) + to_rms(pink(n, rng, fs), -70)   # a room floor, so the releases between hits are defined
    S["snare"] = (stereo(y), fs, "snare-like noise burst train, backbeat + ghosts, peak -6 dBFS, -70 dBFS floor")
    # sustained pad with slow swells: a four-note chord, six harmonics each, detuned pairs, 4 s raised-cosine swells, 12 s
    T2 = 12.0; n2 = int(T2 * fs); t2 = np.arange(n2) / fs; y = np.zeros(n2)
    for f0 in (110.0, 164.81, 220.0, 277.18):
        for k in range(1, 7):
            for det in (-0.3, 0.3):
                y += np.sin(2 * np.pi * (f0 * k + det) * t2 + rng.uniform(0, 2 * np.pi)) / k ** 1.5
    swell = 10 ** ((-20 + 14 * (0.5 - 0.5 * np.cos(2 * np.pi * t2 / 4.0))) / 20)
    y = y / np.abs(y).max() * swell
    S["pad"] = (stereo(y), fs, "sustained pad, 4 s swells from -20 to -6 dBFS peak")
    # pink noise with 10 dB level steps every 2 s
    steps = [-36, -26, -16, -26, -16, -36]; y = np.concatenate([to_rms(pink(int(2 * fs), rng, fs), s) for s in steps])
    S["pinksteps"] = (stereo(y), fs, "pink noise 20 Hz-24 kHz, rms steps " + " ".join(str(s) for s in steps) + " dBFS every 2 s")
    # bass-heavy sustained tone plus hi-hat clicks (sidechain filter test), 8 s
    y = 10 ** (-6 / 20) * np.sin(2 * np.pi * 55 * t); hp = sps.butter(2, [6000, 14000], btype="band", fs=fs, output="sos")
    for k in range(int(T * 8)):
        i0 = int(k * 0.125 * fs); i1 = min(i0 + int(0.03 * fs), n); tt = t[: i1 - i0]
        y[i0:i1] += 10 ** (-14 / 20) * to_peak(sps.sosfilt(hp, rng.standard_normal(i1 - i0)), 0) * np.exp(-tt / 0.008)
    S["basshats"] = (stereo(y), fs, "55 Hz tone at -6 dBFS peak + hi-hat clicks at -14 dBFS every 125 ms")
    # wide stereo with anti-phase low end (stereo link test): a mid chord and uncorrelated pink per side, a 50 Hz tone in anti-phase
    mid = np.zeros(n)
    for f0 in (329.63, 392.0, 493.88, 659.26):
        for k in range(1, 5): mid += np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) / k ** 1.2
    mid = to_peak(mid, -14) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t) ** 2)
    side = 10 ** (-8 / 20) * np.sin(2 * np.pi * 50 * t)
    pl = to_rms(pink(n, rng, fs), -26); pr = to_rms(pink(n, rng, fs), -26)
    S["widestereo"] = (stereo(mid + side + pl, mid - side + pr), fs, "wide stereo: shared mid chord, uncorrelated pink per side, 50 Hz anti-phase at -8 dBFS")
    return S

def load_mix(path):
    from pedalboard.io import AudioFile
    with AudioFile(path) as f:
        x, fs = f.read(f.frames), int(f.samplerate)
    return x.astype(np.float32), fs

def all_signals(mix_path, quick=False):
    S = {}
    if mix_path and os.path.exists(mix_path):
        x, fs = load_mix(mix_path)
        if quick: x = x[:, : int(12 * fs)]
        for g in (0, -6, -12):
            S[f"mix{g:+d}"] = ((x * 10 ** (g / 20)).astype(np.float32), fs, f"{os.path.basename(mix_path)} at {g:+d} dB (peak {db(np.abs(x).max()) + g:.1f}, rms {db(rms(x)) + g:.1f} dBFS)")
    else:
        print(f"mix not found at {mix_path}: synthetic signals only")
    S.update(synth_signals())
    return S

# ------------------------------------------------------------------------------------------------------------ settings
OPTO = dict(optical_bypass="In", optical_gain=12)
DISC = dict(discrete_bypass="In", discrete_gain=12, discrete_attack=1.0, discrete_recover="0.5 s")
def S_(*bases, **kw):
    """a setting: the defaults, then each base dict, then the keyword overrides (later wins)"""
    s = dict(mode="Stereo", transformer="Nickel", sidechain_filter="Out")
    for b in bases: s.update(b)
    s.update(kw); return s
SETTINGS = {
    "A both gentle (opto 18 + 2:1 thr14 5ms/0.5s, SCF in, Iron)": S_(OPTO, DISC, discrete_gain=9, optical_threshold=18, discrete_threshold=14, discrete_ratio="2:1", discrete_attack=5.0, sidechain_filter="In", transformer="Iron"),
    "B both heavy (opto 22 + FLOOD thr20 0.1ms/0.1s, Iron)": S_(OPTO, DISC, optical_threshold=22, discrete_threshold=20, discrete_ratio="Flood", discrete_attack=0.1, discrete_recover="0.1 s", transformer="Iron"),
    "C opto alone gentle (thr 16)": S_(OPTO, optical_threshold=16),
    "D opto alone heavy (thr 22)": S_(OPTO, optical_threshold=22),
    "E disc 1.2:1 thr12 1ms/0.5s": S_(DISC, discrete_threshold=12, discrete_ratio="1.2:1"),
    "F disc 2:1 thr12 1ms/0.5s": S_(DISC, discrete_threshold=12, discrete_ratio="2:1"),
    "G disc 3:1 thr12 1ms/0.5s": S_(DISC, discrete_threshold=12, discrete_ratio="3:1"),
    "H disc 4:1 thr12 1ms/0.5s": S_(DISC, discrete_threshold=12, discrete_ratio="4:1"),
    "I disc 6:1 thr12 1ms/0.5s": S_(DISC, discrete_threshold=12, discrete_ratio="6:1"),
    "J disc FLOOD thr12 1ms/0.5s": S_(DISC, discrete_threshold=12, discrete_ratio="Flood"),
    "J2 disc FLOOD thr18 1ms/0.5s": S_(DISC, discrete_threshold=18, discrete_ratio="Flood"),
    "F2 disc 2:1 thr16 1ms/0.5s": S_(DISC, discrete_threshold=16, discrete_ratio="2:1"),
    "K disc 4:1 thr16 0.1ms/0.1s": S_(DISC, discrete_threshold=16, discrete_ratio="4:1", discrete_attack=0.1, discrete_recover="0.1 s"),
    "L disc 4:1 thr16 30ms/1.2s": S_(DISC, discrete_threshold=16, discrete_ratio="4:1", discrete_attack=30.0, discrete_recover="1.2 s"),
    "M disc 4:1 thr16 1ms/DUAL": S_(DISC, discrete_threshold=16, discrete_ratio="4:1", discrete_recover="Dual"),
    "N disc 4:1 thr16 1ms/0.5s": S_(DISC, discrete_threshold=16, discrete_ratio="4:1"),
    "O disc 4:1 thr16 1ms/0.5s SCF in": S_(DISC, discrete_threshold=16, discrete_ratio="4:1", sidechain_filter="In"),
    "P disc 4:1 thr16 1ms/0.5s DUAL MONO": S_(DISC, discrete_threshold=16, discrete_ratio="4:1", mode="Dual Mono"),
    "Q both gentle, MIX 50 %": S_(OPTO, DISC, discrete_gain=9, optical_threshold=18, discrete_threshold=14, discrete_ratio="2:1", discrete_attack=5.0, sidechain_filter="In", transformer="Iron", mix=50.0),
    "R both gentle, SCF out, Nickel": S_(OPTO, DISC, discrete_gain=9, optical_threshold=18, discrete_threshold=14, discrete_ratio="2:1", discrete_attack=5.0),
    "S opto alone thr 22 SCF in": S_(OPTO, optical_threshold=22, sidechain_filter="In"),
    "T xfmr Nickel only": S_(),
    "U xfmr Iron only": S_(transformer="Iron"),
    "V xfmr Steel only": S_(transformer="Steel"),
}

# ------------------------------------------------------------------------------------------------------------ renderers
class Renderers:
    def __init__(self):
        self.P = pa.Ref(); self.M = hvmc_core.Model()
        if not self.M.fitted: sys.exit("the C interface library is not the fitted build: run scripts/build-capi.sh")
    def ref(self, x, fs, s): return self.P.run(x, fs, **pa.both(**s)).astype(np.float64)
    def ours(self, x, fs, s, cal=None): return self.M.render(x, fs, s, profile=0, quality=0, cal=cal)
    def shifted(self, which, delta):
        """the calibration with the threshold table of one stage shifted so that +delta dB means more gain reduction"""
        cal = self.M.cal.copy()
        if which == "disc": cal[self.M.field("d_thr_db")] -= delta
        else: cal[self.M.field("o_thr_db")] += delta
        return cal

# ------------------------------------------------------------------------------------------------------------ parts
def fmt_bands(b): return " ".join(f"{v:+.2f}" for v in b)

def part_grid(R, SIG, only=None):
    rows = {}
    print(f"\n== grid: {len(SIG)} signals x {len(SETTINGS)} settings\n")
    print(f"{'setting':58s} {'signal':11s} {'lvl':>6s} {'GRref':>6s} {'d10rms':>6s} {'d10max':>7s} {'at':>6s} {'class':7s} {'d100rms':>7s} {'d100max':>7s} {'null':>6s} {'crest x/ref/ours':>17s}  bands ({', '.join(BAND_NAMES)})")
    for sname, s in SETTINGS.items():
        for gname, (x, fs, _) in SIG.items():
            if only and (sname, gname) not in only: continue
            yr = R.ref(x, fs, s); yo = R.ours(x, fs, s)
            m = metrics(x, yo, yr, fs); rows[(sname, gname)] = m
            print(f"{sname:58s} {gname:11s} {m['level_db']:+6.2f} {m['gr_ref_db']:6.1f} {m['d10_rms']:6.2f} {m['d10_max']:+7.2f} {m['d10_max_t']:6.2f} {m['d10_max_class']:7s} {m['d100_rms']:7.2f} {m['d100_max']:+7.2f} {m['null_db']:6.1f} {m['crest_in']:5.1f}/{m['crest_ref']:4.1f}/{m['crest_ours']:4.1f}   {fmt_bands(m['bands_db'])}")
    return rows

def part_localise(R, SIG, grid):
    """attribute every departing case: other stage out, Nickel, one gain offset, one threshold offset, shape"""
    print("\n== localise: cases with |level| or 100 ms trajectory rms over %.1f dB\n" % DEPART)
    out = {}
    hdr = f"{'case':70s} {'lvl':>6s} {'d100':>5s} | {'oth.stage out lvl/d100':>23s} {'Nickel lvl/d100':>16s} | {'gain off':>8s} {'resid':>6s} {'null':>6s} | {'thr off':>8s} {'stage':5s} {'resid':>6s} {'lvl':>6s} | {'onset':>6s} {'rel':>6s} {'steady':>6s}"
    print(hdr)
    for (sname, gname), m in grid.items():
        if abs(m["level_db"]) < DEPART and m["d100_rms"] < DEPART: continue
        s = SETTINGS[sname]; x, fs, _ = SIG[gname]; yr = R.ref(x, fs, s); yo = R.ours(x, fs, s)
        r = {"level_db": m["level_db"], "d100_rms": m["d100_rms"]}
        opto_in, disc_in = s.get("optical_bypass") == "In", s.get("discrete_bypass") == "In"
        # the other stage bypassed (both plugins), when both are in
        if opto_in and disc_in:
            for lab, key in (("opto_only", "discrete_bypass"), ("disc_only", "optical_bypass")):
                s2 = {**s, key: "Out"}; mm = metrics(x, R.ours(x, fs, s2), R.ref(x, fs, s2), fs)
                r[lab] = {"level_db": mm["level_db"], "d100_rms": mm["d100_rms"], "gr_ref_db": mm["gr_ref_db"]}
        if s.get("transformer", "Nickel") != "Nickel":
            s2 = {**s, "transformer": "Nickel"}; mm = metrics(x, R.ours(x, fs, s2), R.ref(x, fs, s2), fs)
            r["nickel"] = {"level_db": mm["level_db"], "d100_rms": mm["d100_rms"]}
        # one gain offset: the mean trajectory difference; the residual is what a static offset cannot remove
        d, mask, ein = traj(x, yo, yr, fs, 100); goff = float(np.mean(d[mask])); resid = float(rms(d[mask] - goff))
        r["gain_offset"] = {"db": goff, "resid_d100": resid, "null_after": null_db(yo * 10 ** (-goff / 20), yr)}
        # one threshold offset on the stage that is in (both stages tried when both are in), fitted on the 100 ms trajectory
        best = None
        for which in (["disc"] if disc_in and not opto_in else ["opto"] if opto_in and not disc_in else ["disc", "opto"] if opto_in and disc_in else []):
            def cost(delta):
                dd, mk, _ = traj(x, R.ours(x, fs, s, cal=R.shifted(which, delta)), yr, fs, 100); return rms(dd[mk])
            res = minimize_scalar(cost, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 0.02})
            yo2 = R.ours(x, fs, s, cal=R.shifted(which, res.x)); mm = metrics(x, yo2, yr, fs)
            cand = {"stage": which, "db": float(res.x), "resid_d100": float(res.fun), "level_after": mm["level_db"], "null_after": mm["null_db"], "d10_rms_after": mm["d10_rms"]}
            if best is None or cand["resid_d100"] < best["resid_d100"]: best = cand
        r["thr_offset"] = best
        r["shape"] = {c: m[f"d10_{c}"] for c in ("onset", "release", "steady")}
        out[(sname, gname)] = r
        os_ = r.get("opto_only"); ds_ = r.get("disc_only"); nk = r.get("nickel"); to = r["thr_offset"]
        print(f"{(sname[:44] + ' | ' + gname):70s} {m['level_db']:+6.2f} {m['d100_rms']:5.2f} | "
              f"{('o ' + f'{os_['level_db']:+.2f}/{os_['d100_rms']:.2f}' + ' d ' + f'{ds_['level_db']:+.2f}/{ds_['d100_rms']:.2f}') if os_ else '':>23s} "
              f"{(f'{nk['level_db']:+.2f}/{nk['d100_rms']:.2f}') if nk else '':>16s} | "
              f"{goff:+8.2f} {resid:6.2f} {r['gain_offset']['null_after']:6.1f} | "
              f"{(f'{to['db']:+8.2f} {to['stage']:5s} {to['resid_d100']:6.2f} {to['level_after']:+6.2f}') if to else '':>29s} | "
              f"{m['d10_onset']:+6.2f} {m['d10_release']:+6.2f} {m['d10_steady']:+6.2f}")
    return out

def charge_signals(fs=FS, secs=3.0):
    rng = np.random.default_rng(3); n = int(secs * fs); t = np.arange(n) / fs; S = {}
    S["sine 1k"] = np.sin(2 * np.pi * 1000 * t)
    S["sine 100"] = np.sin(2 * np.pi * 100 * t)
    S["2-tone 1k+1.3k"] = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1300 * t)
    S["2-tone 60+1k"] = np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 1000 * t)
    S["10-tone 100k"] = sum(np.sin(2 * np.pi * 100 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 11))
    S["40-tone log"] = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in np.geomspace(50, 8000, 40))
    sq = np.zeros(n)
    for k in range(1, 20, 2): sq += np.sin(2 * np.pi * 1000 * k * t) / k
    S["square 1k"] = sq
    S["pink"] = pink(n, rng, fs)
    return S

def bypassed(s):
    """the same setting with both stages out: the no-gain-reduction baseline. Threshold position 1 is not a baseline for the model,
    whose node rests where the soft ratios' curves are not zero (0.73 dB of standing gain reduction at 2:1); make-up cancels to the
    law group's 0.05 dB."""
    return {**s, "optical_bypass": "Out", "discrete_bypass": "Out"}

def steady_gr(R, x, fs, s, base):
    """gain reduction as level in over level out of the last 1.5 s, referred to the same signal with the stages bypassed"""
    n0 = -int(1.5 * fs); out = {}
    for lab in ("ref", "ours"):
        f = R.ref if lab == "ref" else R.ours
        y = f(x, fs, s); y0 = f(x, fs, base)
        out[lab] = float(db(rms(y0[:, n0:])) - db(rms(y[:, n0:])))
    return out

def part_charge(R):
    print("\n== charge law: steady gain reduction (dB) on tones, multi-tones, square and pink noise; ref / ours / ours-ref\n")
    sig = charge_signals(); fs = FS; out = {}
    cases = [("disc 4:1 thr16 0.1ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1", discrete_attack=0.1)),
             ("disc 4:1 thr16 1ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1")),
             ("disc 4:1 thr16 5ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1", discrete_attack=5.0)),
             ("disc 4:1 thr16 30ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1", discrete_attack=30.0)),
             ("disc 2:1 thr12 1ms/0.5s", S_(DISC, discrete_threshold=12, discrete_ratio="2:1")),
             ("disc 2:1 thr12 5ms/0.5s", S_(DISC, discrete_threshold=12, discrete_ratio="2:1", discrete_attack=5.0)),
             ("disc FLOOD thr16 1ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="Flood")),
             ("disc FLOOD thr16 0.1ms/0.1s", S_(DISC, discrete_threshold=16, discrete_ratio="Flood", discrete_attack=0.1, discrete_recover="0.1 s")),
             ("disc FLOOD thr16 30ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="Flood", discrete_attack=30.0)),
             ("opto thr20", S_(OPTO, optical_threshold=20))]
    cases = [(n, s, bypassed(s)) for n, s in cases]
    for norm in ("peak -10 dBFS", "rms -13 dBFS (the -10 dBFS sine's rms)"):
        print(f"-- normalised to {norm}")
        print(f"{'case':28s} " + " ".join(f"{k:>20s}" for k in sig))
        for cname, s, base in cases:
            cells = []
            for k, y in sig.items():
                x = stereo(to_peak(y, -10) if norm.startswith("peak") else to_rms(y, -13.01))
                g = steady_gr(R, x, fs, s, base); out[(norm, cname, k)] = g
                cells.append(f"{g['ref']:5.1f}/{g['ours']:5.1f}/{g['ours'] - g['ref']:+5.2f}")
            print(f"{cname:28s} " + " ".join(f"{c:>20s}" for c in cells))
    return out

def part_routing(R, SIG):
    out = {}
    fs = FS; t = np.arange(int(3 * fs)) / fs
    tones = {"sine 1k -10": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t)), "sine 3k -10": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 3000 * t)),
             "sine 300 -10": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 300 * t)), "sine 50 -10": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 50 * t))}
    print("\n== sidechain filter insertion: gain reduction with the filter IN minus OUT (dB; + = more GR with the filter in); ref / ours / ours-ref\n")
    for cname, s in (("disc 4:1 thr16 1ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1")), ("disc 2:1 thr12", S_(DISC, discrete_threshold=12, discrete_ratio="2:1")), ("opto thr20", S_(OPTO, optical_threshold=20)), ("opto thr16", S_(OPTO, optical_threshold=16))):
        cells = []
        items = list(tones.items()) + [(g, SIG[g][0]) for g in ("mix+0", "pad", "basshats", "kickbass") if g in SIG]
        for gname, x in items:
            fsx = SIG[gname][1] if gname in SIG else fs; n0 = -int(1.5 * fsx) if gname in tones else 0
            r = {}
            for lab, f in (("ref", R.ref), ("ours", R.ours)):
                yi = f(x, fsx, {**s, "sidechain_filter": "In"}); yo = f(x, fsx, {**s, "sidechain_filter": "Out"})
                r[lab] = float(db(rms(yo[:, n0:])) - db(rms(yi[:, n0:])))
            out[("scf", cname, gname)] = r; cells.append(f"{gname}: {r['ref']:+.2f}/{r['ours']:+.2f}/{r['ours'] - r['ref']:+.2f}")
        print(f"{cname:26s} " + "  ".join(cells))
    print("\n== stereo link: gain reduction in STEREO minus DUAL MONO per channel (dB; - = less GR when linked); ref / ours / ours-ref\n")
    n = int(3 * fs)
    links = {"L -10 / R -40 (1k)": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t), 10 ** (-40 / 20) * np.sin(2 * np.pi * 1000 * t)),
             "L -10 / R -16 (1k)": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t), 10 ** (-16 / 20) * np.sin(2 * np.pi * 1000 * t)),
             "L 1k / R 1.1k, -10": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t), 10 ** (-10 / 20) * np.sin(2 * np.pi * 1100 * t)),
             "L -10 / R -10 anti": stereo(10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t), -10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t)),
             "L pink / R pink -16": stereo(to_rms(pink(n, np.random.default_rng(5), fs), -16), to_rms(pink(n, np.random.default_rng(6), fs), -16))}
    for cname, s in (("disc 4:1 thr16 1ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1")), ("opto thr20", S_(OPTO, optical_threshold=20))):
        cells = []
        items = list(links.items()) + [(g, SIG[g][0]) for g in ("widestereo", "mix+0") if g in SIG]
        for gname, x in items:
            fsx = SIG[gname][1] if gname in SIG else fs; n0 = -int(1.5 * fsx) if gname in links else 0
            r = {}
            for lab, f in (("ref", R.ref), ("ours", R.ours)):
                ys = f(x, fsx, {**s, "mode": "Stereo"}); yd = f(x, fsx, {**s, "mode": "Dual Mono"})
                r[lab] = [float(db(rms(yd[c, n0:])) - db(rms(ys[c, n0:]))) for c in (0, 1)]
            out[("link", cname, gname)] = r
            cells.append(f"{gname}: L {r['ref'][0]:+.2f}/{r['ours'][0]:+.2f}/{r['ours'][0] - r['ref'][0]:+.2f} R {r['ref'][1]:+.2f}/{r['ours'][1]:+.2f}/{r['ours'][1] - r['ref'][1]:+.2f}")
        print(f"{cname:26s} " + "  ".join(cells))
    print("\n== stereo link, absolute: per-channel gain reduction in STEREO on unequal material (dB); ref / ours / ours-ref\n")
    for cname, s in (("disc 4:1 thr16 1ms/0.5s", S_(DISC, discrete_threshold=16, discrete_ratio="4:1")),):
        base = bypassed(s); cells = []
        for gname, x in links.items():
            r = {}
            for lab, f in (("ref", R.ref), ("ours", R.ours)):
                ys = f(x, fs, s); y0 = f(x, fs, base); r[lab] = [float(db(rms(y0[c, -int(1.5 * fs):])) - db(rms(ys[c, -int(1.5 * fs):]))) for c in (0, 1)]
            out[("link_abs", cname, gname)] = r
            cells.append(f"{gname}: L {r['ref'][0]:5.2f}/{r['ours'][0]:5.2f}/{r['ours'][0] - r['ref'][0]:+.2f} R {r['ref'][1]:5.2f}/{r['ours'][1]:5.2f}/{r['ours'][1] - r['ref'][1]:+.2f}")
        print(f"{cname:26s} " + "  ".join(cells))
    return out

def part_steps(R, SIG):
    """the pink-noise level steps as a static curve on noise: per-step steady gain reduction, ref / ours, per ratio"""
    if "pinksteps" not in SIG: return {}
    print("\n== pink-noise level steps as a static curve: gain reduction per 2 s step (last 1 s of each), ref / ours / ours-ref\n")
    x, fs, _ = SIG["pinksteps"]; steps = [-36, -26, -16, -26, -16, -36]; out = {}
    cases = [(k, v) for k, v in SETTINGS.items() if k[0] in "CDEFGHIJKLM"]
    print(f"{'setting':40s} " + " ".join(f"{'step ' + str(s):>18s}" for s in steps))
    for cname, s in cases:
        base = bypassed(s)
        cells = []; yr = R.ref(x, fs, s); yo = R.ours(x, fs, s); yr0 = R.ref(x, fs, base); yo0 = R.ours(x, fs, base)
        for i, lv in enumerate(steps):
            a, b = int((2 * i + 1) * fs), int((2 * i + 2) * fs)
            gr = float(db(rms(yr0[:, a:b])) - db(rms(yr[:, a:b]))); go = float(db(rms(yo0[:, a:b])) - db(rms(yo[:, a:b])))
            out[(cname, i)] = {"level": lv, "ref": gr, "ours": go}; cells.append(f"{gr:5.2f}/{go:5.2f}/{go - gr:+5.2f}")
        print(f"{cname:40s} " + " ".join(f"{c:>18s}" for c in cells))
    return out

def part_series(R, SIG):
    """the two-stage cases against their parts: the discrete stage of each plugin fed with the REFERENCE's optical output, so that both
    discrete stages see the same, optically compressed programme; then the optical part of the difference on its own"""
    print("\n== series: two-stage settings decomposed. columns: both stages (grid) | opto alone | disc alone on the raw signal | disc alone on the reference's optical output (same input to both) ; ours - ref level dB / d100 rms\n")
    out = {}
    for sname in ("A both gentle (opto 18 + 2:1 thr14 5ms/0.5s, SCF in, Iron)", "B both heavy (opto 22 + FLOOD thr20 0.1ms/0.1s, Iron)"):
        s = SETTINGS[sname]
        for gname in ("mix+0", "mix-12", "basshats", "kickbass", "pinksteps", "snare"):
            if gname not in SIG: continue
            x, fs, _ = SIG[gname]
            both = metrics(x, R.ours(x, fs, s), R.ref(x, fs, s), fs)
            so = {**s, "discrete_bypass": "Out"}; sd = {**s, "optical_bypass": "Out"}
            yo_opto = R.ref(x, fs, so)   # the reference's optical output
            opto = metrics(x, R.ours(x, fs, so), yo_opto, fs)
            disc_raw = metrics(x, R.ours(x, fs, sd), R.ref(x, fs, sd), fs)
            xo = yo_opto.astype(np.float32)
            disc_on_opto = metrics(xo, R.ours(xo, fs, sd), R.ref(xo, fs, sd), fs)
            out[(sname, gname)] = {"both": both["level_db"], "opto": opto["level_db"], "disc_raw": disc_raw["level_db"], "disc_on_ref_opto": disc_on_opto["level_db"],
                                   "both_d100": both["d100_rms"], "disc_raw_d100": disc_raw["d100_rms"], "disc_on_ref_opto_d100": disc_on_opto["d100_rms"],
                                   "gr_ref_disc_raw": disc_raw["gr_ref_db"], "gr_ref_disc_on_opto": disc_on_opto["gr_ref_db"], "crest_raw": crest_db(x), "crest_after_opto": crest_db(xo)}
            print(f"{sname[:40]:40s} {gname:10s} both {both['level_db']:+.2f}/{both['d100_rms']:.2f} | opto {opto['level_db']:+.2f} | disc(raw) {disc_raw['level_db']:+.2f}/{disc_raw['d100_rms']:.2f} (ref GR {disc_raw['gr_ref_db']:.1f}) | disc(ref opto out) {disc_on_opto['level_db']:+.2f}/{disc_on_opto['d100_rms']:.2f} (ref GR {disc_on_opto['gr_ref_db']:.1f}; crest {crest_db(x):.1f} -> {crest_db(xo):.1f})")
    return out

def part_ratio_ballistics(R):
    """does the reference's detector change with the ratio switch? A -50 -> -10 dBFS step and a 300 ms burst at 4:1 and at FLOOD, 1 ms /
    0.5 s, threshold 16: the gain per 1 kHz period from the step, normalised by the settled gain reduction, ours against the reference"""
    print("\n== ratio ballistics: gain per period after a -50 -> -10 dBFS step (dB, referred to the settled value; 1 = settled), 1 kHz, thr 16, 1 ms / 0.5 s; ref / ours\n")
    fs = FS; t = np.arange(int(2.0 * fs)) / fs; lvl = np.where(t < 1.0, 10 ** (-50 / 20), 10 ** (-10 / 20)); x = stereo(lvl * np.sin(2 * np.pi * 1000 * t))
    out = {}
    for ratio in ("4:1", "Flood", "2:1"):
        for att in (0.1, 1.0, 30.0):
            s = S_(DISC, discrete_threshold=16, discrete_ratio=ratio, discrete_attack=att); rows = {}
            for lab, f in (("ref", R.ref), ("ours", R.ours)):
                y = f(x, fs, s); y0 = f(x, fs, bypassed(s)); n = int(fs / 1000); i0 = int(1.0 * fs)
                g = [float(db(rms(y[:, i0 + k * n: i0 + (k + 1) * n])) - db(rms(y0[:, i0 + k * n: i0 + (k + 1) * n]))) for k in range(0, 400)]
                settled = float(np.mean(g[300:400])); rows[lab] = {"gr_settled": -settled, "frac": [gi / settled for gi in g]}
            out[(ratio, att)] = rows
            ks = [0, 1, 2, 3, 5, 10, 20, 50, 100]
            print(f"{ratio:5s} {att:4.1f} ms  settled GR ref {rows['ref']['gr_settled']:5.2f} ours {rows['ours']['gr_settled']:5.2f}; fraction of settled GR at period " + " ".join(f"{k}:{rows['ref']['frac'][k]:.2f}/{rows['ours']['frac'][k]:.2f}" for k in ks))
    return out

def check_vst3(R, SIG):
    import hvmc_pb
    bundle = hvmc_pb.default_bundle(); p = hvmc_pb.load(bundle)
    gname = "mix+0" if "mix+0" in SIG else next(iter(SIG)); x, fs, _ = SIG[gname]; s = SETTINGS["A both gentle (opto 18 + 2:1 thr14 5ms/0.5s, SCF in, Iron)"]
    hvmc_pb.apply_reference(p, s); p.process(np.zeros((2, int(0.05 * fs)), np.float32), fs, reset=True)
    yv = p.process(x, fs, reset=True).astype(np.float64); yc = R.ours(x, fs, s)
    print(f"\n== VST3 ({os.path.basename(bundle)}) against the C interface on {gname}, setting A: null {null_db(yc, yv):.1f} dB")

def plots(R, SIG, grid, outdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    worst = sorted(grid.items(), key=lambda kv: -abs(kv[1]["d100_rms"]))[:6]
    for (sname, gname), m in worst:
        s = SETTINGS[sname]; x, fs, _ = SIG[gname]; yr = R.ref(x, fs, s); yo = R.ours(x, fs, s)
        ein = env_db(x, fs, 10); gr = env_db(yr, fs, 10) - ein; go = env_db(yo, fs, 10) - ein; t = np.arange(len(ein)) * 0.01
        gr[ein < MASK_DBFS] = np.nan; go[ein < MASK_DBFS] = np.nan
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax[0].plot(t, ein, color="0.6", lw=0.7, label="input (dBFS, 10 ms)"); ax[0].set_ylabel("dBFS"); ax[0].legend(loc="lower right", fontsize=8)
        ax[1].plot(t, gr, lw=0.8, label="reference gain"); ax[1].plot(t, go, lw=0.8, label="ours"); ax[1].plot(t, go - gr, lw=0.8, color="crimson", label="ours - reference")
        ax[1].set_ylabel("dB"); ax[1].set_xlabel("s"); ax[1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{sname} | {gname}: level {m['level_db']:+.2f} dB, d100 rms {m['d100_rms']:.2f}, null {m['null_db']:.1f} dB", fontsize=9)
        fig.tight_layout(); fn = os.path.join(outdir, f"{sname[0]}_{gname}.png".replace("+", "p")); fig.savefig(fn, dpi=110); plt.close(fig)
        print("wrote", fn)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mix", default=os.path.expanduser("~/shadow/listen_test/01_dry_house_mix.wav"))
    ap.add_argument("--parts", default="grid,localise,steps,charge,routing,series,ballistics")
    ap.add_argument("--quick", action="store_true", help="12 s of the mix only")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "program_audit"), help="folder for results.json and figures")
    ap.add_argument("--plots", action="store_true"); ap.add_argument("--check-vst3", action="store_true")
    a = ap.parse_args()
    parts = set(a.parts.split(",")); t0 = time.time()
    R = Renderers(); SIG = all_signals(a.mix, a.quick)
    print(f"reference {R.P.version} sha256 {R.P.sha256[:12]}; model calibration {'fitted' if R.M.fitted else 'PRIORS'}")
    for k, (x, fs, lab) in SIG.items(): print(f"  {k:11s} {x.shape[1] / fs:5.1f} s at {fs} Hz  peak {db(np.abs(x).max()):6.1f}  rms {db(rms(x)):6.1f} dBFS  crest {crest_db(x):4.1f}  {lab}")
    if a.check_vst3: check_vst3(R, SIG)
    res = {}
    grid = part_grid(R, SIG) if "grid" in parts else {}
    res["grid"] = {f"{k[0]} | {k[1]}": v for k, v in grid.items()}
    if "localise" in parts and grid: res["localise"] = {f"{k[0]} | {k[1]}": v for k, v in part_localise(R, SIG, grid).items()}
    if "steps" in parts: res["steps"] = {f"{k[0]} | step {k[1]}": v for k, v in part_steps(R, SIG).items()}
    if "charge" in parts: res["charge"] = {" | ".join(k): v for k, v in part_charge(R).items()}
    if "routing" in parts: res["routing"] = {" | ".join(k): v for k, v in part_routing(R, SIG).items()}
    if "series" in parts: res["series"] = {" | ".join(k): v for k, v in part_series(R, SIG).items()}
    if "ballistics" in parts: res["ballistics"] = {f"{k[0]} | {k[1]}": v for k, v in part_ratio_ballistics(R).items()}
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "results.json")
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update({k: v for k, v in res.items() if v}); res = old
    json.dump(res, open(fn, "w"), indent=1, default=lambda o: None if isinstance(o, float) and np.isnan(o) else float(o) if isinstance(o, np.floating) else str(o))
    if a.plots and grid: plots(R, SIG, grid, os.path.join(a.out, "figs"))
    print(f"\ndone in {time.time() - t0:.0f} s; results in {a.out}/results.json")

if __name__ == "__main__":
    main()
