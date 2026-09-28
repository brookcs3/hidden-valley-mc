#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage: does the detector depend on the RATIO switch, what is the 0.1 ms attack, and what does SIDECHAIN FILTER IN do to the
stage's gain? Programme-audit findings 2, 4 and 5 (research/program-audit.md, sections 3.2, 3.4, 3.5).

Three parts, each a flag; results and every number go to build/disc-ratio-ballistics/:
  --capture  drive the reference (fit/measure/pa.py) and the engine (fit/hvmc_core.py) on one measurement set, write measure.json:
               charge   steady gain reduction on a sine, 100 Hz sine, two-tone, 10-tone, 40-tone, square and pink noise at equal peak
                        (-10 dBFS) and at equal rms (-13 dBFS), every ratio x every attack, recover 0.5 s, threshold 16;
               fine     the static curve of every ratio at threshold 16 in 0.5 dB steps (1 ms / 0.5 s), the ruler that turns a gain
                        into an equivalent steady-sine level, so the detector's reading can be compared across ratio positions
                        without a model of either the curve or the node;
               ar/al/af the steady table (attack x recover at -10 dBFS, level -25 / +2, 100 Hz / 5 kHz) at every ratio, not only 4:1;
               burst    -50 -> L dBFS steps at 1 kHz per ratio, attack 0.1 / 0.5 / 1 / 30 ms, L = -30 / -20 / -10 / 0, the per-period
                        lock-in gain (the protocol's 'env') and the per-period rms gain (the audit's), recover 0.5 s (and 0.1 s);
               trim     SIDECHAIN FILTER in against out: the discrete stage at threshold 1 against level, frequency, ratio, make-up,
                        stereo mode, attack; with the stages out; with the optical stage in; at threshold 16 (through the curves);
                        the optical stage's no-compression gain at -50 / -70 dBFS at all 24 threshold positions, its static curve at
                        threshold 20 and its harmonics at -10 dBFS, filter in and out.
  --analyse  the tables: the reference's detector reading per ratio in equivalent sine level (finding 2), the onset fractions and the
             steady table at 0.1 ms (finding 4), the trim and the leak (finding 5).
  --fit      the candidates (finding 2 and 4), in the stage-3 mirror style: fit/stages/stage3_discrete.py's detector, curve_at and
             residual definitions imported unchanged, plus per-ratio parameters:
               R0  the stage-3 detector as fitted (control: one node for every ratio)
               R1  per-ratio attack scale: ta_r = ta * s_r                       (FLOOD and 6:1 free, the rest 1)
               R2  per-ratio conductance law: Sv_r                                (the level dependence per ratio)
               R3  per-ratio release scale: tr_r = tr * q_r
               R4  per-ratio attack AND release scale (R1 + R3)
               R5  per-ratio depth (rest_r = T - depth_r)
             each with and without the 0.1 ms / 0.5 ms attack constants refitted on the new items (the charge law at 0.1 and 0.5 ms
             and the first periods of the 0.1 ms bursts), which is finding 4.
The stage-3 facts are kept in every fit: the 4:1 steady table, the level and frequency checks, every 4:1 burst and the DUAL items
(the stage-3b residual, same weights), so a candidate that fixes FLOOD by breaking 4:1 shows it.
usage: cd <repo> && python3 -u fit/tools/candidates/disc-ratio-ballistics.py --capture [--quick]
       python3 -u fit/tools/candidates/disc-ratio-ballistics.py --analyse
       python3 -u fit/tools/candidates/disc-ratio-ballistics.py --fit [--modes R0,R1,R2,R3,R4,R5] [--nfev 60]
Needs the licensed reference plug-in for --capture; --analyse and --fit read measure.json."""
import os, sys, json, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "fit")); sys.path.insert(0, os.path.join(ROOT, "fit", "measure")); sys.path.insert(0, os.path.join(ROOT, "fit", "stages"))
import protocol  # noqa: E402

FS = protocol.FS
RATIOS = protocol.RATIOS; ATTACKS = protocol.ATTACKS; RECOVERS = protocol.RECOVERS
OUT = os.path.join(ROOT, "build", "disc-ratio-ballistics"); os.makedirs(OUT, exist_ok=True)
MEAS = os.path.join(OUT, "measure.json")
ARGS = sys.argv[1:]
DISC = {"discrete_bypass": "In"}; OPTO = {"optical_bypass": "In"}
BYPASS = {"optical_bypass": "Out", "discrete_bypass": "Out"}
LEVEL_PEAK = -10.0; LEVEL_RMS = LEVEL_PEAK - 20 * np.log10(np.sqrt(2.0))   # the -10 dBFS sine's rms


def db(x): return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-15))
def rms(x): return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


# ------------------------------------------------------------------------------------------------ signals
def pink(n, rng, fs=FS, hp_hz=20.0):
    """pink noise band-limited to hp_hz .. fs/2, unit rms (the audit's construction: without the high pass half the power is subsonic)"""
    X = np.fft.rfft(rng.standard_normal(n)); f = np.fft.rfftfreq(n, 1 / fs)
    X[1:] /= np.sqrt(f[1:]); X[0] = 0.0; X[f < hp_hz] = 0.0
    y = np.fft.irfft(X, n); return y / rms(y)


def charge_signals(secs=3.0, fs=FS):
    rng = np.random.default_rng(3); n = int(secs * fs); t = np.arange(n) / fs; S = {}
    S["sine1k"] = np.sin(2 * np.pi * 1000 * t)
    S["sine100"] = np.sin(2 * np.pi * 100 * t)
    S["2tone"] = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1300 * t)
    S["10tone"] = sum(np.sin(2 * np.pi * 100 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 11))
    S["40tone"] = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in np.geomspace(50, 8000, 40))
    sq = np.zeros(n)
    for k in range(1, 20, 2): sq += np.sin(2 * np.pi * 1000 * k * t) / k
    S["square"] = sq
    S["pink"] = pink(n, rng, fs)
    return S


def to_peak(x, dbfs): return x / np.abs(x).max() * 10 ** (dbfs / 20)
def to_rms(x, dbfs): return x / rms(x) * 10 ** (dbfs / 20)


def sine(level, f=1000.0, secs=2.5, fs=FS):
    t = np.arange(int(round(secs * fs))) / fs
    return 10 ** (level / 20) * np.sin(2 * np.pi * f * t)


def burst(level, pre=-50.0, pre_s=0.5, burst_s=2.0, post_s=3.5, f=1000.0, fs=FS):
    return protocol.stimulus({"kind": "burst", "pre": pre, "level": level, "pre_s": pre_s, "burst_s": burst_s, "post_s": post_s, "f": f}, fs)[0]


def lockin_gain(y, x, f, fs, last_s):
    """gain of the fundamental over the last last_s seconds (whole periods), dB"""
    per = fs / f; nper = max(1, int(round(last_s * f)))
    n1 = int(round(int(len(y) / per) * per)); n0 = int(round(n1 - nper * per))
    return float(db(abs(protocol.lockin(y, f, fs, n0, n1))) - db(abs(protocol.lockin(x, f, fs, n0, n1))))


def env_lockin(y, x, f, fs, m=None):
    per = fs / f; m = int(len(y) / per) if m is None else m
    out = np.empty(m)
    for k in range(m):
        n0 = int(round(k * per)); n1 = int(round((k + 1) * per))
        out[k] = db(abs(protocol.lockin(y, f, fs, n0, n1))) - db(abs(protocol.lockin(x, f, fs, n0, n1)))
    return out


def env_rms(y, x, f, fs, m):
    per = int(round(fs / f)); out = np.empty(m)
    for k in range(m):
        out[k] = db(rms(y[k * per:(k + 1) * per])) - db(rms(x[k * per:(k + 1) * per]))
    return out


# ------------------------------------------------------------------------------------------------ the two plug-ins
class Both:
    def __init__(self):
        import pa, hvmc_core
        self.P = pa.Ref(); self.pa = pa; self.M = hvmc_core.Model()
        print(f"reference {self.P.version} {self.P.sha256[:12]}; engine fitted={self.M.fitted} layout {self.M.layout_hash}")

    def ref(self, x, fs, s):
        return np.asarray(self.P.run(x, fs, **self.pa.both(**s)), dtype=np.float64)

    def ours(self, x, fs, s):
        return self.M.render(x, fs, s)

    def run(self, x, fs, s):
        return {"ref": self.ref(x, fs, s), "ours": self.ours(x, fs, s)}


def S_(*bases, **kw):
    s = {}
    for b in bases: s.update(b)
    s.update(kw); return s


# ------------------------------------------------------------------------------------------------ capture
def capture(quick=False):
    R = Both(); fs = FS; M = {"meta": {"reference": R.P.version, "sha256": R.P.sha256, "fs": fs, "engine_layout": str(R.M.layout_hash), "quick": quick}}
    t0 = time.time()
    ratios = RATIOS if not quick else ["4:1", "Flood"]
    attacks = ATTACKS if not quick else [0.1, 1.0, 30.0]

    # ---- charge law: every ratio x attack, recover 0.5 s, thr 16, equal peak and equal rms
    print("charge ...", flush=True)
    sig = charge_signals(); ch = {}
    for norm, lvl in (("peak", LEVEL_PEAK), ("rms", LEVEL_RMS)):
        xs = {k: (to_peak(v, lvl) if norm == "peak" else to_rms(v, lvl)) for k, v in sig.items()}
        base = {k: {lab: db(rms(y[0, -int(1.5 * fs):])) for lab, y in R.run(x, fs, S_(BYPASS)).items()} for k, x in xs.items()}
        for r in ratios:
            for a in attacks:
                s = S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a)
                for k, x in xs.items():
                    y = R.run(x, fs, s)
                    ch[f"{norm}|{r}|{a}|{k}"] = {lab: float(base[k][lab] - db(rms(y[lab][0, -int(1.5 * fs):]))) for lab in ("ref", "ours")}
        # recover 0.1 s at the two fast attacks, 4:1 and FLOOD (the audit's K and B settings)
        for r in ("4:1", "Flood"):
            for a in (0.1, 1.0):
                s = S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a, discrete_recover="0.1 s")
                for k, x in xs.items():
                    y = R.run(x, fs, s)
                    ch[f"{norm}|{r}|{a}|rec0.1|{k}"] = {lab: float(base[k][lab] - db(rms(y[lab][0, -int(1.5 * fs):]))) for lab in ("ref", "ours")}
    M["charge"] = ch; print(f"  {len(ch)} cells, {time.time() - t0:.0f} s", flush=True)

    # ---- fine static curve per ratio at threshold 16 (and 12 at FLOOD / 4:1 as the shift check), 1 ms / 0.5 s
    print("fine statics ...", flush=True)
    fine = {}; levels = np.arange(-42.0, 6.01, 0.5)
    for r in ratios:
        for thr in ((16, 12) if r in ("4:1", "Flood") else (16,)):
            for L in levels:
                x = sine(L, secs=2.5); y = R.run(x, fs, S_(DISC, discrete_threshold=thr, discrete_ratio=r))
                fine[f"{r}|{thr}|{L}"] = {lab: lockin_gain(y[lab][0], x, 1000.0, fs, 0.5) for lab in ("ref", "ours")}
    M["fine"] = fine; print(f"  {len(fine)} cells, {time.time() - t0:.0f} s", flush=True)

    # ---- the steady table per ratio
    print("steady table ...", flush=True)
    ar = {}
    for r in ratios:
        for a in attacks:
            for rc in RECOVERS:
                secs = 12.0 if rc == "Dual" else 6.0
                x = sine(-10.0, secs=secs); y = R.run(x, fs, S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a, discrete_recover=rc))
                ar[f"ar|{r}|{a}|{rc}"] = {lab: lockin_gain(y[lab][0], x, 1000.0, fs, 1.0) for lab in ("ref", "ours")}
            for L in (-25.0, 2.0):
                x = sine(L, secs=4.0); y = R.run(x, fs, S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a))
                ar[f"al|{r}|{a}|{L}"] = {lab: lockin_gain(y[lab][0], x, 1000.0, fs, 0.5) for lab in ("ref", "ours")}
            if a in (0.1, 1.0, 30.0):
                for f in (100.0, 5000.0):
                    x = sine(-10.0, f=f, secs=4.0); y = R.run(x, fs, S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a))
                    ar[f"af|{r}|{a}|{f}"] = {lab: lockin_gain(y[lab][0], x, f, fs, 0.5) for lab in ("ref", "ours")}
    M["steady"] = ar; print(f"  {len(ar)} cells, {time.time() - t0:.0f} s", flush=True)

    # ---- bursts per ratio: attack x level, recover 0.5 s; plus recover 0.1 s at -10 dBFS for the two fast attacks
    print("bursts ...", flush=True)
    bu = {}
    cases = [(r, a, L, "0.5 s") for r in ratios for a in ((0.1, 0.5, 1.0, 30.0) if not quick else (0.1, 1.0)) for L in (-30.0, -20.0, -10.0, 0.0)]
    cases += [(r, a, -10.0, "0.1 s") for r in ratios for a in (0.1, 1.0)]
    for r, a, L, rc in cases:
        x = burst(L); y = R.run(x, fs, S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a, discrete_recover=rc))
        m = int(len(x) / (fs / 1000.0))
        bu[f"{r}|{a}|{L}|{rc}"] = {lab: {"lock": np.round(env_lockin(y[lab][0], x, 1000.0, fs, m), 4).tolist(),
                                          "rms": np.round(env_rms(y[lab][0], x, 1000.0, fs, 120), 4).tolist()} for lab in ("ref", "ours")}
    M["burst"] = bu; print(f"  {len(bu)} envelopes, {time.time() - t0:.0f} s", flush=True)

    # ---- the SIDECHAIN FILTER trim and the optical leak
    print("trim ...", flush=True)
    tr = {}
    def g(x, s, f=1000.0, last=1.0, key=None):
        y = R.run(x, fs, s); tr[key] = {lab: lockin_gain(y[lab][0], x, f, fs, last) for lab in ("ref", "ours")}
    for scf in ("Out", "In"):
        for mode in ("Dual Mono", "Stereo"):
            for L in (-60.0, -40.0, -20.0, 0.0):
                g(sine(L, secs=3.0), S_(DISC, discrete_threshold=1, sidechain_filter=scf, mode=mode), key=f"disc_thr1|{scf}|{mode}|L{L}")
        for r in RATIOS:
            g(sine(-30.0, secs=3.0), S_(DISC, discrete_threshold=1, sidechain_filter=scf, discrete_ratio=r), key=f"disc_thr1|{scf}|ratio{r}")
            g(sine(-10.0, secs=4.0), S_(DISC, discrete_threshold=16, sidechain_filter=scf, discrete_ratio=r), key=f"disc_thr16|{scf}|ratio{r}")
        for f in (30.0, 100.0, 1000.0, 10000.0):
            g(sine(-30.0, f=f, secs=3.0), S_(DISC, discrete_threshold=1, sidechain_filter=scf), f=f, key=f"disc_thr1|{scf}|f{f}")
        for gp in (1, 7, 12, 24):
            g(sine(-30.0, secs=3.0), S_(DISC, discrete_threshold=1, sidechain_filter=scf, discrete_gain=gp), key=f"disc_thr1|{scf}|gain{gp}")
        for a in (0.1, 30.0):
            g(sine(-30.0, secs=3.0), S_(DISC, discrete_threshold=1, sidechain_filter=scf, discrete_attack=a, discrete_recover="Dual"), key=f"disc_thr1|{scf}|att{a}_dual")
        g(sine(-30.0, secs=3.0), S_(BYPASS, sidechain_filter=scf), key=f"stages_out|{scf}")
        g(sine(-50.0, secs=3.0), S_(DISC, OPTO, discrete_threshold=1, optical_threshold=1, sidechain_filter=scf), key=f"both_thr1|{scf}")
        g(sine(-50.0, secs=3.0), S_(OPTO, optical_threshold=1, sidechain_filter=scf), key=f"opto_thr1|{scf}")
        g(sine(-50.0, secs=3.0), S_(DISC, discrete_threshold=1, sidechain_filter=scf, transformer="Iron"), key=f"disc_thr1|{scf}|Iron")
        for thr in range(1, 25):
            for L in (-50.0, -70.0):
                g(sine(L, secs=3.0), S_(OPTO, optical_threshold=thr, sidechain_filter=scf), key=f"opto_nogr|{scf}|thr{thr}|L{L}")
        for L in np.arange(-40.0, 4.01, 2.0):
            g(sine(L, secs=2.5), S_(OPTO, optical_threshold=20, sidechain_filter=scf), last=0.5, key=f"opto_static20|{scf}|L{L}")
        for f in (100.0, 1000.0):
            x = sine(-10.0, f=f, secs=3.0); y = R.run(x, fs, S_(OPTO, optical_threshold=20, sidechain_filter=scf))
            for lab in ("ref", "ours"):
                hd, thd = R.pa.harmonics(y[lab][0, -int(1.0 * fs):], f, fs)
                tr[f"opto_harm20|{scf}|f{f}|{lab}"] = {"h": [round(h, 2) if h is not None else None for h in hd[:4]], "gain": lockin_gain(y[lab][0], x, f, fs, 1.0)}
    M["trim"] = tr; print(f"  {len(tr)} cells, {time.time() - t0:.0f} s", flush=True)
    json.dump(M, open(MEAS, "w"))
    print(f"written {MEAS} ({os.path.getsize(MEAS) // 1024} kB) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if "--capture" in ARGS: capture(quick="--quick" in ARGS)
