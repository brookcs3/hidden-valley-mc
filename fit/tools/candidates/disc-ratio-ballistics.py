#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discrete stage: does the detector depend on the RATIO switch, what is the 0.1 ms attack, and what does SIDECHAIN FILTER IN do to the
stage's gain? Programme-audit findings 2, 4 and 5 (research/program-audit.md, sections 3.2, 3.4, 3.5). Results: docs/disc-ballistics.md.

Flags; every result goes to build/disc-ratio-ballistics/:
  --capture [--quick] [--parts charge,fine,steady,burst,trim,thrdep]
             drive the reference (fit/measure/pa.py) and the engine (fit/hvmc_core.py) on one measurement set, write measure.json:
               charge   steady gain reduction on a sine, 100 Hz sine, two-tone, 10-tone, 40-tone, square and pink noise at equal peak
                        (-10 dBFS) and at equal rms (-13 dBFS), every ratio x every attack, recover 0.5 s, threshold 16;
               fine     the static curve of every ratio at threshold 16 in 0.5 dB steps (1 ms / 0.5 s): the ruler that turns a gain
                        into an equivalent steady-sine level (Leq), so the detector's reading can be compared across ratio positions
                        without a model of either the curve or the node;
               steady   the steady table (attack x recover at -10 dBFS, level -25 / +2, 100 Hz / 5 kHz) at every ratio, not only 4:1;
               burst    -50 -> L dBFS steps at 1 kHz per ratio, attack 0.1 / 0.5 / 1 / 30 ms, L = -30 / -20 / -10 / 0, the per-period
                        lock-in gain (the protocol's 'env') and the per-period rms gain (the audit's), recover 0.5 s (and 0.1 s);
               trim     SIDECHAIN FILTER in against out: the discrete stage at threshold 1 against level, frequency, ratio, make-up,
                        stereo mode, attack, transformer; with the stages out; with the optical stage in; at threshold 16 (through the
                        curves); the optical stage's no-compression gain at -50 / -70 dBFS at all 24 threshold positions, its static
                        curve at threshold 20 and its harmonics at -10 dBFS, filter in and out;
               thrdep   the 1 -> 30 ms and 0.1 -> 1 ms sag at a fixed level above the threshold, at four threshold positions (is the
                        attack law referenced to the threshold or to an absolute level?).
  --analyse  the tables (tables.md): the trim and the leak (finding 5); the rulers, the steady table of every ratio in Leq relative to
             4:1, the sag, the charge law in dB of gain and in Leq, the onset fractions, the 30 ms onset rate per dB of gap, the
             release rates and the threshold dependence (finding 2).
  --fit-leq [--modes ..] [--ratios ..] [--nfev n] [--fix k=v,..] [--wlevel L=w,..] [--wburst n] [--tag name] [--verbose]
             per-ratio candidates in the Leq domain: the stage-3 node (node_r with a0 = alpha = 1, sa = sx = inf is
             fit/stages/stage3_discrete.py's detector() exactly) with per-ratio parameters fitted on that ratio's steady table and
             bursts (Leq of the reference against node - d0 of the model), everything else at the stage-3 constants, 4:1 untouched:
               R0  control (the stage-3 detector at every ratio)   R1  attack scale s: ta_r = s ta       R2  Sv_r
               R3  release scale q: tr_r = q tr                    R4  s and q                            R5  depth_r
               R7  conductance intercept a0: f = a0 + xe / Sv      R8  level reference alpha (input / node mix)
               R9  s and Sv                                        R10 a0 and Sv                          R11 s and alpha
               S1  saturating charge current (gap u -> u / (1 + u / Sa), detector-fix mode 34)   S2  saturating level law (Sx)
               S3..S6  Sa with Sx / Sv / s;  R1S, R7S, R9S, R10S: the R modes with Sa
  --fit-fast [--nfev n]
             finding 4: the 0.1 ms and 0.5 ms attack constants at 4:1 through the stage-3 curve, on the fast rows of the steady table,
             the charge law and the fast bursts (stage-3 weights), with the first-period fractions and H3 at 0.1 ms reported.
  --check ratio [--theta s=..,sa=..] [--fast ta0,ta1]
             one candidate through the ratio's curve in dB of gain (the audit's units): the charge law, the steady table and the bursts
             against the reference, with the ratio's own detector offset folded into the curve (what the stage-3 resampling must do).
usage: cd <repo> && python3 -u fit/tools/candidates/disc-ratio-ballistics.py --capture
       python3 -u fit/tools/candidates/disc-ratio-ballistics.py --analyse
       python3 -u fit/tools/candidates/disc-ratio-ballistics.py --fit-leq --modes R0,R1,R7S --tag w3
       python3 -u fit/tools/candidates/disc-ratio-ballistics.py --fit-fast
       python3 -u fit/tools/candidates/disc-ratio-ballistics.py --check Flood --theta a0=8.36,sa=4.38 --fast 0.199e-3,0.721e-3
Needs the licensed reference plug-in for --capture; the other flags read measure.json. Nothing under src/, fit/stages/ or fit/data/ is
touched; the stage-3 constants are read through fit/common.py.

RESULTS (2026-09-28, docs/disc-ballistics.md): finding 5: the trim is +0.638 dB on the discrete stage's input, audio and sidechain, only
with the optical stage out (with it in, the input gain is the +2.14 dB inter-stage gain whether the filter is in or out); the filter does
not remove the optical idle leak, it replaces the threshold-dependent leak by a fixed one (cond0 = 0.0245, -0.21 dB at every position).
Finding 4: ta[0] = 0.199 ms, ta[1] = 0.721 ms (charge law 0.44 -> 0.07 dB rms, fast bursts 0.022 -> 0.011, H3 unchanged). Finding 2: the
ratio switch sets the attack conductance: the soft ratios are slower (s = 2.3 / 1.3 / 1.2 at 1.2:1 / 2:1 / 3:1), 6:1 is 4:1, and FLOOD is a
current-limited limiter attack with a slower bleed (a0 = 6.6, Sa = 5.4 dB, q = 1.13; through the curve: steady 1.03 -> 0.10 dB rms,
charge law 1.29 -> 0.18, bursts 0.134 -> 0.081)."""
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


def env_rms(y, x, f, fs, k0, k1):
    """per-period rms gain (the audit's definition) for periods k0 .. k1-1"""
    per = int(round(fs / f)); out = np.empty(k1 - k0)
    for i, k in enumerate(range(k0, k1)):
        out[i] = db(rms(y[k * per:(k + 1) * per])) - db(rms(x[k * per:(k + 1) * per]))
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
def capture(quick=False, parts=("charge", "fine", "steady", "burst", "trim")):
    R = Both(); fs = FS
    M = json.load(open(MEAS)) if (os.path.exists(MEAS) and set(parts) != {"charge", "fine", "steady", "burst", "trim", "thrdep"}) else {}
    M["meta"] = {"reference": R.P.version, "sha256": R.P.sha256, "fs": fs, "engine_layout": str(R.M.layout_hash), "quick": quick}
    t0 = time.time()
    ratios = RATIOS if not quick else ["4:1", "Flood"]
    attacks = ATTACKS if not quick else [0.1, 1.0, 30.0]

    if "charge" in parts:
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

    if "fine" in parts:
        # ---- fine static curve per ratio at threshold 16 (and 12 at FLOOD / 4:1 as the shift check), 1 ms / 0.5 s
        print("fine statics ...", flush=True)
        fine = {}; levels = np.arange(-42.0, 6.01, 0.5)
        for r in ratios:
            for thr in ((16, 12) if r in ("4:1", "Flood") else (16,)):
                for L in levels:
                    x = sine(L, secs=2.5); y = R.run(x, fs, S_(DISC, discrete_threshold=thr, discrete_ratio=r))
                    fine[f"{r}|{thr}|{L}"] = {lab: lockin_gain(y[lab][0], x, 1000.0, fs, 0.5) for lab in ("ref", "ours")}
        M["fine"] = fine; print(f"  {len(fine)} cells, {time.time() - t0:.0f} s", flush=True)

    if "steady" in parts:
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

    if "burst" in parts:
        # ---- bursts per ratio: attack x level, recover 0.5 s; plus recover 0.1 s at -10 dBFS for the two fast attacks
        print("bursts ...", flush=True)
        bu = {}
        cases = [(r, a, L, "0.5 s") for r in ratios for a in ((0.1, 0.5, 1.0, 30.0) if not quick else (0.1, 1.0)) for L in (-30.0, -20.0, -10.0, 0.0)]
        cases += [(r, a, -10.0, "0.1 s") for r in ratios for a in (0.1, 1.0)]
        for r, a, L, rc in cases:
            x = burst(L); y = R.run(x, fs, S_(DISC, discrete_threshold=16, discrete_ratio=r, discrete_attack=a, discrete_recover=rc))
            m = int(len(x) / (fs / 1000.0))
            bu[f"{r}|{a}|{L}|{rc}"] = {lab: {"lock": np.round(env_lockin(y[lab][0], x, 1000.0, fs, m), 4).tolist(),
                                              "rms": np.round(env_rms(y[lab][0], x, 1000.0, fs, 495, 620), 4).tolist()} for lab in ("ref", "ours")}
        M["burst"] = bu; print(f"  {len(bu)} envelopes, {time.time() - t0:.0f} s", flush=True)

    if "thrdep" in parts:
        # ---- is the attack law referenced to the threshold or to an absolute level? the 30 ms / 1 ms / 0.1 ms sag at a fixed level
        # above each threshold position (the level that is -10 dBFS at position 16), per ratio
        print("threshold dependence ...", flush=True)
        td = {}; Tt = Cal().T
        for r in ("1.2:1", "4:1", "Flood"):
            for thr in (4, 10, 16, 22):
                L = -10.0 + (Tt[thr - 1] - Tt[15])
                for a in (0.1, 1.0, 30.0):
                    for rc in ("0.1 s", "0.5 s"):
                        x = sine(L, secs=6.0); y = R.run(x, fs, S_(DISC, discrete_threshold=thr, discrete_ratio=r, discrete_attack=a, discrete_recover=rc))
                        td[f"{r}|{thr}|{a}|{rc}"] = {"L": L, **{lab: lockin_gain(y[lab][0], x, 1000.0, fs, 1.0) for lab in ("ref", "ours")}}
        M["thrdep"] = td; print(f"  {len(td)} cells, {time.time() - t0:.0f} s", flush=True)

    if "trim" in parts:
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


# ================================================================================================ analysis: rulers and Leq
def load_meas():
    return json.load(open(MEAS))


class Ruler:
    """the static curve of a ratio at threshold 16 (1 ms / 0.5 s) as a ruler: gain -> equivalent steady-sine level. Model-free: a shared
    detector gives the same equivalent level at every ratio for the same signal and setting, whatever the curves are."""
    LEVELS = np.arange(-42.0, 6.01, 0.5)

    def __init__(self, M, lab):
        self.g = {r: np.array([M["fine"][f"{r}|16|{L}"][lab] for L in self.LEVELS]) for r in RATIOS}
        self.g0 = {r: float(self.g[r][0]) for r in RATIOS}
        self.knee = {}
        for r in RATIOS:
            gr = self.g0[r] - self.g[r]
            self.knee[r] = float(self.LEVELS[np.argmax(gr > 0.1)])

    def leq(self, r, gain):
        gr = self.g0[r] - self.g[r]; want = self.g0[r] - gain
        if want <= 0.05: return np.nan
        m = gr > 0.05
        return float(np.interp(want, gr[m], self.LEVELS[m]))

    def leq_arr(self, r, gains):
        return np.array([self.leq(r, g) for g in gains])

    def gain_of(self, r, L):
        """the inverse: gain of the reference's static curve at level L (linear interpolation on the 0.5 dB grid)"""
        return float(np.interp(L, self.LEVELS, self.g[r]))


# ================================================================================================ candidates (Leq domain)
from numba import njit  # noqa: E402


@njit(cache=False)   # not cached: the harness is also loaded by name from scratch scripts, and a cache written under one module name poisons the other
def node_r(a, fs, ta, tr, dual, t2, c2, Tk, depth, sv, alpha, a0, sa, sx):
    """the stage-3 detector (fit/stages/stage3_discrete.py detector(), same equations) with the generalisations the candidates use:
    attack conductance factor f = a0 + (X / sv) / (1 + X / sx), X = alpha * xe + (1 - alpha) * xv (xe = level above rest, xv = node
    above rest), and a saturating charge current: the gap u = e - v enters as u / (1 + u / sa). a0 = 1, alpha = 1, sa = sx = inf is
    the stage-3 law exactly (the exact one-pole step 1 - exp(-rA f) is kept for the linear part). Returns the node (dB) per sample."""
    n = a.shape[0]
    v = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    k2 = 1.0 - np.exp(-1.0 / (t2 * fs)) if dual else 0.0
    floor_db = -100.0; floor_lin = 10.0 ** (floor_db / 20.0)
    rest = Tk - depth
    x = rest; w = rest
    for i in range(n):
        e = 20.0 * np.log10(a[i]) if a[i] > floor_lin else floor_db
        dv = (rest - x) * kR
        if e > x:
            xe = e - rest; xv = x - rest
            if xe < 0.0: xe = 0.0
            if xv < 0.0: xv = 0.0
            X = alpha * xe + (1.0 - alpha) * xv
            f = a0 + (X / sv) / (1.0 + X / sx)
            if f < 1e-3: f = 1e-3
            u = e - x
            us = u / (1.0 + u / sa)   # saturating charge current: u for u << sa, sa for u >> sa
            dv += us * (1.0 - np.exp(-rA * f))
        x += dv
        if dual:
            flow = (x - w) * k2
            x -= flow
            w += flow / c2
        if x < rest + 1e-9: x = rest
        v[i] = x
    return v


class Cal:
    def __init__(self):
        from common import load_cal, MODEL
        cal = load_cal()
        self.ta = [float(v) for v in cal[MODEL.field("d_tatt")]]; self.tr = [float(v) for v in cal[MODEL.field("d_trel")]]
        self.depth = float(cal[MODEL.field("d_rel_depth_db")][0]); self.sv = float(cal[MODEL.field("d_att_sv_db")][0])
        self.T = [float(v) for v in cal[MODEL.field("d_thr_db")]]; self.t2 = float(cal[MODEL.field("d_dual_t2")][0]); self.c2 = float(cal[MODEL.field("d_dual_c2")][0])
        self.curves = np.asarray(cal[MODEL.field("d_curve")]).reshape(6, -1).copy()
        self.gain12 = float(cal[MODEL.field("d_gain_db")][11] + cal[MODEL.fields["x_gain_db"][0]])


# parameter layout of a per-ratio candidate: dict name -> value, applied over the stage-3 constants
MODES = {
    "R0": [],                                   # control: the stage-3 detector for every ratio
    "R1": ["s"],                                # attack scale ta_r = ta * s
    "R2": ["sv"],                               # conductance law scale Sv_r
    "R3": ["q"],                                # release scale tr_r = tr * q
    "R4": ["s", "q"],                           # attack and release scales
    "R5": ["depth"],                            # rest depth per ratio
    "R6": ["s", "q", "sv", "depth"],            # everything (the ceiling of this family)
    "R7": ["a0"],                               # conductance intercept: f = a0 + x / Sv
    "R8": ["alpha"],                            # level reference: f = 1 + (alpha xe + (1 - alpha) xv) / Sv
    "R9": ["s", "sv"],                          # attack scale and conductance scale
    "R10": ["a0", "sv"],                        # intercept and slope of the conductance law
    "R11": ["s", "alpha"],                      # attack scale and level reference
    "S1": ["sa"],                               # shared: saturating charge current (gap), detector-fix mode 34's Sa
    "S2": ["sx"],                               # shared: saturating level law
    "S3": ["sa", "sx"],
    "S4": ["sa", "sv"],                         # gap saturation with the level law's scale refitted
    "S5": ["sx", "sv"],
    "S6": ["sa", "s"],                          # gap saturation with the attack constants rescaled
    "R1S": ["s", "sa"],                         # per-ratio attack scale on top of a gap saturation (sa shared: fixed by --fix)
    "R7S": ["a0", "sa"],
    "R9S": ["s", "sv", "sa"],
    "R10S": ["a0", "sv", "sa"],
    "R7SQ": ["a0", "sa", "q"],                  # FLOOD: intercept, current limit and the release scale
    "R1SQ": ["s", "sa", "q"],
}
P_INIT = {"s": 0.0, "q": 0.0, "sv": 0.0, "depth": None, "a0": 1.0, "alpha": 1.0, "sa": 1.3, "sx": 1.6}   # log10 for s, q, sv, sa, sx
P_LO = {"s": -1.3, "q": -1.0, "sv": -1.0, "depth": -20.0, "a0": 0.05, "alpha": -1.0, "sa": 0.3, "sx": 0.5}
P_HI = {"s": 1.3, "q": 1.0, "sv": 1.5, "depth": 40.0, "a0": 10.0, "alpha": 2.0, "sa": 3.0, "sx": 3.0}
P_XS = {"s": 0.1, "q": 0.1, "sv": 0.1, "depth": 1.0, "a0": 0.3, "alpha": 0.2, "sa": 0.2, "sx": 0.2}
W_BURST = 6000.0   # --wburst n: per-sample burst weight 1 / sqrt(n / 100), n = 6000 is the stage-3 envelope length (6 s at 1 kHz)
W_LEVEL = {}   # --wlevel L=w,...: weight of the bursts at a step level (e.g. 0=0.1 to take the 0 dBFS steps out of a ratio fit)
FIXED = {}   # --fix name=value,...: parameters held at a value (in the natural units: sa, sx in dB) instead of fitted


def theta_of(names, p, C):
    th = {"s": 1.0, "q": 1.0, "sv": C.sv, "depth": C.depth, "a0": 1.0, "alpha": 1.0, "sa": 1e9, "sx": 1e9}
    th.update(FIXED)
    for n, v in zip(names, p):
        if n in ("s", "q", "sa", "sx"): th[n] = 10.0 ** v
        elif n == "sv": th[n] = C.sv * 10.0 ** v
        else: th[n] = v
    return th


class LeqFit:
    """per-ratio fit of a candidate in the Leq domain: model Leq = node - d0_r, reference Leq = ruler(gain)"""
    def __init__(self, M, C, ratio, fast_only=False):
        self.M = M; self.C = C; self.r = ratio; self.ruler = Ruler(M, "ref"); self.knee = self.ruler.knee[ratio]
        self.T = C.T[15]
        # steady cells: (key, attack index, recover index, level, f, Leq_ref)
        self.steady = []
        for ai, a in enumerate(ATTACKS):
            for rci, rc in enumerate(RECOVERS):
                self.steady.append((ai, rci, -10.0, 1000.0, self.ruler.leq(ratio, M["steady"][f"ar|{ratio}|{a}|{rc}"]["ref"])))
            for L in (-25.0, 2.0):
                self.steady.append((ai, 2, L, 1000.0, self.ruler.leq(ratio, M["steady"][f"al|{ratio}|{a}|{L}"]["ref"])))
            if a in (0.1, 1.0, 30.0):
                for f in (100.0, 5000.0):
                    self.steady.append((ai, 2, -10.0, f, self.ruler.leq(ratio, M["steady"][f"af|{ratio}|{a}|{f}"]["ref"])))
        # bursts: per period Leq after the onset (300 periods) and after the end (300 periods)
        self.bursts = []
        for key, item in M["burst"].items():
            r, a, L, rc = key.split("|")
            if r != ratio: continue
            a = float(a); L = float(L)
            e = np.asarray(item["ref"]["lock"])
            on = self.ruler.leq_arr(ratio, e[500:800]); off = self.ruler.leq_arr(ratio, e[2500:2800])
            self.bursts.append((ATTACKS.index(a), RECOVERS.index(rc), L, on, off))
        self.sig_cache = {}
        self.fast_only = fast_only

    def sig(self, kind, L, f=1000.0, secs=4.0):
        k = (kind, L, f, secs)
        if k not in self.sig_cache:
            if kind == "sine":
                self.sig_cache[k] = np.abs(sine(L, f=f, secs=secs))
            else:
                self.sig_cache[k] = np.abs(burst(L, post_s=0.3))
        return self.sig_cache[k]

    def node(self, a, ai, rci, th, level_secs=None):
        dual = rci == 5
        return node_r(a, float(FS), self.C.ta[ai] * th["s"], self.C.tr[rci] * th["q"], dual, self.C.t2, self.C.c2, self.T, th["depth"], th["sv"], th["alpha"], th["a0"], th["sa"], th["sx"])

    def d0(self, th):
        v = self.node(self.sig("sine", -10.0, secs=2.5), 2, 2, th)
        return float(v[-24000:].mean() + 10.0)

    def residual(self, th, detail=False):
        d0 = self.d0(th); out = []; parts = {}
        ss = []
        for ai, rci, L, f, ref in self.steady:
            if not np.isfinite(ref): ss.append(0.0); continue
            secs = 12.0 if rci == 5 else 4.0
            v = self.node(self.sig("sine", L, f=f, secs=secs), ai, rci, th)
            ss.append(float(v[-48000:].mean()) - d0 - ref)
        ss = np.array(ss); out.append(3.0 * ss); parts["steady"] = ss
        gg = []
        for (ai, rci, L, f, ref), d in zip(self.steady, ss):
            gg.append(0.0 if not np.isfinite(ref) else self.ruler.gain_of(self.r, ref + d) - self.ruler.gain_of(self.r, ref))
        parts["steady_gain"] = np.array(gg)
        bb = []
        for ai, rci, L, on, off in self.bursts:
            v = self.node(self.sig("burst", L), ai, rci, th)
            per = 48
            mon = np.array([v[(500 + k) * per:(501 + k) * per].mean() for k in range(300)]) - d0
            moff = np.array([v[(2500 + k) * per:(2501 + k) * per].mean() for k in range(300)]) - d0
            wL = W_LEVEL.get(L, 1.0)
            for m, ref, w0 in ((mon, on, wL), (moff, off, wL)):
                ok = np.isfinite(ref) & (ref > self.knee + 0.5)
                d = np.where(ok, m - ref, 0.0)
                # first three onset periods of the fast attacks: the lock-in of a rising gain is not the curve of the mean node; down-weight
                if m is mon and ai <= 2: d[:3] *= 0.25
                bb.append(w0 * d / np.sqrt(W_BURST / 100.0))
        bb = np.concatenate(bb); out.append(bb); parts["bursts"] = bb
        res = np.concatenate(out)
        return (res, parts, d0) if detail else res


def fit_leq(modes, nfev=60, ratios=None):
    from scipy.optimize import least_squares
    M = load_meas(); C = Cal(); results = {}
    ratios = ratios or RATIOS
    print(f"stage-3 constants: ta {np.round(np.array(C.ta) * 1e3, 3).tolist()} ms, tr {np.round(C.tr, 4).tolist()} s, depth {C.depth:.3f}, Sv {C.sv:.2f}")
    for mode in modes:
        names = [n for n in MODES[mode] if n not in FIXED]; print(f"\n== {mode}: per-ratio parameters {names or 'none'}" + (f" (fixed {FIXED})" if FIXED else "") + (f" (burst level weights {W_LEVEL})" if W_LEVEL else ""))
        tot = {"steady": [], "bursts": []}
        for r in ratios:
            F_ = LeqFit(M, C, r)
            p0 = [C.depth if n == "depth" else P_INIT[n] for n in names]
            if names:
                fn = lambda p: F_.residual(theta_of(names, p, C))
                res = least_squares(fn, p0, bounds=([P_LO[n] for n in names], [P_HI[n] for n in names]), x_scale=[P_XS[n] for n in names], diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
                p = res.x; nf = res.nfev
            else:
                p = []; nf = 0
            th = theta_of(names, p, C)
            resid, parts, d0 = F_.residual(th, detail=True)
            ssr = parts["steady"]; bbr = parts["bursts"]; sgr = parts["steady_gain"]
            tot["steady"].append(ssr); tot["bursts"].append(bbr); tot.setdefault("gain", []).append(sgr)
            ptxt = " ".join(f"{n}={th[n]:.3f}" for n in names)
            print(f"  {r:5s}: {ptxt:44s} d0 {d0:+.2f} | steady Leq rms {np.sqrt(np.mean(ssr ** 2)):.3f} max {np.max(np.abs(ssr)):.2f} (gain {np.sqrt(np.mean(sgr ** 2)):.3f} / {np.max(np.abs(sgr)):.2f}) | bursts rms {np.sqrt(np.mean(bbr ** 2)):.4f} max {np.max(np.abs(bbr)):.2f} | nfev {nf}")
            if "--verbose" in ARGS:
                k = 0
                for ai, a in enumerate(ATTACKS):
                    row = ssr[k:k + 10]; k += 10 if a in (0.1, 1.0, 30.0) else 8
                    print(f"      atk {a:5.1f}: " + " ".join(f"{v:+5.2f}" for v in row))
            results[(mode, r)] = {"theta": th, "steady_rms": float(np.sqrt(np.mean(ssr ** 2))), "steady_gain_rms": float(np.sqrt(np.mean(sgr ** 2))), "bursts_rms": float(np.sqrt(np.mean(bbr ** 2))), "bursts_max": float(np.max(np.abs(bbr)))}
        S = np.concatenate(tot["steady"]); B = np.concatenate(tot["bursts"]); G = np.concatenate(tot["gain"])
        print(f"  all ratios: steady Leq rms {np.sqrt(np.mean(S ** 2)):.3f} max {np.max(np.abs(S)):.2f} (gain {np.sqrt(np.mean(G ** 2)):.3f} / {np.max(np.abs(G)):.2f}) | bursts rms {np.sqrt(np.mean(B ** 2)):.4f} max {np.max(np.abs(B)):.2f}")
        sys.stdout.flush()
    tag = ARGS[ARGS.index("--tag") + 1] if "--tag" in ARGS else "fit_leq"
    json.dump({f"{m}|{r}": v for (m, r), v in results.items()}, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    return results


# ================================================================================================ finding 4: the fastest attacks
class FastFit:
    """the 0.1 ms and 0.5 ms attack constants at 4:1, fitted through the stage-3 curve on: the 0.1 / 0.5 ms rows of the steady table
    (recover x 6, level -25 / +2, 100 Hz / 5 kHz), the charge law (seven signals at equal peak and at equal rms) and the burst
    envelopes (four step levels at 0.5 s, -10 dBFS at 0.1 s), reference gains in dB, stage-3 weights (steady 3, bursts per sample
    over 1 / sqrt(60)). Everything else (Sv, depth, recovers, the curve, T) stays at the stage-3 values."""
    def __init__(self, M, C, ratio="4:1"):
        import stage3_discrete as S3
        self.S3 = S3; self.M = M; self.C = C; self.r = ratio; self.ri = RATIOS.index(ratio); self.T = C.T[15]
        self.curve = C.curves[self.ri]; self.gain12 = C.gain12
        self.sig = charge_signals()
        self.items = []   # (kind, attack index, recover index, level, f, x, ref)
        for a in (0.1, 0.5):
            ai = ATTACKS.index(a)
            for rci, rc in enumerate(RECOVERS):
                self.items.append(("steady", ai, rci, -10.0, 1000.0, np.abs(sine(-10.0, secs=12.0 if rc == "Dual" else 4.0)), M["steady"][f"ar|{ratio}|{a}|{rc}"]["ref"]))
            for L in (-25.0, 2.0):
                self.items.append(("steady", ai, 2, L, 1000.0, np.abs(sine(L, secs=4.0)), M["steady"][f"al|{ratio}|{a}|{L}"]["ref"]))
            if a == 0.1:
                for f in (100.0, 5000.0):
                    self.items.append(("steady", ai, 2, -10.0, f, np.abs(sine(-10.0, f=f, secs=4.0)), M["steady"][f"af|{ratio}|{a}|{f}"]["ref"]))
            for norm, lvl in (("peak", LEVEL_PEAK), ("rms", LEVEL_RMS)):
                for k, s in self.sig.items():
                    x = to_peak(s, lvl) if norm == "peak" else to_rms(s, lvl)
                    self.items.append(("charge", ai, 2, lvl, k, x, M["charge"][f"{norm}|{ratio}|{a}|{k}"]["ref"]))
            for L, rc in ((-30.0, "0.5 s"), (-20.0, "0.5 s"), (-10.0, "0.5 s"), (0.0, "0.5 s"), (-10.0, "0.1 s")):
                if f"{ratio}|{a}|{L}|{rc}" not in M["burst"]: continue   # the 0.1 s recover bursts exist for 0.1 and 1 ms only
                e = np.asarray(M["burst"][f"{ratio}|{a}|{L}|{rc}"]["ref"]["lock"])
                self.items.append(("burst", ai, RECOVERS.index(rc), L, 1000.0, burst(L, post_s=0.3), e))

    def gr(self, v):
        g = np.maximum(self.S3.curve_at(self.curve, v - self.T), 0.0); g[v <= self.T - self.C.depth] = 0.0; return g

    def node(self, a, ai, rci, ta):
        return node_r(a, float(FS), ta[ai], self.C.tr[rci], rci == 5, self.C.t2, self.C.c2, self.T, self.C.depth, self.C.sv, 1.0, 1.0, 1e9, 1e9)

    def residual(self, ta, detail=False):
        out = {"steady": [], "charge": [], "burst": []}
        for kind, ai, rci, L, f, x, ref in self.items:
            if kind == "steady":
                v = self.node(x, ai, rci, ta); g = self.gr(v)
                y = x * 10 ** (-g / 20.0)   # |x| times the gain: the lock-in of the fundamental of a rectified sine is not needed, the mean gain over the last second suffices to 0.01 dB
                n = int(1.0 * FS); gm = 20 * np.log10(np.mean(y[-n:]) / np.mean(x[-n:])) + self.gain12
                out["steady"].append(3.0 * (gm - ref))
            elif kind == "charge":
                v = self.node(np.abs(x), ai, rci, ta); g = self.gr(v); y = x * 10 ** (-g / 20.0); n = int(1.5 * FS)
                grm = db(rms(x[-n:])) - db(rms(y[-n:])) - self.gain12
                out["charge"].append(3.0 * (grm - ref))
            else:
                v = self.node(np.abs(x), ai, rci, ta); g = self.gr(v); y = x * 10 ** (-g / 20.0)
                m = env_lockin(y, x, 1000.0, FS, 800) + self.gain12
                n = 800; w = np.where(ref[:n] < self.gain12 - 0.5, 1.0, 0.25) / np.sqrt(6000.0 / 100.0)
                seg = np.concatenate([np.arange(495, 800), np.arange(2495, 2800)]) if len(ref) > 2800 else np.arange(495, 800)
                # onset (from 5 periods before the step) and the release from the burst end: the env is computed for 800 periods only,
                # so the release is taken from a second render
                d_on = w[495:800] * (m[495:800] - ref[495:800])
                out["burst"].append(d_on)
                xr = x; yr = y
                m2 = env_lockin(yr, xr, 1000.0, FS, 2800)[2495:2800] + self.gain12 if len(ref) > 2800 else np.zeros(0)
                if len(m2): out["burst"].append(w[0] * np.where(ref[2495:2800] < self.gain12 - 0.5, 1.0, 0.25) * (m2 - ref[2495:2800]))
        parts = {k: np.concatenate([np.atleast_1d(o) for o in v]) for k, v in out.items()}
        res = np.concatenate([parts["steady"], parts["charge"], parts["burst"]])
        return (res, parts) if detail else res

    def report(self, ta, label):
        res, parts = self.residual(ta, detail=True)
        print(f"  {label}: ta0 {ta[0] * 1e3:.4f} ms ta1 {ta[1] * 1e3:.4f} ms | steady rms {np.sqrt(np.mean((parts['steady'] / 3) ** 2)):.3f} max {np.max(np.abs(parts['steady'] / 3)):.2f} | charge rms {np.sqrt(np.mean((parts['charge'] / 3) ** 2)):.3f} max {np.max(np.abs(parts['charge'] / 3)):.2f} | bursts rms {np.sqrt(np.mean(parts['burst'] ** 2)):.4f} max {np.max(np.abs(parts['burst'])):.2f}")
        # first-period fractions and the charge table
        for a in (0.1, 0.5):
            ai = ATTACKS.index(a); row = []
            for L in (-30.0, -20.0, -10.0, 0.0):
                e = np.asarray(self.M["burst"][f"{self.r}|{a}|{L}|0.5 s"]["ref"]["lock"]); x = burst(L, post_s=0.3)
                v = self.node(np.abs(x), ai, 2, ta); g = self.gr(v); y = x * 10 ** (-g / 20.0)
                m = env_lockin(y, x, 1000.0, FS, 800) + self.gain12; ss = m[700:800].mean(); ssr = e[2000:2400].mean()
                row.append(f"{L:+3.0f}: {(self.gain12 - m[500]) / (self.gain12 - ss):.3f}/{(self.gain12 - e[500]) / (self.gain12 - ssr):.3f}")
            print(f"     {a} ms first-period fraction model/ref: " + "  ".join(row))
            ch = []
            for k, s in self.sig.items():
                x = to_peak(s, LEVEL_PEAK); v = self.node(np.abs(x), ai, 2, ta); g = self.gr(v); y = x * 10 ** (-g / 20.0); n = int(1.5 * FS)
                ch.append(f"{k}: {db(rms(x[-n:])) - db(rms(y[-n:])) - self.gain12:.2f}/{self.M['charge'][f'peak|{self.r}|{a}|{k}']['ref']:.2f}")
            print(f"     {a} ms charge law at equal peak, model/ref GR: " + "  ".join(ch))
        # H3 at 0.1 ms against the protocol's disc_harm_gr items (reference -38.6 / -58.5 / -70.7 dBc at 100 Hz / 1 kHz / 5 kHz)
        try:
            from common import F as REFF
            h = []
            for f in (100.0, 1000.0, 5000.0):
                x = sine(-10.0, f=f, secs=2.0); v = self.node(np.abs(x), 0, 2, ta); y = x * 10 ** (-self.gr(v) / 20.0)
                t = np.arange(len(x)) / FS; n = int(0.5 * FS); per = int(round(FS / f)); n = (n // per) * per; ys = y[-n:]; ts = t[-n:]
                c1 = abs(np.mean(ys * np.exp(-2j * np.pi * f * ts))); c3 = abs(np.mean(ys * np.exp(-2j * np.pi * 3 * f * ts)))
                h.append(f"{f:.0f} Hz: {20 * np.log10(c3 / c1 + 1e-30):.1f}/{REFF[f'disc_harm_gr_0.1_f{int(f)}']['h'][1]:.1f}")
            print("     H3 at 0.1 ms, model/ref dBc: " + "  ".join(h))
        except Exception as ex:
            print("     (H3 check skipped:", ex, ")")


def fit_fast(nfev=40):
    from scipy.optimize import least_squares
    M = load_meas(); C = Cal(); F_ = FastFit(M, C)
    base = list(C.ta)
    print("== finding 4: the 0.1 ms and 0.5 ms attack constants at 4:1 (everything else at the stage-3 values)")
    F_.report(base, "stage-3 constants")
    def fn(p):
        ta = list(base); ta[0] = 10.0 ** p[0]; ta[1] = 10.0 ** p[1]; return F_.residual(ta)
    res = least_squares(fn, [np.log10(0.15e-3), np.log10(0.6e-3)], bounds=([-6.0, -5.0], [-2.5, -2.0]), x_scale=[0.1, 0.1], diff_step=1e-3, max_nfev=nfev, loss="soft_l1", f_scale=1.0)
    ta = list(base); ta[0] = 10.0 ** res.x[0]; ta[1] = 10.0 ** res.x[1]
    F_.report(ta, f"fitted (nfev {res.nfev})")
    for t0, t1 in ((0.2e-3, 0.65e-3), (0.15e-3, 0.5e-3), (0.25e-3, 0.8e-3)):
        tt = list(base); tt[0] = t0; tt[1] = t1; F_.report(tt, "probe")
    json.dump({"ta0": ta[0], "ta1": ta[1]}, open(os.path.join(OUT, "fit_fast.json"), "w"))
    return ta


# ================================================================================================ analysis tables
def analyse():
    M = load_meas(); T = M["trim"]; out = []
    P = lambda s="": out.append(s)
    ref = lambda k: T[k]["ref"]; our = lambda k: T[k]["ours"]
    P("## Finding 5: SIDECHAIN FILTER in against out\n")
    P("Discrete stage alone, threshold 1 (no gain reduction), 1 kHz unless stated; gain of the fundamental, filter In minus Out, dB.\n")
    P("| item | reference Out | reference In | In - Out (ref) | In - Out (ours) |"); P("|---|---|---|---|---|")
    for mode in ("Dual Mono", "Stereo"):
        for L in (-60.0, -40.0, -20.0, 0.0):
            a, b = f"disc_thr1|Out|{mode}|L{L}", f"disc_thr1|In|{mode}|L{L}"
            P(f"| {mode}, {L:.0f} dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    for r in RATIOS:
        a, b = f"disc_thr1|Out|ratio{r}", f"disc_thr1|In|ratio{r}"; P(f"| ratio {r}, -30 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    for f in (30.0, 100.0, 1000.0, 10000.0):
        a, b = f"disc_thr1|Out|f{f}", f"disc_thr1|In|f{f}"; P(f"| {f:.0f} Hz, -30 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    for g in (1, 7, 12, 24):
        a, b = f"disc_thr1|Out|gain{g}", f"disc_thr1|In|gain{g}"; P(f"| DISCRETE GAIN {g}, -30 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    for a_ in (0.1, 30.0):
        a, b = f"disc_thr1|Out|att{a_}_dual", f"disc_thr1|In|att{a_}_dual"; P(f"| attack {a_} ms, DUAL, -30 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    a, b = "disc_thr1|Out|Iron", "disc_thr1|In|Iron"; P(f"| Iron, -50 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    a, b = "stages_out|Out", "stages_out|In"; P(f"| both stages out, -30 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    a, b = "opto_thr1|Out", "opto_thr1|In"; P(f"| optical alone, threshold 1, -50 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    a, b = "both_thr1|Out", "both_thr1|In"; P(f"| both stages in, thresholds 1, -50 dBFS | {ref(a):+.4f} | {ref(b):+.4f} | {ref(b) - ref(a):+.4f} | {our(b) - our(a):+.4f} |")
    P(f"\nSum of the parts with both stages in, filter out: optical {ref('opto_thr1|Out'):+.4f} + discrete {ref('disc_thr1|Out|Dual Mono|L-60.0'):+.4f} - stages-out {ref('stages_out|Out'):+.4f} = {ref('opto_thr1|Out') + ref('disc_thr1|Out|Dual Mono|L-60.0') - ref('stages_out|Out'):+.4f}; measured {ref('both_thr1|Out'):+.4f}: the inter-stage gain is {ref('both_thr1|Out') - (ref('opto_thr1|Out') + ref('disc_thr1|Out|Dual Mono|L-60.0') - ref('stages_out|Out')):+.3f} dB. With the filter in: parts {ref('opto_thr1|In') + ref('disc_thr1|In|Dual Mono|L-60.0') - ref('stages_out|In'):+.4f}, measured {ref('both_thr1|In'):+.4f}: {ref('both_thr1|In') - (ref('opto_thr1|In') + ref('disc_thr1|In|Dual Mono|L-60.0') - ref('stages_out|In')):+.3f} dB.\n")
    P("Discrete stage at threshold 16, -10 dBFS: net gain change with the filter in (the trim seen through each ratio's curve).\n")
    P("| ratio | ref Out | ref In | net (ref) | net (ours) |"); P("|---|---|---|---|---|")
    for r in RATIOS:
        a, b = f"disc_thr16|Out|ratio{r}", f"disc_thr16|In|ratio{r}"; P(f"| {r} | {ref(a):+.3f} | {ref(b):+.3f} | {ref(b) - ref(a):+.3f} | {our(b) - our(a):+.3f} |")
    P("\nOptical stage alone: no-compression gain (dB) against threshold position at -50 dBFS (identical at -70 dBFS to 0.001 dB), filter Out / In.\n")
    P("| position | ref Out | ref In | ours Out | ours In |"); P("|---|---|---|---|---|")
    for thr in range(1, 25):
        a, b = f"opto_nogr|Out|thr{thr}|L-50.0", f"opto_nogr|In|thr{thr}|L-50.0"; P(f"| {thr} | {ref(a):+.3f} | {ref(b):+.3f} | {our(a):+.3f} | {our(b):+.3f} |")
    P("\nOptical stage, threshold 20, static curve: gain (dB) filter Out / In, and the difference.\n")
    P("| level dBFS | ref Out | ref In | In - Out (ref) | ours Out | ours In |"); P("|---|---|---|---|---|---|")
    for L in np.arange(-40.0, 4.01, 2.0):
        a, b = f"opto_static20|Out|L{L}", f"opto_static20|In|L{L}"; P(f"| {L:.0f} | {ref(a):+.3f} | {ref(b):+.3f} | {ref(b) - ref(a):+.3f} | {our(a):+.3f} | {our(b):+.3f} |")
    P("\nOptical stage, threshold 20, -10 dBFS: H2 / H3 / H4 / H5 (dBc) and gain, filter Out / In.\n")
    for f in (100.0, 1000.0):
        for scf in ("Out", "In"):
            P(f"- {f:.0f} Hz, filter {scf}: reference {T[f'opto_harm20|{scf}|f{f}|ref']['h']} gain {T[f'opto_harm20|{scf}|f{f}|ref']['gain']:+.3f}; ours {T[f'opto_harm20|{scf}|f{f}|ours']['h']} gain {T[f'opto_harm20|{scf}|f{f}|ours']['gain']:+.3f}")

    # ---- finding 2
    R = {lab: Ruler(M, lab) for lab in ("ref", "ours")}
    P("\n## Finding 2: the detector against the ratio switch\n")
    P("Rulers: the reference's static curve of each ratio at threshold 16, 1 ms / 0.5 s, in 0.5 dB steps.\n")
    P("| ratio | GR at -10 dBFS (ref / ours) | slope at -10 dB/dB (ref / ours) | knee, first level with GR > 0.1 dB (ref / ours) |"); P("|---|---|---|---|")
    for r in RATIOS:
        g = R["ref"].g[r]; go = R["ours"].g[r]; i = 64
        P(f"| {r} | {R['ref'].g0[r] - g[i]:.2f} / {R['ours'].g0[r] - go[i]:.2f} | {-(g[i + 1] - g[i - 1]):.3f} / {-(go[i + 1] - go[i - 1]):.3f} | {R['ref'].knee[r]:.1f} / {R['ours'].knee[r]:.1f} |")
    P("\nSteady table of the reference in equivalent sine level Leq (dB), minus the same cell at 4:1; a detector shared by the ratios gives 0 in every cell. Columns: recover 0.1 / 0.25 / 0.5 / 0.8 / 1.2 s / DUAL at -10 dBFS, then level -25 / +2 dBFS at 0.5 s, then 100 Hz / 5 kHz at 0.5 s (0.1, 1, 30 ms only). Ours gives 0.00 in every cell (checked).\n")
    def leq_row(lab, r, a):
        cells = [R[lab].leq(r, M["steady"][f"ar|{r}|{a}|{rc}"][lab]) for rc in RECOVERS] + [R[lab].leq(r, M["steady"][f"al|{r}|{a}|{L}"][lab]) for L in (-25.0, 2.0)]
        cells += [R[lab].leq(r, M["steady"][f"af|{r}|{a}|{f}"][lab]) for f in (100.0, 5000.0)] if a in (0.1, 1.0, 30.0) else [np.nan, np.nan]
        return np.array(cells)
    base = {a: leq_row("ref", "4:1", a) for a in ATTACKS}
    P("| ratio | attack | 0.1 s | 0.25 s | 0.5 s | 0.8 s | 1.2 s | DUAL | -25 dBFS | +2 dBFS | 100 Hz | 5 kHz |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in RATIOS:
        for a in ATTACKS:
            row = leq_row("ref", r, a) - (0.0 if r == "4:1" else base[a])
            P(f"| {r if r != '4:1' else '4:1 (absolute)'} | {a} ms | " + " | ".join("" if np.isnan(v) else f"{v:+.2f}" for v in row) + " |")
    P("\nSag of the steady gain reduction (dB of gain) from 1 ms to 30 ms attack and from 0.1 ms to 1 ms, -10 dBFS, threshold 16, reference / ours.\n")
    P("| ratio | 1 -> 30 ms at 0.1 s | at 0.5 s | at 1.2 s | 0.1 -> 1 ms at 0.5 s |"); P("|---|---|---|---|---|")
    for r in RATIOS:
        g = lambda a, rc, lab: M["steady"][f"ar|{r}|{a}|{rc}"][lab]
        P(f"| {r} | " + " | ".join(f"{g(1.0, rc, 'ref') - g(30.0, rc, 'ref'):.2f} / {g(1.0, rc, 'ours') - g(30.0, rc, 'ours'):.2f}" for rc in ("0.1 s", "0.5 s", "1.2 s")) + f" | {g(0.1, '0.5 s', 'ref') - g(1.0, '0.5 s', 'ref'):.2f} / {g(0.1, '0.5 s', 'ours') - g(1.0, '0.5 s', 'ours'):.2f} |")
    P("\nCharge law: steady gain reduction on the seven signals at equal peak (-10 dBFS), ours minus reference in dB of gain reduction, recover 0.5 s, threshold 16 (negative: ours compresses less). Signals: 1 kHz sine, 100 Hz sine, two-tone 1 + 1.3 kHz, 10-tone 100 Hz harmonic series, 40-tone log-spaced 50 Hz to 8 kHz, band-limited 1 kHz square, pink noise 20 Hz to 24 kHz.\n")
    sigs = ["sine1k", "sine100", "2tone", "10tone", "40tone", "square", "pink"]
    P("| ratio | attack | " + " | ".join(sigs) + " |"); P("|---|---|" + "---|" * len(sigs))
    for r in RATIOS:
        for a in ATTACKS:
            P(f"| {r} | {a} ms | " + " | ".join(f"{M['charge'][f'peak|{r}|{a}|{s}']['ours'] - M['charge'][f'peak|{r}|{a}|{s}']['ref']:+.2f}" for s in sigs) + " |")
    P("\nThe same in the reference's own units: Leq(signal) - Leq(1 kHz sine), dB, i.e. how much higher or lower than a sine of the same peak the detector reads each signal; reference, then ours in brackets where they differ by more than 0.1 dB.\n")
    P("| ratio | attack | " + " | ".join(sigs[1:]) + " |"); P("|---|---|" + "---|" * (len(sigs) - 1))
    for r in RATIOS:
        for a in (0.1, 1.0, 30.0):
            cells = []
            for s in sigs[1:]:
                vr = R["ref"].leq(r, R["ref"].g0[r] - M["charge"][f"peak|{r}|{a}|{s}"]["ref"]) - R["ref"].leq(r, R["ref"].g0[r] - M["charge"][f"peak|{r}|{a}|sine1k"]["ref"])
                vo = R["ours"].leq(r, R["ours"].g0[r] - M["charge"][f"peak|{r}|{a}|{s}"]["ours"]) - R["ours"].leq(r, R["ours"].g0[r] - M["charge"][f"peak|{r}|{a}|sine1k"]["ours"])
                cells.append(f"{vr:+.2f}" + (f" ({vo:+.2f})" if abs(vo - vr) > 0.1 else ""))
            P(f"| {r} | {a} ms | " + " | ".join(cells) + " |")
    P("\nBurst onsets: fraction of the settled gain reduction reached in the first period of a -50 -> L dBFS step at 1 kHz (per-period lock-in gain; the audit's per-period rms figure is 0.05 to 0.09 lower), reference / ours.\n")
    P("| ratio | attack | -30 dBFS | -20 | -10 | 0 |"); P("|---|---|---|---|---|---|")
    for r in RATIOS:
        for a in (0.1, 0.5, 1.0):
            cells = []
            for L in (-30.0, -20.0, -10.0, 0.0):
                it = M["burst"][f"{r}|{a}|{L}|0.5 s"]; fr = []
                for lab in ("ref", "ours"):
                    e = np.asarray(it[lab]["lock"]); g0 = R[lab].g0[r]; ss = e[2000:2400].mean(); fr.append((g0 - e[500]) / (g0 - ss))
                cells.append(f"{fr[0]:.2f} / {fr[1]:.2f}")
            P(f"| {r} | {a} ms | " + " | ".join(cells) + " |")
    P("\nSlow attack (30 ms / 0.5 s): the reference's node rise between periods 10 and 50 after the step, in dB of Leq per period, and divided by the mean gap to the settled level (rate per dB of gap); a dB one-pole with a level-independent conductance gives a constant rate per gap across levels.\n")
    P("| ratio | -30 dBFS rate / per gap | -20 | -10 | 0 |"); P("|---|---|---|---|---|")
    for r in RATIOS:
        cells = []
        for L in (-30.0, -20.0, -10.0, 0.0):
            e = np.asarray(M["burst"][f"{r}|30.0|{L}|0.5 s"]["ref"]["lock"]); lq = R["ref"].leq_arr(r, e[500:800]); ss = R["ref"].leq(r, e[2000:2400].mean())
            rate = (lq[50] - lq[10]) / 40.0; gap = ss - 0.5 * (lq[10] + lq[50])
            cells.append(f"{rate:.3f} / {rate / gap:.4f}" if np.isfinite(rate) and gap > 0 else "-")
        P(f"| {r} | " + " | ".join(cells) + " |")
    P("\nRelease after the -10 dBFS burst, 1 ms attack: the reference's Leq fall per period between periods 5 and 40, and 40 and 120, after the burst end; recover 0.5 s and 0.1 s. Ours gives the same figure at every ratio (0.079 / 0.066 and 0.229 / 0.128).\n")
    P("| ratio | 0.5 s: 5-40 | 40-120 | 0.1 s: 5-40 | 40-120 |"); P("|---|---|---|---|---|")
    for r in RATIOS:
        cells = []
        for rc in ("0.5 s", "0.1 s"):
            e = np.asarray(M["burst"][f"{r}|1.0|-10.0|{rc}"]["ref"]["lock"]); lq = R["ref"].leq_arr(r, e[2500:2800])
            cells += [f"{-(lq[40] - lq[5]) / 35.0:.4f}", f"{-(lq[120] - lq[40]) / 80.0:.4f}"]
        P(f"| {r} | " + " | ".join(cells) + " |")
    if "thrdep" in M:
        P("\nThreshold dependence: sag of the steady gain (dB) from 1 ms to 30 ms and from 0.1 ms to 1 ms at a level 27.7 dB above the threshold (the level that is -10 dBFS at position 16), reference / ours. Position 4 puts the tone at +22 dBFS, where the reference's transformer core clips; the other three positions are the test.\n")
        P("| ratio | recover | position (level) | gain at 1 ms ref / ours | 1 -> 30 ms ref / ours | 0.1 -> 1 ms ref / ours |"); P("|---|---|---|---|---|---|")
        for r in ("1.2:1", "4:1", "Flood"):
            for rc in ("0.1 s", "0.5 s"):
                for thr in (4, 10, 16, 22):
                    g = {a: M["thrdep"][f"{r}|{thr}|{a}|{rc}"] for a in (0.1, 1.0, 30.0)}
                    P(f"| {r} | {rc} | {thr} ({g[1.0]['L']:+.1f} dBFS) | {g[1.0]['ref']:+.3f} / {g[1.0]['ours']:+.3f} | {g[1.0]['ref'] - g[30.0]['ref']:.3f} / {g[1.0]['ours'] - g[30.0]['ours']:.3f} | {g[0.1]['ref'] - g[1.0]['ref']:.3f} / {g[0.1]['ours'] - g[1.0]['ours']:.3f} |")
    txt = "\n".join(out); open(os.path.join(OUT, "tables.md"), "w").write(txt); print(txt)


# ================================================================================================ check: a candidate through the curve
def check_ratio(ratio, th, ta_override=None, label=""):
    """the model through the ratio's stage-3 curve with a per-ratio theta: the charge law at equal peak (all attacks), the steady sag
    table and the burst onset fractions, model against reference, in dB of gain (the audit's units)"""
    import stage3_discrete as S3
    M = load_meas(); C = Cal(); ri = RATIOS.index(ratio); cur = C.curves[ri]; T = C.T[15]
    ta = list(C.ta) if ta_override is None else list(ta_override)
    def node(a, ai, rci):
        return node_r(a, float(FS), ta[ai] * th["s"], C.tr[rci] * th["q"], rci == 5, C.t2, C.c2, T, th["depth"], th["sv"], th["alpha"], th["a0"], th["sa"], th["sx"])
    # the ratio's curve was resampled onto the node domain with the 4:1 detector's offset at the capture setting (1 ms / 0.5 s); a
    # per-ratio detector has its own offset there, so the curve is read at v - T - (d0_r - d0_4:1): the stage-3 resampling redone per ratio
    x20 = np.abs(sine(-20.0, secs=2.5)); n = int(0.5 * FS)
    d0_r = float(node(x20, 2, 2)[-n:].mean() + 20.0)
    th41 = theta_of([], [], C); d0_4 = float(node_r(x20, float(FS), C.ta[2], C.tr[2], False, C.t2, C.c2, T, C.depth, C.sv, 1.0, 1.0, 1e9, 1e9)[-n:].mean() + 20.0)
    dd0 = d0_r - d0_4
    def gr(v):
        g = np.maximum(S3.curve_at(cur, v - T - dd0), 0.0); g[v <= T - th["depth"]] = 0.0; return g
    print(f"== check {ratio} {label}: theta " + " ".join(f"{k}={v:.3f}" for k, v in th.items() if v not in (1e9,)) + f" ta {np.round(np.array(ta) * 1e3, 3).tolist()} ms; d0 at the capture setting {d0_r:+.3f} (4:1 detector {d0_4:+.3f}): the curve is read {dd0:+.3f} dB lower")
    sig = charge_signals(); n = int(1.5 * FS); errs = []
    print("   charge law at equal peak, ours - ref (dB of gain): attack | " + " ".join(f"{k:>7s}" for k in sig))
    for a in ATTACKS:
        ai = ATTACKS.index(a); row = []
        for k, s in sig.items():
            x = to_peak(s, LEVEL_PEAK); v = node(np.abs(x), ai, 2); y = x * 10 ** (-gr(v) / 20.0)
            gm = db(rms(x[-n:])) - db(rms(y[-n:])) - C.gain12; d = gm - M["charge"][f"peak|{ratio}|{a}|{k}"]["ref"]; row.append(d); errs.append(d)
        print(f"     {a:5.1f} ms | " + " ".join(f"{d:+7.2f}" for d in row))
    print(f"   charge rms {np.sqrt(np.mean(np.square(errs))):.3f} max {np.max(np.abs(errs)):.2f}")
    print("   steady table, ours - ref (dB of gain): attack | recover 0.1 0.25 0.5 0.8 1.2 Dual | level -25 +2 | 100 Hz 5 kHz")
    errs = []
    for a in ATTACKS:
        ai = ATTACKS.index(a); row = []
        for rci, rc in enumerate(RECOVERS):
            x = np.abs(sine(-10.0, secs=12.0 if rc == "Dual" else 4.0)); v = node(x, ai, rci); g = gr(v); y = x * 10 ** (-g / 20.0); m = int(FS)
            row.append(20 * np.log10(np.mean(y[-m:]) / np.mean(x[-m:])) + C.gain12 - M["steady"][f"ar|{ratio}|{a}|{rc}"]["ref"])
        for L in (-25.0, 2.0):
            x = np.abs(sine(L, secs=4.0)); v = node(x, ai, 2); g = gr(v); y = x * 10 ** (-g / 20.0); m = int(FS)
            row.append(20 * np.log10(np.mean(y[-m:]) / np.mean(x[-m:])) + C.gain12 - M["steady"][f"al|{ratio}|{a}|{L}"]["ref"])
        if a in (0.1, 1.0, 30.0):
            for f in (100.0, 5000.0):
                x = np.abs(sine(-10.0, f=f, secs=4.0)); v = node(x, ai, 2); g = gr(v); y = x * 10 ** (-g / 20.0); m = int(FS)
                row.append(20 * np.log10(np.mean(y[-m:]) / np.mean(x[-m:])) + C.gain12 - M["steady"][f"af|{ratio}|{a}|{f}"]["ref"])
        errs += row
        print(f"     {a:5.1f} ms | " + " ".join(f"{d:+6.2f}" for d in row[:6]) + " | " + " ".join(f"{d:+6.2f}" for d in row[6:8]) + " | " + " ".join(f"{d:+6.2f}" for d in row[8:]))
    print(f"   steady rms {np.sqrt(np.mean(np.square(errs))):.3f} max {np.max(np.abs(errs)):.2f}")
    print("   bursts (per-period lock-in gain, stage-3 weights): attack, level | rms max | first-period fraction ours/ref | settled ours/ref")
    allb = []
    for a in (0.1, 0.5, 1.0, 30.0):
        ai = ATTACKS.index(a)
        for L in (-30.0, -20.0, -10.0, 0.0):
            e = np.asarray(M["burst"][f"{ratio}|{a}|{L}|0.5 s"]["ref"]["lock"]); x = burst(L); v = node(np.abs(x), ai, 2); y = x * 10 ** (-gr(v) / 20.0)
            m = env_lockin(y, x, 1000.0, FS) + C.gain12; k = min(len(m), len(e)); w = np.where(e[:k] < C.gain12 - 0.5, 1.0, 0.25) / np.sqrt(k / 100.0)
            d = w * (m[:k] - e[:k]); allb.append(d)
            print(f"     {a:5.1f} ms {L:+3.0f} dBFS | {np.sqrt(np.mean(d ** 2)):.4f} {np.max(np.abs(d)):.2f} | {(C.gain12 - m[500]) / (C.gain12 - m[2000:2400].mean()):.3f}/{(C.gain12 - e[500]) / (C.gain12 - e[2000:2400].mean()):.3f} | {m[2000:2400].mean():+.2f}/{e[2000:2400].mean():+.2f}")
    allb = np.concatenate(allb); print(f"   bursts weighted rms {np.sqrt(np.mean(allb ** 2)):.4f} max {np.max(np.abs(allb)):.2f}")


if __name__ == "__main__":
    if "--fix" in ARGS:
        for kv in ARGS[ARGS.index("--fix") + 1].split(","):
            k, v = kv.split("="); FIXED[k] = float(v)
    if "--wburst" in ARGS:
        W_BURST = float(ARGS[ARGS.index("--wburst") + 1])
    if "--wlevel" in ARGS:
        for kv in ARGS[ARGS.index("--wlevel") + 1].split(","):
            k, v = kv.split("="); W_LEVEL[float(k)] = float(v)
    if "--check" in ARGS:
        # --check ratio [--theta s=..,q=..,sv=..,a0=..,sa=..] [--fast ta0,ta1] : one candidate through the curve, in dB of gain
        C_ = Cal(); th = theta_of([], [], C_)
        if "--theta" in ARGS:
            for kv in ARGS[ARGS.index("--theta") + 1].split(","):
                k, v = kv.split("="); th[k] = float(v)
        tao = None
        if "--fast" in ARGS:
            tao = list(C_.ta); tao[0], tao[1] = [float(v) for v in ARGS[ARGS.index("--fast") + 1].split(",")]
        check_ratio(ARGS[ARGS.index("--check") + 1], th, tao, label=ARGS[ARGS.index("--theta") + 1] if "--theta" in ARGS else "stage-3")
    if "--analyse" in ARGS:
        analyse()
    if "--fit-fast" in ARGS:
        fit_fast(nfev=int(ARGS[ARGS.index("--nfev") + 1]) if "--nfev" in ARGS else 40)
    if "--fit-leq" in ARGS:
        modes = ARGS[ARGS.index("--modes") + 1].split(",") if "--modes" in ARGS else ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
        nfev = int(ARGS[ARGS.index("--nfev") + 1]) if "--nfev" in ARGS else 60
        ratios = ARGS[ARGS.index("--ratios") + 1].split(",") if "--ratios" in ARGS else None
        fit_leq(modes, nfev=nfev, ratios=ratios)
    if "--capture" in ARGS:
        parts = tuple(ARGS[ARGS.index("--parts") + 1].split(",")) if "--parts" in ARGS else ("charge", "fine", "steady", "burst", "trim", "thrdep")
        capture(quick="--quick" in ARGS, parts=parts)
