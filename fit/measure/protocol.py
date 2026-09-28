# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""The measurement protocol: every stimulus, every control setting and every feature extractor, shared by the reference capture
(capture.py, which drives the reference plugin) and by the fitting and tests (which drive our model on the same stimuli and extract the
same features). A protocol item is a dict: {id, group, fs, stim: {...}, set: {...}, feat: {...}}. Settings use the reference's parameter
names with the _1/_2 suffix dropped (both channels get the same value) unless an item gives per-channel values.

Levels: dBFS peak of a sine, 1.0 = 0 dBFS = +14 dBu. Features:
  gain_db   steady-state gain of the fundamental over the last `last_s` seconds, by lock-in (whole periods), left channel (or `ch`)
  env       gain of the fundamental per period (lock-in over one period, hop one period), dB, from the start of the render
  harm      H2..H8 in dBc and the phase of H2/H3 relative to the fundamental, over the last `last_s` seconds
  resp      small-signal magnitude response (dB) at RESP_FREQS from an exponential sweep, by deconvolution
"""
import numpy as np

FS = 48000
RESP_FREQS = [10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 8000, 10000, 12000,
              14000, 15000, 16000, 18000, 20000, 22000]
RATIOS = ["1.2:1", "2:1", "3:1", "4:1", "6:1", "Flood"]
ATTACKS = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
RECOVERS = ["0.1 s", "0.25 s", "0.5 s", "0.8 s", "1.2 s", "Dual"]
CORES = ["Nickel", "Iron", "Steel"]


# ------------------------------------------------------------------------------------------------ stimuli
def stimulus(st, fs=FS):
    """returns a (2, n) float64 array"""
    k = st["kind"]
    if k == "sine":
        n = int(round(st["secs"] * fs)); t = np.arange(n) / fs
        a = 10 ** (st["level"] / 20)
        x = a * np.sin(2 * np.pi * st["f"] * t)
        r = x.copy()
        if "level_r" in st:
            r = 10 ** (st["level_r"] / 20) * np.sin(2 * np.pi * st.get("f_r", st["f"]) * t + st.get("phase_r", 0.0))
        return np.stack([x, r])
    if k == "burst":
        # pre level for pre_s, burst level for burst_s, pre level again for post_s; steps on whole periods of f
        per = fs / st["f"]
        n0 = int(round(round(st["pre_s"] * st["f"]) * per)); n1 = int(round(round(st["burst_s"] * st["f"]) * per))
        n2 = int(round(round(st["post_s"] * st["f"]) * per))
        n = n0 + n1 + n2; t = np.arange(n) / fs
        amp = np.full(n, 10 ** (st["pre"] / 20)); amp[n0:n0 + n1] = 10 ** (st["level"] / 20)
        x = amp * np.sin(2 * np.pi * st["f"] * t)
        return np.stack([x, x])
    if k == "pulses":
        # pre_s at pre level, then `count` pulses of on_s at level and off_s at pre level, then post_s at pre level
        per = fs / st["f"]; q = lambda s: int(round(round(s * st["f"]) * per))
        segs = [(q(st["pre_s"]), st["pre"])]
        for _ in range(st["count"]):
            segs += [(q(st["on_s"]), st["level"]), (q(st["off_s"]), st["pre"])]
        segs.append((q(st["post_s"]), st["pre"]))
        n = sum(s for s, _ in segs); t = np.arange(n) / fs
        amp = np.concatenate([np.full(s, 10 ** (l / 20)) for s, l in segs])
        x = amp * np.sin(2 * np.pi * st["f"] * t)
        return np.stack([x, x])
    if k == "logsweep":
        secs, f1, f2 = st["secs"], st.get("f1", 5.0), st.get("f2", 23000.0)
        n = int(round(secs * fs)); t = np.arange(n) / fs
        K = secs * 2 * np.pi * f1 / np.log(f2 / f1)
        x = 10 ** (st["level"] / 20) * np.sin(K * (np.exp(t * np.log(f2 / f1) / secs) - 1))
        tail = int(fs * 0.5)
        x = np.concatenate([x, np.zeros(tail)])
        return np.stack([x, x])
    if k == "square":
        from scipy.signal import butter, sosfilt
        n = int(round(st["secs"] * fs)); t = np.arange(n) / fs
        sq = np.sign(np.sin(2 * np.pi * st["f"] * t)) * 10 ** (st["level"] / 20)
        sq = sosfilt(butter(8, 18000, fs=fs, output="sos"), sq)
        return np.stack([sq, sq])
    raise ValueError(k)


def fundamental_ref(st, fs=FS):
    """the input sine used as the lock-in reference (unit amplitude, same phase), left channel"""
    n = stimulus(st, fs).shape[1]
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * st["f"] * t)


# ------------------------------------------------------------------------------------------------ features
def lockin(y, f, fs, n0, n1):
    """complex amplitude of the component at f over samples [n0, n1) (whole periods expected)"""
    t = np.arange(n0, n1) / fs
    seg = y[n0:n1]
    return 2.0 * np.mean(seg * np.exp(-2j * np.pi * f * t))


def feature(item, y, fs=FS):
    """y: (2, n) output. Returns the feature value (float, list or dict)."""
    ft, st = item["feat"], item["stim"]
    ch = ft.get("ch", 0)
    if ft["type"] == "gain_db":
        f = st["f"]; per = fs / f
        nper = max(1, int(round(ft.get("last_s", 0.5) * f)))
        n1 = int(round(int(y.shape[1] / per) * per)); n0 = int(round(n1 - nper * per))
        lvl = st["level_r"] if (ch == 1 and "level_r" in st) else st["level"]
        c = lockin(y[ch], st.get("f_r", f) if ch == 1 else f, fs, n0, n1)
        return float(20 * np.log10(abs(c) + 1e-30) - lvl)
    if ft["type"] == "env":
        f = st["f"]; per = fs / f
        hop = max(1, int(round(ft.get("hop_periods", 1))))
        ref = fundamental_ref(st, fs)
        # input amplitude envelope, piecewise constant per period by construction
        x = stimulus(st, fs)[0]
        m = int(y.shape[1] / (per * hop))
        out = []
        for k in range(m):
            n0 = int(round(k * hop * per)); n1 = int(round((k + 1) * hop * per))
            cy = lockin(y[ch], f, fs, n0, n1); cx = lockin(x, f, fs, n0, n1)
            out.append(round(float(20 * np.log10(abs(cy) / (abs(cx) + 1e-30) + 1e-30)), 4))
        return out
    if ft["type"] == "harm":
        f = st["f"]; per = fs / f
        nper = max(4, int(round(ft.get("last_s", 0.5) * f)))
        n1 = int(round(int(y.shape[1] / per) * per)); n0 = int(round(n1 - nper * per))
        c1 = lockin(y[ch], f, fs, n0, n1)
        hs, ph = [], []
        for k in range(2, 9):
            if k * f >= fs / 2 * 0.95:
                hs.append(None); ph.append(None); continue
            ck = lockin(y[ch], k * f, fs, n0, n1)
            hs.append(round(float(20 * np.log10(abs(ck) / (abs(c1) + 1e-30) + 1e-30)), 2))
            ph.append(round(float(np.angle(ck) - k * np.angle(c1)), 3))
        return {"h": hs, "phase": ph, "gain_db": round(float(20 * np.log10(abs(c1) + 1e-30) - st["level"]), 4)}
    if ft["type"] == "resp":
        x = stimulus(st, fs)[0]
        nf = 1 << int(np.ceil(np.log2(len(x) * 2)))
        H = np.fft.rfft(y[ch], nf) / (np.fft.rfft(x, nf) + 1e-30)
        fr = np.fft.rfftfreq(nf, 1 / fs)
        out = []
        for f in RESP_FREQS:
            if f >= fs / 2 * 0.98:
                out.append(None); continue
            band = (fr > f / 1.02) & (fr < f * 1.02)
            out.append(round(float(20 * np.log10(np.median(np.abs(H[band])))), 4))
        return out
    raise ValueError(ft["type"])


# ------------------------------------------------------------------------------------------------ the protocol
def items():
    I = []
    def add(id_, group, stim, set_, feat, fs=FS):
        I.append({"id": id_, "group": group, "fs": fs, "stim": stim, "set": set_, "feat": feat})
    sine = lambda level, f=1000.0, secs=2.0, **kw: {"kind": "sine", "level": level, "f": f, "secs": secs, **kw}
    G = {"type": "gain_db", "last_s": 0.5}
    OPTO = {"optical_bypass": "In"}; DISC = {"discrete_bypass": "In"}

    # 1. switch laws
    for g in range(1, 25):
        add(f"law_opt_gain_{g}", "law", sine(-40), {**OPTO, "optical_threshold": 1, "optical_gain": g}, G)
        add(f"law_disc_gain_{g}", "law", sine(-50), {**DISC, "discrete_threshold": 1, "discrete_gain": g}, G)
    for c in CORES:
        add(f"law_core_gain_{c}", "law", sine(-30), {"transformer": c}, G)
    add("law_bypass_stages_out", "law", sine(-30), {}, G)
    for v in ("In", "Out"):
        add(f"law_hardwire_{v}", "law", sine(-30), {"hardwire_bypass": v}, G)

    # 2. opto static: every threshold, level sweep, 1 kHz; plus frequency dependence at threshold 20
    for thr in range(1, 25):
        for lvl in range(-50, 15, 2):
            add(f"opto_static_t{thr}_{lvl}", "opto_static", sine(lvl, secs=2.5), {**OPTO, "optical_threshold": thr}, G)
    for f in (100.0, 3000.0, 8000.0):
        for lvl in range(-40, 11, 2):
            add(f"opto_static_f{int(f)}_{lvl}", "opto_static_f", sine(lvl, f=f, secs=2.5), {**OPTO, "optical_threshold": 20}, G)

    # 3. opto dynamics: bursts at several depths, burst-length series, repeated bursts
    for lb in (-26, -18, -10, -2):
        add(f"opto_burst_{lb}", "opto_dyn", {"kind": "burst", "pre": -50, "level": lb, "pre_s": 0.5, "burst_s": 2.0, "post_s": 3.0, "f": 1000.0},
            {**OPTO, "optical_threshold": 20}, {"type": "env"})
    for bl in (0.05, 0.2, 1.0, 4.0):
        add(f"opto_blen_{bl}", "opto_dyn", {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": bl, "post_s": 3.0, "f": 1000.0},
            {**OPTO, "optical_threshold": 20}, {"type": "env"})
    add("opto_pulses", "opto_dyn", {"kind": "pulses", "pre": -50, "level": -10, "pre_s": 0.5, "on_s": 0.1, "off_s": 0.15, "count": 8, "post_s": 2.0, "f": 1000.0},
        {**OPTO, "optical_threshold": 20}, {"type": "env"})
    add("opto_burst_thr12", "opto_dyn", {"kind": "burst", "pre": -50, "level": 2, "pre_s": 0.5, "burst_s": 2.0, "post_s": 3.0, "f": 1000.0},
        {**OPTO, "optical_threshold": 12}, {"type": "env"})

    # 4. opto distortion under gain reduction
    for thr in (1, 14, 18, 22):
        for lvl in (-20, -10, 0):
            for f in (100.0, 1000.0, 4000.0):
                add(f"opto_harm_t{thr}_{lvl}_f{int(f)}", "opto_harm", sine(lvl, f=f, secs=3.0), {**OPTO, "optical_threshold": thr},
                    {"type": "harm", "last_s": 1.0})

    # 5. discrete static: every threshold and ratio, level sweep
    for r in RATIOS:
        for thr in range(1, 25):
            for lvl in range(-60, 13, 3):
                add(f"disc_static_{r}_t{thr}_{lvl}", "disc_static", sine(lvl, secs=2.5), {**DISC, "discrete_threshold": thr, "discrete_ratio": r}, G)

    # 6. discrete steady state against attack and recover (detector averaging), at three levels
    for a in ATTACKS:
        for rc in RECOVERS:
            add(f"disc_ar_{a}_{rc}", "disc_ar", sine(-10, secs=12.0), {**DISC, "discrete_threshold": 16, "discrete_attack": a, "discrete_recover": rc},
                {"type": "gain_db", "last_s": 1.0})
    for a in ATTACKS:
        for lvl in (-25, 2):
            add(f"disc_al_{a}_{lvl}", "disc_ar", sine(lvl, secs=4.0), {**DISC, "discrete_threshold": 16, "discrete_attack": a}, G)
    for f in (100.0, 5000.0):
        for a in (0.1, 1.0, 30.0):
            add(f"disc_af_{a}_f{int(f)}", "disc_ar", sine(-10, f=f, secs=4.0), {**DISC, "discrete_threshold": 16, "discrete_attack": a}, G)

    # 7. discrete dynamics
    for a in ATTACKS:
        for rc in RECOVERS:
            post = 9.0 if rc == "Dual" else 3.5
            add(f"disc_burst_{a}_{rc}", "disc_dyn", {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": post, "f": 1000.0},
                {**DISC, "discrete_threshold": 16, "discrete_attack": a, "discrete_recover": rc}, {"type": "env"})
    for lb in (-30, -20, 0):
        add(f"disc_bdepth_{lb}", "disc_dyn", {"kind": "burst", "pre": -50, "level": lb, "pre_s": 0.5, "burst_s": 2.0, "post_s": 3.0, "f": 1000.0},
            {**DISC, "discrete_threshold": 16, "discrete_attack": 5.0, "discrete_recover": "0.25 s"}, {"type": "env"})
    for bl in (0.1, 0.5, 2.0, 8.0):
        add(f"disc_dual_blen_{bl}", "disc_dyn", {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": bl, "post_s": 9.0, "f": 1000.0},
            {**DISC, "discrete_threshold": 16, "discrete_attack": 1.0, "discrete_recover": "Dual"}, {"type": "env"})
    add("disc_dual_pulses", "disc_dyn", {"kind": "pulses", "pre": -50, "level": -10, "pre_s": 0.5, "on_s": 0.2, "off_s": 0.3, "count": 10, "post_s": 6.0, "f": 1000.0},
        {**DISC, "discrete_threshold": 16, "discrete_attack": 1.0, "discrete_recover": "Dual"}, {"type": "env"})
    # the soft ratios' post-burst tails (the ratio switch lowers the release reference and slows the bleed, docs/disc-knee-fix.md)
    for r in ("1.2:1", "2:1", "3:1"):
        for rc in ("0.1 s", "0.5 s", "1.2 s"):
            add(f"disc_tail_{r}_{rc}", "disc_dyn", {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 8.5, "f": 1000.0},
                {**DISC, "discrete_threshold": 16, "discrete_attack": 1.0, "discrete_recover": rc, "discrete_ratio": r}, {"type": "env"})

    # 8. discrete distortion: no GR against level; under GR against attack and frequency
    for lvl in (-30, -20, -10, 0, 6):
        add(f"disc_harm_nogr_{lvl}", "disc_harm", sine(lvl, secs=2.0), {**DISC, "discrete_threshold": 1, "discrete_gain": 7}, {"type": "harm", "last_s": 1.0})
    for a in (0.1, 1.0, 10.0):
        for f in (100.0, 1000.0, 5000.0):
            add(f"disc_harm_gr_{a}_f{int(f)}", "disc_harm", sine(-10, f=f, secs=4.0), {**DISC, "discrete_threshold": 16, "discrete_attack": a},
                {"type": "harm", "last_s": 1.0})

    # 9. discrete makeup against gain reduction (the small interaction)
    for g in range(1, 25):
        add(f"disc_gi_{g}", "disc_gi", sine(-10, secs=2.5), {**DISC, "discrete_threshold": 16, "discrete_gain": g}, G)

    # 10. stereo link: L fixed, R swept; in phase, anti-phase
    for stage, st_ in (("disc", {**DISC, "discrete_threshold": 16}), ("opto", {**OPTO, "optical_threshold": 20})):
        for lr in (-80, -40, -30, -20, -15, -10, -5):
            for ch in (0, 1):
                add(f"link_{stage}_{lr}_ch{ch}", "link", sine(-10, secs=2.5, level_r=lr), {**st_, "mode": "Stereo"}, {"type": "gain_db", "last_s": 0.5, "ch": ch})
        add(f"link_{stage}_anti", "link", sine(-10, secs=2.5, level_r=-10, phase_r=np.pi), {**st_, "mode": "Stereo"}, G)
        add(f"link_{stage}_f1100", "link", sine(-10, secs=2.5, level_r=-10, f_r=1100.0), {**st_, "mode": "Stereo"}, G)

    # 11. sidechain filter: GR against frequency, both stages
    for stage, st_ in (("disc", {**DISC, "discrete_threshold": 16}), ("opto", {**OPTO, "optical_threshold": 20})):
        for f in (20.0, 30.0, 40.0, 60.0, 90.0, 130.0, 200.0, 400.0, 1000.0):
            for sc in ("Out", "In"):
                add(f"scf_{stage}_{int(f)}_{sc}", "scf", sine(-10, f=f, secs=3.0), {**st_, "sidechain_filter": sc}, G)
                add(f"scf_{stage}_{int(f)}_{sc}_lo", "scf", sine(-50, f=f, secs=3.0), {**st_, "sidechain_filter": sc}, G)

    # 12. transformer path: harmonic grid and small-signal response per core
    for c in CORES:
        for f in (20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 320.0, 1000.0, 5000.0):
            for lvl in range(-12, 25, 3):
                add(f"xf_{c}_f{int(f)}_{lvl}", "xfmr", sine(lvl, f=f, secs=2.0), {"transformer": c}, {"type": "harm", "last_s": 0.5})
        add(f"xf_resp_{c}", "xfmr_resp", {"kind": "logsweep", "level": -30, "secs": 8.0}, {"transformer": c}, {"type": "resp"})
        add(f"xf_resp_{c}_hot", "xfmr_resp", {"kind": "logsweep", "level": 0, "secs": 8.0}, {"transformer": c}, {"type": "resp"})
    add("stage_resp_opto", "xfmr_resp", {"kind": "logsweep", "level": -40, "secs": 8.0}, {**OPTO, "optical_threshold": 1}, {"type": "resp"})
    add("stage_resp_disc", "xfmr_resp", {"kind": "logsweep", "level": -40, "secs": 8.0}, {**DISC, "discrete_threshold": 1}, {"type": "resp"})

    # 13. stage distortion without gain reduction, against level (the stages' own amplifiers)
    for lvl in (-30, -20, -10, 0, 6, 12):
        add(f"opto_harm_nogr_{lvl}", "stage_harm", sine(lvl, secs=2.0), {**OPTO, "optical_threshold": 1}, {"type": "harm", "last_s": 1.0})

    # 14. sample-rate invariance
    for fs in (44100, 96000):
        add(f"sr_opto_{fs}", "sr", {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 2.0, "f": 1000.0},
            {**OPTO, "optical_threshold": 20}, {"type": "env"}, fs=fs)
        add(f"sr_disc_{fs}", "sr", {"kind": "burst", "pre": -50, "level": -10, "pre_s": 0.5, "burst_s": 2.0, "post_s": 2.0, "f": 1000.0},
            {**DISC, "discrete_threshold": 16, "discrete_attack": 5.0, "discrete_recover": "0.25 s"}, {"type": "env"}, fs=fs)

    # 15. both stages, and mix
    for lvl in (-30, -20, -10, 0):
        add(f"both_{lvl}", "both", sine(lvl, secs=3.0), {**OPTO, **DISC, "optical_threshold": 18, "discrete_threshold": 14, "discrete_ratio": "2:1"}, G)
    for m in (0.0, 25.0, 50.0, 75.0):
        add(f"mix_{int(m)}", "mix", sine(-10, secs=2.5), {**DISC, "discrete_threshold": 20, "mix": m, "transformer": "Iron"}, G)
    return I
