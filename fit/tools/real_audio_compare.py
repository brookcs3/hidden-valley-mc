# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Real programme through both: a mix through our plugin (REFERENCE profile, STANDARD) and through the licensed reference plug-in
at the same panel settings, compared as level, short-term envelope of the gain difference, null depth and band-wise spectrum.
Also, when the folder holds level-matched renders made by hand, the null depth of ours against those after level matching.
usage: python3 fit/tools/real_audio_compare.py [mix.wav] [our .vst3]   (defaults: ~/shadow/listen_test/01_dry_house_mix.wav,
build/macos/bin/HiddenValleyMC.vst3). Needs the reference plug-in (fit/measure/pa.py); not part of the test suite."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "fit", "measure")); sys.path.insert(0, os.path.join(ROOT, "tests"))
from pedalboard.io import AudioFile
import pa, hvmc_pb

MIX = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/shadow/listen_test/01_dry_house_mix.wav")
OURS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build", "macos", "bin", "HiddenValleyMC.vst3")
SETTINGS = {
    "gentle: opto 18 / discrete 14 2:1 5 ms 0.5 s, filter in, Iron": dict(ref=dict(optical_bypass="In", optical_threshold=18, optical_gain=12, discrete_bypass="In", discrete_threshold=14, discrete_ratio="2:1", discrete_attack=5.0, discrete_recover="0.5 s", discrete_gain=9, sidechain_filter="In", transformer="Iron", mode="Stereo"),
        ours=dict(l_optical="IN", l_optical_threshold="18", l_optical_gain="12", l_discrete="IN", l_discrete_threshold="14", l_discrete_ratio="2:1", l_discrete_attack_ms="5", l_discrete_recover_s="0.5", l_discrete_gain="9", l_sidechain_filter="IN", l_transformer="IRON", stereo="STEREO")),
    "squash: opto 22 / discrete 20 FLOOD 0.1 ms 0.1 s, Iron": dict(ref=dict(optical_bypass="In", optical_threshold=22, optical_gain=12, discrete_bypass="In", discrete_threshold=20, discrete_ratio="Flood", discrete_attack=0.1, discrete_recover="0.1 s", discrete_gain=12, sidechain_filter="Out", transformer="Iron", mode="Stereo"),
        ours=dict(l_optical="IN", l_optical_threshold="22", l_optical_gain="12", l_discrete="IN", l_discrete_threshold="20", l_discrete_ratio="FLOOD", l_discrete_attack_ms="0.1", l_discrete_recover_s="0.1", l_discrete_gain="12", l_sidechain_filter="OUT", l_transformer="IRON", stereo="STEREO")),
    "4:1 DUAL: discrete 16 4:1 1 ms DUAL, opto out, Nickel": dict(ref=dict(optical_bypass="Out", discrete_bypass="In", discrete_threshold=16, discrete_ratio="4:1", discrete_attack=1.0, discrete_recover="Dual", discrete_gain=12, sidechain_filter="Out", transformer="Nickel", mode="Stereo"),
        ours=dict(l_optical="OUT", l_discrete="IN", l_discrete_threshold="16", l_discrete_ratio="4:1", l_discrete_attack_ms="1", l_discrete_recover_s="DUAL", l_discrete_gain="12", l_sidechain_filter="OUT", l_transformer="NICKEL", stereo="STEREO")),
    "transformers only, Steel": dict(ref=dict(optical_bypass="Out", discrete_bypass="Out", transformer="Steel", mode="Stereo"),
        ours=dict(l_optical="OUT", l_discrete="OUT", l_transformer="STEEL", stereo="STEREO")),
}

def db(x): return 20 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))
def rms(x): return float(np.sqrt(np.mean(np.square(x))))
def env(x, fs, ms=100):
    n = int(fs * ms / 1000); m = len(x) // n
    return np.sqrt(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1))
def bands(x, fs):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; f = np.fft.rfftfreq(len(x), 1 / fs)
    edges = [20, 60, 150, 400, 1000, 2500, 6000, 12000, 20000]
    return np.array([10 * np.log10(X[(f >= a) & (f < b)].sum() + 1e-30) for a, b in zip(edges[:-1], edges[1:])]), edges

with AudioFile(MIX) as f:
    x, fs = f.read(f.frames), f.samplerate
x = x.astype(np.float32); fs = int(fs)
P = pa.Ref()
ours = hvmc_pb.load(OURS)
print(f"mix {os.path.basename(MIX)}: {x.shape[1] / fs:.1f} s at {fs} Hz, peak {db(np.abs(x).max()):.1f} dBFS, rms {db(rms(x)):.1f} dBFS")
print(f"{'setting':62s} {'level diff':>10s} {'env rms':>8s} {'env max':>8s} {'null':>7s}  band diff dB (20-60, 60-150, 150-400, 400-1k, 1-2.5k, 2.5-6k, 6-12k, 12-20k)")
results = {}
for name, st in SETTINGS.items():
    for k, v in {**pa.BASE, **pa.both(**{kk: vv for kk, vv in st["ref"].items() if kk != "mode"}), "mode": st["ref"]["mode"]}.items():
        setattr(P.p, k, v)
    yr = P.p(x, fs, buffer_size=1024, reset=True)
    for k in st["ours"]:
        pass
    for k, v in st["ours"].items():
        vv = list(ours.parameters[k].valid_values)
        match = [u for u in vv if str(u) == str(v) or (isinstance(u, (int, float)) and abs(float(u) - float(v)) < 1e-9)]
        setattr(ours, k, match[0] if match else v)
    ours.profile = "REFERENCE"; ours.quality = "STANDARD"
    yo = ours(x, fs, reset=True)
    n = min(yr.shape[1], yo.shape[1]); yr, yo = yr[:, :n].astype(np.float64), yo[:, :n].astype(np.float64)
    lvl = db(rms(yo)) - db(rms(yr))
    e = db(env(yo[0], fs)) - db(env(yr[0], fs))
    null = db(rms(yo - yr)) - db(rms(yr))
    bo, edges = bands(yo[0], fs); br, _ = bands(yr[0], fs)
    results[name] = (yo, yr)
    print(f"{name:62s} {lvl:+10.2f} {rms(e):8.3f} {np.abs(e).max():8.2f} {null:7.1f}  {np.round(bo - br, 2)}")
    print(f"{'':62s} gain reduction (ref) rms {db(rms(x)) - db(rms(yr)):.1f} dB; ours {db(rms(x)) - db(rms(yo)):.1f} dB")
# the hand-made level-matched renders, if present
folder = os.path.dirname(MIX)
for fname, key in (("02_squash_opto_plus_flood_iron_LEVELMATCHED.wav", "squash: opto 22 / discrete 20 FLOOD 0.1 ms 0.1 s, Iron"), ("03_squash_4to1_dual_nickel_LEVELMATCHED.wav", "4:1 DUAL: discrete 16 4:1 1 ms DUAL, opto out, Nickel")):
    p = os.path.join(folder, fname)
    if not os.path.exists(p): continue
    with AudioFile(p) as f: y = f.read(f.frames).astype(np.float64)
    yo, yr = results[key]; n = min(y.shape[1], yo.shape[1]); y = y[:, :n]
    for lab, z in (("ours", yo[:, :n]), ("reference (our settings)", yr[:, :n])):
        g = rms(y) / rms(z); zz = z * g
        print(f"{fname} vs {lab:26s}: level-matched null {db(rms(zz - y)) - db(rms(y)):6.1f} dB, envelope diff rms {rms(db(env(zz[0], fs)) - db(env(y[0], fs))):.2f} dB (settings of that file are unknown; a shallow null means different settings, not a different model)")
