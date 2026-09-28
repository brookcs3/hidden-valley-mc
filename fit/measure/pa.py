# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Driving the reference plugin (Plugin Alliance "Shadow Hills Mastering Compressor", the licensed copy installed on the measuring
machine) offline through Pedalboard. Black-box input/output measurement only: nothing here reads, disassembles or extracts anything from
the plugin binary. Levels: dBFS peak of a sine in, 1.0 = 0 dBFS; 0 dBFS = +14 dBu (the reference's default meter calibration).
usage: import pa; p = pa.Ref(); y = p.run(x, **settings)"""
import hashlib, os, numpy as np
from pedalboard import load_plugin

REF_PATH = os.environ.get("HVMC_REFERENCE", "/Library/Audio/Plug-Ins/VST3/Shadow Hills Mastering Compressor.vst3")
FS = 48000

BASE = dict(hardwire_bypass="In", bypass=False, mix=100.0, mode="Dual Mono", key_in="Out", bank="A",
            optical_bypass_1="Out", discrete_bypass_1="Out", optical_bypass_2="Out", discrete_bypass_2="Out",
            transformer_1="Nickel", transformer_2="Nickel", meter_select_1="Output", meter_select_2="Output",
            sidechain_filter_1="Out", sidechain_filter_2="Out", sidechain_hp_freq_hz=90.0,
            optical_threshold_1=1, optical_threshold_2=1, optical_gain_1=12, optical_gain_2=12,
            discrete_threshold_1=1, discrete_threshold_2=1, discrete_gain_1=12, discrete_gain_2=12,
            discrete_ratio_1="4:1", discrete_ratio_2="4:1", discrete_attack_1=1.0, discrete_attack_2=1.0,
            discrete_recover_1="0.5 s", discrete_recover_2="0.5 s")

def both(**kw):
    """expand {optical_threshold: 5} to {optical_threshold_1: 5, optical_threshold_2: 5}"""
    out = {}
    for k, v in kw.items():
        if k in BASE or k in ("mode", "mix", "hardwire_bypass", "sidechain_hp_freq_hz"):
            out[k] = v
        else:
            out[k + "_1"] = v; out[k + "_2"] = v
    return out

class Ref:
    def __init__(self, path=REF_PATH):
        self.path = path
        self.p = load_plugin(path)
        binp = os.path.join(path, "Contents", "MacOS")
        b = os.path.join(binp, os.listdir(binp)[0]) if os.path.isdir(binp) else path
        self.sha256 = hashlib.sha256(open(b, "rb").read()).hexdigest() if os.path.isfile(b) else "unknown"
        info = os.path.join(path, "Contents", "Info.plist")
        self.version = "unknown"
        if os.path.isfile(info):
            import plistlib
            self.version = plistlib.load(open(info, "rb")).get("CFBundleShortVersionString", "unknown")

    def run(self, x, fs=FS, **settings):
        for k, v in {**BASE, **settings}.items():
            setattr(self.p, k, v)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x])
        return self.p(x, fs, buffer_size=1024, reset=True)

def sine(dbfs_peak, f=1000.0, secs=2.0, fs=FS):
    t = np.arange(int(round(secs * fs))) / fs
    return (10 ** (dbfs_peak / 20) * np.sin(2 * np.pi * f * t)).astype(np.float64)

def rms_db(a):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(np.asarray(a, dtype=np.float64)))) + 1e-20))

def harmonics(sig, f0, fs=FS, nh=7):
    """[H2..H(nh+1)] in dBc and THD (dB) of a steady sine segment (Blackman window, peak bin search)"""
    sig = np.asarray(sig, dtype=np.float64); n = len(sig)
    S = np.abs(np.fft.rfft(sig * np.blackman(n)))
    def amp(f):
        i = int(round(f * n / fs))
        return S[max(i - 4, 0):i + 5].max() if i + 5 < len(S) else 0.0
    fund = amp(f0); h = [amp(f0 * k) for k in range(2, nh + 2)]
    hd = [float(20 * np.log10(a / fund + 1e-30)) if a > 0 else None for a in h]
    thd = float(20 * np.log10(np.sqrt(sum(a * a for a in h)) / fund + 1e-30))
    return hd, thd
