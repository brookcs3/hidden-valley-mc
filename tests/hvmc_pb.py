# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Shared by the Pedalboard tests: where the built bundle is, how to load it (registering a macOS Audio Unit built outside the plug-in
folders), and how to set the plugin's stepped parameters by position or from the reference's setting names used by the measurement
protocol (fit/measure/protocol.py). Pedalboard exposes every parameter under its lower-cased name (L_optical_threshold becomes
l_optical_threshold) and, for each, the panel legends as valid_values in position order: numeric legends as floats (1.0 .. 24.0, 0.1 ms),
OFF/ON as False/True, the rest as strings ("4:1", "DUAL", "IRON"). Pedalboard's VST3 host adds one parameter of its own,
latency_frames, from DPF's latency reporting; the Audio Unit has no such parameter."""
import os, sys
import pedalboard

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import au_register  # noqa: E402

NAME = "HiddenValleyMC"
HOST_PARAMS = {"latency_frames"}   # parameters a host adds that are not the plugin's own
RATIOS = ["1.2:1", "2:1", "3:1", "4:1", "6:1", "Flood"]
ATTACKS = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
RECOVERS = ["0.1 s", "0.25 s", "0.5 s", "0.8 s", "1.2 s", "Dual"]
CORES = ["Nickel", "Iron", "Steel", "Gold", "Uranium", "Germanium", "Plutonium"]
# the reference's settings every protocol item starts from (fit/measure/pa.py BASE), in the protocol's names
REF_BASE = dict(hardwire_bypass="In", mix=100.0, mode="Dual Mono", optical_bypass="Out", discrete_bypass="Out", transformer="Nickel",
                sidechain_filter="Out", optical_threshold=1, optical_gain=12, discrete_threshold=1, discrete_gain=12,
                discrete_ratio="4:1", discrete_attack=1.0, discrete_recover="0.5 s", meter_select="Output")


def default_bundle():
    if sys.platform == "darwin":
        return os.path.join(ROOT, "build", "macos", "bin", NAME + ".vst3")
    return os.path.join(ROOT, "build", "bin", NAME + ".vst3")


def load(bundle):
    bundle = os.path.abspath(os.path.expanduser(bundle))
    au_register.register(bundle)
    return pedalboard.load_plugin(bundle)


def set_position(p, name, pos):
    """set a stepped parameter to a position index (0 = first legend)"""
    setattr(p, name, list(p.parameters[name].valid_values)[int(pos)])


def positions_from_reference(s):
    """reference-named settings (a protocol item's 'set') -> {pedalboard name: position} for our plugin"""
    s = {**REF_BASE, **s}
    out = {}
    for ch, pre in ((0, "l_"), (1, "r_")):
        g = lambda k: s.get(f"{k}_{ch + 1}", s[k])
        out[pre + "optical"] = 1 if g("optical_bypass") == "In" else 0
        out[pre + "optical_threshold"] = int(g("optical_threshold")) - 1
        out[pre + "optical_gain"] = int(g("optical_gain")) - 1
        out[pre + "discrete"] = 1 if g("discrete_bypass") == "In" else 0
        out[pre + "discrete_threshold"] = int(g("discrete_threshold")) - 1
        out[pre + "discrete_ratio"] = RATIOS.index(g("discrete_ratio"))
        out[pre + "discrete_attack_ms"] = ATTACKS.index(float(g("discrete_attack")))
        out[pre + "discrete_recover_s"] = RECOVERS.index(g("discrete_recover"))
        out[pre + "discrete_gain"] = int(g("discrete_gain")) - 1
        out[pre + "sidechain_filter"] = 1 if g("sidechain_filter") == "In" else 0
        out[pre + "transformer"] = CORES.index(g("transformer"))
    out["stereo"] = 1 if s["mode"] == "Stereo" else 0
    out["hardwire_bypass"] = 1 if s["hardwire_bypass"] == "In" else 0
    out["mix_percent"] = int(round(float(s["mix"])))
    return out


def apply_reference(p, s, profile=0, quality=0):
    """put the plugin in a protocol item's setting: the reference-named controls, REFERENCE profile, STANDARD quality, the rest default"""
    for name, pos in positions_from_reference(s).items():
        set_position(p, name, pos)
    set_position(p, "profile", profile)
    set_position(p, "quality", quality)
    for name in ("opto_memory", "exhibition"):
        set_position(p, name, 0)
    set_position(p, "temperature_c", 10)   # 25 C
    for pre in ("l_", "r_"):
        set_position(p, pre + "meter_select", 2)


def core_render(x, fs, named, host="fresh", profile=None, quality=None):
    """the same signal through the engine's C interface (fit/hvmc_core.py). named: {our parameter name (L_optical_threshold, quality,
    ...): position}; the rest at the plugin's defaults. host: the order of settings and prepares before the audio, which the engine's
    start depends on (see tests/crosscheck.py):
      "fresh"  the settings, then one prepare: a fresh engine, as the fitting runs it;
      "vst3"   prepared at load with the default settings; the settings reach the plugin with the first block, after the host's
               prepare (Pedalboard's VST3 host, JUCE, delivers parameter changes in the process call, and DPF's VST3 wrapper applies
               controller-side values only in its separate-controller build);
      "au"     prepared at load with the defaults, the settings applied at once, then prepared twice more by Pedalboard's AU host.
    Both host models were established by comparing the plugin's output with the engine's, sample for sample (tests/crosscheck.py).
    Returns (output (2, n) float64, latency)."""
    sys.path.insert(0, os.path.join(ROOT, "fit"))
    import ctypes, numpy as np, hvmc_core
    M = hvmc_core.Model(); L = hvmc_core.lib()
    pos = {i: L.hvmc_param_default(i) for i in range(M.nin)}
    for k, v in named.items():
        pos[M.index[k]] = v
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if x.ndim == 1:
        x = np.stack([x, x])
    e = L.hvmc_new()
    try:
        if host != "fresh":
            L.hvmc_prepare(e, float(fs))
        for k, v in pos.items():
            L.hvmc_set_param(e, int(k), int(v))
        for _ in range({"fresh": 1, "vst3": 0, "au": 2}[host]):
            L.hvmc_prepare(e, float(fs))
        y = np.zeros_like(x); fp = ctypes.POINTER(ctypes.c_float)
        L.hvmc_process(e, x[0].ctypes.data_as(fp), x[1].ctypes.data_as(fp), y[0].ctypes.data_as(fp), y[1].ctypes.data_as(fp), x.shape[1])
        return y.astype(np.float64), L.hvmc_latency(e)
    finally:
        L.hvmc_free(e)


def host_model(bundle):
    """how Pedalboard drives this bundle before its first process(): see core_render"""
    return "au" if bundle.rstrip("/").endswith(".component") else "vst3"
