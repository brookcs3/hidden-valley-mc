#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Cross-check of the three ways the engine runs: the C++ unit tests (tests/test_dsp.cpp, which write cpp_reference.json), the same
engine through the C interface used by the fitting (fit/hvmc_core.py, build/capi/libhvmc), and the built plugin through Pedalboard.
 A. the ctypes core reproduces cpp_reference.json (three renders' RMS) within 0.001 dB;
 B. the plugin and the ctypes core give the same samples, bit for bit, on five renders: a tone through both stages, a burst into the
    DUAL release, dual-mono noise with Iron and Steel on different channels and different settings, the tone in HQ 2X (the plugin's
    output is aligned by the 39-sample latency, which Pedalboard removes), and Gold/Uranium in exhibition mode. The core is driven the
    way the host drives the plugin (tests/hvmc_pb.py core_render: a VST3 is prepared at load with the default settings and receives the
    user's settings with the first block; an Audio Unit has them applied at once and is prepared twice more), because the engine's
    start depends on that order: Engine::prepare() carries the previous settings into the detector's rest level and the control
    smoothers, and settings applied after a prepare slew in from the previous ones, so the first quarter second depends on the settings
    before. The same renders against a fresh engine (settings, then one prepare, as the fitting runs it) are reported too and gated by
    START_TOL; see the note there.
usage: python3 tests/crosscheck.py [cpp_reference.json] [bundle]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "fit"))
import hvmc_pb, hvmc_core  # noqa: E402

# START_TOL: the plugin against a fresh engine given the same settings before its prepare(). Engine::prepare() and the first block after
# it reset every stage to the settings in force (parameters that arrive between the host's prepare and the first block count as
# initial settings), so the two renders are identical and the tolerance is zero.
START_TOL = 0.0

def default_ref():
    if sys.platform == "darwin":
        import platform
        return os.path.join(ROOT, "build", "macos", "tests", platform.machine(), "cpp_reference.json")
    return os.path.join(ROOT, "build", "tests", "cpp_reference.json")

ref_path = sys.argv[1] if len(sys.argv) > 1 else default_ref()
bundle = sys.argv[2] if len(sys.argv) > 2 else hvmc_pb.default_bundle()
M = hvmc_core.Model(); L = hvmc_core.lib()
fails = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok: fails.append(name)
def rms_db(v):
    return 20 * np.log10(np.sqrt(np.mean(np.square(v))) + 1e-30)
def defaults():
    return {i: L.hvmc_param_default(i) for i in range(M.nin)}
def core_render(x, fs, named, host="fresh"):
    return hvmc_pb.core_render(x, fs, named, host)[0]

# A. the C++ unit test's renders
J = json.load(open(ref_path)); fs = float(J["fs"]); worst = 0.0
t = np.arange(int(1.5 * fs)) / fs
for r in J["renders"]:
    x = 10 ** (r["level_dbfs"] / 20) * np.sin(2 * np.pi * 1000 * t)
    y = core_render(x, fs, {"L_optical_threshold": r["optical_threshold"] - 1, "L_discrete_threshold": r["discrete_threshold"] - 1})
    got = rms_db(y[0, int(fs):]); worst = max(worst, abs(got - r["out_rms_db"]))
check(f"ctypes core reproduces cpp_reference.json ({len(J['renders'])} renders)", worst < 0.001, f"max |RMS difference| {worst:.2e} dB")

# B. plugin against the ctypes core
host = hvmc_pb.host_model(bundle)
fs = 48000; t = np.arange(int(2.5 * fs)) / fs
tone = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2) * 10 ** (-10 / 20)
per = fs / 1000; n0 = int(0.5 * 1000 * per); n1 = int(1.5 * 1000 * per); n2 = int(2.0 * 1000 * per)
amp = np.concatenate([np.full(n0, 10 ** (-50 / 20)), np.full(n1, 10 ** (-10 / 20)), np.full(n2, 10 ** (-50 / 20))])
tb = np.arange(len(amp)) / fs; burst = np.stack([amp * np.sin(2 * np.pi * 1000 * tb)] * 2)
rng = np.random.default_rng(7); noise = 0.2 * rng.standard_normal((2, 2 * fs))
cases = [
    ("tone, both stages, STEREO", tone, {"L_optical_threshold": 17, "L_discrete_threshold": 19, "L_discrete_attack_ms": 0, "L_discrete_recover_s": 1}),
    ("burst into the DUAL release", burst, {"L_optical": 0, "L_discrete_threshold": 15, "L_discrete_attack_ms": 2, "L_discrete_recover_s": 5}),
    ("dual-mono noise, L Iron 3:1 / R Steel FLOOD", noise, {"stereo": 0, "L_transformer": 1, "L_discrete_threshold": 13, "L_discrete_ratio": 2,
                                                          "R_transformer": 2, "R_discrete_threshold": 9, "R_optical_threshold": 19, "R_discrete_ratio": 5}),
    ("tone, both stages, HQ 2X", tone, {"L_optical_threshold": 17, "L_discrete_threshold": 19, "quality": 1}),
    ("Gold L / Uranium R, exhibition, 40 C", noise, {"stereo": 0, "L_transformer": 3, "R_transformer": 4, "exhibition": 1, "temperature_c": 25,
                                                   "L_optical_threshold": 19, "R_optical_threshold": 19, "L_discrete_threshold": 13, "R_discrete_threshold": 13}),
]
for name, x, named in cases:
    q = hvmc_pb.load(bundle)
    for k, v in named.items(): hvmc_pb.set_position(q, k.lower(), v)
    yp = q.process(x.astype(np.float32), fs, reset=True).astype(np.float64)
    lat = int(q.reported_latency_samples); n = x.shape[1] - lat
    like_host = core_render(x, fs, named, host)[:, lat:lat + n]
    fresh = core_render(x, fs, named)[:, lat:lat + n]
    d_host = float(np.max(np.abs(yp[:, :n] - like_host))); d_fresh = float(np.max(np.abs(yp[:, :n] - fresh)))
    check(f"plugin = core driven as the {host} host does, bit for bit: {name}", d_host == 0.0, f"max |diff| {d_host:.3g} (latency {lat})")
    check(f"plugin vs a fresh engine within START_TOL: {name}", d_fresh <= START_TOL, f"max |diff| {d_fresh:.3g}")
print("PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
