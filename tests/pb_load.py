#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Load the bundle with Pedalboard in this process and check the panel against the plugin's own parameter table (src/HVMCParams.hpp,
read through the C interface build/capi/libhvmc): the 32 input parameters and 7 read-only meters are all there under their names, every
input parameter offers exactly the panel legends in position order, each defaults to the documented position, and every position of
every input parameter sets and reads back as its legend (481 positions). Then the plugin's promises to a host: latency 0 in STANDARD and
39 samples in HQ 2X, and the reported latency is the true delay (in HQ 2X with HARDWIRE BYPASS out, Pedalboard's latency-compensated
output is the input bit for bit); HARDWIRE BYPASS out and MIX 0 % are bit-transparent in STANDARD; 2 s of noise at 44.1, 48 and 96 kHz
comes out finite and identical whatever the block size (64, 1024, 8192). On a macOS Audio Unit the meters are checked too: after a tone
at discrete threshold 20 the gain-reduction meters read the reduction and a value written to a meter is overwritten by processing
(read-only). Pedalboard's VST3 host (JUCE) caches parameter values and applies a plugin's output-parameter changes only from its message
loop, which never runs here, so through the .vst3 the meters read stale and are not checked (the C++ unit tests cover them).
usage: python3 tests/pb_load.py [bundle] [out.json]   (default build/macos/bin/HiddenValleyMC.vst3, or build/bin/... on Linux)"""
import ctypes, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "fit"))
import hvmc_pb, hvmc_core  # noqa: E402

bundle = sys.argv[1] if len(sys.argv) > 1 else hvmc_pb.default_bundle()
outjson = sys.argv[2] if len(sys.argv) > 2 else None
is_au = bundle.rstrip("/").endswith(".component")
fails = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok: fails.append(name)

# the plugin's own table, through the C interface
L = hvmc_core.lib()
nin, nall = L.hvmc_num_input_params(), L.hvmc_num_params()
def cname(i):
    b = ctypes.create_string_buffer(64); L.hvmc_param_name(i, b, 64); return b.value.decode()
def clabel(i, v):
    b = ctypes.create_string_buffer(64); L.hvmc_param_label(i, v, b, 64); return b.value.decode()
table = []
for i in range(nall):
    labels = [clabel(i, v) for v in range(L.hvmc_param_steps(i))] if i < nin else []
    table.append({"index": i, "name": cname(i), "pb_name": cname(i).lower(), "labels": labels, "default": L.hvmc_param_default(i) if i < nin else None})

def expected_values(labels):
    """what Pedalboard makes of a legend list: floats if every legend is a number, False/True for OFF/ON, else the strings"""
    if labels == ["OFF", "ON"]: return [False, True]
    try: return [float(l) for l in labels]
    except ValueError: return list(labels)

p = hvmc_pb.load(bundle)
print("loaded:", p.name, "| parameters:", len(p.parameters), "| reported latency:", p.reported_latency_samples)
names = set(p.parameters)
ours = {t["pb_name"] for t in table}
extra = names - ours
check(f"all {nall} parameters present ({nin} inputs, {nall - nin} meters)", ours <= names, f"missing {sorted(ours - names)}" if ours - names else "")
check("no parameters besides the plugin's own and the host's latency parameter", extra <= hvmc_pb.HOST_PARAMS, f"extra {sorted(extra)}")

# legends, defaults and read-back of every position
out = {}
bad_valid, bad_default, bad_readback, nset = [], [], [], 0
for t in table[:nin]:
    prm = p.parameters[t["pb_name"]]
    vv = list(prm.valid_values); exp = expected_values(t["labels"])
    if vv != exp or prm.num_steps != len(t["labels"]): bad_valid.append(f"{t['pb_name']}: {vv[:4]}.. steps {prm.num_steps} vs {exp[:4]}.. {len(exp)}")
    if prm.string_value != t["labels"][t["default"]]: bad_default.append(f"{t['pb_name']}: {prm.string_value!r} vs {t['labels'][t['default']]!r}")
    for v, lab in enumerate(t["labels"]):
        hvmc_pb.set_position(p, t["pb_name"], v); nset += 1
        if prm.string_value != lab: bad_readback.append(f"{t['pb_name']} pos {v}: {prm.string_value!r} vs {lab!r}")
    hvmc_pb.set_position(p, t["pb_name"], t["default"])
    out[t["pb_name"]] = {"raw_name": prm.name, "legends": t["labels"], "default": t["labels"][t["default"]], "valid_values": [str(v) for v in vv]}
    print(f"{t['pb_name']:22s} default {t['labels'][t['default']]:>10s}  {len(t['labels']):3d} positions  {t['labels'] if len(t['labels']) <= 8 else t['labels'][:3] + ['...'] + t['labels'][-2:]}")
for t in table[nin:]:
    prm = p.parameters[t["pb_name"]]
    out[t["pb_name"]] = {"raw_name": prm.name, "value": float(getattr(p, t["pb_name"]))}
    print(f"{t['pb_name']:22s} meter, reads {float(getattr(p, t['pb_name'])):+.2f}")
check("every input parameter offers exactly the panel legends, in position order", not bad_valid, "; ".join(bad_valid[:3]))
check("every input parameter defaults to the documented position", not bad_default, "; ".join(bad_default[:3]))
check(f"every position of every input parameter sets and reads back as its legend ({nset} positions)", not bad_readback, "; ".join(bad_readback[:3]))

# latency and transparency
fs = 48000
rng = np.random.default_rng(20260927); noise = (0.25 * rng.standard_normal((2, 2 * fs))).astype(np.float32)
check("STANDARD reports latency 0", p.reported_latency_samples == 0, str(p.reported_latency_samples))
p.hardwire_bypass = "OUT"
y = p.process(noise, fs, reset=True)
check("HARDWIRE BYPASS out is bit-transparent (STANDARD)", np.array_equal(y, noise), f"max |diff| {float(np.max(np.abs(y - noise))):.3g}")
p.quality = "HQ 2X"
y = p.process(noise, fs, reset=True)   # the host learns the new latency when the plugin next runs
check("HQ 2X reports latency 39", p.reported_latency_samples == 39, str(p.reported_latency_samples))
check("HQ 2X: the reported latency is the true delay (bypass through Pedalboard's compensation is bit-transparent)", y.shape == noise.shape and np.array_equal(y, noise),
      f"shape {y.shape}, max |diff| {float(np.max(np.abs(y - noise))):.3g}")
p.quality = "STANDARD"; p.hardwire_bypass = "IN"; p.mix_percent = 0
p.l_discrete_threshold = 24; p.l_optical_threshold = 24
y = p.process(noise, fs, reset=True)
check("MIX 0 % is bit-transparent", np.array_equal(y, noise), f"max |diff| {float(np.max(np.abs(y - noise))):.3g}")
p.mix_percent = 100; p.l_discrete_threshold = 16; p.l_optical_threshold = 18
for sr in (44100, 48000, 96000):
    p.process(noise[:, :4096], sr, reset=True)   # settle the settings at this rate first (see the note in tests/crosscheck.py on start-up)
    ys = [p.process(noise, sr, buffer_size=bs, reset=True) for bs in (64, 1024, 8192)]
    same = all(np.array_equal(ys[0], yy) for yy in ys[1:])
    check(f"{sr} Hz: finite output, identical for block sizes 64, 1024, 8192", same and all(np.all(np.isfinite(yy)) for yy in ys),
          f"peak {float(np.max(np.abs(ys[0]))):.3f}")

# meters (Audio Unit only, see the docstring)
if is_au:
    q = hvmc_pb.load(bundle)
    q.l_discrete_threshold = 20; q.l_optical_threshold = 18; q.l_meter_select = "DISCRETE"; q.r_meter_select = "OPTICAL"
    q.l_meter_db = 7.5
    t = np.arange(int(2.0 * fs)) / fs; tone = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32) * 10 ** (-10 / 20)
    q.process(tone, fs, reset=True)
    m = {n: float(getattr(q, n)) for n in ("l_meter_db", "l_gr_optical_db", "l_gr_discrete_db", "r_meter_db", "r_gr_optical_db", "r_gr_discrete_db", "magic_eye_db")}
    check("AU meters: the discrete GR meter reads the reduction (< -3 dB) and the optical one some (< 0)", m["l_gr_discrete_db"] < -3.0 and m["l_gr_optical_db"] < 0.0, str(m))
    check("AU meters: METER SELECT routes the per-channel meter (L DISCRETE, R OPTICAL)", abs(m["l_meter_db"] - m["l_gr_discrete_db"]) < 1e-6 and abs(m["r_meter_db"] - m["r_gr_optical_db"]) < 1e-6, str(m))
    check("AU meters: a value written to a meter is overwritten by processing (read-only)", abs(m["l_meter_db"] - 7.5) > 0.5, f"{m['l_meter_db']:+.2f}")
    check("AU meters: the magic eye follows the output peak (dBFS, negative)", -40.0 < m["magic_eye_db"] < 0.0, f"{m['magic_eye_db']:+.2f}")
else:
    print("SKIP meters: through a .vst3 Pedalboard's host does not refresh output parameters (run this on the .component for the meter checks)")

if outjson:
    json.dump({"bundle": bundle, "name": p.name, "n_parameters": len(p.parameters), "reported_latency": int(p.reported_latency_samples), "parameters": out},
              open(outjson, "w"), indent=1)
print("PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
