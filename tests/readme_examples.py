#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Runs the Python example of README.md exactly as written, against a built bundle, and checks what it does. EXAMPLE below is the
example's text; when README.md exists its first ```python block must be this text, character for character, so the two cannot drift.
The example loads the plugin from "~/.vst3/HiddenValleyMC.vst3" and reads "mix.wav": the test copies the bundle under test to that path
inside a scratch HOME (build/readme-examples/) and writes a test mix.wav there (32-bit float stereo, 2 s: a 1 kHz tone at -10 dBFS on the
left, 40 Hz plus 1 kHz on the right, both quiet for the first 0.4 s). A macOS Audio Unit cannot be loaded from a .vst3 path, so for a
.component bundle that one path becomes ~/Library/Audio/Plug-Ins/Components/HiddenValleyMC.component in the scratch HOME and the AU is
registered in this process; nothing else changes. Checks: the example prints the plugin's name, keeps the length (the host removes the HQ
latency), compresses (the loud part comes out more than 1 dB quieter, no make-up is set), writes mix_hvmc.wav, and its output is the
same, bit for bit, as the same settings through the engine's C interface (fit/hvmc_core.py) driven as the host drives the plugin
(tests/crosscheck.py explains).
usage: python3 tests/readme_examples.py [bundle]"""
import contextlib, io, os, re, runpy, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "fit"))
import hvmc_pb  # noqa: E402
from pedalboard.io import AudioFile  # noqa: E402

EXAMPLE = '''import os
from pedalboard import load_plugin
from pedalboard.io import AudioFile

comp = load_plugin(os.path.expanduser("~/.vst3/HiddenValleyMC.vst3"))
print(comp.name)

with AudioFile("mix.wav") as f:
    audio, sr = f.read(f.frames), f.samplerate

# the panel legends, as printed on the unit; in STEREO the left controls drive both channels
comp.stereo = "STEREO"
comp.l_optical = "IN"
comp.l_optical_threshold = 18
comp.l_optical_gain = 12
comp.l_discrete = "IN"
comp.l_discrete_threshold = 14
comp.l_discrete_ratio = "2:1"
comp.l_discrete_attack_ms = 5
comp.l_discrete_recover_s = "0.5"
comp.l_discrete_gain = 9
comp.l_sidechain_filter = "IN"
comp.l_transformer = "IRON"
comp.quality = "HQ 2X"   # twice the rate inside; the host is told the 39-sample latency

out = comp(audio, sr)
with AudioFile("mix_hvmc.wav", "w", sr, out.shape[0]) as f:
    f.write(out)
'''
# the example's settings as engine positions, for the cross-check
POSITIONS = {"stereo": 1, "L_optical": 1, "L_optical_threshold": 17, "L_optical_gain": 11, "L_discrete": 1, "L_discrete_threshold": 13,
             "L_discrete_ratio": 1, "L_discrete_attack_ms": 3, "L_discrete_recover_s": 2, "L_discrete_gain": 8, "L_sidechain_filter": 1,
             "L_transformer": 1, "quality": 1}

bundle = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else hvmc_pb.default_bundle())
fails = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok: fails.append(name)

readme = os.path.join(ROOT, "README.md")
if os.path.exists(readme):
    blocks = re.findall(r"```python\n(.*?)```", open(readme).read(), re.S)
    check("README.md's first python example is this test's EXAMPLE, verbatim", bool(blocks) and blocks[0] == EXAMPLE,
          "no python block in README.md" if not blocks else "" if blocks[0] == EXAMPLE else "differs")
else:
    print("note: README.md not present; running EXAMPLE as the example text")
example = EXAMPLE; path = "~/.vst3/HiddenValleyMC.vst3"
if bundle.endswith(".component"):
    au_path = "~/Library/Audio/Plug-Ins/Components/HiddenValleyMC.component"
    example = example.replace(path, au_path); path = au_path
    print(f"Audio Unit: the example loads {au_path} in place of ~/.vst3/HiddenValleyMC.vst3; everything else runs as written")
work = os.path.join(ROOT, "build", "readme-examples")
shutil.rmtree(work, ignore_errors=True)
if sys.platform == "win32" and os.path.isdir(bundle):
    # Pedalboard's Windows host loads a single-file VST3 but not a bundle folder: install the DLL as ~/.vst3/HiddenValleyMC.vst3
    os.makedirs(os.path.dirname(os.path.join(work, path[2:])), exist_ok=True)
    shutil.copy2(os.path.join(bundle, "Contents", "x86_64-win", os.path.basename(bundle)), os.path.join(work, path[2:]))
else:
    shutil.copytree(bundle, os.path.join(work, path[2:]))
hvmc_pb.au_register.register(os.path.join(work, path[2:]))
open(os.path.join(work, "readme_example.py"), "w").write(example)
fs = 48000; t = np.arange(2 * fs) / fs
env = np.where(t < 0.4, 10 ** (-20 / 20), 1.0)
x = np.stack([env * 10 ** (-10 / 20) * np.sin(2 * np.pi * 1000 * t),
              env * (10 ** (-16 / 20) * np.sin(2 * np.pi * 40 * t) + 10 ** (-16 / 20) * np.sin(2 * np.pi * 1000 * t))]).astype(np.float32)
with AudioFile(os.path.join(work, "mix.wav"), "w", fs, 2, bit_depth=32) as f:
    f.write(x)
os.environ["HOME"] = os.environ["USERPROFILE"] = work; os.chdir(work)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = runpy.run_path(os.path.join(work, "readme_example.py"))
printed = buf.getvalue().strip(); print("example printed:", printed)
check("the example prints the plugin's name", printed == "Hidden Valley Mastering Compressor", repr(printed))
audio, sr, out = ns["audio"], ns["sr"], ns["out"]
check("the example reads mix.wav unchanged", sr == fs and np.array_equal(audio, x), f"sr {sr}")
check("the output keeps the length (the host removes the HQ latency)", out.shape == audio.shape, str(out.shape))
check("the output is finite", bool(np.all(np.isfinite(out))))
rms = lambda v: 20 * np.log10(np.sqrt(np.mean(np.square(v.astype(np.float64)))))
loud = slice(int(0.8 * fs), int(1.9 * fs))
gr = rms(audio[0, loud]) - rms(out[0, loud])
check("the loud part comes out more than 1 dB quieter (compression, no make-up)", gr > 1.0, f"{gr:.2f} dB quieter on the left")
with AudioFile(os.path.join(work, "mix_hvmc.wav")) as f:
    check("mix_hvmc.wav is written with the output's length", f.frames == out.shape[1] and f.num_channels == 2, f"{f.frames} frames, {f.num_channels} channels")
core, lat = hvmc_pb.core_render(x, fs, POSITIONS, hvmc_pb.host_model(bundle))
n = x.shape[1] - lat
d = float(np.max(np.abs(out[:, :n].astype(np.float64) - core[:, lat:lat + n])))
check("the output equals the engine's C interface, bit for bit (HQ, aligned by the 39-sample latency)", d == 0.0, f"max |diff| {d:.3g}")
os.chdir(ROOT); shutil.rmtree(work, ignore_errors=True)
print("PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
