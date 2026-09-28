# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Discriminating measurements of the reference's plugin-only controls: the external key input (which stages it feeds; Pedalboard
cannot drive a sidechain bus, so the test is whether an empty key silences the detectors) and the variable sidechain high-pass
(order, range, insertion gain, both stages). Writes fit/data/discriminate_key_hpf.json. usage: python3 -u fit/measure/discriminate_key_hpf.py"""
sys.path.insert(0, os.path.dirname(__file__))
import pa
from pa import both, sine, rms_db, FS
P = pa.Ref()
R = {"meta": {"version": P.version, "sha256": P.sha256}}
# key input: with KEY IN and no key signal (Pedalboard feeds the main bus only), does each stage stop compressing?
x = sine(-10, secs=2.5)
for stage, st in (("disc", both(discrete_bypass="In", discrete_threshold=16)), ("opto", both(optical_bypass="In", optical_threshold=20))):
    rows = {ki: round(rms_db(P.run(x, **st, key_in=ki)[0, -FS:]) - rms_db(x[-FS:]), 2) for ki in ("Out", "In")}
    R[f"key_{stage}_gain_db"] = rows; print(f"key_in with an empty key bus, {stage}: gain", rows)
def g(y, x): return rms_db(y[0, -FS:]) - rms_db(x[-FS:])
freqs = [30.0, 60.0, 90.0, 130.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
R["freqs"] = freqs
def gr(f, **kw):
    xf = sine(-10, f=f, secs=3); x0 = sine(-50, f=f, secs=3)
    return g(P.run(x0, **kw), x0) - g(P.run(xf, **kw), xf)
ref = {f: gr(f, **both(discrete_bypass="In", discrete_threshold=16, sidechain_filter="Out")) for f in freqs}
R["disc_out"] = ref
for fc in (20.0, 90.0, 200.0, 400.0, 666.0):
    row = {f: gr(f, **both(discrete_bypass="In", discrete_threshold=16, sidechain_filter="In", sidechain_hp_freq_hz=fc)) for f in freqs}
    R[f"disc_fc{int(fc)}"] = row
    d = {int(f): round(row[f] - ref[f], 2) for f in freqs}
    print(f"discrete, filter IN at {fc:.0f} Hz, GR change vs OUT:", d)
refo = {f: gr(f, **both(optical_bypass="In", optical_threshold=20, sidechain_filter="Out")) for f in (30.0, 90.0, 200.0, 400.0, 1600.0)}
for fc in (90.0, 400.0):
    row = {f: gr(f, **both(optical_bypass="In", optical_threshold=20, sidechain_filter="In", sidechain_hp_freq_hz=fc)) for f in refo}
    R[f"opto_fc{int(fc)}"] = row; R["opto_out"] = refo
    print(f"optical, filter IN at {fc:.0f} Hz, GR change vs OUT:", {int(f): round(row[f] - refo[f], 2) for f in refo})
json.dump(R, open(os.path.join(os.path.dirname(__file__), "..", "data", "discriminate_key_hpf.json"), "w"), indent=1, default=float)
