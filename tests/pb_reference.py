#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""The plugin against the reference: every item of the measurement protocol (fit/measure/protocol.py, 5,226 stimuli and settings) is
rendered through the built plugin with Pedalboard (REFERENCE profile, STANDARD quality) and its feature is compared with the one
captured from the reference plugin (fit/data/reference_features.json): steady gains in dB, harmonics in dBc (odd orders H3/H5/H7 and
even orders H2/H4/H6/H8 pooled apart, where the reference's harmonic is above -80 dBc: below that it is inaudible next to the
fundamental and the reference's own figures scatter), burst envelopes in dB per period, small-signal responses in dB. The residuals are
pooled per protocol group and feature kind and printed as rms and largest |error|; each pair must stay within TOL, the tolerance table below, which is the
repository's claim of how close the model is. --quick renders a fixed subset (about 900 items, half a minute) instead of the whole
protocol (a few minutes). Before each item the plugin runs a short silence with the item's settings, so that the item's own render
starts from a prepare made with those settings: the engine takes the detector's rest level and the control smoothers from the
settings of the previous prepare (tests/crosscheck.py), and the reference was captured from a fresh instance per item.
The transformer grid's +24 dBFS points (+38 dBu) are pooled apart as xfmr_ceiling: there the reference runs into an output ceiling
(H2 jumps to -12 dBc between +21 and +24 dBFS at 320 Hz and 1 kHz on every core) that the model, which has no clipper, does not
reproduce; the figures are reported and gated loosely so the rest of the grid can be held to its own tolerance.
usage: python3 tests/pb_reference.py [bundle] [out.json] [--quick] [--groups a,b,...]"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "fit", "measure"))
import hvmc_pb, protocol  # noqa: E402

# (rms, max) in dB per group and feature kind, set from the full protocol on 2026-09-27 (fit stages 1-5 as of that run) with a margin
# of about 30 % on the rms and 20 % on the max. PROVISIONAL marks the groups whose figures come from parts of the model still being
# fitted or with known defects at that time, to be tightened when those land:
#   stage 3 (discrete detector): its rest level sits 0.25 dB below threshold where the fitted curves are not zero (1.2:1 0.6 dB, 2:1 1.4,
#   3:1 0.9, 4:1 0.2 dB of standing gain reduction on any signal below threshold: law, disc_static, disc_gi), and the detector's dynamics
#   (disc_ar, disc_dyn, sr, link, scf, both);
#   stage 4 (optical stage): its gain reduction against frequency (opto_static_f, up to 4 dB at 8 kHz), its distortion under gain
#   reduction, which is not fitted (opto_harm odd/even), and its dynamics (opto_dyn).
# The transformer's even harmonics are a known gap of stage 2: where the reference's are quiet (-75 to -100 dBc) the model's sit at
# -45 to -50 dBc, and at a few points the other way round; the odd orders, which carry the core's sound, are held tighter.
TOL = {
    # rms / max in dB per group and kind, set from the fitted model's achieved figures on the full protocol (2026-09-27, stages 1-5 final)
    # with about 1.5x margin on rms and 1.3x on max; a change that pushes a group past these is a regression, not noise.
    "law":           {"gain": (0.01, 0.02)},
    "opto_static":   {"gain": (0.12, 0.45)},
    "opto_static_f": {"gain": (0.45, 1.70)},                          # 8 kHz above +4 dBFS: the reference's own HF ceiling (docs/opto-fix.md 6.5)
    "opto_dyn":      {"env": (0.12, 1.20)},                           # max: the first 20 ms of a below-knee release (docs/opto-fix.md 6.3)
    "opto_harm":     {"gain": (0.25, 0.70), "odd": (3.5, 8.0), "even": (15.0, 35.0)},   # odd: H3 within about 2 dB, H5 within 5 (an open item)
    "disc_static":   {"gain": (0.10, 1.20)},                          # max: the soft ratios' last dB below the knee, an open item (docs/disc-knee-fix.md)
    "disc_ar":       {"gain": (0.16, 0.50)},
    "disc_dyn":      {"env": (0.20, 7.50)},                           # max: one period at the onset of the 0 dBFS burst (docs/detector-fix.md 5.1)
    "disc_harm":     {"gain": (0.12, 0.25), "odd": (1.5, 2.5), "even": (1.0, 1.5)},
    "disc_gi":       {"gain": (0.06, 0.08)},
    "link":          {"gain": (0.10, 0.25)},
    "scf":           {"gain": (0.35, 1.00)},
    "xfmr":          {"gain": (0.06, 0.40), "odd": (4.5, 60.0), "even": (17.0, 41.0)},   # even: open item (docs/xfmr-even-fix.md); odd max: one point, xf_Iron_f1000_21 (reference H5 -71 dBc, model none)
    "xfmr_ceiling":  {"gain": (2.00, 4.00), "odd": (40.0, 80.0), "even": (40.0, 80.0)},  # the reference's output ceiling at +24 dBFS, not modelled
    "xfmr_resp":     {"resp": (0.20, 1.40)},
    "stage_harm":    {"gain": (0.10, 0.20), "odd": (2.0, 2.5), "even": (2.0, 2.5)},
    "sr":            {"env": (0.10, 1.00)},
    "both":          {"gain": (0.85, 1.20)},                          # the two stages in series at three levels; carries both stages' knee errors
    "mix":           {"gain": (0.05, 0.06)},
}
CEILING_DBFS = 24   # transformer-grid levels at or above this go to xfmr_ceiling
HARM_FLOOR = -80.0   # dBc: harmonics the reference has below this are not compared
MODEL_FLOOR = -120.0

def quick_subset(items):
    keep = []
    for it in items:
        g, i = it["group"], it["id"]; s = it["set"]
        if g == "opto_static": ok = s["optical_threshold"] in (1, 12, 20, 24)
        elif g == "disc_static":
            lvl = int(i.rsplit("_", 1)[1]); ok = s["discrete_ratio"] in ("1.2:1", "4:1", "Flood") and s["discrete_threshold"] in (1, 6, 12, 16, 20, 24) and lvl % 9 == 0
        elif g == "disc_dyn": ok = i.startswith("disc_burst_") and (i.endswith("_0.1 s") or i.endswith("_Dual")) or i.startswith("disc_bdepth")
        elif g == "xfmr": ok = int(i.rsplit("_", 1)[1]) in (-12, 6, 18, 24)
        else: ok = True
        if ok: keep.append(it)
    return keep

def residuals(item, model, ref):
    """-> {kind: array of dB residuals}"""
    t = item["feat"]["type"]
    if t == "gain_db": return {"gain": np.array([model - ref])}
    if t == "env":
        n = min(len(ref), len(model)); return {"env": np.asarray(model[:n]) - np.asarray(ref[:n])}
    if t == "resp":
        return {"resp": np.array([a - b for a, b in zip(model, ref) if a is not None and b is not None])}
    if t == "harm":
        out = {"gain": np.array([model["gain_db"] - ref["gain_db"]]), "odd": [], "even": []}
        for k, (a, b) in enumerate(zip(model["h"], ref["h"]), start=2):
            if a is None or b is None or b <= HARM_FLOOR: continue
            out["odd" if k % 2 else "even"].append(max(a, MODEL_FLOOR) - b)
        return {k: np.asarray(v) for k, v in out.items()}
    raise ValueError(t)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bundle = args[0] if args else hvmc_pb.default_bundle(); outjson = args[1] if len(args) > 1 else None
    quick = "--quick" in sys.argv
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--groups="): only = set(a[len("--groups="):].split(","))
    REF = json.load(open(os.path.join(ROOT, "fit", "data", "reference_features.json"))); F = REF["features"]
    items = protocol.items()
    if quick: items = quick_subset(items)
    for it in items:
        if it["group"] == "xfmr" and int(it["id"].rsplit("_", 1)[1]) >= CEILING_DBFS: it["group"] = "xfmr_ceiling"
    if only: items = [it for it in items if it["group"] in only]
    p = hvmc_pb.load(bundle)
    print(f"{p.name}: {len(items)} protocol items{' (quick subset)' if quick else ''}{' (groups ' + ','.join(sorted(only)) + ')' if only else ''} against {REF['meta'].get('reference', 'the reference')} {REF['meta'].get('version', '')}")
    pooled = {}; worst = {}; t0 = time.time(); n = 0
    for it in items:
        if it["id"] not in F: continue
        fs = it["fs"]; x = protocol.stimulus(it["stim"], fs).astype(np.float32)
        hvmc_pb.apply_reference(p, it["set"])
        p.process(np.zeros((2, int(0.05 * fs)), np.float32), fs, reset=True)   # settle the settings (see the docstring)
        y = p.process(x, fs, reset=True).astype(np.float64)
        feat = protocol.feature(it, y, fs)
        for kind, r in residuals(it, feat, F[it["id"]]).items():
            if len(r) == 0: continue
            key = (it["group"], kind); pooled.setdefault(key, []).append(r)
            m = float(np.max(np.abs(r)))
            if m > worst.get(key, (0, ""))[0]: worst[key] = (m, it["id"])
        n += 1
    print(f"rendered {n} items in {time.time() - t0:.0f} s\n")
    print(f"{'group':14s} {'kind':5s} {'items':>5s} {'values':>6s} {'rms dB':>7s} {'max dB':>7s}   {'tol rms':>7s} {'tol max':>7s}  worst item")
    fails = []; table = {}
    for key in sorted(pooled):
        g, kind = key; v = np.concatenate(pooled[key]); rms = float(np.sqrt(np.mean(v ** 2))); mx = float(np.max(np.abs(v)))
        tol = TOL.get(g, {}).get(kind)
        if tol is None: fails.append(f"{g}/{kind}: no tolerance"); status = "?"
        else: status = "PASS" if (rms <= tol[0] and mx <= tol[1]) else "FAIL"
        if status == "FAIL": fails.append(f"{g}/{kind}: rms {rms:.3f} max {mx:.2f} (tol {tol[0]}, {tol[1]})")
        table[f"{g}/{kind}"] = {"items": len(pooled[key]), "values": int(len(v)), "rms": rms, "max": mx, "worst_item": worst[key][1], "tol": tol, "status": status}
        print(f"{g:14s} {kind:5s} {len(pooled[key]):5d} {len(v):6d} {rms:7.3f} {mx:7.2f}   {tol[0] if tol else float('nan'):7.2f} {tol[1] if tol else float('nan'):7.2f}  {worst[key][1]}  {status}")
    if outjson:
        json.dump({"bundle": bundle, "quick": quick, "items": n, "groups": table, "reference": REF["meta"]}, open(outjson, "w"), indent=1)
    for f in fails: print("FAIL", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    sys.exit(0 if not fails else 1)

if __name__ == "__main__":
    main()
