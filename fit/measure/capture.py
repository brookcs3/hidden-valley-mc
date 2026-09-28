# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Capture the reference features: run every protocol item (protocol.py) through the reference plugin and store the features, with the
reference's version and binary hash, in fit/data/reference_features.json. Needs a licensed copy of the reference installed; the features
file is what the fitting and the tests use, so nobody else needs the reference.
usage: python3 fit/measure/capture.py [--group g1,g2]   (re-captures only those groups and merges)"""
import json, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pa, protocol

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reference_features.json")

def settings_for(item):
    s = {}
    for k, v in item["set"].items():
        if k in ("mode", "mix", "hardwire_bypass"):
            s[k] = v
        else:
            s[k + "_1"] = v; s[k + "_2"] = v
    return s

def main():
    groups = None
    if "--group" in sys.argv:
        groups = set(sys.argv[sys.argv.index("--group") + 1].split(","))
    P = pa.Ref()
    data = json.load(open(OUT)) if (groups and os.path.exists(OUT)) else {"meta": {}, "features": {}}
    data["meta"] = {"reference": "Plugin Alliance / Brainworx 'Shadow Hills Mastering Compressor' (VST3, licensed copy)",
                    "version": P.version, "sha256": P.sha256, "host": "pedalboard", "buffer": 1024,
                    "captured": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level_convention": "0 dBFS = +14 dBu, dBFS peak of a sine",
                    "base_settings": pa.BASE}
    its = [it for it in protocol.items() if groups is None or it["group"] in groups]
    t0 = time.time()
    for i, it in enumerate(its):
        x = protocol.stimulus(it["stim"], it["fs"])
        y = np.asarray(P.run(x, fs=it["fs"], **settings_for(it)), dtype=np.float64)
        data["features"][it["id"]] = protocol.feature(it, y, it["fs"])
        if i % 250 == 0:
            print(f"{i}/{len(its)} {it['id']}  {time.time() - t0:.0f} s", flush=True)
    json.dump(data, open(OUT, "w"), separators=(",", ":"))
    print(f"wrote {len(data['features'])} features to {OUT} in {time.time() - t0:.0f} s")

if __name__ == "__main__":
    main()
