# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Shared helpers for the fitting stages: the reference features, the protocol, rendering an item through the model, and residuals
between model and reference features in dB."""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "measure"))
import hvmc_core, protocol  # noqa: E402

DATA = os.path.join(HERE, "data")
REF = json.load(open(os.path.join(DATA, "reference_features.json")))
F = REF["features"]
ITEMS = {it["id"]: it for it in protocol.items()}
MODEL = hvmc_core.Model()
FS = protocol.FS


def render_item(item, cal, profile=0, quality=0):
    x = protocol.stimulus(item["stim"], item["fs"])
    y = MODEL.render(x, item["fs"], item["set"], cal=cal, profile=profile, quality=quality)
    return protocol.feature(item, y, item["fs"])


def feat_residual(item_id, model_feat, floor=-120.0, hweights=(1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.25), gain_w=5.0):
    """dB residual vector between a model feature and the reference feature of the same item"""
    ref = F[item_id]; t = ITEMS[item_id]["feat"]["type"]
    if t == "gain_db":
        return np.array([model_feat - ref])
    if t == "env":
        n = min(len(ref), len(model_feat))
        return np.asarray(model_feat[:n]) - np.asarray(ref[:n])
    if t == "harm":
        r = [gain_w * (model_feat["gain_db"] - ref["gain_db"])]
        for k, (a, b) in enumerate(zip(model_feat["h"], ref["h"])):
            if a is None or b is None:
                continue
            a, b = max(a, floor), max(b, floor)
            if a <= floor and b <= floor:
                r.append(0.0); continue
            r.append(hweights[k] * (a - b) * (0.3 if b < -110 else 1.0))
        return np.array(r)
    if t == "resp":
        return np.array([0.0 if (a is None or b is None) else a - b for a, b in zip(model_feat, ref)])
    raise ValueError(t)


def load_cal(path=os.path.join(DATA, "constants.json")):
    """the current fitted values (a dict name -> list), applied over the priors"""
    cal = MODEL.priors.copy()
    if os.path.exists(path):
        d = json.load(open(path))
        for name, vals in d["values"].items():
            if name not in MODEL.fields:
                raise KeyError(f"{path}: field {name} is not in the engine's calibration layout (fields renamed? rerun the stage that writes it)")
            off, cnt = MODEL.fields[name]
            if len(vals) != cnt:
                raise ValueError(f"{path}: field {name} has {len(vals)} values, the engine expects {cnt}")
            cal[off:off + cnt] = np.asarray(vals, dtype=float)
    return cal


def save_cal(cal, notes, fields=None, path=os.path.join(DATA, "constants.json")):
    """write the named fields (all if None) into constants.json; other fields keep the file's values, so stages can run in any order"""
    if fields is not None:
        unknown = set(fields) - set(MODEL.fields)
        if unknown:
            raise KeyError(f"save_cal: not calibration fields: {sorted(unknown)}")
    old = json.load(open(path)) if os.path.exists(path) else {"values": {}, "notes": {}}
    for name, (off, cnt) in MODEL.fields.items():
        if fields is not None and name not in fields:
            continue
        old["values"][name] = [float(v) for v in cal[off:off + cnt]]
    old["notes"].update(notes)
    old["layout_hash"] = str(MODEL.layout_hash)
    old["reference"] = {k: REF["meta"][k] for k in ("reference", "version", "sha256")}
    json.dump(old, open(path, "w"), indent=1, sort_keys=True)
