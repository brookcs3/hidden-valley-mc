#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Control coverage through Pedalboard: every position of every input parameter is rendered on a fixed two-tone burst (40 Hz and 1 kHz,
-30 dB for 0.5 s, full for 0.7 s, -30 dB for 0.3 s) from one base state (DUAL MONO, both stages in with moderate gain reduction, Nickel,
HARDWARE profile, STANDARD) and compared with the base render. A left-channel control must change the left output and leave the right
output bit-identical, a right-channel control the other way round (the channels are independent in DUAL MONO), a global control must
change at least one channel; in STEREO the left controls must move both channels and the right controls none. Positions that cannot be
heard from the base state are listed in NEUTRAL with the reason, and each is checked to be exactly neutral there and, where a second
context makes it live (another transformer, exhibition scale), live there:
  meter_select: routes the meter output only;  profile REFERENCE = HARDWARE with Nickel (HARDWARE adds only Iron's low-frequency lift
  and noise), live with Iron;  exhibition: scales the sub-audible material effects, so nothing to scale with a hardware core, live with
  Uranium;  temperature_c: acts on the material positions (Germanium leakage and bias, Plutonium's thermistor, Uranium's winding heat),
  nothing with Nickel, live with Germanium. (opto_memory is audible on the burst already.) In STEREO the two channels are identical
  except with Iron, whose noise generator is seeded per channel.
usage: python3 tests/pb_coverage.py [bundle] [out.json] [--survey]   (--survey only lists what is neutral, and passes)"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hvmc_pb  # noqa: E402

FS = 48000
CH = {"optical": 1, "optical_threshold": 17, "optical_gain": 10, "discrete": 1, "discrete_threshold": 15, "discrete_ratio": 2,
      "discrete_attack_ms": 2, "discrete_recover_s": 2, "discrete_gain": 6, "sidechain_filter": 1, "transformer": 0, "meter_select": 2}
BASE = {**{f"l_{k}": v for k, v in CH.items()}, **{f"r_{k}": v for k, v in CH.items()},
        "stereo": 0, "hardwire_bypass": 1, "mix_percent": 100, "profile": 1, "quality": 0, "opto_memory": 0, "temperature_c": 10, "exhibition": 0}
IRON, URANIUM, GERMANIUM = 1, 4, 5
# {parameter: (positions or "all", context that makes it live or None, reason)}
NEUTRAL = {
    "l_meter_select": ("all", None, "routes the meter output only"),
    "r_meter_select": ("all", None, "routes the meter output only"),
    "profile": ([0], {"l_transformer": IRON, "r_transformer": IRON}, "REFERENCE differs from HARDWARE only through Iron's low-frequency lift and noise"),
    "exhibition": ("all", {"l_transformer": URANIUM, "r_transformer": URANIUM}, "scales sub-audible material effects; a hardware core has none"),
    "temperature_c": ("all", {"l_transformer": GERMANIUM, "r_transformer": GERMANIUM}, "acts on the material positions only"),
}

def signal(secs=1.5, fs=FS):
    t = np.arange(int(secs * fs)) / fs
    env = np.where(t < 0.5, 10 ** (-30 / 20), np.where(t < 1.2, 1.0, 10 ** (-30 / 20)))
    x = env * (0.25 * np.sin(2 * np.pi * 40 * t) + 0.25 * np.sin(2 * np.pi * 1000 * t))
    return np.stack([x, x]).astype(np.float32)

class Renderer:
    def __init__(self, bundle):
        self.p = hvmc_pb.load(bundle); self.x = signal(); self.count = 0
    def render(self, over, x=None):
        st = {**BASE, **over}
        for k, v in st.items(): hvmc_pb.set_position(self.p, k, v)
        x = self.x if x is None else x
        self.p.process(np.zeros((2, 2048), np.float32), FS, reset=True)   # settle the settings (see tests/crosscheck.py)
        self.count += 1
        return self.p.process(x, FS, reset=True).astype(np.float64)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bundle = args[0] if args else hvmc_pb.default_bundle(); outjson = args[1] if len(args) > 1 else None
    survey = "--survey" in sys.argv
    R = Renderer(bundle); p = R.p
    inputs = [n for n in p.parameters if n not in hvmc_pb.HOST_PARAMS and not n.endswith("_db")]
    fails, neutral_found, results = [], [], {}
    def check(name, ok, detail=""):
        if survey and not ok: neutral_found.append(name + (" " + detail if detail else "")); return
        if not ok: fails.append(name + (" " + detail if detail else ""))
    t0 = time.time()
    base = R.render({})
    same = lambda a, b: np.array_equal(a, b)
    for name in inputs:
        steps = len(list(p.parameters[name].valid_values))
        side = "l" if name.startswith("l_") else "r" if name.startswith("r_") else "g"
        doc = NEUTRAL.get(name); doc_pos = set(range(steps)) if doc and doc[0] == "all" else set(doc[0]) if doc else set()
        live, quiet = [], []
        for pos in range(steps):
            if pos == BASE[name]: continue
            y = R.render({name: pos})
            dl, dr = float(np.max(np.abs(y[0] - base[0]))), float(np.max(np.abs(y[1] - base[1])))
            if side == "l": is_live = dl > 0; check(f"{name} pos {pos}: right channel untouched", dr == 0, f"{dr:.3g}")
            elif side == "r": is_live = dr > 0; check(f"{name} pos {pos}: left channel untouched", dl == 0, f"{dl:.3g}")
            else: is_live = dl > 0 or dr > 0
            (live if is_live else quiet).append(pos)
            if is_live and pos in doc_pos: check(f"{name} pos {pos} is documented neutral but changes the output", False, f"L {dl:.3g} R {dr:.3g}")
            if not is_live and pos not in doc_pos: check(f"{name} pos {pos} changes nothing from the base state", False)
        results[name] = {"positions": steps, "live": live, "neutral": quiet}
        print(f"{name:22s} {steps:3d} positions: {len(live):3d} live, {len(quiet):3d} neutral" + (f" {quiet if len(quiet) <= 8 else str(quiet[:4]) + '..'}" if quiet else ""))
    # the documented neutral positions come alive in their context
    for name, (poss, ctx, reason) in NEUTRAL.items():
        if ctx is None: continue
        steps = len(list(p.parameters[name].valid_values)); poss = range(steps) if poss == "all" else poss
        ref = R.render(ctx); ys = [R.render({**ctx, name: pos}) for pos in poss if pos != BASE[name]]
        alive = sum(not same(y, ref) for y in ys)
        check(f"{name}: documented-neutral positions are live in their context ({reason})", alive == len(ys), f"{alive} of {len(ys)} live")
        results[name]["live_in_context"] = alive
    # STEREO: the left controls govern both channels, the right controls nothing
    sbase = R.render({"stereo": 1})
    check("STEREO: both channels identical on identical input", same(sbase[0], sbase[1]))
    moved = 0; rq = 0; nl = nr = 0
    for k in CH:
        for pos in (0 if BASE[f"l_{k}"] != 0 else 1, 23 if k.endswith(("threshold", "gain")) and not k.endswith("_ratio") else None):
            if pos is None or pos == BASE[f"l_{k}"]: continue
            if k == "meter_select": continue
            y = R.render({"stereo": 1, f"l_{k}": pos}); nl += 1
            moved += (not same(y, sbase)) and (same(y[0], y[1]) or k == "transformer")   # Iron's noise is seeded per channel
            y = R.render({"stereo": 1, f"r_{k}": pos}); nr += 1; rq += same(y, sbase)
    check(f"STEREO: every left control moves both channels, identically but for Iron's per-channel noise ({nl} settings)", moved == nl, f"{moved} of {nl}")
    check(f"STEREO: no right control changes anything ({nr} settings)", rq == nr, f"{rq} of {nr}")
    print(f"{R.count} renders in {time.time() - t0:.0f} s")
    if survey:
        print("neutral or unexpected (survey):"); [print("  ", n) for n in neutral_found]
    for f in fails: print("FAIL", f)
    if outjson: json.dump({"bundle": bundle, "renders": R.count, "results": results, "fail": fails}, open(outjson, "w"), indent=1)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    sys.exit(0 if not fails else 1)

if __name__ == "__main__":
    main()
