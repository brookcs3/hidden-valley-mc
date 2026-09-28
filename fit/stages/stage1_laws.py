# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Stage 1: the switch laws that are measured directly. The make-up gain of every OPTICAL GAIN and DISCRETE GAIN position is the
measured gain of the unit at that position (stage in, threshold 1, a level far below any compression) minus the gain of the Nickel
transformer path with both stages out. The Nickel midband gain itself comes from the transformer stage (stage 2)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import F, MODEL, load_cal, save_cal  # noqa: E402

def run():
    cal = load_cal()
    base = F["law_core_gain_Nickel"]
    o = [F[f"law_opt_gain_{g}"] - base for g in range(1, 25)]
    d = [F[f"law_disc_gain_{g}"] - base for g in range(1, 25)]
    cal[MODEL.field("o_gain_db")] = o
    cal[MODEL.field("d_gain_db")] = d
    save_cal(cal, {"stage1": "make-up laws from law_opt_gain_*, law_disc_gain_* minus law_core_gain_Nickel"}, fields=["o_gain_db", "d_gain_db"])
    print("optical gain dB:", np.round(o, 2)); print("discrete gain dB:", np.round(d, 2))

if __name__ == "__main__":
    run()
