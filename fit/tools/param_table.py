# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Print the plugin's parameter table as markdown, straight from the compiled tables (src/HVMCParams.hpp through the C interface),
so README.md cannot drift from the plugin: python3 fit/tools/param_table.py  (needs scripts/build-capi.sh)"""
import os, sys, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hvmc_core  # noqa: E402

L = hvmc_core.lib()
L.hvmc_param_description.restype = ctypes.c_char_p
L.hvmc_param_description.argtypes = [ctypes.c_int]
buf = ctypes.create_string_buffer(64)


def name(i):
    L.hvmc_param_name(i, buf, 64); return buf.value.decode()


def label(i, v):
    L.hvmc_param_label(i, v, buf, 64); return buf.value.decode()


def positions(i):
    n = L.hvmc_param_steps(i)
    labs = [label(i, v) for v in range(n)]
    if n > 8 and all(l.lstrip("-").isdigit() for l in labs):
        return f"{labs[0]} to {labs[-1]} ({n} steps)"
    return " / ".join(labs)


def main():
    nin = L.hvmc_num_input_params(); nall = L.hvmc_num_params()
    print("| # | parameter | positions | default | what it does |")
    print("|---|---|---|---|---|")
    for i in range(nin):
        if i >= 12 and i < 24:
            continue   # the right channel repeats the left with the R_ prefix
        nm = name(i)
        if nm.startswith("L_"):
            nm = nm[2:] + " (L_ and R_)"
        print(f"| {i} | `{nm}` | {positions(i)} | {label(i, L.hvmc_param_default(i))} | {L.hvmc_param_description(i).decode()} |")
    print()
    print("Read-only outputs (the host shows them as meters):")
    print()
    print("| # | output | what it reads |")
    print("|---|---|---|")
    for i in range(nin, nall):
        print(f"| {i} | `{name(i)}` | {L.hvmc_param_description(i).decode()} |")

if __name__ == "__main__":
    main()
