# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Run the whole fit: build the C interface to the engine, run the stages in order (each writes its fields into fit/data/constants.json
and its log into build/fit/stageN.log), then write src/dsp/FittedConstants.hpp.
usage: python3 fit/run_all.py                 all stages, then the header
       python3 fit/run_all.py --from 3        stages 3, 4, 5, then the header
       python3 fit/run_all.py --only 2        one stage, then the header
       python3 fit/run_all.py --capture       first re-measure the reference plug-in (needs the licensed reference installed;
                                              fit/measure/capture.py) and the discriminating tests (fit/measure/discriminate.py)
       python3 fit/run_all.py --check         no fitting: exit 1 if the header on disk is not what constants.json produces
The stages take several hours in total (stage 3 and 4 render the engine thousands of times); the fit is deterministic."""
import os, sys, subprocess, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STAGES = ["stage1_laws.py", "stage2_transformers.py", "stage3_discrete.py", "stage4_opto.py", "stage5_routing.py"]


def sh(cmd, log=None):
    print("==", " ".join(cmd), flush=True)
    if log:
        with open(log, "w") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                sys.stdout.write(line); f.write(line)
            p.wait()
            rc = p.returncode
    else:
        rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f"!! failed ({rc}): {' '.join(cmd)}"); sys.exit(rc)


def main():
    argv = sys.argv[1:]
    if "--check" in argv:
        sh([sys.executable, "fit/make_constants.py", "--check"]); return
    sh(["bash", "scripts/build-capi.sh"])
    os.makedirs(os.path.join(ROOT, "build", "fit"), exist_ok=True)
    if "--capture" in argv:
        sh([sys.executable, "-u", "fit/measure/capture.py"], os.path.join(ROOT, "build", "fit", "capture.log"))
        sh([sys.executable, "-u", "fit/measure/discriminate.py"], os.path.join(ROOT, "build", "fit", "discriminate.log"))
    first, only = 1, None
    if "--from" in argv: first = int(argv[argv.index("--from") + 1])
    if "--only" in argv: only = int(argv[argv.index("--only") + 1])
    for n, s in enumerate(STAGES, 1):
        if (only is not None and n != only) or (only is None and n < first):
            continue
        t0 = time.time()
        sh([sys.executable, "-u", os.path.join("fit", "stages", s)], os.path.join(ROOT, "build", "fit", f"stage{n}.log"))
        print(f"== stage {n} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    sh([sys.executable, "fit/make_constants.py"])

if __name__ == "__main__":
    main()
