#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Package a built bundle into one release zip per architecture, plus SHA256SUMS.

Each zip unpacks to HiddenValleyMC.vst3/ holding Contents/<arch>-linux/HiddenValleyMC.so and, in Contents/Resources/, LICENSE
and THIRD_PARTY_NOTICES.md (the notices that the licences of the bundled third-party code ask binary copies to carry). Unzipping into
~/.vst3 installs it; unzipping both architectures there gives one bundle for both.
Entries are sorted and time-stamped from SOURCE_DATE_EPOCH (else the newest input file), so the same inputs give the same zip.
The version comes from getVersion() in src/PluginHiddenValleyMC.cpp.
usage: python3 scripts/package.py [bundle dir] [dist dir]   (defaults: build/bin/HiddenValleyMC.vst3, dist/)"""
import hashlib, os, re, stat, sys, time, zipfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = "HiddenValleyMC"
bundle = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "build", "bin", NAME + ".vst3")
dist = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(ROOT, "dist")
m = re.search(r"d_version\((\d+),\s*(\d+),\s*(\d+)\)", open(os.path.join(ROOT, "src", "PluginHiddenValleyMC.cpp")).read())
version = ".".join(m.groups())
contents = os.path.join(bundle, "Contents")
arches = sorted(d[:-len("-linux")] for d in os.listdir(contents) if d.endswith("-linux") and os.path.isdir(os.path.join(contents, d)))
if not arches:
    sys.exit(f"no Contents/<arch>-linux folder in {bundle}")
resources = [("LICENSE", os.path.join(ROOT, "LICENSE")), ("THIRD_PARTY_NOTICES.md", os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"))]
inputs = [p for _, p in resources] + [os.path.join(contents, f"{a}-linux", NAME + ".so") for a in arches]
epoch = int(os.environ.get("SOURCE_DATE_EPOCH", max(int(os.stat(p).st_mtime) for p in inputs)))
stamp = time.gmtime(max(epoch, 315532800))[:6]   # zip time stamps start in 1980

def add_dir(z, name):
    zi = zipfile.ZipInfo(name + "/", stamp); zi.create_system = 3
    zi.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
    z.writestr(zi, b"")

def add_file(z, name, path, mode):
    zi = zipfile.ZipInfo(name, stamp); zi.create_system = 3; zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (stat.S_IFREG | mode) << 16
    with open(path, "rb") as f:
        z.writestr(zi, f.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

os.makedirs(dist, exist_ok=True)
sums = []
for a in arches:
    zname = f"{NAME}-{version}-linux-{a}.zip"
    top = NAME + ".vst3"
    with zipfile.ZipFile(os.path.join(dist, zname), "w") as z:
        for d in (top, f"{top}/Contents", f"{top}/Contents/Resources", f"{top}/Contents/{a}-linux"):
            add_dir(z, d)
        for rname, path in resources:
            add_file(z, f"{top}/Contents/Resources/{rname}", path, 0o644)
        add_file(z, f"{top}/Contents/{a}-linux/{NAME}.so", os.path.join(contents, f"{a}-linux", NAME + ".so"), 0o755)
    h = hashlib.sha256(open(os.path.join(dist, zname), "rb").read()).hexdigest()
    sums.append(f"{h}  {zname}\n")
    print(f"{zname}  {h}")
with open(os.path.join(dist, "SHA256SUMS"), "w") as f:
    f.writelines(sorted(sums, key=lambda s: s.split()[1]))
print("time stamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch)), "| wrote", os.path.join(dist, "SHA256SUMS"))
