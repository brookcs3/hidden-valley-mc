#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Package the built bundles into release zips, plus SHA256SUMS. Each build that is present is packaged:
  Linux    build/bin/HiddenValleyMC.vst3 (scripts/build.sh, scripts/docker-build.sh): one zip per architecture,
           HiddenValleyMC-<version>-linux-<arch>.zip, unpacking to HiddenValleyMC.vst3/ with Contents/<arch>-linux/
           HiddenValleyMC.so. Unzipping into ~/.vst3 installs it; unzipping both architectures there gives one bundle for both.
  macOS    build/macos/bin/HiddenValleyMC.vst3, .clap and .component (scripts/build-macos.sh): HiddenValleyMC-<version>-macos-universal.zip,
           both bundles exactly as built and signed (they already carry LICENSE and THIRD_PARTY_NOTICES.md, inside the signature).
  Windows  build/windows/bin/HiddenValleyMC.vst3 (scripts/build-windows.sh, scripts/docker-build.sh windows-x64):
           HiddenValleyMC-<version>-windows-x64.zip, unpacking to HiddenValleyMC.vst3/ with Contents/x86_64-win/HiddenValleyMC.vst3.
The Linux and Windows zips get LICENSE and THIRD_PARTY_NOTICES.md in the bundle's Contents/Resources/ (the notices that the licences of
the bundled third-party code ask binary copies to carry). Entries are sorted and time-stamped from SOURCE_DATE_EPOCH (else the newest
input file), so the same inputs give the same zip. SHA256SUMS lists every HiddenValleyMC-<version>-*.zip in the dist folder.
The version comes from getVersion() in src/PluginHVMC.cpp.
usage: python3 scripts/package.py [dist dir]   (default dist/)"""
import hashlib, os, re, stat, sys, time, zipfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = "HiddenValleyMC"
dist = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "dist")
m = re.search(r"d_version\((\d+),\s*(\d+),\s*(\d+)\)", open(os.path.join(ROOT, "src", "PluginHVMC.cpp")).read())
version = ".".join(m.groups())
resources = [("LICENSE", os.path.join(ROOT, "LICENSE")), ("THIRD_PARTY_NOTICES.md", os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"))]
top = NAME + ".vst3"

linux = os.path.join(ROOT, "build", "bin", top)
linux_arches = []
if os.path.isdir(os.path.join(linux, "Contents")):
    linux_arches = sorted(d[:-len("-linux")] for d in os.listdir(os.path.join(linux, "Contents"))
                          if d.endswith("-linux") and os.path.isdir(os.path.join(linux, "Contents", d)))
macos = [os.path.join(ROOT, "build", "macos", "bin", NAME + ext) for ext in (".vst3", ".clap", ".component")]
have_macos = all(os.path.isfile(os.path.join(b, "Contents", "MacOS", NAME)) for b in macos)
windows_dll = os.path.join(ROOT, "build", "windows", "bin", top, "Contents", "x86_64-win", top)
have_windows = os.path.isfile(windows_dll)
if not (linux_arches or have_macos or have_windows):
    sys.exit("no build to package: build/bin (Linux), build/macos/bin (macOS) or build/windows/bin (Windows)")

def tree(path):
    """every directory and file under path, sorted, as paths relative to path's parent; no symlinks expected in a bundle"""
    out = []
    for d, dirs, files in os.walk(path):
        dirs.sort()
        out.append(d)
        for f in sorted(files):
            p = os.path.join(d, f)
            if os.path.islink(p):
                sys.exit(f"unexpected symlink in a bundle: {p}")
            out.append(p)
    return out

inputs = [p for _, p in resources]
inputs += [os.path.join(linux, "Contents", f"{a}-linux", NAME + ".so") for a in linux_arches]
if have_macos:
    inputs += [p for b in macos for p in tree(b) if os.path.isfile(p)]
if have_windows:
    inputs.append(windows_dll)
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

def add_bundle_with_notices(z, binary_dir, binary_path):
    """a one-binary bundle (Linux, Windows) with the notices added in Contents/Resources"""
    for d in (top, f"{top}/Contents", f"{top}/Contents/Resources", f"{top}/Contents/{binary_dir}"):
        add_dir(z, d)
    for rname, path in resources:
        add_file(z, f"{top}/Contents/Resources/{rname}", path, 0o644)
    add_file(z, f"{top}/Contents/{binary_dir}/{os.path.basename(binary_path)}", binary_path, 0o755)

def add_tree(z, path):
    """a bundle exactly as it is on disk (macOS: the signature covers every file), executables 0755, the rest 0644"""
    base = os.path.dirname(path)
    for p in tree(path):
        name = os.path.relpath(p, base).replace(os.sep, "/")
        if os.path.isdir(p):
            add_dir(z, name)
        else:
            add_file(z, name, p, 0o755 if os.stat(p).st_mode & stat.S_IXUSR else 0o644)

os.makedirs(dist, exist_ok=True)
written = []
for a in linux_arches:
    zname = f"{NAME}-{version}-linux-{a}.zip"
    with zipfile.ZipFile(os.path.join(dist, zname), "w") as z:
        add_bundle_with_notices(z, f"{a}-linux", os.path.join(linux, "Contents", f"{a}-linux", NAME + ".so"))
    written.append(zname)
if have_macos:
    zname = f"{NAME}-{version}-macos-universal.zip"
    with zipfile.ZipFile(os.path.join(dist, zname), "w") as z:
        for b in macos:
            add_tree(z, b)
    written.append(zname)
if have_windows:
    zname = f"{NAME}-{version}-windows-x64.zip"
    with zipfile.ZipFile(os.path.join(dist, zname), "w") as z:
        add_bundle_with_notices(z, "x86_64-win", windows_dll)
    written.append(zname)

def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

for zname in written:
    print(f"{zname}  {sha256(os.path.join(dist, zname))}")
zips = sorted(f for f in os.listdir(dist) if f.startswith(f"{NAME}-{version}-") and f.endswith(".zip"))
with open(os.path.join(dist, "SHA256SUMS"), "w") as f:
    f.writelines(f"{sha256(os.path.join(dist, z))}  {z}\n" for z in zips)
print("time stamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch)), "| wrote", os.path.join(dist, "SHA256SUMS"),
      f"({len(zips)} zips)")
