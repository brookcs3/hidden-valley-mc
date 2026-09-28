# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Lets the tests load the macOS Audio Unit (HiddenValleyMC.component) from wherever it was built.
Pedalboard, like other AU hosts, finds an Audio Unit through the system's component registry, which lists only the bundles installed in
~/Library/Audio/Plug-Ins/Components or /Library/Audio/Plug-Ins/Components. For a bundle anywhere else, register(bundle) loads the
bundle's own binary into this process and registers its factory with AudioComponentRegister, under the type, subtype and manufacturer
in its Info.plist; pedalboard.load_plugin(bundle) then instantiates that binary. Nothing is installed and nothing outlives the process.
It does nothing for a .vst3 bundle, on other systems, or for a bundle that is already in one of the two folders. If the system already
has a component with the same codes (an installed copy), the bundle is registered all the same and must then be the first match for its
codes, which is the one hosts (Pedalboard among them) instantiate; if it is not, this stops with an error rather than let the tests
load the installed copy."""
import ctypes, os, plistlib, sys

_keep = []   # the loaded binaries and registered components, alive until the process exits


class _Desc(ctypes.Structure):
    _fields_ = [("componentType", ctypes.c_uint32), ("componentSubType", ctypes.c_uint32), ("componentManufacturer", ctypes.c_uint32),
                ("componentFlags", ctypes.c_uint32), ("componentFlagsMask", ctypes.c_uint32)]


def _fourcc(s):
    return int.from_bytes(s.encode("ascii"), "big")


_registered = set()


def register(bundle):
    bundle = os.path.abspath(os.path.expanduser(bundle)).rstrip("/")
    if sys.platform != "darwin" or not bundle.endswith(".component") or bundle in _registered:
        return
    _registered.add(bundle)
    import pwd   # Unix only: imported here so Windows can import this module
    home = pwd.getpwuid(os.getuid()).pw_dir   # the real home: the registry does not follow $HOME
    folders = {os.path.join(home, "Library/Audio/Plug-Ins/Components"), "/Library/Audio/Plug-Ins/Components"}
    if os.path.dirname(bundle) in folders:
        return
    info = plistlib.load(open(os.path.join(bundle, "Contents", "Info.plist"), "rb"))
    ac = info["AudioComponents"][0]
    desc = _Desc(_fourcc(ac["type"]), _fourcc(ac["subtype"]), _fourcc(ac["manufacturer"]), 0, 0)
    at = ctypes.CDLL("/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    at.AudioComponentFindNext.restype = ctypes.c_void_p
    at.AudioComponentFindNext.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Desc)]
    at.AudioComponentRegister.restype = ctypes.c_void_p
    at.AudioComponentRegister.argtypes = [ctypes.POINTER(_Desc), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]

    def found():
        out, c = [], None
        while True:
            c = at.AudioComponentFindNext(c, ctypes.byref(desc))
            if not c:
                return out
            out.append(c)

    codes = f"{ac['type']} {ac['subtype']} {ac['manufacturer']}"
    installed = len(found())
    lib = ctypes.CDLL(os.path.join(bundle, "Contents", "MacOS", info["CFBundleExecutable"]))
    factory = ctypes.cast(lib[ac["factoryFunction"]], ctypes.c_void_p).value
    name = cf.CFStringCreateWithCString(None, ac["name"].encode("utf-8"), 0x08000100)   # kCFStringEncodingUTF8
    comp = at.AudioComponentRegister(ctypes.byref(desc), name, int(ac["version"]), factory)
    now = found()
    # macOS may already list a bundle it has seen (LaunchServices registers a bundle when a host loads it from any path) and then
    # replaces that entry with this registration instead of adding one, so the count is not a reliable check: what matters is that
    # the registered component exists and is the first match for its codes
    if not comp or comp not in now:
        sys.exit(f"could not register the Audio Unit {codes} from {bundle}")
    if now[0] != comp:
        sys.exit(f"an installed Audio Unit with the codes {codes} comes before {bundle}, so Pedalboard would load that one. "
                 "Test the installed copy by its path, or remove it first.")
    _keep.extend([lib, comp])
    note = f" (ahead of {installed} installed copy with the same codes)" if installed else ""
    print(f"registered the Audio Unit {codes} from {bundle} in this process{note}")
