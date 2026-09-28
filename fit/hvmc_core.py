# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""ctypes bridge to the plugin's own Engine (src/dsp/Engine.hpp through src/capi/hvmc_capi.cpp), for the fitting and the tests.
Build the library first: scripts/build-capi.sh (writes build/capi/libhvmc.{so,dylib}).

    m = Model()                               # calibration = the compiled defaults (fitted constants, or the priors)
    y = m.render(x, 48000, {"optical_threshold": 20, ...}, ref_names=True, profile=0)
    cal = m.cal.copy(); cal[m.field("o_n")] = 1.8; y = m.render(x, 48000, s, cal=cal)
"""
import ctypes, os, sys, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_LIB = None

def lib():
    global _LIB
    if _LIB is None:
        names = {"darwin": ("libhvmc.dylib",), "win32": ("hvmc.dll",)}.get(sys.platform, ("libhvmc.so",))   # the shared build/ may hold other platforms' libraries
        for name in names:
            p = os.path.join(ROOT, "build", "capi", name)
            if os.path.exists(p):
                _LIB = ctypes.CDLL(p)
                break
        if _LIB is None:
            sys.exit("build/capi/libhvmc not found: run scripts/build-capi.sh")
        L = _LIB
        L.hvmc_new.restype = ctypes.c_void_p
        L.hvmc_free.argtypes = [ctypes.c_void_p]
        L.hvmc_prepare.argtypes = [ctypes.c_void_p, ctypes.c_double]
        L.hvmc_set_param.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.hvmc_get_param.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fp = ctypes.POINTER(ctypes.c_float)
        L.hvmc_process.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, ctypes.c_int]
        L.hvmc_latency.argtypes = [ctypes.c_void_p]
        L.hvmc_meter.argtypes = [ctypes.c_void_p, ctypes.c_int]; L.hvmc_meter.restype = ctypes.c_double
        L.hvmc_cal_field.restype = ctypes.c_char_p
        L.hvmc_cal_field.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        L.hvmc_cal_layout_hash.restype = ctypes.c_ulonglong
        dp = ctypes.POINTER(ctypes.c_double)
        L.hvmc_cal_defaults.argtypes = [dp]; L.hvmc_cal_priors.argtypes = [dp]
        L.hvmc_set_calibration.argtypes = [ctypes.c_void_p, dp]
        L.hvmc_param_name.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        L.hvmc_param_label.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    return _LIB

RATIOS = ["1.2:1", "2:1", "3:1", "4:1", "6:1", "Flood"]
ATTACKS = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
RECOVERS = ["0.1 s", "0.25 s", "0.5 s", "0.8 s", "1.2 s", "Dual"]
CORES = ["Nickel", "Iron", "Steel", "Gold", "Uranium", "Germanium", "Plutonium"]
# the reference's settings that every protocol item starts from (fit/measure/pa.py BASE)
REF_BASE = dict(hardwire_bypass="In", mix=100.0, mode="Dual Mono", optical_bypass="Out", discrete_bypass="Out", transformer="Nickel",
                sidechain_filter="Out", optical_threshold=1, optical_gain=12, discrete_threshold=1, discrete_gain=12,
                discrete_ratio="4:1", discrete_attack=1.0, discrete_recover="0.5 s", meter_select="Output")


class Model:
    def __init__(self):
        L = lib()
        self.cal_size = L.hvmc_cal_size()
        self.fields = {}
        for i in range(L.hvmc_cal_field_count()):
            off, cnt = ctypes.c_int(), ctypes.c_int()
            name = L.hvmc_cal_field(i, ctypes.byref(off), ctypes.byref(cnt)).decode()
            self.fields[name] = (off.value, cnt.value)
        self.layout_hash = L.hvmc_cal_layout_hash()
        self.cal = np.zeros(self.cal_size)
        self.fitted = bool(L.hvmc_cal_defaults(self.cal.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
        self.priors = np.zeros(self.cal_size)
        L.hvmc_cal_priors(self.priors.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.nin = L.hvmc_num_input_params(); self.nparams = L.hvmc_num_params()
        self.names = []
        for i in range(self.nparams):
            b = ctypes.create_string_buffer(64); L.hvmc_param_name(i, b, 64); self.names.append(b.value.decode())
        self.index = {n: i for i, n in enumerate(self.names)}

    def field(self, name):
        """slice of the calibration vector for a named field"""
        off, cnt = self.fields[name]
        return slice(off, off + cnt)

    # -------------------------------------------------------------------------------------------- settings
    def ours(self, s):
        """reference-named settings (protocol items) -> our parameter positions {index: position}"""
        s = {**REF_BASE, **s}
        out = {}
        for ch, pre in ((0, "L_"), (1, "R_")):
            g = lambda k: s.get(f"{k}_{ch + 1}", s[k])
            out[self.index[pre + "optical"]] = 1 if g("optical_bypass") == "In" else 0
            out[self.index[pre + "optical_threshold"]] = int(g("optical_threshold")) - 1
            out[self.index[pre + "optical_gain"]] = int(g("optical_gain")) - 1
            out[self.index[pre + "discrete"]] = 1 if g("discrete_bypass") == "In" else 0
            out[self.index[pre + "discrete_threshold"]] = int(g("discrete_threshold")) - 1
            out[self.index[pre + "discrete_ratio"]] = RATIOS.index(g("discrete_ratio"))
            out[self.index[pre + "discrete_attack_ms"]] = ATTACKS.index(float(g("discrete_attack")))
            out[self.index[pre + "discrete_recover_s"]] = RECOVERS.index(g("discrete_recover"))
            out[self.index[pre + "discrete_gain"]] = int(g("discrete_gain")) - 1
            out[self.index[pre + "sidechain_filter"]] = 1 if g("sidechain_filter") == "In" else 0
            out[self.index[pre + "transformer"]] = CORES.index(g("transformer"))
        out[self.index["stereo"]] = 1 if s["mode"] == "Stereo" else 0
        out[self.index["hardwire_bypass"]] = 1 if s["hardwire_bypass"] == "In" else 0
        out[self.index["mix_percent"]] = int(round(float(s["mix"])))
        return out

    def render(self, x, fs, settings=None, ref_names=True, profile=0, quality=0, cal=None, extra=None):
        """x: (2, n) or (n,) float array; returns (2, n) float64 (the engine computes in double, outputs float32 like the plugin)"""
        L = lib()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x])
        x = np.ascontiguousarray(x)
        e = L.hvmc_new()
        try:
            c = np.ascontiguousarray(self.cal if cal is None else cal, dtype=np.float64)
            L.hvmc_set_calibration(e, c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            pos = self.ours(settings or {}) if ref_names else dict(settings or {})
            pos[self.index["profile"]] = profile
            pos[self.index["quality"]] = quality
            for k, v in (extra or {}).items():
                pos[self.index[k] if isinstance(k, str) else k] = v
            for k, v in pos.items():
                L.hvmc_set_param(e, int(k), int(v))
            L.hvmc_prepare(e, float(fs))
            y = np.zeros_like(x)
            fp = ctypes.POINTER(ctypes.c_float)
            L.hvmc_process(e, x[0].ctypes.data_as(fp), x[1].ctypes.data_as(fp), y[0].ctypes.data_as(fp), y[1].ctypes.data_as(fp), x.shape[1])
            return y.astype(np.float64)
        finally:
            L.hvmc_free(e)
