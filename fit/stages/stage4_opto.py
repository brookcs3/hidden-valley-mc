# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Stage 4: the optical stage (docs/opto-fix.md). A feedback divider whose cell is lit by a panel with a hard turn-on, an idle light
leak that grows with the sidechain drive, three conductance states with a self-quenched release, and a second-order low pass in the
sidechain. The loop has no closed form, so the fit renders it; a numba mirror of OptoStage::process (validated against the C++
engine at the start of the run, and the engine is what writes the final report) makes the joint fit tractable.

4a. Stage amplifier b2, b3 from the no-GR harmonic series (threshold 1), rendered through the engine.
4b. Shape, dynamics and harmonics jointly, on the mirror: one knee level per fitted threshold position (6, 10, 14, 18, 20, 22, 24),
    the turn-on, the cell exponent, the persistence, the three states, the leak law and the release quench, on the static level
    series of those positions, the nine burst envelopes and thirteen harmonic items (1 kHz, 100 Hz, 4 kHz).
4c. Threshold table: with the shape fixed, one knee per position on its 33-level static series; position 1 compresses nothing in
    range and is placed 8 dB below position 2.
4d. Sidechain low pass corner and Q on the 3 kHz and 8 kHz static series, rendered through the engine (the Nickel path's own
    high-frequency shelf is then in the loop, as in the plugin).
Report: static residuals per position, the no-GR gain law, bursts, the 27 harmonic items, the sample-rate replicas, and the
discriminating captures (fine knee, no-GR rows, above-knee steps) when fit/data/discriminate_opto.json is present.
usage: python3 -u fit/stages/stage4_opto.py [--validate-only] [--quick]"""
import os, sys, json, time, numpy as np
from numba import njit, prange
from scipy.optimize import least_squares, minimize_scalar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import F, ITEMS, MODEL, load_cal, save_cal, render_item, feat_residual, protocol, FS, DATA  # noqa: E402

np.set_printoptions(linewidth=200, suppress=True)
FSF = float(FS)
LEVELS = list(range(-50, 15, 2))
FIT_POS = (6, 10, 14, 18, 20, 22, 24)
BURSTS = [f"opto_burst_{l}" for l in (-26, -18, -10, -2)] + [f"opto_blen_{b}" for b in (0.05, 0.2, 1.0, 4.0)] + ["opto_pulses"]
HARMS = [f"opto_harm_t{k}_{l}_f1000" for k in (14, 18, 22) for l in (-20, -10, 0)] + ["opto_harm_t18_-10_f100", "opto_harm_t22_0_f100", "opto_harm_t18_-10_f4000", "opto_harm_t22_0_f4000"]
ALL_HARMS = [i for i in ITEMS if i.startswith("opto_harm_t")]


def cidx(name, i=0): return MODEL.fields[name][0] + i


# ------------------------------------------------------------------------------------------------ the mirror
@njit(cache=True)
def lp_matched(fc, Q, fs):
    w0 = 2.0 * np.pi * fc / fs; q = 1.0 / (2.0 * Q)
    if q <= 1.0: a1 = -2.0 * np.exp(-q * w0) * np.cos(np.sqrt(1.0 - q * q) * w0)
    else: a1 = -2.0 * np.exp(-q * w0) * np.cosh(np.sqrt(q * q - 1.0) * w0)
    a2 = np.exp(-2.0 * q * w0)
    A0 = (1.0 + a1 + a2) ** 2; A1 = (1.0 - a1 + a2) ** 2; A2 = -4.0 * a2
    p1 = np.sin(w0 / 2.0) ** 2; p0 = 1.0 - p1; p2 = 4.0 * p0 * p1
    R1 = (A0 * p0 + A1 * p1 + A2 * p2) * Q * Q
    B0 = A0; B1 = (R1 - B0 * p0) / p1
    b0 = 0.5 * (np.sqrt(B0) + np.sqrt(B1)); b1 = np.sqrt(B0) - b0
    return b0, b1, 0.0, a1, a2


@njit(cache=True)
def run_one(x, n, fs, A, vth, nexp, gam, tau_el, w, tatt, trel, L0, mu, b2, b3, fc, Q, out):
    """OptoStage::process with the sidechain filter out, no hwUnit, no material options: divider v = xa / (1 + cond); sidechain
    low pass; d = A |s|; light (d - vth)^n above the turn-on; persistence; target = (L + L0)^gam; three states, quenched release"""
    kEl = 1.0 - np.exp(-1.0 / (tau_el * fs))
    kA0 = 1.0 - np.exp(-1.0 / (tatt[0] * fs)); kA1 = 1.0 - np.exp(-1.0 / (tatt[1] * fs)); kA2 = 1.0 - np.exp(-1.0 / (tatt[2] * fs))
    r0 = 1.0 / (trel[0] * fs); r1 = 1.0 / (trel[1] * fs); r2 = 1.0 / (trel[2] * fs)
    lb0, lb1, lb2, la1, la2 = lp_matched(fc, Q, fs)
    c0 = L0 ** gam
    z1 = 0.0; z2 = 0.0; L = 0.0; s0 = c0; s1 = c0; s2 = c0; cond = c0
    for i in range(n):
        xi = x[i]
        xa = xi + b2 * xi * xi + b3 * xi * xi * xi
        v = xa / (1.0 + cond)
        out[i] = v
        y = lb0 * v + z1
        z1 = lb1 * v - la1 * y + z2
        z2 = lb2 * v - la2 * y
        e = A * abs(y) - vth
        Linst = e ** nexp if e > 0.0 else 0.0
        L += (Linst - L) * kEl
        target = (L + L0) ** gam
        q0 = 1.0 - np.exp(-(1.0 + mu * s0) * r0); q1 = 1.0 - np.exp(-(1.0 + mu * s1) * r1); q2 = 1.0 - np.exp(-(1.0 + mu * s2) * r2)
        s0 += (target - s0) * (kA0 if target > s0 else q0)
        s1 += (target - s1) * (kA1 if target > s1 else q1)
        s2 += (target - s2) * (kA2 if target > s2 else q2)
        cond = w[0] * s0 + w[1] * s1 + w[2] * s2


@njit(cache=True, parallel=True)
def run_batch(X, lens, fs, A, vth, nexp, gam, tau_el, w, tatt, trel, L0, mu, b2, b3, fc, Q, out):
    for j in prange(X.shape[0]):
        run_one(X[j], lens[j], fs, A[j], vth, nexp, gam, tau_el, w, tatt, trel, L0[j], mu, b2, b3, fc, Q, out[j])


class Batch:
    """a set of items rendered together on the mirror: stimuli padded into one array, features through protocol.feature"""
    def __init__(self, ids):
        self.ids = ids
        xs = [protocol.stimulus(ITEMS[i]["stim"], ITEMS[i]["fs"])[0] for i in ids]
        self.lens = np.array([len(x) for x in xs]); self.X = np.zeros((len(ids), self.lens.max()))
        for j, x in enumerate(xs): self.X[j, :len(x)] = x
        self.Y = np.zeros_like(self.X)
        self.thr = np.array([int(ITEMS[i]["set"]["optical_threshold"]) for i in ids])

    def render(self, sh, g0_lin):
        A = np.array([10.0 ** (sh.thr_db(k) / 20.0) for k in self.thr]); L0 = np.array([sh.leak_light(k) for k in self.thr])
        run_batch(self.X, self.lens, FSF, A, sh.vth, sh.n, sh.gam, sh.tau_el, sh.w, sh.tatt, sh.trel, L0, sh.mu, sh.b2, sh.b3, sh.fc, sh.Q, self.Y)
        feats = []
        for j, i in enumerate(self.ids):
            y = self.Y[j, :self.lens[j]] * g0_lin
            feats.append(protocol.feature(ITEMS[i], np.stack([y, y]), ITEMS[i]["fs"]))
        return feats


class Shape:
    """the optical stage's parameters in the C++ calibration's terms; thr_db[k] for the 24 positions (1-based k)"""
    def __init__(self, cal):
        self.b2 = float(cal[cidx("o_b2")]); self.b3 = float(cal[cidx("o_b3")])
        self.thr = np.array(cal[MODEL.field("o_thr_db")], dtype=float).copy()
        self.n = float(cal[cidx("o_n")]); self.gam = float(cal[cidx("o_gamma")]); self.vth = float(cal[cidx("o_vth")]); self.tau_el = float(cal[cidx("o_tau_el")])
        self.w = np.array([cal[cidx("o_w", i)] for i in range(3)], dtype=float)
        self.tatt = np.array([cal[cidx("o_tatt", i)] for i in range(3)], dtype=float); self.trel = np.array([cal[cidx("o_trel", i)] for i in range(3)], dtype=float)
        self.mu = float(cal[cidx("o_rel_mu")]); self.leak = float(cal[cidx("o_leak")]); self.leak_q = float(cal[cidx("o_leak_q")])
        self.fc = float(cal[cidx("o_sc_lp_hz")]); self.Q = float(cal[cidx("o_sc_lp_q")])

    def thr_db(self, k): return float(self.thr[k - 1])
    def knee(self, k): return 20.0 * np.log10(self.vth) - self.thr_db(k)          # dBFS at the divider output
    def set_knee(self, k, knee): self.thr[k - 1] = 20.0 * np.log10(self.vth) - knee
    def cond0(self, k): return self.leak * 10.0 ** (self.leak_q * (self.thr_db(k) - self.thr_db(20)) / 20.0)
    def leak_light(self, k):
        c0 = self.cond0(k); return c0 ** (1.0 / self.gam) if c0 > 0.0 else 0.0

    def write(self, cal):
        cal[MODEL.field("o_thr_db")] = self.thr
        cal[cidx("o_n")] = self.n; cal[cidx("o_gamma")] = self.gam; cal[cidx("o_vth")] = self.vth; cal[cidx("o_tau_el")] = self.tau_el
        for i in range(3): cal[cidx("o_w", i)] = self.w[i]; cal[cidx("o_tatt", i)] = self.tatt[i]; cal[cidx("o_trel", i)] = self.trel[i]
        cal[cidx("o_rel_mu")] = self.mu; cal[cidx("o_leak")] = self.leak; cal[cidx("o_leak_q")] = self.leak_q
        cal[cidx("o_sc_lp_hz")] = self.fc; cal[cidx("o_sc_lp_q")] = self.Q; cal[cidx("o_b2")] = self.b2; cal[cidx("o_b3")] = self.b3


# 4b parameter vector: knees (7), log10 vth, gamma, log10 tau_el, w (3), log10 tatt (3), log10 trel (3), log10 leak, leak_q, log10 mu
def pack(sh):
    return np.array([sh.knee(k) for k in FIT_POS] + [np.log10(sh.vth), sh.gam, np.log10(sh.tau_el)] + list(sh.w) + list(np.log10(sh.tatt)) + list(np.log10(sh.trel)) + [np.log10(sh.leak), sh.leak_q, np.log10(sh.mu)])


def unpack(sh, p):
    s = Shape.__new__(Shape); s.__dict__.update(sh.__dict__); s.thr = sh.thr.copy()
    s.vth = 10.0 ** p[7]; s.gam = float(p[8]); s.tau_el = 10.0 ** p[9]
    w = np.abs(np.array(p[10:13])); s.w = w / w.sum()
    s.tatt = 10.0 ** np.array(p[13:16]); s.trel = 10.0 ** np.array(p[16:19])
    s.leak = 10.0 ** p[19]; s.leak_q = float(p[20]); s.mu = 10.0 ** p[21]
    # the knees are levels at the divider output; the drive table follows vth
    for k, knee in zip(FIT_POS, p[:7]): s.set_knee(k, knee)
    lo, hi = FIT_POS[0], FIT_POS[-1]
    for k in range(1, 25):                       # positions not fitted here keep their spacing from the nearest fitted ones
        if k in FIT_POS: continue
        ks = [q for q in FIT_POS]; j = np.searchsorted(ks, k)
        if j == 0: s.thr[k - 1] = s.thr[lo - 1] - (sh.thr[lo - 1] - sh.thr[k - 1])
        elif j == len(ks): s.thr[k - 1] = s.thr[hi - 1] + (sh.thr[k - 1] - sh.thr[hi - 1])
        else:
            a, b = ks[j - 1], ks[j]; t = (sh.thr[k - 1] - sh.thr[a - 1]) / max(1e-9, sh.thr[b - 1] - sh.thr[a - 1])
            s.thr[k - 1] = s.thr[a - 1] + t * (s.thr[b - 1] - s.thr[a - 1])
    return s


LO = [-45.0] * 7 + [-0.5, 0.8, -5.0] + [0.005] * 3 + [-4.0] * 3 + [-3.0] * 3 + [-4.0, 0.5, -1.0]
HI = [5.0] * 7 + [1.5, 3.0, -1.5] + [1.0] * 3 + [0.0] * 3 + [1.0] * 3 + [-0.5, 4.0, 3.0]
XS = [0.5] * 7 + [0.1, 0.05, 0.1] + [0.05] * 3 + [0.1] * 3 + [0.1] * 3 + [0.1, 0.1, 0.1]


def burst_weight(iid, n):
    ref = np.asarray(F[iid])[:n]
    return np.where(ref < ref[:20].mean() - 0.3, 1.0, 0.3) / np.sqrt(n / 100.0)


def harm_resid(iid, m, wg=2.0, w3=1.0, w5=0.3):
    ref = F[iid]
    out = [wg * (m["gain_db"] - ref["gain_db"])]
    for idx, wt in ((1, w3), (3, w5)):
        if m["h"][idx] is not None and ref["h"][idx] is not None:
            out.append(wt * (max(m["h"][idx], -110.0) - max(ref["h"][idx], -110.0)))
    return out


def run(validate_only=False, quick=False):
    t0 = time.time()
    cal = load_cal()
    G0 = float(cal[cidx("o_gain_db", 11)] + cal[cidx("x_gain_db", 0)]); g0_lin = 10.0 ** (G0 / 20.0)
    sh = Shape(cal); sh.n = 1.0; cal[cidx("o_n")] = 1.0   # the light law exponent is held at 1 (docs/opto-fix.md 2.3)
    # ---- mirror against the engine (the engine is the truth; the mirror is only a faster copy)
    chk = ["opto_static_t20_-10", "opto_static_t20_2", "opto_static_t10_-20", "opto_static_f8000_2", "opto_burst_-10", "opto_harm_t18_-10_f1000"]
    b = Batch(chk); mf = b.render(sh, g0_lin)
    worst = 0.0
    for i, m in zip(chk, mf):
        e = render_item(ITEMS[i], cal)
        if isinstance(m, dict): d = max(abs(m["gain_db"] - e["gain_db"]), abs(m["h"][1] - e["h"][1]))
        elif isinstance(m, list): d = float(np.max(np.abs(np.array(m) - np.array(e)[:len(m)])))
        else: d = abs(m - e)
        if "f8000" not in i: worst = max(worst, d)   # the 8 kHz item carries the Nickel path's HF shelf, which only the engine has
        print(f"  mirror vs engine {i}: max |diff| {d:.4f} dB")
    if worst > 0.02:
        print("!! the mirror does not reproduce the engine; stopping"); sys.exit(1)
    if validate_only: return
    # ---- 4a stage amplifier
    ids = [f"opto_harm_nogr_{l}" for l in (-30, -20, -10, 0, 6, 12)]
    def rh(p):
        c = cal.copy(); c[cidx("o_b2")] = p[0]; c[cidx("o_b3")] = p[1]
        return np.concatenate([feat_residual(i, render_item(ITEMS[i], c))[:3] for i in ids])
    r = least_squares(rh, [sh.b2, sh.b3], bounds=([-0.05, -0.05], [0.05, 0.05]), x_scale=[1e-5, 1e-4], diff_step=1e-3)
    sh.b2, sh.b3 = float(r.x[0]), float(r.x[1]); cal[cidx("o_b2")] = sh.b2; cal[cidx("o_b3")] = sh.b3
    print(f"  4a stage amplifier: b2 {sh.b2:.3e}, b3 {sh.b3:.3e}, rms {np.sqrt(np.mean(r.fun ** 2)):.2f} dB  [{time.time() - t0:.0f} s]")
    # ---- 4b joint fit on the mirror
    levels = LEVELS[::2] if quick else LEVELS
    stat_ids = [f"opto_static_t{k}_{l}" for k in FIT_POS for l in levels]
    bs, bb, bh = Batch(stat_ids), Batch(BURSTS), Batch(HARMS)
    def resid(p):
        s = unpack(sh, p)
        out = [2.0 * (np.array(bs.render(s, g0_lin)) - np.array([F[i] for i in stat_ids]))]
        for i, m in zip(BURSTS, bb.render(s, g0_lin)):
            n = min(len(m), len(F[i])); out.append(burst_weight(i, n) * (np.array(m[:n]) - np.array(F[i][:n])))
        for i, m in zip(HARMS, bh.render(s, g0_lin)): out.append(np.array(harm_resid(i, m)))
        return np.concatenate(out)
    starts = [pack(sh)]
    phys = pack(sh).copy(); phys[7] = np.log10(3.0); phys[8] = 1.63; phys[10:13] = [0.05, 0.70, 0.25]; phys[13:16] = np.log10([1.0, 0.007, 0.05]); phys[16:19] = np.log10([1.0, 0.007, 0.05])
    starts.append(phys)
    best = None
    for si, p0 in enumerate(starts):
        p0 = np.clip(p0, LO, HI)
        rr = least_squares(resid, p0, bounds=(LO, HI), x_scale=XS, diff_step=2e-3, max_nfev=20 if quick else 60, loss="soft_l1", f_scale=1.0)
        print(f"  4b start {si}: cost {rr.cost:.2f}, nfev {rr.nfev}  [{time.time() - t0:.0f} s]")
        if best is None or rr.cost < best.cost: best = rr
    rr = least_squares(resid, best.x, bounds=(LO, HI), x_scale=XS, diff_step=5e-4, max_nfev=30 if quick else 120, xtol=1e-10, ftol=1e-6)
    if rr.cost < best.cost or True: best = rr
    sh = unpack(sh, best.x)
    ns = len(stat_ids); res = best.fun
    print(f"  4b: vth {sh.vth:.3f} gamma {sh.gam:.3f} tau_el {sh.tau_el * 1e3:.4f} ms | w {np.round(sh.w, 3)} attack ms {np.round(sh.tatt * 1e3, 2)} release ms {np.round(sh.trel * 1e3, 1)} mu {sh.mu:.2f} | leak {sh.leak:.4f} q {sh.leak_q:.3f}")
    print(f"      knees dBFS: {dict(zip(FIT_POS, np.round([sh.knee(k) for k in FIT_POS], 2)))}")
    print(f"      statics rms {np.sqrt(np.mean((res[:ns] / 2.0) ** 2)):.3f} dB max {np.max(np.abs(res[:ns] / 2.0)):.2f} | bursts weighted rms {np.sqrt(np.mean(res[ns:ns + sum(min(len(F[i]), b_) for i, b_ in zip(BURSTS, bb.lens // 48))] ** 2)):.3f} | cost {best.cost:.2f}  [{time.time() - t0:.0f} s]")
    # ---- 4c threshold table
    per_pos = {}
    for k in range(2, 25):
        ids_k = [f"opto_static_t{k}_{l}" for l in LEVELS]; bk = Batch(ids_k); ref = np.array([F[i] for i in ids_k])
        def cost(knee):
            s = Shape.__new__(Shape); s.__dict__.update(sh.__dict__); s.thr = sh.thr.copy(); s.set_knee(k, knee)
            return float(np.sum((np.array(bk.render(s, g0_lin)) - ref) ** 2))
        r1 = minimize_scalar(cost, bounds=(-45.0, 40.0), method="bounded", options={"xatol": 1e-3})
        sh.set_knee(k, r1.x)
        m = np.array(bk.render(sh, g0_lin)); e = m - ref; comp = ref < ref[0] - 0.3
        per_pos[k] = (float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e))), float(np.sqrt(np.mean(e[comp] ** 2))) if comp.any() else 0.0, float(m[0] - ref[0]))
    sh.thr[0] = sh.thr[1] - 8.0
    print("  4c threshold drives (dB):", np.round(sh.thr, 2))
    print("     steps (dB):", np.round(np.diff(sh.thr), 2))
    print("     per position rms / max / compressing rms / no-GR gain error:")
    for k in range(2, 25): print(f"       {k:2d}: {per_pos[k][0]:.3f} / {per_pos[k][1]:.2f} / {per_pos[k][2]:.3f} / {per_pos[k][3]:+.3f}")
    print(f"     family rms {np.sqrt(np.mean([per_pos[k][0] ** 2 for k in per_pos])):.3f}, worst max {max(per_pos[k][1] for k in per_pos):.2f}  [{time.time() - t0:.0f} s]")
    sh.write(cal)
    # ---- 4d sidechain low pass, through the engine
    lp_ids = [f"opto_static_f{f}_{l}" for f in (3000, 8000) for l in range(-40, 11, 4)]
    def rl(p):
        c = cal.copy(); c[cidx("o_sc_lp_hz")] = p[0]; c[cidx("o_sc_lp_q")] = p[1]
        return np.array([(0.3 if (i.startswith("opto_static_f8000") and int(i.rsplit("_", 1)[1]) > 4) else 1.0) * (render_item(ITEMS[i], c) - F[i]) for i in lp_ids])
    r2 = least_squares(rl, [sh.fc, sh.Q], bounds=([3000.0, 0.5], [9000.0, 1.0]), x_scale=[500.0, 0.1], diff_step=1e-3, max_nfev=40)
    sh.fc, sh.Q = float(r2.x[0]), float(r2.x[1]); sh.write(cal)
    print(f"  4d sidechain low pass: {sh.fc:.0f} Hz, Q {sh.Q:.3f}, 3 kHz + 8 kHz statics rms {np.sqrt(np.mean(r2.fun ** 2)):.3f} dB  [{time.time() - t0:.0f} s]")
    save_cal(cal, {"stage4": "optical stage (docs/opto-fix.md): amplifier terms (opto_harm_nogr_*), joint shape/dynamics/harmonics on the mirror (opto_static_t{6,10,14,18,20,22,24}_*, opto_burst_*, opto_blen_*, opto_pulses, 13 opto_harm_t* items), knee per position (opto_static_t*), sidechain low pass (opto_static_f3000_*, f8000_*)"},
             fields=["o_thr_db", "o_n", "o_gamma", "o_vth", "o_tau_el", "o_w", "o_tatt", "o_trel", "o_rel_mu", "o_leak", "o_leak_q", "o_sc_lp_hz", "o_sc_lp_q", "o_b2", "o_b3"])
    # ---- report, through the engine
    for k in (12, 20):
        e = np.array([render_item(ITEMS[f"opto_static_t{k}_{l}"], cal) - F[f"opto_static_t{k}_{l}"] for l in LEVELS])
        print(f"  engine static t{k}: rms {np.sqrt(np.mean(e ** 2)):.3f} max {np.max(np.abs(e)):.2f}")
    e = np.array([render_item(ITEMS[f"opto_static_f100_{l}"], cal) - F[f"opto_static_f100_{l}"] for l in range(-40, 11, 4)])
    print(f"  engine static 100 Hz: rms {np.sqrt(np.mean(e ** 2)):.3f} max {np.max(np.abs(e)):.2f}")
    for iid in BURSTS + ["opto_burst_thr12", "sr_opto_44100", "sr_opto_96000"]:
        if iid not in ITEMS: continue
        e = feat_residual(iid, render_item(ITEMS[iid], cal))
        print(f"  engine {iid}: rms {np.sqrt(np.mean(e ** 2)):.3f} dB, max {np.max(np.abs(e)):.2f}")
    h3 = []; h5 = []
    for iid in ALL_HARMS:
        m = render_item(ITEMS[iid], cal); ref = F[iid]
        if m["h"][1] is not None and ref["h"][1] is not None and ref["h"][1] > -100: h3.append(m["h"][1] - ref["h"][1])
        if m["h"][3] is not None and ref["h"][3] is not None and ref["h"][3] > -100: h5.append(m["h"][3] - ref["h"][3])
        print(f"  engine {iid}: gain {m['gain_db']:.2f}/{ref['gain_db']:.2f} H3 {m['h'][1]:.1f}/{ref['h'][1]:.1f} H5 {m['h'][3]:.1f}/{ref['h'][3]:.1f}")
    print(f"  harmonics over {len(h3)} items: H3 error rms {np.sqrt(np.mean(np.square(h3))):.1f} dB, H5 rms {np.sqrt(np.mean(np.square(h5))):.1f} dB")
    disc = os.path.join(DATA, "discriminate_opto.json")
    if os.path.exists(disc):
        D = json.load(open(disc))
        for thr, kn in D["knee"].items():
            lv = np.array(kn["levels"]); ref = np.array(kn["gain_db"])
            m = np.array([MODEL.render(protocol.stimulus({"kind": "sine", "level": float(l), "f": 1000.0, "secs": 3.0}, FS), FS, {"optical_bypass": "In", "optical_threshold": int(thr)}, cal=cal)[0] for l in lv])
            g = np.array([20 * np.log10(np.abs(protocol.lockin(mm, 1000.0, FS, len(mm) - FS, len(mm)))) - l for mm, l in zip(m, lv)])
            print(f"  (A) fine knee, threshold {thr}: rms {np.sqrt(np.mean((g - ref) ** 2)):.3f} dB, max {np.max(np.abs(g - ref)):.2f}")
        row = D["nogr"]["-50.0"]
        errs = {k: round(float(MODEL.render(protocol.stimulus({"kind": "sine", "level": -50.0, "f": 1000.0, "secs": 2.0}, FS), FS, {"optical_bypass": "In", "optical_threshold": int(k)}, cal=cal)[0][-FS:].std() * np.sqrt(2) / 10 ** (-50 / 20)) and 0, 3) for k in row}
        g = {k: round(float(20 * np.log10(np.abs(protocol.lockin(MODEL.render(protocol.stimulus({"kind": "sine", "level": -50.0, "f": 1000.0, "secs": 2.0}, FS), FS, {"optical_bypass": "In", "optical_threshold": int(k)}, cal=cal)[0], 1000.0, FS, FS, 2 * FS))) + 50 - v), 3) for k, v in row.items()}
        print("  (D) no-GR gain model - reference by threshold at -50 dBFS:", g)
        for key, st in D["steps"].items():
            a, b_ = st["from"], st["to"]; n = int(2.0 * FS); t = np.arange(3 * n) / FS
            env = np.full(3 * n, 10 ** (a / 20)); env[n:2 * n] = 10 ** (b_ / 20)
            x = env * np.sin(2 * np.pi * 1000.0 * t)
            y = MODEL.render(np.stack([x, x]), FS, {"optical_bypass": "In", "optical_threshold": 20}, cal=cal)[0]
            per = 48; mm = len(x) // per
            gm = 20 * np.log10(np.sqrt(np.mean(y[:mm * per].reshape(mm, per) ** 2, axis=1)) / np.sqrt(np.mean(x[:mm * per].reshape(mm, per) ** 2, axis=1)))
            gr = np.array(st["per_cycle_gain_db"]); nn = min(len(gm), len(gr))
            rel = gm[2 * n // per:nn] - gr[2 * n // per:nn]; att = gm[n // per:2 * n // per] - gr[n // per:2 * n // per]
            print(f"  (E) step {a:+.0f}->{b_:+.0f}->{a:+.0f}: attack rms {np.sqrt(np.mean(att ** 2)):.3f} max {np.max(np.abs(att)):.2f} | release rms {np.sqrt(np.mean(rel ** 2)):.3f} max {np.max(np.abs(rel)):.2f} | tail at +0.5 s {rel[500] if len(rel) > 500 else float('nan'):+.3f} dB")
    print(f"  done [{time.time() - t0:.0f} s]")

if __name__ == "__main__":
    run(validate_only="--validate-only" in sys.argv, quick="--quick" in sys.argv)
