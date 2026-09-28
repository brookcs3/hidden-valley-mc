# Detector fix: the discrete stage's storage node

Status: design document, 2026-09-27. Synthesises the four candidate harnesses under `fit/tools/candidates/` and the verified
re-runs of their winners. Nothing in `src/` or `fit/` has been changed yet; this document says exactly what to change.

The problem: `DiscreteStage::process` (mode E in the harnesses: log rectifier, attack one-pole toward the rectified level, release
one-pole toward the threshold reference, the two branches mutually exclusive) reproduces the burst envelopes but not the
steady-state table against attack and recover. At 30 ms attack / 0.1 s recover it has 1.25 dB too much gain reduction; overall
0.29 dB rms on the steady cells, 0.046 weighted rms on the bursts. Four harnesses fitted 60-odd topologies on the same
residual; three independent winners emerged and were reproduced from their documented starts. This document adopts the best
of them and specifies the code, the calibration and the fit.

Numbers below are the harness residual (steady cells weighted 3, bursts weighted per sample; "cost" is `least_squares`'s
soft-L1 cost), on the stage-3 curves and thresholds as they are in `fit/data/constants.json` now. "Steady" is rms / max over the
54 steady cells in dB of gain; "bursts" is weighted rms / max over the 33 burst envelopes (DUAL items are not in the harnesses;
see section 5).

## 1. The winning topology

### 1.1 In words

Keep the log rectifier and the one storage node in dB. Two changes to the node:

1. **The release conductance is always on.** The node is a capacitor with a bleed resistor to the threshold reference; the
   bleed does not switch off while the attack diode conducts. This is the physical parallel RC (diode + attack resistor charging
   a capacitor that has a resistor across it to a reference), and it alone takes the steady rms from 0.29 to 0.175 dB and the
   bursts from 0.046 to 0.011 (mode 15 in `mixed-domain.py`, 20 in `cascaded-and-decoupled.py`, 21 in
   `rate-limited-and-diode.py`, 24 in `rms-and-power-domain.py`: all four harnesses found it independently).

2. **The attack conductance grows with the rectified level above the reference.** The charging rate is
   `kA (e - v) (1 + max(e - rest, 0) / Sv)` with `Sv = 17.4 dB`: a signal 17.4 dB above the reference charges twice as fast as one
   at the reference. This is the part that fixes the level dependence (fact 5) that the plain parallel RC over-predicts
   (-0.74 / +0.65 dB at -25 / +2 dBFS for 30 ms) and it takes the steady rms from 0.175 to 0.079 dB (mode 30 in `mixed-domain.py`).

Everything else stays: full-wave rectifier, log floor at -100 dB, the fixed curve per ratio read at `v - T`, the make-up offset
added to the rectified level, the DUAL second node coupled to the first, the optional germanium leak.

### 1.2 Equations

Per sample, `e` the rectified log level (dB), `T` the threshold (dB), `rest = T - depth` the release reference:

```
dv  = (rest - v) * kR                                   release: always
if e > v:
    f  = 1 + max(e - rest, 0) / Sv                      attack conductance factor
    dv += (e - v) * kA(f)                                attack: while the diode conducts
v  += dv
GR  = C_r(v - T)
```

with `kR = 1 - exp(-1 / (tr fs))` and, in continuous time, an attack rate `f / ta`. The harnesses used `kA(f) = kA * f`
with `kA = 1 - exp(-1 / (ta fs))`; section 2.7 recommends the exact `kA(f) = 1 - exp(-f / (ta fs))` for the C++ and the
mirror. The two agree to 1.5 % for the 1 ms position and slower at 48 kHz, differ by 6 % at the 0.5 ms position and are
materially different at 0.1 ms (where `kA * f` exceeds 1).

In continuous time:

```
dv/dt = (rest - v) / tr + [e > v] (e - v) (1 + max(e - rest, 0) / Sv) / ta
```

Fitted values (mode 30, `python3 fit/tools/candidates/mixed-domain.py --modes 30 --verbose`, reproduced exactly from the
default start, nfev 18, cost 10.27):

| | 0.1 | 0.5 | 1 | 5 | 10 | 30 ms label |
|---|---|---|---|---|---|---|
| `ta` (raw, at the reference level) | 0.035 | 0.418 | 1.756 | 6.426 | 19.25 | 58.3 ms |
| effective at -10 dBFS, threshold 16 (`f` about 2.6) | 0.013 | 0.16 | 0.66 | 2.4 | 7.2 | 21.9 ms |
| step tau recovered through the curve (recover 0.5 s) | | | 1.3 | 4.7 | 12.3 | 30.6 ms |
| reference, same method | | | 1.2 | 3.9 | 10.4 | 27.4 ms |

| | 0.1 | 0.25 | 0.5 | 0.8 | 1.2 s label |
|---|---|---|---|---|---|
| `tr` | 0.0947 | 0.1356 | 0.3336 | 0.3341 | 0.4887 s |

`depth = 0.55 dB`, `Sv = 17.43 dB`, `toff = -0.10 dB` (a global threshold shift the harness needs because the stored
thresholds carry mode E's own detector offset; it disappears in the stage-3 refit, section 4).

### 1.3 Why it reproduces facts 1 to 7

**Fact 1, shift invariance.** Every place the threshold enters is a difference: `rest = T - depth`, `e - rest`, `v - T`.
Shifting the level and the threshold together shifts `e`, `v` and `rest` by the same amount and leaves `GR` unchanged. Verified
numerically: max |dGR| 1.3e-12 dB under joint shifts of -45, -20 and +7 dB, and threshold 24 at -60 dBFS equals threshold 16
at -37.9 dBFS to 1e-15. No absolute floor is involved (the -100 dB log floor is 40 dB below the lowest reference and the node
never goes there: with `e = floor < v` only the release acts, toward `rest`).

**Fact 2, the dB one-pole attack.** While the diode conducts the node obeys a linear first-order equation in dB with rate
`f / ta + 1 / tr`, so a level step is approached as a one-pole in dB. The measured time constant through the curve is 1.3 /
4.7 / 12.3 / 30.6 ms for the 1 / 5 / 10 / 30 ms labels against the reference's 1.2 / 3.9 / 10.4 / 27.4 (recover 0.5 s): within
25 %, the closest of the three winners. A consequence the task's fact 2 does not list but the same probe measured: the
reference's attack time constant depends on the recover position (30 ms label: 23.4 / 27.4 / 29.7 ms at 0.1 / 0.5 / 1.2 s;
10 ms: 8.2 / 10.4 / 10.9). That is exactly what an always-on release predicts (`tau = 1 / (kA + kR)`) and no switched-branch
topology can produce; mode 30 gives 22.2 / 30.6 / 33.0 ms.

**Fact 3, the steady-state table.** On a sine the node charges only over the part of each half cycle where the log level is
above it and bleeds all the time. The balance sets `d = v - L`: a slower attack charges less per cycle (`d` falls), a slower
release bleeds less (`d` rises). Because the charge term `(e - v)+` is a nonlinear function of `d` (the log of a sine near its
peak), and because for attack constants near the carrier period the node ripples within the cycle, `d` is not a function of
`ta / tr` alone; the sample-rate simulation captures both. Mode 30's residual table, model minus reference gain in dB (rows
attack, columns recover 0.1 / 0.25 / 0.5 / 0.8 / 1.2 s, then level -25 / +2 dBFS at recover 0.5 s):

```
0.1 ms: -0.08 -0.04  0.02  0.02  0.03 | -0.05  0.05
0.5 ms: -0.05 -0.01  0.04  0.04  0.06 | -0.07  0.07
1 ms:   -0.08  0.01  0.05  0.05  0.07 | -0.05  0.08
5 ms:   -0.03 -0.04  0.04  0.04  0.08 | -0.01  0.05
10 ms:  -0.04  0.03 -0.04 -0.04 -0.04 | -0.10  0.01
30 ms:   0.27  0.10 -0.16 -0.17 -0.13 | -0.10 -0.08
```

Mode E's worst cell (-1.25 at 30 ms / 0.1 s) is +0.27; the 0.5 s and 0.8 s columns come out identical because the fit puts
both constants at 0.334 s.

**Fact 4, frequency independence.** The node's balance is over a cycle of the log-sine, whose shape in dB does not depend on
the period; only the within-cycle ripple does, and it is small at the attack constants where `d` is large. Residuals at 100 Hz
/ 5 kHz: +0.13 / 0.00 (0.1 ms), +0.07 / +0.03 (1 ms), -0.15 / -0.17 (30 ms); mode E had +0.44 / -0.26 at 0.1 ms.

**Fact 5, the mild level dependence.** This is what the level-scaled attack is for. With a fixed attack conductance the bleed
current grows with the level above the reference (`v - rest`) while the charge per cycle does not, so `d` falls linearly with
level: the plain parallel RC gives -0.74 / +0.65 dB at -25 / +2 dBFS (30 ms). Scaling the attack conductance by
`1 + (e - rest) / Sv` makes the charge grow with level too, and the balance saturates to the measured -2.87 / -4.03 / -4.42
(mode 30 residuals -0.10 / -0.08 at 30 ms, -0.10 / +0.01 at 10 ms, -0.05 / +0.08 at 1 ms). The sibling probes pin the law: a
free exponent on the level factor lands at 0.79 with the same shape over the working range (mode 36), the reference is the
input level and not the node (mode 37: alpha = 1.000 exactly), and neither a saturating attack current (mode 34, Sa to its
bound) nor a soft diode knee (mode 35) is wanted.

**Fact 6, ripple and odd harmonics.** The gain reduction rises through each half cycle (charging) and drops back across the
zero crossing (the always-on bleed, plus no charge while `e < v`); the modulation is symmetric in the two half cycles, so only
odd harmonics appear. H3 at 100 Hz / 1 kHz / 5 kHz: -38.3 / -57.9 / -71.0 dBc with 0.1 ms (reference -38.6 / -58.5 / -70.7)
and -41.3 / -61.2 / -74.4 with 10 ms (reference -42.0 / -61.9 / -74.0). Mode 34 is 0.2 dB closer and also matches H5; the
direct ripple figure is discussed in section 5.

**Fact 7, release.** After the burst `e` is below the node, only the bleed acts and the node is a one-pole toward `rest`
with `tr`: nearly constant slope for the first 20 ms (20 ms is a fifth of the shortest `tr`), then slowing; the slope through
the curve scales with `v - rest`, which is why it is 0.29 dB/ms at 0.1 s and 0.084 at 0.5 s. All -10 dBFS burst envelopes are
within 0.12 dB (weighted max) of the reference. DUAL keeps its two-node network (section 2.3): the fast partial release is C1
sharing charge with C2 through `t2`, the multi-second tail is C2 emptying through the coupling and the bleed, about
`(1 + c2) tr` = 4.2 s with the current `c2 = 7.7`, `tr = 0.48` (measured 4.3 s).

### 1.4 Numbers against the runner-up and against mode E

| topology | harness / mode | steady rms / max | bursts rms / max | cost | extra params |
|---|---|---|---|---|---|
| current C++ (mode E, node started at rest) | mixed-domain 40, cascaded 4 | 0.264 / 1.13 | 0.038 / 0.68 | 148 | depth |
| mode E as in `detector_candidates.py` (node started at -100 dB) | all four, mode 4 | 0.286-0.290 / 1.28-1.31 | 0.046 / 1.07 | 208 | depth |
| always-on release, fixed attack conductance | mixed 15 = cascaded 20 = rate-limited 21 = rms 24 | 0.175 / 0.74 | 0.011 / 0.63 | 18.2 | depth |
| always-on release, constant discharge while conducting | rms 29 = cascaded 33 | 0.134 / 0.29 (0.127 / 0.31 with xoff) | 0.013 / 0.56 | 20.5 | depth, L = 17.2 dB |
| **runner-up**: always-on release toward `max(rest, e - D0)` | cascaded 34 REFMAX | 0.113 / 0.53 | 0.009 / 0.49 | 11.22 | depth, xoff, D0 = 22.05 dB |
| **winner**: always-on release, attack conductance scaled by level above rest | mixed 30 | **0.079 / 0.27** | 0.010 / 0.91 | **10.27** | depth, toff, Sv = 17.43 dB |

Where the two leaders differ:

- Steady state: 30 wins everywhere except that its 30 ms row keeps a small structured residual (+0.27 at 0.1 s, -0.16 at
  0.5 s); 34's worst cells are the level checks at 10 / 30 ms (-0.32 / -0.53 at -25 dBFS), which is the very thing 30 fixes.
- Bursts: 34 is marginally better in rms (0.009 vs 0.010) and clearly better on the 0 dBFS step (`disc_bdepth_0`): one
  period after onset the reference gain is -7.0 dB, mode 34 gives -9.9, mode 30 -12.4 (max weighted error 0.49 vs 0.91;
  unweighted 3.6 vs 6.8 dB at that one period). This is the single measured fact that argues against 30's mechanism: the
  reference is slower than a dB one-pole at 0 dBFS and the level-scaled conductance is faster there. It costs one period.
- Attack constants: 34's are 0.55-0.86x the panel labels; 30's raw constants are 1.3-1.9x the labels and its effective
  constants at -10 dBFS are 0.5-0.75x. Neither reads as "label = RC"; the through-the-curve step tau is what is measured, and
  30 is closest (1.3 / 4.7 / 12.3 / 30.6 vs 34's 1.4 / 5.1 / 13.3 / 33.6, reference 1.2 / 3.9 / 10.4 / 27.4).
- Harmonics: both within 0.6 dB of the reference's H3; 34 also matches H5 (-44.6 vs -44.2 at 100 Hz, 0.1 ms).
- Cost in C++: 34 is one `max` per sample; 30 is one multiply-add plus (section 2.7) one `exp` while the diode conducts.

The winner is 30 on the residual that was the problem (the steady state, by 30 % in rms and by 2x in max) with a burst rms
that ties. The 0 dBFS onset and the raw attack constants are recorded as open points in section 5, not hidden.

### 1.5 What was tested and rejected

Everything the task listed was run; the digest is in the four harness docstrings. The families that failed, with the reason
the data gives:

- Power-domain (RMS-style) detectors, dbx 2252 lineage (Blackmer, US 3,681,618, https://patents.google.com/patent/US3681618A;
  THAT 2252 datasheet, https://www.thatcorp.com/datashts/THAT_2252_Datasheet.pdf): rms modes 10-12, 14-17, 20-21; mixed 25-29.
  The release in the power domain is a constant dB/s ramp; the reference's release is a one-pole in dB (fact 7), so the recover
  constants collapse to 20-130 ms and the bursts get 2-3x worse. An RMS term is a constant -3 dB on a sine and cannot produce
  the attack/recover dependence (mode 16: the fit removes it, w = 0.97).
- Giannoulis, Massberg and Reiss 2012, "Digital Dynamic Range Compressor Design, A Tutorial and Analysis", JAES 60(6)
  (https://www.eecs.qmul.ac.uk/~josh/documents/2012/GiannoulisMassbergReiss-dynamicrangecompression-JAES2012.pdf): the smooth
  decoupled form (release one-pole feeding an attack one-pole), cascaded 21-22: the linear attack pole passes the mean of its
  input, so `d` no longer depends on the attack (30 ms row -3.4 to -8.9 dB off; 1.9 dB rms). The smooth branching form is mode
  A / mode E, the baseline.
- Pre-smoothing before the branching (cascaded 23-24, rms 15): breaks frequency independence (-2.4 dB at 100 Hz, +2.3 at
  5 kHz); the fit drives the pole to zero. A post-pole after the node (cascaded 25, 28, 32): driven to zero by the 0.1 ms burst
  onsets (13.9 dB error in the first period when forced to 0.5 ms).
- Saturating, slew-limited, exponential (diode) or soft-knee attack drives (rate-limited 10, 14, 16, 22-23, 32; mixed 19-22,
  34-35): every knee or clamp goes to its "linear" limit; the burst edge is a clean one-pole in dB and forbids them.
- Half-wave rectifier (mixed 12, 26, 28): helps mode E on its own but adds nothing over the always-on release.
- Attack in the linear amplitude or power domain with a dB release (rms 12-13, 20, 26; mixed 10; rate-limited 30): bursts 2-3x
  worse; the attack edge must be a one-pole in dB.
- Two release paths (toward the reference and toward the instantaneous level; rate-limited 12, 25; mixed 16-17; cascaded 35):
  the second path is switched off by the fit (the log dips at the zero crossings make a path toward `e` harmful).
- Charge target `e + k`, threshold as a bias current before the log converter, a clamp at the reference, a slow leak to
  `T - 60`: all no-ops or driven to zero (rate-limited 13, 24, 33-36; rms 22-23; cascaded 36; mixed 29).
- A transdiode log amp whose bandwidth grows with current (mixed 31-32): wrong sign on the level cells and breaks fact 4.
- Attack conductance scaled by the node level rather than the input (mixed 24): fixes the level cells only at the price of
  +1.5 dB at 30 ms / 0.1 s.

## 2. Per-sample C++ and the numba mirror

### 2.1 States and coefficients

States (unchanged in number): `v` (the node, dB), `w` (DUAL's second node, dB), plus the existing smoothers, `gr` and the ratio
fade. Coefficients set in `setTimes()` / `prepare()`:

```
rA    = 1.0 / (c[kc_d_tatt + cfg.attack] * fsr)     attack rate per sample at the reference level (dimensionless)
kA    = 1 - exp(-rA)                                 (kept for the f = 1 case; see 2.7)
kR    = onePoleK(c[kc_d_trel + cfg.recover], fsr)
invSv = 1.0 / c[kc_d_att_sv_db]
k2, c2, kLeak                                        as now
```

### 2.2 `DiscreteStage::process`

Replace the detector block (the two `if (e > v) ... else ...` lines) with:

```cpp
inline double process(double u, double h)
{
    // detector: log rectifier, then the storage node(s). The node is a capacitor with a bleed resistor to the threshold
    // reference (always conducting) and an attack diode from the log level whose conductance grows with the level above the
    // reference (docs/detector-fix.md).
    const double a = std::fabs(h);
    const double e = (a > floorLin ? linToDb(a) : floorDb) + goffS.tick(c[kc_d_goff_db + cfg.gain]);
    const double thr = thrS.tick(c[kc_d_thr_db + cfg.thr]);
    const double rest = thr - c[kc_d_rel_depth_db];
    double dv = (rest - v) * kR;                              // release: always on
    if (e > v) {                                              // attack: while the diode conducts
        const double xe = e - rest;
        const double f = xe > 0.0 ? 1.0 + xe * invSv : 1.0;   // conductance factor, level above the reference
        dv += (e - v) * (1.0 - std::exp(-rA * f));            // exact one-pole step for rate f / ta (never exceeds e)
    }
    v += dv;
    if (cfg.recover == kRecovers - 1) {   // DUAL: the second node, unchanged
        const double flow = (v - w) * k2;
        v -= flow;
        w += flow / c2;
    }
    if (kLeak > 0.0) v += (floorDb - v) * kLeak;              // material option, unchanged
    // gain computer, ratio fade, gain cell: unchanged
    const double x = v - thr;
    double g = curves[cfg.ratio].at(x);
    if (fadePos < fadeLen) {
        const double t = double(fadePos++) / fadeLen, wf = 0.5 - 0.5 * std::cos(kPi * t);
        g = curves[prevRatio].at(x) * (1.0 - wf) + g * wf;
    }
    if (g < 0.0) g = 0.0;
    gr = g;
    const double a2 = (cfg.hwUnit ? c[kc_d_hwunit_a2] : c[kc_d_a2]) + cfg.a2Extra * (g * 0.1);
    const double ui = u + a2 * u * u + c[kc_d_a3] * u * u * u;
    return ui * dbToLin(gainS.tick(c[kc_d_gain_db + cfg.gain]) - g);
}
```

and in `setTimes()`:

```cpp
void setTimes()
{
    if (!c) return;
    rA = 1.0 / (c[kc_d_tatt + cfg.attack] * fsr);
    kA = 1.0 - std::exp(-rA);
    kR = onePoleK(c[kc_d_trel + cfg.recover], fsr);
    kLeak = cfg.leakRatio > 0.0 ? onePoleK(c[kc_d_trel + cfg.recover] / cfg.leakRatio, fsr) : 0.0;
}
```

with `invSv = 1.0 / c[kc_d_att_sv_db];` added to `prepare()` (after `c2`), and `double rA = 1.0, invSv = 0.0;` among the members.
`dv` is computed from the old `v` for both terms, as in the harness (simultaneous, not sequential, updates).

The header comment at the top of `Discrete.hpp` (lines 5-11) should be rewritten to match: the release branch is not "the rest
of the cycle" any more, it is always on, and the attack conductance is level-dependent.

### 2.3 How DUAL's second node attaches

Unchanged: after the node update, `flow = (v - w) k2; v -= flow; w += flow / c2`. The bleed resistor stays on `v` only (C1);
`w` has no path of its own to the reference. What changes is the operating point: with the bleed always on, C1 loses charge
during the burst as well, so `w` fills toward a slightly different level and the two time constants that the fit reads off
the DUAL burst-length and pulse items shift. `d_dual_t2` and `d_dual_c2` must be refitted (section 4); the candidate
harnesses never contained the DUAL items, so there is no number for them yet. If the refit's `disc_dual_*` residuals come out
worse than today's (MODEL.md quotes about 0.05 dB), the one alternative worth a run is moving the bleed to `w` (the larger
capacitor) with the coupling resistor between; do not guess, fit it.

### 2.4 How the threshold reference enters

Three places, all as differences from `thr` (the smoothed `d_thr_db[cfg.thr]`):

1. the release reference `rest = thr - d_rel_depth_db` (the bleed's target);
2. the attack conductance factor `1 + (e - rest) / d_att_sv_db`, clamped at 1 below the reference;
3. the gain computer's argument `v - thr`.

The make-up offset `d_goff_db[cfg.gain]` is added to `e` as now; because `e` also drives the conductance factor, the offset
acts exactly like an input level change, which is what "detector offset" means.

### 2.5 Initial state at rest

`reset()` already puts both nodes at the release reference: `v = w = c[kc_d_thr_db + cfg.thr] - c[kc_d_rel_depth_db]`. Keep it.
With silence (`e = floorDb = -100 < v`) only the bleed acts and the node stays exactly at `rest`, so there is no start-up
transient and no drift toward the floor. (The harness result that mode E's original -100 dB start costs it 0.008 dB of burst
rms is the reason the rest start matters for the fit as well.)

### 2.6 The numba mirror (`fit/stages/stage3_discrete.py`, `detector`)

```python
@njit(cache=True)
def detector(a, fs, ta, tr, floor_db, dual, t2, c2, goff, Tk, depth, sv):
    """a: |sidechain| samples. Returns the node v (dB) per sample. Same equations as DiscreteStage::process: the release
    conductance always on toward Tk - depth, the attack diode conducting toward the rectified log level with a conductance
    that grows with the level above the reference (1 + (e - rest) / sv)."""
    n = a.shape[0]
    v = np.empty(n)
    rA = 1.0 / (ta * fs); kR = 1.0 - np.exp(-1.0 / (tr * fs))
    k2 = 1.0 - np.exp(-1.0 / (t2 * fs)) if dual else 0.0
    floor_lin = 10.0 ** (floor_db / 20.0)
    rest = Tk - depth
    x = rest; w = rest
    for i in range(n):
        e = (20.0 * np.log10(a[i]) if a[i] > floor_lin else floor_db) + goff
        dv = (rest - x) * kR
        if e > x:
            xe = e - rest
            f = 1.0 + xe / sv if xe > 0.0 else 1.0
            dv += (e - x) * (1.0 - np.exp(-rA * f))
        x += dv
        if dual:
            flow = (x - w) * k2
            x -= flow
            w += flow / c2
        v[i] = x
    return v
```

### 2.7 The discrete-time form (why `1 - exp(-rA f)` and not `kA f`)

The harness scaled the per-sample coefficient, `kA * f`. That is not sample-rate invariant for fast attacks and it exceeds 1 at
the 0.1 ms position (at 48 kHz, `ta = 0.035 ms` gives `kA = 0.45`; times 2.6 at -10 dBFS is 1.17, so the node overshoots the
level it is charging toward by 17 % of the gap every sample). Scaling the rate, `1 - exp(-rA f)`, is the exact one-pole step for
a conductance `f / ta`, is bounded by 1, and gives the same result at 44.1 / 48 / 96 kHz (the protocol has sample-rate replicas
and rate invariance is a stated property of the model). The difference from the fitted form is 1.5 % or less for the 1 ms
position and slower, 6 % at 0.5 ms, and only the 0.1 ms position will move noticeably when stage 3 refits. The cost is one `exp`
per sample while the diode conducts, next to the `log` the rectifier already pays. If bit-parity with the harness figures is
ever wanted for a check, `min(1.0, kA * f)` reproduces them for all but the 0.1 ms position.

## 3. Calibration fields (`src/dsp/Calibration.hpp`)

Add one scalar and update three comments and the priors. Adding a field changes `calLayoutHash()`, so the C++ falls back to
the priors until `fit/make_constants.py` rewrites `FittedConstants.hpp`; `load_cal()` applies the stored JSON over the priors,
so the new field takes its prior until stage 3 writes it, no migration needed. The table grows from 773 to 774 values
(update the count in `docs/MODEL.md`).

```
A(d_tatt, kAttacks)         /* attack time constants at the release reference level, log domain; the conductance grows by
                               1 / d_att_sv_db per dB of rectified level above the reference, so the effective constant at
                               X dB above it is d_tatt / (1 + X / d_att_sv_db) */
A(d_trel, kRecovers)        /* release (bleed) time constants, log domain, always conducting; the Dual entry is its first node */
S(d_dual_t2, 0.076)         /* Dual: coupling time constant between the two storage nodes (R2 C1) */
S(d_dual_c2, 7.7)           /* Dual: second storage capacitor, relative to the first */
S(d_floor_db, -100.0)       /* log rectifier floor (far below any signal; the release does not go there) */
S(d_rel_depth_db, 0.55)     /* the bleed discharges toward the threshold minus this depth */
S(d_att_sv_db, 17.4)        /* attack conductance law: factor 1 + (level above the release reference) / this, dB */
```

Priors in `calPriors()`:

```
const double at[kAttacks]  = { 0.035e-3, 0.42e-3, 1.76e-3, 6.4e-3, 19.3e-3, 58.0e-3 };
const double rt[kRecovers] = { 0.095, 0.136, 0.334, 0.334, 0.489, 0.48 };
```

(the Dual entry keeps today's fitted 0.484 s as its prior; the DUAL pair keeps today's fitted 0.076 s / 7.7).

Bounds used by the fit (section 4), not stored in the header:

| field | prior | lower | upper | fit domain | x_scale |
|---|---|---|---|---|---|
| `d_tatt[i]` | above | 1e-5 s | 1 s | linear | 1e-3 |
| `d_trel[i]` | above | 0.01 s | 5 s | linear | 0.05 |
| `d_dual_t2` | 0.076 | 0.005 | 1 | linear | 0.01 |
| `d_dual_c2` | 7.7 | 1.5 | 60 | linear | 2 |
| `d_rel_depth_db` | 0.55 | -20 | 40 | linear | 1 |
| `d_att_sv_db` | 17.4 | 2 (10^0.3) | 1000 (10^3) | log10 | 0.3 |

`Sv` is well determined: 17.43 from a start of 12.6, and 16.5-18.6 across the sibling modes 34-39. The upper bound is the
"switched off" limit (mode 15); if a refit ever drifts there, the level cells will say so.

No field is renamed. `d_rel_depth_db` keeps its name and meaning (the reference is `T - depth`); only its comment and prior
change. Nothing else in `HVMC_CAL_FIELDS` is touched.

## 4. Changes to the stage-3 joint fit (`fit/stages/stage3_discrete.py`)

3a (the static family) is unchanged. In 3b:

1. `detector()` as in section 2.6 (one new argument `sv`). Thread `sv` through `d0_for()` and `env_of_item()` (both call
   `detector`; add `sv` as a keyword with no default, so nothing calls the old form by accident).
2. Parameter vector: `p[0:6]` attack, `p[6:12]` recover (the 12th is the Dual first node), `p[12]` `t2`, `p[13]` `c2`,
   `p[14]` depth, **`p[15]` `log10(Sv)`**. `unpack()` returns the sixth value as `10 ** p[15]`.
3. Start and bounds:
   ```python
   p0 = [0.035e-3, 0.42e-3, 1.76e-3, 6.4e-3, 19.3e-3, 58.0e-3] + [0.095, 0.136, 0.334, 0.334, 0.489, tr_dual] + [t2, c2, 0.55, np.log10(17.4)]
   lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, -20.0, 0.3]; hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 40.0, 3.0]
   x_scale = [1e-3] * 6 + [0.05] * 6 + [0.01, 2.0, 1.0, 0.3]
   ```
   (`tr_dual`, `t2`, `c2` from the loaded calibration as now). Keep `loss="soft_l1"`, `f_scale=1.0`, `diff_step=1e-3`; the
   harness converged in 18-24 evaluations from a cruder start, so `max_nfev=120` is enough.
4. `resid_all()` is otherwise unchanged: same items, same 3.0 weight on the steady cells, same burst weighting. Its residual
   definition is identical to the harnesses' (verified: all three winners reproduce in all three harnesses), so the numbers in
   section 1.4 are the targets: expect about 0.08 / 0.27 steady and 0.010 bursts on the non-DUAL items, plus whatever the
   DUAL items add.
5. The report line prints `Sv` next to depth, and, new, the residual of the four `disc_dual_blen_*` and `disc_dual_pulses`
   items on their own (they were never in a candidate fit; this is the first number for them under the new node).
6. `d0` at the capture setting (1 ms, 0.5 s) is recomputed with the new detector as now, and `T = T' + d0`. This is what
   absorbs the harness's `toff` (-0.10 dB): the stored thresholds carry mode E's `d0`, the refit replaces it.
7. The node-domain resampling of the curves (the `dx` loop) stays. Under mode 30 the offset's level dependence at the capture
   setting is about 0.03 dB over the grid (fact 5, 1 ms row), so `dx` will be small; the printed range is the check.
8. 3c (make-up offsets) and 3d (cell terms) are unchanged. 3c reads `goff` off the 4:1 curve at the capture point; because
   `goff` now also scales the attack conductance, the printed offsets may move by a few hundredths of a dB.
9. Order of operations after the code change: rebuild `hvmc_core` (the C API exposes the new layout), run
   `python3 fit/stages/stage3_discrete.py`, then `python3 fit/make_constants.py`, rebuild, run `tests/pb_reference.py` and the
   C++ unit tests (`tests/test_dsp.cpp` pins the discrete GR at thresholds 1 / 12 / 24 and the meter; those values shift by the
   new `d0`, so their expectations may need re-reading, not the test logic). Then run
   `python3 fit/tools/detector_candidates.py --e-only` once more: with the refitted curves and thresholds its mode-E baseline
   is the "before" number for the docs.
10. `docs/MODEL.md`: the Detector subsection (lines 122-141), the field table (`d_tatt`, `d_trel`, `d_rel_depth_db`, new
    `d_att_sv_db`), the 773 count, and the "Assumed" paragraph ("a rectifier into an RC network in the log domain with a
    threshold-referenced release" becomes "with an always-conducting bleed to the threshold reference and a level-dependent
    attack conductance, chosen by residual among some sixty candidates in four harnesses").

## 5. Still unexplained

1. **The 0 dBFS step onset.** One period after a -50 to 0 dBFS step the reference gain is -7.0 dB; a dB one-pole with the
   fitted constants gives -12.4 (mode 30), -9.9 (34), -10.3 (29). The reference is slower than a one-pole at high level, and the
   level-scaled conductance is faster there. Every saturating attack law was rejected by the -10 dBFS steps, which are clean
   one-poles, and sidechain headroom (mixed 39) went to its bound. One period, 0.91 weighted; unresolved.
2. **The 30 ms row.** Mode 30 leaves +0.27 at 0.1 s and -0.16 at 0.5 s, the same sign pattern as mode 29 (+0.24 / -0.28) at a
   third of the size. The slow-attack / fast-recover corner still wants something the node does not have. The two winners fix
   different cells (34 the onsets and harmonics, 30 the level cells and the table); a run of 34's `max(rest, e - D0)` release
   under 30's attack law is the obvious next candidate and has not been run.
3. **The ripple figure.** The task quotes 2.3 / 2.5 dB of within-cycle gain ripple at 100 Hz (0.1 / 30 ms). Mode 29's harness
   measured 1.24 / 0.36 dB for the detector-only chain, while every winner matches the reference's H3 at 100 Hz within 0.6 dB,
   and H3 is a direct measure of that ripple. Either the quoted ripple includes the gain cell's own distortion or the two
   measurements differ in method; mode 30's direct ripple number was not produced. Measure it the same way on both before
   reading anything into it.
4. **What the attack constants mean.** Raw `d_tatt` is 1.3-1.9x the panel labels, the effective constant at -10 dBFS is
   0.5-0.75x, and the runner-ups' are 0.55-0.86x. The step time constant recovered through the curve is what matches the labels
   (within 12 %), and that is a property of the whole node, not of one resistor. No candidate makes "label = RC".
5. **Physical identity of the conductance law.** `1 + (e - rest) / Sv` is phenomenological. The probes establish its shape
   (linear in dB over the working range, free exponent 0.79), its reference (the input level, alpha = 1.000) and that neither
   a knee nor a saturation is wanted, but not the circuit that produces a diode conductance growing with the log level. A log
   amp whose output impedance falls with current was tried in one form (the transdiode bandwidth, mixed 31) and failed on
   fact 4; other forms have not been run.
6. **DUAL.** No candidate harness included the DUAL items and the steady-state matrix has no DUAL column, so the two-node
   network's constants under the new node, and whether the bleed belongs on the first or the second capacitor, are open until
   the stage-3 refit prints its `disc_dual_*` residuals.
7. **0.5 s and 0.8 s.** The fit puts both at 0.334 s, reproducing the plugin's identical behaviour at the two positions. Whether
   to tie `d_trel[2]` and `d_trel[3]` explicitly is a choice, not a finding.
