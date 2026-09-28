# Discrete stage: ratio-dependent ballistics, the fastest attack, and the SIDECHAIN FILTER trim

Status: research document, 2026-09-28. Measured on the licensed reference plug-in (1.5.1, sha256 b1f977dd3897...) with the fitted
constants as of that date; nothing under `src/`, `fit/stages/` or `fit/data/` was changed. Harness:
`fit/tools/candidates/disc-ratio-ballistics.py` (capture, tables, candidate fits, checks); every number below is in
`build/disc-ratio-ballistics/` (`measure.json`, `tables.md`, the `fit_*.log` and `check*.log` files). It answers programme-audit findings 2,
4 and 5 (`research/program-audit.md`, sections 3.2, 3.4, 3.5) and says what to change in `src/dsp/Discrete.hpp`, `src/dsp/Engine.hpp`,
`src/dsp/Opto.hpp`, `src/dsp/Calibration.hpp` and `fit/stages/stage3_discrete.py`.

The short version:

- **Finding 5.** SIDECHAIN FILTER IN is a +0.638 dB trim on the discrete stage's input, audio and sidechain alike, independent of level,
  frequency, ratio, make-up, attack, recover, stereo mode and transformer to 0.002 dB, and present only with the optical stage **out**:
  with the optical stage in, the discrete input gain is the +2.14 dB inter-stage gain whether the filter is in or out. The engine as of
  today multiplies the two (2.78 dB with both in and the filter in); it must crossfade between them. On the optical stage the filter does
  **not** remove the idle leak: it replaces the threshold-dependent leak by a fixed one, `cond0 = 0.0245` (-0.21 dB at every threshold
  position, the leak the filter-out law has at position 20.7), so the no-compression gain becomes +0.351 dB at all 24 positions. The
  engine as of today sets the leak to zero with the filter in (+0.561 dB at every position): 0.21 dB too loud on everything quiet
  whenever the filter is in.
- **Finding 4.** The 0.1 ms position is a real time constant: `d_tatt[0] = 0.199 ms` and `d_tatt[1] = 0.721 ms` (raw, at the release
  reference; 0.07 and 0.25 ms effective at -10 dBFS), fitted on the charge law, the first periods of the fast bursts and the fast rows of
  the steady table. The charge-law error at 0.1 / 0.5 ms falls from 0.44 to 0.07 dB rms (40-tone and pink noise from +0.85 / +1.23 to
  0.00 / +0.07), the fast bursts from 0.022 to 0.011, the sine rows lose 0.03 dB rms, and H3 at 0.1 ms does not move.
- **Finding 2.** The ratio switch sets the attack conductance. In the reference's own units (the equivalent steady-sine level a gain
  corresponds to on that ratio's static curve, so no model of either the curve or the node is involved) the node at 30 ms attack sits
  2.4 / 0.46 / 0.27 dB lower than at 4:1 for 1.2:1 / 2:1 / 3:1, 0.23 dB higher at 6:1 and 1.75 dB higher at FLOOD, at every recover
  position and both check levels; the same order holds for the 1 ms burst onsets (FLOOD reaches 0.37 of its settled gain reduction in
  the first period at -30 dBFS, 4:1 0.19, 1.2:1 0.11) and for dense material (FLOOD reads the 40-tone 1.0 dB and pink noise 1.0 dB
  higher than 4:1 does at 1 ms). The release is the same to 2 % for 2:1 to 6:1 and 5 % slower at FLOOD. The law stays referenced to the
  threshold (the sag is identical at positions 10, 16 and 22 to 0.01 dB). The smallest change that reproduces it: a per-ratio attack
  time-constant scale (`2.34 / 1.29 / 1.21 / 1 / 1` for 1.2:1 to 6:1), and for FLOOD a limiter timing network: a fixed conductance
  about 6.6 times the 4:1 reference-level conductance with no level dependence, a current limit of 5.4 dB on the charge (a slew-limited
  attack) and a bleed 13 % slower. Through the curves that takes FLOOD's steady table from 1.03 to 0.10 dB rms (max 3.15 to 0.37), its
  charge law from 1.29 to 0.18 (the audit's -1.00 / -0.93 on the 40-tone and pink at 1 ms become about +0.05, the 30 ms sine from -1.88
  to +0.05) and its bursts from 0.134 to 0.081; 1.2:1 / 2:1 / 3:1 from 1.09 / 0.33 / 0.23 to 0.14 / 0.11 / 0.11 dB rms. 4:1 is untouched
  by construction and 6:1 is 4:1. What the per-ratio work exposes and this document leaves open: a shared current limit of about 35 dB
  improves every ratio's 30 ms high-level onsets (4:1 bursts 0.071 to 0.045) at a 0.05 dB cost on the steady table, which is detector-fix
  open point 5.1 seen from a new angle, and at FLOOD the current limit that fits the 30 ms onsets leaves the 0.5 and 1 ms onsets at low
  level too slow (section 5).

Units: dB of gain of the fundamental unless stated; "Leq" is the equivalent steady-sine level (section 2.1); GR is gain reduction.
Residuals "through the curve" are in dB of gain, model minus reference, on the same items and weights as stage 3b (steady cells 3,
burst samples 1 / sqrt(60), samples above 0.5 dB of GR weighted 1 and the rest 0.25).

## 1. Finding 5: the SIDECHAIN FILTER trim and the optical leak

### 1.1 The discrete trim

Discrete stage alone at threshold 1 (no gain reduction at any level here), gain of the fundamental, filter In minus Out. Reference /
engine as built today (`d_scf_trim_db = 0.64`):

| item | reference Out | reference In | In - Out (ref) | In - Out (ours) |
|---|---|---|---|---|
| Dual Mono, -60 / -40 / -20 dBFS | +2.2963 | +2.9339 | +0.6376 | +0.6400 |
| Dual Mono, 0 dBFS | +2.2861 | +2.9222 | +0.6360 | +0.6414 |
| Stereo, -60 / -40 / -20 / 0 dBFS | as Dual Mono | as Dual Mono | +0.6376 / +0.6360 | +0.6400 / +0.6414 |
| every ratio position, -30 dBFS | +2.2963 | +2.9339 | +0.6376 | +0.6400 |
| 30 Hz / 100 Hz / 1 kHz / 10 kHz, -30 dBFS | +2.2837 / +2.2963 / +2.2963 / +2.1934 | +0.6376 above each | +0.6376 | +0.6400 |
| DISCRETE GAIN 1 / 7 / 12 / 24, -30 dBFS | -10.6488 / +0.0168 / +2.2963 / +19.5688 | | +0.6295 / +0.6369 / +0.6376 / +0.6397 | +0.6400 |
| attack 0.1 ms or 30 ms, DUAL, -30 dBFS | +2.2963 | +2.9339 | +0.6376 | +0.6400 |
| Iron, -50 dBFS | +2.1119 | +2.7495 | +0.6376 | +0.6400 |
| both stages out, -30 dBFS | -0.0301 | -0.0301 | 0.0000 | 0.0000 |
| optical alone, threshold 1, -50 dBFS | +0.5608 | +0.3514 | **-0.2094** | +0.0001 |
| both stages in, thresholds 1, -50 dBFS | +5.0310 | +4.8212 | **-0.2098** | **+0.6401** |

So the trim is **+0.638 dB** (0.6376 at every level from -60 to -20 dBFS, 0.6360 at 0 dBFS where the gain cell's own 0.01 dB of
compression sees a 0.64 dB hotter input), independent of frequency from 30 Hz to 10 kHz (so it is in the audio path, not a filter),
of ratio, attack, recover, stereo mode and transformer, and of make-up to 0.01 dB (0.6295 at position 1, 0.6397 at 24: the reference's
own gain law rounding, not a level effect). At threshold 16 the trim is seen through each ratio's curve: net -0.064 at 4:1, -0.167 at
FLOOD, +0.271 at 1.2:1 (reference), against -0.030 / -0.080 / +0.279 in the engine: the sidechain sees the same +0.64 dB as the audio
(the earlier "+0.75 dB sidechain insertion gain" was this trim measured as gain reduction), and the residual 0.03 to 0.09 dB at the steep
ratios is the curve's slope at the trimmed level, not a second gain.

**It does not add to the inter-stage gain.** With both stages in at thresholds 1 the filter changes the output by -0.2098 dB, which is
the optical stage's own change (-0.2094, section 1.2) and nothing from the discrete side. Sum of parts with the filter out: optical
+0.5608 + discrete +2.2963 - stages-out -0.0301 = +2.8871, measured +5.0310, inter-stage gain +2.144 dB; with the filter in: parts
+3.3154, measured +4.8212, +1.506 dB, i.e. 2.144 - 0.638. The discrete stage's input gain is therefore +2.14 dB with the optical stage in
(filter in or out), +0.64 dB with it out and the filter in, 0 with both out. The engine today (`Engine.hpp`, `internal()`):

```cpp
const double gi = (1.0 + wo * (interLin - 1.0)) * k.scfTrim.tick(k.disc.filterIn() ? scfLin : 1.0);
```

gives +2.78 dB with both in and the filter in (measured above: ours +0.6401 against the reference's -0.2098 on that row), which is the
audit's setting A (both stages, filter in, the most used configuration). Replace by a crossfade keyed on the optical bypass state:

```cpp
// the discrete stage's input gain: the inter-stage gain when the optical stage is in (following its crossfade), the SIDECHAIN FILTER's
// trim when it is out and the filter is in; the two do not add (measured: both stages in, the filter changes nothing on the discrete side)
const double gTrim = k.scfTrim.tick(k.disc.filterIn() ? scfLin : 1.0);
const double gi = wo * interLin + (1.0 - wo) * gTrim;
```

with `d_scf_trim_db = 0.638` (calibration field, `S(d_scf_trim_db, 0.638)`; the current prior 0.64 is 0.002 dB off, fine either way)
and `d_inter_db = 2.144` (measured here to three decimals; the current 2.14 is fine). Level-independent, so a gain and not a
level-dependent block; stereo-mode-independent, so before the link sum (as it is); ratio- and attack-independent, so ahead of the
detector and not inside it. The filter's frequency response itself is right (unchanged from the audit).

### 1.2 The optical leak with the filter in

Optical stage alone, no-compression gain against threshold position at -50 dBFS (identical at -70 dBFS to 0.001 dB):

| position | 1 | 4 | 8 | 12 | 16 | 18 | 20 | 21 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| reference, filter Out | +0.561 | +0.560 | +0.555 | +0.531 | +0.488 | +0.446 | +0.381 | +0.336 | +0.276 | +0.116 |
| reference, filter In | +0.351 | +0.351 | +0.351 | +0.351 | +0.351 | +0.351 | +0.351 | +0.351 | +0.351 | +0.351 |
| engine today, filter Out | +0.561 | +0.559 | +0.545 | +0.504 | +0.452 | +0.409 | +0.347 | +0.306 | +0.256 | +0.127 |
| engine today, filter In | +0.561 | +0.561 | +0.561 | +0.561 | +0.561 | +0.561 | +0.561 | +0.561 | +0.561 | +0.561 |

With the filter in the gain is +0.351 dB at every position: threshold-independent, as the audit said, but 0.21 dB **below** the leak-free
gain (+0.561, which the reference shows at position 1 with the filter out and the engine shows at every position with the filter in). So
the filter does not remove the leak; it removes its threshold dependence and leaves a fixed idle conductance: `1 / (1 + c0) = 10^(-0.21/20)`
gives **`c0 = 0.0245`**, which is the filter-out leak of about position 20.7 (position 20 is +0.381, 21 is +0.336) and, to 0.004 dB,
the fitted `o_leak = 0.0250` (the position-20 leak constant). Above the knee the filter changes nothing: the threshold-20 static curve
with the filter in is within +0.01 dB of filter-out from -26 to +4 dBFS (the two outliers, -0.23 at -2 dBFS and +0.13 at +2 dBFS, are the
reference's own ripple at those two points, present in both states), and the harmonics at -10 dBFS are the same to 0.1 dB at 1 kHz.
Reading: the reference's idle bias enters its sidechain ahead of the high pass and scales with the threshold gain (our `o_leak_q` law);
with the high pass in, that path is blocked and a second, fixed bias after the threshold gain remains. Whether the second bias is also
there with the filter out (and merely swamped) cannot be told from the output; the implementation below does not need to know.

`Opto.hpp` today: `double leakFor() const { return cfg.scFilter ? 0.0 : leakLightFor(cfg.thr); }`. Change to a fixed leak with the
filter in, expressed through the existing constant (no new field) or through one new scalar:

```cpp
// the idle light: with the SIDECHAIN FILTER in the threshold-dependent leak is blocked and a fixed one remains (measured: +0.351 dB at
// all 24 positions against +0.561 leak-free, i.e. cond0 = 0.0245, the filter-out leak of position 20.7)
double leakFor() const
{
    const double cond0 = cfg.scFilter ? c[kc_o_leak_scf] : leakCondFor(cfg.thr);   // leakCondFor: the cond0 of leakLightFor
    return cond0 > 0.0 ? std::pow(cond0, 1.0 / c[kc_o_gamma]) : 0.0;
}
```

with `S(o_leak_scf, 0.0245)` (idle conductance with the sidechain filter in; if one prefers no new field, `c[kc_o_leak]` reproduces
it to 0.004 dB). `grDb()` already divides by `(1 + leak conductance)`, so the meter stays at zero at idle in both states. The check is
the table above: +0.351 at every position with the filter in, and the filter-out column unchanged.

The engine's filter-out column is 0.02 to 0.04 dB low from position 12 to 21 (the `o_leak_q` law overshoots mid-range: +0.347 against
+0.381 at 20) and 0.01 high at 24; that is the stage-4 leak fit, not this finding, and it is within the tolerance the tests carry.

## 2. Finding 2: the detector against the ratio switch

### 2.1 Method: the static curve as a ruler

Everything measured on the discrete stage alone at threshold 16, Nickel, make-up 12, DUAL MONO. The static curve of each ratio was
captured in 0.5 dB steps from -42 to +6 dBFS at the capture setting (1 ms / 0.5 s), reference and engine. For a gain `g` measured at any
other setting or signal, `Leq(g)` is the steady 1 kHz sine level that gives `g` on that ratio's own curve. Since the reference's gain is
a fixed curve of its node (section 2 of `docs/MODEL.md`: threshold a pure offset, one curve per ratio), `Leq` is the node's reading in
input-level units, offset by whatever the node sits at on a 1 ms / 0.5 s sine. A detector that does not know the ratio position gives the
same `Leq` at every ratio for the same signal and setting; the engine does (checked: 0.00 in every cell of the table below). The rulers:

| ratio | GR at -10 dBFS (ref / ours) | slope at -10 dBFS, dB/dB | knee (first level with GR > 0.1 dB) |
|---|---|---|---|
| 1.2:1 | 17.05 / 17.04 | 0.577 / 0.593 | -39.5 / -38.0 |
| 2:1 | 25.73 / 25.70 | 1.006 / 1.010 | -39.5 / -38.0 |
| 3:1 | 26.66 / 26.66 | 0.993 / 1.002 | -38.5 / -37.5 |
| 4:1 | 27.55 / 27.51 | 1.077 / 1.071 | -37.5 / -37.5 |
| 6:1 | 27.82 / 27.78 | 1.094 / 1.090 | -36.5 / -36.5 |
| FLOOD | 31.27 / 31.23 | 1.170 / 1.171 | -34.0 / -34.0 |

(The soft ratios' knees are 1.5 dB lower in the reference: that is audit finding 3, the soft-ratio knee, handled elsewhere.)

### 2.2 The steady table in Leq, relative to 4:1

Reference, `Leq` minus the same cell at 4:1, dB. Columns: recover 0.1 / 0.25 / 0.5 / 0.8 / 1.2 s / DUAL at -10 dBFS; level -25 / +2 dBFS
at 0.5 s; 100 Hz / 5 kHz at 0.5 s. The 4:1 row is absolute (its 1 ms / 0.5 s cell is -10.00 by construction).

| ratio | attack | 0.1 s | 0.25 s | 0.5 s | 0.8 s | 1.2 s | DUAL | -25 dBFS | +2 dBFS | 100 Hz | 5 kHz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4:1 (absolute) | 0.1 ms | -9.76 | -9.70 | -9.62 | -9.62 | -9.60 | -9.62 | -24.69 | +2.42 | -9.69 | -9.64 |
| | 1 ms | -10.67 | -10.40 | -10.00 | -10.00 | -9.90 | -9.97 | -25.00 | +2.00 | -10.02 | -10.02 |
| | 30 ms | -18.12 | -16.64 | -13.89 | -13.89 | -13.02 | -13.63 | -27.79 | -2.26 | -13.88 | -13.90 |
| 1.2:1 | 0.1 ms | +0.29 | +0.35 | +0.46 | +0.46 | +0.50 | +0.47 | +0.11 | +0.27 | +0.49 | +0.48 |
| | 1 ms | -0.71 | -0.46 | 0.00 | 0.00 | +0.14 | +0.04 | 0.00 | 0.00 | +0.01 | +0.02 |
| | 5 ms | -1.45 | -1.19 | -0.64 | -0.64 | -0.38 | -0.57 | -0.19 | -0.38 | | |
| | 10 ms | -2.58 | -2.15 | -1.32 | -1.32 | -1.11 | -1.27 | -0.53 | -1.23 | | |
| | 30 ms | -4.18 | -3.64 | -2.40 | -2.40 | -2.03 | -2.31 | -1.16 | -3.08 | -2.42 | -2.39 |
| 2:1 | 0.1 ms | +0.03 | +0.04 | +0.06 | +0.06 | +0.06 | +0.06 | +0.05 | +0.05 | +0.06 | +0.06 |
| | 1 ms | -0.07 | -0.05 | 0.00 | 0.00 | +0.02 | +0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| | 10 ms | -0.51 | -0.42 | -0.20 | -0.20 | -0.16 | -0.19 | -0.31 | -0.17 | | |
| | 30 ms | -0.97 | -0.82 | -0.46 | -0.46 | -0.38 | -0.48 | -0.67 | -0.38 | -0.48 | -0.47 |
| 3:1 | 30 ms | -0.59 | -0.48 | -0.27 | -0.27 | -0.23 | -0.28 | -0.39 | -0.23 | -0.28 | -0.28 |
| 6:1 | 0.1 ms | -0.01 | -0.02 | -0.02 | -0.02 | -0.02 | -0.02 | -0.07 | -0.02 | -0.02 | -0.02 |
| | 10 ms | +0.20 | +0.17 | +0.08 | +0.08 | +0.06 | +0.08 | +0.11 | +0.07 | | |
| | 30 ms | +0.40 | +0.33 | +0.23 | +0.23 | +0.15 | +0.19 | +0.24 | +0.16 | +0.22 | +0.22 |
| FLOOD | 0.1 ms | -0.10 | -0.13 | -0.17 | -0.17 | -0.17 | -0.17 | -0.18 | -0.15 | -0.20 | -0.17 |
| | 0.5 ms | +0.04 | -0.02 | -0.11 | -0.11 | -0.14 | -0.12 | -0.12 | -0.09 | | |
| | 1 ms | +0.33 | +0.20 | 0.00 | 0.00 | -0.04 | -0.01 | 0.00 | 0.00 | -0.01 | 0.00 |
| | 5 ms | +0.86 | +0.66 | +0.28 | +0.28 | +0.17 | +0.24 | +0.22 | +0.28 | | |
| | 10 ms | +1.85 | +1.44 | +0.77 | +0.77 | +0.57 | +0.72 | +0.69 | +0.73 | | |
| | 30 ms | +1.96 | +2.40 | +1.75 | +1.75 | +1.36 | +1.62 | +1.56 | +1.12 | +1.73 | +1.74 |

(Full table with every row in `build/disc-ratio-ballistics/tables.md`.) Reading: the ratio positions order the detector's charge
against its bleed monotonically, 1.2:1 slowest, FLOOD fastest. The effect grows with the attack constant and with the bleed rate (largest
at 30 ms / 0.1 s), is the same at 100 Hz and 5 kHz, and at the fastest attack reverses sign (1.2:1 reads +0.46 at 0.1 ms, FLOOD -0.17):
at 0.1 ms every position reads the peak, so the sign reversal is the ruler's own offset, i.e. at 1 ms / 0.5 s the 1.2:1 node already sits
0.46 dB further below the peak than the 4:1 node, and FLOOD's 0.17 dB less. In dB of gain the sag from 1 ms to 30 ms at 0.5 s is
4.83 / 4.42 / 4.28 / 4.12 / 3.97 / 2.60 for 1.2:1 to FLOOD in the reference against 2.77 / 3.84 / 3.83 / 3.89 / 3.91 / 4.53 in the engine;
at 0.1 s recover 8.35 / 7.96 / 7.19 (1.2:1 / 4:1 / FLOOD) against 6.13 / 8.30 / 9.78.

### 2.3 Bursts: onsets and releases per ratio

First-period fraction of the settled gain reduction after a -50 -> L dBFS step (per-period lock-in gain; the audit's per-period rms
figure is 0.05 to 0.09 lower), reference / engine:

| ratio | attack | -30 dBFS | -20 | -10 | 0 |
|---|---|---|---|---|---|
| 1.2:1 | 0.1 ms | 0.54 / 0.94 | 0.67 / 0.96 | 0.68 / 0.98 | 0.75 / 0.98 |
| 1.2:1 | 1 ms | 0.11 / 0.26 | 0.16 / 0.24 | 0.18 / 0.28 | 0.26 / 0.35 |
| 4:1 | 0.1 ms | 0.69 / 0.93 | 0.81 / 0.96 | 0.88 / 0.97 | 0.83 / 0.98 |
| 4:1 | 0.5 ms | 0.40 / 0.46 | 0.56 / 0.61 | 0.58 / 0.67 | 0.48 / 0.71 |
| 4:1 | 1 ms | 0.19 / 0.13 | 0.32 / 0.24 | 0.29 / 0.30 | 0.23 / 0.35 |
| 6:1 | 1 ms | 0.19 / 0.06 | 0.33 / 0.21 | 0.29 / 0.29 | 0.23 / 0.34 |
| FLOOD | 0.1 ms | 0.84 / 0.89 | 0.92 / 0.95 | 0.92 / 0.97 | 0.78 / 0.99 |
| FLOOD | 0.5 ms | 0.63 / 0.21 | 0.72 / 0.52 | 0.56 / 0.61 | 0.44 / 0.66 |
| FLOOD | 1 ms | 0.37 / 0.00 | 0.38 / 0.12 | 0.27 / 0.23 | 0.21 / 0.30 |

The engine's fractions fall with the ratio position because the curves' knees rise (FLOOD's is 3.5 dB above 4:1's: a node starting at
rest needs longer to reach it); the reference's rise. Two things in the reference stand out: FLOOD's onset is fast at **low** level
(0.63 at 0.5 ms / -30 dBFS against 0.40 at 4:1) and its fraction falls as the level rises (0.78 at 0.1 ms / 0 dBFS against 0.92 at -10),
i.e. FLOOD's charge is not proportional to the gap. The 30 ms onsets say the same in rate terms, the node rise between periods 10 and 50
divided by the mean gap to the settled level (a dB one-pole with a level-independent conductance gives one number across levels; the
stage-3 law `1 + xe / Sv` gives a number rising with level):

| ratio | -30 dBFS | -20 | -10 | 0 |
|---|---|---|---|---|
| 1.2:1 | 0.0127 | 0.0185 | 0.0203 | 0.0230 |
| 2:1 | 0.0159 | 0.0240 | 0.0307 | 0.0263 |
| 4:1 | 0.0169 | 0.0265 | 0.0313 | 0.0236 |
| 6:1 | 0.0174 | 0.0274 | 0.0312 | 0.0229 |
| FLOOD | 0.0281 | 0.0373 | 0.0267 | 0.0178 |

At 4:1 the conductance grows from -30 to -10 dBFS (stage-3's level law) and then falls at 0 dBFS (the high-level slowness of
detector-fix 5.1); at FLOOD it is highest at -30 / -20 dBFS and falls from there: a fixed conductance with a current limit, not a
level-dependent one. Releases after the -10 dBFS burst (Leq fall per period, 1 ms attack; engine: 0.079 / 0.066 at 0.5 s, 0.229 / 0.128
at 0.1 s at every ratio):

| ratio | 0.5 s, periods 5-40 | 40-120 | 0.1 s, 5-40 | 40-120 |
|---|---|---|---|---|
| 1.2:1 | 0.0697 | 0.0540 | 0.1898 | 0.1274 |
| 2:1 | 0.0764 | 0.0659 | 0.2330 | 0.1374 |
| 3:1 | 0.0772 | 0.0657 | 0.2338 | 0.1341 |
| 4:1 | 0.0779 | 0.0648 | 0.2313 | 0.1283 |
| 6:1 | 0.0770 | 0.0638 | 0.2278 | 0.1244 |
| FLOOD | 0.0738 | 0.0621 | 0.2205 | 0.1137 |

2:1 to 6:1 within 2 %; FLOOD 5 % slower (or its rest 1 dB higher: the two are not separable from the release alone); 1.2:1 10 to 18 %
slower, but its ruler is shallow (0.58 dB/dB) and the release runs into its knee, so that figure is the least certain of the table.

### 2.4 The charge law per ratio

Steady gain reduction on seven signals at equal peak (-10 dBFS), engine minus reference in dB of GR (negative: ours compresses less),
recover 0.5 s; the audit's cells are the 4:1 and FLOOD rows:

| ratio | attack | 1 kHz | 100 Hz | 2-tone | 10-tone | 40-tone | square | pink |
|---|---|---|---|---|---|---|---|---|
| 4:1 | 0.1 ms | 0.00 | -0.08 | +0.13 | +0.08 | +0.83 | +0.12 | +1.20 |
| 4:1 | 1 ms | -0.04 | -0.04 | +0.07 | +0.07 | +0.07 | +0.15 | +0.15 |
| 4:1 | 30 ms | +0.19 | +0.19 | 0.00 | -0.01 | -0.09 | +0.18 | -0.03 |
| 6:1 | 30 ms | -0.07 | -0.07 | -0.30 | -0.31 | -0.44 | +0.04 | -0.37 |
| FLOOD | 0.1 ms | +0.19 | +0.15 | +0.20 | +0.17 | +0.43 | +0.26 | +0.66 |
| FLOOD | 1 ms | -0.04 | -0.04 | -0.29 | -0.33 | -1.11 | -0.10 | -1.04 |
| FLOOD | 5 ms | -0.33 | -0.33 | -0.82 | -0.85 | -1.73 | -0.09 | -1.66 |
| FLOOD | 30 ms | -1.97 | -1.96 | -2.72 | -2.79 | -3.06 | -1.07 | -2.98 |
| 1.2:1 | 30 ms | +2.05 | +2.07 | +1.98 | +1.96 | +1.04 | +1.73 | +1.07 |

In the reference's units the 40-tone reads 5.69 dB below the sine at 4:1 / 1 ms and 4.69 at FLOOD (engine 5.69 and 5.55); pink 5.87 and
4.87 (engine 5.71 and 5.67): FLOOD's faster charge reads the peaks of dense material 1.0 dB higher, which is the audit's "FLOOD reads
dense material about 1 dB higher". The 0.1 ms row (+0.83 / +1.20 at 4:1) is finding 4.

### 2.5 Threshold dependence

Sag of the steady gain from 1 ms to 30 ms and from 0.1 ms to 1 ms at a level 27.7 dB above the threshold (the level that is -10 dBFS at
position 16), reference / engine. Position 4 puts the tone at +22 dBFS where the reference's core clips; positions 10, 16 and 22 are the test:

| ratio | recover | position 10 (+5.9 dBFS) | 16 (-10 dBFS) | 22 (-26.4 dBFS) |
|---|---|---|---|---|
| 1.2:1 | 0.5 s | 4.824 / 2.767 | 4.828 / 2.767 | 4.827 / 2.767 |
| 4:1 | 0.5 s | 4.121 / 3.890 | 4.124 / 3.890 | 4.123 / 3.890 |
| FLOOD | 0.5 s | 2.601 / 4.531 | 2.602 / 4.531 | 2.602 / 4.531 |
| FLOOD | 0.1 s | 7.196 / 9.781 | 7.192 / 9.781 | 7.189 / 9.781 |

Identical to 0.01 dB across 32 dB of absolute level: the attack law, at every ratio, is referenced to the threshold (stage-3's `rest`
and `e - rest`), not to an absolute level, and the shift family (fact 1) holds at 30 ms as it does at 1 ms.

### 2.6 Candidates

All fitted in the Leq domain per ratio: the stage-3 detector (`node_r` in the harness with `a0 = alpha = 1`, `Sa = Sx = inf` is
`fit/stages/stage3_discrete.py`'s `detector()` equation for equation) with the stage-3 constants (`d_tatt`, `d_trel`, `d_rel_depth_db`,
`d_att_sv_db`, DUAL) and per-ratio parameters fitted on that ratio's steady table (54 cells, weight 3) and its 18 burst envelopes (300
periods after the onset and 300 after the end, stage-3 weights); model `Leq = node - d0` with `d0` the model's own offset at 1 ms / 0.5 s.
Residuals below are the steady table in dB of gain (through the ruler's slope) and the weighted burst rms / max; the control R0 is the
engine's detector applied to every ratio.

| mode | per-ratio parameters | 1.2:1 | 2:1 | 3:1 | 4:1 | 6:1 | FLOOD |
|---|---|---|---|---|---|---|---|
| R0 | none (control) | 1.087, 0.180 / 0.93 | 0.333, 0.101 / 0.96 | 0.232, 0.084 / 1.02 | 0.107, 0.071 / 1.05 | 0.125, 0.075 / 1.03 | 1.010, 0.163 / 1.16 |
| R1 | attack scale `s` | 2.335: 0.148, 0.090 / 0.52 | 1.292: 0.114, 0.082 / 0.57 | 1.212: 0.117, 0.068 / 0.72 | 1.097: 0.125, 0.063 / 0.90 | 1.011: 0.136, 0.074 / 1.02 | 0.590: 0.313, 0.176 / 1.88 |
| R2 | `Sv` | 102: 0.186 | 21.9: 0.127 | 19.8: 0.123 | | | 7.4 (first run): 0.29 |
| R3 | release scale `q` | 1.048: 1.48 (Leq) | 0.963: 0.29 | 0.972: 0.20 | 1.003 | 1.036: 0.09 | 1.144: 0.65 |
| R4 | `s`, `q` | 2.44, 1.06: 0.155, 0.082 | 1.27, 0.98: 0.116, 0.080 | 1.20, 0.98: 0.119, 0.067 | 1.11, 1.01 | 1.05, 1.05: 0.129, 0.069 | 0.67, 1.15: 0.305, 0.152 / 1.73 |
| R5 | rest depth | 0.33: 1.42 (Leq) | 1.79: 0.30 | 1.33: 0.21 | 0.50 | -0.22: 0.10 | -2.38: 0.73 |
| R7 | conductance intercept `a0` (`f = a0 + xe / Sv`) | 0.05 (bound): 0.659 | 0.366: 0.094 | 0.530: 0.101 | 0.800: 0.108 | 1.029: 0.116 | 3.047: 0.287, 0.169 / 1.80 |
| R8 | level reference `alpha` (input / node) | 0.0: 1.26 (Leq) | 0.19: 0.24 | 0.23: 0.16 | 0.43: 0.13 | 0.69: 0.16 | 2.0 (bound): 0.70 |
| R10 | `a0`, `Sv` | 0.47, 35.8: 0.147 | 0.40, 14.9: 0.095 | 0.76, 16.9: 0.114 | 1.48, 24.1: 0.144 | 2.14, 40.1: 0.169 | 4.63, 460 (bound): 0.332, 0.158 / 1.60 |
| S1 | shared-form current limit `Sa` (u -> u / (1 + u / Sa)) | 9.9: 0.658 | | | 50.2: 0.159, **0.046 / 0.46** | | 1000 (off): 1.017 |
| S4 | `Sa`, `Sv` | off, 98: 0.189, 0.097 | 67.7, 18.3: 0.143, 0.079 / 0.31 | 50.3, 16.0: 0.142, 0.058 / 0.31 | 37.3, 13.3: 0.154, 0.045 / 0.40 | 31.1, 11.6: 0.171, 0.056 / 0.47 | 5.46, 3.32: 0.186, 0.149 / 0.86 |
| R1S | `s`, `Sa` | 2.31, off: 0.152, 0.090 | 1.19, 80: 0.130, 0.078 / 0.31 | 1.07, 52: 0.140, 0.058 / 0.32 | 0.93, 35: 0.155, 0.043 / 0.38 | 0.83, 28: 0.169, 0.053 / 0.42 | 0.284, 4.47: 0.146, 0.135 / 0.67 |
| R7S | `a0`, `Sa` | 0.05 (bound), 28: 0.456 | 0.52, 86: 0.118, 0.076 | 0.82, 51: 0.138, 0.058 | 1.27, 33: 0.157, 0.041 / 0.34 | 1.66, 26: 0.170, 0.047 / 0.36 | **8.36, 4.38: 0.114, 0.120 / 0.50** |
| R9S | `s`, `Sv`, `Sa` | 2.14, 16.5, off: 0.150, 0.091 | 2.50, 5.3, 72: 0.120, 0.076 | 1.25, 11.6, 51: 0.138, 0.058 | 0.63, 29, 37: 0.159, 0.039 / 0.33 | 0.43, 57, 31: 0.174, 0.043 / 0.33 | 0.110, 460, 4.88: 0.112, 0.118 / 0.49 |
| R10S | `a0`, `Sv`, `Sa` | 0.47, 35, off: 0.150, 0.091 | 0.40, 13.3, 72: 0.120, 0.076 | 0.80, 14.4, 51: 0.138, 0.058 | 1.60, 18.4, 37: 0.159, 0.039 | 2.32, 24.7, 31: 0.174, 0.043 | 9.55, 460, 4.92: 0.111, 0.117 / 0.49 |

(Where a cell shows one number it is the steady rms; the S and R*S modes were run on the same data; the full logs are `fit_w3.log`,
`fit_w3sat.log`, `fit_w3comb.log`, `fit_leq.log`, `fit_noL0.log`.) What the table says:

1. **The soft ratios are a slower attack.** One number, the attack scale `s` (R1), takes 1.2:1 / 2:1 / 3:1 from 1.09 / 0.33 / 0.23 to
   0.15 / 0.11 / 0.12 dB rms and their bursts from 0.18 / 0.10 / 0.08 to 0.09 / 0.08 / 0.07. Nothing else does it as one number: the
   conductance intercept (R7) fails at 1.2:1 (`a0` to its bound, 0.66 rms), the level reference (R8) and the depth (R5) do nothing, and
   the release scale (R3) confirms the release is not it (`q` within 4 % of 1). The two-parameter modes buy 0.00 to 0.02 dB over R1.
2. **6:1 is 4:1.** `s = 1.01`, `a0 = 1.03`, `q = 1.04`: its 0.125 rms against the 4:1 control's 0.107 is the same residual pattern (the
   30 ms row), and its bursts are 4:1's. The audit's "6:1 mildly" is this 0.02 dB plus the 5.1 onset item.
3. **FLOOD is a different network.** The attack scale alone (R1, `s = 0.59`) fixes its steady table (1.01 to 0.31) but makes its bursts
   worse (0.163 to 0.176, max 1.88): the high-level 30 ms onsets overshoot, because a faster level-dependent conductance is fastest where
   the reference is slowest. What fits both is a large **fixed** conductance with a **current limit** (R7S: `a0 = 8.36` with the level
   term left as it is, `Sa = 4.38 dB`; R9S / R10S with the level term freed drive it to the bound, `Sv = 460`, and land at the same
   residual): steady 0.114, bursts 0.120 / 0.50; with the release scale fitted alongside (`R7SQ`: `a0 = 6.60`, `Sa = 5.37`, `q = 1.131`)
   0.094 and 0.092 / 0.51, which is the chosen set (section 2.7). In words, at FLOOD the charge rate is about 7 to 8 times the 4:1
   reference-level rate, independent of level, and for gaps over about 5 dB the charge current is constant: a slew-limited limiter
   attack, which is what a "limiter" position with its own timing network would be, with a bleed 13 % slower. The current limit is what
   keeps the 30 ms high-level onsets slow while the 1 ms and 0.5 ms low-level onsets are fast (section 2.3), and the fixed conductance is
   what reads dense material 1 dB higher.
4. **A shared current limit is a separate, open item.** With one `Sa` per ratio and nothing else (S1 / S4), 4:1 and 6:1 want
   `Sa = 31 to 50 dB` and gain 0.03 on their bursts (0.071 to 0.045, max 1.05 to 0.40: the 0 dBFS 30 ms onset of detector-fix 5.1) at a
   0.05 dB cost on the steady table (0.107 to 0.154). That is the trade-off detector-fix 5.1 already names, seen with more data; it is
   not a ratio item and this document does not adopt it (section 4).

### 2.7 The chosen change, checked through the curves

The candidates above are in the Leq domain. The check below is the model through each ratio's stage-3 curve, in dB of gain, on the
same items (`--check`), with finding 4's constants in every row, and with the ratio's own detector offset at the capture setting folded
into the curve (`d0_r - d0_4:1`: +0.24 dB at FLOOD, -0.30 at 1.2:1, -0.07 / -0.05 at 2:1 / 3:1; this is the stage-3 resampling step
redone per ratio, section 3.3, and without it every FLOOD cell carries a -0.25 dB offset).

| ratio | parameters | steady rms / max: before -> after | charge law rms / max | bursts rms / max |
|---|---|---|---|---|
| 1.2:1 | `s = 2.335` | 1.087 / 2.80 -> 0.142 / 0.32 | 0.222 / 0.52 | 0.180 / 0.93 -> 0.030 / 0.33 |
| 2:1 | `s = 1.292` | 0.333 / 0.85 -> 0.110 / 0.46 | 0.060 / 0.12 | 0.101 / 0.96 -> 0.050 / 0.54 |
| 3:1 | `s = 1.212` | 0.232 / 0.67 -> 0.110 / 0.58 | 0.071 / 0.17 | 0.084 / 1.02 -> 0.034 / 0.72 |
| 4:1 | none (finding 4 only) | 0.099 / 0.35 -> 0.104 / 0.35 | 0.284 / 1.23 -> 0.091 / 0.22 | 0.027 / 1.14 -> 0.027 / 1.11 |
| 6:1 | none (finding 4 only) | 0.125 / 0.63 -> 0.133 / 0.70 | 0.138 / 0.41 | 0.075 / 1.03 -> 0.043 / 1.13 |
| FLOOD | **`a0 = 6.60`, `Sa = 5.37 dB`, `q = 1.131`** (chosen) | 1.028 / 3.15 -> **0.104 / 0.37** | 1.288 / 3.03 -> **0.180 / 0.52** | 0.134 / 1.48 -> **0.081 / 0.91** |
| FLOOD | `a0 = 8.36`, `Sa = 4.38 dB` (no release scale) | -> 0.124 / 0.49 | 0.197 / 0.56 | 0.111 / 1.01 |
| FLOOD | `s = 0.284`, `Sa = 4.47` (scale instead of intercept) | -> 0.161 / 0.61 | 0.298 / 1.03 | 0.114 / 1.05 |
| FLOOD | `s = 0.59` (scale alone) | -> 0.323 / 1.03 | 0.498 / 1.20 | 0.124 / 2.39 |

The release scale is fitted with the attack network (`R7SQ` in `fit_floodq.log`: `a0 = 6.60`, `Sa = 5.37`, `q = 1.131`, Leq steady
0.094 dB of gain, bursts 0.092 / 0.51; `R1SQ`, the scale form with `q`, 0.128 / 0.111). It removes the release residual that every FLOOD
burst carried without it (0.11 rms at every attack, all after the burst end) and improves the steady table too, because the slower bleed
and the smaller intercept trade against each other in the 30 ms row. FLOOD's table with the chosen change (engine minus reference, dB of
gain; rows attack, columns recover 0.1 / 0.25 / 0.5 / 0.8 / 1.2 s / DUAL, then level -25 / +2, then 100 Hz / 5 kHz; without `q` the
pattern is the same with the 0.1 s column +0.35 / +0.49 at 30 ms):

```
0.1 ms: +0.11 +0.11 +0.11 +0.11 +0.11 +0.11 | +0.11 +0.09 | +0.14 +0.09
0.5 ms: +0.07 +0.08 +0.10 +0.10 +0.10 +0.10 | +0.11 +0.08
1 ms:   -0.05  0.00 +0.03 +0.03 +0.06 +0.03 | +0.09 +0.04 | +0.07 +0.01
5 ms:   -0.13 -0.08 -0.01 -0.01 +0.03 -0.02 | +0.04  0.00
10 ms:  +0.16 -0.01 -0.12 -0.12 -0.07 -0.17 |  0.00 -0.06
30 ms:  +0.35 +0.49 -0.02 -0.02 -0.18 -0.22 | -0.03 -0.25 | -0.01 -0.04     (a0 = 8.36, Sa = 4.38, no q; the chosen set: rms 0.104, max 0.37)
```

and its charge law (40-tone / pink at 1 ms: +0.03 / +0.08 against the audit's -1.00 / -0.93; at 30 ms -0.56 / -0.52 against -3.06 / -2.98;
the 30 ms sine +0.05 against -1.97). The 0.1 ms row's +0.11 is the ruler offset (the FLOOD ruler is 0.17 dB from the peak, section 2.2)
that a per-ratio `d_tatt[0]` scale would remove; not worth a parameter. What is left in FLOOD's bursts after the change is the onsets of
the fast attacks at low level: first-period fractions 0.77 / 0.78 / 0.70 / 0.61 at 0.1 ms against the reference's 0.84 / 0.92 / 0.92 /
0.78, and 0.35 / 0.41 / 0.36 / 0.32 at 0.5 ms against 0.63 / 0.72 / 0.56 / 0.44: the 5.4 dB current limit that the 30 ms onsets need is too
tight for the 0.5 ms and 1 ms onsets at -30 / -20 dBFS (section 5).

## 3. Finding 4: the fastest attacks

`d_tatt[0]` sits at its fit bound (1e-5 s) because the stage-3 residual is all sines. Fitting `d_tatt[0]` and `d_tatt[1]` at 4:1 through
the stage-3 curve on the 0.1 / 0.5 ms rows of the steady table (18 cells), the charge law (14 cells at equal peak and 14 at equal rms) and
the fast bursts (ten envelopes), everything else at the stage-3 values (`--fit-fast`, `fit_fast.log`):

| | `ta[0]` | `ta[1]` | steady rms / max | charge rms / max | bursts rms / max | first period at -30 / -20 / -10 / 0 dBFS, 0.1 ms (ref 0.69 / 0.81 / 0.88 / 0.83) | 0.5 ms (ref 0.40 / 0.56 / 0.58 / 0.48) | H3 at 0.1 ms, 100 Hz / 1 kHz / 5 kHz (ref -38.6 / -58.5 / -70.7) |
|---|---|---|---|---|---|---|---|---|
| stage 3 | 0.010 ms | 0.412 ms | 0.045 / 0.11 | 0.441 / 1.23 | 0.0221 / 1.14 | 0.93 / 0.96 / 0.98 / 0.98 | 0.46 / 0.61 / 0.67 / 0.71 | -38.1 / -57.7 / -70.8 |
| **fitted** | **0.199 ms** | **0.721 ms** | 0.072 / 0.10 | 0.068 / 0.18 | 0.0108 / 0.49 | 0.65 / 0.78 / 0.83 / 0.86 | 0.32 / 0.47 / 0.54 / 0.58 | -38.1 / -57.9 / -71.1 |
| probe | 0.150 | 0.500 | 0.049 / 0.09 | 0.189 / 0.42 | 0.0146 / 0.92 | 0.71 / 0.83 / 0.87 / 0.90 | 0.41 / 0.56 / 0.63 / 0.66 | |
| probe | 0.250 | 0.800 | 0.089 / 0.10 | 0.094 / 0.15 | 0.0130 / 0.37 | 0.59 / 0.73 / 0.78 / 0.82 | 0.29 / 0.44 / 0.51 / 0.55 | |

The charge law at 0.1 ms after the fit (model / reference GR): sine 25.58 / 25.63, 100 Hz 25.49 / 25.55, two-tone 25.28 / 25.32, 10-tone
25.27 / 25.32, 40-tone 21.55 / 21.54, square 25.61 / 25.51, pink 21.45 / 21.38; at 0.5 ms: 40-tone 20.43 / 20.43, pink 20.29 / 20.24.
The steady sine rows lose 0.03 dB rms (the 0.1 ms row reads 0.05 dB low at every recover position: a real time constant sags a little
even on a sine); the first periods come within 0.05 of the reference at -30 / -20 / -10 dBFS and pass it at 0 dBFS (the reference's
0.83 there is the 5.1 slowness again); H3 does not move. The effective constants at -10 dBFS (`ta / (1 + 27.3 / 14.6)`) are 0.07 and
0.25 ms: the panel's 0.1 and 0.5 ms labels, near enough, in the same sense as the other four positions (detector-fix 5.4). The audit's
setting K (4:1, 0.1 ms / 0.1 s: -0.8 on the mix, -1.6 on pink) should close to within its static offset.

## 4. The change, as equations

Per sample in the discrete stage, `e` the rectified log level, `T` the threshold, `rest = T - depth`, `r` the ratio position:

```
dv   = (rest - v) * kR[r]                                     bleed, always on; kR from tr[recover] * q[r] (q = 1 except FLOOD)
if e > v:
    u   = e - v                                               the gap
    us  = Sa[r] > 0 ? u / (1 + u / Sa[r]) : u                 current limit: the charge current saturates at Sa[r] dB of gap (FLOOD only)
    f   = a0[r] + max(e - rest, 0) / Sv                       conductance: per-ratio intercept, shared level law (a0 = 1 except FLOOD)
    dv += us * (1 - exp(-f / (ta[attack] * s[r] * fs)))       exact one-pole step at rate f / (ta s[r]); s[r] = 1 except the soft ratios
v   += dv
GR   = C_r(v - T)
```

with `ta[0] = 0.199 ms`, `ta[1] = 0.721 ms` (finding 4) and

| ratio | 1.2:1 | 2:1 | 3:1 | 4:1 | 6:1 | FLOOD |
|---|---|---|---|---|---|---|
| `s[r]`, attack scale | 2.335 | 1.292 | 1.212 | 1 | 1 | 1 |
| `a0[r]`, conductance intercept | 1 | 1 | 1 | 1 | 1 | 6.60 |
| `Sa[r]`, current limit, dB (0 = none) | 0 | 0 | 0 | 0 | 0 | 5.37 |
| `q[r]`, release scale | 1 | 1 | 1 | 1 | 1 | 1.131 |

Everything else in the detector (the always-on bleed's form, the DUAL network, the log floor, the make-up offset, the germanium leak) is
unchanged, and so is everything at 4:1: the stage-3 table, bursts, DUAL and level checks are reproduced exactly at 4:1 (and at 6:1)
because every per-ratio entry is the identity there. DUAL at FLOOD: the second node is unchanged and the first node's bleed is the
scaled one; the FLOOD DUAL column of the steady table is within 0.22 dB (section 2.7) and no FLOOD DUAL burst was captured, so the
`disc_dual_*`-style items at FLOOD are the one thing the stage-3 refit should add and print before this is called closed. Shift invariance holds at every ratio (all three terms are differences from `T`,
section 2.5 measured it). The scale and the intercept are the same operation on the rate (`f / (ta s)` with `a0 = 1` against
`a0 / ta`), kept as two tables because the soft ratios need the level law scaled with them (a pure time-constant scale) and FLOOD needs
it swamped (a fixed conductance): a single `a0` table cannot do 1.2:1 (R7) and a single `s` table cannot do FLOOD (R1). If one table is
wanted, `s = 1 / a0` with `Sv` per ratio (R9S: 2.14 / 2.50 / 1.25 / 1 / 1 / 0.11 with `Sv` 16.5 / 5.3 / 11.6 / 14.6 / 14.6 / off) lands
on the same residuals with two tables of six either way.

### 4.1 `src/dsp/Calibration.hpp`

Four arrays of `kRatios` and no renamed field:

```
A(d_att_scale, kRatios)     /* attack time-constant multiplier per RATIO position (1 except the soft ratios: 2.335, 1.292, 1.212, 1, 1, 1) */
A(d_att_a0, kRatios)        /* attack conductance intercept per RATIO position: f = a0 + (level above the reference) / d_att_sv_db
                               (1 except FLOOD, 6.60: a fixed conductance that swamps the level law) */
A(d_att_sat_db, kRatios)    /* current limit on the charge per RATIO position, dB of gap (0 = none; FLOOD 5.37: a slew-limited attack) */
A(d_rel_scale, kRatios)     /* release (bleed) time-constant multiplier per RATIO position (1 except FLOOD, 1.131) */
```

Priors: `{2.335, 1.292, 1.212, 1, 1, 1}`, `{1, 1, 1, 1, 1, 6.60}`, `{0, 0, 0, 0, 0, 5.37}`, `{1, 1, 1, 1, 1, 1.131}`; `d_tatt` priors
`{0.199e-3, 0.721e-3, 1.976e-3, 6.98e-3, 21.1e-3, 64.3e-3}`. Adding fields changes the layout hash, so `fit/make_constants.py` must be rerun after stage 3;
`load_cal()` applies the priors until then. Finding 5's fields: `d_scf_trim_db` prior 0.638, `d_inter_db` 2.144, plus
`S(o_leak_scf, 0.0245)` (section 1.2).

### 4.2 `src/dsp/Discrete.hpp`

`setTimes()` and `prepare()`:

```cpp
void setTimes()
{
    if (!c) return;
    rA = 1.0 / (c[kc_d_tatt + cfg.attack] * c[kc_d_att_scale + cfg.ratio] * fsr);
    a0 = c[kc_d_att_a0 + cfg.ratio];
    invSa = c[kc_d_att_sat_db + cfg.ratio] > 1e-3 ? 1.0 / c[kc_d_att_sat_db + cfg.ratio] : 0.0;
    const double tr = c[kc_d_trel + cfg.recover] * c[kc_d_rel_scale + cfg.ratio];
    kR = onePoleK(tr, fsr);
    kLeak = cfg.leakRatio > 0.0 ? onePoleK(tr / cfg.leakRatio, fsr) : 0.0;
}
```

(members `double a0 = 1.0, invSa = 0.0;`; `configure()` already calls `setTimes()` on every change, and the ratio fade only crossfades
the curves: when the ratio switch moves, the node keeps its value and the new timing applies at once, which is what the reference does
within a period as far as the bursts show; no extra smoothing.) In `process()`, the attack block:

```cpp
if (e > v) {
    const double xe = e - rest;
    const double f = a0 + (xe > 0.0 ? xe * invSv : 0.0);        // conductance: per-ratio intercept plus the level law
    const double u = e - v;
    const double us = invSa > 0.0 ? u / (1.0 + u * invSa) : u;   // current limit (FLOOD): the charge saturates at Sa dB of gap
    dv += us * (1.0 - std::exp(-rA * f));                         // exact one-pole step for the linear part; never overshoots e
}
```

Everything after (`v += dv`, DUAL, the leak, the gain computer, the cell) unchanged. Cost: one divide per sample at FLOOD. The header
comment at the top of the file should say that the ratio switch selects the attack network (a slower attack for the soft ratios, a
current-limited limiter attack at FLOOD), not only the curve.

### 4.3 `src/dsp/Engine.hpp` and `src/dsp/Opto.hpp`

Section 1: the crossfade `gi = wo * interLin + (1 - wo) * gTrim` in `internal()`, and the fixed filter-in leak in `leakFor()`.

### 4.4 `fit/stages/stage3_discrete.py`

3a unchanged. In 3b:

1. `detector()` gains four arguments, `s`, `a0`, `sa`, `q` (the attack constant times `s`, the release constant times `q`), and the
   attack block becomes the equations of section 4 (the harness's `node_r` with `alpha = 1`, `sx = inf` is that function; copy it).
   `d0_for()` and `env_of_item()` thread them through; `env_of_item()` reads the ratio position of its item and passes that ratio's entries.
2. The residual gains the per-ratio items captured here (`build/disc-ratio-ballistics/measure.json`, to be moved into
   `fit/data/reference_features.json` by adding the items to `fit/measure/protocol.py`): the steady table at every ratio (`ar`, `al`,
   `af` per ratio, weight 3), the charge-law cells at equal peak and equal rms for 4:1 and FLOOD at 0.1, 0.5, 1 and 30 ms (weight 3; these
   are what pin `d_tatt[0:2]` and FLOOD's `Sa`), and the burst envelopes at every ratio for 0.1 / 0.5 / 1 / 30 ms at -30 / -20 / -10 /
   0 dBFS (stage-3 weights). The 4:1 items already in the residual stay as they are, so the 4:1 fit is the same problem with more rows.
3. Parameter vector: the current sixteen, plus `s[0:3]` (log domain, start 2.335 / 1.292 / 1.212, bounds 0.3 to 5), `a0[5]` (start 6.60,
   bounds 1 to 30), `sa[5]` (log domain, start 5.37 dB, bounds 1 to 100 dB), `q[5]` (start 1.131, bounds 0.7 to 1.5). 4:1 and 6:1
   entries are fixed at the identity, not fitted
   (6:1 measured as 4:1 within 0.02 dB; leave it a fixed identity until a measurement says otherwise). `d_tatt[0]`'s lower bound goes
   from 1e-5 to 1e-6 (irrelevant now that the charge law holds it at 0.2 ms, but the bound should not be what sets a constant).
4. **The node-domain resampling of the curves must use the ratio's own detector.** Today `dx` is computed once with the 4:1 detector
   and applied to all six curves. With per-ratio timing the node's offset at the capture setting differs per ratio (-0.14 at FLOOD,
   -0.68 at 1.2:1 against -0.38 at 4:1, section 2.7), so the `dx` loop runs per ratio with that ratio's `s`, `a0`, `sa`, and each curve is
   resampled with its own `xnode`. `T = T' + d0` stays defined by the 4:1 detector (the threshold table is shared); the per-ratio offset
   difference is in the curve, which is what the shift family allows (one curve per ratio, one offset per threshold).
5. The report prints the residual per ratio (steady, charge, bursts) so a regression at any position is visible, and the FLOOD
   first-period fractions at 0.1 / 0.5 / 1 ms next to the reference's.
6. 3c and 3d unchanged. `d_goff_db` is read off the 4:1 curve at the capture setting as now.

Order: add the fields, rebuild the C API, add the protocol items and capture them (or convert `measure.json`), run stage 3, regenerate
the header, rerun `tests/pb_reference.py` (the `disc_static` tolerance at the soft ratios will move with the per-ratio resampling; the
4:1 items must not) and the programme audit (J, J2, K, B and the 6:1 / soft-ratio cases are the ones to watch).

## 5. Still open

1. **The high-level 30 ms onsets** (detector-fix 5.1, audit 3.6): at 4:1 and 6:1 a current limit of 31 to 50 dB takes the bursts from
   0.071 to 0.045 (max 1.05 to 0.40) and costs 0.05 dB rms on the steady table. It is the same mechanism this document gives FLOOD at
   4.4 dB, so a single `Sa` law across the ratios (falling from about 50 dB at the soft end to 4.4 at FLOOD, S4's column) is the natural
   next candidate; it needs the steady trade-off understood first (which cells pay: the level cells at 30 ms, in S1's run), and it is not
   adopted here because the 4:1 table is a fact the task keeps.
2. **FLOOD's fast onsets at low level.** With the chosen network the 0.5 ms and 1 ms onsets at -30 / -20 dBFS reach 0.35 / 0.41 and
   0.03 / 0.14 of the settled gain reduction in the first period against the reference's 0.63 / 0.72 and 0.37 / 0.38 (section 2.7): the
   5.4 dB current limit that the 30 ms onsets need is too tight for the fast attacks. A current limit that loosens as the attack constant
   shortens (a limit in dB per unit of `ta`, i.e. `Sa` proportional to some power of `ta`) is the obvious form and was not run; it would
   be a second FLOOD-only constant. The release scale (`q = 1.131`, the release 5 % slower in Leq, section 2.3) is adopted, so the
   audit's `J_snare` release figure should close; the rest-1-dB-higher reading of the same data was not preferred by any fit (R5).
3. **1.2:1** keeps 0.14 dB rms after its scale, with a pattern that grows with the recover time at slow attacks (+0.44 at 30 ms / 0.1 s,
   -0.07 at 30 ms / DUAL) that no single extra parameter in this family removes (`q`, depth, `Sv` and the intercept were all tried). Its
   ruler is shallow and its knee sits in the audit's finding 3 territory; revisit after that fix, with the knee-band capture that
   document asks for.
4. **What the FLOOD network is.** A fixed conductance with a current limit reads as a limiter timing network (a resistor and a
   current-limited charge instead of the level-dependent diode); the reference may equally have a per-position table of constants. The
   measurements cannot tell a circuit from a table; the equations here are the smallest description of the data, not an identification.
5. **The level-law reference.** With per-ratio timing in, detector-fix 5.5 (the physical identity of `1 + xe / Sv`) is unchanged; the
   rate-per-gap table (section 2.3) says the conductance at 4:1 rises from -30 to -10 dBFS and falls at 0 dBFS, which the linear law
   plus a current limit reproduces, and a saturating level law (`Sx`, the harness's S2 / S5) does not do better.
6. **The optical filter-in leak's origin** (section 1.2): a fixed bias after the threshold gain reproduces the output; whether it is also
   present with the filter out cannot be measured from the output and does not matter for the implementation.
7. **The engine's optical leak law mid-range** is 0.02 to 0.04 dB low from positions 12 to 21 with the filter out (section 1.2); a
   stage-4 item, listed so it is not mistaken for this finding.
