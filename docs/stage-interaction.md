# The interaction between the optical and the discrete stage in the reference

Status: measured 2026-09-27/28 against the licensed reference plug-in (version 1.5.1, sha256 b1f977dd3897...), 1,100 renders, all
cached in `build/stage-interaction/measure.json`; tool `fit/tools/candidates/stage-interaction.py` (every number below is reproducible
from it: `--parts analyse` regenerates `build/stage-interaction/tables.md` and `fit.json` from the cache; `--parts measure,...` re-measures).
Ours means the engine through `fit/hvmc_core.py` with the inter-stage gain (`d_inter_db` = +2.14) and the filter trim
(`d_scf_trim_db` = +0.64) already in. Nothing under `src/`, `fit/stages/` or `fit/data/` was changed. This answers
`research/program-audit.md` finding 3.1 ("the optical stage backs off when the discrete stage compresses hard").

## 0. The result

The optical stage does not back off. The reference adds, **after the output transformer**, one two-hundredth of the signal the optical
cell shunts away:

```
y = xfmr( G_dm g_d [ 1.28 G_om v ] )  +  (1/200) (x - v)          v = x g_o, the optical divider output before make-up
```

`x` is the channel input, `v` the optical divider output (before the OPTICAL GAIN make-up), `g_o` the optical cell's gain, `g_d` the
discrete cell's gain, `G_om`, `G_dm` the two make-ups, 1.28 the +2.14 dB inter-stage gain. The bleed is a clean, in-phase, flat copy of
`x - v` at -46.02 dB; it goes through neither make-up, neither gain cell, nor the transformer, and the discrete detector never sees it.
One constant, κ = 10^(-46.02/20) = 1/200.0, fits 411 steady-state cells (five optical thresholds, five discrete threshold/ratio pairs,
both make-ups from 1 to 24, levels -40 to +6 dBFS, discrete threshold 1 to 24) to 0.01 dB rms, 0.13 dB worst, and the -50 to 0 dBFS
step per period to 0.05 dB at three attack settings. The same κ comes out of every subset (per ratio, per make-up, per threshold,
per level, with the discrete stage compressing nothing: 200.0 to 200.7).

Everything the audit saw follows from it. The "excess" is `20 log(1 + κ (1 - g_o) / (G g_o g_d))`: it grows with input level (g_o
and g_d fall), with the discrete threshold (g_d), it grows as either make-up is lowered (the main path shrinks, the bleed does not), it
is on the compressing channel only (the R channel at -40 dBFS has g_o = 1), and it is instantaneous (it follows g_d). Two things in
the audit were misread: the optical stage's *alone* curves already contain the bleed (0.05 to 0.5 dB less gain reduction than the loop
really applies at make-up 12, 1 to 2 dB at make-up 1), which is why the excess looked like it depended on the discrete stage when it was
referred to them; and the first period after a step is an averaging artefact (section 3.7).

## 1. Method

**The probe.** STEREO, left channel 1 kHz at the test level, right channel 1 kHz at -40 dBFS. The discrete detectors are linked (summed
before the rectifier) and the optical ones are not (section 2 of `docs/MODEL.md`), and -40 dBFS is under every optical knee used, so
the right channel carries the same discrete gain reduction as the left and no optical gain reduction: **R's level change reads GR_d**,
and L minus R reads the optical stage in situ. Verified: the discrete stage alone gives identical gain reduction on both channels
(within 0.03 dB from 21 to 48 dB, including its curve plateaus; section 3.2).

**Baselines.** Every level is referred to the same setting with both channels at -60 dBFS (no gain reduction anywhere), so the
make-ups, the inter-stage gain and the transformer cancel. A first pass referred R to itself at -40 dBFS; at discrete thresholds 20
and 22 that probe alone puts the linked detector 2 to 3 dB into gain reduction, which biased GR_d low; every number here is from the
corrected baselines. The probe's own contribution to the linked detector with L loud is 0.5 to 1.4 dB (in-phase sum) and is included
where the discrete law is evaluated.

**Definitions per cell.** `GR_d` from R; `GR_L` the left channel's total gain reduction; `GR_o,alone` the optical stage alone at the same
threshold, make-up and level; `excess = GR_o,alone + GR_d - GR_L` (positive = the reference is louder than the series of its parts with the
+2.14 in); `k` = the extra signal as a fraction of the input, `20 log(out_lin - main_lin)`. Levels are rms over the last second of a
3 s tone; gains are rms differences.

**The alone contamination.** The optical stage alone is `x (g_o G_om) + κ x (1 - g_o)`, so its measured gain reduction understates
the loop's: at threshold 22, make-up 12, the alone reading is 18.25 dB at 0 dBFS where the loop applies 18.56, and 21.45 where it
applies 21.92 (+6 dBFS). The fit solves the true `g_o` from the alone measurement with the same κ, so both measurements are explained by one
constant. This also means the captured optical statics in `fit/data/reference_features.json` carry the bleed (section 6).

## 2. Hypotheses and residuals

The static prediction of each hypothesis is a fixed point on the engine's stage laws (its optical stage alone per threshold and make-up,
its discrete stage with the optical in at threshold 1, so the +2.14 is inside), evaluated on 43 cells (optical 14 and 22, FLOOD 20, 4:1 22,
4:1 16, 2:1 14, make-ups 1 and 12, L at -10, 0, +6 dBFS). Two residuals: the left channel's excess (measured minus predicted) and the R
reading (measured GR_d minus predicted). The R reading is the discriminator: any hypothesis in which the optical stage's audio output
changes must change what the linked discrete detector sees, and R shows it.

| hypothesis | parameter | L excess rms / max (dB) | R reading rms / max (dB) | verdict |
|---|---|---|---|---|
| series of parts, +2.14 in (the engine as it is) | | 6.71 / 15.91 | 1.70 / 5.16 | the R reading is the engine's own discrete-law error (0.14 dB rms against the reference's own law, section 3.4); the L excess is the thing to explain |
| (1) optical sidechain = the discrete cell output before make-up | | 3.81 / 8.79 | 14.47 / 25.71 | refuted: with FLOOD's slope above 1 dB/dB the loop opens fully, and R would read 14 dB more GR_d |
| (2) optical sidechain = own output + f × discrete output (post make-up) | f = 0.1 / 0.3 / 1.0 | 6.71 / 15.91 | 1.68 to 1.76 / 4.8 to 5.1 | refuted: the discrete output is 30 to 60 dB down, the share changes nothing; also the excess does not scale with the discrete make-up while this would |
| (3) optical conductance × exp(-β GR_d) | β = 0.02 / 0.05 / 0.1 | 6.67 / 5.94 / 4.15 | 2.52 / 6.43 / 11.87 | refuted: any β that removes the L excess puts 6 to 12 dB on R; the discrete detector sees no back-off |
| (4) the +2.14 on the audio only, after the discrete detector | | (excess unchanged) | 2.30 / 3.79 | refuted: R reads the detector *with* the +2.14 (row 1, 0.14 rms against the reference's law); the +2.14 is a constant, 2.143 to 2.144 dB at every optical threshold and ratio (2.06 to 2.40 at the make-up extremes, section 3.1) and takes no part in the bleed |
| (5) d_goff on an over-compressing FLOOD curve | | excess at FLOOD 20, 0 dBFS by discrete gain 1..24: 10.6, 6.1, 5.5, 4.6, 2.1, 0.6 | R reads GR_d 28.3, 30.4, 30.6, 30.9, 31.4, 31.7 | refuted: the make-up's detector offset is in the R reading and in the series alike (3.4 dB across the range, as `d_goff_db` says); it leaves the excess untouched |
| **the bleed, κ (x - v) after the transformer** | κ = 1/200 | **0.01 / 0.13** (411 cells) | 0.14 / 0.97 (reference's own law) | the law |

Our engine on the same 43 cells: ours minus the reference 7.21 dB rms (17.86 max); with `κ (x - v)` added after our render, 0.41 dB rms
(1.72 max), the remainder being our own stage laws at make-up 1 (our 4:1 curve at discrete gain 1 reads 0.8 to 1.7 dB high; our optical
loop at optical gain 1 reads 0.4 to 0.5 dB low; both are stage-alone items, section 7).

## 3. Measurements

### 3.1 The inter-stage gain

No compression (L and R at -60 dBFS), both minus optical minus discrete plus none, dB:

| optical threshold | 1 | 14 | 18 | 20 | 22 | 24 | FLOOD thr 1 | 2:1 thr 1 | 1.2:1 thr 1 |
|---|---|---|---|---|---|---|---|---|---|
| extra | +2.144 | +2.144 | +2.144 | +2.144 | +2.143 | +2.143 | +2.144 | +2.144 | +2.144 |

Against the make-ups (optical threshold 22, discrete threshold 1): +2.14 ± 0.03 at optical gain 6 to 24 and discrete gain 1 to 24, except
optical gain 1: +2.40 / +2.07 / +1.99 at discrete gain 1 / 12 / 24 (the bleed is part of that row: at optical gain 1 the "optical alone"
term is itself 0.2 dB bled at -60 dBFS through the leak conductance).

### 3.2 The probe is valid: the link gives both channels the same discrete gain reduction

Discrete alone, STEREO, L at -10 / 0 / +6 dBFS with R at -40, gain reduction on L / on R, dB:

| setting | -10 | 0 | +6 |
|---|---|---|---|
| 4:1 thr 24 | 42.29 / 42.29 | 46.47 / 46.47 | 47.51 / 47.51 |
| 2:1 thr 24 | 39.48 / 39.48 | 44.52 / 44.51 | 45.93 / 45.92 |
| FLOOD thr 24 | 45.05 / 45.05 | 47.54 / 47.54 | 48.01 / 48.01 |
| FLOOD thr 20 | 36.85 / 36.85 | 44.38 / 44.38 | 46.59 / 46.59 |
| 4:1 thr 16 | 21.42 / 21.42 | 31.82 / 31.81 | 37.51 / 37.48 |

With the optical in at threshold 1 (the +2.14 present) the same to 0.02 dB at 40 to 51 dB. So the discrete cell's attenuation is the same on
a channel at +6 dBFS and on one at -40 dBFS, up to its curve plateaus (which are the reference's curves, not a feed-through: 4:1 at
threshold 22 reaches 49.4 dB with the optical in), and the discrete stage alone has no bleed.

### 3.3 The grid, optical threshold 22, make-ups 12 (excerpt; the full 411-cell table is in `build/stage-interaction/tables.md`)

Columns: GR_d from R | optical alone | optical in situ (L minus R) | excess | L output dBFS rms | k = extra / input, dB.

| discrete | L dBFS | GR_d | opto alone | opto in situ | excess | out | k |
|---|---|---|---|---|---|---|---|
| FLOOD 20 | -20 | 23.10 | 5.41 | 5.14 | 0.27 | -26.5 | -53.7 |
| FLOOD 20 | -10 | 27.02 | 11.77 | 10.38 | 1.39 | -35.7 | -49.2 |
| FLOOD 20 | 0 | 30.88 | 18.25 | 13.66 | 4.59 | -42.8 | -47.5 |
| FLOOD 20 | +6 | 33.85 | 21.45 | 13.50 | 7.95 | -45.6 | -47.0 |
| 4:1 22 | -20 | 26.56 | 5.41 | 4.99 | 0.42 | -29.8 | -53.4 |
| 4:1 22 | -10 | 29.91 | 11.77 | 9.85 | 1.93 | -38.0 | -49.0 |
| 4:1 22 | 0 | 33.10 | 18.25 | 12.64 | 5.61 | -44.0 | -47.4 |
| 4:1 22 | +6 | 35.60 | 21.45 | 12.37 | 9.08 | -46.2 | -47.0 |
| 4:1 16 | -10 | 12.34 | 11.77 | 11.59 | 0.18 | -22.2 | -53.1 |
| 4:1 16 | 0 | 15.46 | 18.25 | 17.51 | 0.74 | -31.2 | -50.0 |
| 4:1 16 | +6 | 18.06 | 21.45 | 19.87 | 1.58 | -36.2 | -48.8 |
| 2:1 14 | 0 | 9.59 | 18.25 | 18.01 | 0.24 | -25.9 | -54.1 |
| 2:1 14 | +6 | 11.79 | 21.45 | 20.86 | 0.58 | -30.9 | -51.6 |
| FLOOD 12 | 0 | 2.75 | 18.25 | 18.30 | -0.05 | -19.3 | none |

At optical threshold 14 into FLOOD 20: excess 0.43 / 1.19 / 2.52 / 4.58 / 6.17 / 8.03 at -15 / -10 / -5 / 0 / +3 / +6 dBFS (GR_d 31 to 39,
optical alone 3.7 to 17.0), k -55.6 to -47.5. The discrete threshold sweep at optical 22, L +6: FLOOD threshold 12 / 14 / 16 / 18 / 20 / 22 /
24 gives GR_d 6.1 / 13.1 / 20.1 / 27.1 / 33.9 / 39.9 / 44.7 and excess 0.09 / 0.74 / 2.05 / 4.39 / 7.95 / 12.19 / 16.09; 4:1 the same to
within the curves (0.17 to 12.96); 2:1 0.17 to 11.44. The excess is a function of the two gains only, not of the ratio position.

In the k column the extra signal approaches -47 dB of the input as the gain reductions grow and falls off where they are small; that
fall-off is entirely the alone reference's own bleed (the k column is computed against the alone reading, not the true loop): against the
true optical gain reduction every cell gives κ (1 - g_o) with κ = 1/200 (section 4).

### 3.4 The discrete detector does not see the excess

GR_d read on R against the reference's own discrete law (both stages in, optical at threshold 1, per ratio and threshold) evaluated at the
series level (input, minus the true optical gain reduction, plus the make-up difference and the +2.14, plus the probe's share):

| optical make-up | cells | mean | rms | max |
|---|---|---|---|---|
| 12 | 275 | -0.06 | 0.14 | 0.97 |
| 6 | 8 | -0.55 | 0.57 | 0.74 |
| 18 | 8 | +0.66 | 1.38 | 2.25 |
| 1 / 24 | 8 / 6 | -3.15 / +0.36 | 3.28 / 2.55 | 4.85 / 5.16 |

Had the detector seen the in-situ optical output (the alone output plus the excess), it would read +5.7 dB more on average over the 56
cells with excess above 2 dB (+7.6 to +10.6 at the heaviest). So the discrete detector is fed with the optical divider output as the loop
produces it, and the extra signal is added after the detector tap. (The make-up 1 / 18 / 24 rows are a separate item, section 7.)

### 3.5 The make-up axes

Optical 22 into FLOOD 20, L at 0 dBFS. GR_d | optical in situ | excess | L output dBFS rms | k:

| discrete gain | 1 | 4 | 7 | 12 | 18 | 24 |
|---|---|---|---|---|---|---|
| GR_d | 28.34 | 30.45 | 30.63 | 30.88 | 31.42 | 31.66 |
| optical in situ | 7.64 | 12.13 | 12.73 | 13.66 | 16.15 | 17.62 |
| excess | 10.61 | 6.12 | 5.52 | 4.59 | 2.10 | 0.62 |
| L out | -47.2 | -44.5 | -43.9 | -42.8 | -37.8 | -30.3 |
| k | -47.2 | -47.4 | -47.4 | -47.5 | -48.2 | -50.5 |

| optical gain | 1 | 6 | 12 | 18 | 24 |
|---|---|---|---|---|---|
| GR_d | 11.93 | 26.93 | 30.88 | 40.79 | 48.25 |
| optical in situ | 15.40 | 14.10 | 13.66 | 12.45 | 10.30 |
| excess (vs alone at the same make-up) | 1.69 | 4.04 | 4.59 | 5.90 | 7.50 |
| L out | -39.8 | -42.2 | -42.8 | -44.1 | -45.5 |
| k | -51.8 | -47.8 | -47.5 | -47.3 | -47.2 |

The extra signal is the same absolute level (-47.2 to -47.5 dB of the input) at discrete gain 1 to 12 (a 13 dB change of make-up) and at
optical gain 6 to 24 (a 19 dB change): it passes through neither make-up. The excess in dB is simply the ratio of that fixed signal to a
main path that the make-ups move. The two ends (discrete gain 24, optical gain 1) are the small-excess corners where the alone
contamination dominates the k estimate; against the true loop they give the same κ (200.7 and 200.1).

With the discrete stage compressing nothing (threshold 1) and both make-ups at 1 (the main path 24 dB down), the excess is 1.91 / 2.60 dB
at 0 / +6 dBFS (optical 22) and 2.14 / 2.94 (optical 24): the bleed needs no discrete gain reduction at all; it is a property of the
optical stage (its alone curve at make-up 1 is 1.2 to 2.0 dB under its make-up-12 curve for the same reason).

### 3.6 The bleed against the optical gain alone: the cross-drive grid

R at 20 kHz drives the linked discrete detector (flat there: the discrete stage alone reads 20 kHz 0.3 to 0.8 dB under 1 kHz) while L's
optical loop, whose sidechain low pass is 24 dB down at 20 kHz, sees only L (the optical stage alone gives R at 20 kHz 0.00 dB of gain
reduction up to -10 dBFS). Optical threshold 22, FLOOD 20; k = extra / input, dB, measured against the true optical GR, and the law
κ (1 - g_o):

| L (1 kHz) | GR_o alone | GR_d (R at -20 dBFS) | k measured | k law | GR_d (R at -10) | k measured | k law |
|---|---|---|---|---|---|---|---|
| -30 | 0.00 | 29.93 | none (excess -0.02) | none | 39.56 | none (-0.05) | none |
| -20 | 5.41 | 31.55 | -52.9 | -52.7 | 39.94 | -52.9 | -52.7 |
| -10 | 11.77 | 33.27 | -48.6 | -48.6 | 40.51 | -48.6 | -48.6 |
| 0 | 18.25 | 35.25 | -47.1 | -47.1 | 41.25 | -47.1 | -47.1 |
| +6 | 21.45 | 37.01 | -46.8 | -46.7 | 41.94 | -46.8 | -46.7 |

The same at 4:1 threshold 16 (GR_d 14 to 26): identical k to 0.1 dB. With no optical gain reduction there is no bleed whatever the discrete
stage does (40 dB of it); with it, the extra signal is (1 - g_o) times a constant: the signal the optical cell shunts, `x - v`.

### 3.7 Time: the step

L steps from -50 to 0 dBFS with R at -40, optical 22; per 1 kHz period, the law applied to the measured optical-alone envelope and to
GR_d from R; L measured minus predicted, dB:

| discrete | attack | p1 | p2 | p3 | p5 | p10 | p20 | p50 | p100 | p400 | rms p1..400 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLOOD 20 | 0.1 ms | +0.03 | +0.04 | +0.04 | +0.05 | +0.05 | +0.05 | +0.04 | +0.04 | +0.04 | 0.04 |
| FLOOD 20 | 1 ms | +0.03 | +0.01 | +0.02 | +0.02 | +0.02 | +0.01 | +0.01 | 0.00 | 0.00 | 0.01 |
| FLOOD 20 | 30 ms | -0.01 | -0.02 | -0.02 | -0.03 | -0.04 | -0.06 | -0.15 | -0.21 | -0.22 | 0.21 |
| 4:1 16 | 0.1 / 1 / 30 ms | ≤ 0.05 | | | | | | | | | 0.00 |

No time constant of its own: the excess follows g_o(t) and g_d(t) sample by sample. The audit's "+2.3 dB in the first period, +9.4 at 1 ms,
+10.5 at 5 ms" is the rms of a product of two gains that both fall within the period against the product of their rms values (the two are
positively correlated, so the in-situ rms exceeds the product); from the second period on the law holds to 0.05 dB. The 30 ms row's
-0.2 dB tail is the probe's own baseline gain reduction at that attack setting, not the law.

### 3.8 What the extra signal is: phase, spectrum, harmonics, placement

*Phase and flatness.* Probe tones at -40 dBFS at 100 Hz, 7 and 15 kHz riding on the 1 kHz tone at 0 and +6 dBFS, one instance against the
series of parts, complex lock-in: the extra signal is -47.5 / -47.5 / -47.3 / -47.2 dB of the input at 100 Hz / 1 / 7 / 15 kHz with phase
-2 / 0 / +1 / +1 degrees (0 dBFS, FLOOD 20), -47.0 / -47.0 / -46.8 / -46.9 dB at +6 dBFS; the same at 4:1 22. A clean, flat, in-phase
copy of the input: not through the sidechain low pass (which would lag 100 degrees at 7 kHz), not through the transformer's shelf.

*Harmonics.* One instance against the series at FLOOD 20: H3 -45.7 against -40.3 dBc at 0 dBFS (excess 4.6 dB), -45.9 against -36.7 at
+6 (excess 8.0), -46.9 against -40.3 at 4:1 22 / 0 dBFS (5.6). A clean fundamental added to the series output dilutes its harmonics by the
excess: predicted -44.9, -44.7, -45.9. The optical loop's own ripple is unchanged (a loop that had backed off from 18.3 to 13.7 dB would sit
at -43 dBc); the cell is where the alone measurement says it is.

*Placement.* At the bleed-dominated cell (optical 22 into FLOOD 20, discrete gain 1, +6 dBFS: the output is 15 dB above the series) the
L output is -42.147 / -42.165 / -42.134 dBFS on Nickel / Iron / Steel, whose gains differ by -0.18 / +0.14 dB; at discrete gain 24
(main-path dominated) it follows them, -29.906 / -30.050 / -29.797. The bleed is added after the transformer. With the SIDECHAIN FILTER in
the cell is 0.01 dB different (the +0.64 trim is in the main path, not in the bleed).

## 4. The fit

`out = G0 g_o g_d x + κ (1 - g_o)^p h(GR_d) x`, `G0` the chain's no-compression gain at the setting (both make-ups, the +2.14, the core),
`g_d` from R, `g_o` solved from the alone measurement with the same κ. Fitted on the 186 cells with excess ≥ 0.3 dB and optical make-up ≥ 6,
evaluated on all 411:

| form of h(GR_d) | κ dB | p | h parameters | h(0) | rms / max on the fitted cells | rms / max on all cells |
|---|---|---|---|---|---|---|
| constant (h = 1) | -46.02 | 1 | | 1 | 0.01 / 0.13 | 0.01 / 0.13 |
| constant, p free | -46.02 | 1.00 | | 1 | 0.01 / 0.13 | |
| ramp in dB, min(0, a (GR_d - D)) | -46.01 | 1 | a -0.0002, D -11 | 1.00 | 0.01 / 0.13 | 0.01 / 0.13 |
| h0 + (1 - h0)(1 - g_d)^q | -46.02 | 1 | h0 1.01, q 1.3 | 1.01 | 0.01 / 0.12 | |
| 1 - (1 - h0) / (1 + (GR_d / D)^m) | -46.02 | 1 | h0 1.01, D 9, m 2.4 | 1.01 | 0.01 / 0.12 | |

Every form collapses to the constant. The constant on subsets (h = 1, p = 1):

| subset | κ dB | 1/κ | rms / max (dB) | n |
|---|---|---|---|---|
| all | -46.02 | 200.0 | 0.009 / 0.126 | 186 |
| FLOOD / 4:1 / 2:1 only | -46.02 | 200.1 / 200.0 / 200.0 | 0.015 / 0.002 / 0.000 | 69 / 96 / 21 |
| excess over 3 dB | -46.02 | 200.0 | 0.002 / 0.015 | 81 |
| discrete gain 1 / 18-24 | -46.02 / -46.05 | 200.1 / 200.7 | 0.004 / 0.034 | 15 / 13 |
| optical gain 6 / 18-24 | -46.02 / -46.03 | 200.0 / 200.1 | 0.000 / 0.004 | 7 / 15 |
| optical threshold 14 / 24 | -46.02 | 200.0 | 0.000 / 0.000 | 20 / 21 |
| discrete at threshold 1 (no discrete GR) | -46.03 | 200.1 | 0.000 | 4 |
| L at -10 / +6 dBFS | -46.02 | 200.0 / 200.1 | 0.001 / 0.016 | 20 / 66 |

The 0.13 dB worst case is one FLOOD cell at +6 dBFS with discrete gain 24 (excess 1.5 dB); the L +6 subset carries the plateau cells. A
constant of exactly 0.005 is a designed number, not a fitted one.

## 5. The law in the engine's terms

Per channel, in `Engine::internal`, with `Opto.hpp`'s divider (`v = g xa`, `g = 1/(1 + c)`):

```
bleed = o_bleed * (xa - v) = o_bleed * xa * c / (1 + c)        xa the optical amplifier's output (x with its b2, b3 terms), c the cell conductance
wet   = transformer(discrete(inter * makeup_o * v)) + bleed
```

- **Where.** After the transformer, before the mix. It is not in the optical sidechain (the loop's gain reduction is the alone value to
  0.14 dB rms, section 3.4), not in the discrete detector, not through either make-up, not through the discrete cell or the core.
- **What signal.** `xa - v`, the signal across the optical divider's series element (the cell current times the series resistor), including
  the idle leak conductance `cond0` (at threshold 24 the leak alone gives `(1 - g) = 0.05`, a -72 dB bleed, which is what the +2.40 / +1.99
  at optical gain 1 in section 3.1 read). Whether the reference takes `x` or `xa` cannot be told (b2, b3 are 1e-5, 6e-4).
- **Calibration.** One field, `o_bleed` = 0.005 (-46.02 dB). It does not depend on threshold, make-up, ratio, attack, recover, transformer,
  mode or the filter. Fitted 1/κ: 200.0 ± 0.3 across every subset.
- **With the optical stage out** the bleed is out (c = 0 gives `xa - v = 0` anyway). With the discrete stage out it stays (the optical
  alone curves carry it): the optical bypass crossfade should fade it with the stage, the discrete bypass must not touch it.
- **Profiles.** It is a property of the reference's model; no hardware evidence speaks to it either way. Keep it in REFERENCE and HARDWARE
  (the dynamics come from the reference in every profile); a `-inf dB` value switches it off if a profile wants a clean chain.

**What it does to the fit.** The optical statics captured alone contain the bleed: at make-up 12 the reference's alone gain reduction is
under the loop's by 0.05 dB at 5 dB of gain reduction, 0.31 at 18, 0.47 at 22 (threshold 22; the same at 14 and 24), so stage 4b/4c fitted
knees and slope to a curve that is 0.3 to 0.5 dB shallow at the top. Once the engine renders the bleed, the same protocol items fit the true
loop (the fit renders through the engine); expect the slope `o_gamma / (1 + o_gamma)` and the upper knees to move by that much, and the
`opto_static_t*` items at make-up 12 to close rather than open. The `both` protocol group and the audit's A and B cases (-0.5 to -3.8 dB
remainders, all on the compressing channel and all at low output levels) are this bleed seen on programme: at setting B on basshats the
main path sits 40 dB down and the bleed dominates.

## 6. What remains unexplained

- **The discrete detector at optical make-up 1, 18 and 24.** GR_d read on R follows the reference's own discrete law at make-up 12 to
  0.14 dB rms, but at make-up 1 it reads 3.2 dB less than a -14 dB level shift predicts (FLOOD 20: 11.9 measured, 17.2 predicted), at 18
  +0.7 more, at 24 up to +1.8 more at FLOOD 20 and 5 dB *less* at 4:1 22 (41.4 against 46.6 near that curve's plateau). The audio path's
  make-up is right (the inter-stage gain is +2.14 ± 0.15 at every position, section 3.1), so it is the sidechain that sees the optical
  make-up at other than 1 dB per dB at the ends of its range, or the discrete law's plateau behaves differently when reached through the
  optical make-up. Not part of the bleed (the bleed fit is exact at those cells because GR_d is measured, not predicted); a separate
  discrete-detector item, to be measured as GR_d against optical make-up at fixed detector level.
- **The optical loop's own make-up dependence.** With the bleed removed, the loop's gain reduction still varies with OPTICAL GAIN by
  about -0.2 (position 3), +0.1 (18), -0.5 dB (24) around position 12 (alone, threshold 22, 0 dBFS: 18.06 / 18.13 / 18.25 / 18.35 / 17.80 at
  3 / 6 / 12 / 18 / 24, and 17.10 at position 1 of which 1.2 dB is bleed). The section-2 tap test ("10.1 to 10.5 dB at every position, 9.9
  at 24") is this; it is small, non-monotone, and not modelled.
- **The first period after a step** cannot be tested by per-period rms (section 3.7); a sample-level comparison against the engine once the
  bleed is in would settle whether anything else happens in the first millisecond.
- **The step at 30 ms** shows -0.2 dB at the tail (the probe's baseline at that attack was not measured separately); not a law item.
- **Physical meaning.** A fixed 0.5 % of the optical cell's current appearing at the output after the transformer is not a circuit the
  hardware could have; it reads as a modelling choice in the reference (or a mixing constant in its optical block). No hardware evidence
  exists for or against it.

## 7. Files

- `fit/tools/candidates/stage-interaction.py`: `--parts measure` (grid, make-ups, step, waveform), `measure2` (make-up baselines,
  cross-drive, probe tone, H3), `measure3` (baselines, phase), `measure4` (no-discrete-GR cells, optical make-up sweep, discrete threshold
  sweep), `measure5` (link symmetry at the plateaus), `measure6` (transformer, mode, filter placement), `measure7` (inter-stage gain against
  make-ups), `analyse` (cells, detector check, fits, step, hypotheses on the engine, engine with the law).
- `build/stage-interaction/measure.json` (the cache), `tables.md` (every table in full), `fit.json` (the fitted constants), `ours.json`
  (our engine's renders on the same probes), `*.log`.
