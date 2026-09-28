# Transformer even-order fix: the driver's DC, the knee asymmetry's decay and the saturation pulse

Status: design document, 2026-09-28. Synthesises the three transformer even-harmonic hypothesis harnesses under
`fit/tools/candidates/xfmr-even-*.py` (keys `xfmr-even-a-driver`, `xfmr-even-b-remanence`, `xfmr-even-c-output-stage`, the last
one adversarially verified), their logs (`build_xfmr_even_*.log` in the repository root, the `.result.<core>.<variant>.json` files
next to the remanence harness), and one synthesis run of my own (`fit/tools/candidates/xfmr-even-synthesis.py`, logs
`build_xfmr_even_synthesis.log` and `build_xfmr_even_synthesis_cpp.log`, JSON in `/tmp/xfmr-even-synthesis*.json`) that puts the
surviving pieces on one mirror and one metric and adds the one form none of the keys ran. Nothing under `src/`, `fit/stages/` or
`fit/data/` has been changed; this document says exactly what to change.

The problem: stage 2 (`fit/stages/stage2_transformers.py`, `src/dsp/Transformer.hpp`) fits the transformer grid's gain within
0.3 dB and H3 within about 1 dB, but its even harmonics are 10 to 25 dB too strong wherever the reference sits below -75 dBc
(Iron 60 Hz +18 dBFS: model H2 -47 dBc, reference -66.5). The test suite (`tests/pb_reference.py`, group `xfmr`) reports even-order
error 13.0 dB rms over the grid, odd 3.5. The cause is not one thing but three, and each of the keys found one of them.

Conventions. Levels are dBFS peak of a sine, 1.0 = 0 dBFS = +14 dBu. The grid is `xf_{core}_f{f}_{level}`, f in {20, 30, 40, 60,
80, 120, 160, 320, 1000, 5000} Hz, level -12..+21 dBFS in 3 dB steps (the +24 dBFS row is the reference's output ceiling and is
pooled apart as `xfmr_ceiling` by the test suite; every harness excluded it from the fit). Harmonics are dBc; "even" pools H2/H4/H6/H8
and "odd" H3/H5/H7 where the reference's harmonic is above -80 dBc, the model floored at -120 dBc (the test suite's metric); rms/max
in dB per core, "pooled" is the n-weighted rms over the three cores. The flux drive `D = level - 20 log10(f / 20)` dB is the level
of a 20 Hz sine with the same peak flux (the 6 dB per octave law of the core). All fits use the stage-2 residual
(`fit/common.feat_residual`: gain x5, H2/H3 x1, H4/H5 x0.5, H6..H8 x0.25) with bounded least squares (soft L1, f_scale 2, max 60
function evaluations) per core; every figure below is from a run whose log is named.

## 1. The evidence

### 1.1 The reference law

From `python3 -u fit/tools/candidates/xfmr-even-output.py --law` (rerun 2026-09-28, `/tmp/xfmr-even-law.log`; the three keys'
extractions agree):

| feature | Nickel | Iron | Steel |
|---|---|---|---|
| below the knee: H2 at 1 kHz, -12..+21 dBFS | -119.5 .. -86.6 dBc, 1.0 dB/dB (a2 = 8.3e-6) | -99.7 .. -66.6, 1.0 dB/dB (a2 = 8.4e-5) | -119.5 .. -86.6 (Nickel's) |
| at the onset, D = 0 / 3 / 6 dB, 20 Hz: H2 | -78.8 / -60.5 / -59.4 | -78.3 / -60.4 / -59.6 | -78.8 / -60.5 / -59.7 |
| the same at 40 Hz (+6 / +9 / +12) | -78.9 / -60.6 / -60.8 | -77.2 / -60.4 / -60.6 | -78.8 / -60.5 / -60.7 |
| the same at 80 Hz (+12 / +15 / +18) | -78.8 / -59.7 / -59.2 | -74.0 / -59.3 / -57.7 | -78.7 / -59.5 / -58.0 |
| H4 at D = 0 / 3 (20 Hz): equal to H2 within | 0.0 / 0.3 dB | 0.3 / 0.1 | 0.0 / 0.2 |
| the floor at 20 Hz, +9 .. +21: H2 | -63.3, -64.4, -65.8, -67.3, -68.9 (-0.47 dB/dB) | -65.3, -65.6, -65.8, -65.6, -64.9 | -66.0, -67.4, -69.0, -70.7, -72.5 (-0.54 dB/dB) |
| the floor at equal drive D = 9: 20 / 40 / 80 Hz | -63.3 / -71.0 / -80.2 (8.5 dB per octave) | -65.3 / -67.7 / -63.7 | -66.0 / -72.0 / -77.1 (5.5 dB per octave) |
| floor minus the driver's own H2 (1 kHz line), 20 Hz +9 .. +21 | +35.3 .. +17.7 | +13.5 .. +1.7 | +32.7 .. +14.1 |
| the 120 / 160 Hz +21 dBFS points (D = 5.4 / 3): H2 | -47.5 / -54.1 | -46.5 / -53.6 | -46.6 / -53.8 |
| H3 at equal drive D = 3 / 9: 20 vs 80 Hz | -40.8 / -8.4 vs -42.2 / -8.4 | -43.6 / -9.0 vs -75.6 / -9.5 | -43.3 / -9.1 vs -45.1 / -9.1 |

What a mechanism must produce, in the fitted range:

1. Below the knee, only the driver's quadratic: 1 dB per dB, the same at every frequency, so the driver's even term must not
   magnetise the core (a flux offset proportional to A^2 would put a 2 dB/dB term there, and does in the model as built).
2. At the flux-law onset a burst with a flat even spectrum (H2 = H4 within 0.3 dB, an impulsive once-per-cycle feature) of
   -60 dBc at D = 3 .. 6 that is the same on all three cores (Iron's driver even term is ten times Nickel's and its burst is the
   same) and the same at 20, 40 and 80 Hz: the burst is a flux-domain feature, a function of D.
3. Past the burst a floor that decays with drive at fixed frequency (about -0.5 dB/dB on Nickel and Steel) and, at equal flux
   drive, falls 5 to 8 dB per octave with frequency. The second is the discriminating fact: any static or history-dependent
   asymmetry of the flux curve gives a feature of fixed flux shape at equal D, the differentiator scales it with omega exactly as it
   scales the fundamental, and its dBc comes out the same at every frequency. A feature whose dBc falls 6 dB per octave at equal D has
   a fixed voltage size, not a fixed flux size. Iron's floor is its driver's H2 passing through (1.7 to 13.5 dB over the 1 kHz line).
4. The odd order is untouched by any of this: H3 at equal D is the same at 20 and 80 Hz within 1.5 dB (a flux-law saturator with a
   hard ceiling), and the fix must not move it (odd rms must stay under 3.5 dB pooled).

Direction (c), an even term after the core, is refuted analytically before any fit (output key, `--law`): a quadratic after the
core sees `y = A1 cos + A3 cos 3 + ...`, its 2f term is `a2 (A1^2 / 2 + A1 A3 + ...)`, the 1 kHz line pins a2, and the most it can add
at the onset is `20 log10(1 + 2 h3)` = +0.15 dB at h3 = -41 dBc: -104.5 dBc where the reference has -60.5 (Iron -84.6 against -60.4).
An even term that is a function of the flux (not of y) is not covered by that bound; that is what the synthesis run tests.

### 1.2 The candidates

Even / odd figures are rms / max in dB per core (Nickel / Iron / Steel), pooled in brackets; "current" is the shipped
`fit/data/constants.json` rendered through each key's validated mirror (0.010 dB worst against the C++ engine).

| hypothesis (harness, log) | what it is | even rms/max N / I / S [pooled] | odd rms N / I / S [pooled] | gain rms N / I / S | verdict |
|---|---|---|---|---|---|
| current model (stage 2 as built) | a2 u^2 + a3 u^3 before the flux core, leak r phi, static knee-hardness asymmetry qp/qn = q (1 +- asym) | 11.83/25.0, 13.77/33.7, 13.22/28.8 [13.04] | 2.53/11.6, 5.02/49.4, 1.73/6.4 [3.54] | 0.040, 0.057, 0.019 | baseline |
| (a) `xfmr-even-driver.py` sym / a2env / classA / caenv (`build_xfmr_even_driver_{Nickel,Iron,Steel,post}.log`) | symmetric core, even order from the driver alone: fixed a2; a2 (1 + k env); asymmetric tanh ceiling before the core; the repo's ClassA.hpp module | sym 19.77, 16.61, 20.97 [18.99]; a2env [19.19]; classA [18.47]; caenv [19.12] | sym [3.42]; classA [2.11] (Iron's odd 4.95 -> 2.28 from a symmetric ceiling at +36 dBFS in its driver) | 0.040 / 0.036 | refuted: the onset burst is missing (20 Hz +3: sym H2 -101.7 dBc Nickel, -85.2 Iron, against -60.5 / -60.4) |
| (a) dc / dcenv / dcdec | symmetric core with a standing bias in the primary (constant, envelope-shifted, collapsing with drive) | dc 13.30, 13.44, 15.90 [14.19]; dcenv [12.74]; dcdec [15.20] | [3.27 .. 3.39] | 0.082 / 0.055 | refuted: the bias makes the burst (dc: -70 / -62 dBc at 20 Hz +3) but is then 12-23 dB too loud deep in (Nickel 20 Hz +12: -53.0 against -64.4); the required flux offset spans 100x across frequency at one level (`--offsetmap`), which a level-only mechanism cannot deliver |
| (a) acdc / acdec | the same with the driver's rectified DC removed from the drive | acdc 13.06, 11.25, 15.40 [13.17]; acdec 12.48, 10.97, 14.49 [12.58] | [3.32 / 3.34] | 0.063 / 0.058 | refuted for the same reason (even max 26-30) |
| (a) acasym (control) | the current knee asymmetry, driver AC-coupled | 10.03/24.6, 9.37/27.1, 10.62/31.2 [9.96] | 1.84, 5.09, 1.84 [3.46] | 0.046, 0.047, 0.020 | supported: the driver's DC integrating into the flux is a third of the excess |
| (b) `xfmr-even-remanence.py` V0 (`build_xfmr_even_r5_*_V0.log`) | the C++ refitted with a quiet-floor residual (control) | 10.77/26.3, 13.33/31.5, 12.69/28.2 | 1.60, 4.85, 1.67 | 0.073, 0.127, 0.025 | baseline of that key |
| (b) V2 / V6 | remanence proper: a flux offset built from the excess flux, relaxing with tau (V6 with the driver AC-coupled) | V2 13.79, 14.65, 14.90; V6 31.2, 21.4, 17.2 | 2.43 / 1.85 (N) | | refuted: the offset's mean is a DC bias proportional to the rectified excess, so the even order grows with depth |
| (b) V3 | depth-decaying asymmetry asym / (1 + (env / d0)^2), env a peak tracker of the flux with release tau | 6.52/16.1, 13.93/35.1, 6.43/15.4 | 1.60, 4.93, 1.65 | 0.071, 0.064, 0.023 | Nickel/Steel halved, Iron not: its ten-times driver DC still magnetises the core |
| (b) V5 | V3 with the driver AC-coupled (one-pole high pass at fdc) | 6.28/16.3, 6.49/23.1, 7.14/28.1 [about 6.6] | 1.75, 4.92, 1.68 | 0.046, 0.065, 0.020 | supported; d0 at its 0.3 bound on Iron and Steel, fdc 0.6-0.9 Hz, tau 177-297 ms |
| (b) V8 | peak-hold power-law decay asym / max(env, 0.3)^p, driver AC-coupled; p, fdc, tau free | 6.06/15.8, 6.14/21.5, 7.11/27.6 [about 6.4] | 1.63, 5.00, 1.78 | 0.064, 0.072, 0.019 | the key's best, but Iron's fdc ran to 3.84 Hz and breaks the fitted linear response (`xf_resp_Iron` max 0.597 dB against 0.162 for V0), and Nickel's tau ran to the 2 s bound (the render length): two of its three constants are not identified by steady sines |
| (b) V9 | V8 with p = 2, fdc = 1 Hz, tau = 300 ms fixed (no new field) | 9.30/27.0, 6.99/23.0, 8.73/33.4 [about 8.3] | 1.76, 4.90, 1.66 | 0.044, 0.065, 0.022 | safe form; p = 2 costs Nickel 3 dB |
| (c,d) `xfmr-even-output.py` B0 (`build_xfmr_even_output.log`, rerun `/tmp/xfmr-even-output-rerun.log` identical) | the C++ refitted on the <= +21 grid (control) | 11.89/25.1, 13.23/31.4, 13.35/28.9 [12.88] | 2.50, 4.88, 1.66 [3.44] | 0.039, 0.138, 0.023 | baseline of that key |
| (c) C1 / C2 / C1L | a2 after the core; a2 before but AC-coupled (50 ms mean of u^2 removed); C1 with a level limit | C1 9.89/24.5, 8.82/30.5, 10.61/31.3 [9.72]; C2 10.00, 9.41, 10.62 [9.97]; C1L [9.76] | C1 1.98, 5.10, 1.86 [3.49] | 0.034, 0.046, 0.021 | supported as the DC finding only: C1 = C2, so the whole gain is the removal of the driver's rectified DC from the flux integrator; C1L's limit runs to 29-47 (inactive) |
| (d) D1 / D8 | the leak on the saturated flux, phi' = v - r S(phi) (D8 with a ceiling asymmetry) | D1 25.78/63.1, 10.63/30.6, 27.24/67.9; D8 25.23, 18.63, 29.10 | 1.99, 5.00, 1.37 | 0.029 .. 0.129 | refuted: the leak vanishes in saturation and the flux walks off (even max 62-68) |
| (d) D4 / D7 | a rate loss re y in the integrator; a polarity-asymmetric leak r (1 +- rasym) phi | D4 9.94, 8.98, 10.57 [9.78]; D7 9.83, 8.58, 10.51 [9.58] | [3.5 .. 3.7] | 0.090 / 0.041 | refuted: re fits to 0.007-0.011 and rasym to 2e-5 from a 0.1 start: the loss term carries no even-order information |
| (c) D3 / D5 / D6 | ceiling asymmetry phik (1 +- asym); flux offset; both | D3 14.44, 12.84, 16.29 [14.5]; D5 13.03, 11.53, 15.62 [13.4]; D6 [11.2] | | | refuted: even order growing with drive, as before |
| (c) D2 | hardness asymmetry only above a flux level thr | 10.03, 8.77, 10.56 [9.73] | | | refuted: thr runs to 0.17-0.22 phik and D2 collapses onto C1 |
| (c) D2r (the output key's best, verified) | hardness asymmetry only below thr: asym_eff = asym thr^8 / (thr^8 + (phi/phik)^8), symmetric ceiling; a2 after the core | 9.12/23.5, 7.09/27.2, 8.10/24.6 [8.06] | 1.80, 4.95, 1.86 [3.38] | 0.042, 0.060, 0.024 | improves: thr 0.95 / 0.80 / 0.81 phik, asym 0.033 / 0.052 / 0.050; matches the onset within 2 dB at every frequency, leaves the post-burst dip 8-21 dB too deep, the 20 Hz floor 4-6 dB low, the 40 Hz floor 2-4 dB high, the 120/160 Hz +21 points 7-12 dB low |
| synthesis D2r (`build_xfmr_even_synthesis.log`) | the same, refitted on the synthesis mirror (start thr 0.85) | 9.12/23.3, 7.15/22.7, 8.11/24.6 [8.08] | 1.77, 4.93, 1.78 [3.35] | 0.045, 0.066, 0.018 | reproduced (thr 0.948 / 0.818 / 0.800, asym 0.034 / 0.048 / 0.050) |
| synthesis PH / PH2 | V8's decay with fdc and tau fixed (a2 after the core, hold 300 ms): p per core / p = 2 | PH 8.61/25.2, 5.40/25.6, 8.83/33.0 [7.60]; PH2 9.65, 5.37, 8.96 [8.00] | PH 1.91, 4.92, 2.07 [3.41] | 0.032, 0.049, 0.044 | Iron prefers the decay (p = 1.96), Nickel/Steel are indifferent between the gate and the decay (p 0.75 / 1.10) |
| synthesis M1 | the fixed-voltage saturation pulse alone, symmetric knee: y += ke Vk P(phi), P = (phi/phik)^q / (1 + (phi/phik)^q), Vk the 20 Hz knee level | 16.98/41.4, 16.91/59.1, 19.26/40.3 [17.68] | 1.98, 5.00, 1.59 [3.39] | 0.030, 0.056, 0.034 | refuted alone: with no asymmetry the fit sets ke to -0.005 .. -0.014 and the residual grows with frequency (Nickel 11 dB rms at 20 Hz, 32 at 160 Hz): the pulse alone cannot make the burst without spoiling the sub-onset points |
| synthesis M1a / M1r | the pulse with the static asymmetry / with the D2r gate | M1a 9.44, 9.13, 10.55 [9.67]; M1r 7.53/23.8, 7.16/22.9, 7.84/24.9 [7.48] | [3.48 / 3.35] | | with the static asymmetry the fit turns the pulse off (ke -0.0007); with the gate it helps Nickel and Steel (ke 0.0014 / 0.0011) and not Iron (ke -0.0004) |
| synthesis M1p | the pulse with the decay (a2 after the core) | 5.93/15.7, 6.40/24.6, 6.67/30.8 [6.34] | 2.01, 4.88, 1.95 [3.39] | 0.025, 0.055, 0.032 | the best on Nickel and Steel; Iron's fit landed in a worse minimum than PH (cost 911.6 against 729.6; ke 0.0059), corrected in M1pc below by a third start |
| synthesis D2c / PHc (`build_xfmr_even_synthesis_cpp.log`) | D2r / PH with the driver's even term before the core, AC-coupled (50 ms mean of u^2 removed) | D2c 9.79/33.1, 8.31/22.3, 8.15/24.8 [8.74]; PHc 9.25/28.1, 7.43/22.6, 9.32/33.3 [8.61] | [3.38 / 3.33] | 0.042 / 0.045 | the before-the-core placement costs 0.5-1 dB against the after-the-core one for the gate and the decay alone |
| **synthesis M1pc (adopted)** | the pulse with the decay, driver even term before the core and AC-coupled, the pulse DC-blocked the same way (the C++-ready form) | **5.65/15.2, 5.71/25.0, 6.61/30.2 [5.98]** | **2.05/9.3, 4.96/49.4, 1.97/6.9 [3.44]** | **0.023/0.10, 0.044/0.24, 0.034/0.14** | adopted: the same three constants on every core (|asym| 0.012 / 0.014 / 0.012, p 1.65 / 1.99 / 1.76, |ke| 0.0015 / 0.0016 / 0.0011), which is the reference's "same burst on every core"; the linear response is untouched; the odd order and the gain are unchanged |

Three things survive every key and the synthesis:

1. **The driver's rectified DC must not reach the flux integrator.** `a2 u^2` has a mean `a2 A^2 / 2`; the leaky integrator gives
   it a DC gain of `1 / r` and turns it into a flux offset that grows with `A^2`, which is the 0.5-1 dB/dB growth of the model's even
   order where the reference's decays. Every key found it independently (acasym, V1/V5/V8, C1/C2) and it is worth 3 dB rms on its
   own (13.0 -> 9.7-10.0 pooled) at unchanged odd order. Physically the primary is AC-coupled to the driver (or, for Iron's gapped
   core with a DC-biased Class-A stage, the standing current is a constant bias absorbed into the operating point, not a
   level-dependent one).
2. **The knee asymmetry must not persist at the ceiling, and it must know the peak.** A static asymmetry confined to the knee
   (D2r, memoryless, instantaneous flux) and an asymmetry that decays with the peak flux the core was recently driven to (V5/V8/PH,
   one held state) both take the pooled even from 9.7 to 7.6-8.1; the decaying form is 1.5 dB better on Iron and, combined with
   the pulse, 1.5 dB better on all cores than the gate (M1p [6.34] against M1r [7.48]). The reference's floor falls 0.5 dB/dB with
   drive at fixed frequency, and the asymmetry at the knee crossing has to shrink with the peak the cycle reaches, which an
   instantaneous curve cannot do (at the crossing it does not know the peak). The hold's release (300 ms) is not identified by
   steady sines: anything longer than a cycle gives the same grid.
3. **An even-symmetric feature of fixed voltage size at each excursion into saturation.** This is what the 5-8 dB per octave law
   asks for and what no depth-only law can give. Fitted as `ke Vk P(phi)` with `P` the saturation indicator and `Vk` the 20 Hz knee
   level, it is 0.11-0.16 % of the knee level (-56 to -59 dB) on all three cores, its sign opposite on Iron. On its own it is refuted
   (it cannot make the burst without spoiling the sub-onset points), with the static asymmetry the fit switches it off, with the
   decaying asymmetry it takes the pooled even from 7.6 to 6.0 and flattens the 20-80 Hz floors from -5..-13 dB (D2r) to +-3 dB.
   The physical reading (an interpretation, not a measurement): the driver's source impedance differs between sourcing and
   sinking, so the magnetising-current pulses a saturating core draws on both polarities leave an even-symmetric voltage drop
   `(R+ - R-) / 2 |i_m|` of the same size at every frequency, a narrow pulse per half cycle at the onset (a flat even spectrum) and a
   notch at the zero crossings deep in that narrows with drive (the even order decays). An even function of a half-wave-symmetric
   waveform has only even harmonics and DC, so it cannot touch the odd order (the fits confirm: odd 3.44 against 3.35-3.49 for
   everything else). The DC is not physical (a secondary passes none) and is removed with the same 50 ms mean as the driver's.

The remanence proper (a relaxing flux offset), a flux offset, a ceiling asymmetry, the loss term in any form, and any driver-only
mechanism with a symmetric core are refuted with the figures above; none is worth another round.

## 2. The chosen model

### 2.1 In words

Keep the driver polynomial, the leaky flux integrator, the hard ceiling `S(phi)` with its knee hardness `q`, the differentiator and
the linear tilt. Change three things in `TransformerCore`:

1. **AC-couple the driver's even term.** The core is driven by `u + a2 (u^2 - <u^2>) + a3 u^3`, `<u^2>` a 50 ms one-pole mean.
   The linear path and the odd term are untouched; only the even term's rectified DC is kept out of the flux.
2. **Let the knee asymmetry decay with the held peak flux.** `asym_eff = x_asym / max(env, 0.3)^x_asym_p`, `env` the peak of
   `|phi| / phi_k` held with a 300 ms release; `qp / qn = q (1 +- asym_eff)`. At the onset (`env` about 1) the asymmetry is `x_asym`;
   at D = 12 dB (`env` = 4) it is 10-16 times smaller. Below `env` = 0.3 the core is linear and the floor has no effect.
3. **Add the saturation pulse on the core's output.** `y += x_pulse Vk (P - <P>)`, `P = az^q / (1 + az^q)` with `az = |phi| / phi_k`
   and the symmetric `q`, `Vk = 10^(x_sat_db / 20)` the level of a 20 Hz sine at the knee, `<P>` the same 50 ms mean.

### 2.2 Equations, in the order the C++ runs them

Per sample; `x` the block input after the position gain, `fs` the sample rate, `T = 1 / fs`; per core `g = 10^(x_gain_db/20)`,
`r = 2 pi x_fl_hz`, `Vk = 10^(x_sat_db/20)`, `phi_k = Vk / (2 pi 20)`, `kc = 1 - exp(-1 / (0.05 fs))`, `kh = 1 - exp(-1 / (0.3 fs))`:

```
u    = g x
uc   = clamp(u, -uf, uf),  uf = 1 / sqrt(-3 a3)                     the driver's fold clamp                       [unchanged]
u2   = uc^2
m2  += (u2 - m2) kc                                                 running mean of the even term (50 ms)         [NEW]
v    = uc + a2 (u2 - m2) + a3 u2 uc                                 driver, its even term AC-coupled              [changed]
phi += T (v - r phi)                                                leaky flux integrator                         [unchanged]
az   = |phi| / phi_k
env  = az > env ? az : env + (az - env) kh                          peak flux held, 300 ms release                [NEW]
ae   = asym / max(env, 0.3)^p                                       the asymmetry seen by this sample             [NEW]
qp   = max(q (1 + ae), 1),  qn = max(q (1 - ae), 1)
S    = phi / (1 + az^(qp or qn))^(1 / (qp or qn))                   hard ceiling, polarity's hardness             [unchanged form]
y    = (S - S_prev) fs;  S_prev = S                                 differentiator                                [unchanged]
P    = az^q / (1 + az^q)   (0 for az <= 0.02, 1 where q ln az > 700) saturation indicator, symmetric q            [NEW]
mP  += (P - mP) kc                                                  its running mean (50 ms)                      [NEW]
y   += pulse Vk (P - mP)                                            the saturation pulse, DC-blocked              [NEW]
y    = LP(HS(y)) ...                                                high shelf, low pass, profile extras          [unchanged]
```

`P` reuses `ln az` from `S`: one extra `exp` per sample for the indicator and one `exp`/`log` pair for `env^p`; no iteration.

### 2.3 Parameters

Fitted values from `build_xfmr_even_synthesis_cpp.log` (M1pc), Nickel / Iron / Steel; the linear fields are the shipped ones
(unchanged by this fit, `fit/data/constants.json`).

| symbol | meaning | unit | Nickel | Iron | Steel | calibration field |
|---|---|---|---|---|---|---|
| `a2` | driver even term, on the driver input, AC-coupled | 1 | 8.274e-6 | 8.326e-5 | 8.137e-6 | `x_a2` (existing; bounds now positive) |
| `a3` | driver odd term (compressive) | 1 | -1.3753e-4 | -8.4828e-4 | -1.3174e-4 | `x_a3` (existing) |
| `sat_db` | knee: level of a 20 Hz sine whose flux peak is `phi_k` | dBFS | 5.502 | 5.330 | 5.861 | `x_sat_db` (existing) |
| `q` | knee hardness | 1 | 8.434 | 8.884 | 8.474 | `x_q` (existing) |
| `asym` | knee-hardness asymmetry at the onset (`env` = 1) | 1 | -0.01224 | +0.01397 | -0.01240 | `x_asym` (existing; meaning changed: at the onset, before the decay) |
| `p` | decay of the asymmetry with the held peak flux | 1 | 1.648 | 1.990 | 1.764 | `x_asym_p` (new) |
| `pulse` | saturation pulse size relative to the knee level `Vk` | 1 | +1.473e-3 (-56.6 dB) | -1.638e-3 (-55.7 dB) | +1.128e-3 (-59.0 dB) | `x_pulse` (new) |
| 0.05 | coupling mean's time constant (driver even term, pulse) | s | fixed | fixed | fixed | code constant (`kCoupleTau`) |
| 0.3 | the peak hold's release | s | fixed | fixed | fixed | code constant (`kHoldTau`); not identified by the grid, see section 4 |
| 0.3 | the decay's floor in knee-flux units | 1 | fixed | fixed | fixed | code constant (`kEnvMin`) |
| `gain_db`, `fl_hz`, `hs_hz`, `hs_db`, `lp_hz` | linear tilt | dB, Hz, Hz, dB, Hz | -0.0288, 1.7087, 9512, -0.2007, 71904 | -0.2152, 4.3845, 12205, -0.3759, 71724 | 0.1080, 1.4395, 14234, -0.6726, 24375 | existing, unchanged |

The signs: with `a2 > 0` on every core, Nickel and Steel fit `asym < 0, pulse > 0` and Iron `asym > 0, pulse < 0`; `(a2, asym, pulse)
-> (-a2, -asym, -pulse)` flips the phase of every even harmonic together and is a symmetry of the magnitudes, so `a2` is held
positive and the relative signs are what the fit identifies. Nickel and Steel, which the reference gives the same nonlinearity to
0.1 dB, get the same signs and the same magnitudes; Iron's Class-A stage is the mirror image.

### 2.4 The chosen model's residual

Test-suite metric, levels -12..+21 dBFS, `build_xfmr_even_synthesis_cpp.log`:

| | Nickel | Iron | Steel | pooled |
|---|---|---|---|---|
| gain rms / max | 0.023 / 0.10 | 0.044 / 0.24 | 0.034 / 0.14 | 0.034 |
| odd rms / max (n) | 2.05 / 9.3 (128) | 4.96 / 49.4 (158) | 1.97 / 6.9 (129) | 3.44 |
| even rms / max (n) | 5.65 / 15.2 (93) | 5.71 / 25.0 (120) | 6.61 / 30.2 (93) | 5.98 |
| per order rms / max: H2, H4, H6, H8 | 4.2/12, 6.1/15, 6.7/13, 5.8/11 | 3.0/11, 5.4/13, 8.4/25, 8.4/14 | 4.2/13, 6.0/15, 7.6/14, 9.9/30 | |
| per order rms / max: H3, H5, H7 | 0.6/1, 2.0/5, 3.4/9 | 1.9/15, 9.5/49, 2.9/6 | 0.7/2, 2.0/4, 3.2/7 | |
| even per frequency rms: 20 / 30 / 40 / 60 / 80 / 120 / 160 Hz | 5.5 / 4.7 / 3.6 / 5.9 / 4.8 / 9.0 / 9.2 | 4.7 / 5.3 / 4.3 / 5.4 / 5.8 / 10.4 / 8.4 | 4.2 / 5.3 / 3.9 / 5.3 / 6.2 / 10.5 / 15.2 | |
| worst even point | `xf_Nickel_f120_21` H4 | `xf_Iron_f120_21` H6 | `xf_Steel_f160_21` H8 | |
| worst odd point | `xf_Nickel_f20_12` H7 | `xf_Iron_f1000_21` H5 (the cubic driver, every variant) | `xf_Steel_f20_12` H7 | |

Against the shipped model: even 11.83 / 13.77 / 13.22 -> 5.65 / 5.71 / 6.61 (pooled 13.04 -> 5.98), odd 2.53 / 5.02 / 1.73 ->
2.05 / 4.96 / 1.97 (3.54 -> 3.44), gain 0.040 / 0.057 / 0.019 -> 0.023 / 0.044 / 0.034. Model H2 minus reference H2 in dB where the
reference sits above -80 dBc (columns -12 .. +21 dBFS in 3 dB steps; "." where the reference is below -80 dBc):

```
Nickel                                                          Steel
   20  .  .  .  .  -0.7  -4.1   0.2   0.7   0.6  -0.5  -1.7  -3.0      20  .  .  .  .  -1.3  -4.8   0.6   1.2   1.4   0.5  -0.5  -1.5
   30  .  .  .  .     .     .  -3.8  -0.4   1.9   1.4   0.9  -0.4      30  .  .  .  .     .     .  -4.6  -0.5   2.6   1.5   0.9  -0.5
   40  .  .  .  .     .     .  -0.6  -4.0   1.2   2.5   2.6   2.1      40  .  .  .  .     .     .  -1.2  -4.8   1.4   2.0   1.2   0.3
   60  .  .  .  .     .     .     .     .  -4.2  -1.2   9.8     .      60  .  .  .  .     .     .     .     .  -5.0  -1.7   5.3   0.2
   80  .  .  .  .     .     .     .     .  -0.8  -5.2  -0.5     .      80  .  .  .  .     .     .     .     .  -1.5  -6.0  -1.3   3.9
  120  .  .  .  .     .     .     .     .     .     .  -6.4 -11.7     120  .  .  .  .     .     .     .     .     .     .  -7.3 -12.5
  160  .  .  .  .     .     .     .     .     .     .  -1.6 -11.1     160  .  .  .  .     .     .     .     .     .     .  -2.3 -12.0
Iron
   20  .  .  .  .  -0.1  -4.1   1.6   4.4   3.4   1.3  -1.4  -4.2
   30  .  .  .  .     .     .  -3.4   0.5   4.8   2.0  -1.3  -4.8
   40  .  .  .  .     .     .   0.4  -3.5   2.2   3.0  -0.9  -4.7
   60  .  .  .  .     .     .     .   0.6  -3.5  -0.7   2.0  -3.5
   80  .  .  .  .     .     .     .   0.1   0.1  -4.6  -0.4  -0.7
  120  .  .  .  .     .     .     .  -0.0   0.0   0.1  -5.3 -11.3
  160  .  .  .  .     .     .     .  -0.0  -0.0   0.1  -0.1  -9.8
```

For comparison the output key's D2r left the 20 Hz row of Nickel at -0.7, -0.5, 1.5, -13.5, -5.0, -4.7, -5.2, -5.6 and the 30 Hz
row at -0.7, 2.4, -21.1, -1.1, 0.1, -0.2 (`build_xfmr_even_output.log`): the post-burst dip and the floor's frequency law are what
the pulse fixes.

## 3. The changes

### 3.1 `src/dsp/Transformer.hpp`

`CoreParams` gains two fields; `configure()` two coefficients and the knee level; `process()` changes the driver line; `core()`
gains the held peak, the decaying asymmetry and the pulse; `sat()` takes the polarity hardness as arguments; `reset()` and
`seedFrom()` carry the three new states. Everything else (fold clamp, Class-A profile, germanium stage, materials, filters, noise,
heat, the crossfade block) stays. Written against the file as it is on disk today (fold clamp and exp guard present):

```cpp
struct CoreParams {
    double gainDb = 0.0, a2 = 0.0, a3 = 0.0;
    double flHz = 1.7, satDb = 2.0, q = 9.0, asym = 0.0;   // asym: knee-hardness asymmetry at the onset of saturation
    double asymP = 1.8;                     // its decay with the held peak flux: asym / max(peak, kEnvMin)^asymP     [NEW]
    double pulse = 0.0;                     // saturation pulse, relative to the knee level (even-symmetric, DC-blocked) [NEW]
    ...
    bool operator!=(const CoreParams& o) const
    {
        return gainDb != o.gainDb || a2 != o.a2 || a3 != o.a3 || flHz != o.flHz || satDb != o.satDb || q != o.q || asym != o.asym ||
               asymP != o.asymP || pulse != o.pulse ||                                                             // [NEW]
               ...
    }
};

class TransformerCore {
public:
    static constexpr double kCoupleTau = 0.05;   // s: the driver's even term and the pulse are AC-coupled through this mean
    static constexpr double kHoldTau = 0.3;      // s: release of the held peak flux (not identified by the grid; see docs)
    static constexpr double kEnvMin = 0.3;       // knee-flux units: the decay's floor (the core is linear below it)

    void configure(const CoreParams& p, double fs, uint64_t seed)
    {
        P = p; fsr = fs; T = 1.0 / fs;
        g = dbToLin(p.gainDb);
        r = kTwoPi * p.flHz;
        vk = dbToLin(p.satDb);                                 // [NEW] the 20 Hz knee level
        phik = vk / (kTwoPi * 20.0);
        kCouple = onePoleK(kCoupleTau, fs);                     // [NEW]
        kHold = onePoleK(kHoldTau, fs);                         // [NEW]
        ... (the qp / qn members go away: the hardness is computed per sample)
        reset();
    }
    void seedFrom(const TransformerCore& o)
    {
        phi = o.phi; hs = o.hs; hs2 = o.hs2; lp = o.lp; lift = o.lift; noiseLp = o.noiseLp; geLp = o.geLp;
        env = o.env; tDie = o.tDie; heat = o.heat; pw = o.pw; caStage = o.caStage;
        peak = o.peak; m2 = o.m2; mP = o.mP;                    // [NEW] the held peak and the two coupling means carry over
        sPrev = P.linear ? phi : sat(phi, std::fabs(phi) / phik, asymNow());
    }
    void reset()
    {
        phi = 0.0; sPrev = 0.0; peak = 0.0; m2 = 0.0; mP = 0.0;   // [NEW states zeroed]
        ...
    }

    inline double process(double x)
    {
        double u = g * x;
        if (P.geClassA) u = classA(u);
        else {
            const double uf = P.a3 < 0.0 ? 1.0 / std::sqrt(-3.0 * P.a3) : 1e30;
            const double uc = u > uf ? uf : (u < -uf ? -uf : u);
            const double u2 = uc * uc;
            m2 += (u2 - m2) * kCouple;                          // [NEW] the even term is AC-coupled: its mean does not magnetise the core
            u = uc + P.a2 * (u2 - m2) + P.a3 * u2 * uc;         // [changed]
            if (P.classA) u = caStage.tick(u);
        }
        double y = P.linear ? linearCore(u) : core(u);
        ... unchanged ...
    }

private:
    // the asymmetry seen by the knee: the onset value divided by a power of the peak flux the core has recently been driven to
    inline double asymNow() const { return P.asym / std::exp(P.asymP * std::log(peak > kEnvMin ? peak : kEnvMin)); }

    // saturating flux: S(phi) = phi / (1 + |phi / phi_k|^q)^(1/q), a hard ceiling at +-phi_k; the polarity's hardness is q (1 +- ae)
    inline double sat(double ph, double az, double ae) const
    {
        if (az <= 0.02) return ph;
        double q = ph > 0.0 ? P.q * (1.0 + ae) : P.q * (1.0 - ae);
        if (q < 1.0) q = 1.0;
        const double lz = q * std::log(az);
        if (lz > 700.0) return ph > 0.0 ? phik : -phik;
        return ph / std::exp(std::log1p(std::exp(lz)) / q);
    }

    inline double core(double v)
    {
        phi += T * (v - r * phi);
        const double az = std::fabs(phi) / phik;
        if (az > peak) peak = az; else peak += (az - peak) * kHold;        // [NEW] held peak flux, 300 ms release
        const double s = sat(phi, az, asymNow());
        double y = (s - sPrev) * fsr;
        sPrev = s;
        // [NEW] the saturation pulse: an even-symmetric term of fixed voltage size at each excursion into saturation, DC-blocked
        double Pz = 0.0;
        if (az > 0.02) {
            const double lz = P.q * std::log(az);
            Pz = lz > 700.0 ? 1.0 : 1.0 / (1.0 + std::exp(-lz));
        }
        mP += (Pz - mP) * kCouple;
        y += P.pulse * vk * (Pz - mP);
        return y;
    }

    CoreParams P;
    double fsr = 48000.0, T = 1.0 / 48000.0, g = 1.0, r = 0.0, phik = 0.01, vk = 1.0, kCouple = 1.0, kHold = 1.0;
    double phi = 0.0, sPrev = 0.0, peak = 0.0, m2 = 0.0, mP = 0.0;
    ...
};

inline CoreParams hwCore(const double* c, int k)
{
    CoreParams p;
    p.gainDb = c[kc_x_gain_db + k]; p.a2 = c[kc_x_a2 + k]; p.a3 = c[kc_x_a3 + k];
    p.flHz = c[kc_x_fl_hz + k]; p.satDb = c[kc_x_sat_db + k]; p.q = c[kc_x_q + k]; p.asym = c[kc_x_asym + k];
    p.asymP = c[kc_x_asym_p + k]; p.pulse = c[kc_x_pulse + k];                                        // [NEW]
    p.hsHz = c[kc_x_hs_hz + k]; p.hsDb = c[kc_x_hs_db + k]; p.lpHz = c[kc_x_lp_hz + k];
    return p;
}
```

`1 / (1 + exp(-lz))` is `az^q / (1 + az^q)` without overflow for negative `lz`. The material positions (GOLD, URANIUM, PLUTONIUM)
take the new fields through `hwCore()` unchanged; the linear cores (GOLD-PURE, GERMANIUM) do not run `core()` and are unaffected.
The header comment's sentence "the measured even harmonics peak at the onset of saturation and fade as the drive rises, which a
knee asymmetry does and a flux offset does not" should become: "the knee asymmetry fades with the peak flux the core has been driven
to and a fixed-size pulse marks each excursion into saturation (docs/xfmr-even-fix.md)".

### 3.2 `src/dsp/Calibration.hpp`

Two array fields after `x_asym` in `HVMC_CAL_FIELDS` (the layout hash changes; `FittedConstants.hpp` must be regenerated):

```cpp
    A(x_asym, kHwCores)         /* knee-hardness asymmetry at the onset of saturation (decays with the held peak flux, x_asym_p) */  \
    A(x_asym_p, kHwCores)       /* its decay: asym_eff = x_asym / max(peak, 0.3)^x_asym_p, peak = held |flux| / knee flux */        \
    A(x_pulse, kHwCores)        /* saturation pulse: even-symmetric output term x_pulse * 10^(x_sat_db/20) * (P - <P>), P the knee indicator */ \
```

Priors in `calPriors()` (the fitted values, as the existing `x_a2` / `x_a3` priors are; the signs are the fit's):

```cpp
    const double asym[kHwCores] = { -0.012, 0.014, -0.012 }, asymP[kHwCores] = { 1.65, 1.99, 1.76 };
    const double pulse[kHwCores] = { 1.5e-3, -1.6e-3, 1.1e-3 };
    for (int c = 0; c < kHwCores; ++c) {
        ...
        v[kc_x_asym + c] = asym[c]; v[kc_x_asym_p + c] = asymP[c]; v[kc_x_pulse + c] = pulse[c];
    }
```

(`x_asym`'s prior is 0.02 for every core today; with the decay law its fitted magnitude is 0.012-0.014 and its sign matters
relative to `x_a2`.) `docs/MODEL.md` section 4 gets two rows (`x_asym_p`: "decay of the knee asymmetry with the held peak flux",
unit 1, stage 2 nonlinear pass; `x_pulse`: "saturation pulse relative to the knee level", unit 1, stage 2 nonlinear pass) and the
`x_asym` row reads "knee-hardness asymmetry at the onset of saturation".

### 3.3 `fit/stages/stage2_transformers.py`

Items: the +24 dBFS row is the reference's output ceiling (frequency selective: at +24 it clips at 160 and 320 Hz and is clean at
1 and 5 kHz) and is not modelled; the test suite pools it apart and every harness excluded it, so the stage should too:

```python
grid = [i for i in ITEMS if i.startswith(f"xf_{core}_f") and int(i.rsplit("_", 1)[1]) < 24]
```

Nonlinear pass:

```python
NL = ["x_a2", "x_a3", "x_sat_db", "x_q", "x_asym", "x_asym_p", "x_pulse"]
lo = [1e-8, -5e-2, -6.0, 1.5, -0.5, 0.0, -0.02]
hi = [5e-3,  5e-2, 12.0, 30.0, 0.5, 6.0,  0.02]
x_scale = [1e-5, 1e-4, 0.5, 1.0, 0.01, 0.2, 0.01]
```

`x_a2` is held positive (the sign symmetry above). Starts: the calibration's values (the priors on a fresh run); a second start
with `x_pulse` negated and the lowest cost kept. The pulse's basin is not found from every start: on Iron the M1p run reached only
cost 911.6 (`x_pulse` 0.0059) from +-0.01 where the M1pc run reached 769.4 (`x_pulse` -0.0016) from +0.01, and its other two starts
(-0.01, +0.001) stopped at 812-813; on Nickel and Steel the three starts land within 7 cost units of each other. Order: the linear pass on `xf_resp_{core}` first
(unchanged), then the nonlinear pass; `max_nfev` 60 per start, `diff_step` 1e-3, soft L1 with `f_scale` 2 as now. Expected result
per core on the <= +21 grid (this document's figures, section 2.4): even 5.7 / 5.7 / 6.6 dB rms, odd 2.0 / 5.0 / 2.0, gain within
0.05 dB rms; the stage-2 weighted rms it prints should land near the synthesis costs (588 / 769 / 584 with the same residual).
The docstring's "flux offset" becomes "knee asymmetry at the onset, its decay with the held peak flux, and the saturation pulse".

### 3.4 Tests and docs

`tests/pb_reference.py` `TOL["xfmr"]["even"]` from `(17.0, 41.0)` to `(9.0, 40.0)` (1.5x the expected 6.0 rms; the max stays
near 30 because of the 120/160 Hz +21 points, section 4). Odd and gain bands unchanged. The comment "The transformer's even
harmonics are a known gap of stage 2" becomes "The transformer's even harmonics are fitted to about 6 dB rms (docs/xfmr-even-fix.md);
what remains is the onset burst's last 4 dB and the 120/160 Hz +21 dBFS points at the foot of the +24 dBFS ceiling."

`docs/MODEL.md` 3.3: replace "The two polarities have slightly different hardness, `q (1 +/- x_asym)`: the reference's even harmonics
peak at the onset of saturation and fade as the drive rises, which a knee asymmetry does and a flux offset does not." with the
three mechanisms of section 2.1 and the law of section 1.1 (the burst, its flat spectrum, the floor's decay with drive and its
5-8 dB per octave fall with frequency), and add the equations of section 2.2 to the code block.

### 3.5 Verifying the change

1. `python3 fit/run_all.py --only 2` (stage 2 with the new fields, then the header).
2. `python3 -u fit/tools/candidates/xfmr-even-synthesis.py --validate-only`: the mirror against the rebuilt engine must be within
   0.05 dB on the grid with the new constants. The mirror's `run_one` with `gate = 2, a2_after = 2` and `ke != 0` is the adopted
   form; if the engine differs by more, the C++ transcription is wrong, not the fit.
3. `python3 tests/pb_reference.py --groups xfmr,xfmr_resp,xfmr_ceiling`: even rms under 9, odd under 4.5, `xfmr_resp` unchanged
   (the linear path is untouched: `resp` rms 0.046 / 0.032 / 0.019 dB as now, `xfmr-even-remanence.result.*.V0.json`).

## 4. What remains unexplained

With the adopted model (section 2.4) the even residual is 6.0 dB rms pooled, 15-30 max. Where it sits:

1. **The onset burst is 4 to 5 dB weak on every core at every frequency** (D = 3 column: Nickel -4.1 / -3.8 / -4.0 / -4.2 / -5.2 at
   20 / 30 / 40 / 60 / 80 Hz; Iron -4.1 / -3.4 / -3.5 / -3.5 / -4.6; Steel -4.8 / -4.6 / -4.8 / -5.0 / -6.0), and its H4 is 4-5 dB
   weak too. The fit trades the burst against the floor: a larger `x_asym` makes the burst but overshoots the D = 6 .. 9 points. The
   uniformity of the deficit says the burst's shape (its rise between D = 0 and 3) is not this law's: in the reference H2 climbs
   18 dB between D = 0 and D = 3 and 1 dB more to D = 6; the model's climb is slower.
2. **The 120 Hz and 160 Hz +21 dBFS points are 10-12 dB low** (Nickel -11.7 / -11.1, Iron -11.3 / -9.8, Steel -12.5 / -12.0, the
   worst even points of every candidate in every key). At 120 Hz +21 (D = 5.4) the reference's burst is -47 dBc, 13 dB stronger
   than at the same D at 20-80 Hz; at 160 Hz +21 (D = 3) -54, 6 dB stronger. These sit at the foot of the +24 dBFS anomaly (at +24 the
   reference clips at 160 Hz, H2 -29 dBc, and 320 Hz, H2 -12 with H2 above H3 and 2-3 dB of gain loss, but is clean at 1 kHz and
   5 kHz), a frequency-selective ceiling between about 100 and 400 Hz that is neither the core's 6 dB per octave law nor an output
   clipper, and is not modelled. It contributes the even max (25-30) and the odd max at 160 Hz.
3. **The reference's nulls.** Nickel 60 Hz +18 (H2 -79.5 against the 1 kHz line's -70: the driver's and the core's even terms cancel
   there) the model puts at -69.7 (+9.8); Steel 60 Hz +18 +5.3; Iron 80 Hz +21 H4 +12.2; Nickel 20 Hz +9 H4 +11.2 (the reference's
   H4 notch at D = 9, -74.8 between -60.8 and -67.0). The model has the same interference (its H2 phase flips where the driver's
   term takes over) but not at the same points: the relative phase of the three even sources is 20-30 degrees off.
4. **The floors are within +-3 dB but with structure**: the 20 Hz floor is 1-4 dB high on Iron at +9 .. +12 and 3-4 dB low at +21
   on all cores (the model's decay with drive is slightly steeper than -0.5 dB/dB); the 40 Hz floor is 2-3 dB high at +12 .. +18 on
   Nickel and Iron.
5. **Not identified by the data.** The hold's release (300 ms): the remanence key's free fits put it at 163 ms (Steel), 297 ms
   (Iron) and 2 s (Nickel, the render length), because a steady sine cannot tell them apart. It sets how long after a loud passage
   the onset burst stays suppressed on quieter material and is the one audible consequence of this model the reference data
   neither confirm nor refute. The measurement that would fix it: 20 or 40 Hz at +15 dBFS for 1 s, then a step to the onset level
   (+3 / +9 dBFS), H2 tracked per cycle over the next second (the research note's test 5). The coupling mean's 50 ms is likewise a
   choice; the data say only that the driver's DC must not reach the flux at the 2 s scale.
6. **Iron's opposite signs** (`asym > 0, pulse < 0` against Nickel's and Steel's `asym < 0, pulse > 0`) are what the fit needs; whether
   they are the Class-A stage's inversion or a different mechanism cannot be told from magnitudes alone (the capture has no phase).
7. **Iron's odd max** (49.4 dB at `xf_Iron_f1000_21` H5: reference -71 dBc, model none) is the cubic driver and is not this hole; the
   driver key's side finding stands: a symmetric soft ceiling at about +36 dBFS in Iron's driver (classA / caenv) takes its odd rms
   from 4.95 to 2.28 and the max to 18.5 without touching the even order (`build_xfmr_even_driver_Iron.log`).

## 5. Is the hole solved?

Partly. The even order goes from 13.0 to 6.0 dB rms pooled (Nickel 11.8 -> 5.7, Iron 13.8 -> 5.7, Steel 13.2 -> 6.6), max 34 ->
30, at unchanged odd order (3.5 -> 3.4) and gain, with the linear response untouched, the hard knee and the flux law kept, no
clipper, and three constants that come out the same on all three cores. The task's example point (Iron 60 Hz +18 dBFS) goes from
H2 -47 dBc against -66.5 to +2.0 dB off. That is worth adopting now. What it does not reach is the burst's last 4 dB, the
120/160 Hz +21 points and the exact position of the nulls, which together hold the residual at 6 dB rms and 15-30 dB max.

Directions worth a further round, in order:

1. **The burst's rise.** The reference's H2 climbs 18 dB between D = 0 and 3 and then stops; the adopted law's climb is slower and
   the fit gives away 4-5 dB uniformly. A knee indicator harder than `q` for the pulse (its own hardness, or `P` raised to a power),
   or an asymmetry that switches on with the excursion rather than scaling with the peak, is the cheapest test: one parameter on the
   synthesis mirror, and the D = 0 / 3 / 6 columns say yes or no in one run.
2. **The frequency-selective ceiling between 100 and 400 Hz.** The +24 dBFS row (H2 -29 at 160 Hz, -12 at 320 Hz, clean at 1 and
   5 kHz) and its foot at 120/160 Hz +21 are one feature; a resonant element ahead of the core (the research note's series coupling
   capacitor against the primary inductance, an under-damped second-order high pass) would raise the flux drive in that band and
   produce both. Fitting it needs the +24 row back in the grid with a ceiling model, and it would also settle the "1 dB bump at
   40 / 100 Hz" claims of the manuals.
3. **The nulls' phase.** The relative phase of the driver's term, the knee asymmetry and the pulse fixes where the even order
   cancels; the model's cancellations are 20-30 degrees off. A capture with harmonic phase (the protocol's lock-in has it; the
   reference features store magnitude only) would turn the sign ambiguity of section 2.3 into a measurement and pin the pulse's
   placement (before the differentiator as a flux-domain feature, or after it as a voltage one; the synthesis tested only the latter).
4. **The hold time**, by the step measurement of section 4 item 5, before anyone tunes it by ear.

Not worth another round: a flux offset or remanence in any form, a ceiling asymmetry, the loss term, a level-only driver mechanism,
an even term as a function of the output (all refuted above with the figures), and the memoryless gate on its own (D2r), which is
1.4 to 3.5 dB rms behind the adopted form per core.
