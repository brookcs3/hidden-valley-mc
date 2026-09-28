# Opto fix: the optical stage's law, cell and sidechain

Status: design document, 2026-09-27. Synthesises the four optical-stage hypothesis harnesses under `fit/tools/candidates/opto-*.py`,
their adversarial verifications, and one synthesis run of my own (`fit/tools/candidates/opto-synthesis.py`, 17 s, log
`/tmp/opto_synthesis.log`) that puts the chosen pieces together and scores them on everything this document states. Nothing under
`src/` or `fit/stages/` has been changed; this document says exactly what to change.

The problem: `OptoStage::process` as fitted by stage 4 (`build_stage4.log`) has the right topology (a feedback divider whose
sidechain is the divider output) but the wrong degrees of freedom: the fit pinned the panel turn-on `o_vth` at its 0.5 bound and
the persistence at its lower bound, and the three linear conductance states cannot both hold the multi-hundred-millisecond tail
after a burst that ends below the knee and release in 13 ms after a step that stays above it. Result: a soft progressive knee
where the reference has a hard one (0.93 dB rms / 4.3 max at threshold 20), H5 15-20 dB too high, 0.1-0.9 dB release tails
the reference does not have, and no account of the no-GR gain falling with the threshold position.

Level convention throughout: dBFS peak of a sine, 1.0 = 0 dBFS = +14 dBu; "gain" includes the make-up at position 12 (+0.561 dB
with the Nickel path at threshold 1). "Static" residuals are model minus reference gain in dB over the protocol's 33 levels
(-50..+14 dBFS, 1 kHz); "bursts" is the stage-4 weighted rms over the nine `opto_burst_*` / `opto_blen_*` / `opto_pulses`
envelopes; (A), (B), (C), (D), (E) are the discriminating captures in `fit/data/discriminate_opto.json`.

## 1. The evidence

| hypothesis (harness) | what it is | static rms / max | bursts | H3 / H5 | (A) fine knee | (D) no-GR bias | (E) above-knee steps | verdict (author / verifier) |
|---|---|---|---|---|---|---|---|---|
| stage 4 as built (C++ engine, `constants.json`) | feedback, soft law `(d - 0.5)^1.40`, `L^1.42`, 3 linear states | 0.342 / 1.20 over t6/10/14/20/24 (0.371 / 0.78 at t20) | 0.035 | t18/-10: H3 -44.5 (ref -48.0), H5 -53.6 (ref -72.4) | 0.5 dB crossing 2.5 dB early, width 6.4 vs 4.25 dB | none (0.000) | tails 0.10 / 0.08 / 0.92 dB at 100 ms, t90 24 / 15 / 68 ms (ref 17 / 7 / 13) | baseline |
| `linear-db-feedback` (stage D, cond-driven) | feedback, peak detector 0.5 ms, `GR* = k max(0, S_dB + D)`, `c* = 10^(GR*/20) - 1`, 3 states with rates x `(1 + cond)^2.23` | 0.089 / 0.46 over 7 positions (make-up + trim per position read from the data; 0.117 on compressing points) | 0.006 | 10 items: H3 rms 3.0, H5 rms 3.3 | 0.04 rms, crossings within 0.06 dB | not produced by the law; a trim of -1.45e-4 A^2 dB added by hand | reproduced with alpha (1/5, 0/1, 1/12 ms vs 1/4, 2/7, 3/13), tails < 0.03 | supported / confirmed; 100 Hz statics degrade to 0.60 rms; alpha and the raw constants degenerate |
| `feedforward-fixed-ratio` (best P; the hypothesis as stated is A) | feed-forward computer on the input, fixed ratio, quadratic knee, smoothed target | P 0.110 / 0.40 (compressing 0.151); A 0.200 / 0.70 | P 0.035 plain (0.004 w); A 0.173 | P: H3 -45.5 / H5 -76.3 at t18/-10 | P 0.08 | absorbed in the per-position make-up | fails: target-dependent smoothers miss the release identity by 0.6-0.8 dB; the two that pass (K, P) do so by construction and part at cycle 8-14 instead of 5 | refuted / confirmed (feedback J reproduces the identity untuned: 0.014 dB, parting -0.44 at cycle 9) |
| `physical-law-hard-turnon` (V6) | feedback, hard turn-on `(A|v| - vth)^p` with `n = 1`, `gamma = p = 1.657`, light leak `cond0 ~ A^2.15`, 3 states whose release rate is x `(1 + 7.69 s_i)` | 0.068 / 0.31 over t6/10/14/20/24 (0.083 / 0.31 at t20) | 0.009 | 3 fitted items: H3 err 0.0 / +1.6 / -0.9, H5 err -1.2 / +1.3 / -0.8 | 0.033 rms, width 4.27 vs 4.25, 0.5 dB crossing -26.01 vs -26.01 | reproduced: 0.173 / 0.291 / 0.477 vs 0.180 / 0.284 / 0.445 at t20 / 22 / 24 | 5/16, 2/6, 4/16 ms vs 4/17, 2/7, 3/13; tails 0.000 | supported / confirmed; 15 unfitted positions 0.067 / 0.36; unfitted H7 6-7 dB off |
| `sidechain-shape-and-taper` (analysis, no loop) | static solve of the reference features; sidechain low pass from the HF rows; leak law; drive table | law A family 0.065 / 0.375 over 24 x 33 (0.096 on compressing points) | not in scope | not in scope | law A 0.026 / 0.028 rms on the two sweeps | `c0 = 0.0209 A^2.08`, 0.001 rms over 24 positions; -70 = -50 dBFS to 0.00000 dB | measured: 90 % release 7-17 ms, residual 0.0000 at 0.1-2 s | supported / weakened: the envelope-vs-instantaneous threshold claim was an unrefitted comparison (refit: 0.035 vs 0.024 rms, a mild preference only); the cpp recipe mixed a gain-factor drive table with an in-loop leak |
| synthesis (this document's run) | V6 shape, one knee per position, in-loop leak, matched second-order sidechain low pass 5147 Hz Q 0.718 | 0.065 / 0.36 over 24 x 33 (0.095 on the 362 compressing points; 15 unfitted positions 0.067 / 0.36) | 0.010 (plain 0.078 / 1.00) | 1 kHz, 9 items: H3 rms 1.2, H5 rms 9.3; 100 Hz: 2.9 / 7.1; 4 kHz: 8.7 / 12.9 (1.2 / 9.6 with the persistence pole removed) | t20 0.023 rms, t10 0.029 | max 0.025 dB over 8 positions at -70 and -50 | 5/16, 2/6, 4/15 ms; tails 0.000 | adopted |

Two hypotheses survive their verification. Between them the physical law wins on every axis the task asked for: it fits the
statics, the bursts and the (fitted) harmonics at once, it produces the no-GR bias from inside the loop instead of as a trim, it
keeps the 100 Hz statics (0.11 rms against 0.60), its dynamics parameters are identified (the linear-dB fit's `alpha` is degenerate
with its raw time constants and hits bounds in two of three variants), and it changes the least in `Opto.hpp`. The feed-forward
topology is out: the release from -10 dBFS toward -20 and toward -50 dBFS coincide for five cycles and part exactly when the
divider output crosses the knee, which only a sidechain fed from the divider output can do. The sidechain analysis contributes
three things the loop harnesses could not: the second-order low pass, the no-GR law over all 24 positions, and the demonstration
that the threshold is a pure drive shift (slope multiplier 0.998 +- 0.003 across positions 3..24).

## 2. The chosen model

### 2.1 In words

Keep the feedback divider, the tap before the make-up, the 88.5 Hz first-order sidechain high pass, the amplifier terms and the
three conductance states. Change four things:

1. **The panel has a hard turn-on.** The light is `(A |s| - vth)` above `vth = 3.06` drive units and zero below, with the exponent
   on the cell (`n = 1`, `gamma = 1.657`). The static slope above the knee is `p / (1 + p) = 0.624` dB per dB of gain
   (ratio 2.66:1, output slope 0.376), and the knee is as hard as the fine sweep shows (0.5 -> 3 dB of GR in 4.28 dB, measured 4.25).
   Stage 4's bound of 0.5 on `o_vth` was the blocker.
2. **A light leak proportional to a power of the sidechain drive.** The idle conductance is `cond0 = 0.0202 (A / A20)^2.154`,
   entered as light before the cell law. It reproduces the threshold-dependent no-GR gain (D) inside the loop, to 0.025 dB over
   the 24 positions at -70 and -50 dBFS.
3. **The release is self-quenched.** Each state's release rate is `(1 + 7.69 s_i) / trel_i`, i.e. `dc/dt = -(c + 7.69 c^2) / trel`:
   bimolecular recombination in the cell. The reference's release trajectories read as `d(1/c)/dt` about constant (80-83 /s
   measured on the envelopes by the feed-forward harness, m = 1.94-2.00 at every burst depth). This is what lets one set of
   constants release in 6-16 ms from 6-16 dB of GR above the knee and leave a 0.9 / 0.26 / 0.01 dB tail at 0.1 / 0.3 / 1 s after a
   burst that ends below it.
4. **A second-order low pass in the sidechain,** 5147 Hz, Q 0.718, magnitude-matched to the analog curve (not a bilinear biquad,
   section 5). It accounts for the 8 kHz static series (4.18 rms -> 0.53) and the HF rows (6.1 / 6.9 rms -> 0.26 / 0.13).

### 2.2 Equations, in the order the C++ runs them

Per sample; `x` the stage input, `thr` the OPTICAL THRESHOLD position (0..23), `gain` the OPTICAL GAIN position:

```
xa   = x + b2 x^2 + b3 x^3                                   stage amplifier                     [unchanged]
g    = 1 / (1 + cond)                                        divider, cond from the previous sample
v    = g xa (+ noise)                                        divider output; the audio output before make-up
hp   = HPF1(v, sc_hz)                                        first-order high pass, 88.5 Hz       [unchanged]
s    = scFilter ? hp : v
s2   = LP2(s, o_sc_lp_hz, o_sc_lp_q)                         matched second-order low pass        [NEW]
A    = 10^(o_thr_db[thr] / 20)                               sidechain drive (smoothed switch)    [table revalued]
d    = | A s2 |
e    = d - o_vth                                             hard turn-on, o_vth = 3.058          [prior/bound changed]
L*   = e > 0 ? e^o_n : 0                                     o_n = 1.0
L    += (L* - L) kEl                                         persistence, o_tau_el = 6.6e-5 s (kEl = 1 - exp(-1/(o_tau_el fs)))
L0   = cond0^(1 / o_gamma), cond0 = o_leak (A / A20)^o_leak_q   light leak, A20 = 10^(o_thr_db[19]/20)   [NEW; per configure()]
c*   = (L + L0)^o_gamma                                      cell law, o_gamma = 1.657
for i in 0..2:
    if c* > st_i:  st_i += (c* - st_i) * kAtt_i              kAtt_i = 1 - exp(-1 / (o_tatt[i] fs))
    else:          st_i += (c* - st_i) * (1 - exp(-(1 + o_rel_mu st_i) / (o_trel[i] fs)))     [NEW: self-quenched release]
cond = sum_i o_w[i] st_i
out  = v * 10^(o_gain_db[gain] / 20)                          make-up                             [unchanged]
grDb = 20 log10((1 + cond) / (1 + cond0))                     meter: zero at idle                 [changed]
```

Continuous-time reading of the release: `d st_i / dt = -(st_i + o_rel_mu st_i^2) / o_trel[i]`, so in the bimolecular regime
(`st_i >> 1 / o_rel_mu = 0.13`) `d(1/st_i)/dt = o_rel_mu / o_trel[i]` = 44 / 58 / 82 /s for the three states, and for small
conductance a plain exponential with 175 / 132 / 94 ms, which is the sub-second tail after a burst. The attack is unchanged
(a linear one-pole per state); the loop's overdrive is what makes the measured attack reach 90 % in 1-4 ms with 8-16 ms constants.

Static law (ripple ignored, `v^` the amplitude of the divider output): `cond = (A v^ - o_vth)^p + leak`, so the knee sits at
`v^ = o_vth / A`, i.e. at `20 log10(o_vth) - o_thr_db[thr]` = 9.71 - 37.28 = -27.57 dBFS at position 20 (GR onset measured
between -27.0 and -26.5 dBFS), and above it `x = v (1 + cond)` gives the 0.624 dB/dB gain slope.

### 2.3 Parameters

| symbol | meaning | unit | value (this run) | calibration field |
|---|---|---|---|---|
| `b2`, `b3` | stage amplifier even / odd term | 1, 1 | 2.54e-5, -5.33e-4 | `o_b2`, `o_b3` (unchanged) |
| `sc_hz` | sidechain high-pass corner | Hz | 88.54 | `sc_hz` (unchanged, shared with the discrete stage) |
| `fc_lp`, `Q_lp` | sidechain low pass corner, Q | Hz, 1 | 5147, 0.718 | `o_sc_lp_hz`, `o_sc_lp_q` (new) |
| `A_thr` | sidechain drive per threshold position | dB | table, section 4 (37.28 at position 20) | `o_thr_db[24]` (existing, revalued) |
| `vth` | panel turn-on | drive units | 3.058 | `o_vth` (existing; prior 3.06, bound 0.5..20) |
| `n` | panel light exponent | 1 | 1.0 (fixed) | `o_n` (existing; prior 1.0, held) |
| `gamma` (= p) | cell law exponent | 1 | 1.657 | `o_gamma` (existing; prior 1.66) |
| `tau_el` | phosphor persistence | s | 6.65e-5 (a 2.4 kHz pole; see section 6.2, expect the refit to send it toward 1e-5) | `o_tau_el` (existing; prior 6.6e-5, lower bound 1e-5) |
| `w_i` | state weights (sum 1) | 1 | 0.223, 0.590, 0.187 | `o_w[3]` (existing) |
| `tatt_i` | state attack constants | s | 8.44e-3, 15.73e-3, 8.54e-3 | `o_tatt[3]` (existing) |
| `trel_i` | state release constants at zero conductance | s | 0.1751, 0.1320, 0.0939 | `o_trel[3]` (existing) |
| `mu` | release-rate multiplier per unit conductance | 1 | 7.69 | `o_rel_mu` (new) |
| `b_leak` | idle conductance at position 20 | 1 | 0.0202 (the direct fit of the no-GR rows gives 0.0209 with q 2.08; equivalent, section 4) | `o_leak` (new) |
| `q_leak` | growth of the leak with the drive, `(A/A20)^q` | 1 | 2.154 | `o_leak_q` (new) |
| `G_i` | make-up per gain position | dB | unchanged | `o_gain_db[24]` (unchanged) |

Raw V6 vector (log10 seconds for the time constants), for reproducing the harness: `logC 0.804515, p 1.657263,
log_tau_el -4.177286, w 0.29804 / 0.787815 / 0.249402 (normalised in the model), la -2.073795 / -1.803165 / -2.068737,
lr -0.756595 / -0.879338 / -1.027291, logb -1.694579, q 2.154234, logmuc 0.885802`; the harness's `C = 10^logC = 6.376` maps to
`o_vth = C^(1/p)` and its knee levels to `o_thr_db = 20 log10(o_vth) - knee_dBFS`.

## 3. The changes

### 3.1 `src/dsp/Opto.hpp`

`prepare()` gains the low-pass coefficients and `configure()` the leak; `process()` changes in three places (the low pass, the
leak before the cell law, the release rate). Everything else (the transformer-like amplifier terms, the HPF running always, the
hwUnit low passes, the memory and half-life options, the drive ceiling, the switch smoothers) stays. Pseudo-code close enough to
transcribe:

```cpp
// members added
Biquad2 scLp;                      // second-order low pass, DF2T: b0 b1 b2 a1 a2, state z1 z2
double kRelBase[kOptoStates] = {}; // 1 / (o_trel[i] * fs), so the per-sample release rate is 1 - exp(-(1 + mu st) * kRelBase)
double leakL = 0.0;                // L0 of the current threshold position (light units)
Smoother leakS;                    // the same 15 ms smoothing as thrS, so switching the threshold does not step the idle gain

void prepare(double fs, const double* cal)
{
    ... as now ...
    for (int s = 0; s < kOptoStates; ++s) { kAtt[s] = onePoleK(c[kc_o_tatt + s], fs); kRelBase[s] = 1.0 / (c[kc_o_trel + s] * fs); }
    scLp.setMatchedLowPass(c[kc_o_sc_lp_hz], c[kc_o_sc_lp_q], fs);     // section 5.2 gives the formulas
    leakS.set(0.015, fs);
    ...
}

void reset()
{
    ... as now, plus scLp.reset(); leakS.reset(leakLightFor(cfg.thr));
}

// the leak's light for a threshold position: cond0 = o_leak * (A / A20)^o_leak_q, entered before the cell law as L0 = cond0^(1/gamma)
double leakLightFor(int thr) const
{
    const double ratioDb = c[kc_o_thr_db + thr] - c[kc_o_thr_db + 19];
    const double cond0 = c[kc_o_leak] * std::pow(10.0, c[kc_o_leak_q] * ratioDb / 20.0);
    return cond0 > 0.0 ? std::pow(cond0, 1.0 / c[kc_o_gamma]) : 0.0;
}

inline double process(double x, double noise = 0.0)
{
    const double xa = x + c[kc_o_b2] * x * x + c[kc_o_b3] * x * x * x;
    const double g = 1.0 / (1.0 + cond);
    double v = g * xa + noise;
    if (cfg.hwUnit) { ... as now ... }
    gLast = g;
    const double hp = sc.tick(v);
    const double s = cfg.scFilter ? hp : v;
    const double s2 = scLp.tick(s);                                    // NEW: the sidechain low pass, after the high pass
    const double A = dbToLin(thrS.tick(c[kc_o_thr_db + cfg.thr]));
    double d = std::fabs(A * s2);
    if (d > cfg.driveCeiling) d = cfg.driveCeiling;
    const double e = d - c[kc_o_vth];                                  // hard turn-on: o_vth is now about 3, not 0.5
    const double Linst = e > 0.0 ? std::pow(e, c[kc_o_n]) * cfg.lightGain : 0.0;   // o_n = 1: pow is a no-op the fit may keep
    L += (Linst - L) * kEl;
    L2 += (L - L2) * kEl2;
    const double L0 = leakS.tick(leakLightFor(cfg.thr)); leakL = L0;   // NEW: the leak, smoothed with the switch
    const double target = std::pow(L2 + L0, c[kc_o_gamma]);            // NEW: leak enters as light; target > 0 at idle
    double fast = 0.0;
    for (int i = 0; i < kOptoStates; ++i) {
        const double k = target > st[i] ? kAtt[i]
                                        : 1.0 - std::exp(-(1.0 + c[kc_o_rel_mu] * st[i]) * kRelBase[i]);   // NEW: self-quenched release
        st[i] += (target - st[i]) * k;
        fast += c[kc_o_w + i] * st[i];
    }
    ... halfLife / memory / cond = fast, as now ...
    return v * dbToLin(gainS.tick(c[kc_o_gain_db + cfg.gain]));
}

// meter: the leak is idle conductance, the meter reads zero at idle
double grDb() const
{
    const double c0 = std::pow(leakL, c[kc_o_gamma]);                 // the current smoothed L0 raised back to a conductance
    return linToDb((1.0 + cond) / (1.0 + c0));
}
```

Notes for the transcription:

- `leakLightFor()` is two `pow`s; call it from `configure()` and on each `thrS` change rather than per sample if the profiler
  minds (the smoother then targets a cached value). The harness computes it once per render.
- The release rate uses `st[i]` before the update, as the harness does. The exact `exp` form is what was fitted; the
  first-order `min(1, kRel[i] * (1 + mu st[i]))` differs by 7 % at 10 dB of GR (`(1 + 7.69 * 2.16) / (0.094 * 48000)` = 0.0039 per
  sample, where `1 - exp(-0.0039)` = 0.00389) and is acceptable if the exp is unwanted, but then the fit must use the same form.
- `kEl2`, `lightGain`, `driveCeiling`, the memory and half-life branches are untouched. The light-memory option remains
  contradicted by the data (identical tails after 0.05 s and 4 s bursts), which `docs/MODEL.md` already says.
- The comment header: replace "light L = |d|^n" and "static slope p/(1+p) from a power law with a soft knee" by: the panel
  emits only above `o_vth`, the static curve is a hard knee at `20 log10(o_vth) - o_thr_db` dBFS with a fixed slope
  `gamma / (1 + gamma)` above it; the release is bimolecular (`dc/dt ~ -(c + mu c^2)`); the idle light leak sets the no-GR gain
  per threshold position; the sidechain low pass is what the 8 kHz statics measure.
- The persistence pole stays a field. At 6.6e-5 s it is a 2.4 kHz pole on the light (kEl = 0.27 at 48 kHz), which shapes the
  ripple's frequency dependence (section 6.2); the refit is expected to move it.

### 3.2 `src/dsp/Calibration.hpp`

In `HVMC_CAL_FIELDS`, optical block:

```cpp
    S(sc_hz, 90.0)              /* unchanged */
    A(o_thr_db, kSteps24)       /* sidechain drive gain per OPTICAL THRESHOLD position, dB; the knee is at 20log10(o_vth) - this */
    A(o_gain_db, kSteps24)      /* unchanged */
    S(o_n, 1.0)                 /* EL panel light law exponent above the turn-on: light ~ (drive - o_vth)^n  (was 2.0; held at 1) */
    S(o_gamma, 1.66)            /* CdS conductance law exponent: conductance ~ light^gamma; static slope gamma/(1+gamma)  (was 0.85) */
    S(o_vth, 3.06)              /* EL turn-on, in drive units: a hard knee  (was 0.0; the fit bound was 0.5) */
    S(o_tau_el, 6.6e-5)         /* EL phosphor persistence  (was 5e-4) */
    A(o_w, kOptoStates)         /* unchanged meaning */
    A(o_tatt, kOptoStates)      /* unchanged meaning */
    A(o_trel, kOptoStates)      /* release time constant of each state at zero conductance */
    S(o_rel_mu, 7.7)            /* release rate multiplier per unit conductance: rate = (1 + o_rel_mu * s) / o_trel (bimolecular recombination) */
    S(o_leak, 0.020)            /* idle light leak, as a conductance at threshold position 20 (sets the no-GR gain per position) */
    S(o_leak_q, 2.15)           /* its growth with the sidechain drive: cond0 = o_leak * (A / A20)^o_leak_q */
    S(o_sc_lp_hz, 5150.0)       /* sidechain second-order low pass corner (opto only, after the high pass), magnitude-matched design */
    S(o_sc_lp_q, 0.72)          /* its Q */
    S(o_b2, 0.0) ... S(o_hw_grloss_hz, 60000.0)   /* unchanged */
```

Nothing is removed: `o_n` stays (fitted at 1 and held there; a future fit may free it), `o_tau_el` stays, the `o_mem_*` option
fields stay. Five scalars are added, so `calLayoutHash()` changes and `FittedConstants.hpp` must be regenerated by the fit
before `kCalFitted` is true again.

In `calPriors()`, replace the `o_thr_db` line and the state priors:

```cpp
    static const double thr[kSteps24] = { -9.4, -1.43, 6.27, 11.37, 15.54, 17.99, 20.26, 22.41, 24.53, 26.62, 28.63, 29.57,
                                          30.56, 31.48, 32.46, 33.39, 34.37, 35.32, 36.30, 37.28, 38.29, 39.32, 40.40, 41.36 };
    for (int i = 0; i < kSteps24; ++i) v[kc_o_thr_db + i] = thr[i];
    const double w[kOptoStates] = { 0.223, 0.590, 0.187 }, ta[kOptoStates] = { 0.0084, 0.0157, 0.0085 }, tr[kOptoStates] = { 0.175, 0.132, 0.094 };
```

(position 1's -9.4 is 8 dB below position 2, section 4.1). The `o_gain_db` prior line is unchanged.

### 3.3 `fit/stages/stage4_opto.py`

Reorder and reparametrise. The stage keeps rendering the C++ engine through `render_item` (the source of truth); the numba mirror
in `fit/tools/candidates/opto-synthesis.py` (validated against the engine to 0.0007 dB on statics, 0.0056 dB on envelopes,
identical H3/H5) is 50-100x faster and may be used for the search with the engine for the final residuals. Whichever renders,
the stage must print a mirror-versus-engine check first if it uses the mirror.

**4a. Stage amplifier** (moved first: it does not depend on the light chain and the statics at +10..+14 dBFS do depend on it):
`o_b2`, `o_b3` on `opto_harm_nogr_{-30,-20,-10,0,6,12}`, H2..H4 of each, exactly as the current 4c.

**4b. Shape, dynamics and harmonics, jointly.** Fit in the harness's parametrisation, then write the fields:

```
parameters (22):  knee_k for k in (6, 10, 14, 18, 20, 22, 24)   dBFS      x0 from the table (section 4.1)   bounds (-45, 5)     x_scale 0.5
                  logC                                        log10     0.8045                            (-2, 3)             0.1
                  p                                           1         1.657                             (0.8, 3.0)          0.05
                  log_tau_el                                  log10 s   -4.18                             (-5.0, -1.5)        0.1
                  w0, w1, w2                                  1         0.298, 0.788, 0.249               (0.005, 1)          0.05   (normalised: w /= sum)
                  la0, la1, la2                               log10 s   -2.074, -1.803, -2.069            (-4, 0)             0.1
                  lr0, lr1, lr2                               log10 s   -0.757, -0.879, -1.027            (-3, 1)             0.1
                  logb                                        log10     -1.695                            (-4, -0.5)          0.1
                  q                                           1         2.154                             (0.5, 4)            0.1
                  logmuc                                      log10     0.886                             (-1, 3)             0.1
mapping:          o_n = 1; o_gamma = p; o_vth = C^(1/p); o_thr_db[k-1] = 20 log10(o_vth) - knee_k; o_tau_el = 10^log_tau_el;
                  o_w = w; o_tatt = 10^la; o_trel = 10^lr; o_rel_mu = 10^logmuc; o_leak = 10^logb; o_leak_q = q
                  (A20 in the leak is the position-20 drive of the same vector, so the leak and the knees are fitted consistently)
items:            statics opto_static_t{6,10,14,18,20,22,24}_{-50..14 step 2}         weight 2.0 per dB
                  bursts  opto_burst_{-26,-18,-10,-2}, opto_blen_{0.05,0.2,1.0,4.0}, opto_pulses   stage-4 weights
                          (1.0 where the reference is > 0.3 dB below its flat gain, else 0.3, divided by sqrt(n/100))
                  harmonics opto_harm_t{14,18,22}_{-20,-10,0}_f1000 (9), opto_harm_t18_-10_f100, opto_harm_t22_0_f100,
                          opto_harm_t18_-10_f4000, opto_harm_t22_0_f4000 (4): gain weight 2.0, H3 weight 1.0, H5 weight 0.3
optimiser:        least_squares(bounds, x_scale, diff_step 2e-3, max_nfev 60, loss "soft_l1", f_scale 1.0), two starts (the priors
                  above and the "physical" start of the harness: logC log10(3), p 1.63, w 0.05/0.70/0.25, ta 1.0/0.007/0.05 s,
                  tr 1.0/0.007/0.05 s), keep the lower cost; then one refine pass from it (diff_step 5e-4, linear loss, xtol 1e-10,
                  ftol 1e-6, max_nfev 120)
```

The 4 kHz items and the widened `log_tau_el` bound are the changes from the harness's objective: with V6's three-item harmonic
set the persistence pole was unconstrained and landed where it kills the 4 kHz ripple (section 6.2). H5's weight is 0.3, not 0.7,
because the model's H5 pattern across levels is structurally wrong (section 6.1) and must not be allowed to distort H3.

**4c. Threshold table.** With the shape fixed, one knee per position `k = 1..24` on `opto_static_t{k}_*` (33 levels, unweighted):
`minimize_scalar(cost, bounds=(-45, 40), method="bounded", xatol 1e-3)` on `knee_k`, `o_thr_db[k-1] = 20 log10(o_vth) - knee_k`.
Position 1 compresses nothing within the range (its 0.107 dB fall by +14 dBFS is the amplifier droop) so its cost is flat: write
`o_thr_db[0] = o_thr_db[1] - 8.0`. Position 2 has one compressing point (+14 dBFS, 1.28 dB) and is determined by it. Report the
per-position rms/max, the compressing-only rms, and the no-GR gain model - reference per position (the leak law's check, expected
within 0.03 dB).

**4d. Sidechain low pass.** `o_sc_lp_hz`, `o_sc_lp_q` on `opto_static_f3000_*` and `opto_static_f8000_*` (13 levels each, all
weight 1; the 8 kHz points above +4 dBFS may be given weight 0.3, section 6.5), `least_squares` from (5150, 0.72), bounds
(3000, 9000) x (0.5, 1.0), x_scale (500, 0.1), diff_step 1e-3. Report `opto_static_f100_*` as a check (expected 0.11 rms; the low
pass moves it by 0.003 dB). If `fit/data/discriminate_opto.json` is present, also report the HF rows at -40 / -10 / 0 dBFS
(expected 0.10 / 0.26 / 0.13 rms with the prior values).

**Report section** (not fitted): all 27 under-GR harmonic items H3/H5/H7 by frequency, `opto_burst_thr12`, `sr_opto_44100/96000`,
and, if the discriminate file is present, (A), (B), (D), (E) as `opto-synthesis.py` prints them.

`save_cal` fields: `o_thr_db, o_n, o_gamma, o_vth, o_tau_el, o_w, o_tatt, o_trel, o_rel_mu, o_leak, o_leak_q, o_sc_lp_hz,
o_sc_lp_q, o_b2, o_b3`.

## 4. The threshold law and the no-GR gain law

### 4.1 Threshold = sidechain drive, by table

The static curve of every position lays onto position 20's by a pure level shift (slope multiplier 0.998 +- 0.003, positions
3..24, `sidechain-shape-and-taper`). With the V6 shape and one free knee per position (my run; the seven fitted positions were
refitted the same way, and moved by at most 0.07 dB):

```
pos   1      2      3      4      5      6      7      8      9     10     11     12
knee  (+13.6)  11.14   3.43  -1.66  -5.83  -8.28 -10.56 -12.71 -14.82 -16.91 -18.92 -19.86   dBFS (divider output)
thr   -9.4  -1.43   6.27  11.37  15.54  17.99  20.26  22.41  24.53  26.62  28.63  29.57   o_thr_db, dB
rms   0.003  0.010  0.019  0.009  0.019  0.032  0.042  0.041  0.047  0.053  0.057  0.055   all 33 levels
max   0.01   0.05   0.09   0.04   0.08   0.12   0.15   0.14   0.14   0.16   0.18   0.22

pos  13     14     15     16     17     18     19     20     21     22     23     24
knee -20.85 -21.77 -22.75 -23.68 -24.66 -25.61 -26.59 -27.57 -28.58 -29.61 -30.69 -31.66
thr   30.56  31.48  32.46  33.39  34.37  35.32  36.30  37.28  38.29  39.32  40.40  41.36
rms   0.069  0.074  0.093  0.075  0.089  0.077  0.092  0.076  0.096  0.079  0.099  0.084
max   0.21   0.26   0.36   0.26   0.33   0.27   0.30   0.27   0.30   0.26   0.31   0.26
```

Steps: 7.71, 5.10, 4.16, 2.46, 2.27, 2.15, 2.11, 2.09, 2.01 dB for positions 2 -> 11, then 0.92-1.08 dB per step to 24. Family
over 24 x 33 points: 0.065 rms, 0.36 max; over the 362 compressing points 0.095; the fifteen positions the shape was never fitted
on: 0.067 rms, 0.36 max. Position 1's knee is unconstrained (the fit lands at +13.6 dBFS with rms 0.003 because nothing below
+14 dBFS compresses); -9.4 dB continues the taper 8 dB below position 2 and keeps the knee at +19 dBFS.

The best two-segment law on positions 5..24 (break at position 11, 2.14 dB/step below, 0.968 above) misses the table by 0.117 dB
rms and 0.29 max, and positions 3-4 (steps 7.7, 5.1) fit no such law. Keep the 24-entry table; the C++ already has it.

### 4.2 The no-GR gain

The reference's flat gain is identical at -70 and -50 dBFS (to 0.00000 dB) and falls with the threshold position: +0.561 at 1,
+0.555 at 8, +0.531 at 12, +0.488 at 16, +0.381 at 20, +0.276 at 22, +0.116 at 24 dB. It is a bias, not a compression floor, and
it grows as the square of the sidechain drive (exponent 2.05 by direct log-log fit on positions 8..24; 2.08 when the drive is the
fitted shift; 2.154 in the joint V6 fit). The law:

```
g0(pos) = g_amp - 20 log10(1 + cond0(pos)),   cond0(pos) = o_leak * (A_pos / A_20)^o_leak_q,   g_amp = +0.561 dB (make-up 12 + Nickel)
```

entered as light before the cell law (so that above the knee it is swamped by the signal conductance, which is what the family
prefers: in-loop 0.019 rms versus 0.021 as a gain factor in the static solve, and what V6 fitted). My run, model minus reference
flat gain by position with `o_leak 0.0202, o_leak_q 2.154`: within +0.007 dB from position 1 to 20, -0.001 at 22, -0.012 at 23,
-0.025 at 24 (rms 0.0069); with the direct-fit pair `0.0209, 2.08`: rms 0.0062, the same -0.025 at 24. The two are equivalent;
the fit in 4b will settle the pair with the knees. Against the discriminate rows at -70 and -50 dBFS (positions 1, 4, 8, 12, 16,
20, 22, 24) the largest error is 0.025 dB (position 24), the rest under 0.01.

The GR meter should subtract the leak (`grDb()` above): at position 24 it is otherwise 0.45 dB at idle.

## 5. The sidechain's high-frequency behaviour

### 5.1 What is measured

The audio path is flat: at -40 dBFS the gain moves by -0.004 / -0.018 / -0.041 / -0.070 / -0.103 / -0.135 / -0.175 dB at 2 / 4 /
6 / 8 / 10 / 12 / 16 kHz relative to 1 kHz (`stage_resp_opto` agrees: -0.07 dB at 8 kHz). The gain reduction, however, falls
with frequency: converting the HF rows at -10 and 0 dBFS to equivalent 1 kHz levels through the static curve gives a sidechain
attenuation of 0 / 0.07 / 1.24 / 4.7 / 8.58 / 11.78 / 14.27 / >= 17 dB (at -10 dBFS) and 0 / -0.4 / 0.85 / 4.41 / 8.55 / 11.73 /
14.5 / 19.99 dB (at 0 dBFS) at 1 / 2 / 4 / 6 / 8 / 10 / 12 / 16 kHz. A second-order low pass fits at 0.27 dB rms (free order
N = 2.00; fc 5147 Hz Q 0.718, or Butterworth 5170 Hz); a first-order pole is refuted (2.23 rms, 5.1 max), two real poles 1.07,
third order 1.64. Post-rectifier averaging cannot do it (bounded at 0.47 dB between frequencies). The 8 kHz static series says
the same (knee 8.3 dB higher than at 1 kHz, no-GR gain moved by 0.07 dB).

### 5.2 What to implement

A second-order low pass on the sidechain only, after the high pass and before the rectifier, `fc = o_sc_lp_hz = 5147 Hz`,
`Q = o_sc_lp_q = 0.718`, **magnitude-matched to the analog curve**. This matters: an RBJ bilinear biquad at 48 kHz matches the
analog response at fc only and its warp adds attenuation toward Nyquist, which the reference does not have:

```
                  2k     4k     6k     8k    10k    12k    16k    20k   (dB relative to 1 kHz, fs 48 kHz)
analog          -0.06  -1.24  -4.42  -8.26 -11.77 -14.81 -19.73 -23.58
RBJ bilinear    -0.05  -1.17  -4.59  -9.15 -13.76 -18.26 -27.77 -41.11
Vicanek matched -0.06  -1.24  -4.42  -8.25 -11.75 -14.75 -19.45 -22.56
```

With the bilinear design the HF rows come out 1.6 / 3.1 / 4.3 dB short of GR at 10 / 12 / 16 kHz, 0 dBFS (rms 0.85 / 1.96 at
-10 / 0 dBFS); with the matched design 0.26 / 0.13 rms, max 0.45 / 0.21. The 8 kHz static series goes from 4.18 rms / 6.16 max
(no low pass) to 0.53 / 1.74; the 3 kHz series from 0.60 / 1.49 to 0.455 / 1.14; 100 Hz is untouched (0.108 -> 0.111). A refit
of (fc, Q) on the 3 kHz + 8 kHz statics gives 5544 Hz / 0.661 (statics 0.417 / 0.414, HF rows 0.126 / 0.400): the HF-row values are
kept as the prior because they are the direct measurement; either is within 0.4 dB everywhere that is not section 6.5.

Coefficients (Vicanek, "Matched Second Order Digital Filters", 2016; validated by the table above), for `w0 = 2 pi fc / fs`,
`q = 1 / (2 Q)`, direct form II transposed `y = b0 u + z1; z1 = b1 u - a1 y + z2; z2 = b2 u - a2 y`:

```cpp
void Biquad2::setMatchedLowPass(double fc, double Q, double fs)
{
    const double w0 = 2.0 * M_PI * fc / fs, q = 1.0 / (2.0 * Q);
    a1 = q <= 1.0 ? -2.0 * std::exp(-q * w0) * std::cos(std::sqrt(1.0 - q * q) * w0)
                  : -2.0 * std::exp(-q * w0) * std::cosh(std::sqrt(q * q - 1.0) * w0);
    a2 = std::exp(-2.0 * q * w0);
    const double A0 = (1.0 + a1 + a2) * (1.0 + a1 + a2), A1 = (1.0 - a1 + a2) * (1.0 - a1 + a2), A2 = -4.0 * a2;
    const double p1 = std::sin(0.5 * w0) * std::sin(0.5 * w0), p0 = 1.0 - p1, p2 = 4.0 * p0 * p1;
    const double R1 = (A0 * p0 + A1 * p1 + A2 * p2) * Q * Q;
    const double B0 = A0, B1 = (R1 - B0 * p0) / p1;
    b0 = 0.5 * (std::sqrt(B0) + std::sqrt(B1)); b1 = std::sqrt(B0) - b0; b2 = 0.0;
}
```

At 96 kHz the two designs agree within 1 dB to 16 kHz, so this is also what keeps the model sample-rate invariant in the
sidechain (the states already are: `sr_opto_44100 / 96000` envelopes 0.067 / 0.066 rms against 0.067 at 48 kHz).

The audio-path loss (-0.07 dB at 8 kHz, -0.17 at 16 kHz, -0.18 at 20 kHz) is not modelled in the default profile; the Measured
Unit profile's `o_hw_lp_hz` is the hook if it is ever wanted. The sidechain high pass (88.5 Hz, first order, `scf_opto_*` items)
is settled and unchanged. The stereo optical stages stay unlinked, as the engine already has them (`link_opto_*`: the left
channel's gain is -10.103 dB whatever the right channel does).

## 6. What remains unexplained

All numbers from `/tmp/opto_synthesis.log` (the chosen model with the matched low pass) unless stated.

1. **H5 across the level grid.** On the three items V6 was fitted on, H5 is within 1.3 dB; over the nine 1 kHz items it is
   9.3 dB rms (max 17.9): the model has a null at t22 / -10 dBFS (-82.9 dBc, reference -65.1) and is 15 dB too high at t14 / -10
   (-69.8 vs -84.8) and 10 dB too low at t14 / 0 (-72.9 vs -63.3), while the reference's H5 is monotonic in level at every
   position (t22: -81.0, -65.1, -58.6 at -20 / -10 / 0 dBFS). The hard turn-on on the instantaneous rectified sine makes the
   light a pulse train whose 4f component passes through zero as the duty cycle changes with level; the reference's ripple has no
   such null. At 100 Hz H5 is 7.1 dB rms off (max 13.9). H7 is 6-7 dB low at 1 kHz and 8 dB high at 100 Hz. H3, the audible one,
   is 1.2 dB rms over the nine 1 kHz items (max 3.1) and 2.9 dB at 100 Hz (max 6.8 at t14 / -20, a -64.6 dBc item). Nothing tested
   fixes the H5 pattern: the linear-dB law with a 0.5 ms peak detector has H5 3.3 dB rms over ten items but worse H3 (2.3 rms at
   1 kHz), the 100 Hz statics and no (D); the RMS-detector variants of the physical law made H5 worse. This is the open question
   for the ripple mechanism: something between the rectifier and the cell smooths the light over about a quarter cycle without
   softening the static knee.
2. **The ripple's frequency dependence, and the persistence pole.** With V6's `tau_el` = 0.066 ms the 4 kHz H3 is 8.7 dB rms low
   (t18 / -10: -69.2 vs -61.0 dBc). The reference's H3 falls 6 dB per octave from 100 Hz to 4 kHz (-27.9 / -48.0 / -61.0), the
   model's 10.5 dB per octave above 1 kHz: the 2.4 kHz persistence pole. With `tau_el` set to 0.01 ms and nothing else changed the
   4 kHz H3 error is 1.2 dB rms (max 2.3), the 1 kHz one rises to 2.9 rms (the model then has 1.5-3 dB too much ripple:
   t18 / -10 -46.1 vs -48.0, t22 / 0 -38.2 vs -41.4), 100 Hz stays 2.9, and the gain error over the 27 items improves from 0.21 to
   0.19 rms. The fit in 3.3 includes the 4 kHz items so this is resolved by the refit, at the cost of about 2-3 dB rms H3 at all
   three frequencies rather than 1.2 dB at 1 kHz alone.
3. **The first 20 ms of a release that ends below the knee.** After the -10 dBFS burst the model's gain is 0.58 / 0.44 / 0.12 dB
   below the reference at +5 / +20 / +50 ms (it releases more slowly at the top), after the -2 dBFS burst 0.99 / 0.63 / 0.18 dB,
   after `opto_burst_thr12` 0.78 / 0.59 / 0.21; then 0.09-0.16 dB above it at +100..+300 ms (the tail is slightly small) and
   within 0.02 dB from +1 s. The same at the top of the big above-knee step (-20 -> +5 -> -20: model -11.93 / -9.09 / -7.35 dB at
   3 / 6 / 9 ms, reference -10.61 / -7.84 / -6.30; 50 / 90 % at 4 / 15 ms vs 3 / 13). The stage-4 weighting hides this (weighted
   rms 0.010, plain 0.078 / max 1.00). A release law steeper than `c^2` at large `c`, or a fourth, faster state, would be the
   things to try.
4. **The reference's own step pattern above 0 dBFS.** At every position the 2 dB steps of the reference alternate between about
   1.1 and 1.4 dB (position 20, per 2 dB from -6 dBFS: 1.15, 1.12, 1.39, 1.45, 1.09, 1.07, 1.05, 1.34, 1.19, 1.38), and the wobble about a
   straight line is the same function of GR at positions 10 and 20 (correlation +0.93, 0.03 rms in the fine sweeps). It looks like
   a tabulated or piecewise law in the reference. Every model's residual carries it as a 0.2-0.3 dB pattern (max 0.36 in the
   family); it is the floor of the static fit.
5. **High level at high frequency.** Above +4 dBFS at 8 kHz the reference output is pinned near -10.3 dBFS (gain steps of 1.94 /
   1.92 / 1.90 / 1.87 dB per 2 dB of input, slope 0.95 dB/dB) where the low-pass model gives the 0.62 slope: residual +0.34 /
   +0.87 / +1.33 / +1.74 dB at +4..+10 dBFS. At 3 kHz from -2 dBFS up the model has 0.4-1.1 dB too much GR (the reference's steps
   above +2 dBFS are 1.08, 1.27, 1.79, 1.23 dB per 2 dB). Some high-level HF mechanism (sidechain saturation, or the reference's own
   internal filtering) that the second-order low pass does not capture; it is outside normal operating levels (+4 dBFS at 8 kHz
   is +18 dBu).
6. **Above +14 dBFS (B).** The extended rows at position 20: model -28.70 / -32.97 dB at +20 / +26 dBFS, reference -28.20 /
   -30.86: the reference's slope softens to 0.53 then 0.44 dB/dB where the law holds 0.62. Outside the fitted window and the
   plugin's normal range (+26 dBFS = +40 dBu); a drive ceiling (`OptoConfig::driveCeiling`, already a hook) is the natural form
   if it is ever fitted.
7. **The slope drifts slightly with position.** The reference's slope above 3 dB of GR is 0.640 at position 6 and 0.618 at 24
   (fine sweeps: 0.669 at position 10, 0.620 at 20); a shared `gamma` gives 0.647 / 0.643 at 10 / 20. Worth 0.03-0.05 dB rms per
   position (the closed-form floor per position is 0.04-0.11 dB), consistent with the 0.095 rms on compressing points.
8. **Burst onset.** In periods 2-4 of a burst the model attacks 0.5-0.6 dB faster than the reference at -18 dBFS
   (-2.88 / -3.62 / -4.04 vs -2.27 / -2.99 / -3.45 dB) and 0.5-0.7 dB at -10 dBFS; the first period and the settled value are
   right, as are the step attacks (50 / 90 % at 1 / 3, 1 / 2, 0 / 1 ms vs 1 / 4, 1 / 3, 0 / 1).
9. **Three states are more than the data needs.** The fitted states are nearly degenerate (attack 8.4 / 15.7 / 8.5 ms, release
   175 / 132 / 94 ms, all with the same quench). A single quenched state with one fast ripple state may do the same job; test it
   in 4b before committing to the array fields' values, but keep `kOptoStates = 3` in the layout.
10. **The audio-path HF loss** of item 5.1 (-0.07 dB at 8 kHz, -0.17 at 16 kHz) is not modelled in the default profile.

What is settled and should not be reopened without new data: feedback from the divider output (the release identity), the hard
knee at a fixed ratio (fine sweeps at two positions to 0.03 dB), threshold as a pure drive shift (24 positions), the no-GR gain as
an in-loop `A^2` leak (-70 = -50 dBFS), the absence of any above-knee tail (0.0000 dB at 0.1..2 s), the sub-second below-knee
tail as a consequence of the quenched release rather than a slow state, the second-order sidechain low pass near 5.15 kHz, and
the unlinked stereo stages.
