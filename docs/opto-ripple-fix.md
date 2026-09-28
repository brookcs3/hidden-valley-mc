# Opto ripple fix: the optical stage's harmonics under gain reduction

Status: design document, 2026-09-28, the synthesis of the three ripple harnesses under `fit/tools/candidates/opto-ripple-*.py`
(directions (a) to (e) of `docs/opto-fix.md` section 6) and of one run of my own, `fit/tools/candidates/opto-ripple-synthesis.py`
(549 s, log `/tmp/opto_ripple_synthesis.log`), which puts the adopted pieces together and scores them on everything this document
states. Nothing under `src/`, `fit/stages/` or `fit/data/` has been changed; section 3 says exactly what to change.

The short version. The hole as stated in the brief (H5 9.3 dB rms at 1 kHz with nulls, H3 10.5 dB per octave above 1 kHz, H7
6-8 dB off) was measured against the V6 constants of `docs/opto-fix.md`; the stage-4 refit that followed (the current
`fit/data/constants.json`: `o_n` 1.404, `o_gamma` 1.342, `o_vth` 6.479, `o_tau_el` 10 us, attacks 100-255 ms, releases 36-482 ms,
`o_rel_mu` 7.49) already took the 1 kHz H5 to 3.3 dB rms, H3 to 1.3, and the frequency law of H3 from 100 Hz to 1 kHz to 6.0 dB per octave
(reference 6.0; t18 / -10: -27.8 / -47.8 against -27.9 / -48.0), with 7.3 dB per octave from 1 to 4 kHz against the reference's
6.5. What is left is smaller and sharper: the position and depth of the H5 null, H7 at 1 kHz, H3 just above the knee, and
the 4 kHz rows. Every mechanism proposed for it was tested and refuted, one of them (a free quench exponent) by this synthesis; the
reference's own harmonics say why (section 1.2). Two things are adopted: the amplifier terms move to the divider output (the H2 rows
under gain reduction, direction (e)), and the fit's harmonic objective changes (all 27 items, H7, the fine knee sweeps, the 4 kHz H5
rows unfitted). The odd-harmonic hole itself is not solved; section 5 ranks what is worth a further round.

Level convention: dBFS peak of a sine, 1.0 = 0 dBFS = +14 dBu; harmonics in dBc; "gain" includes the make-up at position 12 and the
Nickel path's midband (G0 = +0.562 dB). "Statics t20" is the model minus reference gain rms over the 33 protocol levels at threshold
20; "knee" is the rms over the fine sweeps (A) of `fit/data/discriminate_opto.json`; "bursts" is the stage-4 weighted rms over the
nine `opto_burst_*` / `opto_blen_*` / `opto_pulses` envelopes; "steps" are the above-knee steps (E). All harnesses render the loop on
a numba mirror of `OptoStage::process` validated against the C++ engine at the start of each run (statics to 0.0007 dB, H3/H5 to
0.03 dB; the engine's H2 is 0.4-0.9 dB above the mirror's because the Nickel driver term `x_a2` sits after the stage in the engine).

## 1. The evidence

### 1.1 Where the current model stands

The baseline of every table below is `fit/data/constants.json` as of this morning, rendered on the mirror at 48 kHz (my run, section
1 of the log; the loop-filter harness's baseline in `/tmp/opto_ripple_loop_verify.log` is identical to the digit):

| | H3 rms 100 / 1k / 4k | H5 rms 100 / 1k / 4k | H7 rms 100 / 1k | H2 rms | gain | statics t20 | knee t20 / t10 | bursts | 3 kHz / 8 kHz |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.51 / 1.34 / 2.42 | 2.18 / 3.33 / 7.19 | 3.60 / 5.73 | 8.65 | 0.173 | 0.085 | 0.035 / 0.021 | 0.0076 | 0.238 / 0.365 |

The facts that must survive all hold on this baseline: the fine knee is 0.035 / 0.021 dB rms with the 0.5-to-3 dB width 4.24 dB
against the reference's 4.25 (t20) and 4.21 against 4.22 (t10); the top slope is -0.644 against -0.620 / -0.669; the no-GR rows (D) are
within 0.035 dB; the steps (E) release in 4-5 / 17-18 ms (50 / 90 percent; reference 4 / 17) with a tail of -0.010 dB at 100 ms and
0.000 at 500 ms (reference 0.000 / 0.000).

Per item, the baseline's odd harmonics at 1 kHz (model / reference, dBc), sorted by gain reduction:

| item | GR dB | H3 | H5 | H7 |
|---|---|---|---|---|
| t14 / -20 | 0.6 | -81.0 / -84.7 | -88.5 / -91.9 | -95.8 / -98.7 |
| t18 / -20 | 2.9 | -62.1 / -62.5 | -75.1 / -75.8 | -101.6 / -105.0 |
| t22 / -20 | 5.4 | -54.1 / -54.5 | -75.1 / -81.0 | -84.5 / -96.6 |
| t14 / -10 | 6.9 | -51.5 / -51.2 | -78.9 / -84.8 | -83.6 / -89.0 |
| t18 / -10 | 9.4 | -47.8 / -48.0 | -75.8 / -72.4 | -83.9 / -78.1 |
| t22 / -10 | 11.8 | -45.0 / -44.8 | -66.1 / -65.1 | -75.0 / -70.1 |
| t14 / 0 | 13.4 | -43.8 / -44.3 | -63.2 / -63.3 | -71.6 / -67.5 |
| t18 / 0 | 15.5 | -42.0 / -41.3 | -59.3 / -58.8 | -66.2 / -64.0 |
| t22 / 0 | 18.2 | -40.1 / -41.4 | -56.1 / -58.6 | -62.2 / -62.5 |

The pattern is the same at 100 Hz, 20 dB higher: the model's H5 dip sits one item too high in level (t18 / -10, where the reference
has -72.4 and the model -75.8) instead of at t14 / -10 (reference -84.8, model -78.9), and H7 is too high from 9 to 13 dB of gain
reduction and too low at 5 dB.

### 1.2 What the reference's harmonics say

The loop-filter harness reads the 27 reference items directly (`/tmp/opto_ripple_loop_verify.log`, section 1). The law any
mechanism has to produce, in numbers:

1. **One 1/f integrator and nothing else between the light and the divider.** H3, H5 and H7 fall 6.0, 6.9 and 6.2 dB per octave
   from 100 Hz to 1 kHz, and H3 keeps 6.3 dB per octave to 4 kHz. No pole below about 10 kHz acts on the light or on the conductance.
   The baseline already has this law (its persistence pole is at 10 us, 16 kHz; the fit pinned it at its lower bound).
2. **The integrator's time constant falls with conductance.** From the 2f ripple the reference implies tau_eff = 78.7 / 22.3 / 10.7 /
   4.6 ms at 0.6 / 2.9 / 6.9 / 18.3 dB of gain reduction, the same at 100 Hz and 1 kHz to 18 percent; the relative ripple is
   dc/cbar = 0.0087 cbar^0.47 (1 kHz / f) to 0.6 dB. The model's quenched release trel / (1 + mu c) gives 83.6 / 33.1 / 13.1 / 2.4 ms at
   the same points: right at the knee, 20-30 percent long from 3 to 7 dB, and half the reference's at 18 dB.
3. **The ripple's harmonic shape is a property of the waveform, not of a filter.** H5-H3 is the same at 100 Hz and 1 kHz within
   5 dB at every gain reduction; at 1 kHz it is -7 dB at the knee, dips to -34 dB at 6.9 dB of gain reduction and recovers to -17 dB
   at 18 dB. H7-H3 is -14 at the knee, dips to -43 at 2.9 dB and recovers to -21. A fixed-time smoother (a quarter cycle at 1 kHz is
   0.25 ms, nothing at 100 Hz) cannot produce a cycle-locked shape; whatever places the dip scales with the period.
4. **The model's dip comes from the states' attack / release asymmetry, not from the pulse train.** The light pulse
   max(0, sin - r)^1.884 has its 4f null at 30-42 dB of gain reduction (r 0.05-0.10); with linear symmetric states the rendered loop has
   no dip at all (`/tmp/opto_ripple_asym_diag2.log`, section 5); with the fitted asymmetric states its deepest conductance 4f/2f is
   -39 dB at 9.4 dB of gain reduction. The reference's dip is at 6.9 dB at both frequencies and only -34 dB deep. The mechanism has to
   move the dip and fill it, not remove it.
5. **The even harmonics belong to an amplifier after the divider.** H2 under gain reduction equals the no-GR H2 at the same input
   level minus the gain reduction, to 1.0 / 0.5 / 0.4 dB rms at 100 Hz / 1 kHz / 4 kHz (against 11-12 dB rms if the term were on the
   input). The b2 term acts on the divider output and takes no part in the ripple.
6. **The 4 kHz H5 rows are not ripple.** They sit at 20 kHz; their 1k-to-4k slopes range from 0.0 to 7.6 dB per octave against 6.9
   plus or minus 0.6 for 100 Hz to 1 kHz.

### 1.3 The candidates

Each row is the fitted result of one harness, scored against the same baseline (constants.json) with the harness's own objective;
"verdict" applies the brief's rule: the fine knee and the burst rms may not degrade by more than 0.01 dB, the statics, the fixed slope,
the release identity, the absence of an above-knee tail and the sub-second below-knee tail must survive. Logs: (a)
`/tmp/opto-ripple-pre-smooth.log`, (b)(c) `/tmp/opto_ripple_asym_fit5.log` and `_fs96.log`, (d)(e) `/tmp/opto_ripple_loop_rebase_full.log`
(the loop-filter verifier's rerun, `/tmp/opto_ripple_loop_verify.log`, reproduces its numbers), F1-F3 and the 96 kHz / HQ 2X rows
`/tmp/opto_ripple_synthesis.log`.

| candidate | added parameters (fitted value) | H3 100 / 1k / 4k | H5 100 / 1k / 4k | H7 100 / 1k | gain | t20 | knee 20 / 10 | bursts | 3k / 8k | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | | 1.5 / 1.3 / 2.4 | 2.2 / 3.3 / 7.2 | 3.6 / 5.7 | 0.173 | 0.085 | 0.035 / 0.021 | 0.008 | 0.238 / 0.365 | |
| (a) rc, tau forced 0.25 ms | dA -2.5, tau_el, n | 4.7 / 11.1 / 20.1 | 4.0 / 7.8 / 21.0 | 6.0 / 6.7 | 1.370 | 2.111 | 0.278 / 0.262 | 0.087 | 2.41 / 1.12 | refuted: the statics |
| (a) peak, rms, slew, rclevel, pre, forced at a quarter cycle | as stated in the harness | 1.5-7.4 / 4.9-14.7 / 16-21 | 3.6-6.0 / 5.4-61.7 / 15-57 | 4.7-7.7 / 4.7-67 | 1.2-2.3 | 0.5-2.8 | 0.16-0.82 | 0.05-0.13 | | refuted: the statics |
| (a) rc, rms, rclevel, pre, free | tau to its no-effect bound (10 us) | 1.2-1.5 / 1.1-1.3 / 2.4-2.8 | 2.1 / 3.3-3.6 / 4.9-5.1 | 3.5-3.7 / 5.8-5.9 | 0.23 | 0.10 | 0.046-0.066 / 0.023-0.039 | 0.008 | | nothing added; the change is dA and n |
| (a) peak, free | tau_rise 0.5 us, tau_fall 20 us, dA -0.22, n 1.406 | 0.8 / 0.8 / 2.4 | 1.9 / 2.3 / 5.5 | 3.6 / 6.0 | 0.249 | 0.103 | 0.089 / 0.063 | 0.009 | 0.416 / 0.389 | refused: knee +0.054 / +0.042, 100 Hz statics 0.134 |
| (b) asymmetric persistence | tau_fall | 1.8 / 1.5 / 3.9 | 2.3 / 3.1 / 5.0 | 4.1 | | 0.075 | 0.029 / 0.032 | 0.0073 | | fits tau_fall = tau_rise: nothing added |
| (b') fall rate (1 + kappa L) | tau_fall, kappa | 1.5 / 1.3 / 2.5 | 2.3 / 4.0 / 5.3 | 4.4 | | 0.084 | 0.033 / 0.028 | 0.0073 | | no gain |
| (c1) light-dependent cell pole | tau, kappa | 1.8 / 1.3 / 3.7 | 2.2 / 4.0 / 5.3 | 4.8 | | 0.066 | 0.027 / 0.031 | 0.0071 | | no-effect bound |
| (c1') illumination-dependent attack | mu_a, attack scale | 2.0 / 1.9 / 1.7 | 2.8 / 4.3 / 5.3 | 5.5 | | 0.083 | 0.028 / 0.035 | 0.0072 | | no gain, knee t10 +0.014 |
| (c2) second cell in parallel | share 0.3, its pole, its speed | 2.7 / 2.3 / 3.2 | 2.7 / 3.0 / 5.6 | 4.3 | | 0.079 | 0.031 / 0.034 | 0.0053 | | 0.002 dB of bursts for +1.2 dB of H3 |
| (c3) cell law after the integration | n, gamma, states refit | 1.8 / 1.8 / 1.7 | 3.6 / 5.4 / 5.3 | 6.7 | | 0.063 | 0.022 / 0.034 | 0.0034 | | trades H5 (+2.1) for bursts |
| (d1) sidechain low pass fc + Q | fc 5207, Q 0.770, tau_el | 1.1 / 1.3 / 0.6 | 2.1 / 3.3 / 1.4 | 3.6 / 6.3 | 0.254 | 0.084 | 0.044 / 0.022 | 0.007 | 0.271 / 0.427 | refused: its gain is the aliased 4 kHz rows; 1 kHz unchanged; 8 kHz statics +0.06, gain +0.08 |
| (d2) added sidechain pole | fp 16.2 kHz | 1.5 / 1.3 / 2.7 | 2.2 / 3.6 / 4.6 | 3.6 / 5.7 | 0.207 | 0.076 | 0.035 / 0.021 | 0.007 | 0.331 / 0.788 | no-effect bound; 8 kHz statics broken |
| (d3) sidechain lead / lag | fz 5.5 kHz, fp 18 kHz | 1.5 / 1.5 / 0.7 | 2.1 / 3.3 / 4.2 | 3.5 / 6.5 | 0.106 | 0.072 | 0.031 / 0.024 | 0.007 | 0.160 / 0.648 | 4 kHz only; 8 kHz statics +0.28 |
| (d4) pole after the cell law | tau_post 8 us | 1.5 / 1.3 / 2.3 | 2.2 / 3.4 / 6.1 | 3.6 / 5.8 | 0.181 | 0.088 | 0.037 / 0.021 | 0.008 | 0.261 / 0.359 | no-effect bound |
| (d5) pole on the conductance | tau_cond 3 us | 1.6 / 1.5 / 1.8 | 2.2 / 3.7 / 5.4 | 3.6 / 6.2 | 0.222 | 0.089 | 0.031 / 0.027 | 0.007 | 0.214 / 0.835 | no-effect bound; 8 kHz broken |
| (d6) mu + attack scale | mu 8.09, attack x0.99 | 1.9 / 1.7 / 1.9 | 2.2 / 3.3 / 6.0 | 3.4 / 5.7 | 0.172 | 0.093 | 0.033 / 0.030 | 0.005 | 0.223 / 0.361 | no harmonic gain |
| (d9) gamma, vth, mu, attack, no knee guard | gamma 1.423, vth 6.69, mu 8.19, attack x1.54, dthr +0.54 | 1.9 / 1.7 / 2.0 | 1.7 / 2.1 / 5.0 | 2.4 / 4.6 | 0.181 | 0.101 | 0.048 / 0.038 | 0.006 | 0.263 / 0.356 | refused: knee +0.013 / +0.017, statics +0.016 |
| (d9k) the same, knee in the objective | gamma 1.392, vth 6.50, mu 8.13, attack x1.23, dthr +0.20 | 1.9 / 1.7 / 2.0 | 1.9 / 2.7 / 5.2 | 2.8 / 5.2 | 0.177 | 0.095 | 0.042 / 0.031 | 0.006 | 0.245 / 0.352 | at the limit (knee +0.007 / +0.010, statics +0.010); a refit, no mechanism |
| (e) amplifier on the divider output | none | 1.5 / 1.4 / 2.3 | 2.2 / 3.3 / 7.2 | 3.6 / 5.7 | 0.173 | 0.086 | 0.035 / 0.021 | 0.0075 | 0.237 / 0.366 | **adopted**: H2 8.65 to 2.43 dB rms, nothing else moves |
| loop at 96 kHz (mirror) | none | 1.5 / 1.3 / 2.9 | 2.2 / 3.3 / 3.9 | 3.6 / 5.6 | | | | | | the 4 kHz H5 error is the 48 kHz loop's aliasing |
| C++ engine, HQ 2X | none | 1.5 / 1.3 / 3.0 | 2.2 / 3.3 / 4.0 | 3.6 / 5.6 | | | | | | same through the engine (STANDARD: 2.5 / 7.2) |
| F1 (e) + gamma, vth, mu, attack, dthr | gamma 1.356, vth 10.24, mu 8.44, attack x2.6, dthr +4.0 (bound) | 1.7 / 1.5 / 2.0 | 1.6 / 2.0 / 6.2 | 2.3 / 4.5 | 0.222 | 0.083 | 0.035 / 0.022 | 0.0057 | 0.367 / 0.633 | refused: 3 / 8 kHz statics +0.13 / +0.27; at a bound |
| F2 (e) + quench exponent rho, mu | rho 0.998, mu 7.95, dthr +0.05 | 1.8 / 1.6 / 1.9 | 2.2 / 3.2 / 5.9 | 3.4 / 5.7 | 0.175 | 0.096 | 0.034 / 0.026 | 0.0058 | 0.227 / 0.362 | rho returns to 1: the bimolecular quench is confirmed |
| F3 (e) + rho, gamma, vth, mu, attack, dthr | rho 1.10, gamma 1.456, vth 10.13, mu 8.11, attack x3.6, dthr +4.0 (bound) | 1.5 / 1.4 / 2.2 | 1.5 / 1.9 / 6.8 | 1.4 / 3.5 | 0.216 | 0.074 | 0.031 / 0.028 | 0.0040 | 0.382 / 0.716 | refused: 3 / 8 kHz statics +0.14 / +0.35; at a bound |

Two rows deserve a sentence. F2 is the one hypothesis none of the three harnesses had fitted (the release quench's exponent, rate =
(1 + mu s^rho) / trel, the shape the tau_eff law of 1.2 seems to ask for); started at rho 0.6 and at 1.0 it converges to 0.998 both
times. The quench is bimolecular; no new field. F1 and F3 show what the harmonic objective wants when the loop's own constants are
free: a light gain about 1.9 times larger relative to the idle leak (reached only by moving `o_vth` and the whole drive table together,
hence the bound), attacks 2.6-3.6 times longer and mu 8.1-8.4. They buy H5 at 1 kHz (3.3 to 1.9-2.0), H7 (5.7 to 3.5-4.5), the
bursts (0.0076 to 0.0040-0.0057) and the steps (release trajectory rms 0.027 / 0.024 / 0.080 to 0.010 / 0.017 / 0.019) while holding
the fine knee and the 1 kHz statics, but they break the 3 kHz and 8 kHz static series (0.24 / 0.37 to 0.38 / 0.72) and the 4 kHz
gains (t14 / 0: -12.31 against -11.76): a stronger attack / release asymmetry biases the mean conductance by an amount that depends on
the ripple, and the ripple depends on frequency. That is a frequency-dependent static, and the sidechain low pass fitted in stage 4d
would have to absorb it. They are pointers for the joint fit (section 5), not a change.

### 1.4 The 4 kHz rows

With the same constants the 4 kHz H5 error is 7.19 dB rms with the loop at 48 kHz (the model's dip on t18 / -10 at -96.9 against
-81.8), 3.93 with the loop at 96 kHz (dip on t14 / -10 at -89.0 against -91.4, where the reference has it at every frequency) and 4.03
through the C++ engine in HQ 2X. The light of a hard turn-on on a rectified 4 kHz sine has harmonics at 8, 16, 24, 32 kHz and beyond;
in a 48 kHz loop the 32 kHz component folds onto 16 kHz, which is what H5 carries. The remaining 3.9 dB at 96 kHz is the rows
t18 / -20 (-96.5 / -90.9) and t22 / -20 (-88.4 / -81.5), where the reference's H5 sits above the law of its other rows (1.2, point 6).
STANDARD mode runs at the host rate by design (`README.md`, `docs/MODEL.md` section on quality: the reference does not oversample
either); the 4 kHz H5 rows are therefore not to be fitted as ripple, and the model's value for them is the HQ 2X one.

## 2. The chosen change

The optical stage after the change, in the order `OptoStage::process` runs it, one sample at the internal rate fs. Everything
that is not marked "changed" is the current engine.

```
1. divider (changed: the stage input goes straight into the divider)
     g   = 1 / (1 + cond)
     v   = g * x + noise                       noise: the uranium decay injection, 0 otherwise
   Measured Unit profile only: v -> hwLp(v) -> grLoss(v)   (first-order poles on the divider output, as now)

2. sidechain, from the divider output (feedback; unchanged)
     hp  = HP1(v, sc_hz)                       first-order high pass, 88.5 Hz; used when SIDECHAIN FILTER is in
     s   = scFilter ? hp : v
     s2  = LP2(s, o_sc_lp_hz, o_sc_lp_q)       second-order low pass, magnitude-matched, 5025 Hz, Q 0.680
     A   = 10^(o_thr_db[thr] / 20)             smoothed over 15 ms on a switch
     d   = |A * s2|                            (material hook: min(d, driveCeiling))

3. panel (unchanged)
     e      = d - o_vth                        hard turn-on, o_vth 6.479 drive units
     Linst  = e > 0 ? e^o_n * lightGain : 0    o_n 1.404
     L     += (Linst - L) * kEl                kEl = 1 - exp(-1 / (o_tau_el fs)), o_tau_el 10 us
     L2    += (L - L2) * kEl2                  material hook (uranium second pole); kEl2 = 1 otherwise

4. cell (unchanged)
     leak   = o_leak * (A / A20)^o_leak_q      as a conductance, o_leak 0.02497, o_leak_q 1.484, A20 = 10^(o_thr_db[20] / 20)
     L0     = leak^(1 / o_gamma)               the leak as light
     target = (L2 + L0)^o_gamma                o_gamma 1.342
     for each state i in 0..2:
       k_i  = target > st_i ? kAtt_i : 1 - exp(-(1 + o_rel_mu * st_i) / (o_trel_i fs))     kAtt_i = 1 - exp(-1 / (o_tatt_i fs))
       st_i += (target - st_i) * k_i
     cond   = sum_i o_w_i * st_i               (light-memory and half-life options as now)

5. stage amplifier (changed: after the divider, before the make-up)
     b2 = o_b2, b3 = o_b3                      CLASS A: b2 += ca_a2 (1 + ca_a2_env env(v)), b3 += ca_a3, env tracking v
     uf = b3 < 0 ? 1 / sqrt(-3 b3) : inf       the fold of the polynomial: clip there instead of inverting
     vc = clamp(v, -uf, uf)
     va = vc + b2 vc^2 + b3 vc^3

6. make-up (unchanged)
     out = va * 10^(o_gain_db[gain] / 20)      smoothed over 15 ms on a switch; CLASS A: ceiling(out)
```

Why this is the physical order: a T4B-style stage is a passive series-resistor / cell divider followed by the gain stage; the
divider cannot generate an even harmonic, the amplifier after it generates one proportional to its own input level, which is the
divider output. The reference says exactly that (1.2, point 5). The sidechain now sees the clean divider output rather than the
distorted stage input; at the fitted b2 (2.5e-5) that changes H3 / H5 by 0.1 dB and nothing else (table row (e)).

Parameters of the stage after the change (all existing fields; no field is added, none is removed):

| parameter | unit | fitted value (`fit/data/constants.json`) | field | set by |
|---|---|---|---|---|
| sidechain high-pass corner | Hz | 88.54 | `sc_hz` | stage 5a |
| drive gain per threshold position | dB | -3.06 ... 48.00 (24 entries; position 20: 43.78) | `o_thr_db` | stage 4b (7 positions), 4c (all) |
| make-up per gain position | dB | -13.57 ... 16.90 | `o_gain_db` | stage 1 |
| panel light law exponent | 1 | 1.404 | `o_n` | stage 4b |
| cell law exponent | 1 | 1.342 | `o_gamma` | stage 4b |
| panel turn-on | drive units | 6.479 | `o_vth` | stage 4b |
| phosphor persistence | s | 1.0e-5 (at its lower bound) | `o_tau_el` | stage 4b |
| state weights | 1 | 0.045, 0.350, 0.605 | `o_w` | stage 4b |
| state attack constants | s | 0.2546, 0.1380, 0.0997 | `o_tatt` | stage 4b |
| state release constants at zero conductance | s | 0.4821, 0.0360, 0.1613 | `o_trel` | stage 4b |
| release quench per unit conductance | 1 | 7.489 (rho = 1, confirmed by F2) | `o_rel_mu` | stage 4b |
| idle leak at position 20, as a conductance | 1 | 0.02497 | `o_leak` | stage 4b |
| leak growth with the drive | 1 | 1.484 | `o_leak_q` | stage 4b |
| sidechain low pass corner, Q | Hz, 1 | 5025, 0.680 | `o_sc_lp_hz`, `o_sc_lp_q` | stage 4d |
| amplifier even term, **on the divider output** | 1 | 2.53e-5 in constants.json; 2.551e-5 from the 4a pass running this morning (`build_stage4b.log`, not yet saved) | `o_b2` | stage 4a |
| amplifier odd term, **on the divider output** | 1 | -6.09e-4 in constants.json; -4.941e-4 from the same 4a pass | `o_b3` | stage 4a |

The fit-side change that goes with it (section 3.3) alters no equation; it alters what stage 4b is asked to match.

## 3. Exact changes

### 3.1 `src/dsp/Opto.hpp`

Lines as read this morning (the file may be mid-edit by the engine's owner; the anchors are the statements, not the numbers).

Header comment, line 8: replace `stage input x -> amplifier (x + b2 x^2 + b3 x^3) -> divider v = g x -> make-up -> out` with
`stage input x -> divider v = g x -> amplifier (v + b2 v^2 + b3 v^3) -> make-up -> out`, and add after the sentence on the idle leak:
`the amplifier sits after the divider (measured: H2 under gain reduction is the no-GR H2 at the same input level minus the gain
reduction, to 0.4-1.0 dB rms), so its even term scales with the output level and takes no part in the loop`.

`process()`, lines 110-116, currently:

```cpp
double b2 = c[kc_o_b2], b3 = c[kc_o_b3];
if (cfg.classA) { caStage.track(x); b2 += caStage.a2Now(); b3 += caStage.a3(); }
const double uf = b3 < 0.0 ? 1.0 / std::sqrt(-3.0 * b3) : 1e30;
const double xc = x > uf ? uf : (x < -uf ? -uf : x);
const double xa = xc + b2 * xc * xc + b3 * xc * xc * xc;
const double g = 1.0 / (1.0 + cond);
double v = g * xa + noise;
```

becomes:

```cpp
const double g = 1.0 / (1.0 + cond);
double v = g * x + noise;   // the divider sees the stage input; the amplifier comes after it (docs/opto-ripple-fix.md section 2)
```

and lines 164-165, currently:

```cpp
const double out = v * dbToLin(gainS.tick(c[kc_o_gain_db + cfg.gain]));
return cfg.classA ? caStage.ceiling(out) : out;
```

become:

```cpp
// the stage amplifier, after the divider and before the make-up: its even term scales with the divider output (the H2 rows)
double b2 = c[kc_o_b2], b3 = c[kc_o_b3];
if (cfg.classA) { caStage.track(v); b2 += caStage.a2Now(); b3 += caStage.a3(); }   // the Class-A module's terms on the amplifier
const double uf = b3 < 0.0 ? 1.0 / std::sqrt(-3.0 * b3) : 1e30;   // beyond the polynomial's fold it clips instead of inverting
const double vc = v > uf ? uf : (v < -uf ? -uf : v);
const double va = vc + b2 * vc * vc + b3 * vc * vc * vc;
const double out = va * dbToLin(gainS.tick(c[kc_o_gain_db + cfg.gain]));
return cfg.classA ? caStage.ceiling(out) : out;   // the module's ceiling, outside the loop
```

The Measured Unit filters (`hwLp`, `grLoss`, lines 117-123) stay on `v` before the sidechain tap, so the polynomial acts on the
filtered divider output; `gLast = g` and the sidechain block (lines 124-135) are untouched. `grDb()` is untouched. The Class-A
envelope now tracks the module's input, which is the divider output; `docs/class-a-profile.md` section 3.4 ("on the stage input where
`o_b2, o_b3` already act") needs the one-line correction, and its statement that the placement inside the loop is invisible at
-87 dBc stays true.

### 3.2 `src/dsp/Calibration.hpp`

No field is added and no prior changes. The file gained `o_bleed` from another agent while this document was being written; the
layout hash moves for that reason, not for this one. Comments to correct:

```cpp
S(o_b2, 0.0)                /* stage amplifier: quadratic term, on the divider output (before the make-up) */
S(o_b3, 0.0)                /* stage amplifier: cubic term, same place */
S(o_rel_mu, 7.7)            /* release rate multiplier per unit conductance: rate = (1 + o_rel_mu s) / o_trel (bimolecular; a free exponent on s fits to 1) */
```

The `docs/MODEL.md` section 4 rows for `o_b2`, `o_b3` ("on the stage input") change accordingly.

### 3.3 `fit/stages/stage4_opto.py`

1. **The mirror** (`run_one`, lines 53-79): move the polynomial to the output, in the C++ order:
   ```python
   uf = 1.0 / np.sqrt(-3.0 * b3) if b3 < 0.0 else 1e30      # before the loop
   ...
   v = xi / (1.0 + cond)
   vc = uf if v > uf else (-uf if v < -uf else v)
   out[i] = vc + b2 * vc * vc + b3 * vc * vc * vc
   ```
   (the sidechain low pass reads `v`; this is what `opto-ripple-synthesis.py`'s `run_one` does with `P[iAMPOUT] = 1`, validated
   against the engine on the current placement to 0.03 dB in H3 / H5). The validation set at the top of `run()` keeps comparing H3 / H5 only; the engine's H2 carries the Nickel
   driver term the mirror does not have.
2. **4a** is unchanged in form (threshold 1; the leak conductance at position 1 is below 1e-5, so b2 and b3 fit the same values);
   rerun it after the C++ change so the constants file records the new placement.
3. **4b items** (line 30): `HARMS` becomes every `opto_harm_t*` item (27; the list `ALL_HARMS` already exists at line 31), not the
   13 of today.
4. **4b harmonic residual** (`harm_resid`, line 168): `wg=2.0, w3=1.0, w5=0.5, w7=0.25`, with the 4 kHz H5 rows at weight 0
   (`w5 = 0.0 if ITEMS[iid]["stim"]["f"] == 4000 else 0.5`, section 1.4) and H7 added (index 5 of the `h` list, skipped where the
   reference is None or below -100 dBc); the model's odd harmonics floored at -110 dBc as now.
5. **4b objective** (`resid`, lines 208-215): add the fine knee sweeps (A) of `discriminate_opto.json` at both positions (20 and 10;
   74 levels, rendered at 1.5 s, gain over the last 0.5 s) at weight 10 per dB, and the 3 kHz and 8 kHz static series at threshold 20
   at weight 2 per dB (14 levels each, from `opto_static_f3000_*` and `_f8000_*`), so that the harmonic terms cannot buy their gains
   with a frequency-dependent static (the F1 / F3 failure of section 1.3). The statics stay at x2, the bursts at their weights.
   Order of the residual vector: statics, bursts, harmonics, knee sweeps, 3 kHz / 8 kHz statics. `DISC` is already loaded for the
   report (line 276); load it at module level.
6. **Bounds and scales** (lines 158-160): unchanged except the persistence, whose lower bound moves from -5.0 to -6.0 (log10 s); the
   fit sits on -5.0 today and every harness that freed it went below (5-8 us). `x_scale` unchanged.
7. **Order**: 4a, 4b, 4c, 4d as now; 4d (the low pass on the 3 kHz / 8 kHz series through the engine) still runs after 4b, and the
   3 kHz / 8 kHz guard inside 4b makes the two consistent.
8. **Report** (lines 269-275): print H2 model / reference per item next to H3 / H5 (the even rows are now a prediction of the
   placement with no free parameter), and the 27-item H3 / H5 / H7 rms by frequency; add the HQ 2X (`quality=1`) rendering of the
   4 kHz items as the model's value for those rows.

What this is expected to give, from the runs that used the same ingredients: the (e) row of section 1.3 for the even harmonics
(2.4 dB rms on the mirror, less through the engine which carries the Nickel driver term), and for the odd ones the d9k row (H5 at
1 kHz 3.3 to 2.7, H7 4.7 to 4.1, bursts 0.0076 to 0.006) at the limit of the knee and static tolerances, or better if the joint fit
finds the light-gain direction of F1 / F3 without the frequency-dependent static (section 5, direction 1).

## 4. What remains unexplained

With (e) in place and the constants as fitted (the (e) row; every number from `/tmp/opto_ripple_synthesis.log`, section 3):

- **The H5 dip.** The model's is at 9.4 dB of gain reduction and -39 dB deep in the conductance's 4f / 2f (t18 / -10 at 1 kHz:
  -75.8 against -72.4); the reference's is at 6.9 dB and -34 dB (t14 / -10: model -78.9 against -84.8). Six dB the wrong way on two
  items, at 100 Hz as at 1 kHz. Nothing that acts in time moves it (1.2, point 3); what moves it is the states' attack / release
  asymmetry against gain reduction (the knob sweep in `/tmp/opto_ripple_loop_verify.log`: gamma x0.85 puts it in the right place with
  a 1 kHz H5 error of 1.4 dB and a static error of 0.79 dB).
- **H7 at 1 kHz** (5.7 dB rms): too high from 9 to 13 dB of gain reduction (t18 / -10 -83.9 against -78.1 is too low, t22 / -10 -75.0
  against -70.1, t14 / 0 -71.6 against -67.5 too high) and 12 dB too low at t22 / -20 (-84.5 against -96.6); at 100 Hz 3.6 dB rms with
  the same signs. The 6f content of the model's ripple is not the reference's just above and just below the dip.
- **H3 just above the knee**: t14 / -20 (0.6 dB of gain reduction) is 3.7-4.4 dB too high at 100 Hz, 1 kHz and 4 kHz alike (-60.2 /
  -64.6 at 100 Hz); the states' mean bias near the knee, or the leak's share of the target there. Everything from 3 dB of gain
  reduction up is within 1.3 dB.
- **H3 at 4 kHz**, 2.3 dB rms with mixed signs (t18 / -20 -83.8 against -79.3, t22 / -20 -71.1 against -68.4 too low; t14 / 0 -56.6
  against -54.6, t18 / 0 -53.6 against -51.9 too high): partly the aliased loop (2.8-2.9 at 96 kHz, so not only), partly the 4 kHz gain
  itself (0.3-0.6 dB) which is the sidechain low pass's business (stage 4d).
- **The 4 kHz H5 rows**: 4.0 dB rms in HQ 2X, 7.2 in STANDARD; not ripple (1.2, point 6; 1.4).
- **H2**: 2.4 dB rms on the mirror with the model 2-4 dB low at 1 kHz (t22 / -10: -119.9 against -117.2); the mirror lacks the Nickel
  driver term that the engine and the reference carry (0.4-0.9 dB in the validation rows), so the engine's figure will be lower. What is
  left is the 4a fit of b2 on six no-GR levels.

## 5. Verdict, and the directions worth a further round

The hole is not solved. Directions (a), (b), (c), (d) are closed by their harnesses: anything that smooths the drive before the turn-on
softens the knee and the statics (2 dB rms when forced to a quarter cycle); anything that smooths the light after it, symmetric or
not, fits to no effect; the sidechain's poles and the low pass fit only the aliased 4 kHz rows; a second cell buys 0.002 dB of bursts
for 1.2 dB of H3; the quench exponent fits to 1. Direction (e) is adopted for the even harmonics, and the fit's objective is
corrected. The reference's own law says what a solution must be: a cycle-locked asymmetry in the cell's response whose mean does not
depend on the ripple. Ranked by the evidence:

1. **The joint refit under the corrected objective, with the light gain and the leak decoupled.** F1 and F3 hit the +4 dB bound of
   the drive-table shift with `o_vth` co-moving, which is the only route the current parametrisation has to a larger light relative to
   the idle leak; in the joint fit (4b) the leak, the knees, `o_vth`, `o_gamma`, the states and mu are all free, so the direction can be
   taken without the bound and with the 3 kHz / 8 kHz series and the knee sweeps as guards. Expected: some of F3's H5 (1.5 / 1.9 at
   100 Hz / 1 kHz) and H7 (1.4 / 3.5), bursts 0.004 and steps 0.010-0.019, if the frequency-dependent static can be avoided; the run
   itself says whether it can. No C++ change; one run of stage 4.
2. **The light pulse's shape at high drive.** Beyond the dip the reference's 4f / 2f recovers to -11 dB at 18 dB of gain reduction
   where the model's (sin - r)^1.884 pulse gives -23: the reference's light pulse is squarer than the model's. A soft ceiling on the
   light far above the knee (a compressive panel law or an EL driver swing limit: `driveCeiling` is already a material hook in
   `OptoConfig`; as a fitted field, `L -> L / (1 + L / o_l_sat)` after the turn-on, one parameter) is cycle-locked, leaves the knee
   untouched by construction, and has not been tested. Run it at 96 kHz so the 4 kHz rows do not mislead.
3. **A conductance-dependent attack** as the knob for the dip's position (9.4 to 6.9 dB of gain reduction): (c1') and (c5) went to
   their no-effect bounds under the asym harness's burst-dominated weights at 48 kHz; retest under the knee-constrained objective
   with the 3 kHz / 8 kHz guard. It keeps the release identity (the quench is untouched) and the sub-second tail.
4. **The near-knee ripple** (H3 +4 dB at 0.6 dB of gain reduction, H7 too low at 5 dB): the leak's share of the target and the
   states' mean bias at small light; test in the joint fit with the -20 dBFS items weighted up before adding anything.
5. Not worth another round: pre-turn-on smoothing of any kind, asymmetric or light-dependent persistence, poles anywhere in the
   loop, the sidechain low pass as a ripple shaper, a second cell, the quench exponent.
