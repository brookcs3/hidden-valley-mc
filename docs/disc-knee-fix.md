# Discrete knee fix: how the soft ratios fade to zero below the threshold reference

Synthesis of the soft-ratio knee round, 2026-09-28. Every figure below is from a run that was actually made; the run is named
beside it. Harnesses: `fit/tools/candidates/disc-knee-feedthrough.py` (direction a), `disc-knee-per-ratio.py` (direction b),
`disc-knee-second-bleed.py` (directions c, d, b'), `disc-knee-gr-domain.py` (the synthesis check of the previous pass at this
document: the node after the gain computer), and two synthesis checks under `build/` (`disc-knee-synth-check.py`,
`disc-knee-synth-check2.py`, logs `/tmp/disc-knee-synth-check.log`, `/tmp/disc-knee-synth-check2.log`) that run the gr-domain
harness's mirror on the reference's per-ratio steady table and bursts captured by `disc-ratio-ballistics.py`
(`build/disc-ratio-ballistics/measure.json`, 07:52), which none of the knee harnesses had.

**The tree moved while this was written.** State at the time of writing:

| file | time | what it is |
|---|---|---|
| `fit/data/constants.json` | 08:25 | rewritten by a stage-3 run during the round (attack 0.029 / 0.503 / 2.095 / 7.23 / 21.8 / 66.2 ms, depth 0.389, Sv 13.99, T16 -37.755); the round's harnesses ran on the 21:47 file (attack 0.01 / 0.412 / 1.976 / 6.98 / 21.1 / 64.3 ms, depth 0.396, Sv 14.57, T16 -37.746) |
| `src/dsp/Discrete.hpp`, `src/dsp/Calibration.hpp`, `build/capi/libhvmc.dylib` | 08:30 | another agent's implementation of the per-ratio form (section 3.5): `d_ratio_off_db` {-2.84, -2.26, -1.29, 0, 0, 0} dB, `d_ratio_rel` {1.371, 1.106, 1.060, 1, 1, 1.131}, on top of the hard zero at rest (22:41) and the snap to rest (23:09) |
| `fit/measure/protocol.py` | 08:28 | gained `disc_tail_{ratio}_{recover}` items (-50 -> -10 dBFS for 2 s -> -50, 8.5 s post-roll) for the soft ratios |
| `fit/stages/stage3_discrete.py` | 23:06 | does not know the per-ratio fields; the on-disk curves were fitted with one shared rest, so the engine of 08:30 runs per-ratio offsets with curves that were not built for them until stage 3 is changed (section 3.5) |

The task's "current best" (statics 0.216 / 0.73, steady ~0.1, bursts ~0.01, DUAL 0.011) was the engine before the hard zero:
the 0.216 / 0.73 is the standing gain reduction the PCHIP returned at the rest point (0.326 / 0.731 / 0.451 dB at 1.2:1 / 2:1 /
3:1), not a knee figure; the stage-3 report set holds no knee point at all. With the hard zero the engine reads that set at
0.009 / 0.03 (verifier rerun, 22:44) and the knee lives in the 198-item set (0.087 / 1.01) and the knee band (0.153-0.168 / 1.01).

## 1. Evidence

### 1.1 What the reference does (measured this round)

Threshold 16 throughout (model rest -38.14 dBFS on the 21:47 constants), gain 12, sidechain filter out; GR in dB.

1. **Static knees (R1, second-bleed `--reference`; per-ratio harness on the grid).** Hard, nearly linear onsets at different
   levels: 1.2:1 and 2:1 start at rest - 1.9 dB (0.056 / 0.067 dB at rest - 1.75, then 0.26 / 0.60 dB per dB), 3:1 at rest - 0.85,
   4:1 at rest + 0.45, 6:1 at rest + 1.15, FLOOD at rest + 3.1. At 2:1: 1.065 dB at rest, 0.470 at rest - 1, 0.001 at rest - 2. Zero
   at low level at every ratio (0.002 dB noise floor).
2. **Post-burst tails (R2; feedthrough `--reference-tail`).** After -50 -> -10 dBFS for 2 s -> -50, attack 1 ms, recover 0.5 s,
   the GR at 0.25 / 0.5 / 1 / 2 / 4 / 8 s is 8.033 / 3.452 / 0.67 / 0.07 / 0.05 / 0.001 at 1.2:1, 12.244 / 5.265 / 0.998 / 0.082 /
   0.05 / 0.001 at 2:1, (n/a) / 5.467 / 1.035 / 0.083 / 0.05 / 0.001 at 3:1, 13.162 / 5.669 / 1.07 / 0.084 / 0.05 / 0.001 at 4:1.
   As a fraction of the in-burst GR (17.05 / 25.73 / 26.66 / 27.55) the tail is the same at every ratio: 0.471 / 0.202 / 0.039
   (1.2:1), 0.476 / 0.205 / 0.039 (2:1), 0.478 / 0.206 / 0.039 (4:1) at 0.25 / 0.5 / 1 s, although the static curves run from 0.27
   to 0.6 dB per dB at 1.2:1 and 0.94 to 1.08 at 4:1. Recover 0.1 s: 1.051 / 0.095 / 0.05 (1.2:1), 1.617 / 0.12 / 0.05 (2:1), 1.742 /
   0.126 / 0.05 (4:1); recover 1.2 s: 10.259 / 5.765 / 1.823 / 0.223 (1.2:1), 15.565 / 8.759 / 2.758 / 0.313 (2:1), 16.707 / 9.406 /
   2.962 / 0.333 (4:1). At every ratio the tail ends on a 0.05 dB plateau that lasts a few recover constants (reached at 1 s
   with 0.1 s recover, 4 s with 0.5 s, 4-8 s with 1.2 s) before it reads 0.001.
3. **Onset after a quiet pre-roll (R3, R3x).** With a 30 ms attack the 4:1 onset after a 2 s tone at -120, -80, -50 dBFS or at rest
   - 3 / -2 / -1 / -0.25 / +0.25 is identical to 0.01 dB per period (0.97 / 1.57 / 2.16 / 3.32 / 4.98 / 7.1 / 10.76 / 17.32 dB at periods
   1 / 2 / 3 / 5 / 8 / 12 / 20 / 40): the 4:1 state is floored below its onset. At 2:1 the standing GR before the onset varies
   smoothly through the 4:1 rest (0.223 / 0.458 / 0.546 / 0.635 / 0.725 / 0.922 / 1.347 at rest - 1 / -0.25 / 0 / +0.25 / +0.5 /
   +1 / +2) and the onset rises by the standing value.
4. **A tone below the 4:1 rest (R4).** At 2:1, rest - 1 reads 0.47 from silence and settles back to 0.47 within 2 s of a burst
   (0.998 at 1 s, on the tail of item 2); at 1.2:1 0.223 both ways; rest - 2 reads 0.001 both ways. The soft ratios follow a tone
   down to 1.9 dB below the 4:1 rest with no memory.
5. **The steady table at every ratio (ballistics finding 2, `build/disc-ratio-ballistics/tables.md`).** The sag of the steady GR
   from 1 ms to 30 ms attack at -10 dBFS is nearly the same in dB of gain at every ratio: -8.35 / -8.34 / -8.23 / -7.96 / -7.75 /
   -7.19 at recover 0.1 s and -4.83 / -4.42 / -4.27 / -4.12 / -3.94 / -2.60 at 0.5 s for 1.2:1 / 2:1 / 3:1 / 4:1 / 6:1 / FLOOD; the
   0.1 -> 1 ms sag is -0.49 / -0.44 / -0.43 / -0.41 / -0.39 / -0.25. A detector shared by the ratios in the level domain gives a sag
   proportional to the curve's slope (engine: -6.13 / -7.80 / -8.03 / -8.30 / -8.50 / -9.78 at 0.1 s). This is 2.2 dB at 1.2:1 and
   2.6 dB at FLOOD, larger than the knee hole, and no candidate of this round fixes it (section 4).
6. **Release fall per period (same table).** In equivalent sine level the release is slower at 1.2:1 than at 4:1 (0.0697 against
   0.0779 dB per period over periods 5-40 at 0.5 s; 0.0540 against 0.0648 over 40-120), which the shared level node cannot
   produce (0.079 / 0.066 at every ratio).

### 1.2 The evidence table

Conventions: "54" is the stage-3 rendered report set (thresholds 4 / 12 / 20 x levels -30 / -15 / 0, all six ratios), "198" the
same thresholds at levels -30..0 step 3, "grid" the whole 6 x 24 x 25 static grid through the mirror by shift invariance,
"knee" the band rest - 4 < L <= rest + 2 (282 points; the per-ratio harness's 54 knee items are -4 < x < 4 at t4 / 12 / 20).
"silence" is the standing GR with no signal per ratio. Dynamics are the stage-3b residuals (steady table x 3, every burst envelope
per period at the stage weights, DUAL items apart), lock-in envelopes; the harnesses' DUAL accounting differs by convention
(second-bleed 0.011 / feedthrough and gr-domain 0.018 on the same constants), so each candidate is read against its own
harness's baseline. "R2" is the post-burst tail rms / max over the 54 reference numbers of fact 2 (three ratios, three recover
positions), "R3" and "R4" the onset and below-rest tone probes of facts 3 and 4, "F2" the per-ratio steady table of fact 5
(324 cells, my check, rms over all / at 1.2:1). rms / max in dB.

| candidate (harness, run) | 54 | 198 | grid | knee | silence (1.2:1 / 2:1 / 3:1 / 4:1) | steady | bursts | DUAL | R2 | R3 | R4 | F2 all / 1.2:1 | new constants | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| engine before the hard zero (21:47 constants; feedthrough, per-ratio, second-bleed baselines) | 0.216 / 0.73 | 0.197 / 0.73 | 0.243 / 0.73 | 0.220 / 0.73 | 0.326 / 0.731 / 0.451 / 0.044 | 0.080-0.084 / 0.25 | 0.010-0.012 / 1.00 | 0.011 (0.018) / 0.07 | - | - | - | - | 0 | the task's "current best"; the probes were not run against this engine |
| engine with the hard zero and the snap (22:41 / 23:09; verifier rerun 22:44, gr-domain baseline 07:5x) | 0.009-0.011 / 0.03-0.04 | 0.087 / 1.01 | 0.046 / 1.01 | 0.153 / 1.01 | 0.000 from reset; 0.326 / 0.731 / 0.450 / 0.044 standing for > 4 s after any signal (the snap needs v - rest < 1e-9) | 0.084 / 0.25 | 0.012 / 1.00 | 0.018 / 0.07 | 0.536 / 1.15 | 0.743 / 1.13 | 0.473 / 1.13 | 0.633 / 1.082 | 0 | baseline |
| (a) V2 feed-through, GR = C(v + alpha min(w - (rest - woff), 0) - T) (feedthrough, pass A) | 0.019 / 0.04 | 0.022 / 0.14 | 0.024 / 0.18 | 0.035 / 0.16 | 0 / 0 / 0 / 0 | 0.084 / 0.23 | 0.011 / 0.98 | 0.018 / 0.09 | predicts 0.94-1.25 dB standing for ~10 s after a burst (measured 0.082 at 2 s) | - | predicts 0.94 / 0.42 for 10 s (measured 0.47 / 0.223) | - | 3 (alpha 0.63, woff 0.24, trw at its 10 s bound) | refuted by fact 2 and 4 |
| (a) V1 blend / V5 1 ms feed-through / V4 gate (feedthrough) | 0.133 / 0.65, 0.116 / 0.53, 0.028 / 0.30 (grid) | | | | | dynamics unchanged | | | | | | | 1-3 | refuted (alpha fitted to 0.006-0.019; V4 needs the 10 s follower) |
| (b) F2 per-ratio depth, curves refit in the true node domain (per-ratio; engine pre-clamp) | 0.012 / 0.03 | 0.014 / 0.07 | 0.017 / 0.13 | 0.020 / 0.12 (54 knee items) | 0.001 / 0.001 / 0.001 / 0.006 | 0.089 / 0.25 | 0.013 / 0.99 | 0.019 / 0.10 | not evaluated (identical to mode 6 by the F3 == F2 identity: 2:1 tail 0.7 dB low at 1 s) | - | - | - | 3 (d = 2.75 / 2.75 / 1.5 dB) | statics solved; tails open |
| (b') mode 6 per-ratio reference T_r = T + off_r (second-bleed, log G, start off -2.2 / -2.2 / -1.0) | 0.024 / 0.07 | 0.019 / 0.08 | 0.019 / 0.14 | 0.015 / 0.10 | 0 / 0 / 0 / 0 (C++ -0.0005) | 0.080 / 0.26 | 0.010 / 1.00 | 0.011 / 0.07 | 0.532 / 1.80 (2:1 11.24 / 4.55 / 0.72 against 12.24 / 5.27 / 1.00) | as baseline | 0.439 against 0.47 | - | 3 | statics solved; tails 0.7 dB low; fitting the tails breaks the knee (0.092 / 0.47) |
| (b'') mode 9 = mode 6 + per-ratio bleed scale (second-bleed, log E, fitted with the probes in the residual) | 0.020 / 0.07 | 0.018 / 0.08 | 0.019 / 0.12 | 0.016 / 0.10 | 0 / 0 / 0 / 0 | 0.080 / 0.26 | 0.010 / 1.00 | 0.011 / 0.07 | 0.087 / 0.24 (in sample; 2:1 12.16 / 5.30 / 1.01) | 0.53 (4:1, as baseline) | 0.446 / 0.216 against 0.47 / 0.223 | 0.633 / 1.082 (4:1 unchanged; soft ratios not refit) | 6: off -2.79 / -2.31 / -1.0 (held), scale 1.371 / 1.109 / 1.05 (held); C++ of 08:30 uses -2.84 / -2.26 / -1.29 and 1.371 / 1.106 / 1.060 / 1 / 1 / 1.131 | solves the hole by fitting; 3:1 and FLOOD pairs unmeasured |
| (c) mode 4 / 5 one-way bleed + leak to rest - D, held D 2.5 tg 1 s (second-bleed) | 0.016 / 0.05 | 0.015 / 0.06 | 0.018 / 0.11 | 0.017 / 0.10 | 0 | 0.461 / 1.87 | 0.144 / 0.88 | 0.172 / 0.28 | - | 4:1 onset after rest - 0.25 pre-roll 0.13 / 0.84 / 1.63 against 0.00 / 0.01 / 0.31 | - | - | 2 | refuted (4:1 dynamics) |
| (c) mode 4 / 5 joint fit | 0.020 / 0.21 (grid) at the scan corner; joint walks D to 0.395 = removes the fade: grid 0.051 / 0.93, knee 0.169 / 0.93 | | | | 0 | 0.080 / 0.26 | 0.011 / 0.98 | 0.013 / 0.08 | 2:1 tail hangs at 0.50 dB at 2 s | | | | 2 | refuted |
| (c) modes 1 / 2 two-way bleed + leak (second-bleed) | grid 0.073 / 1.11, knee 0.251 / 1.11 | | | | 0 | 0.079 | 0.011 | 0.011 | | | | | 2 | algebraically the current node with a deeper rest; refuted |
| (c) mode 3 leak to the floor (GERMANIUM hook) | grid 0.021 / 0.26 (threshold-dependent, forbidden by the shift family) | | | | | 0.176 / 0.92 | 0.087 / 0.90 | 0.117 / 0.19 | | | | | 1 | refuted |
| (c) mode 7 one-way bleed + leak to the floor | | | | | node sits at the pre-roll level | 0.204 / 0.77 | 0.042 / 0.87 | 0.123 / 0.87 | | onset stays at 0 for 20 periods | | | 1 | refuted outright by fact 3 |
| (c) mode 8 sinking reference (second-bleed, log F, fitted) | 0.006 / 0.02 | 0.024 / 0.27 | 0.034 / 0.67 | 0.103 / 0.67 | 0 | 0.091 / 0.36 | 0.019 / 1.02 | 0.039 / 0.15 | 0.486 / 1.44 (all probes) | | | | 2 (D 2.03, tg 1.0; depth -0.89) | refuted |
| (d) mode 0 hard knee, curves in the true node domain (second-bleed, log C) | 0.024 / 0.07 | 0.110 / 1.01 | 0.072 / 1.10 | 0.248 / 1.10 | 0 | 0.080 / 0.26 | 0.010 / 0.99 | 0.011 / 0.09 | | | tone at rest - 1 reads 0 against 0.47 | | 0 | refuted: the node has no ripple at or below rest |
| GR node, law L, pass B (gr-domain, `/tmp/gr-domain-B.log`, probes out of the residual) | 0.018 / 0.064 | 0.018 / 0.078 | 0.018 / 0.136 | 0.019 / 0.102 | 0.0012 x 6 (fixed-point residue, section 3.3) | 0.081 / 0.39 | 0.010 / 1.05 | 0.005 / 0.10 | 0.065 / 0.18 (out of sample) | 0.999 / 1.67 | 0.031 / 0.079 | 0.674 / 1.210 | 0 | solves the hole; see the costs |
| **GR node, law G, pass B** (gr-domain, same log; my checks) | 0.018 / 0.064 | 0.018 / 0.078 | 0.018 / 0.136 | 0.019 / 0.103 | 0.0012 x 6 | 0.084 / 0.41 | 0.010 / 1.05 | 0.005 / 0.10 | 0.065 / 0.18 (out of sample; 3:1 predicted 5.495 / 1.041 at 0.5 / 1 s against 5.467 / 1.035) | 0.903 / 1.65 | 0.031 / 0.079 | 0.557 / 0.935 | 0 | **chosen** |
| GR node, law G, pass A (detector constants as on disk) | 0.018 / 0.064 | 0.017 / 0.077 | 0.018 / 0.136 | 0.019 / 0.101 | 0.0012 | 0.136 / 0.36 | 0.047 / 1.06 | 0.057 / 0.10 | 0.438 / 0.90 | 0.985 / 1.80 | 0.317 / 0.81 | 0.570 / 1.060 | 0 | the level-domain constants do not transfer; the refit is required |

Three things the table cannot show:

* The static columns test almost nothing on their own. Every candidate refits its curves in its own node domain inside the
  evaluation, so any mechanism whose steady-state argument is strictly increasing in level across the knee reproduces the grid
  (the V2 start with a wrong follower already had grid 0.025; the verifier's point). The discriminating evidence is entirely in
  the dynamics: facts 2 to 6.
* Direction (b) and (b') are one model (per-ratio depth d_r and a per-ratio offset o_r = depth - d_r applied to node and curve
  alike are the same equations); direction (b'') adds the bleed scale that (b) lacks. Without it the 2:1 tail runs 0.7 dB low at
  1 s and 1.0 dB low at 0.25 s (mode 6), because a level node released toward a reference 2.2 dB lower passes every level
  1.2 dB earlier and the progressive 2:1 curve turns that into 0.7-1.0 dB of gain.
* The GR node reproduces fact 2 by construction (a node that stores the gain reduction releases every ratio on one
  trajectory, 0.474 / 0.206 / 0.039 at every ratio in the mirror against 0.471-0.478 / 0.202-0.206 / 0.039 measured) and it did so
  without seeing the probes; the 3:1 tail it had never seen is reproduced to 0.03 dB. Mode 9 reproduces the same numbers by
  fitting two constants per ratio and its 3:1 and FLOOD pairs are guesses.

### 1.3 Reading

Directions (a), (c) and (d) are refuted on stated facts: (a) needs a follower that never crosses rest during any protocol
window and predicts a 10 s memory the reference does not have (fact 2, 4); every second bleed either costs the 4:1 dynamics
(modes 3, 4 / 5 held, 7, 8) or is fitted back into the current node (modes 1, 2, 4 / 5 joint); a hard knee read off the node
gives nothing below rest because the node has no ripple there.

Two forms reproduce the knee, silence and the tails, and they fit the protocol facts equally: the per-ratio level node
(mode 9: steady 0.080 / 0.26, bursts 0.010, DUAL 0.011, statics at the shift family's floor) and the node after the gain
computer (law G: steady 0.084 / 0.41, bursts 0.010, DUAL 0.005, statics at the floor). The task's rule is to prefer the physically
motivated form when two fit equally, and the GR node is that form on four counts: it is the ordinary feed-forward VCA topology
(rectifier, log, threshold and ratio network, then the timing capacitor on the control voltage, then the cell), whereas the
per-ratio form has to invent a ratio switch that moves the release reference by 2-3 dB and changes the bleed resistance by
37 % at the same time; it explains fact 2 instead of fitting it, and predicted the 3:1 tail; it needs no per-ratio constants
where the other needs twelve, three of which (3:1, FLOOD) no measurement sets; and it makes silence exact with no pin, snap or
depth. It also does slightly better on the per-ratio steady table (fact 5: 0.557 against 0.633 rms), although neither form
explains that table (section 4).

The costs of the GR node, stated plainly so that nobody has to discover them: the 30 ms onset after a quiet pre-roll (fact 3,
R3) is 0.6 dB faster than the level node at 4:1 (1.31 / 2.14 / 2.93 / 4.42 / 6.43 dB at periods 1 / 2 / 3 / 5 / 8 against 0.97 /
1.57 / 2.16 / 3.32 / 4.98 measured and 0.77 / 1.55 / 2.28 / 3.67 / 5.64 for the level node); the release after a 0 dBFS burst at
3:1 to 6:1 is 0.6-0.9 dB rms off over the first 0.3 s where the level node is 0.3 (check 2); and the +2 dBFS level cells at 10 and
30 ms attack read +0.19 and +0.41 against -0.17 for the level node (the 0.41 is the steady max). All three sit in the slow-attack
and high-level regime where both forms are already 2-3 dB wrong at the other ratios (fact 5) and which the next round has to
address either way (section 5); none touches a fact the task lists.

If the maintainers weigh the 30 ms onset and the 0 dBFS release above the structural argument, the per-ratio form already in
the tree is the fallback; section 3.5 gives what it still needs and the single measurement that decides between the two.

## 2. The chosen change: the node after the gain computer

Per sample, in the order `DiscreteStage::process` runs them. `u` is the stage input, `h` the (possibly linked) sidechain
sample, `T` the smoothed threshold, `g` the storage node in dB of gain reduction, `w` the DUAL node.

```
a    = |h|
e    = 20 log10(a)  if a > floorLin else d_floor_db          the log rectifier                                        [dB]
e   += d_goff_db[gain position]                                the make-up interaction, as now                          [dB]
x    = e - T                                                   T = d_thr_db[threshold position], smoothed as now         [dB]
gt   = max(C_ratio(x), 0)                                      target gain reduction: the ratio position's curve         [dB]
       (during a ratio fade: gt = C_prev(x) (1 - wf) + C_new(x) wf, wf the raised cosine over 20 ms)
dg   = -g kR                                                   bleed toward zero gain reduction, always on
if gt > g:                                                     the attack diode conducts
    f   = 1 + gt / d_att_sv_db                                 law G: its conductance grows with the drive               [1]
    dg += (gt - g) (1 - exp(-f / (d_tatt[attack position] fs)))   exact one-pole step, never overshoots gt
g   += dg
if recover == DUAL:  flow = (g - w) k2;  g -= flow;  w += flow / d_dual_c2
if leakRatio > 0:    g -= g kLeak                              GERMANIUM hook: extra bleed at d_trel / leakRatio
if g < 1e-7: g = 0                                             denormal guard (1e-7 dB is inaudible; silence is exactly 0)
gr   = g
ui   = u + a2 u^2 + a3 u^3;  out = ui 10^((d_gain_db[gain] - g) / 20)      cell and make-up, as now
```

with `kR = 1 - exp(-1 / (d_trel[recover] fs))`, `k2 = 1 - exp(-1 / (d_dual_t2 fs))`. At reset `g = w = 0`. Silence is zero
gain reduction at every ratio from reset and after any signal, because the bleed's target is zero and the curve is zero below
its onset. The knee of each ratio is the zero of its own curve (onsets at x = -2 / -2 / -1 / 0 / +1 / +4 dB on the 1 dB grid),
so a tone 1.9 dB below the 4:1 onset compresses at 2:1 and not at 4:1 (facts 1, 3, 4) with one shared node. The release is
`g(t) = g0 exp(-t / d_trel)` in dB of gain at every ratio (fact 2). The steady gain reduction on a sine is the node's equilibrium
under the within-cycle ripple of `gt`, which is why the curves are fitted through the node (section 3.3) and why the threshold
no longer carries a detector offset: `d_thr_db` is stage 3a's `T'` directly.

Parameters (values from `disc-knee-gr-domain.py --fit`, pass B law G, `build/disc-knee-gr-domain.json`, a joint fit stopped
at its 12-evaluation cap from the 21:47 constants, so they are the starting point of the stage-3 refit of section 3.3, not its
result; every one an existing field):

| field | unit | fitted value | note |
|---|---|---|---|
| `d_thr_db[24]` | dBFS | stage 3a's `T'` (position 16: -37.369; 2.69 dB per step) | no `d0` fold-in any more: +0.386 dB against the 21:47 values |
| `d_curve[6 x 96]` | dB of gain reduction against `e - T`, 1 dB grid from -20 | `curves.G.C` of the JSON | target curves through the node's steady state; zero below each onset |
| `d_tatt[6]` | s | 0.010e-3 (at its bound), 0.587e-3, 2.203e-3, 7.114e-3, 21.57e-3, 66.33e-3 | at zero drive; effective constant at drive `gt` is `d_tatt / (1 + gt / d_att_sv_db)` |
| `d_att_sv_db` | dB of gain reduction | 14.28 | law G scale (law L, on the level above `T - dref`, fitted 13.68 and is not chosen) |
| `d_trel[6]` | s | 0.0848, 0.1226, 0.3006, 0.3006, 0.4393, 0.3347 (DUAL first node) | |
| `d_dual_t2`, `d_dual_c2` | s, 1 | 0.0556, 10.42 | |
| `d_floor_db` | dB | -100 | unchanged |
| `d_rel_depth_db` | dB | removed | law G does not use a reference below the threshold |
| `d_ratio_off_db[6]`, `d_ratio_rel[6]` | dB, 1 | removed | the per-ratio fields of 08:30 have no meaning for a node that rests at zero |

The 0.1 ms attack constant at its 0.01 ms bound is the same finding as `disc-ratio-ballistics.py --fit-fast` (0.199 / 0.721 ms
reproduce the first-period fractions and the charge law; the priors of 08:30 already carry them); the stage-3 refit of section
3.3 should start from those and include that harness's items, or the position stays at the bound.

## 3. Exact changes

### 3.1 `src/dsp/Discrete.hpp`

1. State: replace `double v = -80.0, w = -80.0` by `double g = 0.0, w = 0.0`; `reset()` sets `g = w = 0.0` and no longer reads
   `d_rel_depth_db`, `d_ratio_off_db` or a `refFor()`; delete `refFor()`.
2. `configure()`: the DUAL switch line becomes `if (!first && nc.recover != cfg.recover) w = g;`.
3. `setTimes()`: `kR = onePoleK(c[kc_d_trel + cfg.recover], fsr)` (no `d_ratio_rel`), `kLeak` as now.
4. `process()`: replace everything from `const double thr = ...` to `gr = g;` with the sequence of section 2:

```cpp
const double thr = thrS.tick(c[kc_d_thr_db + cfg.thr]);
const double x = e - thr;
double gt = curves[cfg.ratio].at(x);
if (fadePos < fadeLen) {
    const double t = double(fadePos++) / fadeLen, wf = 0.5 - 0.5 * std::cos(kPi * t);
    gt = curves[prevRatio].at(x) * (1.0 - wf) + gt * wf;
    if (fadePos >= fadeLen && pendingRatio >= 0) { prevRatio = cfg.ratio; cfg.ratio = pendingRatio; pendingRatio = -1; fadePos = 0; }
}
if (gt < 0.0) gt = 0.0;
double dg = -g * kR;                                   // the bleed toward zero gain reduction, always on
if (gt > g) {
    const double f = 1.0 + gt * invSv;                 // law G: the diode's conductance grows with the drive
    dg += (gt - g) * (1.0 - std::exp(-rA * f));        // the exact one-pole step for a rate f / ta: never overshoots gt
}
g += dg;
if (cfg.recover == kRecovers - 1) { const double flow = (g - w) * k2; g -= flow; w += flow / c2; }
if (kLeak > 0.0) g -= g * kLeak;
if (g < 1e-7) g = 0.0;
gr = g;
```

   The hard zero `if (g < 0.0 || v <= rest)` and the snap `if (v < rest + 1e-9) v = rest` go: there is no rest point. The gain
   cell lines that follow use `g` as now (`a2Extra * (g * 0.1)`, `dbToLin(-g)`).
5. `prepare()`: `invSv = c[kc_d_att_sv_db] > 1e-3 ? 1.0 / c[kc_d_att_sv_db] : 0.0;` stays; nothing else changes.
6. The file header comment: "detector: full-wave rectifier -> log -> gain computer -> one storage node in dB of gain
   reduction (the timing capacitor on the control voltage): a bleed to zero, always conducting, and an attack diode whose
   conductance grows with the drive. The knee of each ratio is the zero of its curve; silence is zero at every ratio; the
   release is one trajectory in dB of gain at every ratio (measured, docs/disc-knee-fix.md)."

`Engine.hpp` reads `disc.grDb()` only; nothing else in `src/` touches the node.

### 3.2 `src/dsp/Calibration.hpp`

Remove three fields and rewrite three comments; the layout hash changes, so the build falls back to the priors until
`fit/make_constants.py` rewrites `FittedConstants.hpp` (the table shrinks by 13 values: 825 at the time of writing, since other
rounds have been adding fields, to 812; update the count in `docs/MODEL.md`):

```
A(d_curve, kRatios * kCurveN) /* target gain reduction (dB) against detector level above threshold, per ratio position; the
                                 node's steady state on a sine is what the static grid measures, so the curves are fitted
                                 through the node (stage 3b) */
A(d_tatt, kAttacks)         /* attack time constants at zero drive, dB-of-gain domain; the conductance grows by 1 / d_att_sv_db
                               per dB of target gain reduction, so the effective constant at drive G is d_tatt / (1 + G / d_att_sv_db) */
A(d_trel, kRecovers)        /* release (bleed) time constants toward zero gain reduction, always conducting (the Dual entry is its first node) */
S(d_att_sv_db, 14.3)        /* attack conductance law: factor 1 + (target gain reduction) / this, dB */
```

Delete `S(d_rel_depth_db, ...)`, `A(d_ratio_off_db, kRatios)`, `A(d_ratio_rel, kRatios)` and their prior lines
(`roff`, `rrel`) in `calPriors()`. Priors that change: `at[] = { 0.199e-3, 0.721e-3, 2.2e-3, 7.1e-3, 21.6e-3, 66.3e-3 }`,
`rt[] = { 0.085, 0.123, 0.301, 0.301, 0.439, 0.335 }`, `d_dual_t2` 0.056, `d_dual_c2` 10.4. `kCurveX0 = -20` and the 96-point grid
stay (the onsets lie at -2..+4; the twenty points below are zero). The prior curves in `calPriors()` (`slope[r] * softplus`)
are already zero-based and remain valid priors.

### 3.3 `fit/stages/stage3_discrete.py`

3a unchanged: `fit_static()` still returns `T'` (24) and the level-domain family; from the family build the target
`G_r(y)`, `y = L - T'_k`, as the pooled isotonic regression per ratio on the fine grid `YS` (-20..-8 by 1, -8..8 by 0.25, 8..76
by 1; `Static` in the gr-domain harness; floor 0.0046 rms over 3600 points).

3b becomes the joint fit of fifteen constants with the curves refitted inside every evaluation:

1. Mirror: `node_gr()` of the gr-domain harness replaces `detector()` (it returns the gain reduction per sample; `curve_at`
   is replaced by the PCHIP mirror `pchip_build` / `pchip_at`, since the curve now sits inside the loop and the C++ reads it
   with the monotone cubic). `d0_for()` goes: there is no detector offset; `T = T'`.
2. Curves by fixed point (`fit_curves_gr`): start from `G_r` interpolated onto the 1 dB grid; simulate the node on every `YS`
   point at the capture setting (attack 1 ms, recover 0.5 s, threshold 16, 2.5 s sines, lock-in gain over the last 0.5 s, the
   parallel `sine_grid`); correct the table by 0.8 x the residual, re-monotonise, repeat (2 iterations per evaluation from
   the previous curves, 6 at the end: the last residual is the static term of the report, 0.005 rms). Then pin the grid values
   to exactly zero below each ratio's onset (the first grid point whose target exceeds 0.03 dB): the damped fixed point leaves
   0.0012 dB there and the node would stand on it in silence.
3. Parameter vector `p[0:6]` attack, `p[6:12]` recover (the 12th the Dual first node), `p[12]` `t2`, `p[13]` `c2`, `p[14]`
   `log10(Sv)`; `unpack` returns `10 ** p[14]`. Start from the loaded calibration (first run: the priors of 3.2). Bounds
   `lo = [1e-5] * 6 + [0.01] * 6 + [0.005, 1.5, 0.3]`, `hi = [1.0] * 6 + [5.0] * 6 + [1.0, 60.0, 3.0]`, `x_scale = [1e-3] * 6 +
   [0.05] * 6 + [0.01, 2.0, 0.3]`, `loss="soft_l1"`, `f_scale=1.0`, `diff_step=1e-3`, `max_nfev=120`. The harness's pass B was
   capped at 12 evaluations (`fit: nfev 12 cost 15.449`, `/tmp/gr-domain-B.log`), so the values of section 2 are a stopped fit
   from the level-domain constants and will move under the full budget; the stage should print the final cost and nfev.
4. Residual, in this order: 3.0 x the 54 steady cells (`disc_ar_*`, `disc_al_*`, `disc_af_*`, the model's lock-in gain of a 3 s
   sine); every burst item at the stage weights (`w = (ref < GAIN12 - 0.5 ? 1 : 0.25) / sqrt(n / 100)`, `disc_burst_*`,
   `disc_dual_blen_*`, `disc_dual_pulses`, `disc_bdepth_*`, envelopes by lock-in per period, not the per-period mean gain);
   the fixed-point residual on `YS` at weight `sqrt(3600 / (6 len(YS)))`; and, new, the `disc_tail_{ratio}_{recover}` items
   of 08:28 at the burst weights (they are the tails of fact 2 and the only dynamic items at a soft ratio). Add the per-ratio
   steady table of the ballistics capture as items (`disc_ar/al/af` at every ratio) once it is in the protocol: it is the test of
   the attack law (section 5, item 1).
5. Report: the fitted constants; steady / bursts / DUAL as now, plus the tails per ratio; the rendered check through the engine
   on the 54-item set, the 198-item set and the knee items (-4 < x < 4 at t4 / 12 / 20), the standing GR at silence per ratio
   (-70 dBFS at threshold 1 and 24) and 4 s after `disc_tail_2:1_0.5 s`, all through the engine, because the stage's own print
   at the rest point never reflected the PCHIP.
6. 3c (make-up offsets): the offset is still read off the 4:1 behaviour at the capture point, but through the node: use the
   fixed point's steady GR against `x` on `YS` (`P_4:1`) and `x_needed = np.interp(grm, P_4:1, YS)`, then
   `goff[g] = x_needed - (-10 - T[15])`, `goff -= goff[11]`. 3d unchanged.
7. `save_cal` fields: drop `d_rel_depth_db`; the list is `d_thr_db, d_curve, d_tatt, d_trel, d_floor_db, d_att_sv_db, d_dual_t2,
   d_dual_c2, d_goff_db, d_a2, d_a3`.
8. Order of operations: apply 3.1 and 3.2, rebuild the C API (`hvmc_core` exposes the new layout), run
   `python3 fit/stages/stage3_discrete.py`, `python3 fit/make_constants.py`, rebuild, run `tests/pb_reference.py` and the C++
   unit tests (`tests/test_dsp.cpp` pins the discrete GR at thresholds 1 / 12 / 24: the thresholds move by +0.386 dB and the
   curves change, so those expectations must be re-read), then stage 5 (`d_inter_db`, `d_scf_trim_db`, `d_link` are measured at
   no GR and do not move). `docs/MODEL.md` section 3.2 (the Detector subsection and the per-sample listing), the field table and
   section 8 follow section 2 of this document.

### 3.4 Protocol

Keep the `disc_tail_*` items of 08:28 (their reference values are fact 2's numbers) and add: the reference's per-ratio steady
table (`ar`, `al`, `af` at 1.2:1, 2:1, 3:1, 6:1, FLOOD; the ballistics capture already holds it) and one below-onset tone at 2:1
(-39.5 dBFS at threshold 16, from silence and after a burst: 0.266 / 0.266 measured by the feedthrough harness, 0.47 at rest - 1
by the second-bleed harness). Targets for any model on the tails: 0.082 dB at 2 s and 0.001 at 8 s after the burst at 2:1 with
the 4:1 tail unchanged (0.084 / 0.001).

### 3.5 The per-ratio form now in the tree

`Discrete.hpp` and `Calibration.hpp` of 08:30 implement mode 9: `rest = T + d_ratio_off_db[ratio] - d_rel_depth_db`, the
level law on `e - rest`, the bleed constant `d_trel[recover] d_ratio_rel[ratio]`, the curve read at `v - T - d_ratio_off_db[ratio]`
with the hard zero at rest and the snap. If it is kept instead of section 2, it is not self-consistent until stage 3 is changed:

* the on-disk curves (08:25) were built with one shared rest and pinned at grid point -1; under an offset of -2.84 dB the 1.2:1
  curve's origin moves 2.84 dB down while the reference's knee is 2.17 dB below the 4:1 rest, so the engine's 1.2:1 knee sits
  about 0.6 dB too low until the curves are refit per ratio in the true node domain with `T_r = T + off_r` (the
  per-ratio harness's F2 procedure: simulate the node per static point, isotonic fit, pin at `rest_r`, refine the grid values
  through the PCHIP), and the silence knot must land on a grid point or the PCHIP reads a standing value between the pinned
  and the unpinned point (the 0.33 / 0.73 / 0.45 dB of the task's baseline);
* `off_r` and `rel_r` need a fit: bounds off [-6, 0] dB, rel [0.5, 2], `x_scale` 0.5 / 0.1, on the `disc_tail_*` items with the
  4:1 items held; the values in the header (-2.84 / -2.26 / -1.29, 1.371 / 1.106 / 1.060 / 1 / 1 / 1.131) come from mode 9's
  fit (-2.79 / -2.31, 1.371 / 1.109) plus choices no knee harness made (the 3:1 pair, FLOOD's 1.131);
* the snap must reach rest in finite time after every signal (it does, 1e-9 dB) and the standing GR must be checked through
  the engine after a burst, not from reset;
* physically the form says the RATIO switch moves the release reference and the bleed resistance together (a Thevenin
  equivalent of a switched divider); nothing measured contradicts it, and nothing measured needs it once the node sits after
  the gain computer.

The one measurement that decides: render `disc_tail_3:1_0.5 s` through each engine. The GR node predicts 5.495 / 1.041 / 0.037
dB at 0.5 / 1 / 2 s from constants that never saw a 3:1 tail; the reference reads 5.467 / 1.035 / 0.083; the per-ratio form's
prediction depends on its guessed 3:1 pair. A per-ratio steady table (fact 5) refit is the second: the GR node reaches
0.557 rms over the 324 cells with no ratio-specific constant; the level node needs a per-ratio attack scale of 2.2-2.8 at 1.2:1
and 0.4-0.7 at FLOOD on top of the rest and bleed pairs (`build/disc-ratio-ballistics/fit_leq.log`, `fit_w3.log`).

## 4. What remains unexplained (residuals)

1. **The per-ratio steady sag (fact 5).** Reference 1 -> 30 ms sag at 0.1 s: -8.35 / -8.34 / -8.23 / -7.96 / -7.75 / -7.19;
   level node -6.13 / -7.80 / -8.03 / -8.30 / -8.50 / -9.78; GR node law G -6.05 / -7.89 / -8.06 / -8.22 / -8.30 / -8.98. Steady
   table over all ratios: level 0.633 / 3.15 rms / max (1.2:1 1.082 / 2.75, FLOOD 1.027 / 3.15), GR node 0.557 / 2.69 (1.2:1
   0.935 / 2.69, FLOOD 0.934 / 2.61); at 4:1 both 0.10. In the GR node the slow-attack deficit is `Delta ~ g kR / (ka f) +
   ripple`, which is ratio-invariant only if `f` grows in proportion to `gt` with no constant term; the fitted `1 + gt / 14.3`
   is in between, and the reference's mild reverse trend (1.2:1 sags more than FLOOD) is not produced by that law either.
2. **The 30 ms onset after a quiet pre-roll (fact 3).** At 4:1, periods 1 / 2 / 3 / 5 / 8 / 12 / 20 / 40: reference 0.97 / 1.57 /
   2.16 / 3.32 / 4.98 / 7.1 / 10.76 / 17.32; level node 0.77 / 1.55 / 2.28 / 3.67 / 5.64 / 8.0 / 11.75 / 17.57 (rms 0.53); GR node
   1.31 / 2.14 / 2.93 / 4.42 / 6.43 / 8.76 / 12.43 / 17.9 (rms 1.13). Both are fast from period 5 on; the GR node also in the first
   three. The same stimulus is the protocol's `disc_burst_30.0_0.5 s` (level node: 0.9 dB more GR than the reference at
   period 510), down-weighted by the stage's burst weights.
3. **The 0 dBFS bursts.** First period after a -50 -> 0 dBFS step: 5-12 dB too much GR in both forms (detector-fix.md section
   5.1, unchanged). Release over the first 0.3 s after a 0 dBFS burst: level node 0.31 rms at 4:1, GR node 0.90 (2:1 0.41 /
   0.64, 6:1 0.44 / 0.96): the reference releases from a high GR faster than `exp(-t / d_trel)` early, which a level node with a
   progressive curve does and a GR node with a linear bleed cannot; at -10 and -30 dBFS the two are equal (0.16 / 0.17, 0.04 /
   0.05 at 4:1) and the GR node is far better at the soft ratios (1.2:1 at 1 ms / -10: 0.03 against 0.93).
4. **The +2 dBFS level cells at slow attack.** GR node +0.19 (10 ms) and +0.41 (30 ms) against -0.17 for the level node; the
   -25 dBFS cells are within 0.05 in both.
5. **The 0.05 dB plateau.** At every ratio and recover position the reference's tail ends on 0.05 dB for a few recover
   constants (1 s at 0.1 s, 4 s at 0.5 s, 4-8 s at 1.2 s) before reading 0.001; both forms pass straight through (0.000-0.04 at
   4 s, 0.000 at 8 s with 0.5 s recover). A small slow component, about 0.05 dB and scaling with the recover position, absent
   from the model.
6. **The 0.1 s recover tails at 0.25 s.** Both forms read 0.15-0.2 dB high (GR: 1.794 / 1.19 / 1.92 against 1.617 / 1.051 / 1.742
   at 2:1 / 1.2:1 / 4:1).
7. **The fixed-point residue.** 0.0012 dB of standing GR at silence in the mirror, from the damped curve fit; removed by the
   pin of 3.3 item 2, not yet rendered through an engine (no GR-node engine exists at the time of writing).
8. **Harmonics at the soft ratios.** Under the GR node the within-cycle ripple of the gain is the ripple of `gt`, i.e. the
   level ripple times the curve's local slope, so H3 under gain reduction at 1.2:1 should be lower than at 4:1 by the slope
   (about 0.58): not measured; the protocol's `disc_harm_gr_*` items are at 4:1.

## 5. What is worth a further round, in order

1. **Refit the GR node on the per-ratio steady table and bursts** (`build/disc-ratio-ballistics/measure.json`, plus the
   `disc_tail_*` items), with the attack law free between `1 + gt / Sv`, `gt / Sv` and the level form, and the 0.1 / 0.5 ms
   constants from the fast-attack finding. It is the test of whether the ratio-invariant sag of fact 5 follows from the
   diode law alone; the gr-domain harness has every function it needs (`node_gr`, `steady_cells`, `fit_curves_gr`) and my
   check scripts show how the table is read. One run, under an hour.
2. **The bleed law.** A release conductance that grows with the stored gain reduction (`kR (1 + g / Sg)`, the opto's
   bimolecular form) releases faster from a high GR and keeps `exp(-t / d_trel)` at the -10 dBFS tails: it addresses item 3 of
   section 4 and, through the steady equilibrium, part of item 1. Test on the -30 / -10 / 0 dBFS bursts at all ratios.
3. **The 30 ms onset** (item 2): with items 1 and 2 in place, the remaining first-period and period-5-to-20 errors say
   whether the attack diode needs a forward drop or a saturating conductance; every saturating law was rejected at 4:1 on the
   -10 dBFS steps (detector-fix.md), so the new data at the soft ratios is the only place a different answer could come from.
4. **The slow 0.05 dB component** (item 5): a second, much smaller storage with a several-second constant, coupled like DUAL's
   node; it is 0.05 dB and invisible in every protocol item, so it only matters if the tails are ever rendered to that precision.
5. **Harmonics per ratio** (item 8): one capture of `disc_harm_gr` at 1.2:1 and FLOOD decides between the two node positions
   independently of the ballistics; cheap and clean.
