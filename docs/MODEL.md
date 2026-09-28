# The model

Hidden Valley Mastering Compressor is a behavioural model of a two-stage mastering compressor of the Shadow Hills Mastering Compressor kind: an optical stage (electroluminescent panel and CdS cell) into a discrete feed-forward VCA stage, then a choice of output transformers. It is not a circuit clone. No schematic of the hardware is public, and no hardware unit was on the bench. Every constant in the model came from one of three places, and this document says which:

- **measured**: a black-box measurement of a reference (a licensed copy of a commercial plug-in that models the hardware), or a published hardware measurement;
- **assumed**: a structural choice made from circuit lineage, the hardware manual, photographs of one unit, or physics, and then fitted or left at a physically motivated value;
- **invented**: the four digital-only transformer positions, which are described in `docs/MATERIALS.md` and are not claimed to be measurements of anything.

The code is in `src/dsp/`. The fitting pipeline that produced `src/dsp/FittedConstants.hpp` is in `fit/`. The research behind the structural choices lives outside this repository.

## 1. The reference and the level convention

The measurable reference is the Plugin Alliance / Brainworx "Shadow Hills Mastering Compressor" VST3, version 1.5.1 (binary sha256 `b1f977dd38978d054e1876b41db3b37dee65403c5a84e193aec52301fcb2393c`), driven offline through Pedalboard at 48 kHz with a 1024-sample buffer and a reset before every render. Only its audio input and output were used. Nothing was read from its binary, presets or resources, and none of its audio is redistributed. The captured features (`fit/data/reference_features.json`, 5,226 items) are what the fit and the tests use, so the reference is not needed to build or test this plugin.

Levels are dBFS peak of a sine. The reference's default meter calibration puts 0 VU at -14 dBFS, so **0 dBFS = +14 dBu** everywhere in this repository, in the calibration table and in the documentation. Whether the reference's own model was calibrated to that anchor is not known; the hardware numbers quoted below are converted with it.

The reference is a model of one hardware unit, made by people who had the schematics. Where public hardware evidence disagrees with it, the difference is carried as a calibration **profile** (section 7), never mixed into the fitted constants.

## 2. The measured block diagram, and how each block was told apart

Before any fitting, a set of discriminating tests (`fit/measure/discriminate.py`, results in `fit/data/discriminate.json`) fixed the structure. Every row below was decided by a measurement of the reference unless it says otherwise.

| Question | Test | Result | Decision |
|---|---|---|---|
| Stage order | Both stages in at -10 dBFS: optical alone 10.7 dB of gain reduction, discrete alone 15.2 dB, both 16.5 dB | A series chain predicts 17.0 dB, a parallel sum 25.8 dB | Optical first, discrete second, the discrete detector sees the optically reduced signal |
| Where the optical sidechain is tapped | Fixed input, threshold 20, OPTICAL GAIN stepped 1 to 24 | Gain reduction 10.1 to 10.5 dB at every position (9.9 dB at 24) | The sidechain is taken before the make-up gain; make-up does not alter the loop |
| Optical topology | Static curves at thresholds 12, 18, 24 | Above the knee the output rises 0.36 to 0.38 dB per dB over more than 20 dB, no creep | Feedback detection from the divider output: a feedback loop with a super-linear light law gives a constant ratio, a feed-forward one over-compresses (assumed structure, consistent with the LA-2A lineage and the manual's "fixed ratio") |
| Discrete topology | Static curves at 4:1 and FLOOD | Output falls as input rises (local slopes up to 1.07 and 1.32 dB per dB) | Feed-forward, as the reference's manual also states: a feedback loop cannot exceed 1 dB per dB |
| Discrete threshold | Curves at thresholds 6, 12, 20 for two ratios | Each curve is the same shape shifted by 2.69 to 2.70 dB per step, spread 0.05 dB | The threshold is a pure offset of the detector level; one curve per ratio position, one offset per threshold position |
| Discrete ratio | Six positions, 24 thresholds, 25 levels each | Each position is a fixed, wide, progressive curve; the labels are nominal | The gain computer is a fitted table per position, not a (threshold, ratio, knee) triple |
| Does DISCRETE GAIN change gain reduction? | Fixed input, gain 1 to 24 | Gain reduction rises from 25.3 to 28.2 dB across the range | A small make-up interaction; carried as a detector offset per make-up position |
| Detector type | Sine and band-limited square at equal peak, then at equal RMS | Square at equal peak: 1.9 dB more gain reduction than the sine; at equal RMS: 1.3 dB less | Neither a peak nor an RMS detector; a rectifier into a storage network in the log domain (see the Detector subsection) |
| Steady state against attack | 1 kHz tone, attack 0.1 to 30 ms | Gain reduction falls 4 dB from the fastest to the slowest attack | The detector's storage node sits below the peak by an amount set by the attack/release ratio; a textbook dB one-pole does not do this |
| Release shape | Bursts at every RECOVER position | Exponential in dB (constant ratio per unit time), not a constant dB/s ramp | An RC in the log domain, not a current-source slew |
| DUAL | Bursts of 0.1 to 8 s, pulse trains | A fast partial release then a multi-second tail whose share depends on burst length | A second, larger storage node coupled to the first |
| Stereo link, discrete | STEREO, left -10 dBFS, right swept | 6.1 dB less gain reduction with one channel 30 dB down; anti-phase pairs do not compress | The two sidechain signals are summed before the rectifier |
| Stereo link, optical | Same test on the optical stage | Identical in STEREO and DUAL MONO | Not linked (matches an owner's report and the unpopulated link header on the photographed unit) |
| STEREO controls | Right threshold 4 with left 16 | Right channel follows the left controls, including the transformer position | Left controls govern both channels in STEREO, as the manual says |
| Sidechain filter | 50 Hz and 1 kHz tones, filter in and out | 4 to 6.5 dB less gain reduction at 50 Hz on both stages, none at 1 kHz | One first-order high pass near 90 Hz on both detectors (a second-order filter would take 10.6 dB) |
| Mix | Mix 0 % with heavy compression | Output equals the input bit for bit, not the transformer path (45 dB apart) | The plug-in's mix blends the processed signal with the untouched input |
| Hardwire bypass | OUT | A straight wire | Bit-transparent bypass |
| Transformer path, linear | Log sweeps at -30 dBFS per core | Gains -0.03 / -0.21 / +0.11 dB; first-order low-frequency corners 1.7 / 4.4 / 1.4 Hz; small high-frequency tilts | Per-core linear response: gain, one low-frequency corner, one high shelf, one low pass |
| Transformer path, nonlinear | THD against level at 20 Hz to 1 kHz | THD reaches -40 dB at +3 to +4 dBFS at 20 Hz, +9 to +10 dBFS at 40 Hz, +15 to +16 dBFS at 80 Hz, and never at 160 Hz and above (Nickel, Steel); Nickel and Steel identical at 1 kHz, Iron 16 dB more distorted with a strong second harmonic | A flux-domain core with a hard ceiling that rises 6 dB per octave, and a driver polynomial per core; Iron's extra even order is its Class-A stage |
| Sample rate | The same renders at 44.1, 48 and 96 kHz | Harmonic levels identical | Fit at 48 kHz; nothing in the reference is rate-dependent |
| Latency, determinism | Impulse position, repeated renders | 0 samples; bit-identical | Zero latency in STANDARD; deterministic |

Two facts about the hardware that the reference does not show are carried in the profiles (section 7): the Iron position's low-frequency lift and noise floor seen on hardware renders, and one torn-down unit's high-frequency roll-off and VCA second harmonic.

## 3. Signal flow

Per channel, in order:

```
in -> optical stage -> discrete stage -> transformer (Nickel / Iron / Steel / material) -> wet
out = dry + hardwire * (dry + mix * (wet - dry) - dry)
```

`dry` is the untouched input (delayed by the latency in HQ). HARDWIRE BYPASS out and MIX 0 % both return the input bit for bit. In STEREO the left channel's controls govern both channels (except METER SELECT) and the discrete detectors are linked; the optical detectors are not.

### 3.1 The optical stage (`src/dsp/Opto.hpp`)

The stage is a shunt divider (a series resistor, a CdS cell to ground) whose cell is lit by an electroluminescent panel driven at audio rate from the stage's own output. That structure is assumed from the hardware lineage (the manufacturer's own copy calls it "the same T4B optical attenuator as the LA2A and LA3A") and the photographs (two bare CdS cells under a cloth hood, an NE5532 and a complementary TO-39 pair as the panel driver, no rectifier or timing network in the area). Its behaviour is fitted to the reference.

Per sample, with `x` the stage input:

```
xa   = x + b2 x^2 + b3 x^3                      the stage amplifier's small even and odd terms
g    = 1 / (1 + c)                              the divider: c is the cell's conductance, in units of the series resistor
v    = g xa                                     the divider output (plus the material decay noise, if any)
s    = HPF(v, sc_hz) if SIDECHAIN FILTER is in, else v
s2   = LP2(s, o_sc_lp_hz, o_sc_lp_q)            the sidechain's own second-order low pass (about 5.1 kHz), magnitude-matched design
d    = | A_thr s2 |                             A_thr = 10^(o_thr_db[position]/20): the threshold is the sidechain gain
e    = d - o_vth                                the panel's turn-on: no light at all below it
L*   = e^o_n  (0 for e <= 0)                    light: the panel law (o_n is held at 1)
L    = onepole(L*, o_tau_el)                    phosphor persistence
L2   = onepole(L, tau_el2)                      a second persistence pole (material option; a pass-through otherwise)
L0   = cond0^(1/o_gamma), cond0 = o_leak (A_thr / A_20)^o_leak_q    the idle light leak, growing with the drive squared
c*   = (L2 + L0)^o_gamma                        the cell's target conductance
s_i  += (c* - s_i) * k_i                        three conductance states, i = 1..3:
       k_i = kAtt_i                                 charging (c* > s_i), a linear one-pole with o_tatt[i]
       k_i = 1 - exp(-(1 + o_rel_mu s_i) / (o_trel[i] fs))   discharging: the rate grows with the state's own conductance
c    = sum_i o_w[i] s_i                         (or the memory or half-life variants below)
out  = v * 10^(o_gain_db[position]/20)          make-up after the loop
```

Because the sidechain is taken from `v` (feedback) and the light turns on hard at `o_vth`, the static curve is a hard knee at `20 log10(o_vth) - o_thr_db` dBFS (measured under 1 dB wide) with a fixed slope of `p / (1 + p)` dB per dB above it, `p = o_n o_gamma` (with `o_n` held at 1, `o_gamma / (1 + o_gamma)`; measured 0.62, a ratio near 2.6:1, constant to +14 dBFS). The threshold switch is a pure shift of that knee: the 24-entry drive table is fitted one position at a time and lays every position's curve onto every other's (slope multiplier 0.998 across positions 3 to 24). The idle leak is what makes the no-compression gain fall slightly as the threshold is raised (0.45 dB at position 24); it is identical at -70 and -50 dBFS on the reference, so it is a bias inside the loop, not a compression floor. The self-quenched release (`ds/dt = -(s + mu s^2) / trel`, bimolecular recombination in the cell) is what lets one set of constants recover in 6 to 16 ms while the signal stays above the knee and still leave a sub-second tail once the light goes out; the reference has no long tail above the knee at all. The sidechain low pass is what the 8 kHz static series and the gain reduction against frequency measure: the audio path is flat to 16 kHz, the detector is not. Each of these was chosen by residual among four independent hypothesis harnesses (`fit/tools/candidates/opto-*.py`, synthesised in `docs/opto-fix.md`); a feed-forward computer with opto-style smoothing was refuted by the release identity (the release from -10 dBFS toward -20 and toward -50 dBFS coincide for five cycles and part exactly when the divider output crosses the knee).

The panel responds to the instantaneous drive on both half cycles, so the fastest state carries a ripple at twice the signal frequency; that ripple modulates the gain and is where the stage's odd harmonics under gain reduction come from. This is why the detector runs at audio rate and why the model's distortion under gain reduction is fitted only through the loop's dynamics, never as a separate term.

Gain reduction, as the meter reads it, is `20 log10((1 + c) / (1 + cond0))`: zero at idle whatever the threshold position.

**Options.**

- *Light memory* (`opto_memory`, off by default): a fourth, slow state `mem` with the attack of the second state and a release `o_mem_trel / (1 + o_mem_mu * expo)`, where `expo` tracks the recent exposure with time constant `o_mem_tm`; `c = (1 - o_mem_beta) fast + o_mem_beta mem`. The reference has no light memory (its release after a 0.1 s burst and after an 8 s burst are identical to 0.01 dB), so this is off for reference parity. The hardware manual describes a two-stage release whose slow part "takes over a second, depending on the amount of attenuation applied", which is what this option gives. Its four constants are priors, not fitted.
- *Half-life release* (URANIUM only): a three-member decay chain on a 20 % share of the fast conductance, described in `docs/MATERIALS.md`. In equilibrium every member's activity equals its source, so the static value is unchanged and only the release is shaped.
- *Measured Unit profile*: a first-order low pass at `o_hw_lp_hz` on the divider output, and a second one whose corner is `o_hw_grloss_hz / (1 + o_hw_grloss_k * c)`, so it falls as the cell conducts (section 7).
- *Material hooks*: a light gain multiplying `L*` (GOLD's window), `tau_el2` (URANIUM's glass), a drive ceiling (unused by any shipped position), and the noise added at the divider node (URANIUM's decay events).

The threshold and make-up switches are smoothed over 15 ms. The sidechain high pass runs whether or not the filter is in, so switching it is click-free.

### 3.2 The discrete stage (`src/dsp/Discrete.hpp`)

A feed-forward compressor around a discrete gain cell. The photographs show a plug-in card with about twelve unmarked small-signal transistors, several glued together for thermal tracking, beside an NE5532 and the timing components; the lineage (a Blackmer-type log/antilog cell in the dbx tradition) is the most likely reading but is not proven, and the model does not depend on it. What the model depends on was measured (section 2): feed-forward operation, the threshold as a pure offset, a fixed curve per ratio position, the make-up interaction, the link law, and the stage's own distortion.

Per sample, with `u` the stage input and `h` the (possibly linked) sidechain signal:

```
h    = HPF90(u) if SIDECHAIN FILTER is in, else u        (the engine sums h over both channels in STEREO)
e    = 20 log10 |h| + d_goff_db[gain position]           the log rectifier (floored at d_floor_db, -100 dB)
v    = detector(e)                                       the storage node, in dB (see Detector)
x    = v - d_thr_db[threshold position]                  level above threshold
g    = C_ratio(x)                                        gain reduction in dB from the ratio position's curve, never below 0
ui   = u + a2 u^2 + a3 u^3                               the gain cell's own even and odd terms, on its input
out  = ui * 10^((d_gain_db[gain position] - g) / 20)
```

**Ratio curves.** `d_curve` holds six curves of gain reduction against level above threshold, on a 1 dB grid from -20 to +75 dB (96 points each). The engine evaluates them with a Fritsch-Carlson monotone cubic (PCHIP), so the curves never wiggle between points and extrapolate linearly above the grid. When the ratio switch moves, the old and new curves are crossfaded over 20 ms.

**Threshold.** Twenty-four offsets, `d_thr_db`, about 2.7 dB apart (position 1 at about +3 dBFS, position 24 at about -60 dBFS of detector level).

**Make-up.** `d_gain_db` is the measured gain of each DISCRETE GAIN position; `d_goff_db` is the detector offset that comes with each position, which reproduces the small rise of gain reduction with make-up seen in the reference (the offset is zero at position 12, where the static grid was captured).

**Gain cell.** `d_a2` and `d_a3` act on the cell's input, before the gain, so the stage's distortion does not fall with gain reduction. The reference's second harmonic rises one dB per dB of level (-87 dBc at -30 dBFS, -67 dBc at -10 dBFS, -51 dBc at +6 dBFS) and its third at two dB per dB, which is what a fixed polynomial on the input gives. Under gain reduction the reference also grows odd harmonics; those come from the detector's ripple, not from the cell, and the model reproduces them the same way.

**Stereo.** In STEREO the engine replaces both channels' sidechain signals by `d_link * (h_L + h_R)` before the rectifier. With `d_link = 0.5` that is the mean, which reproduces the measured 6.1 dB drop with one channel 30 dB down and the absence of compression on anti-phase pairs.

#### Detector

The detector is a full-wave rectifier followed by a log converter and one storage node in dB. The node is a capacitor with a bleed resistor to the threshold reference, always conducting, and an attack diode from the log level whose conductance grows with the level above that reference:

```
rest = T - d_rel_depth_db                                  T = d_thr_db[threshold position]
dv   = (rest - v) * kR                                     bleed, always on; kR from d_trel[recover position]
if e > v:
    f   = 1 + max(e - rest, 0) / d_att_sv_db               attack conductance factor
    dv += (e - v) * (1 - exp(-f / (d_tatt[attack position] * fs)))
v   += dv
```

Two things distinguish it from a textbook branching detector. The bleed does not switch off while the diode conducts: on a sine at a slow attack the diode conducts for most of each cycle, and a detector that pauses its discharge then under-counts it by about four times, which is exactly the 1.25 dB error the earlier switched-branch form had at 30 ms / 0.1 s. And the attack conductance is not fixed: a signal 17 dB above the reference charges the node twice as fast as one at the reference, which is what reproduces the mild level dependence of the steady gain reduction. This form was chosen by residual among some sixty candidate topologies fitted in four independent harnesses (`fit/tools/candidates/`, synthesised in `docs/detector-fix.md`): power-domain and RMS detectors, decoupled and cascaded smoothers, pre- and post-filters, saturating and slew-limited attacks, half-wave rectification and two-path releases were all run on the same steady-state and burst data and rejected on stated residuals. The always-on bleed was found by all four harnesses independently.

On a steady sine the node settles below the peak by an amount set by the charge per cycle against the bleed. That is why the steady gain reduction falls as the attack is made slower, why the attack time constant measured through a level step depends on the recover position (`1 / (kA + kR)`, which no switched-branch detector produces), and why the static curves are captured at one attack and recover setting (1 ms, 0.5 s): the curve family is defined at the node level, and the detector offset at that setting, `d0`, is recomputed from the fitted detector and folded into the thresholds (`T = T' + d0`). At rest (silence) only the bleed acts and the node sits exactly at `rest`, so there is no start-up transient.

**DUAL.** The last RECOVER position couples a second, larger storage node `w` to the first through a resistor: `flow = (v - w) * k2; v -= flow; w += flow / c2`, with `k2` from `d_dual_t2` (the coupling time constant) and `c2 = d_dual_c2` (the second capacitor relative to the first). Short bursts leave the second node nearly empty, so the release is fast; long bursts fill it, so the release becomes a multi-second tail. The topology is a fit (a two-node network reproduces the reference's DUAL release to about 0.05 dB), not an identification of the circuit.

**Leak.** A material hook (GERMANIUM): a third branch that discharges the node toward the log floor with a time constant of `d_trel / leakRatio`, off for every stock position.

**Fitting the detector (stage 3b).** A numba mirror of the C++ detector renders the node trajectory for the steady-tone matrix (attack × recover at -10 dBFS, plus level and frequency checks) and for every burst item, the gain trajectory `C_ratio(v(t) - T)` is compared with the reference's per-period gain envelope, and the six attack constants, six recover constants, the DUAL pair, the release depth and the conductance law's scale are fitted jointly with `scipy.optimize.least_squares` (`loss="soft_l1"`), steady-state points weighted 3 to 1 against burst samples. The calibration fields this sets are `d_tatt`, `d_trel`, `d_dual_t2`, `d_dual_c2`, `d_rel_depth_db`, `d_att_sv_db`, and it fixes `d_floor_db` at -100 dB. Because the node offset varies slowly with level above the release reference, the static curves are resampled onto the node domain after the detector fit so that the static family stays exact.

### 3.3 The transformer path (`src/dsp/Transformer.hpp`)

One parametric core, one parameter set per position:

```
x -> gain -> driver stage (x + a2 x^2 + a3 x^3) -> magnetic core -> high shelf -> low pass -> [profile and material extras] -> out
```

**The core** is a flux-domain model: the flux is the leaky integral of the drive, the core saturates the flux with a hard ceiling, and the output is the rate of change of the saturated flux (integrate, saturate, differentiate):

```
phi   += T * (v - r * phi)                       forward (explicit) Euler leaky integrator; r = 2 pi x_fl_hz sets the low-frequency corner
S(phi) = phi / (1 + |phi / phi_k|^q)^(1/q)        a hard ceiling at +/- phi_k with knee hardness q
y      = (S(phi) - S(phi_prev)) * fs
```

With `S` linear this is exactly a first-order high pass at `x_fl_hz`, which is the magnetising inductance against the source resistance. Driven hard at low frequency the flux cannot exceed `phi_k`, so the output is limited to `omega * phi_k`, a ceiling that rises 6 dB per octave: saturation at 20 Hz sets in 6 dB below where it does at 40 Hz, which is what the reference measures (section 2) and what a memoryless waveshaper cannot do. `x_sat_db` is the level, in dBFS peak, of a 20 Hz sine whose flux peak is the knee flux (`phi_k = 10^(x_sat_db/20) / (2 pi 20)`). The two polarities have slightly different hardness, `q (1 +/- x_asym)`: the reference's even harmonics peak at the onset of saturation and fade as the drive rises, which a knee asymmetry does and a flux offset does not. The core's third harmonic comes out in anti-phase with the driver stage's, which is why the reference's third harmonic dips where the two cross.

**The driver stage** is a polynomial with an even term `x_a2` and a compressive odd term `x_a3` (negative). Nickel and Steel share one nonlinearity to within 0.1 dB in the reference; Iron's is about ten times larger with a much stronger even term, which is its extra Class-A stage (every manual and the manufacturer's own copy put that stage in the Iron path alone; the model treats it as part of Iron's parameter set rather than a separate block).

**The linear tilt** per core is the gain `x_gain_db`, the corner `x_fl_hz`, a first-order high shelf (`x_hs_hz`, `x_hs_db`) and a first-order low pass (`x_lp_hz`, zero for none). All filters are bilinear with prewarping (TPT), so the responses hold at any sample rate.

**Switching.** A position change configures a second core, warms it up on the last 80 ms of input, and crossfades to it over 20 ms, so automation is click-free. A change that arrives during a fade is applied when the fade ends.

The invented positions (GOLD, URANIUM, GERMANIUM, PLUTONIUM) reuse this block with their own parameter sets and a few extra elements (a second shelf, a peaking section, a noise source, a Class-A variant, a winding thermal element); see `docs/MATERIALS.md`.

### 3.4 Bypasses, mix, switching

The stage bypass switches, the hardwire bypass and the mix are crossfaded over 10 ms; the make-up and threshold switches are smoothed over 15 ms; the ratio switch crossfades its curve over 20 ms; the transformer switch is described above. At the first buffer after `prepare` every smoother starts on its target, so a render from a fresh instance is bit-identical to one from an instance that has been sitting on the same settings, and the bypasses are exact.

### 3.5 Meters (`src/dsp/Meters.hpp`)

The plugin has no GUI. It publishes read-only output parameters: per channel the meter as METER SELECT shows it (optical or discrete gain reduction in dB, negative, or the output VU), the two gain reductions on their own, and the magic eye. The VU is the full-wave rectified output through a second-order low pass (natural frequency 2.05 Hz, damping 0.81, which reaches 99 % of a step in about 300 ms with about 1 % overshoot, as IEC 60268-17 asks), scaled so a steady sine reads its RMS level, with 0 VU at a sine of -14 dBFS peak. The gain-reduction meters are the stages' own values through the same needle. The magic eye is the mono output's peak, 1 ms attack, 300 ms release, in dBFS. The hardware's optical meter is a second CdS cell under the same panel; the model reads the audio cell.

## 4. The calibration table

Every fitted constant lives in one flat table of doubles (`src/dsp/Calibration.hpp`, the `HVMC_CAL_FIELDS` macro), 793 values in all. The fitting pipeline reads this layout through the C interface (`src/capi/hvmc_capi.cpp`), fits the values, and writes `src/dsp/FittedConstants.hpp`, which `Calibration.hpp` includes. The header carries an FNV-1a hash of the field names and sizes; if the header was written for another layout, or is missing, the plugin falls back to the priors in `calPriors()` and the unit tests report it. A release build must have the fitted table.

| Field | Size | Meaning | Units | Set by |
|---|---|---|---|---|
| `sc_hz` | 1 | Sidechain high-pass corner, first order, both detectors | Hz | stage 5a |
| `o_thr_db` | 24 | Sidechain drive gain per OPTICAL THRESHOLD position; the knee is at 20 log10(o_vth) minus this | dB | stage 4b (seven positions), 4c (all 24) |
| `o_gain_db` | 24 | Make-up gain per OPTICAL GAIN position | dB | stage 1 |
| `o_n` | 1 | Panel light law exponent above the turn-on, light ~ (drive - o_vth)^n (held at 1) | | stage 4b |
| `o_gamma` | 1 | Cell law exponent, conductance ~ light^gamma; static slope gamma/(1+gamma) | | stage 4b |
| `o_vth` | 1 | Panel turn-on, in drive units: a hard knee | | stage 4b |
| `o_tau_el` | 1 | Phosphor persistence | s | stage 4b |
| `o_w` | 3 | Weights of the cell's three conductance states (sum 1) | | stage 4b |
| `o_tatt`, `o_trel` | 3 each | Attack time constant of each state; release time constant at zero conductance | s | stage 4b |
| `o_rel_mu` | 1 | Release rate multiplier per unit conductance (bimolecular recombination) | | stage 4b |
| `o_leak` | 1 | Idle light leak as a conductance at threshold position 20 | | stage 4b |
| `o_leak_q` | 1 | Growth of the leak with the sidechain drive, cond0 = o_leak (A/A20)^q | | stage 4b |
| `o_sc_lp_hz`, `o_sc_lp_q` | 1 each | Sidechain second-order low pass corner and Q (optical stage only) | Hz, 1 | stage 4d |
| `o_b2`, `o_b3` | 1 each | Stage amplifier even and odd terms, on the stage input | | stage 4c |
| `o_mem_beta`, `o_mem_trel`, `o_mem_mu`, `o_mem_tm` | 1 each | Light-memory option: slow share, its base release, its lengthening at full exposure, the exposure time constant | , s, , s | priors (the option is not in the reference) |
| `o_hw_lp_hz` | 1 | Measured Unit: first-order low pass in the optical path | Hz | stage 5c, from the teardown sweep |
| `o_hw_grloss_k`, `o_hw_grloss_hz` | 1 each | Measured Unit: high-frequency loss that grows with gain reduction, corner `hz / (1 + k c)` | , Hz | stage 5c, from one owner's data |
| `d_thr_db` | 24 | Threshold per DISCRETE THRESHOLD position, dBFS of the detector level | dB | stage 3a (shift family) and 3b (offset `d0`) |
| `d_gain_db` | 24 | Make-up gain per DISCRETE GAIN position | dB | stage 1 |
| `d_goff_db` | 24 | Detector offset that comes with each make-up position | dB | stage 3c |
| `d_curve` | 6 × 96 | Gain reduction against level above threshold, per ratio position, -20 to +75 dB in 1 dB steps | dB | stage 3a, resampled in 3b |
| `d_tatt` | 6 | Attack time constants at the release reference level, log domain (the conductance grows by 1 / `d_att_sv_db` per dB above it) | s | stage 3b (Detector) |
| `d_trel` | 6 | Release (bleed) time constants, log domain, always conducting; the DUAL entry is its first node | s | stage 3b (Detector) |
| `d_dual_t2`, `d_dual_c2` | 1 each | DUAL: coupling time constant, second capacitor relative to the first | s, | stage 3b (Detector) |
| `d_floor_db` | 1 | Log rectifier floor | dB | fixed at -100 in stage 3b |
| `d_rel_depth_db` | 1 | Release reference below the threshold (the node's rest point) | dB | stage 3b (Detector) |
| `d_att_sv_db` | 1 | Attack conductance law: factor 1 + (level above the release reference) / this | dB | stage 3b (Detector) |
| `d_inter_db` | 1 | Gain between the stages when both are in (the reference is this much above the sum of the two stages' laws) | dB | stage 5 |
| `d_scf_trim_db` | 1 | SIDECHAIN FILTER in: trim on the discrete stage's input, audio and sidechain alike | dB | stage 5 |
| `d_link` | 1 | STEREO: detector input = `d_link` (left + right) | | stage 5b |
| `d_a2`, `d_a3` | 1 each | Gain cell even and odd terms, on the cell input | | stage 3d |
| `d_hwunit_a2` | 1 | Measured Unit: the torn-down unit's even-order term | | stage 5c, from the teardown FFT |
| `x_gain_db` | 3 | Midband gain, Nickel / Iron / Steel | dB | stage 2, linear pass |
| `x_a2`, `x_a3` | 3 each | Driver stage even and odd terms (`x_a3` negative: compressive) | | stage 2, nonlinear pass |
| `x_fl_hz` | 3 | Low-frequency corner (magnetising inductance against source resistance) | Hz | stage 2, linear pass |
| `x_sat_db` | 3 | Core saturation: the level of a 20 Hz sine whose flux peak is the knee flux | dBFS | stage 2, nonlinear pass |
| `x_q` | 3 | Knee hardness | | stage 2, nonlinear pass |
| `x_asym` | 3 | Knee-hardness asymmetry between the polarities | | stage 2, nonlinear pass |
| `x_hs_hz`, `x_hs_db` | 3 each | High shelf corner and gain | Hz, dB | stage 2, linear pass |
| `x_lp_hz` | 3 | First-order low pass (0 = none) | Hz | stage 2, linear pass |
| `ca_in_db`, `ca_out_db` | 1 each | CLASS A: input and output level offsets | dB | derived (docs/class-a-profile.md) |
| `ca_in_fl_hz`, `ca_in_lp_hz` | 1 each | CLASS A: input transformer corners (0 = none) | Hz | derived |
| `ca_a2`, `ca_a3`, `ca_a2_env` | 1 each | CLASS A: module even and odd terms, envelope walk of the even term | | derived |
| `ca_ceil_db`, `ca_ceil_q`, `ca_ceil_asym_db` | 1 each | CLASS A: module ceiling, knee hardness, polarity asymmetry | dBFS, 1, dB | derived |
| `ca_d_a2_scale`, `ca_noise_db` | 1 each | CLASS A: cell even-term scale, cell noise floor | 1, dBFS | derived |
| `hw_iron_lift_db`, `hw_iron_lift_hz`, `hw_iron_lift_q` | 1 each | Hardware profile: Iron's low-frequency lift, a peaking section | dB, Hz, | priors from hardware renders (stage 5c writes them unchanged) |
| `hw_iron_noise_db` | 1 | Hardware profile: Iron's low-frequency noise floor | dBFS RMS | prior from the teardown FFT (stage 5c writes it unchanged) |

Values are in `fit/data/constants.json` (the fit's output, with a note per stage recording what data it used) and, identically, in `src/dsp/FittedConstants.hpp`. The generated header lists any field still at its prior.

## 5. The fitting pipeline

### 5.1 Measurement protocol

`fit/measure/protocol.py` defines 5,226 items, each a stimulus, a control setting and a feature extractor, shared by the capture of the reference and by the fit and tests of the model. Stimuli are steady sines, bursts and pulse trains (all on whole periods), an exponential sweep with a silent tail, and a band-limited square. Features are: the steady gain of the fundamental by lock-in over the last half or full second; the gain per period from the start of a render (the burst envelopes); the harmonics H2 to H8 in dBc with the phase of H2 and H3 relative to the fundamental; and the small-signal magnitude response at 28 frequencies by deconvolution of the sweep. The groups: switch laws; optical statics (24 thresholds × 33 levels at 1 kHz, plus 100 Hz, 3 kHz and 8 kHz series), optical bursts (four depths, four lengths, a pulse train), optical harmonics under gain reduction; discrete statics (6 ratios × 24 thresholds × 25 levels), the steady-state attack × recover matrix, level and frequency checks, bursts at every attack and recover position, DUAL burst-length and pulse series, discrete harmonics with and without gain reduction, the make-up interaction; the stereo link (left fixed, right swept, in phase, anti-phase, 1 kHz against 1.1 kHz) for both stages; the sidechain filter (nine frequencies, in and out, two levels) for both stages; the transformer harmonic grid (10 frequencies × 13 levels per core) and sweeps at two levels; the stage responses and the stages' own harmonics; sample-rate replicas at 44.1 and 96 kHz; both stages together; and mix. `fit/measure/capture.py` runs every item through the reference (64 s in all) and writes the features with the reference's version and hash.

### 5.2 The stages

Each stage is a script in `fit/stages/`; each writes only its own fields into `fit/data/constants.json` (`save_cal(..., fields=...)`), so the stages can be run in any order or one at a time. Renders of the model go through `fit/hvmc_core.py`, a ctypes bridge to the same `Engine` the plugin runs, built as a shared library by `scripts/build-capi.sh`; the model that is fitted is therefore the code that ships, not a Python re-implementation.

**Stage 1, the switch laws** (`stage1_laws.py`). The make-up gain of every OPTICAL GAIN and DISCRETE GAIN position is the measured gain of the stage at that position (threshold 1, a level far below compression) minus the gain of the Nickel path with both stages out. Closed form, no fitting. The result is the hardware manual's "zoomed taper", which a three-point pilot had missed: both switches have three segments, about 4.5 to 5 dB per step from position 1 to 3, about 0.46 to 0.49 dB per step from 3 to 13, and about 1.4 to 1.5 dB per step from 13 to 24; unity falls at optical position 11 (+0.11 dB) and discrete position 7 (+0.05 dB), which is where the manual puts it.

**Stage 2, the transformer paths** (`stage2_transformers.py`). Per core, two passes. The linear pass fits gain, low-frequency corner, high shelf and low pass to the -30 dBFS sweep response in closed form: `digital_response()` is the exact response of the discrete filters the C++ runs (a forward (explicit) Euler leaky integrator, TPT shelf and low pass), so the fit is analytic and needs no renders. Results: Nickel -0.029 dB, 1.71 Hz, a -0.20 dB shelf from 9.5 kHz, a 71.9 kHz pole; Iron -0.215 dB, 4.38 Hz, -0.38 dB from 12.2 kHz, 71.7 kHz; Steel +0.108 dB, 1.44 Hz, -0.67 dB from 14.2 kHz, a 24.4 kHz pole. The residuals over the 28 sweep frequencies are 0.0045, 0.0014 and 0.0052 dB rms (0.014, 0.005 and 0.015 dB at worst). The nonlinear pass then fits `x_a2`, `x_a3`, `x_sat_db`, `x_q` and `x_asym` by rendering the 130-point harmonic grid through the engine, with the fundamental's gain weighted 5 to 1 against H2 and H3 and the higher harmonics weighted less (`fit/common.py`, `feat_residual`), bounded `least_squares` with the soft-L1 loss. Results: knee levels 5.40, 5.42 and 5.98 dBFS at 20 Hz, hardness 8.3, 8.2 and 8.2, asymmetry 0.024, 0.030 and 0.032; driver terms 7.8e-6 / -1.42e-4 (Nickel), 7.9e-5 / -8.62e-4 (Iron), 7.4e-6 / -1.36e-4 (Steel). The weighted harmonic residual the stage prints (4.8, 5.0 and 5.0 dB rms) is over H2 to H8 at every grid point, most of it in harmonics near the -120 dBc capture floor; the fundamental and the third harmonic match far more closely than that figure suggests. Two structural choices were made from these residuals: a core whose magnetising current grows with flux (a resistor-inductor core) put the third harmonic in quadrature with the driver's and could not produce the reference's dip, and a flux offset made the even harmonics grow with drive when the reference's fade, which is why the model has a hard ceiling with a knee asymmetry.

**Stage 3, the discrete stage** (`stage3_discrete.py`). Four parts. *3a, the static family*: for every ratio `r`, threshold `k` and level `L` the measured gain reduction is `GR(r, k, L)`; the model says `GR = C_r(L - T'_k)`, one curve per ratio and one shift per threshold. It is fitted by alternating projections: pool-adjacent-violators (isotonic) regression gives each ratio's monotone curve on the shifted data, then a one-dimensional search gives each threshold position's shift; six rounds. The printed residual is the test of the shift model itself. *3b, the detector*: described under Detector. *3c, the make-up interaction*: the detector offset per DISCRETE GAIN position, read off the 4:1 curve from the measured gain reduction at each position. *3d, the gain cell*: `d_a2` and `d_a3` from the no-gain-reduction harmonic series at five levels, rendered through the whole model so the Nickel path is accounted for.

**Stage 4, the optical stage** (`stage4_opto.py`). The stage is a feedback loop with no closed form, so it is fitted by rendering it; a numba mirror of `OptoStage::process` (checked against the C++ engine at the start of every run, to 0.001 dB on statics and 0.005 dB on envelopes) makes the joint fit tractable, and the engine writes the final report. *4a*: `o_b2` and `o_b3` from the stage's harmonics with no gain reduction. *4b*: jointly, one knee level per fitted threshold position (6, 10, 14, 18, 20, 22, 24), the turn-on, the cell exponent, the persistence, the three states, the leak law and the release quench, on those positions' 1 kHz static series (weight 2), the nine burst envelopes and thirteen harmonic items at 1 kHz, 100 Hz and 4 kHz (gain weight 2, H3 weight 1, H5 weight 0.3); two starts, `soft_l1`, then a linear-loss refinement. *4c*: with the shape fixed, one knee per position on its 33-level series; position 1 compresses nothing in range and is placed 8 dB below position 2. *4d*: the sidechain low pass corner and Q on the 3 kHz and 8 kHz static series, rendered through the engine so that the Nickel path's own HF shelf is in the loop as in the plugin. The stage then reports, through the engine, the per-position statics, the no-compression gain law, the bursts, all 27 harmonic items, the sample-rate replicas and the discriminating captures (fine knee, no-GR rows at -70 and -50 dBFS, above-knee steps) from `fit/data/discriminate_opto.json`.

**Stage 5, routing and profiles** (`stage5_routing.py`). *5a*: `sc_hz` from the discrete stage's gain reduction against frequency with the filter in, rendered, one scalar; the filter-out items and the optical items are reported and must not move. *5b*: `d_link` from the stereo link sweep on the discrete stage; the anti-phase and 1 kHz / 1.1 kHz pairs and the optical link items are reported. *5c*: the Measured Unit constants, which are not fitted to the reference but taken from the hardware evidence and written with a note so the header records their provenance: an 18 kHz first-order pole (the torn-down unit's -1.1 dB at 10 kHz and -3.8 dB at 20 kHz bracket 16.9 to 18.6 kHz), the corner and slope of the gain-reduction-dependent loss solved from one owner's two points (-0.25 dB at 10 kHz at no gain reduction, -1.5 dB at 3 dB), and `d_hwunit_a2` for a second harmonic of -50 dBc at the teardown's level (about -3 dBu, -17 dBFS).

### 5.3 Running it

```
scripts/build-capi.sh                 # the engine as a shared library, for the Python side
python3 fit/run_all.py                # all five stages in order, logs in build/fit/, then the header
python3 fit/run_all.py --from 3       # stages 3, 4, 5, then the header
python3 fit/run_all.py --only 2       # one stage, then the header
python3 fit/run_all.py --capture      # re-measure the reference first (needs the licensed reference installed)
python3 fit/make_constants.py         # write src/dsp/FittedConstants.hpp from fit/data/constants.json
python3 fit/make_constants.py --check # exit 1 if the header on disk is not what constants.json produces
```

The fit is deterministic (fixed initial points, bounds and iteration limits, no randomness), so `--check` is the regression test that the committed header and the committed constants agree; a stage that changes a constant must be followed by regenerating the header. Stages 3 and 4 render the engine thousands of times and take hours; stages 1, 2 and 5 take minutes.

## 6. Profiles

The `profile` switch chooses which truth each block follows. The dynamics (gain computers, ballistics, link, filter, control laws) come from the reference in every profile, because no hardware dynamics data exist. The profiles differ only in constants and in which optional blocks are switched in, never in the fitted structure, and each addition is listed here with its source.

| Profile | What it adds | Source | Confidence |
|---|---|---|---|
| REFERENCE | Nothing: every block as fitted to the reference | the capture | the fit residuals |
| HARDWARE (default) | Iron: a +0.4 dB peaking lift at 32 Hz, Q 0.55 (`hw_iron_*`) | hardware renders of a real unit through a converter, where Iron alone rises about +0.36 to +0.38 dB at 20 to 40 Hz while Nickel and Steel are flat | one unit, program material, the converter included |
| | Iron: a low-frequency noise floor at -96 dBFS RMS, shaped by a 60 Hz one-pole (`hw_iron_noise_db`) | the teardown's spectra, where Iron's floor rises smoothly to about 200 uV at 20 Hz, about 20 dB above Nickel's, with no mains structure: the signature of an extra biased stage | one unit |
| MEASURED UNIT | Everything in HARDWARE, plus: a first-order 18 kHz pole in the optical path (`o_hw_lp_hz`) | the teardown's response sweep (-1.1 dB at 10 kHz, -3.8 dB at 20 kHz, identical on all three cores, so upstream of the transformers; placed in the optical path because users of another model localised the same roll-off there) | one unit, and contested: other owners report -1 dB at 16 kHz |
| | an optical high-frequency loss that grows with gain reduction (`o_hw_grloss_*`) | one owner's data: -0.25 dB at 10 kHz at no gain reduction, -1.5 dB at 3 dB | one owner, two points |
| | the discrete cell's even-order term raised to the torn-down unit's -50 dBc (`d_hwunit_a2`) | the teardown's FFT at about -3 dBu; the two channels of that unit differ (0.34 % against 0.24 %), which reads as a trim, so it is not the default | one unit, likely a trim state |

| CLASS A | Everything in HARDWARE (not the MEASURED UNIT extras), plus a derived all-Class-A signal path (`ca_*`, `src/dsp/ClassA.hpp`): +2.5 dB at the input ahead of both compressors (the vendors' "1 to 3 dB hotter" seen through the fitted static curves), a single-ended module block (even term 8.0e-5 walking with the 5 Hz envelope, odd term -7.2e-4, a soft ceiling at +13 dBFS with the cut-off polarity 1 dB lower) on the optical amplifier and after the optical make-up, after the discrete make-up, and on the transformer driver of the three hardware cores; the discrete cell keeps its terms and gains a -96 dBFS white noise floor | `docs/class-a-profile.md`: the module terms borrowed from the fitted Iron stage (the one measured single-ended Class-A stage in the family), the ceiling from the Pye 4060's published maximum output and the BA283 rating, the offset derived from the reported range; the Class A edition is the original Model 1 form of the unit and the Providence edition its current form | no measurement of any Class A unit exists: derived and borrowed, labelled as such in the document |

The Iron additions apply to the Iron position only; the optical and discrete additions apply in every transformer position; the CLASS A driver block applies to the three hardware cores only (the material positions keep their own definitions).

## 7. Sample rate, latency, HQ

Everything is designed in continuous time and discretised at the running rate: time constants through `1 - exp(-1 / (tau fs))`, the filters by the bilinear transform with prewarping, the core by a backward-Euler integrator and a first difference at `fs`. The reference itself is rate-invariant (section 2), and the protocol carries 44.1 and 96 kHz replicas of the burst items to check that the model is too.

**STANDARD** runs at the host rate with zero latency, like the reference. Nothing looks ahead and nothing is oversampled; the nonlinearities are mild enough at mastering levels that the reference does not oversample either, and its aliasing was not copied because the fit is to rate-invariant features.

**HQ 2X** runs the whole channel (both stages and the transformer) at twice the host rate between two half-band filters (`src/dsp/Oversampler.hpp`): a linear-phase 79-tap half-band FIR designed with the Parks-McClellan algorithm (`fit/tools/halfband.py`; passband to 20 kHz and stopband from 28 kHz at a 96 kHz internal rate, 0.00007 dB ripple, -108 dB stopband), with the half-band structure imposed exactly, so only 40 taps are computed per rate change. Each filter delays by 39 internal samples; the round trip is 39 samples at the host rate, which the plugin reports as its latency and by which it delays the dry path so that MIX and HARDWIRE BYPASS stay aligned. Switching quality restarts the engine, and the host is told the new latency.

## 8. What is measured, what is assumed, what is invented

**Measured** (from the reference, or from published hardware data where the profiles say so): the stage order; the optical sidechain tap, static curves, threshold law, make-up law, attack and release trajectories, distortion under gain reduction; the discrete feed-forward structure, the threshold as an offset, the six ratio curves, the make-up law and its interaction, the attack and recover constants, DUAL, the link law, the cell's distortion; the sidechain filter's order and corner; the mix and bypass laws; the per-core linear response and harmonic grid; zero latency; rate invariance.

**Assumed**: that the optical stage is a shunt divider with feedback detection and a three-state cell with a quenched release (the feed-forward alternative was refuted by the release identity, section 3.1; the leak, the hard turn-on and the quench are phenomenological forms chosen by residual, not traced components); that the discrete detector is a rectifier into an RC network in the log domain with an always-conducting bleed to the threshold reference and a level-dependent attack conductance (chosen by residual among some sixty candidates in four harnesses; see Detector); that the gain cell's distortion is a fixed polynomial on its input; that the transformer core is a flux-domain saturator with a knee asymmetry (chosen by residual over a resistor-inductor core and a flux offset); that Iron's extra even order is a Class-A stage in that path alone (documented, not traced); the Measured Unit constants' placement (the 18 kHz pole in the optical path rather than on the input transformer's secondary); the light-memory option's constants; that 0 dBFS = +14 dBu is the anchor the reference's model was built to.

**Invented**: GOLD, URANIUM, GERMANIUM and PLUTONIUM, the TEMPERATURE dial and the EXHIBITION switch, all of which are described with their physical basis and every departure from it in `docs/MATERIALS.md`. Also plugin conveniences the hardware lacks: MIX, HQ, the profiles, and the read-only meters.

**Not modelled**: the input transformer (linear at mastering levels; its 20 Hz saturation starts near +19 dBu), the line-amplifier modules' clipping (rails unknown), an output clipper (the reference's hard limit at 40 Hz is the flux ceiling and needs no separate clipper), unit-to-unit spread, an optical link option, a load-dependent low-frequency corner, and the hardware's magic-eye and meter-cell physics.
