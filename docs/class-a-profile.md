# The CLASS A profile

A fourth calibration profile for the model, derived rather than measured. It stands for the all-Class-A edition of the unit: the original Model 1 form, the 2012 Vintage King "Class A" run of fifty units (red lamps, Lundahl input transformers, "updated Class-A discrete compressor section") and the 2026 Providence Edition (silver face, "fully discrete Class A signal path compared to the standard Class A/B design", Lundahl input transformers). The standard green-lamp unit that the reference plug-in models sits between them; everything the model currently is was fitted to that unit's plug-in.

No measurement of any Class A unit exists. What exists is: the vendors' sentences about the difference, the teardown of one standard unit, the fitted constants of the standard unit's reference, the datasheets of the two input transformers, and, as a donor for the numbers the Shadow Hills sources do not give, the Pye 4060, a documented 1960s British discrete Class-A transformer-coupled compressor with a published specification and bench-measured transformers. This document says which of those each number came from, tags it, and proposes the calibration field to hold it. It follows the conventions of `docs/MODEL.md`: levels are dBFS peak of a sine, 0 dBFS = +14 dBu, and every constant is measured, assumed (here: derived, borrowed or estimated) or invented.

Confidence tags used below:

- [MEASURED]: a published measurement or datasheet figure, or a constant fitted to the reference in this repository.
- [DERIVED]: follows from a measured figure and stated physics or arithmetic.
- [BORROWED from Pye 4060]: a number taken from the Pye 4060 specification or its bench data and transplanted, with the argument for why it transfers.
- [ESTIMATE]: a physically motivated value with no measurement behind it, given with a range.

Fitted constants quoted below are from `src/dsp/FittedConstants.hpp` (the reference, Plugin Alliance / Brainworx "Shadow Hills Mastering Compressor" 1.5.1).

## 1. What is physically different in a Class-A edition of this unit

### 1.1 What the sources say, and what they do not

Every description of the Class A editions reduces to three items: "an updated Class-A discrete compressor section, Lundahl input transformers and hand wired each compressor with Mogami Cable" (Vintage King, https://vintageking.com/shadow-hills-mastering-compressor-class-a-vk-limited-edition); "upgraded Class A discrete gain cells" (Universal Audio FAQ, https://help.uaudio.com/hc/en-us/articles/40897140942996-FAQ-Shadow-Hills-Mastering-Compressor-Class-A); "an all-Class A discrete design" with Lundahl input transformers "for a smoother, more hi-fi sound", against "the standard Class A/B design" (Providence Sound and Vision, https://provsv.com/shadow-hills-mastering-compressor-providence-edition/). The audible outcome, as the only party that A/B'd two hardware units put it: "a smoother compression sound, a punchier VCA, more pronounced sonic dimension, and a larger sense of 'size' and perceived depth. With the gain set to identical values on both versions of the Shadow Hills Mastering Compressor, the Class A hardware version clocks in roughly 1-3 dB hotter, delivering a thicker and richer tone" (Plugin Alliance, https://www.plugin-alliance.com/products/mastering-compressor-class-a); "1 to 3 dB hotter for added presence, with faster transients, more depth, and a richer low end" (Universal Audio, https://www.uaudio.com/products/shadow-hills-mastering-compressor-class-a).

No source says component by component what changed: not the op-amp module topology, not the gain cell transistors, not the rails or bias, not whether the 2N3055 driver or the output transformers changed, not which Lundahl part. The output transformer trio is described identically for every edition, and the ratio, attack and recover switches are the same.

### 1.2 The standard unit's signal path, from the teardown

The torn-down standard unit (realgearonline, poster "gainstaging", 2026-09-07, https://realgearonline.com/thread/20829/shadow-hills-mastering-compressor-teardown) has, per channel:

- a Jensen JT-11 line input transformer (the JT-11P-1 family, 1:1, https://www.jensen-transformers.com/wp-content/uploads/2014/08/jt-11p-1.pdf);
- three identical plug-in discrete op-amp modules, "built from individual transistors. The small-signal positions include PN4250 devices, while the output/driver area uses MJE181/MJE182-family plastic power transistors";
- the optical attenuator (EL panel, two CdS cells) and the discrete gain cell module, "a discrete gain cell with a physical module format, and its pinout and construction point toward the older dbx VCA tradition";
- "a 2N3055 metal-can transistor. That part immediately suggests the design language of older Class-A transformer-driving stages, especially the Neve BA183 / BA283 family", switched into the Iron path only ("This position has an additional Class-A amplifier section adding even-ordered harmonic distortion", every manual);
- three custom output transformers.

MJE181 and MJE182 are a complementary NPN/PNP pair. A complementary pair in the output position of a discrete op-amp module is a push-pull output stage, which is what Providence means by "the standard Class A/B design" [DERIVED]. So the standard unit is: Class-AB op-amp modules, an AB (or lean-AB) gain cell, and one single-ended Class-A 2N3055 stage that only the Iron position uses.

### 1.3 What "all Class A" therefore changes, and what each change does to the signal

**(a) The three op-amp modules run single-ended Class A** (or Class-A push-pull; the sources do not say which, see section 5). A single-ended Class-A stage has one device carrying the whole signal around a standing current. Its transfer curve is the device's exponential law bent by degeneration and feedback: compressive on one half cycle and expansive on the other, so its dominant product is second order, proportional to level ("in class-A amplifiers second harmonic distortion is proportional to the signal amplitude while the third harmonic distortion is proportional to the square of the signal amplitude", Burosch, https://www.burosch.de/en/audio/999-distortion-measurement-in-audio-amplifiers.html; Pass Labs: "the 2nd order type declines inversely to the output voltage ... and the 3rd order type declines inversely to the square of the voltage", https://www.passlabs.com/technical_article/audio-distortion-and-feedback/). In the model's polynomial form `u + a2 u^2 + a3 u^3` that is exactly a fixed `a2`: H2/H1 = a2 A / 2, rising one dB per dB of level, and H3/H1 = |a3| A^2 / 4, rising two dB per dB. There is no crossover product, so the distortion is monotonic: it falls smoothly with level and does not rise again at low level ("Air's distortion characteristic is monotonic, which is to say its distortion products decrease smoothly as the acoustic level decreases", Pass, https://www.passlabs.com/technical_article/single-ended-class-a/). The stage runs out of current on one polarity (cut-off, at about twice the standing current) and into the rail on the other, so its ceiling is soft and asymmetric, and the bias point walks with the signal envelope through the coupling capacitors, which makes the even term itself drift slightly with level. The perceptual claims attached to bias current in the literature ("As the bias is reduced the perception of stage depth and ambiance will generally decrease", Pass, same article) are what the vendors' "larger perceived depth" most plausibly refers to.

The one such stage that has been measured in this family is the Iron position's 2N3055 stage, through the reference: Iron's driver terms are `x_a2 = 7.94e-5, x_a3 = -8.62e-4` against Nickel's `7.80e-6, -1.42e-4` and Steel's `7.38e-6, -1.36e-4` [MEASURED]. The Nickel and Steel pair is odd-dominant and symmetric (a push-pull signature); the Iron extra is ten times the even term and six times the odd term. That pair of differences, `+8.0e-5` and `-7.2e-4`, is the model's own template for "one single-ended Class-A silicon stage at this unit's operating level", and section 3 reuses it.

**(b) The gain cell is Class A.** In the Blackmer lineage this has a specific meaning. dbx's only Class-A VCA was the 2001 (Bob Adams, 1980): "Unlike earlier Blackmer cells that operated in lean class AB, the dbx2001 operated in class A. Distortion dropped to less than 0.001% but the noise and dynamic range of the dbx2001 were inferior to those of class AB circuits" (THAT Corporation, "A Brief History of VCAs", https://thatcorp.com/a-brief-history-of-vcas/). A Class-A log/antilog core keeps all its transistors conducting through the gain range, so the hand-off products of an AB core are absent and the even order is set by core matching and the symmetry trim, at the cost of a higher noise floor. The reference's cell has `d_a2 = 2.86e-3, d_a3 = +1.58e-3` (H2 -67 dBc at -10 dBFS, -57 dBc at 0 dBFS) and no crossover term at all; the torn-down standard unit's cell measured -50 dBc at about -17 dBFS (`d_hwunit_a2 = 0.0448`, sixteen times the reference, read in `docs/MODEL.md` section 6 as a trim state). So the physics says a Class-A cell is cleaner and noisier, not dirtier; the vendors' "punchier VCA" is consistent with a cell that never hands off between halves and has current to spare, and the model has nothing to remove because the reference never showed a crossover term.

**(c) The input transformer is a Lundahl instead of the Jensen JT-11P-1.** The part number is not published; Lundahl's catalogue line-input part for high levels is the LL1540 (1+1:1+1, mu-metal core and can, 5 Hz to 50 kHz within 0.2 dB from a 600 ohm source into 15 kohm, 0.5 dB loss at 1 kHz with that termination, distortion below 0.1 percent at +20 dBu and below 1 percent at +30 dBu at 50 Hz, self-resonance above 60 kHz; https://www.lundahltransformers.com/wp-content/uploads/datasheets/1540.pdf) [MEASURED, datasheet; the choice of part is an ESTIMATE]. The Jensen JT-11P-1 is 1:1, -3 dB at 0.25 Hz and 95 kHz, rated to +20 dBu at 20 Hz at 0.025 percent THD, 13 kohm input impedance with a 10 kohm load, and in Jensen's test circuit has -2.3 dB typical voltage gain at 1 kHz and -0.15 dB at 20 Hz [MEASURED, datasheet]. Two things follow. First, both parts are flat to within 0.2 dB across the audio band and both are far from saturation at mastering levels (+20 dBu is +6 dBFS here), so the linear and nonlinear differences between them are below the reference's own fit residuals; the "smoother, more hi-fi" claim has no datasheet basis and is not modelled. Second, the insertion loss differs: the Jensen loses about 2.3 dB in its published test circuit and the Lundahl about 0.5 dB in its published termination. The conditions are not identical, so the difference is somewhere between a fraction of a dB and 1.8 dB, but its sign is fixed: the Lundahl unit is hotter at the input, before either compressor [DERIVED]. This matters for where the level offset goes (section 3.2).

**(d) The Iron 2N3055 stage and the output transformers are unchanged** as far as any source says. In the Class A profile the Iron path is therefore two single-ended Class-A stages in cascade (the module and the 2N3055 driver); Nickel and Steel get one.

**(e) What does not change.** The optical attenuator (the same EL panel and CdS cells, "the same T4B optical attenuator as the LA2A and LA3A", Vintage King), its feedback loop and the fitted light law; the discrete detector, its threshold offsets, the six ratio curves, the attack and recover constants, DUAL and the link law; the sidechain filter; the transformer cores' flux ceilings, corners and shelves. None of these is a Class-A question, and there are no Class A dynamics data to fit against anyway. `docs/MODEL.md` section 6 states the rule: profiles differ only in constants, never in structure. This profile keeps that rule.

## 2. What the Pye 4060 contributes

### 2.1 What the Pye 4060 is

A mono compression amplifier made by Pye Limited, Cambridge, from about 1965, used by the BBC (Science Museum Group object co8355804, "Pye Compression amplifier type 4060, c.1965 used by the BBC", which credits it with a "unique rich, thick, gluey sound", https://collection.sciencemuseumgroup.org.uk/objects/co8355804/pye-4060-compression-amplifier), by Olympic Studios and Thames Television, and sold by dealers as "one of the first Class A solid-state compressor/limiters" (Vintage King, https://vintageking.com/pye-4060-compressor-limiter-pair-40-52-vintage). The basic compression amplifier module is the 845751 (/00 without and /01 with the noise gate board); the 4060 rack carries one. Its gain element is not a VCA: it is a mark-space (pulse-width) chopper driven by a Hartley blocking oscillator at about 250 kHz, with a diode bridge deriving the control feedback (GroupDIY, https://groupdiy.com/threads/pye-compressor-limiter-thread-boards-shipping-bom-up.43351/). That part of the Pye contributes nothing to this profile: the Shadow Hills cell is a Blackmer-type log/antilog cell and the Pye's is a switch.

### 2.2 The Pye specification [MEASURED, manufacturer's specification]

Reproduced from the Pye service manual by a seller listing (https://www.worthpoint.com/worthopedia/pye-845751-01-compressor-420397799; the manual itself is sold by Studio Electronics, https://www.studioelectronics.biz/sunshop/index.php?l=product_detail&p=1860, and the 845751/01 schematic is hosted at https://elektrotanya.com/pye_845751_01_compression_amplifier_schematic.pdf/download.html behind a captcha; neither was readable in this session):

- Gain (no compression): 0 dB, plus or minus 0.5 dB.
- Frequency response: +0.5 to -1 dB, 30 Hz to 15 kHz, in all modes; +0.2 to -0.5 dB, 60 Hz to 8 kHz.
- Output level: +24 dBm maximum. Input level: +24 dBm maximum.
- Distortion: less than 1 percent into a 600 ohm load, measured at 30 Hz, 1 kHz and 8 kHz.
- Signal to noise: greater than 60 dB.
- Input impedance 10 kohm; output impedance less than 50 ohm.
- Attack: compression below 0.5 ms, limiting 1 ms; decay 100 to 3200 ms in six steps; ratios 1:1, 2:1, 3:1, 5:1 and limiting; threshold in 2 dB steps.

In this repository's convention, +24 dBm (a sine of +24 dBu RMS) is +10 dBFS peak.

### 2.3 The Pye output stage and transformer, from bench data

From a restoration thread with the unit on the bench (GroupDIY, https://groupdiy.com/threads/pye-compressor.59067/) [MEASURED, one hobbyist, instruments not stated]:

- Output transformer 772246: turns ratio primary to secondary 1:1.08, primary to feedback winding 1:1.44; primary inductance 3.11 H at 20 Hz; DC resistance primary 2.1 to 2.2 ohm, secondary 10.6 ohm, feedback winding 56 ohm; the primary is centre-tapped; "will take +8 dBu without saturating and is almost flat from 10 Hz to 20 kHz" (source impedance for that test not stated).
- Output transistors: two, 2N3053 (BFY56 in some schematic revisions), on heat sinks, running hot; "the standing current is too high" was the restorer's reading of the as-found bias.
- Supply about 16 V (run at 15.5 V on the bench); the whole unit draws 170 mA.

Two output transistors on a centre-tapped primary with a tertiary feedback winding is a Class-A push-pull output stage with local negative feedback taken from the transformer, not a single-ended stage [DERIVED]. Pye's own house style supports the reading (Pye Limited, GB797082A, "Current feedback circuit for push-pull amplifiers", 1955, https://patents.google.com/patent/GB797082A). This is contrary to the brief's premise, and it changes what the Pye can donate: a Class-A push-pull pair cancels its even order, so the Pye output stage is not a template for the second-harmonic profile. The single-ended template is the Shadow Hills' own Iron stage (section 1.3a), which is the better donor in any case because it was measured in this unit at this level convention.

What the Pye does donate is the envelope of a discrete Class-A transformer-coupled line stage of the same era and lineage as the BA283-style stage in the Iron path:

1. **The ceiling and the distortion at the ceiling.** +24 dBm maximum with less than 1 percent THD into 600 ohm at 30 Hz, 1 kHz and 8 kHz: the stage reaches +10 dBFS with THD no worse than -40 dB, at the low end as well as the midband. The Neve BA283, the stage the teardown names for the Iron driver, is rated +26 dBu (+12 dBFS) and measured about 0.07 percent at its normal level in a Sound On Sound comparison of the transformer-coupled original against its electronically balanced replacement (https://www.soundonsound.com/reviews/neve-1073opx). Those two numbers bracket the Class-A profile's soft ceiling (section 3.4).
2. **How the ceiling relates to rail and standing current.** A Class-A stage's maximum output is set by the rail (about 16 V here) and its standing current; the unit's 170 mA and the restorer's note that the pair runs hot are the reason the stage can deliver +24 dBm at all. In the Shadow Hills the rails are unknown; the Pye and Neve figures stand in for them.
3. **The transformer corner.** 3.11 H driven from a source below 50 ohm gives a first-order corner of about 2.6 Hz (R / 2 pi L) [DERIVED], inside the range of the reference's fitted corners (Nickel 1.71 Hz, Iron 4.38 Hz, Steel 1.44 Hz). A Class-A driver into a transformer of this class does not need a different low-frequency corner from the one already fitted, which is why section 3 leaves `x_fl_hz` alone.
4. **The low-frequency saturation margin.** THD below -40 dB at +10 dBFS at 30 Hz. The reference's cores reach -40 dB THD at +3 to +4 dBFS at 20 Hz (`docs/MODEL.md` section 2), which is about +7 dBFS at 30 Hz along the 6 dB per octave flux ceiling. The Pye transformer has about 3 dB more low-frequency headroom than the reference's cores; same order, and no reason to move `x_sat_db`.
5. **Bandwidth is not copied.** The Pye's -1 dB at 15 kHz is a 1965 transformer and a 16 V stage; the reference is flat to 16 kHz and the standard unit's teardown sweep is -1.1 dB at 10 kHz. The Class A profile keeps the reference's high-frequency response and does not add the Measured Unit's 18 kHz pole, which is the only reading of "faster transients" the model can honour (section 3.6).
6. **Output impedance below 50 ohm** with a feedback winding around the transformer: the Pye's response is flat into 600 ohm, so no load-dependent corner is needed here either.

## 3. The profile, as equations in the engine's blocks

### 3.1 Placement in the engine

`kGProfile` gains a fourth value, `kProfileClassA`, after MEASURED UNIT. CLASS A builds on HARDWARE (Iron's +0.4 dB lift at 32 Hz and its -96 dBFS low-frequency floor are kept, since they are the Iron stage and transformer, unchanged in every edition) and does not include the MEASURED UNIT extras (the 18 kHz pole, the gain-reduction-dependent loss, the raised `d_hwunit_a2`): those are one standard unit's contested numbers and are not evidence about a different build.

The new fields, in the style of `Calibration.hpp`:

```
/* CLASS A profile */
S(ca_in_db, 2.5)          /* input level offset (the Lundahl input and module gain structure), before the optical stage, dB */
S(ca_out_db, 0.0)         /* remainder of the vendors' 1-3 dB, if any is found at the output, dB */
S(ca_in_fl_hz, 0.0)       /* added first-order input high pass for the Lundahl (0 = none) */
S(ca_in_lp_hz, 0.0)       /* added first-order input low pass for the Lundahl (0 = none) */
S(ca_a2, 8.0e-5)          /* one single-ended Class-A module: even term, on the module input */
S(ca_a3, -7.2e-4)         /* one single-ended Class-A module: compressive odd term */
S(ca_a2_env, 0.25)        /* even term walks with the 5 Hz envelope of the module input: a2 (1 + ca_a2_env * env) */
S(ca_ceil_db, 13.0)       /* module output ceiling, dBFS peak (+27 dBu) */
S(ca_ceil_q, 8.0)         /* ceiling knee hardness, the core's form */
S(ca_ceil_asym_db, 1.0)   /* the cut-off polarity's ceiling sits this much lower than the rail polarity's */
S(ca_d_a2_scale, 1.0)     /* gain cell even term relative to d_a2 (Class-A cell: unchanged) */
S(ca_noise_db, -96.0)     /* Class-A cell noise floor, white, dBFS RMS, at the cell output before make-up */
```

Twelve scalars. `calLayoutHash()` changes, so the fitted header must be regenerated (`python3 fit/make_constants.py`); the stages do not touch these fields and a stage-5 note records them as derived, as it does for the MEASURED UNIT constants.

### 3.2 The level offset: +2.5 dB at the input [DERIVED]

`ca_in_db = +2.5 dB`, range +1.5 to +3.0, applied to the wet path after the dry tap and before the optical stage:

```
xin = x * 10^(ca_in_db / 20)           in Engine::internal, before k.opto.process
```

Why at the input and not at the output. The vendors did not report one number; they reported a range, "1-3 dB hotter with the gain set to identical values". A fixed offset at the output would be one number. An offset at the input is seen through the static curves: below both knees the whole offset comes out; above the optical knee the output rises `o_gamma / (1 + o_gamma)` = 0.59 dB per dB (`o_gamma` about 1.34 to 1.42 depending on the fit in force, see `FittedConstants.hpp`), so +2.5 dB in becomes +1.5 dB out; with the discrete stage compressing as well, less again. An input offset of +2.5 to +3 dB produces exactly a 1 to 3 dB spread at the output across settings and levels, which is what was reported. The input transformer comparison (section 1.3c) puts part of it, a fraction of a dB to about 1.8 dB, at the input independently; the rest is module gain structure, which in a hand-built Class-A module is also ahead of the compressors. `ca_out_db = 0.0`, range 0 to +1.0, is kept for the case where a bench session finds the static family shifted along the output axis instead (section 4).

Consequence, stated plainly: at the same panel settings the CLASS A profile compresses about 2.5 dB earlier than REFERENCE and HARDWARE, which is about one position of the discrete threshold (2.7 dB per step) and one to two of the optical table. The detector laws, the 24-entry tables and the six curves are untouched; the signal reaching them is hotter. That is what "smoother compression at the same settings" most plausibly was: more of the optical stage's 2.6:1 knee in play at the settings a standard unit's owner is used to. Anyone wanting panel parity instead moves the value to `ca_out_db`.

### 3.3 The input transformer [DERIVED, datasheets]

`ca_in_fl_hz = 0.0` (none), range 0 to 1.1 Hz. The LL1540's "5 Hz to 50 kHz within 0.2 dB" places a first-order corner no higher than 5 x sqrt(10^(0.2/10) - 1) = 1.1 Hz; the reference's fitted corners (1.4 to 4.4 Hz per core) already contain whatever the Jensen contributed, so an added corner at 1.1 Hz is -0.011 dB at 20 Hz, below the fit residual. Default off; the field exists so a bench sweep can set it.

`ca_in_lp_hz = 0.0` (none), range 60 to 100 kHz. The LL1540 is flat to 50 kHz and self-resonant above 60 kHz; the JT-11P-1 is -3 dB at 95 kHz. The difference at 20 kHz is under 0.1 dB either way. Default off.

Saturation: not modelled. LL1540 is rated below 0.1 percent at +20 dBu (+6 dBFS) at 50 Hz and below 1 percent at +30 dBu (+16 dBFS); the JT-11P-1 0.025 percent at +20 dBu at 20 Hz. Both are above the output cores' knees (`x_sat_db` 5.4 to 6.0 dBFS at 20 Hz), so the output cores saturate first at every frequency, as in the standard unit. If a Class A unit is ever swept at 20 Hz and shows more low-frequency distortion than the cores account for, an input core with the existing flux form would be the place for it; no field is reserved until then.

### 3.4 The Class-A module transfer [BORROWED from the fitted Iron stage; ceiling BORROWED from Pye 4060]

One block, applied at the output of each module the signal passes:

```
classA(u):
    env  += (|u| - env) * k5Hz                                   the bias walk (the germanium block's form, kEnv)
    a2    = ca_a2 * (1 + ca_a2_env * env)
    y     = u + a2 u^2 + ca_a3 u^3                                the single-ended even and compressive odd terms
    C+    = 10^(ca_ceil_db / 20)                                  the rail polarity
    C-    = C+ * 10^(-ca_ceil_asym_db / 20)                       the cut-off polarity, 1 dB lower
    y     = y / (1 + |y / C|^q)^(1/q),   C = C+ for y > 0, C- for y < 0, q = ca_ceil_q
```

The ceiling is the `sat()` form already in `TransformerCore`, on the signal rather than the flux, with the same polarity asymmetry trick that the cores use (`x_asym`), because it is the same physics: a hard knee that is slightly harder on one side.

Values:

- `ca_a2 = 8.0e-5`, range 4e-5 to 1e-3. The fitted Iron extra (7.94e-5 minus Nickel's 7.80e-6). The upper end of the range is the teardown's lesson that the reference under-reports this unit's even order: the standard unit's cell measured sixteen times the reference's `d_a2`; if the output stages are under-reported by even a tenth of that, `ca_a2` is nearer 1e-3 (H2 -66 dBc at 0 dBFS). [BORROWED from the fitted Iron stage; range ESTIMATE]
- `ca_a3 = -7.2e-4`, range -3.6e-4 to -8.6e-4. The fitted Iron extra in the odd term. [BORROWED from the fitted Iron stage]
- `ca_a2_env = 0.25`, range 0 to 0.5. The envelope-driven bias shift of a single-ended stage (Eichas and Zoelzer's feed-forward bias term, `~/shadow/research/workflow/class-a-path.md` section 2.2b); the germanium block uses 0.5 for a much worse device. At 0 dBFS the envelope of a sine is 0.64, so the even term rises 16 percent. [ESTIMATE]
- `ca_ceil_db = +13.0 dBFS` (+27 dBu peak), range +12 to +14. Pye 4060: +24 dBm maximum at less than 1 percent THD (+10 dBFS at THD no worse than -40 dB). BA283: +26 dBu (+12 dBFS). With `q = 8`, a ceiling at +13 dBFS gives THD -43 dB at +10 dBFS, -60 dB at +6 dBFS and -31 dB at +12 dBFS on the Nickel path (computed on the polynomial plus ceiling, section 4.1); +12 dBFS would give -37.5 dB at +10 dBFS and miss the Pye bound, which is why +13 is the value. [BORROWED from Pye 4060 and the BA283 rating]
- `ca_ceil_q = 8.0`, range 4 to 12. The fitted cores' knee hardness (8.22 to 8.29) reused: the reference's own hard knees are the best evidence of how this unit's ceilings bend. Below 6 dB under the ceiling the knee is inert (0.004 dB of compression at -6 dB re ceiling), so the mid-level harmonics are the polynomial's alone. [BORROWED from the fitted cores]
- `ca_ceil_asym_db = 1.0`, range 0 to 3. The cut-off polarity of a single-ended stage clips first. The sign (which polarity) is a listening decision and a bench question (H2 phase); the model's `x_asym` convention (positive means the positive polarity is harder) is kept. [ESTIMATE]

### 3.5 Where the block is applied, per stage

**Optical stage.** The module ahead of the divider gets the even and odd terms added to the fitted amplifier terms, on the stage input where `o_b2, o_b3` already act (`o_b2 = 2.54e-5, o_b3 = -5.33e-4`, fitted):

```
b2 = o_b2 + ca_a2 * (1 + ca_a2_env * env),   b3 = o_b3 + ca_a3
xa = x + b2 x^2 + b3 x^3
```

and the ceiling is applied to the stage output after make-up (`out = classA_ceiling(v * makeup)`), which is outside the loop: the sidechain is tapped from `v`. The added even term is inside the loop's signal, but at -87 dBc it changes the divider's drive by nothing the fitted light law can see; the loop's constants are not touched. [DERIVED]

**Discrete stage.** The gain cell keeps `d_a2` (`ca_d_a2_scale = 1.0`, range 1 to 16) and `d_a3`; section 1.3b gives the reason: a Class-A Blackmer cell is cleaner, not dirtier, and the reference has no crossover term to remove. The range's upper end is `d_hwunit_a2 / d_a2` (0.0448 / 0.00286 = 15.6), the standard unit's measured state, for a listener who decides the vendors' "thicker" needs a cell-level second harmonic to be audible (see section 4.3 for why the output stages cannot supply it). The cell's Class-A noise is `ca_noise_db = -96 dBFS RMS` white, range -100 to -88, added at the cell output before make-up so it scales with DISCRETE GAIN as a cell's noise does (dbx: the Class-A 2001's "noise and dynamic range ... were inferior"; no figure is published, and the model has no discrete noise term to be relative to, so this is the same value as the HARDWARE Iron floor). [ESTIMATE] The module after the cell (the stage's output and make-up amplifier) gets the full block after make-up:

```
y = ui * 10^((d_gain_db[gain] - g) / 20) + noise
out = classA(y)
```

**Transformer driver.** Per core, the module block (its own polynomial and ceiling) follows the fitted driver polynomial in `TransformerCore::process`, before the core (as implemented; the two polynomials in cascade are the summed form to first order):

```
u = g x;  u = u + x_a2 u^2 + x_a3 u^3;  u = classA(u);  y = core(u)
```

For Iron this makes two single-ended stages in cascade (the module and the 2N3055 driver): the even terms add if the stages are non-inverting and subtract if both invert, so Iron's total even term lies anywhere from 0 to 1.6e-4 [DERIVED]; the profile adds (total 1.6e-4, H2 -82 dBc at 0 dBFS) and a bench H2 phase measurement on a Class A unit decides. The block applies to Nickel, Iron and Steel only; the four invented positions keep their own definitions (GERMANIUM already carries its own Class-A block).

### 3.6 What is deliberately not changed

- The optical loop: `o_thr_db, o_gain_db, o_gamma, o_vth, o_tau_el, o_w, o_tatt, o_trel, o_rel_mu, o_leak, o_leak_q, o_sc_lp_*`. No Class A dynamics data exist and the attenuator is the same part.
- The discrete detector and curves: `d_thr_db, d_gain_db, d_goff_db, d_curve, d_tatt, d_trel, d_dual_*, d_rel_depth_db, d_att_sv_db, d_link, sc_hz`. Same reason; the Class A editions list the same switch values.
- The cores: `x_gain_db, x_fl_hz, x_sat_db, x_q, x_asym, x_hs_*, x_lp_*`. Same transformers in every edition; the Pye's corner and headroom land on the fitted values (section 2.3).
- No 18 kHz pole, no gain-reduction-dependent loss: the profile keeps the reference's bandwidth. "Faster transients" cannot be produced by a static transfer; the one thing the model can do about it is not to add high-frequency loss, and a Class-A stage with more standing current is if anything wider, not narrower.
- No slew term, no thermal memory beyond the 5 Hz envelope: at line level neither is measurable (`class-a-path.md` section 2.1f, g).

## 4. Verification without a unit

### 4.1 Numbers the profile produces (computed, so they can be checked against anything)

Polynomial plus ceiling of the Nickel Class-A path, 1 kHz, before the core (the core adds its own products only at low frequency), against the standard Nickel driver:

| Level (dBFS peak) | Standard Nickel H2 / H3 (dBc) | CLASS A Nickel H2 / H3 (dBc) | CLASS A THD (dB) | Peak compression (dB) |
|---|---|---|---|---|
| -10 | -118 / -109 | -97 / -93 | | 0.00 |
| 0 | -108 / -89 | -87 / -73 | | -0.01 |
| +6 | -102 / -77 | -72 / -59 | -60 | -0.03 |
| +10 | -98 / -69 | -53 / -44 | -43 | -0.1 |
| +12 | -96 / -65 | | -31.5 | -0.4 |

The gain cell, unchanged: H2 -87 dBc at -30 dBFS, -67 at -10, -57 at 0, -51 at +6; H3 -68 dBc at 0 dBFS. Iron CLASS A (two stages): H2 -82 dBc, H3 -68 dBc at 0 dBFS.

The Pye bound is satisfied (THD below -40 dB at +10 dBFS, the Pye's +24 dBm), the BA283 ceiling is respected (+12 dBFS usable), and at mastering levels (-6 to +3 dBFS peaks) the Class-A stages sit 20 to 30 dB below the gain cell's own second harmonic.

### 4.2 The published descriptions the profile must satisfy, and how each maps

| Published description | Source | What in the profile answers it | Test |
|---|---|---|---|
| "1-3 dB hotter with the gain set to identical values" | Plugin Alliance, Universal Audio | `ca_in_db = +2.5` seen through the static curves gives +2.5 dB below the knees, about +1.5 dB above the optical knee, less with both stages compressing | Render the static series at threshold 12 in REFERENCE and CLASS A; the level difference must fall inside 1 to 3 dB at every point and shrink with gain reduction |
| "smoother compression sound" | Plugin Alliance | The same offset: at equal panel settings the optical knee is reached one to two positions earlier, so more of the program sits in the 2.6:1 region | Listening at matched output level; the null between profiles should be dominated by gain-reduction differences, not by harmonics |
| "punchier VCA", "faster transients" | Plugin Alliance, Universal Audio | Nothing is added that slows anything: no HF pole, no slew term; the discrete detector is untouched | Burst renders in REFERENCE and CLASS A must have identical gain trajectories once the input offset is compensated |
| "larger size and perceived depth", "more pronounced sonic dimension" | Plugin Alliance, Universal Audio | The even terms and their sign (`ca_a2`, `ca_ceil_asym_db`), per the bias-current literature; honestly the weakest link, since at 0 dBFS they are 30 dB under the cell | ABX on full mixes at matched level with `ca_a2` at 8e-5, 3e-4 and 1e-3, and with the sign flipped; if only the largest is audible, that says the reference's scale is not the hardware's |
| "richer low end" | Universal Audio | Carried from HARDWARE: Iron's +0.4 dB lift at 32 Hz, plus the level | Sweep in CLASS A Iron: +0.4 dB at 32 Hz over REFERENCE Iron, nothing else |
| "thicker and richer tone" | Plugin Alliance | The level offset and the even terms | As above |

### 4.3 What a null test says now

At the fitted values the audible content of CLASS A is the +2.5 dB and the compression shift that follows from it. The Class-A transfer terms at 8e-5 are a correction 20 to 30 dB below the gain cell's own second harmonic at mastering levels; they only surface above +6 dBFS, where the ceiling begins. This is stated here so nobody expects the profile to change the harmonic spectrum audibly at 0 dBFS: it does not, and the reference's own numbers say it should not unless the hardware's stages are far dirtier than the reference reports, which the teardown suggests is possible (section 3.4, the range of `ca_a2`).

### 4.4 The cheap intermediate reference: the Class A plug-in

The Plugin Alliance "Shadow Hills Mastering Compressor Class A" (Brainworx, 2020, "a faithful 1:1 model" of a VK unit) and its 2025 UAD port are the only Class A data that exist, second-hand as they are. If a licence is available, run `fit/measure/protocol.py` against it with TMT fixed to channel 1, Headroom at its default and the Extra Unit features off, and compare with the standard plug-in's capture. Three items are decisive and cheap:

1. **Static series at a fixed threshold, both stages, five levels.** If the Class A curve is the standard curve shifted along the input axis by a constant, the offset is at the input (`ca_in_db`); along the output axis, at the output (`ca_out_db`). A shift along both is the transformer-plus-module split.
2. **Harmonic grid per core with both stages bypassed.** Gives `ca_a2, ca_a3` directly from H2 and H3 against level on Nickel and Steel (which have no Class-A stage in the standard model), the ceiling from where THD climbs above -40 dB, and `ca_ceil_asym_db` from the H2 phase.
3. **Discrete harmonics with no gain reduction, five levels.** Gives `ca_d_a2_scale`. Silence renders give `ca_noise_db`.

Caveats: it is Brainworx's model, not hardware; TMT randomises per-channel component values and must be held; and the standard plug-in was measured on v1.5.1, so the Class A capture should use its contemporaneous release.

### 4.5 What a bench session on a Class A unit would measure

The protocol in `~/shadow/research/workflow/class-a-path.md` section 2.4, plus, specific to this profile:

- Static level at matched panel settings against a standard unit, at -30, -20, -10 and 0 dBFS with both stages in and with both out (the placement question, settled on hardware).
- The harmonic grid per core with both stages bypassed (the Class-A stage terms and the ceiling), then with the discrete stage in at no gain reduction (the cell), then the Iron Bias trimmer at as-found and two other settings (the sign and magnitude of the even term).
- A sweep with HARDWIRE in and both stages out at -30 dBFS (the input transformer's corner and any shelf against the standard unit).
- A 20 Hz level series to +16 dBFS (input transformer against output cores).
- Silence at DISCRETE GAIN 7, 12 and 24 (the cell's noise and its scaling with make-up).
- The bursts at every attack and recover position, which must match the standard unit's trajectories once the input offset is compensated; if they do not, the profile's premise that the dynamics are unchanged is wrong and this document is superseded.

## 5. What is borrowed, what is known

**Known (published or fitted):** the vendors' three-item description of the Class A editions and the "1-3 dB hotter" figure; the standard unit's parts (Jensen JT-11, three discrete modules with an MJE181/MJE182 pair, a dbx-tradition gain cell, a 2N3055 driver in the Iron path); the reference's fitted terms for every stage, including the Iron single-ended stage's even and odd terms, which this profile reuses as its template; the LL1540 and JT-11P-1 datasheets; the Pye 4060 specification (ceiling +24 dBm, THD below 1 percent at 30 Hz, 1 kHz and 8 kHz, response, impedances) and its bench-measured output transformer.

**Derived:** that the standard modules are Class AB (from the complementary pair); that the Pye output stage is Class-A push-pull (from the centre-tapped primary, two heat-sinked devices and the feedback winding), and therefore that the Pye cannot donate an even-order template; that the Lundahl unit is hotter at the input (from the insertion-loss figures, sign only); that the "1-3 dB" range is the signature of an input-side offset seen through the fitted static curves; that the added even terms are inaudible at 0 dBFS against the fitted gain cell.

**Borrowed from the Pye 4060:** the ceiling level and hardness constraint (+13 dBFS, q = 8, from "+24 dBm at less than 1 percent"), corroborated by the BA283's +26 dBu; the check that a Class-A driver into a transformer of this class keeps the fitted low-frequency corner and saturation margin. Nothing about the Pye's gain element, bandwidth or sound was used.

**Estimated:** the Lundahl part (LL1540); the envelope walk (0.25); the ceiling asymmetry (1 dB) and its sign; the split of the level offset between input and output (all at the input); the cell's noise floor (-96 dBFS RMS); the range of `ca_a2` above the fitted Iron value; that the cell's even term is unchanged.

**Not known, and not pretended:** whether the Class-A modules are single-ended or Class-A push-pull (if push-pull, `ca_a2` goes to zero and `ca_a3` stays, and the profile reduces to the level offset, the ceiling and the noise); the rails and bias currents; whether the gain cell changed at all; whether the 2N3055 stage or the transformers changed; the Class A editions' dynamics, which are assumed identical because the switches are identical and nothing else is known; and whether any of this is what the vendors heard. The whole profile is a hypothesis with its numbers stated so that one afternoon with a Class A unit, or one capture of the Class A plug-in, can falsify it field by field.

## Sources

- Vintage King, Class A VK Limited Edition: https://vintageking.com/shadow-hills-mastering-compressor-class-a-vk-limited-edition
- Plugin Alliance, Shadow Hills Mastering Compressor Class A: https://www.plugin-alliance.com/products/mastering-compressor-class-a
- Universal Audio, Shadow Hills Mastering Compressor Class A: https://www.uaudio.com/products/shadow-hills-mastering-compressor-class-a and FAQ https://help.uaudio.com/hc/en-us/articles/40897140942996-FAQ-Shadow-Hills-Mastering-Compressor-Class-A
- Providence Sound and Vision, Providence Edition: https://provsv.com/shadow-hills-mastering-compressor-providence-edition/
- Standard unit teardown: https://realgearonline.com/thread/20829/shadow-hills-mastering-compressor-teardown
- Jensen JT-11P-1 datasheet: https://www.jensen-transformers.com/wp-content/uploads/2014/08/jt-11p-1.pdf
- Lundahl LL1540 datasheet: https://www.lundahltransformers.com/wp-content/uploads/datasheets/1540.pdf
- Pye 4060 specification (from the service manual, reproduced in a listing): https://www.worthpoint.com/worthopedia/pye-845751-01-compressor-420397799
- Pye 4060 bench data (transformer 772246, output transistors, rail): https://groupdiy.com/threads/pye-compressor.59067/
- Pye 4060 circuit (PWM chopper, oscillator): https://groupdiy.com/threads/pye-compressor-limiter-thread-boards-shipping-bom-up.43351/ and https://groupdiy.com/threads/pye-845751-01-i-am-looking-for-a-service-manual-or-schematic.86931/
- Pye 4060 in the Science Museum Group collection: https://collection.sciencemuseumgroup.org.uk/objects/co8355804/pye-4060-compression-amplifier
- Pye 4060 dealer copy ("one of the first Class A solid-state compressor/limiters"): https://vintageking.com/pye-4060-compressor-limiter-pair-40-52-vintage
- Pye 845751/01 schematic (captcha-walled): https://elektrotanya.com/pye_845751_01_compression_amplifier_schematic.pdf/download.html ; service manual for sale: https://www.studioelectronics.biz/sunshop/index.php?l=product_detail&p=1860
- Pye Limited push-pull feedback patent: https://patents.google.com/patent/GB797082A
- Neve BA283 stage (single-ended 2N3055, transformer as collector load, +26 dBu): https://groupdiy.com/threads/neve-ba283-output-stage.64054/ and https://www.soundonsound.com/reviews/neve-1073opx
- Single-ended Class-A distortion physics: https://www.passlabs.com/technical_article/single-ended-class-a/ , https://www.passlabs.com/technical_article/audio-distortion-and-feedback/ , https://www.burosch.de/en/audio/999-distortion-measurement-in-audio-amplifiers.html
- Blackmer VCA lineage, Class A 2001 against Class AB 2150: https://thatcorp.com/a-brief-history-of-vcas/
- Brainworx TMT (component tolerances "from 0.1% on some parts up to 20%"): https://files.plugin-alliance.com/products/bx_console_n/bx_console_n_manual_en.pdf
- This repository: `docs/MODEL.md` sections 2, 3, 6 and 8; `src/dsp/FittedConstants.hpp`; `~/shadow/research/feature-inventory.md` sections C, D2, E and addendum; `~/shadow/research/workflow/class-a-path.md`
