# The material positions

The transformer switch has three positions the hardware has, NICKEL, IRON and STEEL, and four it does not: GOLD, URANIUM, GERMANIUM and PLUTONIUM. The four are **invented**. They are not measurements of any hardware, no such transformer exists, and no one has heard one. What they are is an honest answer to the question "what would this circuit do if it were made of that material", worked out from published material science and then rendered through the same model as the real positions. Where a real effect is too small to hear, the position says so, ships it at its true scale, and lets the EXHIBITION switch raise it to an audible, documented level.

This document covers the physics each position rests on, every constant in `src/dsp/Materials.hpp` and the material branches of `Engine::apply()` (`src/dsp/Engine.hpp`) and `coreFor()` (`src/dsp/Transformer.hpp`), and a ledger of what is physical scale and what is exhibition scale. The research behind it is a set of reports on each material, a ferromagnetism primer and a design synthesis with its adversarial critique, which live outside this repository; the sources those reports cite are named here by author or title.

Every constant carries a tag in the code and here: **[P]** physical, a published or derived number used as is; **[E]** exhibition, a real but sub-audible effect scaled up, with the true value stated; **[D]** design, a listenability choice that physics bounds but does not fix; **[INFERRED]** and **[ESTIMATE]** as in the research.

## 1. The shared fact, and the shared controls

**None of the four elements can be a transformer core.** Gold and germanium are diamagnetic, uranium and plutonium are weak paramagnets; all four have a relative permeability within a thousandth of one (from the CRC Handbook's molar susceptibilities). On the same winding as an 80 % nickel core, each is an air core with its low-frequency corner about 15 octaves higher, tens to hundreds of kilohertz: no audio at all. So a material position is not "a core made of X". It is an assignment of the material to the blocks where it really acts, and the core, where a core is needed, is a real compound that contains the material (Au4Mn for gold, a UFe10Si2-class silicide for uranium) or a plain nickel core (plutonium) or an ideal one (germanium).

| Position | Core | Windings | Junctions | Light path | Heat and radiation |
|---|---|---|---|---|---|
| GOLD | Au4Mn, a real 22.4-karat ferromagnet | gold | stock | 10 nm gold-flash window | none |
| URANIUM | UFe10Si2-class, a real room-temperature magnet | uranium | stock | uranium-glass window | alpha decay, fission fragments, winding heat |
| GERMANIUM | ideal, linear | stock | all 1960s germanium | stock | die temperature |
| PLUTONIUM | the real nickel core | delta-phase plutonium-gallium, plus an eddy slug | stock | stock | winding heat |

**TEMPERATURE** (`temperature_c`, 15 to 120 C, default 25) is the room temperature. It drives GOLD's core (its saturation and its Curie point) and GERMANIUM's junctions (leakage, die temperature, noise). The studio range is 15 to 45 C; above that the dial is a range extension, real material laws at an operating condition no studio reaches. It has no effect on NICKEL, IRON, STEEL, URANIUM or PLUTONIUM.

**EXHIBITION** (`exhibition`, off by default) raises the sub-audible effects to their documented exhibition scale: URANIUM's winding heat, its decay noise and its fission rate, and PLUTONIUM's winding bloom and heat. Off, every one of those runs at its true, physical size, which for the radiation effects is silence. This is the honesty contract: an exhibition knob's zero is the truth.

The positions plug into the model at these points (`docs/MODEL.md` describes the blocks): the transformer core's parameter set (`CoreParams`), a second shelf, a peaking section and a noise source in the transformer block, a winding thermal element after the core, a Class-A variant of the driver stage, a light gain and a second persistence pole in the optical stage, a noise input at the optical divider node, the half-life release, and in the discrete stage a detector leak and an extra even-order term in the gain cell.

## 2. GOLD

### 2.1 Physics

Pure gold is diamagnetic (volume susceptibility -3.45e-5): a solid gold core is an air core, perfectly linear, no colour. But there is a real gold ferromagnet: **ordered Au4Mn**, 80 atomic percent gold, 93.5 % by mass, 22.4 karat (He, Gercsi and others, Physical Review B 106, 214414, 2022, arXiv 2210.14069). It orders at a Curie temperature of 385 K (about 112 C), carries 4.1 Bohr magnetons per formula unit at 4 K and 2.8 at room temperature (a saturation flux of about 0.39 T at 20 C, half of an 80 % nickel core's 0.78 T), and is strongly uniaxial (K1 about 100 kJ/m3 at room temperature). The authors fit its magnetisation against temperature with mean-field theory for spin S = 2, which is what the code uses.

Three consequences are physical and audible or borderline audible:

- **Early saturation.** Half the saturation flux is 6.0 dB less headroom than Nickel at the same geometry.
- **Thermal drift.** At 20 to 35 C the core runs at 0.76 to 0.80 of its Curie temperature, so its saturation falls 0.66 dB over that span and 1.19 dB by 45 C; nickel-family cores move less than 0.06 dB. Past 112 C it stops being a magnet.
- **A soft, rotation-dominated knee.** Below its anisotropy field the magnetisation of a uniaxial polycrystal is mostly reversible rotation, which approaches saturation gradually. This is physically motivated, not measured; Au4Mn's coercivity and initial permeability are unpublished.

One consequence is catastrophic and must be scaled: Au4Mn's permeability is far too low to pass bass (rotation alone gives a relative permeability near 1.4; the domain-wall part is unmeasured). Shipped as is, the core would roll off somewhere between hundreds of hertz and tens of kilohertz. The model keeps its low-frequency corner at 3 Hz, a documented exhibition factor of a hundred to ten thousand.

Two gold effects outside the core: gold windings have 1.30 to 1.32 times copper's resistance (about -0.3 dB into 600 ohms, -0.02 dB into a bridging load, no tone); and gold's reflectance sits on its 5d band edge exactly where a ZnS:Cu panel emits and a CdS cell listens (0.41 at 460 nm, 0.68 at 528 nm, 0.79 at 550 nm, Johnson and Christy's optical constants), so gold in the light path is a real mechanism. A gilded hood would flip sign against the hardware's black cloth hood, so the model uses the one gold optic whose sign does not depend on the unknown hood interior: a **10 nm gold-flash window**, a near-neutral filter passing about 70 % of the light (-1.5 dB), computed from the thin-film Fresnel equations with those constants.

The audio-industry "gold sound" (warm, silky, expensive) is folklore in every controlled test the research found; nothing here claims to deliver it. What the position delivers is a core that saturates 6 dB early with the softest knee on the switch and drifts with the room, and an optical stage seeing 1.5 dB less light.

### 2.2 As implemented

`coreFor()` starts from the fitted Nickel set and changes:

| Element | Value | Tag |
|---|---|---|
| Knee level | Nickel's `x_sat_db` + `goldKneeDb(T)`, where `goldKneeDb = -6.0 dB + 20 log10(m(T) / m(20 C))` and `m` is the mean-field Brillouin magnetisation for S = 2 and Tc = 385 K (`mat::brillouinM`, solved by fixed-point iteration when the dial moves, never per sample) | [P]: `kGoldKneeDbAt20C` -6.0, `kGoldTc` 385, `kGoldS` 2 |
| Knee hardness | Nickel's `x_q` × 0.5 (`kGoldQScale`): the softest knee on the switch | [D], physically motivated |
| Low-frequency corner | 3.0 Hz (`kGoldFlHz`) | [E]: physical 0.8 to 55 kHz |
| Midband gain | -0.06 dB (`kGoldGainDb`): the gold winding into a bridging load | [P] |
| Driver stage, shelf, low pass, asymmetry | Nickel's | |
| Optical light gain | 0.708 (`kGoldLightDb` -1.5 dB applied as a power ratio, `dbToLin(2 × -1.5)`) multiplying the panel's light before the persistence pole | [P] |

At 25 C the knee sits 6.2 dB below Nickel's; at 35 C 6.7 dB; at 45 C 7.2 dB; at 60 C 8.2 dB; at 80 C 10.1 dB; at 90 C 11.7 dB. Above 90 C the mean-field curve is increasingly wrong (its critical exponent is 0.5 where real magnets show about 0.36), so the last twenty degrees are model-dependent.

**GOLD above its Curie point.** When the dial is at or above 111.85 C (T ≥ 385 K), `coreFor()` switches to a linear core: no saturation, low-frequency corner 6 Hz, gain -0.3 dB, no low pass, a second shelf of -0.8 dB from 10 kHz. That is the "24 k air core": above Tc the alloy is a paramagnet and the transformer has no core colour at all. Physically, a real transformer passing its Curie point would also lose its bass and 6 to 11 dB of level; those are kept at exhibition scale (6 Hz and -0.3 dB), and the alloy is a Curie-Weiss paramagnet, not diamagnetic gold, so the label is a relabel of an honest morph. The gold window stays in the light path.

Not implemented, and listed in the design as optional: a gold-doped driver (a hypothetical device; gold in silicon is a lifetime killer), a leaky gold-doped detector diode, a treble-desensitised sidechain (its citation was refuted), and any session warm-up: the temperature is a dial, not a clock.

## 3. URANIUM

### 3.1 Physics

Uranium metal is a Pauli paramagnet (relative permeability 1.0004; the 1952 susceptibility measurements found no ferromagnetism from 20 to 350 C). A uranium core is an air core. Room-temperature uranium-bearing ferromagnets exist, but the iron carries the moment: the best documented is **UFe10Si2** (ThMn12 structure, Curie temperature 640 ± 10 K, 19.5 Bohr magnetons per formula unit, about 28 % uranium by mass, with a measurable uranium contribution to the anisotropy; Berlureau and others, JMMM 102, 1991, and related work). Its room-temperature saturation is estimated at 1.1 to 1.3 T, between Nickel's 0.78 T and Steel's 2.0 T: a knee 3.0 to 4.4 dB above Nickel's. Its lattice constants, coercivity and permeability are not in the sources, so its loop is a design choice; its temperature drift (0.07 to 0.12 dB over 20 to 35 C) is inaudible and not modelled.

**Uranium windings** are real: uranium is ductile and rolled to foil, its resistivity at 300 K is 28.6 microohm centimetres, 17 times copper, with a temperature coefficient of +2.55e-3 per kelvin (the Los Alamos and OSTI handbook data). A uranium winding heats 26.5 times faster than copper at the same geometry and current, and its resistance rises as it heats: thermal compression with memory, real in sign and rate. Its size at line level is tiny: the research's own check puts the heating at 12 mK at +4 dBu and 0.8 K at +18 dBu (0.008 dB of compression), so the "heavy, hot" uranium is exhibition in magnitude.

**Radioactivity** is electrically real and inaudible. Depleted uranium's specific activity is 14.8 kBq/g (IAEA). About 63 alpha particles per second leave each square centimetre of a thick slab, about 2,400 per second from a 300 g piece, each ionising about 75,000 ion pairs in air (12 fC) over an ion transit of 1 to 7 ms. Collected at a high-impedance node that is low-frequency rumble at about -125 dBu, 25 to 60 dB below hearing (-139 dBFS at the divider node in the model's units). Spontaneous fission of U-238 (about 6.8 per second per kilogram) sends about 7 fragments per hour out of a 40 cm2 surface, each about 15 times an alpha's charge. There is no "radioactive distortion": radiation adds no harmonics and no energy to the audio, and the model does not pretend otherwise.

**Uranium glass** (uranyl in borosilicate) glows green at 526 nm under the panel's light with a fluorescence lifetime of about 370 microseconds (NIST SRM 2941 and related work). That is 24 to 190 times faster than the CdS cell's own response, so it adds no audible time constant; it is a sub-millisecond smoothing pole on the light and the best visual cue the position has.

**The half-life release.** The U-238 chain (U-238 → Th-234, 24.1 days → Pa-234m, 1.17 minutes → U-234, 245,000 years) has a real topology and a real order of half-lives, and in secular equilibrium every member's activity equals the source. The ratios span seventeen orders of magnitude, so no mapping preserves them; the model maps the time axis with a power law, `t_map = 50 ms × (t_real / 70.2 s)^0.161`, chosen so Pa-234m lands at 50 ms and U-234 at 3.0 s, which puts Th-234 at 0.26 s. What survives is the chain's topology, the order of its half-lives, secular equilibrium, and the fact that activity, not atom count, is what the gain follows. What does not survive is the ratios; the design calls the result decay-chain-inspired, and so does this document. The research also found that a faithful two-member chain gives a nearly single-exponential release, which is why the three-member chain with U-234 as the exposure memory is the one used.

### 3.2 As implemented

`coreFor()` starts from the fitted Steel set and changes:

| Element | Value | Tag |
|---|---|---|
| Knee level | **Nickel's** `x_sat_db` + 3.7 dB (`kUraniumKneeDb`) | [E-estimate]: 1.2 T against Nickel's 0.78 T |
| Knee hardness | Steel's `x_q` × 0.6 (`kUraniumQScale`): softer than Steel, for a high-anisotropy core with lower initial permeability | [D] |
| Low-frequency corner | Steel's `x_fl_hz` × 2 (`kUraniumFlScale`) | [INFERRED] |
| Gain, driver terms, shelf, low pass, asymmetry | Steel's | |
| Winding thermal element | resistance change -0.01 dB per kelvin (`kUraniumHeatDbPerK`, from 17× copper into 600 ohms with the +2.55e-3/K coefficient); the winding warms by `heatKPerFs2 × mean-square output`, with 3 s heating and 8 s cooling time constants (`kUraniumHeatTau`, `kUraniumCoolTau`) | [P] slope; [D] time constants |
| `heatKPerFs2`, true scale | 0.64 K per unit mean-square, i.e. 0.8 K for a +4 dBFS (+18 dBu) sine, the research's physical estimate | [P] |
| `heatKPerFs2`, exhibition | 20 K per unit mean-square (`kUraniumHeatK`): 10 K for a 0 dBFS sine, -0.1 dB; the design's default detent | [E] |

The winding element (`TransformerCore::process`) tracks the output power through a 0.3 s one-pole, drives the winding temperature toward `heatKPerFs2 × power` with the two time constants, and scales the output by `10^(heatDbPerK × heat / 20)`. The static insertion loss of the uranium winding (-5.3 dB in the research's example) is not applied; the make-up would remove it.

**Optical stage** (`Engine::apply`): `tauEl2 = 370 us` (`kUraniumTauEl2`), a second persistence pole after the phosphor's; and `halfLife = true`, which replaces the plain cell conductance with:

```
src = 0.20 × fast                     the slow share (kUraniumSlowShare; hardware two-stage release 15 to 25 %)
n1 += src - n1 kTh                    Th-234, mapped half-life 0.26 s
n2 += n1 kTh - n2 kPa                 Pa-234m, 0.050 s
n3 += n2 kPa - n3 kU                  U-234, 3.0 s
c   = 0.80 × fast + (n1 kTh + n2 kPa + n3 kU) / 3
```

where each `k` is the per-sample decay fraction `1 - exp(-ln 2 / (t_half fs))`. In equilibrium each member's activity equals `src`, so the static gain reduction is unchanged; on release the chain drains in order, and long, loud passages leave a longer tail because U-234 fills only under long exposure. The half-lives are the `Materials.hpp` constants `kThHalf`, `kPaHalf` and `kU234Half`, read in `OptoStage::prepare`; the slow share is `kUraniumSlowShare`.

**Decay noise** (`DecayNoise` in `Engine.hpp`), injected as a current at the optical divider node, the highest-impedance node in the hardware's opto path:

| Element | True scale | Exhibition |
|---|---|---|
| Alpha rate | 2,400 events per second (`kAlphaRate`), Poisson, sample-accurate | same |
| Alpha level at the node | -139 dBFS RMS (`kAlphaLevelDb`, -125 dBu) | +45 dB (`kAlphaExhibitionDb`): -94 dBFS |
| Charge per event | flat distribution in [0, 1] of the calibrated amplitude, random sign (the residual range of an alpha leaving a thick slab is flat) | same |
| Kernel | two cascaded one-poles at 60 Hz (ion transit 1 to 7 ms, 25 to 150 Hz); the amplitude is calibrated so the filtered train has the stated RMS (Campbell's theorem) | same |
| Fission fragments | 7 per hour (`kFissionRate`); a fragment's amplitude is `kFissionScale` (15) times half the alpha scale, drawn between half and full of that, so about ten to fifteen times the mean alpha charge | 0.5 per second (`kFissionExhibitionRate`) |

At true scale the generator is on and inaudible by design: the honest default of a radiation effect is silence. The generator is deterministic (a counter-based splitmix64 seeded per channel), so renders repeat.

Not implemented: the "age since casting" law (Th-234 ingrowth over 24 days scales only inaudible terms), the radioluminescent light floor, the UO2 oxide-contact element (no audio-band data exist), natural against depleted activity, and the cryogenic sub-mode.

## 4. GERMANIUM

### 4.1 Physics

Germanium is the one material of the four whose circuit role is history rather than thought experiment: it was the transistor and diode material from about 1948 to 1968, so every junction in this compressor has a documented germanium counterpart. Its magnetic role is null (diamagnetic; an air core), so the germanium position has the cleanest transformer on the switch, an ideal linear core, and all of its character comes from the semiconductors.

The traits belong to the era's alloy-junction devices more than to the element (germanium's carrier mobility is higher than silicon's):

- **Leakage** is microamps, not nanoamps (the 2N404 datasheet: 5 uA at 25 C, 90 uA at 80 C), and doubles roughly every 8 to 10 C. In a Class-A stage the collector current is `beta × I_B + (beta + 1) × I_CBO`, so the operating point walks with temperature; hot devices "gate" or "choke" as one polarity's headroom collapses. This is the Fuzz Face behaviour the guitar literature documents at length.
- **Low Early voltage** (8 to 20 V measured on OC44 and AC128 devices, Holmes, Holters and van Walstijn, DAFx-17, against 50 to 100 V for silicon) makes the stage's gain vary with the collector swing: an estimated 5 to 10 times more second harmonic at equal swing, growing with output level.
- **Slow power devices**: the germanium TO-3 power transistors that would replace the 2N3055 (2N176, 2N669) have a current-gain corner of 3 to 7 kHz against the 2N3055's 10 kHz minimum and about 50 kHz typical, so loop gain falls through the top octaves, distortion rises with frequency and the top softens.
- **Polarity**: germanium power devices were PNP, which mirrors the even-order asymmetry.
- **Maximum junction temperature** 75 to 100 C against 150 to 200 C for silicon: "hot" is a real failure edge.
- **Detector diodes** (1N34A class) leak 3 to 10 uA at 1 V, of the same order as, or more than, the release current of a detector with megohm release resistors and a microfarad capacitor. A literal swap would collapse the release; a period designer would have re-scaled the impedances, and so does the model.
- **Gain-cell mismatch**: unselected germanium pairs differ by an estimated 2 to 28 mV of base-emitter voltage where a monolithic silicon cell is trimmed to within 0.5 to 2.5 mV (THAT 2181 datasheet), so the cell's even-order residual grows with gain reduction.
- The famous soft germanium knee does **not** survive this topology: the discrete detector is a precision rectifier, which divides the diode drop by its loop gain. The model does not fake it.

### 4.2 As implemented

**Transformer block** (`coreFor()`, starting from the fitted Iron set): a linear core (no saturation), low-frequency corner 2 Hz, a first-order low pass at 20 kHz, no first shelf, a second shelf of -1.0 dB from 10 kHz (`kGeHfShelfDb`), gain as Iron's.

**Class-A stage** (`TransformerCore::classA`, in the Iron position's place):

| Element | Value | Tag |
|---|---|---|
| Loop-gain low pass ahead of the shaper | one pole at 5 kHz (`kGeLoopLpHz`) | [P]: the 3 to 7 kHz current-gain corner |
| Even-order term | -6 × Iron's `x_a2` (`kGeH2Scale`, the sign mirrored for PNP), multiplied by `(1 + 0.5 × envelope)` (a 5 Hz envelope follower: the Early-effect term grows with swing) and by `(1 + 0.05 × (leak - 1))` (leakage-driven bias walk) | [E-estimate] scale; [INFERRED] laws |
| Odd-order term | Iron's `x_a3` | |
| Die temperature | `T_die` follows `room + 15 C` (`kGeDieRiseC`, the quiescent rise) with a 120 s time constant (`kGeThermalTau`) | [D] rise; [P-estimate] time constant |
| Leakage factor | `leak = 2^((T_die - 25) / 9)` (`kGeLeakDoubleC`) | [P] |
| Choke | when `T_die` exceeds 75 C (`kGeChokeStartC`) the negative half-cycle's ceiling falls from +12 dBFS toward -18 dBFS at 100 C (`kGeChokeFullC`), through a `tanh` soft limit: the one-sided gating of a hot germanium stage. With a 15 C quiescent rise this needs a room above 60 C, in the dial's extension range | [P] limits; [D] shape |
| Noise | -100 dBFS RMS (`kGeNoiseDb`) at 25 C, +1.5 dB per 9 C of room temperature (the shot noise of the leakage), shaped by a 300 Hz one-pole (`kGeNoiseLpHz`) standing in for 1/f | [P-estimate] |

**Discrete stage** (`Engine::apply` and `DiscreteStage`): a detector leak toward the log floor with a conductance `leakRatio` times the release conductance, `leakRatio = 0.3 × 2^((room - 25) / 9)` (`kGeLeakRatio25`), so about 1.4 at 45 C; the research's physical ratio with the hardware's impedances is 2.4 to 43, which would collapse the release, so 0.3 is the re-scaled design value. And a gain-cell mismatch: the cell's even-order term rises by `4 × d_a2` per 10 dB of gain reduction (`kGeVcaMismatchA2`).

Not implemented: a per-unit seed for the device spread, control-voltage feedthrough, the op-amp modules' lower headroom, a true 1/f noise shape (the shipped noise is white below a 300 Hz one-pole corner, `kGeNoiseLpHz`, and falls 6 dB per octave above it), the germanium panel driver's swing limit (`kGeElCeilingDb` is defined but not applied), self-heating from the program (the die rise is a constant), and the passive-detector topology swap that would give the Chandler-style soft knee.

## 5. PLUTONIUM

### 5.1 Physics

Scope: only the public handbook electrical, magnetic, optical, thermal and ordinary-decay properties of plutonium metal and its dioxide, as used for sound design. Nothing else about the material is modelled, implied or discussed.

Plutonium is a paramagnet with no magnetic order at all: "no evidence for ordered or disordered magnetic moments, static or dynamic, in either phase" (Lashley, Lawson, McQueeney and Lander, Physical Review B 72, 2005), and no Curie point. Its relative permeability is about 1.0006. So the forbidden metal is the most magnetically silent core of the five: no saturation, no hysteresis, no Barkhausen grit, and no audio either, since a plutonium core is an air core. That irony is the position's design: the metal is kept out of the core, which stays the real nickel core, and put everywhere else.

- **Windings.** Gallium-stabilised delta-phase plutonium is "roughly as strong and malleable as aluminium" with 35 % elongation (Hecker, Los Alamos Science 26, 2000), so it can be drawn; the brittle alpha phase cannot. Its resistivity is about 100 microohm centimetres, **60 times copper**, and, unusually for a metal, it falls slightly as it warms near room temperature (a negative temperature coefficient whose sign is verified and whose magnitude is unsourced, bounded at -1.5e-4 to -1e-3 per kelvin). Sixty times the winding resistance raises the low-frequency corner of the magnetising inductance, damps the leakage resonance at the top, and costs several dB of insertion loss.
- **An eddy slug.** A centimetre-scale delta-plutonium body in the coil's field is a lossy but non-shorting eddy load whose loss corner falls at a few kilohertz (skin depth 16 mm at 1 kHz, 3.6 mm at 20 kHz); copper in the same place would short the coil across the whole band. Plutonium's poor conductivity is what makes it a musical eddy load.
- **Decay heat.** Pu-239 metal self-heats at 1.96 mW/g, which keeps a 100 to 300 g body 3 to 18 K above the room, on or off. On a nickel core that changes the saturation by less than 0.1 dB: inaudible.
- **Radiation.** A bare surface emits about 9 million alphas per second per square centimetre: dense Poisson, which is Gaussian hiss and DC drift at a high-impedance node, not crackle; crackle is how a Geiger counter renders it. Any coating tens of micrometres thick removes it. Nothing of this is modelled.

### 5.2 As implemented

`coreFor()` starts from the fitted Nickel set (a real permalloy core) and changes:

| Element | Value | Tag |
|---|---|---|
| Low-frequency corner | 15 Hz (`kPuFlHz`): the winding resistance across the magnetising inductance | [P-direction], provisional value: the hardware's winding resistances and source impedance, which set the true value, are unknown |
| Eddy slug | a second shelf of -2 dB from 1.5 kHz (`kPuShelfHz`, `kPuShelfDb`) | [ESTIMATE] corner; [D] depth |
| Winding bloom | resistance falls as it warms: +0.001 dB per kelvin at true scale (`kPuBloomDbPerK`), ×30 in exhibition (`kPuBloomExhibition`) | [P] sign and true slope; [E] |
| Program heating of the winding | 0.1 K per unit mean-square at true scale, 10 K in exhibition; 2 s heating, 20 s cooling | [P] true; [E] |

So at true scale the bloom is about a ten-thousandth of a decibel; in exhibition a sustained passage at full-scale mean-square rises by about 0.3 dB (0.15 dB for a 0 dBFS sine) over a couple of seconds and relaxes over twenty. The constant decay-heat offset (`kPuDecayHeatK`, 10 K) is defined but not applied, because it changes nothing audible on a nickel core. The position is listed on the switch like the others; the design's hidden unlock, its heater macro that walks the nickel core to its Curie point, the bare-metal toggles and the Geiger-counter depiction are not implemented.

## 6. Every constant in `src/dsp/Materials.hpp`

| Constant | Value | Tag | Used by |
|---|---|---|---|
| `kGoldTc` | 385 K | [P] arXiv 2210.14069 | `goldKneeDb`, the paramagnetic switch in `coreFor` |
| `kGoldS` | 2 | [P] mean-field fit in the same paper | `goldKneeDb` |
| `kGoldKneeDbAt20C` | -6.0 dB | [P] 0.39 T against 0.78 T | `goldKneeDb` |
| `kGoldQScale` | 0.5 | [D] | GOLD knee hardness |
| `kGoldFlHz` | 3.0 Hz | [E] physical 0.8 to 55 kHz | GOLD corner |
| `kGoldGainDb` | -0.06 dB | [P] 1.31× copper into a bridging load | GOLD gain |
| `kGoldLightDb` | -1.5 dB | [P] 10 nm film, spectrally weighted | optical light gain (power ratio 0.708) |
| `kUraniumKneeDb` | +3.7 dB | [E-estimate] | URANIUM knee, relative to Nickel |
| `kUraniumQScale` | 0.6 | [D] | URANIUM knee hardness, relative to Steel |
| `kUraniumFlScale` | 2.0 | [INFERRED] | URANIUM corner, relative to Steel |
| `kUraniumTauEl2` | 370 us | [P] uranyl fluorescence lifetime | second optical persistence pole |
| `kUraniumSlowShare` | 0.20 | [P-range] hardware 0.15 to 0.25 | the share of the conductance routed through the decay chain (`OptoStage::process`) |
| `kThHalf`, `kPaHalf`, `kU234Half` | 0.26, 0.050, 3.0 s | [E] mapped time axis | `OptoStage::prepare`, the per-sample decay fractions of the chain |
| `kAlphaRate` | 2,400 per second | [P] 300 g of depleted uranium | `DecayNoise` |
| `kAlphaLevelDb` | -139 dBFS RMS | [P] -125 dBu at the divider node | `DecayNoise` |
| `kAlphaExhibitionDb` | +45 dB | [E] | `DecayNoise` in exhibition |
| `kFissionRate` | 7 per hour | [P] | `DecayNoise` |
| `kFissionExhibitionRate` | 0.5 per second | [E] | `DecayNoise` in exhibition |
| `kFissionScale` | 15 | [P] mean fragment burst against one alpha | `DecayNoise` |
| `kUraniumHeatDbPerK` | -0.01 dB/K | [P] | URANIUM winding element |
| `kUraniumHeatK` | 20 K per unit mean-square | [E] true 0.8 K at +18 dBu | URANIUM winding element in exhibition |
| `kUraniumHeatTau`, `kUraniumCoolTau` | 3 s, 8 s | [D] | URANIUM winding element |
| `kGeH2Scale` | 6 | [E-estimate] | GERMANIUM Class-A even-order term |
| `kGeLoopLpHz` | 5 kHz | [P] alloy-junction power devices 3 to 7 kHz | GERMANIUM loop low pass |
| `kGeHfShelfDb` | -1.0 dB | [D] | GERMANIUM output shelf (corner 10 kHz) |
| `kGeLeakDoubleC` | 9 C | [P] | GERMANIUM leakage law, detector leak, Class-A bias walk |
| `kGeLeakRatio25` | 0.3 | [D] physical 2.4 to 43 | GERMANIUM detector leak |
| `kGeDieRiseC` | 15 C | [D] | GERMANIUM die temperature |
| `kGeThermalTau` | 120 s | [P-estimate] | GERMANIUM die time constant |
| `kGeChokeStartC`, `kGeChokeFullC` | 75 C, 100 C | [P] germanium junction limits | GERMANIUM choke |
| `kGeNoiseDb` | -100 dBFS RMS | [P-estimate] | GERMANIUM noise floor |
| `kGeNoiseLpHz` | 300 Hz | [P-estimate] | one-pole corner on that noise (`Transformer.hpp`, `coreFor`) |
| `kGeVcaMismatchA2` | 4 | [D] | GERMANIUM gain-cell mismatch, per 10 dB of gain reduction |
| `kGeElCeilingDb` | -3 dB | [P-estimate] | defined, not applied |
| `kPuFlHz` | 15 Hz | [P-direction], provisional | PLUTONIUM corner |
| `kPuShelfHz`, `kPuShelfDb` | 1.5 kHz, -2 dB | [ESTIMATE] corner, [D] depth | PLUTONIUM eddy slug |
| `kPuDecayHeatK` | 10 K | [P] | defined, not applied |
| `kPuBloomDbPerK` | +0.001 dB/K | [P] | PLUTONIUM winding bloom |
| `kPuBloomExhibition` | 30 | [E] | PLUTONIUM winding bloom in exhibition |

`mat::brillouinM(t, S)` solves the mean-field self-consistency `m = B_S(3S m / ((S + 1) t))` by damped fixed-point iteration, returns 1 at zero temperature and 0 at or above the Curie temperature, and is called only when the temperature control moves.

## 7. The ledger: physical against exhibition

| Position | Effect | Physical | Shipped, EXHIBITION off | Shipped, EXHIBITION on |
|---|---|---|---|---|
| GOLD | Knee 6.0 dB below Nickel at 20 C, drifting with the room | physical | as physical | same |
| GOLD | Core permeability | rotation-only relative permeability about 1.4; corner 0.8 to 55 kHz | corner 3 Hz: ×100 to ×10,000 | same |
| GOLD | Knee hardness, coercivity | unpublished | half of Nickel's exponent: design | same |
| GOLD | Window, -1.5 dB of light | physical (film model) | as physical | same |
| GOLD | Dial above 45 C, Curie point at 112 C | real law, unreal room | reachable; model-dependent above 90 C | same |
| GOLD | Air core above Tc | 64 Hz corner, -6 to -11 dB, top loss | 6 Hz, -0.3 dB, -0.8 dB shelf | same |
| URANIUM | Knee 3.7 dB above Nickel | estimate on 1.1 to 1.3 T | as estimated | same |
| URANIUM | Winding heat | 0.8 K, 0.008 dB at +18 dBu | as physical | 10 K, 0.1 dB for a 0 dBFS sine |
| URANIUM | Alpha rumble | -139 dBFS, inaudible | as physical: silent | -94 dBFS |
| URANIUM | Fission pops | 7 per hour | as physical | 0.5 per second |
| URANIUM | Half-life release | the chain's order and topology | mapped time axis, 50 ms to 3 s | same |
| URANIUM | Glass persistence | 370 us | as physical (inaudible) | same |
| GERMANIUM | Bias walk, Early-effect even order, slow power device, leaky detector, cell mismatch, noise | physical mechanisms, estimated magnitudes | as estimated; detector leak re-scaled from 2.4 to 43 down to 0.3 | same |
| GERMANIUM | Core | air core, no audio | ideal linear core | same |
| GERMANIUM | Choke | real above 75 C junction | reachable from a 60 C room | same |
| PLUTONIUM | Windings, 60× copper | corner rise and damping, exact size unknown | 15 Hz, provisional | same |
| PLUTONIUM | Eddy slug | corner a few kHz, depth unknown | -2 dB from 1.5 kHz: design | same |
| PLUTONIUM | Winding bloom | +0.001 dB/K, 0.1 K of program heating | as physical (about 1e-4 dB) | ×30 and 10 K: about 0.3 dB |
| PLUTONIUM | Decay heat, radiation | 3 to 18 K, inaudible; hiss and DC at bare nodes | not modelled | not modelled |

## 8. Sources named in this document

He, Gercsi and others, "Au4Mn, a localized ferromagnet with strong spin-orbit coupling, long-range ferromagnetic exchange and high Curie temperature", Physical Review B 106, 214414 (2022), arXiv 2210.14069. Johnson and Christy, optical constants of the noble metals, Physical Review B 6, 4370 (1972), via refractiveindex.info. The CRC Handbook's table of magnetic susceptibilities. Berlureau and others on UFe10Si2, JMMM 102 (1991), and the OSTI record 171215. The OSTI compilation of uranium resistivity and thermal data (1433931). The IAEA page on depleted uranium. The LNHB nuclide tables for U-238. The NIST SRM 2941 uranyl-glass fluorescence work and the ORNL and Chemical Physics Letters lifetime measurements. Holmes, Holters and van Walstijn, "Comparison of germanium bipolar junction transistor models for real-time circuit simulation", DAFx-17. The Central Semiconductor 2N404 and 2N1302 datasheets, the Motorola 2N176/2N669 data-book pages, the onsemi 2N3055 datasheet, the BKC 1N34A datasheet, and the THAT 2181 datasheet. Keen's "Technology of the Fuzz Face" and the Small Bear Fuzz Face FAQ. Hecker, "Plutonium and its alloys", Los Alamos Science 26 (2000). Lashley, Lawson, McQueeney and Lander, "Absence of magnetic moments in plutonium", Physical Review B 72, 054416 (2005). Janoschek and others on valence fluctuations in plutonium, Science Advances 1 (2015). The Wikipedia articles on plutonium, plutonium-238 and radioluminescence for the decay-heat and phosphor facts.
