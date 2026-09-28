# Future work

What was deliberately left out of 1.0, why, and what it would take.

## Needs a hardware unit

- **A bench session on a real unit.** No public measurement of the hardware's dynamics exists (no static curve, no attack or release, no knee, no sidechain-filter response), so every dynamics constant is fitted to the reference plug-in. Half a day with a unit and an analyser would settle, in this order: the transformer path with both stages out (response per core at four levels into 600 ohms and 10 kohms, THD against frequency and level per core, the clip point, the noise floor), which is the contested colour reference; the discrete static grid and time constants, including whether RECOVER 0.5 and 0.8 s differ (they are identical in the reference); the optical statics, dynamics, memory and high-frequency loss against gain reduction; and the same measurements on the reference at the same dBu anchor. Deferred because it needs a unit and outward contact, which are the owner's call.
- **The operating-level anchor.** Every hardware-against-plug-in comparison assumes 0 dBFS = +14 dBu (the reference's meter default). Whether the reference's model was built to that anchor is unknown; the bench session above, or a question to the reference's maker, would fix it.
- **Which physical transformer is Iron and which is Steel**, whether the Class-A stage is switched in or always running, and the transformers' winding resistances, inductance and source impedance. These do not change the model (the stage is a routing flag and the corners are fitted), but they set the true value of PLUTONIUM's provisional 15 Hz corner and URANIUM's insertion loss.
- **A second unit's high-frequency response** (the torn-down unit's -3.8 dB at 20 kHz against other owners' -1 dB at 16 kHz) and a unit with the discrete stage bypassed, to separate the transformer paths' own distortion from the VCA's.

## Model options the plan describes and 1.0 does not ship

- **Class-A edition mode.** A flag that routes an Iron-type asymmetric stage into all three positions with its own bias, plus the vendors' +1 to +3 dB level offset and the fixed optical high-frequency roll-off that model has. It needs the other edition's model measured at matched settings before its constants exist.
- **Opto link option** (`opto_link = average`) for units with the link header cabled, and a selectable **optical sidechain source** (output or input) until a hardware gain-against-threshold test settles the topology. Both are small; the defaults match the reference.
- **Load parameter** (600 ohms / 10 kohms) scaling the transformer path's low-frequency corner and insertion gain; unverifiable on the reference, needs the hardware numbers above.
- **Per-channel mismatch** (cell speed, sidechain trim, cell symmetry) as fixed, documented constants; the two channels of the torn-down unit differ by 3 dB in second harmonic. Not implemented; the two channels are identical.
- **Control-voltage feedthrough** in the discrete gain cell (a DC step proportional to a gain change), a property of the cell family that the reference does not show and no hardware measurement quantifies.
- **Jiles-Atherton hysteresis** in the transformer core (steel's low-level grit and magnetisation memory). The flux-domain saturator reproduces everything the reference does; hysteresis would only be justified by a hardware measurement of loop opening at 20 to 40 Hz, and it costs oversampling.
- **Light memory constants.** The option ships with priors from the hardware manual's description and LA-2A-family data; the reference has no memory to fit them to.
- **Anti-derivative anti-aliasing in STANDARD mode.** The plan proposed first-order ADAA on the memoryless shapers; the nonlinearities are mild enough at mastering levels that 1.0 runs them plainly at the host rate, as the reference does, with HQ 2X for the rest. Worth revisiting if a use case drives the stages hard at 44.1 kHz.

## Material positions

- **Uranium CRYO**: the "uranium is a magnet only when cold" sub-mode (US, UFe2, UGe2 below their ordering points, UCoAl's metamagnetic step, superconducting U6Fe windings). Deferred because the compounds' permeabilities and switching fields would need exhibition scaling by four or more orders of magnitude, which the critique asked to be ledgered first.
- **Literal demos**: a per-position air-core "ghost" (the core bypassed, an exhibition high pass at 80 to 150 Hz, labelled with the true same-winding corner). Nearly free once a UI exists; pointless without one.
- **Curie macro** for PLUTONIUM: a heater state that walks the nickel core up its magnetisation curve to its Curie point (400 C for the alloy the fit uses, 455 C for HyMu 80) and crossfades the core to linear above it. Needs a heater-to-core coupling, the alloy's Curie point settled, and permalloy `a(T)`, `k(T)` laws; the hidden unlock it was to sit behind is a UX decision.
- **Ge EL ceiling**: a germanium panel driver would swing about 3 dB less (`kGeElCeilingDb` is defined, the drive ceiling hook exists in the optical stage, nothing sets it). A one-line change, left out until the effect on maximum gain reduction is measured and documented.
- **Germanium extras**: a per-unit seed for the device spread, the passive-detector topology swap (the Chandler-style soft knee, labelled as not this topology), 1/f noise, program-dependent die heating, the op-amp modules' lower headroom.
- **Gold extras**: the gold-doped driver (a hypothetical device), leaky gold-doped detector diodes, the ruby-glass optic, a session warm-up model.
- **Uranium extras**: the age-since-casting law, the radioluminescent light floor, the UO2 oxide-contact element (no audio-band data), natural against depleted activity.
- **Blind listening.** Every word the research attaches to a material ("warm", "heavy", "alive") is inference. Free-descriptor A/B tests of GOLD against NICKEL, GERMANIUM against IRON and URANIUM against STEEL should happen before any such word appears in user-facing copy.
- **Honesty metadata.** The design asked for each material constant to carry its shipped value, physical value, factor and source in a machine-readable form from which the ledger in `docs/MATERIALS.md` is generated, so the two cannot drift. The ledger is written by hand in 1.0.

## Validation

- **ABX sessions** on program material, level-matched, logged; the plan's acceptance table (waveform null and gain-trajectory error per condition) is only partly covered by the automated tests.
- **A re-measurement against the reference's current version.** The capture is of 1.5.1; later versions list no DSP change, but a diff of the derived curves on 1.7.0 would confirm it.
- **Windows CI** and the cross-compiled Windows build, as the sibling project has them.

## GUI

There is none; hosts show generic controls and the read-only meters. A panel with the VU meters and the magic eye is the obvious next thing.

- **CLASS A profile against a Class A unit or the Class A plug-in**: the profile's twelve constants are derived and borrowed (`docs/class-a-profile.md`). The cheapest check is a capture of the separate Class A plug-in with the existing protocol (TMT held on one channel): a static series at fixed settings places the level offset at the input or the output, and the per-core harmonic grid with both stages bypassed gives the module terms and the ceiling directly. A bench session on a unit settles the H2 phase (which polarity clips first) and the input transformer's corners.
- **Programme-material audit findings** (`research/program-audit.md`): on a real mix the model over-compresses by 0.35 to 1.1 dB rms against the reference at settings that engage the stages, uniformly across frequency (the transformers alone null to -38 dB). The audit localises this; the candidates are the soft-ratio knee, the detector's charge per cycle on dense material, the sidechain filter's insertion gains and the optical loop on programme envelopes.
