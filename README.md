# Hidden Valley Mastering Compressor

Hidden Valley Mastering Compressor is a stereo mastering compressor built as a VST3 and CLAP plugin (and on macOS also as an Audio Unit) with no GUI of its own (hosts show their generic controls) and meant to be driven from code (for example from Spotify's Pedalboard in Python). It is a behavioural model of the Shadow Hills Mastering Compressor: an optical stage into a discrete VCA stage into a choice of three output transformers, with every panel control stepped like the hardware. The model is fitted to measurements of a licensed copy of the commercial plug-in model of that unit and to published hardware data, and it says which is which.

The name is a nod. Hidden Valley Mastering Compressor is not affiliated with, endorsed by or derived from Shadow Hills Industries, Plugin Alliance or Brainworx, and contains no code, presets or resources of theirs.

## What makes it different

- **Two compressors in series, each modelled as the circuit it is.** The optical stage is a feedback shunt divider: an electroluminescent panel lit from the stage's own output drives a CdS cell, with a hard turn-on, an idle light leak that grows with the drive, and a release whose rate grows with the cell's own conductance. That one structure gives the fixed 2.6:1 slope above a knee under 1 dB wide, the fast recovery while the signal stays above the knee and the sub-second tail once it drops below, and the odd harmonics under gain reduction, without a separate term for any of them. The discrete stage is a feed-forward gain cell whose timing capacitor sits on the control voltage: the gain computer's target is smoothed by a node in dB of gain reduction with a bleed that never switches off and an attack diode that conducts harder the larger the drive, which is why the steady gain reduction falls as the attack is slowed, why the attack time depends on the recover setting, and why every ratio releases along one trajectory. Between the stages sit two measured details of the reference: a +2.14 dB gain when both are in, and a 0.5 % bleed of the signal the optical cell shunts away that reaches the output after the transformer. Each topology was chosen by residual among some sixty candidates fitted to the same measurements, then a second round on the soft ratios (`docs/detector-fix.md`, `docs/disc-knee-fix.md`, `docs/opto-fix.md`, `docs/stage-interaction.md`).
- **Three transformers as flux-domain cores, not EQ curves.** NICKEL, IRON and STEEL each integrate the signal into flux, saturate it against a hard ceiling with their own knee hardness and asymmetry, and differentiate it back, behind a driver with its own even and odd terms; IRON carries the extra Class-A stage the hardware has. Low frequencies at high level hit the ceiling, 1 kHz does not, and the harmonic spectra follow the reference's across a grid of ten frequencies and thirteen levels.
- **Four material positions that could not be built.** GOLD (an Au4Mn core, Curie point 385 K, a gold-flash window on the cell), URANIUM (a UFe10Si2 core, uranyl-glass persistence, a release that follows the U-238 decay chain, alpha-decay rumble), GERMANIUM (alloy-junction Class-A stage with its thermal bias walk, leaky detector) and PLUTONIUM (a nickel core behind delta-Pu windings with an NTC bloom). Each is derived from the material's real properties at true physical scale, where most of the effects are inaudible, with an EXHIBITION switch that scales them up and a TEMPERATURE control from 15 to 120 C. `docs/MATERIALS.md` keeps the ledger of what is physical and what is exaggerated.
- **Four calibration profiles.** REFERENCE reproduces the commercial plug-in for null tests. HARDWARE (the default) adds the uncontested hardware evidence, Iron's low-frequency lift and noise floor. MEASURED UNIT adds one torn-down unit's contested data: an 18 kHz pole in the optical path, an HF loss that grows with optical gain reduction, and that unit's mis-trimmed VCA at H2 −50 dBc. CLASS A is a derived all-Class-A signal path for the original Model 1 and Providence editions: a hotter input, single-ended module terms and soft ceilings borrowed from the fitted Iron stage and from the Pye 4060's published limits, with every number labelled derived, borrowed or estimated in `docs/class-a-profile.md`.
- **Every front-panel control, stepped like the hardware, for both channels.** Per channel: OPTICAL in/out, threshold and gain (24 positions each), DISCRETE in/out, threshold (24), ratio (1.2:1 to FLOOD), attack (0.1 to 30 ms), recover (0.1 s to DUAL), gain (24), SIDECHAIN FILTER, TRANSFORMER and METER SELECT; then STEREO / DUAL MONO, HARDWIRE BYPASS, and the plugin's own MIX, PROFILE, QUALITY, OPTO MEMORY, TEMPERATURE, EXHIBITION, SIDECHAIN HP and VU REFERENCE. That is 34 automatable parameters, each showing the panel's own legends (the sidechain corner is a number in hertz), plus seven read-only meter outputs.
- **Zero latency in STANDARD, 39 samples in HQ 2X.** STANDARD runs the whole model at the host rate with no oversampling, as the reference does. HQ 2X runs it at twice the rate behind a 79-tap half-band pair and reports its latency to the host.

## Install

Download the zip for your system and `SHA256SUMS` from the repository's Releases page. Every bundle carries `LICENSE` and `THIRD_PARTY_NOTICES.md` in `Contents/Resources/`. All formats are stereo in and out.

### Linux

`HiddenValleyMC-1.0.0-linux-x86_64.zip` (or `-linux-aarch64.zip`):

```bash
sudo apt-get install -y unzip        # minimal images lack it
sha256sum --ignore-missing -c SHA256SUMS
mkdir -p ~/.vst3
unzip -o HiddenValleyMC-1.0.0-linux-x86_64.zip -d ~/.vst3
```

That gives `~/.vst3/HiddenValleyMC.vst3`, which VST3 hosts on Linux find by themselves. Unzipping both zips there gives one bundle for both architectures.

### macOS

`HiddenValleyMC-1.0.0-macos-universal.zip` holds a VST3, a CLAP and an Audio Unit, each one universal binary for Apple Silicon and Intel, for macOS 11 or newer. In Terminal, in the folder you downloaded to:

```bash
shasum -a 256 --ignore-missing -c SHA256SUMS
unzip -o HiddenValleyMC-1.0.0-macos-universal.zip
mkdir -p ~/Library/Audio/Plug-Ins/VST3 ~/Library/Audio/Plug-Ins/CLAP ~/Library/Audio/Plug-Ins/Components
mv HiddenValleyMC.vst3 ~/Library/Audio/Plug-Ins/VST3/
mv HiddenValleyMC.clap ~/Library/Audio/Plug-Ins/CLAP/
mv HiddenValleyMC.component ~/Library/Audio/Plug-Ins/Components/
```

The bundles are signed ad hoc, not notarised, so macOS may refuse them on first use; right-click each bundle, choose Open, or allow it in System Settings, Privacy & Security. The Audio Unit appears in Logic and GarageBand after a rescan (`killall -9 AudioComponentRegistrar`, then reopen the host); `auval -v aufx HVmc Cbrk` validates it.

### Windows

`HiddenValleyMC-1.0.0-windows-x64.zip` holds the VST3. Unzip it into `C:\Program Files\Common Files\VST3\`.

## Build

Everything builds from source with `make`, a C++17 compiler and git; the build fetches [DPF](https://github.com/DISTRHO/DPF) at a pinned commit into `third_party/` on first run, then builds and runs the DSP unit tests (a failing test stops the build) before the plugin.

```bash
scripts/build.sh            # Linux: build/bin/HiddenValleyMC.vst3 and the CLAP
scripts/build-macos.sh      # macOS: build/macos/bin/HiddenValleyMC.vst3, .clap and .component, universal, ad-hoc signed
scripts/build-windows.sh    # Windows x64 with MinGW-w64, cross-compiling on Linux or in an MSYS2 MINGW64 shell
scripts/docker-build.sh aarch64 x86_64 windows-x64   # the release binaries, in throwaway Debian 12 containers
python3 scripts/package.py  # reproducible release zips and SHA256SUMS into dist/
```

The macOS build applies `scripts/dpf-au-parameter-strings.patch` to its copy of DPF so the Audio Unit shows the panel legends as the VST3 does. `-fno-fast-math` is required: the fitted constants and the tests assume IEEE arithmetic.

## Use it from Python with Pedalboard

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pedalboard==0.9.21
```

Pass the outer `.vst3` folder, not the binary inside it. Pedalboard lowercases the parameter names (the plugin declares `L_optical_threshold`; Python sees `l_optical_threshold`). Every value must be one of the parameter's positions, given as the panel legend: `comp.parameters["l_discrete_ratio"].valid_values` lists them. In STEREO the left-channel controls drive both channels and the `r_` parameters do nothing, as on the unit; METER SELECT stays independent per channel.

```python
import os
from pedalboard import load_plugin
from pedalboard.io import AudioFile

comp = load_plugin(os.path.expanduser("~/.vst3/HiddenValleyMC.vst3"))
print(comp.name)

with AudioFile("mix.wav") as f:
    audio, sr = f.read(f.frames), f.samplerate

# the panel legends, as printed on the unit; in STEREO the left controls drive both channels
comp.stereo = "STEREO"
comp.l_optical = "IN"
comp.l_optical_threshold = 18
comp.l_optical_gain = 12
comp.l_discrete = "IN"
comp.l_discrete_threshold = 14
comp.l_discrete_ratio = "2:1"
comp.l_discrete_attack_ms = 5
comp.l_discrete_recover_s = "0.5"
comp.l_discrete_gain = 9
comp.l_sidechain_filter = "IN"
comp.l_transformer = "IRON"
comp.quality = "HQ 2X"   # twice the rate inside; the host is told the 39-sample latency

out = comp(audio, sr)
with AudioFile("mix_hvmc.wav", "w", sr, out.shape[0]) as f:
    f.write(out)
```

That is the manufacturer's suggested mastering direction (a little optical, a little more discrete at a low ratio) with the filter in and the Iron transformer. The example is run as written by `tests/readme_examples.py`. Notes for scripted use:

- **Start state:** a render with `reset=True` (Pedalboard's default) starts exactly on the current settings and is bit-identical to a fresh instance; the engine resets every stage to the settings in force at the first block, so parameters set between loading and processing count as initial settings.
- **Changes while playing:** switches are smoothed over 15 ms, ratio changes crossfade over 20 ms, and a transformer change warms the new core on the last 80 ms of input and crossfades over 20 ms, so automation is click-free.
- **Meters:** the seven read-only outputs (`l_meter_db`, `l_gr_optical_db`, `l_gr_discrete_db`, the right-channel three, and `magic_eye_db`) read as the panel meters would. Pedalboard's VST3 host does not refresh output parameters between process calls; the Audio Unit route reads them correctly.
- **QUALITY is a mode, not an automation target:** switching it restarts the engine at the new rate (a level jump, and the latency changes). Set it before playback. In the Audio Unit the host is not told about a runtime latency change (a limitation of DPF's AU wrapper), so set QUALITY before the transport runs there.
- **Teardown:** DPF may print `asked to delete component while audio processor still active` when Python exits. It is harmless.

## Parameters

| # | parameter | positions | default | what it does |
|---|---|---|---|---|
| 0 | `optical (L_ and R_)` | OUT / IN | IN | OPTICAL BYPASS: IN puts the optical (electroluminescent panel and CdS cell) compressor in circuit |
| 1 | `optical_threshold (L_ and R_)` | 1 to 24 (24 steps) | 1 | OPTICAL THRESHOLD, 24 positions: 1 is the least compression, 24 the most (sidechain drive into the panel) |
| 2 | `optical_gain (L_ and R_)` | 1 to 24 (24 steps) | 11 | OPTICAL GAIN, 24 positions of make-up; unity is near 11, fine steps around unity, coarse at the ends |
| 3 | `discrete (L_ and R_)` | OUT / IN | IN | DISCRETE BYPASS: IN puts the discrete (VCA) compressor in circuit |
| 4 | `discrete_threshold (L_ and R_)` | 1 to 24 (24 steps) | 1 | DISCRETE THRESHOLD, 24 positions, 1 least to 24 most compression (about 2.7 dB per step) |
| 5 | `discrete_ratio (L_ and R_)` | 1.2:1 / 2:1 / 3:1 / 4:1 / 6:1 / FLOOD | 1.2:1 | DISCRETE RATIO: the labels are nominal; each position is a fixed, progressive curve (FLOOD over-compresses) |
| 6 | `discrete_attack_ms (L_ and R_)` | 0.1 / 0.5 / 1 / 5 / 10 / 30 | 30 | DISCRETE ATTACK in milliseconds |
| 7 | `discrete_recover_s (L_ and R_)` | 0.1 / 0.25 / 0.5 / 0.8 / 1.2 / DUAL | 0.1 | DISCRETE RECOVER in seconds; DUAL is a two-stage, program-dependent release |
| 8 | `discrete_gain (L_ and R_)` | 1 to 24 (24 steps) | 7 | DISCRETE GAIN, 24 positions of make-up; unity is at 7 |
| 9 | `sidechain_filter (L_ and R_)` | OUT / IN | IN | SIDECHAIN FILTER: IN high-passes both detectors (90 Hz, first order); audio is unfiltered |
| 10 | `transformer (L_ and R_)` | NICKEL / IRON / STEEL / GOLD / URANIUM / GERMANIUM / PLUTONIUM | NICKEL | Output transformer: NICKEL, IRON (with its extra Class-A stage), STEEL; GOLD, URANIUM, GERMANIUM and PLUTONIUM are digital-only material positions (see docs/MATERIALS.md) |
| 11 | `meter_select (L_ and R_)` | OPTICAL / DISCRETE / OUTPUT | OUTPUT | METER SELECT: what this channel's meter output shows |
| 24 | `stereo` | DUAL MONO / STEREO | STEREO | DUAL MONO or STEREO: in STEREO the left-channel controls govern both channels and the discrete detectors are linked |
| 25 | `hardwire_bypass` | OUT / IN | IN | HARDWIRE BYPASS: OUT connects input to output; IN engages the line amplifiers and the transformer |
| 26 | `mix_percent` | 0 to 100 (101 steps) | 100 | Parallel mix with the unprocessed input (plugin addition), percent wet |
| 27 | `profile` | REFERENCE / HARDWARE / MEASURED UNIT / CLASS A | HARDWARE | Calibration profile: REFERENCE (fitted to the reference plugin), HARDWARE (plus uncontested hardware evidence), MEASURED UNIT (plus one torn-down unit's contested data), CLASS A (HARDWARE plus a derived all-Class-A signal path: hotter input, single-ended module terms and ceilings; docs/class-a-profile.md) |
| 28 | `quality` | STANDARD / HQ 2X | STANDARD | STANDARD runs at the host rate with zero latency; HQ 2X runs the whole channel at twice the rate (reports latency) |
| 29 | `opto_memory` | OFF / ON | OFF | Opto light memory: the CdS cell's release lengthens after long, deep compression (off matches the reference) |
| 30 | `temperature_c` | 15 to 120 (106 steps) | 25 | Room temperature for the material positions (15-45 C physical, above that an exhibition range) |
| 31 | `exhibition` | OFF / ON | OFF | Exhibition scale for sub-audible material effects (off = true physical scale) |
| 32 | `sidechain_hp_hz` | 20 to 666 (647 steps) | 90 | Sidechain high-pass corner in Hz, first order, both detectors, when SIDECHAIN FILTER is in (the hardware's fixed 90 Hz is the default) |
| 33 | `vu_reference_dbfs` | -18 / -14 / -9 | -14 | VU reference: the sine peak level in dBFS that reads 0 VU on the OUTPUT meter (-14 = 0 dBu at +14 dBu full scale) |

Read-only outputs (the host shows them as meters):

| # | output | what it reads |
|---|---|---|
| 34 | `L_meter_db` | Left meter as METER SELECT shows it: optical or discrete gain reduction (dB, negative) or output VU (0 VU = -14 dBFS peak sine) |
| 35 | `L_gr_optical_db` | Left optical gain reduction, dB (negative), ballistic like the panel meter |
| 36 | `L_gr_discrete_db` | Left discrete gain reduction, dB (negative), ballistic like the panel meter |
| 37 | `R_meter_db` | Right meter as METER SELECT shows it (see L_meter_db) |
| 38 | `R_gr_optical_db` | Right optical gain reduction, dB (negative) |
| 39 | `R_gr_discrete_db` | Right discrete gain reduction, dB (negative) |
| 40 | `magic_eye_db` | Mono output peak in dBFS, the signal the magic-eye tube follows |

The discrete stage's ratio legends are nominal: each position is a fixed, progressive curve of gain reduction against level fitted to the reference, and 4:1 and FLOOD over-compress (the 2008 manual calls FLOOD 20:1; current dealer copy lists the same position as 10:1 and DUAL as AUTO). The optical stage's nominal 2:1 measures as about 2.6:1.

## Verification in numbers

Measured on the macOS build through Pedalboard 0.9.21 on the `arm64` VST3 and Audio Unit natively (the `x86_64` half of each universal binary runs the C++ unit tests under Rosetta 2 in `scripts/build-macos.sh`; the Pedalboard checks have not been run on that half); `scripts/test.sh` runs every check below and stops at the first failure. The tolerance table in `tests/pb_reference.py` carries about 1.5 times the achieved rms and 1.3 times the achieved maximum of each group, so a change that pushes a group past it is a regression, not noise; the figures below are the achieved ones.

- **Against the reference, 5,235 protocol items** (`tests/pb_reference.py`, `fit/data/reference_features.json`): the make-up laws within 0.001 dB; the optical static family, 24 threshold positions by 33 levels, 0.079 dB rms and 0.52 dB worst; the discrete static family, six ratios by 24 thresholds by 25 levels, 0.029 dB rms and 0.15 dB worst, with exact silence at every ratio; steady-state gain reduction against attack and recover 0.087 dB rms; burst envelopes 0.060 dB rms (optical) and 0.079 dB rms (discrete, including the soft ratios' post-burst tails; the largest error is one period into a 0 dBFS onset); the two stages together 0.084 dB rms; the stereo link 0.051 dB rms; the sidechain filter 0.048 dB rms; the transformer path's gain 0.033 dB rms, odd harmonics 3.4 dB rms and even harmonics 6.0 dB rms across the 20 Hz to 5 kHz, −12 to +21 dBFS grid (+24 dBFS is pooled apart as the reference's output ceiling); the optical stage's odd harmonics under gain reduction 1.8 dB rms over 28 items (the third alone 1.4 dB rms over 32 items, the fifth 3.3 dB).
- **Every position of every control is live and does what its legend says** (`tests/pb_coverage.py`, about 590 renders): the only audio-neutral positions are METER SELECT, REFERENCE profile with Nickel, EXHIBITION with a hardware core and TEMPERATURE with a non-material core, each by design; STEREO makes the right-channel controls neutral and DUAL MONO makes the channels independent.
- **Bit-exact:** the plugin equals the engine driven through its C interface sample for sample, and equals a fresh engine given the same settings, on tones, bursts into the DUAL release, noise through Iron and Steel, HQ 2X and the material positions (`tests/crosscheck.py`); HARDWIRE BYPASS out and MIX 0 % return the input bit for bit; 44.1, 48 and 96 kHz renders are identical across block sizes of 64, 1024 and 8192.
- **Latency 0 in STANDARD, 39 samples in HQ 2X,** and the HQ output through the host's compensation is bit-transparent with the bypass out (`tests/pb_load.py`).
- **The Audio Unit passes Apple's validation** (`auval -v aufx HVmc Cbrk`) with the panel legends as value strings.
- **Programme material** (`fit/tools/real_audio_compare.py` and `fit/tools/program_audit.py`, a 30 s house mix plus synthetic programme through both plugins, run locally against the licensed reference): the transformer path alone nulls to −38 dB; both stages at a gentle mastering setting (optical 18, discrete 14 at 2:1, filter in, Iron) land within 0.08 dB rms of the reference at a −33 dB null, 4:1 with DUAL within 0.01 dB at −32 dB, and a FLOOD squash (34 dB of gain reduction) within 0.22 dB at −26 dB. The FLOOD remainder is the ratio-dependent attack network still to be fitted under the new node (`FUTURE_WORK.md`).

## Limits

- **It is a model of a model, plus hardware data.** Every dynamic and static behaviour is fitted to measurements of the commercial plug-in model of the unit (version 1.5.1) and the profiles add published hardware evidence; no unit was on the bench for this version. `docs/MODEL.md` section 8 lists what is measured, what is assumed and what is invented.
- **Known residuals.** The optical stage's fifth harmonic under gain reduction is within about 3 dB of the reference's and its fourth-kilohertz rows at 48 kHz are aliasing of the light pulse (HQ 2X removes it); the transformer cores' onset burst of even harmonics is matched to about 6 dB rms, with the 120 and 160 Hz points at +21 dBFS at the foot of the reference's output ceiling the worst; the discrete detector's attack differs slightly between ratio positions in the reference (FLOOD in particular), which the model carries as hooks not yet fitted. Each is tracked in `FUTURE_WORK.md`.
- **Not modelled.** The reference's output ceiling near +22 dBFS (+36 dBu), its softening above +20 dBFS, and its external key input. The reference's variable sidechain corner and VU reference are both here (`sidechain_hp_hz`, 20 to 666 Hz, within 0.11 dB of the reference's at 200 and 400 Hz; `vu_reference_dbfs`). The Class A and Providence editions of the hardware have no measurements; the CLASS A profile is derived, not fitted, and says so.
- **Standard mode is not oversampled**, as the reference is not; harmonics above Nyquist alias at 44.1 and 48 kHz in both. HQ 2X halves that at the cost of 39 samples of latency.

## Repository layout

| path | what it is |
|---|---|
| `src/PluginHVMC.cpp`, `src/HVMCParams.hpp`, `src/DistrhoPluginInfo.h`, `src/Makefile` | the DPF plugin: the 32 parameters and 7 meter outputs, one engine for both channels |
| `src/dsp/Engine.hpp`, `Opto.hpp`, `Discrete.hpp`, `Transformer.hpp`, `Oversampler.hpp`, `Meters.hpp`, `Common.hpp` | the model: the two stages, the transformer path, HQ resampling, meters |
| `src/dsp/Calibration.hpp`, `src/dsp/FittedConstants.hpp`, `src/dsp/Materials.hpp` | the calibration table and its fitted values (generated), the material constants |
| `src/capi/hvmc_capi.cpp` | the C interface the fitting and the tests drive the engine through |
| `fit/` | the measurement protocol, the captures of the reference (`fit/data/`), the five fitting stages, `run_all.py` and `make_constants.py` |
| `tests/` | `test_dsp.cpp` (C++ unit tests), `crosscheck.py`, `pb_load.py`, `pb_reference.py`, `pb_coverage.py`, `readme_examples.py`, `au_register.py` |
| `scripts/` | `build.sh`, `build-macos.sh`, `build-windows.sh`, `docker-build.sh`, `test.sh`, `docker-test.sh`, `package.py`, `dpf-au-parameter-strings.patch` |
| `docs/` | `MODEL.md`, `MATERIALS.md`, the research syntheses (`detector-fix.md`, `opto-fix.md`) |

The reference plug-in and its captured features are the measurement, not the model: `fit/data/reference_features.json` holds numbers, nothing of the product.

## Credits

- [DPF](https://github.com/DISTRHO/DPF), the DISTRHO Plugin Framework, by Filipe Coelho (ISC).
- [Pedalboard](https://github.com/spotify/pedalboard) by Spotify (GPL-3.0), the host the tests run in.
- The realgearonline teardown thread and the Flotown Mastering four-unit shoot-out, for the hardware data the profiles carry.
- Peter Reardon designed the unit this is modelled on. VST is a registered trademark of Steinberg Media Technologies GmbH.

The full notices are in `THIRD_PARTY_NOTICES.md`.

## Licence

GPL-3.0-only: see `LICENSE`. DPF keeps its ISC licence, listed with its notice in `THIRD_PARTY_NOTICES.md`.

## Author

Cameron Brooks.
