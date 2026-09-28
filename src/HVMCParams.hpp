// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The front panel as plugin parameters: every knob and switch of both channels with the panel's own positions, then the global
// switches, then the read-only meters. Names are written for a reader (human or agent): L_/R_ channel prefix, the unit in the name
// where there is one (attack_ms, recover_s, mix_percent, temperature_c). Every input parameter is a stepped enumeration whose value is
// the position index (0 = first legend); hosts that read the labels (Pedalboard does) show the legends.
#pragma once
#include <cstdio>

namespace hvmc {

enum ChannelParam {
    kPOptical,          // OUT / IN
    kPOpticalThreshold, // 1 .. 24
    kPOpticalGain,      // 1 .. 24
    kPDiscrete,         // OUT / IN
    kPDiscreteThreshold,
    kPDiscreteRatio,    // 1.2:1 2:1 3:1 4:1 6:1 FLOOD
    kPDiscreteAttack,   // 0.1 0.5 1 5 10 30 ms
    kPDiscreteRecover,  // 0.1 0.25 0.5 0.8 1.2 s, DUAL
    kPDiscreteGain,     // 1 .. 24
    kPSidechainFilter,  // OUT / IN
    kPTransformer,      // NICKEL IRON STEEL GOLD URANIUM GERMANIUM PLUTONIUM
    kPMeterSelect,      // OPTICAL DISCRETE OUTPUT
    kPerChannel
};

enum GlobalParam {
    kGStereo = 2 * kPerChannel, // DUAL MONO / STEREO
    kGHardwire,                 // OUT / IN (IN = the unit is in circuit)
    kGMix,                      // 0 .. 100 %
    kGProfile,                  // REFERENCE / HARDWARE / MEASURED UNIT / CLASS A
    kGQuality,                  // STANDARD / HQ 2X
    kGOptoMemory,               // OFF / ON
    kGTemperature,              // 15 .. 120 C
    kGExhibition,               // OFF / ON
    kGScHpHz,                   // 20 .. 666 Hz: the sidechain high-pass corner when SIDECHAIN FILTER is in (plugin control of the reference)
    kGVuRef,                    // -18 / -14 / -9 dBFS: the sine peak level that reads 0 VU on the OUTPUT meter
    kNumInputParams
};

enum OutputParam {
    kOMeterL = kNumInputParams, // the meter as METER SELECT shows it: optical or discrete GR (dB, negative) or output VU
    kOGrOpticalL,
    kOGrDiscreteL,
    kOMeterR,
    kOGrOpticalR,
    kOGrDiscreteR,
    kOMagicEye,                 // mono output peak, dBFS (the VT-138 follows it)
    kNumParams
};

enum Transformer { kNickel, kIron, kSteel, kGold, kUranium, kGermanium, kPlutonium, kNumTransformers };
enum Profile { kProfileReference, kProfileHardware, kProfileMeasuredUnit, kProfileClassA };
enum MeterSelect { kMeterOptical, kMeterDiscrete, kMeterOutput };

static constexpr int kTempMin = 15, kTempMax = 120, kTempDefault = 25;
static constexpr int kScHpMin = 20, kScHpMax = 666;
static constexpr double kVuRefDbfs[3] = { -18.0, -14.0, -9.0 };

inline bool isOutput(int index) { return index >= kNumInputParams; }
// a parameter declared to the host in its own unit rather than as an enumeration (DPF's enumeration count is 8-bit): the host value is
// the position plus this offset
inline int hostOffset(int index) { return index == kGScHpHz ? kScHpMin : 0; }

// number of positions of an input parameter
inline int paramSteps(int index)
{
    if (index < 2 * kPerChannel) {
        switch (index % kPerChannel) {
        case kPOptical: case kPDiscrete: case kPSidechainFilter: return 2;
        case kPOpticalThreshold: case kPOpticalGain: case kPDiscreteThreshold: case kPDiscreteGain: return 24;
        case kPDiscreteRatio: case kPDiscreteAttack: case kPDiscreteRecover: return 6;
        case kPTransformer: return kNumTransformers;
        case kPMeterSelect: return 3;
        }
    }
    switch (index) {
    case kGStereo: case kGHardwire: case kGQuality: case kGOptoMemory: case kGExhibition: return 2;
    case kGMix: return 101;
    case kGProfile: return 4;
    case kGTemperature: return kTempMax - kTempMin + 1;
    case kGScHpHz: return kScHpMax - kScHpMin + 1;
    case kGVuRef: return 3;
    }
    return 1;
}

// default position: stages in and at rest (threshold 1, unity make-up), the manual's mastering settings for the rest
inline int paramDefault(int index)
{
    if (index < 2 * kPerChannel) {
        switch (index % kPerChannel) {
        case kPOptical: case kPDiscrete: return 1;
        case kPOpticalThreshold: return 0;
        case kPOpticalGain: return 10;          // position 11: unity (+0.08 dB)
        case kPDiscreteThreshold: return 0;
        case kPDiscreteRatio: return 0;         // 1.2:1
        case kPDiscreteAttack: return 5;        // 30 ms
        case kPDiscreteRecover: return 0;       // 0.1 s
        case kPDiscreteGain: return 6;          // position 7: unity (+0.02 dB)
        case kPSidechainFilter: return 1;       // IN
        case kPTransformer: return kNickel;
        case kPMeterSelect: return kMeterOutput;
        }
    }
    switch (index) {
    case kGStereo: return 1;
    case kGHardwire: return 1;
    case kGMix: return 100;
    case kGProfile: return kProfileHardware;
    case kGQuality: return 0;
    case kGOptoMemory: return 0;
    case kGTemperature: return kTempDefault - kTempMin;
    case kGExhibition: return 0;
    case kGScHpHz: return 90 - kScHpMin;
    case kGVuRef: return 1;
    }
    return 0;
}

inline void positionLabel(int index, int v, char* out, int cap)
{
    static const char* inout[] = { "OUT", "IN" };
    static const char* ratios[] = { "1.2:1", "2:1", "3:1", "4:1", "6:1", "FLOOD" };
    static const char* attacks[] = { "0.1", "0.5", "1", "5", "10", "30" };
    static const char* recovers[] = { "0.1", "0.25", "0.5", "0.8", "1.2", "DUAL" };
    static const char* xfmr[] = { "NICKEL", "IRON", "STEEL", "GOLD", "URANIUM", "GERMANIUM", "PLUTONIUM" };
    static const char* meters[] = { "OPTICAL", "DISCRETE", "OUTPUT" };
    static const char* profiles[] = { "REFERENCE", "HARDWARE", "MEASURED UNIT", "CLASS A" };
    static const char* onoff[] = { "OFF", "ON" };
    if (index < 2 * kPerChannel) {
        switch (index % kPerChannel) {
        case kPOptical: case kPDiscrete: case kPSidechainFilter: std::snprintf(out, size_t(cap), "%s", inout[v]); return;
        case kPDiscreteRatio: std::snprintf(out, size_t(cap), "%s", ratios[v]); return;
        case kPDiscreteAttack: std::snprintf(out, size_t(cap), "%s", attacks[v]); return;
        case kPDiscreteRecover: std::snprintf(out, size_t(cap), "%s", recovers[v]); return;
        case kPTransformer: std::snprintf(out, size_t(cap), "%s", xfmr[v]); return;
        case kPMeterSelect: std::snprintf(out, size_t(cap), "%s", meters[v]); return;
        default: std::snprintf(out, size_t(cap), "%d", v + 1); return;   // the 24-position switches, 1 .. 24
        }
    }
    switch (index) {
    case kGStereo: std::snprintf(out, size_t(cap), "%s", v ? "STEREO" : "DUAL MONO"); return;
    case kGHardwire: std::snprintf(out, size_t(cap), "%s", inout[v]); return;
    case kGProfile: std::snprintf(out, size_t(cap), "%s", profiles[v]); return;
    case kGQuality: std::snprintf(out, size_t(cap), "%s", v ? "HQ 2X" : "STANDARD"); return;
    case kGOptoMemory: case kGExhibition: std::snprintf(out, size_t(cap), "%s", onoff[v]); return;
    case kGScHpHz: std::snprintf(out, size_t(cap), "%d", v + kScHpMin); return;
    case kGVuRef: std::snprintf(out, size_t(cap), "%d", int(kVuRefDbfs[v])); return;
    case kGTemperature: std::snprintf(out, size_t(cap), "%d", v + kTempMin); return;
    default: std::snprintf(out, size_t(cap), "%d", v); return;   // mix 0 .. 100
    }
}

inline void paramName(int index, char* out, int cap)
{
    static const char* ch[] = { "optical", "optical_threshold", "optical_gain", "discrete", "discrete_threshold", "discrete_ratio",
                                "discrete_attack_ms", "discrete_recover_s", "discrete_gain", "sidechain_filter", "transformer",
                                "meter_select" };
    static const char* gl[] = { "stereo", "hardwire_bypass", "mix_percent", "profile", "quality", "opto_memory", "temperature_c",
                                "exhibition", "sidechain_hp_hz", "vu_reference_dbfs" };
    static const char* outs[] = { "L_meter_db", "L_gr_optical_db", "L_gr_discrete_db", "R_meter_db", "R_gr_optical_db",
                                  "R_gr_discrete_db", "magic_eye_db" };
    if (index < 2 * kPerChannel) { std::snprintf(out, size_t(cap), "%s_%s", index < kPerChannel ? "L" : "R", ch[index % kPerChannel]); return; }
    if (index < kNumInputParams) { std::snprintf(out, size_t(cap), "%s", gl[index - kGStereo]); return; }
    std::snprintf(out, size_t(cap), "%s", outs[index - kNumInputParams]);
}

inline const char* paramDescription(int index)
{
    if (index < 2 * kPerChannel) {
        switch (index % kPerChannel) {
        case kPOptical: return "OPTICAL BYPASS: IN puts the optical (electroluminescent panel and CdS cell) compressor in circuit";
        case kPOpticalThreshold: return "OPTICAL THRESHOLD, 24 positions: 1 is the least compression, 24 the most (sidechain drive into the panel)";
        case kPOpticalGain: return "OPTICAL GAIN, 24 positions of make-up; unity is near 11, fine steps around unity, coarse at the ends";
        case kPDiscrete: return "DISCRETE BYPASS: IN puts the discrete (VCA) compressor in circuit";
        case kPDiscreteThreshold: return "DISCRETE THRESHOLD, 24 positions, 1 least to 24 most compression (about 2.7 dB per step)";
        case kPDiscreteRatio: return "DISCRETE RATIO: the labels are nominal; each position is a fixed, progressive curve (FLOOD over-compresses)";
        case kPDiscreteAttack: return "DISCRETE ATTACK in milliseconds";
        case kPDiscreteRecover: return "DISCRETE RECOVER in seconds; DUAL is a two-stage, program-dependent release";
        case kPDiscreteGain: return "DISCRETE GAIN, 24 positions of make-up; unity is at 7";
        case kPSidechainFilter: return "SIDECHAIN FILTER: IN high-passes both detectors (90 Hz, first order); audio is unfiltered";
        case kPTransformer: return "Output transformer: NICKEL, IRON (with its extra Class-A stage), STEEL; GOLD, URANIUM, GERMANIUM and PLUTONIUM are digital-only material positions (see docs/MATERIALS.md)";
        case kPMeterSelect: return "METER SELECT: what this channel's meter output shows";
        }
    }
    switch (index) {
    case kGStereo: return "DUAL MONO or STEREO: in STEREO the left-channel controls govern both channels and the discrete detectors are linked";
    case kGHardwire: return "HARDWIRE BYPASS: OUT connects input to output; IN engages the line amplifiers and the transformer";
    case kGMix: return "Parallel mix with the unprocessed input (plugin addition), percent wet";
    case kGProfile: return "Calibration profile: REFERENCE (fitted to the reference plugin), HARDWARE (plus uncontested hardware evidence), MEASURED UNIT (plus one torn-down unit's contested data), CLASS A (HARDWARE plus a derived all-Class-A signal path: hotter input, single-ended module terms and ceilings; docs/class-a-profile.md)";
    case kGQuality: return "STANDARD runs at the host rate with zero latency; HQ 2X runs the whole channel at twice the rate (reports latency)";
    case kGOptoMemory: return "Opto light memory: the CdS cell's release lengthens after long, deep compression (off matches the reference)";
    case kGTemperature: return "Room temperature for the material positions (15-45 C physical, above that an exhibition range)";
    case kGExhibition: return "Exhibition scale for sub-audible material effects (off = true physical scale)";
    case kGScHpHz: return "Sidechain high-pass corner in Hz, first order, both detectors, when SIDECHAIN FILTER is in (the hardware's fixed 90 Hz is the default)";
    case kGVuRef: return "VU reference: the sine peak level in dBFS that reads 0 VU on the OUTPUT meter (-14 = 0 dBu at +14 dBu full scale)";
    case kOMeterL: return "Left meter as METER SELECT shows it: optical or discrete gain reduction (dB, negative) or output VU (0 VU = -14 dBFS peak sine)";
    case kOGrOpticalL: return "Left optical gain reduction, dB (negative), ballistic like the panel meter";
    case kOGrDiscreteL: return "Left discrete gain reduction, dB (negative), ballistic like the panel meter";
    case kOMeterR: return "Right meter as METER SELECT shows it (see L_meter_db)";
    case kOGrOpticalR: return "Right optical gain reduction, dB (negative)";
    case kOGrDiscreteR: return "Right discrete gain reduction, dB (negative)";
    case kOMagicEye: return "Mono output peak in dBFS, the signal the magic-eye tube follows";
    }
    return "";
}

} // namespace hvmc
