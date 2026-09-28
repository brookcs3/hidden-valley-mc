// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The digital-only material positions (GOLD, URANIUM, GERMANIUM, PLUTONIUM): their physical constants and how they turn into the
// model's parameters. None of these metals can be a transformer core by itself (all have |mu_r - 1| < 1e-3); each position is an
// assignment of materials to the blocks where the material really acts. The derivations, sources, and every departure from physical
// scale are in docs/MATERIALS.md; the numbers below carry a tag: [P] physical, [E] exhibition (scaled up, documented), [D] design choice.
#pragma once
#include <cmath>
#include "Common.hpp"

namespace hvmc {
namespace mat {

// ------------------------------------------------------------------------------------------------ magnetisation against temperature
// Mean-field reduced magnetisation m = M(T)/M(0) for spin S: m = B_S((3S/(S+1)) m / t), t = T/Tc. Solved by fixed-point iteration
// (called when the temperature control moves, never per sample). Returns 0 at or above Tc.
inline double brillouinM(double t, double S)
{
    if (t >= 1.0) return 0.0;
    if (t <= 1e-6) return 1.0;
    double m = std::sqrt(1.0 - t) + 0.2;
    if (m > 1.0) m = 1.0;
    const double a = 3.0 * S / (S + 1.0);
    for (int it = 0; it < 200; ++it) {
        const double x = a * m / t;
        const double c1 = (2.0 * S + 1.0) / (2.0 * S), c2 = 1.0 / (2.0 * S);
        const double b = c1 / std::tanh(c1 * x) - c2 / std::tanh(c2 * x);
        const double mn = 0.5 * m + 0.5 * b;
        if (std::fabs(mn - m) < 1e-12) { m = mn; break; }
        m = mn;
    }
    return m;
}

// GOLD: ordered Au4Mn, a ferromagnet that is 93.5 % gold by weight. Tc 385 K, mean-field S = 2 fits its room-temperature moment.
static constexpr double kGoldTc = 385.0;                 // [P] arXiv 2210.14069
static constexpr double kGoldS = 2.0;                    // [P] mean-field fit in the same paper
static constexpr double kGoldKneeDbAt20C = -6.0;         // [P] saturation flux 0.39 T against 80 % Ni 0.78 T
static constexpr double kGoldQScale = 0.5;               // [D] softest knee on the switch (knee exponent half of Nickel's)
static constexpr double kGoldFlHz = 3.0;                 // [E] physical corner is 0.8-55 kHz (permeability ~1.4-100)
static constexpr double kGoldGainDb = -0.06;             // [P] 1.31x copper winding resistance, bridging load
static constexpr double kGoldLightDb = -1.5;             // [P] 10 nm gold-flash window on the opto (spectrally weighted)

// knee level relative to Nickel at temperature tC, dB: Au4Mn's saturation follows m(T); -6.0 dB at 20 C
inline double goldKneeDb(double tC)
{
    const double m = brillouinM((tC + 273.15) / kGoldTc, kGoldS);
    const double m20 = brillouinM((20.0 + 273.15) / kGoldTc, kGoldS);
    if (m <= 1e-4) return -80.0;                          // above Tc: paramagnetic (see coreIsLinear)
    return kGoldKneeDbAt20C + linToDb(m / m20);
}

// URANIUM: a UFe10Si2-class core (28 wt % U, Tc 640 K), uranium windings, uranium-glass window, alpha decay.
static constexpr double kUraniumKneeDb = 3.7;            // [E-estimate] saturation 1.2 T, between Nickel and Steel
static constexpr double kUraniumQScale = 0.6;            // [D] softer knee than Steel (high anisotropy, lower initial permeability)
static constexpr double kUraniumFlScale = 2.0;           // [INFERRED] twice Steel's corner
static constexpr double kUraniumTauEl2 = 0.00037;        // [P] uranyl glass fluorescence persistence
static constexpr double kUraniumSlowShare = 0.20;        // [P-range] slow share of the CdS release (hardware 0.15-0.25)
// half-life release: the U-238 chain mapped to audio time, t_map = 50 ms * (t_real / 70.2 s)^0.161 [E: time axis mapped]
static constexpr double kThHalf = 0.26, kPaHalf = 0.050, kU234Half = 3.0;
// alpha decay: ~2,400 events/s for a 300 g depleted-uranium core [P]; ionisation current pulses of 1-7 ms (ion transit)
static constexpr double kAlphaRate = 2400.0;             // [P]
static constexpr double kAlphaLevelDb = -139.0;          // [P] rumble at the opto divider node, dBFS RMS (-125 dBu)
static constexpr double kAlphaExhibitionDb = 45.0;       // [E] exhibition gain on it
static constexpr double kFissionRate = 7.0 / 3600.0;     // [P] spontaneous-fission fragments reaching the node, per second
static constexpr double kFissionExhibitionRate = 0.5;    // [E]
static constexpr double kFissionScale = 15.0;            // [P] mean fragment burst relative to one alpha
static constexpr double kUraniumHeatDbPerK = -0.01;      // [P] uranium winding: 17x copper resistance, TCR +2.55e-3/K
static constexpr double kUraniumHeatK = 20.0;            // [E] full-scale program heats the winding 20 K (true: <= 0.8 K)
static constexpr double kUraniumHeatTau = 3.0, kUraniumCoolTau = 8.0;   // [D]

// GERMANIUM: vintage alloy-junction germanium devices in the Class-A stage, the detector diodes and the gain cell; ideal core.
static constexpr double kGeH2Scale = 6.0;                // [E-estimate] even-order drive against the silicon Iron stage
static constexpr double kGeLoopLpHz = 5000.0;            // [P] alpha cut-off of alloy-junction power devices, 3-7 kHz
static constexpr double kGeHfShelfDb = -1.0;             // [D] output HF shelf, -1 dB from 10 kHz (the corner is set in coreFor)
static constexpr double kGeLeakDoubleC = 9.0;            // [P] germanium leakage doubles about every 9 C
static constexpr double kGeLeakRatio25 = 0.3;            // [D] detector leak against release at 25 C (physical 2.4-43 with SHMC values)
static constexpr double kGeDieRiseC = 15.0;              // [D] Class-A die above room temperature at quiescent power
static constexpr double kGeThermalTau = 120.0;           // [P-estimate] die and heatsink warm-up
static constexpr double kGeChokeStartC = 75.0, kGeChokeFullC = 100.0;   // [P] germanium junction limits
static constexpr double kGeNoiseDb = -100.0;             // [P-estimate] added noise floor, dBFS RMS
static constexpr double kGeNoiseLpHz = 300.0;            // [P-estimate] one-pole corner standing in for the 1/f shape of germanium noise
static constexpr double kGeVcaMismatchA2 = 4.0;          // [D] gain-cell even order against the silicon cell, scales with GR depth
static constexpr double kGeElCeilingDb = -3.0;           // [P-estimate] germanium EL driver swings 3 dB less

// PLUTONIUM (hidden): delta-phase Pu-Ga windings (60x copper resistance, negative temperature coefficient) on a real Nickel core,
// a plutonium eddy-current slug, decay heat.
static constexpr double kPuFlHz = 15.0;                  // [P-direction, provisional value] winding resistance across Lm
static constexpr double kPuShelfHz = 1500.0, kPuShelfDb = -2.0;   // [ESTIMATED corner, D depth] eddy slug
static constexpr double kPuDecayHeatK = 10.0;            // [P] core runs ~10 K above room from 1.96 mW/g decay heat
static constexpr double kPuBloomDbPerK = 0.001;          // [P] conducts better when hot: +0.001 dB/K (true scale)
static constexpr double kPuBloomExhibition = 30.0;       // [E]

} // namespace mat
} // namespace hvmc
