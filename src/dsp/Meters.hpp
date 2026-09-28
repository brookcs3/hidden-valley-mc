// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// Meter ballistics, published as the plugin's read-only output parameters (there is no GUI of its own).
//  - VU: full-wave rectified output through a second-order low pass (damping 0.81, natural frequency 2.05 Hz), which rises to 99 % of
//    a step in about 300 ms with about 1 % overshoot (IEC 60268-17); scaled so a steady sine reads its RMS level; 0 VU = a sine at
//    -14 dBFS peak (the reference's default: 0 dBu = -14 dBFS).
//  - Gain reduction: the stages' own values, through the same needle ballistics.
//  - Magic eye (VT-138): the mono output's peak, 1 ms attack and 300 ms release, in dBFS.
#pragma once
#include <cmath>
#include "Common.hpp"

namespace hvmc {

struct Needle {
    Biquad lp;
    double last = 0.0;
    void prepare(double fs)
    {
        const double f0 = 2.05, zeta = 0.81;
        lp.setLowPass(f0, 1.0 / (2.0 * zeta), fs);
        lp.reset(); last = 0.0;
    }
    inline double tick(double x) { last = lp.tick(x); return last; }
};

struct VuMeter {
    Needle n;
    double value = 0.0;
    void prepare(double fs) { n.prepare(fs); value = 0.0; }
    inline void tick(double x) { value = n.tick(std::fabs(x)); }
    // VU: average-responding, sine-calibrated (mean |sin| = 2/pi of the peak); 0 VU = a sine of refDbfs peak (default -14 dBFS)
    double vu(double refDbfs = -14.0) const
    {
        const double ref = 0.6366197723675814 * dbToLin(refDbfs);
        return linToDb((value > 1e-12 ? value : 1e-12) / ref);
    }
};

struct PeakMeter {
    double value = 0.0, ka = 1.0, kr = 1.0;
    void prepare(double fs) { ka = onePoleK(0.001, fs); kr = onePoleK(0.3, fs); value = 0.0; }
    inline void tick(double x) { const double a = std::fabs(x); value += (a - value) * (a > value ? ka : kr); }
    double db() const { return linToDb(value > 1e-7 ? value : 1e-7); }
};

} // namespace hvmc
