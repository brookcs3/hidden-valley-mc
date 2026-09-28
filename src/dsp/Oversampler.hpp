// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// 2x up- and down-sampling for the HQ mode: a linear-phase 79-tap half-band FIR, designed with the Parks-McClellan (Remez) algorithm
// (passband 0-20 kHz and stopband from 28 kHz at a 96 kHz internal rate, i.e. 0.2083 and 0.2917 of the internal rate), with the
// half-band structure imposed exactly: centre tap 1/2, the other taps at even offsets from the centre zero. Passband ripple 0.00007 dB,
// stopband -108 dB. A half-band filter needs only its 40 odd-offset taps per rate change; each of the two filters delays by 39 internal
// samples, so the round trip delays by 39 samples at the host rate, which the plugin reports as its latency in HQ mode.
// (Design script: fit/tools/halfband.py; its output is pasted below.)
#pragma once

namespace hvmc {

static constexpr int kHbHalf = 20;              // unique odd-offset taps (the filter is symmetric)
static constexpr int kHbTaps = 2 * kHbHalf;     // odd-offset taps in all
static constexpr int kOversampleLatency = 39;   // host-rate samples, up + down
static constexpr double kHbCoef[kHbHalf] = {
    -9.81219801604945e-06,
    3.1341509067166537e-05,
    -7.8502241313082718e-05,
    0.00016856705368511956,
    -0.00032617961916305585,
    0.00058470078761872323,
    -0.00098707769957056884,
    0.0015874840781196532,
    -0.0024524800885871532,
    0.0036638060113515464,
    -0.0053224973791969705,
    0.0075581470196502713,
    -0.010546534162111465,
    0.014547385144035528,
    -0.019985642526544639,
    0.027646165769623997,
    -0.039204687973462035,
    0.05903752060497456,
    -0.10326774090594422,
    0.31735437054863891
};

struct HalfbandUp {
    double line[kHbTaps] = {};
    double center[kHbHalf] = {};   // delay line for the pure-delay phase (x[m - 19])
    int pos = 0, cpos = 0;
    void reset() { for (double& v : line) v = 0.0; for (double& v : center) v = 0.0; pos = cpos = 0; }
    // one input sample -> two output samples at twice the rate (gain 1)
    inline void process(double x, double& y0, double& y1)
    {
        line[pos] = x;
        double acc = 0.0;
        for (int i = 0; i < kHbTaps; ++i) {
            int idx = pos - i; if (idx < 0) idx += kHbTaps;
            const double c = i < kHbHalf ? kHbCoef[i] : kHbCoef[kHbTaps - 1 - i];
            acc += c * line[idx];
        }
        pos = pos + 1 == kHbTaps ? 0 : pos + 1;
        y0 = 2.0 * acc;
        // odd phase: the centre tap, a delay of 19 input samples
        const double d = center[cpos];
        center[cpos] = x;
        cpos = cpos + 1 == 19 ? 0 : cpos + 1;
        y1 = d;
    }
};

struct HalfbandDown {
    double even[kHbTaps] = {};
    double odd[kHbHalf + 1] = {};
    int pos = 0, opos = 0;
    void reset() { for (double& v : even) v = 0.0; for (double& v : odd) v = 0.0; pos = opos = 0; }
    // two input samples at twice the rate -> one output sample
    inline double process(double w0, double w1)
    {
        even[pos] = w0;
        double acc = 0.0;
        for (int i = 0; i < kHbTaps; ++i) {
            int idx = pos - i; if (idx < 0) idx += kHbTaps;
            const double c = i < kHbHalf ? kHbCoef[i] : kHbCoef[kHbTaps - 1 - i];
            acc += c * even[idx];
        }
        pos = pos + 1 == kHbTaps ? 0 : pos + 1;
        // centre tap: w[2m - 39] = the odd sample from 20 input pairs ago
        const double d = odd[opos];
        odd[opos] = w1;
        opos = opos + 1 == kHbHalf ? 0 : opos + 1;
        return acc + 0.5 * d;
    }
};

} // namespace hvmc
