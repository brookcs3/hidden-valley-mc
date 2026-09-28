// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The discrete stage: a feed-forward compressor around a discrete gain cell (Blackmer lineage). Measured on the reference (docs/MODEL.md,
// docs/detector-fix.md, docs/disc-knee-fix.md):
//  - detector: full-wave rectifier -> log -> gain computer -> one storage node in dB of gain reduction (the timing capacitor on the
//    control voltage): a bleed to zero, always conducting, and an attack diode whose conductance grows with the drive. The knee of
//    each ratio is the zero of its own curve, silence is zero gain reduction at every ratio, and the release is one trajectory in dB
//    of gain at every ratio (all measured). On a sine the node settles under the within-cycle ripple of the target, which is why the
//    steady gain reduction falls as the attack is made slower and why the attack time constant depends on the recover position.
//    Chosen by residual among some sixty candidate topologies in four harnesses and a second round on the soft ratios
//    (fit/tools/candidates/). DUAL couples a second, larger storage node: a fast partial release, then a multi-second tail.
//  - threshold: a pure offset of the detector level (about 2.7 dB per step);
//  - ratio: each position is a fixed, progressive curve of gain reduction against the level above threshold (a table fitted to the
//    reference through the node; the panel labels are nominal, and 4:1 and FLOOD over-compress). The per-ratio attack tables
//    (d_att_scale, d_att_a0, d_att_sat_db) are hooks for the ratio-dependent attack network measured in docs/disc-ballistics.md,
//    carried at the identity until refitted under this node;
//  - stereo: the two channels' sidechain signals are summed before the rectifier (anti-phase content does not compress);
//  - gain cell: even- and odd-order terms on its input (so its distortion does not fall with gain reduction), then the gain, then make-up.
#pragma once
#include <cmath>
#include "Common.hpp"
#include "Calibration.hpp"
#include "ClassA.hpp"

namespace hvmc {

struct DiscreteConfig {
    int thr = 0, ratio = 0, attack = 5, recover = 0, gain = 6;
    bool scFilter = true;
    double scHz = 0.0;            // sidechain high-pass corner (0: the calibration's sc_hz)
    bool hwUnit = false;
    double leakRatio = 0.0;      // material (germanium): detector leakage against the release conductance
    double a2Extra = 0.0;        // material (germanium): gain-cell mismatch, even order, per 10 dB of gain reduction
    bool classA = false;         // CLASS A profile: the output module after make-up, the cell's noise, the cell's even-term scale
    ClassAParams ca;
    double a2Scale = 1.0, noiseDb = -400.0;
};

// monotone cubic (Fritsch-Carlson) interpolation of one gain-reduction curve on the uniform grid
struct Curve {
    double y[kCurveN], m[kCurveN];
    void build(const double* src)
    {
        double d[kCurveN - 1];
        for (int i = 0; i < kCurveN; ++i) y[i] = src[i];
        for (int i = 0; i < kCurveN - 1; ++i) d[i] = (y[i + 1] - y[i]) / kCurveDx;
        m[0] = d[0]; m[kCurveN - 1] = d[kCurveN - 2];
        for (int i = 1; i < kCurveN - 1; ++i) m[i] = (d[i - 1] * d[i] <= 0.0) ? 0.0 : 0.5 * (d[i - 1] + d[i]);
        for (int i = 0; i < kCurveN - 1; ++i) {
            if (d[i] == 0.0) { m[i] = m[i + 1] = 0.0; continue; }
            const double a = m[i] / d[i], b = m[i + 1] / d[i], s = a * a + b * b;
            if (s > 9.0) { const double t = 3.0 / std::sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
        }
    }
    inline double at(double x) const
    {
        const double u = (x - kCurveX0) / kCurveDx;
        if (u <= 0.0) return y[0];
        if (u >= double(kCurveN - 1)) return y[kCurveN - 1] + m[kCurveN - 1] * (x - (kCurveX0 + kCurveDx * (kCurveN - 1)));
        const int i = int(u);
        const double t = u - i, t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * kCurveDx * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * kCurveDx * m[i + 1];
    }
};

class DiscreteStage {
public:
    void prepare(double fs, const double* cal)
    {
        fsr = fs; c = cal;
        sc.set(FirstOrder::kHighPass, cfg.scHz > 0.0 ? cfg.scHz : c[kc_sc_hz], fs);
        for (int r = 0; r < kRatios; ++r) curves[r].build(c + kc_d_curve + r * kCurveN);
        floorDb = c[kc_d_floor_db];
        floorLin = dbToLin(floorDb);
        k2 = onePoleK(c[kc_d_dual_t2], fs);
        c2 = c[kc_d_dual_c2] > 1e-6 ? c[kc_d_dual_c2] : 1e-6;
        invSv = c[kc_d_att_sv_db] > 1e-3 ? 1.0 / c[kc_d_att_sv_db] : 0.0;
        thrS.set(0.015, fs); gainS.set(0.015, fs); goffS.set(0.015, fs); caStage.prepare(fs);
        fadeLen = int(0.020 * fs); if (fadeLen < 8) fadeLen = 8;
        setTimes();
        configured = false;   // the first configure() after a prepare() resets the stage to the settings it is given
        reset();
    }

    void reset()
    {
        sc.reset(); caStage.reset();
        g = w = 0.0;   // at rest the node holds no gain reduction: exact silence at every ratio
        gr = 0.0; fadePos = fadeLen; prevRatio = cfg.ratio; pendingRatio = -1;
        thrS.reset(c[kc_d_thr_db + cfg.thr]); gainS.reset(c[kc_d_gain_db + cfg.gain]); goffS.reset(c[kc_d_goff_db + cfg.gain]);
    }

    void configure(const DiscreteConfig& next)
    {
        DiscreteConfig nc = next;
        const bool first = !configured;
        if (!first && nc.ratio != cfg.ratio) {
            if (fadePos < fadeLen) { pendingRatio = nc.ratio; nc.ratio = cfg.ratio; }   // a change during a fade waits for it
            else { prevRatio = cfg.ratio; fadePos = 0; }
        }
        if (!first && nc.recover != cfg.recover) w = g;   // the second node starts level with the first when DUAL is switched
        if (c && nc.scHz != cfg.scHz) sc.set(FirstOrder::kHighPass, nc.scHz > 0.0 ? nc.scHz : c[kc_sc_hz], fsr);   // retune, state kept
        cfg = nc; configured = true;
        setTimes();
        caStage.set(cfg.ca); noiseAmp = cfg.classA && cfg.noiseDb > -300.0 ? dbToLin(cfg.noiseDb) : 0.0;
        if (first) reset();
    }

    void seed(uint64_t s) { noise.seed(s); }
    bool filterIn() const { return cfg.scFilter; }

    // the channel's sidechain signal (the engine sums the two channels' in stereo before calling process)
    inline double sidechain(double u)
    {
        const double hp = sc.tick(u);   // runs either way: switching is click-free
        return cfg.scFilter ? hp : u;
    }

    inline double process(double u, double h)
    {
        // detector: log rectifier, gain computer, then the storage node in dB of gain reduction
        const double a = std::fabs(h);
        const double e = (a > floorLin ? linToDb(a) : floorDb) + goffS.tick(c[kc_d_goff_db + cfg.gain]);
        const double thr = thrS.tick(c[kc_d_thr_db + cfg.thr]);
        const double x = e - thr;
        double gt = curves[cfg.ratio].at(x);   // the target gain reduction: the ratio position's curve
        if (fadePos < fadeLen) {
            const double t = double(fadePos++) / fadeLen, wf = 0.5 - 0.5 * std::cos(kPi * t);
            gt = curves[prevRatio].at(x) * (1.0 - wf) + gt * wf;
            if (fadePos >= fadeLen && pendingRatio >= 0) { prevRatio = cfg.ratio; cfg.ratio = pendingRatio; pendingRatio = -1; fadePos = 0; setTimes(); }
        }
        if (gt < 0.0) gt = 0.0;
        double dg = -g * kR;   // the bleed toward zero gain reduction, always on
        if (gt > g) {          // the attack diode conducts
            const double f = a0 + gt * invSv;                            // its conductance grows with the drive (the ratio's intercept plus the law)
            const double gap = gt - g;
            const double gs = invSa > 0.0 ? gap / (1.0 + gap * invSa) : gap;   // current limit, where a ratio position has one
            dg += gs * (1.0 - std::exp(-rA * f));                       // the exact one-pole step for a rate f / ta: never overshoots gt
        }
        g += dg;
        if (cfg.recover == kRecovers - 1) {   // DUAL: the second node
            const double flow = (g - w) * k2;
            g -= flow;
            w += flow / c2;
        }
        if (kLeak > 0.0) g -= g * kLeak;      // material option (germanium): an extra bleed
        if (g < 1e-7) g = 0.0;                // 1e-7 dB is inaudible; silence is exactly zero
        gr = g;
        // gain cell
        const double a2 = (cfg.hwUnit ? c[kc_d_hwunit_a2] : c[kc_d_a2]) * cfg.a2Scale + cfg.a2Extra * (g * 0.1);
        const double a3 = c[kc_d_a3];
        const double uf = a3 < 0.0 ? 1.0 / std::sqrt(-3.0 * a3) : 1e30;   // beyond the polynomial's fold it clips instead of inverting
        const double uc = u > uf ? uf : (u < -uf ? -uf : u);
        const double ui = uc + a2 * uc * uc + a3 * uc * uc * uc;
        const double cell = ui * dbToLin(-g) + (noiseAmp > 0.0 ? noise.gauss() * noiseAmp : 0.0);   // the cell, then its noise
        const double out = cell * dbToLin(gainS.tick(c[kc_d_gain_db + cfg.gain]));
        return cfg.classA ? caStage.tick(out) : out;   // CLASS A: the output module after make-up
    }

    double grDb() const { return gr; }

private:
    void setTimes()
    {
        if (!c) return;
        const double trel = c[kc_d_trel + cfg.recover];
        rA = 1.0 / (c[kc_d_tatt + cfg.attack] * c[kc_d_att_scale + cfg.ratio] * fsr);
        a0 = c[kc_d_att_a0 + cfg.ratio];
        invSa = c[kc_d_att_sat_db + cfg.ratio] > 1e-3 ? 1.0 / c[kc_d_att_sat_db + cfg.ratio] : 0.0;
        kR = onePoleK(trel, fsr);
        kLeak = cfg.leakRatio > 0.0 ? onePoleK(trel / cfg.leakRatio, fsr) : 0.0;
    }

    DiscreteConfig cfg;
    bool configured = false;
    const double* c = nullptr;
    double fsr = 48000.0;
    FirstOrder sc;
    Curve curves[kRatios];
    Smoother thrS, gainS, goffS;
    ClassAStage caStage;
    Noise noise;
    double noiseAmp = 0.0;
    double floorDb = -80.0, floorLin = 1e-4, rA = 1.0, invSv = 0.0, a0 = 1.0, invSa = 0.0, kR = 1.0, k2 = 0.0, c2 = 1.0, kLeak = 0.0;
    double g = 0.0, w = 0.0, gr = 0.0;
    int prevRatio = 0, fadePos = 0, fadeLen = 960, pendingRatio = -1;
};

} // namespace hvmc
