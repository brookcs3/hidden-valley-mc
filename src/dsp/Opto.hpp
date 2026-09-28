// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The optical stage: a T4B-style shunt divider (series resistor, CdS cell to ground) whose cell is lit by an electroluminescent panel
// driven, at audio rate, from the stage's own output (feedback detection, sidechain tapped before the make-up gain; both measured on the
// reference, docs/MODEL.md, docs/opto-fix.md).
//
//   stage input x -> amplifier (x + b2 x^2 + b3 x^3) -> divider v = g x -> make-up -> out
//   sidechain: v -> [90 Hz high pass] -> second-order low pass (about 5.1 kHz) -> drive d = A_thr |v| -> light L = (d - vth)^n above
//   the turn-on vth, nothing below it (panel persistence tau_el) -> plus an idle light leak that grows with the drive squared ->
//   cell target c* = (L + L0)^gamma -> cell conductance c = sum_i w_i s_i, each state s_i chasing c* with its own attack constant and
//   a release whose rate grows with its own conductance (bimolecular recombination: ds/dt = -(s + mu s^2) / trel)
//   divider gain g = 1 / (1 + c), gain reduction 20 log10((1 + c) / (1 + c0)) with c0 the idle leak
//
// Consequences, all measured on the reference: the panel emits only above vth, so the static curve is a hard knee at
// 20 log10(vth) - A_thr dBFS with a fixed slope gamma / (1 + gamma) (about 0.62 dB per dB, a ratio near 2.6:1) above it; the
// threshold switch is a pure drive shift; the idle leak sets the small loss of no-compression gain at high threshold positions; the
// quenched release recovers in 6 to 16 ms while the signal stays above the knee and leaves a sub-second tail once it falls below;
// the low pass is what the 8 kHz static series measures. The panel responds to the instantaneous drive (it emits on both half
// cycles), so the cell keeps a ripple at twice the signal frequency; that ripple modulates the gain and is where the stage's odd
// harmonics under gain reduction come from, rising toward low frequencies as in an LA-2A-style cell.
//
// Options: light memory (a slow, exposure-dependent share of the release); the Measured Unit profile's opto-path low pass and its HF loss
// that grows with gain reduction; and the material hooks (light gain, a second persistence pole, the uranium half-life release, a
// noise injection at the divider node, a drive ceiling).
#pragma once
#include <cmath>
#include "Common.hpp"
#include "Calibration.hpp"
#include "Materials.hpp"
#include "ClassA.hpp"

namespace hvmc {

struct OptoConfig {
    int thr = 0, gain = 10;          // switch positions 0..23
    bool scFilter = true;
    bool memory = false;
    bool hwUnit = false;             // Measured Unit profile
    double lightGain = 1.0;          // material: multiplies the light
    double tauEl2 = 0.0;             // material: second persistence pole (uranium glass)
    bool halfLife = false;           // material: uranium half-life (Bateman) release on a slow share
    double driveCeiling = 1e9;       // material: EL driver swing limit (drive units)
    bool classA = false;             // CLASS A profile: the module's terms on the amplifier, its ceiling after make-up
    ClassAParams ca;
};

class OptoStage {
public:
    void prepare(double fs, const double* cal)
    {
        fsr = fs; c = cal;
        sc.set(FirstOrder::kHighPass, c[kc_sc_hz], fs);
        kEl = onePoleK(c[kc_o_tau_el], fs);
        for (int s = 0; s < kOptoStates; ++s) kAtt[s] = onePoleK(c[kc_o_tatt + s], fs);
        kMemAtt = onePoleK(c[kc_o_tatt + 1], fs); kMemRel = onePoleK(c[kc_o_mem_trel], fs); kMem = onePoleK(c[kc_o_mem_tm], fs);
        const double ln2 = 0.6931471805599453;
        kTh = 1.0 - std::exp(-ln2 / (mat::kThHalf * fs)); kPa = 1.0 - std::exp(-ln2 / (mat::kPaHalf * fs)); kU = 1.0 - std::exp(-ln2 / (mat::kU234Half * fs));
        for (int s = 0; s < kOptoStates; ++s) kRelBase[s] = 1.0 / (c[kc_o_trel + s] * fs);
        scLp.setMatchedLowPass(c[kc_o_sc_lp_hz], c[kc_o_sc_lp_q], fs);
        thrS.set(0.015, fs); gainS.set(0.015, fs); leakS.set(0.015, fs); caStage.prepare(fs);
        hwLp.set(FirstOrder::kLowPass, c[kc_o_hw_lp_hz], fs);
        configured = false;   // the first configure() after a prepare() resets the stage to the settings it is given
        reset();
    }

    void reset()
    {
        sc.reset(); hwLp.reset(); grLoss.reset(); scLp.reset(); caStage.reset();
        leakTarget = leakLightFor(cfg.thr); leakS.reset(leakTarget); leakL = leakTarget;
        // at rest the cell sits on the idle leak: no start-up transient at the divider
        const double c0 = std::pow(leakTarget, c[kc_o_gamma]);
        L = L2 = 0.0; for (int s = 0; s < kOptoStates; ++s) st[s] = c0;
        mem = expo = 0.0; n1 = n2 = n3 = 0.0; cond = c0; gLast = 1.0 / (1.0 + c0);
        thrS.reset(c[kc_o_thr_db + cfg.thr]); gainS.reset(c[kc_o_gain_db + cfg.gain]);
        grLossFc = -1.0;
    }

    void configure(const OptoConfig& nc)
    {
        const bool first = !configured;
        cfg = nc; configured = true;
        kEl2 = cfg.tauEl2 > 0.0 ? onePoleK(cfg.tauEl2, fsr) : 1.0;
        leakTarget = leakLightFor(cfg.thr);
        caStage.set(cfg.ca);
        if (first) reset();
    }

    // the idle light for a threshold position: cond0 = o_leak (A / A20)^o_leak_q, entered before the cell law as L0 = cond0^(1/gamma)
    double leakLightFor(int thr) const
    {
        const double ratioDb = c[kc_o_thr_db + thr] - c[kc_o_thr_db + 19];
        const double cond0 = c[kc_o_leak] * std::pow(10.0, c[kc_o_leak_q] * ratioDb / 20.0);
        return cond0 > 0.0 ? std::pow(cond0, 1.0 / c[kc_o_gamma]) : 0.0;
    }

    // one sample; `noise` is added at the divider node (material decay noise), in signal units
    inline double process(double x, double noise = 0.0)
    {
        double b2 = c[kc_o_b2], b3 = c[kc_o_b3];
        if (cfg.classA) { caStage.track(x); b2 += caStage.a2Now(); b3 += caStage.a3(); }   // the Class-A module's terms on the amplifier
        const double xa = x + b2 * x * x + b3 * x * x * x;
        const double g = 1.0 / (1.0 + cond);
        double v = g * xa + noise;
        if (cfg.hwUnit) {
            v = hwLp.tick(v);
            // HF loss that grows with gain reduction: the cell's capacitance shunts more as its resistance falls
            const double fc = c[kc_o_hw_grloss_hz] / (1.0 + c[kc_o_hw_grloss_k] * cond);
            if (std::fabs(fc - grLossFc) > 1.0) { grLoss.set(FirstOrder::kLowPass, fc, fsr); grLossFc = fc; }
            v = grLoss.tick(v);
        }
        gLast = g;
        // sidechain from the divider output (feedback), before the make-up
        const double hp = sc.tick(v);   // the filter runs either way, so switching it is click-free
        const double s = cfg.scFilter ? hp : v;
        const double s2 = scLp.tick(s);   // the sidechain's own low pass, after the high pass
        const double A = dbToLin(thrS.tick(c[kc_o_thr_db + cfg.thr]));
        double d = std::fabs(A * s2);
        if (d > cfg.driveCeiling) d = cfg.driveCeiling;
        const double e = d - c[kc_o_vth];   // hard turn-on
        const double Linst = e > 0.0 ? std::pow(e, c[kc_o_n]) * cfg.lightGain : 0.0;
        L += (Linst - L) * kEl;
        L2 += (L - L2) * kEl2;
        leakL = leakS.tick(leakTarget);
        const double target = std::pow(L2 + leakL, c[kc_o_gamma]);   // the leak enters as light: the target is positive at idle
        double fast = 0.0;
        for (int i = 0; i < kOptoStates; ++i) {
            const double k = target > st[i] ? kAtt[i] : 1.0 - std::exp(-(1.0 + c[kc_o_rel_mu] * st[i]) * kRelBase[i]);   // quenched release
            st[i] += (target - st[i]) * k;
            fast += c[kc_o_w + i] * st[i];
        }
        if (cfg.halfLife) {
            // uranium: a three-member decay chain (Th-234 -> Pa-234m -> U-234) carries the slow share; in equilibrium each member's
            // activity equals the source, so the static value is unchanged and only the release is shaped
            // per sample: the source feeds Th-234 (src atoms), each member decays a fraction k of its atoms into the next; a member's
            // decays per sample are its activity, and in equilibrium every activity equals src
            const double beta = mat::kUraniumSlowShare, src = beta * fast;
            const double a1 = n1 * kTh, a2 = n2 * kPa, a3 = n3 * kU;
            n1 += src - a1;
            n2 += a1 - a2;
            n3 += a2 - a3;
            cond = (1.0 - beta) * fast + (a1 + a2 + a3) / 3.0;
        } else if (cfg.memory) {
            const double beta = c[kc_o_mem_beta];
            const double k = target > mem ? kMemAtt : kMemRel / (1.0 + c[kc_o_mem_mu] * expo);
            mem += (target - mem) * k;
            expo += ((target > 1.0 ? 1.0 : target) - expo) * kMem;
            cond = (1.0 - beta) * fast + beta * mem;
        } else {
            cond = fast;
        }
        const double out = v * dbToLin(gainS.tick(c[kc_o_gain_db + cfg.gain]));
        return cfg.classA ? caStage.ceiling(out) : out;   // the module's ceiling, outside the loop
    }

    // gain reduction of the divider above its idle leak, dB (what the meter cell reads, calibrated to the audio cell; zero at idle)
    double grDb() const { return linToDb((1.0 + cond) / (1.0 + std::pow(leakL, c[kc_o_gamma]))); }

private:
    OptoConfig cfg;
    bool configured = false;
    const double* c = nullptr;
    double fsr = 48000.0;
    FirstOrder sc, hwLp, grLoss;
    Biquad scLp;
    double grLossFc = -1.0;
    Smoother thrS, gainS, leakS;
    ClassAStage caStage;
    double leakTarget = 0.0, leakL = 0.0;
    double kEl = 1.0, kEl2 = 1.0, kAtt[kOptoStates] = {}, kRelBase[kOptoStates] = {}, kMemAtt = 1.0, kMemRel = 1.0, kMem = 1.0;
    double kTh = 0.0, kPa = 0.0, kU = 0.0;
    double L = 0.0, L2 = 0.0, st[kOptoStates] = {}, mem = 0.0, expo = 0.0, n1 = 0.0, n2 = 0.0, n3 = 0.0, cond = 0.0, gLast = 1.0;
};

} // namespace hvmc
