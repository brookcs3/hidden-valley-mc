// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The whole stereo unit. Signal order per channel (docs/MODEL.md):
//
//   in -> [optical stage] -> [discrete stage] -> transformer (Nickel / Iron / Steel / material) -> wet
//   out = dry + hardwire * (dry + mix * (wet - dry) - dry)        (dry = the untouched input, delayed by the latency in HQ)
//
// Stage bypass switches, the hardwire bypass and the mix are crossfaded (10 ms), the make-up and threshold switches are smoothed, and
// transformer changes crossfade between two cores (the new one warmed on the last 80 ms of input). In STEREO the left-channel controls
// govern both channels (all of them except METER SELECT, as on the hardware) and the discrete detectors are linked; the optical
// detectors are not (measured on the reference). In HQ the channel DSP runs at twice the host rate between two half-band filters.
//
// The same class runs in the plugin, in the unit tests and, through src/capi/, in the Python fitting and tests.
#pragma once
#include <cmath>
#include <cstring>
#include "../HVMCParams.hpp"
#include "Common.hpp"
#include "Calibration.hpp"
#include "Materials.hpp"
#include "Opto.hpp"
#include "Discrete.hpp"
#include "Transformer.hpp"
#include "Oversampler.hpp"
#include "Meters.hpp"

namespace hvmc {

// Poisson impulses through a two-pole low pass: the ionisation current of alpha decay near a high-impedance node (uranium)
struct DecayNoise {
    Noise rng;
    double p = 0.0, pf = 0.0, amp = 0.0, famp = 0.0, k = 1.0, s1 = 0.0, s2 = 0.0;
    bool on = false;
    void configure(double fs, double rate, double rmsLin, double fissionRate, uint64_t seed)
    {
        rng.seed(seed);
        on = rmsLin > 0.0 && rate > 0.0;
        p = rate / fs; pf = fissionRate / fs;
        k = onePoleK(1.0 / (kTwoPi * 60.0), fs);   // ion transit 1-7 ms: about 25-150 Hz
        // power gain of the two cascaded one-poles (h[n] = k^2 (n+1) r^n, r = 1 - k): sum h^2 = k^4 (1 + r^2) / (1 - r^2)^3
        const double rr = (1.0 - k) * (1.0 - k);
        const double g = k * k * k * k * (1.0 + rr) / ((1.0 - rr) * (1.0 - rr) * (1.0 - rr));
        // flat charge distribution in [0, 1]: E[s^2] = 1/3
        amp = on ? rmsLin / std::sqrt(p * g / 3.0) : 0.0;
        famp = mat::kFissionScale * 0.5 * amp;
        s1 = s2 = 0.0;
    }
    inline double tick()
    {
        if (!on) return 0.0;
        double x = 0.0;
        if (rng.uniform() < p) x = (rng.uniform() < 0.5 ? -1.0 : 1.0) * amp * rng.uniform();
        if (pf > 0.0 && rng.uniform() < pf) x += (rng.uniform() < 0.5 ? -1.0 : 1.0) * famp * (0.5 + 0.5 * rng.uniform());
        s1 += (x - s1) * k; s2 += (s1 - s2) * k;
        return s2;
    }
};

class Engine {
public:
    Engine()
    {
        std::memcpy(cal, calDefaults().v, sizeof(cal));
        for (int i = 0; i < kNumInputParams; ++i) ctl[i] = paramDefault(i);
    }
    Engine(const Engine&) = delete;               // the stages keep pointers into this object's own tables
    Engine& operator=(const Engine&) = delete;

    // replace the calibration (the fitter does this); takes effect at the next prepare()
    void setCalibration(const double* v) { std::memcpy(cal, v, sizeof(cal)); if (prepared) prepare(hostFs); }   // the stages read cal in place
    const double* calibration() const { return cal; }

    void prepare(double fs)
    {
        hostFs = fs;
        quality = ctl[kGQuality];
        os = quality ? 2 : 1;
        const double fi = fs * os;
        for (int c = 0; c < 2; ++c) {
            Ch& h = ch[c];
            h.opto.prepare(fi, cal);
            h.disc.prepare(fi, cal); h.disc.seed(0xC1A5 + 613ull * uint64_t(c));
            h.xf.prepare(fi, 0x51ED + 977ull * uint64_t(c));
            h.up.reset(); h.keyUp.reset(); h.down.reset();
            h.optoIn.set(0.010, fi); h.discIn.set(0.010, fi); h.scfTrim.set(0.015, fi); h.scfTrim.reset(1.0);
            h.caHp.reset(); h.caLp.reset();
            h.vu.prepare(fs); h.grO.prepare(fs); h.grD.prepare(fs);
            h.grOv = h.grDv = 0.0;
            h.uranium = false;   // forces the decay-noise generator to be set up again for this rate
        }
        mixS.set(0.010, fs); hwS.set(0.010, fs);
        eye.prepare(fs);
        for (int i = 0; i < kDelayMax; ++i) dry[0][i] = dry[1][i] = 0.0;
        dpos = 0;
        prepared = true; first = true;
        apply();
    }

    void reset() { if (prepared) prepare(hostFs); }

    // value = position index (0 .. steps-1)
    void setParam(int index, int value)
    {
        if (index < 0 || index >= kNumInputParams) return;
        const int steps = paramSteps(index);
        if (value < 0) value = 0;
        if (value >= steps) value = steps - 1;
        if (ctl[index] != value) { ctl[index] = value; dirty = true; }
    }
    int param(int index) const { return (index >= 0 && index < kNumInputParams) ? ctl[index] : 0; }

    int latency() const { return os == 2 ? kOversampleLatency : 0; }
    int latencyFor(int q) const { return q ? kOversampleLatency : 0; }

    // keyL / keyR: the external key (sidechain) input, or null when the host has none (KEY IN then hears silence)
    void process(const float* inL, const float* inR, float* outL, float* outR, int n, const float* keyL = nullptr, const float* keyR = nullptr)
    {
        if (!prepared) prepare(48000.0);
        if (ctl[kGQuality] != quality) prepare(hostFs);   // the resamplers change: restart (the host is told the new latency)
        if (dirty) apply();
        const int L = latency();
        for (int i = 0; i < n; ++i) {
            // a non-finite or absurd input sample must not poison the stages for the rest of the session
            double x[2] = { double(inL[i]), double(inR[i]) };
            for (int c = 0; c < 2; ++c) { if (!std::isfinite(x[c])) x[c] = 0.0; else if (x[c] > 64.0) x[c] = 64.0; else if (x[c] < -64.0) x[c] = -64.0; }
            double kx[2] = { keyL ? double(keyL[i]) : 0.0, keyR ? double(keyR[i]) : 0.0 };
            for (int c = 0; c < 2; ++c) { if (!std::isfinite(kx[c])) kx[c] = 0.0; else if (kx[c] > 64.0) kx[c] = 64.0; else if (kx[c] < -64.0) kx[c] = -64.0; }
            double wet[2];
            if (os == 1) {
                internal(x, wet, kx);
            } else {
                double u0[2], u1[2], w0[2], w1[2], k0[2], k1[2];
                for (int c = 0; c < 2; ++c) { ch[c].up.process(x[c], u0[c], u1[c]); ch[c].keyUp.process(kx[c], k0[c], k1[c]); }
                internal(u0, w0, k0);
                internal(u1, w1, k1);
                for (int c = 0; c < 2; ++c) wet[c] = ch[c].down.process(w0[c], w1[c]);
            }
            double m = mixS.tick(ctl[kGMix] * 0.01), hw = hwS.tick(ctl[kGHardwire] ? 1.0 : 0.0);
            if (m < 1e-9) m = 0.0;
            if (hw < 1e-9) hw = 0.0;
            double y[2];
            for (int c = 0; c < 2; ++c) {
                double d;
                if (L == 0) d = x[c];
                else { d = dry[c][(dpos - L) & (kDelayMax - 1)]; dry[c][dpos] = x[c]; }
                if (!std::isfinite(wet[c])) { recover(c); wet[c] = d; }   // should never happen; if it does, the channel restarts
                const double processed = m > 0.0 ? d + m * (wet[c] - d) : d;   // a bypass is a bypass, whatever the wet path holds
                y[c] = hw > 0.0 ? d + hw * (processed - d) : d;
                ch[c].vu.tick(y[c]);
                ch[c].grOv = ch[c].opto.grDb(); ch[c].grDv = ch[c].disc.grDb();   // once per host sample: the needles are slow
                ch[c].grO.tick(ch[c].grOv); ch[c].grD.tick(ch[c].grDv);
            }
            if (L) dpos = (dpos + 1) & (kDelayMax - 1);
            eye.tick(0.5 * (y[0] + y[1]));
            outL[i] = float(y[0]); outR[i] = float(y[1]);
        }
        first = false;
    }

    // read-only meters, in the order of OutputParam
    double meter(int outIndex) const
    {
        const int c = (outIndex >= kOMeterR && outIndex <= kOGrDiscreteR) ? 1 : 0;
        const Ch& h = ch[c];
        switch (outIndex) {
        case kOMeterL: case kOMeterR: {
            const int sel = ctl[c == 0 ? kPMeterSelect : kPerChannel + kPMeterSelect];
            if (sel == kMeterOptical) return -h.grO.last;
            if (sel == kMeterDiscrete) return -h.grD.last;
            return h.vu.vu(kVuRefDbfs[ctl[kGVuRef]]);
        }
        case kOGrOpticalL: case kOGrOpticalR: return -h.grO.last;
        case kOGrDiscreteL: case kOGrDiscreteR: return -h.grD.last;
        case kOMagicEye: return eye.db();
        }
        return 0.0;
    }

private:
    static constexpr int kDelayMax = 64;

    struct Ch {
        Smoother scfTrim;
        FirstOrder caHp, caLp;
        OptoStage opto;
        DiscreteStage disc;
        TransformerBlock xf;
        HalfbandUp up, keyUp;
        HalfbandDown down;
        Smoother optoIn, discIn;
        DecayNoise decay;
        Needle grO, grD;
        VuMeter vu;
        double grOv = 0.0, grDv = 0.0;
        bool uranium = false;
    };

    // one sample at the internal rate for both channels
    inline void internal(const double* x, double* out, const double* key)
    {
        const bool keyOn = ctl[kGKeyIn] != 0;
        double a[2], ad[2], h[2], wo_[2];
        for (int c = 0; c < 2; ++c) {
            Ch& k = ch[c];
            const double noise = k.uranium ? k.decay.tick() : 0.0;
            double xin = x[c] * caIn;   // CLASS A: the hotter input, ahead of both compressors (1.0 otherwise)
            xin = k.caLp.tick(k.caHp.tick(xin));   // CLASS A input transformer corners (pass-through unless set)
            const double yo = k.opto.process(xin, noise, keyOn, key[c]);
            const double wo = k.optoIn.tick(optoOn[c] ? 1.0 : 0.0); wo_[c] = wo;
            a[c] = xin + wo * (yo - xin);
            // the discrete stage's input: the inter-stage gain when both stages are in (following the optical crossfade), and the
            // SIDECHAIN FILTER's trim, both on audio and sidechain alike
            const double gTrim = k.scfTrim.tick(k.disc.filterIn() ? scfLin : 1.0);
            const double gi = wo * interLin + (1.0 - wo) * gTrim;   // the two do not add: with the optical stage in the filter changes nothing here
            ad[c] = a[c] * gi;
            h[c] = k.disc.sidechain(keyOn ? key[c] : ad[c]);   // KEY IN: the key replaces the detector's own signal, before the filter and the stereo sum
        }
        if (stereo) { const double s = cal[kc_d_link] * (h[0] + h[1]); h[0] = h[1] = s; }
        for (int c = 0; c < 2; ++c) {
            Ch& k = ch[c];
            const double yd = k.disc.process(ad[c], h[c]);
            const double wd = k.discIn.tick(discOn[c] ? 1.0 : 0.0);
            const double b = a[c] + wd * (yd - a[c]);
            // after the transformer, the optical cell's shunted signal bleeds through at o_bleed (measured 1/200), following the
            // optical bypass crossfade and bypassing both make-ups (docs/stage-interaction.md)
            out[c] = (k.xf.process(b) + wo_[c] * bleed * k.opto.shunted()) * caOut;
        }
    }

    // restart one channel's stages at their current settings (after a non-finite sample slipped through)
    void recover(int c)
    {
        Ch& h = ch[c];
        h.opto.reset(); h.disc.reset(); h.xf.reset(); h.up.reset(); h.keyUp.reset(); h.down.reset();
        h.caHp.reset(); h.caLp.reset();
        h.optoIn.reset(optoOn[c] ? 1.0 : 0.0); h.discIn.reset(discOn[c] ? 1.0 : 0.0);
    }

    // controls -> stage configurations (not per sample)
    void apply()
    {
        dirty = false;
        stereo = ctl[kGStereo] != 0;
        const int profile = ctl[kGProfile];
        const double roomC = double(ctl[kGTemperature] + kTempMin);
        const bool exhibition = ctl[kGExhibition] != 0;
        const double fi = hostFs * os;
        const bool classA = profile == kProfileClassA;
        ClassAParams ca;
        ca.a2 = cal[kc_ca_a2]; ca.a3 = cal[kc_ca_a3]; ca.a2Env = cal[kc_ca_a2_env]; ca.ceilDb = cal[kc_ca_ceil_db]; ca.q = cal[kc_ca_ceil_q]; ca.asymDb = cal[kc_ca_ceil_asym_db];
        caIn = classA ? dbToLin(cal[kc_ca_in_db]) : 1.0; caOut = classA ? dbToLin(cal[kc_ca_out_db]) : 1.0;
        interLin = dbToLin(cal[kc_d_inter_db]); scfLin = dbToLin(cal[kc_d_scf_trim_db]); bleed = cal[kc_o_bleed];
        for (int c = 0; c < 2; ++c) {
            ch[c].caHp.set(FirstOrder::kHighPass, classA ? cal[kc_ca_in_fl_hz] : 0.0, fi);
            ch[c].caLp.set(FirstOrder::kLowPass, classA ? cal[kc_ca_in_lp_hz] : 0.0, fi);
        }
        for (int c = 0; c < 2; ++c) {
            const int src = (stereo && c == 1) ? 0 : c;   // STEREO: the left controls govern both channels
            auto P = [&](int p) { return ctl[src * kPerChannel + p]; };
            const int xfmr = P(kPTransformer);
            OptoConfig oc;
            const double scHz = cal[kc_sc_hz] * double(ctl[kGScHpHz] + kScHpMin) / 90.0;   // the fit's corner is the reference's 90 Hz setting
            oc.scHz = scHz;
            oc.thr = P(kPOpticalThreshold); oc.gain = P(kPOpticalGain); oc.scFilter = P(kPSidechainFilter) != 0;
            oc.memory = ctl[kGOptoMemory] != 0; oc.hwUnit = profile == kProfileMeasuredUnit;
            oc.classA = classA; oc.ca = ca;
            if (xfmr == kGold) oc.lightGain = dbToLin(2.0 * mat::kGoldLightDb);   // light is a power: -1.5 dB is x0.708
            if (xfmr == kUranium) { oc.tauEl2 = mat::kUraniumTauEl2; oc.halfLife = true; }
            ch[c].opto.configure(oc);
            DiscreteConfig dc;
            dc.scHz = scHz;
            dc.thr = P(kPDiscreteThreshold); dc.ratio = P(kPDiscreteRatio); dc.attack = P(kPDiscreteAttack);
            dc.recover = P(kPDiscreteRecover); dc.gain = P(kPDiscreteGain); dc.scFilter = P(kPSidechainFilter) != 0;
            dc.hwUnit = profile == kProfileMeasuredUnit;
            dc.classA = classA; dc.ca = ca; dc.a2Scale = classA ? cal[kc_ca_d_a2_scale] : 1.0; dc.noiseDb = classA ? cal[kc_ca_noise_db] : -400.0;
            if (xfmr == kGermanium) {
                dc.leakRatio = mat::kGeLeakRatio25 * std::exp2((roomC - 25.0) / mat::kGeLeakDoubleC);
                dc.a2Extra = mat::kGeVcaMismatchA2 * cal[kc_d_a2];
            }
            ch[c].disc.configure(dc);
            CoreContext ctx; ctx.profile = profile; ctx.roomC = roomC; ctx.exhibition = exhibition; ctx.ca = ca;
            if (first) ch[c].xf.snap(coreFor(cal, xfmr, ctx)); else ch[c].xf.setParams(coreFor(cal, xfmr, ctx));
            optoOn[c] = P(kPOptical) != 0; discOn[c] = P(kPDiscrete) != 0;
            const bool u = xfmr == kUranium;
            if (u != ch[c].uranium || exhibition != lastExhibition) {
                const double lvl = mat::kAlphaLevelDb + (exhibition ? mat::kAlphaExhibitionDb : 0.0);
                ch[c].decay.configure(fi, mat::kAlphaRate, dbToLin(lvl), exhibition ? mat::kFissionExhibitionRate : mat::kFissionRate,
                                      0xA1FA + 31ull * uint64_t(c));
            }
            ch[c].uranium = u;
            if (first) {   // nothing heard yet: the stages start at rest on these settings, no crossfades, no slewing
                ch[c].opto.reset(); ch[c].disc.reset();
                ch[c].optoIn.reset(optoOn[c] ? 1.0 : 0.0); ch[c].discIn.reset(discOn[c] ? 1.0 : 0.0);
            }
        }
        lastExhibition = exhibition;
        if (first) { mixS.reset(ctl[kGMix] * 0.01); hwS.reset(ctl[kGHardwire] ? 1.0 : 0.0); }
    }

    double cal[kCalSize];
    int ctl[kNumInputParams];
    bool dirty = true, prepared = false, first = true, stereo = true, lastExhibition = false;
    double caIn = 1.0, caOut = 1.0, interLin = 1.0, scfLin = 1.0, bleed = 0.0;
    bool optoOn[2] = { true, true }, discOn[2] = { true, true };
    double hostFs = 48000.0;
    int quality = 0, os = 1;
    Ch ch[2];
    Smoother mixS, hwS;
    PeakMeter eye;
    double dry[2][kDelayMax];
    int dpos = 0;
};

} // namespace hvmc
