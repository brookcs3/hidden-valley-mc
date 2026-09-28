// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The output line amplifier and transformer: one parametric core, one parameter set per position.
//
//   x -> gain -> driver stage (x + a2 x^2 + a3 x^3) -> magnetic core -> high shelf -> low pass -> [profile / material extras] -> out
//
// The core is a flux-domain model: the flux is the (leaky) integral of the drive, the core saturates the flux with a hard ceiling, and the
// output is the rate of change of the saturated flux (integrate -> saturate -> differentiate). Linear, this is the first-order
// low-frequency corner of the magnetising inductance against the source resistance. Driven hard at low frequency, the flux cannot
// exceed the knee, so the output is limited to a ceiling that rises 6 dB per octave: saturation at 20 Hz sets in 6 dB below where it
// does at 40 Hz, as measured on the reference, and the core's third harmonic comes out in anti-phase with the driver stage's, which
// is what makes the reference's third harmonic dip where the two cross (docs/MODEL.md). The form is explicit: no iteration per sample.
#pragma once
#include <cmath>
#include "Common.hpp"
#include "Calibration.hpp"
#include "Materials.hpp"
#include "ClassA.hpp"

namespace hvmc {

struct CoreParams {
    double gainDb = 0.0, a2 = 0.0, a3 = 0.0;
    double flHz = 1.7, satDb = 2.0, q = 9.0, asym = 0.0;   // asym: knee-hardness asymmetry between the polarities
    bool linear = false;                    // no saturation (material positions whose core is ideal)
    double hsHz = 0.0, hsDb = 0.0, lpHz = 0.0;
    double hs2Hz = 0.0, hs2Db = 0.0;        // a second shelf (plutonium eddy slug, germanium output shelf)
    double liftDb = 0.0, liftHz = 32.0, liftQ = 0.55;   // Hardware profile, Iron
    double noiseDb = -400.0, noiseLpHz = 0.0;           // added noise floor (Hardware Iron: low-frequency; germanium: 1/f)
    // germanium Class-A stage (in the Iron position's place)
    bool geClassA = false;
    double geLoopLpHz = 0.0, geA2 = 0.0, dieRiseC = 0.0, roomC = 25.0;
    // winding thermal element (uranium: resistance rises with heat; plutonium: falls)
    double heatDbPerK = 0.0, heatKPerFs2 = 0.0, heatTau = 3.0, coolTau = 8.0;
    // CLASS A profile: the driver module's terms and ceiling before the core (hardware cores only)
    bool classA = false;
    ClassAParams ca;
    bool operator!=(const CoreParams& o) const
    {
        return gainDb != o.gainDb || a2 != o.a2 || a3 != o.a3 || flHz != o.flHz || satDb != o.satDb || q != o.q || asym != o.asym ||
               linear != o.linear || hsHz != o.hsHz || hsDb != o.hsDb || lpHz != o.lpHz || hs2Hz != o.hs2Hz || hs2Db != o.hs2Db ||
               liftDb != o.liftDb || liftHz != o.liftHz || liftQ != o.liftQ || noiseDb != o.noiseDb || noiseLpHz != o.noiseLpHz ||
               geClassA != o.geClassA || geLoopLpHz != o.geLoopLpHz || geA2 != o.geA2 || dieRiseC != o.dieRiseC ||
               roomC != o.roomC || heatDbPerK != o.heatDbPerK || heatKPerFs2 != o.heatKPerFs2 || classA != o.classA ||
               ca.a2 != o.ca.a2 || ca.a3 != o.ca.a3 || ca.a2Env != o.ca.a2Env || ca.ceilDb != o.ca.ceilDb || ca.q != o.ca.q || ca.asymDb != o.ca.asymDb;
    }
};

class TransformerCore {
public:
    void configure(const CoreParams& p, double fs, uint64_t seed)
    {
        P = p; fsr = fs; T = 1.0 / fs;
        g = dbToLin(p.gainDb);
        r = kTwoPi * p.flHz;
        phik = dbToLin(p.satDb) / (kTwoPi * 20.0);
        qp = p.q * (1.0 + p.asym); qn = p.q * (1.0 - p.asym);
        if (qp < 1.0) qp = 1.0;
        if (qn < 1.0) qn = 1.0;
        hs.set(FirstOrder::kHighShelf, p.hsHz, fs, p.hsDb);
        hs2.set(FirstOrder::kHighShelf, p.hs2Hz, fs, p.hs2Db);
        lp.set(FirstOrder::kLowPass, p.lpHz, fs);
        lift.setPeak(p.liftHz, p.liftQ, p.liftDb, fs);
        noiseLp.set(FirstOrder::kLowPass, p.noiseLpHz, fs);
        noiseAmp = p.noiseDb > -300.0 ? dbToLin(p.noiseDb) : 0.0;
        noise.seed(seed);
        geLp.set(FirstOrder::kLowPass, p.geClassA ? p.geLoopLpHz : 0.0, fs);
        kEnv = onePoleK(1.0 / (kTwoPi * 5.0), fs);
        caStage.prepare(fs); caStage.set(p.ca);
        kDie = onePoleK(mat::kGeThermalTau, fs);
        kHeat = onePoleK(p.heatTau, fs); kCool = onePoleK(p.coolTau, fs); kPow = onePoleK(0.3, fs);
        reset();
    }

    // continue from another core's state (a position change): flux, filters and slow states carry over; the saturated flux is
    // re-evaluated under this core's law so the differentiator does not see a step
    void seedFrom(const TransformerCore& o)
    {
        phi = o.phi; hs = o.hs; hs2 = o.hs2; lp = o.lp; lift = o.lift; noiseLp = o.noiseLp; geLp = o.geLp;
        env = o.env; tDie = o.tDie; heat = o.heat; pw = o.pw; caStage = o.caStage;
        sPrev = P.linear ? phi : sat(phi);
    }
    void reset()
    {
        phi = 0.0; sPrev = 0.0;
        hs.reset(); hs2.reset(); lp.reset(); lift.reset(); noiseLp.reset(); geLp.reset(); caStage.reset();
        env = 0.0; tDie = P.roomC + P.dieRiseC; heat = 0.0; pw = 0.0;
    }

    inline double process(double x)
    {
        double u = g * x;
        if (P.geClassA) u = classA(u);
        else {
            const double uf = P.a3 < 0.0 ? 1.0 / std::sqrt(-3.0 * P.a3) : 1e30;   // beyond the polynomial's fold it clips instead of inverting
            const double uc = u > uf ? uf : (u < -uf ? -uf : u);
            u = uc + P.a2 * uc * uc + P.a3 * uc * uc * uc;
            if (P.classA) u = caStage.tick(u);   // CLASS A: the driver module's own terms and ceiling before the core
        }
        double y = P.linear ? linearCore(u) : core(u);
        y = hs.tick(y); y = hs2.tick(y); y = lp.tick(y);
        if (P.liftDb != 0.0) y = lift.tick(y);
        if (noiseAmp > 0.0) {
            const double n = noise.gauss() * noiseAmp;
            y += P.noiseLpHz > 0.0 ? noiseLp.tick(n) * noiseLpScale() : n;
        }
        if (P.heatKPerFs2 != 0.0) {
            // winding temperature follows the program power; its resistance changes the insertion loss (the static part is made up)
            pw += (y * y - pw) * kPow;
            const double target = P.heatKPerFs2 * pw;
            heat += (target - heat) * (target > heat ? kHeat : kCool);
            if (heat > 100.0) heat = 100.0;   // a winding does not run past 100 K over ambient; the gain term stays bounded
            y *= dbToLin(P.heatDbPerK * heat);
        }
        return y;
    }

private:
    // saturating flux: S(phi) = phi / (1 + |phi / phi_k|^q)^(1/q), a hard ceiling at +-phi_k with knee hardness q. The two polarities
    // have slightly different hardness (the measured even harmonics peak at the onset of saturation and fade as the drive rises,
    // which a knee asymmetry does and a flux offset does not).
    inline double sat(double ph) const
    {
        const double az = std::fabs(ph / phik);
        if (az <= 0.02) return ph;
        const double q = ph > 0.0 ? qp : qn;
        const double lz = q * std::log(az);
        if (lz > 700.0) return ph > 0.0 ? phik : -phik;   // far past the ceiling: the exp would overflow to a wrong zero
        return ph / std::exp(std::log1p(std::exp(lz)) / q);
    }

    // the core: flux = leaky integral of the drive (forward Euler; the leak is the low-frequency corner), output = the rate of change
    // of the saturated flux. With S linear this is exactly the first-order high pass; with S saturating, the output is limited to
    // omega * phi_k, a ceiling that rises 6 dB per octave.
    inline double core(double v)
    {
        phi += T * (v - r * phi);
        const double s = sat(phi);
        const double y = (s - sPrev) * fsr;
        sPrev = s;
        return y;
    }
    inline double linearCore(double v)
    {
        const double prev = phi;
        phi += T * (v - r * phi);
        return (phi - prev) * fsr;
    }

    // germanium Class-A stage: slow loop gain ahead of the shaper, mirrored (PNP) asymmetry, even order that walks with the envelope
    // and with die temperature (leakage doubles every 9 C), and a one-sided choke as the die approaches the germanium limit
    inline double classA(double u)
    {
        u = geLp.tick(u);
        env += (std::fabs(u) - env) * kEnv;
        tDie += (P.roomC + P.dieRiseC - tDie) * kDie;
        const double leak = std::exp2((tDie - 25.0) / mat::kGeLeakDoubleC);
        const double a2 = P.geA2 * (1.0 + 0.5 * env) * (1.0 + 0.05 * (leak - 1.0));
        double y = u + a2 * u * u + P.a3 * u * u * u;
        if (tDie > mat::kGeChokeStartC) {
            // one polarity's ceiling collapses as the operating point runs out (the "gating" of hot germanium stages)
            const double k = clampd((tDie - mat::kGeChokeStartC) / (mat::kGeChokeFullC - mat::kGeChokeStartC), 0.0, 1.0);
            const double ceil = dbToLin(12.0 - 30.0 * k);
            if (y < 0.0) y = -ceil * std::tanh(-y / ceil);
        }
        return y;
    }
    // a one-pole low pass passes pi fc / fs of white noise's power: scale so the filtered noise has the stated RMS
    double noiseLpScale() const { return P.noiseLpHz > 0.0 ? 1.0 / std::sqrt(kPi * P.noiseLpHz / fsr) : 1.0; }

    CoreParams P;
    double fsr = 48000.0, T = 1.0 / 48000.0, g = 1.0, r = 0.0, phik = 0.01, qp = 9.0, qn = 9.0;
    double phi = 0.0, sPrev = 0.0;
    FirstOrder hs, hs2, lp, noiseLp, geLp;
    ClassAStage caStage;
    Biquad lift;
    Noise noise;
    double noiseAmp = 0.0, env = 0.0, tDie = 25.0, kEnv = 1.0, kDie = 1.0;
    double heat = 0.0, pw = 0.0, kHeat = 1.0, kCool = 1.0, kPow = 1.0;
};

// ------------------------------------------------------------------------------------------------ parameters per position
struct CoreContext {
    int profile = 1;            // 0 reference, 1 hardware, 2 measured unit, 3 class A
    ClassAParams ca;
    double roomC = 25.0;
    bool exhibition = false;
};

inline CoreParams hwCore(const double* c, int k)
{
    CoreParams p;
    p.gainDb = c[kc_x_gain_db + k]; p.a2 = c[kc_x_a2 + k]; p.a3 = c[kc_x_a3 + k];
    p.flHz = c[kc_x_fl_hz + k]; p.satDb = c[kc_x_sat_db + k]; p.q = c[kc_x_q + k]; p.asym = c[kc_x_asym + k];
    p.hsHz = c[kc_x_hs_hz + k]; p.hsDb = c[kc_x_hs_db + k]; p.lpHz = c[kc_x_lp_hz + k];
    return p;
}

inline CoreParams coreFor(const double* c, int position, const CoreContext& ctx)
{
    CoreParams p;
    switch (position) {
    case 0: case 1: case 2:
        p = hwCore(c, position);
        if (position == 1 && ctx.profile >= 1) {
            p.liftDb = c[kc_hw_iron_lift_db]; p.liftHz = c[kc_hw_iron_lift_hz]; p.liftQ = c[kc_hw_iron_lift_q];
            p.noiseDb = c[kc_hw_iron_noise_db]; p.noiseLpHz = 60.0;
        }
        if (ctx.profile == 3) { p.classA = true; p.ca = ctx.ca; }   // CLASS A: the driver module before the core
        break;
    case 3: {   // GOLD: an Au4Mn core (half Nickel's saturation, drifting with temperature), the softest knee; paramagnetic above 112 C
        p = hwCore(c, 0);
        const double tK = ctx.roomC + 273.15;
        if (tK >= mat::kGoldTc) {   // GOLD-PURE: the core has become paramagnetic (a linear air core, exhibition-scaled)
            p.linear = true; p.flHz = 6.0; p.gainDb = -0.3; p.lpHz = 0.0; p.hsHz = 0.0; p.hs2Hz = 10000.0; p.hs2Db = -0.8;
        } else {
            p.satDb = c[kc_x_sat_db] + mat::goldKneeDb(ctx.roomC);
            p.q = c[kc_x_q] * mat::kGoldQScale;
            p.flHz = mat::kGoldFlHz;
            p.gainDb = mat::kGoldGainDb;
        }
        break;
    }
    case 4:     // URANIUM: a UFe10Si2 core between Nickel and Steel, softer knee; uranium windings heat with the program
        p = hwCore(c, 2);
        p.satDb = c[kc_x_sat_db] + mat::kUraniumKneeDb;
        p.q = c[kc_x_q + 2] * mat::kUraniumQScale;
        p.flHz = c[kc_x_fl_hz + 2] * mat::kUraniumFlScale;
        p.heatDbPerK = mat::kUraniumHeatDbPerK;
        // true scale: <= 0.8 K at +4 dBFS sustained; exhibition: 20 K at full scale
        p.heatKPerFs2 = ctx.exhibition ? mat::kUraniumHeatK : 0.8 / (dbToLin(4.0) * dbToLin(4.0) * 0.5);
        p.heatTau = mat::kUraniumHeatTau; p.coolTau = mat::kUraniumCoolTau;
        break;
    case 5:     // GERMANIUM: ideal core, germanium Class-A stage in the Iron position's place
        p = hwCore(c, 1);
        p.linear = true; p.flHz = 2.0; p.lpHz = 20000.0; p.hsHz = 0.0; p.hsDb = 0.0;
        p.hs2Hz = 10000.0; p.hs2Db = mat::kGeHfShelfDb;
        p.geClassA = true; p.geLoopLpHz = mat::kGeLoopLpHz;
        p.geA2 = -mat::kGeH2Scale * c[kc_x_a2 + 1];   // mirrored asymmetry (PNP), six times the silicon stage's even order
        p.a3 = c[kc_x_a3 + 1];
        p.dieRiseC = mat::kGeDieRiseC; p.roomC = ctx.roomC;
        p.noiseDb = mat::kGeNoiseDb + 1.5 * (ctx.roomC - 25.0) / 9.0; p.noiseLpHz = mat::kGeNoiseLpHz;
        break;
    case 6:     // PLUTONIUM (hidden): the real Nickel core behind delta-Pu windings and an eddy-current slug
        p = hwCore(c, 0);
        p.flHz = mat::kPuFlHz;
        p.hs2Hz = mat::kPuShelfHz; p.hs2Db = mat::kPuShelfDb;
        p.heatDbPerK = mat::kPuBloomDbPerK * (ctx.exhibition ? mat::kPuBloomExhibition : 1.0);
        p.heatKPerFs2 = ctx.exhibition ? 10.0 : 0.1;   // program heating of the winding (true about 0.1 K at full scale)
        p.heatTau = 2.0; p.coolTau = 20.0;
        break;
    }
    return p;
}

// one channel's transformer: the active core, and a second one warmed on recent input and crossfaded in when the position changes
class TransformerBlock {
public:
    void prepare(double fs, uint64_t seed)
    {
        fsr = fs; seedBase = seed;
        fadeLen = int(0.02 * fs); if (fadeLen < 16) fadeLen = 16;
        fading = false; configured = false;
    }
    void setParams(const CoreParams& p)
    {
        if (!configured) {
            cur = p; a.configure(p, fsr, seedBase); act = &a; nxt = &b; configured = true; return;
        }
        if (fading) { pending = p; hasPending = true; return; }
        if (!(p != cur)) return;
        startFade(p);
    }
    void reset() { a.reset(); b.reset(); fading = false; hasPending = false; }
    // start on these parameters at rest, with no crossfade (the first block after a prepare)
    void snap(const CoreParams& p) { reset(); configured = false; setParams(p); }
    inline double process(double x)
    {
        if (!fading) return act->process(x);
        const double ya = act->process(x), yb = nxt->process(x);
        const double t = double(fpos) / fadeLen, w = 0.5 - 0.5 * std::cos(kPi * t);
        if (++fpos >= fadeLen) {
            TransformerCore* tmp = act; act = nxt; nxt = tmp; fading = false;
            if (hasPending) { hasPending = false; if (pending != cur) startFade(pending); }
        }
        return ya + w * (yb - ya);
    }
private:
    void startFade(const CoreParams& p)
    {
        cur = p;
        nxt->configure(p, fsr, ++seedBase);
        nxt->seedFrom(*act);   // the flux and filter states carry over: no history replay in the audio callback
        fading = true; fpos = 0;
    }
    TransformerCore a, b;
    TransformerCore* act = &a;
    TransformerCore* nxt = &b;
    CoreParams cur, pending;
    bool configured = false, fading = false, hasPending = false;
    double fsr = 48000.0;
    int fadeLen = 960, fpos = 0;
    uint64_t seedBase = 1;
};

} // namespace hvmc
