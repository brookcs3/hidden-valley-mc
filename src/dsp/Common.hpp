// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// Small shared pieces: constants, dB conversions, one-pole smoothers and filters, a counter-based noise source, and the denormal guard.
// Everything here is header-only, allocation-free and real-time safe.
#pragma once
#include <cmath>
#include <cstdint>

namespace hvmc {

static constexpr double kPi = 3.14159265358979323846;
static constexpr double kTwoPi = 2.0 * kPi;

inline double dbToLin(double db) { return std::exp(db * 0.11512925464970228); }        // 10^(db/20)
inline double linToDb(double x) { return 8.685889638065035 * std::log(x > 1e-30 ? x : 1e-30); }   // 20 log10 x
inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

// coefficient of a one-pole smoother with time constant tau (seconds) at sample rate fs: y += (x - y) * k
inline double onePoleK(double tau, double fs) { return tau <= 0.0 ? 1.0 : 1.0 - std::exp(-1.0 / (tau * fs)); }

struct Smoother {
    double y = 0.0, k = 1.0;
    void set(double tau, double fs) { k = onePoleK(tau, fs); }
    void reset(double v) { y = v; }
    inline double tick(double target) { y += (target - y) * k; return y; }
};

// first-order sections by the bilinear transform with prewarping (TPT form): high pass, low pass, high shelf
struct FirstOrder {
    enum Kind { kLowPass, kHighPass, kHighShelf, kLowShelf };
    double g = 0.0, s = 0.0, gainLin = 1.0;
    int kind = kLowPass;
    bool bypass = true;
    // fc <= 0 means "no filter" (the section passes its input); a corner at or above 0.49 fs is clamped there
    void set(int k, double fc, double fs, double shelfDb = 0.0)
    {
        kind = k;
        bypass = !(fc > 0.0);
        if (bypass) return;
        if (fc > 0.49 * fs) fc = 0.49 * fs;
        g = std::tan(kPi * fc / fs);
        gainLin = dbToLin(shelfDb);
    }
    void reset() { s = 0.0; }
    inline double tick(double x)
    {
        if (bypass) return x;
        const double v = (x - s) * g / (1.0 + g);
        const double lp = v + s;
        s = lp + v;
        const double hp = x - lp;
        switch (kind) {
        case kLowPass: return lp;
        case kHighPass: return hp;
        case kHighShelf: return lp + gainLin * hp;
        default: return gainLin * lp + hp;
        }
    }
};

// transposed direct form II biquad
struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, s1 = 0, s2 = 0;
    void reset() { s1 = s2 = 0.0; }
    inline double tick(double x)
    {
        const double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return y;
    }
    // RBJ peaking equaliser
    void setPeak(double f0, double q, double gainDb, double fs)
    {
        if (!(f0 > 0.0) || std::fabs(gainDb) < 1e-9) { b0 = 1; b1 = b2 = a1 = a2 = 0; return; }
        const double A = std::pow(10.0, gainDb / 40.0), w = kTwoPi * f0 / fs, al = std::sin(w) / (2.0 * q), c = std::cos(w);
        const double a0 = 1.0 + al / A;
        b0 = (1.0 + al * A) / a0; b1 = -2.0 * c / a0; b2 = (1.0 - al * A) / a0; a1 = -2.0 * c / a0; a2 = (1.0 - al / A) / a0;
    }
    // RBJ low pass
    void setLowPass(double f0, double q, double fs)
    {
        const double w = kTwoPi * f0 / fs, al = std::sin(w) / (2.0 * q), c = std::cos(w), a0 = 1.0 + al;
        b0 = (1.0 - c) * 0.5 / a0; b1 = (1.0 - c) / a0; b2 = b0; a1 = -2.0 * c / a0; a2 = (1.0 - al) / a0;
    }
    // magnitude-matched second-order low pass (Vicanek, "Matched Second Order Digital Filters", 2016): poles by impulse invariance,
    // zeros matched at DC, f0 and Nyquist, so the response follows the analog curve up to Nyquist instead of the bilinear warp
    void setMatchedLowPass(double f0, double Q, double fs)
    {
        if (!(f0 > 0.0)) { b0 = 1; b1 = b2 = a1 = a2 = 0; return; }
        const double w0 = kTwoPi * f0 / fs, q = 1.0 / (2.0 * Q);
        a1 = q <= 1.0 ? -2.0 * std::exp(-q * w0) * std::cos(std::sqrt(1.0 - q * q) * w0)
                      : -2.0 * std::exp(-q * w0) * std::cosh(std::sqrt(q * q - 1.0) * w0);
        a2 = std::exp(-2.0 * q * w0);
        const double A0 = (1.0 + a1 + a2) * (1.0 + a1 + a2), A1 = (1.0 - a1 + a2) * (1.0 - a1 + a2), A2 = -4.0 * a2;
        const double p1 = std::sin(0.5 * w0) * std::sin(0.5 * w0), p0 = 1.0 - p1, p2 = 4.0 * p0 * p1;
        const double R1 = (A0 * p0 + A1 * p1 + A2 * p2) * Q * Q;
        const double B0 = A0, B1 = (R1 - B0 * p0) / p1;
        b0 = 0.5 * (std::sqrt(B0) + std::sqrt(B1)); b1 = std::sqrt(B0) - b0; b2 = 0.0;
    }
};

// deterministic noise: splitmix64 on a counter, so a render is reproducible from its seed and sample position
struct Noise {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    void seed(uint64_t s) { state = s * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull; }
    inline uint64_t next()
    {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    inline double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }   // [0, 1)
    inline double gauss()   // Box-Muller, one value per call
    {
        const double u1 = uniform() + 1e-300, u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
    }
};

// flush denormals to zero for the duration of a process call (x86 SSE and AArch64); from the sibling project, unchanged
struct DenormalGuard {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
    unsigned int saved;
    DenormalGuard() { saved = __builtin_ia32_stmxcsr(); __builtin_ia32_ldmxcsr(saved | 0x8040u); }
    ~DenormalGuard() { __builtin_ia32_ldmxcsr(saved); }
#elif defined(__aarch64__)
    unsigned long saved;
    DenormalGuard() { __asm__ __volatile__("mrs %0, fpcr" : "=r"(saved)); const unsigned long v = saved | (1ul << 24); __asm__ __volatile__("msr fpcr, %0" : : "r"(v)); }
    ~DenormalGuard() { __asm__ __volatile__("msr fpcr, %0" : : "r"(saved)); }
#else
    DenormalGuard() {}
#endif
};

} // namespace hvmc
