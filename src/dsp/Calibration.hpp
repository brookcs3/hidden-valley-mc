// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// Every fitted constant of the model, in one flat table of doubles with named fields. The fitting pipeline (fit/) reads this layout
// through the C API (src/capi/hvmc_capi.cpp), fits the values against the reference features, and writes FittedConstants.hpp, which
// this file includes. If FittedConstants.hpp is missing or was written for a different layout, the model falls back to the priors in
// calPriors() and kCalFitted is false (the unit tests report it; a release build must have it true).
//
// Level convention: 1.0 = 0 dBFS = +14 dBu (the reference's default meter calibration). Times in seconds, frequencies in Hz.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

namespace hvmc {

static constexpr int kSteps24 = 24;
static constexpr int kRatios = 6;
static constexpr int kAttacks = 6;
static constexpr int kRecovers = 6;       // 0.1, 0.25, 0.5, 0.8, 1.2 s, Dual
static constexpr int kHwCores = 3;        // Nickel, Iron, Steel
static constexpr int kCurveN = 96;        // discrete gain-computer table: detector level above threshold, -20 .. +75 dB, 1 dB grid
static constexpr double kCurveX0 = -20.0;
static constexpr double kCurveDx = 1.0;
static constexpr int kOptoStates = 3;

// S(name, prior) is a scalar, A(name, count) an array (its priors come from calPriors()).
#define HVMC_CAL_FIELDS(S, A)                                                                                                        \
    S(sc_hz, 90.0)              /* sidechain high-pass corner, first order, both detectors */                                       \
    /* optical stage */                                                                                                                \
    A(o_thr_db, kSteps24)       /* sidechain drive gain per OPTICAL THRESHOLD position, dB */                                        \
    A(o_gain_db, kSteps24)      /* make-up gain per OPTICAL GAIN position, dB */                                                      \
    S(o_n, 1.0)               /* EL panel light law exponent above the turn-on: light ~ (drive - o_vth)^n (held at 1 by the fit) */    \
    S(o_gamma, 1.66)          /* CdS conductance law exponent: conductance ~ light^gamma; static slope above the knee gamma/(1+gamma) */    \
    S(o_vth, 3.06)            /* EL turn-on in drive units: a hard knee at 20log10(o_vth) - o_thr_db dBFS */    \
    S(o_tau_el, 6.6e-5)       /* EL phosphor persistence, s */    \
    A(o_w, kOptoStates)         /* weights of the cell's conductance states (sum 1) */                                                \
    A(o_tatt, kOptoStates)      /* attack time constant of each state */                                                              \
    A(o_trel, kOptoStates)      /* release time constant of each state */                                                             \
    S(o_rel_mu, 7.7)            /* release rate multiplier per unit conductance: rate = (1 + o_rel_mu s) / o_trel (bimolecular) */ \
    S(o_leak, 0.020)            /* idle light leak as a conductance at threshold position 20 (sets the no-GR gain per position) */  \
    S(o_leak_q, 2.15)           /* its growth with the sidechain drive: cond0 = o_leak (A / A20)^o_leak_q */                       \
    S(o_sc_lp_hz, 5150.0)       /* optical sidechain second-order low pass corner (after the high pass), magnitude-matched */     \
    S(o_sc_lp_q, 0.72)          /* its Q */                                                                                        \
    S(o_b2, 0.0)                /* stage amplifier: quadratic term, on the stage input */                                             \
    S(o_b3, 0.0)                /* stage amplifier: cubic term */                                                                     \
    S(o_mem_beta, 0.2)          /* light-memory option: share of the slow, exposure-dependent state */                                \
    S(o_mem_trel, 1.0)          /*   its base release time constant */                                                                \
    S(o_mem_mu, 2.0)            /*   release lengthening at full exposure */                                                          \
    S(o_mem_tm, 5.0)            /*   exposure memory time constant */                                                                 \
    S(o_hw_lp_hz, 18000.0)      /* Measured Unit profile: first-order low pass in the opto path */                                    \
    S(o_hw_grloss_k, 0.30)      /* Measured Unit profile: HF loss that grows with opto GR (corner falls as 1/(1+k*c)) */              \
    S(o_hw_grloss_hz, 60000.0)  /*   corner of that loss at zero GR */                                                                \
    /* discrete stage */                                                                                                               \
    A(d_thr_db, kSteps24)       /* threshold per DISCRETE THRESHOLD position, dBFS of the detector level */                           \
    A(d_gain_db, kSteps24)      /* make-up gain per DISCRETE GAIN position, dB */                                                     \
    A(d_goff_db, kSteps24)      /* detector offset that comes with each make-up position (the measured GR interaction), dB */         \
    A(d_curve, kRatios * kCurveN) /* gain reduction (dB) against detector level above threshold, per ratio position */               \
    A(d_tatt, kAttacks)         /* attack time constants at the release reference level, log domain (see d_att_sv_db) */               \
    A(d_trel, kRecovers)        /* release (bleed) time constants, log domain, always conducting (the Dual entry is its first node) */ \
    S(d_dual_t2, 0.052)         /* Dual: coupling time constant between the two storage nodes (R2 C1) */                             \
    S(d_dual_c2, 10.4)          /* Dual: second storage capacitor, relative to the first */                                          \
    S(d_floor_db, -100.0)       /* log rectifier floor (far below any signal; the release does not go there) */               \
    S(d_rel_depth_db, 0.55)     /* the bleed discharges toward the threshold minus this depth (the node's rest point) */                        \
    S(d_att_sv_db, 17.4)        /* attack conductance law: factor 1 + (level above the release reference) / this, in dB */          \
    S(d_link, 0.5)              /* stereo: detector input = d_link * (left + right) */                                                \
    S(d_inter_db, 2.14)         /* gain between the stages when both are in (measured: both in at no GR is this much above the sum) */ \
    S(d_scf_trim_db, 0.64)      /* SIDECHAIN FILTER in: a trim on the discrete stage's input, audio and sidechain alike */             \
    S(d_a2, 0.0)                /* gain cell even-order term, on the cell input */                                                   \
    S(d_a3, 0.0)                /* gain cell odd-order term */                                                                        \
    S(d_hwunit_a2, 0.0)         /* Measured Unit profile: the torn-down unit's even-order term */                                     \
    /* transformer path: 0 Nickel, 1 Iron, 2 Steel */                                                                                  \
    A(x_gain_db, kHwCores)      /* midband gain */                                                                                     \
    A(x_a2, kHwCores)           /* driver stage quadratic term */                                                                     \
    A(x_a3, kHwCores)           /* driver stage cubic term (negative: compressive) */                                                \
    A(x_fl_hz, kHwCores)        /* low-frequency corner: source resistance against magnetising inductance */                          \
    A(x_sat_db, kHwCores)       /* core saturation: the level (dBFS peak) of a 20 Hz sine whose flux peak is the knee flux */         \
    A(x_q, kHwCores)            /* knee hardness: magnetising current ~ flux * (1 + |flux / knee|^q) */                               \
    A(x_asym, kHwCores)         /* knee-hardness asymmetry between the polarities (even harmonics at saturation onset) */                                              \
    A(x_hs_hz, kHwCores)        /* high shelf corner */                                                                                \
    A(x_hs_db, kHwCores)        /* high shelf gain */                                                                                  \
    A(x_lp_hz, kHwCores)        /* first-order low pass (0 = none) */                                                                  \
    /* Hardware profile */                                                                                                             \
    S(hw_iron_lift_db, 0.40)    /* Iron: LF lift measured on hardware (Flotown renders), peaking section */                          \
    S(hw_iron_lift_hz, 32.0)                                                                                                           \
    S(hw_iron_lift_q, 0.55)                                                                                                            \
    S(hw_iron_noise_db, -96.0)  /* Iron: low-frequency noise floor (teardown FFT), dBFS RMS */ \
    S(ca_in_db, 2.5)            /* CLASS A profile: input level offset (Lundahl input and module gain structure), dB */               \
    S(ca_out_db, 0.0)           /* CLASS A: remainder of the vendors' 1-3 dB at the output, dB */                                       \
    S(ca_in_fl_hz, 0.0)         /* CLASS A: added first-order input high pass for the Lundahl (0 = none) */                             \
    S(ca_in_lp_hz, 0.0)         /* CLASS A: added first-order input low pass for the Lundahl (0 = none) */                              \
    S(ca_a2, 8.0e-5)            /* CLASS A: one single-ended module's even term, on the module input */                                 \
    S(ca_a3, -7.2e-4)           /* CLASS A: its compressive odd term */                                                                 \
    S(ca_a2_env, 0.25)          /* CLASS A: the even term walks with the 5 Hz envelope: a2 (1 + ca_a2_env env) */                        \
    S(ca_ceil_db, 13.0)         /* CLASS A: module output ceiling, dBFS peak (+27 dBu) */                                               \
    S(ca_ceil_q, 8.0)           /* CLASS A: ceiling knee hardness, the cores' form */                                                    \
    S(ca_ceil_asym_db, 1.0)     /* CLASS A: the cut-off polarity's ceiling sits this much lower */                                      \
    S(ca_d_a2_scale, 1.0)       /* CLASS A: gain cell even term relative to d_a2 (a Class-A Blackmer cell is cleaner, so 1) */          \
    S(ca_noise_db, -96.0)       /* CLASS A: cell noise floor, white, dBFS RMS, at the cell output before make-up */
enum CalField : int {
#define HVMC_S(name, prior) kc_##name,
#define HVMC_A(name, n) kc_##name, kc_##name##_last = kc_##name + (n) - 1,
    HVMC_CAL_FIELDS(HVMC_S, HVMC_A)
#undef HVMC_S
#undef HVMC_A
    kCalSize
};

struct CalFieldInfo { const char* name; int offset; int count; };

inline const CalFieldInfo* calFields(int* count)
{
    static const CalFieldInfo fields[] = {
#define HVMC_S(name, prior) { #name, kc_##name, 1 },
#define HVMC_A(name, n) { #name, kc_##name, (n) },
        HVMC_CAL_FIELDS(HVMC_S, HVMC_A)
#undef HVMC_S
#undef HVMC_A
    };
    *count = int(sizeof(fields) / sizeof(fields[0]));
    return fields;
}

// FNV-1a over the field names and sizes: a fitted header written for another layout is rejected
inline uint64_t calLayoutHash()
{
    uint64_t h = 1469598103934665603ull;
    int n = 0;
    const CalFieldInfo* f = calFields(&n);
    for (int i = 0; i < n; ++i) {
        for (const char* p = f[i].name; *p; ++p) { h ^= uint8_t(*p); h *= 1099511628211ull; }
        h ^= uint64_t(f[i].count) + 0x9E37ull; h *= 1099511628211ull;
    }
    // the meaning of the curve tables depends on the grid, so a grid change must also invalidate a fitted header
    const double grid[3] = { double(kCurveN), kCurveX0, kCurveDx };
    const unsigned char* gb = reinterpret_cast<const unsigned char*>(grid);
    for (size_t i = 0; i < sizeof(grid); ++i) { h ^= gb[i]; h *= 1099511628211ull; }
    return h;
}

// Priors: physically motivated starting values, used by the fitter as its initial point and by the build if no fit exists.
static constexpr double kOptoThrPrior[kSteps24] = { -9.4, -1.43, 6.27, 11.37, 15.54, 17.99, 20.26, 22.41, 24.53, 26.62, 28.63, 29.57,
                                                    30.56, 31.48, 32.46, 33.39, 34.37, 35.32, 36.30, 37.28, 38.29, 39.32, 40.40, 41.36 };
inline void calPriors(double* v)
{
    std::memset(v, 0, sizeof(double) * kCalSize);
#define HVMC_S(name, prior) v[kc_##name] = (prior);
#define HVMC_A(name, n)
    HVMC_CAL_FIELDS(HVMC_S, HVMC_A)
#undef HVMC_S
#undef HVMC_A
    for (int i = 0; i < kSteps24; ++i) {
        v[kc_o_thr_db + i] = kOptoThrPrior[i];
        v[kc_o_gain_db + i] = -5.0 + 0.9 * i;
        v[kc_d_thr_db + i] = 2.8 - 2.69 * i;         // 2.69 dB per step, threshold 1 at +2.8 dBFS
        v[kc_d_gain_db + i] = -3.0 + 0.9 * i;
        v[kc_d_goff_db + i] = 0.0;
    }
    const double w[kOptoStates] = { 0.223, 0.590, 0.187 }, ta[kOptoStates] = { 0.0084, 0.0157, 0.0085 }, tr[kOptoStates] = { 0.175, 0.132, 0.094 };
    for (int s = 0; s < kOptoStates; ++s) { v[kc_o_w + s] = w[s]; v[kc_o_tatt + s] = ta[s]; v[kc_o_trel + s] = tr[s]; }
    const double slope[kRatios] = { 0.45, 0.8, 0.9, 1.0, 1.1, 1.3 };
    for (int r = 0; r < kRatios; ++r)
        for (int i = 0; i < kCurveN; ++i) {
            const double x = kCurveX0 + kCurveDx * i, wk = 4.0;
            const double sp = (x > 30.0 * wk) ? x : wk * std::log1p(std::exp(x / wk));
            v[kc_d_curve + r * kCurveN + i] = slope[r] * sp;
        }
    const double at[kAttacks] = { 0.035e-3, 0.42e-3, 1.76e-3, 6.4e-3, 19.3e-3, 58.0e-3 };   // at the release reference level
    const double rt[kRecovers] = { 0.095, 0.136, 0.334, 0.334, 0.489, 0.48 };
    for (int i = 0; i < kAttacks; ++i) v[kc_d_tatt + i] = at[i];
    for (int i = 0; i < kRecovers; ++i) v[kc_d_trel + i] = rt[i];
    const double g[kHwCores] = { -0.03, -0.21, 0.11 }, a2[kHwCores] = { 8e-6, 8e-5, 8e-6 }, a3[kHwCores] = { -1.4e-4, -8e-4, -1.4e-4 };
    const double fl[kHwCores] = { 1.66, 4.4, 1.45 }, sat[kHwCores] = { 2.0, 3.0, 3.0 }, hs[kHwCores] = { 10000.0, 0.0, 0.0 };
    const double hsdb[kHwCores] = { -0.2, 0.0, 0.0 }, lp[kHwCores] = { 0.0, 62000.0, 50000.0 };
    for (int c = 0; c < kHwCores; ++c) {
        v[kc_x_gain_db + c] = g[c]; v[kc_x_a2 + c] = a2[c]; v[kc_x_a3 + c] = a3[c]; v[kc_x_fl_hz + c] = fl[c];
        v[kc_x_sat_db + c] = sat[c]; v[kc_x_q + c] = 9.0; v[kc_x_asym + c] = 0.02;
        v[kc_x_hs_hz + c] = hs[c]; v[kc_x_hs_db + c] = hsdb[c]; v[kc_x_lp_hz + c] = lp[c];
    }
}

} // namespace hvmc

#if defined(__has_include)
#if __has_include("FittedConstants.hpp")
#include "FittedConstants.hpp"
#endif
#endif

namespace hvmc {

// the calibration the plugin runs: the fitted table when it matches this layout, the priors otherwise (built once, thread-safe)
struct CalTable {
    double v[kCalSize];
    bool fitted = false;
    CalTable()
    {
        calPriors(v);
#ifdef HVMC_FITTED_CONSTANTS
        if (kFittedLayoutHash == calLayoutHash() && kFittedCount == kCalSize) {
            std::memcpy(v, kFittedValues, sizeof(v));
            fitted = true;
        }
#endif
    }
};

inline const CalTable& calDefaults()
{
    static const CalTable table;
    return table;
}

} // namespace hvmc
