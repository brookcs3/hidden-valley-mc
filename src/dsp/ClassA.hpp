// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// The CLASS A profile's module block (docs/class-a-profile.md): one single-ended Class-A amplifier module as the Class A editions of
// the unit use in place of the standard unit's Class-A/B modules. Even and compressive odd terms on the module input, the even term
// walking with the 5 Hz envelope of the signal (the bias shift of a single-ended stage), then a soft output ceiling in the same form
// as the transformer cores' flux knee, with the cut-off polarity's ceiling a little lower than the rail polarity's. The terms are
// borrowed from the fitted Iron stage (the one measured single-ended Class-A stage in the family), the ceiling from the Pye 4060's
// published maximum output and the BA283's rating; none of it is a measurement of a Class A unit.
#pragma once
#include <cmath>
#include "Common.hpp"

namespace hvmc {

struct ClassAParams {
    double a2 = 0.0, a3 = 0.0, a2Env = 0.0, ceilDb = 13.0, q = 8.0, asymDb = 1.0;
};

class ClassAStage {
public:
    void prepare(double fs) { kEnv = onePoleK(1.0 / (kTwoPi * 5.0), fs); reset(); }
    void set(const ClassAParams& p)
    {
        P = p;
        cp = dbToLin(p.ceilDb); cn = cp * dbToLin(-p.asymDb);
        invQ = p.q > 0.0 ? 1.0 / p.q : 0.0;
    }
    void reset() { env = 0.0; }
    // the even and odd terms, with the envelope-walked even term
    inline double poly(double u)
    {
        env += (std::fabs(u) - env) * kEnv;
        const double a2 = P.a2 * (1.0 + P.a2Env * env);
        return u + a2 * u * u + P.a3 * u * u * u;
    }
    // the soft ceiling: y / (1 + |y/C|^q)^(1/q), C per polarity
    inline double ceiling(double y) const
    {
        if (invQ <= 0.0) return y;
        const double C = y > 0.0 ? cp : cn;
        return y / std::pow(1.0 + std::pow(std::fabs(y / C), P.q), invQ);
    }
    inline double tick(double u) { return ceiling(poly(u)); }
    // the even-term increment for a stage that already has its own polynomial (the optical amplifier)
    inline double a2Now() const { return P.a2 * (1.0 + P.a2Env * env); }
    inline void track(double u) { env += (std::fabs(u) - env) * kEnv; }
    double a3() const { return P.a3; }

private:
    ClassAParams P;
    double kEnv = 1.0, env = 0.0, cp = 4.47, cn = 3.98, invQ = 0.125;
};

} // namespace hvmc
