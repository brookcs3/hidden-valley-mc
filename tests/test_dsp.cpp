// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// Unit tests for the DSP core (no plugin host needed). scripts/build.sh and scripts/build-macos.sh build and run them before the
// plugin; by hand, from the repository root:
//   c++ -std=gnu++17 -O2 -Wall -Wextra -Isrc -o build/tests/test_dsp tests/test_dsp.cpp && build/tests/test_dsp build/tests
// Checks, each printed with its measured figure and PASS/FAIL:
//  1. calibration: the compiled constants are the fitted set (layout hash matches), not the priors;
//  2. transparency: HARDWIRE BYPASS out, and MIX 0 %, return the input bit for bit; stages out at unity gain within 0.1 dB;
//  3. determinism: two renders of the same noise with different block sizes are identical;
//  4. every position of every input parameter renders finite, bounded output;
//  5. HQ mode: latency is 39 samples and a 1 kHz tone through HQ matches STANDARD after the delay;
//  6. compression sanity: threshold up gives more gain reduction, and the meter reads it;
//  7. click test: stepping ratio, transformer and bypass switches while noise plays stays bounded;
//  8. transformer ceiling: a 20 Hz tone at +18 dBFS comes out limited (the flux ceiling), a 1 kHz tone does not;
// 10. KEY IN: a silent key stops the compression, a loud key compresses a quiet programme;
//  9. robustness: NaN, Inf and absurd input samples leave the engine finite and working, and the bypass stays transparent.
// Also writes <out_dir>/cpp_reference.json (a few renders' RMS figures) for tests/crosscheck.py.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#include <limits>
#include "dsp/Engine.hpp"

using namespace hvmc;

static int gFail = 0;
static void report(const char* name, bool ok, const char* detail)
{
    std::printf("[%s] %s: %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) ++gFail;
}

struct Render { std::vector<float> l, r; };

static Render run(Engine& e, const std::vector<float>& inL, const std::vector<float>& inR, int block = 512)
{
    Render o; o.l.resize(inL.size()); o.r.resize(inL.size());
    for (size_t i = 0; i < inL.size(); i += size_t(block)) {
        const int n = int(std::min(size_t(block), inL.size() - i));
        e.process(inL.data() + i, inR.data() + i, o.l.data() + i, o.r.data() + i, n);
    }
    return o;
}

static std::vector<float> sine(double dbfs, double f, double secs, double fs)
{
    std::vector<float> x(size_t(secs * fs));
    const double a = std::pow(10.0, dbfs / 20.0);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(a * std::sin(2 * kPi * f * double(i) / fs));
    return x;
}

static double rmsDb(const std::vector<float>& v, size_t from)
{
    double s = 0; size_t n = 0;
    for (size_t i = from; i < v.size(); ++i) { s += double(v[i]) * v[i]; ++n; }
    return 20 * std::log10(std::sqrt(s / double(n ? n : 1)) + 1e-30);
}

static bool finite(const std::vector<float>& v, float* peak)
{
    *peak = 0;
    for (float x : v) { if (!std::isfinite(x)) return false; if (std::fabs(x) > *peak) *peak = std::fabs(x); }
    return true;
}

static void stagesOut(Engine& e)
{
    e.setParam(kPOptical, 0); e.setParam(kPDiscrete, 0);
    e.setParam(kPerChannel + kPOptical, 0); e.setParam(kPerChannel + kPDiscrete, 0);
    e.setParam(kGStereo, 0);
}

int main(int argc, char** argv)
{
    const std::string outDir = argc > 1 ? argv[1] : ".";
    const double fs = 48000.0;
    char buf[512];

    // 1. calibration
    {
        const CalTable& t = calDefaults();
        std::snprintf(buf, sizeof buf, "fitted constants %s, layout hash %016llx, %d values", t.fitted ? "present" : "ABSENT (priors in use)",
                      (unsigned long long)calLayoutHash(), int(kCalSize));
        report("calibration", t.fitted, buf);
    }

    std::mt19937 rng(20260927);
    std::normal_distribution<double> gauss(0.0, 0.1);
    std::vector<float> noiseL(size_t(2 * fs)), noiseR(noiseL.size());
    for (size_t i = 0; i < noiseL.size(); ++i) { noiseL[i] = float(gauss(rng)); noiseR[i] = float(gauss(rng)); }

    // 2. transparency
    {
        Engine e; e.setParam(kGHardwire, 0); e.prepare(fs);
        Render o = run(e, noiseL, noiseR);
        bool exact = true;
        for (size_t i = 0; i < noiseL.size(); ++i) if (o.l[i] != noiseL[i] || o.r[i] != noiseR[i]) { exact = false; break; }
        report("hardwire bypass OUT is bit-transparent", exact, exact ? "identical" : "differs");
        Engine e2; e2.setParam(kGMix, 0); e2.setParam(kPOpticalThreshold, 23); e2.setParam(kPDiscreteThreshold, 23); e2.prepare(fs);
        o = run(e2, noiseL, noiseR);
        exact = true;
        for (size_t i = 0; i < noiseL.size(); ++i) if (o.l[i] != noiseL[i] || o.r[i] != noiseR[i]) { exact = false; break; }
        report("MIX 0 % is bit-transparent", exact, exact ? "identical" : "differs");
        Engine e3; stagesOut(e3); e3.prepare(fs);
        std::vector<float> s = sine(-20.0, 1000.0, 1.0, fs);
        o = run(e3, s, s);
        const double g = rmsDb(o.l, size_t(fs / 2)) - rmsDb(s, size_t(fs / 2));
        std::snprintf(buf, sizeof buf, "stages out, Nickel: gain %+.3f dB", g);
        report("unity through the transformer path", std::fabs(g) < 0.1, buf);
    }

    // 3. determinism
    {
        Engine a, b;
        for (Engine* e : { &a, &b }) { e->setParam(kPOpticalThreshold, 19); e->setParam(kPDiscreteThreshold, 15); e->prepare(fs); }
        Render oa = run(a, noiseL, noiseR), ob = run(b, noiseL, noiseR, 333);
        bool same = true;
        for (size_t i = 0; i < oa.l.size(); ++i) if (oa.l[i] != ob.l[i] || oa.r[i] != ob.r[i]) { same = false; break; }
        report("determinism across block sizes", same, same ? "identical with 512- and 333-sample blocks" : "differs");
    }

    // 4. every position renders finite, bounded output
    {
        int bad = 0, count = 0; float worst = 0;
        std::vector<float> s = sine(-6.0, 80.0, 0.5, fs);
        for (int p = 0; p < kNumInputParams; ++p) {
            for (int v = 0; v < paramSteps(p); ++v) {
                if (p == kGTemperature && v % 15 != 0) continue;
                if (p == kGMix && v % 25 != 0) continue;
                if (p == kGScHpHz && v % 100 != 0) continue;   // sample the sidechain corner
                Engine e; e.setParam(kPOpticalThreshold, 17); e.setParam(kPDiscreteThreshold, 15); e.setParam(p, v); e.prepare(fs);
                Render o = run(e, s, s);
                float pk1, pk2; const bool ok = finite(o.l, &pk1) && finite(o.r, &pk2);
                const float pk = std::max(pk1, pk2);
                if (!ok || pk > 4.0f) ++bad;
                if (pk > worst) worst = pk;
                ++count;
            }
        }
        std::snprintf(buf, sizeof buf, "%d renders, %d bad, worst peak %.2f", count, bad, worst);
        report("every parameter position renders finite, bounded output", bad == 0, buf);
    }

    // 5. HQ mode
    {
        Engine hq; hq.setParam(kGQuality, 1); hq.setParam(kPDiscreteThreshold, 15); hq.prepare(fs);
        Engine st; st.setParam(kPDiscreteThreshold, 15); st.prepare(fs);
        std::snprintf(buf, sizeof buf, "latency %d samples (standard %d)", hq.latency(), st.latency());
        report("HQ reports 39 samples of latency, STANDARD 0", hq.latency() == kOversampleLatency && st.latency() == 0, buf);
        std::vector<float> s = sine(-12.0, 1000.0, 2.0, fs);
        Render a = run(hq, s, s), b = run(st, s, s);
        const double d = rmsDb(a.l, size_t(fs)) - rmsDb(b.l, size_t(fs));
        double err = 0; size_t n = 0;
        for (size_t i = size_t(fs); i + size_t(hq.latency()) < a.l.size(); ++i) { err += std::fabs(double(a.l[i + size_t(hq.latency())]) - b.l[i]); ++n; }
        std::snprintf(buf, sizeof buf, "level difference %+.3f dB, mean |delta| after alignment %.5f", d, err / double(n));
        report("HQ matches STANDARD on a 1 kHz tone", std::fabs(d) < 0.05 && err / double(n) < 0.01, buf);
    }

    // 6. compression sanity and meters
    {
        double grs[3], meters[3];
        const int thr[3] = { 0, 11, 23 };
        std::vector<float> s = sine(-10.0, 1000.0, 2.0, fs), quiet = sine(-50.0, 1000.0, 2.0, fs);
        auto setup = [&](Engine& e, int t) {
            e.setParam(kPDiscreteThreshold, t); e.setParam(kPDiscreteRatio, 3); e.setParam(kPDiscreteGain, 6);
            e.setParam(kPOptical, 0); e.setParam(kPerChannel + kPOptical, 0); e.setParam(kGStereo, 1); e.prepare(fs);
        };
        Engine q; setup(q, 0);
        Render oq = run(q, quiet, quiet);
        const double staticGain = rmsDb(oq.l, size_t(fs)) - rmsDb(quiet, size_t(fs));   // make-up and transformer path, no compression
        for (int i = 0; i < 3; ++i) {
            Engine e; setup(e, thr[i]);
            Render o = run(e, s, s);
            grs[i] = staticGain - (rmsDb(o.l, size_t(fs)) - rmsDb(s, size_t(fs)));
            meters[i] = -e.meter(kOGrDiscreteL);
        }
        std::snprintf(buf, sizeof buf, "discrete GR at threshold 1/12/24: %.2f / %.2f / %.2f dB; meter reads %.2f / %.2f / %.2f",
                      grs[0], grs[1], grs[2], meters[0], meters[1], meters[2]);
        report("more threshold, more gain reduction, and the meter follows",
               grs[0] < 0.5 && grs[1] > 2.0 && grs[2] > grs[1] + 5.0 && std::fabs(meters[2] - grs[2]) < 1.5, buf);
    }

    // 7. click test
    {
        Engine e; e.setParam(kPDiscreteThreshold, 17); e.setParam(kPOpticalThreshold, 17); e.prepare(fs);
        std::vector<float> outL(noiseL.size()), outR(noiseL.size());
        for (size_t i = 0; i < noiseL.size(); i += 256) {
            const int n = int(std::min(size_t(256), noiseL.size() - i));
            const int step = int(i / 256);
            if (step % 40 == 10) e.setParam(kPDiscreteRatio, (step / 40) % 6);
            if (step % 40 == 20) e.setParam(kPTransformer, (step / 40) % kNumTransformers);
            if (step % 40 == 30) e.setParam(kPOptical, (step / 40) % 2);
            if (step % 40 == 35) e.setParam(kGHardwire, (step / 40) % 2);
            e.process(noiseL.data() + i, noiseR.data() + i, outL.data() + i, outR.data() + i, n);
        }
        float pk1, pk2; const bool ok = finite(outL, &pk1) && finite(outR, &pk2);
        const float worst = std::max(pk1, pk2);
        std::snprintf(buf, sizeof buf, "worst peak %.3f (input peak about 0.45)", worst);
        report("switching ratio, transformer and bypasses while noise plays stays bounded", ok && worst < 2.0f, buf);
    }

    // 8. transformer ceiling
    {
        Engine e; stagesOut(e); e.prepare(fs);
        Engine e2; stagesOut(e2); e2.prepare(fs);
        std::vector<float> lo = sine(18.0, 20.0, 1.0, fs), hi = sine(18.0, 1000.0, 1.0, fs);
        Render a = run(e, lo, lo), b = run(e2, hi, hi);
        const double gl = rmsDb(a.l, size_t(fs / 2)) - rmsDb(lo, size_t(fs / 2)), gh = rmsDb(b.l, size_t(fs / 2)) - rmsDb(hi, size_t(fs / 2));
        std::snprintf(buf, sizeof buf, "+18 dBFS: 20 Hz gain %+.2f dB, 1 kHz gain %+.2f dB", gl, gh);
        report("the core limits low frequencies, not 1 kHz", gl < -3.0 && gh > -1.0, buf);
    }

    // 9. a non-finite input sample must not poison the session, and the bypasses must stay transparent through it
    {
        Engine e; e.setParam(kPOpticalThreshold, 17); e.setParam(kPDiscreteThreshold, 15); e.setParam(kPTransformer, kIron); e.prepare(fs);
        std::vector<float> s = sine(-10.0, 1000.0, 2.0, fs);
        s[1000] = std::numeric_limits<float>::quiet_NaN(); s[1500] = std::numeric_limits<float>::infinity(); s[2000] = 1e30f;
        Render o = run(e, s, s);
        float pk; const bool ok = finite(o.l, &pk);
        const double tail = rmsDb(o.l, size_t(fs)), ref = rmsDb(s, size_t(fs));
        std::snprintf(buf, sizeof buf, "output %s, last-second level %+.2f dB re input (compressing)", ok ? "finite" : "NOT finite", tail - ref);
        report("a NaN, an Inf and a 1e30 input sample leave the engine finite and working", ok && tail - ref < -3.0 && tail - ref > -40.0, buf);
        Engine b; b.setParam(kGHardwire, 0); b.setParam(kPTransformer, kIron); b.prepare(fs);
        std::vector<float> t((size_t)fs, 0.1f); t[10] = std::numeric_limits<float>::quiet_NaN();
        Render ob = run(b, t, t);
        bool clean = true;
        for (size_t i = 0; i < t.size(); ++i) if (i != 10 && ob.l[i] != t[i]) { clean = false; break; }
        report("HARDWIRE OUT stays bit-transparent after a NaN sample (the NaN itself becomes 0)", clean && ob.l[10] == 0.0f, clean ? "identical elsewhere" : "differs");
    }

    // 10. KEY IN: with a silent key nothing compresses; with a loud key a quiet programme is compressed by it
    {
        std::vector<float> loud = sine(-10.0, 1000.0, 2.0, fs), quiet = sine(-40.0, 1000.0, 2.0, fs), none(loud.size(), 0.0f);
        auto render = [&](const std::vector<float>& prog, const std::vector<float>& key, int keyIn) {
            Engine e; e.setParam(kPOpticalThreshold, 19); e.setParam(kPDiscreteThreshold, 15); e.setParam(kGKeyIn, keyIn); e.setParam(kGStereo, 1); e.prepare(fs);
            std::vector<float> l(prog.size()), r(prog.size());
            e.process(prog.data(), prog.data(), l.data(), r.data(), int(prog.size()), key.data(), key.data());
            return rmsDb(l, size_t(fs)) - rmsDb(prog, size_t(fs));
        };
        const double own = render(loud, none, 0), silentKey = render(loud, none, 1), noGr = render(quiet, none, 0), keyed = render(quiet, loud, 1);
        std::snprintf(buf, sizeof buf, "loud programme: own sidechain %+.2f dB, silent key %+.2f dB; quiet programme: %+.2f dB alone, %+.2f dB keyed by a loud key",
                      own, silentKey, noGr, keyed);
        report("KEY IN replaces the detectors' signal", silentKey > own + 5.0 && keyed < noGr - 5.0 && std::fabs(silentKey - noGr) < 1.0, buf);
    }

    // cpp_reference.json for the Python cross-check
    {
        const std::string path = outDir + "/cpp_reference.json";
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            std::fprintf(f, "{\"fs\":48000,\"renders\":[");
            const int thr[3] = { 0, 11, 23 };
            for (int i = 0; i < 3; ++i) {
                Engine e; e.setParam(kPOpticalThreshold, thr[i]); e.setParam(kPDiscreteThreshold, thr[i]); e.prepare(fs);
                std::vector<float> s = sine(-10.0, 1000.0, 1.5, fs);
                Render o = run(e, s, s);
                std::fprintf(f, "%s{\"optical_threshold\":%d,\"discrete_threshold\":%d,\"level_dbfs\":-10,\"out_rms_db\":%.6f}",
                             i ? "," : "", thr[i] + 1, thr[i] + 1, rmsDb(o.l, size_t(fs)));
            }
            std::fprintf(f, "]}\n");
            std::fclose(f);
        }
    }

    std::printf("%s (%d failure%s)\n", gFail ? "FAILED" : "ALL PASSED", gFail, gFail == 1 ? "" : "s");
    return gFail ? 1 : 0;
}
