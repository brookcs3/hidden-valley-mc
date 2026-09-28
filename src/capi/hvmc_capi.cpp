// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// A C interface to the same Engine the plugin runs, built as a shared library for the Python fitting and tests (fit/, tests/), which
// load it with ctypes. Not part of the plugin. Build: see scripts/build-capi.sh.
#include <cstring>
#include "../dsp/Engine.hpp"

using namespace hvmc;

extern "C" {

void* hvmc_new() { return new Engine(); }
void hvmc_free(void* e) { delete static_cast<Engine*>(e); }
void hvmc_prepare(void* e, double fs) { static_cast<Engine*>(e)->prepare(fs); }
void hvmc_set_param(void* e, int index, int value) { static_cast<Engine*>(e)->setParam(index, value); }
int hvmc_get_param(void* e, int index) { return static_cast<Engine*>(e)->param(index); }
void hvmc_process(void* e, const float* inL, const float* inR, float* outL, float* outR, int n)
{
    static_cast<Engine*>(e)->process(inL, inR, outL, outR, n);
}
int hvmc_latency(void* e) { return static_cast<Engine*>(e)->latency(); }
double hvmc_meter(void* e, int outIndex) { return static_cast<Engine*>(e)->meter(outIndex); }

// calibration: layout, defaults (fitted or priors), and replacement
int hvmc_cal_size() { return kCalSize; }
int hvmc_cal_field_count() { int n = 0; calFields(&n); return n; }
const char* hvmc_cal_field(int i, int* offset, int* count)
{
    int n = 0;
    const CalFieldInfo* f = calFields(&n);
    if (i < 0 || i >= n) return nullptr;
    *offset = f[i].offset; *count = f[i].count;
    return f[i].name;
}
unsigned long long hvmc_cal_layout_hash() { return calLayoutHash(); }
int hvmc_cal_defaults(double* out) { std::memcpy(out, calDefaults().v, sizeof(double) * kCalSize); return calDefaults().fitted ? 1 : 0; }
void hvmc_cal_priors(double* out) { calPriors(out); }
void hvmc_set_calibration(void* e, const double* v) { static_cast<Engine*>(e)->setCalibration(v); }

// parameters: table shared with the plugin
int hvmc_num_input_params() { return kNumInputParams; }
int hvmc_num_params() { return kNumParams; }
int hvmc_param_steps(int index) { return paramSteps(index); }
int hvmc_param_default(int index) { return paramDefault(index); }
void hvmc_param_name(int index, char* out, int cap) { paramName(index, out, cap); }
void hvmc_param_label(int index, int value, char* out, int cap) { positionLabel(index, value, out, cap); }
const char* hvmc_param_description(int index) { return paramDescription(index); }

}
