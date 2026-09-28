// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// Hidden Valley Mastering Compressor: the DPF plugin. Holds the panel parameters, hands them to the Engine (src/dsp/Engine.hpp), and
// publishes the meters as read-only output parameters. All processing is in src/dsp/; see README.md for the controls and docs/ for the
// model and how it was measured.
#include "DistrhoPlugin.hpp"
#include <atomic>
#include <cmath>
#include "HVMCParams.hpp"
#include "dsp/Engine.hpp"

START_NAMESPACE_DISTRHO

class HiddenValleyPlugin : public Plugin
{
public:
    HiddenValleyPlugin()
        : Plugin(hvmc::kNumParams, 0, 0)
    {
        for (int i = 0; i < hvmc::kNumInputParams; ++i) fValues[i] = float(hvmc::paramDefault(i));
        for (int i = hvmc::kNumInputParams; i < hvmc::kNumParams; ++i) fValues[i] = 0.0f;
        fEngine = new hvmc::Engine();
        fDirty.store(true);
        setLatency(0);
    }

    ~HiddenValleyPlugin() override { delete fEngine; }

protected:
    const char* getLabel() const override { return "HiddenValleyMC"; }
    const char* getDescription() const override
    {
        return "Stereo mastering compressor after the Shadow Hills Mastering Compressor: an optical stage (electroluminescent panel "
               "and CdS cell, feedback detection) into a discrete feed-forward VCA stage, then a choice of output transformers "
               "(Nickel, Iron with its Class-A stage, Steel) plus four digital-only material positions (Gold, Uranium, Germanium, "
               "Plutonium). Every panel control, stepped like the hardware. A behavioural model fitted to measurements of a reference "
               "and to published hardware data; not affiliated with Shadow Hills Industries.";
    }
    const char* getMaker() const override { return "Cameron Brooks"; }
    const char* getHomePage() const override { return "https://github.com/brookcs3/hidden-valley-mc"; }
    const char* getLicense() const override { return "GPL-3.0-only"; }
    uint32_t getVersion() const override { return d_version(1, 0, 0); }
    int64_t getUniqueId() const override { return d_cconst('H', 'V', 'm', 'c'); }

    void initAudioPort(bool input, uint32_t index, AudioPort& port) override
    {
        port.groupId = kPortGroupStereo;
        Plugin::initAudioPort(input, index, port);
    }

    void initParameter(uint32_t index, Parameter& p) override
    {
        char buf[64];
        hvmc::paramName(int(index), buf, sizeof(buf));
        p.name = buf;
        p.shortName = buf;
        p.symbol = buf;
        p.unit = "";   // units live in the names (Pedalboard appends the unit field to the name otherwise)
        p.description = hvmc::paramDescription(int(index));
        if (hvmc::isOutput(int(index))) {
            p.hints = kParameterIsOutput;
            p.ranges.min = -80.0f;
            p.ranges.max = 20.0f;
            p.ranges.def = 0.0f;
            if (index == uint32_t(hvmc::kOMagicEye)) p.ranges.max = 0.0f;
            return;
        }
        const int steps = hvmc::paramSteps(int(index));
        p.hints = kParameterIsAutomatable | kParameterIsInteger;
        if (steps == 2) p.hints |= kParameterIsBoolean;
        p.ranges.min = 0.0f;
        p.ranges.max = float(steps - 1);
        p.ranges.def = float(hvmc::paramDefault(int(index)));
        ParameterEnumerationValue* ev = new ParameterEnumerationValue[steps];
        for (int v = 0; v < steps; ++v) {
            hvmc::positionLabel(int(index), v, buf, sizeof(buf));
            ev[v].value = float(v);
            ev[v].label = buf;
        }
        p.enumValues.count = uint8_t(steps);
        p.enumValues.restrictedMode = true;
        p.enumValues.values = ev;
        p.enumValues.deleteLater = true;
    }

    float getParameterValue(uint32_t index) const override
    {
        return index < uint32_t(hvmc::kNumParams) ? fValues[index].load(std::memory_order_relaxed) : 0.0f;
    }

    void setParameterValue(uint32_t index, float value) override
    {
        if (index >= uint32_t(hvmc::kNumInputParams)) return;   // meters are read-only
        const int steps = hvmc::paramSteps(int(index));
        float v = std::round(value);
        if (v < 0.0f) v = 0.0f;
        if (v > float(steps - 1)) v = float(steps - 1);
        if (v != fValues[index].load(std::memory_order_relaxed)) {
            fValues[index].store(v, std::memory_order_relaxed);
            fDirty.store(true);
            if (index == uint32_t(hvmc::kGQuality)) setLatency(uint32_t(fEngine->latencyFor(int(v))));
        }
    }

    void activate() override
    {
        pushControls();
        fEngine->prepare(getSampleRate());
        setLatency(uint32_t(fEngine->latency()));
    }

    void sampleRateChanged(double newSampleRate) override
    {
        pushControls();
        fEngine->prepare(newSampleRate);
    }

    void run(const float** inputs, float** outputs, uint32_t frames) override
    {
        const hvmc::DenormalGuard guard;
        if (fDirty.exchange(false)) pushControls();
        fEngine->process(inputs[0], inputs[1], outputs[0], outputs[1], int(frames));
        for (int i = hvmc::kNumInputParams; i < hvmc::kNumParams; ++i) {
            double m = fEngine->meter(i);
            const double hi = i == hvmc::kOMagicEye ? 0.0 : 20.0;   // each output stays inside its declared range
            if (m < -80.0) m = -80.0;
            if (m > hi) m = hi;
            fValues[i].store(float(m), std::memory_order_relaxed);
        }
    }

private:
    void pushControls()
    {
        for (int i = 0; i < hvmc::kNumInputParams; ++i) fEngine->setParam(i, int(fValues[i].load(std::memory_order_relaxed)));
    }

    std::atomic<float> fValues[hvmc::kNumParams];   // written by the host's thread and by run(), read by both
    hvmc::Engine* fEngine;
    std::atomic<bool> fDirty;

    DISTRHO_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HiddenValleyPlugin)
};

Plugin* createPlugin()
{
    return new HiddenValleyPlugin();
}

END_NAMESPACE_DISTRHO
