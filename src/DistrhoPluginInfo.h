// SPDX-FileCopyrightText: 2026 Cameron Brooks
// SPDX-License-Identifier: GPL-3.0-only
//
// Hidden Valley Mastering Compressor: DPF plugin description.
// The VST3 class id is { 'DPF ', 'clas', getUniqueId() = 'HVmc', DISTRHO_PLUGIN_BRAND_ID = 'Cbrk' } (DPF builds it from these two).
#ifndef DISTRHO_PLUGIN_INFO_H_INCLUDED
#define DISTRHO_PLUGIN_INFO_H_INCLUDED

#define DISTRHO_PLUGIN_BRAND   "Cameron Brooks"
#define DISTRHO_PLUGIN_NAME    "Hidden Valley Mastering Compressor"
#define DISTRHO_PLUGIN_URI     "https://github.com/brookcs3/hidden-valley-mc"
#define DISTRHO_PLUGIN_CLAP_ID "io.github.brookcs3.hidden-valley-mc"

#define DISTRHO_PLUGIN_BRAND_ID  Cbrk
#define DISTRHO_PLUGIN_UNIQUE_ID HVmc

#define DISTRHO_PLUGIN_HAS_UI        0
#define DISTRHO_PLUGIN_IS_RT_SAFE    1
#define DISTRHO_PLUGIN_NUM_INPUTS    2
#define DISTRHO_PLUGIN_NUM_OUTPUTS   2
#define DISTRHO_PLUGIN_WANT_PROGRAMS 0
#define DISTRHO_PLUGIN_WANT_STATE    0
#define DISTRHO_PLUGIN_WANT_LATENCY  1
#define DISTRHO_PLUGIN_CLAP_FEATURES "audio-effect", "compressor", "stereo"
#define DISTRHO_PLUGIN_VST3_CATEGORIES "Fx|Dynamics|Stereo"
#define DISTRHO_PLUGIN_AU_TYPE aufx

#endif // DISTRHO_PLUGIN_INFO_H_INCLUDED
