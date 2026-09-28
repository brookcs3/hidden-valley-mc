# SPDX-FileCopyrightText: 2026 Cameron Brooks
# SPDX-License-Identifier: GPL-3.0-only
"""Design the HQ-mode half-band filter (src/dsp/Oversampler.hpp) and print its unique taps and figures."""
import numpy as np
from scipy.signal import remez, freqz
n = 79
h = remez(n, [0, 20 / 96, 28 / 96, 0.5], [1, 0], fs=1.0, maxiter=200)
c = (n - 1) // 2; offs = np.arange(n) - c
h[(offs % 2 == 0) & (offs != 0)] = 0.0; h[c] = 0.5
w, H = freqz(h, worN=16384, fs=1.0)
pb = np.abs(H[w <= 20 / 96]); sb = np.abs(H[w >= 28 / 96])
print(f"passband ripple {20*np.log10(pb.max()/pb.min()):.6f} dB, stopband {20*np.log10(sb.max()):.1f} dB")
for v in h[(offs % 2 != 0) & (offs < 0)]:
    print(f"    {v:.17g},")
