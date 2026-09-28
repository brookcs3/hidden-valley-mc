# Third-party notices

Hidden Valley Mastering Compressor is licensed under the GNU General Public License, version 3 only (see LICENSE). The build
compiles the following third-party software into the plugin binaries: DPF, and for the CLAP target the CLAP API headers. Nothing else is linked in: the DSP is this repository's own
code, and the Python fitting and test tools (NumPy, SciPy, numba, Pedalboard) are development dependencies that are not distributed
with the plugin.

## DPF (DISTRHO Plugin Framework)

- Source: https://github.com/DISTRHO/DPF, at the commit pinned in `scripts/build.sh` (`DPF_COMMIT`)
- Use: the VST3, CLAP and Audio Unit wrappers, and the plugin entry points. On macOS the AU is built from a copy of DPF with
  `scripts/dpf-au-parameter-strings.patch` applied (a change to DPF's AU wrapper only, under the same licence).
- Licence: ISC. The text below is DPF's LICENSE file.

```
Copyright (C) 2012-2025 Filipe Coelho <falktx@falktx.com>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## CLAP API headers

- Source: https://github.com/free-audio/clap, as vendored by DPF in `distrho/src/clap/`
- Use: the CLAP target only (`HiddenValleyMC.clap`); the VST3 target uses DPF's own ISC-licensed interface headers.
- Licence: MIT.

```
MIT License

Copyright (c) 2021 Alexandre BIQUE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Reference measurements

The fitted constants in `src/dsp/FittedConstants.hpp` were produced by measuring a licensed copy of a commercial plug-in with test
signals (see `docs/MODEL.md`). No code, presets, resources or documentation from that product are included in, or were consulted
for, this repository. "Shadow Hills" and the names of that product and its publisher are trademarks of their owners; this project is
not affiliated with, endorsed by or derived from any of them.
