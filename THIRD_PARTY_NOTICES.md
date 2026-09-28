# Third-party notices

Hidden Valley Mastering Compressor is licensed under the GNU General Public License, version 3 only (see LICENSE). The build
compiles the following third-party software into the plugin binaries. Nothing else is linked in: the DSP is this repository's own
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

## Reference measurements

The fitted constants in `src/dsp/FittedConstants.hpp` were produced by measuring a licensed copy of a commercial plug-in with test
signals (see `docs/MODEL.md`). No code, presets, resources or documentation from that product are included in, or were consulted
for, this repository. "Shadow Hills" and the names of that product and its publisher are trademarks of their owners; this project is
not affiliated with, endorsed by or derived from any of them.
