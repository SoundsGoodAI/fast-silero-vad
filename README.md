# Fast Silero VAD

<p>
  <a href="https://github.com/SoundsGoodAI/fast-silero-vad/actions/workflows/ci.yml"><img src="https://github.com/SoundsGoodAI/fast-silero-vad/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://pypi.org/project/fast-silero-vad/"><img src="https://img.shields.io/pypi/v/fast-silero-vad" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12, 3.13, and 3.14">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-4C566A" alt="Linux and macOS">
  <img src="https://img.shields.io/badge/typing-typed-2F80ED" alt="Typed Python package">
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/lint-Ruff-261230?logo=ruff&amp;logoColor=white" alt="Linted with Ruff"></a>
  <a href="https://github.com/SoundsGoodAI/fast-silero-vad/blob/v0.1.2/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license"></a>
</p>

**Fast Silero VAD** combines a streamlined ONNX graph with an optimized spectral
frontend to improve CPU throughput while preserving numerically equivalent
speech probabilities.

## Benchmarks

<table>
  <thead>
    <tr>
      <th>
        <div align="center"><big>Apple M4 Max (single thread)</big></div>
      </th>
      <th>
        <div align="center"><big>AMD EPYC 9655 (single thread)</big></div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><img src="https://raw.githubusercontent.com/SoundsGoodAI/fast-silero-vad/v0.1.2/docs/plots/duration_sweep_combined_speedup_m4.svg" width="100%" alt="Speedup by input duration on Apple M4 Max"></td>
      <td width="50%"><img src="https://raw.githubusercontent.com/SoundsGoodAI/fast-silero-vad/v0.1.2/docs/plots/duration_sweep_combined_speedup_amd.svg" width="100%" alt="Speedup by input duration on AMD EPYC 9655"></td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td width="50%"><img src="https://raw.githubusercontent.com/SoundsGoodAI/fast-silero-vad/v0.1.2/docs/plots/duration_sweep_combined_rtfx_m4.svg" width="100%" alt="RTFx by input duration on Apple M4 Max"></td>
      <td width="50%"><img src="https://raw.githubusercontent.com/SoundsGoodAI/fast-silero-vad/v0.1.2/docs/plots/duration_sweep_combined_rtfx_amd.svg" width="100%" alt="RTFx by input duration on AMD EPYC 9655"></td>
    </tr>
  </tbody>
</table>

**Solid lines** measure probability inference.
**Dashed lines** include speech segmentation.
**RTFx** is a ratio of input audio duration to processing time.

- **Upstream throughput plateaus.** From 32 to 1024 ms, official Python Silero
  moves only from 392 to 457 RTFx on AMD and from 371 to 390 RTFx on M4. The
  official C++ implementation is effectively flat at 574 to 571 RTFx on AMD
  and 444 to 437 RTFx on M4.
- **The gain reproduces across CPU families.** At 1024 ms, Fast RFFT reaches
  1,448 RTFx on Linux/x86-64 and 1,165 RTFx on macOS/arm64: 3.17x and 2.98x
  the throughput of official Python, and 2.53x and 2.67x that of official C++.
- **The graph rewrite pays off without RFFT.** Standard Fast ONNX reaches 1,147
  RTFx on AMD and 755 RTFx on M4 at 1024 ms. That is 2.51x and
  1.93x the Python baseline, or 2.01x and 1.73x the C++ baseline.
- **RFFT provides the largest incremental gain.** The custom spectral frontend
  adds another 26% on AMD and 54% on M4 over the standard Fast ONNX graph at
  1024 ms.
- **The full VAD inference with segmentation keeps the gain.** The complete
  RFFT pipeline with `Numba`-optimized segmentation reaches 1,438 RTFx on
  AMD and 1,162 RTFx on M4, only 0.7% and 0.3% below probability inference
  alone. It delivers 3.15x and 2.98x the throughput of official Python
  Silero VAD, and 2.52x and 2.66x that of the official Silero VAD C++
  implementation.

Measurements use deterministic 16 kHz mono audio, one ONNX Runtime thread, and
50 warmups. Values are medians of 500 runs. Model loading and WAV decoding are
excluded.

The Python baseline uses `load_silero_vad(onnx=True).audio_forward()`. The C++
baseline uses the pinned upstream
[`examples/c++`](https://github.com/snakers4/silero-vad/tree/b163605b3f44c3aadf28f97b125a2f7c461e9a7f/examples/c%2B%2B)
implementation compiled with C++20, `-O3`, and `-DNDEBUG`.

## Quick Start

Install Fast Silero VAD with the export dependencies:

```bash
pip install "fast-silero-vad[export]"
```

The RFFT export requires a C++20 compiler available as `c++` and network access
on the first run.

### Export

Export a 16 kHz offline bundle with the optimized RFFT frontend:

```bash
fast-silero-vad-export \
  --output-dir-path models/fast_silero_vad_16k \
  --model-type offline_vad \
  --vad-branch 16k \
  --threshold 0.01 \
  --min-speech-duration-ms 100 \
  --max-speech-duration-ms 30000 \
  --min-silence-duration-ms 1000 \
  --speech-pad-ms 300 \
  --use-onnxruntime-custom-op
```

The exporter uses the JIT checkpoint bundled with `silero-vad>=6.2.1` and
writes the runtime and segmenter settings to `model_config.yaml`. For streaming,
use a new output directory and `--model-type streaming_vad`. Use
`--vad-branch 8k` to export an 8 kHz bundle.

### CLI

```bash
fast-silero-vad \
  --model-dir models/fast_silero_vad_16k \
  --wav-dir /path/to/wav/files \
  --output-path segments.tsv
```

### Python

The API accepts one-dimensional normalized floating-point audio. Offline models
reset after each call:

```python
import numpy as np

from fast_silero_vad import VAD

audio = np.zeros(16000, dtype=np.float32)

vad = VAD("models/fast_silero_vad_16k")
vad.apply_samplerate(16000)
segments = vad(audio)
```

A streaming model preserves state until `final=True`:

```python
vad = VAD("models/fast_silero_vad_16k_streaming")
vad.apply_samplerate(16000)

segments = []
for start in range(0, len(audio), 1600):
    end = min(len(audio), start + 1600)
    segments.extend(vad(audio[start:end], final=end == len(audio)))
```

## Why Fast Silero VAD Scales

At 16 kHz, each Silero window is 512 samples, or 32 ms. The official wrapper
executes one ONNX Runtime call per window (see
[`audio_forward`](https://github.com/snakers4/silero-vad/blob/b163605b3f44c3aadf28f97b125a2f7c461e9a7f/src/silero_vad/utils_vad.py#L98-L109)
and
[`get_speech_timestamps`](https://github.com/snakers4/silero-vad/blob/b163605b3f44c3aadf28f97b125a2f7c461e9a7f/src/silero_vad/utils_vad.py#L312-L330)).
Longer audio therefore increases the number of calls linearly and leaves RTFx
roughly flat.

Fast Silero VAD returns the same 32 ms probability sequence with two independent
optimizations:

1. **Multi-window execution:** one ONNX Runtime call processes multiple 32 ms
   windows.
2. **RFFT frontend:** a C++ ONNX Runtime custom operator replaces the Conv1d
   Fourier basis with a real FFT.

The standard Fast ONNX graph gains more as input grows because one ONNX Runtime
call handles more 32 ms windows, amortizing Python, tensor binding, and graph
dispatch overhead across the input. At 32 ms there is only one window, so
multi-window execution offers almost no advantage in this case; most of the
speedup comes from the real FFT frontend.

## Numerical Equivalence

Numerically equivalent means float32-close, not bit-identical. The
[`export equivalence tests`](https://github.com/SoundsGoodAI/fast-silero-vad/blob/v0.1.2/tests/test_export_equivalence.py) run 12 seeded PCM
chunks statefully through the original Silero JIT checkpoint and through both
standard and real FFT exports at 8 kHz and 16 kHz. Every speech probability must
match within `rtol=1e-4` and `atol=1e-6`.

## Reproducing

These commands reproduce the duration sweep. They require
[`uv`](https://docs.astral.sh/uv/getting-started/installation/), a C++20 compiler
named `c++`, and network access on the first run. ONNX Runtime is limited to one
thread.

### Setup

```bash
pip install uv
uv sync --frozen --extra export --extra plot
mkdir -p benchmarks/output/m4_max

EXPORT_ARGS=(
  --model-type offline_vad
  --vad-branch 16k
  --threshold 0.01
  --min-speech-duration-ms 100
  --max-speech-duration-ms 30000
  --min-silence-duration-ms 1000
  --speech-pad-ms 0
)

uv run --frozen --extra export fast-silero-vad-export \
  --output-dir-path models/benchmark_standard_16k \
  "${EXPORT_ARGS[@]}"

uv run --frozen --extra export fast-silero-vad-export \
  --output-dir-path models/benchmark_custom_op_16k \
  "${EXPORT_ARGS[@]}" \
  --use-onnxruntime-custom-op
```

Python generates its input in memory. Create the equivalent mono 16-bit WAV for
the original Silero VAD C++ harness:

```bash
uv run --frozen --extra export python - <<'PY'
import wave
from pathlib import Path

import numpy as np

from benchmarks.utils import make_synthetic_audio

output_path = Path("benchmarks/output/m4_max/synthetic_1024ms.wav")
audio = make_synthetic_audio(1.024, 16000)
pcm = np.clip(audio * (2**15 - 1), -32768, 32767).astype(np.int16)

with wave.open(str(output_path), "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(16000)
    output.writeframes(pcm.tobytes())
PY
```

### Python

```bash
uv run --frozen --extra export python -m benchmarks.duration_sweep \
  --standard-model-dir models/benchmark_standard_16k \
  --custom-op-model-dir models/benchmark_custom_op_16k \
  --durations-ms 32 64 128 256 512 1024 \
  --samplerate 16000 \
  --threshold 0.01 \
  --min-speech-duration-ms 100 \
  --max-speech-duration-ms 30000 \
  --min-silence-duration-ms 1000 \
  --speech-pad-ms 0 \
  --warmup 50 \
  --repeats 500 \
  --output-tsv-path benchmarks/output/m4_max/python.tsv
```

### C++

The builder downloads the pinned upstream source and matching ONNX Runtime
headers, then packages the runtime library beside the executable.

```bash
uv run --frozen --extra export python benchmarks/cpp/build.py

OFFICIAL_ONNX=$(find .venv -path '*/silero_vad/data/silero_vad.onnx' -print -quit)

benchmarks/cpp/build/vad_benchmark \
  "$OFFICIAL_ONNX" \
  benchmarks/output/m4_max/cpp.tsv \
  50 500 \
  benchmarks/output/m4_max/synthetic_1024ms.wav \
  0 \
  32 64 128 256 512 1024
```

### Plots

```bash
uv run --frozen --extra plot python -m benchmarks.plot_benchmark \
  --input-tsv-path \
    benchmarks/output/m4_max/python.tsv \
    benchmarks/output/m4_max/cpp.tsv \
  --output-dir-path docs/plots/m4_max \
  --formats svg
```

## License

Fast Silero VAD source code is licensed under
[Apache-2.0](https://github.com/SoundsGoodAI/fast-silero-vad/blob/v0.1.2/LICENSE).
Exported model weights are derived from Silero VAD and remain subject to the
[Silero VAD MIT License](https://github.com/SoundsGoodAI/fast-silero-vad/blob/v0.1.2/src/fast_silero_vad/licenses/SILERO-VAD-MIT.txt).
This project is independent and is not affiliated with or endorsed by the
Silero Team. See the project
[NOTICE](https://github.com/SoundsGoodAI/fast-silero-vad/blob/v0.1.2/NOTICE).
