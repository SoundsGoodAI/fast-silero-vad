# Fast Silero VAD

<p>
  <a href="https://github.com/SoundsGoodAI/fast-silero-vad/actions/workflows/ci.yml"><img src="https://github.com/SoundsGoodAI/fast-silero-vad/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <img src="https://img.shields.io/badge/Python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12, 3.13, and 3.14">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS-4C566A" alt="Linux and macOS">
  <img src="https://img.shields.io/badge/typing-typed-2F80ED" alt="Typed Python package">
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/lint-Ruff-261230?logo=ruff&amp;logoColor=white" alt="Linted with Ruff"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license"></a>
</p>

**Fast Silero VAD** combines a streamlined ONNX graph with an optimized spectral frontend to reduce inference overhead and improve CPU throughput while preserving speech-probability outputs numerically equivalent to the original Silero VAD.

<table>
  <tr>
    <th colspan="2">
      <div align="center"><big>AMD EPYC 9655 (single thread)</big></div>
    </th>
  </tr>
  <tr>
    <td width="50%"><img src="docs/plots/duration_sweep_combined_rtfx.svg" width="100%" alt="RTFx by input duration on AMD EPYC 9655"></td>
    <td width="50%"><img src="docs/plots/duration_sweep_combined_speedup.svg" width="100%" alt="Speedup by input duration on AMD EPYC 9655"></td>
  </tr>
</table>

<small>
<strong>Solid lines</strong> measure probability inference.
<strong>Dashed lines</strong> include speech segmentation.
<strong>RTFx</strong> is a ratio of processed audio duration to processing time.<br><br>
</small>

- The Fast RFFT Silero VAD engine rises from **657 RTFx** (**1.68x** speedup) at
  32 ms to **1,448 RTFx** (**3.17x** speedup) at 1024 ms.
- With the Numba-compiled Segmenter included, the full Fast RFFT pipeline reaches
  **593 RTFx** (**1.51x** speedup) at 32 ms and **1,438 RTFx** (**3.15x**
  speedup) at 1024 ms.
- The standard Fast Silero VAD graph reaches **2.51x** speedup at 1024 ms; the
  full pipeline with Segmenter reaches **2.49x**.
- Official Python and C++ Silero VAD settle into approximately constant throughput as
  input duration grows.

Silero ONNX (Python) calls `load_silero_vad(onnx=True).audio_forward()` from the
installed official `silero-vad` package.
Silero ONNX (C++) uses the upstream Silero VAD
[`examples/c++`](https://github.com/snakers4/silero-vad/tree/b163605b3f44c3aadf28f97b125a2f7c461e9a7f/examples/c%2B%2B)
ONNX implementation. Its unchanged `silero.cc` is
compiled with C++20, `-O3`, and `-DNDEBUG` and linked against ONNX Runtime.
Values are median latencies from 500 runs after 50 warmups.
Model loading and WAV decoding are excluded.

## Quick Start

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on Linux
or macOS with the official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS, Homebrew can be used instead:

```bash
brew install uv
```

Install the project and its export dependencies:

```bash
uv sync --extra export
```

Export a 16 kHz offline bundle with the optimized real FFT frontend. The
exporter uses the JIT checkpoint bundled with the official `silero-vad`
package:

```bash
uv run --frozen --extra export fast-silero-vad-export \
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

The exporter stores the segmenter policy in `model_config.yaml`; both offline
and streaming runtimes use these values:

| parameter | value | behavior |
|---|---:|---|
| `threshold` | `0.01` | Opens or continues speech when the model probability is at least this value. |
| `min_speech_duration_ms` | `100` | Discards detected speech shorter than 100 ms. |
| `max_speech_duration_ms` | `30000` | Splits longer speech near a low-probability region while keeping both sides at least the minimum speech duration. |
| `min_silence_duration_ms` | `1000` | Closes an active segment after 1000 ms of below-threshold probabilities. |
| `speech_pad_ms` | `300` | Adds 300 ms context around natural VAD boundaries. Forced maximum-duration splits are always adjacent and unpadded. |

The example creates an `offline_vad` bundle. Fast Silero VAD supports both
runtime modes:

| model type | input behavior | state behavior |
|---|---|---|
| `offline_vad` | One complete audio array per call | Resets after every call |
| `streaming_vad` | Successive audio chunks | Preserved until `final=True` |

To create a streaming bundle, rerun the export command with a different output
directory and `--model-type streaming_vad`. The command-line detector reads the
model type from `model_config.yaml` and handles either mode automatically:

```bash
uv run --frozen fast-silero-vad \
  --model-dir models/fast_silero_vad_16k \
  --wav-dir /path/to/wav/files \
  --output-path segments.tsv
```

The Python API accepts one-dimensional normalized floating-point audio. An
offline model processes the complete recording in one call:

```python
import numpy as np

from fast_silero_vad import VAD

audio = np.zeros(16000, dtype=np.float32)

vad = VAD("models/fast_silero_vad_16k")
vad.apply_samplerate(16000)
segments = vad(audio)
```

A streaming model preserves inference and segmentation state between chunks:

```python
import numpy as np

from fast_silero_vad import VAD

audio = np.zeros(16000, dtype=np.float32)

vad = VAD("models/fast_silero_vad_16k_streaming")
vad.apply_samplerate(16000)

segments = []
chunk_samples = 1600
for start in range(0, len(audio), chunk_samples):
    end = min(len(audio), start + chunk_samples)
    segments.extend(vad(audio[start:end], final=end == len(audio)))
```

## Why Fast Silero VAD Scales

At 16 kHz, one Silero model window is 512 samples:

```text
512 samples / 16,000 samples per second = 0.032 seconds = 32 ms
```

The official low-level ONNX wrapper accepts exactly one 512-sample window per
call. Its long-audio paths pad the final window when necessary and iterate over
the input in 512-sample steps:

- [`OnnxWrapper.audio_forward`](https://github.com/snakers4/silero-vad/blob/b163605b3f44c3aadf28f97b125a2f7c461e9a7f/src/silero_vad/utils_vad.py#L98-L109)
- [`get_speech_timestamps`](https://github.com/snakers4/silero-vad/blob/b163605b3f44c3aadf28f97b125a2f7c461e9a7f/src/silero_vad/utils_vad.py#L312-L330)

Consequently, doubling the audio duration doubles the number of ONNX Runtime
calls. Processing time grows almost linearly with audio duration, leaving the
official Silero VAD RTFx approximately constant. The upstream official C++ ONNX
implementation follows the same pattern, but its lower language overhead gives
it a higher constant RTFx.

Fast Silero VAD accepts several model windows in one ONNX graph call and returns the
same sequence of 32 ms probabilities. This produces two independent gains:

1. **Single-call multi-window execution.** The standard graph amortizes Python,
   tensor binding, dispatch, and ONNX Runtime call overhead across all windows.
2. **Optimized real FFT frontend.** The custom-op graph replaces the original
   Conv1d Fourier basis with a C++ real FFT using precomputed Hann-window and
   twiddle values.

The standard graph does not improve the 32 ms case because one model window
leaves no repeated call overhead to amortize. The custom real FFT frontend still
provides a measurable improvement at that duration.

## Reproducing

The benchmark tools are repository-only. Run the commands below from a project
checkout after installing its dependencies with `uv`.

### Python

```bash
taskset -c 0 uv run --frozen --extra export python -m benchmarks.duration_sweep \
  --standard-model-dir /path/to/standard_bundle \
  --custom-op-model-dir /path/to/custom_op_bundle \
  --wav-path /path/to/audio.wav \
  --wav-offset-sec 0 \
  --durations-ms 32 64 128 256 512 1024 \
  --threshold 0.005 \
  --min-speech-duration-ms 100 \
  --max-speech-duration-ms 30000 \
  --min-silence-duration-ms 1000 \
  --speech-pad-ms 0 \
  --warmup 50 \
  --repeats 500 \
  --output-tsv-path benchmarks/results/python.tsv
```

### C++

The build script downloads the pinned upstream `examples/c++` implementation,
compiles its ONNX path with C++20, and links it against the exact ONNX Runtime
shared library installed in the project environment. Matching C/C++ headers
are downloaded from the corresponding ONNX Runtime tag when absent from the
local cache.

```bash
uv run --frozen python benchmarks/cpp/build.py

taskset -c 0 benchmarks/cpp/build/vad_benchmark \
  /path/to/silero_vad.onnx \
  benchmarks/results/cpp.tsv \
  50 500 \
  /path/to/audio.wav \
  0 \
  32 64 128 256 512 1024
```

### Plots

```bash
uv run --frozen --extra plot python -m benchmarks.plot_benchmark \
  --input-tsv-path benchmarks/results/python.tsv benchmarks/results/cpp.tsv \
  --output-dir-path docs/plots \
  --formats svg
```
