#!/usr/bin/env python3
"""Benchmark Silero VAD engines and complete Fast pipelines by input duration."""

import argparse
import csv
import os
import platform
import socket
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime
from omegaconf import OmegaConf

from fast_silero_vad.constants import VAD_CONFIG_FILE
from fast_silero_vad.segmenter.segmenter import SpeechSegmenter
from fast_silero_vad.utils import validate_model_config
from fast_silero_vad.vad import VAD

from .utils import (
    get_cpu_model,
    get_official_model,
    make_synthetic_audio,
    read_wav,
    run_fast_engine,
    run_fast_pipeline,
    run_original_onnx,
    time_function,
)

DEFAULT_DURATIONS_MS = (32, 64, 128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    """Parse duration-sweep command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed model, audio, segmenter, timing, and output options.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark official, standard, and custom-op Silero VAD runtimes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--standard-model-dir",
        required=True,
        type=str,
        help="Standard Fast Silero bundle.",
    )
    parser.add_argument(
        "--custom-op-model-dir",
        required=True,
        type=str,
        help="Real FFT Fast Silero bundle.",
    )
    parser.add_argument(
        "--durations-ms",
        nargs="+",
        type=int,
        default=DEFAULT_DURATIONS_MS,
        help="Input durations to benchmark in milliseconds.",
    )
    parser.add_argument(
        "--samplerate", type=int, default=16000, help="Synthetic sample rate."
    )
    parser.add_argument(
        "--wav-path",
        type=str,
        default="",
        help="Optional mono 16-bit PCM WAV used instead of synthetic audio.",
    )
    parser.add_argument(
        "--wav-offset-sec",
        type=float,
        default=0.0,
        help="Start offset of the benchmark excerpt within --wav-path.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01, help="Segmenter threshold."
    )
    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=100,
        help="Segmenter minimum speech duration.",
    )
    parser.add_argument(
        "--max-speech-duration-ms",
        type=int,
        default=30000,
        help="Segmenter maximum speech duration.",
    )
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=1000,
        help="Segmenter minimum silence duration.",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=0,
        help="Segmenter speech padding.",
    )
    parser.add_argument(
        "--repeats", type=int, default=200, help="Timed repeats per runtime."
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup repeats per runtime."
    )
    parser.add_argument(
        "--output-tsv-path",
        required=True,
        type=str,
        help="Destination TSV for benchmark rows.",
    )
    return parser.parse_args()


def write_results(
    output_path: str,
    args: argparse.Namespace,
    samplerate: int,
    rows: list[dict[str, str]],
) -> None:
    """Write duration-sweep metadata and measurements to a TSV file.

    Host and runtime metadata are repeated on every row so the TSV remains
    self-contained when filtered or combined with C++ benchmark results.
    ``compiler`` and ``build`` describe the Python interpreter used to run the
    benchmark.

    Parameters
    ----------
    output_path : str
        Destination TSV path.
    args : argparse.Namespace
        Benchmark options containing timing and segmenter parameters.
    samplerate : int
        Sample rate of the benchmark audio in Hz.
    rows : list[dict[str, str]]
        Measurement rows to combine with the shared metadata.
    """

    timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    metadata = {
        "timestamp": timestamp,
        "host": socket.gethostname(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": get_cpu_model(),
        "logical_cores": str(os.cpu_count() or ""),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "silero_vad": version("silero-vad"),
        "numpy": np.__version__,
        "samplerate": str(samplerate),
        "warmup": str(args.warmup),
        "repeats": str(args.repeats),
        "threshold": str(args.threshold),
        "min_speech_duration_ms": str(args.min_speech_duration_ms),
        "max_speech_duration_ms": str(args.max_speech_duration_ms),
        "min_silence_duration_ms": str(args.min_silence_duration_ms),
        "speech_pad_ms": str(args.speech_pad_ms),
        "compiler": platform.python_compiler(),
        "build": " ".join(platform.python_build()),
    }
    fieldnames = [
        *metadata,
        "language",
        "benchmark",
        "audio_duration_ms",
        "audio_duration_sec",
        "best_sec",
        "median_sec",
        "rtfx",
        "speedup",
        "outputs",
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(metadata | row)


def configure_segmenter(model: VAD, model_dir: str, args: argparse.Namespace) -> None:
    """Apply benchmark segmenter parameters without modifying the bundle.

    Parameters
    ----------
    model : VAD
        Loaded Fast Silero VAD model whose segmenter should be replaced.
    model_dir : str
        Model bundle directory containing ``model_config.yaml``.
    args : argparse.Namespace
        Benchmark options containing the segmenter parameters.
    """

    model_config = OmegaConf.load(Path(model_dir) / VAD_CONFIG_FILE)
    model_config.threshold = args.threshold
    model_config.min_speech_duration_ms = args.min_speech_duration_ms
    model_config.max_speech_duration_ms = args.max_speech_duration_ms
    model_config.min_silence_duration_ms = args.min_silence_duration_ms
    model_config.speech_pad_ms = args.speech_pad_ms
    validate_model_config(model_config)
    model.segmenter = SpeechSegmenter(model_config)


def run(args: argparse.Namespace) -> None:
    """Run the requested duration sweep and write its results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed duration-sweep options.

    Raises
    ------
    ValueError
        If durations, sample rate, warmup count, or repeat count are invalid.
    """

    for duration_ms in args.durations_ms:
        if duration_ms <= 0:
            raise ValueError(f"Durations must be positive, got {duration_ms} ms.")
    if args.samplerate <= 0:
        raise ValueError(f"Sample rate must be positive, got {args.samplerate}.")
    if args.warmup < 0:
        raise ValueError(f"Warmup count must be non-negative, got {args.warmup}.")
    if args.repeats <= 0:
        raise ValueError(f"Repeat count must be positive, got {args.repeats}.")

    official_model = get_official_model()
    standard_model = VAD(args.standard_model_dir)
    custom_op_model = VAD(args.custom_op_model_dir)
    configure_segmenter(standard_model, args.standard_model_dir, args)
    configure_segmenter(custom_op_model, args.custom_op_model_dir, args)
    rows = []

    max_duration_sec = max(args.durations_ms) / 1000.0
    if args.wav_path:
        source_audio, samplerate = read_wav(
            args.wav_path, args.wav_offset_sec, max_duration_sec
        )
    else:
        samplerate = args.samplerate
        source_audio = make_synthetic_audio(max_duration_sec, samplerate)

    for duration_ms in args.durations_ms:
        audio = source_audio[: round(duration_ms * samplerate / 1000)]
        duration_sec = len(audio) / samplerate
        benchmarks = (
            (
                "official_silero_onnx",
                lambda audio=audio: run_original_onnx(
                    official_model, audio, samplerate
                ),
            ),
            (
                "fast_silero_standard",
                lambda audio=audio: run_fast_engine(standard_model, audio, samplerate),
            ),
            (
                "fast_silero_standard_segmenter",
                lambda audio=audio: run_fast_pipeline(
                    standard_model, audio, samplerate
                ),
            ),
            (
                "fast_silero_custom_op",
                lambda audio=audio: run_fast_engine(custom_op_model, audio, samplerate),
            ),
            (
                "fast_silero_custom_op_segmenter",
                lambda audio=audio: run_fast_pipeline(
                    custom_op_model, audio, samplerate
                ),
            ),
        )
        results = [
            time_function(name, fn, args.warmup, args.repeats)
            for name, fn in benchmarks
        ]
        baseline_rtfx = duration_sec / results[0][2]
        for benchmark, best, median, outputs in results:
            rtfx = duration_sec / median
            rows.append(
                {
                    "language": "python",
                    "benchmark": benchmark,
                    "audio_duration_ms": str(duration_ms),
                    "audio_duration_sec": f"{duration_sec:.6f}",
                    "best_sec": f"{best:.9f}",
                    "median_sec": f"{median:.9f}",
                    "rtfx": f"{rtfx:.6f}",
                    "speedup": f"{rtfx / baseline_rtfx:.6f}",
                    "outputs": str(outputs),
                }
            )

    write_results(args.output_tsv_path, args, samplerate, rows)


def main() -> None:
    """Parse arguments and run the duration sweep."""

    run(parse_args())


if __name__ == "__main__":
    main()
