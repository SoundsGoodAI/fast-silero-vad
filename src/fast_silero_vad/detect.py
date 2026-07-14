#!/usr/bin/env python3
"""Command-line WAV voice activity detection.

This module loads a packaged Fast Silero VAD bundle, reads mono 16-bit PCM WAV
files, applies the WAV sample rate once at the beginning of each audio file,
and writes detected speech segments to a TSV file.
"""

import argparse
import csv
import wave
from pathlib import Path

import numpy as np

from .vad import VAD

VAD_OUTPUT_HEADER = ("wav_path", "start", "end")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for WAV VAD.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options. Exactly one input source is required:
        ``--wav-dir`` for recursive WAV discovery or ``--wav-list-path`` for an
        explicit list of WAV paths.
    """

    parser = argparse.ArgumentParser(
        description="Detect speech segments with a packaged VAD model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Packaged VAD model directory.",
    )
    wav_input_group = parser.add_mutually_exclusive_group(required=True)
    wav_input_group.add_argument(
        "--wav-dir",
        type=str,
        help="Directory containing WAV files.",
    )
    wav_input_group.add_argument(
        "--wav-list-path",
        type=str,
        help="Text file containing WAV paths, one path per line.",
    )
    parser.add_argument(
        "--streaming-vad-chunk-sec",
        type=float,
        default=0.1,
        help="Chunk size used only when the model config selects streaming_vad.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="TSV output path, i.e. output_dir/result.tsv",
    )
    return parser.parse_args()


def read_wav(wav_path: str | Path) -> tuple[np.typing.NDArray[np.float32], int]:
    """Read one mono 16-bit PCM WAV file.

    Parameters
    ----------
    wav_path : str | Path
        Path to a WAV file.

    Returns
    -------
    tuple[np.typing.NDArray[np.float32], int]
        Normalized mono float32 audio and the WAV sample rate in Hz.

    Raises
    ------
    ValueError
        Raised when the WAV is not mono or not 16-bit PCM.
    """

    with wave.open(str(wav_path), "rb") as wav:
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()
        sampling_rate = wav.getframerate()
        data = wav.readframes(wav.getnframes())

    if channels != 1:
        raise ValueError(
            f"Expected mono WAV audio for {wav_path}, but found {channels} channels.",
        )
    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV audio for {wav_path}, but found "
            f"{8 * sample_width}-bit samples.",
        )

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    audio = np.clip(audio / np.float32(2**15 - 1), -1.0, 1.0)
    return audio, sampling_rate


def read_wav_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve WAV input paths from a directory or explicit path list.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    list[Path]
        Sorted WAV paths to process.
    """

    if args.wav_list_path:
        with open(args.wav_list_path, encoding="utf8") as wav_list_file:
            return sorted(
                Path(line.strip()) for line in wav_list_file if line.strip() != ""
            )
    return sorted(Path(args.wav_dir).rglob("*.wav"))


def run(args: argparse.Namespace) -> None:
    """Run VAD for requested WAV files and write a segment TSV.

    The model sample rate is applied exactly once per WAV file before any audio
    is processed. For ``offline_vad`` bundles the full PCM signal is processed
    in one call. For ``streaming_vad`` bundles the model is reset for each WAV,
    the sample rate is applied, and chunks are fed through the model
    with ``final=True`` only on the last chunk.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Raises
    ------
    FileNotFoundError
        Raised when the selected input source does not contain any WAV paths.
    """

    wav_paths = read_wav_paths(args)
    if not wav_paths:
        source = args.wav_list_path if args.wav_list_path else args.wav_dir
        raise FileNotFoundError(f"No WAV files found for {source}")

    model = VAD(args.model_dir)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=VAD_OUTPUT_HEADER,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for wav_path in wav_paths:
            audio, sampling_rate = read_wav(wav_path)
            if not model.is_offline:
                model.reset()
                model.apply_samplerate(sampling_rate)
                segments = []
                chunk_samples = max(
                    1, round(args.streaming_vad_chunk_sec * sampling_rate)
                )
                for start in range(0, max(1, len(audio)), chunk_samples):
                    end = min(len(audio), start + chunk_samples)
                    segments.extend(
                        model(audio[start:end], final=end >= len(audio)),
                    )
            else:
                model.apply_samplerate(sampling_rate)
                segments = model(audio)

            for segment in segments:
                writer.writerow(
                    {
                        "wav_path": str(wav_path),
                        "start": segment["start"],
                        "end": segment["end"],
                    },
                )


def main() -> None:
    """Parse CLI arguments and run VAD detection."""
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
