#!/usr/bin/env python3
"""Shared runtime and timing helpers for Silero VAD benchmarks."""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch

from fast_silero_vad.vad import VAD

if TYPE_CHECKING:
    from silero_vad.utils_vad import OnnxWrapper


def read_wav(
    path: str | Path, offset_sec: float = 0.0, duration_sec: float | None = None
) -> tuple[np.typing.NDArray[np.float32], int]:
    """Read a mono 16-bit PCM WAV file or an exact excerpt from it.

    Parameters
    ----------
    path : str | Path
        WAV file path.
    offset_sec : float
        Non-negative start offset in seconds.
    duration_sec : float | None
        Exact duration to read in seconds. Reads through the end when omitted.

    Returns
    -------
    tuple[np.typing.NDArray[np.float32], int]
        Normalized mono float32 audio and sample rate in Hz.

    Raises
    ------
    ValueError
        If the offset or duration is invalid, the WAV is not mono 16-bit PCM,
        or the requested excerpt extends beyond the available audio.
    """

    if offset_sec < 0.0:
        raise ValueError(f"WAV offset must be non-negative, got {offset_sec}.")
    if duration_sec is not None and duration_sec <= 0.0:
        raise ValueError(f"WAV duration must be positive, got {duration_sec}.")

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV, got {wav.getnchannels()} channels.")
        if wav.getsampwidth() != 2:
            raise ValueError(
                f"Expected 16-bit PCM WAV, got {8 * wav.getsampwidth()} bits."
            )
        samplerate = wav.getframerate()
        start_frame = round(offset_sec * samplerate)
        num_frames = (
            wav.getnframes() - start_frame
            if duration_sec is None
            else round(duration_sec * samplerate)
        )
        if (
            start_frame > wav.getnframes()
            or start_frame + num_frames > wav.getnframes()
        ):
            raise ValueError(
                f"WAV {path} does not contain the requested audio at offset "
                f"{offset_sec} seconds."
            )
        wav.setpos(start_frame)
        data = wav.readframes(num_frames)

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return np.clip(audio / (2**15 - 1), -1.0, 1.0), samplerate


def make_synthetic_audio(
    duration_sec: float, samplerate: int
) -> np.typing.NDArray[np.float32]:
    """Create deterministic normalized mono audio for repeatable benchmarks.

    Parameters
    ----------
    duration_sec : float
        Synthetic audio duration in seconds.
    samplerate : int
        Synthetic audio sample rate in Hz.

    Returns
    -------
    np.typing.NDArray[np.float32]
        One-dimensional normalized float32 audio.
    """

    num_samples = round(duration_sec * samplerate)
    timeline = np.arange(num_samples, dtype=np.float32) / samplerate
    waveform = 0.05 * np.sin(2.0 * np.pi * 220.0 * timeline)
    waveform += 0.02 * np.sin(2.0 * np.pi * 440.0 * timeline)
    waveform *= (np.sin(2.0 * np.pi * 0.8 * timeline) > -0.4).astype(np.float32)
    return waveform.astype(np.float32)


def get_official_model() -> OnnxWrapper:
    """Load the ONNX model bundled with the official ``silero-vad`` package."""

    # The ONNX wrapper does not use torchaudio, but silero-vad imports it eagerly.
    sys.modules.setdefault("torchaudio", ModuleType("torchaudio"))
    from silero_vad import load_silero_vad

    return load_silero_vad(onnx=True)


def run_original_onnx(
    model: OnnxWrapper, audio: np.typing.NDArray[np.float32], samplerate: int
) -> int:
    """Run the official ``silero-vad`` Python ONNX pipeline.

    Parameters
    ----------
    model : OnnxWrapper
        ONNX wrapper loaded from the official ``silero-vad`` package.
    audio : np.typing.NDArray[np.float32]
        Normalized mono float32 audio.
    samplerate : int
        Audio sample rate in Hz.

    Returns
    -------
    int
        Number of model probabilities produced.
    """

    probabilities = model.audio_forward(
        torch.from_numpy(audio).unsqueeze(0), samplerate
    )
    return probabilities.shape[1]


def run_fast_engine(
    model: VAD, audio: np.typing.NDArray[np.float32], samplerate: int
) -> int:
    """Run only the Fast Silero VAD inference engine.

    Parameters
    ----------
    model : VAD
        Fast Silero VAD runtime wrapper.
    audio : np.typing.NDArray[np.float32]
        Normalized mono float32 audio.
    samplerate : int
        Audio sample rate in Hz.

    Returns
    -------
    int
        Number of probability chunks produced by the engine.
    """

    model.engine.reset()
    model.engine.apply_samplerate(samplerate)
    return len(model.engine(audio, final=True))


def run_fast_pipeline(
    model: VAD, audio: np.typing.NDArray[np.float32], samplerate: int
) -> int:
    """Run the complete offline Fast Silero VAD pipeline.

    Parameters
    ----------
    model : VAD
        Fast Silero VAD runtime wrapper.
    audio : np.typing.NDArray[np.float32]
        Normalized mono float32 audio.
    samplerate : int
        Audio sample rate in Hz.

    Returns
    -------
    int
        Number of speech segments produced by the pipeline.
    """

    model.apply_samplerate(samplerate)
    return len(model(audio))


def time_function(
    name: str, fn: Callable[[], int], warmup: int, repeats: int
) -> tuple[str, float, float, int]:
    """Measure one benchmark function.

    Parameters
    ----------
    name : str
        Benchmark label.
    fn : Callable[[], int]
        Function being measured.
    warmup : int
        Number of unmeasured warmup calls.
    repeats : int
        Number of timed calls.

    Returns
    -------
    tuple[str, float, float, int]
        Benchmark label, best elapsed seconds, median elapsed seconds, and
        output count from the final timed call.
    """

    for _ in range(warmup):
        fn()

    output_count = 0
    elapsed_values = []
    for _ in range(repeats):
        start = time.perf_counter()
        output_count = fn()
        elapsed_values.append(time.perf_counter() - start)

    return name, min(elapsed_values), statistics.median(elapsed_values), output_count


def get_cpu_model() -> str:
    """Return the host CPU model used for benchmark metadata.

    Linux reads the first model name from ``/proc/cpuinfo``. macOS queries
    ``machdep.cpu.brand_string`` through ``sysctl``. Other platforms, or
    failed platform-specific queries, fall back to Python's platform data.

    Returns
    -------
    str
        Concise CPU model, processor, or machine architecture string.
    """

    if sys.platform == "linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf8").splitlines()
        except OSError:
            cpuinfo = ()
        for line in cpuinfo:
            name, separator, value = line.partition(":")
            if separator and name.strip() == "model name":
                return value.strip()

    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ("sysctl", "-n", "machdep.cpu.brand_string"), text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass

    return platform.processor() or platform.machine()
