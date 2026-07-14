import wave
from argparse import Namespace
from array import array
from pathlib import Path

import numpy as np
import pytest

import fast_silero_vad.detect as detect


def _write_wav(
    wav_path: Path,
    samples: list[int],
    *,
    samplerate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(samplerate)
        pcm = array("h", samples)
        wav.writeframes(pcm.tobytes())


def _args(tmp_path: Path, *, wav_dir: Path, output_path: Path) -> Namespace:
    return Namespace(
        model_dir=str(tmp_path / "model"),
        wav_dir=str(wav_dir),
        wav_list_path=None,
        streaming_vad_chunk_sec=0.002,
        output_path=str(output_path),
    )


def test_read_wav_rejects_stereo_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, [1, 2, 3, 4], channels=2)

    with pytest.raises(ValueError, match="Expected mono WAV"):
        detect.read_wav(wav_path)


def test_read_wav_rejects_non_pcm16_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "pcm8.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(16000)
        wav.writeframes(bytes([1, 2, 3]))

    with pytest.raises(ValueError, match="Expected 16-bit PCM"):
        detect.read_wav(wav_path)


def test_detect_writes_offline_tsv_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "sample.wav"
    output_path = tmp_path / "segments.tsv"
    _write_wav(wav_path, [1, 2, 3])

    class FakeOfflineVAD:
        def __init__(self) -> None:
            self.is_offline = True
            self.samplerates: list[int] = []
            self.audio_segments: list[np.typing.NDArray[np.float32]] = []

        def apply_samplerate(self, samplerate: int) -> None:
            self.samplerates.append(samplerate)

        def __call__(
            self, audio: np.typing.NDArray[np.float32]
        ) -> list[dict[str, float]]:
            self.audio_segments.append(audio.copy())
            return [{"start": 0.0, "end": 0.1}]

    fake_vad = FakeOfflineVAD()
    monkeypatch.setattr(detect, "VAD", lambda model_dir: fake_vad)

    detect.run(_args(tmp_path, wav_dir=wav_dir, output_path=output_path))

    assert output_path.read_text(encoding="utf8") == (
        f"wav_path\tstart\tend\n{wav_path}\t0.0\t0.1\n"
    )
    assert fake_vad.samplerates == [16000]
    np.testing.assert_allclose(
        fake_vad.audio_segments[0],
        np.array([1, 2, 3], dtype=np.float32) / np.float32(2**15 - 1),
    )


def test_detect_streaming_sets_final_only_on_last_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "sample.wav"
    output_path = tmp_path / "segments.tsv"
    _write_wav(wav_path, [1, 2, 3, 4, 5], samplerate=1000)

    class FakeStreamingVAD:
        def __init__(self) -> None:
            self.is_offline = False
            self.calls: list[tuple[np.typing.NDArray[np.float32], bool]] = []
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def apply_samplerate(self, samplerate: int) -> None:
            self.samplerate = samplerate

        def __call__(
            self, audio: np.typing.NDArray[np.float32], final: bool
        ) -> list[dict[str, float]]:
            self.calls.append((audio.copy(), final))
            if final:
                return [{"start": 0.0, "end": 0.005}]
            return []

    fake_vad = FakeStreamingVAD()
    monkeypatch.setattr(detect, "VAD", lambda model_dir: fake_vad)

    detect.run(_args(tmp_path, wav_dir=wav_dir, output_path=output_path))

    assert fake_vad.reset_count == 1
    assert fake_vad.samplerate == 1000
    assert [final for _, final in fake_vad.calls] == [False, False, True]
    np.testing.assert_allclose(
        np.hstack([audio for audio, _ in fake_vad.calls]),
        np.arange(1, 6, dtype=np.float32) / np.float32(2**15 - 1),
    )
    assert output_path.read_text(encoding="utf8") == (
        f"wav_path\tstart\tend\n{wav_path}\t0.0\t0.005\n"
    )


def test_detect_streaming_finalizes_empty_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    wav_path = wav_dir / "empty.wav"
    output_path = tmp_path / "segments.tsv"
    _write_wav(wav_path, [])

    class FakeStreamingVAD:
        is_offline = False

        def __init__(self) -> None:
            self.calls: list[tuple[np.typing.NDArray[np.float32], bool]] = []

        def reset(self) -> None:
            pass

        def apply_samplerate(self, samplerate: int) -> None:
            pass

        def __call__(
            self,
            audio: np.typing.NDArray[np.float32],
            final: bool,
        ) -> list[dict[str, float]]:
            self.calls.append((audio, final))
            return []

    fake_vad = FakeStreamingVAD()
    monkeypatch.setattr(detect, "VAD", lambda model_dir: fake_vad)

    detect.run(_args(tmp_path, wav_dir=wav_dir, output_path=output_path))

    assert len(fake_vad.calls) == 1
    assert len(fake_vad.calls[0][0]) == 0
    assert fake_vad.calls[0][1] is True
