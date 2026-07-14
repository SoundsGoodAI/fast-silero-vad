from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from fast_silero_vad import VAD
from fast_silero_vad.constants import VAD_CONFIG_FILE
from fast_silero_vad.utils import VADInferenceError


def _patch_engine(monkeypatch: pytest.MonkeyPatch, engine_class: type) -> None:
    monkeypatch.setattr("fast_silero_vad.vad.VADEngine", engine_class)


def _write_model_config(tmp_path: Path, *, model_type: str, device: str = "cpu") -> str:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / VAD_CONFIG_FILE).write_text(
        f"model_type: {model_type}\n"
        f"device: {device}\n"
        "model_samplerate: 8000\n"
        "threshold: 0.5\n"
        "min_speech_duration_ms: 32\n"
        "min_silence_duration_ms: 100\n"
        "max_speech_duration_ms: 64\n"
        "speech_pad_ms: 0\n",
        encoding="utf8",
    )
    return str(model_dir)


def test_vad_validates_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated_model_dirs = []
    monkeypatch.setattr(
        "fast_silero_vad.vad.validate_model_dir", validated_model_dirs.append
    )
    model_dir = _write_model_config(tmp_path, model_type="offline_vad")

    class FakeEngine:
        def __init__(self, model_dir: str, model_config: DictConfig) -> None:
            pass

    class FakeSegmenter:
        def __init__(self, model_config: DictConfig) -> None:
            pass

    _patch_engine(monkeypatch, FakeEngine)
    monkeypatch.setattr("fast_silero_vad.vad.SpeechSegmenter", FakeSegmenter)

    VAD(model_dir)

    assert validated_model_dirs == [model_dir]


@pytest.mark.parametrize("model_type", ("offline_vad", "streaming_vad"))
def test_vad_casts_audio_to_float32_and_validates_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    model_dir = _write_model_config(tmp_path, model_type=model_type)

    class FakeEngine:
        def __init__(self, model_dir: str, model_config: DictConfig) -> None:
            self.total_samples = 0
            self.audio: np.typing.NDArray[np.float32] | None = None

        def reset(self) -> None:
            self.total_samples = 0

        def __call__(
            self, audio_segment: np.typing.NDArray[np.float32], final: bool
        ) -> np.typing.NDArray[np.float32]:
            self.audio = audio_segment
            self.total_samples = len(audio_segment)
            return np.empty(0, dtype=np.float32)

    _patch_engine(monkeypatch, FakeEngine)
    vad = VAD(model_dir, validate=False)
    assert vad.is_offline is (model_type == "offline_vad")

    audio = np.zeros(256)
    if not vad.is_offline:
        vad(audio, final=False)
    else:
        vad(audio)

    assert vad.engine.audio is not None
    assert vad.engine.audio.dtype == np.float32

    with pytest.raises(VADInferenceError, match="must be one-dimensional"):
        if not vad.is_offline:
            vad(np.zeros((2, 256)), final=False)
        else:
            vad(np.zeros((2, 256)))

    with pytest.raises(
        VADInferenceError, match="samplerate must be a positive integer"
    ):
        vad.apply_samplerate(0)
    with pytest.raises(
        VADInferenceError, match="samplerate must be a positive integer"
    ):
        vad.apply_samplerate(16000.5)


def test_offline_vad_handles_empty_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _write_model_config(tmp_path, model_type="offline_vad")

    class FakeEngine:
        def __init__(self, model_dir: str, model_config: DictConfig) -> None:
            self.total_samples = 0
            self.reset_count = 0
            self.call_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(
            self, audio_segment: np.typing.NDArray[np.float32], final: bool
        ) -> np.typing.NDArray[np.float32]:
            self.call_count += 1
            return np.empty(0, dtype=np.float32)

    _patch_engine(monkeypatch, FakeEngine)

    vad = VAD(model_dir, validate=False)

    assert vad(np.empty(0, dtype=np.float32)) == []
    assert vad.engine.reset_count == 1
    assert vad.engine.call_count == 1
    assert vad.timing["engine_sec"] >= 0.0
    assert vad.timing["segmenter_sec"] >= 0.0
    assert vad.timing["calls"] == 1


def test_offline_vad_forces_final_and_resets_after_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _write_model_config(tmp_path, model_type="offline_vad")

    class FakeEngine:
        def __init__(self, model_dir: str, model_config: DictConfig) -> None:
            self.total_samples = 0
            self.reset_count = 0
            self.segments: list[list[float]] = []
            self.finals: list[bool] = []

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(
            self, audio_segment: np.typing.NDArray[np.float32], final: bool
        ) -> np.typing.NDArray[np.float32]:
            self.segments.append(audio_segment.tolist())
            self.finals.append(final)
            self.total_samples = 123
            return np.array([0.8, 0.2], dtype=np.float32)

    class FakeSegmenter:
        def __init__(self, model_config: DictConfig) -> None:
            self.reset_count = 0
            self.calls: list[tuple[list[float], int, bool]] = []

        def reset(self) -> None:
            self.reset_count += 1

        def __call__(
            self,
            probabilities: np.typing.NDArray[np.float32],
            total_samples: int,
            final: bool,
        ) -> list[dict[str, float]]:
            self.calls.append((probabilities.tolist(), total_samples, final))
            return [{"start": 0.0, "end": 0.1}]

    _patch_engine(monkeypatch, FakeEngine)
    monkeypatch.setattr("fast_silero_vad.vad.SpeechSegmenter", FakeSegmenter)

    vad = VAD(model_dir, validate=False)

    segments = vad(np.array([0.1, 0.2, 0.3], dtype=np.float32), final=False)

    assert segments == [{"start": 0.0, "end": 0.1}]
    assert vad.engine.reset_count == 1
    np.testing.assert_allclose(vad.engine.segments[0], [0.1, 0.2, 0.3])
    assert vad.engine.finals == [True]
    assert vad.segmenter.reset_count == 1
    assert vad.segmenter.calls == [
        ([0.800000011920929, 0.20000000298023224], 123, True)
    ]
    assert vad.timing["engine_sec"] >= 0.0
    assert vad.timing["segmenter_sec"] >= 0.0
    assert vad.timing["calls"] == 1


def test_streaming_vad_preserves_state_until_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = _write_model_config(tmp_path, model_type="streaming_vad")

    class FakeEngine:
        def __init__(self, model_dir: str, model_config: DictConfig) -> None:
            self.total_samples = 0
            self.reset_count = 0
            self.samplerate = 0

        def reset(self) -> None:
            self.reset_count += 1
            self.total_samples = 0

        def apply_samplerate(self, samplerate: int) -> None:
            self.samplerate = samplerate

        def __call__(
            self, audio_segment: np.typing.NDArray[np.float32], final: bool
        ) -> np.typing.NDArray[np.float32]:
            self.total_samples += len(audio_segment)
            return np.array([0.8] if len(audio_segment) > 0 else [], dtype=np.float32)

    _patch_engine(monkeypatch, FakeEngine)

    vad = VAD(model_dir, validate=False)

    vad.apply_samplerate(16000)
    assert vad.engine.samplerate == 16000

    assert vad(np.zeros(256, dtype=np.float32), final=False) == []
    assert vad.engine.reset_count == 0
    assert vad.segmenter.triggered is True
    assert vad.timing["calls"] == 1

    assert vad(np.empty(0, dtype=np.float32), final=True) == [
        {"start": 0.0, "end": 0.032}
    ]
    assert vad.engine.reset_count == 1
    assert vad.segmenter.triggered is False
    assert vad.timing["engine_sec"] >= 0.0
    assert vad.timing["segmenter_sec"] >= 0.0
    assert vad.timing["calls"] == 2
