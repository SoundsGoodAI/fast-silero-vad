from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from fast_silero_vad.constants import VAD_CONFIG_FILE, VAD_ONNX_FILE
from fast_silero_vad.utils import VADInitializationError, validate_model_dir

INPUT_NAMES = [
    "input_vad",
    "cached_left_context",
    "h_recurrent_state",
    "c_recurrent_state",
]
OUTPUT_NAMES = [
    "output_vad",
    "output_cached_left_context",
    "output_h_recurrent_state",
    "output_c_recurrent_state",
]
HIDDEN_DIM = 128


def _write_model_bundle(
    tmp_path: Path,
    *,
    model_type: str = "streaming_vad",
    device: str = "cpu",
    model_samplerate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 32,
    min_silence_duration_ms: int = 100,
    max_speech_duration_ms: int = 64,
    speech_pad_ms: int = 0,
    custom_op: str | None = None,
) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)

    custom_op_config = ""
    if custom_op is not None:
        custom_op_config = f"onnx_custom_op: {custom_op}\n"

    (model_dir / VAD_CONFIG_FILE).write_text(
        f"model_type: {model_type}\n"
        f"device: {device}\n"
        f"model_samplerate: {model_samplerate}\n"
        f"threshold: {threshold}\n"
        f"min_speech_duration_ms: {min_speech_duration_ms}\n"
        f"min_silence_duration_ms: {min_silence_duration_ms}\n"
        f"max_speech_duration_ms: {max_speech_duration_ms}\n"
        f"speech_pad_ms: {speech_pad_ms}\n"
        f"{custom_op_config}",
        encoding="utf8",
    )
    (model_dir / VAD_ONNX_FILE).write_bytes(b"model")

    return model_dir


def _context_samples_for(model_samplerate: int) -> int:
    chunk_samples = 512 if model_samplerate == 16000 else 256
    return chunk_samples // 8


def _patch_onnx_signature(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    input_shapes: list[tuple[int | str, ...]] | None = None,
    output_shapes: list[tuple[int | str, ...]] | None = None,
) -> None:
    def fake_session(
        path: str, model_config: DictConfig
    ) -> tuple[
        object,
        list[str],
        list[str],
        list[tuple[int | str, ...]],
        list[tuple[int | str, ...]],
    ]:
        context_samples = _context_samples_for(model_config.model_samplerate)
        return (
            object(),
            input_names if input_names is not None else INPUT_NAMES,
            output_names if output_names is not None else OUTPUT_NAMES,
            input_shapes
            if input_shapes is not None
            else [
                (1, "input_vad_len"),
                (1, context_samples),
                (1, HIDDEN_DIM),
                (1, HIDDEN_DIM),
            ],
            output_shapes
            if output_shapes is not None
            else [
                (1, "output_vad_len"),
                (1, context_samples),
                (1, HIDDEN_DIM),
                (1, HIDDEN_DIM),
            ],
        )

    monkeypatch.setattr("fast_silero_vad.utils.get_onnxruntime_session", fake_session)


@pytest.mark.parametrize("model_samplerate", (8000, 16000))
def test_validation_accepts_cpu_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_samplerate: int
) -> None:
    _patch_onnx_signature(monkeypatch)
    model_dir = _write_model_bundle(tmp_path, model_samplerate=model_samplerate)

    validate_model_dir(str(model_dir))


def test_validation_rejects_cuda_bundle(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(tmp_path, model_type="offline_vad", device="cuda")

    with pytest.raises(VADInitializationError, match="CPU-only"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_missing_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(VADInitializationError):
        validate_model_dir(str(model_dir))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "asr"),
        ("device", "tpu"),
        ("model_samplerate", 44100),
        ("threshold", 1.5),
        ("min_speech_duration_ms", -1),
        ("min_silence_duration_ms", -1),
        ("max_speech_duration_ms", -1),
        ("speech_pad_ms", -1),
    ],
)
def test_validation_rejects_invalid_config_values(
    tmp_path: Path, field: str, value: str | int | float
) -> None:
    model_dir = _write_model_bundle(tmp_path)
    model_config_path = model_dir / VAD_CONFIG_FILE
    model_config = OmegaConf.load(model_config_path)
    model_config[field] = value
    OmegaConf.save(model_config, model_config_path)

    with pytest.raises(VADInitializationError):
        validate_model_dir(str(model_dir))


def test_validation_rejects_overlapping_speech_padding(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(
        tmp_path, min_silence_duration_ms=32, speech_pad_ms=17
    )

    with pytest.raises(VADInitializationError, match="at least twice speech_pad_ms"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_too_short_max_speech_duration(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(
        tmp_path, min_speech_duration_ms=100, max_speech_duration_ms=199
    )

    with pytest.raises(VADInitializationError, match="at least twice"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_max_speech_duration_below_two_min_chunks(
    tmp_path: Path,
) -> None:
    model_dir = _write_model_bundle(
        tmp_path, min_speech_duration_ms=33, max_speech_duration_ms=66
    )

    with pytest.raises(VADInitializationError, match="chunk rounding"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_max_speech_duration_below_one_chunk(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(
        tmp_path, min_speech_duration_ms=32, max_speech_duration_ms=0
    )

    with pytest.raises(VADInitializationError, match="chunk rounding"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_min_speech_duration_below_one_chunk(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(tmp_path, min_speech_duration_ms=31)

    with pytest.raises(VADInitializationError, match="at least 32 ms"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_min_silence_duration_below_one_chunk(
    tmp_path: Path,
) -> None:
    model_dir = _write_model_bundle(tmp_path, min_silence_duration_ms=31)

    with pytest.raises(VADInitializationError, match="at least 32 ms"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_missing_required_artifact(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(tmp_path)
    (model_dir / VAD_ONNX_FILE).unlink()

    with pytest.raises(VADInitializationError):
        validate_model_dir(str(model_dir))


def test_validation_rejects_invalid_onnx_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(
        monkeypatch,
        input_shapes=[(1, "input_vad_len"), (1, 32), (1, HIDDEN_DIM), (1, HIDDEN_DIM)],
    )
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="cached_left_context"):
        validate_model_dir(str(model_dir))


def test_validation_accepts_inferred_hidden_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden_dim = 96
    _patch_onnx_signature(
        monkeypatch,
        input_shapes=[(1, "input_vad_len"), (1, 64), (1, hidden_dim), (1, hidden_dim)],
        output_shapes=[
            (1, "output_vad_len"),
            (1, 64),
            (1, hidden_dim),
            (1, hidden_dim),
        ],
    )
    model_dir = _write_model_bundle(tmp_path)

    validate_model_dir(str(model_dir))


@pytest.mark.parametrize("hidden_shape", ((1, "hidden_dim"), (1, 0), (128,), (2, 128)))
def test_validation_rejects_invalid_hidden_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hidden_shape: tuple[int | str, ...],
) -> None:
    _patch_onnx_signature(
        monkeypatch,
        input_shapes=[(1, "input_vad_len"), (1, 64), hidden_shape, (1, HIDDEN_DIM)],
    )
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="fixed positive hidden_dim"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_wrong_cached_context_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(
        monkeypatch,
        output_shapes=[
            (1, "output_vad_len"),
            (1, 32),
            (1, HIDDEN_DIM),
            (1, HIDDEN_DIM),
        ],
    )
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="output_cached_left_context"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_wrong_onnx_input_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(monkeypatch, input_names=INPUT_NAMES[:-1])
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="exactly four inputs"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_wrong_onnx_output_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(monkeypatch, output_names=OUTPUT_NAMES[:-1])
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="exactly four outputs"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_wrong_onnx_input_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(
        monkeypatch,
        input_names=["input_vad", "cached_left_context", "h_recurrent_state", "wrong"],
    )
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="Expected ONNX inputs"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_wrong_onnx_output_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(
        monkeypatch,
        output_names=[
            "output_vad",
            "output_cached_left_context",
            "output_h_recurrent_state",
            "wrong",
        ],
    )
    model_dir = _write_model_bundle(tmp_path)

    with pytest.raises(VADInitializationError, match="Expected ONNX outputs"):
        validate_model_dir(str(model_dir))


def test_validation_rejects_missing_custom_op_library(tmp_path: Path) -> None:
    model_dir = _write_model_bundle(tmp_path, custom_op="silero_frontend.so")

    with pytest.raises(VADInitializationError):
        validate_model_dir(str(model_dir))


def test_validation_accepts_existing_custom_op_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_onnx_signature(monkeypatch)
    model_dir = _write_model_bundle(tmp_path, custom_op="silero_frontend.so")
    (model_dir / "silero_frontend.so").write_bytes(b"custom-op")

    validate_model_dir(str(model_dir))
