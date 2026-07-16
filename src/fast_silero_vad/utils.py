"""Validation and packaging helpers for Fast Silero VAD bundles."""

from __future__ import annotations

import json
import shutil
import tarfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import onnxruntime
from omegaconf import DictConfig, OmegaConf

from .constants import (
    FAST_SILERO_VAD_VERSION,
    VAD_CONFIG_FILE,
    VAD_INPUT_NAMES,
    VAD_MODEL_TARBALL,
    VAD_MODEL_TYPES,
    VAD_ONNX_FILE,
    VAD_OUTPUT_NAMES,
)


class VADInitializationError(Exception):
    """Raised when a model bundle cannot be initialized or validated."""


class VADInferenceError(Exception):
    """Raised when runtime input is incompatible with VAD inference."""


def get_initial_states(
    context_samples: int, hidden_dim: int
) -> tuple[
    np.typing.NDArray[np.float32],
    np.typing.NDArray[np.float32],
    np.typing.NDArray[np.float32],
]:
    """Create empty recurrent and audio-context states for one VAD stream.

    Parameters
    ----------
    context_samples : int
        Number of model-rate samples cached as left context.
    hidden_dim : int
        LSTM hidden-state dimension used by the exported Silero branch.

    Returns
    -------
    tuple[np.typing.NDArray[np.float32], ...]
        Cached left context, hidden state, and cell state in ONNX input shapes.
    """

    left_context = np.zeros((1, context_samples), dtype=np.float32)
    h_recurrent_state = np.zeros((1, hidden_dim), dtype=np.float32)
    c_recurrent_state = np.zeros((1, hidden_dim), dtype=np.float32)

    return left_context, h_recurrent_state, c_recurrent_state


def get_onnxruntime_session(
    onnx_model_path: str, model_config: DictConfig
) -> tuple[
    onnxruntime.InferenceSession,
    list[str],
    list[str],
    list[tuple[int | str, ...]],
    list[tuple[int | str, ...]],
]:
    """Create a single-threaded CPU ONNX Runtime session.

    Parameters
    ----------
    onnx_model_path : str
        Path to the packaged ``vad.onnx`` model.
    model_config : DictConfig
        Bundle configuration loaded from ``model_config.yaml``. If
        ``onnx_custom_op`` is present, the referenced shared library is loaded
        relative to the model directory.

    Returns
    -------
    tuple[
        onnxruntime.InferenceSession,
        list[str],
        list[str],
        list[tuple[int | str, ...]],
        list[tuple[int | str, ...]],
    ]
        Session, input names, output names, input shapes, and output shapes.
    """

    onnxruntime.set_default_logger_severity(4)
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 4
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    model_dir_path = Path(onnx_model_path).parent
    custom_op_library = model_config.get("onnx_custom_op")
    if custom_op_library is not None:
        session_opts.register_custom_ops_library(
            str(model_dir_path / custom_op_library)
        )

    session = onnxruntime.InferenceSession(
        onnx_model_path, session_opts, providers=["CPUExecutionProvider"]
    )
    input_names = [session_input.name for session_input in session.get_inputs()]
    output_names = [session_output.name for session_output in session.get_outputs()]
    input_shapes = [
        tuple(session_input.shape) for session_input in session.get_inputs()
    ]
    output_shapes = [
        tuple(session_output.shape) for session_output in session.get_outputs()
    ]

    return session, input_names, output_names, input_shapes, output_shapes


def validate_model_dir(model_dir: str) -> None:
    """Validate a packaged Fast Silero VAD model directory.

    Parameters
    ----------
    model_dir : str
        Path to an unpacked model bundle.

    Raises
    ------
    VADInitializationError
        Raised when required files are missing, configuration values are
        unsupported, custom-op libraries are absent, or the ONNX graph
        signature is incompatible with the runtime.
    """

    model_dir_path = Path(model_dir)
    model_config_path = model_dir_path / VAD_CONFIG_FILE
    if not model_config_path.exists():
        raise VADInitializationError(
            f"Missing VAD model configuration {model_config_path}.",
        )

    model_config = OmegaConf.load(model_config_path)
    validate_model_config(model_config)

    for required_file in (VAD_CONFIG_FILE, VAD_ONNX_FILE):
        if not (model_dir_path / required_file).exists():
            raise VADInitializationError(
                f"Missing required VAD bundle file {model_dir_path / required_file}.",
            )

    custom_op_library = model_config.get("onnx_custom_op")
    if (
        custom_op_library is not None
        and not (model_dir_path / custom_op_library).exists()
    ):
        raise VADInitializationError(
            "Missing ONNX Runtime custom-op library "
            f"{model_dir_path / custom_op_library}.",
        )

    validate_onnx_model(model_dir_path, model_config)


def validate_model_config(model_config: DictConfig) -> None:
    """Validate runtime configuration values independent of model files.

    Parameters
    ----------
    model_config : DictConfig
        OmegaConf configuration loaded from ``model_config.yaml`` or prepared
        by the export pipeline.

    Raises
    ------
    VADInitializationError
        Raised when the configuration declares unsupported model metadata or
        impossible segmenter timing values.
    """

    if model_config.model_type not in VAD_MODEL_TYPES:
        raise VADInitializationError(
            "Expected model_type to be one of "
            f"{', '.join(VAD_MODEL_TYPES)}, got {model_config.model_type}.",
        )
    if model_config.device != "cpu":
        raise VADInitializationError(
            f"Fast Silero VAD is CPU-only, got device={model_config.device}.",
        )
    if not isinstance(
        model_config.model_samplerate, int
    ) or model_config.model_samplerate not in (8000, 16000):
        raise VADInitializationError(
            "The runtime supports only 8000 Hz and 16000 Hz models, "
            f"got {model_config.model_samplerate}.",
        )
    if (
        not isinstance(model_config.threshold, float)
        or not 0.0 <= model_config.threshold <= 1.0
    ):
        raise VADInitializationError(
            f"Expected threshold in [0.0, 1.0], got {model_config.threshold}.",
        )

    for key in (
        "min_speech_duration_ms",
        "max_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
    ):
        value = model_config[key]
        if not isinstance(value, int) or value < 0:
            raise VADInitializationError(
                "Expected non-negative integer segmenter parameter "
                f"{key}, got {value}.",
            )

    chunk_samples = 512 if model_config.model_samplerate == 16000 else 256
    chunk_ms = 1000 * chunk_samples // model_config.model_samplerate
    if model_config.min_speech_duration_ms < chunk_ms:
        raise VADInitializationError(
            f"min_speech_duration_ms must be at least {chunk_ms} ms "
            "to ensure at least one model-rate chunk is processed.",
        )
    if model_config.min_silence_duration_ms < chunk_ms:
        raise VADInitializationError(
            f"min_silence_duration_ms must be at least {chunk_ms} ms "
            "to observe at least one non-speech chunk.",
        )

    ms_samples = round(model_config.model_samplerate / 1000)
    min_speech_samples = model_config.min_speech_duration_ms * ms_samples
    max_speech_samples = model_config.max_speech_duration_ms * ms_samples
    min_speech_chunks = min_speech_samples // chunk_samples + (
        1 if min_speech_samples % chunk_samples > 0 else 0
    )
    max_speech_chunks = max_speech_samples // chunk_samples + (
        1 if max_speech_samples % chunk_samples > 0 else 0
    )

    if max_speech_chunks < 2 * min_speech_chunks:
        raise VADInitializationError(
            "max_speech_duration_ms must cover at least twice "
            "min_speech_duration_ms after model-rate chunk rounding.",
        )
    if model_config.min_silence_duration_ms < 2 * model_config.speech_pad_ms:
        raise VADInitializationError(
            "min_silence_duration_ms must be at least twice "
            "speech_pad_ms to keep padded segments from overlapping.",
        )


def validate_onnx_model(model_dir: Path, model_config: DictConfig) -> None:
    """Validate the CPU ONNX graph signature against runtime expectations.

    Parameters
    ----------
    model_dir : Path
        Path to the unpacked model bundle directory.
    model_config : DictConfig
        OmegaConf model configuration loaded from ``model_config.yaml``.

    Raises
    ------
    VADInitializationError
        Raised when the graph does not expose the expected four inputs, four
        outputs, or recurrent state tensor shapes.
    """

    _, input_names, output_names, input_shapes, output_shapes = get_onnxruntime_session(
        str(model_dir / VAD_ONNX_FILE), model_config
    )

    if len(input_names) != 4:
        raise VADInitializationError(
            "Expected exactly four inputs for the VAD ONNX model, "
            f"but found {len(input_names)} inputs.",
        )
    if len(output_names) != 4:
        raise VADInitializationError(
            "Expected exactly four outputs for the VAD ONNX model, "
            f"but found {len(output_names)} outputs.",
        )

    expected_input_names = sorted(VAD_INPUT_NAMES)
    if sorted(input_names) != expected_input_names:
        raise VADInitializationError(
            f"Expected ONNX inputs {expected_input_names}, got {sorted(input_names)}.",
        )

    expected_output_names = sorted(VAD_OUTPUT_NAMES)
    if sorted(output_names) != expected_output_names:
        raise VADInitializationError(
            f"Expected ONNX outputs {expected_output_names}, got "
            f"{sorted(output_names)}.",
        )

    input_shapes_by_name = dict(zip(input_names, input_shapes, strict=True))
    output_shapes_by_name = dict(zip(output_names, output_shapes, strict=True))

    context_shape = (1, 64 if model_config.model_samplerate == 16000 else 32)
    hidden_shape = input_shapes_by_name["h_recurrent_state"]
    if (
        len(hidden_shape) != 2
        or hidden_shape[0] != 1
        or not isinstance(hidden_shape[1], int)
        or hidden_shape[1] <= 0
    ):
        raise VADInitializationError(
            "Expected ONNX tensor h_recurrent_state to have shape "
            f"(1, hidden_dim) with a fixed positive hidden_dim, got {hidden_shape}.",
        )

    expected_shapes = {
        "cached_left_context": context_shape,
        "c_recurrent_state": hidden_shape,
        "output_cached_left_context": context_shape,
        "output_h_recurrent_state": hidden_shape,
        "output_c_recurrent_state": hidden_shape,
    }
    all_shapes = input_shapes_by_name | output_shapes_by_name
    for name, expected_shape in expected_shapes.items():
        if all_shapes[name] != expected_shape:
            raise VADInitializationError(
                f"Expected ONNX tensor {name} shape {expected_shape}, got "
                f"{all_shapes[name]}.",
            )


def make_version_file(
    onnxruntime_version: str,
    onnx_opset_version: int,
    opts: Namespace,
    output_path: str | Path,
) -> None:
    """Write lightweight export provenance into ``VERSION``.

    Parameters
    ----------
    onnxruntime_version : str
        ONNX Runtime version used during export.
    onnx_opset_version : int
        ONNX opset version used for the exported graph.
    opts : Namespace
        Packaging options with source checkpoint and VAD branch metadata.
    output_path : str | Path
        Destination VERSION file.
    """

    data: dict[str, str | int | bool] = {
        "fast_silero_vad_version": FAST_SILERO_VAD_VERSION,
        "onnxruntime_version": onnxruntime_version,
        "onnx_opset_version": onnx_opset_version,
        "source_model_license": "MIT",
        "source_model_project": "snakers4/silero-vad",
        "source_model_sha256": opts.source_model_sha256,
        "vad_branch": opts.vad_branch,
        "use_onnxruntime_custom_op": bool(opts.use_onnxruntime_custom_op),
    }
    with open(output_path, "w", encoding="utf8") as version_file:
        version_file.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def prepare_output_dir(output_dir_path: str | Path) -> Path:
    """Create an empty output model directory.

    Parameters
    ----------
    output_dir_path : str | Path
        Directory to recreate for model packaging.

    Returns
    -------
    Path
        Prepared output directory path.
    """

    output_dir = Path(output_dir_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def prepare_vad_config(opts: Namespace, output_dir_path: Path) -> None:
    """Write the runtime ``model_config.yaml`` for a packaged VAD bundle.

    Parameters
    ----------
    opts : Namespace
        Packaging options and inferred model metadata.
    output_dir_path : Path
        Model bundle directory where the config should be written.
    """

    model_config = {
        "model_type": opts.model_type,
        "device": "cpu",
        "model_samplerate": opts.model_samplerate,
        "threshold": opts.threshold,
        "min_speech_duration_ms": opts.min_speech_duration_ms,
        "max_speech_duration_ms": opts.max_speech_duration_ms,
        "min_silence_duration_ms": opts.min_silence_duration_ms,
        "speech_pad_ms": opts.speech_pad_ms,
    }
    if opts.onnx_custom_op is not None:
        model_config["onnx_custom_op"] = opts.onnx_custom_op

    config = OmegaConf.create(model_config)
    validate_model_config(config)
    OmegaConf.save(config=config, f=output_dir_path / VAD_CONFIG_FILE)


def tar_model(output_dir_path: str | Path) -> Path:
    """Create ``model.tgz`` next to the unpacked model directory.

    Parameters
    ----------
    output_dir_path : str | Path
        Model bundle directory to archive.

    Returns
    -------
    Path
        Created tarball path.
    """

    output_dir = Path(output_dir_path)
    tar_path = output_dir / VAD_MODEL_TARBALL
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(output_dir.iterdir()):
            if path.name == VAD_MODEL_TARBALL:
                continue
            tar.add(path, arcname=path.name)
    return tar_path
